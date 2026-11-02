import asyncio
import gc
import hmac
import html
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import quote, urljoin, urlparse

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rednote-mm-bot")

# httpx/httpcore log full request URLs at INFO.  For Telegram those URLs
# contain ``/bot<TOKEN>/...``; two production tokens were exposed that way
# when Render logs were copied.  Keep those libraries at WARNING and also
# redact any secret-shaped text that reaches a handler.
for _noisy_logger in ("httpx", "httpcore", "hpack", "h11"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


def _redact_secret_text(value, limit: int = 0) -> str:
    """Return text safe for logs/Telegram errors without credentials."""
    text = str(value or "")
    for name in (
        "BOT_TOKEN",
        "GEMINI_API_KEY",
        "WHISPER_API_KEY",
        "WORKER_API_KEY",
        "N8N_WEBHOOK_SECRET",
        "XHS_COOKIE",
    ):
        secret = str(globals().get(name) or "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(https?://api\.telegram\.org/(?:file/)?bot)[^/\s'\"]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b", "[REDACTED_BOT_TOKEN]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)((?:api[_-]?key|cookie|token|secret|x-worker-key)\s*[:=]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:api[_-]?key|key|token|access_token|xsec_token)=)[^&\s]+",
        r"\1[REDACTED]",
        text,
    )
    if limit and len(text) > int(limit):
        text = text[: int(limit)].rstrip() + "…"
    return text


def _safe_error_text(value, limit: int = 1000) -> str:
    text = _redact_secret_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > int(limit):
        text = text[: int(limit)].rstrip() + "…"
    return text


class _SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            rendered = record.getMessage()
            redacted = _redact_secret_text(rendered)
            if redacted != rendered:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True


_SECRET_REDACTION_FILTER = _SecretRedactionFilter()
logger.addFilter(_SECRET_REDACTION_FILTER)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_SECRET_REDACTION_FILTER)


class SourceDownloadError(RuntimeError):
    """A source-acquisition failure with a safe, user-actionable reason."""

    def __init__(self, message: str, *, detail: str = "", user_message: str = ""):
        safe_message = _safe_error_text(message, 1000)
        super().__init__(safe_message)
        self.detail = _safe_error_text(detail or message, 2000)
        self.user_message = (
            _safe_error_text(user_message, 1000) if user_message else safe_message
        )


def setup_ffmpeg() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = "/tmp/ffbin"
        os.makedirs(bin_dir, exist_ok=True)
        target = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(target):
            try:
                os.symlink(exe, target)
            except Exception:
                shutil.copy2(exe, target)
                os.chmod(target, 0o755)
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        return target
    except Exception:
        return "ffmpeg"


FFMPEG_BIN = setup_ffmpeg()
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else "/tmp"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()


def _clean_public_base_url(value: str) -> str:
    return str(value or "").strip().rstrip("/")


def _resolve_public_base_url() -> str:
    """Resolve this service's real public URL for Telegram webhook setup.

    Priority matters.  A stale, manually copied ``WEBHOOK_BASE_URL`` /
    ``PUBLIC_BASE_URL`` / ``RENDER_EXTERNAL_URL`` can otherwise make a
    freshly deployed bot re-register Telegram's webhook to an older
    service, leaving the user talking to obsolete code even though the
    new deploy is healthy.  Render's platform-provided hostname describes
    the instance that is actually running this process, so when an
    explicit value points at a *different* ``*.onrender.com`` host it is
    stale by definition and is ignored in favour of the real hostname.
    An explicit non-Render custom domain is still respected.
    """
    hostname = str(os.environ.get("RENDER_EXTERNAL_HOSTNAME") or "").strip()
    hostname = hostname.replace("https://", "").replace("http://", "")
    hostname = hostname.strip("/").split("/")[0].strip()
    actual_render_url = f"https://{hostname}" if hostname else ""

    explicit = _clean_public_base_url(
        os.environ.get("WEBHOOK_BASE_URL")
        or os.environ.get("PUBLIC_BASE_URL")
        or ""
    )
    if explicit:
        explicit_host = urlparse(explicit).netloc.lower()
        if (
            actual_render_url
            and explicit_host.endswith(".onrender.com")
            and explicit_host != hostname.lower()
        ):
            logger.warning(
                "Ignoring stale public URL %s; this instance is %s",
                explicit,
                actual_render_url,
            )
            return actual_render_url
        return explicit

    if actual_render_url:
        return actual_render_url

    render_url = _clean_public_base_url(os.environ.get("RENDER_EXTERNAL_URL"))
    if render_url:
        return render_url

    # Local/legacy fallback only; on Render the hostname above is present.
    return "https://rednote-bot-gzd0.onrender.com"


RENDER_EXTERNAL_URL = _resolve_public_base_url()
PUBLIC_BASE_URL = RENDER_EXTERNAL_URL
RENDER_GIT_COMMIT = os.environ.get("RENDER_GIT_COMMIT", "").strip()
RENDER_GIT_BRANCH = os.environ.get("RENDER_GIT_BRANCH", "").strip()
RENDER_GIT_REPO_SLUG = os.environ.get("RENDER_GIT_REPO_SLUG", "").strip()
RENDER_SERVICE_NAME = os.environ.get("RENDER_SERVICE_NAME", "").strip()
RENDER_SERVICE_ID = os.environ.get("RENDER_SERVICE_ID", "").strip()

def _parse_int_env(name: str, default: int) -> int:
    """Env က ဂဏန်းမဟုတ်ရင် bot မပျက်အောင် default သုံးတယ် (deploy ပျက်ခြင်း ကာကွယ်)."""
    try:
        return int((os.environ.get(name, "") or "").strip() or default)
    except (TypeError, ValueError):
        logger.warning(f"env {name} ဂဏန်း မဟုတ်ပါ — default {default} သုံးပါမယ်")
        return default


def _parse_id_list_env(name: str) -> set:
    out = set()
    for part in (os.environ.get(name, "") or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                logger.warning(f"env {name} ထဲက '{part}' ကို ကျော်ပါမယ် (ဂဏန်း မဟုတ်ပါ)")
    return out


ADMIN_ID = _parse_int_env("ADMIN_ID", 8430004524)
ALLOWED_USER_IDS = {8855067536} | _parse_id_list_env("ALLOWED_IDS")

# --- n8n worker mode -------------------------------------------------
# PIPELINE_MODE=direct (default) keeps the original all-in-one bot flow.
# PIPELINE_MODE=n8n keeps this Telegram bot as the front door/status UI,
# forwards each accepted link to an n8n Webhook, and lets n8n call the
# authenticated stage endpoints below on this same Render service.
PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "direct").strip().lower()
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "").strip()
N8N_WEBHOOK_SECRET = os.environ.get("N8N_WEBHOOK_SECRET", "").strip()
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "").strip()
WORKER_JOBS_DIR = os.path.join(WORK_ROOT, "n8n_jobs")
os.makedirs(WORKER_JOBS_DIR, exist_ok=True)

# Telegram's normal user-facing flow is the manual two-minute-part flow:
# script the whole item first, send each original part with its own Recap
# button, render only a clicked part, and join only after an explicit tap.
# RECAP_FLOW=auto restores the former one-shot behaviour (and n8n handoff
# for URL jobs when PIPELINE_MODE=n8n).
RECAP_FLOW = os.environ.get("RECAP_FLOW", "manual").strip().lower()
MANUAL_PARTS_ENABLED = RECAP_FLOW not in ("auto", "automatic", "direct", "n8n")
MANUAL_SESSION_TTL_HOURS = max(
    1, min(168, _parse_int_env("MANUAL_SESSION_TTL_HOURS", 24))
)

# Source download configuration.  Cookies are optional and must be set in
# Render's Environment/Secret Files UI, never pasted into chat or logs.
TELEGRAM_BOT_FILE_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
SOURCE_DOWNLOAD_TIMEOUT_SECONDS = max(
    60, _parse_int_env("SOURCE_DOWNLOAD_TIMEOUT_SECONDS", 1800)
)
YTDLP_USER_AGENT = (
    os.environ.get("YTDLP_USER_AGENT", "").strip()
    or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
YTDLP_COOKIES_FILE = (
    os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    or os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
)
XHS_COOKIE = (
    os.environ.get("XHS_COOKIE", "").strip()
    or os.environ.get("REDNOTE_COOKIE", "").strip()
)

# Whisper is called through an OpenAI-compatible transcription API.
# This works with OpenAI by default; another compatible provider can be
# selected by changing WHISPER_API_BASE/WHISPER_MODEL/WHISPER_API_KEY.
WHISPER_API_BASE = os.environ.get(
    "WHISPER_API_BASE", "https://api.openai.com/v1"
).strip().rstrip("/")
WHISPER_API_KEY = (
    os.environ.get("WHISPER_API_KEY", "").strip()
    or os.environ.get("OPENAI_API_KEY", "").strip()
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-1").strip() or "whisper-1"
# The whole intake pipeline is built around two-minute source parts, so
# Whisper audio is never sent in pieces longer than the video segments.
WHISPER_CHUNK_SECONDS = max(
    60, min(120, _parse_int_env("WHISPER_CHUNK_SECONDS", 120))
)

WORKER_STAGES = ("download", "transcribe", "rewrite", "tts", "render", "deliver")
WORKER_STAGE_LABELS = {
    "download": "📥 မူရင်းဗီဒီယို ဒေါင်းနေပါသည်...",
    "transcribe": "📝 Whisper ဖြင့် transcript ထုတ်နေပါသည်...",
    "rewrite": "🧠 Gemini ဖြင့် ပထမလူ script ပြန်ရေးနေပါသည်...",
    "tts": "🎙️ Gemini TTS ဖြင့် အသံသွင်းနေပါသည်...",
    "render": "🎬 အသံကို မထိဘဲ video ကို အချိန်ကိုက် ဖြတ်ဆက်နေပါသည်...",
    "deliver": "📤 Final video ကို ပို့နေပါသည်...",
}
WORKER_TASKS: dict = {}
WORKER_JOB_LOCK = asyncio.Lock()
WORKER_STAGE_SEMAPHORE = asyncio.Semaphore(1)

SAMPLE_RATE = 24000
MAX_TG_BYTES = 45 * 1024 * 1024
# Product rule: download the complete source first and split it into
# two-minute parts (last part may be shorter).  In the default Telegram
# flow the whole script is then written, original parts are sent back with
# per-part Recap buttons, and joining is a separate explicit tap.  This is
# fixed for every source duration, e.g. 3 hours -> 90 source parts.
SOURCE_SEGMENT_SECONDS = 120
CHUNK_DURATION = SOURCE_SEGMENT_SECONDS
PAUSE_AFTER_SCENE = 0.20
TELEGRAM_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpeg", ".mpg",
    ".3gp", ".3g2", ".ts", ".mts", ".m2ts",
}

# --- Final-duration target (input -> final video) ---
# The product requirement is about the *actual final video duration*, not
# a character estimate: e.g. a 600-second source at the default 90% setting
# must deliver a final video of about 540 seconds.  Character counts are
# only the initial estimate; after TTS the measured narration duration is
# fed back into Gemini expansion/trimming rounds until it is on target.
SCRIPT_TARGET_RATIO = 0.90      # default final duration = 90% of input
MINIMUM_SCRIPT_COVERAGE_PERCENT = 90  # every job/link must target at least 90%
SCRIPT_CHARS_PER_SEC = 6.5      # initial estimate only; TTS is re-measured
SCRIPT_MAX_RATIO = 1.05         # initial script cap before measured calibration
SCRIPT_TOPUP_ATTEMPTS = 6       # character-estimate top-ups before TTS
SCRIPT_MIN_CHAR_RATIO = 0.98    # initial script must reach 98% of estimate
SCRIPT_MIN_GROUP_CHAR_RATIO = 0.75  # one-liners do not count as covered
SCRIPT_MIN_NARRATION_RATIO = 0.90  # legacy metric name; final gates use window below
TRANSCRIPT_MIN_SPAN_RATIO = 0.75  # transcript must span most of the source
DURATION_CALIBRATION_ROUNDS = 4  # measured TTS -> expand/trim -> TTS again
FIRST_PERSON_REPAIR_ROUNDS = 3   # rewrite scenes missing ကျွန်တော်/ကျွန်မ
FINAL_DURATION_TOLERANCE_RATIO = 0.02  # final must be target ±2% of source
FINAL_DURATION_TOLERANCE_MIN_SECONDS = 0.35

PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
os.makedirs(PUBLISH_CACHE_DIR, exist_ok=True)

CANCELLED_JOBS: set = set()
BACKGROUND_TASKS: set = set()
MANUAL_SESSIONS_BY_CHAT: dict = {}
MANUAL_RENDER_TASKS: dict = {}
MANUAL_SESSION_LOCK = asyncio.Lock()
PERSISTENT_KEYBOARD_CHATS: set = set()
TELEGRAM_WEBHOOK_ALLOWED_UPDATES = [
    "message",
    "edited_message",
    "callback_query",
]
video_render_semaphore = asyncio.Semaphore(1)


def _spawn_background(coro, *, name: str = ""):
    """Create a task and retain a strong reference until it finishes.

    Python may garbage-collect an otherwise unreferenced task.  Telegram
    webhook/button handling must not depend on a fire-and-forget task that
    can disappear before it has replied.
    """
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.add(task)

    def _finished(done_task):
        BACKGROUND_TASKS.discard(done_task)
        if done_task.cancelled():
            return
        exc = done_task.exception()
        if exc:
            logger.error(
                "Background task %s failed: %s: %s",
                name or done_task.get_name(),
                type(exc).__name__,
                exc,
            )

    task.add_done_callback(_finished)
    return task

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
PREFERRED_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"]
SPINNING_EMOJIS = ["⏳", "⌛", "⏳", "⌛"]

DEEP_VOICE_FILTER = (
    "highpass=f=55,"
    "asetrate=22080,aresample=24000,atempo=1.0870,"
    "equalizer=f=110:t=q:w=1.0:g=4.5,"
    "equalizer=f=230:t=q:w=1.1:g=2.5,"
    "equalizer=f=6500:t=q:w=1.0:g=-1.5,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)


def get_anim_emoji(frame: int) -> str:
    return SPINNING_EMOJIS[frame % len(SPINNING_EMOJIS)]


# =====================================================================
# 🎛️ BUTTON MENU SYSTEM + FEATURE REGISTRY  (features အသစ် ထည့်နည်း)
# ---------------------------------------------------------------------
# Feature အသစ်ထည့်ချင်ရင် FEATURES list ထဲမှာ dict တစ်ခု ထည့်ရုံပါပဲ —
# menu ခလုတ်တွေ အလိုအလျောက် ပေါ်လာပါလိမ့်မယ်။
#   {"key": "coverage", "icon": "🎯", "label": "Script ပမာဏ",
#    "desc": "...", "kind": "cycle", "options": [(90, "90%"), ...]}
# kind အမျိုးအစားများ:
#   "cycle"  = ရွေးစရာများထဲမှ တစ်ခု (နှိပ်ရင် ရွေးချယ်မီနူး ပွင့်တယ်)
#   "toggle" = On/Off ခလုတ် (နှိပ်လိုက်တာနဲ့ တန်း ပြောင်းတယ်)
# ပြီးရင် DEFAULT_SETTINGS ထဲမှာ default တန်ဖိုး ထည့်ပေးပါ။
# =====================================================================

DEFAULT_SETTINGS = {
    "coverage": 90,        # 1  🎯 Script output = input ၏ %
    "voice": "Charon",     # 2  🎙 အသံ
    "deep": True,          # 3  🔊 Deep အသံ
    "speed": 1.0,          # 4  ⏩ အသံအမြန်နှုန်း
    "volume": 100,         # 5  🔉 အသံအတိုးအကျယ် %
    "resolution": 1280,    # 6  📐 ဗီဒီယို အရွယ်အစား (long edge px)
    "quality": 23,         # 7  💎 အရည်အသွေး (CRF, နည်းလေ ကောင်းလေ)
    "shield": True,        # 8  🛡️ Crop+Flip shield
    "color": True,         # 9  🎨 အရောင် မြှင့်တင်မှု
    "pause": 0.20,         # 10 ⏸ ခန်းကြား နားချိန် (စက္ကန့်)
    "cut": 8,              # 11 ✂️ ဖြတ်ထုတ်ခွင့် %
    "chunk": SOURCE_SEGMENT_SECONDS,  # 12 📦 မူရင်းကို ၂ မိနစ်စီခွဲ (fixed)
    "rate": 6.5,           # 13 🗣 Script အလျင် (စာလုံး/စက္ကန့်)
    "pov": "first",        # 14 👤 ပြောပြသူ ရှုထောင့်
    "tone": "natural",     # 15 🎭 ဇာတ်ကြောင်း လေသံ
    "split_tg": True,      # 16 📤 Telegram အတွက် ခွဲပို့ခြင်း
    "caption": True,       # 17 💬 Video caption ထည့်ခြင်း
    "notify": True,        # 18 🔔 ပြီးဆုံး အကြောင်းကြားခြင်း
    "fast_mode": False,    # 19 ⚡ မြန်ဆန်မုဒ် (preset အမြန်)
    "auto_start": True,    # 20 🚀 Link ပို့လိုက်တာနဲ့ တန်း စခြင်း
}

FEATURES = [
    {"key": "coverage", "icon": "🎯", "label": "Final ကြာချိန်", "kind": "cycle",
     "desc": "Final video အမှန်တကယ်ကြာချိန်သည် မူရင်း video ကြာချိန်၏ ဘယ်နှုန်း% ဖြစ်မလဲ (အနည်းဆုံး 90%)",
     "options": [(90, "90% ⭐"), (95, "95%"), (100, "100%")]},
    {"key": "voice", "icon": "🎙️", "label": "အသံ (Voice)", "kind": "cycle",
     "desc": "TTS အသံသရုပ်ဆောင် ရွေးပါ",
     "options": [("Charon", "Charon (ယောက်ျား·နက်)"), ("Fenrir", "Fenrir (ယောက်ျား)"),
                 ("Puck", "Puck (ယောက်ျား·သေးသေး)"), ("Kore", "Kore (မိန်းမ)"), ("Aoede", "Aoede (မိန်းမ·နူး)")]},
    {"key": "deep", "icon": "🔊", "label": "Deep အသံ", "kind": "toggle",
     "desc": "အသံကို နက်ရှိုင်း/ရင်အုပ်သံ ဖြစ်အောင် လုပ်ခြင်း"},
    {"key": "speed", "icon": "⏩", "label": "အမြန်နှုန်း", "kind": "cycle",
     "desc": "အသံဖတ် အမြန်နှုန်း",
     "options": [(0.9, "0.9x နှေး"), (1.0, "1.0x ပုံမှန်"), (1.1, "1.1x"), (1.2, "1.2x မြန်")]},
    {"key": "volume", "icon": "🔉", "label": "အသံအတိုးအကျယ်", "kind": "cycle",
     "desc": "အသံ volume %",
     "options": [(80, "80%"), (100, "100% ပုံမှန်"), (120, "120%"), (150, "150% ကျယ်")]},
    {"key": "resolution", "icon": "📐", "label": "ဗီဒီယို အရွယ်", "kind": "cycle",
     "desc": "Output ဗီဒီယို အရွယ်အစား",
     "options": [(720, "720p ပေါ့"), (1280, "1280p ပုံမှန်"), (1920, "1920p ကြီး")]},
    {"key": "quality", "icon": "💎", "label": "အရည်အသွေး", "kind": "cycle",
     "desc": "CRF — နည်းလေ အရည်အသွေးကောင်းလေ ဖိုင်ကြီးလေ",
     "options": [(18, "အကောင်းဆုံး"), (23, "ပုံမှန် ⭐"), (28, "ဖိုင်သေး")]},
    {"key": "shield", "icon": "🛡️", "label": "Shield (Crop+Flip)", "kind": "toggle",
     "desc": "ဗီဒီယိုကို crop + ဘေးပြောင်း လုပ်ခြင်း"},
    {"key": "color", "icon": "🎨", "label": "အရောင် မြှင့်", "kind": "toggle",
     "desc": "Contrast/Saturation မြှင့်တင်ခြင်း"},
    {"key": "pause", "icon": "⏸️", "label": "ခန်းကြား နားချိန်", "kind": "cycle",
     "desc": "ခန်းတစ်ခုနဲ့ တစ်ခုကြား တိတ်ဆိတ်ချိန်",
     "options": [(0.0, "မနား"), (0.2, "0.2s ⭐"), (0.5, "0.5s"), (1.0, "1.0s")]},
    {"key": "cut", "icon": "✂️", "label": "ဖြတ်ထုတ်ခွင့်", "kind": "cycle",
     "desc": "မလိုတဲ့ အပိုင်း ဖြတ်ထုတ်ခွင့် အများဆုံး %",
     "options": [(0, "လုံးဝ မဖြတ်"), (5, "5%"), (8, "8% ⭐"), (15, "15%")]},
    {"key": "chunk", "icon": "📦", "label": "မူရင်းပိုင်းချိန်", "kind": "cycle",
     "desc": "Source ကို download ပြီးတာနဲ့ အမြဲတမ်း ၂ မိနစ်စီခွဲပြီး Script အရင်ရေးကာ အပိုင်းတစ်ခုချင်း Recap ခလုတ်ပြမည်",
     "options": [(SOURCE_SEGMENT_SECONDS, "2 မိနစ် ⭐ (အမြဲတမ်း)")]},
    {"key": "rate", "icon": "🗣️", "label": "Script အလျင် ချိန်ညှိ", "kind": "cycle",
     "desc": "TTS ဖတ်နှုန်း (စာလုံး/စက္ကန့်) — စာသား ပမာဏ တွက်ရန်",
     "options": [(5.5, "5.5 နှေး"), (6.5, "6.5 ⭐"), (7.5, "7.5 မြန်")]},
    {"key": "pov", "icon": "👤", "label": "ပြောပြသူ ရှုထောင့်", "kind": "cycle",
     "desc": "အဓိကဇာတ်ကောင် ကိုယ်တိုင် ပြန်ပြောပြသလို ပထမလူ — အမြဲတမ်း",
     "options": [("first", "ပထမလူ (ကျွန်တော်/ကျွန်မ) ⭐")]},
    {"key": "tone", "icon": "🎭", "label": "ဇာတ်ကြောင်း လေသံ", "kind": "cycle",
     "desc": "Script ရဲ့ ခံစားချက်အရောင်",
     "options": [("natural", "သဘာဝ ⭐"), ("dramatic", "ဒရာမာ ဆန်ဆန်"), ("emotional", "စိတ်လှုပ်ရှားဖွယ်"), ("calm", "အေးအေးဆေးဆေး")]},
    {"key": "split_tg", "icon": "📤", "label": "Telegram ခွဲပို့", "kind": "toggle",
     "desc": "ဖိုင်ကြီးရင် 45MB အပိုင်းတွေ ခွဲပို့ခြင်း"},
    {"key": "caption", "icon": "💬", "label": "Video Caption", "kind": "toggle",
     "desc": "ပို့တဲ့ ဗီဒီယိုမှာ caption စာသား ထည့်ခြင်း"},
    {"key": "notify", "icon": "🔔", "label": "ပြီး အကြောင်းကြား", "kind": "toggle",
     "desc": "အလုပ်ပြီးရင် သီးသန့် အကြောင်းကြား စာပို့ခြင်း"},
    {"key": "fast_mode", "icon": "⚡", "label": "မြန်ဆန်မုဒ်", "kind": "toggle",
     "desc": "Render preset ကို superfast ပြောင်းပြီး မြန်မြန် ထုတ်ခြင်း"},
    {"key": "auto_start", "icon": "🚀", "label": "Video/Link တန်းစ", "kind": "toggle",
     "desc": "Video file သို့မဟုတ် link ပို့လိုက်တာနဲ့ မေးမနေဘဲ တန်းစခြင်း (Off ဆိုရင် အတည်ပြု ခလုတ် ပြမယ်)"},
]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}

SETTINGS_FILE = os.path.join(WORK_ROOT, "bot_settings.json")
_settings_lock = asyncio.Lock()
ACTIVE_JOBS: dict = {}          # chat_id -> job_id
PENDING_LINKS: dict = {}        # legacy chat_id -> url (confirm စောင့်နေတာ)
PENDING_INPUTS: dict = {}       # chat_id -> source spec (link or Telegram video)
JOB_STATS: dict = {}            # chat_id -> {"done":..,"failed":..,"cancelled":..}


def _load_persisted():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("users", {}), data.get("stats", {})
    except Exception:
        return {}, {}


_PERSISTED_USERS, _PERSISTED_STATS = _load_persisted()
JOB_STATS.update({int(k): v for k, v in _PERSISTED_STATS.items()})


def _save_persisted():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump({"users": _PERSISTED_USERS, "stats": JOB_STATS}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"settings save failed: {e}")


def get_user_settings(chat_id) -> dict:
    """User ရဲ့ settings (မရှိရင် default) — အမြဲ မိတ္တူ dict ပြန်ပေးတယ်။"""
    s = dict(DEFAULT_SETTINGS)
    saved = _PERSISTED_USERS.get(str(chat_id), {})
    if isinstance(saved, dict):
        s.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS})
    # 👤 ရေးဟန် = ပထမလူ အမြဲတမ်း — အရင် သိမ်းထားသော ဆက်တင်ဟောင်း ရှိလျှင် ပြန်ညှိသည်။
    if s.get("pov") != "first":
        s["pov"] = "first"
    # Every job/link targets at least 90%, even if a stale user setting
    # saved by an older version contains a lower coverage value.
    return _with_required_coverage(s)


def set_user_setting(chat_id, key: str, value):
    if key not in DEFAULT_SETTINGS:
        return get_user_settings(chat_id)
    _PERSISTED_USERS.setdefault(str(chat_id), {})[key] = value
    _save_persisted()
    return get_user_settings(chat_id)


def reset_user_settings(chat_id):
    _PERSISTED_USERS.pop(str(chat_id), None)
    _save_persisted()
    return get_user_settings(chat_id)


def bump_stat(chat_id, field: str):
    st = JOB_STATS.setdefault(int(chat_id), {"done": 0, "failed": 0, "cancelled": 0})
    st[field] = st.get(field, 0) + 1
    _save_persisted()


def is_allowed(user_id) -> bool:
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return False
    return uid == ADMIN_ID or uid in ALLOWED_USER_IDS


def feature_value_label(feat: dict, settings: dict) -> str:
    val = settings.get(feat["key"], DEFAULT_SETTINGS.get(feat["key"]))
    if feat["kind"] == "toggle":
        return "🟢 On" if val else "🔴 Off"
    for v, label in feat.get("options", []):
        if v == val:
            return label
    return str(val)


def settings_summary_text(chat_id) -> str:
    s = get_user_settings(chat_id)
    lines = ["📋 <b>လက်ရှိ ဆက်တင်များ</b>\n"]
    for f in FEATURES:
        lines.append(f"{f['icon']} {f['label']}: <b>{feature_value_label(f, s)}</b>")
    return "\n".join(lines)


def btn(text: str, data: str) -> dict:
    return {"text": text, "callback_data": data}


def main_menu_keyboard(chat_id) -> dict:
    s = get_user_settings(chat_id)
    rows = [
        [btn("🎬 Recap စတင်မယ်", "menu:recap"), btn("⚙️ ဆက်တင်များ (20)", "menu:settings")],
        [btn(f"🎯 Final: မူရင်း၏ {s['coverage']}%", "feat:coverage"), btn(f"🎙️ အသံ: {s['voice']}", "feat:voice")],
        [btn("📊 အလုပ်အခြေအနေ", "menu:status"), btn("🛑 လုပ်နေတာ ရပ်မယ်", "menu:cancel")],
        [btn("📋 ဆက်တင် အကျဉ်း", "menu:summary"), btn("📈 စာရင်းချုပ်", "menu:stats")],
        [btn("❓ အသုံးပြုနည်း", "menu:help"), btn("🔄 ဆက်တင် ပြန်ချိန်", "menu:reset")],
    ]
    return {"inline_keyboard": rows}


def settings_keyboard(chat_id) -> dict:
    """FEATURES registry ကနေ အလိုအလျောက် တည်ဆောက်တဲ့ ခလုတ်တွေ — feature အသစ်ထည့်ရင် ဒီမှာ တန်းပေါ်တယ်။"""
    s = get_user_settings(chat_id)
    rows, row = [], []
    for f in FEATURES:
        label = f"{f['icon']} {f['label']}"
        if f["kind"] == "toggle":
            label += " 🟢" if s.get(f["key"]) else " 🔴"
        row.append(btn(label, f"feat:{f['key']}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([btn("🏠 ပင်မ မီနူးသို့", "menu:main")])
    return {"inline_keyboard": rows}


def feature_options_keyboard(feat: dict, chat_id) -> dict:
    s = get_user_settings(chat_id)
    cur = s.get(feat["key"])
    rows = []
    for v, label in feat.get("options", []):
        mark = "✅ " if v == cur else ""
        rows.append([btn(f"{mark}{label}", f"set:{feat['key']}:{json.dumps(v)}")])
    rows.append([btn("⬅️ ဆက်တင်များသို့", "menu:settings"), btn("🏠 ပင်မ", "menu:main")])
    return {"inline_keyboard": rows}


def confirm_keyboard() -> dict:
    return {"inline_keyboard": [[
        btn("✅ စတင်မယ်", "job:start"), btn("❌ မလုပ်တော့ဘူး", "job:discard"),
    ]]}


def job_progress_keyboard() -> dict:
    """Buttons that stay attached to the live progress message."""
    return {"inline_keyboard": [
        [btn("🛑 ဒီအလုပ်ကို ရပ်မယ်", "menu:cancel"), btn("📊 အခြေအနေ", "menu:status")],
        [btn("⚙️ ဆက်တင်များ", "menu:settings"), btn("🏠 ပင်မ မီနူး", "menu:main")],
    ]}


def manual_part_keyboard(session_id: str, part_index: int, status: str = "ready") -> dict:
    """Button attached below one original two-minute Telegram video."""
    callback = f"part:recap:{session_id}:{int(part_index)}"
    if status == "rendering":
        rows = [[btn("⏳ Recap လုပ်နေပြီ...", "part:noop")]]
    elif status == "done":
        rows = [[btn("✅ Recap ပြီးပြီ", "part:noop")]]
    elif status == "failed":
        rows = [[btn("🔁 Recap ပြန်လုပ်မယ်", callback)]]
    else:
        rows = [[btn("🎬 Recap လုပ်မယ်", callback)]]
    return {"inline_keyboard": rows}


def manual_control_keyboard(session_id: str) -> dict:
    return {"inline_keyboard": [
        [btn("🔗 Recap အားလုံး ပြန်ဆက်မယ်", f"part:join:{session_id}")],
        [
            btn("📊 အပိုင်းအခြေအနေ", f"part:status:{session_id}"),
            btn("🗑️ Session ပယ်မယ်", f"part:discard:{session_id}"),
        ],
    ]}


PERSISTENT_BUTTON_RECAP = "🎬 Recap စတင်မယ်"
PERSISTENT_BUTTON_SETTINGS = "⚙️ ဆက်တင်များ"
PERSISTENT_BUTTON_STATUS = "📊 အလုပ်အခြေအနေ"
PERSISTENT_BUTTON_CANCEL = "🛑 လုပ်နေတာ ရပ်မယ်"
PERSISTENT_BUTTON_SUMMARY = "📋 ဆက်တင် အကျဉ်း"
PERSISTENT_BUTTON_HELP = "❓ အသုံးပြုနည်း"


def persistent_menu_keyboard() -> dict:
    """Telegram reply keyboard shown persistently under the chat input."""
    return {
        "keyboard": [
            [
                {"text": PERSISTENT_BUTTON_RECAP},
                {"text": PERSISTENT_BUTTON_SETTINGS},
            ],
            [
                {"text": PERSISTENT_BUTTON_STATUS},
                {"text": PERSISTENT_BUTTON_CANCEL},
            ],
            [
                {"text": PERSISTENT_BUTTON_SUMMARY},
                {"text": PERSISTENT_BUTTON_HELP},
            ],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
        "input_field_placeholder": "ဗီဒီယို သို့မဟုတ် link ပို့ပါ…",
    }


MAIN_MENU_TEXT = (
    "🎬 <b>RedNote Recap Bot</b>\n"
    "━━━━━━━━━━━━━━━\n"
    "Video file သို့မဟုတ် RedNote/video link ပို့လိုက်တာနဲ့ မြန်မာ Recap "
    "ဗီဒီယို လုပ်ပေးပါတယ်။\n\n"
    "👇 ခလုတ်လေးတွေ နှိပ်ပြီး သုံးနိုင်ပါတယ်\n"
    "🎬 <b>Recap စတင်မယ်</b> — video/link ပို့ဖို့\n"
    "⚙️ <b>ဆက်တင်များ</b> — feature ၂၀ လုံး ချိန်ညှိရန်\n"
    "🛑 <b>ရပ်မယ်</b> — လုပ်နေတဲ့ အလုပ် ပယ်ဖျက်ရန်"
)


def start_features_text(chat_id=None) -> str:
    """Full plain-language explanation shown when the user presses /start.

    This is generated from the actual FEATURES registry and the user's
    current settings, so /start describes what this deployed bot really
    does (and the values currently selected), rather than a stale manual
    list copied into a welcome message.
    """
    try:
        settings = get_user_settings(chat_id) if chat_id is not None else dict(DEFAULT_SETTINGS)
    except Exception:
        settings = dict(DEFAULT_SETTINGS)
    coverage = settings.get("coverage", MINIMUM_SCRIPT_COVERAGE_PERCENT)
    version = globals().get("BUILD_VERSION", "")
    feature_lines = []
    for index, feat in enumerate(FEATURES, start=1):
        feature_lines.append(
            f"{index:02d}. {feat['icon']} {feat['label']}: "
            f"<b>{feature_value_label(feat, settings)}</b>"
        )
    version_line = f"\n🆕 Version: <code>{version}</code>" if version else ""
    auto_line = (
        "Video/Link ပို့တာနဲ့ တန်းစပါမယ်။"
        if settings.get("auto_start")
        else "Video/Link ပို့ပြီးရင် ✅ စတင်မယ် ခလုတ်အရင် ပြပါမယ်။"
    )
    return (
        "🎬 <b>RedNote Recap Bot — လုပ်ဆောင်ချက်များ</b>\n"
        "━━━━━━━━━━━━━━━\n\n"
        "<b>📥 ပို့နိုင်တာတွေ</b>\n"
        "• Telegram ထဲ တိုက်ရိုက်ပို့တဲ့ <b>video file</b> "
        "(Telegram Bot download ကန့်သတ်ချက် <b>20MB အောက်</b>)\n"
        "• Document အဖြစ် ပို့တဲ့ video file\n"
        "• <b>RedNote link</b>၊ share စာသားထဲက link နှင့် အခြား video link\n\n"
        "<b>⚙️ Bot လုပ်မယ့် အစီအစဉ်</b>\n"
        "1️⃣ Source video အပြည့်အစုံကို အရင် download လုပ်မယ်။ "
        "20MB ထက်ကြီးသော file ကို link အနေနဲ့ ပို့ပါ။\n"
        "2️⃣ ဘယ်လောက်ရှည်ရှည် <b>၂ မိနစ်စီ</b> ခွဲမယ်။ "
        "ဥပမာ ၃ နာရီ = ၉၀ ပိုင်း၊ နောက်ဆုံးပိုင်းက ကျန်ချိန်အတိုင်းပဲ။\n"
        "3️⃣ တစ်ခန်းလုံးရဲ့ <b>Script ကို အရင်ရေးပြီးသိမ်းမယ်</b>။ အကျဉ်းချုပ်"
        "မဟုတ်ဘဲ အဓိကဇာတ်ကောင် ကိုယ်တိုင် <b>«ကျွန်တော်/ကျွန်မ»</b> အဖြစ် "
        "အချိန်လိုက် ပြန်ပြောမယ်။ ဇာတ်ကောင်နာမည်တွေ၊ ရှေ့အကျဉ်းနဲ့ နောက်ဆုံးမြင်ကွင်းကို "
        "အပိုင်းတိုင်း ဆက်သယ်သွားမယ်။\n"
        "4️⃣ ပြီးရင် မူရင်း ၂ မိနစ်အပိုင်းတွေကို တစ်ပိုင်းချင်း ပြန်ပို့ပြီး "
        "အောက်မှာ <b>🎬 Recap လုပ်မယ်</b> ခလုတ်ထားပေးမယ်။ Auto မ Run ဘူး။\n"
        f"5️⃣ ခလုတ်နှိပ်မှသာ အဲဒီအပိုင်းကိုပဲ သိမ်းထားတဲ့ Script နဲ့ recap လုပ်မယ်။ "
        f"TTS အသံကို အမှန်တိုင်းပြီး အပိုင်းတစ်ခုချင်း ကြာချိန်ကို မူရင်းအပိုင်း၏ "
        f"<b>{coverage}%</b> ဖြစ်အောင် ညှိမယ်။ အသံကို stretch/tempo မလုပ်ဘူး။\n"
        "6️⃣ အပိုင်းအားလုံး Recap ပြီးမှ <b>🔗 ပြန်ဆက်မယ်</b> ခလုတ်ကို နှိပ်ရင် "
        "Recap အားလုံးကို အစဉ်အတိုင်း ပြန်ဆက်ပြီး final video ပို့မယ်။ File ကြီးရင် Telegram အတွက် ခွဲပို့မယ်။\n\n"
        "<b>⌨️ ခလုတ် / Command တွေ</b>\n"
        "🎬 Recap စတင်မယ် · ⚙️ ဆက်တင်များ · 📊 အခြေအနေ · "
        "🛑 ရပ်မယ် · 📋 အကျဉ်း · ❓ အကူအညီ\n"
        "/start · /menu · /settings · /status · /cancel · /summary · /help\n"
        f"🚀 {auto_line}\n\n"
        "<b>🎛️ လက်ရှိ Feature ၂၀ နဲ့ တန်ဖိုးများ</b>\n"
        + "\n".join(feature_lines)
        + version_line
        + "\n\n🎬 စဖို့အတွက် video file ဒါမှမဟုတ် link ကို ဒီအတိုင်း ပို့လိုက်ပါ။"
    )


HELP_TEXT = (
    "❓ <b>အသုံးပြုနည်း</b>\n"
    "━━━━━━━━━━━━━━━\n"
    "1️⃣ Telegram video file (20MB အောက်)၊ document video သို့မဟုတ် "
    "RedNote/video link ပို့ပါ\n"
    "2️⃣ Bot က source အပြည့်ဒေါင်း → ၂ မိနစ်စီခွဲ → တစ်ခန်းလုံးရဲ့ ပထမလူ "
    "Script အရင်ရေး → မူရင်းအပိုင်းတွေကို 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ ပြန်ပို့ပါမယ်\n"
    "3️⃣ Recap လိုချင်တဲ့ အပိုင်းအောက်က ခလုတ်ကို နှိပ်ပါ။ အဲဒီအပိုင်းကိုပဲ လုပ်ပေးပါမယ်\n"
    "4️⃣ အပိုင်းတိုင်း ပြီးရင် 🔗 Recap အားလုံး ပြန်ဆက်မယ် ကို နှိပ်၍ Final ရယူနိုင်ပါတယ်\n\n"
    "⌨️ <b>Commands:</b>\n"
    "/start သို့မဟုတ် /menu — လုပ်ဆောင်ချက် အပြည့်အစုံ + ပင်မ မီနူး\n"
    "/settings — ဆက်တင်များ\n"
    "/status — အလုပ်အခြေအနေ\n"
    "/cancel — လုပ်နေတာ ရပ်ရန်\n"
    "/summary — ဆက်တင်အကျဉ်း\n"
    "/help — ဒီစာမျက်နှာ\n\n"
    "💡 ဆက်တင်တွေ ပြောင်းထားရင် နောက်ဗီဒီယိုက စပြီး အကျိုးသက်ရောက်ပါမယ်။"
)


def fmt_progress_bar(pct: int) -> str:
    pct = max(0, min(100, pct))
    filled = int(round(pct / 10))
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_cmd(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        out, err = proc.communicate()
        return subprocess.CompletedProcess(cmd, -9, out, err)


class ProgressTracker:
    def __init__(self, chat_id: int, message_id: int, job_id: str = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.job_id = job_id
        self.last_ts = 0.0
        self.last_text = ""
        self.anim_frame = 0

    async def update(self, text: str, force: bool = False):
        if not self.message_id or text == self.last_text:
            return
        now = time.monotonic()
        if not force and (now - self.last_ts) < 3.0:
            return
        self.last_ts = now
        self.last_text = text
        self.anim_frame += 1
        animated_text = f"{get_anim_emoji(self.anim_frame)} {text}"
        try:
            await tg_api("editMessageText", {
                "chat_id": self.chat_id,
                "message_id": self.message_id,
                "text": animated_text,
                "reply_markup": job_progress_keyboard(),
            })
        except Exception:
            pass


async def tg_api(method: str, data: dict = None, timeout: float = 90.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                resp = await client.post(url, json=data or {})
                return resp.json()
        except Exception:
            await asyncio.sleep(1.5)
    return {"ok": False}


async def send_text(chat_id: int, text: str, reply_to: int = None, reply_markup: dict = None):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "reply_to_message_id": reply_to,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    res = await tg_api("sendMessage", payload)
    if not res.get("ok"):
        # HTML အမှားကြောင့် ပျက်ရင် plain text နဲ့ ပြန်ပို့ (မီနူး မပျောက်အောင်)
        payload.pop("parse_mode", None)
        res = await tg_api("sendMessage", payload)
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str, reply_markup: dict = None):
    if not message_id:
        return
    try:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        res = await tg_api("editMessageText", payload)
        if not (res or {}).get("ok"):
            payload.pop("parse_mode", None)
            await tg_api("editMessageText", payload)
    except Exception:
        pass


async def edit_reply_markup(chat_id: int, message_id: int, reply_markup: dict = None):
    if not message_id:
        return
    try:
        await tg_api(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup or {"inline_keyboard": []},
            },
        )
    except Exception:
        pass


async def answer_callback(callback_id: str, text: str = ""):
    try:
        await tg_api("answerCallbackQuery", {
            "callback_query_id": callback_id,
            "text": text,
            "show_alert": False,
        })
    except Exception:
        pass


async def ensure_persistent_keyboard(
    chat_id,
    reply_to: int = None,
    *,
    force: bool = False,
) -> bool:
    """Install Telegram's bottom reply keyboard for this chat.

    Telegram only shows a reply keyboard after the bot sends a message
    carrying it; registering commands at startup is not enough for an old
    chat.  Send one small installer message the first time a chat talks to
    a newly deployed process, and again whenever ``/start`` forces it.
    """
    try:
        chat_key = int(chat_id)
    except (TypeError, ValueError):
        return False
    if not force and chat_key in PERSISTENT_KEYBOARD_CHATS:
        return False

    PERSISTENT_KEYBOARD_CHATS.add(chat_key)
    version = globals().get("BUILD_VERSION", "")
    text = "⌨️ အောက်ခြေခလုတ်တွေ အဆင်သင့်ဖြစ်ပါပြီ။ Video file သို့မဟုတ် Link ကို ဒီအတိုင်းပို့လိုက်ရုံပါပဲ။"
    if version:
        text += f"\n🆕 <code>{version}</code>"
    try:
        message_id = await send_text(
            chat_key,
            text,
            reply_to=reply_to,
            reply_markup=persistent_menu_keyboard(),
        )
    except Exception as e:
        PERSISTENT_KEYBOARD_CHATS.discard(chat_key)
        logger.warning(f"persistent keyboard send failed for {chat_key}: {e}")
        return False
    if not message_id:
        PERSISTENT_KEYBOARD_CHATS.discard(chat_key)
        return False
    return True


async def send_main_menu(
    chat_id: int,
    reply_to: int = None,
    with_persistent_keyboard: bool = False,
):
    if with_persistent_keyboard:
        await ensure_persistent_keyboard(
            chat_id, reply_to=reply_to, force=True
        )
    # /start and /menu explain the complete current workflow and all 20
    # currently selected features before the user sends a video/link.
    text = start_features_text(chat_id)
    return await send_text(
        chat_id,
        text,
        reply_to=reply_to,
        reply_markup=main_menu_keyboard(chat_id),
    )


async def tg_upload_media(method: str, data: dict, field_name: str, file_name: str, file_path: str, mime_type: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    payload_bytes = await asyncio.to_thread(lambda: open(file_path, "rb").read())
    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    resp = await client.post(
                        url,
                        data=data,
                        files={field_name: (file_name, payload_bytes, mime_type)},
                    )
                    res = resp.json()
                    if res.get("ok"):
                        return res
            except Exception:
                pass
            await asyncio.sleep(2.5)
        return {"ok": False}
    finally:
        del payload_bytes
        gc.collect()


def _safe_video_extension(file_name: str = "", mime_type: str = "") -> str:
    ext = os.path.splitext(str(file_name or ""))[1].lower()
    if ext in TELEGRAM_VIDEO_EXTENSIONS:
        return ext
    mime = str(mime_type or "").lower()
    mime_extensions = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-m4v": ".m4v",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
        "video/mpeg": ".mpeg",
        "video/3gpp": ".3gp",
        "video/mp2t": ".ts",
    }
    return mime_extensions.get(mime, ".mp4")


def _telegram_video_attachment(msg: dict) -> dict | None:
    """Normalize a video uploaded directly in Telegram.

    Users can send a normal Telegram video, an animation/video note, or a
    video as a document (the latter preserves the original file).  Return
    only safe metadata needed to download it later; never trust a client
    supplied filename as a filesystem path.
    """
    if not isinstance(msg, dict):
        return None

    candidates = []
    for kind in ("video", "animation", "video_note"):
        value = msg.get(kind)
        if isinstance(value, dict) and value.get("file_id"):
            candidates.append((kind, value))

    document = msg.get("document")
    if isinstance(document, dict) and document.get("file_id"):
        mime = str(document.get("mime_type") or "").lower()
        ext = os.path.splitext(str(document.get("file_name") or ""))[1].lower()
        if mime.startswith("video/") or ext in TELEGRAM_VIDEO_EXTENSIONS:
            candidates.append(("document", document))

    if not candidates:
        return None
    kind, value = candidates[0]
    file_name = os.path.basename(str(value.get("file_name") or "telegram-video"))
    return {
        "kind": "telegram_file",
        "telegram_kind": kind,
        "file_id": str(value.get("file_id") or ""),
        "file_unique_id": str(value.get("file_unique_id") or ""),
        "file_name": file_name,
        "mime_type": str(value.get("mime_type") or ""),
        "file_size": int(value.get("file_size") or 0),
        "duration": float(value.get("duration") or 0),
        "extension": _safe_video_extension(
            file_name, str(value.get("mime_type") or "")
        ),
    }


def _telegram_attachment_label(attachment: dict) -> str:
    name = str((attachment or {}).get("file_name") or "Telegram video")
    size = int((attachment or {}).get("file_size") or 0)
    duration = float((attachment or {}).get("duration") or 0)
    details = []
    if duration > 0:
        details.append(fmt_time(duration))
    if size > 0:
        details.append(f"{size / (1024 * 1024):.1f}MB")
    return name + (f" ({', '.join(details)})" if details else "")


def _extract_first_url(text: str) -> str:
    """Extract the first HTTP(S) URL from a bare link or share caption."""
    match = re.search(r"https?://[^\s<>\"'“”‘’]+", str(text or ""), re.IGNORECASE)
    if not match:
        return ""
    value = match.group(0).strip()
    # Share text commonly wraps links in punctuation/brackets.
    while value and value[-1] in ")]}>.,!?;:'\"":
        # Do not strip a balanced closing parenthesis that belongs to a
        # Wikipedia-style URL; share links used here do not need one.
        value = value[:-1]
    return value


def _url_hostname(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower().strip()
    except Exception:
        return ""


def _is_rednote_url(url: str) -> bool:
    host = _url_hostname(url)
    return bool(
        host == "xiaohongshu.com"
        or host.endswith(".xiaohongshu.com")
        or host == "rednote.com"
        or host.endswith(".rednote.com")
        or host == "xhslink.com"
        or host.endswith(".xhslink.com")
        or host == "xhslink.cn"
        or host.endswith(".xhslink.cn")
        or host.endswith(".xhscdn.com")
    )


def _is_direct_media_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or ""))
        path = (parsed.path or "").lower()
        host = (parsed.hostname or "").lower()
    except Exception:
        return False
    return bool(
        path.endswith(tuple(TELEGRAM_VIDEO_EXTENSIONS))
        or path.endswith(".m3u8")
        or host.endswith(".xhscdn.com")
        or host.endswith("googlevideo.com")
        or "sns-video" in host
    )


def _configured_cookie_file_path() -> str:
    configured = str(globals().get("YTDLP_COOKIES_FILE") or "").strip()
    candidates = [configured] if configured else []
    candidates.extend(
        ["/etc/secrets/yt-dlp-cookies.txt", "/etc/secrets/cookies.txt"]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def _normalise_cookie_domain(domain: str = "") -> str:
    value = str(domain or "").strip().lower()
    if not value or "xiaohongshu.com" in value or "rednote.com" in value or "xhslink" in value:
        return ".xiaohongshu.com"
    if "xhscdn.com" in value:
        return ".xhscdn.com"
    return value if value.startswith(".") or "." in value else ".xiaohongshu.com"


def _cookie_records_from_value(value) -> list[tuple[str, str, str, str]]:
    raw = str(value or "").strip()
    if not raw:
        return []

    if raw[:1] in ("[", "{"):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if data is not None:
            if isinstance(data, dict) and isinstance(data.get("cookies"), list):
                items = data["cookies"]
            elif isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                return [
                    (str(name), str(cookie_value), ".xiaohongshu.com", "/")
                    for name, cookie_value in data.items()
                    if str(name).strip() and not isinstance(cookie_value, (dict, list))
                ]
            else:
                items = []
            records = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("key") or "").strip()
                cookie_value = item.get("value")
                if name and cookie_value is not None:
                    records.append(
                        (
                            name,
                            str(cookie_value),
                            _normalise_cookie_domain(str(item.get("domain") or "")),
                            str(item.get("path") or "/") or "/",
                        )
                    )
            if records:
                return records

    records = []
    for line in raw.splitlines():
        clean = line.strip()
        if clean.startswith("#HttpOnly_"):
            clean = clean[len("#HttpOnly_") :]
        elif not clean or clean.startswith("#"):
            continue
        fields = clean.split("\t")
        if len(fields) < 7:
            continue
        name = str(fields[5] or "").strip()
        if name:
            records.append(
                (
                    name,
                    "\t".join(fields[6:]),
                    _normalise_cookie_domain(fields[0]),
                    str(fields[2] or "/") or "/",
                )
            )
    if records:
        return records

    return [
        (name, cookie_value, ".xiaohongshu.com", "/")
        for name, cookie_value in _parse_cookie_header(raw).items()
    ]


def _source_cookie_records() -> list[tuple[str, str, str, str]]:
    records = _cookie_records_from_value(globals().get("XHS_COOKIE") or "")
    if records:
        return records
    cookie_file = _configured_cookie_file_path()
    if not cookie_file:
        return []
    try:
        with open(cookie_file, "r", encoding="utf-8", errors="ignore") as fh:
            return _cookie_records_from_value(fh.read(1024 * 1024))
    except OSError:
        return []


def _yt_dlp_cookies_file() -> str:
    configured_file = _configured_cookie_file_path()
    if configured_file:
        return configured_file
    records = _cookie_records_from_value(globals().get("XHS_COOKIE") or "")
    if not records:
        return ""
    lines = ["# Netscape HTTP Cookie File"]
    for name, cookie_value, domain, path in records:
        safe_name = str(name).replace("\t", "").replace("\r", "").replace("\n", "")
        safe_value = (
            str(cookie_value).replace("\t", "").replace("\r", "").replace("\n", "")
        )
        if not safe_name:
            continue
        lines.append(
            "\t".join(
                [
                    domain,
                    "TRUE" if domain.startswith(".") else "FALSE",
                    path or "/",
                    "FALSE",
                    "0",
                    safe_name,
                    safe_value,
                ]
            )
        )
    if len(lines) <= 1:
        return ""
    content = "\n".join(lines) + "\n"
    target = os.path.join(str(globals().get("WORK_ROOT") or "/tmp"), ".yt-dlp-cookies.txt")
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        existing = ""
        if os.path.isfile(target):
            with open(target, "r", encoding="utf-8", errors="ignore") as fh:
                existing = fh.read()
        if existing != content:
            temp_target = f"{target}.tmp"
            with open(temp_target, "w", encoding="utf-8") as fh:
                fh.write(content)
            try:
                os.chmod(temp_target, 0o600)
            except OSError:
                pass
            os.replace(temp_target, target)
        return target
    except OSError:
        return ""


def _rednote_cookie_configured() -> bool:
    return bool(
        str(globals().get("XHS_COOKIE") or "").strip()
        or _configured_cookie_file_path()
    )


def _parse_cookie_header(value: str) -> dict:
    cookies = {}
    for item in str(value or "").split(";"):
        if "=" not in item:
            continue
        name, cookie_value = item.split("=", 1)
        name = name.strip()
        if name:
            cookies[name] = cookie_value.strip()
    return cookies


def _source_browser_headers(url: str = "", referer: str = "") -> dict:
    headers = {
        "User-Agent": YTDLP_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,my;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    elif _is_rednote_url(url):
        headers["Referer"] = "https://www.xiaohongshu.com/"
    return headers


def _rednote_httpx_cookies() -> httpx.Cookies:
    jar = httpx.Cookies()
    for name, value, domain, path in _source_cookie_records():
        # xhslink starts on another host and redirects to Xiaohongshu.
        # Bind configured cookies explicitly to the destination domain so
        # they survive that redirect instead of being attached only to the
        # short-link host.
        jar.set(name, value, domain=domain, path=path or "/")
    return jar


async def download_telegram_file(
    file_id: str,
    out_path: str,
    tracker=None,
    *,
    expected_size: int = 0,
    raise_on_error: bool = False,
) -> float:
    """Download an uploaded Telegram file to disk and return its duration.

    The public Telegram Bot API can only download files up to 20MB through
    ``getFile``.  Previously every failure here collapsed to ``0`` and the
    user only saw the generic “source download failed”.  When
    ``raise_on_error`` is set, return the actionable reason instead.
    """

    def fail(message: str, detail: str = "") -> float:
        logger.warning("Telegram file download failed: %s", _safe_error_text(detail or message))
        if raise_on_error:
            raise SourceDownloadError(message, detail=detail or message, user_message=message)
        return 0.0

    try:
        known_size = int(expected_size or 0)
    except (TypeError, ValueError):
        known_size = 0
    if known_size > TELEGRAM_BOT_FILE_DOWNLOAD_LIMIT_BYTES:
        size_mb = known_size / (1024 * 1024)
        return fail(
            "Telegram video file က 20MB ထက်ကြီးလို့ Bot က တိုက်ရိုက် download မလုပ်နိုင်ပါ။ "
            f"ပို့ထားတဲ့ file က {size_mb:.1f}MB ပါ။ Video link အနေနဲ့ပို့ပါ၊ သို့မဟုတ် "
            "20MB အောက် compress/split လုပ်၍ ပြန်ပို့ပါ။",
            f"Telegram Bot API getFile limit exceeded: {known_size} bytes",
        )
    if not file_id:
        return fail("Telegram file_id မရပါ။ Video ကို ပြန်ပို့ကြည့်ပါ။")
    if not BOT_TOKEN:
        return fail("BOT_TOKEN မသတ်မှတ်ရသေးလို့ Telegram file download မလုပ်နိုင်ပါ။")

    result = await tg_api("getFile", {"file_id": file_id}, timeout=120.0)
    if not (result or {}).get("ok"):
        description = _safe_error_text((result or {}).get("description") or "unknown error")
        if "file is too big" in description.lower() or "too big" in description.lower():
            return fail(
                "Telegram video file က 20MB ထက်ကြီးလို့ Bot က တိုက်ရိုက် download မလုပ်နိုင်ပါ။ "
                "Video link အနေနဲ့ပို့ပါ၊ သို့မဟုတ် 20MB အောက် compress/split လုပ်၍ ပြန်ပို့ပါ။",
                description,
            )
        return fail(
            "Telegram file ကို download လုပ်ဖို့ getFile မအောင်မြင်ပါ။ Video ကို ပြန်ပို့ကြည့်ပါ။",
            description,
        )
    result_data = (result or {}).get("result") or {}
    file_path = str(result_data.get("file_path") or "")
    try:
        remote_size = int(result_data.get("file_size") or 0)
    except (TypeError, ValueError):
        remote_size = 0
    if remote_size > TELEGRAM_BOT_FILE_DOWNLOAD_LIMIT_BYTES:
        return fail(
            "Telegram video file က 20MB ထက်ကြီးလို့ Bot က တိုက်ရိုက် download မလုပ်နိုင်ပါ။ "
            f"File အရွယ်က {remote_size / (1024 * 1024):.1f}MB ပါ။ Video link အနေနဲ့ပို့ပါ။",
            f"Telegram Bot API getFile limit exceeded: {remote_size} bytes",
        )
    if not file_path:
        return fail("Telegram getFile က file path မပြန်ပေးပါ။ Video ကို ပြန်ပို့ကြည့်ပါ။")

    # Never log this URL: it contains the bot token.
    download_url = (
        f"https://api.telegram.org/file/bot{BOT_TOKEN}/"
        f"{quote(file_path.lstrip('/'), safe='/')}"
    )
    downloaded = 0
    last_report = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=1800.0, follow_redirects=True) as client:
            async with client.stream("GET", download_url) as response:
                status_code = int(getattr(response, "status_code", 0) or 0)
                if status_code >= 400:
                    if known_size > TELEGRAM_BOT_FILE_DOWNLOAD_LIMIT_BYTES or status_code == 400:
                        detail = f"Telegram file download HTTP {status_code}; likely over the 20MB Bot API limit"
                    else:
                        detail = f"Telegram file download HTTP {status_code}"
                    return fail(
                        "Telegram file download မအောင်မြင်ပါ။ File က 20MB ထက်ကြီးရင် link အနေနဲ့ပို့ပါ။",
                        detail,
                    )
                with open(out_path, "wb") as fh:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if tracker and time.monotonic() - last_report >= 4.0:
                            last_report = time.monotonic()
                            await tracker.update(
                                "📥 Telegram video ကို လက်ခံနေပါသည်...\n"
                                f"📦 {downloaded / (1024 * 1024):.1f} MB",
                                force=True,
                            )
    except SourceDownloadError:
        raise
    except Exception as e:
        return fail(
            "Telegram file download ကြားဖြတ်ပျက်သွားပါသည်။ ခဏနေမှ ပြန်ပို့ကြည့်ပါ။",
            f"{type(e).__name__}: {_safe_error_text(e)}",
        )

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        return fail(
            "Telegram က ရလာတဲ့ video file က အလွန်သေး/မပြည့်စုံလို့ ဖတ်မရပါ။ ပြန်ပို့ကြည့်ပါ။",
            f"downloaded Telegram file is only {size} bytes",
        )
    duration = float(probe_video_info(out_path)[0] or 0)
    if duration <= 0:
        return fail(
            "Telegram file ကို download ရသော်လည်း video duration ဖတ်မရပါ။ "
            "File ပျက်နေခြင်း သို့မဟုတ် codec/container မထောက်ပံ့ခြင်း ဖြစ်နိုင်ပါတယ်။ MP4 အဖြစ် ပြန်ပို့ကြည့်ပါ။",
            f"ffprobe/ffmpeg could not read a duration from {os.path.getsize(out_path)} bytes",
        )
    return duration


def probe_video_info(path: str) -> tuple[float, int, int]:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        dur = float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    w, h = 0, 0
    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d{2,5})x(\d{2,5})", out)
    if vm:
        w, h = int(vm.group(2)), int(vm.group(3))
    return dur, w, h


def compute_output_dims(src_w: int, src_h: int, long_edge_target: int = 1280) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1280, 720
    long_edge = max(src_w, src_h)
    scale = float(long_edge_target) / float(long_edge)
    out_w = max(2, int(round(src_w * scale / 2.0)) * 2)
    out_h = max(2, int(round(src_h * scale / 2.0)) * 2)
    return out_w, out_h


def _yt_dlp_command_prefix() -> list[str]:
    try:
        if importlib.util.find_spec("yt_dlp") is not None:
            return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass
    return ["yt-dlp"]


def _installed_yt_dlp_version() -> str:
    try:
        return importlib.metadata.version("yt-dlp")
    except Exception:
        return "unknown"


def _yt_dlp_impersonation_available() -> bool:
    try:
        return importlib.util.find_spec("curl_cffi") is not None
    except Exception:
        return False


def _yt_dlp_base_args(url: str, *, impersonate: bool = False) -> list[str]:
    args = [
        "--no-playlist",
        "--no-warnings",
        "--socket-timeout",
        "30",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--user-agent",
        YTDLP_USER_AGENT,
    ]
    cookies_file = _yt_dlp_cookies_file()
    if cookies_file:
        args.extend(["--cookies", cookies_file])
    if _is_rednote_url(url):
        args.extend(["--referer", "https://www.xiaohongshu.com/"])
    if impersonate and _yt_dlp_impersonation_available():
        args.extend(["--impersonate", "chrome"])
    return args


def _download_error_from_detail(detail: str, url: str = "") -> SourceDownloadError:
    raw_lines = [
        _safe_error_text(line, 700)
        for line in str(detail or "").splitlines()
        if str(line or "").strip()
    ]
    error_lines = [
        line
        for line in raw_lines
        if "error" in line.lower() or "unable" in line.lower() or "failed" in line.lower()
    ]
    concise = (error_lines[-1] if error_lines else (raw_lines[-1] if raw_lines else "unknown download error"))
    concise = re.sub(r"^ERROR:\s*", "", concise, flags=re.IGNORECASE).strip()
    lowered = concise.lower()
    if "unsupported url" in lowered or "no suitable extractor" in lowered:
        user_message = (
            "ဒီ link ကို download extractor က မဖတ်နိုင်ပါ။ "
            "Video page link အပြည့်အစုံ သို့မဟုတ် direct video link ကို ပြန်ပို့ကြည့်ပါ။"
        )
    elif any(
        token in lowered
        for token in ("http error 403", "403 forbidden", "forbidden", "captcha", "blocked")
    ):
        user_message = (
            "Website က server download ကို ပိတ်ထားပါတယ် (403/anti-bot)။ "
            "Public link အသစ်ပြန်ကူးပို့ကြည့်ပါ။ RedNote ဆိုရင် cookie လိုအပ်နိုင်ပါတယ်။"
        )
    elif any(
        token in lowered
        for token in (
            "login",
            "sign in",
            "cookies",
            "private",
            "http error 401",
            "unauthorized",
            "authentication",
        )
    ):
        user_message = (
            "ဒီ link က private ဖြစ်နေ၍ဖြစ်စေ login/cookie လိုအပ်၍ဖြစ်စေ download မရပါ။ "
            "Public ဖွင့်ထားတဲ့ link ကို ပြန်ပို့ကြည့်ပါ။"
        )
    elif any(
        token in lowered
        for token in ("requested format is not available", "no video formats", "no formats")
    ):
        user_message = "ဒီ link မှာ download လုပ်နိုင်တဲ့ video format မတွေ့ပါ။"
    elif any(
        token in lowered
        for token in ("timed out", "timeout", "connection", "network", "name resolution", "temporary failure")
    ):
        user_message = "Download server/network ကြောင့် အချိန်ကုန်သွားပါတယ်။ ခဏနေမှ ပြန်ပို့ကြည့်ပါ။"
    elif any(
        token in lowered
        for token in ("http error 404", "not found", "removed", "deleted", "unavailable")
    ):
        user_message = "ဒီ video link က မရှိတော့/ဖျက်ခံရ/သက်တမ်းကုန်နေပါတယ်။ Link အသစ်ပြန်ကူးပို့ပါ။"
    else:
        user_message = "မူရင်းဗီဒီယို download မရပါ။"
    return SourceDownloadError(user_message, detail=concise, user_message=user_message)


async def _resolve_source_url(url: str) -> str:
    original = _extract_first_url(url) or str(url or "").strip()
    if not original.startswith(("http://", "https://")):
        return original
    host = _url_hostname(original)
    should_resolve = _is_rednote_url(original) or host in {
        "bit.ly",
        "t.co",
        "tinyurl.com",
        "shorturl.at",
    }
    if not should_resolve:
        return original
    try:
        cookies = _rednote_httpx_cookies() if _is_rednote_url(original) else None
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers=_source_browser_headers(original),
            cookies=cookies,
        ) as client:
            async with client.stream("GET", original) as response:
                resolved = str(response.url or original)
        return resolved or original
    except Exception as e:
        logger.warning(
            "source URL resolve failed for host=%s: %s",
            host or "unknown",
            _safe_error_text(e),
        )
        return original


def _source_url_candidates(original: str, resolved: str) -> list[str]:
    candidates: list[str] = []

    def add(value: str):
        clean = _extract_first_url(value) or str(value or "").strip()
        if clean.startswith(("http://", "https://")) and clean not in candidates:
            candidates.append(clean)

    add(resolved)
    add(original)
    for value in list(candidates):
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        if host == "rednote.com" or host.endswith(".rednote.com"):
            add(parsed._replace(netloc="www.xiaohongshu.com").geturl())
    return candidates


def _extract_direct_video_urls_from_text(text: str, base_url: str = "") -> list[str]:
    raw = html.unescape(str(text or ""))
    raw = raw.replace("\\/", "/")
    raw = re.sub(r"\\u002[fF]", "/", raw)
    raw = re.sub(r"\\u0026", "&", raw)
    candidates: list[str] = []

    def add(value: str):
        clean = str(value or "").strip().strip("\"'")
        if not clean:
            return
        clean = urljoin(base_url or "", clean)
        if clean.startswith(("http://", "https://")) and clean not in candidates:
            candidates.append(clean)

    for match in re.finditer(
        r"(?:masterUrl|master_url|videoUrl|video_url|playUrl|play_url)"
        r"[\"']?\s*[:=]\s*[\"']([^\"']+)",
        raw,
        re.IGNORECASE,
    ):
        add(match.group(1))
    for match in re.finditer(
        r"originVideoKey[\"']?\s*[:=]\s*[\"']([^\"']+)", raw, re.IGNORECASE
    ):
        key = str(match.group(1) or "").lstrip("/")
        if key:
            add(f"https://sns-video-bd.xhscdn.com/{key}")
    media_suffixes = tuple(TELEGRAM_VIDEO_EXTENSIONS) + (".m3u8",)
    for match in re.finditer(r"https?://[^\s\"'<>\\]+", raw, re.IGNORECASE):
        value = match.group(0).rstrip("),.;")
        path = (urlparse(value).path or "").lower()
        if path.endswith(media_suffixes) or "sns-video" in value.lower():
            add(value)
    return candidates[:10]


async def _extract_rednote_direct_urls(url: str) -> list[str]:
    if not _is_rednote_url(url):
        return []
    try:
        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers=_source_browser_headers(url),
            cookies=_rednote_httpx_cookies(),
        ) as client:
            async with client.stream("GET", url) as response:
                if int(getattr(response, "status_code", 0) or 0) >= 400:
                    logger.warning(
                        "RedNote page fallback HTTP %s host=%s",
                        response.status_code,
                        _url_hostname(url),
                    )
                    return []
                final_url = str(response.url or url)
                content_type = str(response.headers.get("content-type") or "").lower()
                if content_type.startswith("video/"):
                    return [final_url]
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes(1024 * 256):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= 4 * 1024 * 1024:
                        break
        text = b"".join(chunks).decode(errors="ignore")
        return _extract_direct_video_urls_from_text(text, final_url)
    except Exception as e:
        logger.warning(
            "RedNote direct URL extraction failed host=%s: %s",
            _url_hostname(url),
            _safe_error_text(e),
        )
        return []


async def _download_direct_media(
    url: str,
    out_path: str,
    tracker=None,
    total_dur: float = 0,
    *,
    referer: str = "",
) -> float:
    clean_url = _extract_first_url(url) or str(url or "").strip()
    path = (urlparse(clean_url).path or "").lower()
    if path.endswith(".m3u8"):
        raise SourceDownloadError(
            "HLS playlist ကို yt-dlp ဖြင့် download လုပ်ရန် လိုအပ်ပါတယ်။",
            detail="direct downloader does not handle .m3u8 playlists",
        )
    downloaded = 0
    last_report = time.monotonic()
    content_type = ""
    try:
        cookies = _rednote_httpx_cookies() if _is_rednote_url(clean_url) else None
        async with httpx.AsyncClient(
            timeout=1800.0,
            follow_redirects=True,
            headers=_source_browser_headers(clean_url, referer=referer),
            cookies=cookies,
        ) as client:
            async with client.stream("GET", clean_url) as response:
                status_code = int(getattr(response, "status_code", 0) or 0)
                if status_code >= 400:
                    raise _download_error_from_detail(
                        f"HTTP Error {status_code} while downloading direct media",
                        clean_url,
                    )
                content_type = str(response.headers.get("content-type") or "").lower()
                is_known_media = bool(
                    content_type.startswith("video/")
                    or "octet-stream" in content_type
                    or _is_direct_media_url(clean_url)
                    or "sns-video" in clean_url.lower()
                )
                if content_type and not is_known_media:
                    raise SourceDownloadError(
                        "Link က video file မဟုတ်ဘဲ web page ပြန်ပေးနေပါတယ်။",
                        detail=f"direct URL returned content-type {content_type}",
                    )
                try:
                    content_length = int(response.headers.get("content-length") or 0)
                except (TypeError, ValueError):
                    content_length = 0
                with open(out_path, "wb") as fh:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if tracker and time.monotonic() - last_report >= 4.0:
                            last_report = time.monotonic()
                            pct = (
                                min(95, int(downloaded * 100 / content_length))
                                if content_length > downloaded
                                else min(95, int(downloaded / (10 * 1024 * 1024)))
                            )
                            await tracker.update(
                                "📥 မူရင်းဗီဒီယို ဒေါင်းနေသည်...\n"
                                f"📦 {downloaded / (1024 * 1024):.0f} MB\n"
                                f"📊 {fmt_progress_bar(pct)}",
                                force=True,
                            )
    except SourceDownloadError:
        raise
    except Exception as e:
        raise _download_error_from_detail(
            f"{type(e).__name__}: {_safe_error_text(e)}", clean_url
        ) from e

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        raise SourceDownloadError(
            "Direct video download က မပြည့်စုံပါ။",
            detail=f"direct download produced only {size} bytes; content-type={content_type}",
        )
    duration = float(probe_video_info(out_path)[0] or 0)
    if duration <= 0:
        raise SourceDownloadError(
            "Download ရလာတဲ့ file ကို video အဖြစ် ဖတ်မရပါ။",
            detail=(
                f"direct download produced unreadable media "
                f"({os.path.getsize(out_path)} bytes, content-type={content_type})"
            ),
        )
    return duration


async def _read_stream_tail(stream, max_bytes: int = 65536) -> bytes:
    if stream is None:
        return b""
    buffer = bytearray()
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            del buffer[: len(buffer) - max_bytes]
    return bytes(buffer)


async def _run_ytdlp_info(url: str, *, impersonate: bool = False) -> dict:
    cmd = [*_yt_dlp_command_prefix(), *_yt_dlp_base_args(url, impersonate=impersonate), "-j", url]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        raise SourceDownloadError(
            "yt-dlp downloader မတွေ့ပါ။",
            detail="yt-dlp executable is not installed",
        ) from e
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise _download_error_from_detail("yt-dlp info timed out", url) from None
    stdout_text = (out or b"").decode(errors="ignore")
    stderr_text = (err or b"").decode(errors="ignore")
    if proc.returncode != 0:
        raise _download_error_from_detail(stderr_text or stdout_text, url)
    for line in stdout_text.splitlines():
        clean = line.strip()
        if not clean.startswith("{"):
            continue
        try:
            data = json.loads(clean)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    raise SourceDownloadError(
        "Video အချက်အလက်ကို yt-dlp က မပြန်ပေးပါ။",
        detail=_safe_error_text(stderr_text or stdout_text[-1000:]),
    )


def _duration_from_ytdlp_info(data: dict) -> float:
    candidates = [data] if isinstance(data, dict) else []
    entries = data.get("entries") if isinstance(data, dict) else []
    if isinstance(entries, list):
        candidates.extend(item for item in entries if isinstance(item, dict))
    for item in candidates:
        try:
            duration = float(item.get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            return duration
        formats = item.get("formats") or []
        format_durations = []
        if isinstance(formats, list):
            for fmt in formats:
                if not isinstance(fmt, dict):
                    continue
                try:
                    value = float(fmt.get("duration") or 0)
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0:
                    format_durations.append(value)
        if format_durations:
            return max(format_durations)
    return 0.0


async def get_source_duration(url: str, *, raise_on_error: bool = False) -> float:
    original = _extract_first_url(url) or str(url or "").strip()
    if not original.startswith(("http://", "https://")):
        error = SourceDownloadError(
            "Video link မမှန်ပါ။ http/https link အပြည့်အစုံ ပို့ပါ။",
            detail="source URL is empty or not http/https",
        )
        if raise_on_error:
            raise error
        return 0.0
    resolved = await _resolve_source_url(original)
    candidates = _source_url_candidates(original, resolved)
    attempts: list[tuple[str, bool]] = [(item, False) for item in candidates]
    if candidates and _is_rednote_url(candidates[0]) and _yt_dlp_impersonation_available():
        attempts.append((candidates[0], True))
    errors: list[SourceDownloadError] = []
    for candidate, impersonate in attempts:
        try:
            info = await _run_ytdlp_info(candidate, impersonate=impersonate)
            duration = _duration_from_ytdlp_info(info)
            if duration > 0:
                return duration
            errors.append(
                SourceDownloadError(
                    "Video duration ကို link က မရပါ။",
                    detail="yt-dlp info did not include a duration",
                )
            )
        except SourceDownloadError as e:
            errors.append(e)
        except Exception as e:
            errors.append(
                _download_error_from_detail(
                    f"{type(e).__name__}: {_safe_error_text(e)}", candidate
                )
            )
    if errors:
        logger.warning(
            "get_source_duration failed host=%s: %s",
            _url_hostname(original),
            errors[0].detail,
        )
        if raise_on_error:
            raise errors[0]
    return 0.0


def _normalize_downloaded_video(produced_path: str, out_path: str) -> bool:
    if os.path.abspath(produced_path) == os.path.abspath(out_path):
        return os.path.exists(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
    result = run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            produced_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out_path,
        ],
        1200,
    )
    if result.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            os.remove(produced_path)
        except Exception:
            pass
        return True
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        shutil.move(produced_path, out_path)
        return bool(os.path.exists(out_path) and os.path.getsize(out_path) > 0)
    except Exception:
        return False


async def _download_with_ytdlp_once(
    url: str,
    out_path: str,
    tracker=None,
    total_dur: float = 0,
    *,
    format_selector: str = "bestvideo[height<=720]+bestaudio/best[height<=720]",
    impersonate: bool = False,
) -> float:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(out_path) or "source.mp4"
    stem = os.path.splitext(filename)[0] or "source"
    download_prefix = f"{stem}_download."
    template = os.path.join(out_dir, f"{stem}_download.%(ext)s")

    def current_files() -> list[str]:
        try:
            names = os.listdir(out_dir)
        except Exception:
            return []
        paths = []
        for name in names:
            if not name.startswith(download_prefix):
                continue
            path = os.path.join(out_dir, name)
            try:
                if os.path.isfile(path) and os.path.getsize(path) > 0:
                    paths.append(path)
            except Exception:
                continue
        return sorted(paths, key=lambda p: os.path.getsize(p), reverse=True)

    def cleanup_files(except_path: str = ""):
        for path in current_files():
            if except_path and os.path.abspath(path) == os.path.abspath(except_path):
                continue
            try:
                os.remove(path)
            except Exception:
                pass

    cleanup_files()
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass

    cmd = [
        *_yt_dlp_command_prefix(),
        *_yt_dlp_base_args(url, impersonate=impersonate),
        "--no-part",
        "-f",
        format_selector,
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except FileNotFoundError as e:
        raise SourceDownloadError(
            "yt-dlp downloader မတွေ့ပါ။",
            detail="yt-dlp executable is not installed",
        ) from e
    stdout_task = asyncio.create_task(_read_stream_tail(proc.stdout))
    stderr_task = asyncio.create_task(_read_stream_tail(proc.stderr))
    started = time.monotonic()
    last_report = started
    try:
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
                break
            except asyncio.TimeoutError:
                if time.monotonic() - started > SOURCE_DOWNLOAD_TIMEOUT_SECONDS:
                    raise _download_error_from_detail(
                        "yt-dlp download timed out", url
                    ) from None
                if tracker and time.monotonic() - last_report >= 4.0:
                    last_report = time.monotonic()
                    files = current_files()
                    mb = (
                        os.path.getsize(files[0]) / (1024 * 1024)
                        if files
                        else 0.0
                    )
                    if total_dur > 0:
                        pct = min(95, int((mb / max(1, total_dur)) * 100 / 4))
                    else:
                        pct = min(95, int(mb / 10))
                    await tracker.update(
                        "📥 မူရင်းဗီဒီယို ဒေါင်းနေသည်...\n"
                        f"📦 {mb:.0f} MB\n"
                        f"📊 {fmt_progress_bar(pct)}",
                        force=True,
                    )
    except BaseException:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        stdout_task.cancel()
        stderr_task.cancel()
        raise
    stdout_tail, stderr_tail = await asyncio.gather(stdout_task, stderr_task)
    combined_output = (
        (stderr_tail or b"").decode(errors="ignore")
        + "\n"
        + (stdout_tail or b"").decode(errors="ignore")
    )
    files = current_files()
    if proc.returncode != 0 or not files:
        cleanup_files()
        raise _download_error_from_detail(combined_output, url)

    produced = files[0]
    normalized = await asyncio.to_thread(
        _normalize_downloaded_video, produced, out_path
    )
    cleanup_files(except_path=out_path)
    if not normalized or not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
        size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
        detail = _safe_error_text(combined_output[-1000:])
        raise SourceDownloadError(
            "Download ရလာတဲ့ video file က မပြည့်စုံ/ဖတ်မရပါ။",
            detail=f"yt-dlp produced {size} bytes. {detail}",
        )
    actual_dur = float(probe_video_info(out_path)[0] or 0)
    if actual_dur <= 0:
        raise SourceDownloadError(
            "Download ရလာတဲ့ file ကို video အဖြစ် ဖတ်မရပါ။",
            detail=(
                f"yt-dlp output was {os.path.getsize(out_path)} bytes but "
                "ffmpeg could not read a video duration"
            ),
        )
    return actual_dur


async def download_full(
    url: str,
    out_path: str,
    tracker=None,
    total_dur: float = 0,
    *,
    raise_on_error: bool = False,
) -> float:
    original = _extract_first_url(url) or str(url or "").strip()
    if not original.startswith(("http://", "https://")):
        error = SourceDownloadError(
            "Video link မမှန်ပါ။ http/https link အပြည့်အစုံ ပို့ပါ။",
            detail="source URL is empty or not http/https",
        )
        if raise_on_error:
            raise error
        return 0.0

    resolved = await _resolve_source_url(original)
    candidates = _source_url_candidates(original, resolved)
    direct_errors: list[SourceDownloadError] = []
    ytdlp_errors: list[SourceDownloadError] = []
    fallback_errors: list[SourceDownloadError] = []

    # A direct media/CDN URL is often faster and more reliable than asking
    # yt-dlp's generic extractor to rediscover the same URL.
    for candidate in candidates:
        candidate_path = (urlparse(candidate).path or "").lower()
        if _is_direct_media_url(candidate) and not candidate_path.endswith(".m3u8"):
            try:
                return await _download_direct_media(
                    candidate, out_path, tracker, total_dur
                )
            except SourceDownloadError as e:
                direct_errors.append(e)
                logger.warning(
                    "direct media download failed host=%s: %s",
                    _url_hostname(candidate),
                    e.detail,
                )

    selectors = [
        "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "best[height<=720]/best",
        "best",
    ]
    attempts: list[tuple[str, str, bool]] = []
    if candidates:
        for selector in selectors:
            attempts.append((candidates[0], selector, False))
        for candidate in candidates[1:]:
            attempts.append((candidate, selectors[0], False))
            attempts.append((candidate, "best", False))
        if _is_rednote_url(candidates[0]) and _yt_dlp_impersonation_available():
            attempts.append((candidates[0], selectors[0], True))
            attempts.append((candidates[0], "best", True))

    for candidate, selector, impersonate in attempts:
        try:
            return await _download_with_ytdlp_once(
                candidate,
                out_path,
                tracker,
                total_dur,
                format_selector=selector,
                impersonate=impersonate,
            )
        except SourceDownloadError as e:
            ytdlp_errors.append(e)
            logger.warning(
                "yt-dlp download failed host=%s selector=%s impersonate=%s: %s",
                _url_hostname(candidate),
                selector,
                impersonate,
                e.detail,
            )

    # Last RedNote fallback: the page's initial state normally contains
    # masterUrl/backupUrls/originVideoKey.  Parsing those direct CDN URLs
    # can still work when the extractor or an anti-bot layer fails.
    if any(_is_rednote_url(candidate) for candidate in candidates):
        for candidate in candidates:
            direct_urls = await _extract_rednote_direct_urls(candidate)
            for direct_url in direct_urls[:5]:
                try:
                    direct_path = (urlparse(direct_url).path or "").lower()
                    if direct_path.endswith(".m3u8") or not _is_direct_media_url(direct_url):
                        return await _download_with_ytdlp_once(
                            direct_url,
                            out_path,
                            tracker,
                            total_dur,
                            format_selector="best",
                        )
                    return await _download_direct_media(
                        direct_url,
                        out_path,
                        tracker,
                        total_dur,
                        referer=candidate,
                    )
                except SourceDownloadError as e:
                    fallback_errors.append(e)
                    logger.warning(
                        "RedNote direct fallback failed host=%s: %s",
                        _url_hostname(direct_url),
                        e.detail,
                    )

    error = (
        ytdlp_errors[0]
        if ytdlp_errors
        else (
            direct_errors[0]
            if direct_errors
            else (
                fallback_errors[0]
                if fallback_errors
                else SourceDownloadError(
                    "မူရင်းဗီဒီယို download မရပါ။",
                    detail="all source download attempts failed without a detailed error",
                )
            )
        )
    )
    logger.warning(
        "download_full failed host=%s: %s",
        _url_hostname(original),
        error.detail,
    )
    if raise_on_error:
        raise error
    return 0.0


def build_source_segment_plan(
    total_duration: float,
    segment_seconds: float = SOURCE_SEGMENT_SECONDS,
) -> list[dict]:
    """Build non-overlapping two-minute windows for any source duration."""
    try:
        duration = float(total_duration or 0)
    except (TypeError, ValueError):
        return []
    if duration <= 0:
        return []
    step = max(1.0, float(segment_seconds or SOURCE_SEGMENT_SECONDS))
    count = max(1, int(math.ceil(duration / step - 1e-9)))
    plan = []
    for index in range(count):
        start = index * step
        end = min(duration, start + step)
        if end - start <= 0.01:
            continue
        plan.append(
            {
                "index": len(plan) + 1,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
        )
    return plan


def make_keep_ranges(duration: float, cuts: list, max_remove_ratio: float = 0.08):
    candidates = []
    for item in cuts:
        try:
            start = max(0.0, min(duration, float(item["start"])))
            end = max(0.0, min(duration, float(item["end"])))
            if end > start:
                candidates.append((start, end))
        except (KeyError, TypeError, ValueError):
            continue

    candidates.sort()
    merged = []
    for start, end in candidates:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    removal_limit = duration * max_remove_ratio
    accepted_cuts = []
    removed = 0.0
    for start, end in merged:
        remaining = removal_limit - removed
        if remaining <= 0:
            break
        cut_len = min(end - start, remaining)
        if cut_len > 0:
            accepted_cuts.append((start, start + cut_len))
            removed += cut_len

    keep_ranges = []
    cursor = 0.0
    for start, end in accepted_cuts:
        if start > cursor:
            keep_ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep_ranges.append((cursor, duration))
    return keep_ranges


def split_pcm_by_weights(master: np.ndarray, scenes: list) -> list[np.ndarray]:
    if len(scenes) == 0:
        return []
    if len(scenes) == 1 or len(master) < SAMPLE_RATE:
        return [master]

    weights = np.array([max(5, len(s.get("burmese", ""))) for s in scenes], dtype=np.float64)
    cum_ratios = np.cumsum(weights)[:-1] / np.sum(weights)

    hop = int(0.02 * SAMPLE_RATE)
    n_frames = max(1, (len(master) - hop) // hop)
    rms = np.array([
        float(np.sqrt(np.mean(master[i * hop:i * hop + hop] ** 2)))
        for i in range(n_frames)
    ])

    cuts = [0]
    for r in cum_ratios:
        expected = int(r * n_frames)
        win = max(25, int(0.15 * n_frames))
        lo = max(cuts[-1] // hop + 10, expected - win)
        hi = min(n_frames - 10, expected + win)
        if hi <= lo:
            best = max(cuts[-1] // hop + 1, expected)
        else:
            dists = np.abs(np.arange(lo, hi) - expected) / max(1.0, float(win))
            score = rms[lo:hi] + 0.015 * dists
            best = lo + int(np.argmin(score))
        cuts.append(min(len(master), best * hop))
    cuts.append(len(master))

    clips = []
    for i in range(len(scenes)):
        seg = master[cuts[i]:cuts[i + 1]].copy()
        if len(seg) < 200:
            seg = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        clips.append(seg)
    return clips


def build_voice_filter(settings: dict = None) -> str:
    """Settings အလိုက် TTS အသံ filter တည်ဆောက်တယ် (deep / speed / volume)."""
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    if s.get("deep"):
        base = DEEP_VOICE_FILTER
        speed = float(s.get("speed") or 1.0)
        if abs(speed - 1.0) > 0.01:
            base += f",atempo={speed:.3f}"
    else:
        speed = float(s.get("speed") or 1.0)
        base = "loudnorm=I=-16:TP=-1.5:LRA=11"
        if abs(speed - 1.0) > 0.01:
            base = f"atempo={speed:.3f}," + base
    vol = float(s.get("volume") or 100) / 100.0
    if abs(vol - 1.0) > 0.01:
        base += f",volume={vol:.2f}"
    return base


async def _tts_one_batch(batch: list, batch_idx: int, total_batches: int, work_dir: str, settings: dict = None):
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    voice_name = s.get("voice") or "Charon"
    voice_filter = build_voice_filter(s)
    client = genai.Client(api_key=GEMINI_API_KEY)
    joined = "\n\n".join(s_["burmese"] for s_ in batch)

    for model in PREFERRED_TTS_MODELS:
        for attempt in range(2):
            tmp_raw = os.path.join(work_dir, f"tts_{batch_idx}_raw.bin")
            tmp_pcm = os.path.join(work_dir, f"tts_{batch_idx}.pcm")
            try:
                cfg = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                        )
                    ),
                )
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=f"""
အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် သဘာဝကျသော စကားပြောဟန်ဖြင့် ဖတ်ပါ။
အသံသည် နက်ရှိုင်းသော (deep, low), ရင်အုပ်သံပါသော, ပဲ့တင်သံမပါသော အသံဖြစ်ရမည်။
ဇာတ်လမ်းကို ကိုယ်တိုင်ကြုံတွေ့ခဲ့သော အဓိကဇာတ်ကောင်က ပြန်ပြောပြနေသလို အသံသွင်းပါ။
အလွန်မြန်မြန်၊ တစ်သံတည်း မဖတ်ပါနှင့်။
စာသားတွင် မပါသော စကားလုံးများကို မထည့်ပါနှင့်။

{joined}
""",
                    config=cfg,
                )
                cands = getattr(resp, "candidates", None) or []
                if not cands or not getattr(cands[0], "content", None):
                    raise RuntimeError("no candidates")
                parts = cands[0].content.parts
                if not parts or not getattr(parts[0], "inline_data", None):
                    raise RuntimeError("no inline_data")
                data = parts[0].inline_data.data
                if not data or len(data) < 1000:
                    raise RuntimeError(f"too small ({len(data) if data else 0} bytes)")

                with open(tmp_raw, "wb") as f:
                    f.write(data)

                result = run_cmd([
                    FFMPEG_BIN, "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", tmp_raw,
                    "-af", voice_filter,
                    "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    tmp_pcm,
                ], 180)

                if result.returncode != 0 or not os.path.exists(tmp_pcm):
                    raise RuntimeError("ffmpeg convert failed")
                if os.path.getsize(tmp_pcm) < 400:
                    raise RuntimeError("pcm too small")

                master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                clips = split_pcm_by_weights(master, batch)
                logger.info(f"TTS batch {batch_idx + 1}/{total_batches} OK ({model})")
                return clips

            except Exception as e:
                err = str(e)[:200]
                logger.warning(f"TTS batch {batch_idx + 1} attempt {attempt + 1} ({model}): {err}")
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    await asyncio.sleep(8)
                else:
                    await asyncio.sleep(2)
            finally:
                for p in (tmp_raw, tmp_pcm):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

    return None


async def synthesize_tts(scenes: list, work_dir: str, settings: dict = None) -> list[np.ndarray]:
    BATCH_SIZE = 8
    batches = [scenes[i:i + BATCH_SIZE] for i in range(0, len(scenes), BATCH_SIZE)]
    logger.info(f"TTS: {len(scenes)} scenes in {len(batches)} batches")
    all_clips = []

    for b_idx, batch in enumerate(batches):
        clips = await _tts_one_batch(batch, b_idx, len(batches), work_dir, settings)
        if clips is None:
            logger.error(f"TTS batch {b_idx + 1}/{len(batches)} failed — silence fallback")
            clips = [np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32) for _ in batch]
        all_clips.extend(clips)
        await asyncio.sleep(1.5)

    return all_clips


FIRST_PERSON_MALE = "ကျွန်တော်"
FIRST_PERSON_FEMALE = "ကျွန်မ"
FIRST_PERSON_PRONOUNS = (FIRST_PERSON_MALE, FIRST_PERSON_FEMALE)


def _normalize_burmese_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return (
        normalized
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        # Possessive ကျွန်တော့် must count as the same first-person pronoun.
        .replace("ကျွန်တော့်", FIRST_PERSON_MALE)
    )


def _first_person_counts(text: str) -> dict:
    normalized = _normalize_burmese_text(text)
    return {
        FIRST_PERSON_MALE: normalized.count(FIRST_PERSON_MALE),
        FIRST_PERSON_FEMALE: normalized.count(FIRST_PERSON_FEMALE),
    }


def _other_first_person_pronoun(pronoun: str) -> str:
    return FIRST_PERSON_FEMALE if pronoun == FIRST_PERSON_MALE else FIRST_PERSON_MALE


def _detect_pronoun(text: str) -> str:
    """Return the dominant first-person pronoun in text, or "" if absent."""
    counts = _first_person_counts(text)
    male = counts[FIRST_PERSON_MALE]
    female = counts[FIRST_PERSON_FEMALE]
    if male > female:
        return FIRST_PERSON_MALE
    if female > male:
        return FIRST_PERSON_FEMALE
    return ""


def _voice_first_person_pronoun(settings: dict = None) -> str:
    voice = str((settings or {}).get("voice") or "")
    return FIRST_PERSON_FEMALE if voice in ("Kore", "Aoede") else FIRST_PERSON_MALE


def _choose_first_person_pronoun(
    texts=None,
    preferred: str = "",
    response_pronoun: str = "",
    settings: dict = None,
) -> str:
    if preferred in FIRST_PERSON_PRONOUNS:
        return preferred
    joined = " ".join(str(t or "") for t in (texts or []))
    detected = _detect_pronoun(joined)
    if detected:
        return detected
    if response_pronoun in FIRST_PERSON_PRONOUNS:
        return response_pronoun
    return _voice_first_person_pronoun(settings)


def _scene_is_first_person(text: str, pronoun: str = "") -> bool:
    """A scene passes only if the narrator visibly speaks as ``I``.

    The JSON ``pronoun`` field is not enough: the actual Burmese narration
    must contain the selected pronoun and must not also use the opposite
    pronoun for the narrator.  This is intentionally strict because the
    previous prompt-only approach silently delivered third-person scripts.
    """
    value = str(text or "").strip()
    if not value:
        return False
    counts = _first_person_counts(value)
    if pronoun in FIRST_PERSON_PRONOUNS:
        other = _other_first_person_pronoun(pronoun)
        return counts.get(pronoun, 0) > 0 and counts.get(other, 0) == 0
    detected = _detect_pronoun(value)
    if not detected:
        return False
    other = _other_first_person_pronoun(detected)
    return counts.get(other, 0) == 0


def _first_person_bad_scene_indexes(scenes: list, pronoun: str = "") -> list:
    bad = []
    for index, scene in enumerate(scenes or []):
        text = ""
        if isinstance(scene, dict):
            text = str(scene.get("burmese") or scene.get("text") or "")
        elif isinstance(scene, str):
            text = scene
        if not _scene_is_first_person(text, pronoun):
            bad.append(index)
    return bad


def _first_person_metrics(texts: list, pronoun: str = "") -> dict:
    counts = {FIRST_PERSON_MALE: 0, FIRST_PERSON_FEMALE: 0}
    bad_indexes = []
    for index, text in enumerate(texts or []):
        value = str(text or "")
        item_counts = _first_person_counts(value)
        for key in counts:
            counts[key] += int(item_counts.get(key) or 0)
        if not _scene_is_first_person(value, pronoun):
            bad_indexes.append(index)
    return {
        "pronoun": pronoun or _detect_pronoun(" ".join(str(t or "") for t in (texts or []))),
        "male_pronoun_count": int(counts[FIRST_PERSON_MALE]),
        "female_pronoun_count": int(counts[FIRST_PERSON_FEMALE]),
        "scene_count": len(texts or []),
        "first_person_scene_count": len(texts or []) - len(bad_indexes),
        "bad_scene_indexes": bad_indexes,
        "all_scenes_first_person": bool(texts) and not bad_indexes,
    }


async def extract_story_summary(scenes: list, client) -> dict:
    if not scenes:
        return {"summary": "", "characters": [], "last_scene": "", "protagonist": "", "pronoun": ""}
    all_text = "\n".join(s["burmese"] for s in scenes)
    detected_pronoun = _detect_pronoun(all_text)
    prompt = f"""
အောက်ပါ မြန်မာဇာတ်လမ်းအပိုင်းကို ဖတ်ပြီး နောက်အပိုင်းအတွက် အကျဉ်းချုပ် ထုတ်ပေးပါ။
မှတ်ချက် — ဇာတ်ကြောင်းကို အဓိကဇာတ်ကောင် ကိုယ်တိုင် ပထမလူ (ကျွန်တော်/ကျွန်မ) ဖြင့် ပြောနေခြင်းဖြစ်သည်။
ပြောနေသူသည် အဓိကဇာတ်ကောင် ကိုယ်တိုင်ပင် ဖြစ်သည်။

{all_text}

Return JSON:
{{"characters": ["နာမည်1"], "protagonist": "အဓိကဇာတ်ကောင် (ပြောနေသူ ကိုယ်တိုင်)", "pronoun": "ပြောနေသူ သုံးထားသည့် နာမ်စား — ကျွန်တော် သို့မဟုတ် ကျွန်မ", "summary": "အဓိကအကြောင်းအရာ ၂-၃ ကြောင်း", "last_scene": "နောက်ဆုံးအခြေအနေ"}}
    """.strip()
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=PREFERRED_MODELS[0],
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = json.loads(resp.text.strip())
        if not isinstance(data, dict):
            raise ValueError("summary JSON is not an object")
        # စာသားထဲမှ တိုက်ရိုက် ရှာတွေ့သည့် နာမ်စား ကို model အဖြေထက် ပို ယုံကြည်သည်။
        model_pronoun = data.get("pronoun") if data.get("pronoun") in ("ကျွန်တော်", "ကျွန်မ") else ""
        data["pronoun"] = detected_pronoun or model_pronoun
        return data
    except Exception as e:
        logger.warning(f"Summary error: {e}")
        return {
            "characters": [],
            "protagonist": "",
            "pronoun": detected_pronoun,
            "summary": all_text[-500:],
            "last_scene": scenes[-1]["burmese"],
        }


def _with_required_coverage(settings: dict = None) -> dict:
    """Return job settings with product-wide required values applied.

    A saved/stale coverage value below 90 must never make an individual
    source produce a short recap.  The source is also always partitioned
    into fixed two-minute parts, regardless of a stale 3/5/10-minute
    setting saved by an older bot version.
    """
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    try:
        coverage = float(s.get("coverage") or MINIMUM_SCRIPT_COVERAGE_PERCENT)
    except (TypeError, ValueError):
        coverage = float(MINIMUM_SCRIPT_COVERAGE_PERCENT)
    coverage = max(float(MINIMUM_SCRIPT_COVERAGE_PERCENT), min(100.0, coverage))
    s["coverage"] = int(coverage) if coverage.is_integer() else coverage
    s["chunk"] = SOURCE_SEGMENT_SECONDS
    s["pov"] = "first"
    return s


def _settings_coverage_ratio(settings: dict = None) -> float:
    s = _with_required_coverage(settings)
    return float(s["coverage"]) / 100.0


def _settings_script_rate(settings: dict = None) -> float:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    try:
        rate = float(s.get("rate") or SCRIPT_CHARS_PER_SEC)
    except (TypeError, ValueError):
        rate = SCRIPT_CHARS_PER_SEC
    return rate if rate > 0 else SCRIPT_CHARS_PER_SEC


def _minimum_script_chars(target_chars: int) -> int:
    return int(math.ceil(max(0, int(target_chars or 0)) * SCRIPT_MIN_CHAR_RATIO))


def _settings_speed(settings: dict = None) -> float:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    try:
        speed = float(s.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    return speed if speed > 0 else 1.0


def _target_final_seconds(source_duration: float, settings: dict = None) -> float:
    """Actual final-video seconds requested for a source duration."""
    return max(0.0, float(source_duration or 0)) * _settings_coverage_ratio(settings)


def _final_duration_tolerance_seconds(source_duration: float) -> float:
    source = max(0.0, float(source_duration or 0))
    return max(
        FINAL_DURATION_TOLERANCE_MIN_SECONDS,
        source * FINAL_DURATION_TOLERANCE_RATIO,
    )


def _final_duration_window(source_duration: float, settings: dict = None) -> tuple:
    target = _target_final_seconds(source_duration, settings)
    tolerance = _final_duration_tolerance_seconds(source_duration)
    return target, max(0.0, target - tolerance), target + tolerance


def _duration_is_on_target(
    actual_seconds: float,
    source_duration: float,
    settings: dict = None,
) -> bool:
    if float(source_duration or 0) <= 0:
        return False
    _, low, high = _final_duration_window(source_duration, settings)
    return low <= float(actual_seconds or 0) <= high


def _duration_status(actual_seconds: float, source_duration: float, settings: dict = None) -> str:
    if float(source_duration or 0) <= 0:
        return "unknown"
    _, low, high = _final_duration_window(source_duration, settings)
    actual = float(actual_seconds or 0)
    if actual < low:
        return "short"
    if actual > high:
        return "long"
    return "on_target"


def _settings_pause_seconds(settings: dict = None) -> float:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    try:
        return max(0.0, float(s.get("pause") or 0))
    except (TypeError, ValueError):
        return PAUSE_AFTER_SCENE


def _clip_array(clip) -> np.ndarray:
    try:
        return np.asarray(clip if clip is not None else [], dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _clip_is_silent(clip) -> bool:
    arr = _clip_array(clip)
    return bool(arr.size == 0 or float(np.max(np.abs(arr))) < 0.001)


def _narration_measurements(scenes: list, clips: list, settings: dict = None) -> dict:
    """Measure actual post-TTS duration and this job's real chars/second."""
    scene_list = list(scenes or [])
    clip_list = list(clips or [])
    pause_seconds = _settings_pause_seconds(settings)
    speech_seconds = 0.0
    spoken_chars = 0
    spoken_seconds = 0.0
    silent_count = 0
    for index, clip in enumerate(clip_list):
        arr = _clip_array(clip)
        seconds = float(arr.size) / SAMPLE_RATE if SAMPLE_RATE else 0.0
        speech_seconds += seconds
        if _clip_is_silent(arr):
            silent_count += 1
            continue
        text = ""
        if index < len(scene_list) and isinstance(scene_list[index], dict):
            text = str(scene_list[index].get("burmese") or "").strip()
        if text and seconds > 0:
            spoken_chars += len(text)
            spoken_seconds += seconds
    pause_total = pause_seconds * max(0, len(scene_list) - 1)
    measured_rate = (spoken_chars / spoken_seconds) if spoken_seconds > 0 else 0.0
    if measured_rate <= 0:
        measured_rate = _settings_script_rate(settings) * _settings_speed(settings)
    return {
        "speech_seconds": round(float(speech_seconds), 6),
        "pause_seconds": round(float(pause_seconds), 6),
        "pause_total_seconds": round(float(pause_total), 6),
        "narration_seconds": round(float(speech_seconds + pause_total), 6),
        "silent_count": int(silent_count),
        "measured_chars_per_second": round(float(measured_rate), 6),
        "spoken_chars_measured": int(spoken_chars),
    }


def script_target_chars(chunk_dur: float, settings: dict = None) -> int:
    """Initial character estimate for the requested final duration.

    Voice speed is included so a faster voice needs more text to reach the
    same actual final duration.  The estimate is later corrected using the
    measured TTS rate for this exact job/voice.
    """
    ratio = _settings_coverage_ratio(settings)
    rate = _settings_script_rate(settings)
    speed = _settings_speed(settings)
    return int(max(0.0, float(chunk_dur or 0)) * ratio * rate * speed)


def script_total_chars(scenes: list) -> int:
    return sum(len((s.get("burmese") or "").strip()) for s in scenes if isinstance(s, dict))


def sort_scenes(scenes: list) -> list:
    def _start(s):
        try:
            return float(s.get("start", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0
    valid = [s for s in scenes if isinstance(s, dict) and (s.get("burmese") or "").strip()]
    return sorted(valid, key=_start)


def _trim_text_to_char_budget(text: str, budget: int) -> str:
    value = str(text or "").strip()
    budget = max(0, int(budget or 0))
    if len(value) <= budget:
        return value
    if budget <= 0:
        return ""
    cut = value.rfind("။", 0, budget)
    # Keep a complete sentence when it does not remove too much of the
    # scene; otherwise use the exact character budget as a last resort.
    if cut >= max(25, int(budget * 0.55)):
        return value[:cut + 1].strip()
    return value[:budget].strip()


def trim_scenes_proportionally(scenes: list, desired_total_chars: int) -> list:
    """Shorten every scene proportionally instead of deleting the ending.

    This is used after actual TTS measurement shows the narration is too
    long.  Keeping the beginning of every chronological scene preserves the
    first-person framing and avoids the old behaviour of dropping all late
    scenes simply because the total estimate was over target.
    """
    valid = [dict(s) for s in (scenes or []) if isinstance(s, dict)]
    desired = max(0, int(desired_total_chars or 0))
    total = script_total_chars(valid)
    if not valid or desired <= 0 or total <= desired:
        return valid

    ratio = desired / float(total)
    trimmed = []
    for scene in valid:
        text = str(scene.get("burmese") or "").strip()
        budget = max(1, int(math.floor(len(text) * ratio))) if text else 0
        new_scene = dict(scene)
        new_scene["burmese"] = _trim_text_to_char_budget(text, budget)
        trimmed.append(new_scene)

    # Sentence-boundary choices can leave a small remainder either way.
    # Remove any remaining excess from the longest scenes first, while
    # preserving at least a short first-person opening where possible.
    excess = script_total_chars(trimmed) - desired
    while excess > 0:
        candidates = [
            i for i, scene in enumerate(trimmed)
            if len(str(scene.get("burmese") or "")) > 30
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda i: len(str(trimmed[i].get("burmese") or "")),
        )
        current = str(trimmed[index].get("burmese") or "")
        reduction = min(excess, max(1, len(current) - 25))
        trimmed[index]["burmese"] = _trim_text_to_char_budget(
            current, len(current) - reduction
        )
        new_excess = script_total_chars(trimmed) - desired
        if new_excess >= excess:
            break
        excess = new_excess
    return [s for s in trimmed if str(s.get("burmese") or "").strip()]


def trim_scenes_to_target(scenes: list, target_chars: int) -> list:
    """Cap an initially over-long script without deleting its ending."""
    limit = int(max(0, int(target_chars or 0)) * SCRIPT_MAX_RATIO)
    if script_total_chars(scenes) <= limit:
        return scenes
    return trim_scenes_proportionally(scenes, int(target_chars or 0))


def _direct_first_person_repair_prompt(
    scenes: list,
    bad_indexes: list,
    pronoun: str,
    protagonist: str = "",
) -> str:
    payload = []
    for index in bad_indexes or []:
        if index < 0 or index >= len(scenes or []):
            continue
        scene = scenes[index] if isinstance(scenes[index], dict) else {}
        payload.append({
            "scene_index": int(index),
            "start": scene.get("start"),
            "end": scene.get("end"),
            "current_burmese_chars": len(str(scene.get("burmese") or "").strip()),
            "burmese": str(scene.get("burmese") or ""),
        })
    payload_json = json.dumps(payload, ensure_ascii=False)
    other = _other_first_person_pronoun(pronoun)
    return f"""
FIRST-PERSON REPAIR ONLY — အောက်ပါ scenes များသည် ပထမလူပုံစံ မဖြစ်သေးပါ။
အကြောင်းအရာ၊ အစဉ်နှင့် အသေးစိတ်အားလုံးကို မလျော့ဘဲ အဓိကဇာတ်ကောင် ကိုယ်တိုင် ပြန်ပြောနေသလို ပြန်ရေးပါ။
ပြောနေသူ၏ နာမ်စားမှာ «{pronoun}» တစ်မျိုးတည်းသာ ဖြစ်ရမည်။ «{other}» ကို ပြောနေသူအဖြစ် လုံးဝမသုံးပါနှင့်။
Scene တိုင်းတွင် «{pronoun}» ကို ပထမဝါကျ သို့မဟုတ် အစပိုင်းတွင် အနည်းဆုံးတစ်ကြိမ် ထည့်ပါ။
ကိုယ်တိုင်မရှိသော မြင်ကွင်းဆိုလျှင် «အဲဒီတုန်းက {pronoun} မသိသေးပေမယ့်…» သို့မဟုတ် «နောက်မှ {pronoun} သိလိုက်ရတာက…» ဟုစပါ။
အဓိကဇာတ်ကောင်အမည်: {protagonist or "မသိရသေး"}
ပြန်ရေးပြီးစာသားအရှည်သည် current_burmese_chars ထက် မနည်းရပါ။

Scenes:
{payload_json}

JSON ကိုသာ ပြန်ပေးပါ:
{{"pronoun": "{pronoun}", "scenes": [{{"scene_index": 0, "start": 0.0, "end": 12.0, "burmese": "ပြန်ရေးပြီး ပထမလူစာသား"}}]}}
""".strip()


def _parse_direct_scene_replacements(data, allowed_indexes) -> dict:
    allowed = {int(i) for i in (allowed_indexes or [])}
    raw_scenes = []
    if isinstance(data, dict) and isinstance(data.get("scenes"), list):
        raw_scenes = data["scenes"]
    elif isinstance(data, list):
        raw_scenes = data
    replacements = {}
    for item in raw_scenes:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("scene_index"))
        except (TypeError, ValueError):
            continue
        text = str(item.get("burmese") or item.get("text") or "").strip()
        if index in allowed and text:
            replacements[index] = text
    return replacements


async def _upload_video_to_gemini(client, chunk_path: str):
    video_file = await asyncio.to_thread(client.files.upload, file=chunk_path)
    for _ in range(40):
        state_obj = getattr(video_file, "state", None)
        state = getattr(state_obj, "name", str(state_obj or ""))
        if state in ("ACTIVE", "FAILED"):
            break
        await asyncio.sleep(2)
        video_file = await asyncio.to_thread(client.files.get, name=video_file.name)
    state_obj = getattr(video_file, "state", None)
    state = getattr(state_obj, "name", str(state_obj or ""))
    if state != "ACTIVE":
        raise RuntimeError("Gemini က ဗီဒီယိုကို စစ်ဆေး၍မရပါ။")
    return video_file


async def _delete_gemini_file(client, video_file) -> None:
    if not video_file:
        return
    try:
        await asyncio.to_thread(client.files.delete, name=video_file.name)
    except Exception as e:
        logger.warning(f"Gemini delete error: {e}")


async def build_chunk_recap(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    story_context: dict,
    chunk_index: int,
    tracker: ProgressTracker,
    settings: dict = None,
    *,
    script_only: bool = False,
    prepared_script: dict = None,
    extract_summary: bool = True,
    keep_source_chunk: bool = False,
    output_path: str = "",
):
    st = _with_required_coverage(settings)
    has_prepared_script = bool(
        isinstance(prepared_script, dict) and prepared_script.get("scenes")
    )
    if script_only and has_prepared_script:
        raise ValueError("script_only cannot be combined with prepared_script")
    coverage_ratio = _settings_coverage_ratio(st)
    chars_rate = _settings_script_rate(st)
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_fmt = fmt_time(global_start)
    end_fmt = fmt_time(global_end)

    # Output script must equal coverage% (default 90%) of the input chunk.
    target_chars = script_target_chars(chunk_dur, st)
    target_seconds = chunk_dur * coverage_ratio
    scene_count = max(15, int(target_seconds / 8))
    chars_per_scene = max(80, target_chars // scene_count)
    # 👤 ရေးဟန် = ပထမလူ အမြဲတမ်း (fixed) — အဓိကဇာတ်ကောင် ကိုယ်တိုင်
    # «ကျွန်တော်/ကျွန်မ» အနေနှင့် ပြန်ပြောပြသလို ရေးသည်။ အရင်သိမ်းထားသော
    # narrator ဆက်တင် အဟောင်းရှိလျှင်လည်း ပထမလူပင် ဖြစ်ရမည်။
    tone_line = {
        "natural": "သဘာဝကျသော လေသံဖြင့် ရေးပါ။",
        "dramatic": "ဒရာမာဆန်ဆန်၊ တင်းမာမှုရှိသော လေသံဖြင့် ရေးပါ။",
        "emotional": "စိတ်လှုပ်ရှားဖွယ်၊ ခံစားချက်ပြင်းသော လေသံဖြင့် ရေးပါ။",
        "calm": "အေးအေးဆေးဆေး၊ တည်ငြိမ်သော လေသံဖြင့် ရေးပါ။",
    }.get(st.get("tone"), "သဘာဝကျသော လေသံဖြင့် ရေးပါ။")
    protagonist = story_context.get("protagonist", "") if chunk_index > 1 else ""
    pronoun = story_context.get("pronoun", "") if chunk_index > 1 else ""
    if pronoun not in ("ကျွန်တော်", "ကျွန်မ"):
        pronoun = ""
    voice_pronoun = "ကျွန်မ" if st.get("voice") in ("Kore", "Aoede") else "ကျွန်တော်"
    if pronoun:
        other_pronoun = "ကျွန်မ" if pronoun == "ကျွန်တော်" else "ကျွန်တော်"
        pronoun_line = (
            f"ရှေ့ပိုင်းမှာ သုံးပြီးသား နာမ်စား «{pronoun}» ကိုသာ ဆက်သုံးပါ။ "
            f"«{other_pronoun}» သို့ လှည့် မသုံးပါနှင့်။"
        )
    else:
        pronoun_line = (
            "အဓိကဇာတ်ကောင်က အမျိုးသားဖြစ်လျှင် «ကျွန်တော်»၊ အမျိုးသမီးဖြစ်လျှင် «ကျွန်မ» သုံးပါ။ "
            f"ဗီဒီယိုမှ ကျား/မကို အတိအကျ မသိနိုင်လျှင် «{voice_pronoun}» သုံးပါ။ "
            "ရွေးပြီးနောက် ဤအပိုင်း တစ်လျှောက် လုံးဝ မပြောင်းပါနှင့်။"
        )

    if chunk_index == 1:
        context_section = (
            "🔵 ဤသည်မှာ ဇာတ်လမ်း၏ အစဖြစ်သည်။ ပထမဆုံး ဝါကျကစပြီး အဓိကဇာတ်ကောင် "
            "ကိုယ်တိုင် «ကျွန်တော်/ကျွန်မ» အနေနှင့် ပြန်ပြောနေသလို ရေးပါ။ "
            "«ဇာတ်လမ်းအကျဉ်း» ပုံစံ မိတ်ဆက်မထည့်ဘဲ ဇာတ်လမ်းထဲသို့ တိုက်ရိုက် ဝင်ပါ။"
        )
    else:
        chars_str = ", ".join(story_context.get("characters", [])[:10])
        protag_line = f"- အဓိကဇာတ်ကောင် = ပြောနေသူ ကိုယ်တိုင် (မပြောင်းရ): {protagonist}\n" if protagonist else ""
        speaker_line = (
            f"- ပြောနေသူ ကိုယ့်ကိုယ်ကို သုံးသည့် နာမ်စား (မပြောင်းရ): «{pronoun}»\n"
            if pronoun else ""
        )
        context_section = f"""
🔵 ယခင်ဖြစ်ပျက်ခဲ့သည့် ဇာတ်လမ်းအကျဉ်း (အချက်အလက်သိရုံသာ — ပြန် မရေးရ) —
{protag_line}{speaker_line}- ဇာတ်ကောင်များ: {chars_str}
- အဓိကဖြစ်ရပ်များ: {story_context.get('summary', '')}
- နောက်ဆုံးအခြေအနေ: {story_context.get('last_scene', '')}

🔵 ယခင်အပိုင်းကို အကျဉ်းချုပ် ပြန် မပြောပါနှင့် — အထက်ပါ နောက်ဆုံးအခြေအနေကနေ
အဓိကဇာတ်ကောင် တစ်ယောက်တည်းကပင် ဆက်ပြောနေသလို တန်းဆက်ရေးပါ။
ပြောနေသူ၊ နာမ်စား («ကျွန်တော်/ကျွန်မ») နှင့် အဓိကဇာတ်ကောင် လုံးဝ မပြောင်းပါနှင့်။
"""

    prompt = f"""
သင်သည် ဗီဒီယိုထဲရှိ ဇာတ်လမ်းကို အဓိကဇာတ်ကောင် ကိုယ်တိုင် «ကျွန်တော်/ကျွန်မ» အဖြစ် ကိုယ်တိုင်ကြုံခဲ့ရသမျှကို အပြည့်အဝ ပြန်ပြောပြနေသလိုမျိုး မြန်မာဘာသာဖြင့် ရေးပေးရမည့် ဇာတ်ကြောင်းရေးသူ ဖြစ်သည်။

ဤဗီဒီယိုသည် မူရင်းဗီဒီယို၏ {start_fmt} မှ {end_fmt} အထိ ဖြစ်သည်။ ဤအပိုင်းကိုသာ ကြည့်ပြီး အောက်ပါ ညွှန်ကြားချက်များအတိုင်း ဇာတ်ကြောင်းရေးပါ။

{context_section}
- ရှုထောင့်: ပထမလူ (First person) အမြဲတမ်း — အောက်ပါ စည်းကမ်းများကို အတိအကျလိုက်နာပါ။
- နာမ်စား: {pronoun_line}
- လေသံ: {tone_line}

## ပထမလူ အမြင်ဖြင့် ပြောပုံ (ခန်းတိုင်း အမြဲလိုက်နာရန်)
- ပြောနေသူသည် အဓိကဇာတ်ကောင် ကိုယ်တိုင်သာ ဖြစ်သည်။ ကိုယ့်ကိုယ်ကို ညွှန်းတိုင်း «ကျွန်တော်» (အမျိုးသားဆိုလျှင်) သို့မဟုတ် «ကျွန်မ» (အမျိုးသမီးဆိုလျှင်) ဟုသာ သုံးပါ။ ဥပမာ — «အဲဒီအချိန်မှာ ကျွန်တော် ...»၊ «ကျွန်မတော့ ... လို့ ပြန်ဖြေလိုက်တယ်»။ **Scene တိုင်း၏ ပထမဝါကျ သို့မဟုတ် အစပိုင်းတွင် ရွေးထားသော နာမ်စား အနည်းဆုံး တစ်ကြိမ် မဖြစ်မနေပါရမည်။** JSON ၏ pronoun field ထည့်ရုံနဲ့ မလုံလောက်ပါ — burmese စာသားထဲမှာတင် ပါရပါမည်။
- ကိုယ့်အကြောင်းကို ကိုယ့်အမည် သို့မဟုတ် «သူ/သူမ» သုံး၍ တတိယလူကဲ့သို့ လုံးဝ မရေးပါနှင့်။ အခြားဇာတ်ကောင်များကိုမူ ၎င်းတို့၏ အမည် သို့မဟုတ် «သူ/သူမ» ဖြင့် သုံးနိုင်သည်။
- အဓိကဇာတ်ကောင်၏ အမည်ကို ဗီဒီယိုမှ မသိနိုင်ပါက မတီထွင်ပါနှင့်။ (ကျား/မ မသေချာလျှင် အထက်ပါ «နာမ်စား» စည်းကမ်းအတိုင်း ရွေးပြီး မပြောင်းပါနှင့်။)
- ကိုယ် မရှိခဲ့၊ ကိုယ်တိုင် မမြင်ရသေးသည့် မြင်ကွင်းများကိုလည်း လုံးဝ မကျော်ပါနှင့် — «အဲဒီတုန်းက ကျွန်တော် မသိသေးပေမယ့် ...»၊ «နောက်မှသာ သိလိုက်ရတာက ...» ဟု ဆက်စပ်ပြောပြပါ။ သို့သော် ကိုယ်တိုင် မြင်ခဲ့သယောင် မပြောပါနှင့်။
- ကိုယ့်အတွင်းစိတ် အတွေး၊ ခံစားချက်များကို ဗီဒီယိုထဲရှိ မျက်နှာအမူအရာ၊ အပြုအမူနှင့် စကားပြောချက်များပေါ် အခြေခံပြီးသာ ရေးပါ။ ဗီဒီယိုတွင် မပါသည်ကို မတီထွင်ပါနှင့်။

## ဗီဒီယိုကို အပြည့်အဝ ပြန်ပြောပုံ (အကျဉ်းချုပ် မဟုတ်ပါ)
- အတိုချုပ် (summary) မရေးပါနှင့်။ နားထောင်နေသူက မူရင်းဗီဒီယိုကို ပြန်ပင် ကြည့်နေရသလို ထင်ရအောင် အစက အဆုံးအထိ အစဉ်လိုက် အပြည့်အဝ ပြန်ပြောပြပါ။
- ခန်းတိုင်းတွင် ဘယ်သူတွေ ရှိနေလဲ၊ ဘယ်မှာလဲ၊ ဘာလုပ်နေလဲ၊ ဘာတွေ ပြောလဲ (အဓိပ္ပာယ်)၊ တစ်ဖက်က ဘယ်လို တုံ့ပြန်လဲ၊ ဘာကြောင့်လဲနှင့် ဘာဆက်ဖြစ်သည်ကို ရှင်းလင်းစွာ ဖော်ပြပါ။
- ဖြစ်ရပ် များစွာကို ဝါကျတစ်ကြောင်းတည်းနှင့် ချုံ့ မပစ်ပါနှင့်။ အလယ်ပိုင်းကို ကျော်၍ အဆုံးသို့ တန်းမရောက်ပါနှင့် — နောက်ဆုံးခန်းသည် ဤအပိုင်း၏ နောက်ဆုံး မြင်ကွင်းအထိ ရောက်အောင် ရေးပါ။
- အဓိက ဖြစ်ရပ်များအပြင် ဇာတ်လမ်းကို နားလည်ရန် လိုအပ်သည့် အသေးစိတ်များ — ဇာတ်ကောင်တို့၏ ဆက်နွယ်မှု၊ အငြင်းပွားမှု၊ လျှို့ဝှက်ချက်၊ အလှည့်အပြောင်းနှင့် နောင်ပိုင်း သက်ရောက်မည့် ပစ္စည်း၊ စာ၊ ဖုန်းခေါ်ဆိုမှု စသည်တို့ကို တစ်ခု မကျန် ထည့်ပါ။
- စကားပြောချက်များကို စာလုံးချင်း ကူးစရာ မလိုသော်လည်း ဘယ်သူက ဘယ်သူ့ကို ဘာပြောခဲ့သည်ဟူသော အဓိပ္ပာယ်ကို အတိအကျ ထိန်းသိမ်းပြီး ပထမလူအပြောထဲ သဘာဝကျကျ ပေါင်းစည်း ရေးပါ။ ဥပမာ — «သူက ကျွန်တော့်ကို ... လို့ ပြောလာတော့ ကျွန်တော်လည်း ... လို့ ပြန်ပြောလိုက်တယ်»။
- ဖြစ်ရပ်တစ်ခုမှ နောက်တစ်ခုသို့ «အဲဒီအချိန်မှာတော့ ...»၊ «ဒါပေမယ့် ...»၊ «အဲဒီနောက်မှာတော့ ...» စသည့် စကားဆက်များကို လိုအပ်သလို အမျိုးမျိုး သုံးပါ။ တစ်မျိုးတည်းကို ထပ်ခါထပ်ခါ မသုံးပါနှင့်။
- ဖယ်ပစ်နိုင်သည်မှာ ဇာတ်လမ်းနှင့် မသက်ဆိုင်သည့် ကြာညောင်းသော တိတ်ဆိတ်မှု၊ ထပ်နေသော မြင်ကွင်း/စကားနှင့် မလိုအပ်သော နားချိန်များသာ ဖြစ်သည်။ အဓိကကျသည့် မည်သည့်အရာမှ မကျန်စေရ။
- ဤအပိုင်းအဆုံးတွင် ဇာတ်လမ်းဆက်နေပါက နောက်တွင် ဘာဖြစ်မည်ကို ခန့်မှန်း မရေးဘဲ ဤအပိုင်း၌ မြင်ရသည့် နေရာတွင်ပင် ရပ်ပါ။

## ကြာချိန်နှင့် ဗီဒီယိုတည်းဖြတ်ရန် အချိန်မှတ်ချက်များ
- နောက်ဆုံး final video ၏ အမှန်တကယ်ကြာချိန်သည် မူရင်းအပိုင်း၏ {st['coverage']}% ဖြစ်ရမည်။ စာလုံးပမာဏသည် အဲဒီကြာချိန်ကို ရအောင်တွက်ထားသော ပမာဏဖြစ်ပြီး အကျဉ်းချုပ်၍ မရပါ။ အရေးမကြီးသော တိတ်ဆိတ်နေမှု၊ ထပ်နေသောမြင်ကွင်းနှင့် မလိုအပ်သော နားချိန်များကိုသာ ဖြတ်ပါ။
- ဇာတ်ကြောင်းစာသားကို ရည်မှန်းထားသည့် ကြာချိန်နှင့် ကိုက်ညီအောင် လုံလောက်စွာ အသေးစိတ်ရေးပါ။ အလွန်တိုသော အကျဉ်းချုပ် မရေးပါနှင့်။
- အနည်းဆုံး ခန်း {scene_count} ခန်း ရေးပါ။ ခန်းတစ်ခန်းလျှင် အနည်းဆုံး {chars_per_scene} လုံး ပါဝင်ပါ။
- အပိုင်းတစ်ခုစီအတွက် "start" နှင့် "end" တွင် မူရင်းဗီဒီယို၏ ဤအပိုင်းအစမှ စတွက်ထားသော စက္ကန့်ကို ထည့်ပါ။
- ပထမခန်း၏ "start" သည် အပိုင်းအစ (0.0 ပတ်ဝန်းကျင်) မှ စပြီး နောက်ဆုံးခန်း၏ "end" သည် အပိုင်းဆုံး ({chunk_dur:.0f} စက္ကန့် ပတ်လည်) အထိ ရောက်ရမည်။ အလယ်တွင် ဇာတ်လမ်းနှင့် ဆိုင်သည့် အချိန်ကွက် တစ်ခုမှ မလွတ်ရပါ။
- အရေးမကြီးသောအပိုင်းကို ဖြတ်သင့်ပါက ၎င်း၏ မူရင်းစတင်ချိန်နှင့် ဆုံးချိန်ကို "cuts" စာရင်းတွင် ထည့်ပြီး ဖြတ်ရသည့်အကြောင်းကို တိုတိုရေးပါ။
- ထိန်းသိမ်းမည့် မြင်ကွင်းများကို မူရင်းအစဉ်အတိုင်း စီပါ။ မြင်ကွင်းအချိန်များ ထပ်မနေစေရ၊ အရေးကြီးသောအပိုင်းများလည်း မကျန်ခဲ့စေရ။

## မထုတ်ခင် ကိုယ့်ဘာသာ စစ်ဆေးရန်
JSON မထုတ်ခင် အောက်ပါတို့ကို စစ်ပါ — (၁) စာသား တစ်လျှောက် နာမ်စား တစ်မျိုးတည်း («ကျွန်တော်» သို့မဟုတ် «ကျွန်မ») ပင် သုံးထားသလား၊ ကိုယ့်ကို «သူ/သူမ» ဟု မပြောင်းမိဘူးလား။ (၂) ဗီဒီယို အစကနေ အဆုံး အစဉ်လိုက် တစ်ခု မကျန် ပါပြီလား။ (၃) ဗီဒီယိုထဲ မပါသည့် အရာကို မတီထွင် ထားဘူးလား။ (၄) စုစုပေါင်း စာလုံးအရေအတွက် {target_chars} လုံး ဝန်းကျင် ပြည့်ရဲ့လား။ စစ်ဆေးပြီးမှသာ JSON ထုတ်ပါ။

## အဖြေထုတ်ပုံ
အောက်ပါပုံစံအတိုင်း JSON ကိုသာ ပြန်ပေးပါ။ JSON ပြင်ပတွင် ရှင်းလင်းချက် သို့မဟုတ် အခြားစာသား မရေးပါနှင့်။

{{
  "protagonist": "အဓိကဇာတ်ကောင်နာမည် (မသိရင် ကွက်လပ်ထား)",
  "pronoun": "ကျွန်တော် သို့မဟုတ် ကျွန်မ (စာသားတွင် သုံးထားသည့်အတိုင်း)",
  "scenes": [
    {{
      "start": 0.0,
      "end": 12.0,
      "burmese": "«ကျွန်တော်/ကျွန်မ» အဖြစ် ပထမလူဖြင့် ရေးထားသော မြန်မာဇာတ်ကြောင်း"
    }}
  ],
  "cuts": [
    {{
      "start": 12.0,
      "end": 15.0,
      "reason": "ထပ်နေသော မြင်ကွင်း"
    }}
  ]
}}
""".strip()

    video_file = None
    scenes = []
    cuts = []
    new_protagonist = protagonist
    new_pronoun = pronoun

    if has_prepared_script:
        # Manual per-part flow: the script was already generated and stored
        # before the original part was sent.  Rendering starts from those
        # exact scenes; Gemini is only asked for more text later if the
        # measured TTS duration proves the stored script is too short.
        scenes = json.loads(
            json.dumps(prepared_script.get("scenes") or [], ensure_ascii=False)
        )
        cuts = json.loads(
            json.dumps(prepared_script.get("cuts") or [], ensure_ascii=False)
        )
        new_protagonist = str(
            prepared_script.get("protagonist") or protagonist or ""
        )
        prepared_pronoun = str(prepared_script.get("pronoun") or "")
        if prepared_pronoun in ("ကျွန်တော်", "ကျွန်မ"):
            new_pronoun = prepared_pronoun
        await tracker.update(
            "📝 သိမ်းထားပြီးသား Script ကို သုံး၍ Recap ပြင်ဆင်နေပါသည်...",
            force=True,
        )
    else:
        await tracker.update("🎧 Gemini ဆီ ဗီဒီယိုတင်ပြီး ဇာတ်လမ်းစာသား ရေးနေပါသည်...", force=True)
        video_file = await _upload_video_to_gemini(client, chunk_path)
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=PREFERRED_MODELS[0],
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            parsed = json.loads(resp.text.strip())
            if isinstance(parsed, dict):
                scenes = parsed.get("scenes", [])
                cuts = parsed.get("cuts", [])
                if not protagonist:
                    new_protagonist = parsed.get("protagonist", "") or protagonist
                if parsed.get("pronoun") in ("ကျွန်တော်", "ကျွန်မ"):
                    new_pronoun = parsed["pronoun"]
            elif isinstance(parsed, list):
                scenes = parsed
        except Exception as e:
            logger.error(f"AI script error: {e}")

    scenes = sort_scenes(scenes)
    new_pronoun = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes],
        preferred=pronoun,
        response_pronoun=new_pronoun,
        settings=st,
    )
    total_chars = script_total_chars(scenes)
    logger.info(
        f"Chunk {chunk_index}: {len(scenes)} scenes, {total_chars}/{target_chars} chars "
        f"({(total_chars / chars_rate / chunk_dur * 100) if chunk_dur else 0:.0f}% of input, target {st['coverage']}%)"
    )

    # Top up repeatedly until the script reaches 90% of the input.
    # (Previously only ONE top-up ran, so short scripts slipped through
    # and the remaining video played with silent narration gaps.)
    # A prepared/manual script was already topped up when it was written;
    # measured-TTS calibration below is the only stage allowed to add more.
    for attempt in range(0 if has_prepared_script else SCRIPT_TOPUP_ATTEMPTS):
        if total_chars >= target_chars:
            break
        missing = target_chars - total_chars
        await tracker.update(
            f"⚠️ Script {total_chars / max(1, target_chars) * 100:.0f}% ပဲရှိသေးသည် (ပစ်မှတ် {st['coverage']}%)။ "
            f"လိုနေသော {missing} လုံး ထပ်ဖြည့်နေပါသည်... ({attempt + 1}/{SCRIPT_TOPUP_ATTEMPTS})",
            force=True,
        )
        extra_prompt = f"""
မူရင်းဗီဒီယိုကို ပြန်ကြည့်ပြီး ယခင် script မှာ ကျန်ခဲ့သော သို့မဟုတ် အတိုချုပ်လွန်းသော အစိတ်အပိုင်းများကို အသေးစိတ် ဖြည့်စွက် ရေးပေးပါ။
ယခင် ရေးပြီးသား အကြောင်းအရာများကို စကားလုံး ထပ်မရေးပါနှင့်။ ယခင် scenes များ၏ ကြားလပ်နေသော အချိန်များ၊ ကျန်ခဲ့သော ဖြစ်ရပ်များ၊ စကားပြောချက်များ၏ အဓိပ္ပာယ်၊ ဇာတ်ကောင်တို့၏ တုံ့ပြန်မှု၊ အကြောင်းရင်းနှင့် အကျိုးဆက်များကို အစဉ်အတိုင်း ဖြည့်ရေးပါ။
ပထမလူ အမြင် (First person) ကို ဆက်ထိန်းပါ — ပြောနေသူသည် အဓိကဇာတ်ကောင် ကိုယ်တိုင်ဖြစ်၍ နာမ်စား «{new_pronoun or pronoun or 'ယခင် script တွင် သုံးထားသော ကျွန်တော်/ကျွန်မ'}» ကိုသာ ဆက်သုံးပါ။ ကိုယ့်အကြောင်းကို «သူ/သူမ» ဟု မပြောင်းပါနှင့်။
ကိုယ် မရှိခဲ့၊ မမြင်ရသေးသည့် မြင်ကွင်းဖြစ်လျှင် «အဲဒီတုန်းက ကျွန်တော် မသိသေးပေမယ့် ...» ဟု ဆက်စပ် ဖော်ပြပြီး ဘာမှ မကျန်အောင် ထည့်ပါ။
မူရင်းဗီဒီယိုတွင် မပါသော အရာကို မတီထွင်ပါနှင့်။
🎯 အနည်းဆုံး {missing} လုံး အသစ် ထပ်ဖြည့်ပါ။ စုစုပေါင်း {target_chars} လုံး (မူရင်း၏ {st['coverage']}%) ပြည့်ရပါမည်။

ယခင် script:
{json.dumps(scenes, ensure_ascii=False)}

အောက်ပါပုံစံအတိုင်း JSON object ကိုသာ ပြန်ပေးပါ။ JSON ပြင်ပတွင် မည်သည့်စာသားမှ မရေးပါနှင့်။
{{"pronoun": "ကျွန်တော် သို့မဟုတ် ကျွန်မ", "scenes": [{{"start": 0.0, "end": 12.0, "burmese": "ထပ်ဖြည့်စာသား"}}], "cuts": []}}
        """.strip()
        try:
            resp2 = await asyncio.to_thread(
                client.models.generate_content,
                model=PREFERRED_MODELS[0],
                contents=[video_file, extra_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            extra = json.loads(resp2.text.strip())
            if isinstance(extra, dict):
                scenes.extend(extra.get("scenes", []))
                cuts.extend(extra.get("cuts", []))
                if not new_protagonist:
                    new_protagonist = extra.get("protagonist", "") or new_protagonist
                if not new_pronoun and extra.get("pronoun") in ("ကျွန်တော်", "ကျွန်မ"):
                    new_pronoun = extra["pronoun"]
            elif isinstance(extra, list):
                scenes.extend(extra)
        except Exception as e:
            logger.warning(f"Extra script error (attempt {attempt + 1}): {e}")
        scenes = sort_scenes(scenes)
        total_chars = script_total_chars(scenes)
        logger.info(f"Chunk {chunk_index} top-up {attempt + 1}: {total_chars}/{target_chars} chars")

    # Cap an initially over-long script proportionally so the ending is not
    # simply deleted.  Actual duration is corrected again after TTS below.
    # Do not silently rewrite a stored manual script before measuring it.
    if not has_prepared_script:
        scenes = trim_scenes_to_target(scenes, target_chars)
    new_pronoun = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes],
        preferred=new_pronoun or pronoun,
        settings=st,
    )

    # Prompt-only first-person instructions are not sufficient.  Validate
    # every scene's actual Burmese text and ask Gemini to rewrite only the
    # scenes that still sound like a third-person summary.
    for repair_round in range(0 if has_prepared_script else FIRST_PERSON_REPAIR_ROUNDS):
        bad_indexes = _first_person_bad_scene_indexes(scenes, new_pronoun)
        if not bad_indexes:
            break
        await tracker.update(
            "👤 ပထမလူအပြောပုံစံ မဖြစ်သေးသော scene များ ပြန်ပြင်နေပါသည်... "
            f"({repair_round + 1}/{FIRST_PERSON_REPAIR_ROUNDS})",
            force=True,
        )
        accepted = 0
        for start_index in range(0, len(bad_indexes), 5):
            batch_indexes = bad_indexes[start_index:start_index + 5]
            repair_prompt = _direct_first_person_repair_prompt(
                scenes, batch_indexes, new_pronoun, new_protagonist
            )
            try:
                repair_resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=PREFERRED_MODELS[0],
                    contents=[video_file, repair_prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                repair_data = json.loads(repair_resp.text.strip())
            except Exception as e:
                logger.warning(f"Direct first-person repair failed: {e}")
                continue
            replacements = _parse_direct_scene_replacements(
                repair_data, batch_indexes
            )
            for scene_index, new_text in replacements.items():
                old_text = str(scenes[scene_index].get("burmese") or "")
                if (
                    _scene_is_first_person(new_text, new_pronoun)
                    and len(new_text.strip()) >= max(1, int(len(old_text.strip()) * 0.9))
                ):
                    scenes[scene_index] = {
                        **scenes[scene_index],
                        "burmese": new_text.strip(),
                    }
                    accepted += 1
        if accepted <= 0:
            break

    scenes = sort_scenes(scenes)
    total_chars = script_total_chars(scenes)
    spoken_est = total_chars / chars_rate if chars_rate else 0.0
    coverage_pct = (spoken_est / chunk_dur * 100) if chunk_dur else 0
    first_person_metrics = _first_person_metrics(
        [str(s.get("burmese") or "") for s in scenes], new_pronoun
    )
    logger.info(
        f"Chunk {chunk_index} final script: {len(scenes)} scenes, {total_chars} chars, "
        f"~{spoken_est:.0f}s spoken = {coverage_pct:.0f}% of {chunk_dur:.0f}s input "
        f"(target {st['coverage']}%), first-person="
        f"{first_person_metrics.get('first_person_scene_count')}/{len(scenes)}"
    )

    minimum_chars = _minimum_script_chars(target_chars)
    bad_first_person = first_person_metrics.get("bad_scene_indexes") or []
    if not scenes or total_chars < minimum_chars or bad_first_person:
        await _delete_gemini_file(client, video_file)
        raise RuntimeError(
            "Script failed duration/first-person validation after repairs: "
            f"{total_chars}/{target_chars} chars "
            f"(~{spoken_est:.1f}s of {chunk_dur:.1f}s input, "
            f"target {float(st.get('coverage') or 90):.0f}%), "
            f"third-person scene indexes={bad_first_person}; "
            "refusing to render a shortened or third-person recap"
        )

    await tracker.update(
        f"📝 Script ပြီးပါပြီ — input ၏ {coverage_pct:.0f}% (ပစ်မှတ် {st['coverage']}%)၊ {total_chars} လုံး",
        force=True,
    )

    if script_only:
        await _delete_gemini_file(client, video_file)
        if extract_summary:
            script_summary = await extract_story_summary(scenes, client)
        else:
            script_summary = {
                "characters": list((story_context or {}).get("characters") or []),
                "summary": str((story_context or {}).get("summary") or ""),
                "last_scene": str((scenes[-1].get("burmese") or "")) if scenes else "",
                "protagonist": new_protagonist,
                "pronoun": new_pronoun,
            }
        if new_protagonist and not script_summary.get("protagonist"):
            script_summary["protagonist"] = new_protagonist
        if new_pronoun and not script_summary.get("pronoun"):
            script_summary["pronoun"] = new_pronoun
        script_payload = {
            "version": 1,
            "part_index": int(chunk_index),
            "global_start": round(float(global_start or 0), 3),
            "global_end": round(float(global_end or 0), 3),
            "duration": round(float(chunk_dur or 0), 3),
            "scenes": json.loads(json.dumps(scenes, ensure_ascii=False)),
            "cuts": json.loads(json.dumps(cuts, ensure_ascii=False)),
            "protagonist": new_protagonist,
            "pronoun": new_pronoun,
            "story_context_before": json.loads(
                json.dumps(story_context or {}, ensure_ascii=False)
            ),
            "story_context_after": script_summary,
            "settings": json.loads(json.dumps(st, ensure_ascii=False)),
            "metrics": {
                "target_chars": int(target_chars),
                "actual_chars": int(total_chars),
                "estimated_spoken_seconds": round(float(spoken_est), 3),
                "estimated_coverage_percent": round(float(coverage_pct), 2),
                "target_final_seconds": round(float(target_seconds), 3),
                "first_person": first_person_metrics,
            },
        }
        return script_payload, script_summary

    target_final_seconds, duration_low, duration_high = _final_duration_window(
        chunk_dur, st
    )
    await tracker.update(f"🎙️ {st.get('voice', 'Charon')} AI မှ အသံသွင်းနေပါသည်...", force=True)

    # Character counts are only an estimate.  Measure the real TTS audio;
    # if it is not already at the requested final duration, use this job's
    # measured chars/second to ask Gemini for precisely the missing amount
    # (or proportionally trim every scene if it is too long), then TTS again.
    clips = []
    audio_seconds = 0.0
    duration_status = "unknown"
    calibration_history = []
    try:
        clips = await synthesize_tts(scenes, os.path.dirname(chunk_path), st)
        for calibration_round in range(DURATION_CALIBRATION_ROUNDS + 1):
            if clips is None or len(clips) < len(scenes):
                raise RuntimeError("Gemini TTS returned too few clips")
            clips = list(clips)[:len(scenes)]
            measurements = _narration_measurements(scenes, clips, st)
            audio_seconds = float(measurements.get("narration_seconds") or 0)
            silent_count = int(measurements.get("silent_count") or 0)
            if silent_count == len(clips) or silent_count > max(1, int(len(clips) * 0.25)):
                raise RuntimeError(
                    f"Gemini TTS produced too many silent scenes: {silent_count}/{len(clips)}"
                )
            duration_status = _duration_status(audio_seconds, chunk_dur, st)
            calibration_history.append({
                "round": calibration_round,
                **measurements,
                "status": duration_status,
            })
            logger.info(
                f"Chunk {chunk_index} measured TTS round {calibration_round}: "
                f"{audio_seconds:.1f}s actual, target {target_final_seconds:.1f}s "
                f"from {chunk_dur:.1f}s input, status={duration_status}, "
                f"measured_rate={measurements.get('measured_chars_per_second')} chars/s"
            )
            if duration_status == "on_target":
                break
            if calibration_round >= DURATION_CALIBRATION_ROUNDS:
                break

            measured_rate = float(
                measurements.get("measured_chars_per_second")
                or (chars_rate * _settings_speed(st))
            )
            desired_speech_seconds = max(
                0.1,
                target_final_seconds - float(measurements.get("pause_total_seconds") or 0),
            )
            current_chars = script_total_chars(scenes)

            if duration_status == "short":
                desired_total_chars = int(math.ceil(
                    desired_speech_seconds * measured_rate * 1.01
                ))
                missing_chars = max(1, desired_total_chars - current_chars)
                await tracker.update(
                    f"⏱️ အသံအမှန် {fmt_time(audio_seconds)} ပဲရှိသေးလို့ "
                    f"ပစ်မှတ် {fmt_time(target_final_seconds)} ပြည့်အောင် "
                    f"script ထပ်ဖြည့်နေပါသည်... ({calibration_round + 1}/{DURATION_CALIBRATION_ROUNDS})",
                    force=True,
                )
                extra_prompt = f"""
မူရင်းဗီဒီယိုကို ပြန်ကြည့်ပြီး ယခင် script မှာ အတိုချုပ်လွန်းသော/ကျန်ခဲ့သော အပိုင်းများကို scene အသစ်များအဖြစ် ထပ်ဖြည့်ပါ။
ယခင် scenes များကို ပြန်မပေးပါနှင့် — အသစ်ထပ်ထည့်ရမည့် scenes များကိုသာ ပေးပါ။
အသစ်စာသား စုစုပေါင်း အနည်းဆုံး {missing_chars} လုံး ရှိရမည်ဖြစ်ပြီး ရှိပြီးသားနှင့်ပေါင်းလျှင် {desired_total_chars} လုံး ပြည့်ရမည်။
ပြောနေသူမှာ အဓိကဇာတ်ကောင် ကိုယ်တိုင်ဖြစ်ပြီး နာမ်စား «{new_pronoun}» တစ်မျိုးတည်းကိုသာ သုံးရမည်။
Scene တိုင်း၏ အစပိုင်းတွင် «{new_pronoun}» အနည်းဆုံးတစ်ကြိမ် ပါရမည်။
ဗီဒီယိုထဲမပါသော အရာကို မတီထွင်ပါနှင့်။ စကားလုံးထပ်ဖြည့်ခြင်းမဟုတ်ဘဲ ကျန်ခဲ့သော ဖြစ်ရပ်၊ စကားပြောချက်၊ တုံ့ပြန်မှု၊ အကြောင်းရင်းနှင့် အကျိုးဆက်များကို ထည့်ပါ။

ရှိပြီးသား script:
{json.dumps(scenes, ensure_ascii=False)}

JSON ကိုသာပြန်ပေးပါ:
{{"pronoun": "{new_pronoun}", "scenes": [{{"start": 0.0, "end": 12.0, "burmese": "အသစ်ထပ်ဖြည့်သော ပထမလူစာသား"}}], "cuts": []}}
""".strip()
                try:
                    if video_file is None:
                        video_file = await _upload_video_to_gemini(client, chunk_path)
                    resp2 = await asyncio.to_thread(
                        client.models.generate_content,
                        model=PREFERRED_MODELS[0],
                        contents=[video_file, extra_prompt],
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    extra = json.loads(resp2.text.strip())
                except Exception as e:
                    logger.warning(f"Direct duration calibration top-up failed: {e}")
                    break
                raw_extra_scenes = []
                if isinstance(extra, dict):
                    raw_extra_scenes = extra.get("scenes", [])
                    cuts.extend(extra.get("cuts", []))
                elif isinstance(extra, list):
                    raw_extra_scenes = extra
                accepted_scenes = [
                    scene for scene in raw_extra_scenes
                    if isinstance(scene, dict)
                    and _scene_is_first_person(
                        str(scene.get("burmese") or ""), new_pronoun
                    )
                ]
                if not accepted_scenes:
                    logger.warning("Direct duration calibration returned no first-person scenes")
                    break
                scenes.extend(accepted_scenes)
                scenes = sort_scenes(scenes)
                if script_total_chars(scenes) <= current_chars:
                    break
            elif duration_status == "long":
                desired_total_chars = int(math.floor(
                    desired_speech_seconds * measured_rate * 0.99
                ))
                await tracker.update(
                    f"⏱️ အသံအမှန် {fmt_time(audio_seconds)} ရှည်နေလို့ "
                    f"ပစ်မှတ် {fmt_time(target_final_seconds)} ဖြစ်အောင် ချိန်နေပါသည်...",
                    force=True,
                )
                trimmed = trim_scenes_proportionally(scenes, desired_total_chars)
                if script_total_chars(trimmed) >= current_chars:
                    break
                scenes = sort_scenes(trimmed)
            else:
                break

            clips = await synthesize_tts(scenes, os.path.dirname(chunk_path), st)
    finally:
        await _delete_gemini_file(client, video_file)

    if duration_status != "on_target":
        raise RuntimeError(
            "Measured TTS duration is outside the final-duration target after "
            f"calibration: {audio_seconds:.1f}s actual, target "
            f"{target_final_seconds:.1f}s (acceptable {duration_low:.1f}-"
            f"{duration_high:.1f}s from {chunk_dur:.1f}s input); "
            "refusing to render a video with the wrong duration"
        )

    out_w, out_h = compute_output_dims(
        *probe_video_info(chunk_path)[1:], int(st.get("resolution") or 1280)
    )
    pause_seconds = _settings_pause_seconds(st)
    audio_durations = [
        float(_clip_array(clip).size) / SAMPLE_RATE for clip in clips
    ]
    # Align each source scene to its measured narration clip.  The old
    # direct renderer kept almost the whole source chunk and padded audio
    # with silence, so neither the spoken content nor final duration was
    # actually controlled by the 90% setting.
    plan = _build_alignment_plan(
        scenes, audio_durations, chunk_dur, pause_seconds=pause_seconds
    )
    expected_final_seconds = sum(
        float(item.get("desired_output_seconds") or 0) for item in plan
    )
    if not _duration_is_on_target(expected_final_seconds, chunk_dur, st):
        raise RuntimeError(
            "Direct render plan is outside the final-duration target: "
            f"{expected_final_seconds:.1f}s planned, target "
            f"{target_final_seconds:.1f}s from {chunk_dur:.1f}s input"
        )

    master_pcm_path = os.path.join(os.path.dirname(chunk_path), "master_audio.pcm")
    pause_samples = int(round(pause_seconds * SAMPLE_RATE))
    with open(master_pcm_path, "wb") as pcm_f:
        for clip_index, clip in enumerate(clips):
            arr = _clip_array(clip)
            np.clip(arr * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)
            if pause_samples > 0 and clip_index < len(clips) - 1:
                np.zeros(pause_samples, dtype=np.int16).tofile(pcm_f)

    master_wav = os.path.join(os.path.dirname(chunk_path), "master_audio.wav")
    result = run_cmd([
        FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", master_pcm_path, master_wav,
    ], 120)
    if os.path.exists(master_pcm_path):
        os.remove(master_pcm_path)
    if result.returncode != 0 or not os.path.exists(master_wav):
        raise RuntimeError("master narration WAV creation failed")

    seg_files = []
    await tracker.update("🎬 ဗီဒီယို ပြင်ဆင်နေပါသည်...", force=True)

    for index, item in enumerate(plan):
        segment_path = os.path.join(
            os.path.dirname(chunk_path), f"seg_{index:03d}.mp4"
        )
        ffmpeg_filter = _worker_video_filter(
            out_w, out_h, st, float(item.get("freeze_seconds") or 0)
        )
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-ss", f"{float(item.get('source_start') or 0):.3f}",
            "-i", chunk_path,
            "-t", f"{max(0.04, float(item.get('input_seconds') or 0.04)):.3f}",
            "-an",
            "-vf", ffmpeg_filter,
            "-c:v", "libx264",
            "-preset", "superfast" if st.get("fast_mode") else "ultrafast",
            "-crf", str(int(st.get("quality") or 23)),
            segment_path,
        ], 300)

        if result.returncode != 0 or not os.path.exists(segment_path):
            raise RuntimeError(f"segment {index + 1} failed")
        seg_files.append(segment_path)

    concat_txt = os.path.join(os.path.dirname(chunk_path), "concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for tp in seg_files:
            escaped = tp.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    video_only = os.path.join(os.path.dirname(chunk_path), "video_only.mp4")
    result = await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", video_only,
    ], 300)
    if result.returncode != 0 or not os.path.exists(video_only):
        raise RuntimeError("direct video concat failed")

    out_chunk = str(output_path or "") or chunk_path.replace(".mp4", "_recap.mp4")
    if os.path.dirname(os.path.abspath(out_chunk)):
        os.makedirs(os.path.dirname(os.path.abspath(out_chunk)), exist_ok=True)
    result = await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y",
        "-i", video_only,
        "-i", master_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", f"{expected_final_seconds:.3f}",
        "-movflags", "+faststart",
        out_chunk,
    ], 300)
    if result.returncode != 0 or not os.path.exists(out_chunk):
        raise RuntimeError("direct final mux failed")

    final_chunk_duration, _, _ = probe_video_info(out_chunk)
    if not _duration_is_on_target(final_chunk_duration, chunk_dur, st):
        raise RuntimeError(
            "Direct rendered chunk duration is outside target: "
            f"{final_chunk_duration:.1f}s actual, target "
            f"{target_final_seconds:.1f}s (acceptable {duration_low:.1f}-"
            f"{duration_high:.1f}s from {chunk_dur:.1f}s input)"
        )
    logger.info(
        f"Chunk {chunk_index} direct final duration verified: "
        f"{final_chunk_duration:.2f}s / {chunk_dur:.2f}s = "
        f"{final_chunk_duration / chunk_dur * 100 if chunk_dur else 0:.1f}%"
    )

    cleanup_files = [video_only, master_wav, concat_txt] + seg_files
    if not keep_source_chunk:
        cleanup_files.insert(0, chunk_path)
    for f in cleanup_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    if extract_summary:
        new_summary = await extract_story_summary(scenes, client)
    else:
        prepared_summary = (
            (prepared_script or {}).get("story_context_after")
            if isinstance(prepared_script, dict)
            else {}
        )
        new_summary = (
            dict(prepared_summary)
            if isinstance(prepared_summary, dict)
            else {
                "characters": list((story_context or {}).get("characters") or []),
                "summary": str((story_context or {}).get("summary") or ""),
                "last_scene": str((scenes[-1].get("burmese") or "")) if scenes else "",
                "protagonist": new_protagonist,
                "pronoun": new_pronoun,
            }
        )
    if new_protagonist and not new_summary.get("protagonist"):
        new_summary["protagonist"] = new_protagonist
    if new_pronoun and not new_summary.get("pronoun"):
        new_summary["pronoun"] = new_pronoun

    if isinstance(prepared_script, dict):
        # Let the manual-session caller persist any measured-TTS changes
        # made during this render back into the stored per-part script.
        prepared_script.update(
            {
                "scenes": json.loads(json.dumps(scenes, ensure_ascii=False)),
                "cuts": json.loads(json.dumps(cuts, ensure_ascii=False)),
                "protagonist": new_protagonist,
                "pronoun": new_pronoun,
                "story_context_after": new_summary,
            }
        )
        metrics = dict(prepared_script.get("metrics") or {})
        metrics.update(
            {
                "actual_chars_after_calibration": int(script_total_chars(scenes)),
                "measured_narration_seconds": round(float(audio_seconds), 3),
                "rendered_duration_seconds": round(float(final_chunk_duration), 3),
                "duration_calibration_history": calibration_history,
            }
        )
        prepared_script["metrics"] = metrics
    return out_chunk, new_summary


async def generate_chunk_script(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    story_context: dict,
    chunk_index: int,
    tracker: ProgressTracker,
    settings: dict = None,
) -> dict:
    """Write and validate one part's first-person script, without TTS/render."""
    script, _summary = await build_chunk_recap(
        chunk_path,
        chunk_dur,
        global_start,
        global_end,
        story_context,
        chunk_index,
        tracker,
        settings,
        script_only=True,
        extract_summary=True,
        keep_source_chunk=True,
    )
    return script


async def render_chunk_recap_from_script(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    script: dict,
    chunk_index: int,
    tracker: ProgressTracker,
    settings: dict = None,
    *,
    output_path: str = "",
) -> tuple[str, dict]:
    """Render one clicked manual part from its previously stored script."""
    working_script = json.loads(json.dumps(script or {}, ensure_ascii=False))
    story_context = working_script.get("story_context_before") or {
        "characters": [],
        "summary": "",
        "last_scene": "",
        "protagonist": working_script.get("protagonist") or "",
        "pronoun": working_script.get("pronoun") or "",
    }
    out_chunk, _summary = await build_chunk_recap(
        chunk_path,
        chunk_dur,
        global_start,
        global_end,
        story_context,
        chunk_index,
        tracker,
        settings,
        prepared_script=working_script,
        extract_summary=False,
        keep_source_chunk=True,
        output_path=output_path,
    )
    return out_chunk, working_script


def _extract_source_segment_file(
    source_path: str,
    output_path: str,
    start: float,
    duration: float,
) -> bool:
    """Accurately extract one planned source window as a standalone MP4."""
    result = run_cmd(
        [
            FFMPEG_BIN, "-y",
            "-ss", f"{max(0.0, float(start)):.3f}",
            "-i", source_path,
            "-t", f"{max(0.01, float(duration)):.3f}",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ],
        1200,
    )
    return bool(
        result.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )


async def _prepare_source_segments(
    source_path: str,
    work_dir: str,
    total_duration: float,
    segment_seconds: float = SOURCE_SEGMENT_SECONDS,
) -> list[dict]:
    """Physically split a downloaded source into two-minute video files.

    Prefer FFmpeg's one-pass segment muxer: it partitions the source with
    no overlaps/gaps.  If keyframes/container make that unusable or the
    count is not the planned two-minute count, accurately extract every
    planned window instead.  A JSON manifest is retained in the job dir
    while the job runs so progress/logs can identify each part.
    """
    plan = build_source_segment_plan(total_duration, segment_seconds)
    if not plan:
        return []
    segment_dir = os.path.join(work_dir, "source_segments")
    os.makedirs(segment_dir, exist_ok=True)

    candidates = await asyncio.to_thread(
        _split_video_once, source_path, segment_dir, float(segment_seconds)
    )
    prepared: list[dict] = []
    cursor = 0.0
    for index, path in enumerate(candidates, start=1):
        duration, _, _ = probe_video_info(path)
        if duration <= 0:
            prepared = []
            break
        prepared.append(
            {
                "index": index,
                "path": path,
                "start": round(cursor, 3),
                "end": round(cursor + duration, 3),
                "duration": round(float(duration), 3),
            }
        )
        cursor += float(duration)

    total_tolerance = max(2.0, float(total_duration) * 0.02)
    max_candidate_duration = max(
        (float(item["duration"]) for item in prepared), default=0.0
    )
    # Stream-copy segmentation can only cut at keyframes.  Accept it only
    # when its boundaries are still the requested two-minute boundaries;
    # otherwise use the accurate extraction fallback below rather than
    # silently recapping, for example, a 170-second part as "2 minutes".
    boundary_tolerance = 2.0
    boundaries_ok = len(prepared) == len(plan) and all(
        abs(float(candidate["start"]) - float(planned["start"]))
        <= boundary_tolerance
        and abs(float(candidate["end"]) - float(planned["end"]))
        <= boundary_tolerance
        for candidate, planned in zip(prepared, plan)
    )
    one_pass_ok = bool(
        boundaries_ok
        and abs(cursor - float(total_duration)) <= total_tolerance
        and max_candidate_duration <= float(segment_seconds) + boundary_tolerance
    )
    if not one_pass_ok:
        shutil.rmtree(segment_dir, ignore_errors=True)
        os.makedirs(segment_dir, exist_ok=True)
        prepared = []
        for item in plan:
            output_path = os.path.join(
                segment_dir, f"source_part_{int(item['index']):04d}.mp4"
            )
            ok = await asyncio.to_thread(
                _extract_source_segment_file,
                source_path,
                output_path,
                float(item["start"]),
                float(item["duration"]),
            )
            actual_duration = probe_video_info(output_path)[0] if ok else 0.0
            if not ok or actual_duration <= 0:
                raise RuntimeError(
                    f"၂ မိနစ်ပိုင်း {item['index']}/{len(plan)} ဖြတ်၍မရပါ။"
                )
            prepared.append(
                {
                    **item,
                    "path": output_path,
                    "duration": round(float(actual_duration), 3),
                    "end": round(float(item["start"]) + float(actual_duration), 3),
                }
            )

    manifest_path = os.path.join(work_dir, "source_segments.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_segment_seconds": float(segment_seconds),
                    "source_duration_seconds": round(float(total_duration), 3),
                    "segments": [
                        {
                            "index": int(item["index"]),
                            "file": os.path.basename(str(item["path"])),
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                            "duration": float(item["duration"]),
                        }
                        for item in prepared
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"source segment manifest write failed: {e}")
    return prepared


async def _acquire_source_file(
    source_spec: dict,
    work_dir: str,
    tracker=None,
) -> tuple[str, float]:
    """Download either a Telegram upload or a URL and return path+duration."""
    source_spec = dict(source_spec or {})
    source_kind = str(source_spec.get("kind") or "url")
    os.makedirs(work_dir, exist_ok=True)
    if source_kind == "telegram_file":
        extension = _safe_video_extension(
            str(source_spec.get("file_name") or ""),
            str(source_spec.get("mime_type") or ""),
        )
        source_path = os.path.join(work_dir, f"source{extension}")
        if tracker:
            await tracker.update(
                f"📥 Telegram video ကို လက်ခံနေပါသည်... ({_telegram_attachment_label(source_spec)})",
                force=True,
            )
        try:
            expected_size = int(source_spec.get("file_size") or 0)
        except (TypeError, ValueError):
            expected_size = 0
        duration = await download_telegram_file(
            str(source_spec.get("file_id") or ""),
            source_path,
            tracker,
            expected_size=expected_size,
            raise_on_error=True,
        )
        return source_path, float(duration or 0)

    url = _extract_first_url(str(source_spec.get("url") or ""))
    if not url:
        raise SourceDownloadError(
            "Video link မတွေ့ပါ။ http/https link အပြည့်အစုံ ပို့ပါ။",
            detail="no http/https URL found in source spec",
        )
    if tracker:
        await tracker.update("🔍 ဗီဒီယို အချိန်စစ်ဆေးနေပါသည်...", force=True)
    # A failed duration probe is not fatal by itself: download_full has
    # additional direct/RedNote fallbacks and will surface the real error.
    try:
        probe_duration = await get_source_duration(url)
    except Exception as e:
        logger.warning("source duration probe failed: %s", _safe_error_text(e))
        probe_duration = 0.0
    source_path = os.path.join(work_dir, "source.mp4")
    if tracker:
        await tracker.update("📥 မူရင်းဗီဒီယို ဒေါင်းနေပါသည်...", force=True)
    duration = await download_full(
        url,
        source_path,
        tracker,
        probe_duration,
        raise_on_error=True,
    )
    return source_path, float(duration or 0)


def _make_telegram_preview_copy(source_path: str, output_path: str) -> bool:
    """Create a <=45MB preview for an original segment Telegram cannot accept."""
    duration, src_w, src_h = probe_video_info(source_path)
    if duration <= 0:
        return False
    target_bytes = int(MAX_TG_BYTES * 0.92)
    total_bps = int(target_bytes * 8 / max(1.0, duration))
    audio_bps = 96_000 if total_bps > 320_000 else 64_000
    video_bps = max(180_000, total_bps - audio_bps)
    out_w, out_h = compute_output_dims(src_w, src_h, 1280)
    video_filter = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result = run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            str(video_bps),
            "-maxrate",
            str(video_bps),
            "-bufsize",
            str(video_bps * 2),
            "-c:a",
            "aac",
            "-b:a",
            str(audio_bps),
            "-movflags",
            "+faststart",
            output_path,
        ],
        1200,
    )
    return bool(
        result.returncode == 0
        and os.path.exists(output_path)
        and 0 < os.path.getsize(output_path) <= MAX_TG_BYTES
    )


def _concat_video_parts(parts: list[str], output_path: str, concat_path: str) -> bool:
    """Join recap parts in order into one MP4 and verify the result."""
    valid_parts = [
        str(part)
        for part in (parts or [])
        if part and os.path.exists(str(part)) and os.path.getsize(str(part)) > 0
    ]
    if not valid_parts:
        return False
    with open(concat_path, "w", encoding="utf-8") as f:
        for part in valid_parts:
            escaped = part.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    result = run_cmd(
        [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ],
        1200,
    )
    return bool(
        result.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 0
    )


def _split_video_once(path: str, out_dir: str, segment_seconds: float) -> list[str]:
    """Split one MP4 at keyframes. Returns non-empty parts in order."""
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "part_%04d.mp4")
    result = run_cmd([
        FFMPEG_BIN, "-y",
        "-i", path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", f"{max(1.0, float(segment_seconds)):.3f}",
        "-reset_timestamps", "1",
        pattern,
    ], 1200)
    if result.returncode != 0:
        return []
    return sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if (
            f.startswith("part_")
            and f.endswith(".mp4")
            and os.path.getsize(os.path.join(out_dir, f)) > 0
        )
    )


async def split_for_telegram(path: str, work_dir: str) -> list[str]:
    """Return Telegram-sized parts, recursively resplitting any oversized part.

    A fixed 180-second split is not enough for high-bitrate files: a 3-minute
    part can still exceed MAX_TG_BYTES.  Estimate the needed segment length
    from this file's own bitrate, then verify every returned part and split
    again when necessary.
    """
    if not os.path.exists(path):
        return []
    if os.path.getsize(path) <= MAX_TG_BYTES:
        return [path]

    base_dir = os.path.join(work_dir, "tg_parts")
    os.makedirs(base_dir, exist_ok=True)
    parts = [path]

    for round_no in range(8):
        if all(os.path.getsize(part) <= MAX_TG_BYTES for part in parts):
            break

        new_parts: list[str] = []
        changed = False
        for part_index, part in enumerate(parts):
            size = os.path.getsize(part)
            if size <= MAX_TG_BYTES:
                new_parts.append(part)
                continue

            duration, _, _ = probe_video_info(part)
            if duration <= 0:
                new_parts.append(part)
                continue

            # 10% headroom for bitrate peaks and MP4 overhead.
            estimate = duration * (MAX_TG_BYTES * 0.90 / float(size))
            segment_seconds = max(1.0, min(duration / 2.0, estimate))
            out_dir = os.path.join(base_dir, f"r{round_no}_{part_index}")
            children = await asyncio.to_thread(
                _split_video_once, part, out_dir, segment_seconds
            )
            if len(children) < 2:
                # Keyframes may be sparse.  Try a shorter target once for
                # this file before giving up and returning it unchanged.
                retry_dir = os.path.join(base_dir, f"r{round_no}_{part_index}_retry")
                children = await asyncio.to_thread(
                    _split_video_once, part, retry_dir, max(0.5, segment_seconds / 2.0)
                )
            if len(children) >= 2:
                new_parts.extend(children)
                changed = True
            else:
                new_parts.append(part)

        parts = new_parts
        if not changed:
            break

    return parts or [path]


# =====================================================================
# n8n worker API implementation
# =====================================================================

class WorkerJobError(RuntimeError):
    pass


class WorkerCancelled(RuntimeError):
    pass


def _worker_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_safe_worker_job_id(job_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id or ""))


def _worker_job_dir(job_id: str) -> str:
    if not _is_safe_worker_job_id(job_id):
        raise WorkerJobError("invalid job_id")
    root = os.path.abspath(WORKER_JOBS_DIR)
    path = os.path.abspath(os.path.join(root, job_id))
    if os.path.commonpath([root, path]) != root:
        raise WorkerJobError("invalid job path")
    return path


def _worker_state_file(job_id: str) -> str:
    return os.path.join(_worker_job_dir(job_id), "state.json")


def _default_stage_map() -> dict:
    return {
        stage: {
            "status": "pending",
            "attempts": 0,
            "started_at": "",
            "finished_at": "",
            "error": "",
            "metrics": {},
        }
        for stage in WORKER_STAGES
    }


def _save_worker_job(state: dict) -> dict:
    job_id = str(state.get("job_id") or "")
    job_dir = _worker_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    state["job_dir"] = job_dir
    # A Telegram Cancel tap may persist ``cancel_requested=True`` while a
    # running stage still holds an older in-memory copy.  Never let that
    # older copy overwrite the user's cancel request on its next progress
    # save; otherwise the visible button appears to do nothing.
    if job_id and not state.get("cancel_requested"):
        try:
            existing_path = _worker_state_file(job_id)
            if os.path.exists(existing_path):
                with open(existing_path, encoding="utf-8") as existing_f:
                    existing = json.load(existing_f)
                if existing.get("cancel_requested"):
                    state["cancel_requested"] = True
                    CANCELLED_JOBS.add(job_id)
        except Exception:
            pass
    state["updated_at"] = _worker_now()
    tmp_path = _worker_state_file(job_id) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, _worker_state_file(job_id))
    return state


def _load_worker_job(job_id: str) -> dict:
    path = _worker_state_file(job_id)
    if not os.path.exists(path):
        raise WorkerJobError("worker job not found")
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    state["job_id"] = str(state.get("job_id") or job_id)
    state["job_dir"] = _worker_job_dir(state["job_id"])
    stages = _default_stage_map()
    saved = state.get("stages") or {}
    if isinstance(saved, dict):
        for name in WORKER_STAGES:
            if isinstance(saved.get(name), dict):
                stages[name].update(saved[name])
                if not isinstance(stages[name].get("metrics"), dict):
                    stages[name]["metrics"] = {}
    state["stages"] = stages
    if not isinstance(state.get("artifacts"), dict):
        state["artifacts"] = {}
    if not isinstance(state.get("metrics"), dict):
        state["metrics"] = {}
    if not isinstance(state.get("settings"), dict):
        state["settings"] = get_user_settings(state.get("chat_id"))
    state["settings"] = _with_required_coverage(state.get("settings") or {})
    state["settings"]["pov"] = "first"
    if not isinstance(state.get("story_context"), dict):
        state["story_context"] = {
            "characters": [], "summary": "", "last_scene": "",
            "protagonist": "", "pronoun": "",
        }
    return state


def _worker_path(state: dict, relative_path: str) -> str:
    job_dir = os.path.abspath(state.get("job_dir") or _worker_job_dir(str(state["job_id"])))
    path = os.path.abspath(os.path.join(job_dir, relative_path))
    if os.path.commonpath([job_dir, path]) != job_dir:
        raise WorkerJobError("invalid artifact path")
    return path


def _next_worker_stage(state: dict) -> str:
    for stage in WORKER_STAGES:
        if (state.get("stages") or {}).get(stage, {}).get("status") != "done":
            return stage
    return ""


def _public_worker_job(state: dict) -> dict:
    task = WORKER_TASKS.get(str(state.get("job_id") or ""))
    settings = state.get("settings") or {}
    return {
        "ok": True,
        "job_id": state.get("job_id"),
        "chat_id": state.get("chat_id"),
        "reply_msg_id": state.get("reply_msg_id"),
        "status_message_id": state.get("status_message_id"),
        "url": state.get("url"),
        "status": state.get("status", "queued"),
        "current_stage": state.get("current_stage") or "",
        "next_stage": _next_worker_stage(state),
        "progress": state.get("progress_text", ""),
        "error": state.get("error", ""),
        "task_running": bool(task and not task.done()),
        "created_at": state.get("created_at", ""),
        "updated_at": state.get("updated_at", ""),
        "stages": json.loads(json.dumps(state.get("stages") or {})),
        "artifacts": json.loads(json.dumps(state.get("artifacts") or {})),
        "metrics": json.loads(json.dumps(state.get("metrics") or {})),
        "selected_settings": {
            "coverage": settings.get("coverage"),
            "voice": settings.get("voice"),
            "resolution": settings.get("resolution"),
            "tone": settings.get("tone"),
            "rate": settings.get("rate"),
            "pause": settings.get("pause"),
            "split_tg": settings.get("split_tg"),
        },
    }


def _create_worker_job(
    job_id: str,
    chat_id: int,
    url: str,
    reply_msg_id: int = None,
    status_message_id: int = None,
    settings_override: dict = None,
) -> dict:
    job_id = str(job_id or "").strip()
    if not _is_safe_worker_job_id(job_id):
        raise WorkerJobError("invalid job_id")
    try:
        chat_id = int(chat_id)
    except (TypeError, ValueError) as e:
        raise WorkerJobError("invalid chat_id") from e
    url = _extract_first_url(str(url or "")) or str(url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise WorkerJobError("invalid url")

    active = ACTIVE_JOBS.get(chat_id)
    if active and active != job_id:
        raise WorkerJobError(f"another job is already active: {active}")

    if os.path.exists(_worker_state_file(job_id)):
        state = _load_worker_job(job_id)
        if int(state.get("chat_id") or 0) != chat_id or state.get("url") != url:
            raise WorkerJobError("job_id already belongs to a different request")
        changed = False
        if reply_msg_id and not state.get("reply_msg_id"):
            state["reply_msg_id"] = int(reply_msg_id)
            changed = True
        if status_message_id and not state.get("status_message_id"):
            state["status_message_id"] = int(status_message_id)
            changed = True
        if settings_override:
            settings = get_user_settings(chat_id)
            settings.update({
                k: v for k, v in settings_override.items() if k in DEFAULT_SETTINGS
            })
            settings = _with_required_coverage(settings)
            settings["pov"] = "first"
            if settings != state.get("settings"):
                state["settings"] = settings
                changed = True
        if state.get("status") not in ("delivered", "cancelled", "failed"):
            ACTIVE_JOBS[chat_id] = job_id
        return _save_worker_job(state) if changed else state

    settings = get_user_settings(chat_id)
    if settings_override:
        settings.update({
            k: v for k, v in settings_override.items() if k in DEFAULT_SETTINGS
        })
    settings = _with_required_coverage(settings)
    settings["pov"] = "first"
    os.makedirs(_worker_job_dir(job_id), exist_ok=True)
    state = {
        "job_id": job_id,
        "job_dir": _worker_job_dir(job_id),
        "chat_id": chat_id,
        "reply_msg_id": int(reply_msg_id) if reply_msg_id else None,
        "status_message_id": int(status_message_id) if status_message_id else None,
        "url": url,
        "status": "queued",
        "current_stage": "",
        "progress_text": "🤖 n8n workflow စတင်ရန် ပြင်ဆင်နေပါသည်...",
        "error": "",
        "cancel_requested": False,
        "terminal_stat_counted": False,
        "created_at": _worker_now(),
        "updated_at": _worker_now(),
        "stages": _default_stage_map(),
        "artifacts": {},
        "metrics": {},
        "settings": settings,
        "story_context": {
            "characters": [], "summary": "", "last_scene": "",
            "protagonist": "", "pronoun": "",
        },
    }
    ACTIVE_JOBS[chat_id] = job_id
    return _save_worker_job(state)


def _delete_worker_job_record(job_id: str, clear_active: bool = True):
    job_id = str(job_id or "")
    if not _is_safe_worker_job_id(job_id):
        return
    chat_id = None
    try:
        chat_id = int(_load_worker_job(job_id).get("chat_id") or 0) or None
    except Exception:
        pass
    WORKER_TASKS.pop(job_id, None)
    CANCELLED_JOBS.discard(job_id)
    if clear_active and chat_id and ACTIVE_JOBS.get(chat_id) == job_id:
        ACTIVE_JOBS.pop(chat_id, None)
    shutil.rmtree(_worker_job_dir(job_id), ignore_errors=True)


def _clear_worker_active(state: dict):
    try:
        chat_id = int(state.get("chat_id") or 0)
    except (TypeError, ValueError):
        return
    if ACTIVE_JOBS.get(chat_id) == state.get("job_id"):
        ACTIVE_JOBS.pop(chat_id, None)


def request_cancel_active_job(chat_id) -> str:
    """Cancel the chat's direct or worker job and persist worker intent."""
    try:
        chat_key = int(chat_id)
    except (TypeError, ValueError):
        return ""
    job_id = str(ACTIVE_JOBS.get(chat_key) or "")
    if not job_id:
        return ""
    CANCELLED_JOBS.add(job_id)
    try:
        manual_state = _load_manual_session(job_id)
        if int(manual_state.get("chat_id") or 0) == chat_key:
            manual_state["cancel_requested"] = True
            _save_manual_session(manual_state)
            manual_task = MANUAL_RENDER_TASKS.get(job_id)
            if manual_task and not manual_task.done():
                manual_task.cancel()
            return job_id
    except Exception:
        pass
    try:
        state = _load_worker_job(job_id)
        if int(state.get("chat_id") or 0) == chat_key:
            state["cancel_requested"] = True
            if state.get("status") in ("queued", "running", "stage_done"):
                state["status"] = "cancelling"
            _save_worker_job(state)
            task = WORKER_TASKS.get(job_id)
            if task and not task.done():
                task.cancel()
    except Exception:
        # Direct-pipeline jobs have no worker state file; the in-memory
        # CANCELLED_JOBS marker above is their cancellation signal.
        pass
    return job_id


async def request_cancel_active_job_async(chat_id) -> str:
    """Telegram-cancel a job and finalize an idle worker job immediately.

    If a worker stage task is running, cancelling that task lets
    ``_run_worker_stage`` perform terminal cleanup.  Between stages there
    is no task, however, so finalize here; otherwise the chat would remain
    blocked in ACTIVE_JOBS until n8n happened to call another endpoint.
    """
    job_id = request_cancel_active_job(chat_id)
    if not job_id:
        return ""

    needs_terminal = False
    state = None
    async with WORKER_JOB_LOCK:
        try:
            state = _load_worker_job(job_id)
        except Exception:
            return job_id
        task = WORKER_TASKS.get(job_id)
        if task and not task.done():
            return job_id
        if state.get("status") in ("delivered", "cancelled"):
            return job_id
        current_stage = state.get("current_stage") or _next_worker_stage(state)
        if current_stage:
            info = (state.get("stages") or {}).get(current_stage) or {}
            if info.get("status") not in ("done", "cancelled"):
                _set_worker_stage(
                    state, current_stage, "cancelled", error="cancelled"
                )
        state["cancel_requested"] = True
        _save_worker_job(state)
        needs_terminal = True

    if needs_terminal and state is not None:
        await _worker_terminal(state, "cancelled", "cancelled")
    return job_id


def _worker_tracker(state: dict):
    chat_id = state.get("chat_id")
    status_id = state.get("status_message_id")
    if not chat_id or not status_id:
        return None
    return ProgressTracker(int(chat_id), int(status_id), job_id=str(state.get("job_id") or ""))


async def _worker_progress(state: dict, text: str, tracker=None, force: bool = True):
    state["progress_text"] = text
    _save_worker_job(state)
    tracker = tracker or _worker_tracker(state)
    if tracker:
        await tracker.update(text, force=force)


def _worker_check_cancelled(state: dict):
    if state.get("cancel_requested") or str(state.get("job_id") or "") in CANCELLED_JOBS:
        raise WorkerCancelled("cancelled")


def _set_worker_stage(
    state: dict,
    stage: str,
    status: str,
    error: str = "",
    metrics: dict = None,
):
    info = state.setdefault("stages", _default_stage_map()).setdefault(stage, {})
    info["status"] = status
    if status == "running":
        info["started_at"] = _worker_now()
        info["finished_at"] = ""
        info["error"] = ""
        info["attempts"] = int(info.get("attempts") or 0) + 1
    if status in ("done", "failed", "cancelled"):
        info["finished_at"] = _worker_now()
    if error:
        info["error"] = error[:2000]
    if metrics:
        current = info.get("metrics") if isinstance(info.get("metrics"), dict) else {}
        current.update(metrics)
        info["metrics"] = current


def _merged_duration(intervals: list) -> float:
    clean = []
    for item in intervals or []:
        try:
            start = max(0.0, float(item[0]))
            end = max(start, float(item[1]))
        except Exception:
            continue
        if end > start:
            clean.append((start, end))
    clean.sort()
    total = 0.0
    cur_start = None
    cur_end = None
    for start, end in clean:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def _normalize_whisper_segments(raw_segments: list, offset: float = 0.0) -> list:
    normalized = []
    for seg in raw_segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start") or 0) + float(offset)
            end = float(seg.get("end") or start) + float(offset)
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        if not text or end <= start:
            continue
        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        })
    return normalized


def _group_transcript_segments(segments: list) -> list:
    """Deterministically group Whisper segments before Gemini sees them.

    Gemini is never trusted to invent timestamps.  Each group is one future
    narration scene and it keeps the exact Whisper segment IDs/start/end.
    """
    ordered = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            start = float(seg.get("start") or 0)
            end = float(seg.get("end") or start)
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or "").strip()
        if not text or end <= start:
            continue
        clean = dict(seg)
        clean["id"] = int(seg.get("id") or len(ordered) + 1)
        clean["start"] = start
        clean["end"] = end
        clean["text"] = text
        ordered.append(clean)
    ordered.sort(key=lambda s: (s["start"], s["end"], s["id"]))

    groups = []
    current = []
    current_speech = 0.0
    current_chars = 0

    def flush():
        nonlocal current, current_speech, current_chars
        if not current:
            return
        groups.append({
            "group_id": len(groups) + 1,
            "source_segment_ids": [int(s["id"]) for s in current],
            "start": round(float(current[0]["start"]), 3),
            "end": round(float(current[-1]["end"]), 3),
            "speech_seconds": round(float(current_speech), 3),
            "source_text": " ".join(str(s["text"]).strip() for s in current).strip(),
        })
        current = []
        current_speech = 0.0
        current_chars = 0

    for seg in ordered:
        duration = max(0.0, float(seg["end"]) - float(seg["start"]))
        if current:
            gap = float(seg["start"]) - float(current[-1]["end"])
            if gap >= 1.5 and current_speech >= 5.0:
                flush()
        current.append(seg)
        current_speech += duration
        current_chars += len(str(seg["text"]))
        span = float(current[-1]["end"]) - float(current[0]["start"])
        if current_speech >= 12.0 or span >= 20.0 or current_chars >= 650:
            flush()
    flush()
    return groups


def _batch_transcript_groups(groups: list, max_groups: int = 20, max_chars: int = 12000) -> list:
    batches = []
    batch = []
    batch_chars = 0
    for group in groups or []:
        chars = len(str(group.get("source_text") or ""))
        if batch and (len(batch) >= max_groups or batch_chars + chars > max_chars):
            batches.append(batch)
            batch = []
            batch_chars = 0
        batch.append(group)
        batch_chars += chars
    if batch:
        batches.append(batch)
    return batches


def _calculate_transcript_coverage(
    groups: list,
    scenes_by_group: dict,
    target_by_group: dict = None,
    min_group_char_ratio: float = SCRIPT_MIN_GROUP_CHAR_RATIO,
) -> dict:
    """Measure transcript coverage without treating one-liners as covered.

    Presence alone is not coverage: if per-group character targets are
    supplied, a scene must reach at least ``min_group_char_ratio`` of its
    own target before its source speech counts as covered.  The no-target
    form remains available for simple presence checks/tests.
    """
    targets = {}
    for key, value in (target_by_group or {}).items():
        try:
            targets[int(key)] = max(0, int(value or 0))
        except (TypeError, ValueError):
            continue

    total_speech = 0.0
    covered_speech = 0.0
    total_segments = 0
    covered_segments = 0
    missing_groups = []
    underfilled_groups = []
    for group in groups or []:
        gid = int(group.get("group_id") or 0)
        speech = max(0.0, float(group.get("speech_seconds") or 0))
        ids = [int(x) for x in (group.get("source_segment_ids") or [])]
        total_speech += speech
        total_segments += len(ids)
        text = str((scenes_by_group or {}).get(gid) or "").strip()
        if not text:
            missing_groups.append(gid)
            underfilled_groups.append(gid)
            continue

        target = targets.get(gid, 0)
        required_chars = (
            int(math.ceil(target * float(min_group_char_ratio or 0)))
            if target > 0
            else 1
        )
        if len(text) >= required_chars:
            covered_speech += speech
            covered_segments += len(ids)
        else:
            underfilled_groups.append(gid)
    speech_ratio = (covered_speech / total_speech) if total_speech > 0 else 0.0
    segment_ratio = (covered_segments / total_segments) if total_segments > 0 else 0.0
    return {
        "speech_seconds_total": round(total_speech, 3),
        "speech_seconds_covered": round(covered_speech, 3),
        "speech_coverage_ratio": round(speech_ratio, 4),
        "speech_coverage_percent": round(speech_ratio * 100.0, 2),
        "segment_count_total": total_segments,
        "segment_count_covered": covered_segments,
        "segment_coverage_ratio": round(segment_ratio, 4),
        "segment_coverage_percent": round(segment_ratio * 100.0, 2),
        "missing_group_ids": missing_groups,
        "underfilled_group_ids": underfilled_groups,
    }


async def _worker_stage_download(state: dict, tracker=None):
    source_path = _worker_path(state, "source.mp4")
    duration = 0.0
    if os.path.exists(source_path) and os.path.getsize(source_path) > 10_000:
        duration, width, height = probe_video_info(source_path)
    else:
        width = height = 0

    if duration <= 0:
        _worker_check_cancelled(state)
        await _worker_progress(state, WORKER_STAGE_LABELS["download"], tracker)
        probe_duration = await get_source_duration(str(state.get("url") or ""))
        try:
            duration = await download_full(
                str(state.get("url") or ""),
                source_path,
                tracker,
                probe_duration,
                raise_on_error=True,
            )
        except SourceDownloadError as e:
            detail = e.detail or e.user_message
            raise WorkerJobError(
                f"source download failed: {e.user_message} ({detail})"
            ) from e
        _worker_check_cancelled(state)
        if duration <= 0 or not os.path.exists(source_path):
            raise WorkerJobError("source download failed: no readable video duration")
        _, width, height = probe_video_info(source_path)

    metrics = {
        "duration_seconds": round(float(duration), 3),
        "size_bytes": os.path.getsize(source_path),
        "size_mb": round(os.path.getsize(source_path) / (1024 * 1024), 2),
        "width": int(width or 0),
        "height": int(height or 0),
    }
    state.setdefault("artifacts", {})["source"] = "source.mp4"
    state.setdefault("metrics", {})["source"] = metrics
    _set_worker_stage(state, "download", "done", metrics=metrics)
    _save_worker_job(state)
    await _worker_progress(
        state,
        f"✅ Download ပြီးပါပြီ ({fmt_time(duration)}, {metrics['size_mb']:.1f}MB)",
        tracker,
    )


def _extract_whisper_audio_chunks(source_path: str, out_dir: str, source_duration: float) -> list:
    os.makedirs(out_dir, exist_ok=True)
    if source_duration <= 0:
        source_duration = probe_video_info(source_path)[0]
    chunk_count = max(1, int(math.ceil(float(source_duration) / float(WHISPER_CHUNK_SECONDS))))
    chunks = []
    cumulative = 0.0
    for index in range(chunk_count):
        nominal_start = index * WHISPER_CHUNK_SECONDS
        nominal_duration = min(WHISPER_CHUNK_SECONDS, float(source_duration) - nominal_start)
        if nominal_duration <= 0.05:
            continue
        path = os.path.join(out_dir, f"audio_{index:04d}.mp3")
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            result = run_cmd([
                FFMPEG_BIN, "-y",
                "-ss", f"{nominal_start:.3f}",
                "-i", source_path,
                "-t", f"{nominal_duration:.3f}",
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "64k",
                path,
            ], 1200)
            if result.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) < 1000:
                detail = result.stderr.decode(errors="ignore")[-500:]
                raise WorkerJobError(f"Whisper audio extract failed: {detail}")
        actual_duration = probe_video_info(path)[0] or nominal_duration
        chunks.append({
            "path": path,
            "offset": round(cumulative, 3),
            "duration": round(float(actual_duration), 3),
        })
        cumulative += float(actual_duration)
    if not chunks:
        raise WorkerJobError("no audio chunks extracted for Whisper")
    return chunks


async def _transcribe_whisper_chunk(audio_path: str) -> dict:
    if not WHISPER_API_KEY:
        raise WorkerJobError(
            "Whisper credential missing: set WHISPER_API_KEY or OPENAI_API_KEY"
        )

    def read_bytes() -> bytes:
        with open(audio_path, "rb") as f:
            return f.read()

    payload = await asyncio.to_thread(read_bytes)
    endpoint = f"{WHISPER_API_BASE}/audio/transcriptions"
    form = [
        ("model", WHISPER_MODEL),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
    ]
    headers = {"Authorization": f"Bearer {WHISPER_API_KEY}"}
    last_error = ""
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    data=form,
                    files={
                        "file": (
                            os.path.basename(audio_path),
                            payload,
                            "audio/mpeg",
                        )
                    },
                )
            if 200 <= response.status_code < 300:
                data = response.json()
                if not isinstance(data, dict):
                    raise WorkerJobError("Whisper returned a non-object response")
                return data
            last_error = response.text[:1000]
            if response.status_code not in (408, 409, 429) and response.status_code < 500:
                break
        except WorkerJobError:
            raise
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < 2:
            await asyncio.sleep(2 ** attempt * 2)
    raise WorkerJobError(f"Whisper transcription failed: {last_error}")


async def _worker_stage_transcribe(state: dict, tracker=None):
    _worker_check_cancelled(state)
    source_path = _worker_path(state, (state.get("artifacts") or {}).get("source") or "source.mp4")
    if not os.path.exists(source_path):
        raise WorkerJobError("downloaded source.mp4 is missing")
    source_duration = float(
        ((state.get("metrics") or {}).get("source") or {}).get("duration_seconds") or 0
    ) or probe_video_info(source_path)[0]

    await _worker_progress(state, WORKER_STAGE_LABELS["transcribe"], tracker)
    chunks = await asyncio.to_thread(
        _extract_whisper_audio_chunks,
        source_path,
        _worker_path(state, "whisper_audio"),
        source_duration,
    )
    all_segments = []
    languages = []
    full_text_parts = []
    for index, chunk in enumerate(chunks):
        _worker_check_cancelled(state)
        data = await _transcribe_whisper_chunk(str(chunk["path"]))
        language = str(data.get("language") or "").strip()
        if language:
            languages.append(language)
        chunk_segments = _normalize_whisper_segments(
            data.get("segments") or [], float(chunk.get("offset") or 0)
        )
        chunk_text = str(data.get("text") or "").strip()
        if not chunk_segments and chunk_text:
            start = float(chunk.get("offset") or 0)
            end = start + max(0.1, float(chunk.get("duration") or 0.1))
            chunk_segments = [{"start": round(start, 3), "end": round(end, 3), "text": chunk_text}]
        if not chunk_text and chunk_segments:
            chunk_text = " ".join(s["text"] for s in chunk_segments)
        all_segments.extend(chunk_segments)
        if chunk_text:
            full_text_parts.append(chunk_text)
        await _worker_progress(
            state,
            f"{WORKER_STAGE_LABELS['transcribe']} ({index + 1}/{len(chunks)})",
            tracker,
        )

    all_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    for index, segment in enumerate(all_segments):
        segment["id"] = index + 1
    if not all_segments or not " ".join(full_text_parts).strip():
        raise WorkerJobError("Whisper returned no usable transcript text")

    speech_seconds = _merged_duration([
        (float(s["start"]), float(s["end"])) for s in all_segments
    ])
    transcript = {
        "model": WHISPER_MODEL,
        "language": max(set(languages), key=languages.count) if languages else "",
        "source_duration_seconds": round(float(source_duration), 3),
        "speech_duration_seconds": round(float(speech_seconds), 3),
        "text": " ".join(full_text_parts).strip(),
        "segments": all_segments,
    }
    transcript_path = _worker_path(state, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": WHISPER_MODEL,
        "language": transcript["language"],
        "segment_count": len(all_segments),
        "audio_chunk_count": len(chunks),
        "source_duration_seconds": round(float(source_duration), 3),
        "speech_duration_seconds": round(float(speech_seconds), 3),
        "speech_ratio_of_source": round(
            speech_seconds / source_duration, 4
        ) if source_duration > 0 else 0.0,
    }
    state.setdefault("artifacts", {})["transcript"] = "transcript.json"
    state.setdefault("metrics", {})["transcript"] = metrics
    _set_worker_stage(state, "transcribe", "done", metrics=metrics)
    _save_worker_job(state)
    await _worker_progress(
        state,
        f"✅ Transcript ပြီးပါပြီ — segment {len(all_segments)} ခု၊ speech {fmt_time(speech_seconds)}",
        tracker,
    )


def _rewrite_target_chars_for_duration(duration_seconds: float, settings: dict = None) -> int:
    coverage_ratio = _settings_coverage_ratio(settings)
    rate = _settings_script_rate(settings)
    speed = _settings_speed(settings)
    return max(
        0,
        int(float(duration_seconds or 0) * coverage_ratio * rate * speed),
    )


def _rewrite_target_chars_for_speech(speech_seconds: float, settings: dict = None) -> int:
    return _rewrite_target_chars_for_duration(speech_seconds, settings)


def _group_timeline_seconds(groups: list, source_duration: float = 0.0) -> dict:
    """Seconds of source timeline represented by each transcript group.

    A group owns the timeline from its start until the next group starts;
    the last group owns the remainder of the source.  This prevents long
    pauses/visual beats between lines of dialogue from disappearing from
    the duration target merely because Whisper marked them as non-speech.
    """
    ordered = sorted(
        [g for g in (groups or []) if isinstance(g, dict)],
        key=lambda g: (float(g.get("start") or 0), float(g.get("end") or 0)),
    )
    result = {}
    for index, group in enumerate(ordered):
        try:
            gid = int(group.get("group_id") or 0)
            start = max(0.0, float(group.get("start") or 0))
            own_end = max(start, float(group.get("end") or start))
            if index + 1 < len(ordered):
                limit = float(ordered[index + 1].get("start") or own_end)
            elif float(source_duration or 0) > 0:
                limit = float(source_duration)
            else:
                limit = own_end
        except (TypeError, ValueError):
            continue
        timeline = max(0.0, float(limit) - start)
        if timeline <= 0:
            timeline = max(0.0, float(group.get("speech_seconds") or 0))
        result[gid] = round(float(timeline), 3)
    return result


def _worker_tone_line(settings: dict = None) -> str:
    tone = str((settings or {}).get("tone") or "natural")
    return {
        "natural": "သဘာဝကျသော လေသံဖြင့် ရေးပါ။",
        "dramatic": "ဒရာမာဆန်ဆန်၊ တင်းမာမှုရှိသော လေသံဖြင့် ရေးပါ။",
        "emotional": "စိတ်လှုပ်ရှားဖွယ်၊ ခံစားချက်ပြင်းသော လေသံဖြင့် ရေးပါ။",
        "calm": "အေးအေးဆေးဆေး၊ တည်ငြိမ်သော လေသံဖြင့် ရေးပါ။",
    }.get(tone, "သဘာဝကျသော လေသံဖြင့် ရေးပါ။")


async def _worker_gemini_json(prompt: str, client=None):
    """Ask Gemini for JSON, trying the same preferred models as the direct bot."""
    if client is None:
        if not GEMINI_API_KEY:
            raise WorkerJobError("GEMINI_API_KEY is not configured")
        client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    for model in PREFERRED_MODELS:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            raw = str(getattr(response, "text", "") or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\s*```$", "", raw).strip()
            data = json.loads(raw)
            if isinstance(data, (dict, list)):
                return data
            raise ValueError("Gemini did not return a JSON object/list")
        except Exception as e:
            last_error = e
            logger.warning(f"Worker Gemini JSON failed with {model}: {str(e)[:250]}")
    raise WorkerJobError(f"Gemini rewrite failed: {last_error}")


def _parse_rewrite_response(data, allowed_group_ids) -> tuple[dict, str, str]:
    allowed = {int(gid) for gid in (allowed_group_ids or [])}
    protagonist = ""
    pronoun = ""
    raw_scenes = []
    if isinstance(data, dict):
        protagonist = str(data.get("protagonist") or "").strip()
        p = str(data.get("pronoun") or "").strip()
        if p in ("ကျွန်တော်", "ကျွန်မ"):
            pronoun = p
        if isinstance(data.get("scenes"), list):
            raw_scenes = data["scenes"]
    elif isinstance(data, list):
        raw_scenes = data

    scenes = {}
    for item in raw_scenes:
        if not isinstance(item, dict):
            continue
        try:
            gid = int(item.get("group_id"))
        except (TypeError, ValueError):
            continue
        if gid not in allowed:
            continue
        text = str(item.get("burmese") or item.get("text") or "").strip()
        if text:
            scenes[gid] = text
    return scenes, protagonist, pronoun


def _rewrite_prompt(
    groups: list,
    target_by_group: dict,
    story_context: dict,
    settings: dict,
    existing_by_group: dict = None,
    expand: bool = False,
    repair_first_person: bool = False,
    adjust_mode: str = "",
) -> str:
    st = _with_required_coverage(settings)
    context = story_context or {}
    pronoun = str(context.get("pronoun") or "").strip()
    protagonist = str(context.get("protagonist") or "").strip()
    voice_pronoun = _voice_first_person_pronoun(st)
    if pronoun in FIRST_PERSON_PRONOUNS:
        other = _other_first_person_pronoun(pronoun)
        pronoun_line = (
            f"ပြောနေသူသည် «{pronoun}» ကိုသာ ဆက်တိုက်သုံးပါ။ "
            f"ကိုယ့်ကို «{other}» ဟု လုံးဝမပြောင်းပါနှင့်။"
        )
    else:
        pronoun_line = (
            "အဓိကဇာတ်ကောင်က အမျိုးသားဖြစ်လျှင် «ကျွန်တော်»၊ အမျိုးသမီးဖြစ်လျှင် «ကျွန်မ» သုံးပါ။ "
            f"ကျား/မကို transcript မှ မသေချာလျှင် «{voice_pronoun}» သုံးပြီး မပြောင်းပါနှင့်။"
        )

    existing = {}
    for key, value in (existing_by_group or {}).items():
        try:
            existing[int(key)] = str(value or "").strip()
        except (TypeError, ValueError):
            continue

    payload_groups = []
    for group in groups or []:
        gid = int(group.get("group_id") or 0)
        target_chars = int(target_by_group.get(gid) or 0)
        current_text = existing.get(gid) or ""
        item = {
            "group_id": gid,
            "start": group.get("start"),
            "end": group.get("end"),
            "source_timeline_seconds": group.get("timeline_seconds"),
            "source_segment_ids": group.get("source_segment_ids") or [],
            "target_burmese_chars": target_chars,
            "source_text": str(group.get("source_text") or ""),
        }
        if expand or repair_first_person or adjust_mode:
            item["current_burmese_chars"] = len(current_text)
            item["minimum_additional_chars"] = max(0, target_chars - len(current_text))
            item["desired_burmese_chars"] = target_chars
            if current_text:
                item["existing_burmese"] = current_text
        payload_groups.append(item)

    if repair_first_person:
        task_line = (
            "FIRST-PERSON REPAIR — existing_burmese ၏ အကြောင်းအရာအားလုံးကို မလျော့ဘဲ "
            "အဓိကဇာတ်ကောင် ကိုယ်တိုင် ပြန်ပြောနေသလို scene အပြည့်အစုံ ပြန်ရေးပါ။"
        )
    elif adjust_mode == "shorten":
        task_line = (
            "existing_burmese ကို အဓိကအချက်အားလုံးထိန်းပြီး desired_burmese_chars "
            "ပမာဏအထိ တိုအောင် scene အပြည့်အစုံ ပြန်ရေးပါ။"
        )
    elif expand or adjust_mode == "expand":
        task_line = (
            "existing_burmese ကို အပိုင်းအစလေးမျှသာ ပြန်မပေးဘဲ scene အပြည့်အစုံကို ပြန်ရေးပြီး "
            "source_text ထဲက ကျန်နေသေးသော အချက်အလက်အားလုံးဖြင့် target ပြည့်အောင် ချဲ့ရေးပါ။"
        )
    else:
        task_line = "source_text ကို Burmese ပထမလူ ဇာတ်ကြောင်းအဖြစ် ပြန်ရေးပါ။"

    if adjust_mode == "shorten":
        length_line = (
            "Scene တစ်ခုစီ၏ နောက်ဆုံးအရှည်သည် `desired_burmese_chars` ၏ 95%–105% "
            "အတွင်း ဖြစ်ရမည်။ အဆုံးခန်းတွေကို မဖျက်ဘဲ scene တိုင်းကို အချိုးကျ ချုံ့ပါ။"
        )
    else:
        length_line = (
            "Scene တစ်ခုစီအတွက် Burmese စာသား အရှည်သည် `target_burmese_chars` ထက် မနည်းရပါ။ "
            "\"ခန့်\" မဟုတ်ပါ — ၎င်းသည် အနိမ့်ဆုံးလိုအပ်သော ပမာဏဖြစ်သည်။ Expand mode တွင်မူ "
            "နောက်ဆုံး scene တစ်ခုစီသည် `current_burmese_chars + minimum_additional_chars` "
            "ပြည့်ရမည်ဖြစ်ပြီး အပိုင်းအစလေးမဟုတ်ဘဲ scene အပြည့်အစုံကို ပြန်ပေးရမည်။"
        )
    context_json = json.dumps({
        "protagonist": protagonist,
        "pronoun": pronoun,
        "characters": context.get("characters") or [],
        "previous_summary": str(context.get("summary") or "")[-2500:],
        "last_scene": str(context.get("last_scene") or "")[-1000:],
    }, ensure_ascii=False)
    groups_json = json.dumps(payload_groups, ensure_ascii=False)

    return f"""
သင်သည် Whisper transcript ကို မြန်မာဘာသာဖြင့် အဓိကဇာတ်ကောင် ကိုယ်တိုင် ပြန်ပြောပြသလို ပထမလူဇာတ်ကြောင်းအဖြစ် ပြန်ရေးပေးရမည်။

အဓိကတာဝန်: {task_line}
လေသံ: {_worker_tone_line(st)}
နာမ်စား: {pronoun_line}

လိုက်နာရမည်များ:
- Output သည် Burmese သာ ဖြစ်ရမည်။ မူရင်းဘာသာစကားကို ဘာသာပြန်ပါ။
- အကျဉ်းချုပ် မဟုတ်ပါ။ Group တစ်ခုစီရှိ အကြောင်းအရာအားလုံးနီးပါးကို အပြည့်အဝ ပြန်ပြောပါ။ နောက်ဆုံး final video ၏ အမှန်တကယ်ကြာချိန်သည် မူရင်း video ၏ {st.get('coverage')}% ဖြစ်ရမည်။
- Input group တစ်ခုအတွက် output scene တစ်ခုတည်းသာ ထုတ်ပါ။ group_id ကို အတိအကျ ကူးထည့်ပါ။ Group များကို မပေါင်းပါနှင့်၊ မခွဲပါနှင့်၊ အစဉ်မပြောင်းပါနှင့်။
- start/end သို့မဟုတ် source_segment_ids ကို ကိုယ်တိုင် မဖန်တီးပါနှင့်။ ၎င်းတို့ကို worker က ထည့်ပေးမည်။
- {length_line}
- JSON မထုတ်ခင် scene တစ်ခုချင်းစီ၏ စာလုံးအရေအတွက်ကို ကိုယ့်ဘာသာစစ်ပါ။ Target မပြည့်သေးပါက source_text ထဲရှိ အရာ၊ နေရာ၊ အကြောင်းရင်း၊ စကားပြောချက်၊ တုံ့ပြန်မှုနှင့် အကျိုးဆက်များကို မထပ်ဘဲ ပိုမိုအသေးစိတ်ထည့်ပြီးမှ ထုတ်ပါ။ စကားလုံးထပ်ခြင်း၊ filler ထပ်ခြင်းဖြင့် အရှည်မဖြည့်ပါနှင့်။
- ပြောနေသူမှာ အဓိကဇာတ်ကောင် ကိုယ်တိုင်ဖြစ်သည်။ ကိုယ့်ကိုယ်ကို နာမည် သို့မဟုတ် «သူ/သူမ» ဟု မညွှန်းပါနှင့်။
- **Scene တိုင်း၏ ပထမဝါကျ သို့မဟုတ် အစပိုင်းတွင် ရွေးထားသော နာမ်စား အနည်းဆုံးတစ်ကြိမ် မဖြစ်မနေပါရမည်။** pronoun field ထည့်ရုံနဲ့ မလုံလောက်ပါ — burmese စာသားထဲမှာတင် ပါရပါမည်။
- အဓိကဇာတ်ကောင် မပါဝင်သော အဖြစ်အပျက်များကို မကျော်ပါနှင့်။ «အဲဒီတုန်းက ကျွန်တော်/ကျွန်မ မသိသေးပေမယ့်…» သို့မဟုတ် «နောက်မှသိလိုက်ရတာက…» ပုံစံဖြင့် retrospective ရေးပါ။
- Transcript တွင် မပါသော အဖြစ်အပျက်၊ အမည်၊ အကြောင်းရင်းများကို မတီထွင်ပါနှင့်။
- Dialogue ၏ အဓိပ္ပာယ်နှင့် ပြောသူ/ကြားသူကို ထိန်းသိမ်းပါ။ တိုက်ရိုက်စာလုံးချင်း ကူးနိုင်သလောက် တိကျကျ Burmese အဖြစ် ပြန်ရေးပါ။
- Visual-only အချက်ကို transcript မှ မသိနိုင်လျှင် မှန်းမရေးပါနှင့်။
- Scene text ထဲတွင် timestamp, heading, bullet, quotation label သို့မဟုတ် ရှင်းလင်းချက် မထည့်ပါနှင့်။ TTS ဖတ်မည့် စာသားသာရေးပါ။

ယခင်ဇာတ်လမ်းအခြေအနေ:
{context_json}

ပြန်ရေးရမည့် transcript groups:
{groups_json}

JSON ကိုသာ အောက်ပါပုံစံဖြင့် ပြန်ပေးပါ:
{{
  "protagonist": "အဓိကဇာတ်ကောင်အမည် သို့မဟုတ် ကွက်လပ်",
  "pronoun": "ကျွန်တော် သို့မဟုတ် ကျွန်မ",
  "scenes": [
    {{"group_id": 1, "burmese": "TTS ဖတ်မည့် Burmese ပထမလူစာသား"}}
  ]
}}
""".strip()


async def _repair_worker_first_person_scenes(
    groups: list,
    scenes_by_group: dict,
    target_by_group: dict,
    story_context: dict,
    settings: dict,
    client,
    state: dict = None,
    tracker=None,
    rounds: int = FIRST_PERSON_REPAIR_ROUNDS,
) -> tuple:
    """Rewrite worker scenes whose actual text is not first-person.

    Returns ``(selected_pronoun, remaining_bad_group_ids)`` and mutates
    ``scenes_by_group``/``story_context`` only with validated replacements.
    """
    texts = [str(v or "") for v in (scenes_by_group or {}).values()]
    selected = _choose_first_person_pronoun(
        texts,
        preferred=str((story_context or {}).get("pronoun") or ""),
        settings=settings,
    )
    story_context["pronoun"] = selected
    group_by_id = {}
    for group in groups or []:
        try:
            group_by_id[int(group.get("group_id") or 0)] = group
        except (TypeError, ValueError, AttributeError):
            continue

    def bad_group_ids():
        bad = []
        for gid, group in group_by_id.items():
            text = str(scenes_by_group.get(gid) or "").strip()
            if text and not _scene_is_first_person(text, selected):
                bad.append(gid)
        return sorted(bad)

    for repair_round in range(max(0, int(rounds or 0))):
        bad_ids = bad_group_ids()
        if not bad_ids:
            break
        if state is not None:
            await _worker_progress(
                state,
                "👤 ပထမလူအပြောပုံစံ မဖြစ်သေးသော scene များ ပြန်ပြင်နေပါသည်... "
                f"({repair_round + 1}/{rounds})",
                tracker,
            )
        accepted = 0
        bad_groups = [group_by_id[gid] for gid in bad_ids if gid in group_by_id]
        for batch in _batch_transcript_groups(
            bad_groups, max_groups=5, max_chars=3000
        ):
            allowed = [int(g["group_id"]) for g in batch]
            prompt = _rewrite_prompt(
                batch,
                target_by_group,
                story_context,
                settings,
                existing_by_group=scenes_by_group,
                repair_first_person=True,
            )
            try:
                data = await _worker_gemini_json(prompt, client)
            except Exception as e:
                logger.warning(f"Worker first-person repair failed: {e}")
                continue
            parsed, response_protagonist, _ = _parse_rewrite_response(
                data, allowed
            )
            if response_protagonist and not story_context.get("protagonist"):
                story_context["protagonist"] = response_protagonist
            for gid, new_text in parsed.items():
                old_text = str(scenes_by_group.get(int(gid)) or "")
                if (
                    _scene_is_first_person(new_text, selected)
                    and len(new_text.strip()) >= max(
                        1, int(len(old_text.strip()) * 0.9)
                    )
                ):
                    scenes_by_group[int(gid)] = new_text.strip()
                    accepted += 1
        if accepted <= 0:
            break
    return selected, bad_group_ids()


async def _worker_stage_rewrite(state: dict, tracker=None):
    _worker_check_cancelled(state)
    if not GEMINI_API_KEY:
        raise WorkerJobError("GEMINI_API_KEY is not configured")
    settings = _with_required_coverage(state.get("settings") or {})
    settings["pov"] = "first"

    transcript_rel = (state.get("artifacts") or {}).get("transcript") or "transcript.json"
    transcript_path = _worker_path(state, transcript_rel)
    if not os.path.exists(transcript_path):
        raise WorkerJobError("transcript.json is missing; run transcribe first")
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments") or []
    groups = _group_transcript_segments(segments)
    if not groups:
        raise WorkerJobError("transcript contains no usable timestamped segments")

    speech_seconds = sum(float(g.get("speech_seconds") or 0) for g in groups)
    if speech_seconds <= 0:
        speech_seconds = float(transcript.get("speech_duration_seconds") or 0)

    source_duration = float(transcript.get("source_duration_seconds") or 0)
    if source_duration <= 0:
        source_duration = float(
            ((state.get("metrics") or {}).get("source") or {}).get("duration_seconds") or 0
        )
    transcript_first_start = min(float(g.get("start") or 0) for g in groups)
    transcript_last_end = max(float(g.get("end") or 0) for g in groups)
    transcript_span_seconds = max(0.0, transcript_last_end - transcript_first_start)

    # A transcript that stops after, say, 35 seconds of a 9-minute source
    # cannot support a 90% recap.  Reject that upstream truncation here
    # instead of producing a 35-second "complete" video.
    if (
        source_duration >= 60
        and transcript_last_end < source_duration * TRANSCRIPT_MIN_SPAN_RATIO
    ):
        raise WorkerJobError(
            "Whisper transcript does not span the source video: "
            f"last segment ends at {transcript_last_end:.1f}s of "
            f"{source_duration:.1f}s ({transcript_last_end / source_duration * 100:.1f}%); "
            "refusing to build a shortened recap"
        )

    # The user's coverage setting means narration duration as a percentage
    # of the *source video*, not merely of Whisper's active-speech seconds.
    # Fall back to speech only for legacy test transcripts without a source
    # duration.
    target_basis_seconds = source_duration if source_duration > 0 else speech_seconds
    target_chars = _rewrite_target_chars_for_duration(target_basis_seconds, settings)
    timeline_by_group = _group_timeline_seconds(groups, source_duration)
    for group in groups:
        gid = int(group["group_id"])
        group["timeline_seconds"] = float(timeline_by_group.get(gid) or 0)

    allocation_weights = {}
    for group in groups:
        gid = int(group["group_id"])
        if source_duration > 0:
            weight = float(timeline_by_group.get(gid) or 0)
        else:
            weight = float(group.get("speech_seconds") or 0)
        allocation_weights[gid] = max(0.0, weight)
    total_weight = sum(allocation_weights.values())

    if target_chars > 0 and total_weight > 0:
        minimum = 35 if target_chars >= 35 * len(groups) else max(1, target_chars // len(groups))
        target_by_group = {
            int(g["group_id"]): max(
                minimum,
                int(round(target_chars * allocation_weights[int(g["group_id"])] / total_weight)),
            )
            for g in groups
        }
    else:
        target_by_group = {int(g["group_id"]): 80 for g in groups}
        target_chars = sum(target_by_group.values())

    await _worker_progress(state, WORKER_STAGE_LABELS["rewrite"], tracker)
    client = genai.Client(api_key=GEMINI_API_KEY)
    story_context = {
        "characters": [], "summary": "", "last_scene": "",
        "protagonist": "", "pronoun": "",
    }
    scenes_by_group: dict[int, str] = {}
    batches = _batch_transcript_groups(groups, max_groups=10, max_chars=6000)

    for batch_index, batch in enumerate(batches, start=1):
        _worker_check_cancelled(state)
        allowed = [int(g["group_id"]) for g in batch]
        prompt = _rewrite_prompt(
            batch, target_by_group, story_context, settings,
            existing_by_group=scenes_by_group,
        )
        data = await _worker_gemini_json(prompt, client)
        parsed, response_protagonist, response_pronoun = _parse_rewrite_response(data, allowed)
        for gid, text in parsed.items():
            if text.strip():
                scenes_by_group[int(gid)] = text.strip()

        detected_pronoun = _detect_pronoun(" ".join(parsed.values()))
        if response_protagonist and not story_context.get("protagonist"):
            story_context["protagonist"] = response_protagonist
        if not story_context.get("pronoun"):
            story_context["pronoun"] = detected_pronoun or response_pronoun

        batch_scenes = [
            {"burmese": scenes_by_group[int(g["group_id"])]}
            for g in batch
            if scenes_by_group.get(int(g["group_id"]))
        ]
        if batch_scenes:
            try:
                summary = await extract_story_summary(batch_scenes, client)
            except Exception as e:
                logger.warning(f"Worker story summary update failed: {e}")
                summary = {}
            if isinstance(summary, dict) and summary:
                merged = dict(story_context)
                merged.update(summary)
                if story_context.get("protagonist"):
                    merged["protagonist"] = story_context["protagonist"]
                if story_context.get("pronoun"):
                    merged["pronoun"] = story_context["pronoun"]
                story_context = {
                    "characters": merged.get("characters") or [],
                    "summary": str(merged.get("summary") or ""),
                    "last_scene": str(merged.get("last_scene") or ""),
                    "protagonist": str(merged.get("protagonist") or ""),
                    "pronoun": str(merged.get("pronoun") or ""),
                }
            if not story_context.get("last_scene") and batch_scenes:
                story_context["last_scene"] = batch_scenes[-1]["burmese"]

        await _worker_progress(
            state,
            f"{WORKER_STAGE_LABELS['rewrite']} ({batch_index}/{len(batches)})",
            tracker,
        )

    # Retry only the missing groups so coverage can reach the selected target.
    for _retry_round in range(2):
        missing = [
            g for g in groups if not str(scenes_by_group.get(int(g["group_id"])) or "").strip()
        ]
        if not missing:
            break
        for batch in _batch_transcript_groups(missing, max_groups=6, max_chars=3500):
            _worker_check_cancelled(state)
            allowed = [int(g["group_id"]) for g in batch]
            prompt = _rewrite_prompt(
                batch, target_by_group, story_context, settings,
                existing_by_group=scenes_by_group,
            )
            data = await _worker_gemini_json(prompt, client)
            parsed, response_protagonist, response_pronoun = _parse_rewrite_response(data, allowed)
            for gid, text in parsed.items():
                if text.strip():
                    scenes_by_group[int(gid)] = text.strip()
            if response_protagonist and not story_context.get("protagonist"):
                story_context["protagonist"] = response_protagonist
            if not story_context.get("pronoun"):
                story_context["pronoun"] = (
                    _detect_pronoun(" ".join(parsed.values())) or response_pronoun
                )

    # Validate/repair the actual narration text before length top-ups.  A
    # correct JSON pronoun field does not prove the Burmese text itself is
    # first-person; the previous system accepted third-person summaries.
    selected_pronoun, _ = await _repair_worker_first_person_scenes(
        groups,
        scenes_by_group,
        target_by_group,
        story_context,
        settings,
        client,
        state=state,
        tracker=tracker,
    )
    story_context["pronoun"] = selected_pronoun

    # Expand underfilled groups using only omitted facts from their own
    # source.  Small batches are deliberate: asking Gemini to expand dozens
    # of scenes at once reliably produces another summary.  Stop early only
    # when the hard length gate below will reject the job anyway.
    minimum_chars = _minimum_script_chars(target_chars)
    previous_total_chars = -1
    stagnant_rounds = 0
    for topup_round in range(SCRIPT_TOPUP_ATTEMPTS):
        _worker_check_cancelled(state)
        total_chars = sum(len(v.strip()) for v in scenes_by_group.values())
        if total_chars >= minimum_chars:
            break
        if total_chars <= previous_total_chars:
            stagnant_rounds += 1
            if stagnant_rounds >= 2:
                break
        else:
            stagnant_rounds = 0
        previous_total_chars = total_chars

        candidates = []
        for group in groups:
            gid = int(group["group_id"])
            current_text = str(scenes_by_group.get(gid) or "").strip()
            wanted = int(target_by_group.get(gid) or 0)
            if wanted > 0 and len(current_text) < wanted * SCRIPT_MIN_CHAR_RATIO:
                ratio = len(current_text) / wanted if wanted else 0.0
                weight = float(
                    group.get("timeline_seconds")
                    or group.get("speech_seconds")
                    or 0
                )
                candidates.append((ratio, -weight, group))
        if not candidates:
            break

        candidates.sort(key=lambda item: (item[0], item[1]))
        topup_groups = [item[2] for item in candidates[:40]]
        await _worker_progress(
            state,
            "⚠️ Script ပမာဏ "
            f"{total_chars / max(1, target_chars) * 100:.0f}% ပဲရှိသေးလို့ "
            f"ကျန်အသေးစိတ်များ ထပ်ဖြည့်နေပါသည်... "
            f"({topup_round + 1}/{SCRIPT_TOPUP_ATTEMPTS})",
            tracker,
        )
        for batch in _batch_transcript_groups(
            topup_groups, max_groups=5, max_chars=3000
        ):
            _worker_check_cancelled(state)
            allowed = [int(g["group_id"]) for g in batch]
            prompt = _rewrite_prompt(
                batch, target_by_group, story_context, settings,
                existing_by_group=scenes_by_group, expand=True,
            )
            data = await _worker_gemini_json(prompt, client)
            parsed, _, _ = _parse_rewrite_response(data, allowed)
            for gid, new_text in parsed.items():
                old_text = str(scenes_by_group.get(int(gid)) or "")
                if (
                    len(new_text.strip()) > len(old_text.strip())
                    and _scene_is_first_person(new_text, selected_pronoun)
                ):
                    scenes_by_group[int(gid)] = new_text.strip()

    selected_pronoun, bad_first_person_groups = (
        await _repair_worker_first_person_scenes(
            groups,
            scenes_by_group,
            target_by_group,
            story_context,
            settings,
            client,
            state=state,
            tracker=tracker,
            rounds=2,
        )
    )
    story_context["pronoun"] = selected_pronoun
    coverage_metrics = _calculate_transcript_coverage(
        groups, scenes_by_group, target_by_group=target_by_group
    )
    total_chars = sum(len(v.strip()) for v in scenes_by_group.values())
    required_ratio = _settings_coverage_ratio(settings)
    script_rate = _settings_script_rate(settings) * _settings_speed(settings)
    estimated_spoken_seconds = total_chars / script_rate if script_rate else 0.0
    target_narration_seconds = target_basis_seconds * required_ratio
    coverage_metrics.update({
        "required_speech_coverage_ratio": round(required_ratio, 4),
        "required_speech_coverage_percent": round(required_ratio * 100.0, 2),
        "source_duration_seconds": round(float(source_duration), 3),
        "speech_duration_seconds": round(float(speech_seconds), 3),
        "transcript_span_seconds": round(float(transcript_span_seconds), 3),
        "transcript_last_end_seconds": round(float(transcript_last_end), 3),
        "target_basis_seconds": round(float(target_basis_seconds), 3),
        "target_basis": "source_duration" if source_duration > 0 else "speech_duration",
        "target_narration_seconds": round(float(target_narration_seconds), 3),
        "target_burmese_chars": int(target_chars),
        "minimum_acceptable_burmese_chars": int(minimum_chars),
        "actual_burmese_chars": int(total_chars),
        "char_target_ratio": round(total_chars / target_chars, 4) if target_chars else 0.0,
        "char_target_percent": round(total_chars / target_chars * 100.0, 2) if target_chars else 0.0,
        "estimated_spoken_seconds": round(float(estimated_spoken_seconds), 3),
        "estimated_narration_ratio_of_source": round(
            estimated_spoken_seconds / source_duration, 4
        ) if source_duration > 0 else 0.0,
        "group_count": len(groups),
        "scene_count": len(scenes_by_group),
        "language": str(transcript.get("language") or ""),
    })
    ordered_scene_texts = [
        str(scenes_by_group.get(int(g.get("group_id") or 0)) or "")
        for g in groups
        if str(scenes_by_group.get(int(g.get("group_id") or 0)) or "").strip()
    ]
    first_person_metrics = _first_person_metrics(
        ordered_scene_texts, selected_pronoun
    )
    coverage_metrics["first_person"] = first_person_metrics
    coverage_metrics["first_person_pronoun"] = selected_pronoun
    coverage_metrics["bad_first_person_group_ids"] = bad_first_person_groups
    if not scenes_by_group:
        raise WorkerJobError("Gemini returned no usable rewritten scenes")
    if bad_first_person_groups:
        raise WorkerJobError(
            "Script is not first-person in every scene after repairs: "
            f"pronoun={selected_pronoun}, bad_group_ids={bad_first_person_groups}; "
            "refusing to deliver a third-person recap"
        )
    if float(coverage_metrics.get("speech_coverage_ratio") or 0) < required_ratio:
        raise WorkerJobError(
            "Transcript coverage is below target: "
            f"{coverage_metrics.get('speech_coverage_percent')}% < {required_ratio * 100:.1f}% "
            f"(missing={coverage_metrics.get('missing_group_ids')}, "
            f"underfilled={coverage_metrics.get('underfilled_group_ids')})"
        )
    if total_chars < minimum_chars:
        raise WorkerJobError(
            "Script length below duration target after top-ups: "
            f"{total_chars}/{target_chars} chars "
            f"(~{estimated_spoken_seconds:.1f}s narration; target "
            f"~{target_narration_seconds:.1f}s from {target_basis_seconds:.1f}s basis, "
            f"coverage {required_ratio * 100:.0f}%); "
            "refusing to render a shortened recap"
        )

    scenes = []
    for group in groups:
        gid = int(group["group_id"])
        text = str(scenes_by_group.get(gid) or "").strip()
        if not text:
            continue
        scenes.append({
            "group_id": gid,
            "source_segment_ids": [int(x) for x in (group.get("source_segment_ids") or [])],
            "start": float(group.get("start") or 0),
            "end": float(group.get("end") or group.get("start") or 0),
            "speech_seconds": float(group.get("speech_seconds") or 0),
            "timeline_seconds": float(group.get("timeline_seconds") or 0),
            "burmese": text,
        })

    script = {
        "story_context": story_context,
        "coverage": coverage_metrics,
        "target_by_group": {str(k): int(v) for k, v in target_by_group.items()},
        "groups": groups,
        "scenes": scenes,
    }
    script_path = _worker_path(state, "script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)

    state["story_context"] = story_context
    state.setdefault("artifacts", {})["script"] = "script.json"
    state.setdefault("metrics", {})["script"] = coverage_metrics
    _set_worker_stage(state, "rewrite", "done", metrics=coverage_metrics)
    _save_worker_job(state)
    await _worker_progress(
        state,
        "✅ Script ပြီးပါပြီ — transcript coverage "
        f"{coverage_metrics.get('speech_coverage_percent', 0):.1f}%, စာလုံး {total_chars} လုံး",
        tracker,
    )


def _worker_settings(state: dict) -> dict:
    settings = _with_required_coverage(state.get("settings") or {})
    settings["pov"] = "first"
    return settings


def _load_worker_json(state: dict, artifact_key: str, default_name: str) -> dict:
    relative = (state.get("artifacts") or {}).get(artifact_key) or default_name
    path = _worker_path(state, str(relative))
    if not os.path.exists(path):
        raise WorkerJobError(f"required worker artifact is missing: {relative}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise WorkerJobError(f"invalid worker artifact: {relative}")
    return data


def _write_worker_json(state: dict, relative_path: str, data: dict):
    path = _worker_path(state, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def _worker_script_groups(script: dict, scenes: list) -> list:
    groups = []
    seen = set()
    for group in (script or {}).get("groups") or []:
        if not isinstance(group, dict):
            continue
        try:
            gid = int(group.get("group_id") or 0)
        except (TypeError, ValueError):
            continue
        if gid <= 0 or gid in seen:
            continue
        seen.add(gid)
        clean = dict(group)
        clean["group_id"] = gid
        groups.append(clean)
    if groups:
        return sorted(groups, key=lambda g: int(g.get("group_id") or 0))

    # Legacy script artifacts may not retain transcript groups.  Build a
    # minimal group from each scene so duration repair can still operate;
    # Gemini is instructed to preserve the existing scene's facts.
    for index, scene in enumerate(scenes or [], start=1):
        if not isinstance(scene, dict):
            continue
        try:
            gid = int(scene.get("group_id") or index)
            start = float(scene.get("start") or 0)
            end = float(scene.get("end") or start)
        except (TypeError, ValueError):
            continue
        groups.append({
            "group_id": gid,
            "source_segment_ids": scene.get("source_segment_ids") or [],
            "start": start,
            "end": end,
            "speech_seconds": max(0.0, end - start),
            "timeline_seconds": float(scene.get("timeline_seconds") or max(0.0, end - start)),
            "source_text": str(scene.get("burmese") or ""),
        })
    return groups


def _allocate_worker_duration_targets(
    scenes: list,
    groups: list,
    desired_total_chars: int,
    mode: str,
) -> dict:
    group_by_id = {}
    for group in groups or []:
        try:
            group_by_id[int(group.get("group_id") or 0)] = group
        except (TypeError, ValueError, AttributeError):
            continue
    weights = {}
    currents = {}
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        try:
            gid = int(scene.get("group_id") or 0)
        except (TypeError, ValueError):
            continue
        if gid <= 0:
            continue
        current = len(str(scene.get("burmese") or "").strip())
        currents[gid] = current
        group = group_by_id.get(gid) or {}
        weight = float(
            group.get("timeline_seconds")
            or scene.get("timeline_seconds")
            or group.get("speech_seconds")
            or scene.get("speech_seconds")
            or 0
        )
        if weight <= 0:
            try:
                weight = max(
                    0.0,
                    float(scene.get("end") or 0) - float(scene.get("start") or 0),
                )
            except (TypeError, ValueError):
                weight = 0.0
        weights[gid] = weight if weight > 0 else float(max(1, current))
    total_weight = sum(weights.values())
    desired_total = max(0, int(desired_total_chars or 0))
    targets = {}
    for gid, current in currents.items():
        allocated = (
            int(round(desired_total * weights[gid] / total_weight))
            if total_weight > 0 else current
        )
        if mode == "expand":
            targets[gid] = max(current, allocated)
        elif mode == "shorten":
            targets[gid] = min(current, allocated)
        else:
            targets[gid] = allocated
    return targets


async def _worker_adjust_scenes_for_duration(
    state: dict,
    script: dict,
    scenes: list,
    desired_by_group: dict,
    mode: str,
    settings: dict,
    tracker=None,
) -> tuple:
    """Use Gemini to expand scenes toward measured-duration targets."""
    groups = _worker_script_groups(script, scenes)
    if not groups:
        return scenes, [], _voice_first_person_pronoun(settings)
    scenes_by_group = {}
    scene_by_group = {}
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        try:
            gid = int(scene.get("group_id") or 0)
        except (TypeError, ValueError):
            continue
        if gid > 0:
            scenes_by_group[gid] = str(scene.get("burmese") or "").strip()
            scene_by_group[gid] = scene

    story_context = dict((script or {}).get("story_context") or state.get("story_context") or {})
    selected = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
        preferred=str(story_context.get("pronoun") or ""),
        settings=settings,
    )
    story_context["pronoun"] = selected

    candidates = []
    for group in groups:
        gid = int(group.get("group_id") or 0)
        current = len(str(scenes_by_group.get(gid) or ""))
        desired = int(desired_by_group.get(gid) or current)
        text = str(scenes_by_group.get(gid) or "")
        needs_duration = (
            (mode == "expand" and desired > int(current * 1.02))
            or (mode == "shorten" and desired < int(current * 0.98))
        )
        if needs_duration or not _scene_is_first_person(text, selected):
            candidates.append(group)
    if not candidates:
        return scenes, [], selected

    if not GEMINI_API_KEY:
        raise WorkerJobError("GEMINI_API_KEY is not configured for duration calibration")
    client = genai.Client(api_key=GEMINI_API_KEY)
    changed = []
    for batch in _batch_transcript_groups(candidates, max_groups=5, max_chars=3000):
        _worker_check_cancelled(state)
        allowed = [int(g["group_id"]) for g in batch]
        prompt = _rewrite_prompt(
            batch,
            desired_by_group,
            story_context,
            settings,
            existing_by_group=scenes_by_group,
            expand=(mode == "expand"),
            adjust_mode=mode,
        )
        data = await _worker_gemini_json(prompt, client)
        parsed, response_protagonist, _ = _parse_rewrite_response(data, allowed)
        if response_protagonist and not story_context.get("protagonist"):
            story_context["protagonist"] = response_protagonist
        for gid, new_text in parsed.items():
            gid = int(gid)
            old_text = str(scenes_by_group.get(gid) or "")
            desired = int(desired_by_group.get(gid) or len(old_text))
            new_len = len(new_text.strip())
            if not _scene_is_first_person(new_text, selected):
                continue
            if not _scene_is_first_person(old_text, selected):
                progress = new_len >= max(1, int(len(old_text) * 0.9))
            elif mode == "expand":
                progress = new_len > len(old_text) and abs(new_len - desired) < abs(len(old_text) - desired)
            elif mode == "shorten":
                progress = new_len < len(old_text) and abs(new_len - desired) < abs(len(old_text) - desired)
            else:
                progress = new_len != len(old_text)
            if progress:
                scenes_by_group[gid] = new_text.strip()
                changed.append(gid)

    updated = []
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        new_scene = dict(scene)
        try:
            gid = int(scene.get("group_id") or 0)
        except (TypeError, ValueError):
            gid = 0
        if gid in scenes_by_group:
            new_scene["burmese"] = scenes_by_group[gid]
        updated.append(new_scene)
    script["story_context"] = story_context
    return updated, sorted(set(changed)), selected


async def _resynthesize_changed_worker_clips(
    scenes: list,
    clips: list,
    changed_group_ids: list,
    work_dir: str,
    settings: dict,
) -> list:
    if not changed_group_ids:
        return list(clips or [])
    changed = {int(gid) for gid in changed_group_ids}
    indexes = []
    changed_scenes = []
    for index, scene in enumerate(scenes or []):
        try:
            gid = int(scene.get("group_id") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if gid in changed:
            indexes.append(index)
            changed_scenes.append(scene)
    if not changed_scenes:
        return list(clips or [])
    new_clips = await synthesize_tts(changed_scenes, work_dir, settings)
    if new_clips is None or len(new_clips) < len(changed_scenes):
        raise WorkerJobError("Gemini TTS returned too few recalibration clips")
    updated = list(clips or [])
    if len(updated) < len(scenes or []):
        updated.extend([None] * (len(scenes or []) - len(updated)))
    for index, clip in zip(indexes, list(new_clips)[:len(changed_scenes)]):
        updated[index] = clip
    return updated


def _persist_worker_calibrated_script(
    state: dict,
    script: dict,
    scenes: list,
    measurements: dict,
    calibration_history: list,
    selected_pronoun: str,
    source_seconds: float,
    settings: dict,
):
    script["scenes"] = scenes
    coverage = dict(script.get("coverage") or {})
    total_chars = script_total_chars(scenes)
    target_final, low, high = _final_duration_window(source_seconds, settings)
    coverage.update({
        "actual_burmese_chars": int(total_chars),
        "estimated_spoken_seconds": round(float(measurements.get("speech_seconds") or 0), 3),
        "measured_narration_seconds": round(float(measurements.get("narration_seconds") or 0), 3),
        "measured_chars_per_second": measurements.get("measured_chars_per_second"),
        "target_final_seconds": round(float(target_final), 3),
        "acceptable_final_min_seconds": round(float(low), 3),
        "acceptable_final_max_seconds": round(float(high), 3),
        "duration_calibration_rounds": max(0, len(calibration_history or []) - 1),
        "duration_calibration_history": calibration_history or [],
        "first_person": _first_person_metrics(
            [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
            selected_pronoun,
        ),
        "first_person_pronoun": selected_pronoun,
    })
    script["coverage"] = coverage
    _write_worker_json(state, "script.json", script)
    state.setdefault("metrics", {})["script"] = coverage
    return coverage


async def _worker_stage_tts(state: dict, tracker=None):
    _worker_check_cancelled(state)
    settings = _worker_settings(state)
    script = _load_worker_json(state, "script", "script.json")
    scenes = sort_scenes(script.get("scenes") or [])
    if not scenes:
        raise WorkerJobError("script.json contains no scenes; run rewrite first")

    await _worker_progress(state, WORKER_STAGE_LABELS["tts"], tracker)
    clips = await synthesize_tts(scenes, str(state.get("job_dir") or ""), settings)
    _worker_check_cancelled(state)
    if clips is None:
        raise WorkerJobError("Gemini TTS returned no clips")
    clips = list(clips)
    if len(clips) < len(scenes):
        raise WorkerJobError(
            f"Gemini TTS returned {len(clips)} clips for {len(scenes)} scenes"
        )
    clips = clips[:len(scenes)]

    clips_dir = _worker_path(state, "audio_clips")
    os.makedirs(clips_dir, exist_ok=True)
    try:
        pause_seconds = float(settings.get("pause") or 0)
    except (TypeError, ValueError):
        pause_seconds = PAUSE_AFTER_SCENE
    pause_samples = max(0, int(round(pause_seconds * SAMPLE_RATE)))
    pause_bytes = (np.zeros(pause_samples, dtype="<i2").tobytes() if pause_samples else b"")

    # Calibrate against measured TTS seconds, not only the initial
    # character estimate.  This is the direct fix for scripts that looked
    # long on paper but produced only a fraction of the source duration.
    script_coverage_initial = script.get("coverage") or {}
    source_seconds_initial = float(
        script_coverage_initial.get("source_duration_seconds")
        or ((state.get("metrics") or {}).get("source") or {}).get("duration_seconds")
        or 0
    )
    selected_pronoun = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
        preferred=str(
            (script.get("story_context") or state.get("story_context") or {}).get("pronoun")
            or ""
        ),
        settings=settings,
    )
    calibration_history = []
    if source_seconds_initial > 0:
        target_final_seconds, final_low, final_high = _final_duration_window(
            source_seconds_initial, settings
        )
        # A script artifact from an older rewrite can be third-person even
        # when its duration happens to be right.  Repair it before measuring
        # the final calibration so TTS never voices that version.
        initial_bad_fp = _first_person_bad_scene_indexes(
            [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
            selected_pronoun,
        )
        if initial_bad_fp:
            desired_current = {}
            for scene in scenes:
                try:
                    gid = int(scene.get("group_id") or 0)
                except (TypeError, ValueError, AttributeError):
                    continue
                if gid > 0:
                    desired_current[gid] = len(str(scene.get("burmese") or "").strip())
            await _worker_progress(
                state,
                "👤 Script ကို အဓိကဇာတ်ကောင် ကိုယ်တိုင်ပြောပုံစံဖြင့် ပြန်ပြင်နေပါသည်...",
                tracker,
            )
            scenes, changed_fp, selected_pronoun = (
                await _worker_adjust_scenes_for_duration(
                    state,
                    script,
                    scenes,
                    desired_current,
                    "expand",
                    settings,
                    tracker,
                )
            )
            if changed_fp:
                clips = await _resynthesize_changed_worker_clips(
                    scenes,
                    clips,
                    changed_fp,
                    str(state.get("job_dir") or ""),
                    settings,
                )

        for calibration_round in range(DURATION_CALIBRATION_ROUNDS + 1):
            _worker_check_cancelled(state)
            measurements = _narration_measurements(scenes, clips, settings)
            round_silent = int(measurements.get("silent_count") or 0)
            if not scenes or round_silent == len(scenes):
                raise WorkerJobError("Gemini TTS produced only silent audio")
            if round_silent > max(1, int(len(scenes) * 0.25)):
                raise WorkerJobError(
                    f"Gemini TTS produced too many silent scenes: {round_silent}/{len(scenes)}"
                )
            narration_now = float(measurements.get("narration_seconds") or 0)
            calibration_history.append({
                "round": calibration_round,
                "total_chars": int(
                    measurements.get("spoken_chars_measured")
                    or script_total_chars(scenes)
                ),
                "spoken_seconds": round(float(measurements.get("speech_seconds") or 0), 3),
                "narration_seconds": round(narration_now, 3),
                "measured_chars_per_second": measurements.get("measured_chars_per_second"),
                "target_final_seconds": round(target_final_seconds, 3),
                "ratio_of_source": round(
                    narration_now / source_seconds_initial, 4
                ) if source_seconds_initial > 0 else 0.0,
            })
            if final_low <= narration_now <= final_high:
                break
            if calibration_round >= DURATION_CALIBRATION_ROUNDS:
                break

            total_pauses = float(measurements.get("pause_total_seconds") or 0)
            desired_speech_seconds = max(0.1, target_final_seconds - total_pauses)
            measured_cps = measurements.get("measured_chars_per_second")
            if not measured_cps or float(measured_cps) <= 0:
                measured_cps = _settings_script_rate(settings) * _settings_speed(settings)
            current_chars = int(
                measurements.get("spoken_chars_measured")
                or script_total_chars(scenes)
            )
            desired_total_chars = int(round(desired_speech_seconds * float(measured_cps)))
            if narration_now < final_low:
                ratio_target = int(round(
                    current_chars
                    * (desired_speech_seconds / max(0.01, float(measurements.get("speech_seconds") or 0)))
                    * 1.03
                ))
                desired_total_chars = max(desired_total_chars, ratio_target)
                groups_for_targets = _worker_script_groups(script, scenes)
                desired_by_group = _allocate_worker_duration_targets(
                    scenes,
                    groups_for_targets,
                    desired_total_chars,
                    "expand",
                )
                await _worker_progress(
                    state,
                    "⏱️ အသံကြာချိန် နည်းနေပါသဖြင့် script ကို ပြန်ချဲ့ပြီး "
                    f"ပြန်အသံသွင်းနေပါသည်... ({calibration_round + 1}/{DURATION_CALIBRATION_ROUNDS})",
                    tracker,
                )
                old_total = script_total_chars(scenes)
                scenes, changed_groups, selected_pronoun = (
                    await _worker_adjust_scenes_for_duration(
                        state,
                        script,
                        scenes,
                        desired_by_group,
                        "expand",
                        settings,
                        tracker,
                    )
                )
                if not changed_groups or script_total_chars(scenes) <= old_total:
                    calibration_history[-1]["calibration_stalled"] = True
                    break
                clips = await _resynthesize_changed_worker_clips(
                    scenes,
                    clips,
                    changed_groups,
                    str(state.get("job_dir") or ""),
                    settings,
                )
            else:
                await _worker_progress(
                    state,
                    "⏱️ အသံကြာချိန် များနေပါသဖြင့် script ကို အစမှ အဆုံးအထိ "
                    f"အချိုးကျ ခ်ဳံ့နေပါသည်... ({calibration_round + 1}/{DURATION_CALIBRATION_ROUNDS})",
                    tracker,
                )
                before_text = {
                    int(s.get("group_id") or 0): str(s.get("burmese") or "")
                    for s in scenes if isinstance(s, dict)
                }
                scenes = trim_scenes_proportionally(scenes, desired_total_chars)
                changed_groups = []
                for scene in scenes:
                    try:
                        gid = int(scene.get("group_id") or 0)
                    except (TypeError, ValueError, AttributeError):
                        continue
                    if gid > 0 and before_text.get(gid) != str(scene.get("burmese") or ""):
                        changed_groups.append(gid)
                if not changed_groups:
                    calibration_history[-1]["calibration_stalled"] = True
                    break
                clips = await _resynthesize_changed_worker_clips(
                    scenes,
                    clips,
                    changed_groups,
                    str(state.get("job_dir") or ""),
                    settings,
                )

        final_measurements = _narration_measurements(scenes, clips, settings)
        final_narration = float(final_measurements.get("narration_seconds") or 0)
        bad_fp_groups = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            text = str(scene.get("burmese") or "")
            if text.strip() and not _scene_is_first_person(text, selected_pronoun):
                bad_fp_groups.append(scene.get("group_id"))
        if bad_fp_groups:
            raise WorkerJobError(
                "Narration script is not first-person in every scene after "
                f"calibration: pronoun={selected_pronoun}, bad_group_ids={bad_fp_groups}; "
                "refusing to render a third-person recap"
            )
        if not (final_low <= final_narration <= final_high):
            raise WorkerJobError(
                "TTS narration could not be calibrated to the required final "
                f"duration: {_duration_status(final_narration, source_seconds_initial, settings)}; "
                f"calibration_history={calibration_history}; refusing to render "
                "a video that is not 90% of the source duration"
            )
        _persist_worker_calibrated_script(
            state,
            script,
            scenes,
            final_measurements,
            calibration_history,
            selected_pronoun,
            source_seconds_initial,
            settings,
        )

    entries = []
    total_speech_samples = 0
    silent_count = 0
    narration_path = _worker_path(state, "narration.wav")
    with wave.open(narration_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        for index, (scene, clip) in enumerate(zip(scenes, clips), start=1):
            arr = np.asarray(clip if clip is not None else [], dtype=np.float32).reshape(-1)
            pcm = np.clip(arr * 32767.0, -32768, 32767).astype("<i2")
            filename = f"scene_{index:04d}.pcm"
            pcm.tofile(os.path.join(clips_dir, filename))
            max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
            silent = bool(arr.size == 0 or max_abs < 0.001)
            silent_count += int(silent)
            total_speech_samples += int(arr.size)
            entries.append({
                "scene_index": index,
                "group_id": scene.get("group_id"),
                "file": f"audio_clips/{filename}",
                "samples": int(arr.size),
                "duration_seconds": round(float(arr.size) / SAMPLE_RATE, 6),
                "silent": silent,
                "max_abs": round(max_abs, 6),
            })
            wav_file.writeframes(pcm.tobytes())
            if index < len(scenes) and pause_bytes:
                wav_file.writeframes(pause_bytes)

    if not entries or silent_count == len(entries):
        raise WorkerJobError("Gemini TTS produced only silent audio")
    if silent_count > max(1, int(len(entries) * 0.25)):
        raise WorkerJobError(
            f"Gemini TTS produced too many silent scenes: {silent_count}/{len(entries)}"
        )

    narration_seconds = (
        total_speech_samples / SAMPLE_RATE
        + pause_seconds * max(0, len(entries) - 1)
    )
    script_coverage = script.get("coverage") or {}
    source_seconds = float(
        script_coverage.get("source_duration_seconds")
        or ((state.get("metrics") or {}).get("source") or {}).get("duration_seconds")
        or 0
    )
    coverage_ratio = _settings_coverage_ratio(settings)
    speed = _settings_speed(settings)
    expected_narration_seconds, minimum_narration_seconds, maximum_narration_seconds = (
        _final_duration_window(source_seconds, settings)
        if source_seconds > 0
        else (0.0, 0.0, 0.0)
    )
    if (
        expected_narration_seconds > 0
        and not (minimum_narration_seconds <= narration_seconds <= maximum_narration_seconds)
    ):
        raise WorkerJobError(
            "TTS narration outside required final-duration window: "
            f"{_duration_status(narration_seconds, source_seconds, settings)}; "
            f"coverage {coverage_ratio * 100:.0f}%, speed {speed}; "
            "refusing to render a recap that is not 90% of the source duration"
        )

    audio_data = {
        "sample_rate": SAMPLE_RATE,
        "pause_seconds": pause_seconds,
        "narration_file": "narration.wav",
        "clips": entries,
        "total_speech_seconds": round(total_speech_samples / SAMPLE_RATE, 6),
        "total_narration_seconds": round(narration_seconds, 6),
        "expected_narration_seconds": round(expected_narration_seconds, 6),
        "minimum_acceptable_narration_seconds": round(minimum_narration_seconds, 6),
        "maximum_acceptable_narration_seconds": round(maximum_narration_seconds, 6),
        "target_final_seconds": round(expected_narration_seconds, 6),
        "acceptable_final_min_seconds": round(minimum_narration_seconds, 6),
        "acceptable_final_max_seconds": round(maximum_narration_seconds, 6),
        "duration_calibration_history": calibration_history,
        "measured_chars_per_second": (
            calibration_history[-1].get("measured_chars_per_second")
            if calibration_history else None
        ),
        "first_person_pronoun": selected_pronoun,
        "first_person": _first_person_metrics(
            [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
            selected_pronoun,
        ),
        "silent_scene_count": silent_count,
        "voice": settings.get("voice"),
    }
    _write_worker_json(state, "audio.json", audio_data)
    metrics = {
        "scene_count": len(entries),
        "silent_scene_count": silent_count,
        "speech_seconds": round(total_speech_samples / SAMPLE_RATE, 3),
        "narration_seconds": round(narration_seconds, 3),
        "narration_ratio_of_source": round(
            narration_seconds / source_seconds, 4
        ) if source_seconds > 0 else 0.0,
        "expected_narration_seconds": round(expected_narration_seconds, 3),
        "minimum_acceptable_narration_seconds": round(minimum_narration_seconds, 3),
        "maximum_acceptable_narration_seconds": round(maximum_narration_seconds, 3),
        "target_final_seconds": round(expected_narration_seconds, 3),
        "duration_calibration_rounds": max(0, len(calibration_history) - 1),
        "narration_ratio_of_expected": round(
            narration_seconds / expected_narration_seconds, 4
        ) if expected_narration_seconds > 0 else 0.0,
        "voice": settings.get("voice"),
    }
    state.setdefault("artifacts", {})["audio"] = "audio.json"
    state.setdefault("artifacts", {})["narration"] = "narration.wav"
    state.setdefault("metrics", {})["audio"] = metrics
    _set_worker_stage(state, "tts", "done", metrics=metrics)
    _save_worker_job(state)
    await _worker_progress(
        state,
        f"✅ အသံသွင်းပြီးပါပြီ — {fmt_time(narration_seconds)} ({settings.get('voice')})",
        tracker,
    )


def _build_alignment_plan(scenes: list, audio_durations: list, source_duration: float, pause_seconds: float = 0.0) -> list:
    """Build start-aligned video cuts without changing narration speed.

    - If narration is shorter than its source scene, only the beginning of
      that scene is kept.
    - If narration is longer, following source footage may be borrowed, but
      only until the next selected scene starts.
    - A frozen final frame is used only for any remaining shortage.
    """
    if len(scenes) != len(audio_durations):
        raise WorkerJobError("scene/audio count mismatch while building render plan")
    source_duration = float(source_duration or 0)
    if source_duration <= 0:
        raise WorkerJobError("source duration is unknown")
    try:
        pause_seconds = max(0.0, float(pause_seconds or 0))
    except (TypeError, ValueError):
        pause_seconds = 0.0

    plan = []
    output_cursor = 0.0
    for index, (scene, raw_audio_duration) in enumerate(zip(scenes, audio_durations)):
        try:
            start = float(scene.get("start") or 0)
            end = float(scene.get("end") or start)
            audio_duration = max(0.0, float(raw_audio_duration or 0))
        except (TypeError, ValueError, AttributeError) as e:
            raise WorkerJobError(f"invalid scene timing at index {index + 1}") from e

        start = max(0.0, min(start, max(0.0, source_duration - 0.04)))
        end = max(start, min(end, source_duration))
        if index + 1 < len(scenes):
            try:
                limit = float(scenes[index + 1].get("start") or source_duration)
            except (TypeError, ValueError, AttributeError):
                limit = source_duration
        else:
            limit = source_duration
        limit = max(start, min(float(limit), source_duration))

        pause_after = pause_seconds if index + 1 < len(scenes) else 0.0
        desired_seconds = audio_duration + pause_after
        available_seconds = max(0.0, limit - start)
        input_seconds = min(desired_seconds, available_seconds)
        if input_seconds <= 0:
            input_seconds = min(0.04, source_duration)
            if start + input_seconds > source_duration:
                start = max(0.0, source_duration - input_seconds)
        freeze_seconds = max(0.0, desired_seconds - input_seconds)
        original_scene_seconds = max(0.0, end - start)
        borrowed_seconds = max(0.0, input_seconds - original_scene_seconds)

        plan.append({
            "scene_index": index + 1,
            "group_id": scene.get("group_id"),
            "source_segment_ids": scene.get("source_segment_ids") or [],
            "source_start": round(start, 6),
            "source_end": round(end, 6),
            "source_limit": round(limit, 6),
            "narration_seconds": round(audio_duration, 6),
            "pause_after_seconds": round(pause_after, 6),
            "desired_output_seconds": round(desired_seconds, 6),
            "input_seconds": round(input_seconds, 6),
            "borrowed_seconds": round(borrowed_seconds, 6),
            "freeze_seconds": round(freeze_seconds, 6),
            "output_start": round(output_cursor, 6),
            "output_end": round(output_cursor + desired_seconds, 6),
        })
        output_cursor += desired_seconds
    return plan


def _worker_video_filter(out_w: int, out_h: int, settings: dict, freeze_seconds: float = 0.0) -> str:
    filters = []
    if settings.get("shield"):
        filters.extend(["crop=iw*0.96:ih*0.96", "hflip"])
    filters.extend([
        "fps=25",
        f"scale={int(out_w)}:{int(out_h)}:force_original_aspect_ratio=increase:flags=bicubic",
        f"crop={int(out_w)}:{int(out_h)}",
    ])
    if settings.get("color"):
        filters.append("eq=contrast=1.08:saturation=1.12:gamma=1.02")
    if float(freeze_seconds or 0) > 0.01:
        filters.append(f"tpad=stop_mode=clone:stop_duration={float(freeze_seconds):.3f}")
    filters.append("format=yuv420p")
    return ",".join(filters)


async def _worker_stage_render(state: dict, tracker=None):
    _worker_check_cancelled(state)
    settings = _worker_settings(state)
    script = _load_worker_json(state, "script", "script.json")
    audio = _load_worker_json(state, "audio", "audio.json")
    scenes = sort_scenes(script.get("scenes") or [])
    clips = sorted(
        [c for c in (audio.get("clips") or []) if isinstance(c, dict)],
        key=lambda c: int(c.get("scene_index") or 0),
    )
    if not scenes:
        raise WorkerJobError("script.json contains no scenes")
    if len(clips) != len(scenes):
        raise WorkerJobError(f"scene/audio count mismatch: {len(scenes)} scenes, {len(clips)} clips")

    source_rel = (state.get("artifacts") or {}).get("source") or "source.mp4"
    source_path = _worker_path(state, str(source_rel))
    narration_rel = (state.get("artifacts") or {}).get("narration") or audio.get("narration_file") or "narration.wav"
    narration_path = _worker_path(state, str(narration_rel))
    if not os.path.exists(source_path):
        raise WorkerJobError("downloaded source.mp4 is missing")
    if not os.path.exists(narration_path):
        raise WorkerJobError("narration.wav is missing")

    source_duration, src_w, src_h = probe_video_info(source_path)
    if source_duration <= 0:
        source_duration = float(
            ((state.get("metrics") or {}).get("source") or {}).get("duration_seconds") or 0
        )
    if source_duration <= 0:
        raise WorkerJobError("could not determine source video duration")
    out_w, out_h = compute_output_dims(src_w, src_h, int(settings.get("resolution") or 1280))
    audio_durations = [float(c.get("duration_seconds") or 0) for c in clips]
    try:
        pause_seconds = float(audio.get("pause_seconds", settings.get("pause") or 0) or 0)
    except (TypeError, ValueError):
        pause_seconds = 0.0
    plan = _build_alignment_plan(scenes, audio_durations, source_duration, pause_seconds)

    # Fail before expensive rendering if an older/manual artifact would
    # violate either contract: literal 90%-of-source final duration or
    # protagonist first-person narration in every scene.
    planned_duration = sum(
        float(item.get("desired_output_seconds") or 0) for item in plan
    )
    target_final_pre, final_low_pre, final_high_pre = _final_duration_window(
        source_duration, settings
    )
    if not (final_low_pre <= planned_duration <= final_high_pre):
        raise WorkerJobError(
            "Planned final duration is outside the required 90% window before "
            f"render: {_duration_status(planned_duration, source_duration, settings)}; "
            "run TTS calibration again instead of rendering a shortened recap"
        )
    script_coverage_render = script.get("coverage") or {}
    render_pronoun = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes if isinstance(s, dict)],
        preferred=str(
            script_coverage_render.get("first_person_pronoun")
            or (script.get("story_context") or {}).get("pronoun")
            or ""
        ),
        settings=settings,
    )
    bad_render_groups = [
        s.get("group_id")
        for s in scenes
        if isinstance(s, dict)
        and str(s.get("burmese") or "").strip()
        and not _scene_is_first_person(str(s.get("burmese") or ""), render_pronoun)
    ]
    if bad_render_groups:
        raise WorkerJobError(
            "Script is not first-person in every scene before render: "
            f"pronoun={render_pronoun}, bad_group_ids={bad_render_groups}; "
            "run rewrite again"
        )
    _write_worker_json(state, "render_plan.json", {"items": plan})
    state.setdefault("artifacts", {})["render_plan"] = "render_plan.json"
    _save_worker_job(state)

    await _worker_progress(state, WORKER_STAGE_LABELS["render"], tracker)
    segments_dir = _worker_path(state, "render_segments")
    os.makedirs(segments_dir, exist_ok=True)
    segment_files = []

    async with video_render_semaphore:
        for index, item in enumerate(plan, start=1):
            _worker_check_cancelled(state)
            segment_path = os.path.join(segments_dir, f"segment_{index:04d}.mp4")
            ffmpeg_filter = _worker_video_filter(
                out_w, out_h, settings, float(item.get("freeze_seconds") or 0)
            )
            result = await asyncio.to_thread(run_cmd, [
                FFMPEG_BIN, "-y",
                "-ss", f"{float(item.get('source_start') or 0):.3f}",
                "-i", source_path,
                "-t", f"{max(0.04, float(item.get('input_seconds') or 0.04)):.3f}",
                "-an",
                "-vf", ffmpeg_filter,
                "-c:v", "libx264",
                "-preset", "superfast" if settings.get("fast_mode") else "ultrafast",
                "-crf", str(int(settings.get("quality") or 23)),
                segment_path,
            ], 900)
            if result.returncode != 0 or not os.path.exists(segment_path) or os.path.getsize(segment_path) <= 0:
                detail = result.stderr.decode(errors="ignore")[-600:]
                raise WorkerJobError(f"render segment {index} failed: {detail}")
            segment_files.append(segment_path)
            if index == 1 or index == len(plan) or index % 10 == 0:
                pct = int(index / len(plan) * 100) if plan else 100
                await _worker_progress(
                    state,
                    f"{WORKER_STAGE_LABELS['render']} ({index}/{len(plan)})\n📊 {fmt_progress_bar(pct)}",
                    tracker,
                )

        _worker_check_cancelled(state)
        concat_path = _worker_path(state, "render_concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for segment in segment_files:
                escaped = segment.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        video_only_path = _worker_path(state, "video_only.mp4")
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            video_only_path,
        ], 900)
        if result.returncode != 0 or not os.path.exists(video_only_path) or os.path.getsize(video_only_path) <= 0:
            detail = result.stderr.decode(errors="ignore")[-600:]
            raise WorkerJobError(f"video concat failed: {detail}")

        _worker_check_cancelled(state)
        final_path = _worker_path(state, "final.mp4")
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-i", video_only_path,
            "-i", narration_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            final_path,
        ], 900)
        if result.returncode != 0 or not os.path.exists(final_path) or os.path.getsize(final_path) <= 0:
            detail = result.stderr.decode(errors="ignore")[-600:]
            raise WorkerJobError(f"final mux failed: {detail}")

    final_duration, final_w, final_h = probe_video_info(final_path)
    expected_duration = sum(float(item.get("desired_output_seconds") or 0) for item in plan)
    coverage_ratio = _settings_coverage_ratio(settings)
    speed = _settings_speed(settings)
    target_final_seconds, minimum_final_seconds, maximum_final_seconds = (
        _final_duration_window(source_duration, settings)
    )
    # Small mux/container differences are tolerated by the window itself.
    # Anything outside it is not "90% of source" in the user's literal sense.
    if not (minimum_final_seconds <= final_duration <= maximum_final_seconds):
        raise WorkerJobError(
            "Rendered final duration is outside the required 90% window: "
            f"{_duration_status(final_duration, source_duration, settings)}; "
            f"alignment plan {expected_duration:.1f}s, coverage "
            f"{coverage_ratio * 100:.0f}%, speed {speed}; refusing to deliver"
        )
    metrics = {
        "source_duration_seconds": round(float(source_duration), 3),
        "expected_duration_seconds": round(float(expected_duration), 3),
        "final_duration_seconds": round(float(final_duration), 3),
        "target_final_seconds": round(float(target_final_seconds), 3),
        "minimum_acceptable_final_seconds": round(float(minimum_final_seconds), 3),
        "maximum_acceptable_final_seconds": round(float(maximum_final_seconds), 3),
        "first_person_pronoun": render_pronoun,
        "final_ratio_of_source": round(final_duration / source_duration, 4) if source_duration else 0.0,
        "final_ratio_of_target": round(
            final_duration / target_final_seconds, 4
        ) if target_final_seconds else 0.0,
        "size_bytes": os.path.getsize(final_path),
        "size_mb": round(os.path.getsize(final_path) / (1024 * 1024), 2),
        "width": int(final_w or out_w),
        "height": int(final_h or out_h),
        "segment_count": len(plan),
        "borrowed_seconds": round(sum(float(i.get("borrowed_seconds") or 0) for i in plan), 3),
        "freeze_seconds": round(sum(float(i.get("freeze_seconds") or 0) for i in plan), 3),
        "removed_source_seconds": round(
            max(0.0, source_duration - sum(float(i.get("input_seconds") or 0) for i in plan)), 3
        ),
        "audio_alignment_filters_used": [],
    }
    state.setdefault("artifacts", {})["final_video"] = "final.mp4"
    state.setdefault("artifacts", {})["render_plan"] = "render_plan.json"
    state.setdefault("metrics", {})["render"] = metrics
    _set_worker_stage(state, "render", "done", metrics=metrics)
    _save_worker_job(state)
    await _worker_progress(
        state,
        f"✅ Final video ပြီးပါပြီ — {fmt_time(final_duration)}, {metrics['size_mb']:.1f}MB",
        tracker,
    )


async def _worker_stage_deliver(state: dict, tracker=None):
    _worker_check_cancelled(state)
    settings = _worker_settings(state)
    chat_id = int(state.get("chat_id") or 0)
    if not chat_id:
        raise WorkerJobError("chat_id is missing")
    final_rel = (state.get("artifacts") or {}).get("final_video") or "final.mp4"
    final_path = _worker_path(state, str(final_rel))
    if not os.path.exists(final_path) or os.path.getsize(final_path) <= 0:
        raise WorkerJobError("final.mp4 is missing; run render first")

    # Last gate immediately before Telegram upload.  Even if a stale final
    # file was produced by an older build, do not send it when its actual
    # duration is not 90% of the submitted source.
    final_duration, _, _ = probe_video_info(final_path)
    metrics_all = state.get("metrics") or {}
    delivery_script = {}
    try:
        delivery_script = _load_worker_json(state, "script", "script.json")
    except WorkerJobError:
        delivery_script = {}
    delivery_coverage = delivery_script.get("coverage") or {}
    delivery_source_seconds = float(
        (metrics_all.get("render") or {}).get("source_duration_seconds")
        or (metrics_all.get("source") or {}).get("duration_seconds")
        or delivery_coverage.get("source_duration_seconds")
        or 0
    )
    if delivery_source_seconds > 0:
        if final_duration <= 0:
            raise WorkerJobError(
                "Could not verify final video duration before delivery; "
                "refusing to send an unverified recap"
            )
        if not _duration_is_on_target(
            final_duration, delivery_source_seconds, settings
        ):
            raise WorkerJobError(
                "Final video duration is outside the required 90% window "
                "before delivery: "
                f"{_duration_status(final_duration, delivery_source_seconds, settings)}; "
                "refusing to upload a shortened recap"
            )
    delivery_scenes = delivery_script.get("scenes") or []
    if delivery_scenes:
        delivery_pronoun = _choose_first_person_pronoun(
            [str(s.get("burmese") or "") for s in delivery_scenes if isinstance(s, dict)],
            preferred=str(
                delivery_coverage.get("first_person_pronoun")
                or (delivery_script.get("story_context") or {}).get("pronoun")
                or ""
            ),
            settings=settings,
        )
        bad_delivery_groups = [
            s.get("group_id")
            for s in delivery_scenes
            if isinstance(s, dict)
            and str(s.get("burmese") or "").strip()
            and not _scene_is_first_person(
                str(s.get("burmese") or ""), delivery_pronoun
            )
        ]
        if bad_delivery_groups:
            raise WorkerJobError(
                "Script is not first-person in every scene before delivery: "
                f"pronoun={delivery_pronoun}, bad_group_ids={bad_delivery_groups}; "
                "refusing to upload"
            )

    await _worker_progress(state, WORKER_STAGE_LABELS["deliver"], tracker)
    if settings.get("split_tg", True):
        parts = await split_for_telegram(final_path, str(state.get("job_dir") or ""))
    else:
        parts = [final_path]
    if not parts:
        raise WorkerJobError("could not prepare final video for Telegram")
    for part in parts:
        if not os.path.exists(part) or os.path.getsize(part) <= 0:
            raise WorkerJobError("a Telegram delivery part is missing or empty")
        if os.path.getsize(part) > MAX_TG_BYTES:
            raise WorkerJobError(
                "a final video part still exceeds the Telegram limit; "
                "enable Telegram splitting or lower the output settings"
            )

    for index, part in enumerate(parts, start=1):
        _worker_check_cancelled(state)
        caption = ""
        if settings.get("caption", True):
            caption = f"📹 Recap ပြီးပါပြီ။ ({fmt_time(final_duration)})"
            if len(parts) > 1:
                caption += f" - အပိုင်း {index}/{len(parts)}"
        result = await tg_upload_media(
            "sendVideo",
            {"chat_id": str(chat_id), "caption": caption},
            "video",
            f"recap_{index}.mp4",
            part,
            "video/mp4",
        )
        if not (result or {}).get("ok"):
            raise WorkerJobError(f"Telegram upload failed for part {index}/{len(parts)}")
        await _worker_progress(
            state,
            f"{WORKER_STAGE_LABELS['deliver']} ({index}/{len(parts)})",
            tracker,
        )

    metrics = {
        "part_count": len(parts),
        "part_sizes_bytes": [os.path.getsize(part) for part in parts],
        "final_duration_seconds": round(float(final_duration), 3),
        "source_duration_seconds": round(float(delivery_source_seconds), 3),
        "final_ratio_of_source": round(
            final_duration / delivery_source_seconds, 4
        ) if delivery_source_seconds > 0 else 0.0,
        "target_final_seconds": round(
            _target_final_seconds(delivery_source_seconds, settings), 3
        ) if delivery_source_seconds > 0 else 0.0,
    }
    state.setdefault("artifacts", {})["delivery_parts"] = [
        os.path.relpath(part, str(state.get("job_dir") or "")) for part in parts
    ]
    state.setdefault("metrics", {})["delivery"] = metrics
    _set_worker_stage(state, "deliver", "done", metrics=metrics)
    _save_worker_job(state)

    status_id = state.get("status_message_id")
    if status_id:
        try:
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": int(status_id)})
        except Exception:
            pass
    if settings.get("notify", True):
        await send_text(
            chat_id,
            f"✅ <b>Recap ပြီးပါပြီ!</b> ({fmt_time(final_duration)})\n"
            f"🎯 Final duration {settings.get('coverage')}%"
            + (
                f" (အမှန် {final_duration / delivery_source_seconds * 100:.1f}%)"
                if delivery_source_seconds > 0 else ""
            )
            + f" · 🎙️ {settings.get('voice')}",
            reply_markup=main_menu_keyboard(chat_id),
        )


async def _worker_terminal(state: dict, status: str, error: str = ""):
    state["status"] = status
    state["current_stage"] = ""
    if error:
        state["error"] = error[:2000]
    elif status == "delivered":
        state["error"] = ""

    stat_field = {
        "delivered": "done",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(status)
    if stat_field and not state.get("terminal_stat_counted"):
        state["terminal_stat_counted"] = True
        try:
            bump_stat(int(state.get("chat_id") or 0), stat_field)
        except Exception as e:
            logger.warning(f"worker stat update failed: {e}")

    _clear_worker_active(state)
    _save_worker_job(state)

    if status in ("failed", "cancelled"):
        try:
            chat_id = int(state.get("chat_id") or 0)
            status_id = state.get("status_message_id")
            if chat_id and status_id:
                message = (
                    "🛑 အလုပ်ကို ရပ်လိုက်ပါပြီ။"
                    if status == "cancelled"
                    else f"❌ အမှား: {(error or state.get('error') or 'worker failed')[:300]}"
                )
                await edit_text(
                    chat_id,
                    int(status_id),
                    message,
                    reply_markup=main_menu_keyboard(chat_id),
                )
        except Exception as e:
            logger.warning(f"worker terminal Telegram update failed: {e}")


def _worker_stage_function(stage: str):
    return {
        "download": _worker_stage_download,
        "transcribe": _worker_stage_transcribe,
        "rewrite": _worker_stage_rewrite,
        "tts": _worker_stage_tts,
        "render": _worker_stage_render,
        "deliver": _worker_stage_deliver,
    }.get(stage)


async def _run_worker_stage(job_id: str, stage: str):
    current_task = asyncio.current_task()
    try:
        async with WORKER_STAGE_SEMAPHORE:
            state = _load_worker_job(job_id)
            tracker = _worker_tracker(state)
            stage_info = (state.get("stages") or {}).get(stage) or {}
            if stage_info.get("status") == "done":
                return _public_worker_job(state)

            stage_function = _worker_stage_function(stage)
            if stage_function is None:
                raise WorkerJobError(f"unknown worker stage: {stage}")

            state["status"] = "running"
            state["current_stage"] = stage
            state["error"] = ""
            info = state.setdefault("stages", _default_stage_map()).setdefault(stage, {})
            if info.get("status") != "running":
                info["status"] = "running"
                info["started_at"] = _worker_now()
                info["finished_at"] = ""
                info["error"] = ""
            _save_worker_job(state)

            try:
                _worker_check_cancelled(state)
                await stage_function(state, tracker)
                _worker_check_cancelled(state)

                # Real stage functions mark themselves done with metrics.
                # This also makes small test doubles behave correctly.
                current_info = (state.get("stages") or {}).get(stage) or {}
                if current_info.get("status") != "done":
                    current_info["status"] = "done"
                    current_info["finished_at"] = _worker_now()
                    current_info["error"] = ""
                    state.setdefault("stages", {})[stage] = current_info

                state["current_stage"] = ""
                if stage == "deliver":
                    await _worker_terminal(state, "delivered")
                else:
                    state["status"] = "stage_done"
                    _save_worker_job(state)
                return _public_worker_job(state)
            except WorkerCancelled:
                _set_worker_stage(state, stage, "cancelled", error="cancelled")
                await _worker_terminal(state, "cancelled", "cancelled")
                return _public_worker_job(state)
            except asyncio.CancelledError:
                _set_worker_stage(state, stage, "cancelled", error="cancelled")
                await _worker_terminal(state, "cancelled", "cancelled")
                raise
            except Exception as e:
                logger.exception(f"Worker stage failed: job={job_id} stage={stage}")
                message = f"{type(e).__name__}: {e}"
                _set_worker_stage(state, stage, "failed", error=message)
                await _worker_terminal(state, "failed", message)
                return _public_worker_job(state)
    except asyncio.CancelledError:
        # Covers cancellation while waiting for the stage semaphore or
        # before the inner stage try-block is entered.  If the inner
        # handler already finalized the job, this is a no-op.
        try:
            state = _load_worker_job(job_id)
            if state.get("status") not in ("delivered", "cancelled", "failed"):
                info = (state.get("stages") or {}).get(stage) or {}
                if info.get("status") not in ("done", "cancelled"):
                    _set_worker_stage(
                        state, stage, "cancelled", error="cancelled"
                    )
                await _worker_terminal(state, "cancelled", "cancelled")
        except Exception as cleanup_error:
            logger.warning(
                f"worker cancellation cleanup failed for {job_id}: {cleanup_error}"
            )
        raise
    finally:
        if WORKER_TASKS.get(job_id) is current_task:
            WORKER_TASKS.pop(job_id, None)


async def _forward_job_to_n8n(state: dict) -> bool:
    if not N8N_WEBHOOK_URL:
        return False
    payload = {
        "job_id": state.get("job_id"),
        "chat_id": state.get("chat_id"),
        "reply_msg_id": state.get("reply_msg_id"),
        "status_message_id": state.get("status_message_id"),
        "url": state.get("url"),
        "worker_base_url": RENDER_EXTERNAL_URL.rstrip("/"),
        "create_status_message": not bool(state.get("status_message_id")),
        "settings": state.get("settings") or {},
    }
    headers = {"Content-Type": "application/json"}
    if N8N_WEBHOOK_SECRET:
        headers["X-N8N-Secret"] = N8N_WEBHOOK_SECRET
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(N8N_WEBHOOK_URL, headers=headers, json=payload)
        if 200 <= response.status_code < 300:
            return True
        logger.warning(
            "n8n webhook rejected job %s: HTTP %s %s",
            state.get("job_id"), response.status_code, response.text[:500],
        )
    except Exception as e:
        logger.warning(f"n8n webhook forward failed for {state.get('job_id')}: {e}")
    return False


# =====================================================================
# Manual two-minute-part Telegram flow
# =====================================================================

class ManualSessionError(RuntimeError):
    pass


def _user_error_text(exc: Exception) -> str:
    if isinstance(exc, SourceDownloadError):
        text = exc.user_message or str(exc)
        if exc.detail and exc.detail != text:
            text += f"\nအသေးစိတ်: {exc.detail}"
    else:
        text = f"{type(exc).__name__} - {_safe_error_text(exc, 900)}"
    return html.escape(_safe_error_text(text, 1400), quote=False)


def _manual_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manual_sessions_root() -> str:
    root = os.path.join(WORK_ROOT, "manual_sessions")
    os.makedirs(root, exist_ok=True)
    return root


def _is_safe_manual_session_id(session_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", str(session_id or "")))


def _manual_session_dir(session_id: str) -> str:
    if not _is_safe_manual_session_id(session_id):
        raise ManualSessionError("invalid manual session id")
    root = os.path.abspath(_manual_sessions_root())
    path = os.path.abspath(os.path.join(root, str(session_id)))
    if os.path.commonpath([root, path]) != root:
        raise ManualSessionError("invalid manual session path")
    return path


def _manual_state_file(session_id: str) -> str:
    return os.path.join(_manual_session_dir(session_id), "state.json")


def _save_manual_session(state: dict, *, allow_clear_cancel: bool = False) -> dict:
    session_id = str(state.get("session_id") or "")
    session_dir = _manual_session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)
    state["session_dir"] = session_dir
    if session_id and not state.get("cancel_requested") and not allow_clear_cancel:
        try:
            existing_path = _manual_state_file(session_id)
            if os.path.exists(existing_path):
                with open(existing_path, encoding="utf-8") as existing_file:
                    existing = json.load(existing_file)
                if existing.get("cancel_requested"):
                    state["cancel_requested"] = True
                    CANCELLED_JOBS.add(session_id)
        except Exception:
            pass
    state["updated_at"] = _manual_now()
    tmp_path = _manual_state_file(session_id) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)
        state_file.flush()
        os.fsync(state_file.fileno())
    os.replace(tmp_path, _manual_state_file(session_id))
    try:
        MANUAL_SESSIONS_BY_CHAT[int(state.get("chat_id") or 0)] = session_id
    except (TypeError, ValueError):
        pass
    return state


def _load_manual_session(session_id: str) -> dict:
    path = _manual_state_file(session_id)
    if not os.path.exists(path):
        raise ManualSessionError("manual session not found")
    with open(path, encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict):
        raise ManualSessionError("manual session state is not an object")
    state["session_id"] = str(state.get("session_id") or session_id)
    state["session_dir"] = _manual_session_dir(state["session_id"])
    if not isinstance(state.get("parts"), list):
        state["parts"] = []
    if not isinstance(state.get("source"), dict):
        state["source"] = {}
    if not isinstance(state.get("settings"), dict):
        state["settings"] = get_user_settings(state.get("chat_id"))
    state["settings"] = _with_required_coverage(state.get("settings") or {})
    if not isinstance(state.get("story_context"), dict):
        state["story_context"] = {
            "characters": [],
            "summary": "",
            "last_scene": "",
            "protagonist": "",
            "pronoun": "",
        }
    return state


def _manual_path(state: dict, relative_path: str) -> str:
    session_dir = os.path.abspath(
        state.get("session_dir") or _manual_session_dir(str(state.get("session_id") or ""))
    )
    path = os.path.abspath(os.path.join(session_dir, str(relative_path or "")))
    if os.path.commonpath([session_dir, path]) != session_dir:
        raise ManualSessionError("invalid manual artifact path")
    return path


def _manual_part(state: dict, part_index: int) -> dict | None:
    try:
        wanted = int(part_index)
    except (TypeError, ValueError):
        return None
    for part in state.get("parts") or []:
        if isinstance(part, dict) and int(part.get("index") or 0) == wanted:
            return part
    return None


def _manual_session_expired(state: dict) -> bool:
    try:
        expires_at = float(state.get("expires_at_unix") or 0)
    except (TypeError, ValueError):
        expires_at = 0.0
    return bool(expires_at and time.time() > expires_at)


def _delete_manual_session(
    session_id: str,
    *,
    clear_active: bool = True,
) -> bool:
    session_id = str(session_id or "")
    if not _is_safe_manual_session_id(session_id):
        return False
    chat_id = None
    try:
        state = _load_manual_session(session_id)
        chat_id = int(state.get("chat_id") or 0) or None
    except Exception:
        state = None
    MANUAL_RENDER_TASKS.pop(session_id, None)
    CANCELLED_JOBS.discard(session_id)
    if chat_id and MANUAL_SESSIONS_BY_CHAT.get(chat_id) == session_id:
        MANUAL_SESSIONS_BY_CHAT.pop(chat_id, None)
    if clear_active and chat_id and ACTIVE_JOBS.get(chat_id) == session_id:
        ACTIVE_JOBS.pop(chat_id, None)
    existed = os.path.isdir(_manual_session_dir(session_id))
    shutil.rmtree(_manual_session_dir(session_id), ignore_errors=True)
    return bool(existed or state)


def _find_manual_session_for_chat(
    chat_id,
    *,
    include_terminal: bool = False,
) -> dict | None:
    try:
        chat_key = int(chat_id)
    except (TypeError, ValueError):
        return None
    root = _manual_sessions_root()
    candidate_ids: list[str] = []
    mapped = str(MANUAL_SESSIONS_BY_CHAT.get(chat_key) or "")
    if mapped:
        candidate_ids.append(mapped)
    try:
        entries = []
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                try:
                    entries.append((os.path.getmtime(path), name))
                except Exception:
                    entries.append((0.0, name))
        for _mtime, name in sorted(entries, reverse=True):
            if name not in candidate_ids:
                candidate_ids.append(name)
    except Exception:
        pass

    terminal = {"completed", "cancelled", "expired"}
    for session_id in candidate_ids:
        try:
            state = _load_manual_session(session_id)
        except Exception:
            continue
        if int(state.get("chat_id") or 0) != chat_key:
            continue
        if _manual_session_expired(state):
            if not include_terminal:
                _delete_manual_session(session_id)
                continue
            state["status"] = "expired"
        if not include_terminal and state.get("status") in terminal:
            continue
        MANUAL_SESSIONS_BY_CHAT[chat_key] = session_id
        return state
    return None


def _cleanup_expired_manual_sessions() -> int:
    removed = 0
    root = _manual_sessions_root()
    try:
        names = os.listdir(root)
    except Exception:
        return 0
    for name in names:
        if name in MANUAL_RENDER_TASKS:
            continue
        try:
            state = _load_manual_session(name)
        except Exception:
            continue
        if _manual_session_expired(state) and _delete_manual_session(name):
            removed += 1
    return removed


def _reset_stale_manual_sessions_after_restart() -> int:
    """Make state left by a previous process actionable after a restart.

    Render tasks exist only in process memory.  At startup, therefore, a
    persisted ``rendering``/``joining`` status cannot still be running in
    this process.  Reset only that unit of work for a retry click; an
    interrupted script-first intake cannot safely resume and is marked
    failed so its status/discard controls remain available.
    """
    reset_count = 0
    root = _manual_sessions_root()
    try:
        names = os.listdir(root)
    except Exception:
        return 0
    for name in names:
        if name in MANUAL_RENDER_TASKS:
            continue
        try:
            state = _load_manual_session(name)
        except Exception:
            continue
        status = str(state.get("status") or "")
        changed = False
        if status == "rendering":
            for part in state.get("parts") or []:
                if isinstance(part, dict) and part.get("status") == "rendering":
                    part["status"] = "failed"
                    part["error"] = "Server restart ဖြစ်သွား၍ ဒီအပိုင်းကို ထပ်နှိပ်ပါ။"
                    changed = True
            counts = _manual_counts(state)
            state["status"] = (
                "ready_to_join"
                if counts["total"] and counts["done"] == counts["total"]
                else "ready"
            )
            state["active_render_part"] = None
            changed = True
        elif status == "joining":
            state["status"] = "ready_to_join"
            state["error"] = "Server restart ဖြစ်သွား၍ ပြန်ဆက်ရန် ထပ်နှိပ်ပါ။"
            changed = True
        elif status in ("preparing", "splitting", "scripting", "sending_parts"):
            state["status"] = "failed"
            state["error"] = (
                "Server restart ဖြစ်သွား၍ ဒီလုပ်ငန်းစဉ်ကို ဆက်မလုပ်နိုင်တော့ပါ။ "
                "Session ကို ပယ်၍ Source ကို ပြန်ပို့ပါ။"
            )
            changed = True
        if changed:
            state["cancel_requested"] = False
            CANCELLED_JOBS.discard(str(state.get("session_id") or name))
            try:
                _save_manual_session(state, allow_clear_cancel=True)
                reset_count += 1
            except Exception:
                continue
    return reset_count


def _create_manual_session(
    *,
    session_id: str,
    chat_id: int,
    reply_msg_id: int,
    status_message_id: int,
    source_spec: dict,
    settings: dict,
) -> dict:
    source_spec = dict(source_spec or {})
    source_kind = str(source_spec.get("kind") or "url")
    url = _extract_first_url(str(source_spec.get("url") or ""))
    if source_kind == "telegram_file":
        label = _telegram_attachment_label(source_spec)
    elif url:
        label = _url_hostname(url) or "Video link"
    else:
        label = "Video"
    now_unix = time.time()
    state = {
        "session_id": str(session_id),
        "session_dir": _manual_session_dir(str(session_id)),
        "chat_id": int(chat_id),
        "reply_msg_id": int(reply_msg_id) if reply_msg_id else None,
        "status_message_id": int(status_message_id) if status_message_id else None,
        "control_message_id": None,
        "status": "preparing",
        "source": {
            "kind": source_kind,
            "url": url,
            "label": label,
            "file_name": str(source_spec.get("file_name") or ""),
        },
        "source_duration_seconds": 0.0,
        "source_size_bytes": 0,
        "segment_seconds": SOURCE_SEGMENT_SECONDS,
        "settings": _with_required_coverage(settings or {}),
        "parts": [],
        "story_context": {
            "characters": [],
            "summary": "",
            "last_scene": "",
            "protagonist": "",
            "pronoun": "",
        },
        "full_script_file": "",
        "active_render_part": None,
        "cancel_requested": False,
        "terminal_stat_counted": False,
        "error": "",
        "created_at": _manual_now(),
        "updated_at": _manual_now(),
        "created_at_unix": now_unix,
        "expires_at_unix": now_unix + MANUAL_SESSION_TTL_HOURS * 3600,
    }
    return _save_manual_session(state)


def _manual_counts(state: dict) -> dict:
    parts = [p for p in (state.get("parts") or []) if isinstance(p, dict)]
    counts = {
        "total": len(parts),
        "done": 0,
        "ready": 0,
        "rendering": 0,
        "failed": 0,
        "pending": 0,
    }
    for part in parts:
        status = str(part.get("status") or "")
        if status == "done":
            counts["done"] += 1
        elif status == "rendering":
            counts["rendering"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status in ("ready", "script_ready"):
            counts["ready"] += 1
        else:
            counts["pending"] += 1
    return counts


def _manual_status_text(state: dict) -> str:
    counts = _manual_counts(state)
    settings = state.get("settings") or {}
    source = state.get("source") or {}
    source_label = html.escape(
        str(source.get("label") or source.get("url") or "Video"), quote=False
    )
    done_indexes = []
    ready_indexes = []
    failed_indexes = []
    rendering_indexes = []
    for part in state.get("parts") or []:
        if not isinstance(part, dict):
            continue
        index = int(part.get("index") or 0)
        status = str(part.get("status") or "")
        if status == "done":
            done_indexes.append(index)
        elif status == "rendering":
            rendering_indexes.append(index)
        elif status == "failed":
            failed_indexes.append(index)
        elif status in ("ready", "script_ready"):
            ready_indexes.append(index)

    def fmt_indexes(values: list[int]) -> str:
        return ", ".join(str(v) for v in values) if values else "မရှိ"

    try:
        remaining_hours = max(
            0.0, (float(state.get("expires_at_unix") or 0) - time.time()) / 3600.0
        )
    except (TypeError, ValueError):
        remaining_hours = 0.0
    return (
        "🎬 <b>၂ မိနစ်အပိုင်းများ အခြေအနေ</b>\n"
        "━━━━━━━━━━━━━━━\n"
        f"📥 Source: {source_label} ({fmt_time(float(state.get('source_duration_seconds') or 0))})\n"
        f"✂️ အပိုင်း: {counts['total']} · ✅ Recap ပြီး: {counts['done']} · "
        f"🎬 စောင့်နေ: {counts['ready']} · ❌ Failed: {counts['failed']}\n"
        f"✅ ပြီးသောအပိုင်းများ: {fmt_indexes(done_indexes)}\n"
        f"🎬 Recap မလုပ်ရသေးသောအပိုင်းများ: {fmt_indexes(ready_indexes)}\n"
        f"⏳ လုပ်နေဆဲ: {fmt_indexes(rendering_indexes)} · ❌ Failed: {fmt_indexes(failed_indexes)}\n"
        f"🎯 တစ်ပိုင်းချင်း Final duration: မူရင်းအပိုင်း၏ {settings.get('coverage', 90)}%\n"
        f"🕒 Session ကျန်ချိန်ခန့်မှန်း: {remaining_hours:.1f} နာရီ\n\n"
        "မူရင်းအပိုင်းတစ်ခုအောက်က <b>🎬 Recap လုပ်မယ်</b> ကို နှိပ်ပါ။ "
        "အပိုင်းတိုင်းပြီးမှ 🔗 ပြန်ဆက်နိုင်ပါမယ်။"
    )


def _manual_script_relative(part_index: int) -> str:
    return f"scripts/script_part_{int(part_index):04d}.json"


def _write_manual_json(state: dict, relative_path: str, data: dict) -> str:
    path = _manual_path(state, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(tmp_path, path)
    return path


def _enrich_manual_script(state: dict, part: dict, script: dict) -> dict:
    enriched = json.loads(json.dumps(script or {}, ensure_ascii=False))
    part_start = float(part.get("start") or 0)
    enriched["session_id"] = str(state.get("session_id") or "")
    enriched["part_index"] = int(part.get("index") or enriched.get("part_index") or 0)
    enriched["global_start"] = round(part_start, 3)
    enriched["global_end"] = round(float(part.get("end") or part_start), 3)
    enriched["duration"] = round(float(part.get("duration") or 0), 3)
    scenes = []
    for scene in enriched.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        item = dict(scene)
        try:
            local_start = float(item.get("start") or 0)
        except (TypeError, ValueError):
            local_start = 0.0
        try:
            local_end = float(item.get("end") or local_start)
        except (TypeError, ValueError):
            local_end = local_start
        item["global_start"] = round(part_start + local_start, 3)
        item["global_end"] = round(part_start + local_end, 3)
        scenes.append(item)
    enriched["scenes"] = scenes
    return enriched


def _save_manual_script(state: dict, part: dict, script: dict) -> dict:
    enriched = _enrich_manual_script(state, part, script)
    relative = _manual_script_relative(int(part.get("index") or 0))
    _write_manual_json(state, relative, enriched)
    part["script_file"] = relative
    part["script_chars"] = int(script_total_chars(enriched.get("scenes") or []))
    metrics = enriched.get("metrics") or {}
    part["script_target_chars"] = int(metrics.get("target_chars") or 0)
    return enriched


def _load_manual_script(state: dict, part: dict) -> dict:
    relative = str(part.get("script_file") or _manual_script_relative(int(part.get("index") or 0)))
    path = _manual_path(state, relative)
    if not os.path.exists(path):
        raise ManualSessionError("stored script for this part is missing")
    with open(path, encoding="utf-8") as script_file:
        script = json.load(script_file)
    if not isinstance(script, dict) or not script.get("scenes"):
        raise ManualSessionError("stored script for this part is empty")
    return script


def _write_manual_full_script(state: dict, scripts: list[dict]) -> None:
    payload = {
        "version": 1,
        "session_id": state.get("session_id"),
        "chat_id": state.get("chat_id"),
        "source": state.get("source") or {},
        "source_duration_seconds": state.get("source_duration_seconds"),
        "segment_seconds": state.get("segment_seconds"),
        "settings": state.get("settings") or {},
        "story_context": state.get("story_context") or {},
        "parts": scripts,
    }
    _write_manual_json(state, "full_script.json", payload)
    state["full_script_file"] = "full_script.json"


async def _send_manual_original_part(state: dict, part: dict) -> int | None:
    session_id = str(state.get("session_id") or "")
    chat_id = int(state.get("chat_id") or 0)
    index = int(part.get("index") or 0)
    total_parts = len(state.get("parts") or [])
    source_path = _manual_path(state, str(part.get("source_file") or ""))
    if not os.path.exists(source_path) or os.path.getsize(source_path) <= 0:
        raise ManualSessionError(f"original part {index} is missing")

    send_path = source_path
    preview_path = ""
    files_to_send: list[str] = []
    try:
        if os.path.getsize(send_path) > MAX_TG_BYTES:
            preview_relative = f"previews/source_part_{index:04d}_preview.mp4"
            preview_path = _manual_path(state, preview_relative)
            preview_ok = await asyncio.to_thread(
                _make_telegram_preview_copy, source_path, preview_path
            )
            if preview_ok:
                send_path = preview_path

        files_to_send = [send_path]
        if os.path.getsize(send_path) > MAX_TG_BYTES:
            split_files = await split_for_telegram(
                send_path, str(state.get("session_dir") or "")
            )
            if split_files:
                files_to_send = list(split_files)

        base_caption = (
            f"🎬 မူရင်းအပိုင်း {index}/{total_parts} "
            f"({fmt_time(float(part.get('start') or 0))}–{fmt_time(float(part.get('end') or 0))})\n"
            "📝 Script အသင့်ရှိပြီ — Recap လုပ်ရန် အောက်ကခလုတ်ကို နှိပ်ပါ။"
        )
        message_id = None
        for file_index, path_to_send in enumerate(files_to_send, start=1):
            if not os.path.exists(path_to_send) or os.path.getsize(path_to_send) <= 0:
                raise RuntimeError(f"original part {index} upload file is missing")
            if os.path.getsize(path_to_send) > MAX_TG_BYTES:
                raise RuntimeError(
                    f"original part {index} is still too large for Telegram"
                )
            caption = base_caption
            if len(files_to_send) > 1:
                caption += f"\n📤 Telegram upload အပိုင်း {file_index}/{len(files_to_send)}"
                if file_index < len(files_to_send):
                    caption += " — Recap ခလုတ်ကို နောက်ဆုံး upload အောက်မှာ ထားပါတယ်။"
            payload = {"chat_id": str(chat_id), "caption": caption}
            if file_index == len(files_to_send):
                payload["reply_markup"] = json.dumps(
                    manual_part_keyboard(session_id, index, "ready"),
                    ensure_ascii=False,
                )
            result = await tg_upload_media(
                "sendVideo",
                payload,
                "video",
                f"original_part_{index:04d}.mp4",
                path_to_send,
                "video/mp4",
            )
            if not (result or {}).get("ok"):
                description = _safe_error_text(
                    (result or {}).get("description") or "Telegram upload failed"
                )
                raise RuntimeError(
                    f"Telegram original part {index} upload failed: {description}"
                )
            if file_index == len(files_to_send):
                result_data = (result or {}).get("result") or {}
                if isinstance(result_data, dict) and result_data.get("message_id"):
                    try:
                        message_id = int(result_data["message_id"])
                    except (TypeError, ValueError):
                        message_id = None
        return message_id
    finally:
        for candidate in files_to_send:
            if candidate and os.path.abspath(candidate) != os.path.abspath(source_path):
                try:
                    os.remove(candidate)
                except Exception:
                    pass
        if preview_path and os.path.exists(preview_path):
            try:
                os.remove(preview_path)
            except Exception:
                pass


async def _handle_manual_source_job(
    chat_id: int,
    reply_msg_id: int,
    source_spec: dict,
    session_id: str,
    status_id: int,
    job_settings: dict,
) -> None:
    """Script a whole source first, then send original parts with buttons."""
    tracker = ProgressTracker(chat_id, status_id, job_id=session_id)
    state: dict | None = None
    sent_any_original = False
    try:
        state = _create_manual_session(
            session_id=session_id,
            chat_id=chat_id,
            reply_msg_id=reply_msg_id,
            status_message_id=status_id,
            source_spec=source_spec,
            settings=job_settings,
        )
        session_dir = str(state.get("session_dir") or "")
        source_path, total_dur = await _acquire_source_file(
            source_spec, session_dir, tracker
        )
        if session_id in CANCELLED_JOBS:
            raise Exception("cancelled")
        if total_dur <= 0:
            raise SourceDownloadError(
                "မူရင်းဗီဒီယို download မရပါ။",
                detail="acquired source has no readable duration",
            )
        source_size = os.path.getsize(source_path) if os.path.exists(source_path) else 0
        state["source_duration_seconds"] = round(float(total_dur), 3)
        state["source_size_bytes"] = int(source_size)
        state["status"] = "splitting"
        _save_manual_session(state)
        logger.info(
            "Full source: %.1fs = %s, %.0fMB",
            total_dur,
            fmt_time(total_dur),
            source_size / (1024 * 1024),
        )
        segment_plan = build_source_segment_plan(total_dur, SOURCE_SEGMENT_SECONDS)
        await tracker.update(
            f"✅ မူရင်းရယူပြီး ({fmt_time(total_dur)}, {source_size / (1024 * 1024):.0f}MB)\n"
            f"✂️ ၂ မိနစ်စီ {len(segment_plan)} ပိုင်းခွဲနေပါသည်...",
            force=True,
        )
        source_segments = await _prepare_source_segments(
            source_path, session_dir, total_dur, SOURCE_SEGMENT_SECONDS
        )
        if not source_segments:
            raise RuntimeError("မူရင်းဗီဒီယိုကို ၂ မိနစ်ပိုင်းများ ခွဲ၍မရပါ။")
        parts = []
        for segment in source_segments:
            segment_path = str(segment.get("path") or "")
            parts.append(
                {
                    "index": int(segment.get("index") or len(parts) + 1),
                    "start": round(float(segment.get("start") or 0), 3),
                    "end": round(float(segment.get("end") or 0), 3),
                    "duration": round(float(segment.get("duration") or 0), 3),
                    "source_file": os.path.relpath(segment_path, session_dir),
                    "script_file": "",
                    "status": "pending_script",
                    "message_id": None,
                    "recap_file": "",
                    "recap_duration_seconds": 0.0,
                    "attempts": 0,
                    "error": "",
                }
            )
        state["parts"] = parts
        state["status"] = "scripting"
        _save_manual_session(state)
        try:
            os.remove(source_path)
        except Exception:
            pass
        total_parts = len(parts)
        await tracker.update(
            f"✅ ၂ မိနစ်ပိုင်း {total_parts} ခုရပါပြီ\n"
            "📝 အခု တစ်ခန်းလုံးရဲ့ Script ကို အစဉ်အတိုင်း အရင်ရေးပြီးသိမ်းပါမယ်။ "
            "ပြီးမှ မူရင်းအပိုင်းတွေကို Recap ခလုတ်တွေနဲ့ တစ်ပိုင်းချင်း ပို့ပါမယ်။",
            force=True,
        )

        story_context = {
            "characters": [],
            "summary": "",
            "last_scene": "",
            "protagonist": "",
            "pronoun": "",
        }
        scripts: list[dict] = []
        for part in parts:
            if session_id in CANCELLED_JOBS or state.get("cancel_requested"):
                raise Exception("cancelled")
            part_num = int(part.get("index") or 0)
            chunk_path = _manual_path(state, str(part.get("source_file") or ""))
            actual_dur = float(probe_video_info(chunk_path)[0] or 0)
            if actual_dur <= 0:
                actual_dur = float(part.get("duration") or 0)
            if not os.path.exists(chunk_path) or actual_dur <= 0:
                raise RuntimeError(f"၂ မိနစ်ပိုင်း {part_num} ကြာချိန် မရပါ။")
            part["duration"] = round(actual_dur, 3)
            if float(part.get("end") or 0) <= float(part.get("start") or 0):
                part["end"] = round(float(part.get("start") or 0) + actual_dur, 3)
            part["status"] = "scripting"
            state["story_context"] = story_context
            _save_manual_session(state)
            await tracker.update(
                f"📝 Script ရေးနေသည် — အပိုင်း {part_num}/{total_parts} "
                f"({fmt_time(float(part.get('start') or 0))}–{fmt_time(float(part.get('end') or 0))})\n"
                "👤 အဓိကဇာတ်ကောင် ကိုယ်တိုင် «ကျွန်တော်/ကျွန်မ» အဖြစ် အပြည့်အဝ ပြန်ပြောပါမယ်။",
                force=True,
            )
            script = await generate_chunk_script(
                chunk_path,
                actual_dur,
                float(part.get("start") or 0),
                float(part.get("end") or 0),
                story_context,
                part_num,
                tracker,
                job_settings,
            )
            script = _save_manual_script(state, part, script)
            scripts.append(script)
            next_context = script.get("story_context_after") or {}
            if isinstance(next_context, dict) and next_context:
                story_context = {
                    "characters": list(next_context.get("characters") or []),
                    "summary": str(next_context.get("summary") or ""),
                    "last_scene": str(next_context.get("last_scene") or ""),
                    "protagonist": str(
                        next_context.get("protagonist")
                        or script.get("protagonist")
                        or story_context.get("protagonist")
                        or ""
                    ),
                    "pronoun": str(
                        next_context.get("pronoun")
                        or script.get("pronoun")
                        or story_context.get("pronoun")
                        or ""
                    ),
                }
            state["story_context"] = story_context
            part["status"] = "script_ready"
            _save_manual_session(state)
            logger.info(
                "Manual script part %s/%s ready for session %s (%s chars)",
                part_num,
                total_parts,
                session_id,
                part.get("script_chars"),
            )

        _write_manual_full_script(state, scripts)
        state["status"] = "sending_parts"
        _save_manual_session(state)
        await tracker.update(
            f"✅ တစ်ခန်းလုံး Script {total_parts} ပိုင်းစလုံး ရေးပြီး သိမ်းပြီးပါပြီ\n"
            "📤 အခု မူရင်း ၂ မိနစ်အပိုင်းတွေကို တစ်ပိုင်းချင်း 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ ပို့နေပါသည်...",
            force=True,
        )

        for position, part in enumerate(parts, start=1):
            if session_id in CANCELLED_JOBS or state.get("cancel_requested"):
                raise Exception("cancelled")
            part_num = int(part.get("index") or position)
            await tracker.update(
                f"📤 မူရင်းအပိုင်း {part_num}/{total_parts} ကို ပို့နေပါသည်...",
                force=True,
            )
            message_id = await _send_manual_original_part(state, part)
            sent_any_original = True
            part["message_id"] = message_id
            part["status"] = "ready"
            part["error"] = ""
            _save_manual_session(state)
            if position < total_parts:
                await asyncio.sleep(0.2)

        state["status"] = "ready"
        state["error"] = ""
        _save_manual_session(state)
        await edit_text(
            chat_id,
            status_id,
            f"✅ Script နှင့် မူရင်း ၂ မိနစ်အပိုင်း {total_parts} ခု အသင့်ဖြစ်ပါပြီ။\n"
            "Recap လုပ်ချင်တဲ့ အပိုင်းအောက်က 🎬 Recap လုပ်မယ် ကို နှိပ်ပါ။",
            reply_markup=main_menu_keyboard(chat_id),
        )
        control_id = await send_text(
            chat_id,
            _manual_status_text(state),
            reply_markup=manual_control_keyboard(session_id),
        )
        if control_id:
            state["control_message_id"] = int(control_id)
            _save_manual_session(state)
    except Exception as e:
        logger.exception("Manual source job error")
        if str(e) == "cancelled" or session_id in CANCELLED_JOBS:
            bump_stat(chat_id, "cancelled")
            if state is not None:
                _delete_manual_session(session_id)
            await edit_text(
                chat_id,
                status_id,
                "🛑 အလုပ်ကို ရပ်လိုက်ပါပြီ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
        else:
            bump_stat(chat_id, "failed")
            message = _user_error_text(e)
            if state is not None:
                has_sent_part = sent_any_original or any(
                    isinstance(p, dict) and p.get("message_id")
                    for p in state.get("parts") or []
                )
                state["status"] = "failed"
                state["error"] = message
                try:
                    _save_manual_session(state)
                except Exception:
                    pass
                if not has_sent_part:
                    _delete_manual_session(session_id)
            await edit_text(
                chat_id,
                status_id,
                f"❌ အမှား: {message}",
                reply_markup=main_menu_keyboard(chat_id),
            )
    finally:
        if ACTIVE_JOBS.get(chat_id) == session_id:
            ACTIVE_JOBS.pop(chat_id, None)
        gc.collect()


class _NullProgressTracker:
    async def update(self, _text: str, force: bool = False) -> None:
        return None


async def _claim_manual_part(
    session_id: str,
    part_index: int,
    chat_id: int,
) -> tuple[dict | None, str]:
    async with MANUAL_SESSION_LOCK:
        try:
            state = _load_manual_session(session_id)
        except Exception:
            return None, "expired"
        if int(state.get("chat_id") or 0) != int(chat_id):
            return None, "wrong_chat"
        if _manual_session_expired(state):
            _delete_manual_session(session_id)
            return None, "expired"
        if state.get("status") in ("cancelled", "completed"):
            return state, "closed"
        if state.get("cancel_requested"):
            current_task = MANUAL_RENDER_TASKS.get(session_id)
            if (current_task and not current_task.done()) or state.get("status") in (
                "preparing",
                "splitting",
                "scripting",
                "sending_parts",
                "rendering",
                "joining",
            ):
                return state, "busy"
            # A cancelled one-part render should not permanently close an
            # otherwise ready session if cleanup was interrupted.
            state["cancel_requested"] = False
            CANCELLED_JOBS.discard(session_id)
            _save_manual_session(state, allow_clear_cancel=True)
        part = _manual_part(state, part_index)
        if not part:
            return state, "missing_part"
        if str(part.get("status") or "") == "done":
            return state, "done"
        task = MANUAL_RENDER_TASKS.get(session_id)
        if task and not task.done():
            return state, "busy"
        if any(
            isinstance(item, dict) and item.get("status") == "rendering"
            for item in state.get("parts") or []
        ):
            return state, "busy"
        active = ACTIVE_JOBS.get(int(chat_id))
        if active and active != session_id:
            return state, "active_other"
        if str(part.get("status") or "") not in ("ready", "script_ready", "failed"):
            return state, "not_ready"
        try:
            source_path = _manual_path(state, str(part.get("source_file") or ""))
            script_path = _manual_path(
                state,
                str(part.get("script_file") or _manual_script_relative(part_index)),
            )
        except Exception:
            return state, "missing"
        if not os.path.exists(source_path) or not os.path.exists(script_path):
            return state, "missing"
        part["status"] = "rendering"
        part["attempts"] = int(part.get("attempts") or 0) + 1
        part["error"] = ""
        part["started_at"] = _manual_now()
        state["status"] = "rendering"
        state["active_render_part"] = int(part_index)
        state["error"] = ""
        ACTIVE_JOBS[int(chat_id)] = session_id
        _save_manual_session(state)
        return state, "claimed"


async def _run_manual_part_recap(
    session_id: str,
    part_index: int,
    progress_message_id: int = None,
    original_message_id: int = None,
) -> None:
    current_task = asyncio.current_task()
    chat_id_value = 0
    state: dict | None = None
    part: dict | None = None
    split_files: list[str] = []
    rendered_path = ""
    try:
        state = _load_manual_session(session_id)
        chat_id_value = int(state.get("chat_id") or 0)
        part = _manual_part(state, part_index)
        if not part:
            raise ManualSessionError("manual part not found")
        settings = _with_required_coverage(state.get("settings") or {})
        total_parts = len(state.get("parts") or [])
        tracker = (
            ProgressTracker(chat_id_value, int(progress_message_id), job_id=session_id)
            if progress_message_id
            else _NullProgressTracker()
        )
        source_path = _manual_path(state, str(part.get("source_file") or ""))
        if not os.path.exists(source_path):
            raise ManualSessionError(
                "ဒီအပိုင်းရဲ့ မူရင်း file က session ထဲမှာ မရှိတော့ပါ။ Source ကို ပြန်ပို့ကြည့်ပါ။"
            )
        script = _load_manual_script(state, part)
        actual_dur = float(probe_video_info(source_path)[0] or 0)
        if actual_dur <= 0:
            actual_dur = float(part.get("duration") or 0)
        if actual_dur <= 0:
            raise RuntimeError(f"အပိုင်း {part_index} ကြာချိန် မရပါ။")

        output_relative = f"recaps/recap_part_{int(part_index):04d}.mp4"
        output_path = _manual_path(state, output_relative)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        existing_duration = (
            float(probe_video_info(output_path)[0] or 0)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0
            else 0.0
        )
        updated_script = script
        if existing_duration > 0 and _duration_is_on_target(
            existing_duration, actual_dur, settings
        ):
            rendered_path = output_path
            rendered_duration = existing_duration
            await tracker.update(
                f"✅ အပိုင်း {part_index}/{total_parts} Recap file အသင့်ရှိနေပြီ — ပြန်ပို့နေပါသည်...",
                force=True,
            )
        else:
            async with video_render_semaphore:
                rendered_path, updated_script = await render_chunk_recap_from_script(
                    source_path,
                    actual_dur,
                    float(part.get("start") or 0),
                    float(part.get("end") or 0),
                    script,
                    int(part_index),
                    tracker,
                    settings,
                    output_path=output_path,
                )
            rendered_duration = float(probe_video_info(rendered_path)[0] or 0)
            if not _duration_is_on_target(rendered_duration, actual_dur, settings):
                raise RuntimeError(
                    "Rendered part duration is outside target: "
                    f"{rendered_duration:.1f}s from {actual_dur:.1f}s source part"
                )
            _save_manual_script(state, part, updated_script)

        if session_id in CANCELLED_JOBS:
            raise asyncio.CancelledError()

        files_to_send = [rendered_path]
        if settings.get("split_tg", True) and os.path.getsize(rendered_path) > MAX_TG_BYTES:
            split_files = await split_for_telegram(
                rendered_path, str(state.get("session_dir") or "")
            )
            if split_files:
                files_to_send = list(split_files)
        await tracker.update(
            f"📤 Recap အပိုင်း {part_index}/{total_parts} ကို ပို့နေပါသည်...",
            force=True,
        )
        for file_index, path_to_send in enumerate(files_to_send, start=1):
            if not os.path.exists(path_to_send) or os.path.getsize(path_to_send) <= 0:
                raise RuntimeError("Recap file မတွေ့ပါ။")
            if os.path.getsize(path_to_send) > MAX_TG_BYTES:
                raise RuntimeError("Recap file က Telegram limit ထက် ကြီးနေပါတယ်။")
            caption = ""
            if settings.get("caption", True):
                caption = (
                    f"✅ Recap အပိုင်း {part_index}/{total_parts} "
                    f"({fmt_time(rendered_duration)})"
                )
                if len(files_to_send) > 1:
                    caption += f" — upload {file_index}/{len(files_to_send)}"
            result = await tg_upload_media(
                "sendVideo",
                {"chat_id": str(chat_id_value), "caption": caption},
                "video",
                f"recap_part_{int(part_index):04d}.mp4",
                path_to_send,
                "video/mp4",
            )
            if not (result or {}).get("ok"):
                description = _safe_error_text(
                    (result or {}).get("description") or "Telegram upload failed"
                )
                raise RuntimeError(f"Recap upload failed: {description}")

        async with MANUAL_SESSION_LOCK:
            fresh = _load_manual_session(session_id)
            fresh_part = _manual_part(fresh, part_index)
            if not fresh_part:
                raise ManualSessionError("manual part disappeared")
            fresh_part["status"] = "done"
            fresh_part["recap_file"] = output_relative
            fresh_part["recap_duration_seconds"] = round(float(rendered_duration), 3)
            fresh_part["finished_at"] = _manual_now()
            fresh_part["error"] = ""
            counts = _manual_counts(fresh)
            fresh["status"] = (
                "ready_to_join" if counts["done"] >= counts["total"] and counts["total"] else "ready"
            )
            fresh["active_render_part"] = None
            fresh["error"] = ""
            _save_manual_session(fresh)
            state = fresh
            part = fresh_part

        try:
            os.remove(source_path)
        except Exception:
            pass
        for split_file in split_files:
            if split_file and os.path.abspath(split_file) != os.path.abspath(rendered_path):
                try:
                    os.remove(split_file)
                except Exception:
                    pass

        button_message_id = int(original_message_id or part.get("message_id") or 0) or None
        if button_message_id:
            await edit_reply_markup(
                chat_id_value,
                button_message_id,
                manual_part_keyboard(session_id, int(part_index), "done"),
            )
        counts = _manual_counts(state)
        if progress_message_id:
            await edit_text(
                chat_id_value,
                int(progress_message_id),
                f"✅ <b>Recap အပိုင်း {part_index}/{total_parts} ပြီးပါပြီ</b> "
                f"({fmt_time(rendered_duration)})\n"
                f"📊 ပြီးသောအပိုင်း: {counts['done']}/{counts['total']} · "
                f"ကျန်အပိုင်း: {counts['total'] - counts['done']}",
                reply_markup=manual_control_keyboard(session_id),
            )
        if counts["done"] >= counts["total"] and counts["total"]:
            await send_text(
                chat_id_value,
                "🎉 <b>အပိုင်းအားလုံး Recap ပြီးပါပြီ!</b>\n"
                "အခု 🔗 <b>Recap အားလုံး ပြန်ဆက်မယ်</b> ကို နှိပ်ပြီး Final video ရယူနိုင်ပါတယ်။",
                reply_markup=manual_control_keyboard(session_id),
            )
    except asyncio.CancelledError:
        try:
            async with MANUAL_SESSION_LOCK:
                fresh = _load_manual_session(session_id)
                fresh_part = _manual_part(fresh, part_index)
                if fresh_part and fresh_part.get("status") == "rendering":
                    fresh_part["status"] = "ready"
                    fresh_part["error"] = ""
                fresh["status"] = "ready"
                fresh["active_render_part"] = None
                fresh["cancel_requested"] = False
                _save_manual_session(fresh, allow_clear_cancel=True)
                CANCELLED_JOBS.discard(session_id)
                state = fresh
                chat_id_value = int(fresh.get("chat_id") or chat_id_value)
        except Exception:
            pass
        if chat_id_value:
            button_message_id = int(original_message_id or 0) or None
            if button_message_id:
                await edit_reply_markup(
                    chat_id_value,
                    button_message_id,
                    manual_part_keyboard(session_id, int(part_index), "ready"),
                )
            if progress_message_id:
                await edit_text(
                    chat_id_value,
                    int(progress_message_id),
                    "🛑 ဒီအပိုင်း Recap ကို ရပ်လိုက်ပါပြီ။ ပြန်လုပ်ချင်ရင် Recap ခလုတ်ကို ထပ်နှိပ်ပါ။",
                    reply_markup=manual_control_keyboard(session_id),
                )
        raise
    except Exception as e:
        logger.exception("Manual part recap failed")
        message = _user_error_text(e)
        try:
            async with MANUAL_SESSION_LOCK:
                fresh = _load_manual_session(session_id)
                fresh_part = _manual_part(fresh, part_index)
                if fresh_part:
                    fresh_part["status"] = "failed"
                    fresh_part["error"] = message
                fresh["status"] = "ready"
                fresh["active_render_part"] = None
                fresh["error"] = message
                _save_manual_session(fresh)
                state = fresh
                chat_id_value = int(fresh.get("chat_id") or chat_id_value)
        except Exception:
            pass
        if chat_id_value:
            button_message_id = int(original_message_id or 0) or None
            if button_message_id:
                await edit_reply_markup(
                    chat_id_value,
                    button_message_id,
                    manual_part_keyboard(session_id, int(part_index), "failed"),
                )
            if progress_message_id:
                await edit_text(
                    chat_id_value,
                    int(progress_message_id),
                    f"❌ <b>အပိုင်း {part_index} Recap မအောင်မြင်ပါ</b>\n{message}\n\n"
                    "မူရင်းအပိုင်းအောက်က 🔁 Recap ပြန်လုပ်မယ် ကို နှိပ်ပြီး ပြန်ကြိုးစားနိုင်ပါတယ်။",
                    reply_markup=manual_control_keyboard(session_id),
                )
            else:
                await send_text(
                    chat_id_value,
                    f"❌ <b>အပိုင်း {part_index} Recap မအောင်မြင်ပါ</b>\n{message}",
                    reply_markup=manual_control_keyboard(session_id),
                )
    finally:
        if MANUAL_RENDER_TASKS.get(session_id) is current_task:
            MANUAL_RENDER_TASKS.pop(session_id, None)
        if chat_id_value and ACTIVE_JOBS.get(chat_id_value) == session_id:
            should_clear = True
            try:
                fresh = _load_manual_session(session_id)
                should_clear = not any(
                    isinstance(item, dict) and item.get("status") == "rendering"
                    for item in fresh.get("parts") or []
                ) and fresh.get("status") != "joining"
            except Exception:
                should_clear = True
            if should_clear:
                ACTIVE_JOBS.pop(chat_id_value, None)
        gc.collect()


async def _claim_manual_join(
    session_id: str,
    chat_id: int,
) -> tuple[dict | None, str]:
    async with MANUAL_SESSION_LOCK:
        try:
            state = _load_manual_session(session_id)
        except Exception:
            return None, "expired"
        if int(state.get("chat_id") or 0) != int(chat_id):
            return None, "wrong_chat"
        if _manual_session_expired(state):
            _delete_manual_session(session_id)
            return None, "expired"
        task = MANUAL_RENDER_TASKS.get(session_id)
        if (task and not task.done()) or state.get("status") in ("rendering", "joining"):
            return state, "busy"
        if state.get("cancel_requested"):
            state["cancel_requested"] = False
            CANCELLED_JOBS.discard(session_id)
            _save_manual_session(state, allow_clear_cancel=True)
        active = ACTIVE_JOBS.get(int(chat_id))
        if active and active != session_id:
            return state, "active_other"
        counts = _manual_counts(state)
        if not counts["total"] or counts["done"] < counts["total"]:
            return state, "not_all_done"
        for part in state.get("parts") or []:
            if not isinstance(part, dict):
                return state, "missing"
            try:
                recap_path = _manual_path(state, str(part.get("recap_file") or ""))
            except Exception:
                return state, "missing"
            if not os.path.exists(recap_path) or os.path.getsize(recap_path) <= 0:
                return state, "missing"
        state["status"] = "joining"
        state["error"] = ""
        ACTIVE_JOBS[int(chat_id)] = session_id
        _save_manual_session(state)
        return state, "claimed"


async def _run_manual_join(
    session_id: str,
    progress_message_id: int = None,
) -> None:
    current_task = asyncio.current_task()
    chat_id_value = 0
    state: dict | None = None
    split_files: list[str] = []
    try:
        state = _load_manual_session(session_id)
        chat_id_value = int(state.get("chat_id") or 0)
        settings = _with_required_coverage(state.get("settings") or {})
        tracker = (
            ProgressTracker(chat_id_value, int(progress_message_id), job_id=session_id)
            if progress_message_id
            else _NullProgressTracker()
        )
        parts = sorted(
            [p for p in (state.get("parts") or []) if isinstance(p, dict)],
            key=lambda p: int(p.get("index") or 0),
        )
        recap_paths = [
            _manual_path(state, str(part.get("recap_file") or "")) for part in parts
        ]
        if not recap_paths or any(
            not os.path.exists(path) or os.path.getsize(path) <= 0
            for path in recap_paths
        ):
            raise ManualSessionError("Recap အပိုင်းအားလုံး မရှိသေးလို့ ပြန်ဆက်၍မရပါ။")

        final_relative = "final/final_recap.mp4"
        concat_relative = "final/final_concat.txt"
        final_path = _manual_path(state, final_relative)
        concat_path = _manual_path(state, concat_relative)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        source_duration = float(state.get("source_duration_seconds") or 0)
        final_duration = (
            float(probe_video_info(final_path)[0] or 0)
            if os.path.exists(final_path) and os.path.getsize(final_path) > 0
            else 0.0
        )
        if not (
            final_duration > 0
            and source_duration > 0
            and _duration_is_on_target(final_duration, source_duration, settings)
        ):
            await tracker.update(
                f"🔗 Recap ပြီးတဲ့ {len(recap_paths)} ပိုင်းအားလုံးကို အစဉ်အတိုင်း ပြန်ဆက်နေပါသည်...",
                force=True,
            )
            concat_ok = await asyncio.to_thread(
                _concat_video_parts, recap_paths, final_path, concat_path
            )
            if not concat_ok:
                raise RuntimeError("Recap အပိုင်းအားလုံးကို final video တစ်ခုအဖြစ် ပြန်ဆက်၍မရပါ။")
            final_duration = float(probe_video_info(final_path)[0] or 0)

        target_final, final_low, final_high = _final_duration_window(
            source_duration, settings
        )
        if source_duration <= 0 or not _duration_is_on_target(
            final_duration, source_duration, settings
        ):
            raise RuntimeError(
                "Final joined video duration is outside target: "
                f"{final_duration:.1f}s actual, target {target_final:.1f}s "
                f"(acceptable {final_low:.1f}-{final_high:.1f}s)"
            )
        if session_id in CANCELLED_JOBS:
            raise asyncio.CancelledError()

        files_to_send = [final_path]
        if settings.get("split_tg", True) and os.path.getsize(final_path) > MAX_TG_BYTES:
            split_files = await split_for_telegram(
                final_path, str(state.get("session_dir") or "")
            )
            if split_files:
                files_to_send = list(split_files)
        await tracker.update(
            f"📤 Final Recap ဗီဒီယို ပို့နေပါသည်... ({fmt_time(final_duration)})",
            force=True,
        )
        for file_index, path_to_send in enumerate(files_to_send, start=1):
            if not os.path.exists(path_to_send) or os.path.getsize(path_to_send) <= 0:
                raise RuntimeError("Final video file မတွေ့ပါ။")
            if os.path.getsize(path_to_send) > MAX_TG_BYTES:
                raise RuntimeError("Final video က Telegram limit ထက် ကြီးနေပါတယ်။")
            caption = ""
            if settings.get("caption", True):
                caption = f"📹 Recap ပြီးပါပြီ။ ({fmt_time(final_duration)})"
                if len(files_to_send) > 1:
                    caption += f" - အပိုင်း {file_index}/{len(files_to_send)}"
            result = await tg_upload_media(
                "sendVideo",
                {"chat_id": str(chat_id_value), "caption": caption},
                "video",
                f"recap_final_{file_index}.mp4",
                path_to_send,
                "video/mp4",
            )
            if not (result or {}).get("ok"):
                description = _safe_error_text(
                    (result or {}).get("description") or "Telegram upload failed"
                )
                raise RuntimeError(f"Final upload failed: {description}")

        if not state.get("terminal_stat_counted"):
            bump_stat(chat_id_value, "done")
        if progress_message_id:
            try:
                await tg_api(
                    "deleteMessage",
                    {
                        "chat_id": chat_id_value,
                        "message_id": int(progress_message_id),
                    },
                )
            except Exception:
                pass
        ratio_text = (
            f" (အမှန် {final_duration / source_duration * 100:.1f}%)"
            if source_duration > 0
            else ""
        )
        await send_text(
            chat_id_value,
            f"✅ <b>Final Recap ပြီးပါပြီ!</b> ({fmt_time(final_duration)})\n"
            f"🎯 Final duration {settings.get('coverage')}%{ratio_text} · "
            f"🎙️ {settings.get('voice')}",
            reply_markup=main_menu_keyboard(chat_id_value),
        )
        _delete_manual_session(session_id)
    except asyncio.CancelledError:
        try:
            async with MANUAL_SESSION_LOCK:
                fresh = _load_manual_session(session_id)
                fresh["status"] = "ready_to_join"
                fresh["active_render_part"] = None
                fresh["cancel_requested"] = False
                _save_manual_session(fresh, allow_clear_cancel=True)
                CANCELLED_JOBS.discard(session_id)
                chat_id_value = int(fresh.get("chat_id") or chat_id_value)
        except Exception:
            pass
        if chat_id_value and progress_message_id:
            await edit_text(
                chat_id_value,
                int(progress_message_id),
                "🛑 Final ပြန်ဆက်တာကို ရပ်လိုက်ပါပြီ။ ပြန်နှိပ်၍ ထပ်လုပ်နိုင်ပါတယ်။",
                reply_markup=manual_control_keyboard(session_id),
            )
        raise
    except Exception as e:
        logger.exception("Manual final join failed")
        message = _user_error_text(e)
        try:
            async with MANUAL_SESSION_LOCK:
                fresh = _load_manual_session(session_id)
                fresh["status"] = "ready_to_join"
                fresh["error"] = message
                fresh["active_render_part"] = None
                _save_manual_session(fresh)
                chat_id_value = int(fresh.get("chat_id") or chat_id_value)
        except Exception:
            pass
        if chat_id_value and progress_message_id:
            await edit_text(
                chat_id_value,
                int(progress_message_id),
                f"❌ <b>Final ပြန်ဆက်၍ မရပါ</b>\n{message}",
                reply_markup=manual_control_keyboard(session_id),
            )
        elif chat_id_value:
            await send_text(
                chat_id_value,
                f"❌ <b>Final ပြန်ဆက်၍ မရပါ</b>\n{message}",
                reply_markup=manual_control_keyboard(session_id),
            )
    finally:
        if MANUAL_RENDER_TASKS.get(session_id) is current_task:
            MANUAL_RENDER_TASKS.pop(session_id, None)
        if chat_id_value and ACTIVE_JOBS.get(chat_id_value) == session_id:
            should_clear = True
            try:
                fresh = _load_manual_session(session_id)
                should_clear = fresh.get("status") not in ("rendering", "joining")
            except Exception:
                should_clear = True
            if should_clear:
                ACTIVE_JOBS.pop(chat_id_value, None)
        gc.collect()


async def _handle_manual_part_callback(
    data: str,
    chat_id: int,
    message_id: int = None,
) -> bool:
    if data == "part:noop":
        return True
    if not data.startswith("part:"):
        return False
    bits = data.split(":")
    if len(bits) < 3:
        await send_text(
            chat_id,
            "⌛ ဒီ Recap session က မရှိတော့ပါ။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
            reply_markup=main_menu_keyboard(chat_id),
        )
        return True
    action = bits[1]
    session_id = bits[2]

    if action == "recap":
        if len(bits) != 4:
            await send_text(
                chat_id,
                "⌛ ဒီခလုတ်က သက်တမ်းကုန်နေပါတယ်။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
            return True
        try:
            part_index = int(bits[3])
        except ValueError:
            return True
        state, code = await _claim_manual_part(session_id, part_index, int(chat_id))
        if code != "claimed":
            messages = {
                "expired": "⌛ ဒီ session က သက်တမ်းကုန်/ပျောက်သွားပါပြီ။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
                "wrong_chat": "⛔ ဒီအပိုင်းက အခြား chat ရဲ့ session ပါ။",
                "closed": "⚪ ဒီ session က ပိတ်သွားပါပြီ။",
                "missing_part": "⚠️ ဒီအပိုင်းကို မတွေ့ပါ။",
                "done": f"✅ အပိုင်း {part_index} Recap က ပြီးပြီးသားပါ။",
                "busy": "⏳ အပိုင်းတစ်ခုကို Recap လုပ်နေဆဲပါ။ ပြီးမှ နောက်တစ်ခု နှိပ်ပါ။",
                "active_other": "⚠️ အခြားအလုပ်တစ်ခု လုပ်နေဆဲပါ။ ပြီးအောင်စောင့်ပါ။",
                "not_ready": "⏳ ဒီအပိုင်းရဲ့ Script/Original က အသင့်မဖြစ်သေးပါ။",
                "missing": "⌛ ဒီအပိုင်းရဲ့ မူရင်း file သို့မဟုတ် Script မရှိတော့ပါ။ Source ကို ပြန်ပို့ကြည့်ပါ။",
            }
            await send_text(
                chat_id,
                messages.get(code, "⚠️ Recap စ၍မရသေးပါ။"),
                reply_markup=(
                    manual_control_keyboard(session_id)
                    if state
                    else main_menu_keyboard(chat_id)
                ),
            )
            return True
        if message_id:
            await edit_reply_markup(
                chat_id,
                int(message_id),
                manual_part_keyboard(session_id, part_index, "rendering"),
            )
        total_parts = len((state or {}).get("parts") or [])
        progress_id = await send_text(
            chat_id,
            f"🎬 <b>အပိုင်း {part_index}/{total_parts} Recap စတင်လိုက်ပါပြီ</b>\n"
            "ဒီအပိုင်းတစ်ခုကိုပဲ သိမ်းထားတဲ့ Script နဲ့ လုပ်ပါမယ်။",
            reply_markup=job_progress_keyboard(),
        )
        task = _spawn_background(
            _run_manual_part_recap(
                session_id,
                part_index,
                progress_id,
                int(message_id) if message_id else None,
            ),
            name=f"manual-part-{session_id}-{part_index}",
        )
        MANUAL_RENDER_TASKS[session_id] = task
        return True

    if action == "status":
        try:
            state = _load_manual_session(session_id)
        except Exception:
            await send_text(
                chat_id,
                "⌛ ဒီ session က မရှိတော့ပါ။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
            return True
        if int(state.get("chat_id") or 0) != int(chat_id):
            await send_text(chat_id, "⛔ ဒီ session က ဒီ chat အတွက် မဟုတ်ပါ။", reply_markup=main_menu_keyboard(chat_id))
            return True
        if _manual_session_expired(state):
            _delete_manual_session(session_id)
            await send_text(
                chat_id,
                "⌛ ဒီ session က သက်တမ်းကုန်သွားပါပြီ။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
            return True
        await send_text(
            chat_id,
            _manual_status_text(state),
            reply_markup=manual_control_keyboard(session_id),
        )
        return True

    if action == "discard":
        deleted = False
        busy = False
        should_count_cancelled = False
        async with MANUAL_SESSION_LOCK:
            try:
                state = _load_manual_session(session_id)
            except Exception:
                state = None
            if state and int(state.get("chat_id") or 0) == int(chat_id):
                task = MANUAL_RENDER_TASKS.get(session_id)
                busy = bool(
                    (task and not task.done())
                    or state.get("status")
                    in (
                        "preparing",
                        "splitting",
                        "scripting",
                        "sending_parts",
                        "rendering",
                        "joining",
                    )
                    or any(
                        isinstance(p, dict) and p.get("status") == "rendering"
                        for p in state.get("parts") or []
                    )
                )
                if not busy:
                    should_count_cancelled = bool(
                        not state.get("terminal_stat_counted")
                        and state.get("status") != "completed"
                    )
                    deleted = _delete_manual_session(session_id)
        if busy:
            await send_text(
                chat_id,
                "⏳ အလုပ်/Recap လုပ်နေဆဲ ဖြစ်လို့ session တန်းပယ်၍မရပါ။ အရင်ဆုံး 🛑 ရပ်မယ် ကို နှိပ်ပါ။",
                reply_markup=manual_control_keyboard(session_id),
            )
        elif deleted:
            if should_count_cancelled:
                bump_stat(chat_id, "cancelled")
            await send_text(
                chat_id,
                "🗑️ ဒီ ၂ မိနစ်အပိုင်း session ကို ပယ်လိုက်ပါပြီ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
        else:
            await send_text(
                chat_id,
                "⌛ ဒီ session က မရှိတော့ပါ။",
                reply_markup=main_menu_keyboard(chat_id),
            )
        return True

    if action == "join":
        state, code = await _claim_manual_join(session_id, int(chat_id))
        if code != "claimed":
            if code == "not_all_done" and state:
                await send_text(
                    chat_id,
                    "⚠️ <b>အပိုင်းအားလုံး Recap မပြီးသေးပါ</b>\n\n"
                    + _manual_status_text(state),
                    reply_markup=manual_control_keyboard(session_id),
                )
                return True
            messages = {
                "expired": "⌛ ဒီ session က သက်တမ်းကုန်/ပျောက်သွားပါပြီ။ Video/Link ကို ပြန်ပို့ကြည့်ပါ။",
                "wrong_chat": "⛔ ဒီ session က ဒီ chat အတွက် မဟုတ်ပါ။",
                "busy": "⏳ အပိုင်းတစ်ခု သို့မဟုတ် Final join လုပ်နေဆဲပါ။",
                "active_other": "⚠️ အခြားအလုပ် လုပ်နေဆဲပါ။ ပြီးအောင်စောင့်ပါ။",
                "missing": "⚠️ Recap file အချို့ မရှိတော့လို့ Final ဆက်၍မရပါ။",
            }
            await send_text(
                chat_id,
                messages.get(code, "⚠️ Final ဆက်၍မရသေးပါ။"),
                reply_markup=(
                    manual_control_keyboard(session_id)
                    if state
                    else main_menu_keyboard(chat_id)
                ),
            )
            return True
        progress_id = await send_text(
            chat_id,
            "🔗 <b>Recap အပိုင်းအားလုံးကို Final video အဖြစ် ပြန်ဆက်နေပါသည်...</b>",
            reply_markup=job_progress_keyboard(),
        )
        task = _spawn_background(
            _run_manual_join(session_id, progress_id),
            name=f"manual-join-{session_id}",
        )
        MANUAL_RENDER_TASKS[session_id] = task
        return True

    return True


async def handle_link_job(chat_id: int, reply_msg_id: int, url: str):
    await _handle_source_job(
        chat_id, reply_msg_id, {"kind": "url", "url": str(url or "")}
    )


async def handle_telegram_video_job(chat_id: int, reply_msg_id: int, attachment: dict):
    await _handle_source_job(
        chat_id,
        reply_msg_id,
        {"kind": "telegram_file", **dict(attachment or {})},
    )


async def _handle_source_job(chat_id: int, reply_msg_id: int, source_spec: dict):
    job_settings = _with_required_coverage(get_user_settings(chat_id))
    source_spec = dict(source_spec or {})
    source_kind = str(source_spec.get("kind") or "url")
    url = _extract_first_url(str(source_spec.get("url") or ""))
    if source_kind == "url" and url:
        source_spec["url"] = url
    # Fixed product rule: first split every downloaded source into
    # two-minute parts, regardless of any stale user chunk setting.
    segment_seconds = SOURCE_SEGMENT_SECONDS
    job_settings["chunk"] = segment_seconds
    job_id = f"j{secrets.token_hex(5)}"
    if chat_id in ACTIVE_JOBS:
        await send_text(chat_id, "⚠️ အလုပ်တစ်ခု လုပ်နေဆဲပါ။ ပြီးအောင်စောင့်ပါ သို့မဟုတ် 🛑 ရပ်မယ် ခလုတ်နှိပ်ပါ။",
                        reply_to=reply_msg_id, reply_markup=main_menu_keyboard(chat_id))
        return
    if MANUAL_PARTS_ENABLED:
        existing_manual = _find_manual_session_for_chat(chat_id)
        if existing_manual:
            existing_id = str(existing_manual.get("session_id") or "")
            await send_text(
                chat_id,
                "⚠️ <b>မပြီးသေးတဲ့ ၂ မိနစ်အပိုင်း session တစ်ခု ရှိနေပါသေးတယ်။</b>\n\n"
                + _manual_status_text(existing_manual)
                + "\n\nအသစ်မစခင် အဲဒီ session ကို ဆက်လုပ်ပါ၊ သို့မဟုတ် 🗑️ Session ပယ်မယ် ကို နှိပ်ပါ။",
                reply_to=reply_msg_id,
                reply_markup=manual_control_keyboard(existing_id),
            )
            return
    ACTIVE_JOBS[chat_id] = job_id
    if MANUAL_PARTS_ENABLED:
        initial_text = (
            f"⚡ Recap ပြင်ဆင်နေပါပြီ...\n"
            "📝 Source တစ်ခုလုံးရဲ့ Script ကို အရင်ရေးပြီး၊ မူရင်းကို ၂ မိနစ်စီခွဲကာ "
            "တစ်ပိုင်းချင်း 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ ပြန်ပို့ပါမယ် (Auto မ Run ပါ)\n"
            f"🎯 တစ်ပိုင်းချင်း Final duration: မူရင်းအပိုင်း၏ {job_settings['coverage']}% · "
            f"🎙️ {job_settings['voice']} · 📐 {job_settings['resolution']}p"
        )
    else:
        initial_text = (
            f"⚡ Recap စတင်နေပါပြီ...\n"
            "✂️ မူရင်းကို ၂ မိနစ်စီခွဲပြီး တစ်ပိုင်းချင်း recap လုပ်၍ ပြန်ဆက်ပါမယ်\n"
            f"🎯 Final duration: မူရင်း၏ {job_settings['coverage']}% · 🎙️ {job_settings['voice']} · 📐 {job_settings['resolution']}p"
        )
    status_id = await send_text(
        chat_id,
        initial_text,
        reply_to=reply_msg_id,
        reply_markup=job_progress_keyboard(),
    )

    if MANUAL_PARTS_ENABLED:
        await _handle_manual_source_job(
            chat_id,
            reply_msg_id,
            source_spec,
            job_id,
            status_id,
            job_settings,
        )
        return

    # Optional n8n orchestration.  The Telegram webhook/menu/status remain
    # here; n8n only calls this service's authenticated stage endpoints.
    # If forwarding fails, clean only the worker record and continue with
    # the unchanged direct pipeline below (ACTIVE_JOBS/job_id are kept).
    if source_kind == "url" and PIPELINE_MODE == "n8n" and N8N_WEBHOOK_URL:
        n8n_tracker = ProgressTracker(chat_id, status_id, job_id=job_id)
        forwarded = False
        try:
            async with WORKER_JOB_LOCK:
                worker_state = _create_worker_job(
                    job_id=job_id,
                    chat_id=chat_id,
                    url=url,
                    reply_msg_id=reply_msg_id,
                    status_message_id=status_id,
                    settings_override=job_settings,
                )
            await n8n_tracker.update(
                "🤖 n8n workflow ဆီ ပို့နေပါသည်...",
                force=True,
            )
            forwarded = await _forward_job_to_n8n(worker_state)
            if forwarded:
                return
            logger.warning(f"n8n did not accept job {job_id}; using direct pipeline")
            await n8n_tracker.update(
                "⚠️ n8n မရသေးလို့ မူလလုပ်ငန်းစဉ်ဖြင့် ဆက်လုပ်ပါမည်...",
                force=True,
            )
        except Exception as e:
            logger.warning(f"n8n handoff failed for {job_id}, using direct pipeline: {e}")
            try:
                await n8n_tracker.update(
                    "⚠️ n8n မရသေးလို့ မူလလုပ်ငန်းစဉ်ဖြင့် ဆက်လုပ်ပါမည်...",
                    force=True,
                )
            except Exception:
                pass

        was_cancelled = job_id in CANCELLED_JOBS
        try:
            _delete_worker_job_record(job_id, clear_active=False)
        except Exception as e:
            logger.warning(f"failed to remove unforwarded worker record {job_id}: {e}")
        ACTIVE_JOBS[chat_id] = job_id
        if was_cancelled:
            CANCELLED_JOBS.add(job_id)

    work_dir = tempfile.mkdtemp(prefix="vid_", dir=WORK_ROOT)
    tracker = ProgressTracker(chat_id, status_id, job_id=job_id)

    story_context = {"characters": [], "summary": "", "last_scene": "", "protagonist": "", "pronoun": ""}

    try:
        source_path, total_dur = await _acquire_source_file(
            source_spec, work_dir, tracker
        )
        if total_dur <= 0:
            raise SourceDownloadError(
                "မူရင်းဗီဒီယို download မရပါ။",
                detail="acquired source has no readable duration",
            )

        size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(f"Full source: {total_dur:.1f}s = {fmt_time(total_dur)}, {size_mb:.0f}MB")
        segment_plan = build_source_segment_plan(total_dur, segment_seconds)
        await tracker.update(
            f"✅ မူရင်းရယူပြီး ({fmt_time(total_dur)}, {size_mb:.0f}MB)\n"
            f"✂️ ၂ မိနစ်စီ {len(segment_plan)} ပိုင်းခွဲနေပါသည်...",
            force=True,
        )

        source_segments = await _prepare_source_segments(
            source_path, work_dir, total_dur, segment_seconds
        )
        if not source_segments:
            raise RuntimeError("မူရင်းဗီဒီယိုကို ၂ မိနစ်ပိုင်းများ ခွဲ၍မရပါ။")

        total_parts = len(source_segments)
        logger.info(
            "Prepared %s fixed %ss source segments for job %s",
            total_parts,
            segment_seconds,
            job_id,
        )
        # All source parts now exist, so the full original is no longer
        # needed.  Removing it before rendering keeps long jobs from using
        # roughly double disk space for source + source segments.
        try:
            os.remove(source_path)
        except Exception:
            pass
        await tracker.update(
            f"✅ ၂ မိနစ်ပိုင်း {total_parts} ခုရပါပြီ\n"
            "🎬 တစ်ပိုင်းချင်း အစဉ်အတိုင်း Recap လုပ်ပြီး နောက်ဆုံးမှာ အားလုံးပြန်ဆက်ပါမယ်",
            force=True,
        )

        chunks = []
        for segment in source_segments:
            if job_id in CANCELLED_JOBS:
                raise Exception("cancelled")

            part_num = int(segment.get("index") or (len(chunks) + 1))
            chunk_path = str(segment.get("path") or "")
            start = float(segment.get("start") or 0)
            segment_end = float(segment.get("end") or 0)
            actual_dur, _, _ = probe_video_info(chunk_path)
            if actual_dur <= 0:
                actual_dur = float(segment.get("duration") or 0)
            if not chunk_path or not os.path.exists(chunk_path) or actual_dur <= 0:
                raise RuntimeError(f"၂ မိနစ်ပိုင်း {part_num} ကြာချိန် မရပါ။")
            if segment_end <= start:
                segment_end = start + actual_dur

            logger.info(
                "Source segment %s/%s: %s-%s (%ss)",
                part_num,
                total_parts,
                fmt_time(start),
                fmt_time(segment_end),
                f"{actual_dur:.1f}",
            )
            await tracker.update(
                f"🎧 ၂ မိနစ်ပိုင်း {part_num}/{total_parts} "
                f"({fmt_time(start)}–{fmt_time(segment_end)}) ကို Recap လုပ်နေပါသည်...\n"
                f"📊 {fmt_progress_bar(int((start / total_dur) * 100) if total_dur else 0)}",
                force=True,
            )

            async with video_render_semaphore:
                recap_chunk, new_summary = await build_chunk_recap(
                    chunk_path,
                    actual_dur,
                    start,
                    segment_end,
                    story_context,
                    part_num,
                    tracker,
                    job_settings,
                )

            if not recap_chunk:
                raise RuntimeError(f"အပိုင်း {part_num} Recap မရပါ။")

            chunks.append(recap_chunk)
            story_context = new_summary
            logger.info(
                f"Story context: protagonist='{story_context.get('protagonist', '')}', "
                f"pronoun='{story_context.get('pronoun', '')}', "
                f"{len(story_context.get('characters', []))} chars, "
                f"{len(story_context.get('summary', ''))} summary"
            )

            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass

        if os.path.exists(source_path):
            try:
                os.remove(source_path)
            except Exception:
                pass

        if len(chunks) != total_parts:
            raise RuntimeError(
                f"Recap အပိုင်းမစုံပါ: {len(chunks)}/{total_parts} ပိုင်းသာရပါသည်။"
            )
        await tracker.update(
            f"🔗 Recap ပြီးတဲ့ {total_parts} ပိုင်းအားလုံးကို အစဉ်အတိုင်း ပြန်ဆက်နေပါသည်...",
            force=True,
        )
        final_out = os.path.join(work_dir, "final_recap.mp4")
        concat_final = os.path.join(work_dir, "final_concat.txt")
        concat_ok = await asyncio.to_thread(
            _concat_video_parts, chunks, final_out, concat_final
        )
        if not concat_ok:
            raise RuntimeError("Recap အပိုင်းအားလုံးကို final video တစ်ခုအဖြစ် ပြန်ဆက်၍မရပါ။")

        final_dur, _, _ = probe_video_info(final_out)
        target_final_dur, final_low, final_high = _final_duration_window(
            total_dur, job_settings
        )
        if not _duration_is_on_target(final_dur, total_dur, job_settings):
            raise RuntimeError(
                "Final video duration is outside target: "
                f"{final_dur:.1f}s actual, target {target_final_dur:.1f}s "
                f"(acceptable {final_low:.1f}-{final_high:.1f}s from "
                f"{total_dur:.1f}s source)"
            )
        await tracker.update(
            f"📤 ဗီဒီယို ပို့ပေးနေပါသည်... ({fmt_time(final_dur)})\n"
            f"📊 {fmt_progress_bar(99)}",
            force=True,
        )

        parts = await split_for_telegram(final_out, work_dir) if job_settings.get('split_tg') else [final_out]
        for i, part in enumerate(parts):
            caption = f"📹 Recap ပြီးပါပြီ။ ({fmt_time(final_dur)})" if job_settings.get("caption") else ""
            if len(parts) > 1:
                caption += f" - အပိုင်း {i + 1}/{len(parts)}"
            res = await tg_upload_media(
                "sendVideo",
                {"chat_id": str(chat_id), "caption": caption},
                "video",
                f"recap_{i + 1}.mp4",
                part,
                "video/mp4",
            )
            if not res.get("ok"):
                await edit_text(chat_id, status_id, "⚠️ ဗီဒီယိုပို့ရာတွင် အခက်အခဲရှိပါသည်။")
                break

        bump_stat(chat_id, "done")
        try:
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
        except Exception:
            pass
        if job_settings.get("notify"):
            await send_text(
                chat_id,
                f"✅ <b>Recap ပြီးပါပြီ!</b> ({fmt_time(final_dur)})\n"
                f"🎯 Final duration {job_settings['coverage']}% "
                f"(အမှန် {final_dur / total_dur * 100:.1f}%) · "
                f"🎙️ {job_settings['voice']}",
                reply_markup=main_menu_keyboard(chat_id),
            )

    except Exception as e:
        logger.exception("Job Error")
        if str(e) == "cancelled":
            bump_stat(chat_id, "cancelled")
            await edit_text(chat_id, status_id, "🛑 အလုပ်ကို ရပ်လိုက်ပါပြီ။", reply_markup=main_menu_keyboard(chat_id))
        else:
            bump_stat(chat_id, "failed")
            await edit_text(chat_id, status_id, f"❌ အမှား: {_user_error_text(e)}", reply_markup=main_menu_keyboard(chat_id))
    finally:
        ACTIVE_JOBS.pop(chat_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def job_status_text(chat_id) -> str:
    job = ACTIVE_JOBS.get(chat_id)
    s = get_user_settings(chat_id)
    if job:
        manual_note = ""
        try:
            manual_state = _load_manual_session(str(job))
            if int(manual_state.get("chat_id") or 0) == int(chat_id):
                counts = _manual_counts(manual_state)
                manual_note = (
                    f"\n🎬 Manual အပိုင်းများ: {counts['done']}/{counts['total']} ပြီး"
                )
        except Exception:
            pass
        return (f"📊 <b>အလုပ်အခြေအနေ</b>\n━━━━━━━━━━━━━━━\n🟢 လုပ်နေဆဲ (Job: <code>{job}</code>)\n"
                f"🎯 Final duration: မူရင်း၏ {s['coverage']}% · 🎙️ {s['voice']}"
                f"{manual_note}\n\nရပ်ချင်ရင် 🛑 ခလုတ်နှိပ်ပါ။")
    if MANUAL_PARTS_ENABLED:
        manual_state = _find_manual_session_for_chat(chat_id)
        if manual_state:
            return _manual_status_text(manual_state)
    return ("📊 <b>အလုပ်အခြေအနေ</b>\n━━━━━━━━━━━━━━━\n⚪ လက်ရှိ လုပ်နေတာ မရှိပါ။\n\n"
            "🎬 Recap စဖို့ video file သို့မဟုတ် link ပို့လိုက်ပါ (သို့) ခလုတ်နှိပ်ပါ။")


def _status_keyboard(chat_id) -> dict:
    if MANUAL_PARTS_ENABLED:
        manual_state = _find_manual_session_for_chat(chat_id)
        if manual_state:
            return manual_control_keyboard(str(manual_state.get("session_id") or ""))
    return main_menu_keyboard(chat_id)


async def handle_callback(cb: dict):
    """Inline button ခလုတ်တွေ နှိပ်လာရင် ကိုင်တွယ်တယ်။"""
    data = (cb.get("data") or "").strip()
    from_user = cb.get("from", {}) or {}
    msg = cb.get("message", {}) or {}
    chat_id = msg.get("chat", {}).get("id") or from_user.get("id")
    message_id = msg.get("message_id")
    if not is_allowed(from_user.get("id")):
        await answer_callback(cb.get("id"), "⛔ ခွင့်ပြုထားသူ မဟုတ်ပါ")
        return
    await answer_callback(cb.get("id"))
    # A user may tap an inline button from an older menu after a deploy
    # without ever sending /start again.  Install the persistent bottom
    # buttons once for that chat as well.
    await ensure_persistent_keyboard(chat_id)

    if data.startswith("part:"):
        await _handle_manual_part_callback(data, int(chat_id), message_id)
        return

    async def show(text, markup):
        # While a job is running, the callback may come from the live
        # progress message.  Editing that message into Settings/Status
        # makes the next progress update immediately overwrite it, so the
        # button looks broken.  Send a separate menu message and preserve
        # the progress message.  Cancel is the exception: update the
        # progress message itself to show that the job is stopping.
        if (
            ACTIVE_JOBS.get(chat_id)
            and data.startswith("menu:")
            and data != "menu:cancel"
        ):
            await send_text(chat_id, text, reply_markup=markup)
            return
        await edit_text(chat_id, message_id, text, reply_markup=markup)

    if data == "menu:main":
        await show(start_features_text(chat_id), main_menu_keyboard(chat_id))
    elif data == "menu:settings":
        await show("⚙️ <b>ဆက်တင်များ (Feature ၂၀)</b>\n━━━━━━━━━━━━━━━\nပြောင်းချင်တဲ့ ခလုတ်ကို နှိပ်ပါ။\n🟢 = On · 🔴 = Off", settings_keyboard(chat_id))
    elif data == "menu:recap":
        await show("🎬 <b>Recap စတင်ရန်</b>\n\nTelegram video file (20MB အောက်)၊ document video သို့မဟုတ် RedNote/video link (https://...) ကို ဒီအတိုင်း ပို့လိုက်ပါ။\n✂️ Bot က ၂ မိနစ်စီခွဲပြီး တစ်ခန်းလုံး Script အရင်ရေးမယ်၊ ပြီးရင် မူရင်းအပိုင်းတွေကို 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ တစ်ပိုင်းချင်း ပြန်ပို့ပါမယ်။", main_menu_keyboard(chat_id))
    elif data == "menu:status":
        await show(job_status_text(chat_id), _status_keyboard(chat_id))
    elif data == "menu:summary":
        await show(settings_summary_text(chat_id), main_menu_keyboard(chat_id))
    elif data == "menu:help":
        await show(HELP_TEXT, main_menu_keyboard(chat_id))
    elif data == "menu:stats":
        st = JOB_STATS.get(int(chat_id), {"done": 0, "failed": 0, "cancelled": 0})
        await show(f"📈 <b>စာရင်းချုပ်</b>\n━━━━━━━━━━━━━━━\n✅ ပြီးဆုံး: {st.get('done', 0)}\n❌ ပျက်: {st.get('failed', 0)}\n🛑 ရပ်ခဲ့: {st.get('cancelled', 0)}", main_menu_keyboard(chat_id))
    elif data == "menu:cancel":
        job = await request_cancel_active_job_async(chat_id)
        if job:
            await show("🛑 ရပ်ဖို့ တောင်းဆိုလိုက်ပါပြီ... လက်ရှိအပိုင်း ပြီးရင် ရပ်ပါမယ်။", main_menu_keyboard(chat_id))
        else:
            await show("⚪ လုပ်နေတဲ့ အလုပ် မရှိပါ။", main_menu_keyboard(chat_id))
    elif data == "menu:reset":
        reset_user_settings(chat_id)
        await show("🔄 ဆက်တင်အားလုံး မူရင်း (default) ပြန်ထားလိုက်ပါပြီ။", main_menu_keyboard(chat_id))
    elif data.startswith("feat:"):
        key = data.split(":", 1)[1]
        feat = FEATURE_BY_KEY.get(key)
        if not feat:
            return
        if feat["kind"] == "toggle":
            cur = get_user_settings(chat_id).get(key)
            set_user_setting(chat_id, key, not bool(cur))
            await show("⚙️ <b>ဆက်တင်များ (Feature ၂၀)</b>\n\n" f"{feat['icon']} {feat['label']}: <b>{feature_value_label(feat, get_user_settings(chat_id))}</b>", settings_keyboard(chat_id))
        else:
            await show(f"{feat['icon']} <b>{feat['label']}</b>\n{feat.get('desc', '')}\n\n👇 ရွေးချယ်ပါ:", feature_options_keyboard(feat, chat_id))
    elif data.startswith("set:"):
        _, key, raw = data.split(":", 2)
        feat = FEATURE_BY_KEY.get(key)
        if not feat:
            return
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        set_user_setting(chat_id, key, value)
        await show(f"✅ {feat['icon']} <b>{feat['label']}</b> ကို <b>{feature_value_label(feat, get_user_settings(chat_id))}</b> ပြောင်းလိုက်ပါပြီ။", settings_keyboard(chat_id))
    elif data == "job:start":
        pending = PENDING_INPUTS.pop(chat_id, None)
        legacy_url = PENDING_LINKS.pop(chat_id, None)
        await show("🚀 စတင်လိုက်ပါပြီ...", main_menu_keyboard(chat_id))
        if isinstance(pending, dict) and pending.get("kind") == "telegram_file":
            _spawn_background(
                handle_telegram_video_job(chat_id, message_id, pending),
                name="telegram-video-job",
            )
        else:
            url = ""
            if isinstance(pending, dict):
                url = str(pending.get("url") or "")
            url = url or str(legacy_url or "")
            if url:
                _spawn_background(
                    handle_link_job(chat_id, message_id, url),
                    name="telegram-link-job",
                )
    elif data == "job:discard":
        PENDING_INPUTS.pop(chat_id, None)
        PENDING_LINKS.pop(chat_id, None)
        await show("❌ Video/Link ကို ပယ်လိုက်ပါပြီ။", main_menu_keyboard(chat_id))


async def process_admin_update(update: dict):
    # ခလုတ် (callback) လာရင်
    if update.get("callback_query"):
        await handle_callback(update["callback_query"])
        return
    msg = update.get("message")
    if not msg:
        return
    chat_id = msg.get("chat", {}).get("id")
    from_id = msg.get("from", {}).get("id")
    msg_id = msg.get("message_id")
    # A directly uploaded video normally has no text, only video/document
    # metadata and an optional caption.  Treat captions like text for links
    # or commands, while the attachment itself takes priority as the source.
    text = (msg.get("text") or msg.get("caption") or "").strip()
    source_url = _extract_first_url(text)
    attachment = _telegram_video_attachment(msg)
    if not is_allowed(from_id):
        await send_text(chat_id, "⛔ ဒီ bot ကို သုံးခွင့် မရှိပါ။", reply_to=msg_id)
        return

    # Install the bottom buttons on the first ordinary message/link/video
    # too; the user should not have to discover that /start is required
    # after every deployment.  /start and /menu force-install via menu.
    first_token = text.split()[0].split("@")[0].lower() if text else ""
    if (text or attachment) and first_token not in ("/start", "/menu"):
        await ensure_persistent_keyboard(chat_id, reply_to=msg_id)

    # Persistent reply-keyboard buttons arrive as ordinary text messages,
    # not callback_query updates.  Route their exact labels explicitly;
    # without this mapping the bottom buttons visibly exist but only fall
    # through to the generic menu response.
    if text == PERSISTENT_BUTTON_RECAP:
        await send_text(
            chat_id,
            "🎬 <b>Recap စတင်ရန်</b>\n\nTelegram video file (20MB အောက်)၊ document video သို့မဟုတ် RedNote/video link ကို ဒီအတိုင်း ပို့လိုက်ပါ။\n"
            "✂️ Bot က ၂ မိနစ်စီခွဲပြီး တစ်ခန်းလုံး Script အရင်ရေးကာ မူရင်းအပိုင်းတွေကို 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ ပြန်ပို့ပါမယ်။",
            reply_to=msg_id,
            reply_markup=main_menu_keyboard(chat_id),
        )
        return
    persistent_commands = {
        PERSISTENT_BUTTON_SETTINGS: "/settings",
        PERSISTENT_BUTTON_STATUS: "/status",
        PERSISTENT_BUTTON_CANCEL: "/cancel",
        PERSISTENT_BUTTON_SUMMARY: "/summary",
        PERSISTENT_BUTTON_HELP: "/help",
    }
    cmd = persistent_commands.get(text) or (
        text.split()[0].split("@")[0].lower() if text else ""
    )
    if cmd in ("/start", "/menu"):
        await send_main_menu(
            chat_id, reply_to=msg_id, with_persistent_keyboard=True
        )
    elif cmd == "/settings":
        await send_text(chat_id, "⚙️ <b>ဆက်တင်များ (Feature ၂၀)</b>\nပြောင်းချင်တဲ့ ခလုတ်ကို နှိပ်ပါ။", reply_to=msg_id, reply_markup=settings_keyboard(chat_id))
    elif cmd == "/status":
        await send_text(chat_id, job_status_text(chat_id), reply_to=msg_id, reply_markup=_status_keyboard(chat_id))
    elif cmd == "/cancel":
        job = await request_cancel_active_job_async(chat_id)
        if job:
            await send_text(chat_id, "🛑 ရပ်ဖို့ တောင်းဆိုလိုက်ပါပြီ...", reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
        else:
            await send_text(chat_id, "⚪ လုပ်နေတဲ့ အလုပ် မရှိပါ။", reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
    elif cmd == "/summary":
        await send_text(chat_id, settings_summary_text(chat_id), reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
    elif cmd == "/help":
        await send_text(chat_id, HELP_TEXT, reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
    elif attachment:
        s = get_user_settings(chat_id)
        if s.get("auto_start"):
            await handle_telegram_video_job(chat_id, msg_id, attachment)
        else:
            PENDING_INPUTS[chat_id] = dict(attachment)
            await send_text(
                chat_id,
                "🎬 <b>ဒီ video ကို Recap လုပ်မလား?</b>\n"
                f"{_telegram_attachment_label(attachment)}\n"
                "✂️ ၂ မိနစ်စီခွဲပြီး Script အရင်ရေးကာ မူရင်းအပိုင်းတွေကို 🎬 Recap လုပ်မယ် ခလုတ်နဲ့ ပြန်ပို့ပါမယ်။",
                reply_to=msg_id,
                reply_markup=confirm_keyboard(),
            )
    elif source_url:
        s = get_user_settings(chat_id)
        if s.get("auto_start"):
            await handle_link_job(chat_id, msg_id, source_url)
        else:
            PENDING_LINKS[chat_id] = source_url
            PENDING_INPUTS[chat_id] = {"kind": "url", "url": source_url}
            await send_text(chat_id, "🎬 <b>ဒီ link ကို Recap လုပ်မလား?</b>\n" + html.escape(source_url[:200], quote=False), reply_to=msg_id, reply_markup=confirm_keyboard())
    elif text:
        await send_main_menu(
            chat_id, reply_to=msg_id, with_persistent_keyboard=True
        )
    elif msg.get("document") or msg.get("audio") or msg.get("voice") or msg.get("photo"):
        await send_text(
            chat_id,
            "⚠️ ဒီဖိုင်ကို video အဖြစ် မဖတ်နိုင်ပါ။ Telegram video၊ "
            "document video (MP4/MOV/MKV/WebM စသည်) သို့မဟုတ် video link ပို့ပါ။",
            reply_to=msg_id,
            reply_markup=main_menu_keyboard(chat_id),
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    version = globals().get("BUILD_VERSION", "")
    features = globals().get("BUILD_FEATURES", 20)
    logger.info(
        "🚀 rednote-bot %s စတင်ပါပြီ (features=%s, public_base=%s, "
        "render_commit=%s, render_branch=%s)",
        version,
        features,
        PUBLIC_BASE_URL,
        RENDER_GIT_COMMIT or "unknown",
        RENDER_GIT_BRANCH or "unknown",
    )
    try:
        removed_sessions = _cleanup_expired_manual_sessions()
        if removed_sessions:
            logger.info("Cleaned %s expired manual part sessions", removed_sessions)
        reset_sessions = _reset_stale_manual_sessions_after_restart()
        if reset_sessions:
            logger.info("Reset %s stale manual part sessions after restart", reset_sessions)
    except Exception as e:
        logger.warning("manual session cleanup failed: %s", _safe_error_text(e))
    logger.info(
        "Source downloader: yt-dlp=%s impersonation=%s rednote_cookie_configured=%s recap_flow=%s",
        _installed_yt_dlp_version(),
        _yt_dlp_impersonation_available(),
        _rednote_cookie_configured(),
        "manual_per_part" if MANUAL_PARTS_ENABLED else "auto",
    )
    setup_task = _spawn_background(_setup_bot(), name="telegram-setup")
    try:
        yield
    finally:
        setup_task.cancel()
        try:
            await setup_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

# ဒီ code အစစ် run နေကြောင်း စစ်နိုင်ဖို့ version အမှတ်အသား
BUILD_VERSION = "v3.8-manual-parts-download-fix"
BUILD_FEATURES = 20


@app.get("/")
async def root():
    return JSONResponse({
        "app": "rednote-bot",
        "version": BUILD_VERSION,
        "features": BUILD_FEATURES,
        "status": "online",
        "pipeline_mode": PIPELINE_MODE,
        "recap_flow": "manual_per_part" if MANUAL_PARTS_ENABLED else "auto_concat",
        "manual_parts_enabled": MANUAL_PARTS_ENABLED,
        # Backward-compatible field name from v3.2.  Semantically this is
        # now the minimum selectable *final video duration* percentage.
        "minimum_script_coverage_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "target_final_duration_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "final_duration_enforcement": "measured_tts_and_rendered_file",
        "first_person_enforcement": "per_scene_validated",
        "source_segment_seconds": SOURCE_SEGMENT_SECONDS,
        "segment_pipeline": (
            "download_split_120s_script_first_send_originals_click_recap_explicit_join"
            if MANUAL_PARTS_ENABLED
            else "download_split_120s_recap_each_concat_send"
        ),
        "input_types": ["telegram_video", "telegram_document_video", "video_link"],
        "source_download_diagnostics": "redacted_yt_dlp_stderr_with_direct_and_rednote_fallbacks",
        "telegram_file_download_limit_mb": 20,
        "yt_dlp_version": _installed_yt_dlp_version(),
        "yt_dlp_impersonation_available": _yt_dlp_impersonation_available(),
        "rednote_cookie_configured": _rednote_cookie_configured(),
        "telegram_buttons": "inline_persistent_menu_and_per_part_recap",
        "telegram_webhook_allowed_updates": TELEGRAM_WEBHOOK_ALLOWED_UPDATES,
        "public_base_url": PUBLIC_BASE_URL,
        "public_host": urlparse(PUBLIC_BASE_URL).netloc,
        "render_git_commit": RENDER_GIT_COMMIT,
        "render_git_branch": RENDER_GIT_BRANCH,
        "render_git_repo_slug": RENDER_GIT_REPO_SLUG,
        "render_service_name": RENDER_SERVICE_NAME,
        "render_service_id": RENDER_SERVICE_ID,
        "worker_api_configured": bool(WORKER_API_KEY),
        "whisper_configured": bool(WHISPER_API_KEY),
        "whisper_model": WHISPER_MODEL,
        "n8n_webhook_configured": bool(N8N_WEBHOOK_URL),
    })


# Keep the legacy /healthz path from the previously deployed service as
# well as /health.  If the existing Render service still health-checks
# /healthz, returning 404 there can make every new deploy fail its health
# check and silently leave the old container serving traffic.
@app.get("/health")
@app.get("/healthz")
async def health():
    return JSONResponse({
        "ok": True,
        "app": "rednote-bot",
        "version": BUILD_VERSION,
        "pipeline_mode": PIPELINE_MODE,
        "recap_flow": "manual_per_part" if MANUAL_PARTS_ENABLED else "auto_concat",
        "manual_parts_enabled": MANUAL_PARTS_ENABLED,
        "minimum_script_coverage_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "target_final_duration_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "final_duration_enforcement": "measured_tts_and_rendered_file",
        "first_person_enforcement": "per_scene_validated",
        "source_segment_seconds": SOURCE_SEGMENT_SECONDS,
        "segment_pipeline": (
            "download_split_120s_script_first_send_originals_click_recap_explicit_join"
            if MANUAL_PARTS_ENABLED
            else "download_split_120s_recap_each_concat_send"
        ),
        "input_types": ["telegram_video", "telegram_document_video", "video_link"],
        "source_download_diagnostics": "redacted_yt_dlp_stderr_with_direct_and_rednote_fallbacks",
        "telegram_file_download_limit_mb": 20,
        "yt_dlp_version": _installed_yt_dlp_version(),
        "yt_dlp_impersonation_available": _yt_dlp_impersonation_available(),
        "rednote_cookie_configured": _rednote_cookie_configured(),
        "telegram_buttons": "inline_persistent_menu_and_per_part_recap",
        "telegram_webhook_allowed_updates": TELEGRAM_WEBHOOK_ALLOWED_UPDATES,
        "public_base_url": PUBLIC_BASE_URL,
        "public_host": urlparse(PUBLIC_BASE_URL).netloc,
        "render_git_commit": RENDER_GIT_COMMIT,
        "render_git_branch": RENDER_GIT_BRANCH,
        "render_git_repo_slug": RENDER_GIT_REPO_SLUG,
        "render_service_name": RENDER_SERVICE_NAME,
        "worker_api_configured": bool(WORKER_API_KEY),
    })


# --- Authenticated n8n worker API ------------------------------------
# These endpoints are intentionally disabled unless WORKER_API_KEY is set.
# n8n starts one stage, polls GET /api/v1/jobs/{job_id}, and only starts
# the next stage after the previous one is marked done.


def _worker_auth_error(request: Request):
    if not WORKER_API_KEY:
        return JSONResponse(
            {"ok": False, "error": "worker API is not configured"},
            status_code=503,
        )
    supplied = (request.headers.get("x-worker-key") or "").strip()
    authorization = (request.headers.get("authorization") or "").strip()
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, WORKER_API_KEY):
        return JSONResponse(
            {"ok": False, "error": "unauthorized"},
            status_code=401,
        )
    return None


def _worker_error_response(message: str, status_code: int = 400):
    return JSONResponse({"ok": False, "error": str(message)}, status_code=status_code)


async def _worker_request_json(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _worker_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _worker_public_with_task_state(state: dict) -> dict:
    public = _public_worker_job(state)
    task = WORKER_TASKS.get(str(state.get("job_id") or ""))
    task_alive = bool(task and not task.done())
    public["task_running"] = task_alive
    public["stale_running"] = bool(
        not task_alive
        and any(
            (info or {}).get("status") == "running"
            for info in (state.get("stages") or {}).values()
        )
    )
    return public


@app.post("/api/v1/jobs")
async def worker_create_job(request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error

    payload = await _worker_request_json(request)
    nested_body = payload.get("body")
    if isinstance(nested_body, dict) and not payload.get("url"):
        payload = {**nested_body, **{k: v for k, v in payload.items() if k != "body"}}

    job_id = str(payload.get("job_id") or f"j{secrets.token_hex(5)}").strip()
    settings_override = payload.get("settings") if isinstance(payload.get("settings"), dict) else None
    try:
        async with WORKER_JOB_LOCK:
            state = _create_worker_job(
                job_id=job_id,
                chat_id=payload.get("chat_id"),
                url=str(payload.get("url") or ""),
                reply_msg_id=payload.get("reply_msg_id"),
                status_message_id=payload.get("status_message_id"),
                settings_override=settings_override,
            )
    except WorkerJobError as e:
        message = str(e)
        status_code = 409 if "already active" in message or "already belongs" in message else 400
        return _worker_error_response(message, status_code)
    except Exception as e:
        logger.exception("worker job create failed")
        return _worker_error_response(f"{type(e).__name__}: {e}", 500)

    if _worker_bool(payload.get("create_status_message")) and not state.get("status_message_id"):
        settings = state.get("settings") or get_user_settings(state.get("chat_id"))
        status_id = await send_text(
            int(state["chat_id"]),
            "⚡ Recap စတင်နေပါပြီ...\n"
            f"🎯 Final duration: မူရင်း၏ {settings.get('coverage', 90)}% · "
            f"🎙️ {settings.get('voice', 'Charon')} · 📐 {settings.get('resolution', 1280)}p",
            reply_to=state.get("reply_msg_id"),
            reply_markup=job_progress_keyboard(),
        )
        if status_id:
            async with WORKER_JOB_LOCK:
                try:
                    state = _load_worker_job(job_id)
                    if not state.get("status_message_id"):
                        state["status_message_id"] = int(status_id)
                        _save_worker_job(state)
                except Exception as e:
                    logger.warning(f"could not save worker status message id: {e}")

    return JSONResponse(_worker_public_with_task_state(state), status_code=200)


@app.get("/api/v1/jobs/{job_id}")
async def worker_get_job(job_id: str, request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error
    try:
        state = _load_worker_job(job_id)
    except WorkerJobError:
        return _worker_error_response("worker job not found", 404)
    except Exception as e:
        logger.exception("worker job load failed")
        return _worker_error_response(f"{type(e).__name__}: {e}", 500)
    return JSONResponse(_worker_public_with_task_state(state))


@app.post("/api/v1/jobs/{job_id}/stages/{stage}")
async def worker_start_stage(job_id: str, stage: str, request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error
    if stage not in WORKER_STAGES:
        return _worker_error_response("unknown worker stage", 404)

    async with WORKER_JOB_LOCK:
        try:
            state = _load_worker_job(job_id)
        except WorkerJobError:
            return _worker_error_response("worker job not found", 404)

        stage_info = (state.get("stages") or {}).get(stage) or {}
        if stage_info.get("status") == "done":
            return JSONResponse(_worker_public_with_task_state(state), status_code=200)

        task = WORKER_TASKS.get(job_id)
        if task and not task.done():
            if state.get("current_stage") and state.get("current_stage") != stage:
                return _worker_error_response(
                    f"another worker stage is already running: {state.get('current_stage')}",
                    409,
                )
            return JSONResponse(_worker_public_with_task_state(state), status_code=202)

        if state.get("status") in ("delivered", "cancelled"):
            return _worker_error_response(
                f"worker job is already {state.get('status')}", 409
            )
        if state.get("status") == "failed" and stage_info.get("status") != "failed":
            return _worker_error_response("worker job is failed", 409)
        if job_id in CANCELLED_JOBS or state.get("cancel_requested"):
            return _worker_error_response("worker job cancellation was requested", 409)

        stage_index = WORKER_STAGES.index(stage)
        incomplete = [
            previous
            for previous in WORKER_STAGES[:stage_index]
            if ((state.get("stages") or {}).get(previous) or {}).get("status") != "done"
        ]
        if incomplete:
            return _worker_error_response(
                f"previous worker stage is not done: {incomplete[0]}", 409
            )

        try:
            chat_id = int(state.get("chat_id") or 0)
        except (TypeError, ValueError):
            chat_id = 0
        active = ACTIVE_JOBS.get(chat_id) if chat_id else None
        if active and active != job_id:
            return _worker_error_response(
                f"another job is already active: {active}", 409
            )

        # A failed stage may be explicitly restarted.  Its previous final
        # error/stat remain historical information, but this new attempt
        # becomes the active one for the chat again.
        if state.get("status") == "failed":
            state["terminal_stat_counted"] = False
        state["cancel_requested"] = False
        if chat_id:
            ACTIVE_JOBS[chat_id] = job_id
        _set_worker_stage(state, stage, "running")
        state["status"] = "running"
        state["current_stage"] = stage
        state["error"] = ""
        _save_worker_job(state)
        WORKER_TASKS[job_id] = asyncio.create_task(_run_worker_stage(job_id, stage))
        return JSONResponse(_worker_public_with_task_state(state), status_code=202)


@app.post("/api/v1/jobs/{job_id}/cancel")
async def worker_cancel_job(job_id: str, request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error

    task = None
    needs_terminal = False
    async with WORKER_JOB_LOCK:
        try:
            state = _load_worker_job(job_id)
        except WorkerJobError:
            return _worker_error_response("worker job not found", 404)

        if state.get("status") == "delivered":
            return JSONResponse(_worker_public_with_task_state(state))

        state["cancel_requested"] = True
        CANCELLED_JOBS.add(job_id)
        task = WORKER_TASKS.get(job_id)
        if task and not task.done():
            state["status"] = "cancelling"
            _save_worker_job(state)
            task.cancel()
            return JSONResponse(_worker_public_with_task_state(state), status_code=202)

        current_stage = state.get("current_stage") or _next_worker_stage(state)
        if current_stage:
            info = (state.get("stages") or {}).get(current_stage) or {}
            if info.get("status") not in ("done", "cancelled"):
                _set_worker_stage(state, current_stage, "cancelled", error="cancelled")
        _save_worker_job(state)
        needs_terminal = state.get("status") != "cancelled"

    if needs_terminal:
        await _worker_terminal(state, "cancelled", "cancelled")
    return JSONResponse(_worker_public_with_task_state(state))


@app.post("/api/v1/jobs/{job_id}/fail")
async def worker_fail_job(job_id: str, request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error

    payload = await _worker_request_json(request)
    reason = str(payload.get("reason") or payload.get("error") or "n8n workflow failed").strip()
    if not reason:
        reason = "n8n workflow failed"

    async with WORKER_JOB_LOCK:
        try:
            state = _load_worker_job(job_id)
        except WorkerJobError:
            return _worker_error_response("worker job not found", 404)

        if state.get("status") == "delivered":
            return JSONResponse(_worker_public_with_task_state(state))

        task = WORKER_TASKS.get(job_id)
        if task and not task.done():
            return _worker_error_response("worker stage is still running", 409)

        failed_stage = state.get("current_stage")
        if not failed_stage:
            for name in WORKER_STAGES:
                if ((state.get("stages") or {}).get(name) or {}).get("status") == "failed":
                    failed_stage = name
                    break
        if failed_stage:
            info = (state.get("stages") or {}).get(failed_stage) or {}
            if info.get("status") != "done":
                _set_worker_stage(state, failed_stage, "failed", error=reason)
        state["error"] = reason[:2000]
        _save_worker_job(state)

    await _worker_terminal(state, "failed", reason)
    return JSONResponse(_worker_public_with_task_state(state))


@app.delete("/api/v1/jobs/{job_id}")
async def worker_cleanup_job(job_id: str, request: Request):
    auth_error = _worker_auth_error(request)
    if auth_error:
        return auth_error
    if not _is_safe_worker_job_id(job_id):
        return _worker_error_response("invalid job_id", 400)

    payload = await _worker_request_json(request)
    force = _worker_bool(payload.get("force")) or _worker_bool(
        request.query_params.get("force")
    )
    task = None

    async with WORKER_JOB_LOCK:
        try:
            _load_worker_job(job_id)
        except WorkerJobError:
            return JSONResponse({"ok": True, "job_id": job_id, "deleted": False})

        task = WORKER_TASKS.get(job_id)
        if task and not task.done():
            if not force:
                return _worker_error_response("worker stage is still running", 409)
            CANCELLED_JOBS.add(job_id)
            task.cancel()

    if task and not task.done():
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async with WORKER_JOB_LOCK:
        _delete_worker_job_record(job_id, clear_active=True)

    return JSONResponse({"ok": True, "job_id": job_id, "deleted": True})


BOT_COMMANDS = [
    {"command": "start", "description": "🎬 ပင်မ မီနူး (ခလုတ်များ)"},
    {"command": "settings", "description": "⚙️ ဆက်တင်များ (Feature 20)"},
    {"command": "status", "description": "📊 အလုပ်အခြေအနေ"},
    {"command": "cancel", "description": "🛑 လုပ်နေတာ ရပ်ရန်"},
    {"command": "summary", "description": "📋 ဆက်တင် အကျဉ်း"},
    {"command": "help", "description": "❓ အသုံးပြုနည်း"},
]


def _telegram_result_ok(method: str, response) -> bool:
    ok = bool((response or {}).get("ok"))
    if ok:
        logger.info("Telegram %s: ok", method)
    else:
        logger.warning(
            "Telegram %s failed: %s",
            method,
            (response or {}).get("description") or response,
        )
    return ok


async def _setup_bot_once(expected_webhook_url: str) -> bool:
    """Register commands, the chat menu button, and this exact webhook."""
    me = await tg_api("getMe", {})
    if not _telegram_result_ok("getMe", me):
        return False
    username = str(((me or {}).get("result") or {}).get("username") or "")
    if username:
        logger.info("Telegram bot identity: @%s", username)

    commands_ok = True
    for payload in (
        {"commands": BOT_COMMANDS},
        {
            "commands": BOT_COMMANDS,
            "scope": {"type": "all_private_chats"},
        },
    ):
        result = await tg_api("setMyCommands", payload)
        commands_ok = _telegram_result_ok("setMyCommands", result) and commands_ok

    menu_result = await tg_api(
        "setChatMenuButton",
        {"menu_button": {"type": "commands"}},
    )
    menu_ok = _telegram_result_ok("setChatMenuButton", menu_result)

    webhook_result = await tg_api(
        "setWebhook",
        {
            "url": expected_webhook_url,
            "drop_pending_updates": False,
            "max_connections": 40,
            # Explicitly reset this every deploy.  If omitted, Telegram
            # can retain an older allowed_updates setting and silently
            # stop delivering inline-button callback_query updates.
            "allowed_updates": TELEGRAM_WEBHOOK_ALLOWED_UPDATES,
        },
    )
    webhook_ok = _telegram_result_ok("setWebhook", webhook_result)

    info = await tg_api("getWebhookInfo", {})
    info_ok = _telegram_result_ok("getWebhookInfo", info)
    result = (info or {}).get("result") or {}
    actual_url = str(result.get("url") or "")
    actual_updates = result.get("allowed_updates") or []
    url_ok = actual_url == expected_webhook_url
    updates_ok = (
        not actual_updates or "callback_query" in actual_updates
    )
    logger.info(
        "Telegram webhook verify: url=%s expected=%s allowed_updates=%s "
        "pending=%s last_error_date=%s last_error_message=%s",
        actual_url,
        expected_webhook_url,
        actual_updates,
        result.get("pending_update_count"),
        result.get("last_error_date"),
        result.get("last_error_message") or "",
    )
    return bool(
        commands_ok and menu_ok and webhook_ok and info_ok and url_ok and updates_ok
    )


async def _setup_bot():
    """စတင်တာနဲ့ Telegram command/menu ခလုတ်နှင့် webhook ကို
    ဒီ service ဆီ ပြန်ချိတ်တယ် — အဟောင်း လွဲနေရင် ကိုယ်တိုင် ပြင်သွားအောင်။"""
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN မရှိပါ — Telegram setup ကျော်ပါမည်")
        return
    if not PUBLIC_BASE_URL:
        logger.warning("Public base URL မရှိပါ — Telegram webhook setup ကျော်ပါမည်")
        return

    expected_webhook_url = f"{PUBLIC_BASE_URL.rstrip('/')}/webhook"
    await asyncio.sleep(2)
    for attempt in range(1, 7):
        try:
            if await _setup_bot_once(expected_webhook_url):
                logger.info(
                    "✅ Telegram setup complete: version=%s webhook=%s",
                    BUILD_VERSION,
                    expected_webhook_url,
                )
                return
            logger.warning(
                "Telegram setup incomplete (attempt %s/6)", attempt
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Telegram setup attempt %s/6 failed: %s: %s",
                attempt,
                type(e).__name__,
                e,
            )
        if attempt < 6:
            await asyncio.sleep(min(30, 2 ** attempt))

    logger.error(
        "Telegram setup did not verify after retries; the HTTP app remains "
        "online and /start can still install buttons when reachable"
    )


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    _spawn_background(
        process_admin_update(update), name="telegram-update"
    )
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=max(1, _parse_int_env("PORT", 10000)),
    )

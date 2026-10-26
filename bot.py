import asyncio
import gc
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
import unicodedata
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rednote-mm-bot")


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

    Priority matters.  A stale, manually copied ``RENDER_EXTERNAL_URL`` can
    otherwise make a freshly deployed bot re-register Telegram's webhook to
    an older service, leaving the user talking to obsolete code.  Render's
    platform-provided hostname describes the instance that is actually
    running this process, so prefer it over the generic URL variable.  An
    explicit ``WEBHOOK_BASE_URL`` remains available for custom domains.
    """
    explicit = _clean_public_base_url(
        os.environ.get("WEBHOOK_BASE_URL")
        or os.environ.get("PUBLIC_BASE_URL")
        or ""
    )
    if explicit:
        return explicit

    hostname = str(os.environ.get("RENDER_EXTERNAL_HOSTNAME") or "").strip()
    hostname = hostname.replace("https://", "").replace("http://", "")
    hostname = hostname.strip("/").split("/")[0].strip()
    if hostname:
        return f"https://{hostname}"

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
WHISPER_CHUNK_SECONDS = max(60, _parse_int_env("WHISPER_CHUNK_SECONDS", 600))

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
CHUNK_DURATION = 300
PAUSE_AFTER_SCENE = 0.20

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
    "chunk": 300,          # 12 📦 အပိုင်းခွဲချိန် (စက္ကန့်)
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
    {"key": "chunk", "icon": "📦", "label": "အပိုင်းခွဲ အချိန်", "kind": "cycle",
     "desc": "ရှည်တဲ့ ဗီဒီယိုကို ခွဲလုပ်မည့် အပိုင်းအရှည်",
     "options": [(180, "3 မိနစ်"), (300, "5 မိနစ် ⭐"), (600, "10 မိနစ်")]},
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
    {"key": "auto_start", "icon": "🚀", "label": "Link တန်း စ", "kind": "toggle",
     "desc": "Link ပို့လိုက်တာနဲ့ မေးမနေဘဲ တန်းစခြင်း (Off ဆိုရင် အတည်ပြု ခလုတ် ပြမယ်)"},
]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}

SETTINGS_FILE = os.path.join(WORK_ROOT, "bot_settings.json")
_settings_lock = asyncio.Lock()
ACTIVE_JOBS: dict = {}          # chat_id -> job_id
PENDING_LINKS: dict = {}        # chat_id -> url (auto_start Off က confirm စောင့်နေတာ)
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
        "input_field_placeholder": "ဗီဒီယို link ပို့ပါ…",
    }


MAIN_MENU_TEXT = (
    "🎬 <b>RedNote Recap Bot</b>\n"
    "━━━━━━━━━━━━━━━\n"
    "ဗီဒီယို link ပို့လိုက်တာနဲ့ မြန်မာ Recap ဗီဒီယို လုပ်ပေးပါတယ်။\n\n"
    "👇 ခလုတ်လေးတွေ နှိပ်ပြီး သုံးနိုင်ပါတယ်\n"
    "🎬 <b>Recap စတင်မယ်</b> — link ပို့ဖို့\n"
    "⚙️ <b>ဆက်တင်များ</b> — feature ၂၀ လုံး ချိန်ညှိရန်\n"
    "🛑 <b>ရပ်မယ်</b> — လုပ်နေတဲ့ အလုပ် ပယ်ဖျက်ရန်"
)

HELP_TEXT = (
    "❓ <b>အသုံးပြုနည်း</b>\n"
    "━━━━━━━━━━━━━━━\n"
    "1️⃣ ဗီဒီယို link (http...) ကို ဒီအတိုင်း ပို့လိုက်ပါ\n"
    "2️⃣ Bot က download → ပထမလူ script အပြည့်အစုံ → အသံကြာချိန် တိုင်းတာညှိခြင်း → Final ဗီဒီယို (မူရင်း၏ ဆက်တင်%) လုပ်ပေးပါမယ်\n"
    "3️⃣ ပြီးရင် Recap ဗီဒီယို ပြန်ပို့ပေးပါမယ်\n\n"
    "⌨️ <b>Commands:</b>\n"
    "/start သို့မဟုတ် /menu — ပင်မ မီနူး\n"
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
    text = "⌨️ အောက်ခြေခလုတ်တွေ အဆင်သင့်ဖြစ်ပါပြီ။ Link ကို ဒီအတိုင်းပို့လိုက်ရုံပါပဲ။"
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
    version = globals().get("BUILD_VERSION", "")
    text = MAIN_MENU_TEXT
    if version:
        coverage = get_user_settings(chat_id).get("coverage", 90)
        text += (
            f"\n\n🆕 Version: <code>{version}</code> · "
            f"🎯 Final = မူရင်း၏ {coverage}%"
        )
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


async def get_source_duration(url: str) -> float:
    try:
        cmd = ["yt-dlp", "-j", "--no-warnings", url]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if out:
            data = json.loads(out.decode().strip().split("\n")[0])
            dur = data.get("duration")
            if dur:
                return float(dur)
    except Exception as e:
        logger.warning(f"get_source_duration error: {e}")
    return 0


async def download_full(url: str, out_path: str, tracker=None, total_dur: float = 0) -> float:
    cmd = [
        "yt-dlp", "--no-warnings", "--no-part",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", out_path, url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )

    last_report = time.monotonic()
    while True:
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
            break
        except asyncio.TimeoutError:
            if tracker and time.monotonic() - last_report >= 4.0:
                last_report = time.monotonic()
                if os.path.exists(out_path):
                    mb = os.path.getsize(out_path) / (1024 * 1024)
                    if total_dur > 0:
                        pct = min(95, int((mb / max(1, total_dur)) * 100 / 4))
                    else:
                        pct = min(95, int(mb / 10))
                    await tracker.update(
                        f"📥 မူရင်းဗီဒီယို ဒေါင်းနေသည်...\n"
                        f"📦 {mb:.0f} MB\n"
                        f"📊 {fmt_progress_bar(pct)}",
                        force=True,
                    )

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 10000:
        return 0
    actual_dur, _, _ = probe_video_info(out_path)
    return actual_dur


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
    """Return job settings with the product-wide 90% floor applied.

    A saved/stale value below 90 must never make an individual link produce
    a short recap.  Values above 90 (95/100) remain available.
    """
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    try:
        coverage = float(s.get("coverage") or MINIMUM_SCRIPT_COVERAGE_PERCENT)
    except (TypeError, ValueError):
        coverage = float(MINIMUM_SCRIPT_COVERAGE_PERCENT)
    coverage = max(float(MINIMUM_SCRIPT_COVERAGE_PERCENT), min(100.0, coverage))
    s["coverage"] = int(coverage) if coverage.is_integer() else coverage
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


async def build_chunk_recap(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    story_context: dict,
    chunk_index: int,
    tracker: ProgressTracker,
    settings: dict = None,
):
    st = _with_required_coverage(settings)
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

    await tracker.update("🎧 Gemini ဆီ ဗီဒီယိုတင်ပြီး ဇာတ်လမ်းစာသား ရေးနေပါသည်...", force=True)
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

    scenes = []
    cuts = []
    new_protagonist = protagonist
    new_pronoun = pronoun

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
    for attempt in range(SCRIPT_TOPUP_ATTEMPTS):
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
    scenes = trim_scenes_to_target(scenes, target_chars)
    new_pronoun = _choose_first_person_pronoun(
        [str(s.get("burmese") or "") for s in scenes],
        preferred=new_pronoun or pronoun,
        settings=st,
    )

    # Prompt-only first-person instructions are not sufficient.  Validate
    # every scene's actual Burmese text and ask Gemini to rewrite only the
    # scenes that still sound like a third-person summary.
    for repair_round in range(FIRST_PERSON_REPAIR_ROUNDS):
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
        try:
            await asyncio.to_thread(client.files.delete, name=video_file.name)
        except Exception as e:
            logger.warning(f"Gemini delete error: {e}")
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
        try:
            await asyncio.to_thread(client.files.delete, name=video_file.name)
        except Exception as e:
            logger.warning(f"Gemini delete error: {e}")

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

    out_chunk = chunk_path.replace(".mp4", "_recap.mp4")
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

    for f in [chunk_path, video_only, master_wav, concat_txt] + seg_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    new_summary = await extract_story_summary(scenes, client)
    if new_protagonist and not new_summary.get("protagonist"):
        new_summary["protagonist"] = new_protagonist
    if new_pronoun and not new_summary.get("pronoun"):
        new_summary["pronoun"] = new_pronoun
    return out_chunk, new_summary


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
    url = str(url or "").strip()
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
        duration = await download_full(
            str(state.get("url") or ""), source_path, tracker, probe_duration
        )
        _worker_check_cancelled(state)
        if duration <= 0 or not os.path.exists(source_path):
            raise WorkerJobError("source download failed")
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


async def handle_link_job(chat_id: int, reply_msg_id: int, url: str):
    job_settings = _with_required_coverage(get_user_settings(chat_id))
    chunk_duration = int(job_settings.get("chunk") or CHUNK_DURATION)
    job_id = f"j{secrets.token_hex(5)}"
    if chat_id in ACTIVE_JOBS:
        await send_text(chat_id, "⚠️ အလုပ်တစ်ခု လုပ်နေဆဲပါ။ ပြီးအောင်စောင့်ပါ သို့မဟုတ် 🛑 ရပ်မယ် ခလုတ်နှိပ်ပါ။",
                        reply_to=reply_msg_id, reply_markup=main_menu_keyboard(chat_id))
        return
    ACTIVE_JOBS[chat_id] = job_id
    status_id = await send_text(
        chat_id,
        f"⚡ Recap စတင်နေပါပြီ...\n🎯 Final duration: မူရင်း၏ {job_settings['coverage']}% · 🎙️ {job_settings['voice']} · 📐 {job_settings['resolution']}p",
        reply_to=reply_msg_id,
        reply_markup=job_progress_keyboard(),
    )

    # Optional n8n orchestration.  The Telegram webhook/menu/status remain
    # here; n8n only calls this service's authenticated stage endpoints.
    # If forwarding fails, clean only the worker record and continue with
    # the unchanged direct pipeline below (ACTIVE_JOBS/job_id are kept).
    if PIPELINE_MODE == "n8n" and N8N_WEBHOOK_URL:
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
        await tracker.update("🔍 ဗီဒီယို အချိန်စစ်ဆေးနေပါသည်...", force=True)
        probe_dur = await get_source_duration(url)

        source_path = os.path.join(work_dir, "source.mp4")
        await tracker.update("📥 မူရင်းဗီဒီယို ဒေါင်းနေပါသည်...", force=True)
        total_dur = await download_full(url, source_path, tracker, probe_dur)

        if total_dur <= 0:
            raise RuntimeError("မူရင်းဗီဒီယို ဒေါင်းမရပါ။")

        size_mb = os.path.getsize(source_path) / (1024 * 1024)
        logger.info(f"Full source: {total_dur:.1f}s = {fmt_time(total_dur)}, {size_mb:.0f}MB")
        await tracker.update(
            f"✅ မူရင်းရယူပြီး ({fmt_time(total_dur)}, {size_mb:.0f}MB)\n"
            f"✂️ အပိုင်းခွဲနေပါသည်...",
            force=True,
        )

        total_parts = max(1, int(np.ceil(total_dur / chunk_duration)))
        chunks = []

        for part_num in range(1, total_parts + 1):
            if job_id in CANCELLED_JOBS:
                raise Exception("cancelled")

            start = (part_num - 1) * chunk_duration
            end = min(start + chunk_duration, total_dur)
            chunk_dur = end - start

            chunk_path = os.path.join(work_dir, f"chunk_{part_num}.mp4")

            await tracker.update(
                f"✂️ အပိုင်း {part_num}/{total_parts} ခွဲနေပါသည်...\n"
                f"📊 {fmt_progress_bar(int(((part_num - 1) / total_parts) * 100))}",
                force=True,
            )

            extract_result = await asyncio.to_thread(run_cmd, [
                FFMPEG_BIN, "-y",
                "-ss", f"{start:.3f}",
                "-i", source_path,
                "-t", f"{chunk_dur:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                chunk_path,
            ], 600)

            if extract_result.returncode != 0 or not os.path.exists(chunk_path):
                raise RuntimeError(f"အပိုင်း {part_num} ဖြတ်၍မရပါ။")

            actual_dur, _, _ = probe_video_info(chunk_path)
            if actual_dur <= 0:
                raise RuntimeError(f"အပိုင်း {part_num} ကြာချိန် မရပါ။")

            logger.info(f"Chunk {part_num}/{total_parts}: {actual_dur:.1f}s")

            await tracker.update(
                f"🎧 အပိုင်း {part_num}/{total_parts} ကို Recap လုပ်နေပါသည်...\n"
                f"📊 {fmt_progress_bar(int((start / total_dur) * 100))}",
                force=True,
            )

            async with video_render_semaphore:
                recap_chunk, new_summary = await build_chunk_recap(
                    chunk_path,
                    actual_dur,
                    start,
                    start + actual_dur,
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

        await tracker.update("🔗 အပိုင်းတွေ ပေါင်းစပ်နေပါသည်...", force=True)
        final_out = os.path.join(work_dir, "final_recap.mp4")
        concat_final = os.path.join(work_dir, "final_concat.txt")
        with open(concat_final, "w") as f:
            for c in chunks:
                f.write(f"file '{c}'\n")

        run_cmd([
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_final,
            "-c", "copy",
            "-movflags", "+faststart",
            final_out,
        ], 300)

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
            await edit_text(chat_id, status_id, f"❌ အမှား: {type(e).__name__} - {str(e)[:120]}", reply_markup=main_menu_keyboard(chat_id))
    finally:
        ACTIVE_JOBS.pop(chat_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def job_status_text(chat_id) -> str:
    job = ACTIVE_JOBS.get(chat_id)
    s = get_user_settings(chat_id)
    if job:
        return (f"📊 <b>အလုပ်အခြေအနေ</b>\n━━━━━━━━━━━━━━━\n🟢 လုပ်နေဆဲ (Job: <code>{job}</code>)\n"
                f"🎯 Final duration: မူရင်း၏ {s['coverage']}% · 🎙️ {s['voice']}\n\nရပ်ချင်ရင် 🛑 ခလုတ်နှိပ်ပါ။")
    return ("📊 <b>အလုပ်အခြေအနေ</b>\n━━━━━━━━━━━━━━━\n⚪ လက်ရှိ လုပ်နေတာ မရှိပါ။\n\n"
            "🎬 Recap စဖို့ link ပို့လိုက်ပါ (သို့) ခလုတ်နှိပ်ပါ။")


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
        await show(MAIN_MENU_TEXT, main_menu_keyboard(chat_id))
    elif data == "menu:settings":
        await show("⚙️ <b>ဆက်တင်များ (Feature ၂၀)</b>\n━━━━━━━━━━━━━━━\nပြောင်းချင်တဲ့ ခလုတ်ကို နှိပ်ပါ။\n🟢 = On · 🔴 = Off", settings_keyboard(chat_id))
    elif data == "menu:recap":
        await show("🎬 <b>Recap စတင်ရန်</b>\n\nဗီဒီယို link (http...) ကို ဒီအတိုင်း စာပို့လိုက်ပါ။\nဥပမာ: https://...", main_menu_keyboard(chat_id))
    elif data == "menu:status":
        await show(job_status_text(chat_id), main_menu_keyboard(chat_id))
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
        url = PENDING_LINKS.pop(chat_id, None)
        await show("🚀 စတင်လိုက်ပါပြီ...", main_menu_keyboard(chat_id))
        if url:
            _spawn_background(
                handle_link_job(chat_id, message_id, url),
                name="telegram-link-job",
            )
    elif data == "job:discard":
        PENDING_LINKS.pop(chat_id, None)
        await show("❌ Link ကို ပယ်လိုက်ပါပြီ။", main_menu_keyboard(chat_id))


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
    text = (msg.get("text") or "").strip()
    if not is_allowed(from_id):
        await send_text(chat_id, "⛔ ဒီ bot ကို သုံးခွင့် မရှိပါ။", reply_to=msg_id)
        return

    # Install the bottom buttons on the first ordinary message/link too;
    # the user should not have to discover that /start is required after
    # every deployment.  /start and /menu force-install via send_main_menu.
    first_token = text.split()[0].split("@")[0].lower() if text else ""
    if text and first_token not in ("/start", "/menu"):
        await ensure_persistent_keyboard(chat_id, reply_to=msg_id)

    # Persistent reply-keyboard buttons arrive as ordinary text messages,
    # not callback_query updates.  Route their exact labels explicitly;
    # without this mapping the bottom buttons visibly exist but only fall
    # through to the generic menu response.
    if text == PERSISTENT_BUTTON_RECAP:
        await send_text(
            chat_id,
            "🎬 <b>Recap စတင်ရန်</b>\n\nဗီဒီယို link (http...) ကို ဒီအတိုင်း စာပို့လိုက်ပါ။\n"
            "⏱️ Final ဗီဒီယိုကြာချိန်ကို မူရင်းဗီဒီယို၏ 90% အတိအကျနီးပါး ချိန်ညှိပေးပါမယ်။",
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
        await send_text(chat_id, job_status_text(chat_id), reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
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
    elif text.startswith("http"):
        s = get_user_settings(chat_id)
        if s.get("auto_start"):
            await handle_link_job(chat_id, msg_id, text)
        else:
            PENDING_LINKS[chat_id] = text
            await send_text(chat_id, "🎬 <b>ဒီ link ကို Recap လုပ်မလား?</b>\n" + text[:200], reply_to=msg_id, reply_markup=confirm_keyboard())
    elif text:
        await send_main_menu(
            chat_id, reply_to=msg_id, with_persistent_keyboard=True
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
BUILD_VERSION = "v3.5-button-webhook-healthz-fix"
BUILD_FEATURES = 20


@app.get("/")
async def root():
    return JSONResponse({
        "app": "rednote-bot",
        "version": BUILD_VERSION,
        "features": BUILD_FEATURES,
        "status": "online",
        "pipeline_mode": PIPELINE_MODE,
        # Backward-compatible field name from v3.2.  Semantically this is
        # now the minimum selectable *final video duration* percentage.
        "minimum_script_coverage_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "target_final_duration_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "final_duration_enforcement": "measured_tts_and_rendered_file",
        "first_person_enforcement": "per_scene_validated",
        "telegram_buttons": "inline_persistent_and_menu",
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
        "minimum_script_coverage_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "target_final_duration_percent": MINIMUM_SCRIPT_COVERAGE_PERCENT,
        "final_duration_enforcement": "measured_tts_and_rendered_file",
        "first_person_enforcement": "per_scene_validated",
        "telegram_buttons": "inline_persistent_and_menu",
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

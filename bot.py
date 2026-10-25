# ==============================================================================
# bot.py — Rednote Multi-Platform Recap Bot (3GB+ Video Safe Version)
# Disk-aware, segment-based, low-RAM, streaming download
# ==============================================================================
import asyncio
import gc
import json
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.parse
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

import edge_tts
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rednote-mm-bot")


# ==============================================================================
# FFmpeg Setup
# ==============================================================================
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


# ==============================================================================
# Disk-aware configuration (3GB+ video support)
# ==============================================================================
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else "/tmp"


def _get_free_disk_mb(path: str) -> int:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) // (1024 * 1024)
    except Exception:
        return 512


FREE_DISK_MB = _get_free_disk_mb(WORK_ROOT)
logger.info(f"WORK_ROOT={WORK_ROOT}, FREE_DISK_MB={FREE_DISK_MB}")

# Segment processing budgets
SEGMENT_TARGET_MB = max(80, min(400, FREE_DISK_MB // 4))
SEGMENT_MAX_DURATION = 240  # seconds per segment
SEGMENT_SAFETY_MB = max(150, min(800, FREE_DISK_MB // 2))

# File size limits
MAX_TG_BYTES = 47 * 1024 * 1024
MAX_SOURCE_BYTES = min(
    4 * 1024 * 1024 * 1024,
    max(500 * 1024 * 1024, FREE_DISK_MB * 1024 * 1024 * 2),
)
MAX_BOT_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_VIDEO_SECONDS = 6 * 3600


# ==============================================================================
# Environment
# ==============================================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://rednote-bot-gzd0.onrender.com").strip()

ADMIN_ID = int(os.environ.get("ADMIN_ID", "8430004524").strip() or "8430004524")
ALLOWED_USER_IDS = {8855067536}

SAMPLE_RATE = 24000

# Cache directories
VOICE_STATE_FILE = os.path.join(WORK_ROOT, "admin_voice_clones.json")
REF_CACHE_DIR = os.path.join(WORK_ROOT, "voice_refs")
PREVIEW_CACHE_DIR = os.path.join(WORK_ROOT, "voice_previews")
BGM_CACHE_DIR = os.path.join(WORK_ROOT, "bgm_cache")
FONT_CACHE_DIR = os.path.join(WORK_ROOT, "font_cache")
PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
for _d in (REF_CACHE_DIR, PREVIEW_CACHE_DIR, BGM_CACHE_DIR, FONT_CACHE_DIR, PUBLISH_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)


# ==============================================================================
# Voice & Size presets
# ==============================================================================
BUILTIN_VOICES = OrderedDict([
    ("thiha_deep", {"label": "🎙️ ၁။ ကိုမင်းခန့် (Recap အသံနက် · 🔒)", "short": "🎙️ ၁။ ကိုမင်းခန့် (အသံနက်)", "engine": "edge", "voice": "my-MM-ThihaNeural", "rate": "+2%", "pitch": "-10Hz", "eq": "equalizer=f=180:t=q:w=1.1:g=3.4,equalizer=f=3200:t=q:w=1.2:g=1.8", "gender": "male"}),
    ("thiha_natural", {"label": "🎙️ ၂။ ကိုသီဟ (သဘာဝ အသံကြည် · 🔒)", "short": "🎙️ ၂။ ကိုသီဟ (သဘာဝ)", "engine": "edge", "voice": "my-MM-ThihaNeural", "rate": "+3%", "pitch": "-2Hz", "eq": "", "gender": "male"}),
    ("thiha_fast", {"label": "🎙️ ၃။ ကိုဇော် (TikTok သွက်လက် · 🔒)", "short": "🎙️ ၃။ ကိုဇော် (သွက်လက်)", "engine": "edge", "voice": "my-MM-ThihaNeural", "rate": "+9%", "pitch": "+3Hz", "eq": "equalizer=f=2800:t=q:w=1.2:g=2.0", "gender": "male"}),
    ("nilar_sweet", {"label": "👩 ၄။ မနီလာ (မိန်းကလေး အသံချို · 🔒)", "short": "👩 ၄။ မနီလာ (အသံချို)", "engine": "edge", "voice": "my-MM-NilarNeural", "rate": "+3%", "pitch": "+0Hz", "eq": "", "gender": "female"}),
    ("nilar_calm", {"label": "👩 ၅။ မမေ (Story မိန်းကလေး · 🔒)", "short": "👩 ၅။ မမေ (တည်ငြိမ်)", "engine": "edge", "voice": "my-MM-NilarNeural", "rate": "+4%", "pitch": "-5Hz", "eq": "equalizer=f=220:t=q:w=1.1:g=2.2", "gender": "female"}),
    ("charon", {"label": "🤖 ၆။ Charon AI (Deep Male)", "short": "🤖 ၆။ Charon AI", "engine": "gemini", "voice": "Charon", "fallback": "thiha_deep", "gender": "male"}),
    ("fenrir", {"label": "🤖 ၇။ Fenrir AI (Bold Male)", "short": "🤖 ၇။ Fenrir AI", "engine": "gemini", "voice": "Fenrir", "fallback": "thiha_natural", "gender": "male"}),
    ("kore", {"label": "🤖 ၈။ Kore AI (Soft Female)", "short": "🤖 ၈။ Kore AI", "engine": "gemini", "voice": "Kore", "fallback": "nilar_sweet", "gender": "female"}),
])

VIDEO_SIZES = OrderedDict([
    ("9_16", {"label": "📱 9:16 (TikTok / Reels)", "st1": (720, 1280), "uhd": (2160, 3840)}),
    ("16_9", {"label": "🖥️ 16:9 (YouTube / FB)", "st1": (1280, 720), "uhd": (3840, 2160)}),
    ("orig", {"label": "📐 မူရင်းအချိုးအစားအတိုင်း · HD", "st1": (0, 0), "uhd": (0, 0)}),
    ("1_1", {"label": "🟦 1:1 (Square)", "st1": (960, 960), "uhd": (2160, 2160)}),
    ("4_5", {"label": "🖼️ 4:5 (FB Portrait)", "st1": (864, 1080), "uhd": (2160, 2700)}),
])


# ==============================================================================
# State
# ==============================================================================
VOICE_STATE = {
    "active": "charon",
    "video_size": "orig",
    "copyright_shield": True,
    "bgm_mode": "orig",
    "custom_bgm": None,
    "clones": [],
    "vault_msg_id": 0,
    "loaded_from_tg": False,
}
AWAITING_VOICE_SAMPLE = False
AWAITING_BGM_UPLOAD = False
AWAITING_TITLE_INFO = None
PUBLISHED_JOBS: OrderedDict = OrderedDict()
PENDING_LINKS: OrderedDict = OrderedDict()
JOB_RETRY_ARGS: OrderedDict = OrderedDict()
DOWNLOAD_TOKENS: dict = {}
CANCELLED_JOBS: set = set()

if os.path.exists(VOICE_STATE_FILE):
    try:
        with open(VOICE_STATE_FILE, "r", encoding="utf-8") as vf:
            VOICE_STATE.update(json.load(vf))
    except Exception:
        pass

processed_updates = deque(maxlen=600)
MAX_CONCURRENT_PROJECTS = 2
job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROJECTS)
audio_analysis_semaphore = asyncio.Semaphore(1)
speech_generation_semaphore = asyncio.Semaphore(4)
video_render_semaphore = asyncio.Semaphore(1)
recap_pipeline_semaphore = asyncio.Semaphore(1)
JOB_ADMISSION_LOCK = asyncio.Lock()
RESERVED_JOB_SLOTS = 0
ACTIVE_JOBS = 0
JOB_STATES: dict = {}
JOB_MONITOR_STATE = {
    "state": "idle", "job_id": "", "chat_id": 0, "status_message_id": 0,
    "phase": "အဆင်သင့်", "detail": "လက်ရှိလုပ်ငန်း မရှိပါ။", "progress": 0,
    "chunk": 0, "total_chunks": 0, "started_at": 0,
    "last_progress_at": 0, "last_watchdog_alert": 0, "updated_at": int(time.time()),
}
JOB_MONITOR_LAST_SYNC = 0.0
JOB_MONITOR_SYNC_PENDING = False
STATS = {"completed_hd": 0, "cloned_jobs": 0, "started_at": int(time.time())}

PREFERRED_MODELS = [
    "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-3.5-flash",
    "gemini-flash-latest", "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash-lite", "gemini-2.5-flash",
]
PREFERRED_TTS_MODELS = [
    "gemini-3.8-flash-tts", "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]
TTS_COOLDOWN_UNTIL: dict = {}

VAULT_TAG = "🔐ADMIN_VOICE_VAULT_JSON:"
MYANMAR_DIGITS = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")


# ==============================================================================
# Subprocess helper
# ==============================================================================
def run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class JobCancelled(Exception):
    pass


class VoiceEngineUnavailable(RuntimeError):
    pass


@asynccontextmanager
async def cancellable_semaphore(semaphore: asyncio.Semaphore, job_id: str):
    acquired = False
    while not acquired:
        if job_id in CANCELLED_JOBS:
            yield False
            return
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=0.5)
            acquired = True
        except asyncio.TimeoutError:
            continue
    if job_id in CANCELLED_JOBS:
        semaphore.release()
        yield False
        return
    try:
        yield True
    finally:
        semaphore.release()# ==============================================================================
# Telegram API
# ==============================================================================
async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 90.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    t_cfg = httpx.Timeout(connect=15.0, write=max(30.0, timeout), read=max(30.0, timeout), pool=15.0)
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=t_cfg, http2=False) as client:
                if files:
                    resp = await client.post(url, data=data or {}, files=files)
                else:
                    resp = await client.post(url, json=data or {})
                return resp.json()
        except Exception as e:
            logger.warning(f"tg_api({method}) attempt {attempt + 1}: {e}")
            if attempt == 2:
                return {"ok": False, "description": str(e)}
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"ok": False}


async def tg_upload_media(method: str, data: dict, field_name: str, file_name: str,
                          file_path: str, mime_type: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    payload_bytes = await asyncio.to_thread(_read_file_bytes, file_path)
    t_cfg = httpx.Timeout(connect=20.0, write=300.0, read=300.0, pool=20.0)
    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=t_cfg, http2=False) as client:
                    resp = await client.post(url, data=data,
                                            files={field_name: (file_name, payload_bytes, mime_type)})
                    res = resp.json()
                    if res.get("ok"):
                        return res
                    logger.warning(f"tg_upload_media({method}) resp: {res}")
            except Exception as e:
                logger.warning(f"tg_upload_media({method}) attempt {attempt + 1}: {e}")
            await asyncio.sleep(2.5 * (attempt + 1))
        return {"ok": False}
    finally:
        del payload_bytes
        gc.collect()


def fmt_progress_bar(pct: int) -> str:
    pct = max(0, min(100, pct))
    filled = int(round(pct / 10))
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"


class ProgressTracker:
    def __init__(self, chat_id: int, message_id: int, reply_markup: dict = None,
                 min_interval: float = 3.0, job_id: str = None):
        self.chat_id = chat_id
        self.message_id = message_id
        self.reply_markup = reply_markup
        self.job_id = job_id
        self.min_interval = min_interval
        self.last_ts = 0.0
        self.last_text = ""

    async def update(self, text: str, force: bool = False):
        if not self.message_id or text == self.last_text:
            return
        now = time.monotonic()
        if not force and (now - self.last_ts) < self.min_interval:
            return
        self.last_ts = now
        self.last_text = text
        if self.job_id and self.job_id in JOB_STATES:
            JOB_STATES[self.job_id]["phase"] = text.split("\n")[0][:180]
            JOB_STATES[self.job_id]["detail"] = text[:500]
            JOB_STATES[self.job_id]["updated_at"] = int(time.time())
        await edit_text(self.chat_id, self.message_id, text, reply_markup=self.reply_markup)


async def send_text(chat_id: int, text: str, reply_to: int = None, reply_markup: dict = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    if reply_markup:
        payload["reply_markup"] = reply_markup
    res = await tg_api("sendMessage", payload, timeout=25.0)
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str, reply_markup: dict = None):
    if not message_id:
        return
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        await tg_api("editMessageText", payload, timeout=20.0)
    except Exception:
        pass


def update_job_state(job_id: str, **updates):
    state = JOB_STATES.setdefault(job_id, {"job_id": job_id})
    state.update(updates)
    if JOB_MONITOR_STATE.get("job_id") == job_id:
        JOB_MONITOR_STATE.update(updates)


def format_job_status(chat_id: int = None) -> str:
    candidates = list(JOB_STATES.values())
    if chat_id is not None:
        candidates = [s for s in candidates if s.get("chat_id") == chat_id]
    active = sorted([s for s in candidates if s.get("state") in ("running", "queued")],
                    key=lambda s: int(s.get("started_at") or 0), reverse=True)
    if active:
        lines = [f"🟢 သင့် Video Project {len(active)} ခု လုပ်ဆောင်နေပါသည်။"]
        for i, item in enumerate(active, 1):
            lines.append(f"\n{i}. {item.get('phase') or '—'}")
            lines.append(fmt_progress_bar(int(item.get("progress") or 0)))
        return "\n".join(lines)
    return "🟡 လက်ရှိ video လုပ်ငန်း မရှိပါ။ Link/File ပို့ပါ။"


def save_voice_state_local():
    try:
        with open(VOICE_STATE_FILE, "w", encoding="utf-8") as vf:
            json.dump(VOICE_STATE, vf, ensure_ascii=False)
    except Exception:
        pass


async def sync_voice_state_to_telegram():
    save_voice_state_local()


async def restore_voice_state_from_telegram():
    if VOICE_STATE.get("loaded_from_tg") or not BOT_TOKEN:
        return
    VOICE_STATE["loaded_from_tg"] = True


def detect_platform_label(url: str) -> str:
    u = url.lower()
    if any(k in u for k in ("xiaohongshu.com", "xhslink.com", "rednote.com")): return "📕 Rednote"
    if "douyin.com" in u: return "🎶 Douyin"
    if "tiktok.com" in u: return "🎵 TikTok"
    if "youtu" in u: return "▶️ YouTube/Shorts"
    if "facebook.com" in u or "fb.watch" in u: return "📘 Facebook"
    if "instagram.com" in u: return "📸 Instagram"
    if "bilibili.com" in u or "b23.tv" in u: return "📺 Bilibili"
    return "🌐 Video"


def get_active_voice_id() -> str:
    VOICE_STATE["active"] = "charon"
    return "charon"


def get_active_voice_label() -> str:
    return BUILTIN_VOICES["charon"]["label"]


def get_active_voice_gender() -> str:
    return "male"


def get_active_size_key() -> str:
    return "orig"


def get_active_size_label() -> str:
    return VIDEO_SIZES["orig"]["label"]


def get_public_base_url() -> str:
    return (RENDER_EXTERNAL_URL or "https://rednote-bot-gzd0.onrender.com").rstrip("/")


def studio_menu_text() -> str:
    return "📩 Video Link (သို့) ဖိုင် ပို့ပါ။\n🎙️ Charon AI · မူရင်းအချိုးအစား · HD"


def studio_menu_keyboard() -> dict:
    return {"inline_keyboard": []}


async def download_telegram_file(file_id: str, dest_path: str) -> bool:
    try:
        r = await tg_api("getFile", {"file_id": file_id}, timeout=20.0)
        fpath = (r.get("result") or {}).get("file_path")
        if not fpath:
            return False
        dl_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{fpath}"
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            resp = await client.get(dl_url)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return True
    except Exception as e:
        logger.warning(f"download_telegram_file error: {e}")
    return False


# ==============================================================================
# Download Engine — Segment-aware for 3GB+ videos
# ==============================================================================
def _download_total_bytes(headers) -> int:
    try:
        cr = headers.get("content-range", "")
        m = re.search(r"/([0-9]+)$", cr)
        if m:
            return int(m.group(1))
        return int(headers.get("content-length", "0") or 0)
    except Exception:
        return 0


async def run_yt_dlp_with_progress(cmd: list[str], tracker=None, job_id: str = "",
                                    timeout: int = 1800):
    if job_id and job_id in CANCELLED_JOBS:
        raise JobCancelled("Owner cancelled job")
    actual_cmd = list(cmd)
    actual_cmd[1:1] = ["--newline", "--progress"]
    proc = await asyncio.create_subprocess_exec(
        *actual_cmd, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    output_lines = []
    started = time.monotonic()
    last_transfer = started
    last_reported_pct = -1
    timed_out = False
    cancelled = False
    while True:
        if job_id and job_id in CANCELLED_JOBS:
            cancelled = True
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout or time.monotonic() - last_transfer > 120:
            timed_out = True
            break
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if not line:
            break
        output_lines.append(line)
        if len(output_lines) > 2000:
            output_lines = output_lines[-1000:]
        text = line.decode(errors="ignore").strip()
        match = re.search(r"\[download\]\s+([0-9]+(?:\.[0-9]+)?)%\s+of\s+(.+?)(?:\s+at\s+|\s+ETA|$)", text)
        if match:
            last_transfer = time.monotonic()
            pct = max(0, min(100, int(float(match.group(1)))))
            if pct >= last_reported_pct and tracker:
                last_reported_pct = pct
                overall = 8 + int(round(12 * pct / 100))
                await tracker.update(
                    f"📥 Video ရယူနေပါသည် · {overall}%\n"
                    f"📦 {match.group(1)}% · {match.group(2)[:70]}\n"
                    f"📊 {fmt_progress_bar(overall)}",
                    force=True,
                )
    if cancelled or timed_out:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try: proc.terminate()
            except: pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except: proc.kill()
            await proc.wait()
        if cancelled:
            raise JobCancelled("Owner cancelled job")
        tail = b"".join(output_lines[-25:]) + b"\nDownloader stalled or timed out"
        return subprocess.CompletedProcess(actual_cmd, -9, b"", tail)
    code = await proc.wait()
    return subprocess.CompletedProcess(actual_cmd, code, b"".join(output_lines), b"")


async def _download_segmented_by_time(
    raw_url: str, out_dir: str, total_dur: float,
    tracker: "ProgressTracker", job_id: str,
) -> tuple[list[str], str]:
    """Video ကို SEGMENT_MAX_DURATION စက္ကန့်စီ segment တွေအဖြစ် ခွဲပြီး ဒေါင်းတယ်။"""
    seg_files = []
    seg_count = max(1, int(np.ceil(total_dur / SEGMENT_MAX_DURATION)))
    logger.info(f"Segmented download: {seg_count} segments of {SEGMENT_MAX_DURATION}s")

    for i in range(seg_count):
        if job_id and job_id in CANCELLED_JOBS:
            raise JobCancelled("Owner cancelled job")
        start_sec = i * SEGMENT_MAX_DURATION
        end_sec = min(total_dur, start_sec + SEGMENT_MAX_DURATION)
        seg_path = os.path.join(out_dir, f"seg_{i:03d}.mp4")

        if tracker:
            pct = 8 + int(12 * (i + 1) / seg_count)
            await tracker.update(
                f"📥 Video ရယူနေပါသည် (အပိုင်း {i+1}/{seg_count})\n"
                f"📊 {fmt_progress_bar(pct)}",
                force=True,
            )

        cmd = [
            "yt-dlp", "--no-playlist", "--no-warnings",
            "--download-sections", f"*{start_sec}-{end_sec}",
            "--force-keyframes-at-cuts",
            "--socket-timeout", "45", "--retries", "3",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", seg_path, raw_url,
        ]
        await run_yt_dlp_with_progress(cmd, tracker, job_id, timeout=600)

        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 10000:
            seg_files.append(seg_path)
        else:
            if os.path.exists(seg_path):
                os.remove(seg_path)

    if not seg_files:
        raise RuntimeError("No segments downloaded")
    return seg_files, f"Segmented ({len(seg_files)} parts)"


async def download_video_in_segments(
    raw_url: str, out_dir: str, tracker: "ProgressTracker" = None, job_id: str = "",
) -> tuple[list[str], str]:
    """
    Disk-safe download: 3GB+ videos split into SEGMENT_MAX_DURATION chunks.
    """
    os.makedirs(out_dir, exist_ok=True)
    ul = raw_url.lower()

    # Probe duration first
    probe_cmd = [
        "yt-dlp", "--no-playlist", "--no-warnings",
        "--print", "%(duration)s|%(filesize,filesize_approx)s",
        raw_url,
    ]
    duration = 0.0
    est_size = 0
    try:
        proc = await asyncio.create_subprocess_exec(
            *probe_cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
        if proc.returncode == 0 and out:
            line = out.decode().strip().split("\n")[0]
            parts = line.split("|")
            if parts[0] not in ("NA", "None", ""):
                duration = float(parts[0])
            if len(parts) > 1 and parts[1] not in ("NA", "None", ""):
                est_size = int(float(parts[1]))
    except Exception as e:
        logger.warning(f"Probe failed: {e}")

    logger.info(f"Download probe: duration={duration:.1f}s, size_est={est_size//(1024*1024)}MB")

    # Decide: segment or single?
    if duration > SEGMENT_MAX_DURATION or est_size > SEGMENT_SAFETY_MB * 1024 * 1024:
        logger.info("→ Segmented download")
        return await _download_segmented_by_time(raw_url, out_dir, duration, tracker, job_id)

    # Single file download
    out_template = os.path.join(out_dir, "seg_000.mp4")
    cmd = [
        "yt-dlp", "--no-playlist", "--no-warnings",
        "--socket-timeout", "45", "--retries", "5",
        "--fragment-retries", "5", "--concurrent-fragments", "2",
        "--max-filesize", f"{SEGMENT_TARGET_MB}M",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--no-part",
        "-o", out_template, raw_url,
    ]
    await run_yt_dlp_with_progress(cmd, tracker, job_id, timeout=1800)

    if os.path.exists(out_template) and os.path.getsize(out_template) > 10000:
        return [out_template], "HD Single"

    # Rednote fallback
    if any(k in ul for k in ("xiaohongshu.com", "xhslink.com", "rednote.com")):
        ok = await _rednote_direct_extract(raw_url, out_template, tracker, job_id)
        if ok:
            return [out_template], "Rednote Direct"

    # TikTok/Douyin fallback
    if "tiktok.com" in ul or "douyin.com" in ul:
        ok = await _tiktok_direct_extract(raw_url, out_template, tracker, job_id)
        if ok:
            return [out_template], "TikTok Direct"

    raise RuntimeError("Video download failed")


async def _rednote_direct_extract(raw_url: str, out_path: str,
                                   tracker: "ProgressTracker" = None, job_id: str = "") -> bool:
    """Rednote HTML scraping fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as client:
            r = await client.get(raw_url, headers=headers)
            html = r.text

            urls = []
            for m in re.finditer(r'"masterUrl"\s*:\s*"([^"]+)"', html):
                u = m.group(1).replace("\\u002F", "/").replace("\\/", "/").strip()
                if u.startswith("http"):
                    urls.append(u)
            for m in re.finditer(r'"originVideoKey"\s*:\s*"([^"]+)"', html):
                k = m.group(1).lstrip("/")
                urls.append(f"https://sns-video-bd.xhscdn.com/{k}")

            for vurl in urls[:5]:
                try:
                    async with client.stream("GET", vurl,
                                            headers={"Referer": "https://www.xiaohongshu.com/"},
                                            timeout=360.0) as resp:
                        if resp.status_code not in (200, 206):
                            continue
                        written = 0
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(1024 * 1024):
                                if job_id and job_id in CANCELLED_JOBS:
                                    raise JobCancelled("cancelled")
                                written += len(chunk)
                                if written > MAX_SOURCE_BYTES:
                                    return False
                                f.write(chunk)
                        if os.path.getsize(out_path) > 25000:
                            return True
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Rednote direct extract: {e}")
    return False


async def _tiktok_direct_extract(raw_url: str, out_path: str,
                                  tracker: "ProgressTracker" = None, job_id: str = "") -> bool:
    """TikWM API fallback for TikTok/Douyin."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as c:
            resp = await c.post("https://www.tikwm.com/api/", data={"url": raw_url, "hd": "1"})
            j = resp.json()
            if j.get("code") == 0 and isinstance(j.get("data"), dict):
                d = j["data"]
                for key in ("hdplay", "play", "wmplay"):
                    val = d.get(key)
                    if not val:
                        continue
                    full_u = val if val.startswith("http") else f"https://www.tikwm.com{val}"
                    try:
                        async with c.stream("GET", full_u,
                                            headers={"User-Agent": "Mozilla/5.0"},
                                            timeout=360.0) as r2:
                            if r2.status_code not in (200, 206):
                                continue
                            written = 0
                            with open(out_path, "wb") as f:
                                async for chunk in r2.aiter_bytes(1024 * 1024):
                                    if job_id and job_id in CANCELLED_JOBS:
                                        raise JobCancelled("cancelled")
                                    written += len(chunk)
                                    if written > MAX_SOURCE_BYTES:
                                        return False
                                    f.write(chunk)
                            if os.path.getsize(out_path) > 25000:
                                return True
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"TikTok direct extract: {e}")
    return False


async def download_any_platform_video(raw_url: str, out_path: str,
                                       tracker: "ProgressTracker" = None, job_id: str = ""):
    """Legacy wrapper — single file output."""
    out_dir = os.path.dirname(out_path) or "."
    segs, tier = await download_video_in_segments(raw_url, out_dir, tracker, job_id)
    if len(segs) == 1:
        if segs[0] != out_path:
            os.rename(segs[0], out_path)
        return True, tier
    concat_file = os.path.join(out_dir, "concat_all.txt")
    with open(concat_file, "w") as f:
        for s in segs:
            f.write(f"file '{s}'\n")
    run_cmd([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
             "-c", "copy", out_path], 600)
    for s in segs:
        try: os.remove(s)
        except: pass
    return (os.path.exists(out_path) and os.path.getsize(out_path) > 10000), tier# ==============================================================================
# Video analysis & FFmpeg helpers
# ==============================================================================
def probe_video_info(path: str) -> tuple[float, int, int, str]:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        dur = float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    w, h = 0, 0
    vcodec = "h264"
    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d{2,5})x(\d{2,5})", out)
    if vm:
        vcodec = vm.group(1).lower()
        w, h = int(vm.group(2)), int(vm.group(3))
    rot_m = re.search(r"rotation of\s*(-?\d+)|rotate\s*:\s*(-?\d+)", out, flags=re.I)
    if rot_m:
        ang = abs(int(rot_m.group(1) or rot_m.group(2) or 0)) % 180
        if ang == 90 and w > 0 and h > 0:
            w, h = h, w
    return dur, w, h, vcodec


def compute_stage_dims(src_w: int, src_h: int, size_key: str) -> tuple:
    if src_w <= 0 or src_h <= 0:
        src_w, src_h = 1080, 1920
    if size_key in VIDEO_SIZES and size_key != "orig":
        return VIDEO_SIZES[size_key]["st1"], VIDEO_SIZES[size_key]["uhd"]
    long_e = max(src_w, src_h)
    short_e = min(src_w, src_h)
    s1_scale = 1280.0 / float(long_e)
    s1_w = max(2, int(round(src_w * s1_scale / 2.0)) * 2)
    s1_h = max(2, int(round(src_h * s1_scale / 2.0)) * 2)
    s4_scale = min(3840.0 / float(long_e), 2160.0 / float(short_e))
    u_w = max(2, int(round(src_w * s4_scale / 2.0)) * 2)
    u_h = max(2, int(round(src_h * s4_scale / 2.0)) * 2)
    return (s1_w, s1_h), (u_w, u_h)


def clean_burmese_for_tts(text: str) -> str:
    t = re.sub(r"[*_#`~<>[\]()\"'“”‘’…—–-]", " ", text)
    t = re.sub(r"[\U00010000-\U0010ffff]", " ", t)
    t = t.translate(MYANMAR_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_speakable_text(text: str) -> bool:
    return bool(re.search(r"[\u1000-\u1049a-zA-Z0-9]", text))


async def groq_whisper_transcribe(audio_mp3_path: str) -> list:
    if not GROQ_API_KEY or not os.path.exists(audio_mp3_path):
        return []
    if os.path.getsize(audio_mp3_path) > 24 * 1024 * 1024:
        return []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for model in ("whisper-large-v3-turbo", "whisper-large-v3"):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(audio_mp3_path, "rb") as af:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        data={"model": model, "response_format": "verbose_json",
                              "timestamp_granularities[]": "segment"},
                        files={"file": ("speech.mp3", af, "audio/mpeg")},
                    )
                if resp.status_code == 200:
                    j = resp.json()
                    segs = []
                    for s in j.get("segments", []):
                        txt = (s.get("text") or "").strip()
                        if txt:
                            segs.append({"start": round(float(s.get("start", 0)), 2),
                                         "end": round(float(s.get("end", 0)), 2),
                                         "text": txt})
                    return segs
        except Exception as e:
            logger.warning(f"Groq Whisper: {e}")
    return []


async def transcribe_and_script_scenes(
    audio_mp3_path: str, total_dur: float, gender: str = "male",
    scene_range: tuple = None, visual_frames: list = None, window_index: int = 0,
) -> list:
    whisper_segs = await groq_whisper_transcribe(audio_mp3_path)
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_mp3_path, "rb") as f:
        audio_bytes = f.read()

    if scene_range:
        min_sc, max_sc = scene_range
    elif total_dur <= 90:
        min_sc, max_sc = 4, 11
    elif total_dur <= 300:
        min_sc, max_sc = 8, 18
    else:
        min_sc = max(12, min(22, int(total_dur // 35)))
        max_sc = max(min_sc + 6, min(36, int(total_dur // 20)))

    whisper_hint = ""
    if whisper_segs:
        whisper_hint = f"\nWhisper transcript:\n{json.dumps(whisper_segs[:120], ensure_ascii=False)}\n"

    frame_times = [f"{s:.1f}" for s, _ in (visual_frames or [])]
    visual_hint = f"\nFrames at seconds: {', '.join(frame_times)}.\n" if frame_times else "\nNo frames available.\n"

    prompt = f"""
You are a Myanmar YouTube long-form recap narrator. Analyze this {total_dur:.1f}-second window.
{whisper_hint}
{visual_hint}
Window index: {window_index}.

Write {min_sc}–{max_sc} narration scenes in Burmese covering the source events chronologically.
- Each scene's `start`/`end` within 0.0 and {total_dur:.1f}.
- Do not invent facts. Keep proper names.
- Return ONLY valid JSON array:
[{{"start": 3.2, "end": 8.7, "burmese": "..."}}]
""".strip()

    contents = [prompt]
    if len(audio_bytes) <= 19 * 1024 * 1024:
        contents.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"))
    for sec, fb in (visual_frames or []):
        contents.append(f"Frame @ {sec:.1f}s:")
        contents.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))
    del audio_bytes

    cfg = types.GenerateContentConfig(temperature=0.22, response_mime_type="application/json")
    raw_json = None
    last_err = None
    for model in PREFERRED_MODELS:
        try:
            resp = await asyncio.to_thread(client.models.generate_content,
                                           model=model, contents=contents, config=cfg)
            if resp and resp.text:
                raw_json = resp.text.strip()
                break
        except Exception as e:
            last_err = e
    if raw_json is None:
        raise RuntimeError(f"Gemini error: {last_err}")

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_json, flags=re.M).strip()
    parsed = json.loads(cleaned)
    segments = parsed.get("segments", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

    out = []
    for seg in segments:
        try:
            st = max(0.0, float(seg.get("start", 0.0)))
            en = min(total_dur, float(seg.get("end", st + 4.0)))
            if en <= st + 0.5:
                en = min(total_dur, st + 3.5)
            bm = clean_burmese_for_tts(str(seg.get("burmese", "")))
            if not bm or not has_speakable_text(bm):
                continue
            if not bm.endswith("။"):
                bm += "။"
            out.append({"start": st, "end": en, "burmese": bm})
        except Exception:
            continue
    out.sort(key=lambda x: x["start"])
    return out


# ==============================================================================
# Audio synthesis
# ==============================================================================
def trim_pcm_silence(pcm: np.ndarray, thresh: float = 0.011) -> np.ndarray:
    if len(pcm) == 0:
        return pcm
    hop = int(0.015 * SAMPLE_RATE)
    if len(pcm) < hop * 2:
        return pcm
    active = np.where(np.abs(pcm) > thresh)[0]
    if len(active) == 0:
        return pcm
    pad = int(0.035 * SAMPLE_RATE)
    s = max(0, int(active[0]) - pad)
    e = min(len(pcm), int(active[-1]) + pad)
    return pcm[s:e]


def normalize_clip_loudness(pcm: np.ndarray, target_rms: float = 0.155) -> np.ndarray:
    if len(pcm) < 200:
        return pcm
    active = pcm[np.abs(pcm) > 0.018]
    cur_rms = float(np.sqrt(np.mean(active * active))) if len(active) > 100 else float(np.sqrt(np.mean(pcm * pcm)))
    if cur_rms < 1e-4:
        return pcm
    gain = float(np.clip(target_rms / cur_rms, 0.35, 3.2))
    out = pcm * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.92:
        out = np.tanh(out * (1.15 / peak)) * 0.92
    fade = min(int(0.018 * SAMPLE_RATE), len(out) // 5)
    if fade > 2:
        out = out.copy()
        out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out.astype(np.float32)


async def _synthesize_edge_locked_all(scenes: list, vcfg: dict, work_dir: str) -> list:
    sem = speech_generation_semaphore
    results = [None] * len(scenes)
    vname = vcfg["voice"]
    rate = vcfg["rate"]
    pitch = vcfg["pitch"]
    eq = vcfg.get("eq") or ""

    async def _one(idx: int, text_bm: str):
        async with sem:
            mp3_p = os.path.join(work_dir, f"edg_{idx:03d}.mp3")
            pcm_p = os.path.join(work_dir, f"edg_{idx:03d}.pcm")
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(text_bm, vname, rate=rate, pitch=pitch)
                    await comm.save(mp3_p)
                    if os.path.exists(mp3_p) and os.path.getsize(mp3_p) > 300:
                        break
                except Exception:
                    await asyncio.sleep(1.0 * (attempt + 1))
            if os.path.exists(mp3_p) and os.path.getsize(mp3_p) > 300:
                cmd = [FFMPEG_BIN, "-y", "-i", mp3_p]
                if eq:
                    cmd += ["-af", eq]
                cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
                        "-ar", str(SAMPLE_RATE), pcm_p]
                await asyncio.to_thread(run_cmd, cmd, 30)
                os.remove(mp3_p)
            if os.path.exists(pcm_p) and os.path.getsize(pcm_p) > 200:
                seg = np.fromfile(pcm_p, dtype=np.int16).astype(np.float32) / 32768.0
                os.remove(pcm_p)
                results[idx] = normalize_clip_loudness(trim_pcm_silence(seg))
            else:
                results[idx] = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)

    await asyncio.gather(*(_one(i, sc["burmese"]) for i, sc in enumerate(scenes)))
    return [r if r is not None else np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32) for r in results]


def split_master_pcm_into_n_clips(master_pcm: np.ndarray, scenes: list) -> list:
    n = len(scenes)
    master_pcm = trim_pcm_silence(master_pcm)
    if n <= 1 or len(master_pcm) < SAMPLE_RATE:
        return [normalize_clip_loudness(master_pcm)]
    weights = np.array([max(4, len(s["burmese"])) for s in scenes], dtype=np.float64)
    cum_ratios = np.cumsum(weights)[:-1] / np.sum(weights)
    hop = int(0.02 * SAMPLE_RATE)
    n_frames = max(1, (len(master_pcm) - hop) // hop)
    rms = np.array([float(np.sqrt(np.mean(master_pcm[i*hop:i*hop+hop]**2))) for i in range(n_frames)])
    cut_samples = [0]
    for r in cum_ratios:
        expected = int(r * n_frames)
        win = max(25, int(0.15 * n_frames))
        lo = max(cut_samples[-1] // hop + 10, expected - win)
        hi = min(n_frames - 10, expected + win)
        if hi <= lo:
            best = max(cut_samples[-1] // hop + 1, expected)
        else:
            dists = np.abs(np.arange(lo, hi) - expected) / max(1.0, float(win))
            score = rms[lo:hi] + 0.015 * dists
            best = lo + int(np.argmin(score))
        cut_samples.append(min(len(master_pcm), best * hop))
    cut_samples.append(len(master_pcm))
    clips = []
    for i in range(n):
        seg = trim_pcm_silence(master_pcm[cut_samples[i]:cut_samples[i+1]])
        clips.append(normalize_clip_loudness(seg))
    return clips


async def _synthesize_gemini_locked_all(scenes: list, gemini_voice: str, work_dir: str):
    if not GEMINI_API_KEY:
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    batch_size = 26
    now_m = time.monotonic()
    candidate_models = [m for m in PREFERRED_TTS_MODELS
                        if TTS_COOLDOWN_UNTIL.get(m, 0.0) <= now_m]
    if not candidate_models:
        return None
    style = ("Natural Burmese narration, conversational medium pace, clear syllables, "
             "no drawn-out vowels. Keep Charon speaker identity.")

    for chosen_model in candidate_models:
        all_clips = []
        model_ok = True
        for b_idx, start_i in enumerate(range(0, len(scenes), batch_size)):
            batch = scenes[start_i:start_i + batch_size]
            joined = "\n\n".join(s["burmese"] for s in batch)
            if chosen_model == "gemini-3.8-flash-tts":
                contents_payload = [{"role": "user", "parts": [{"text": joined, "speech_metadata": {"style": style}}]}]
                cfg = {"response_modalities": ["AUDIO"],
                       "speech_config": {"voice_config": {"voice": gemini_voice}}}
            else:
                legacy_prompt = f"Read as natural Burmese narrator. Keep Charon speaker.\n\n{joined}"
                contents_payload = legacy_prompt
                cfg = types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_voice)
                        )
                    ),
                )
            try:
                resp = await asyncio.to_thread(client.models.generate_content,
                                              model=chosen_model, contents=contents_payload, config=cfg)
                cands = getattr(resp, "candidates", None) or []
                if not cands or not getattr(cands[0], "content", None):
                    raise RuntimeError("TTS returned no candidate")
                part = cands[0].content.parts[0].inline_data
                if not part or not part.data or len(part.data) <= 1000:
                    raise RuntimeError("TTS empty audio")
                audio_bytes = part.data
                mime = (part.mime_type or "").lower()
                tmp_raw = os.path.join(work_dir, f"gem_raw_{b_idx}.bin")
                tmp_pcm = os.path.join(work_dir, f"gem_{b_idx}.pcm")
                with open(tmp_raw, "wb") as f:
                    f.write(audio_bytes)
                if audio_bytes[:4] == b"RIFF":
                    run_cmd([FFMPEG_BIN, "-y", "-i", tmp_raw, "-f", "s16le", "-acodec", "pcm_s16le",
                             "-ac", "1", "-ar", str(SAMPLE_RATE), tmp_pcm], 60)
                else:
                    rate_m = re.search(r"rate=(\d+)", mime)
                    pcm_rate = rate_m.group(1) if rate_m else "24000"
                    run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", pcm_rate, "-ac", "1",
                             "-i", tmp_raw, "-f", "s16le", "-acodec", "pcm_s16le",
                             "-ac", "1", "-ar", str(SAMPLE_RATE), tmp_pcm], 60)
                try: os.remove(tmp_raw)
                except: pass
                if not os.path.exists(tmp_pcm) or os.path.getsize(tmp_pcm) <= 1000:
                    raise RuntimeError("TTS empty after conversion")
                master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                os.remove(tmp_pcm)
                b_clips = split_master_pcm_into_n_clips(master, batch)
                if len(b_clips) != len(batch):
                    raise RuntimeError("TTS split mismatch")
                all_clips.extend(b_clips)
                del b_clips, audio_bytes
            except Exception as e:
                err_s = str(e)
                if "429" in err_s or "RESOURCE_EXHAUSTED" in err_s:
                    TTS_COOLDOWN_UNTIL[chosen_model] = time.monotonic() + 1500.0
                logger.warning(f"Gemini TTS ({chosen_model}) failed: {err_s[:200]}")
                model_ok = False
                break
        if model_ok and len(all_clips) == len(scenes):
            return all_clips
        del all_clips
        gc.collect()
    return None


async def synthesize_voice_clips(scenes: list, work_dir: str,
                                  voice_id_override: str = None,
                                  tracker: "ProgressTracker" = None) -> list:
    vid = voice_id_override or get_active_voice_id()
    if vid.startswith("clone_"):
        clone_entry = next((c for c in VOICE_STATE.get("clones", []) if c["id"] == vid), None)
        base_clips = await _synthesize_edge_locked_all(scenes, BUILTIN_VOICES["thiha_natural"], work_dir)
        if clone_entry:
            return base_clips
        return base_clips
    vcfg = BUILTIN_VOICES.get(vid, BUILTIN_VOICES["thiha_deep"])
    if tracker:
        await tracker.update(f"🎙️ [{vcfg['short']}] အသံသွင်းနေပါသည်...\n📊 {fmt_progress_bar(38)}", force=True)
    if vcfg["engine"] == "gemini":
        gem_clips = await _synthesize_gemini_locked_all(scenes, vcfg["voice"], work_dir)
        if gem_clips is not None and len(gem_clips) == len(scenes):
            return gem_clips
        raise VoiceEngineUnavailable("Charon AI အသံ ယာယီမရရှိပါ။")
    return await _synthesize_edge_locked_all(scenes, vcfg, work_dir)


# ==============================================================================
# Cinema recap builder
# ==============================================================================
async def build_cinema_recap_hd(
    orig_video: str, orig_dur: float, src_w: int, src_h: int,
    st1_dims: tuple, scenes: list, voice_clips: list,
    work_dir: str, recap_hd_path: str, copyright_shield: bool = True,
    bgm_mode: str = "orig", custom_bgm_pcm_path: str = None,
    tracker: "ProgressTracker" = None, delete_source: bool = True, job_id: str = "",
):
    out_w, out_h = st1_dims
    src_ratio = (float(src_w) / float(src_h)) if (src_w > 0 and src_h > 0) else (9.0 / 16.0)
    tgt_ratio = float(out_w) / float(out_h)
    need_blur = abs(src_ratio - tgt_ratio) / max(0.1, tgt_ratio) > 0.09
    shield_v = "crop=iw*0.96:ih*0.96,hflip," if copyright_shield else ""
    color = "eq=contrast=1.05:saturation=1.09" if copyright_shield else "eq=contrast=1.03:saturation=1.06"

    # Build master audio
    master_pcm_path = os.path.join(work_dir, "master_mix.pcm")
    video_slices = []
    with open(master_pcm_path, "wb") as pcm_f:
        for i, (sc, v_clip) in enumerate(zip(scenes, voice_clips)):
            src_s = sc["start"]
            avail_e = scenes[i + 1]["start"] if (i + 1 < len(scenes)) else orig_dur
            avail_e = max(src_s + 0.5, min(orig_dur, avail_e))
            norm_voice = normalize_clip_loudness(v_clip)
            voice_dur = len(norm_voice) / SAMPLE_RATE
            target_dur = max(0.65, voice_dur + 0.16)
            target_samples = int(round(target_dur * SAMPLE_RATE))
            avail_len = avail_e - src_s
            cut_src_e = src_s + min(avail_len, target_dur * 1.15) if avail_len >= target_dur else avail_e
            video_slices.append((src_s, cut_src_e, target_dur))
            mixed = np.zeros(target_samples, dtype=np.float32)
            v_len = min(target_samples, len(norm_voice))
            mixed[:v_len] = norm_voice[:v_len]
            peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
            if peak > 0.95:
                mixed *= 0.95 / peak
            np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)

    master_wav = os.path.join(work_dir, "master_mix.wav")
    run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-i", master_pcm_path, master_wav], 60)
    try: os.remove(master_pcm_path)
    except: pass
    gc.collect()

    seg_files = []
    tot = max(1, len(video_slices))
    for idx, (s_t, e_t, tgt_d) in enumerate(video_slices):
        if job_id and job_id in CANCELLED_JOBS:
            raise JobCancelled("cancelled")
        if tracker:
            pct = 42 + int(round((idx * 28) / tot))
            await tracker.update(
                f"⏳ HD Recap တည်းဖြတ်နေပါသည် (အပိုင်း {idx+1}/{tot})\n"
                f"📊 {fmt_progress_bar(pct)}",
                force=(idx == 0 or idx == tot - 1),
            )
        raw_d = max(0.15, e_t - s_t)
        seg_path = os.path.join(work_dir, f"seg_{idx:03d}.mp4")
        need_boom = (tgt_d / raw_d) > 1.48 and raw_d >= 0.6
        eff_raw = raw_d * 2.0 if need_boom else raw_d
        pts = tgt_d / eff_raw
        fps_chain = "minterpolate=fps=25:mi_mode=blend" if pts > 1.12 else "fps=25"
        boom_prefix = ("[0:v]split=2[f1][f2];[f2]reverse[f2r];[f1][f2r]concat=n=2:v=1:a=0[vb];[vb]"
                       if need_boom else "[0:v]")
        if need_blur:
            fc = (f"{boom_prefix}{shield_v}setpts={pts:.4f}*(PTS-STARTPTS),{fps_chain},split=2[bg][fg];"
                  f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=fast_bilinear,"
                  f"crop={out_w}:{out_h},boxblur=20:3,eq=brightness=-0.08:saturation=1.12[bg2];"
                  f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=bicubic,"
                  f"{color}[fg2];"
                  f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]")
        else:
            fc = (f"{boom_prefix}{shield_v}setpts={pts:.4f}*(PTS-STARTPTS),{fps_chain},"
                  f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,"
                  f"crop={out_w}:{out_h},{color},format=yuv420p[vout]")
        async with cancellable_semaphore(video_render_semaphore, job_id) as acq:
            if not acq:
                raise JobCancelled("cancelled")
            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-ss", f"{s_t:.3f}", "-t", f"{raw_d:.3f}",
                 "-i", orig_video, "-an", "-filter_complex", fc,
                 "-map", "[vout]", "-t", f"{tgt_d:.3f}",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                 "-maxrate", "3200k", "-bufsize", "6400k",
                 "-x264-params", "rc-lookahead=0:sync-lookahead=0:ref=1:bframes=0:threads=1",
                 "-video_track_timescale", "90000", "-threads", "1", seg_path], 180,
            )
        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 500:
            seg_files.append(seg_path)

    if not seg_files:
        raise RuntimeError("No video slices generated")
    if delete_source and os.path.exists(orig_video):
        os.remove(orig_video)

    concat_list = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list, "w") as f:
        for tp in seg_files:
            f.write(f"file '{tp}'\n")
    proc = await asyncio.to_thread(
        run_cmd,
        [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-i", master_wav, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", "-movflags", "+faststart", recap_hd_path], 120,
    )
    for tp in seg_files:
        try: os.remove(tp)
        except: pass
    try: os.remove(master_wav)
    except: pass
    if not os.path.exists(recap_hd_path) or os.path.getsize(recap_hd_path) < 5000:
        raise RuntimeError(f"Concat failed: {proc.stderr.decode()[-300:]}")


# ==============================================================================
# Long video recap — segmented
# ==============================================================================
async def build_long_video_recap_hd(
    orig_video: str, total_dur: float, src_w: int, src_h: int,
    stage_dims: tuple, work_dir: str, recap_hd_path: str,
    tracker: "ProgressTracker", job_id: str, gender: str = "male",
) -> float:
    window_sec = min(SEGMENT_MAX_DURATION, 240)
    total_windows = max(1, int(np.ceil(total_dur / window_sec)))
    logger.info(f"Long recap: {total_windows} windows × {window_sec}s")

    all_scenes = []
    analysis_sem = asyncio.Semaphore(1)  # RAM safety
    analysis_counter = {"done": 0}
    analysis_lock = asyncio.Lock()

    async def process_window(idx: int) -> tuple:
        if job_id in CANCELLED_JOBS:
            raise JobCancelled("cancelled")
        async with cancellable_semaphore(analysis_sem, job_id) as acq:
            if not acq:
                raise JobCancelled("cancelled")
            offset = idx * window_sec
            span = min(window_sec, total_dur - offset)
            audio_path = os.path.join(work_dir, f"audio_{idx:03d}.mp3")
            frame_dir = os.path.join(work_dir, f"frames_{idx:03d}")
            os.makedirs(frame_dir, exist_ok=True)
            try:
                proc = await asyncio.to_thread(
                    run_cmd,
                    [FFMPEG_BIN, "-y", "-ss", f"{offset:.3f}", "-t", f"{span:.3f}",
                     "-i", orig_video, "-vn", "-ac", "1", "-ar", "16000",
                     "-b:a", "24k", audio_path], 600,
                )
                if proc.returncode != 0 or not os.path.exists(audio_path):
                    return idx, []
                visual_frames = []
                frame_count = min(6, max(1, int(span / 30)))
                frame_step = max(1.0, span / frame_count)
                pattern = os.path.join(frame_dir, "f_%02d.jpg")
                await asyncio.to_thread(
                    run_cmd,
                    [FFMPEG_BIN, "-y", "-ss", f"{offset:.3f}", "-t", f"{span:.3f}",
                     "-i", orig_video, "-map", "0:v:0", "-an",
                     "-vf", f"fps=1/{frame_step:.2f},scale=320:-2",
                     "-frames:v", str(frame_count), "-q:v", "7", pattern], 180,
                )
                for i, fp in enumerate(sorted(os.listdir(frame_dir))[:frame_count]):
                    fb = await asyncio.to_thread(_read_file_bytes, os.path.join(frame_dir, fp))
                    if fb:
                        visual_frames.append((i * frame_step, fb))
                lo = max(2, min(35, int(np.ceil(span / 12))))
                hi = max(lo + 2, min(40, int(np.ceil(span / 9))))
                scenes = await transcribe_and_script_scenes(
                    audio_path, span, gender=gender,
                    scene_range=(lo, hi), visual_frames=visual_frames, window_index=idx,
                )
                abs_scenes = [{
                    **sc,
                    "start": round(offset + float(sc["start"]), 2),
                    "end": round(min(offset + span, offset + float(sc["end"])), 2),
                } for sc in scenes]
                return idx, abs_scenes
            finally:
                try: os.remove(audio_path)
                except: pass
                shutil.rmtree(frame_dir, ignore_errors=True)

    async def report_done(result):
        async with analysis_lock:
            analysis_counter["done"] += 1
            done = analysis_counter["done"]
            pct = 20 + int(16 * done / total_windows)
            await tracker.update(
                f"🎧 Audio analysis: {done}/{total_windows}\n📊 {fmt_progress_bar(pct)}",
                force=(done == total_windows),
            )
        return result

    async def bounded(idx):
        return await report_done(await process_window(idx))

    # Sequential processing — RAM-safe for 3GB videos
    for idx in range(total_windows):
        _, abs_scenes = await bounded(idx)
        all_scenes.extend(abs_scenes)

    if not all_scenes:
        raise RuntimeError("No spoken recap scenes")

    if job_id in CANCELLED_JOBS:
        raise JobCancelled("cancelled")

    await tracker.update("🎛️ အသံနှင့် HD တည်းဖြတ်မှု စတင်ပါပြီ...", force=True)
    async with cancellable_semaphore(recap_pipeline_semaphore, job_id) as acq:
        if not acq:
            raise JobCancelled("cancelled")
        all_clips = await synthesize_voice_clips(
            all_scenes, work_dir, voice_id_override="charon", tracker=tracker,
        )
        if len(all_clips) != len(all_scenes):
            raise RuntimeError("Voice synthesis incomplete")

        # Split scenes into windows by time
        window_part_paths = []
        for w_idx in range(total_windows):
            w_start_t = w_idx * window_sec
            w_end_t = w_start_t + window_sec
            idxs = [i for i, sc in enumerate(all_scenes)
                    if w_start_t <= sc["start"] < w_end_t]
            if not idxs:
                continue
            window_scenes = [all_scenes[i] for i in idxs]
            window_clips = [all_clips[i] for i in idxs]

            part_path = os.path.join(work_dir, f"recap_w_{w_idx:03d}.mp4")
            await tracker.update(
                f"🎞️ Recap တည်းဖြတ်နေပါသည် ({w_idx+1}/{total_windows})\n"
                f"📊 {fmt_progress_bar(55 + int(20 * (w_idx + 1) / total_windows))}",
                force=True,
            )
            await build_cinema_recap_hd(
                orig_video=orig_video, orig_dur=total_dur,
                src_w=src_w, src_h=src_h, st1_dims=stage_dims,
                scenes=window_scenes, voice_clips=window_clips,
                work_dir=work_dir, recap_hd_path=part_path,
                copyright_shield=True, bgm_mode="orig",
                tracker=tracker, delete_source=False, job_id=job_id,
            )
            window_part_paths.append(part_path)
            del window_clips
            for i in idxs:
                all_clips[i] = None
            gc.collect()

        if not window_part_paths:
            raise RuntimeError("No recap parts built")

        concat_file = os.path.join(work_dir, "recap_all.txt")
        with open(concat_file, "w") as f:
            for p in window_part_paths:
                f.write(f"file '{p}'\n")
        proc = await asyncio.to_thread(
            run_cmd,
            [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
             "-c", "copy", "-movflags", "+faststart", recap_hd_path], 900,
        )
        for p in window_part_paths:
            try: os.remove(p)
            except: pass
        del all_clips
        gc.collect()

        if not os.path.isfile(recap_hd_path) or os.path.getsize(recap_hd_path) < 5000:
            raise RuntimeError(f"Join failed: {proc.stderr.decode()[-300:]}")
        return (await asyncio.to_thread(probe_video_info, recap_hd_path))[0]


# ==============================================================================
# Upload helpers
# ==============================================================================
def save_download_result(video_path: str) -> str:
    now = time.time()
    for key, item in list(DOWNLOAD_TOKENS.items()):
        if item.get("expires", 0) < now:
            DOWNLOAD_TOKENS.pop(key, None)
            try: os.remove(item.get("path", ""))
            except: pass
    token = secrets.token_urlsafe(20)
    dest = os.path.join(PUBLISH_CACHE_DIR, f"recap_{token}.mp4")
    shutil.move(video_path, dest)
    DOWNLOAD_TOKENS[token] = {"path": dest, "expires": now + 6 * 3600}
    return f"{get_public_base_url()}/download/{token}"


async def upload_final_video(chat_id: int, reply_msg_id: int, parts: list,
                              caption_text: str, width: int, height: int,
                              tracker: "ProgressTracker" = None) -> bool:
    for p_idx, part_path in enumerate(parts):
        mb_sz = os.path.getsize(part_path) / (1024 * 1024)
        if tracker:
            await tracker.update(
                f"📤 ပို့ပေးနေပါသည် ({p_idx+1}/{len(parts)} · {mb_sz:.1f} MB)...\n📊 {fmt_progress_bar(99)}",
                force=True,
            )
        cap = caption_text if len(parts) == 1 else f"{caption_text} (အပိုင်း {p_idx+1}/{len(parts)})"
        data_payload = {
            "chat_id": str(chat_id), "caption": cap,
            "reply_to_message_id": str(reply_msg_id),
            "supports_streaming": "true",
            "width": str(width), "height": str(height),
        }
        res = await tg_upload_media("sendVideo", data=data_payload,
                                     field_name="video",
                                     file_name=f"recap_{p_idx+1}.mp4",
                                     file_path=part_path, mime_type="video/mp4")
        if not res.get("ok"):
            return False
    return True


# ==============================================================================
# JOB HANDLER — the main processing function
# ==============================================================================
async def handle_video_job(
    chat_id: int, reply_msg_id: int, url: str, size_key: str,
    voice_id: str, copyright_shield: bool = True, bgm_mode: str = "orig",
    custom_title: str = "", existing_status_id: int = None, input_file_id: str = "",
):
    job_id = f"j{secrets.token_hex(5)}"
    url = url or ""
    size_key, voice_id, copyright_shield, bgm_mode = "orig", "charon", True, "orig"
    plat_lbl = "Telegram Video" if input_file_id else detect_platform_label(url)
    retry_args = {"chat_id": chat_id, "reply_msg_id": reply_msg_id, "url": url,
                  "size_key": size_key, "voice_id": voice_id, "copyright_shield": True,
                  "bgm_mode": "orig", "input_file_id": input_file_id}
    JOB_RETRY_ARGS[job_id] = retry_args
    while len(JOB_RETRY_ARGS) > 30:
        JOB_RETRY_ARGS.popitem(last=False)

    v_gender = "male"
    cancel_kb = {"inline_keyboard": [[
        {"text": "⏹ ရပ်မည်", "callback_data": f"cancel|{job_id}"},
        {"text": "↩️ မီနူး", "callback_data": "back|menu"},
    ]]}
    init_txt = f"⚡ Turbo Recap စတင်နေပါပြီ · {plat_lbl}\n📊 {fmt_progress_bar(5)}"

    if existing_status_id:
        status_id = existing_status_id
        await edit_text(chat_id, status_id, init_txt, reply_markup=cancel_kb)
    else:
        status_id = await send_text(chat_id, init_txt, reply_to=reply_msg_id, reply_markup=cancel_kb)

    tracker = ProgressTracker(chat_id, status_id, reply_markup=cancel_kb,
                              min_interval=3.0, job_id=job_id)
    now_job = int(time.time())
    JOB_STATES[job_id] = {
        "state": "running", "job_id": job_id, "chat_id": chat_id,
        "status_message_id": status_id or 0, "phase": "Video ရယူရန် စတင်",
        "detail": init_txt[:500], "progress": 5, "chunk": 0, "total_chunks": 0,
        "started_at": now_job, "last_progress_at": now_job,
        "last_watchdog_alert": 0, "updated_at": now_job,
    }
    JOB_MONITOR_STATE.update(JOB_STATES[job_id])
    await sync_voice_state_to_telegram()

    try:
        async with cancellable_semaphore(job_semaphore, job_id) as acq:
            if not acq or job_id in CANCELLED_JOBS:
                CANCELLED_JOBS.discard(job_id)
                update_job_state(job_id, state="stopped", phase="အလုပ်မစမီ ရပ်ပြီး")
                await edit_text(chat_id, status_id, "⏹ မစမီ ရပ်လိုက်ပါပြီ။")
                return
            update_job_state(job_id, phase="Video ရယူနေပါသည်", progress=8)
            work_dir = tempfile.mkdtemp(prefix="vid_4k_", dir=WORK_ROOT)
            try:
                # ==== DOWNLOAD ====
                await tracker.update(f"📥 Video ရယူနေပါသည် · {plat_lbl}\n📊 {fmt_progress_bar(8)}", force=True)
                dl_dir = os.path.join(work_dir, "download")
                os.makedirs(dl_dir, exist_ok=True)
                orig_video = os.path.join(dl_dir, "source.mp4")
                dl_tier = ""

                if input_file_id:
                    ok = await download_telegram_file(input_file_id, orig_video)
                    dl_tier = "Telegram"
                else:
                    try:
                        segs, dl_tier = await download_video_in_segments(
                            url, dl_dir, tracker=tracker, job_id=job_id,
                        )
                        if len(segs) == 1:
                            os.rename(segs[0], orig_video)
                        else:
                            cf = os.path.join(dl_dir, "cat.txt")
                            with open(cf, "w") as f:
                                for s in segs:
                                    f.write(f"file '{s}'\n")
                            run_cmd([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
                                     "-i", cf, "-c", "copy", orig_video], 600)
                            for s in segs:
                                try: os.remove(s)
                                except: pass
                        ok = os.path.exists(orig_video) and os.path.getsize(orig_video) > 10000
                    except JobCancelled:
                        raise
                    except Exception as e:
                        logger.exception("Download failed")
                        ok = False
                        dl_tier = str(e)[:120]

                if not ok:
                    update_job_state(job_id, state="failed", phase="ရယူမရ")
                    await edit_text(chat_id, status_id,
                                    "❌ Video link မှ ဖိုင်ရယူမရပါ။ Link စစ်ပါ။",
                                    reply_markup={"inline_keyboard": [[
                                        {"text": "🔁 ပြန်", "callback_data": f"retry|{job_id}"},
                                        {"text": "↩️ မီနူး", "callback_data": "back|menu"},
                                    ]]})
                    return

                await tracker.update(f"✅ ရယူပြီး · {dl_tier}\n📊 {fmt_progress_bar(20)}", force=True)
                dur, src_w, src_h, _ = await asyncio.to_thread(probe_video_info, orig_video)
                if dur <= 0.5:
                    await edit_text(chat_id, status_id, "❌ Video မတွေ့ပါ။")
                    return
                if dur > MAX_VIDEO_SECONDS:
                    await edit_text(chat_id, status_id, "⚠️ ၆ နာရီထက် ရှည်ပါတယ်။")
                    return

                st1_dims, _ = compute_stage_dims(src_w, src_h, size_key)
                recap_hd = os.path.join(work_dir, "recap_hd.mp4")
                await build_long_video_recap_hd(
                    orig_video=orig_video, total_dur=dur,
                    src_w=src_w, src_h=src_h, stage_dims=st1_dims,
                    work_dir=work_dir, recap_hd_path=recap_hd,
                    tracker=tracker, job_id=job_id, gender=v_gender,
                )

                await tracker.update(f"📤 ပို့ရန် ပြင်ဆင်နေပါသည်...\n📊 {fmt_progress_bar(98)}", force=True)
                if job_id in CANCELLED_JOBS:
                    raise JobCancelled("cancelled")

                _, out_w, out_h, _ = await asyncio.to_thread(probe_video_info, recap_hd)
                caption = "📹 HD Recap · မူရင်းအချိုးအစား · Charon AI"

                if os.path.getsize(recap_hd) <= MAX_TG_BYTES:
                    sent = await upload_final_video(
                        chat_id=chat_id, reply_msg_id=reply_msg_id,
                        parts=[recap_hd], caption_text=caption,
                        width=out_w or st1_dims[0], height=out_h or st1_dims[1],
                        tracker=tracker,
                    )
                else:
                    url_dl = await asyncio.to_thread(save_download_result, recap_hd)
                    sent = bool(await send_text(chat_id, f"📹 HD Recap: {url_dl}",
                                                reply_to=reply_msg_id))

                if sent:
                    STATS["completed_hd"] += 1
                    update_job_state(job_id, state="completed", phase="ပြီးပါပြီ", progress=100)
                    await sync_voice_state_to_telegram()
                    try: await tg_api("deleteMessage",
                                     {"chat_id": chat_id, "message_id": status_id})
                    except: pass
                else:
                    update_job_state(job_id, state="failed", phase="ပို့မရ")
                    await edit_text(chat_id, status_id, "⚠️ ပို့မရပါ။",
                                    reply_markup={"inline_keyboard": [[
                                        {"text": "🔁 ပြန်", "callback_data": f"retry|{job_id}"},
                                    ]]}),
            except VoiceEngineUnavailable as e:
                update_job_state(job_id, state="failed", phase="Charon မရ")
                await edit_text(chat_id, status_id, f"⚠️ {str(e)[:200]}",
                                reply_markup={"inline_keyboard": [[
                                    {"text": "🔁 ပြန်", "callback_data": f"retry|{job_id}"},
                                ]]})
            except JobCancelled:
                raise
            except Exception as e:
                if job_id in CANCELLED_JOBS or isinstance(e, JobCancelled):
                    CANCELLED_JOBS.discard(job_id)
                    update_job_state(job_id, state="stopped", phase="ရပ်ပြီး")
                    await edit_text(chat_id, status_id, "⏹ ရပ်လိုက်ပါပြီ။")
                else:
                    logger.exception("Job error")
                    update_job_state(job_id, state="failed", phase=f"Error: {type(e).__name__}")
                    await edit_text(chat_id, status_id,
                                    f"❌ အမှား: {type(e).__name__}. Retry နှိပ်ပါ။",
                                    reply_markup={"inline_keyboard": [[
                                        {"text": "🔁 ပြန်", "callback_data": f"retry|{job_id}"},
                                        {"text": "↩️ မီနူး", "callback_data": "back|menu"},
                                    ]]})
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
                gc.collect()
    finally:
        if JOB_STATES.get(job_id, {}).get("state") in ("running", "queued"):
            update_job_state(job_id, state="ended", phase="ပြီးဆုံး")


async def enqueue_video_job(**job_args) -> bool:
    global RESERVED_JOB_SLOTS
    chat_id = int(job_args["chat_id"])
    reply_msg_id = job_args.get("reply_msg_id")
    busy = False
    async with JOB_ADMISSION_LOCK:
        if ACTIVE_JOBS + RESERVED_JOB_SLOTS >= MAX_CONCURRENT_PROJECTS:
            busy = True
        else:
            RESERVED_JOB_SLOTS += 1
            asyncio.create_task(_run_admitted_video_job(dict(job_args)))
    if busy:
        await send_text(chat_id,
                        "⏳ လက်ရှိ Project ၂ ခု လုပ်နေပါပြီ။ တစ်ခုပြီးမှ ထပ်ပို့ပါ။",
                        reply_to=reply_msg_id)
        return False
    return True


async def _run_admitted_video_job(job_args: dict):
    global ACTIVE_JOBS, RESERVED_JOB_SLOTS
    async with JOB_ADMISSION_LOCK:
        RESERVED_JOB_SLOTS = max(0, RESERVED_JOB_SLOTS - 1)
        ACTIVE_JOBS += 1
    try:
        await handle_video_job(**job_args)
    except Exception:
        logger.exception("Admitted job failed")
    finally:
        async with JOB_ADMISSION_LOCK:
            ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)


# ==============================================================================
# Update processor
# ==============================================================================
async def process_admin_update(update: dict):
    global AWAITING_VOICE_SAMPLE, AWAITING_BGM_UPLOAD, AWAITING_TITLE_INFO
    try:
        cb = update.get("callback_query")
        if cb:
            cb_id = cb.get("id")
            sender_id = (cb.get("from") or {}).get("id", 0)
            data = cb.get("data") or ""
            msg = cb.get("message") or {}
            chat_id = msg.get("chat", {}).get("id")
            msg_id = msg.get("message_id")

            if sender_id not in ({ADMIN_ID} | ALLOWED_USER_IDS):
                await tg_api("answerCallbackQuery",
                             {"callback_query_id": cb_id, "text": "🔒 ခွင့်မပြုပါ။"})
                return

            if data.startswith("cancel|"):
                jid = data.split("|", 1)[1]
                CANCELLED_JOBS.add(jid)
                if jid in JOB_STATES:
                    update_job_state(jid, phase="ရပ်ရန် တောင်းဆိုချက် လက်ခံ", updated_at=int(time.time()))
                await tg_api("answerCallbackQuery",
                             {"callback_query_id": cb_id, "text": "⏹ ရပ်ရန် တောင်းဆိုချက် လက်ခံပါပြီ"})
                await edit_text(chat_id, msg_id, "⏹ ရပ်ရန် တောင်းဆိုချက် လက်ခံပါပြီ။",
                                reply_markup={"inline_keyboard": [[
                                    {"text": "↩️ မီနူး", "callback_data": "back|menu"},
                                ]]})
                return

            if data.startswith("retry|"):
                jid = data.split("|", 1)[1]
                args = JOB_RETRY_ARGS.get(jid)
                if not args:
                    await tg_api("answerCallbackQuery",
                                 {"callback_query_id": cb_id,
                                  "text": "Retry သက်တမ်းကုန်ပါပြီ။ Link ပြန်ပို့ပါ။",
                                  "show_alert": True})
                    return
                await tg_api("answerCallbackQuery",
                             {"callback_query_id": cb_id, "text": "🔁 ပြန်စ"})
                await enqueue_video_job(**args, existing_status_id=msg_id)
                return

            if data == "back|menu":
                await tg_api("answerCallbackQuery", {"callback_query_id": cb_id, "text": "မီနူး"})
                await edit_text(chat_id, msg_id, studio_menu_text(),
                                reply_markup=studio_menu_keyboard())
                return

            await tg_api("answerCallbackQuery", {"callback_query_id": cb_id})
            return

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        chat_id = msg.get("chat", {}).get("id")
        sender_id = (msg.get("from") or {}).get("id", 0)
        msg_id = msg.get("message_id")
        if not chat_id:
            return

        if sender_id != ADMIN_ID and sender_id not in ALLOWED_USER_IDS:
            await send_text(chat_id, "🔒 ခွင့်မပြုပါ။", reply_to=msg_id)
            return

        media_obj = msg.get("video") or msg.get("video_note")
        text = (msg.get("text") or msg.get("caption") or "").strip()
        if not media_obj and msg.get("document"):
            doc = msg["document"]
            mime_d = (doc.get("mime_type") or "").lower()
            fn_d = (doc.get("file_name") or "").lower()
            if mime_d.startswith("video/") or fn_d.endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v")):
                media_obj = doc

        if media_obj and not re.search(r"https?://", text):
            fid = media_obj.get("file_id")
            if int(media_obj.get("file_size") or 0) > MAX_BOT_DOWNLOAD_BYTES:
                await send_text(chat_id,
                                "⚠️ ဖိုင်ကြီးလွန်းပါတယ် (20MB အောက်)။ Link ပို့ပါ။",
                                reply_to=msg_id)
                return
            if fid:
                await enqueue_video_job(
                    chat_id=chat_id, reply_msg_id=msg_id, url="", input_file_id=fid,
                    size_key="orig", voice_id="charon", copyright_shield=True, bgm_mode="orig",
                )
                return

        if not text:
            return

        command = text.split()[0].split("@", 1)[0].lower()

        if command == "/status":
            await send_text(chat_id, format_job_status(chat_id=chat_id), reply_to=msg_id)
            return

        if any(text.startswith(c) for c in ("/start", "/menu")):
            await send_text(chat_id, studio_menu_text(), reply_to=msg_id,
                            reply_markup=studio_menu_keyboard())
            return

        url_match = re.search(r"https?://[^\s，。！？,<>\"'；：）】]+", text)
        if not url_match:
            await send_text(chat_id, studio_menu_text(), reply_to=msg_id,
                            reply_markup=studio_menu_keyboard())
            return

        raw_url = url_match.group(0).rstrip(".,!?;:)")
        await enqueue_video_job(
            chat_id=chat_id, reply_msg_id=msg_id, url=raw_url,
            size_key="orig", voice_id="charon", copyright_shield=True, bgm_mode="orig",
        )
    except Exception:
        logger.exception("process_admin_update error")


# ==============================================================================
# FastAPI app
# ==============================================================================
async def active_job_keepalive_loop():
    while True:
        await asyncio.sleep(180)
        if ACTIVE_JOBS > 0 and RENDER_EXTERNAL_URL:
            try:
                ping_url = RENDER_EXTERNAL_URL.rstrip("/") + "/healthz"
                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.get(ping_url)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    ka_task = asyncio.create_task(active_job_keepalive_loop())
    if BOT_TOKEN and RENDER_EXTERNAL_URL:
        try:
            await tg_api("setWebhook", {"url": f"{RENDER_EXTERNAL_URL.rstrip('/')}/webhook",
                                        "drop_pending_updates": False,
                                        "allowed_updates": ["message", "edited_message", "callback_query"]})
            await tg_api("setMyCommands", {"commands": [
                {"command": "start", "description": "မီနူးဖွင့်ရန်"},
                {"command": "status", "description": "အခြေအနေကြည့်ရန်"},
            ]})
        except Exception as e:
            logger.error(f"Startup init: {e}")
    yield
    ka_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "ok": True, "version": "v16.0-segment-safe",
        "active_jobs": ACTIVE_JOBS, "max_projects": MAX_CONCURRENT_PROJECTS,
        "free_disk_mb": _get_free_disk_mb(WORK_ROOT),
        "segment_max_duration": SEGMENT_MAX_DURATION,
        "segment_target_mb": SEGMENT_TARGET_MB,
        "max_source_bytes": MAX_SOURCE_BYTES,
    })


@app.get("/download/{token}")
async def download_result(token: str):
    item = DOWNLOAD_TOKENS.get(token)
    if not item or item.get("expires", 0) < time.time() or not os.path.isfile(item.get("path", "")):
        DOWNLOAD_TOKENS.pop(token, None)
        return HTMLResponse("သက်တမ်းကုန်ပါပြီ။", status_code=404)
    return FileResponse(item["path"], media_type="video/mp4", filename="recap_hd.mp4")


@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def index(request: Request):
    return HTMLResponse(
        '<!doctype html><html><body style="font:18px system-ui;background:#101827;color:#fff;'
        'display:grid;place-items:center;min-height:90vh">'
        '<main><h1>🎬 Recap Turbo v16</h1><p>3GB+ video safe · Segment-based processing</p>'
        '<p>Video Link ပို့ပါ။</p></main></body></html>'
    )


@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
    except Exception:
        return JSONResponse({"ok": True})
    update_id = update.get("update_id")
    if update_id in processed_updates:
        return JSONResponse({"ok": True})
    if update_id is not None:
        processed_updates.append(update_id)
    asyncio.create_task(process_admin_update(update))
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

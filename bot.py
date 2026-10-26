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
import unicodedata
from contextlib import asynccontextmanager

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
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://rednote-bot-gzd0.onrender.com").strip()

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

SAMPLE_RATE = 24000
MAX_TG_BYTES = 45 * 1024 * 1024
CHUNK_DURATION = 300
PAUSE_AFTER_SCENE = 0.20

# --- Script coverage (input -> script output) ---
# Script (narration) must cover 90% of the input chunk duration.
# e.g. a 300s chunk must produce narration that, when spoken, lasts
# ~270s (0.90 * 300). Burmese TTS (Charon) speaks at roughly
# 6.5 characters/second, so the character target is derived from that.
SCRIPT_TARGET_RATIO = 0.90      # output script = 90% of input duration
SCRIPT_CHARS_PER_SEC = 6.5      # estimated Burmese TTS speaking rate
SCRIPT_MAX_RATIO = 1.05         # never exceed 105% of the 90% target
SCRIPT_TOPUP_ATTEMPTS = 3       # retry top-ups until 90% is reached

PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
os.makedirs(PUBLISH_CACHE_DIR, exist_ok=True)

CANCELLED_JOBS: set = set()
video_render_semaphore = asyncio.Semaphore(1)

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
    {"key": "coverage", "icon": "🎯", "label": "Script ပမာဏ", "kind": "cycle",
     "desc": "Input ကြာချိန်ရဲ့ ဘယ်နှုန်း% script ထွက်မလဲ",
     "options": [(50, "50%"), (70, "70%"), (80, "80%"), (90, "90% ⭐"), (95, "95%"), (100, "100%")]},
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
    return s


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
        [btn(f"🎯 Script: {s['coverage']}%", "feat:coverage"), btn(f"🎙️ အသံ: {s['voice']}", "feat:voice")],
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
    "2️⃣ Bot က download → script (input ၏ ဆက်တင်%) → အသံ → ဗီဒီယို လုပ်ပေးပါမယ်\n"
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


async def send_main_menu(chat_id: int, reply_to: int = None):
    return await send_text(chat_id, MAIN_MENU_TEXT, reply_to=reply_to,
                           reply_markup=main_menu_keyboard(chat_id))


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


def _detect_pronoun(text: str) -> str:
    """Script စာသားထဲမှ ပြောနေသူ အသုံးအများဆုံး ပထမလူ နာမ်စား (ကျွန်တော်/ကျွန်မ)
    ကို ရှာသည် — chunk အကူးတွင် နာမ်စား မလွဲအောင် ထိန်းရန်။ မတွေ့ရင် "" ပြန်သည်။"""
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFC", s or "")
        return s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")

    t = _norm(text)
    male = t.count(_norm("ကျွန်တော်"))
    female = t.count(_norm("ကျွန်မ"))
    if male > female:
        return "ကျွန်တော်"
    if female > male:
        return "ကျွန်မ"
    return ""


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


def script_target_chars(chunk_dur: float, settings: dict = None) -> int:
    """Characters needed so spoken narration = coverage% of input (default 90%)."""
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    ratio = float(s.get("coverage") or 90) / 100.0
    rate = float(s.get("rate") or SCRIPT_CHARS_PER_SEC)
    return int(chunk_dur * ratio * rate)


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


def trim_scenes_to_target(scenes: list, target_chars: int) -> list:
    """Trim scenes (from the end) so total <= SCRIPT_MAX_RATIO * target,
    without ever dropping below the 90% target itself."""
    limit = int(target_chars * SCRIPT_MAX_RATIO)
    if script_total_chars(scenes) <= limit:
        return scenes
    kept, total = [], 0
    for scene in scenes:
        text = (scene.get("burmese") or "").strip()
        if total + len(text) <= limit:
            kept.append(scene)
            total += len(text)
            continue
        # Partially keep this scene: cut at the last sentence end (။)
        # that still keeps us >= target_chars and <= limit.
        room = limit - total
        if room >= 40 and total < target_chars:
            cut = text.rfind("။", 0, room)
            piece = text[:cut + 1].strip() if cut >= 40 else text[:room].strip()
            if piece:
                scene = dict(scene, burmese=piece)
                kept.append(scene)
        break
    return kept if script_total_chars(kept) >= target_chars else scenes[:len(kept)] or scenes


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
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    coverage_ratio = float(st.get("coverage") or 90) / 100.0
    chars_rate = float(st.get("rate") or SCRIPT_CHARS_PER_SEC)
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
- ပြောနေသူသည် အဓိကဇာတ်ကောင် ကိုယ်တိုင်သာ ဖြစ်သည်။ ကိုယ့်ကိုယ်ကို ညွှန်းတိုင်း «ကျွန်တော်» (အမျိုးသားဆိုလျှင်) သို့မဟုတ် «ကျွန်မ» (အမျိုးသမီးဆိုလျှင်) ဟုသာ သုံးပါ။ ဥပမာ — «အဲဒီအချိန်မှာ ကျွန်တော် ...»၊ «ကျွန်မတော့ ... လို့ ပြန်ဖြေလိုက်တယ်»။ (ဝါကျများ ဆက်နေသရွေ့ နာမ်စားချန်ပြီး ရေးနိုင်သော်လည်း ကိုယ့်အကြောင်း အသစ်စတိုင်း «ကျွန်တော်/ကျွန်မ» ဖြင့်ပင် စပါ။)
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
- မူရင်းအပိုင်း၏ {st['coverage']}% ခန့်ကို ထိန်းသိမ်းပါ။ အရေးမကြီးသော တိတ်ဆိတ်နေမှု၊ ထပ်နေသောမြင်ကွင်းနှင့် မလိုအပ်သော နားချိန်များကိုသာ ဖြတ်ပါ။
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

    if total_chars < target_chars:
        logger.warning(
            f"Chunk {chunk_index}: script only {total_chars}/{target_chars} chars "
            f"after {SCRIPT_TOPUP_ATTEMPTS} top-ups — continuing with best effort"
        )

    # Cap over-long scripts at 105% of the 90% target so narration
    # does not overrun the video.
    scenes = trim_scenes_to_target(scenes, target_chars)
    total_chars = script_total_chars(scenes)
    spoken_est = total_chars / chars_rate
    coverage_pct = (spoken_est / chunk_dur * 100) if chunk_dur else 0
    logger.info(
        f"Chunk {chunk_index} final script: {len(scenes)} scenes, {total_chars} chars, "
        f"~{spoken_est:.0f}s spoken = {coverage_pct:.0f}% of {chunk_dur:.0f}s input (target {st['coverage']}%)"
    )
    await tracker.update(
        f"📝 Script ပြီးပါပြီ — input ၏ {coverage_pct:.0f}% (ပစ်မှတ် {st['coverage']}%)၊ {total_chars} လုံး",
        force=True,
    )

    try:
        await asyncio.to_thread(client.files.delete, name=video_file.name)
    except Exception as e:
        logger.warning(f"Gemini delete error: {e}")

    if not scenes:
        scenes = [{"start": 0, "end": 5, "burmese": "စာသားရယူရာတွင် အမှားအယွင်းရှိပါသည်။"}]

    keep_ranges = make_keep_ranges(chunk_dur, cuts, max_remove_ratio=float(st.get('cut') or 0) / 100.0)
    kept_duration = sum(end - start for start, end in keep_ranges)

    if kept_duration < chunk_dur * coverage_ratio:
        logger.warning(f"Kept {kept_duration:.1f}s < {st['coverage']}%, forcing keep-all")
        keep_ranges = [(0.0, chunk_dur)]
        kept_duration = chunk_dur

    await tracker.update(f"🎙️ {st.get('voice', 'Charon')} AI မှ အသံသွင်းနေပါသည်...", force=True)
    clips = await synthesize_tts(scenes, os.path.dirname(chunk_path), st)

    # Verify the actual spoken audio against the 90% target.
    audio_seconds = sum(len(c) for c in clips) / SAMPLE_RATE + float(st.get('pause') or 0) * max(0, len(clips) - 1)
    audio_pct = (audio_seconds / kept_duration * 100) if kept_duration else 0
    logger.info(
        f"Chunk {chunk_index} TTS audio: {audio_seconds:.0f}s = {audio_pct:.0f}% "
        f"of kept video {kept_duration:.0f}s (target ~{st['coverage']}% of input)"
    )
    if audio_seconds < kept_duration * coverage_ratio:
        logger.warning(f"Chunk {chunk_index}: TTS audio below {st['coverage']}% of kept video ({audio_pct:.0f}%)")

    out_w, out_h = compute_output_dims(*probe_video_info(chunk_path)[1:], int(st.get('resolution') or 1280))
    shield_v = "crop=iw*0.96:ih*0.96,hflip," if st.get("shield") else ""
    color = "eq=contrast=1.08:saturation=1.12:gamma=1.02," if st.get("color") else ""

    master_pcm_path = os.path.join(os.path.dirname(chunk_path), "master_audio.pcm")
    with open(master_pcm_path, "wb") as pcm_f:
        for clip in clips:
            np.clip(clip * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)
            pause_samples = int(float(st.get('pause') or 0) * SAMPLE_RATE)
            if pause_samples > 0:
                np.zeros(pause_samples, dtype=np.int16).tofile(pcm_f)

    master_wav = os.path.join(os.path.dirname(chunk_path), "master_audio.wav")
    run_cmd([
        FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", master_pcm_path, master_wav,
    ], 120)
    if os.path.exists(master_pcm_path):
        os.remove(master_pcm_path)

    seg_files = []
    await tracker.update("🎬 ဗီဒီယို ပြင်ဆင်နေပါသည်...", force=True)

    for index, (start, end) in enumerate(keep_ranges):
        segment_duration = end - start
        segment_path = os.path.join(os.path.dirname(chunk_path), f"seg_{index:03d}.mp4")
        fc = (
            f"{shield_v}"
            f"fps=25,"
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,"
            f"crop={out_w}:{out_h},"
            f"{color}format=yuv420p"
        )
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", chunk_path,
            "-t", f"{segment_duration:.3f}",
            "-an",
            "-vf", fc,
            "-c:v", "libx264",
            "-preset", "superfast" if st.get("fast_mode") else "ultrafast",
            "-crf", str(int(st.get("quality") or 23)),
            segment_path,
        ], 300)

        if result.returncode != 0 or not os.path.exists(segment_path):
            raise RuntimeError(f"segment {index + 1} failed")
        seg_files.append(segment_path)

    concat_txt = os.path.join(os.path.dirname(chunk_path), "concat.txt")
    with open(concat_txt, "w") as f:
        for tp in seg_files:
            f.write(f"file '{tp}'\n")

    video_only = os.path.join(os.path.dirname(chunk_path), "video_only.mp4")
    await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", video_only,
    ], 300)

    out_chunk = chunk_path.replace(".mp4", "_recap.mp4")
    await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y",
        "-i", video_only,
        "-i", master_wav,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-af", "apad",
        "-t", f"{kept_duration:.3f}",
        "-movflags", "+faststart",
        out_chunk,
    ], 300)

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


async def split_for_telegram(path: str, work_dir: str) -> list[str]:
    if not os.path.exists(path):
        return []
    if os.path.getsize(path) <= MAX_TG_BYTES:
        return [path]

    out_dir = os.path.join(work_dir, "tg_parts")
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "part_%03d.mp4")
    run_cmd([
        FFMPEG_BIN, "-y",
        "-i", path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", "180",
        "-reset_timestamps", "1",
        pattern,
    ], 300)
    parts = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith("part_") and f.endswith(".mp4")
    )
    return parts or [path]


async def handle_link_job(chat_id: int, reply_msg_id: int, url: str):
    job_settings = get_user_settings(chat_id)
    chunk_duration = int(job_settings.get("chunk") or CHUNK_DURATION)
    job_id = f"j{secrets.token_hex(5)}"
    if chat_id in ACTIVE_JOBS:
        await send_text(chat_id, "⚠️ အလုပ်တစ်ခု လုပ်နေဆဲပါ။ ပြီးအောင်စောင့်ပါ သို့မဟုတ် 🛑 ရပ်မယ် ခလုတ်နှိပ်ပါ။",
                        reply_to=reply_msg_id, reply_markup=main_menu_keyboard(chat_id))
        return
    ACTIVE_JOBS[chat_id] = job_id
    status_id = await send_text(
        chat_id,
        f"⚡ Recap စတင်နေပါပြီ...\n🎯 Script: {job_settings['coverage']}% · 🎙️ {job_settings['voice']} · 📐 {job_settings['resolution']}p",
        reply_to=reply_msg_id,
        reply_markup={"inline_keyboard": [[btn("🛑 ဒီအလုပ်ကို ရပ်မယ်", "menu:cancel")]]},
    )
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
            await send_text(chat_id, f"✅ <b>Recap ပြီးပါပြီ!</b> ({fmt_time(final_dur)})\n🎯 Script {job_settings['coverage']}% · 🎙️ {job_settings['voice']}",
                            reply_markup=main_menu_keyboard(chat_id))

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
                f"🎯 Script {s['coverage']}% · 🎙️ {s['voice']}\n\nရပ်ချင်ရင် 🛑 ခလုတ်နှိပ်ပါ။")
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

    async def show(text, markup):
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
        job = ACTIVE_JOBS.get(chat_id)
        if job:
            CANCELLED_JOBS.add(job)
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
            asyncio.create_task(handle_link_job(chat_id, message_id, url))
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

    cmd = text.split()[0].split("@")[0].lower() if text else ""
    if cmd in ("/start", "/menu"):
        await send_main_menu(chat_id, reply_to=msg_id)
    elif cmd == "/settings":
        await send_text(chat_id, "⚙️ <b>ဆက်တင်များ (Feature ၂၀)</b>\nပြောင်းချင်တဲ့ ခလုတ်ကို နှိပ်ပါ။", reply_to=msg_id, reply_markup=settings_keyboard(chat_id))
    elif cmd == "/status":
        await send_text(chat_id, job_status_text(chat_id), reply_to=msg_id, reply_markup=main_menu_keyboard(chat_id))
    elif cmd == "/cancel":
        job = ACTIVE_JOBS.get(chat_id)
        if job:
            CANCELLED_JOBS.add(job)
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
        await send_main_menu(chat_id, reply_to=msg_id)


app = FastAPI()

# ဒီ code အစစ် run နေကြောင်း စစ်နိုင်ဖို့ version အမှတ်အသား
BUILD_VERSION = "v3.0-new-code-button-menu"
BUILD_FEATURES = 20


@app.get("/")
async def root():
    return JSONResponse({
        "app": "rednote-bot",
        "version": BUILD_VERSION,
        "features": BUILD_FEATURES,
        "status": "online",
    })


@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "app": "rednote-bot", "version": BUILD_VERSION})


BOT_COMMANDS = [
    {"command": "start", "description": "🎬 ပင်မ မီနူး (ခလုတ်များ)"},
    {"command": "settings", "description": "⚙️ ဆက်တင်များ (Feature 20)"},
    {"command": "status", "description": "📊 အလုပ်အခြေအနေ"},
    {"command": "cancel", "description": "🛑 လုပ်နေတာ ရပ်ရန်"},
    {"command": "summary", "description": "📋 ဆက်တင် အကျဉ်း"},
    {"command": "help", "description": "❓ အသုံးပြုနည်း"},
]


async def _setup_bot():
    """စတင်တာနဲ့ (၁) Telegram command မီနူး မှတ်ပုံတင်ပြီး (၂) webhook ကို
    ဒီ service ဆီ ပြန်ချိတ်တယ် — အဟောင်း လွဲနေရင် ကိုယ်တိုင် ပြင်သွားအောင်။"""
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN မရှိပါ")
        return
    await asyncio.sleep(2)
    try:
        res = await tg_api("setMyCommands", {"commands": BOT_COMMANDS})
        logger.info(f"setMyCommands: {res.get('ok')}")
    except Exception as e:
        logger.warning(f"setMyCommands failed: {e}")
    if not RENDER_EXTERNAL_URL:
        return
    url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/webhook"
    try:
        res = await tg_api("setWebhook", {"url": url, "drop_pending_updates": False})
        logger.info(f"setWebhook -> {url}: {res.get('ok')} / version {BUILD_VERSION}")
        info = await tg_api("getWebhookInfo", {})
        logger.info(f"getWebhookInfo: {info.get('result', {})}")
    except Exception as e:
        logger.warning(f"setWebhook failed: {e}")


@app.on_event("startup")
async def _on_startup():
    logger.info(f"🚀 rednote-bot {BUILD_VERSION} စတင်ပါပြီ (features={BUILD_FEATURES})")
    asyncio.create_task(_setup_bot())


@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    asyncio.create_task(process_admin_update(update))
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

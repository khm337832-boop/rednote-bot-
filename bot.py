import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

import edge_tts
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else None

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip()

CHARON_VOICE = "Charon"
FALLBACK_THIHA_VOICE = "my-MM-ThihaNeural"
SAMPLE_RATE = 24000
MAX_TG_BYTES = 47 * 1024 * 1024

processed_updates = deque(maxlen=500)
job_semaphore = asyncio.Semaphore(1)
STATS = {"completed": 0, "completed_4k": 0, "started_at": int(time.time())}

# Short key -> Rednote URL cache for inline buttons (Telegram callback_data max 64 bytes)
URL_CACHE: OrderedDict[str, str] = OrderedDict()
# Per-chat default mode: "ask" (show menu), "std" (original fast recap), "4k" (4K Ultra-HD recap)
USER_DEFAULT_MODE: dict[int, str] = {}

PREFERRED_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]

PREFERRED_TTS_MODELS = [
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]


def remember_url(url: str) -> str:
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    URL_CACHE[key] = url
    URL_CACHE.move_to_end(key)
    while len(URL_CACHE) > 400:
        URL_CACHE.popitem(last=False)
    return key


def mode_selection_keyboard(url_key: str) -> dict:
    return {
        "inline_keyboard": [
            [{"text": "⚡ မူလ Smart Recap (အမြန် · လက်ရှိအတိုင်း)", "callback_data": f"rn|std|{url_key}"}],
            [{"text": "💎 4K Ultra-HD Recap (အကြည်ဆုံး 4K Final)", "callback_data": f"rn|4k|{url_key}"}],
            [{"text": "🎞️ Lossless Master Dub (မဖြတ်ဘဲ မူရင်းအကြည်)", "callback_data": f"rn|lossless|{url_key}"}],
            [{"text": "📥 မူရင်း Video အကြည်ဆုံး ဒေါင်းမယ် (No Watermark)", "callback_data": f"rn|raw|{url_key}"}],
        ]
    }


def followup_keyboard(url_key: str, done_mode: str) -> dict:
    rows = []
    if done_mode != "4k":
        rows.append([{"text": "💎 ဒီ Video ကို 4K Ultra-HD ထပ်ထုတ်မယ်", "callback_data": f"rn|4k|{url_key}"}])
    if done_mode != "std":
        rows.append([{"text": "⚡ မူလ Smart Recap (အမြန်) ထုတ်မယ်", "callback_data": f"rn|std|{url_key}"}])
    if done_mode != "raw":
        rows.append([{"text": "📥 မူရင်း Raw အကြည်ဆုံး ယူမယ်", "callback_data": f"rn|raw|{url_key}"}])
    return {"inline_keyboard": rows}


def default_mode_keyboard(current: str) -> dict:
    def mark(m: str, label: str) -> str:
        return f"✅ {label}" if current == m else label

    return {
        "inline_keyboard": [
            [{"text": mark("ask", "🎛️ Link ပို့တိုင်း Mode ရွေးမယ် (Default)"), "callback_data": "setmode|ask"}],
            [{"text": mark("4k", "💎 အမြဲ 4K Ultra-HD တန်းထုတ်မယ်"), "callback_data": "setmode|4k"}],
            [{"text": mark("std", "⚡ အမြဲ မူလ Smart Recap တန်းထုတ်မယ်"), "callback_data": "setmode|std"}],
        ]
    }


async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 180.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        if files:
            resp = await client.post(url, data=data or {}, files=files)
        else:
            resp = await client.post(url, json=data or {})
        return resp.json()


async def send_text(chat_id: int, text: str, reply_to: int = None, reply_markup: dict = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    if reply_markup:
        payload["reply_markup"] = reply_markup
    res = await tg_api("sendMessage", payload)
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str, reply_markup: dict = None):
    if not message_id:
        return
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        await tg_api("editMessageText", payload)
    except Exception:
        pass


async def register_webhook(base_url: str) -> dict:
    if not BOT_TOKEN or not base_url:
        return {"ok": False, "reason": "Missing BOT_TOKEN or base_url"}
    clean_base = base_url.rstrip("/")
    if clean_base.startswith("http://") and "localhost" not in clean_base and "127.0.0.1" not in clean_base:
        clean_base = "https://" + clean_base[len("http://") :]
    webhook_url = f"{clean_base}/webhook"
    res = await tg_api(
        "setWebhook",
        {
            "url": webhook_url,
            "drop_pending_updates": False,
            "allowed_updates": ["message", "edited_message", "callback_query"],
        },
    )
    logger.info(f"Webhook set to {webhook_url}: {res}")
    return res


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RENDER_EXTERNAL_URL and BOT_TOKEN:
        try:
            await register_webhook(RENDER_EXTERNAL_URL)
        except Exception as e:
            logger.error(f"Startup webhook error: {e}")
    yield


app = FastAPI(lifespan=lifespan)


def render_workflow_dashboard_html(wh_ok: bool, groq_ready: bool, gemini_ready: bool) -> str:
    wh_badge = "ACTIVE · WEBHOOK READY" if wh_ok else "CONNECTING..."
    groq_badge = "GROQ WHISPER-V3 · READY" if groq_ready else "GEMINI AUDIO MODE"
    gem_badge = "CHARON AI VOICE · READY" if gemini_ready else "KEY REQUIRED"
    return f"""<!DOCTYPE html>
<html lang="my">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Smart Recap &amp; 4K Ultra-HD Studio (v4) · Rednote ➔ Myanmar</title>
<style>
  :root {{
    --bg: #070913;
    --panel: rgba(16, 22, 42, 0.84);
    --border: rgba(99, 179, 237, 0.25);
    --cyan: #00f2fe;
    --blue: #4facfe;
    --purple: #a855f7;
    --green: #10b981;
    --gold: #fbbf24;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: radial-gradient(circle at 15% 15%, #131c38 0%, #070913 55%, #04050a 100%);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
    padding: 24px 16px 56px;
  }}
  .grid-bg {{
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
      linear-gradient(rgba(79,172,254,0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(79,172,254,0.06) 1px, transparent 1px);
    background-size: 34px 34px;
  }}
  .wrap {{ position: relative; z-index: 1; max-width: 980px; margin: 0 auto; }}
  .topbar {{
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; padding: 16px 20px; border-radius: 18px;
    background: var(--panel); border: 1px solid var(--border);
    backdrop-filter: blur(14px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
    margin-bottom: 24px;
  }}
  .brand {{ display: flex; align-items: center; gap: 12px; }}
  .pulse-dot {{
    width: 12px; height: 12px; border-radius: 50%; background: var(--green);
    box-shadow: 0 0 14px var(--green);
    animation: pulse 1.8s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(1.35); opacity: 0.65; }}
  }}
  .title {{ font-size: 18px; font-weight: 800; letter-spacing: 0.3px; }}
  .subtitle {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .pill {{
    font-size: 11px; font-weight: 700; padding: 6px 11px; border-radius: 999px;
    background: rgba(16, 185, 129, 0.14); color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    letter-spacing: 0.4px;
  }}
  .pill.ai {{
    background: rgba(168, 85, 247, 0.14); color: #d8b4fe;
    border-color: rgba(168, 85, 247, 0.38);
  }}
  .pill.uhd {{
    background: rgba(251, 191, 36, 0.16); color: #fde68a;
    border-color: rgba(251, 191, 36, 0.45);
  }}
  .hero {{ text-align: center; margin: 18px 0 30px; }}
  .hero h1 {{
    font-size: clamp(22px, 4.5vw, 34px); font-weight: 900;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #fbbf24, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .hero p {{ color: var(--muted); font-size: 14px; }}
  .pipeline {{ display: flex; flex-direction: column; align-items: center; }}
  .node {{
    width: 100%; max-width: 680px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  }}
  .icon-box {{
    width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    background: linear-gradient(135deg, rgba(79,172,254,0.2), rgba(0,242,254,0.08));
    border: 1px solid rgba(0,242,254,0.35);
  }}
  .node.gold .icon-box {{
    background: linear-gradient(135deg, rgba(251,191,36,0.26), rgba(245,158,11,0.1));
    border-color: rgba(251,191,36,0.5);
  }}
  .node.purple .icon-box {{
    background: linear-gradient(135deg, rgba(168,85,247,0.24), rgba(236,72,153,0.1));
    border-color: rgba(168,85,247,0.45);
  }}
  .node.green .icon-box {{
    background: linear-gradient(135deg, rgba(16,185,129,0.24), rgba(6,182,212,0.1));
    border-color: rgba(16,185,129,0.45);
  }}
  .n-body {{ flex: 1; min-width: 0; }}
  .n-step {{ font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--cyan); }}
  .node.gold .n-step {{ color: var(--gold); }}
  .node.purple .n-step {{ color: #d8b4fe; }}
  .node.green .n-step {{ color: #6ee7b7; }}
  .n-title {{ font-size: 16px; font-weight: 800; margin: 3px 0 4px; color: #fff; }}
  .n-desc {{ font-size: 13px; color: var(--muted); line-height: 1.45; }}
  .n-tag {{
    font-size: 11px; font-weight: 700; padding: 5px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.06); color: #e2e8f0; white-space: nowrap;
  }}
  .wire {{
    width: 3px; height: 32px; background: rgba(79,172,254,0.22);
    position: relative; overflow: hidden;
  }}
  .wire::after {{
    content: ""; position: absolute; left: 0; top: -18px;
    width: 100%; height: 18px;
    background: linear-gradient(180deg, transparent, var(--cyan), #fff);
    animation: flow 1.3s linear infinite;
  }}
  @keyframes flow {{ 0% {{ top: -18px; }} 100% {{ top: 34px; }} }}
  .cta-bar {{
    margin-top: 30px; text-align: center;
    display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
  }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 13px 24px; border-radius: 999px; font-weight: 800; font-size: 14px;
    text-decoration: none; color: #041019;
    background: linear-gradient(90deg, var(--cyan), var(--gold));
    box-shadow: 0 0 24px rgba(0, 242, 254, 0.35);
  }}
  .stat-card {{
    display: inline-flex; align-items: center; gap: 10px;
    padding: 12px 20px; border-radius: 999px;
    background: var(--panel); border: 1px solid var(--border);
    font-size: 13px; color: var(--text); font-weight: 600;
  }}
  @media (max-width: 560px) {{ .n-tag {{ display: none; }} .node {{ padding: 14px 15px; }} }}
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="pulse-dot"></div>
      <div>
        <div class="title">AI SMART RECAP &amp; 4K ULTRA-HD ENGINE (v4)</div>
        <div class="subtitle">Original Fast Recap + Highest-Quality Master Downloader + True 4K Output</div>
      </div>
    </div>
    <div class="badges">
      <span class="pill">{wh_badge}</span>
      <span class="pill uhd">4K ULTRA-HD MASTER · READY</span>
      <span class="pill ai">{groq_badge}</span>
      <span class="pill ai">{gem_badge}</span>
    </div>
  </div>

  <div class="hero">
    <h1>Rednote ➔ Myanmar Charon AI + 4K Studio</h1>
    <p>Original Fast Smart Recap (Untouched) + Highest-Bitrate Raw Extractor + 4K Ultra-HD (2160p) Final</p>
  </div>

  <div class="pipeline">
    <div class="node gold">
      <div class="icon-box">💎</div>
      <div class="n-body">
        <div class="n-step">NODE 01 · HIGHEST-QUALITY EXTRACTOR</div>
        <div class="n-title">originVideoKey Raw Master (No-Watermark Highest Bitrate)</div>
        <div class="n-desc">Rednote မှ အကြည်ဆုံး မူရင်း Master Stream (originVideoKey / Largest Bitrate) ကို ဦးစားပေး တိုက်ရိုက်ဆွဲထုတ်ခြင်း</div>
      </div>
      <div class="n-tag">Max Quality</div>
    </div>
    <div class="wire"></div>

    <div class="node purple">
      <div class="icon-box">🎙️</div>
      <div class="n-body">
        <div class="n-step">NODE 02 · UNTOUCHED CHARON AI + BGM</div>
        <div class="n-title">Groq Whisper v3 + Gemini 3 + Charon (1.0x) + BGM Ducking</div>
        <div class="n-desc">မူလအတိုင်း အသံသြသြ သဘာဝအမြန်နှုန်း 100% နှင့် နောက်ခံ BGM သံ ခပ်တိုးတိုးဖြင့် စတူဒီယိုအရည်အသွေး ပေါင်းစပ်ခြင်း</div>
      </div>
      <div class="n-tag">Charon 1.0x</div>
    </div>
    <div class="wire"></div>

    <div class="node green">
      <div class="icon-box">🎬</div>
      <div class="n-body">
        <div class="n-step">NODE 03 · DUAL OUTPUT ENGINE</div>
        <div class="n-title">⚡ Original Fast Smart Recap + 💎 True 4K Ultra-HD (2160×3840)</div>
        <div class="n-desc">မူလ အမြန် Smart Recap သာမက အသစ်ပေါင်းထည့်ထားသော 4K Ultra-HD Final နှင့် Lossless Raw Master များကို စိတ်ကြိုက်ရွေးထုတ်နိုင်ခြင်း</div>
      </div>
      <div class="n-tag">Dual 4K + Fast</div>
    </div>
  </div>

  <div class="cta-bar">
    <a class="btn" href="https://t.me/Aihjklbot" target="_blank">🚀 Open @Aihjklbot on Telegram</a>
    <div class="stat-card">⚡ Total Completed: {STATS["completed"]} · 💎 4K Delivered: {STATS["completed_4k"]}</div>
  </div>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def index(request: Request):
    base = RENDER_EXTERNAL_URL or str(request.base_url)
    wh_ok = False
    if BOT_TOKEN:
        try:
            res = await register_webhook(base)
            wh_ok = bool(res.get("ok"))
        except Exception:
            wh_ok = False
    return HTMLResponse(
        content=render_workflow_dashboard_html(
            wh_ok=wh_ok,
            groq_ready=bool(GROQ_API_KEY),
            gemini_ready=bool(GEMINI_API_KEY),
        )
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

    # Handle inline button taps
    cb = update.get("callback_query")
    if cb:
        cb_id = cb.get("id")
        data = cb.get("data") or ""
        msg = cb.get("message") or {}
        chat_id = msg.get("chat", {}).get("id")
        msg_id = msg.get("message_id")
        asyncio.create_task(tg_api("answerCallbackQuery", {"callback_query_id": cb_id}))

        if data.startswith("setmode|") and chat_id:
            chosen = data.split("|", 1)[1]
            USER_DEFAULT_MODE[chat_id] = chosen
            label_map = {
                "ask": "🎛️ Link ပို့တိုင်း Mode ရွေးမယ်",
                "4k": "💎 အမြဲ 4K Ultra-HD တန်းထုတ်မယ်",
                "std": "⚡ အမြဲ မူလ Smart Recap တန်းထုတ်မယ်",
            }
            asyncio.create_task(
                edit_text(
                    chat_id,
                    msg_id,
                    f"✅ Default Mode ပြောင်းပြီးပါပြီ — {label_map.get(chosen, chosen)}",
                    reply_markup=default_mode_keyboard(chosen),
                )
            )
            return JSONResponse({"ok": True})

        if data.startswith("rn|") and chat_id:
            parts = data.split("|", 2)
            if len(parts) == 3:
                _, mode, url_key = parts
                raw_url = URL_CACHE.get(url_key)
                if not raw_url:
                    asyncio.create_task(
                        send_text(
                            chat_id,
                            "⚠️ ဒီ Link သက်တမ်းကုန်သွားလို့ Rednote Link ကို နောက်တစ်ခေါက် ပြန်ပို့ပေးပါနော်။",
                            reply_to=msg_id,
                        )
                    )
                else:
                    asyncio.create_task(handle_video_job(chat_id, msg_id, raw_url, url_key, mode=mode))
        return JSONResponse({"ok": True})

    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return JSONResponse({"ok": True})

    chat_id = msg.get("chat", {}).get("id")
    text = (msg.get("text") or msg.get("caption") or "").strip()
    msg_id = msg.get("message_id")

    if not chat_id or not text:
        return JSONResponse({"ok": True})

    if text.startswith("/start"):
        asyncio.create_task(
            send_text(
                chat_id,
                "မင်္ဂလာပါ! 🎬💎🔥\n\n"
                "Rednote (小红书) Video Link ပို့ပေးလိုက်ပါ!\n\n"
                "✨ v4 Dual Studio Engine (မူလအကောင်းဆုံး + 4K အသစ်ပေါင်းစပ်ပြီး) —\n"
                "⚡ မူလ Smart Recap (အမြန်) — လက်ရှိအတိုင်း အသံသြသြ + BGM + ဖြတ်/ဆက်\n"
                "💎 4K Ultra-HD Final (အသစ်) — အကြည်ဆုံး Version ဒေါင်းပြီး Final ကို 4K (2160×3840) ပြောင်းပေးမယ်\n"
                "🎞️ Lossless Master Dub — မဖြတ်ဘဲ မူရင်းအကြည်အတိုင်း မြန်မာအသံထည့်ပေးမယ်\n"
                "📥 အကြည်ဆုံး Raw Download — Watermark မပါ အကြည်ဆုံး မူရင်း Video ထုတ်ပေးမယ်\n\n"
                "⚙️ Auto Mode ပြောင်းချင်ရင် /mode ကို နှိပ်နိုင်ပါတယ်!",
                reply_to=msg_id,
            )
        )
        return JSONResponse({"ok": True})

    if text.startswith("/mode"):
        cur = USER_DEFAULT_MODE.get(chat_id, "ask")
        asyncio.create_task(
            send_text(
                chat_id,
                "⚙️ Rednote Link ပို့လိုက်တဲ့အခါ ဘယ်လို အလုပ်လုပ်စေချင်လဲ ရွေးထားနိုင်ပါတယ် 👇",
                reply_to=msg_id,
                reply_markup=default_mode_keyboard(cur),
            )
        )
        return JSONResponse({"ok": True})

    url_match = re.search(r"https?://[^\s，。！？,<>\"'；：）】]+", text)
    if not url_match:
        asyncio.create_task(
            send_text(
                chat_id,
                "⚠️ Rednote Video Link လေး ပို့ပေးပါနော် (ဥပမာ http://xhslink.com/... ပါတဲ့ စာသား)။",
                reply_to=msg_id,
            )
        )
        return JSONResponse({"ok": True})

    raw_url = url_match.group(0).rstrip(".,!?;:)")
    url_key = remember_url(raw_url)
    pref_mode = USER_DEFAULT_MODE.get(chat_id, "ask")

    if pref_mode == "ask":
        asyncio.create_task(
            send_text(
                chat_id,
                "🎬 Rednote Video Link လက်ခံရရှိပါပြီ!\n"
                "ဘယ် Version နဲ့ ထုတ်ပေးရမလဲ ရွေးနှိပ်ပေးပါ 👇",
                reply_to=msg_id,
                reply_markup=mode_selection_keyboard(url_key),
            )
        )
    else:
        asyncio.create_task(handle_video_job(chat_id, msg_id, raw_url, url_key, mode=pref_mode))
    return JSONResponse({"ok": True})


def run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


async def download_rednote_video(raw_url: str, out_path: str) -> tuple[bool, str]:
    """
    Upgraded Highest-Quality Rednote Extractor:
    1. Tier 1: Extracts unwatermarked `originVideoKey` raw creator master (True 4K / 2K / 1080p60) across CDN mirrors.
    2. Tier 2: Probes all `masterUrl` & CDN streams and ranks by LARGEST Content-Length (highest bitrate/resolution).
    3. Tier 3: Falls back to `og:video` direct stream.
    4. Tier 4: Falls back to uncapped `yt-dlp` (`bestvideo+bestaudio/best`).
    """
    mobile_headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    desktop_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    expanded_url = raw_url
    origin_urls: list[str] = []
    master_urls: list[str] = []
    og_urls: list[str] = []

    def _clean(u: str) -> str:
        return (
            u.encode("utf-8")
            .decode("unicode_escape", errors="ignore")
            .replace("\\u002F", "/")
            .replace("\\/", "/")
            .replace("&amp;", "&")
            .strip()
        )

    async with httpx.AsyncClient(follow_redirects=True, timeout=45.0) as client:
        for hdrs in (desktop_headers, mobile_headers):
            try:
                r = await client.get(expanded_url, headers=hdrs)
                expanded_url = str(r.url)
                html = r.text

                # 1. Raw Creator Master (originVideoKey -> Unwatermarked Highest Quality)
                for key in re.findall(r'"originVideoKey"\s*:\s*"([^"]+)"', html):
                    k = _clean(key).lstrip("/")
                    if k:
                        for host in (
                            "https://sns-video-bd.xhscdn.com",
                            "https://sns-video-hw.xhscdn.com",
                            "https://sns-video-al.xhscdn.com",
                        ):
                            origin_urls.append(f"{host}/{k}")

                # 2. High-bitrate masterUrl & CDN streams
                for m in re.findall(r'"masterUrl"\s*:\s*"([^"]+)"', html):
                    u_c = _clean(m)
                    if u_c.startswith("http"):
                        master_urls.append(u_c)

                for c in re.findall(r"https?:(?:\\u002F|/){2}sns-video[^\s\"'<>\\]+", html):
                    u_c = _clean(c)
                    if u_c.startswith("http"):
                        master_urls.append(u_c)

                # 3. og:video fallback
                og_m = re.findall(
                    r'<meta[^>]+(?:name|property)=["\']og:video["\'][^>]+content=["\']([^"\']+)["\']',
                    html,
                    flags=re.I,
                )
                og_m += re.findall(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']og:video["\']',
                    html,
                    flags=re.I,
                )
                for u in og_m:
                    u_c = _clean(u)
                    if u_c.startswith("http"):
                        og_urls.append(u_c)

                if origin_urls or master_urls:
                    break
            except Exception as e:
                logger.warning(f"HTML scrape warning: {e}")

        dl_headers = {
            "User-Agent": desktop_headers["User-Agent"],
            "Referer": "https://www.xiaohongshu.com/",
        }

        async def _probe_and_rank(urls: list[str]) -> list[tuple[int, str]]:
            seen_u = set()
            ranked = []
            for vurl in urls:
                if vurl in seen_u:
                    continue
                seen_u.add(vurl)
                clen = 0
                try:
                    hr = await client.head(vurl, headers=dl_headers, timeout=10.0)
                    if hr.status_code in (200, 206, 301, 302):
                        clen = int(hr.headers.get("content-length", "0") or 0)
                    elif hr.status_code >= 400:
                        continue
                except Exception:
                    clen = 1
                ranked.append((clen, vurl))
            ranked.sort(key=lambda x: x[0], reverse=True)
            return ranked

        async def _stream_to_disk(vurl: str) -> bool:
            try:
                async with client.stream("GET", vurl, headers=dl_headers, timeout=180.0) as resp:
                    if resp.status_code in (200, 206):
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=131072):
                                f.write(chunk)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 25_000:
                            return True
            except Exception as e:
                logger.warning(f"Stream download warning: {e}")
            return False

        # Priority 1: Unwatermarked originVideoKey Master (highest filesize first)
        for _, best_u in await _probe_and_rank(origin_urls):
            if await _stream_to_disk(best_u):
                return True, "Raw Master (No-Watermark)"

        # Priority 2: Highest-bitrate masterUrl / CDN stream
        for _, best_u in await _probe_and_rank(master_urls):
            if await _stream_to_disk(best_u):
                return True, "High-Bitrate CDN"

        # Priority 3: og:video stream
        for _, best_u in await _probe_and_rank(og_urls):
            if await _stream_to_disk(best_u):
                return True, "HD Direct"

    # Priority 4: Uncapped yt-dlp bestvideo+bestaudio
    candidate_urls = [expanded_url]
    if "rednote.com" in expanded_url:
        candidate_urls.append(expanded_url.replace("rednote.com", "xiaohongshu.com"))
    if raw_url not in candidate_urls:
        candidate_urls.append(raw_url)

    for u in candidate_urls:
        if os.path.exists(out_path):
            os.remove(out_path)
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            out_path,
            u,
        ]
        await asyncio.to_thread(run_cmd, cmd, 180)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 25_000:
            return True, "Max Stream (yt-dlp)"

    return False, ""


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

    # Account for 90/270 rotation metadata so portrait/landscape 4K dimensions never swap
    rot_m = re.search(r"rotation of\s*(-?\d+)|rotate\s*:\s*(-?\d+)", out, flags=re.I)
    if rot_m:
        ang = abs(int(rot_m.group(1) or rot_m.group(2) or 0)) % 180
        if ang == 90 and w > 0 and h > 0:
            w, h = h, w

    return dur, w, h, vcodec


def get_media_duration(path: str) -> float:
    return probe_video_info(path)[0]


def target_4k_dims(src_w: int, src_h: int) -> tuple[int, int]:
    """Computes exact aspect-preserving 4K UHD dimensions (e.g. 2160x3840 portrait or 3840x2160 landscape)."""
    if src_w <= 0 or src_h <= 0:
        return 2160, 3840
    long_edge = max(src_w, src_h)
    short_edge = min(src_w, src_h)
    scale = min(3840.0 / long_edge, 2160.0 / short_edge)
    tw = max(2, int(round(src_w * scale / 2.0)) * 2)
    th = max(2, int(round(src_h * scale / 2.0)) * 2)
    return tw, th


MYANMAR_DIGITS = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")


def clean_burmese_for_tts(text: str) -> str:
    t = re.sub(r"[*_#`~<>[\]()\"'“”‘’…—–-]", " ", text)
    t = re.sub(r"[\U00010000-\U0010ffff]", " ", t)
    t = t.translate(MYANMAR_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_speakable_text(text: str) -> bool:
    return bool(re.search(r"[\u1000-\u1049a-zA-Z0-9]", text))


async def groq_whisper_transcribe(audio_mp3_path: str) -> list[dict]:
    if not GROQ_API_KEY:
        return []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for w_model in ("whisper-large-v3-turbo", "whisper-large-v3"):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_mp3_path, "rb") as af:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        data={
                            "model": w_model,
                            "response_format": "verbose_json",
                            "timestamp_granularities[]": "segment",
                        },
                        files={"file": ("speech.mp3", af, "audio/mpeg")},
                    )
                if resp.status_code == 200:
                    j = resp.json()
                    segs = []
                    for s in j.get("segments", []):
                        txt = (s.get("text") or "").strip()
                        if txt:
                            segs.append(
                                {
                                    "start": round(float(s.get("start", 0.0)), 2),
                                    "end": round(float(s.get("end", 0.0)), 2),
                                    "text": txt,
                                }
                            )
                    logger.info(f"Groq {w_model} extracted {len(segs)} segments")
                    return segs
        except Exception as e:
            logger.warning(f"Groq Whisper warning: {e}")
    return []


async def transcribe_and_script_scenes(audio_mp3_path: str, total_dur: float) -> list[dict]:
    whisper_segs = await groq_whisper_transcribe(audio_mp3_path)

    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_mp3_path, "rb") as f:
        audio_bytes = f.read()

    whisper_hint = ""
    if whisper_segs:
        whisper_json = json.dumps(whisper_segs, ensure_ascii=False)
        whisper_hint = f"""
Exact Whisper Large-v3 speech timestamps & transcript:
{whisper_json}
"""

    prompt = f"""
You are a top Myanmar male video narrator / recap scriptwriter.
Listen to this {total_dur:.1f}-second video audio.
{whisper_hint}
Task:
1. Group the spoken content into **4 to 14 smooth, coherent scene blocks** (`start` and `end` timestamps in seconds within `0.0` to `{total_dur:.1f}`).
   - Merge tiny choppy 1-second fragments into natural 3 to 9 second sentences!
   - Skip silent dead air or music-only filler sections (our video editor will automatically trim out unused filler gaps!).
   - If there is NO spoken content at all, return `[]`.
2. Rewrite each scene block into **100% natural, clear, punchy colloquial spoken Burmese (မြန်မာအပြောစကားစစ်စစ်)** for a **single confident male narrator**:
   - NEVER use literary Burmese ("သည်", "၏", "၌", "၎င်း", "သောကြောင့်", "ဖြစ်ပေသည်").
   - ALWAYS use clear everyday spoken Burmese ("တယ်", "ဗျာ", "နော်", "ပေါ့", "လေ", "ကျွန်တော်", "အဲဒါကြောင့်").
   - Write ALL numbers and English words phonetically in pure Myanmar Unicode script (e.g., "အေအိုင်", "တစ်ရာ", "အိုင်ဖုန်း") so the voice pronounces every word clearly!
   - End every scene's Burmese sentence cleanly with `။`.

Return ONLY valid JSON (no markdown fences):
[
  {{
    "start": 0.4,
    "end": 5.2,
    "burmese": "ဒီဗီဒီယိုလေးထဲမှာတော့ တကယ်အသုံးဝင်တဲ့ နည်းလမ်းကောင်းလေးကို ပြောပြပေးသွားမှာပါဗျာ။"
  }}
]
""".strip()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    cfg = types.GenerateContentConfig(temperature=0.25, response_mime_type="application/json")

    raw_json_text = None
    last_err = None
    for model_name in PREFERRED_MODELS:
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=[audio_part, prompt],
                config=cfg,
            )
            if resp and resp.text:
                raw_json_text = resp.text.strip()
                break
        except Exception as e:
            last_err = e

    if raw_json_text is None:
        raise RuntimeError(f"Gemini script error: {last_err}")

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_json_text, flags=re.M).strip()
    segments = json.loads(cleaned)
    if not isinstance(segments, list):
        segments = segments.get("segments", []) if isinstance(segments, dict) else []

    out = []
    for seg in segments:
        try:
            st = max(0.0, float(seg.get("start", 0.0)))
            en = min(total_dur, float(seg.get("end", st + 3.0)))
            if en <= st + 0.3:
                en = min(total_dur, st + 2.5)
            bm = clean_burmese_for_tts(str(seg.get("burmese", "")))
            if not bm or not has_speakable_text(bm):
                continue
            if not bm.endswith("။"):
                bm += "။"
            out.append({"start": st, "end": en, "burmese": bm})
        except Exception:
            continue

    out.sort(key=lambda x: x["start"])
    for i in range(len(out) - 1):
        if out[i]["end"] > out[i + 1]["start"]:
            out[i]["end"] = max(out[i]["start"] + 0.4, out[i + 1]["start"])
    return out


def trim_pcm_silence(pcm: np.ndarray, thresh: float = 0.012) -> np.ndarray:
    if len(pcm) == 0:
        return pcm
    hop = int(0.015 * SAMPLE_RATE)
    if len(pcm) < hop * 2:
        return pcm
    idx_active = np.where(np.abs(pcm) > thresh)[0]
    if len(idx_active) == 0:
        return pcm
    pad = int(0.04 * SAMPLE_RATE)
    s = max(0, int(idx_active[0]) - pad)
    e = min(len(pcm), int(idx_active[-1]) + pad)
    return pcm[s:e]


def split_master_pcm_into_n_clips(master_pcm: np.ndarray, scenes: list[dict]) -> list[np.ndarray]:
    n = len(scenes)
    master_pcm = trim_pcm_silence(master_pcm)
    if n <= 1 or len(master_pcm) < SAMPLE_RATE:
        return [master_pcm]

    weights = np.array([max(4, len(s["burmese"])) for s in scenes], dtype=np.float64)
    cum_ratios = np.cumsum(weights)[:-1] / np.sum(weights)

    hop = int(0.02 * SAMPLE_RATE)
    n_frames = max(1, (len(master_pcm) - hop) // hop)
    rms = np.array(
        [float(np.sqrt(np.mean(master_pcm[i * hop : i * hop + hop] ** 2))) for i in range(n_frames)],
        dtype=np.float32,
    )

    cut_samples = [0]
    for r in cum_ratios:
        expected_frame = int(r * n_frames)
        win = max(25, int(0.14 * n_frames))
        lo = max(cut_samples[-1] // hop + 10, expected_frame - win)
        hi = min(n_frames - 10, expected_frame + win)
        if hi <= lo:
            best_frame = max(cut_samples[-1] // hop + 1, expected_frame)
        else:
            dists = np.abs(np.arange(lo, hi) - expected_frame) / max(1.0, float(win))
            score = rms[lo:hi] + 0.018 * dists
            best_frame = lo + int(np.argmin(score))
        cut_samples.append(min(len(master_pcm), best_frame * hop))
    cut_samples.append(len(master_pcm))

    clips = []
    for i in range(n):
        seg = master_pcm[cut_samples[i] : cut_samples[i + 1]]
        seg = trim_pcm_silence(seg)
        fade = min(int(0.015 * SAMPLE_RATE), len(seg) // 4)
        if fade > 1:
            seg = seg.copy()
            seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        clips.append(seg)
    return clips


async def synthesize_charon_clips(scenes: list[dict], work_dir: str) -> list[np.ndarray]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    joined_script = "\n\n".join(s["burmese"] for s in scenes)

    for tts_model in PREFERRED_TTS_MODELS:
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=tts_model,
                contents=joined_script,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=CHARON_VOICE)
                        )
                    ),
                ),
            )
            part = resp.candidates[0].content.parts[0].inline_data
            audio_bytes = part.data
            mime = (part.mime_type or "").lower()
            if audio_bytes and len(audio_bytes) > 1000:
                tmp_raw = os.path.join(work_dir, "charon_raw.bin")
                tmp_pcm = os.path.join(work_dir, "charon_24k.pcm")
                with open(tmp_raw, "wb") as f:
                    f.write(audio_bytes)

                if audio_bytes[:4] == b"RIFF":
                    run_cmd(
                        [
                            FFMPEG_BIN,
                            "-y",
                            "-i",
                            tmp_raw,
                            "-f",
                            "s16le",
                            "-acodec",
                            "pcm_s16le",
                            "-ac",
                            "1",
                            "-ar",
                            str(SAMPLE_RATE),
                            tmp_pcm,
                        ],
                        45,
                    )
                else:
                    rate_m = re.search(r"rate=(\d+)", mime)
                    pcm_rate = rate_m.group(1) if rate_m else "24000"
                    run_cmd(
                        [
                            FFMPEG_BIN,
                            "-y",
                            "-f",
                            "s16le",
                            "-ar",
                            pcm_rate,
                            "-ac",
                            "1",
                            "-i",
                            tmp_raw,
                            "-f",
                            "s16le",
                            "-acodec",
                            "pcm_s16le",
                            "-ac",
                            "1",
                            "-ar",
                            str(SAMPLE_RATE),
                            tmp_pcm,
                        ],
                        45,
                    )
                if os.path.exists(tmp_pcm) and os.path.getsize(tmp_pcm) > 1000:
                    master_pcm = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    logger.info(f"Charon 1-call master synthesized via {tts_model} ({len(master_pcm)/SAMPLE_RATE:.1f}s)")
                    return split_master_pcm_into_n_clips(master_pcm, scenes)
        except Exception as e:
            logger.warning(f"Charon TTS ({tts_model}) warning: {str(e)[:160]}")

    logger.info("Using fallback Thiha Deep Narrator per scene")
    clips = []
    for idx, sc in enumerate(scenes):
        mp3_p = os.path.join(work_dir, f"thiha_{idx}.mp3")
        pcm_p = os.path.join(work_dir, f"thiha_{idx}.pcm")
        for attempt in range(3):
            try:
                comm = edge_tts.Communicate(sc["burmese"], FALLBACK_THIHA_VOICE, rate="+4%", pitch="-6Hz")
                await comm.save(mp3_p)
                if os.path.exists(mp3_p) and os.path.getsize(mp3_p) > 300:
                    break
            except Exception:
                await asyncio.sleep(1.2 * (attempt + 1))

        if os.path.exists(mp3_p) and os.path.getsize(mp3_p) > 300:
            run_cmd(
                [FFMPEG_BIN, "-y", "-i", mp3_p, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), pcm_p],
                30,
            )
        if os.path.exists(pcm_p) and os.path.getsize(pcm_p) > 200:
            seg = np.fromfile(pcm_p, dtype=np.int16).astype(np.float32) / 32768.0
            clips.append(trim_pcm_silence(seg))
        else:
            clips.append(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))
    return clips


def extract_vocal_suppressed_bgm(orig_video: str, orig_dur: float, work_dir: str) -> np.ndarray:
    bgm_pcm_path = os.path.join(work_dir, "bgm_bed.pcm")
    vocal_suppress_af = (
        "pan=stereo|c0=0.5*c0-0.38*c1|c1=0.5*c1-0.38*c0,"
        "equalizer=f=550:t=q:w=1.1:g=-13,"
        "equalizer=f=1650:t=q:w=1.1:g=-13,"
        "equalizer=f=3000:t=q:w=1.2:g=-10,"
        "lowpass=f=6500"
    )
    run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            orig_video,
            "-vn",
            "-af",
            vocal_suppress_af,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            bgm_pcm_path,
        ],
        90,
    )
    if os.path.exists(bgm_pcm_path) and os.path.getsize(bgm_pcm_path) > 400:
        orig_bgm = np.fromfile(bgm_pcm_path, dtype=np.int16).astype(np.float32) / 32768.0
        os.remove(bgm_pcm_path)
        bgm_peak = float(np.max(np.abs(orig_bgm)))
        if bgm_peak > 0.01:
            orig_bgm = (orig_bgm / bgm_peak) * 0.55
        return orig_bgm
    return np.zeros(int(orig_dur * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)


def prepare_recap_plan_and_audio(
    orig_video: str,
    orig_dur: float,
    scenes: list[dict],
    voice_clips: list[np.ndarray],
    work_dir: str,
) -> tuple[list[tuple[float, float, float]], str]:
    """
    Exact untouched Smart Recap audio + scene slice timing planner:
    - Charon 1.0x natural speed
    - Vocal-suppressed original BGM auto-ducked (16% under voice, 36% during pauses)
    - Frees all numpy buffers before video encoding to stay well under 512MB RAM!
    """
    orig_bgm = extract_vocal_suppressed_bgm(orig_video, orig_dur, work_dir)
    video_slices: list[tuple[float, float, float]] = []
    audio_chunks: list[np.ndarray] = []

    def slice_bgm(s_sec: float, e_sec: float, target_samples: int, vol: float) -> np.ndarray:
        s_i = max(0, min(len(orig_bgm) - 1, int(s_sec * SAMPLE_RATE)))
        e_i = max(s_i + 1, min(len(orig_bgm), int(e_sec * SAMPLE_RATE)))
        raw = orig_bgm[s_i:e_i]
        if len(raw) == 0 or target_samples <= 0:
            return np.zeros(max(0, target_samples), dtype=np.float32)
        if len(raw) != target_samples:
            x_old = np.linspace(0.0, 1.0, len(raw), dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        fade = min(int(0.03 * SAMPLE_RATE), len(raw) // 4)
        if fade > 1:
            raw[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            raw[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return raw * vol

    first_start = scenes[0]["start"]
    if first_start > 0.25:
        intro_dur = min(0.8, first_start)
        video_slices.append((0.0, first_start if first_start <= 1.2 else intro_dur, intro_dur))
        audio_chunks.append(slice_bgm(0.0, intro_dur, int(intro_dur * SAMPLE_RATE), vol=0.55))

    for sc, v_clip in zip(scenes, voice_clips):
        src_s = sc["start"]
        src_e = sc["end"]
        src_len = max(0.4, src_e - src_s)

        c_peak = float(np.max(np.abs(v_clip))) if len(v_clip) > 0 else 0.0
        norm_voice = (v_clip / c_peak) * 0.92 if c_peak > 0.01 else v_clip

        pause_after = 0.28
        voice_dur = len(norm_voice) / SAMPLE_RATE
        target_dur = max(0.6, voice_dur + pause_after)
        target_samples = int(round(target_dur * SAMPLE_RATE))

        if src_len > target_dur * 1.45:
            cut_src_e = src_s + target_dur * 1.35
        else:
            cut_src_e = src_e

        video_slices.append((src_s, cut_src_e, target_dur))

        bgm_seg = slice_bgm(src_s, cut_src_e, target_samples, vol=1.0)
        env = np.full(target_samples, 0.28, dtype=np.float32)
        v_len = min(target_samples, len(norm_voice))
        env[:v_len] = 0.16
        ramp = min(int(0.05 * SAMPLE_RATE), target_samples - v_len)
        if ramp > 2 and v_len + ramp <= target_samples:
            env[v_len : v_len + ramp] = np.linspace(0.16, 0.36, ramp, dtype=np.float32)
            env[v_len + ramp :] = 0.36

        mixed = bgm_seg * env
        mixed[:v_len] += norm_voice[:v_len]
        audio_chunks.append(mixed)

    last_end = scenes[-1]["end"]
    if orig_dur - last_end > 0.25:
        outro_dur = min(0.65, orig_dur - last_end)
        video_slices.append((last_end, last_end + outro_dur, outro_dur))
        audio_chunks.append(slice_bgm(last_end, last_end + outro_dur, int(outro_dur * SAMPLE_RATE), vol=0.45))

    full_audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(SAMPLE_RATE, dtype=np.float32)
    del audio_chunks, orig_bgm
    master_s16 = np.clip(full_audio * 32767.0, -32768, 32767).astype(np.int16)
    del full_audio
    master_wav = os.path.join(work_dir, "master_mix.wav")
    raw_pcm_f = os.path.join(work_dir, "master_mix.pcm")
    master_s16.tofile(raw_pcm_f)
    del master_s16
    gc.collect()

    run_cmd(
        [FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", raw_pcm_f, master_wav],
        30,
    )
    if os.path.exists(raw_pcm_f):
        os.remove(raw_pcm_f)

    return video_slices, master_wav


async def encode_recap_slices_and_mux(
    orig_video: str,
    video_slices: list[tuple[float, float, float]],
    master_wav: str,
    work_dir: str,
    final_video_path: str,
    quality: str = "std",
    src_w: int = 1080,
    src_h: int = 1920,
    progress_cb=None,
):
    """
    - quality == "std": 100% Untouched original fast sequential slice cutting (480p low-RAM).
    - quality == "4k": Edge-sharpened True 4K UHD (e.g. 2160x3840) sequential low-RAM (<275MB) encoder!
    """
    tw, th = target_4k_dims(src_w, src_h)
    seg_files = []
    total_segs = len(video_slices)

    for idx, (s_t, e_t, tgt_d) in enumerate(video_slices):
        if progress_cb and quality == "4k":
            await progress_cb(idx + 1, total_segs, tw, th)

        raw_d = max(0.1, e_t - s_t)
        pts_factor = tgt_d / raw_d
        seg_path = os.path.join(work_dir, f"seg_{idx:03d}.mp4")

        if quality == "4k":
            # Sharpen source edges BEFORE fast_bilinear 4K upscale -> 2.5x faster & <275MB child RSS!
            vf_str = (
                f"setpts={pts_factor:.4f}*(PTS-STARTPTS),"
                f"unsharp=3:3:0.7:3:3:0.0,"
                f"scale={tw}:{th}:flags=fast_bilinear,"
                f"fps=24,format=yuv420p"
            )
            cmd = [
                FFMPEG_BIN,
                "-y",
                "-ss",
                f"{s_t:.3f}",
                "-t",
                f"{raw_d:.3f}",
                "-i",
                orig_video,
                "-an",
                "-vf",
                vf_str,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-x264-params",
                "rc-lookahead=0:sync-lookahead=0:ref=1:bframes=0:subme=0:me=dia:partitions=none:trellis=0:weightp=0:cabac=0:deblock=0:aq-mode=0:sliced-threads=0:threads=1",
                "-video_track_timescale",
                "90000",
                "-threads",
                "1",
                seg_path,
            ]
            slice_timeout = 600
        else:
            # 100% Untouched Original Fast Smart Recap
            vf_str = f"setpts={pts_factor:.4f}*(PTS-STARTPTS),scale='min(480,iw)':-2,fps=24,format=yuv420p"
            cmd = [
                FFMPEG_BIN,
                "-y",
                "-ss",
                f"{s_t:.3f}",
                "-t",
                f"{raw_d:.3f}",
                "-i",
                orig_video,
                "-an",
                "-vf",
                vf_str,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "27",
                "-video_track_timescale",
                "90000",
                "-threads",
                "1",
                seg_path,
            ]
            slice_timeout = 150

        await asyncio.to_thread(run_cmd, cmd, slice_timeout)
        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 500:
            seg_files.append(seg_path)

    if not seg_files:
        raise RuntimeError("No video slices were generated")

    # Free original video disk space before final lossless concat
    if os.path.exists(orig_video):
        os.remove(orig_video)

    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as cf:
        for tp in seg_files:
            cf.write(f"file '{tp}'\n")

    audio_br = "192k" if quality == "4k" else "128k"
    proc = await asyncio.to_thread(
        run_cmd,
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-i",
            master_wav,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            audio_br,
            "-shortest",
            "-movflags",
            "+faststart",
            final_video_path,
        ],
        120,
    )

    for tp in seg_files:
        if os.path.exists(tp):
            os.remove(tp)

    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) < 5000:
        err_tail = proc.stderr.decode(errors="ignore")[-300:]
        raise RuntimeError(f"FFmpeg video concat failed: {err_tail}")


def mux_lossless_master_with_bgm(
    orig_video: str,
    orig_dur: float,
    vcodec: str,
    scenes: list[dict],
    voice_clips: list[np.ndarray],
    work_dir: str,
    final_video_path: str,
):
    """Zero-loss (-c:v copy) master muxer: keeps 100% original video pixels + Charon 1.0x + BGM ducking."""
    total_samples = max(int(round(orig_dur * SAMPLE_RATE)), SAMPLE_RATE)
    orig_bgm = extract_vocal_suppressed_bgm(orig_video, orig_dur, work_dir)
    if len(orig_bgm) < total_samples:
        orig_bgm = np.pad(orig_bgm, (0, total_samples - len(orig_bgm)))
    else:
        orig_bgm = orig_bgm[:total_samples]

    voice_track = np.zeros(total_samples, dtype=np.float32)
    duck_env = np.full(total_samples, 0.36, dtype=np.float32)
    cursor = 0

    for sc, v_clip in zip(scenes, voice_clips):
        if len(v_clip) == 0:
            continue
        c_peak = float(np.max(np.abs(v_clip)))
        norm_v = (v_clip / c_peak) * 0.94 if c_peak > 0.01 else v_clip
        desired_start = int(round(sc["start"] * SAMPLE_RATE))
        place_start = max(cursor, desired_start)
        if place_start >= total_samples:
            break
        avail = min(len(norm_v), total_samples - place_start)
        if avail <= 0:
            continue
        place_end = place_start + avail
        voice_track[place_start:place_end] += norm_v[:avail]
        duck_env[place_start:place_end] = 0.15
        ramp = int(0.05 * SAMPLE_RATE)
        r_s = max(0, place_start - ramp)
        if place_start > r_s:
            duck_env[r_s:place_start] = np.minimum(
                duck_env[r_s:place_start], np.linspace(0.36, 0.15, place_start - r_s, dtype=np.float32)
            )
        r_e = min(total_samples, place_end + ramp)
        if r_e > place_end:
            duck_env[place_end:r_e] = np.minimum(
                duck_env[place_end:r_e], np.linspace(0.15, 0.36, r_e - place_end, dtype=np.float32)
            )
        cursor = place_end + int(0.15 * SAMPLE_RATE)

    mixed = np.clip(orig_bgm * duck_env + voice_track, -0.99, 0.99)
    del orig_bgm, voice_track, duck_env
    master_s16 = (mixed * 32767.0).astype(np.int16)
    del mixed
    gc.collect()

    raw_pcm_f = os.path.join(work_dir, "lossless_mix.pcm")
    master_wav = os.path.join(work_dir, "lossless_mix.wav")
    master_s16.tofile(raw_pcm_f)
    del master_s16
    run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", raw_pcm_f, master_wav], 30)
    if os.path.exists(raw_pcm_f):
        os.remove(raw_pcm_f)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        orig_video,
        "-i",
        master_wav,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
    ]
    if vcodec in ("hevc", "h265"):
        cmd += ["-tag:v", "hvc1"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", final_video_path]
    proc = run_cmd(cmd, 120)
    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) < 5000:
        err_tail = proc.stderr.decode(errors="ignore")[-300:]
        raise RuntimeError(f"Lossless mux failed: {err_tail}")


def split_video_lossless(video_path: str, total_dur: float, work_dir: str) -> list[str]:
    fsize = os.path.getsize(video_path)
    if fsize <= MAX_TG_BYTES:
        return [video_path]

    num_parts = max(2, int(np.ceil(fsize / (41 * 1024 * 1024))))
    part_dur = max(6.0, total_dur / num_parts)
    out_pattern = os.path.join(work_dir, "part_%02d.mp4")

    run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            video_path,
            "-c",
            "copy",
            "-map",
            "0",
            "-f",
            "segment",
            "-segment_time",
            f"{part_dur:.2f}",
            "-reset_timestamps",
            "1",
            out_pattern,
        ],
        90,
    )
    parts = sorted(
        os.path.join(work_dir, fn)
        for fn in os.listdir(work_dir)
        if fn.startswith("part_") and fn.endswith(".mp4") and os.path.getsize(os.path.join(work_dir, fn)) > 1000
    )
    return parts if parts else [video_path]


async def handle_video_job(chat_id: int, reply_msg_id: int, url: str, url_key: str, mode: str = "std"):
    mode_titles = {
        "std": "⚡ မူလ Smart Recap",
        "4k": "💎 4K Ultra-HD Recap",
        "lossless": "🎞️ Lossless Master Dub",
        "raw": "📥 မူရင်းအကြည်ဆုံး Raw Video",
    }
    m_title = mode_titles.get(mode, mode_titles["std"])
    status_id = await send_text(
        chat_id,
        f"⏳ [{m_title}]\nRednote မှ အကြည်ဆုံး မူရင်း Video ကို စတင်ဒေါင်းလုဒ်ဆွဲနေပါပြီ...",
        reply_to=reply_msg_id,
    )

    async with job_semaphore:
        work_dir = tempfile.mkdtemp(prefix="xhs_mm_", dir=WORK_ROOT)
        try:
            orig_video = os.path.join(work_dir, "orig.mp4")
            ok, dl_tier = await download_rednote_video(url, orig_video)
            if not ok:
                await edit_text(
                    chat_id,
                    status_id,
                    "❌ ဒီ Link ကနေ Video ဒေါင်းလို့ မရပါဘူး။\n"
                    "Rednote App ထဲက Share ➔ Copy Link (复制链接) နှိပ်ပြီး ရလာတဲ့ Link အသစ်ကို ပြန်ပို့ကြည့်ပေးပါနော်။",
                )
                return

            dur, src_w, src_h, vcodec = await asyncio.to_thread(probe_video_info, orig_video)
            if dur <= 0.5:
                await edit_text(chat_id, status_id, "❌ ဒါက Video မဟုတ်ဘဲ ပုံ (Photo post) ဖြစ်နေပါတယ်နော်။")
                return
            if dur > 480:
                await edit_text(chat_id, status_id, "⚠️ Video က ၈ မိနစ်ထက် ပိုရှည်နေလို့ Free Server ပေါ်မှာ မနိုင်ပါဘူးနော်။")
                return

            src_res_str = f"{src_w}×{src_h}" if src_w and src_h else "HD"

            # Mode 4: Raw Unwatermarked Highest-Quality Download
            if mode == "raw":
                await edit_text(chat_id, status_id, f"📤 မူရင်းအကြည်ဆုံး Video ({src_res_str} · {dl_tier}) ကို ပို့ပေးနေပါပြီ...")
                parts = await asyncio.to_thread(split_video_lossless, orig_video, dur, work_dir)
                all_ok = True
                for p_idx, part_path in enumerate(parts):
                    cap = (
                        f"📥 မူရင်းအကြည်ဆုံး Rednote Video (No Watermark)\n📐 Resolution: {src_res_str} · {dl_tier}"
                        if len(parts) == 1
                        else f"📥 မူရင်းအကြည်ဆုံး Video (အပိုင်း {p_idx + 1}/{len(parts)}) · {src_res_str}"
                    )
                    kb = followup_keyboard(url_key, "raw") if p_idx == len(parts) - 1 else None
                    data_payload = {
                        "chat_id": str(chat_id),
                        "caption": cap,
                        "reply_to_message_id": str(reply_msg_id),
                        "supports_streaming": "true",
                    }
                    if src_w and src_h:
                        data_payload["width"] = str(src_w)
                        data_payload["height"] = str(src_h)
                    if kb:
                        data_payload["reply_markup"] = json.dumps(kb)
                    with open(part_path, "rb") as vf:
                        send_res = await tg_api(
                            "sendVideo",
                            data=data_payload,
                            files={"video": (f"rednote_raw_{p_idx+1}.mp4", vf, "video/mp4")},
                            timeout=300.0,
                        )
                    if not send_res.get("ok"):
                        all_ok = False
                        await edit_text(chat_id, status_id, f"⚠️ Video ပို့ရာတွင် အခက်အခဲရှိပါသည်: {send_res}")
                        break
                if all_ok:
                    STATS["completed"] += 1
                    try:
                        await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
                    except Exception:
                        pass
                return

            # Modes 1, 2, 3: AI Burmese Dubbing / Smart Recap
            await edit_text(
                chat_id,
                status_id,
                f"📥 Source ({src_res_str} · {dl_tier}) ရပါပြီ!\n🎧 Whisper v3 + Gemini 3 နဲ့ မြန်မာစကားပြော Script ရေးနေပါတယ်...",
            )

            audio_mp3 = os.path.join(work_dir, "speech.mp3")
            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-i", orig_video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_mp3],
                60,
            )

            scenes = await transcribe_and_script_scenes(audio_mp3, dur)
            if os.path.exists(audio_mp3):
                os.remove(audio_mp3)

            if not scenes:
                await edit_text(chat_id, status_id, "⚠️ ဒီ Video ထဲမှာ စကားပြောသံ မပါပါဘူး (သီချင်းသံသက်သက် ဖြစ်နေပါတယ်)။")
                return

            await edit_text(
                chat_id,
                status_id,
                f"🎙️ Charon ယောက်ျားလေးအသံ (သဘာဝအမြန်နှုန်း 100%) သွင်းနေပါပြီ... ({len(scenes)} ပိုင်း)",
            )
            voice_clips = await synthesize_charon_clips(scenes, work_dir)

            final_video = os.path.join(work_dir, "final_mm.mp4")

            if mode == "lossless":
                await edit_text(
                    chat_id,
                    status_id,
                    f"🎞️ မူရင်းအကြည် ({src_res_str}) ကို Pixel မပျက်ဘဲ မြန်မာအသံ + BGM ပေါင်းစပ်နေပါပြီ...",
                )
                await asyncio.to_thread(
                    mux_lossless_master_with_bgm,
                    orig_video,
                    dur,
                    vcodec,
                    scenes,
                    voice_clips,
                    work_dir,
                    final_video,
                )
                del voice_clips
                gc.collect()
                if os.path.exists(orig_video):
                    os.remove(orig_video)
            else:
                video_slices, master_wav = await asyncio.to_thread(
                    prepare_recap_plan_and_audio,
                    orig_video,
                    dur,
                    scenes,
                    voice_clips,
                    work_dir,
                )
                del voice_clips
                gc.collect()

                if mode == "4k":
                    tw, th = target_4k_dims(src_w, src_h)

                    async def _on_4k_slice(cur_i: int, tot_i: int, w4: int, h4: int):
                        pct = int(round((cur_i - 1) * 100 / max(1, tot_i)))
                        await edit_text(
                            chat_id,
                            status_id,
                            f"💎 4K Ultra-HD ({w4}×{h4}) ပြောင်းပြီး ဖြတ်/ဆက်နေပါပြီ...\n"
                            f"📊 အပိုင်း {cur_i}/{tot_i} ({pct}%) · ခဏလေးစောင့်ပေးပါနော်...",
                        )

                    await encode_recap_slices_and_mux(
                        orig_video,
                        video_slices,
                        master_wav,
                        work_dir,
                        final_video,
                        quality="4k",
                        src_w=src_w,
                        src_h=src_h,
                        progress_cb=_on_4k_slice,
                    )
                else:
                    await edit_text(
                        chat_id,
                        status_id,
                        "✂️ အသံနဲ့ကိုက်အောင် Video အပိုင်းတွေကို ဖြတ်/ဆက်ပြီး နောက်ခံသံ ညှိနေပါပြီ...",
                    )
                    await encode_recap_slices_and_mux(
                        orig_video,
                        video_slices,
                        master_wav,
                        work_dir,
                        final_video,
                        quality="std",
                        src_w=src_w,
                        src_h=src_h,
                    )

            final_dur, out_w, out_h, _ = await asyncio.to_thread(probe_video_info, final_video)
            parts = await asyncio.to_thread(split_video_lossless, final_video, final_dur or dur, work_dir)
            if len(parts) > 1 and os.path.exists(final_video):
                os.remove(final_video)

            out_res_str = f"{out_w}×{out_h}" if out_w and out_h else src_res_str
            await edit_text(chat_id, status_id, f"📤 Final Video ({out_res_str}) ကို Telegram ဆီ ပို့ပေးနေပါပြီ...")

            all_ok = True
            for p_idx, part_path in enumerate(parts):
                if mode == "4k":
                    cap_head = f"💎 4K Ultra-HD ({out_res_str}) + မြန်မာစကားပြော ပြီးပါပြီ!\n📥 Source: {src_res_str} ({dl_tier})"
                elif mode == "lossless":
                    cap_head = f"🎞️ Lossless Master ({out_res_str}) + မြန်မာစကားပြော ပြီးပါပြီ!\n📥 Source: {dl_tier}"
                else:
                    cap_head = "✅ မြန်မာစကားပြော + Smart Video Edit ပြီးပါပြီ!"

                cap = cap_head if len(parts) == 1 else f"{cap_head}\n📦 (အပိုင်း {p_idx + 1}/{len(parts)})"
                kb = followup_keyboard(url_key, mode) if p_idx == len(parts) - 1 else None

                data_payload = {
                    "chat_id": str(chat_id),
                    "caption": cap,
                    "reply_to_message_id": str(reply_msg_id),
                    "supports_streaming": "true",
                }
                if out_w and out_h:
                    data_payload["width"] = str(out_w)
                    data_payload["height"] = str(out_h)
                if kb:
                    data_payload["reply_markup"] = json.dumps(kb)

                with open(part_path, "rb") as vf:
                    send_res = await tg_api(
                        "sendVideo",
                        data=data_payload,
                        files={"video": (f"rednote_{mode}_{p_idx+1}.mp4", vf, "video/mp4")},
                        timeout=300.0,
                    )
                if not send_res.get("ok"):
                    all_ok = False
                    await edit_text(chat_id, status_id, f"⚠️ Video ပို့ရာတွင် အခက်အခဲရှိပါသည်: {send_res}")
                    break

            if all_ok:
                STATS["completed"] += 1
                if mode == "4k":
                    STATS["completed_4k"] += 1
                try:
                    await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
                except Exception:
                    pass

        except Exception as e:
            logger.exception("Job processing error")
            await edit_text(chat_id, status_id, f"❌ အမှားအယွင်းတစ်ခု ဖြစ်သွားပါတယ်: {str(e)[:200]}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
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

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip()

FEMALE_EDGE_VOICE = "my-MM-NilarNeural"
MALE_EDGE_VOICE = "my-MM-ThihaNeural"
FEMALE_GEMINI_VOICE = "Kore"
MALE_GEMINI_VOICE = "Puck"
SAMPLE_RATE = 24000
MAX_TG_BYTES = 47 * 1024 * 1024  # 47 MB safe margin under Telegram's 50 MB Bot API limit

processed_updates = deque(maxlen=500)
job_semaphore = asyncio.Semaphore(1)
STATS = {"completed": 0, "started_at": int(time.time())}

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
    "gemini-3.8-flash-lite-tts",
    "gemini-3.8-flash-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]


async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 180.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        if files:
            resp = await client.post(url, data=data or {}, files=files)
        else:
            resp = await client.post(url, json=data or {})
        return resp.json()


async def send_text(chat_id: int, text: str, reply_to: int = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    res = await tg_api("sendMessage", payload)
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str):
    if not message_id:
        return
    try:
        await tg_api("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text})
    except Exception:
        pass


async def register_webhook(base_url: str) -> dict:
    if not BOT_TOKEN or not base_url:
        return {"ok": False, "reason": "Missing BOT_TOKEN or base_url"}
    clean_base = base_url.rstrip("/")
    if clean_base.startswith("http://") and "localhost" not in clean_base and "127.0.0.1" not in clean_base:
        clean_base = "https://" + clean_base[len("http://") :]
    webhook_url = f"{clean_base}/webhook"
    res = await tg_api("setWebhook", {"url": webhook_url, "drop_pending_updates": False})
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
    gem_badge = "GEMINI 3 FLASH + TTS · READY" if gemini_ready else "KEY REQUIRED"
    return f"""<!DOCTYPE html>
<html lang="my">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Autonomous Dubbing Workflow · Rednote ➔ Myanmar</title>
<style>
  :root {{
    --bg: #070913;
    --panel: rgba(16, 22, 42, 0.82);
    --border: rgba(99, 179, 237, 0.25);
    --cyan: #00f2fe;
    --blue: #4facfe;
    --purple: #a855f7;
    --green: #10b981;
    --amber: #f59e0b;
    --pink: #ec4899;
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
  .hero {{
    text-align: center; margin: 18px 0 30px;
  }}
  .hero h1 {{
    font-size: clamp(22px, 4.5vw, 34px); font-weight: 900;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .hero p {{ color: var(--muted); font-size: 14px; }}
  .pipeline {{
    display: flex; flex-direction: column; gap: 0; align-items: center;
  }}
  .node {{
    width: 100%; max-width: 680px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 20px;
    display: flex; align-items: center; gap: 16px;
    position: relative;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
  }}
  .node:hover {{ transform: translateY(-2px); border-color: var(--cyan); }}
  .icon-box {{
    width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    background: linear-gradient(135deg, rgba(79,172,254,0.2), rgba(0,242,254,0.08));
    border: 1px solid rgba(0,242,254,0.35);
    box-shadow: inset 0 0 18px rgba(0,242,254,0.15);
  }}
  .node.purple .icon-box {{
    background: linear-gradient(135deg, rgba(168,85,247,0.24), rgba(236,72,153,0.1));
    border-color: rgba(168,85,247,0.45);
  }}
  .node.amber .icon-box {{
    background: linear-gradient(135deg, rgba(245,158,11,0.24), rgba(239,68,68,0.1));
    border-color: rgba(245,158,11,0.45);
  }}
  .node.green .icon-box {{
    background: linear-gradient(135deg, rgba(16,185,129,0.24), rgba(6,182,212,0.1));
    border-color: rgba(16,185,129,0.45);
  }}
  .n-body {{ flex: 1; min-width: 0; }}
  .n-step {{ font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: var(--cyan); }}
  .node.purple .n-step {{ color: #d8b4fe; }}
  .node.amber .n-step {{ color: #fcd34d; }}
  .node.green .n-step {{ color: #6ee7b7; }}
  .n-title {{ font-size: 16px; font-weight: 800; margin: 3px 0 4px; color: #fff; }}
  .n-desc {{ font-size: 13px; color: var(--muted); line-height: 1.45; }}
  .n-tag {{
    font-size: 11px; font-weight: 700; padding: 5px 10px; border-radius: 8px;
    background: rgba(255,255,255,0.06); color: #e2e8f0; white-space: nowrap;
  }}
  .wire {{
    width: 3px; height: 34px;
    background: rgba(79,172,254,0.22);
    position: relative; overflow: hidden;
  }}
  .wire::after {{
    content: ""; position: absolute; left: 0; top: -18px;
    width: 100%; height: 18px;
    background: linear-gradient(180deg, transparent, var(--cyan), #fff);
    animation: flow 1.3s linear infinite;
  }}
  @keyframes flow {{
    0% {{ top: -18px; }}
    100% {{ top: 36px; }}
  }}
  .cta-bar {{
    margin-top: 30px; text-align: center;
    display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
  }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 13px 24px; border-radius: 999px; font-weight: 800; font-size: 14px;
    text-decoration: none; color: #041019;
    background: linear-gradient(90deg, var(--cyan), var(--blue));
    box-shadow: 0 0 24px rgba(0, 242, 254, 0.4);
  }}
  .stat-card {{
    display: inline-flex; align-items: center; gap: 10px;
    padding: 12px 20px; border-radius: 999px;
    background: var(--panel); border: 1px solid var(--border);
    font-size: 13px; color: var(--text); font-weight: 600;
  }}
  @media (max-width: 560px) {{
    .n-tag {{ display: none; }}
    .node {{ padding: 14px 15px; }}
  }}
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="pulse-dot"></div>
      <div>
        <div class="title">AI VIDEO DUBBING WORKFLOW ENGINE</div>
        <div class="subtitle">24/7 Autonomous Cloud Pipeline · Zero-Touch Automation</div>
      </div>
    </div>
    <div class="badges">
      <span class="pill">{wh_badge}</span>
      <span class="pill ai">{groq_badge}</span>
      <span class="pill ai">{gem_badge}</span>
    </div>
  </div>

  <div class="hero">
    <h1>Rednote ➔ Natural Myanmar Dubbing AI</h1>
    <p>Multi-Agent Audio Timing · Gender-Matched Neural Voice · Lossless Studio Muxer</p>
  </div>

  <div class="pipeline">
    <div class="node">
      <div class="icon-box">📲</div>
      <div class="n-body">
        <div class="n-step">NODE 01 · TRIGGER</div>
        <div class="n-title">Telegram Bot Webhook (@Aihjklbot)</div>
        <div class="n-desc">User ပို့လိုက်သည့် မည်သည့် Rednote (小红书) Link ကိုမဆို ၂၄ နာရီ အလိုအလျောက် လက်ခံဖမ်းယူခြင်း</div>
      </div>
      <div class="n-tag">Webhook Instant</div>
    </div>
    <div class="wire"></div>

    <div class="node">
      <div class="icon-box">📥</div>
      <div class="n-body">
        <div class="n-step">NODE 02 · MEDIA EXTRACTOR</div>
        <div class="n-title">Smart HD Video Stream Downloader</div>
        <div class="n-desc">Watermark မပါသည့် 720p/1080p HD Stream ကို ရွေးချယ်ဆွဲချပြီး မူရင်းအသံလမ်းကြောင်းကို ခွဲထုတ်ခြင်း</div>
      </div>
      <div class="n-tag">CDN + yt-dlp</div>
    </div>
    <div class="wire"></div>

    <div class="node purple">
      <div class="icon-box">🎯</div>
      <div class="n-body">
        <div class="n-step">NODE 03 · TIMING AI AGENT</div>
        <div class="n-title">Groq Whisper Large-v3 Precision ASR</div>
        <div class="n-desc">မူရင်းစကားပြော စတင်ချိန်နှင့် ရပ်နားချိန်ကို 0.01s အတိအကျ အချိန်မှတ်သား၍ စာသားထုတ်ယူခြင်း</div>
      </div>
      <div class="n-tag">Whisper Large-v3</div>
    </div>
    <div class="wire"></div>

    <div class="node purple">
      <div class="icon-box">🧠</div>
      <div class="n-body">
        <div class="n-step">NODE 04 · SCRIPT &amp; GENDER AI AGENT</div>
        <div class="n-title">Google Gemini 3 Flash + Acoustic F0 Pitch</div>
        <div class="n-desc">အသံလှိုင်း (Hz) နှင့် AI နားထောင်မှုပေါင်းစပ်၍ ကျား/မ ခွဲခြားပြီး မြန်မာအပြောစကားစစ်စစ်သို့ အချိန်ကိုက် ပြန်ဆိုခြင်း</div>
      </div>
      <div class="n-tag">Gemini 3 Flash</div>
    </div>
    <div class="wire"></div>

    <div class="node amber">
      <div class="icon-box">🎙️</div>
      <div class="n-body">
        <div class="n-step">NODE 05 · DUAL NEURAL VOICE ENGINE</div>
        <div class="n-title">Dynamic Female &amp; Male Myanmar Voiceover</div>
        <div class="n-desc">မိန်းကလေးနေရာ မိန်းကလေးသံ (👩 Kore/Nilar)၊ ယောက်ျားလေးနေရာ ယောက်ျားလေးသံ (👨 Puck/Thiha) ၃ ဆင့်ခံစနစ်ဖြင့် အသံသွင်းခြင်း</div>
      </div>
      <div class="n-tag">3-Layer Neural TTS</div>
    </div>
    <div class="wire"></div>

    <div class="node green">
      <div class="icon-box">🎬</div>
      <div class="n-body">
        <div class="n-step">NODE 06 · STUDIO MASTERING &amp; DELIVERY</div>
        <div class="n-title">FFmpeg Smart Sync Muxer ➔ Telegram Delivery</div>
        <div class="n-desc">မူရင်းအသံဖျောက်၊ စကားပြောအမြန်နှုန်း ကွက်တိညှိပြီး အရည်အသွေးမကျဘဲ Final MP4 ကို Telegram သို့ ပြန်ပို့ပေးခြင်း</div>
      </div>
      <div class="n-tag">Lossless MP4</div>
    </div>
  </div>

  <div class="cta-bar">
    <a class="btn" href="https://t.me/Aihjklbot" target="_blank">🚀 Open @Aihjklbot on Telegram</a>
    <div class="stat-card">⚡ Processed Videos (Session): {STATS["completed"]}</div>
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
    html = render_workflow_dashboard_html(
        wh_ok=wh_ok,
        groq_ready=bool(GROQ_API_KEY),
        gemini_ready=bool(GEMINI_API_KEY),
    )
    return HTMLResponse(content=html)


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
                "မင်္ဂလာပါ! 🎬⚡\n\n"
                "Rednote (小红书) က ကြိုက်တဲ့ Video Link ကို ဒီထဲ Copy ကူးပြီး ပို့ပေးလိုက်ပါ။\n\n"
                "🤖 Dual-AI Engine (Groq Whisper v3 + Gemini 3 Flash) က အလိုအလျောက် —\n"
                "1️⃣ HD Video ဒေါင်းမယ်\n"
                "2️⃣ စက္ကန့်အတိအကျနဲ့ မြန်မာအပြောစကား ပြန်ရေးမယ်\n"
                "3️⃣ 👩 မိန်းကလေးသံ / 👨 ယောက်ျားလေးသံ ခွဲပြီး အသံသွင်းမယ်\n"
                "4️⃣ မူရင်းအသံဖျောက်ပြီး Final Video ကို ဒီမှာ တန်းပို့ပေးပါမယ်!",
                reply_to=msg_id,
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
    asyncio.create_task(handle_video_job(chat_id, msg_id, raw_url))
    return JSONResponse({"ok": True})


def run_cmd(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


async def download_rednote_video(raw_url: str, out_path: str) -> bool:
    """
    Smart downloader that prefers compact 720p/1080p MP4 streams (<42 MB) over huge raw 4K files
    so Telegram upload is fast and zero CPU re-encoding is needed on free servers.
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
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    expanded_url = raw_url
    direct_video_urls = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for hdrs in (mobile_headers, desktop_headers):
            try:
                r = await client.get(expanded_url, headers=hdrs)
                expanded_url = str(r.url)
                html = r.text

                # 1. Preferog:video (usually pre-compressed mobile-ready H.264 MP4 <25MB)
                og_matches = re.findall(
                    r'<meta[^>]+(?:name|property)=["\']og:video["\'][^>]+content=["\']([^"\']+)["\']',
                    html,
                    flags=re.I,
                )
                og_matches += re.findall(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']og:video["\']',
                    html,
                    flags=re.I,
                )
                for u in og_matches:
                    u_clean = u.replace("\\u002F", "/").replace("&amp;", "&")
                    if u_clean.startswith("http"):
                        direct_video_urls.append(u_clean)

                # 2. Extract masterUrl streams (encoded H264 streams, much smaller than raw originVideoKey)
                master_matches = re.findall(r'"masterUrl"\s*:\s*"([^"]+)"', html)
                for m in master_matches:
                    u_clean = m.encode("utf-8").decode("unicode_escape", errors="ignore").replace("\\u002F", "/")
                    if u_clean.startswith("http"):
                        direct_video_urls.append(u_clean)

                cdn_matches = re.findall(r"https?:(?:\\u002F|/){2}sns-video[^\s\"'<>\\]+", html)
                for c in cdn_matches:
                    direct_video_urls.append(c.replace("\\u002F", "/"))

                # Keep raw originVideoKey last since raw 4K files can exceed 50MB
                origin_keys = re.findall(r'"originVideoKey"\s*:\s*"([^"]+)"', html)
                for key in origin_keys:
                    k_clean = key.encode("utf-8").decode("unicode_escape", errors="ignore").replace("\\u002F", "/")
                    direct_video_urls.append(f"https://sns-video-bd.xhscdn.com/{k_clean}")

                if direct_video_urls:
                    break
            except Exception as e:
                logger.warning(f"HTML scrape attempt warning: {e}")

        seen = set()
        fallback_large_url = None
        for vurl in direct_video_urls:
            if vurl in seen:
                continue
            seen.add(vurl)
            try:
                dl_headers = {
                    "User-Agent": mobile_headers["User-Agent"],
                    "Referer": "https://www.xiaohongshu.com/",
                }
                # Check Content-Length first so we pick a <44MB stream whenever available
                head_r = await client.head(vurl, headers=dl_headers)
                clen = int(head_r.headers.get("content-length", "0") or 0)
                if clen > 44 * 1024 * 1024:
                    if fallback_large_url is None:
                        fallback_large_url = vurl
                    continue

                async with client.stream("GET", vurl, headers=dl_headers) as resp:
                    if resp.status_code in (200, 206):
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 20_000:
                            logger.info(f"Downloaded compact stream ({os.path.getsize(out_path)} B): {vurl[:80]}")
                            return True
            except Exception as e:
                logger.warning(f"Direct CDN download warning: {e}")

    # Next try yt-dlp requesting <=720p under 45MB
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
            "best[height<=720][filesize<45M][ext=mp4]/best[height<=720][ext=mp4]/best[filesize<45M]/best[height<=1080]",
            "--merge-output-format",
            "mp4",
            "-o",
            out_path,
            u,
        ]
        proc = await asyncio.to_thread(run_cmd, cmd, 150)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 20_000:
            logger.info(f"Downloaded via yt-dlp: {u[:80]}")
            return True

    # Last resort: download fallback_large_url if all streams were >44MB (will be split losslessly on delivery)
    if fallback_large_url:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                dl_headers = {
                    "User-Agent": mobile_headers["User-Agent"],
                    "Referer": "https://www.xiaohongshu.com/",
                }
                async with client.stream("GET", fallback_large_url, headers=dl_headers) as resp:
                    if resp.status_code in (200, 206):
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 20_000:
                            return True
        except Exception as e:
            logger.warning(f"Fallback large download failed: {e}")

    return False


def get_media_duration(path: str) -> float:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m:
        h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return h * 3600.0 + mi * 60.0 + s
    return 0.0


def estimate_segment_pitch_hz(wav_samples_8k: np.ndarray, start: float, end: float) -> float:
    sr = 8000
    s_idx = max(0, int(start * sr))
    e_idx = min(len(wav_samples_8k), int(end * sr))
    seg = wav_samples_8k[s_idx:e_idx].astype(np.float32)
    if len(seg) < sr // 4:
        return 0.0
    frame_len = int(0.04 * sr)
    hop = int(0.02 * sr)
    min_lag = int(sr / 300)
    max_lag = int(sr / 75)
    f0s = []
    for idx in range(0, len(seg) - frame_len, hop):
        frame = seg[idx : idx + frame_len]
        frame = frame - np.mean(frame)
        energy = np.mean(frame * frame)
        if energy < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        if max_lag >= len(corr):
            continue
        sub = corr[min_lag:max_lag]
        if len(sub) == 0:
            continue
        peak_idx = int(np.argmax(sub)) + min_lag
        if corr[peak_idx] > 0.32 * corr[0]:
            f0s.append(sr / peak_idx)
    if len(f0s) < 3:
        return 0.0
    return float(np.median(f0s))


MYANMAR_DIGITS = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")


def clean_burmese_for_tts(text: str) -> str:
    t = re.sub(r"[*_#`~<>[\]()]", " ", text)
    t = re.sub(r"[\U00010000-\U0010ffff]", " ", t)
    t = t.translate(MYANMAR_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_speakable_text(text: str) -> bool:
    return bool(re.search(r"[\u1000-\u1049a-zA-Z0-9]", text))


def merge_nearby_segments(segments: list[dict]) -> list[dict]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        combined_dur = seg["end"] - prev["start"]
        if (
            seg["gender"] == prev["gender"]
            and gap < 0.45
            and combined_dur <= 8.5
            and len(prev["burmese"]) + len(seg["burmese"]) <= 220
        ):
            sep = "၊ " if not prev["burmese"].endswith(("။", "၊", "?", "!")) else " "
            prev["burmese"] = f"{prev['burmese']}{sep}{seg['burmese']}"
            prev["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


async def groq_whisper_transcribe(audio_mp3_path: str) -> list[dict]:
    """
    Uses Groq's ultra-accurate Whisper Large-v3 API to obtain exact 0.01s speech timestamps.
    """
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
                    logger.info(f"Groq {w_model} extracted {len(segs)} precise segments")
                    return segs
                else:
                    logger.warning(f"Groq {w_model} HTTP {resp.status_code}: {resp.text[:160]}")
        except Exception as e:
            logger.warning(f"Groq Whisper warning: {e}")
    return []


async def transcribe_and_translate_pipeline(audio_mp3_path: str, total_dur: float, wav_8k: np.ndarray) -> list[dict]:
    whisper_segs = await groq_whisper_transcribe(audio_mp3_path)

    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_mp3_path, "rb") as f:
        audio_bytes = f.read()

    whisper_hint = ""
    if whisper_segs:
        whisper_json = json.dumps(whisper_segs, ensure_ascii=False)
        whisper_hint = f"""
Here are the exact Whisper Large-v3 speech timestamps & transcript for timing reference:
{whisper_json}

Anchor your `start` and `end` timestamps closely to Whisper's exact speech boundaries (you may merge tiny back-to-back fragments spoken by the SAME person into smooth 2–8s sentences, keeping exact start of first fragment and end of last fragment).
"""

    prompt = f"""
You are an elite Myanmar video dubbing director, translator, and voice analyst.
Listen carefully to this {total_dur:.1f}-second audio clip from a short video.
{whisper_hint}
Task:
1. Segment spoken dialogue with accurate `start` and `end` timestamps in seconds (within 0.0 to {total_dur:.1f}). Ignore pure background music or silence without words. If there is no speech at all, return `[]`.
2. For EACH segment, listen to the speaker's actual voice in the audio and label `gender` strictly as `"female"` or `"male"`. Switch `"female"` / `"male"` accurately whenever a different gender speaks!
3. Adapt each segment into **100% natural, smooth colloquial spoken Burmese (မြန်မာအပြောစကားစစ်စစ်)** that fits the exact duration `(end - start)` seconds without rushing:
   - NEVER use stiff literary Burmese ("သည်", "၏", "၌", "၎င်း", "သောကြောင့်", "ဖြစ်ပေသည်").
   - ALWAYS use natural spoken Burmese ("တယ်", "နော်", "ပေါ့", "လေ", "အဲဒါကြောင့်", "ဆိုတော့").
   - Match pronouns/particles naturally to speaker gender (female: "ရှင့်/နော်/ကျွန်မ", male: "ဗျာ/ကျွန်တော်").
   - Write ALL numbers and English words phonetically in Myanmar Unicode words (e.g. "အေအိုင်", "တစ်ရာ", "နှစ်ခု") — NO emojis, NO "..." alone.
   - IMPORTANT PACING RULE: Myanmar speech averages ~4 to 5 syllables per second. Keep each Burmese line concise enough to be spoken comfortably inside `(end - start)` seconds!

Return ONLY valid JSON (no markdown fences):
[
  {{
    "start": 0.3,
    "end": 3.8,
    "gender": "female",
    "burmese": "ဒီနေ့မှာတော့ အရမ်းအသုံးဝင်တဲ့ နည်းလမ်းလေး ပြောပြပေးမယ်နော်။"
  }}
]
""".strip()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
    )

    last_err = None
    raw_json_text = None

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
                logger.info(f"Gemini script adaptation succeeded with: {model_name}")
                break
        except Exception as e:
            last_err = e
            logger.warning(f"Model {model_name} failed: {str(e)[:160]}")

    if raw_json_text is None:
        try:
            avail = await asyncio.to_thread(client.models.list)
            for m in avail:
                mname = getattr(m, "name", "").replace("models/", "")
                if "flash" in mname and "tts" not in mname and "image" not in mname and mname not in PREFERRED_MODELS:
                    try:
                        resp = await asyncio.to_thread(
                            client.models.generate_content,
                            model=mname,
                            contents=[audio_part, prompt],
                            config=cfg,
                        )
                        if resp and resp.text:
                            raw_json_text = resp.text.strip()
                            break
                    except Exception as e2:
                        last_err = e2
        except Exception:
            pass

    if raw_json_text is None:
        raise RuntimeError(f"Gemini API failed: {last_err}")

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_json_text, flags=re.M).strip()
    segments = json.loads(cleaned)
    if not isinstance(segments, list):
        segments = segments.get("segments", []) if isinstance(segments, dict) else []

    valid_segments = []
    for seg in segments:
        try:
            start = max(0.0, float(seg.get("start", 0.0)))
            end = min(total_dur, float(seg.get("end", start + 2.0)))
            if end <= start:
                end = min(total_dur, start + 1.5)
            burmese = clean_burmese_for_tts(str(seg.get("burmese", "")))
            if not burmese or not has_speakable_text(burmese):
                continue

            ai_gender = str(seg.get("gender", "female")).lower().strip()
            pitch_hz = estimate_segment_pitch_hz(wav_8k, start, end)

            if ai_gender in ("male", "m"):
                gender = "male"
                if pitch_hz > 215.0:
                    gender = "female"
            else:
                gender = "female"
                if 75.0 <= pitch_hz <= 145.0:
                    gender = "male"

            valid_segments.append(
                {
                    "start": start,
                    "end": end,
                    "gender": gender,
                    "burmese": burmese,
                }
            )
        except Exception as e:
            logger.warning(f"Skipping malformed segment {seg}: {e}")

    valid_segments.sort(key=lambda x: x["start"])
    return merge_nearby_segments(valid_segments)


async def generate_tts_audio(
    text: str,
    gender: str,
    out_wav_path: str,
    gemini_client: genai.Client,
    state: dict,
) -> bool:
    tmp_raw = out_wav_path + ".raw"

    # Layer 1: Google Gemini Native Neural TTS
    if not state.get("gemini_tts_exhausted"):
        g_voice = FEMALE_GEMINI_VOICE if gender == "female" else MALE_GEMINI_VOICE
        for tts_model in PREFERRED_TTS_MODELS:
            if tts_model in state.get("dead_tts_models", set()):
                continue
            try:
                resp = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=tts_model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=g_voice)
                            )
                        ),
                    ),
                )
                part = resp.candidates[0].content.parts[0].inline_data
                audio_bytes = part.data
                mime = (part.mime_type or "").lower()
                if audio_bytes and len(audio_bytes) > 400:
                    with open(tmp_raw, "wb") as f:
                        f.write(audio_bytes)
                    if audio_bytes[:4] == b"RIFF":
                        run_cmd(
                            [FFMPEG_BIN, "-y", "-i", tmp_raw, "-ac", "1", "-ar", str(SAMPLE_RATE), out_wav_path],
                            30,
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
                                "-ar",
                                str(SAMPLE_RATE),
                                out_wav_path,
                            ],
                            30,
                        )
                    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 400:
                        return True
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini TTS ({tts_model}) warning: {err_str[:160]}")
                if "404" in err_str or "NOT_FOUND" in err_str:
                    state.setdefault("dead_tts_models", set()).add(tts_model)
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    state["gemini_tts_exhausted"] = True
                    break

    # Layer 2: Microsoft Edge Neural TTS
    if not state.get("edge_blocked"):
        try:
            e_voice = FEMALE_EDGE_VOICE if gender == "female" else MALE_EDGE_VOICE
            tmp_mp3 = out_wav_path + ".edge.mp3"
            comm = edge_tts.Communicate(text, e_voice)
            await comm.save(tmp_mp3)
            if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 300:
                run_cmd([FFMPEG_BIN, "-y", "-i", tmp_mp3, "-ac", "1", "-ar", str(SAMPLE_RATE), out_wav_path], 30)
                if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 400:
                    return True
        except Exception as e:
            logger.warning(f"Edge TTS unavailable on this IP, switching to fallback: {e}")
            state["edge_blocked"] = True

    # Layer 3: Google Translate Myanmar TTS (with male pitch transform when needed)
    try:
        chunks = [text[i : i + 170] for i in range(0, len(text), 170)]
        mp3_bytes = bytearray()
        async with httpx.AsyncClient(timeout=20.0) as client:
            for ch in chunks:
                r = await client.get(
                    "https://translate.google.com/translate_tts",
                    params={"ie": "UTF-8", "tl": "my", "client": "tw-ob", "q": ch},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code == 200 and r.content:
                    mp3_bytes.extend(r.content)
        if len(mp3_bytes) > 300:
            tmp_gt = out_wav_path + ".gt.mp3"
            with open(tmp_gt, "wb") as f:
                f.write(mp3_bytes)
            af_extra = []
            if gender == "male":
                af_extra = ["-af", "asetrate=24000*0.84,aresample=24000,atempo=1.19"]
            run_cmd(
                [FFMPEG_BIN, "-y", "-i", tmp_gt, "-ac", "1", "-ar", str(SAMPLE_RATE)] + af_extra + [out_wav_path],
                30,
            )
            if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 400:
                return True
    except Exception as e:
        logger.error(f"Google Translate TTS fallback error: {e}")

    return False


async def synthesize_dubbed_track(segments: list[dict], total_dur: float, work_dir: str, out_audio_path: str):
    total_samples = int(np.ceil((total_dur + 1.0) * SAMPLE_RATE))
    master_pcm = np.zeros(total_samples, dtype=np.float32)
    cursor_sample = 0
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    tts_state: dict = {"dead_tts_models": set()}

    for idx, seg in enumerate(segments):
        if not has_speakable_text(seg["burmese"]):
            continue

        raw_wav = os.path.join(work_dir, f"seg_{idx}_raw.wav")
        trimmed_wav = os.path.join(work_dir, f"seg_{idx}_trim.wav")

        ok = await generate_tts_audio(seg["burmese"], seg["gender"], raw_wav, gemini_client, tts_state)
        if not ok or not os.path.exists(raw_wav):
            continue

        trim_filter = (
            "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-42dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.06:start_threshold=-42dB,"
            "areverse"
        )
        run_cmd(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                raw_wav,
                "-af",
                trim_filter,
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                trimmed_wav,
            ],
            30,
        )
        if not os.path.exists(trimmed_wav) or os.path.getsize(trimmed_wav) < 300:
            trimmed_wav = raw_wav

        tts_dur = get_media_duration(trimmed_wav)
        if tts_dur <= 0.05:
            continue

        start_sec = max(seg["start"], cursor_sample / SAMPLE_RATE)
        if idx + 1 < len(segments):
            next_start = max(start_sec + 0.4, segments[idx + 1]["start"])
            avail_dur = max(0.5, next_start - start_sec - 0.05)
        else:
            avail_dur = max(0.6, total_dur - start_sec)

        final_pcm_path = os.path.join(work_dir, f"seg_{idx}.pcm")
        speed_ratio = tts_dur / avail_dur if avail_dur > 0 else 1.0

        af_chain = []
        if speed_ratio > 1.03:
            clamped_speed = min(1.42, speed_ratio)
            af_chain.append(f"atempo={clamped_speed:.3f}")

        ff_cmd = [FFMPEG_BIN, "-y", "-i", trimmed_wav]
        if af_chain:
            ff_cmd += ["-af", ",".join(af_chain)]
        ff_cmd += ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), final_pcm_path]
        run_cmd(ff_cmd, 30)

        if not os.path.exists(final_pcm_path):
            continue

        seg_pcm = np.fromfile(final_pcm_path, dtype=np.int16).astype(np.float32) / 32768.0
        if len(seg_pcm) == 0:
            continue

        # Normalize individual segment peak so female & male voices have balanced studio loudness
        seg_peak = float(np.max(np.abs(seg_pcm)))
        if seg_peak > 0.02:
            seg_pcm = (seg_pcm / seg_peak) * 0.88

        fade_len = min(int(0.015 * SAMPLE_RATE), len(seg_pcm) // 4)
        if fade_len > 1:
            seg_pcm[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            seg_pcm[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        place_idx = int(start_sec * SAMPLE_RATE)
        end_idx = place_idx + len(seg_pcm)
        if end_idx > len(master_pcm):
            pad = np.zeros(end_idx - len(master_pcm) + SAMPLE_RATE, dtype=np.float32)
            master_pcm = np.concatenate([master_pcm, pad])

        master_pcm[place_idx:end_idx] += seg_pcm
        cursor_sample = end_idx + int(0.04 * SAMPLE_RATE)

    target_len = int(total_dur * SAMPLE_RATE)
    if cursor_sample > target_len:
        target_len = min(cursor_sample, int((total_dur + 1.2) * SAMPLE_RATE))
    master_pcm = master_pcm[:target_len]

    peak = float(np.max(np.abs(master_pcm))) if len(master_pcm) > 0 else 0.0
    if peak > 0.01:
        master_pcm = (master_pcm / peak) * 0.93

    master_s16 = np.clip(master_pcm * 32767.0, -32768, 32767).astype(np.int16)
    raw_master_path = os.path.join(work_dir, "master.pcm")
    master_s16.tofile(raw_master_path)

    run_cmd(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            raw_master_path,
            "-c:a",
            "libmp3lame",
            "-b:a",
            "128k",
            out_audio_path,
        ],
        60,
    )


def split_video_lossless(video_path: str, total_dur: float, work_dir: str) -> list[str]:
    """
    Zero-CPU lossless splitter (-c copy) when a video exceeds Telegram's 50 MB Bot API limit.
    Finishes in <1 second without re-encoding or timing out on 0.1 CPU free servers!
    """
    fsize = os.path.getsize(video_path)
    if fsize <= MAX_TG_BYTES:
        return [video_path]

    num_parts = int(np.ceil(fsize / (42 * 1024 * 1024)))
    part_dur = max(10.0, total_dur / num_parts)
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
        60,
    )
    parts = sorted(
        os.path.join(work_dir, fn)
        for fn in os.listdir(work_dir)
        if fn.startswith("part_") and fn.endswith(".mp4")
    )
    return parts if parts else [video_path]


async def handle_video_job(chat_id: int, reply_msg_id: int, url: str):
    status_id = await send_text(chat_id, "⏳ HD Video ကို စတင်ဒေါင်းလုဒ်ဆွဲနေပါပြီ...", reply_to=reply_msg_id)
    async with job_semaphore:
        work_dir = tempfile.mkdtemp(prefix="xhs_mm_")
        try:
            orig_video = os.path.join(work_dir, "orig.mp4")
            ok = await download_rednote_video(url, orig_video)
            if not ok:
                await edit_text(
                    chat_id,
                    status_id,
                    "❌ ဒီ Link ကနေ Video ဒေါင်းလို့ မရပါဘူး။\n"
                    "Rednote App ထဲက Share ➔ Copy Link (复制链接) နှိပ်ပြီး ရလာတဲ့ Link အသစ်ကို ပြန်ပို့ကြည့်ပေးပါနော်။",
                )
                return

            dur = await asyncio.to_thread(get_media_duration, orig_video)
            if dur <= 0.5:
                await edit_text(chat_id, status_id, "❌ ဒါက Video မဟုတ်ဘဲ ပုံ (Photo post) ဖြစ်နေပါတယ်နော်။")
                return
            if dur > 600:
                await edit_text(chat_id, status_id, "⚠️ Video က ၁၀ မိနစ်ထက် ပိုရှည်နေလို့ Free Server ပေါ်မှာ မနိုင်ပါဘူးနော်။")
                return

            ai_label = "Groq Whisper v3 + Gemini 3" if GROQ_API_KEY else "Gemini 3 AI"
            await edit_text(chat_id, status_id, f"🎧 {ai_label} နဲ့ အချိန်အတိအကျ ဖမ်းယူပြီး မြန်မာလို ပြန်ဆိုနေပါတယ်...")

            audio_mp3 = os.path.join(work_dir, "speech.mp3")
            audio_8k_pcm = os.path.join(work_dir, "speech8k.pcm")

            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-i", orig_video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_mp3],
                60,
            )
            await asyncio.to_thread(
                run_cmd,
                [
                    FFMPEG_BIN,
                    "-y",
                    "-i",
                    orig_video,
                    "-vn",
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    "8000",
                    audio_8k_pcm,
                ],
                60,
            )

            wav_8k = (
                np.fromfile(audio_8k_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if os.path.exists(audio_8k_pcm)
                else np.zeros(8000, dtype=np.float32)
            )

            segments = await transcribe_and_translate_pipeline(audio_mp3, dur, wav_8k)
            if not segments:
                await edit_text(chat_id, status_id, "⚠️ ဒီ Video ထဲမှာ စကားပြောသံ မပါပါဘူး (သီချင်းသံသက်သက် ဖြစ်နေပါတယ်)။")
                return

            f_count = sum(1 for s in segments if s["gender"] == "female")
            m_count = sum(1 for s in segments if s["gender"] == "male")
            await edit_text(
                chat_id,
                status_id,
                f"🎙️ မြန်မာအသံသွင်းပြီး Studio Timing ညှိနေပါပြီ...\n"
                f"(👩 မိန်းကလေးသံ {f_count} ပိုင်း / 👨 ယောက်ျားလေးသံ {m_count} ပိုင်း)",
            )

            dub_mp3 = os.path.join(work_dir, "burmese_dub.mp3")
            await synthesize_dubbed_track(segments, dur, work_dir, dub_mp3)

            final_video = os.path.join(work_dir, "final_mm.mp4")
            await asyncio.to_thread(
                run_cmd,
                [
                    FFMPEG_BIN,
                    "-y",
                    "-i",
                    orig_video,
                    "-i",
                    dub_mp3,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    final_video,
                ],
                90,
            )

            # Remove orig_video immediately to free /tmp RAM disk space before upload/splitting
            if os.path.exists(orig_video):
                os.remove(orig_video)

            parts = await asyncio.to_thread(split_video_lossless, final_video, dur, work_dir)
            await edit_text(chat_id, status_id, "📤 Final Video ကို Telegram ဆီ ပို့ပေးနေပါပြီ...")

            all_ok = True
            for p_idx, part_path in enumerate(parts):
                cap = (
                    "✅ မြန်မာစကားပြော ပြောင်းပြီးပါပြီ!"
                    if len(parts) == 1
                    else f"✅ မြန်မာစကားပြော Video (အပိုင်း {p_idx + 1}/{len(parts)})"
                )
                with open(part_path, "rb") as vf:
                    send_res = await tg_api(
                        "sendVideo",
                        data={
                            "chat_id": str(chat_id),
                            "caption": cap,
                            "reply_to_message_id": str(reply_msg_id),
                            "supports_streaming": "true",
                        },
                        files={"video": (f"rednote_myanmar_{p_idx+1}.mp4", vf, "video/mp4")},
                        timeout=240.0,
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

        except Exception as e:
            logger.exception("Job processing error")
            await edit_text(chat_id, status_id, f"❌ အမှားအယွင်းတစ်ခု ဖြစ်သွားပါတယ်: {str(e)[:200]}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

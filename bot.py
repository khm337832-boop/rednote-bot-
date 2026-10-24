import asyncio
import gc
import json
import logging
import os
import re
import shutil
import signal
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
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else None

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip()

CHARON_VOICE = "Charon"
FALLBACK_THIHA_VOICE = "my-MM-ThihaNeural"
SAMPLE_RATE = 24000
MAX_TG_BYTES = 47 * 1024 * 1024
MAX_VIDEO_SECONDS = 3600  # Supports long videos up to 60 minutes!

processed_updates = deque(maxlen=600)
job_semaphore = asyncio.Semaphore(1)
ACTIVE_JOBS = 0
STATS = {"completed_orig": 0, "completed_4k": 0, "started_at": int(time.time())}

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


def run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
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


async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 240.0):
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


async def active_job_keepalive_loop():
    """Prevents Render Free Tier from spinning down mid-job on long (>8–60 min) videos!"""
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
    if RENDER_EXTERNAL_URL and BOT_TOKEN:
        try:
            await register_webhook(RENDER_EXTERNAL_URL)
        except Exception as e:
            logger.error(f"Startup webhook error: {e}")
    yield
    ka_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return JSONResponse({"ok": True, "active_jobs": ACTIVE_JOBS})


def render_workflow_dashboard_html(wh_ok: bool, groq_ready: bool, gemini_ready: bool) -> str:
    wh_badge = "ACTIVE · WEBHOOK READY" if wh_ok else "CONNECTING..."
    groq_badge = "GROQ WHISPER-V3 · READY" if groq_ready else "GEMINI AUDIO MODE"
    gem_badge = "CHARON AI VOICE · READY" if gemini_ready else "KEY REQUIRED"
    return f"""<!DOCTYPE html>
<html lang="my">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Smart Recap ➔ 4K Ultra-HD Pipeline (v5) · @Aihjklbot</title>
<style>
  :root {{
    --bg: #070913;
    --panel: rgba(16, 22, 42, 0.86);
    --border: rgba(99, 179, 237, 0.25);
    --cyan: #00f2fe;
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
    padding: 24px 16px 56px;
  }}
  .wrap {{ max-width: 960px; margin: 0 auto; }}
  .topbar {{
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; padding: 16px 20px; border-radius: 18px;
    background: var(--panel); border: 1px solid var(--border);
    margin-bottom: 24px;
  }}
  .brand {{ display: flex; align-items: center; gap: 12px; }}
  .pulse-dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--green); box-shadow: 0 0 14px var(--green); }}
  .title {{ font-size: 18px; font-weight: 800; }}
  .subtitle {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
  .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .pill {{
    font-size: 11px; font-weight: 700; padding: 6px 11px; border-radius: 999px;
    background: rgba(16, 185, 129, 0.14); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35);
  }}
  .pill.uhd {{ background: rgba(251, 191, 36, 0.16); color: #fde68a; border-color: rgba(251, 191, 36, 0.45); }}
  .pill.ai {{ background: rgba(168, 85, 247, 0.14); color: #d8b4fe; border-color: rgba(168, 85, 247, 0.38); }}
  .hero {{ text-align: center; margin: 18px 0 28px; }}
  .hero h1 {{
    font-size: clamp(22px, 4.5vw, 34px); font-weight: 900;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #fbbf24, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .hero p {{ color: var(--muted); font-size: 14px; }}
  .pipeline {{ display: flex; flex-direction: column; align-items: center; }}
  .node {{
    width: 100%; max-width: 680px; background: var(--panel); border: 1px solid var(--border);
    border-radius: 18px; padding: 16px 20px; display: flex; align-items: center; gap: 16px;
  }}
  .icon-box {{
    width: 50px; height: 50px; border-radius: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 23px;
    background: rgba(79,172,254,0.16); border: 1px solid rgba(0,242,254,0.35);
  }}
  .node.gold .icon-box {{ background: rgba(251,191,36,0.18); border-color: rgba(251,191,36,0.5); }}
  .n-step {{ font-size: 10px; font-weight: 800; color: var(--cyan); letter-spacing: 1px; }}
  .node.gold .n-step {{ color: var(--gold); }}
  .n-title {{ font-size: 16px; font-weight: 800; margin: 3px 0; color: #fff; }}
  .n-desc {{ font-size: 13px; color: var(--muted); line-height: 1.45; }}
  .wire {{ width: 3px; height: 26px; background: linear-gradient(180deg, var(--cyan), var(--gold)); }}
  .cta-bar {{ margin-top: 28px; text-align: center; display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px; padding: 13px 24px; border-radius: 999px;
    font-weight: 800; font-size: 14px; text-decoration: none; color: #041019;
    background: linear-gradient(90deg, var(--cyan), var(--gold));
  }}
  .stat-card {{
    padding: 12px 20px; border-radius: 999px; background: var(--panel); border: 1px solid var(--border);
    font-size: 13px; font-weight: 600;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="pulse-dot"></div>
      <div>
        <div class="title">AI SMART RECAP ➔ 4K ULTRA-HD ENGINE (v5)</div>
        <div class="subtitle">Long Video Support (Up to 60m) · Original Recap First ➔ 4K Final Upgrade</div>
      </div>
    </div>
    <div class="badges">
      <span class="pill">{wh_badge}</span>
      <span class="pill uhd">60-MIN + 4K PIPELINE · READY</span>
      <span class="pill ai">{groq_badge}</span>
      <span class="pill ai">{gem_badge}</span>
    </div>
  </div>
  <div class="hero">
    <h1>Rednote Best Download ➔ Original Recap ➔ 4K Final</h1>
    <p>အကြည်ဆုံး Version ဒေါင်းမည် ➔ မူလအတိုင်း Smart Recap အရင်ထုတ်ပို့မည် ➔ ထို Video ကို 4K ပြောင်းပို့မည်</p>
  </div>
  <div class="pipeline">
    <div class="node">
      <div class="icon-box">📥</div>
      <div>
        <div class="n-step">STEP 01 · HIGHEST QUALITY DOWNLOAD</div>
        <div class="n-title">originVideoKey Raw Master (No-Watermark · Up to 60 Mins)</div>
        <div class="n-desc">Rednote မှ အကြည်ဆုံး မူရင်း Video ကို ဆွဲထုတ်ခြင်း (၈ မိနစ် ကန့်သတ်ချက် မရှိတော့ပါ — ၁ နာရီအထိ ရသည်)</div>
      </div>
    </div>
    <div class="wire"></div>
    <div class="node">
      <div class="icon-box">✂️</div>
      <div>
        <div class="n-step">STEP 02 · ORIGINAL SMART RECAP FIRST</div>
        <div class="n-title">Charon 1.0x Male Voice + BGM Ducking + Smart Scene Cut</div>
        <div class="n-desc">မူလအတိုင်း အသံသြသြ၊ နောက်ခံ BGM သံနှင့် ဖြတ်/ဆက်လုပ်၍ မူလ Smart Recap Video ကို အရင်ဆုံး ထုတ်ပေးခြင်း</div>
      </div>
    </div>
    <div class="wire"></div>
    <div class="node gold">
      <div class="icon-box">💎</div>
      <div>
        <div class="n-step">STEP 03 · 4K ULTRA-HD POST-UPGRADE</div>
        <div class="n-title">Finished Recap Video ➔ True 4K Ultra-HD (2160×3840)</div>
        <div class="n-desc">မူလအတိုင်း ထွက်လာပြီးသား Recap Video ကိုမှ Edge-Sharpened 4K Ultra-HD သို့ အဆင့်မြှင့်ပြောင်းလဲပြီး ပို့ပေးခြင်း</div>
      </div>
    </div>
  </div>
  <div class="cta-bar">
    <a class="btn" href="https://t.me/Aihjklbot" target="_blank">🚀 Open @Aihjklbot on Telegram</a>
    <div class="stat-card">✅ Recap Sent: {STATS["completed_orig"]} · 💎 4K Upgraded: {STATS["completed_4k"]}</div>
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

    cb = update.get("callback_query")
    if cb:
        cb_id = cb.get("id")
        if cb_id:
            asyncio.create_task(tg_api("answerCallbackQuery", {"callback_query_id": cb_id}))
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
                "Rednote (小红书) Video Link ပို့ပေးလိုက်ပါ! (Video အရှည် ၁ နာရီအထိ ရပါပြီ!)\n\n"
                "✨ v5 အလိုအလျောက် လုပ်ဆောင်ပုံ —\n"
                "1️⃣ Rednote မှ အကြည်ဆုံး မူရင်း Version (No-Watermark) ကို ဒေါင်းမယ်\n"
                "2️⃣ မူလအတိုင်း Charon ယောက်ျားလေးအသံ + BGM + Smart ဖြတ်/ဆက်နဲ့ Video ကို အရင်ထုတ်ပြီး ပို့ပေးမယ်\n"
                "3️⃣ ပြီးတာနဲ့ အဲ့ဒီထွက်လာတဲ့ Video ကိုပဲ 4K Ultra-HD (2160×3840) ဆက်ပြောင်းပြီး ထပ်ပို့ပေးပါမယ်!",
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


async def download_rednote_video(raw_url: str, out_path: str) -> tuple[bool, str]:
    """Downloads the sharpest unwatermarked version from Rednote (`originVideoKey` -> largest `masterUrl` -> uncapped `yt-dlp`)."""
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

                for key in re.findall(r'"originVideoKey"\s*:\s*"([^"]+)"', html):
                    k = _clean(key).lstrip("/")
                    if k:
                        for host in (
                            "https://sns-video-bd.xhscdn.com",
                            "https://sns-video-hw.xhscdn.com",
                            "https://sns-video-al.xhscdn.com",
                        ):
                            origin_urls.append(f"{host}/{k}")

                for m in re.findall(r'"masterUrl"\s*:\s*"([^"]+)"', html):
                    u_c = _clean(m)
                    if u_c.startswith("http"):
                        master_urls.append(u_c)

                for c in re.findall(r"https?:(?:\\u002F|/){2}sns-video[^\s\"'<>\\]+", html):
                    u_c = _clean(c)
                    if u_c.startswith("http"):
                        master_urls.append(u_c)

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
                async with client.stream("GET", vurl, headers=dl_headers, timeout=300.0) as resp:
                    if resp.status_code in (200, 206):
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=131072):
                                f.write(chunk)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 25_000:
                            return True
            except Exception as e:
                logger.warning(f"Stream download warning: {e}")
            return False

        for _, best_u in await _probe_and_rank(origin_urls):
            if await _stream_to_disk(best_u):
                return True, "Raw Master (No-Watermark)"

        for _, best_u in await _probe_and_rank(master_urls):
            if await _stream_to_disk(best_u):
                return True, "High-Bitrate Master"

        for _, best_u in await _probe_and_rank(og_urls):
            if await _stream_to_disk(best_u):
                return True, "HD Direct"

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
            "--socket-timeout",
            "20",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            out_path,
            u,
        ]
        await asyncio.to_thread(run_cmd, cmd, 300)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 25_000:
            return True, "Max Quality (yt-dlp)"

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

    rot_m = re.search(r"rotation of\s*(-?\d+)|rotate\s*:\s*(-?\d+)", out, flags=re.I)
    if rot_m:
        ang = abs(int(rot_m.group(1) or rot_m.group(2) or 0)) % 180
        if ang == 90 and w > 0 and h > 0:
            w, h = h, w

    return dur, w, h, vcodec


def target_4k_dims(src_w: int, src_h: int) -> tuple[int, int]:
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
    if not GROQ_API_KEY or not os.path.exists(audio_mp3_path):
        return []
    if os.path.getsize(audio_mp3_path) > 24 * 1024 * 1024:
        return []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for w_model in ("whisper-large-v3-turbo", "whisper-large-v3"):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
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

    # Scale scene blocks smoothly for both short (<2m) and long (8–60m) videos
    if total_dur <= 300:
        min_sc, max_sc = 4, 14
    else:
        min_sc = max(10, min(18, int(total_dur // 45)))
        max_sc = max(min_sc + 4, min(28, int(total_dur // 25)))

    whisper_hint = ""
    if whisper_segs:
        # Cap hint length cleanly for very long transcripts
        whisper_json = json.dumps(whisper_segs[:220], ensure_ascii=False)
        whisper_hint = f"\nExact Whisper Large-v3 speech timestamps & transcript:\n{whisper_json}\n"

    prompt = f"""
You are a top Myanmar male video narrator / recap scriptwriter.
Listen to this {total_dur:.1f}-second video audio.
{whisper_hint}
Task:
1. Group the spoken content into **{min_sc} to {max_sc} smooth, coherent scene blocks** (`start` and `end` timestamps in seconds within `0.0` to `{total_dur:.1f}`).
   - Merge tiny choppy 1-second fragments into natural 4 to 10 second sentences covering the full story from beginning to end!
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

    contents_payload = [prompt]
    if len(audio_bytes) <= 19 * 1024 * 1024:
        contents_payload.insert(0, types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"))
    del audio_bytes

    cfg = types.GenerateContentConfig(temperature=0.25, response_mime_type="application/json")

    raw_json_text = None
    last_err = None
    for model_name in PREFERRED_MODELS:
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents_payload,
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


async def _synthesize_charon_batch(batch_scenes: list[dict], batch_idx: int, work_dir: str) -> list[np.ndarray] | None:
    client = genai.Client(api_key=GEMINI_API_KEY)
    joined_script = "\n\n".join(s["burmese"] for s in batch_scenes)

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
                tmp_raw = os.path.join(work_dir, f"charon_raw_{batch_idx}.bin")
                tmp_pcm = os.path.join(work_dir, f"charon_24k_{batch_idx}.pcm")
                with open(tmp_raw, "wb") as f:
                    f.write(audio_bytes)

                if audio_bytes[:4] == b"RIFF":
                    run_cmd(
                        [FFMPEG_BIN, "-y", "-i", tmp_raw, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), tmp_pcm],
                        60,
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
                        60,
                    )
                if os.path.exists(tmp_raw):
                    os.remove(tmp_raw)
                if os.path.exists(tmp_pcm) and os.path.getsize(tmp_pcm) > 1000:
                    master_pcm = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    os.remove(tmp_pcm)
                    logger.info(f"Charon batch {batch_idx} via {tts_model} ({len(master_pcm)/SAMPLE_RATE:.1f}s)")
                    return split_master_pcm_into_n_clips(master_pcm, batch_scenes)
        except Exception as e:
            logger.warning(f"Charon TTS ({tts_model}) warning: {str(e)[:160]}")
    return None


async def synthesize_charon_clips(scenes: list[dict], work_dir: str) -> list[np.ndarray]:
    """
    Uses 1 single Charon call for standard videos (<=12 scenes), and cleanly batches by 12 scenes
    for long (8–60 min) videos so Gemini TTS never hits single-call audio duration limits!
    """
    all_clips: list[np.ndarray] = []
    batch_size = 12
    for b_i in range(0, len(scenes), batch_size):
        batch = scenes[b_i : b_i + batch_size]
        b_clips = await _synthesize_charon_batch(batch, b_i // batch_size, work_dir)
        if b_clips is not None and len(b_clips) == len(batch):
            all_clips.extend(b_clips)
            continue

        # Fallback per scene if daily Gemini TTS quota is exhausted
        for idx, sc in enumerate(batch):
            mp3_p = os.path.join(work_dir, f"thiha_{b_i + idx}.mp3")
            pcm_p = os.path.join(work_dir, f"thiha_{b_i + idx}.pcm")
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
                os.remove(mp3_p)
            if os.path.exists(pcm_p) and os.path.getsize(pcm_p) > 200:
                seg = np.fromfile(pcm_p, dtype=np.int16).astype(np.float32) / 32768.0
                os.remove(pcm_p)
                all_clips.append(trim_pcm_silence(seg))
            else:
                all_clips.append(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))
    return all_clips


def build_recap_video_and_bgm_mix(
    orig_video: str,
    orig_dur: float,
    scenes: list[dict],
    voice_clips: list[np.ndarray],
    work_dir: str,
    final_video_path: str,
):
    """
    STAGE 1 — Exact Original Smart Recap Video Builder (Memory-Safe even for 60-minute videos!):
    - Streams vocal-suppressed BGM slices directly from disk (`bgm_bed.pcm` seek) using <5MB RAM!
    - Keeps Charon 1.0x natural speed + vocal-suppressed original BGM auto-ducking.
    - Sequentially cuts/syncs scenes into clean HD Smart Recap MP4 (`recap_orig.mp4`).
    """
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
        300,
    )

    has_bgm = os.path.exists(bgm_pcm_path) and os.path.getsize(bgm_pcm_path) > 400
    total_bgm_samples = (os.path.getsize(bgm_pcm_path) // 2) if has_bgm else 0

    # Find BGM peak in small 1MB streaming chunks so even 60-min BGM uses <2MB RAM
    bgm_gain = 0.55
    if has_bgm:
        peak_i16 = 1
        with open(bgm_pcm_path, "rb") as bf:
            while True:
                buf = bf.read(524288)
                if not buf:
                    break
                arr = np.frombuffer(buf, dtype=np.int16)
                if len(arr):
                    p = int(np.max(np.abs(arr.astype(np.int32))))
                    if p > peak_i16:
                        peak_i16 = p
        if peak_i16 > 300:
            bgm_gain = 0.55 / (peak_i16 / 32768.0)

    def read_bgm_slice(s_sec: float, e_sec: float, target_samples: int, vol: float) -> np.ndarray:
        if not has_bgm or total_bgm_samples < 10 or target_samples <= 0:
            return np.zeros(max(0, target_samples), dtype=np.float32)
        s_i = max(0, min(total_bgm_samples - 1, int(s_sec * SAMPLE_RATE)))
        e_i = max(s_i + 1, min(total_bgm_samples, int(e_sec * SAMPLE_RATE)))
        n_read = e_i - s_i
        with open(bgm_pcm_path, "rb") as bf:
            bf.seek(s_i * 2)
            raw_bytes = bf.read(n_read * 2)
        if not raw_bytes:
            return np.zeros(target_samples, dtype=np.float32)
        raw = (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0) * bgm_gain
        if len(raw) != target_samples:
            x_old = np.linspace(0.0, 1.0, len(raw), dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, target_samples, dtype=np.float32)
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        fade = min(int(0.03 * SAMPLE_RATE), len(raw) // 4)
        if fade > 1:
            raw[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            raw[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return raw * vol

    video_slices: list[tuple[float, float, float]] = []
    raw_pcm_out = os.path.join(work_dir, "master_mix.pcm")

    with open(raw_pcm_out, "wb") as pcm_f:
        first_start = scenes[0]["start"]
        if first_start > 0.25:
            intro_dur = min(0.8, first_start)
            video_slices.append((0.0, first_start if first_start <= 1.2 else intro_dur, intro_dur))
            intro_arr = read_bgm_slice(0.0, intro_dur, int(intro_dur * SAMPLE_RATE), vol=0.55)
            np.clip(intro_arr * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)

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

            bgm_seg = read_bgm_slice(src_s, cut_src_e, target_samples, vol=1.0)
            env = np.full(target_samples, 0.28, dtype=np.float32)
            v_len = min(target_samples, len(norm_voice))
            env[:v_len] = 0.16
            ramp = min(int(0.05 * SAMPLE_RATE), target_samples - v_len)
            if ramp > 2 and v_len + ramp <= target_samples:
                env[v_len : v_len + ramp] = np.linspace(0.16, 0.36, ramp, dtype=np.float32)
                env[v_len + ramp :] = 0.36

            mixed = bgm_seg * env
            mixed[:v_len] += norm_voice[:v_len]
            np.clip(mixed * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)

        last_end = scenes[-1]["end"]
        if orig_dur - last_end > 0.25:
            outro_dur = min(0.65, orig_dur - last_end)
            video_slices.append((last_end, last_end + outro_dur, outro_dur))
            outro_arr = read_bgm_slice(last_end, last_end + outro_dur, int(outro_dur * SAMPLE_RATE), vol=0.45)
            np.clip(outro_arr * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)

    if os.path.exists(bgm_pcm_path):
        os.remove(bgm_pcm_path)

    master_wav = os.path.join(work_dir, "master_mix.wav")
    run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", raw_pcm_out, master_wav], 60)
    if os.path.exists(raw_pcm_out):
        os.remove(raw_pcm_out)
    gc.collect()

    # Cut & time-fit each scene slice cleanly (preserving crisp 720p HD detail for both viewing & Stage 2 4K conversion!)
    seg_files = []
    for idx, (s_t, e_t, tgt_d) in enumerate(video_slices):
        raw_d = max(0.1, e_t - s_t)
        pts_factor = tgt_d / raw_d
        seg_path = os.path.join(work_dir, f"seg_{idx:03d}.mp4")
        vf_str = f"setpts={pts_factor:.4f}*(PTS-STARTPTS),scale='min(720,iw)':-2,fps=24,format=yuv420p"
        run_cmd(
            [
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
                "-video_track_timescale",
                "90000",
                "-threads",
                "1",
                seg_path,
            ],
            180,
        )
        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 500:
            seg_files.append(seg_path)

    if not seg_files:
        raise RuntimeError("No video slices were generated")

    if os.path.exists(orig_video):
        os.remove(orig_video)

    concat_list_path = os.path.join(work_dir, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as cf:
        for tp in seg_files:
            cf.write(f"file '{tp}'\n")

    proc = run_cmd(
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
            "160k",
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
    if os.path.exists(master_wav):
        os.remove(master_wav)

    if not os.path.exists(final_video_path) or os.path.getsize(final_video_path) < 5000:
        err_tail = proc.stderr.decode(errors="ignore")[-300:]
        raise RuntimeError(f"FFmpeg video concat failed: {err_tail}")


async def convert_finished_recap_to_4k(
    recap_video_path: str,
    recap_dur: float,
    src_w: int,
    src_h: int,
    work_dir: str,
    out_4k_path: str,
    progress_cb=None,
):
    """
    STAGE 2 — Takes the ALREADY-FINISHED Original Smart Recap Video (`recap_video_path`)
    and upgrades it to Edge-Sharpened True 4K Ultra-HD (2160x3840 / 3840x2160) in safe 10s chunks (<275MB RAM)!
    """
    tw, th = target_4k_dims(src_w, src_h)
    chunk_sec = 10.0
    total_chunks = max(1, int(np.ceil(recap_dur / chunk_sec)))
    chunk_files = []

    vf_4k = f"unsharp=3:3:0.75:3:3:0.0,scale={tw}:{th}:flags=fast_bilinear,fps=24,format=yuv420p"

    for idx in range(total_chunks):
        st = idx * chunk_sec
        dur_c = min(chunk_sec, max(0.2, recap_dur - st + 0.1))
        if st >= recap_dur:
            break

        if progress_cb:
            await progress_cb(idx + 1, total_chunks, tw, th)

        c_path = os.path.join(work_dir, f"uhd_{idx:03d}.mp4")
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss",
            f"{st:.3f}",
            "-t",
            f"{dur_c:.3f}",
            "-i",
            recap_video_path,
            "-an",
            "-vf",
            vf_4k,
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
            c_path,
        ]
        await asyncio.to_thread(run_cmd, cmd, 600)
        if os.path.exists(c_path) and os.path.getsize(c_path) > 500:
            chunk_files.append(c_path)

    if not chunk_files:
        raise RuntimeError("4K conversion produced no chunks")

    uhd_list = os.path.join(work_dir, "uhd_concat.txt")
    with open(uhd_list, "w", encoding="utf-8") as uf:
        for cp in chunk_files:
            uf.write(f"file '{cp}'\n")

    # Mux lossless 4K video chunks with exact audio track from finished recap_video_path
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
            uhd_list,
            "-i",
            recap_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-movflags",
            "+faststart",
            out_4k_path,
        ],
        180,
    )

    for cp in chunk_files:
        if os.path.exists(cp):
            os.remove(cp)
    if os.path.exists(recap_video_path):
        os.remove(recap_video_path)

    if not os.path.exists(out_4k_path) or os.path.getsize(out_4k_path) < 5000:
        err_tail = proc.stderr.decode(errors="ignore")[-300:]
        raise RuntimeError(f"4K final mux failed: {err_tail}")


def split_video_lossless(video_path: str, total_dur: float, work_dir: str, prefix: str = "part_") -> list[str]:
    fsize = os.path.getsize(video_path)
    if fsize <= MAX_TG_BYTES:
        return [video_path]

    num_parts = max(2, int(np.ceil(fsize / (41 * 1024 * 1024))))
    part_dur = max(6.0, total_dur / num_parts)
    out_pattern = os.path.join(work_dir, f"{prefix}%02d.mp4")

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
        120,
    )
    parts = sorted(
        os.path.join(work_dir, fn)
        for fn in os.listdir(work_dir)
        if fn.startswith(prefix) and fn.endswith(".mp4") and os.path.getsize(os.path.join(work_dir, fn)) > 1000
    )
    return parts if parts else [video_path]


async def upload_video_parts(
    chat_id: int,
    reply_msg_id: int,
    parts: list[str],
    caption_prefix: str,
    file_prefix: str,
    width: int = 0,
    height: int = 0,
) -> bool:
    for p_idx, part_path in enumerate(parts):
        cap = caption_prefix if len(parts) == 1 else f"{caption_prefix}\n📦 (အပိုင်း {p_idx + 1}/{len(parts)})"
        data_payload = {
            "chat_id": str(chat_id),
            "caption": cap,
            "reply_to_message_id": str(reply_msg_id),
            "supports_streaming": "true",
        }
        if width and height:
            data_payload["width"] = str(width)
            data_payload["height"] = str(height)
        with open(part_path, "rb") as vf:
            send_res = await tg_api(
                "sendVideo",
                data=data_payload,
                files={"video": (f"{file_prefix}_{p_idx + 1}.mp4", vf, "video/mp4")},
                timeout=360.0,
            )
        if not send_res.get("ok"):
            return False
    return True


async def handle_video_job(chat_id: int, reply_msg_id: int, url: str):
    global ACTIVE_JOBS
    status_id = await send_text(
        chat_id,
        "⏳ Rednote မှ အကြည်ဆုံး မူရင်း Video ကို စတင်ဒေါင်းလုဒ်ဆွဲနေပါပြီ...",
        reply_to=reply_msg_id,
    )
    ACTIVE_JOBS += 1
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

            dur, src_w, src_h, _ = await asyncio.to_thread(probe_video_info, orig_video)
            if dur <= 0.5:
                await edit_text(chat_id, status_id, "❌ ဒါက Video မဟုတ်ဘဲ ပုံ (Photo post) ဖြစ်နေပါတယ်နော်။")
                return
            if dur > MAX_VIDEO_SECONDS:
                await edit_text(chat_id, status_id, "⚠️ Video က မိနစ် (၆၀) ထက် ပိုရှည်နေပါတယ်နော်။")
                return

            dur_min_str = f"{int(dur // 60)}m {int(dur % 60)}s" if dur >= 60 else f"{dur:.0f}s"
            src_res_str = f"{src_w}×{src_h}" if src_w and src_h else "HD"

            await edit_text(
                chat_id,
                status_id,
                f"📥 အကြည်ဆုံး Source ({src_res_str} · {dur_min_str} · {dl_tier}) ရပါပြီ!\n"
                "🎧 Whisper v3 + Gemini 3 နဲ့ မြန်မာစကားပြော Script ရေးနေပါတယ်...",
            )

            # 24kbps 16kHz mono speech extraction -> even a 45-min video is only ~7.8 MB!
            audio_mp3 = os.path.join(work_dir, "speech.mp3")
            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-i", orig_video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k", audio_mp3],
                180,
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

            # =========================================================================
            # STAGE 1: Produce & Send Exact Original Smart Recap Video FIRST!
            # =========================================================================
            await edit_text(
                chat_id,
                status_id,
                "✂️ [အဆင့် ၁/၂] မူလအတိုင်း Video အပိုင်းတွေကို ဖြတ်/ဆက်ပြီး နောက်ခံ BGM သံ ညှိနေပါပြီ...",
            )
            recap_video = os.path.join(work_dir, "recap_orig.mp4")
            await asyncio.to_thread(
                build_recap_video_and_bgm_mix,
                orig_video,
                dur,
                scenes,
                voice_clips,
                work_dir,
                recap_video,
            )
            del voice_clips
            gc.collect()

            recap_dur, rec_w, rec_h, _ = await asyncio.to_thread(probe_video_info, recap_video)
            recap_parts = await asyncio.to_thread(split_video_lossless, recap_video, recap_dur or dur, work_dir, "origpart_")

            await edit_text(chat_id, status_id, "📤 [အဆင့် ၁/၂] မူလ Smart Recap Video ကို အရင်ပို့ပေးနေပါပြီ...")
            sent_orig = await upload_video_parts(
                chat_id,
                reply_msg_id,
                recap_parts,
                caption_prefix="✅ မူလ Smart Recap + မြန်မာစကားပြော Video ပြီးပါပြီ!\n(💎 4K Ultra-HD Version ကိုလည်း ဆက်ပြောင်းပြီး ထပ်ပို့ပေးနေပါတယ်...)",
                file_prefix="rednote_recap",
                width=rec_w,
                height=rec_h,
            )
            for rp in recap_parts:
                if rp != recap_video and os.path.exists(rp):
                    os.remove(rp)

            if sent_orig:
                STATS["completed_orig"] += 1

            # =========================================================================
            # STAGE 2: Convert THAT Finished Recap Video into True 4K Ultra-HD!
            # =========================================================================
            final_4k = os.path.join(work_dir, "final_4k.mp4")

            async def _on_4k_prog(cur_c: int, tot_c: int, w4: int, h4: int):
                pct = int(round((cur_c - 1) * 100 / max(1, tot_c)))
                await edit_text(
                    chat_id,
                    status_id,
                    f"💎 [အဆင့် ၂/၂] ထွက်လာတဲ့ မူလ Video ကို 4K Ultra-HD ({w4}×{h4}) ပြောင်းနေပါပြီ...\n"
                    f"📊 ပြီးစီးမှု — {pct}% (အပိုင်း {cur_c}/{tot_c})",
                )

            await convert_finished_recap_to_4k(
                recap_video,
                recap_dur or dur,
                rec_w or src_w,
                rec_h or src_h,
                work_dir,
                final_4k,
                progress_cb=_on_4k_prog,
            )

            dur_4k, out_w4, out_h4, _ = await asyncio.to_thread(probe_video_info, final_4k)
            parts_4k = await asyncio.to_thread(split_video_lossless, final_4k, dur_4k or recap_dur, work_dir, "uhdpart_")
            if len(parts_4k) > 1 and os.path.exists(final_4k):
                os.remove(final_4k)

            await edit_text(chat_id, status_id, f"📤 [အဆင့် ၂/၂] 4K Ultra-HD ({out_w4}×{out_h4}) Final Video ကို ပို့ပေးနေပါပြီ...")
            sent_4k = await upload_video_parts(
                chat_id,
                reply_msg_id,
                parts_4k,
                caption_prefix=f"💎 4K Ultra-HD ({out_w4}×{out_h4}) Final Video ပြီးပါပြီ!\n📥 Source: {src_res_str} ({dl_tier})",
                file_prefix="rednote_4k_uhd",
                width=out_w4,
                height=out_h4,
            )

            if sent_4k:
                STATS["completed_4k"] += 1
                try:
                    await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
                except Exception:
                    pass
            else:
                await edit_text(chat_id, status_id, "⚠️ 4K Video ပို့ရာတွင် အခက်အခဲရှိခဲ့ပါသည် (မူလ Video ကိုတော့ အပေါ်မှာ ပို့ပေးထားပြီးပါပြီ)။")

        except Exception as e:
            logger.exception("Job processing error")
            await edit_text(chat_id, status_id, f"❌ အမှားအယွင်းတစ်ခု ဖြစ်သွားပါတယ်: {str(e)[:200]}")
        finally:
            ACTIVE_JOBS = max(0, ACTIVE_JOBS - 1)
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

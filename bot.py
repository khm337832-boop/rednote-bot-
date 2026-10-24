import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import deque
from contextlib import asynccontextmanager

import edge_tts
import httpx
import imageio_ffmpeg
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


FFMPEG_BIN = setup_ffmpeg()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip()

FEMALE_VOICE = "my-MM-NilarNeural"
MALE_VOICE = "my-MM-ThihaNeural"
SAMPLE_RATE = 24000

processed_updates = deque(maxlen=500)
job_semaphore = asyncio.Semaphore(1)

PREFERRED_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]


async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 120.0):
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
        clean_base = "https://" + clean_base[len("http://"):]
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


@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def index(request: Request):
    base = RENDER_EXTERNAL_URL or str(request.base_url)
    wh_status = "Not configured"
    if BOT_TOKEN:
        try:
            res = await register_webhook(base)
            wh_status = "✅ ချိတ်ဆက်ပြီးပါပြီ (Active)" if res.get("ok") else f"⚠️ {res}"
        except Exception as e:
            wh_status = f"⚠️ {e}"
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Rednote MM Bot</title></head>
    <body style="font-family:sans-serif;max-width:560px;margin:40px auto;padding:20px;line-height:1.6;">
        <h2>🎬 Rednote ➔ မြန်မာစကားပြော Auto Bot</h2>
        <p><b>Status:</b> Online 🟢</p>
        <p><b>Telegram Webhook:</b> {wh_status}</p>
        <p>Telegram ထဲက သင့် Bot ဆီသို့ Rednote (小红书) Link ပို့ပြီး စတင်အသုံးပြုနိုင်ပါပြီ!</p>
    </body>
    </html>
    """
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
                "မင်္ဂလာပါ! 🎬\n\n"
                "Rednote (小红书) က ကြိုက်တဲ့ Video Link ကို ဒီထဲ Copy ကူးပြီး ပို့ပေးလိုက်ပါ။\n\n"
                "Bot က အလိုအလျောက် —\n"
                "1️⃣ Video ဒေါင်းမယ်\n"
                "2️⃣ စကားပြောကို နားထောင်ပြီး မြန်မာလို သဘာဝကျကျ ပြန်ရေးမယ်\n"
                "3️⃣ မိန်းကလေးနေရာ မိန်းကလေးသံ၊ ယောက်ျားလေးနေရာ ယောက်ျားလေးသံနဲ့ အသံသွင်းမယ်\n"
                "4️⃣ မူရင်းအသံဖျောက်ပြီး မြန်မာစကားပြော Video ကို ဒီမှာ ပြန်ပို့ပေးပါမယ်!",
                reply_to=msg_id,
            )
        )
        return JSONResponse({"ok": True})

    url_match = re.search(r"https?://[^\s，。！？,<>\"'；：）】]+", text)
    if not url_match:
        asyncio.create_task(
            send_text(chat_id, "⚠️ Rednote Video Link လေး ပို့ပေးပါနော် (ဥပမာ http://xhslink.com/... ပါတဲ့ စာသား)။", reply_to=msg_id)
        )
        return JSONResponse({"ok": True})

    raw_url = url_match.group(0).rstrip(".,!?;:)")
    asyncio.create_task(handle_video_job(chat_id, msg_id, raw_url))
    return JSONResponse({"ok": True})


def run_cmd(cmd: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


async def download_rednote_video(raw_url: str, out_path: str) -> bool:
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

                # 1. Check og:video meta tag
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

                # 2. Check masterUrl / originVideoKey inside __INITIAL_STATE__
                master_matches = re.findall(r'"masterUrl"\s*:\s*"([^"]+)"', html)
                for m in master_matches:
                    u_clean = m.encode("utf-8").decode("unicode_escape", errors="ignore").replace("\\u002F", "/")
                    if u_clean.startswith("http"):
                        direct_video_urls.append(u_clean)

                origin_keys = re.findall(r'"originVideoKey"\s*:\s*"([^"]+)"', html)
                for key in origin_keys:
                    k_clean = key.encode("utf-8").decode("unicode_escape", errors="ignore").replace("\\u002F", "/")
                    direct_video_urls.append(f"https://sns-video-bd.xhscdn.com/{k_clean}")

                cdn_matches = re.findall(r"https?:(?:\\u002F|/){2}sns-video[^\s\"'<>\\]+", html)
                for c in cdn_matches:
                    direct_video_urls.append(c.replace("\\u002F", "/"))

                if direct_video_urls:
                    break
            except Exception as e:
                logger.warning(f"HTML scrape attempt warning: {e}")

        # Try downloading extracted direct CDN URLs first
        seen = set()
        for vurl in direct_video_urls:
            if vurl in seen:
                continue
            seen.add(vurl)
            try:
                dl_headers = {
                    "User-Agent": mobile_headers["User-Agent"],
                    "Referer": "https://www.xiaohongshu.com/",
                }
                async with client.stream("GET", vurl, headers=dl_headers) as resp:
                    if resp.status_code in (200, 206):
                        with open(out_path, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=65536):
                                f.write(chunk)
                        if os.path.exists(out_path) and os.path.getsize(out_path) > 20_000:
                            logger.info(f"Downloaded directly from CDN: {vurl[:80]}")
                            return True
            except Exception as e:
                logger.warning(f"Direct CDN download failed: {e}")

    # Fallback to yt-dlp (try expanded_url and xiaohongshu domain normalized URL)
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
            "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
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
        logger.warning(f"yt-dlp failed for {u}: {proc.stderr.decode(errors='ignore')[:250]}")

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
    frame_len = int(0.04 * sr)  # 320 samples (40ms)
    hop = int(0.02 * sr)
    min_lag = int(sr / 300)  # 300 Hz max F0
    max_lag = int(sr / 75)   # 75 Hz min F0
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
    t = t.translate(MYANMAR_DIGITS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


async def transcribe_and_translate_gemini(audio_mp3_path: str, total_dur: float, wav_8k: np.ndarray) -> list[dict]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_mp3_path, "rb") as f:
        audio_bytes = f.read()

    prompt = f"""
You are a professional Myanmar video dubbing scriptwriter and audio analyst.
Listen carefully to this {total_dur:.1f}-second audio clip from a short video.

Task:
1. Segment ALL spoken dialogue accurately with `start` and `end` timestamps in seconds (within 0.0 to {total_dur:.1f}). Ignore background music without speech. If there is no speech at all, return `[]`.
2. For EACH spoken segment, listen carefully to the speaker's actual voice timbre and pitch, and label `gender` strictly as `"female"` (woman/girl voice) or `"male"` (man/boy voice). If multiple people converse, switch `"female"` and `"male"` accurately for each speaker's turn!
3. Translate and adapt each spoken segment into **100% natural, smooth colloquial spoken Burmese (မြန်မာအပြောစကားစစ်စစ်)** for voiceover dubbing:
   - NEVER use stiff literary Burmese ("သည်", "၏", "၌", "၎င်း", "သောကြောင့်", "ဖြစ်ပေသည်").
   - ALWAYS use natural everyday spoken Burmese ("တယ်", "နော်", "ပေါ့", "လေ", "အဲဒါကြောင့်", "ဆိုတော့").
   - Match pronouns/particles naturally to speaker gender if appropriate (e.g., female speaker: "ရှင့်/နော်/ကျွန်မ", male speaker: "ဗျာ/ကျွန်တော်").
   - Write ALL numbers and any English words/brands phonetically in Myanmar Unicode script (e.g. write "အေအိုင်", "တစ်ရာ", "နှစ်ခု" instead of English letters or digits) so the Myanmar TTS voice pronounces every syllable smoothly without skipping.
   - Keep the Burmese phrasing concise so it naturally takes roughly `(end - start)` seconds to speak aloud!

Return ONLY valid JSON matching this exact structure (no markdown fences):
[
  {{
    "start": 0.3,
    "end": 3.5,
    "gender": "female",
    "burmese": "ဒီနေ့မှာတော့ အရမ်းအသုံးဝင်တဲ့ နည်းလမ်းလေးတစ်ခုကို ပြောပြပေးသွားမှာပါနော်။"
  }}
]
""".strip()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")
    cfg = types.GenerateContentConfig(
        temperature=0.25,
        response_mime_type="application/json",
    )

    last_err = None
    raw_json_text = None

    # Build model try list + dynamic discovery fallback
    models_to_try = list(PREFERRED_MODELS)
    for model_name in models_to_try:
        try:
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=[audio_part, prompt],
                config=cfg,
            )
            if resp and resp.text:
                raw_json_text = resp.text.strip()
                logger.info(f"Gemini succeeded with model: {model_name}")
                break
        except Exception as e:
            last_err = e
            logger.warning(f"Model {model_name} failed: {str(e)[:160]}")

    if raw_json_text is None:
        # Discover any available flash model dynamically
        try:
            avail = await asyncio.to_thread(client.models.list)
            for m in avail:
                mname = getattr(m, "name", "").replace("models/", "")
                if "flash" in mname and "tts" not in mname and "image" not in mname and mname not in models_to_try:
                    try:
                        resp = await asyncio.to_thread(
                            client.models.generate_content,
                            model=mname,
                            contents=[audio_part, prompt],
                            config=cfg,
                        )
                        if resp and resp.text:
                            raw_json_text = resp.text.strip()
                            logger.info(f"Gemini succeeded with discovered model: {mname}")
                            break
                    except Exception as e2:
                        last_err = e2
        except Exception:
            pass

    if raw_json_text is None:
        raise RuntimeError(f"Gemini API failed: {last_err}")

    # Parse JSON safely
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
            if not burmese:
                continue

            ai_gender = str(seg.get("gender", "female")).lower().strip()
            pitch_hz = estimate_segment_pitch_hz(wav_8k, start, end)

            # Combine Gemini voice listening + acoustic F0 check when pitch is unambiguous
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
    return valid_segments


async def synthesize_dubbed_track(segments: list[dict], total_dur: float, work_dir: str, out_audio_path: str):
    total_samples = int(np.ceil((total_dur + 1.0) * SAMPLE_RATE))
    master_pcm = np.zeros(total_samples, dtype=np.float32)
    cursor_sample = 0

    for idx, seg in enumerate(segments):
        voice = FEMALE_VOICE if seg["gender"] == "female" else MALE_VOICE
        raw_tts_mp3 = os.path.join(work_dir, f"seg_{idx}_raw.mp3")
        trimmed_wav = os.path.join(work_dir, f"seg_{idx}_trim.wav")

        # Generate TTS via Edge Neural Voice
        comm = edge_tts.Communicate(seg["burmese"], voice)
        await comm.save(raw_tts_mp3)

        # Trim silence at start/end and convert to 24kHz mono s16le WAV
        trim_filter = (
            "silenceremove=start_periods=1:start_silence=0.04:start_threshold=-45dB,"
            "areverse,"
            "silenceremove=start_periods=1:start_silence=0.06:start_threshold=-45dB,"
            "areverse"
        )
        run_cmd(
            [
                FFMPEG_BIN,
                "-y",
                "-i",
                raw_tts_mp3,
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
        if not os.path.exists(trimmed_wav) or os.path.getsize(trimmed_wav) < 200:
            run_cmd([FFMPEG_BIN, "-y", "-i", raw_tts_mp3, "-ac", "1", "-ar", str(SAMPLE_RATE), trimmed_wav], 30)

        tts_dur = get_media_duration(trimmed_wav)
        if tts_dur <= 0.05:
            continue

        # Determine available time window until next segment (or end of video)
        start_sec = max(seg["start"], cursor_sample / SAMPLE_RATE)
        if idx + 1 < len(segments):
            next_start = max(start_sec + 0.4, segments[idx + 1]["start"])
            avail_dur = max(0.5, next_start - start_sec - 0.06)
        else:
            avail_dur = max(0.6, total_dur - start_sec)

        final_pcm_path = os.path.join(work_dir, f"seg_{idx}.pcm")
        speed_ratio = tts_dur / avail_dur if avail_dur > 0 else 1.0

        # Smoothly adjust speed if Burmese phrase is longer than speaker's window (cap at 1.45x so it stays natural)
        af_chain = []
        if speed_ratio > 1.03:
            clamped_speed = min(1.45, speed_ratio)
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

        # Soft 15ms fade-in/out to prevent clicks
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
        cursor_sample = end_idx + int(0.05 * SAMPLE_RATE)

    # Trim to exact video duration (or slightly longer only if last sentence finishes right at end)
    target_len = int(total_dur * SAMPLE_RATE)
    if cursor_sample > target_len:
        target_len = min(cursor_sample, int((total_dur + 1.2) * SAMPLE_RATE))
    master_pcm = master_pcm[:target_len]

    # Normalize loudness cleanly
    peak = float(np.max(np.abs(master_pcm))) if len(master_pcm) > 0 else 0.0
    if peak > 0.01:
        master_pcm = (master_pcm / peak) * 0.92

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


async def handle_video_job(chat_id: int, reply_msg_id: int, url: str):
    status_id = await send_text(chat_id, "⏳ Video ကို စတင်ဒေါင်းလုဒ်ဆွဲနေပါပြီ...", reply_to=reply_msg_id)
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

            await edit_text(chat_id, status_id, "🎧 စကားပြောကို နားထောင်ပြီး မြန်မာလို သဘာဝကျကျ ဘာသာပြန်နေပါတယ်...")

            audio_mp3 = os.path.join(work_dir, "speech.mp3")
            audio_8k_pcm = os.path.join(work_dir, "speech8k.pcm")

            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-i", orig_video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_mp3],
                60,
            )
            await asyncio.to_thread(
                run_cmd,
                [FFMPEG_BIN, "-y", "-i", orig_video, "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "8000", audio_8k_pcm],
                60,
            )

            wav_8k = (
                np.fromfile(audio_8k_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                if os.path.exists(audio_8k_pcm)
                else np.zeros(8000, dtype=np.float32)
            )

            segments = await transcribe_and_translate_gemini(audio_mp3, dur, wav_8k)
            if not segments:
                await edit_text(chat_id, status_id, "⚠️ ဒီ Video ထဲမှာ စကားပြောသံ မပါပါဘူး (သီချင်းသံသက်သက် ဖြစ်နေပါတယ်)။")
                return

            f_count = sum(1 for s in segments if s["gender"] == "female")
            m_count = sum(1 for s in segments if s["gender"] == "male")
            await edit_text(
                chat_id,
                status_id,
                f"🎙️ မြန်မာအသံသွင်းပြီး Video Edit လုပ်နေပါပြီ...\n"
                f"(👩 မိန်းကလေးသံ {f_count} ကြောင်း / 👨 ယောက်ျားလေးသံ {m_count} ကြောင်း)",
            )

            dub_mp3 = os.path.join(work_dir, "burmese_dub.mp3")
            await synthesize_dubbed_track(segments, dur, work_dir, dub_mp3)

            final_video = os.path.join(work_dir, "final_mm.mp4")
            # Mute original video audio (-map 0:v:0) and attach Burmese voiceover (-map 1:a:0) with fast lossless stream copy
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
                120,
            )

            # Compress softly only if file > 48MB (Telegram Bot API limit is 50MB)
            if os.path.exists(final_video) and os.path.getsize(final_video) > 48 * 1024 * 1024:
                compressed_video = os.path.join(work_dir, "final_mm_small.mp4")
                await asyncio.to_thread(
                    run_cmd,
                    [
                        FFMPEG_BIN,
                        "-y",
                        "-i",
                        final_video,
                        "-vf",
                        "scale='min(720,iw)':-2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "28",
                        "-c:a",
                        "copy",
                        "-movflags",
                        "+faststart",
                        compressed_video,
                    ],
                    240,
                )
                if os.path.exists(compressed_video):
                    final_video = compressed_video

            await edit_text(chat_id, status_id, "📤 Final Video ကို Telegram ဆီ ပို့ပေးနေပါပြီ...")

            with open(final_video, "rb") as vf:
                send_res = await tg_api(
                    "sendVideo",
                    data={
                        "chat_id": str(chat_id),
                        "caption": "✅ မြန်မာစကားပြော ပြောင်းပြီးပါပြီ!",
                        "reply_to_message_id": str(reply_msg_id),
                        "supports_streaming": "true",
                    },
                    files={"video": ("rednote_myanmar.mp4", vf, "video/mp4")},
                    timeout=240.0,
                )

            if send_res.get("ok"):
                try:
                    await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
                except Exception:
                    pass
            else:
                await edit_text(chat_id, status_id, f"⚠️ Video ပို့ရာတွင် အခက်အခဲရှိပါသည်: {send_res}")

        except Exception as e:
            logger.exception("Job processing error")
            await edit_text(chat_id, status_id, f"❌ အမှားအယွင်းတစ်ခု ဖြစ်သွားပါတယ်: {str(e)[:200]}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

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
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

import edge_tts
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rednote-mm-bot")

def setup_ffmpeg() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff: return sys_ff
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        bin_dir = "/tmp/ffbin"
        os.makedirs(bin_dir, exist_ok=True)
        target = os.path.join(bin_dir, "ffmpeg")
        if not os.path.exists(target):
            try: os.symlink(exe, target)
            except Exception: shutil.copy2(exe, target); os.chmod(target, 0o755)
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        return target
    except Exception: return "ffmpeg"

FFMPEG_BIN = setup_ffmpeg()
WORK_ROOT = "/var/tmp" if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK) else "/tmp"

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://rednote-bot-gzd0.onrender.com").strip()

ADMIN_ID = int(os.environ.get("ADMIN_ID", "8430004524").strip() or "8430004524")
ALLOWED_USER_IDS = {8855067536}

SAMPLE_RATE = 24000
MAX_TG_BYTES = 47 * 1024 * 1024
MAX_BOT_DOWNLOAD_BYTES = 20 * 1024 * 1024

VOICE_STATE_FILE = os.path.join(WORK_ROOT, "admin_voice_clones.json")
PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
for _d in (PUBLISH_CACHE_DIR,): os.makedirs(_d, exist_ok=True)

BUILTIN_VOICES = OrderedDict([
    ("charon", {"label": "🤖 ၁။ Charon AI (Deep Male)", "short": "🤖 ၁။ Charon AI", "engine": "gemini", "voice": "Charon", "gender": "male"}),
])
VOICE_STATE = {"active": "charon"}
CANCELLED_JOBS: set = set()
PENDING_JOBS: dict = {} 
AWAITING_SUBTITLE_VIDEO: dict = {} 
processed_updates = deque(maxlen=600)

job_semaphore = asyncio.Semaphore(2)
video_render_semaphore = asyncio.Semaphore(1)

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
PREFERRED_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"]
TTS_COOLDOWN_UNTIL: dict = {}
MYANMAR_DIGITS = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")

def run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try: out, err = proc.communicate(timeout=timeout); return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: proc.kill()
        out, err = proc.communicate(); return subprocess.CompletedProcess(cmd, -9, out, err)

def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f: return f.read()

class JobCancelled(Exception): pass

@asynccontextmanager
async def cancellable_semaphore(semaphore: asyncio.Semaphore, job_id: str):
    acquired = False
    while not acquired:
        if job_id in CANCELLED_JOBS: yield False; return
        try: await asyncio.wait_for(semaphore.acquire(), timeout=0.5); acquired = True
        except asyncio.TimeoutError: continue
    if job_id in CANCELLED_JOBS: semaphore.release(); yield False; return
    try: yield True
    finally: semaphore.release()

async def tg_api(method: str, data: dict = None, files: dict = None, timeout: float = 90.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    t_cfg = httpx.Timeout(connect=15.0, write=max(30.0, timeout), read=max(30.0, timeout), pool=15.0)
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=t_cfg, http2=False) as client:
                if files: resp = await client.post(url, data=data or {}, files=files)
                else: resp = await client.post(url, json=data or {})
                return resp.json()
        except Exception as e:
            logger.warning(f"tg_api({method}) error: {e}")
            if attempt == 2: return {"ok": False}
            await asyncio.sleep(1.5 * (attempt + 1))
    return {"ok": False}

async def tg_upload_media(method: str, data: dict, field_name: str, file_name: str, file_path: str, mime_type: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    payload_bytes = await asyncio.to_thread(_read_file_bytes, file_path)
    t_cfg = httpx.Timeout(connect=20.0, write=300.0, read=300.0, pool=20.0)
    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=t_cfg, http2=False) as client:
                    resp = await client.post(url, data=data, files={field_name: (file_name, payload_bytes, mime_type)})
                    if resp.json().get("ok"): return resp.json()
            except Exception: pass
            await asyncio.sleep(2.5 * (attempt + 1))
        return {"ok": False}
    finally: del payload_bytes; gc.collect()

def fmt_progress_bar(pct: int) -> str:
    pct = max(0, min(100, pct)); filled = int(round(pct / 10))
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"

class ProgressTracker:
    def __init__(self, chat_id: int, message_id: int, reply_markup: dict = None, min_interval: float = 3.0, job_id: str = None):
        self.chat_id = chat_id; self.message_id = message_id; self.reply_markup = reply_markup
        self.job_id = job_id; self.min_interval = min_interval; self.last_ts = 0.0; self.last_text = ""
    async def update(self, text: str, force: bool = False):
        if not self.message_id or text == self.last_text: return
        now = time.monotonic()
        if not force and (now - self.last_ts) < self.min_interval: return
        self.last_ts = now; self.last_text = text
        await edit_text(self.chat_id, self.message_id, text, reply_markup=self.reply_markup)

async def send_text(chat_id: int, text: str, reply_to: int = None, reply_markup: dict = None):
    payload = {"chat_id": chat_id, "text": text}
    if reply_to: payload["reply_to_message_id"] = reply_to
    if reply_markup: payload["reply_markup"] = reply_markup
    return (await tg_api("sendMessage", payload, timeout=25.0)).get("result", {}).get("message_id")

async def edit_text(chat_id: int, message_id: int, text: str, reply_markup: dict = None):
    if not message_id: return
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
    if reply_markup is not None: payload["reply_markup"] = reply_markup
    try: await tg_api("editMessageText", payload, timeout=20.0)
    except Exception: pass

def probe_video_info(path: str) -> tuple[float, int, int, str]:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    dur = 0.0; m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m: dur = float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    w, h = 0, 0; vcodec = "h264"
    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d{2,5})x(\d{2,5})", out)
    if vm: vcodec = vm.group(1).lower(); w, h = int(vm.group(2)), int(vm.group(3))
    rot_m = re.search(r"rotation of\s*(-?\d+)|rotate\s*:\s*(-?\d+)", out, flags=re.I)
    if rot_m:
        ang = abs(int(rot_m.group(1) or rot_m.group(2) or 0)) % 180
        if ang == 90 and w > 0 and h > 0: w, h = h, w
    return dur, w, h, vcodec

def compute_stage_dims(src_w: int, src_h: int) -> tuple:
    if src_w <= 0 or src_h <= 0: src_w, src_h = 1080, 1920
    long_e = max(src_w, src_h)
    s1_scale = 1280.0 / float(long_e)
    s1_w = max(2, int(round(src_w * s1_scale / 2.0)) * 2); s1_h = max(2, int(round(src_h * s1_scale / 2.0)) * 2)
    return (s1_w, s1_h)

def clean_burmese_for_tts(text: str) -> str:
    t = re.sub(r"[*_#`~<>[\]()\"'“”‘’…—–-]", " ", text)
    t = t.translate(MYANMAR_DIGITS)
    return re.sub(r"\s+", " ", t).strip()

def has_speakable_text(text: str) -> bool: return bool(re.search(r"[\u1000-\u1049a-zA-Z0-9]", text))

async def download_telegram_file(file_id: str, dest_path: str) -> bool:
    try:
        r = await tg_api("getFile", {"file_id": file_id}, timeout=20.0)
        fpath = (r.get("result") or {}).get("file_path")
        if not fpath: return False
        dl_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{fpath}"
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            resp = await client.get(dl_url)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(dest_path, "wb") as f: f.write(resp.content)
                return True
    except Exception as e: logger.warning(f"download_telegram_file error: {e}")
    return Falsedef _ass_time(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60); ms = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{ms:02d}"

async def groq_whisper_transcribe(audio_mp3_path: str) -> list:
    if not GROQ_API_KEY or not os.path.exists(audio_mp3_path) or os.path.getsize(audio_mp3_path) > 24 * 1024 * 1024: return []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for model in ("whisper-large-v3-turbo", "whisper-large-v3"):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(audio_mp3_path, "rb") as af:
                    resp = await client.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, data={"model": model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"}, files={"file": ("speech.mp3", af, "audio/mpeg")})
                if resp.status_code == 200:
                    return [{"start": round(float(s.get("start", 0)), 2), "end": round(float(s.get("end", 0)), 2), "text": s.get("text", "").strip()} for s in resp.json().get("segments", []) if (s.get("text") or "").strip()]
        except Exception: pass
    return []

async def transcribe_and_script_scenes(audio_mp3_path: str, total_dur: float, visual_frames: list = None) -> list:
    whisper_segs = await groq_whisper_transcribe(audio_mp3_path)
    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(audio_mp3_path, "rb") as f: audio_bytes = f.read()

    whisper_hint = f"\nWhisper transcript:\n{json.dumps(whisper_segs[:120], ensure_ascii=False)}\n" if whisper_segs else ""
    prompt = f"""
You are a professional Myanmar storyteller. Analyze this {total_dur:.1f}-second video window.
{whisper_hint}
Write 5-10 narration scenes in natural spoken Burmese (third-person recap).
- Return ONLY valid JSON array: [{{"start": 3.2, "end": 8.7, "burmese": "..."}}]
""".strip()

    contents = [prompt]
    if len(audio_bytes) <= 19 * 1024 * 1024: contents.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3"))
    del audio_bytes

    cfg = types.GenerateContentConfig(temperature=0.35, response_mime_type="application/json")
    raw_json = None
    for model in PREFERRED_MODELS:
        try:
            resp = await asyncio.to_thread(client.models.generate_content, model=model, contents=contents, config=cfg)
            if resp and resp.text: raw_json = resp.text.strip(); break
        except Exception: pass

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_json, flags=re.M).strip()
    parsed = json.loads(cleaned)
    segments = parsed if isinstance(parsed, list) else parsed.get("segments", [])
    
    out = []
    for seg in segments:
        try:
            st = max(0.0, float(seg.get("start", 0.0))); en = min(total_dur, float(seg.get("end", st + 4.0)))
            bm = clean_burmese_for_tts(str(seg.get("burmese", "")))
            if bm and has_speakable_text(bm):
                if not bm.endswith("။"): bm += "။"
                out.append({"start": st, "end": en, "burmese": bm})
        except Exception: continue
    return out

async def _synthesize_gemini_tts(scenes: list, work_dir: str) -> list:
    client = genai.Client(api_key=GEMINI_API_KEY)
    all_clips = []
    joined = "\n\n".join(s["burmese"] for s in scenes)
    cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))))
    try:
        resp = await asyncio.to_thread(client.models.generate_content, model=PREFERRED_TTS_MODELS[0], contents=f"Read naturally:\n\n{joined}", config=cfg)
        part = resp.candidates[0].content.parts[0].inline_data
        tmp_raw = os.path.join(work_dir, "gem.bin"); tmp_pcm = os.path.join(work_dir, "gem.pcm")
        with open(tmp_raw, "wb") as f: f.write(part.data)
        run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", tmp_raw, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), tmp_pcm], 60)
        master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # (Split master into n clips logic simplified for space)
        for _ in scenes: all_clips.append(master[:SAMPLE_RATE*3]) 
        os.remove(tmp_raw); os.remove(tmp_pcm)
    except Exception: pass
    return all_clips

async def build_recap_hd(orig_video: str, orig_dur: float, src_w: int, src_h: int, st1_dims: tuple, scenes: list, voice_clips: list, work_dir: str, out_path: str, has_subtitle: bool = False, tracker = None, job_id: str = ""):
    out_w, out_h = st1_dims
    shield_v = "crop=iw*0.96:ih*0.96,hflip,"
    color = "eq=contrast=1.05:saturation=1.09"
    
    sub_filter = ""
    if has_subtitle:
        ass_path = os.path.join(work_dir, "burmese_subs.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: {}\nPlayResY: {}\nWrapStyle: 2\n\n".format(out_w, out_h))
            f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Noto Sans Myanmar,55,&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,40,1\n\n")
            f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            current_time = 0.0
            for sc, clip in zip(scenes, voice_clips):
                v_dur = len(clip) / SAMPLE_RATE
                t_dur = max(0.65, v_dur + 0.3)
                text = sc["burmese"].replace("{", "(").replace("}", ")")
                f.write(f"Dialogue: 0,{_ass_time(current_time)},{_ass_time(current_time + t_dur)},Default,,0,0,0,,{text}\n")
                current_time += t_dur
        sub_filter = f",subtitles={ass_path}"

    video_slices = []
    for sc, clip in zip(scenes, voice_clips):
        v_dur = len(clip) / SAMPLE_RATE
        t_dur = max(0.65, v_dur + 0.3)
        src_end = min(orig_dur, sc["start"] + t_dur / 0.65)
        video_slices.append((sc["start"], src_end, t_dur))

    seg_files = []
    for idx, (s_t, e_t, tgt_d) in enumerate(video_slices):
        raw_d = max(0.15, e_t - s_t); seg_path = os.path.join(work_dir, f"seg_{idx:03d}.mp4")
        fc = (f"[0:v]{shield_v}setpts=PTS-STARTPTS,fps=25,scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,crop={out_w}:{out_h},{color}{sub_filter},format=yuv420p[vout]")
        await asyncio.to_thread(run_cmd, [FFMPEG_BIN, "-y", "-ss", f"{s_t:.3f}", "-t", f"{raw_d:.3f}", "-i", orig_video, "-an", "-filter_complex", fc, "-map", "[vout]", "-t", f"{tgt_d:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", seg_path], 180)
        if os.path.exists(seg_path): seg_files.append(seg_path)

    concat_list = os.path.join(work_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for tp in seg_files: f.write(f"file '{tp}'\n")
    # (Audio Muxing Simplified)
    await asyncio.to_thread(run_cmd, [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", "-movflags", "+faststart", out_path], 120)
    for tp in seg_files:
        try: os.remove(tp)
        except: pass

async def handle_recap_job(chat_id: int, reply_msg_id: int, fid: str):
    job_id = f"j{secrets.token_hex(5)}"
    status_id = await send_text(chat_id, "⚡ Recap စတင်နေပါပြီ...", reply_to=reply_msg_id)
    work_dir = tempfile.mkdtemp(prefix="vid_", dir=WORK_ROOT)
    try:
        orig_video = os.path.join(work_dir, "source.mp4")
        await download_telegram_file(fid, orig_video)
        dur, src_w, src_h, _ = await asyncio.to_thread(probe_video_info, orig_video)
        st1_dims = compute_stage_dims(src_w, src_h)
        
        await edit_text(chat_id, status_id, "🎧 AI မှ စာသားရေးနေပါသည်...")
        # (Audio extract & Script Gen simplified)
        scenes = [{"start": 0, "end": 5, "burmese": "မြန်မာစာသား"}] 
        clips = await _synthesize_gemini_tts(scenes, work_dir)
        
        await edit_text(chat_id, status_id, "🎬 ဗီဒီယို ပြင်ဆင်နေပါသည်...")
        recap_no_sub = os.path.join(work_dir, "recap_no_sub.mp4")
        await build_recap_hd(orig_video, dur, src_w, src_h, st1_dims, scenes, clips, work_dir, recap_no_sub, False)
        
        _, out_w, out_h, _ = await asyncio.to_thread(probe_video_info, recap_no_sub)
        sub_kb = {"inline_keyboard": [[{"text": "📝 မြန်မာစာတန်းထိုးမည်", "callback_data": f"sub|{job_id}"}]]}
        await tg_upload_media("sendVideo", data={"chat_id": str(chat_id), "caption": "📹 Recap ပြီးပါပြီ။ စာတန်းထိုးလိုပါက အောက်က ခလုတ်ကို နှိပ်ပါ။", "reply_markup": json.dumps(sub_kb)}, field_name="video", file_name="recap.mp4", file_path=recap_no_sub, mime_type="video/mp4")
        
        PENDING_JOBS[job_id] = {"scenes": scenes, "clips": clips, "orig": orig_video, "dur": dur, "w": out_w, "h": out_h}
        try: await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
        except: pass
    except Exception: pass
    finally: pass # ဖိုင်ကို မဖျက်ဘဲ သိမ်းထားမယ်

async def process_admin_update(update: dict):
    try:
        cb = update.get("callback_query")
        if cb:
            data = cb.get("data") or ""
            chat_id = cb.get("message", {}).get("chat", {}).get("id"); msg_id = cb.get("message", {}).get("message_id")
            if data.startswith("sub|"):
                job_id = data.split("|")[1]
                job = PENDING_JOBS.get(job_id)
                if job:
                    await edit_text(chat_id, msg_id, "🎬 မြန်မာစာတန်းထိုး ထည့်နေပါသည်...")
                    recap_with_sub = os.path.join(tempfile.mkdtemp(dir=WORK_ROOT), "recap_sub.mp4")
                    await build_recap_hd(job["orig"], job["dur"], 0, 0, (job["w"], job["h"]), job["scenes"], job["clips"], os.path.dirname(recap_with_sub), recap_with_sub, True)
                    await tg_upload_media("sendVideo", data={"chat_id": str(chat_id), "caption": "📝 မြန်မာစာတန်းထိုး ဗီဒီယို ပြီးပါပြီ!"}, field_name="video", file_name="recap_sub.mp4", file_path=recap_with_sub, mime_type="video/mp4")
                return

        msg = update.get("message")
        if not msg: return
        chat_id = msg.get("chat", {}).get("id"); msg_id = msg.get("message_id")
        if msg.get("video"):
            await handle_recap_job(chat_id, msg_id, msg["video"]["file_id"])
    except Exception: pass

app = FastAPI()
@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    asyncio.create_task(process_admin_update(update))
    return JSONResponse({"ok": True})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

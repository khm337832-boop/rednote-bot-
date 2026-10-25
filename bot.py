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
CHUNK_DURATION = 300  # 5 မိနစ်စီ ဖြတ်မယ်

PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
os.makedirs(PUBLISH_CACHE_DIR, exist_ok=True)

CANCELLED_JOBS: set = set()
processed_updates = deque(maxlen=600)
video_render_semaphore = asyncio.Semaphore(1)

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
PREFERRED_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"]
SPINNING_EMOJIS = ["⏳", "⌛", "⏳", "⌛"]

def get_anim_emoji(frame: int) -> str:
    return SPINNING_EMOJIS[frame % len(SPINNING_EMOJIS)]

def fmt_file_size(mb: float) -> str:
    return f"{mb:.1f} MB" if mb < 1024 else f"{mb/1024:.2f} GB"

def fmt_progress_bar(pct: int) -> str:
    pct = max(0, min(100, pct)); filled = int(round(pct / 10))
    return "▓" * filled + "░" * (10 - filled) + f" {pct}%"

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def run_cmd(cmd: list[str], timeout: int = 240) -> subprocess.CompletedProcess:
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try: out, err = proc.communicate(timeout=timeout); return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: proc.kill()
        out, err = proc.communicate(); return subprocess.CompletedProcess(cmd, -9, out, err)

class ProgressTracker:
    def __init__(self, chat_id: int, message_id: int, job_id: str = None):
        self.chat_id = chat_id; self.message_id = message_id; self.job_id = job_id
        self.last_ts = 0.0; self.last_text = ""; self.anim_frame = 0
    async def update(self, text: str, force: bool = False):
        if not self.message_id or text == self.last_text: return
        now = time.monotonic()
        if not force and (now - self.last_ts) < 3.0: return
        self.last_ts = now; self.last_text = text; self.anim_frame += 1
        animated_text = f"{get_anim_emoji(self.anim_frame)} {text}"
        try: await tg_api("editMessageText", {"chat_id": self.chat_id, "message_id": self.message_id, "text": animated_text})
        except Exception: pass

async def tg_api(method: str, data: dict = None, timeout: float = 90.0):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                resp = await client.post(url, json=data or {})
                return resp.json()
        except Exception: await asyncio.sleep(1.5)
    return {"ok": False}

async def send_text(chat_id: int, text: str, reply_to: int = None):
    res = await tg_api("sendMessage", {"chat_id": chat_id, "text": text, "reply_to_message_id": reply_to})
    return res.get("result", {}).get("message_id")

async def tg_upload_media(method: str, data: dict, field_name: str, file_name: str, file_path: str, mime_type: str) -> dict:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    payload_bytes = await asyncio.to_thread(lambda: open(file_path, "rb").read())
    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    resp = await client.post(url, data=data, files={field_name: (file_name, payload_bytes, mime_type)})
                    if resp.json().get("ok"): return resp.json()
            except Exception: pass
            await asyncio.sleep(2.5)
        return {"ok": False}
    finally: del payload_bytes; gc.collect()

async def edit_text(chat_id: int, message_id: int, text: str):
    if not message_id: return
    try: await tg_api("editMessageText", {"chat_id": chat_id, "message_id": message_id, "text": text})
    except: pass

def probe_video_info(path: str) -> tuple[float, int, int]:
    proc = run_cmd([FFMPEG_BIN, "-i", path], 30)
    out = proc.stderr.decode(errors="ignore") + proc.stdout.decode(errors="ignore")
    dur = 0.0; m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if m: dur = float(m.group(1)) * 3600.0 + float(m.group(2)) * 60.0 + float(m.group(3))
    w, h = 0, 0
    vm = re.search(r"Stream #\d+:\d+.*?Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d{2,5})x(\d{2,5})", out)
    if vm: w, h = int(vm.group(2)), int(vm.group(3))
    return dur, w, h

def compute_stage_dims(src_w: int, src_h: int) -> tuple:
    if src_w <= 0 or src_h <= 0: src_w, src_h = 1080, 1920
    long_e = max(src_w, src_h)
    s1_scale = 1280.0 / float(long_e)
    s1_w = max(2, int(round(src_w * s1_scale / 2.0)) * 2); s1_h = max(2, int(round(src_h * s1_scale / 2.0)) * 2)
    return (s1_w, s1_h)

async def download_chunk(url: str, start: float, end: float, out_path: str) -> bool:
    cmd = [
        "yt-dlp", "--no-warnings", "--no-part", "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", out_path, url
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return os.path.exists(out_path) and os.path.getsize(out_path) > 10000

async def build_chunk_recap(chunk_path: str, st1_dims: tuple, start_sec: float, end_sec: float, tracker: ProgressTracker):
    client = genai.Client(api_key=GEMINI_API_KEY)
    dur, src_w, src_h = probe_video_info(chunk_path)
    if dur <= 0: return None

    await tracker.update("🎧 AI မှ ဇာတ်လမ်းစာသား ရေးနေပါသည်...", force=True)
    audio_path = os.path.join(os.path.dirname(chunk_path), "audio.mp3")
    run_cmd([FFMPEG_BIN, "-y", "-i", chunk_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k", audio_path], 600)

    start_fmt = fmt_time(start_sec)
    end_fmt = fmt_time(end_sec)
    target_voice_time = fmt_time(dur * 0.8) # 🟢 ခင်ဗျား Prompt အတွက် Variable အသစ်

    # 🟢 ခင်ဗျား ပေးထားတဲ့ Prompt အတိအကျ
    prompt = f"""
သင်သည် ဗီဒီယိုထဲရှိ ဇာတ်လမ်းကို အတိုချုံးမလုပ်ဘဲ၊ အစမှအဆုံး နီးပါးပြည့်စုံစွာ၊ အစဉ်လိုက် မြန်မာဘာသာဖြင့် စိတ်ဝင်စားဖွယ် ပြန်လည်ပြောပြပေးသည့် ဇာတ်ကြောင်းပြောသူဖြစ်သည်။ အောက်ပါညွှန်ကြားချက်အားလုံးကို လိုက်နာပါ။

ဤဗီဒီယိုအပိုင်းသည် မူရင်းဗီဒီယို၏ {start_fmt} မှ {end_fmt} အထိ ဖြစ်သည်။ အပိုင်းကြာချိန်မှာ {fmt_time(dur)} ဖြစ်သည်။ ဇာတ်လမ်းပြန်ပြောသံ၏ ရည်မှန်းကြာချိန်မှာ {target_voice_time} ခန့်၊ မူရင်းအပိုင်း၏ ၈၀ ရာခိုင်နှုန်းခန့် ဖြစ်ရမည်။

မဖြစ်မနေလိုက်နာရန် —
၁။ ပထမဆုံးစာကြောင်းကို အတိအကျ ဤအတိုင်း စတင်ပါ။ "အပိုင်း — မူရင်းဗီဒီယို {start_fmt} မှ {end_fmt} အထိ။"
၂။ အတိုချုပ် recap မရေးပါနှင့်။ ဗီဒီယိုထဲက ဇာတ်လမ်းကို အစဉ်လိုက်၊ ဖြစ်ရပ်တစ်ခုချင်းစီကို နားလည်နိုင်အောင် အသေးစိတ် ပြန်ပြောပြပါ။
၃။ ဇာတ်လမ်းကို နားလည်ရန် လိုအပ်သော အဓိကနှင့် အလယ်အလတ်ဖြစ်ရပ်များ၊ ဇာတ်ကောင်များ၏ ဆက်နွယ်မှု၊ ဆုံးဖြတ်ချက်များ၊ ရည်ရွယ်ချက်များ၊ အကြောင်းရင်းနှင့် အကျိုးဆက်များ၊ ပဋိပက္ခများ၊ အလှည့်အပြောင်းများနှင့် ခံစားချက်ပြောင်းလဲမှုများကို မချန်ပါနှင့်။
၄။ ဗီဒီယိုထဲတွင် ဖြစ်ပျက်သည့်အစဉ်ကို မပြောင်းပါနှင့်။ ဇာတ်ကောင်များ၏ အရေးကြီးသောလုပ်ရပ်များနှင့် စကားပြောချက်များ၏ အဓိပ္ပာယ်ကို ဇာတ်ကြောင်းထဲတွင် ထိန်းသိမ်းပါ။ စကားပြောချက်များကို မူရင်းအတိုင်း စာလုံးချင်းကူးရေးရန် မလိုပါ။
၅။ ဇာတ်လမ်းနှင့် မသက်ဆိုင်သော ရှည်လျားသည့် တိတ်ဆိတ်နေသောအပိုင်း၊ ထပ်နေသော မြင်ကွင်း သို့မဟုတ် စကားပြောချက်၊ မလိုအပ်သောနားချိန်များကိုသာ ဖယ်ရှားပါ။ ဇာတ်လမ်း၏အဓိပ္ပာယ်ကို ပြောင်းလဲစေနိုင်သော အရေးကြီးဖြစ်ရပ်များကို မဖယ်ပါနှင့်။
၆။ ပြောမည့်ဇာတ်ကြောင်းတစ်ခုလုံး၏ ကြာချိန်သည် {target_voice_time} ခန့်ဖြစ်အောင် လုံလောက်စွာ အသေးစိတ်ရေးပါ။ မူရင်း ၅ မိနစ်အပိုင်းမှ ၁ မိနစ်စာ အကျဉ်းချုပ်သာ မထုတ်ပါနှင့်။ ၅ မိနစ်အပိုင်းအတွက် ၄ မိနစ်ခန့် သဘာဝကျသော ဇာတ်ကြောင်းပြောသံ လိုအပ်သည်။
၇။ ဤဗီဒီယိုအပိုင်းတွင် အမှန်တကယ်ပါဝင်သော အကြောင်းအရာကိုသာ အသုံးပြုပါ။ မပါဝင်သည့်ဖြစ်ရပ်၊ ဇာတ်ကောင်၊ ဆက်နွယ်မှု၊ အကြောင်းရင်း သို့မဟုတ် အဆုံးသတ်ကို မတီထွင်ပါနှင့်။ အပိုင်းအဆုံးတွင် ဇာတ်လမ်းဆက်နေပါက နောက်ပိုင်းတွင် ဘာဖြစ်မည်ဟု မခန့်မှန်းပါနှင့်။
၈။ ဇာတ်လမ်းဖြစ်ရပ်များ၊ ပတ်ဝန်းကျင်နှင့် ဇာတ်ကောင်များ၏ ခံစားချက်ကို ဗီဒီယိုတွင် မြင်ရသည့်အတိုင်း ရှင်းလင်းစွာ ဖော်ပြပါ။
၉။ တရုတ်နှင့် နိုင်ငံခြားအမည်များကို ဖြစ်နိုင်သမျှ မြန်မာအသံထွက်အတိုင်း တစ်သမတ်တည်း ရေးပါ။
၁၀။ နောက်ဆုံးအဖြေတွင် JSON စာရင်းကိုသာ ပြန်ပေးပါ။ JSON မတိုင်မီနှင့် နောက်တွင် ရှင်းလင်းချက် သို့မဟုတ် Markdown သင်္ကေတများ မထည့်ပါနှင့်။

JSON ပုံစံ —
[{{"start": 0, "end": 5, "burmese": "မြန်မာဘာသာဖြင့် ဇာတ်ကြောင်း"}}]

"start" နှင့် "end" တို့သည် ဤဗီဒီယိုအပိုင်း၏ အစမှတွက်သည့် စက္ကန့်များ ဖြစ်ရမည်။ ဇာတ်လမ်းအများအပြားကို စာသားအနည်းငယ်ထဲ အလွန်အကျွံမချုံ့ပါနှင့်။ အပိုင်းတစ်ခုစီတွင် အကြောင်းအရာကို နားလည်ရလွယ်အောင် သဘာဝကျသော စကားပြောဟန်ဖြင့် ရှင်းပြပါ။
    """.strip()

    resp = await asyncio.to_thread(client.models.generate_content, model=PREFERRED_MODELS[0], contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
    try:
        scenes = json.loads(resp.text.strip())
        if isinstance(scenes, dict) and "scenes" in scenes: scenes = scenes["scenes"]
    except Exception:
        scenes = [{"start": 0, "end": 5, "burmese": "စာသားရယူရာတွင် အမှားအယွင်းရှိပါသည်။"}]

    await tracker.update("🎙️ Charon AI မှ အသံသွင်းနေပါသည်...", force=True)
    joined = "\n\n".join(s["burmese"] for s in scenes)
    cfg = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon"))))
    tts_resp = await asyncio.to_thread(client.models.generate_content, model=PREFERRED_TTS_MODELS[0], contents=f"Read naturally:\n\n{joined}", config=cfg)
    
    tmp_raw = os.path.join(os.path.dirname(chunk_path), "gem.bin")
    tmp_pcm = os.path.join(os.path.dirname(chunk_path), "gem.pcm")
    with open(tmp_raw, "wb") as f: f.write(tts_resp.candidates[0].content.parts[0].inline_data.data)
    run_cmd([FFMPEG_BIN, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", tmp_raw, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), tmp_pcm], 60)
    master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    clip_dur = len(master) // len(scenes)
    clips = [master[i * clip_dur: (i + 1) * clip_dur] for i in range(len(scenes))]

    out_w, out_h = st1_dims
    shield_v = "crop=iw*0.96:ih*0.96,hflip,"
    color = "eq=contrast=1.1:saturation=1.15:gamma=1.05"

    video_slices = []
    for sc, clip in zip(scenes, clips):
        v_dur = len(clip) / SAMPLE_RATE
        t_dur = max(0.65, v_dur + 0.3)
        src_end = min(dur, sc["start"] + t_dur / 0.80)
        video_slices.append((sc["start"], src_end, t_dur))

    seg_files = []
    await tracker.update("🎬 ဗီဒီယို ပြင်ဆင်နေပါသည်...", force=True)
    for idx, (s_t, e_t, tgt_d) in enumerate(video_slices):
        raw_d = max(0.15, e_t - s_t); seg_path = os.path.join(os.path.dirname(chunk_path), f"seg_{idx:03d}.mp4")
        fc = f"[0:v]{shield_v}setpts=PTS-STARTPTS,fps=25,scale={out_w}:{out_h}:force_original_aspect_ratio=increase:flags=bicubic,crop={out_w}:{out_h},{color},format=yuv420p[vout]"
        await asyncio.to_thread(run_cmd, [FFMPEG_BIN, "-y", "-ss", f"{s_t:.3f}", "-t", f"{raw_d:.3f}", "-i", chunk_path, "-an", "-filter_complex", fc, "-map", "[vout]", "-t", f"{tgt_d:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", seg_path], 180)
        if os.path.exists(seg_path): seg_files.append(seg_path)

    concat_txt = os.path.join(os.path.dirname(chunk_path), "concat.txt")
    with open(concat_txt, "w") as f:
        for tp in seg_files: f.write(f"file '{tp}'\n")
    
    out_chunk = chunk_path.replace(".mp4", "_recap.mp4")
    await asyncio.to_thread(run_cmd, [FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c:v", "copy", out_chunk], 120)
    
    if os.path.exists(audio_path): os.remove(audio_path)
    if os.path.exists(tmp_raw): os.remove(tmp_raw)
    if os.path.exists(tmp_pcm): os.remove(tmp_pcm)
    if os.path.exists(chunk_path): os.remove(chunk_path)
    for tp in seg_files:
        if os.path.exists(tp): os.remove(tp)
        
    return out_chunk

async def handle_link_job(chat_id: int, reply_msg_id: int, url: str):
    job_id = f"j{secrets.token_hex(5)}"
    status_id = await send_text(chat_id, "⚡ Recap စတင်နေပါပြီ...", reply_to=reply_msg_id)
    work_dir = tempfile.mkdtemp(prefix="vid_", dir=WORK_ROOT)
    tracker = ProgressTracker(chat_id, status_id, job_id=job_id)
    
    try:
        await tracker.update("🔍 ဗီဒီယို အချိန်စစ်ဆေးနေပါသည်...", force=True)
        probe_cmd = ["yt-dlp", "--print", "%(duration)s", url]
        proc = await asyncio.create_subprocess_exec(*probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        total_dur = float(out.decode().strip()) if out else 0
        if total_dur <= 0: raise RuntimeError("Cannot get duration")

        chunks = []
        start = 0
        part_num = 1
        total_parts = int(np.ceil(total_dur / CHUNK_DURATION))

        while start < total_dur:
            if job_id in CANCELLED_JOBS: raise Exception("cancelled")
            end = min(start + CHUNK_DURATION, total_dur)
            
            await tracker.update(f"📥 ဒေါင်းလုဒ်ဆွဲနေပါသည်... (အပိုင်း {part_num}/{total_parts})\n📊 {fmt_progress_bar(int((start/total_dur)*100))}", force=True)
            
            chunk_path = os.path.join(work_dir, f"chunk_{part_num}.mp4")
            success = await download_chunk(url, start, end, chunk_path)
            if not success: raise RuntimeError(f"Download failed for part {part_num}")

            await tracker.update(f"✂️ အပိုင်း {part_num}/{total_parts} ကို Recap လုပ်နေပါသည်...", force=True)
            async with video_render_semaphore:
                recap_chunk = await build_chunk_recap(chunk_path, (1280, 720), start, end, tracker)
            
            if recap_chunk:
                chunks.append(recap_chunk)
            else:
                raise RuntimeError(f"Recap failed for part {part_num}")
                
            start = end
            part_num += 1
            
        await tracker.update("🔗 အပိုင်းတွေ ပေါင်းစပ်နေပါသည်...", force=True)
        final_out = os.path.join(work_dir, "final_recap.mp4")
        concat_final = os.path.join(work_dir, "final_concat.txt")
        with open(concat_final, "w") as f:
            for c in chunks: f.write(f"file '{c}'\n")
            
        run_cmd([FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", concat_final, "-c", "copy", final_out], 120)

        await tracker.update("📤 ဗီဒီယို ပို့ပေးနေပါသည်...\n📊 " + fmt_progress_bar(99), force=True)
        res = await tg_upload_media("sendVideo", {"chat_id": str(chat_id), "caption": "📹 Recap ပြီးပါပြီ။"}, "video", "recap.mp4", final_out, "video/mp4")
        
        if not res.get("ok"):
            await edit_text(chat_id, status_id, "⚠️ ပို့ရာတွင် အခက်အခဲရှိပါသည်။")
            
        try: await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
        except: pass

    except Exception as e:
        logger.exception("Job Error")
        await edit_text(chat_id, status_id, f"❌ အမှား: {type(e).__name__}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()

async def process_admin_update(update: dict):
    msg = update.get("message")
    if not msg: return
    chat_id = msg.get("chat", {}).get("id"); msg_id = msg.get("message_id")
    text = msg.get("text", "").strip()
    
    if text and text.startswith("http"):
        await handle_link_job(chat_id, msg_id, text)

app = FastAPI()
@app.post("/webhook")
async def telegram_webhook(request: Request):
    update = await request.json()
    asyncio.create_task(process_admin_update(update))
    return JSONResponse({"ok": True})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

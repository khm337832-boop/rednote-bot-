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

ADMIN_ID = int(os.environ.get("ADMIN_ID", "8430004524").strip() or "8430004524")
ALLOWED_USER_IDS = {8855067536}

SAMPLE_RATE = 24000
MAX_TG_BYTES = 45 * 1024 * 1024
CHUNK_DURATION = 300
PAUSE_AFTER_SCENE = 0.20

PUBLISH_CACHE_DIR = os.path.join(WORK_ROOT, "publish_cache")
os.makedirs(PUBLISH_CACHE_DIR, exist_ok=True)

CANCELLED_JOBS: set = set()
video_render_semaphore = asyncio.Semaphore(1)

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]
PREFERRED_TTS_MODELS = ["gemini-3.8-flash-tts", "gemini-2.5-flash-preview-tts"]
SPINNING_EMOJIS = ["⏳", "⌛", "⏳", "⌛"]


def get_anim_emoji(frame: int) -> str:
    return SPINNING_EMOJIS[frame % len(SPINNING_EMOJIS)]


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


async def send_text(chat_id: int, text: str, reply_to: int = None):
    res = await tg_api("sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "reply_to_message_id": reply_to,
    })
    return res.get("result", {}).get("message_id")


async def edit_text(chat_id: int, message_id: int, text: str):
    if not message_id:
        return
    try:
        await tg_api("editMessageText", {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        })
    except Exception:
        pass


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


def compute_output_dims(src_w: int, src_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1280, 720
    long_edge = max(src_w, src_h)
    scale = 1280.0 / float(long_edge)
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


async def download_chunk(url: str, start: float, end: float, out_path: str) -> float:
    cmd = [
        "yt-dlp", "--no-warnings", "--no-part",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", out_path, url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
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


async def synthesize_tts(scenes: list, work_dir: str) -> list[np.ndarray]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    joined = "\n\n".join(s["burmese"] for s in scenes)
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
    )
    tmp_raw = os.path.join(work_dir, "tts_raw.bin")
    tmp_pcm = os.path.join(work_dir, "tts.pcm")
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=PREFERRED_TTS_MODELS[0],
            contents=f"""
အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် သဘာဝကျသော စကားပြောဟန်ဖြင့် ဖတ်ပါ။
အသံသည် နက်ရှိုင်းသော (deep, low), ရင်အုပ်သံပါသော, ပဲ့တင်သံမပါသော အသံဖြစ်ရမည်။
ဇာတ်လမ်းကို ကိုယ်တိုင်ကြုံတွေ့ခဲ့သော အဓိကဇာတ်ကောင်က ပြန်ပြောပြနေသလို အသံသွင်းပါ။
အသံကို ရှင်းလင်း၊ နွေးထွေးပြီး ခံစားချက်ပါအောင် ဖတ်ပါ။
စာကြောင်းအဆုံးနှင့် ပုဒ်ဖြတ်အမှတ်အသားများတွင် သဘာဝကျကျ ခဏရပ်ပါ။
အလွန်မြန်မြန်၊ တစ်သံတည်း သို့မဟုတ် စက်ရုပ်ဆန်ဆန် မဖတ်ပါနှင့်။
စာသားတွင် မပါသော စကားလုံးများကို မထည့်ပါနှင့်။ နာမည်များကို သေချာရှင်းလင်းစွာ အသံထွက်ပါ။

{joined}
""",
            config=cfg,
        )
        data = resp.candidates[0].content.parts[0].inline_data.data
        with open(tmp_raw, "wb") as f:
            f.write(data)

        deep_voice_filter = (
            "highpass=f=55,"
            "asetrate=22080,aresample=24000,atempo=1.0870,"
            "equalizer=f=110:t=q:w=1.0:g=4.5,"
            "equalizer=f=230:t=q:w=1.1:g=2.5,"
            "equalizer=f=6500:t=q:w=1.0:g=-1.5,"
            "loudnorm=I=-16:TP=-1.5:LRA=11"
        )

        run_cmd([
            FFMPEG_BIN, "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", tmp_raw,
            "-af", deep_voice_filter,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            tmp_pcm,
        ], 120)

        if os.path.exists(tmp_pcm) and os.path.getsize(tmp_pcm) > 200:
            master = np.fromfile(tmp_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return split_pcm_by_weights(master, scenes)
    except Exception as e:
        logger.error(f"TTS error: {e}")
    finally:
        for p in (tmp_raw, tmp_pcm):
            if os.path.exists(p):
                os.remove(p)
    return [np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32) for _ in scenes]


async def extract_story_summary(scenes: list, client) -> dict:
    if not scenes:
        return {"summary": "", "characters": [], "last_scene": "", "protagonist": ""}
    all_text = "\n".join(s["burmese"] for s in scenes)
    prompt = f"""
အောက်ပါ မြန်မာဇာတ်လမ်းအပိုင်းကို ဖတ်ပြီး နောက်အပိုင်းအတွက် အကျဉ်းချုပ် ထုတ်ပေးပါ။
ဇာတ်ကောင်နာမည်များ၊ အဓိကဇာတ်ကောင် (protagonist)၊ အဓိကဖြစ်ရပ်များ၊ နောက်ဆုံးဖြစ်ပျက်နေသည့် အခြေအနေကို ထည့်ပါ။

ဇာတ်လမ်းအပိုင်း:
{all_text}

Return JSON:
{{"characters": ["နာမည်1"], "protagonist": "အဓိကဇာတ်ကောင်နာမည်", "summary": "အဓိကအကြောင်းအရာ ၂-၃ ကြောင်း", "last_scene": "နောက်ဆုံးအခြေအနေ"}}
    """.strip()
    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=PREFERRED_MODELS[0],
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(resp.text.strip())
    except Exception as e:
        logger.warning(f"Summary error: {e}")
        return {
            "characters": [],
            "protagonist": "",
            "summary": all_text[-500:],
            "last_scene": scenes[-1]["burmese"],
        }


async def build_chunk_recap(
    chunk_path: str,
    chunk_dur: float,
    global_start: float,
    global_end: float,
    story_context: dict,
    chunk_index: int,
    tracker: ProgressTracker,
):
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_fmt = fmt_time(global_start)
    end_fmt = fmt_time(global_end)

    target_chars = int(chunk_dur * 6.5)
    scene_count = max(15, int(chunk_dur / 8))
    chars_per_scene = max(80, target_chars // scene_count)
    protagonist = story_context.get("protagonist", "") if chunk_index > 1 else ""

    if chunk_index == 1:
        context_section = "🔵 ဤသည် ဇာတ်လမ်း၏ အစဖြစ်သည်။ အဓိကဇာတ်ကောင်က ဇာတ်လမ်းကို ပထမလူအနေနဲ့ ပြန်ပြောပြပါ။"
    else:
        chars_str = ", ".join(story_context.get("characters", [])[:10])
        protag_line = f"- အဓိကဇာတ်ကောင် (မပြောင်းရ): {protagonist}\n" if protagonist else ""
        context_section = f"""
🔵 ယခင်ဖြစ်ပျက်ခဲ့သည့် ဇာတ်လမ်းအကျဉ်း —
{protag_line}- ဇာတ်ကောင်များ: {chars_str}
- အဓိကဖြစ်ရပ်များ: {story_context.get('summary', '')}
- နောက်ဆုံးအခြေအနေ: {story_context.get('last_scene', '')}

🔵 ရှေ့အပိုင်းနဲ့ ရှုထောင့်နဲ့ အဓိကဇာတ်ကောင် လုံးဝ မပြောင်းပါနှင့်။
"""

    prompt = f"""
သင်သည် မြန်မာဘာသာဖြင့် ရုပ်ရှင်ဇာတ်လမ်းကို ပြောပြသည့် ကျွမ်းကျင်သူ ဇာတ်ကြောင်းပြောသူဖြစ်သည်။
ပေးထားသော ဗီဒီယိုအပိုင်းကို ကြည့်ရှုပြီး အောက်ပါအတိုင်း ရေးပါ။

ဤဗီဒီယိုအပိုင်းသည် မူရင်းဗီဒီယို၏ {start_fmt} မှ {end_fmt} အထိ ဖြစ်သည်။
ဤအပိုင်း၏ ကြာချိန်မှာ {fmt_time(chunk_dur)} ဖြစ်သည်။

{context_section}

ဇာတ်ကြောင်းပြောဟန် (မဖြစ်မနေ) —
- ဇာတ်လမ်းကို အဓိကဇာတ်ကောင်ရဲ့ ရှုထောင့်ကနေ ပထမလူ (first-person) အနေနဲ့ ပြောပြပါ။
- အဓိကဇာတ်ကောင်ရဲ့ လိင်/နာမည် သိရင် ပထမဆုံးစာကြောင်းကို "ကျွန်တော်ကတော့ [နာမည်] ပါ" သို့မဟုတ် "ကျွန်မကတော့ [နာမည်] ပါ" ဆိုပြီး စပါ။ နာမည် မသိရင် နာမည်မတီထွင်ပါနှင့်။
- "အဲဒီအချိန်မှာတော့...", "ဒါပေမယ့် ကျွန်တော် မသိသေးတာက...", "အဲဒီနောက်မှာတော့..." စတဲ့ သဘာဝကျတဲ့ ဆက်စပ်စကားလုံးတွေကို သင့်တင့်သလို သုံးပါ။ စကားလုံးတစ်လုံးတည်း ထပ်ခါတလဲလဲ မသုံးပါနှင့်။
- လူပြောသလို ရိုးရှင်း၊ နားလည်လွယ်၊ ခံစားချက်ပါတဲ့ ဘာသာစကားနဲ့ ရေးပါ။ စက်ရုပ်ဆန်တဲ့ သို့မဟုတ် တရားဝင်လွန်းတဲ့ စာကြောင်းတွေ မသုံးပါနှင့်။
- ဇာတ်ကောင်ရဲ့ ခံစားချက်၊ မြင်တဲ့အရာ၊ လုပ်တဲ့အပြုအမူကို ဗီဒီယိုထဲက သတင်းအချက်အလက်အပေါ် အခြေခံပြီး ပြောပါ။ ဗီဒီယိုမှာ မပြထားတဲ့ အတွေး၊ ခံစားချက်တွေကို မတီထွင်ပါနှင့်။
- မူရင်းစကားပြောတွေကို တိုက်ရိုက်ကူးချမနေပါနှင့်။ ဒါပေမယ့် အဲဒီစကားရဲ့ အဓိပ္ပာယ်ကို အဓိကဇာတ်ကောင်က သူ့အမြင်နဲ့ ပြောပြနေသလို ဖော်ပြပါ။
- ဇာတ်လမ်းဖြစ်ရပ်တွေကို မကျော်ဘဲ မူရင်းအစဉ်အတိုင်း ဆက်ပြောပါ။ အပိုင်းတွေကြား ရှုထောင့်နဲ့ အဓိကဇာတ်ကောင် မပြောင်းပါနှင့်။

အဓိကရည်မှန်းချက် —
ဤအပိုင်းကို "အတိုချုပ် recap" မလုပ်ပါနှင့်။ မူရင်းဗီဒီယိုထဲက ဇာတ်လမ်းဖြစ်ရပ်များကို အစဉ်လိုက်၊ နီးပါးပြည့်စုံအောင် ပြန်ပြောပြပါ။
ဇာတ်လမ်းနှင့် မသက်ဆိုင်သော တိတ်ဆိတ်ချိန်၊ ထပ်ခါတလဲလဲ မြင်ကွင်း၊ အကြောင်းမဲ့ နားချိန်၊ မလိုအပ်သော အစ/အဆုံးပိုင်းများကိုသာ ဖယ်ရှားပါ။

လိုက်နာရန် —
၁။ မူရင်းဗီဒီယိုထဲတွင် ဖြစ်ပျက်သည့် အစဉ်အတိုင်း ပြောပါ။ အစဉ်ကို မပြောင်းပါနှင့်။
၂။ ဇာတ်ကောင်များ ဘယ်သူဘယ်ဝါ၊ ဘယ်လိုဆက်နွယ်မှု၊ ဘာကြောင့် ထိုလုပ်ရပ်ကို လုပ်သည်၊ ဘာဖြစ်သွားသည် ဆိုတာကို ရှင်းပြပါ။
၃။ မူရင်းတွင် မပါသော ဖြစ်ရပ်၊ ဇာတ်ကောင်၊ စကားပြောချက်၊ အကြောင်းရင်း သို့မဟုတ် အဆုံးသတ်ကို လုံးဝ မတီထွင်ပါနှင့်။
၄။ ဤအပိုင်းအဆုံးတွင် ဇာတ်လမ်းဆက်နေပါက နောက်ပိုင်းတွင် ဘာဖြစ်မည်ကို မခန့်မှန်းပါနှင့်။
၅။ တရုတ်နှင့် နိုင်ငံခြားအမည်များကို မြန်မာအသံထွက်ဖြင့် တစ်သမတ်တည်း ရေးပါ။
၆။ ဤအပိုင်း၏ ဇာတ်လမ်းဖြစ်ရပ်များ နီးပါးအားလုံး ပါဝင်အောင် စာသားရေးပါ။
၇။ အနည်းဆုံး ခန်း {scene_count} ခန်း ရေးပါ။
၈။ ခန်းတစ်ခန်းလျှင် အနည်းဆုံး {chars_per_scene} လုံး ပါဝင်ပါ။
၉။ နောက်ဆုံးတွင် JSON object တစ်ခုတည်းကိုသာ ပြန်ပေးပါ။

JSON ပုံစံ —
{{
  "protagonist": "အဓိကဇာတ်ကောင်နာမည် (မသိရင် ကွက်လပ်ထားပါ)",
  "cuts": [
    {{"start": 12.0, "end": 15.0, "reason": "ထပ်နေသောမြင်ကွင်း"}}
  ],
  "scenes": [
    {{"start": 0.0, "end": 12.0, "burmese": "ဇာတ်ကြောင်း"}},
    {{"start": 15.0, "end": 30.0, "burmese": "ဇာတ်ကြောင်း"}}
  ]
}}

cuts သည် မလိုအပ်သော အပိုင်းများသာ ဖြစ်ရမည်။ မူရင်းဗီဒီယို၏ ၈% ထက် မပိုစေရ။
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
        elif isinstance(parsed, list):
            scenes = parsed
    except Exception as e:
        logger.error(f"AI script error: {e}")

    total_chars = sum(len(s.get("burmese", "")) for s in scenes)
    logger.info(f"Chunk {chunk_index}: {len(scenes)} scenes, {total_chars} chars, target {target_chars}")

    if total_chars < target_chars * 0.9:
        await tracker.update("⚠️ စာသားနည်းနေပါသည်။ ထပ်ဖြည့်နေပါသည်...", force=True)
        extra_prompt = f"""
မူရင်းဗီဒီယိုကို ကြည့်ပြီး ဇာတ်လမ်းကို ပိုမိုအသေးစိတ်အောင် ထပ်မံရေးပေးပါ။
ယခင်ရေးထားသော script ပါ အကြောင်းအရာများကို ထပ်မရေးပါနှင့်။
လွတ်နေသော အရေးကြီးဖြစ်ရပ်များ၊ အကြောင်းရင်းနှင့် အကျိုးဆက်များကို ထပ်ဖြည့်ပါ။
မူရင်းဗီဒီယိုတွင် မပါသော အကြောင်းအရာများကို မတီထွင်ပါနှင့်။
ပထမလူ (first-person) ရှုထောင့်ကို ဆက်ထိန်းသိမ်းပါ။
အနည်းဆုံး {max(100, target_chars - total_chars)} လုံး ထပ်ဖြည့်ပါ။

ယခင် script:
{json.dumps(scenes, ensure_ascii=False)}

JSON object ကိုသာ ပြန်ပေးပါ။
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
            elif isinstance(extra, list):
                scenes.extend(extra)
            logger.info(f"Added extra scenes: {len(scenes)}")
        except Exception as e:
            logger.warning(f"Extra script error: {e}")

    try:
        await asyncio.to_thread(client.files.delete, name=video_file.name)
        logger.info("Gemini ပေါ်ရှိ ယာယီဗီဒီယိုဖိုင် ဖျက်ပြီးပါပြီ။")
    except Exception as e:
        logger.warning(f"Gemini ဗီဒီယိုဖိုင် ဖျက်မရပါ: {e}")

    if not scenes:
        scenes = [{"start": 0, "end": 5, "burmese": "စာသားရယူရာတွင် အမှားအယွင်းရှိပါသည်။"}]

    keep_ranges = make_keep_ranges(chunk_dur, cuts, max_remove_ratio=0.08)
    kept_duration = sum(end - start for start, end in keep_ranges)

    if kept_duration < chunk_dur * 0.90:
        logger.warning(f"Kept duration {kept_duration:.1f}s < 90% of {chunk_dur:.1f}s, forcing keep-all")
        keep_ranges = [(0.0, chunk_dur)]
        kept_duration = chunk_dur

    logger.info(f"Keep ranges: {len(keep_ranges)}, kept {kept_duration:.1f}s / {chunk_dur:.1f}s")

    await tracker.update("🎙️ Charon AI မှ အသံသွင်းနေပါသည်...", force=True)
    clips = await synthesize_tts(scenes, os.path.dirname(chunk_path))

    out_w, out_h = compute_output_dims(*probe_video_info(chunk_path)[1:])
    shield_v = "crop=iw*0.96:ih*0.96,hflip,"
    color = "eq=contrast=1.08:saturation=1.12:gamma=1.02"

    master_pcm_path = os.path.join(os.path.dirname(chunk_path), "master_audio.pcm")
    with open(master_pcm_path, "wb") as pcm_f:
        for clip in clips:
            np.clip(clip * 32767.0, -32768, 32767).astype(np.int16).tofile(pcm_f)
            pause_samples = int(PAUSE_AFTER_SCENE * SAMPLE_RATE)
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
            f"{color},format=yuv420p"
        )
        result = await asyncio.to_thread(run_cmd, [
            FFMPEG_BIN, "-y",
            "-ss", f"{start:.3f}",
            "-i", chunk_path,
            "-t", f"{segment_duration:.3f}",
            "-an",
            "-vf", fc,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            segment_path,
        ], 300)

        if result.returncode != 0 or not os.path.exists(segment_path):
            raise RuntimeError(f"ဗီဒီယိုအပိုင်း {index + 1} ဖန်တီး၍မရပါ။")
        seg_files.append(segment_path)

    concat_txt = os.path.join(os.path.dirname(chunk_path), "concat.txt")
    with open(concat_txt, "w") as f:
        for tp in seg_files:
            f.write(f"file '{tp}'\n")

    video_only = os.path.join(os.path.dirname(chunk_path), "video_only.mp4")
    await asyncio.to_thread(run_cmd, [
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        video_only,
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
    job_id = f"j{secrets.token_hex(5)}"
    status_id = await send_text(chat_id, "⚡ Recap စတင်နေပါပြီ...", reply_to=reply_msg_id)
    work_dir = tempfile.mkdtemp(prefix="vid_", dir=WORK_ROOT)
    tracker = ProgressTracker(chat_id, status_id, job_id=job_id)

    story_context = {"characters": [], "summary": "", "last_scene": "", "protagonist": ""}

    try:
        await tracker.update("🔍 ဗီဒီယို အချိန်စစ်ဆေးနေပါသည်...", force=True)
        total_dur = await get_source_duration(url)
        if total_dur <= 0:
            raise RuntimeError("ဗီဒီယိုကြာချိန် မရပါ။")

        logger.info(f"Source duration: {total_dur:.1f}s = {fmt_time(total_dur)}")
        await tracker.update(f"📹 မူရင်းကြာချိန်: {fmt_time(total_dur)}\n📥 ဒေါင်းလုဒ် စတင်နေပါသည်...", force=True)

        chunks = []
        start = 0
        part_num = 1
        total_parts = int(np.ceil(total_dur / CHUNK_DURATION))

        while start < total_dur and part_num <= total_parts + 5:
            if job_id in CANCELLED_JOBS:
                raise Exception("cancelled")

            end = min(start + CHUNK_DURATION, total_dur)
            await tracker.update(
                f"📥 ဒေါင်းလုဒ်ဆွဲနေပါသည်... (အပိုင်း {part_num}/{total_parts})\n"
                f"📊 {fmt_progress_bar(int((start / total_dur) * 100))}",
                force=True,
            )

            chunk_path = os.path.join(work_dir, f"chunk_{part_num}.mp4")
            actual_dur = await download_chunk(url, start, end, chunk_path)
            if actual_dur <= 0:
                raise RuntimeError(f"အပိုင်း {part_num} ဒေါင်းလုဒ် မရပါ။")

            await tracker.update(
                f"✂️ အပိုင်း {part_num}/{total_parts} ကို Recap လုပ်နေပါသည်...",
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
                )

            if not recap_chunk:
                raise RuntimeError(f"အပိုင်း {part_num} Recap မရပါ။")

            chunks.append(recap_chunk)
            story_context = new_summary
            logger.info(
                f"Story context updated: "
                f"{len(story_context.get('characters', []))} characters, "
                f"protagonist='{story_context.get('protagonist', '')}', "
                f"{len(story_context.get('summary', ''))} summary chars"
            )

            start += actual_dur
            part_num += 1

            if actual_dur < CHUNK_DURATION * 0.9:
                break

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

        parts = await split_for_telegram(final_out, work_dir)
        for i, part in enumerate(parts):
            caption = f"📹 Recap ပြီးပါပြီ။ ({fmt_time(final_dur)})"
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

        try:
            await tg_api("deleteMessage", {"chat_id": chat_id, "message_id": status_id})
        except Exception:
            pass

    except Exception as e:
        logger.exception("Job Error")
        await edit_text(chat_id, status_id, f"❌ အမှား: {type(e).__name__} - {str(e)[:120]}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


async def process_admin_update(update: dict):
    msg = update.get("message")
    if not msg:
        return
    chat_id = msg.get("chat", {}).get("id")
    msg_id = msg.get("message_id")
    text = (msg.get("text") or "").strip()

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

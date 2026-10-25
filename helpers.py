import os
import re
import time
import asyncio
import subprocess
import signal

SPINNING_EMOJIS = ["⏳", "⌛", "⏳", "⌛"]

def get_anim_emoji(frame: int) -> str:
    return SPINNING_EMOJIS[frame % len(SPINNING_EMOJIS)]

def fmt_file_size(mb: float) -> str:
    return f"{mb:.1f} MB" if mb < 1024 else f"{mb/1024:.2f} GB"

async def download_video_never_stop(url: str, out_path: str, tracker=None) -> bool:
    for attempt in range(5):
        if tracker: await tracker.update(f"📥 ဒေါင်းလုဒ်ဆွဲနေပါသည်... (ကြိုးစားမှု {attempt+1}/5)", force=True)
        cmd = ["yt-dlp", "--no-warnings", "--no-part", "-f", "bestvideo+bestaudio/best", "--merge-output-format", "mp4", "-o", out_path, url]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000: return True
        await asyncio.sleep(3)
    return False

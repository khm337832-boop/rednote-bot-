"""Timed Burmese subtitles: validated cues and libass burn-in.

Coordinates and font size are percentages of the video frame. This module
is independent of the recap pipeline and does not translate or accept jobs.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SubtitleStyle:
    x: float = 50.0
    y: float = 90.0
    size: float = 4.5

    def validate(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y, self.size)):
            raise ValueError("Subtitle position and size must be finite")
        if not (0 <= self.x <= 100 and 0 <= self.y <= 100 and 1 <= self.size <= 15):
            raise ValueError("Subtitle position or size out of range")


def _ass_time(value: float) -> str:
    # Round once so a boundary such as 59.999 seconds carries correctly.
    cs = round(value * 100)
    hours, rem = divmod(cs, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, centiseconds = divmod(rem, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


def _ass_text(text: str) -> str:
    # Do not allow generated/source text to inject ASS override tags.
    text = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
    return "\\N".join(line.strip() for line in text.splitlines())


def make_ass(cues: list[SubtitleCue], width: int, height: int, style: SubtitleStyle, *, duration: float | None = None) -> str:
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        raise ValueError("Invalid video dimensions")
    style.validate()
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid video duration")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Myanmar,{height * style.size / 100:.2f},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{max(1, height * .0025):.2f},1,2,{int(width * .04)},{int(width * .04)},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    if not cues:
        raise ValueError("No subtitle cues")
    events = []
    previous = -1.0
    for cue in cues:
        if not all(math.isfinite(v) for v in (cue.start, cue.end)) or cue.start < 0 or cue.end <= cue.start or cue.start < previous:
            raise ValueError("Invalid or out-of-order subtitle timing")
        if duration is not None and cue.end > duration + 0.01:
            raise ValueError("Subtitle extends beyond video")
        if not cue.text.strip():
            raise ValueError("Empty subtitle")
        previous = cue.start
        x, y = round(width * style.x / 100), round(height * style.y / 100)
        events.append(f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Default,,0,0,0,,{{\\pos({x},{y})}}{_ass_text(cue.text)}")
    return header + "\n".join(events) + "\n"


def burn_subtitles(source: str, output: str, cues: list[SubtitleCue], width: int, height: int, style: SubtitleStyle, *, duration: float | None = None) -> None:
    """Burn subtitles and keep all original audio; never publish partial output."""
    ass = make_ass(cues, width, height, style, duration=duration)
    out = Path(output).resolve()
    if out == Path(source).resolve():
        raise ValueError("Output must differ from source")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="subtitle-") as tmp:
        # Safe generated path: filter argument does not contain user-provided paths.
        ass_path = Path(tmp) / "captions.ass"
        ass_path.write_text(ass, encoding="utf-8")
        partial = out.parent / ("." + out.stem + ".partial.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-i", str(source),
                "-vf", f"ass={ass_path}", "-map", "0:v:0", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "copy", "-movflags", "+faststart", str(partial),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            os.replace(partial, out)
        finally:
            partial.unlink(missing_ok=True)

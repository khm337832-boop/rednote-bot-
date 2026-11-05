"""FFmpeg rendering — မြန်မာစာတန်း burn-in (ASS/libass), Blur, Logo, SRT.

ဒီဇိုင်း ဆုံးဖြတ်ချက်များ
------------------------
* မြန်မာစာအတွက် ``drawtext`` ကို **မသုံး**ပါ။ စာလုံး shaping မှားနိုင်သည်။
  ASS ဖိုင် ထုတ်ပြီး libass ၏ ``ass`` filter ဖြင့် burn-in လုပ်သည်။
  ဖောင့်ကို ``fontsdir`` ဖြင့် တိုက်ရိုက် ညွှန်သည် (system font မလိုအပ်)။
* Blur ကို ``crop → boxblur → overlay`` ဖြင့် လုပ်သည်။
* Logo ကို ``scale → rotate (transparent) → colorchannelmixer (opacity) → overlay``။
* စာတန်း၊ Logo အားလုံးကို ဗီဒီယိုအရွယ်အစားနှင့် အချိုးကျ (fraction 0–1)
  အနေဖြင့် သိမ်းထားသဖြင့် resolution မည်သို့ပြောင်းလဲပြောင်း နေရာမှန်သည်။
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import config as cfg
from . import myanmar_text as mt

ProgressFn = Callable[[float | None, str], None]
CancelFn = Callable[[], bool]


# ---------------------------------------------------------------------------
# Video probing (ffprobe when available, otherwise parse `ffmpeg -i`)
# ---------------------------------------------------------------------------


@dataclass
class VideoMeta:
    width: int
    height: int
    duration: float
    fps: float = 0.0
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    container: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 3),
            "fps": round(self.fps, 3),
            "has_audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "container": self.container,
        }


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Video:\s*([A-Za-z0-9_\-]+).*?,\s*(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUDIO_RE = re.compile(r"Audio:\s*([A-Za-z0-9_\-]+)")


def _find_ffprobe(settings: Any = None) -> str | None:
    explicit = os.environ.get("FFPROBE_BINARY") or getattr(settings, "ffprobe_binary", "")
    if explicit and Path(explicit).exists():
        return explicit
    return shutil.which("ffprobe")


def probe_video(path: str | Path, ffmpeg: str | None = None) -> VideoMeta:
    path = str(path)
    ffprobe = _find_ffprobe()
    if ffprobe:
        try:
            out = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if out.returncode == 0:
                import json

                data = json.loads(out.stdout or "{}")
                video = next(
                    (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
                )
                audio = next(
                    (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
                )
                if video:
                    fps = 0.0
                    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
                    if "/" in rate:
                        num, den = rate.split("/", 1)
                        try:
                            fps = float(num) / float(den) if float(den) else 0.0
                        except (ValueError, ZeroDivisionError):
                            fps = 0.0
                    duration = 0.0
                    for candidate in (
                        video.get("duration"),
                        data.get("format", {}).get("duration"),
                    ):
                        try:
                            if candidate and float(candidate) > 0:
                                duration = float(candidate)
                                break
                        except (TypeError, ValueError):
                            continue
                    return VideoMeta(
                        width=int(video.get("width") or 0),
                        height=int(video.get("height") or 0),
                        duration=duration,
                        fps=fps,
                        has_audio=audio is not None,
                        video_codec=str(video.get("codec_name") or ""),
                        audio_codec=str((audio or {}).get("codec_name") or ""),
                        container=str(data.get("format", {}).get("format_name") or ""),
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    # Fallback: parse ffmpeg's stream summary.
    binary = ffmpeg or (cfg.get_ffmpeg_info().path)
    if not binary:
        raise RuntimeError("FFmpeg/ffprobe မတွေ့သဖြင့် ဗီဒီယိုကို စစ်ဆေး၍ မရပါ")
    result = subprocess.run(
        [binary, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    stderr = result.stderr or ""
    duration = 0.0
    match = _DURATION_RE.search(stderr)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    video_match = _VIDEO_RE.search(stderr)
    if not video_match:
        raise RuntimeError("ဗီဒီယို stream မတွေ့ပါ — ဖိုင် ပျက်စီးနေနိုင်သည်")
    codec, width, height = video_match.group(1), int(video_match.group(2)), int(video_match.group(3))
    fps_match = _FPS_RE.search(stderr)
    audio_match = _AUDIO_RE.search(stderr)
    return VideoMeta(
        width=width,
        height=height,
        duration=duration,
        fps=float(fps_match.group(1)) if fps_match else 0.0,
        has_audio=audio_match is not None,
        video_codec=codec,
        audio_codec=audio_match.group(1) if audio_match else "",
    )


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

DEFAULT_STYLE: dict[str, Any] = {
    "fontSize": 0.055,
    "x": 0.5,
    "y": 0.86,
    "width": 0.82,
    "color": "#FFFFFF",
    "outlineColor": "#000000",
    "outlineWidth": 0.08,
    "opacity": 1.0,
    "bold": True,
    "rotation": 0.0,
    "background": False,
    "backgroundColor": "#000000",
    "backgroundOpacity": 0.0,
}


def normalise_style(style: dict[str, Any] | None, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_STYLE)
    if defaults:
        merged.update({k: v for k, v in defaults.items() if v is not None})
    if style:
        merged.update({k: v for k, v in style.items() if v is not None})
    merged["fontSize"] = float(merged.get("fontSize") or 0.055)
    merged["x"] = float(merged.get("x", 0.5))
    merged["y"] = float(merged.get("y", 0.86))
    merged["width"] = float(merged.get("width", 0.82))
    merged["outlineWidth"] = float(merged.get("outlineWidth", 0.08))
    merged["opacity"] = min(1.0, max(0.0, float(merged.get("opacity", 1.0))))
    merged["rotation"] = float(merged.get("rotation", 0.0) or 0.0)
    merged["bold"] = bool(merged.get("bold", True))
    return merged


def _hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    """#RRGGBB → &HAABBGGRR (ASS uses BGR order)."""
    value = str(hex_color or "#FFFFFF").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) == 8:  # #RRGGBBAA → ignore embedded alpha
        value = value[:6]
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        value = "FFFFFF"
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{blue.upper()}{green.upper()}{red.upper()}"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    centiseconds = round(seconds * 100)
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    text = mt.normalize_text(text)
    # Never let user text inject ASS override tags or break the file.
    text = (
        text.replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
        .replace("\u00a0", " ")
    )
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\\N".join(lines)


_FONT_CACHE: dict[tuple[str, int], Any] = {}


def _load_font(path: Path, size_px: int, bold: bool):
    from PIL import ImageFont

    key = (str(path), size_px)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font_path = path if path.exists() else (cfg.FONT_BOLD if bold else cfg.FONT_REGULAR)
    kwargs: dict[str, Any] = {}
    try:  # raqm gives harfbuzz shaping when the Pillow build ships it
        from PIL import features

        if features.check("raqm"):
            kwargs["layout_engine"] = ImageFont.Layout.RAQM
    except Exception:
        kwargs = {}
    try:
        font = ImageFont.truetype(str(font_path), size_px, **kwargs)
    except Exception:
        font = ImageFont.truetype(str(cfg.FONT_REGULAR), size_px)
    _FONT_CACHE[key] = font
    return font


def measure_text(text: str, size_px: int, bold: bool = False) -> float:
    font = _load_font(cfg.FONT_BOLD if bold else cfg.FONT_REGULAR, max(4, int(size_px)), bold)
    try:
        return float(font.getlength(text))
    except Exception:  # pragma: no cover - very old Pillow
        return len(text) * size_px * 0.6


def wrap_for_box(text: str, box_width_px: float, font_size_px: float, bold: bool = False) -> list[str]:
    """Wrap ASS text into lines using real font metrics (fallback: char budget)."""
    if box_width_px <= font_size_px:
        return [line for line in text.split("\n") if line.strip()] or [text.strip()]
    try:
        return mt.wrap_text(
            text,
            max(1.0, box_width_px),
            lambda s: measure_text(s, font_size_px, bold),
        )
    except Exception:  # pragma: no cover - defensive
        capacity = max(4, int(box_width_px / (font_size_px * 0.62)))
        return mt.wrap_by_char_budget(text, capacity)


def segment_style(segment: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    style = dict(segment.get("style") or {})
    for key in ("x", "y", "width", "fontSize", "rotation", "opacity"):
        if key in segment and segment[key] is not None and key not in style:
            style[key] = segment[key]
    return normalise_style(style, defaults)


def build_ass(
    segments: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    defaults: dict[str, Any] | None = None,
    font_family: str = cfg.FONT_FAMILY,
    duration: float = 0.0,
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions")
    header = f"""[Script Info]
; Generated by Myanmar Subtitle Editor
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Myanmar,{font_family},64,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for index, segment in enumerate(segments):
        try:
            start = max(0.0, float(segment.get("start", 0.0)))
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end <= start:
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        style = segment_style(segment, defaults)
        font_size_px = max(8.0, style["fontSize"] * height)
        box_width_px = max(16.0, style["width"] * width)
        wrapped = wrap_for_box(text, box_width_px, font_size_px, style["bold"])
        escaped = _escape_ass_text("\n".join(wrapped))
        if not escaped:
            continue
        alpha = int(round((1.0 - style["opacity"]) * 255))
        outline_px = max(0.0, style["outlineWidth"] * font_size_px)
        pos_x = round(style["x"] * width)
        pos_y = round(style["y"] * height)
        rotation = style["rotation"]  # ASS \frz and Konva both rotate clockwise
        overrides = [
            f"\\an5",
            f"\\pos({pos_x},{pos_y})",
            f"\\fs{font_size_px:.1f}",
            f"\\b{1 if style['bold'] else 0}",
            f"\\c{_hex_to_ass(style['color'])}",
            f"\\3c{_hex_to_ass(style['outlineColor'])}",
            f"\\bord{outline_px:.2f}",
            "\\shad0",
            "\\q2",
        ]
        if alpha > 0:
            overrides.append(f"\\alpha&H{alpha:02X}&")
        if abs(rotation) > 0.01:
            overrides.append(f"\\frz{rotation:.2f}")
        if style.get("background"):
            back_alpha = int(round((1.0 - float(style.get("backgroundOpacity", 0.5))) * 255))
            overrides.append("\\bord0")
            overrides.append(f"\\4c{_hex_to_ass(style.get('backgroundColor', '#000000'))}")
            overrides.append(f"\\4a&H{back_alpha:02X}&")
        dialogue = (
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Myanmar,,0,0,0,,"
            + "{" + "".join(overrides) + "}" + escaped
        )
        lines.append(dialogue + "\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Filter graph
# ---------------------------------------------------------------------------


def _even(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def _clamp_box(box: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x = float(box.get("x", 0.0)) * width
    y = float(box.get("y", 0.0)) * height
    w = float(box.get("w", box.get("width", 0.2))) * width
    h = float(box.get("h", box.get("height", 0.1))) * height
    x = min(max(0.0, x), width - 4)
    y = min(max(0.0, y), height - 4)
    w = max(8.0, min(w, width - x))
    h = max(8.0, min(h, height - y))
    # yuv420 needs even crop offsets/sizes.
    return _even(x) if x > 0 else 0, _even(y) if y > 0 else 0, _even(w), _even(h)


def blur_radius(intensity: float, box_w: int, box_h: int) -> int:
    """Map UI intensity (1–20) to a boxblur radius in pixels.

    Uses the same formula as the browser preview so both look similar:
    ``intensity * (min(w, h) / 720) * 1.2``.
    """
    scale = min(box_w, box_h) / 720.0
    radius = int(round(float(intensity) * max(0.35, scale) * 1.2))
    return max(1, min(radius, 120))


def build_filter_script(
    *,
    width: int,
    height: int,
    blur_boxes: Sequence[dict[str, Any]],
    logos: Sequence[dict[str, Any]],
    logo_input_index: Sequence[int],
    ass_name: str | None = "subs.ass",
    has_audio: bool = True,
    scale_to_height: int = 0,
) -> str:
    """Build the ``-filter_complex`` script (one statement per line)."""
    lines: list[str] = []
    out_w, out_h = width, height
    if scale_to_height and height > scale_to_height:
        out_w = _even(width * scale_to_height / height)
        out_h = _even(scale_to_height)
        lines.append(f"[0:v]scale={out_w}:{out_h}:flags=lanczos[vbase0]")
        current = "vbase0"
    else:
        current = "0:v"

    active_blurs = [
        box
        for box in blur_boxes
        if float(box.get("intensity", 0)) > 0
        and float(box.get("w", box.get("width", 0)) or 0) > 0
        and float(box.get("h", box.get("height", 0)) or 0) > 0
    ]
    if active_blurs:
        count = len(active_blurs)
        labels = [f"bzsrc{i}" for i in range(count)]
        split_out = "".join(f"[{label}]" for label in labels)
        lines.append(f"[{current}]split={count + 1}[vbasesrc]{split_out}")
        base_label = "vbasesrc"
        for index, box in enumerate(active_blurs):
            x, y, w, h = _clamp_box(box, out_w, out_h)
            radius = blur_radius(float(box.get("intensity", 8)), w, h)
            lines.append(
                f"[bzsrc{index}]crop={w}:{h}:{x}:{y},boxblur=luma_radius={radius}:"
                f"luma_power=2:chroma_radius={max(1, radius // 2)}:chroma_power=2[bz{index}]"
            )
            next_label = f"vblur{index}"
            lines.append(f"[{base_label}][bz{index}]overlay={x}:{y}:format=auto[{next_label}]")
            base_label = next_label
        current = base_label

    for index, (logo, input_index) in enumerate(zip(logos, logo_input_index)):
        target_w = max(2, int(round(float(logo.get("w", 0.15)) * out_w)))
        target_h = max(2, int(round(float(logo.get("h", 0.15)) * out_h)))
        target_w, target_h = _even(target_w), _even(target_h)
        opacity = min(1.0, max(0.0, float(logo.get("opacity", 1.0))))
        rotation = float(logo.get("rotation", 0.0) or 0.0)
        chain = [
            "format=rgba",
            f"scale={target_w}:{target_h}:flags=lanczos",
        ]
        if abs(rotation) > 0.01:
            angle = rotation * math.pi / 180.0
            chain.append(
                f"rotate={angle:.5f}:ow=hypot(iw\\,ih):oh=hypot(iw\\,ih):fillcolor=none:c=none"
            )
        if opacity < 0.999:
            chain.append(f"colorchannelmixer=aa={opacity:.3f}")
        logo_label = f"logo{index}"
        lines.append(f"[{input_index}:v]" + ",".join(chain) + f"[{logo_label}]")
        center_x = float(logo.get("x", 0.5)) * out_w + target_w / 2.0
        center_y = float(logo.get("y", 0.5)) * out_h + target_h / 2.0
        rotated_w = math.hypot(target_w, target_h) if abs(rotation) > 0.01 else target_w
        rotated_h = math.hypot(target_w, target_h) if abs(rotation) > 0.01 else target_h
        pos_x = int(round(center_x - rotated_w / 2.0))
        pos_y = int(round(center_y - rotated_h / 2.0))
        next_label = f"vlogo{index}"
        lines.append(
            f"[{current}][{logo_label}]overlay={pos_x}:{pos_y}:format=auto:repeatlast=1[{next_label}]"
        )
        current = next_label

    if ass_name:
        lines.append(f"[{current}]ass={ass_name}:fontsdir=fonts[vout]")
        current = "vout"
    lines.append(f"[{current}]format=yuv420p[vfinal]")
    return ";".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _copy_fonts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("NotoSansMyanmar-Regular.ttf", "NotoSansMyanmar-Bold.ttf"):
        source = cfg.FONTS_DIR / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def _tail(path: Path, limit: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


@dataclass
class RenderResult:
    output: Path
    duration: float
    elapsed: float
    meta: dict[str, Any] = field(default_factory=dict)


def render_video(
    *,
    video_path: str | Path,
    segments: Sequence[dict[str, Any]],
    output_path: str | Path,
    blur_boxes: Sequence[dict[str, Any]] = (),
    logos: Sequence[dict[str, Any]] = (),
    duration: float = 0.0,
    style_defaults: dict[str, Any] | None = None,
    settings: Any = None,
    progress: ProgressFn | None = None,
    cancelled: CancelFn | None = None,
) -> RenderResult:
    """Burn captions + blur boxes + logos into an MP4 (H.264 + AAC)."""
    settings = settings or cfg.get_settings()
    ffmpeg, info = cfg.ffmpeg_path_or_raise(settings)
    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(f"ဗီဒီယိုဖိုင် မတွေ့ပါ: {video_path}")

    def report(fraction: float | None, stage: str) -> None:
        if progress:
            progress(fraction, stage)

    report(0.01, "ဗီဒီယို စစ်ဆေးနေသည်...")
    meta = probe_video(video_path, ffmpeg)
    if duration <= 0:
        duration = meta.duration
    if duration <= 0:
        raise RuntimeError("ဗီဒီယို ကြာချိန် မသိနိုင်ပါ — ဖိုင် ပျက်စီးနေနိုင်သည်")
    if meta.width <= 0 or meta.height <= 0:
        raise RuntimeError("ဗီဒီယို အရွယ်အစား မသိနိုင်ပါ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.WORK_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="mse-render-", dir=str(cfg.WORK_DIR)))
    try:
        _copy_fonts(work_dir / "fonts")
        ass_path = work_dir / "subs.ass"
        ass_text = build_ass(
            segments,
            width=meta.width,
            height=meta.height,
            defaults=style_defaults,
            duration=duration,
        )
        ass_path.write_text(ass_text, encoding="utf-8")

        logo_paths: list[Path] = []
        for logo in logos:
            raw = str(logo.get("src") or "")
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = cfg.UPLOADS_DIR / raw
            if candidate.exists():
                logo_paths.append(candidate)
        active_logos = [
            logo
            for logo, path in zip(logos, logo_paths)
            if path.exists() and float(logo.get("w", 0)) > 0 and float(logo.get("h", 0)) > 0
        ]
        active_logo_paths = [
            path
            for logo, path in zip(logos, logo_paths)
            if path.exists() and float(logo.get("w", 0)) > 0 and float(logo.get("h", 0)) > 0
        ]

        filter_script = build_filter_script(
            width=meta.width,
            height=meta.height,
            blur_boxes=blur_boxes,
            logos=active_logos,
            logo_input_index=[index + 1 for index in range(len(active_logos))],
            ass_name="subs.ass",
            has_audio=meta.has_audio,
            scale_to_height=int(getattr(settings, "max_render_height", 0) or 0),
        )
        (work_dir / "filters.txt").write_text(filter_script, encoding="utf-8")

        command = [ffmpeg, "-hide_banner", "-y", "-i", str(video_path.resolve())]
        for path in active_logo_paths:
            command += ["-loop", "1", "-i", str(path.resolve())]
        # Pass the graph inline (argv, not a shell): no quoting issues and it
        # works on every FFmpeg version — ``-filter_complex_script`` was
        # removed in FFmpeg 9.
        command += ["-filter_complex", filter_script, "-map", "[vfinal]"]
        if meta.has_audio:
            command += ["-map", "0:a:0?"]
        command += [
            "-c:v",
            str(settings.video_codec),
            "-crf",
            str(settings.video_crf),
            "-preset",
            str(settings.video_preset),
            "-pix_fmt",
            "yuv420p",
        ]
        if meta.has_audio:
            command += ["-c:a", "aac", "-b:a", str(settings.audio_bitrate)]
        command += [
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            "-progress",
            "pipe:1",
            "-nostats",
            str(output_path.resolve()),
        ]

        report(0.03, "စာတန်း ရေးဆွဲနေသည်...")
        log_path = work_dir / "ffmpeg.log"
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=log_file,
                text=True,
                errors="replace",
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    if cancelled and cancelled():
                        process.terminate()
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise RuntimeError("ပိတ်လိုက်ပါသည်")
                    line = line.strip()
                    if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                        try:
                            # Both keys are microseconds in recent FFmpeg builds.
                            micro = float(line.split("=", 1)[1])
                        except ValueError:
                            continue
                        seconds = micro / 1_000_000.0
                        fraction = min(0.99, 0.03 + 0.96 * (seconds / duration if duration else 0))
                        report(fraction, "ဗီဒီယို ရေးဆွဲနေသည်...")
                    elif line.startswith("progress=end"):
                        report(0.99, "ပြီးဆုံးအောင် ပြင်ဆင်နေသည်...")
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.kill()
        if returncode != 0:
            tail = _tail(log_path)
            raise RuntimeError(
                "ဗီဒီယို ရေးဆွဲ၍ မရပါ (FFmpeg error).\n" + (tail[-1500:] if tail else "အမှား အသေးစိတ် မရပါ")
            )
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("ရလဒ်ဖိုင် မဖန်တီးနိုင်ပါ — FFmpeg log ကို စစ်ပါ")
        report(1.0, "ပြီးစီးပါပြီ")
        return RenderResult(
            output=output_path,
            duration=duration,
            elapsed=round(time.time() - started, 2),
            meta=meta.to_dict(),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def build_srt(segments: Sequence[dict[str, Any]]) -> str:
    """SRT text (UTF-8, CRLF line endings as required by the format)."""
    blocks: list[str] = []
    index = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(segment.get("start", 0.0)))
            end = float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        index += 1
        blocks.append(
            f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n"
        )
    return "\n".join(blocks)


def _srt_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(milliseconds, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write SRT with a BOM (``utf-8-sig``) so Windows players detect Unicode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = build_srt(segments)
    try:
        import pysrt  # type: ignore

        subs = pysrt.SubRipFile()
        for segment in segments:
            caption = str(segment.get("text") or "").strip()
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            if not caption or end <= start:
                continue
            subs.append(
                pysrt.SubRipItem(
                    index=len(subs) + 1,
                    start=pysrt.SubRipTime(seconds=start),
                    end=pysrt.SubRipTime(seconds=end),
                    text=caption,
                )
            )
        subs.save(str(path), encoding="utf-8-sig")
    except Exception:
        path.write_text(text, encoding="utf-8-sig")
    return path


def extract_thumbnail(video_path: str | Path, out_path: str | Path, at: float = 1.0) -> Path:
    ffmpeg, _ = cfg.ffmpeg_path_or_raise()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-y",
            "-ss",
            f"{max(0.0, at):.2f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            "-q:v",
            "4",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
    )
    if result.returncode != 0 or not out_path.exists():
        raise RuntimeError("ပုံငယ် ထုတ်၍ မရပါ: " + (result.stderr or "")[-400:])
    return out_path


def make_demo_video(output_path: str | Path, duration: float = 22.0) -> Path:
    """Generate a small sample clip so the editor can be tried without a file.

    Uses lavfi test sources only — no network and no external media.
    """
    ffmpeg, _ = cfg.ffmpeg_path_or_raise()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=1280x720:rate=25:duration={duration:.2f}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=320:sample_rate=48000:duration={duration:.2f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-t",
        f"{duration:.2f}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=300)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError("သရုပ်ပြ ဗီဒီယို ဖန်တီး၍ မရပါ: " + (result.stderr or "")[-600:])
    return output_path

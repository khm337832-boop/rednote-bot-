"""ပရောဂျက် ဆက်တင် (config) — model size, thresholds, paths.

Config values are resolved in this order (later wins):

1. defaults in :class:`Settings`
2. ``backend/config.json`` (optional; see ``backend/config.example.json``)
3. environment variables (``MSE_*``)

Everything user-facing is a plain Python value so it can be serialised to JSON
and shown on the dashboard (model, thresholds, ffmpeg status, demo mode).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FONTS_DIR = BASE_DIR / "fonts"
UPLOADS_DIR = Path(os.environ.get("MSE_UPLOADS_DIR", BASE_DIR / "uploads"))
PROJECTS_DIR = Path(os.environ.get("MSE_PROJECTS_DIR", BASE_DIR / "projects"))
WORK_DIR = Path(os.environ.get("MSE_WORK_DIR", UPLOADS_DIR / "_work"))
CONFIG_FILE = Path(os.environ.get("MSE_CONFIG", BASE_DIR / "config.json"))

FONT_REGULAR = FONTS_DIR / "NotoSansMyanmar-Regular.ttf"
FONT_BOLD = FONTS_DIR / "NotoSansMyanmar-Bold.ttf"
FONT_FAMILY = "Noto Sans Myanmar"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".m4v", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}

MODEL_SIZES = ("large-v3", "large-v2", "medium", "small", "base", "tiny")
STT_BACKENDS = ("faster-whisper", "openai-whisper")
LAST_HOLD_MODES = ("cap", "video_end")


def _env(name: str, default: Any, cast=None) -> Any:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is None:
        return raw
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """Runtime settings for transcription, segment timing and rendering."""

    # ---- speech to text -------------------------------------------------
    stt_backend: str = "faster-whisper"
    model_size: str = "large-v3"
    language: str = ""  # empty = auto-detect source language
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto | float16 | int8 | int8_float16 | float32
    beam_size: int = 5
    vad_filter: bool = True
    cpu_threads: int = 0  # 0 = library default

    # ---- segment logic (CapCut style hold) ------------------------------
    gap_threshold: float = 0.5  # token gap (seconds) that starts a new segment
    punctuation_break: bool = True  # break after ။ ၊ ? ! .
    max_chars: int = 35  # Unicode grapheme / syllable count per segment
    max_duration: float = 6.0  # seconds of speech per segment
    min_duration: float = 0.3  # minimum on-screen duration
    max_hold_seconds: float = 3.0  # how long the last caption may hold
    last_hold_mode: str = "cap"  # cap = min(video end, end + max_hold)

    # ---- upload / jobs --------------------------------------------------
    max_upload_mb: int = 500
    chunk_size: int = 1024 * 1024
    job_workers: int = 2
    demo_mode: str = "auto"  # auto | on | off (sample captions when Whisper missing)

    # ---- default caption look ------------------------------------------
    default_font_size: float = 0.055  # fraction of frame height
    default_position_x: float = 0.5
    default_position_y: float = 0.86
    default_box_width: float = 0.82
    default_text_color: str = "#FFFFFF"
    default_outline_color: str = "#000000"
    default_outline_width: float = 0.08  # fraction of font size
    default_opacity: float = 1.0
    default_bold: bool = True

    # ---- rendering ------------------------------------------------------
    ffmpeg_binary: str = ""  # explicit path; auto-detected when empty
    video_codec: str = "libx264"
    video_crf: int = 20
    video_preset: str = "veryfast"
    audio_bitrate: str = "192k"
    max_render_height: int = 0  # 0 = keep source resolution

    # ---- access ---------------------------------------------------------
    access_password: str = ""

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024

    def validate(self) -> list[str]:
        """Return a list of human readable problems (empty when valid)."""
        problems: list[str] = []
        if self.stt_backend not in STT_BACKENDS:
            problems.append(f"stt_backend သည် {STT_BACKENDS} ထဲက တစ်ခု ဖြစ်ရမည်")
        if self.model_size not in MODEL_SIZES:
            problems.append(f"model_size သည် {MODEL_SIZES} ထဲက တစ်ခု ဖြစ်ရမည်")
        if self.last_hold_mode not in LAST_HOLD_MODES:
            problems.append(f"last_hold_mode သည် {LAST_HOLD_MODES} ထဲက တစ်ခု ဖြစ်ရမည်")
        if not (0.05 <= self.gap_threshold <= 5.0):
            problems.append("gap_threshold သည် 0.05 – 5.0 စက္ကန့် အတွင်း ဖြစ်ရမည်")
        if not (5 <= self.max_chars <= 200):
            problems.append("max_chars သည် 5 – 200 အတွင်း ဖြစ်ရမည်")
        if not (0.5 <= self.max_duration <= 30):
            problems.append("max_duration သည် 0.5 – 30 စက္ကန့် အတွင်း ဖြစ်ရမည်")
        if not (0.05 <= self.min_duration <= 3):
            problems.append("min_duration သည် 0.05 – 3 စက္ကန့် အတွင်း ဖြစ်ရမည်")
        if not (0.0 <= self.max_hold_seconds <= 60):
            problems.append("max_hold_seconds သည် 0 – 60 စက္ကန့် အတွင်း ဖြစ်ရမည်")
        if not (1 <= self.max_upload_mb <= 4096):
            problems.append("max_upload_mb သည် 1 – 4096 MB အတွင်း ဖြစ်ရမည်")
        return problems

    # ---- serialisation --------------------------------------------------
    def to_public_dict(self) -> dict[str, Any]:
        """Subset safe to expose on the dashboard/editor UI."""
        return {
            "stt_backend": self.stt_backend,
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "gap_threshold": self.gap_threshold,
            "punctuation_break": self.punctuation_break,
            "max_chars": self.max_chars,
            "max_duration": self.max_duration,
            "min_duration": self.min_duration,
            "max_hold_seconds": self.max_hold_seconds,
            "last_hold_mode": self.last_hold_mode,
            "max_upload_mb": self.max_upload_mb,
            "chunk_size": self.chunk_size,
            "demo_mode": self.demo_mode,
            "font_family": FONT_FAMILY,
            "default_style": {
                "fontSize": self.default_font_size,
                "x": self.default_position_x,
                "y": self.default_position_y,
                "width": self.default_box_width,
                "color": self.default_text_color,
                "outlineColor": self.default_outline_color,
                "outlineWidth": self.default_outline_width,
                "opacity": self.default_opacity,
                "bold": self.default_bold,
            },
            "max_render_height": self.max_render_height,
            "access_protected": bool(self.access_password),
        }

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)


def _cast_for(field_type: Any, value: Any) -> Any:
    """Best-effort cast of a JSON/env string to the dataclass field type."""
    if not isinstance(field_type, str) and field_type is not None:
        pass
    if isinstance(value, str):
        if field_type is bool:
            return value.strip().lower() in {"1", "true", "yes", "on"}
        if field_type is int:
            try:
                return int(float(value))
            except ValueError:
                return value
        if field_type is float:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _env_overrides(settings: Settings) -> Settings:
    mapping = {
        "stt_backend": ("MSE_STT_BACKEND", str),
        "model_size": ("MSE_MODEL_SIZE", str),
        "language": ("MSE_LANGUAGE", str),
        "device": ("MSE_DEVICE", str),
        "compute_type": ("MSE_COMPUTE_TYPE", str),
        "beam_size": ("MSE_BEAM_SIZE", int),
        "vad_filter": ("MSE_VAD_FILTER", bool),
        "gap_threshold": ("MSE_GAP_THRESHOLD", float),
        "max_chars": ("MSE_MAX_CHARS", int),
        "max_duration": ("MSE_MAX_DURATION", float),
        "min_duration": ("MSE_MIN_DURATION", float),
        "max_hold_seconds": ("MSE_MAX_HOLD_SECONDS", float),
        "last_hold_mode": ("MSE_LAST_HOLD_MODE", str),
        "max_upload_mb": ("MSE_MAX_UPLOAD_MB", int),
        "demo_mode": ("MSE_DEMO_MODE", str),
        "ffmpeg_binary": ("MSE_FFMPEG", str),
        "video_crf": ("MSE_VIDEO_CRF", int),
        "video_preset": ("MSE_VIDEO_PRESET", str),
        "max_render_height": ("MSE_MAX_RENDER_HEIGHT", int),
        "access_password": ("EDITOR_PASSWORD", str),
    }
    for name, (env_name, caster) in mapping.items():
        raw = os.environ.get(env_name)
        if raw is None or raw.strip() == "":
            continue
        setattr(settings, name, _cast_for(caster, raw.strip()))
    return settings


def load_settings(config_file: Path | None = None) -> Settings:
    path = config_file or CONFIG_FILE
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    settings = _env_overrides(Settings.from_dict(data) if data else Settings())
    if settings.max_upload_mb <= 0:
        settings.max_upload_mb = 500
    return settings


_settings_lock = threading.Lock()
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = load_settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Used by tests to inject a configured instance."""
    global _settings
    _settings = settings


# ---------------------------------------------------------------------------
# FFmpeg detection
# ---------------------------------------------------------------------------


@dataclass
class FFmpegInfo:
    path: str | None = None
    source: str = "none"
    version: str = ""
    filters: dict[str, bool] = field(default_factory=dict)
    config_flags: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def has_libass(self) -> bool:
        return bool(self.filters.get("ass") and self.filters.get("subtitles"))

    @property
    def has_harfbuzz(self) -> bool:
        return "libharfbuzz" in self.config_flags

    @property
    def has_freetype(self) -> bool:
        return "libfreetype" in self.config_flags

    @property
    def has_boxblur(self) -> bool:
        return bool(self.filters.get("boxblur"))

    @property
    def has_overlay(self) -> bool:
        return bool(self.filters.get("overlay"))

    @property
    def ok(self) -> bool:
        return bool(
            self.path
            and self.has_libass
            and self.has_freetype
            and self.has_boxblur
            and self.has_overlay
            and not self.problems
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "version": self.version,
            "ok": self.ok,
            "has_libass": self.has_libass,
            "has_harfbuzz": self.has_harfbuzz,
            "has_freetype": self.has_freetype,
            "has_boxblur": self.has_boxblur,
            "has_overlay": self.has_overlay,
            "problems": self.problems,
        }


def _candidate_ffmpeg_paths(settings: Settings) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if settings.ffmpeg_binary:
        candidates.append((settings.ffmpeg_binary, "env"))
    env_bin = os.environ.get("FFMPEG_BINARY")
    if env_bin:
        candidates.append((env_bin, "env"))
    system = shutil.which("ffmpeg")
    if system:
        candidates.append((system, "system"))
    try:  # imageio's static build ships libass + harfbuzz
        import imageio_ffmpeg

        candidates.append((imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"))
    except Exception:  # pragma: no cover - optional dependency
        pass
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for path, source in candidates:
        if not path:
            continue
        resolved = str(Path(path).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((resolved, source))
    return unique


def _run_ffmpeg(path: str, args: list[str], timeout: float = 25.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [path, "-hide_banner", *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
    )


def inspect_ffmpeg(path: str, source: str = "system") -> FFmpegInfo:
    info = FFmpegInfo(path=path, source=source)
    try:
        filters_out = _run_ffmpeg(path, ["-filters"]).stdout
        version_out = _run_ffmpeg(path, ["-version"]).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        info.problems.append(f"FFmpeg ကို လည်ပတ်၍ မရပါ: {exc}")
        return info

    wanted = {"ass", "subtitles", "boxblur", "overlay", "scale", "rotate", "crop"}
    for line in filters_out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1]
        if name in wanted:
            info.filters[name] = True
    for line in version_out.splitlines():
        if line.startswith("ffmpeg version"):
            info.version = line.strip()
        if "configuration:" in line:
            info.config_flags = line.strip()
    for filter_name in ("ass", "boxblur", "overlay"):
        if not info.filters.get(filter_name):
            info.problems.append(f"FFmpeg တွင် '{filter_name}' filter မပါပါ")
    if not info.has_libass:
        info.problems.append(
            "FFmpeg တွင် libass မပါပါ — မြန်မာစာတန်း burn-in လုပ်၍ မရပါ။ "
            "libass ပါသော FFmpeg build ထည့်ပါ (README တွင် ကြည့်ပါ)။"
        )
    if not info.has_freetype:
        info.problems.append("FFmpeg တွင် libfreetype မပါပါ — ဖောင့် render မရနိုင်ပါ")
    if info.has_libass and not info.has_harfbuzz:
        info.problems.append(
            "FFmpeg တွင် libharfbuzz မပါပါ — မြန်မာစာ အက္ခရာစီပုံ မှားနိုင်ပါသည်"
        )
    return info


_ffmpeg_lock = threading.Lock()
_ffmpeg_cache: dict[str, FFmpegInfo] = {}


def get_ffmpeg_info(settings: Settings | None = None, *, force: bool = False) -> FFmpegInfo:
    """Pick the first FFmpeg that can burn ASS subtitles; fall back gracefully."""
    settings = settings or get_settings()
    candidates = _candidate_ffmpeg_paths(settings)
    key = "|".join(f"{p}:{s}" for p, s in candidates)
    with _ffmpeg_lock:
        if not force and key in _ffmpeg_cache:
            return _ffmpeg_cache[key]
        inspected = [inspect_ffmpeg(path, source) for path, source in candidates]
        best: FFmpegInfo | None = None
        for info in inspected:
            if info.ok:
                best = info
                break
        if best is None:
            # Prefer a partially working binary so errors stay meaningful.
            usable = [i for i in inspected if i.path and not i.problems[:1]]
            best = usable[0] if usable else (inspected[0] if inspected else FFmpegInfo())
        _ffmpeg_cache[key] = best
        return best


def ffmpeg_path_or_raise(settings: Settings | None = None) -> tuple[str, FFmpegInfo]:
    info = get_ffmpeg_info(settings)
    if not info.path:
        raise RuntimeError(
            "FFmpeg မတွေ့ပါ။ README ပါ installation အဆင့်များအတိုင်း FFmpeg (libass ပါသော build) ထည့်ပါ။"
        )
    if not info.has_libass:
        raise RuntimeError(
            "FFmpeg တွင် libass မပါသောကြောင့် မြန်မာစာတန်း burn-in လုပ်၍ မရပါ။ "
            "`ffmpeg -filters | grep subtitles` ဖြင့် စစ်ပြီး libass ပါသော build သုံးပါ။"
        )
    return info.path, info


# ---------------------------------------------------------------------------
# Whisper availability
# ---------------------------------------------------------------------------

_whisper_state: dict[str, Any] = {"checked": False, "available": False, "reason": ""}


def whisper_available(settings: Settings | None = None) -> tuple[bool, str]:
    """Whether the configured STT backend can be imported right now."""
    settings = settings or get_settings()
    cache_key = settings.stt_backend
    if _whisper_state.get("checked") and _whisper_state.get("backend") == cache_key:
        return bool(_whisper_state["available"]), str(_whisper_state["reason"])
    available, reason = False, ""
    module = "faster_whisper" if settings.stt_backend == "faster-whisper" else "whisper"
    try:
        __import__(module)
        available = True
    except Exception as exc:  # ImportError or binary mismatch
        reason = (
            f"'{module}' module မရှိပါ ({exc}). "
            f"`pip install -r backend/requirements.txt` ဖြင့် ထည့်ပါ။"
        )
    _whisper_state.update(
        {"checked": True, "available": available, "reason": reason, "backend": cache_key}
    )
    return available, reason


def demo_enabled(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    mode = (settings.demo_mode or "auto").lower()
    if mode == "on":
        return True
    if mode == "off":
        return False
    available, _ = whisper_available(settings)
    return not available

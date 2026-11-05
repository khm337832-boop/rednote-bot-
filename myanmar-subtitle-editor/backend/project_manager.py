"""Project save/load (JSON, UTF-8, ensure_ascii=False) — အလိုအလျောက် သိမ်းဆည်းမှု။

Project တစ်ခုလုံးကို ``backend/projects/<id>.json`` တွင် သိမ်းသည်။
Atomic write (tmp → rename) ဖြင့် လျှပ်စစ်ပြတ်သွားလျှင်လည်း ဖိုင် မပျက်စီးစေရ။
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config as cfg
from . import myanmar_text as mt

SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{6,64}$")

_lock = threading.RLock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex[:16]
    return f"{prefix}{value}" if prefix else value


def _safe_id(value: str | None) -> str:
    if value and _ID_RE.match(value):
        return value
    return new_id()


def _projects_dir() -> Path:
    cfg.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return cfg.PROJECTS_DIR


def _path_for(project_id: str) -> Path:
    return _projects_dir() / f"{project_id}.json"


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number:  # NaN
        return fallback
    return min(max(number, low), high)


def normalise_segment(segment: dict[str, Any], index: int) -> dict[str, Any]:
    """Validate one caption and keep the editor's CapCut invariants."""
    style = segment.get("style") if isinstance(segment.get("style"), dict) else {}
    clean_style: dict[str, Any] = {}
    for key in ("fontSize", "x", "y", "width", "rotation", "opacity", "outlineWidth"):
        if key in style:
            clean_style[key] = style[key]
    for key in ("color", "outlineColor", "backgroundColor"):
        if isinstance(style.get(key), str):
            clean_style[key] = style[key]
    for key in ("bold", "background"):
        if key in style:
            clean_style[key] = bool(style[key])
    return {
        "id": _safe_id(str(segment.get("id") or "")) if segment.get("id") else new_id("s"),
        "start": round(max(0.0, _clamp(segment.get("start"), 0, 86_400, 0.0)), 3),
        "end": round(max(0.0, _clamp(segment.get("end"), 0, 86_400, 0.0)), 3),
        "text": mt.normalize_text(str(segment.get("text") or ""))[:2000],
        "style": clean_style,
    }


def normalise_logo(logo: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _safe_id(str(logo.get("id") or "")) if logo.get("id") else new_id("l"),
        "type": "logo",
        "src": str(logo.get("src") or "")[:512],
        "name": str(logo.get("name") or "Logo")[:120],
        "x": round(_clamp(logo.get("x"), -0.5, 1.5, 0.02), 4),
        "y": round(_clamp(logo.get("y"), -0.5, 1.5, 0.02), 4),
        "w": round(_clamp(logo.get("w"), 0.02, 1.5, 0.15), 4),
        "h": round(_clamp(logo.get("h"), 0.02, 1.5, 0.15), 4),
        "rotation": round(_clamp(logo.get("rotation"), -360, 360, 0.0), 2),
        "opacity": round(_clamp(logo.get("opacity"), 0.0, 1.0, 1.0), 3),
        "visible": bool(logo.get("visible", True)),
        "locked": bool(logo.get("locked", False)),
    }


def normalise_blur(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _safe_id(str(box.get("id") or "")) if box.get("id") else new_id("b"),
        "type": "blur",
        "name": str(box.get("name") or "Blur")[:120],
        "x": round(_clamp(box.get("x"), 0.0, 1.0, 0.3), 4),
        "y": round(_clamp(box.get("y"), 0.0, 1.0, 0.3), 4),
        "w": round(_clamp(box.get("w", box.get("width")), 0.01, 1.0, 0.2), 4),
        "h": round(_clamp(box.get("h", box.get("height")), 0.01, 1.0, 0.1), 4),
        "intensity": round(_clamp(box.get("intensity"), 1, 20, 8), 1),
        "visible": bool(box.get("visible", True)),
        "locked": bool(box.get("locked", False)),
    }


def normalise_segments(
    segments: Iterable[dict[str, Any]], duration: float = 0.0
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sort, remove overlaps and enforce ``end[i] = start[i+1]``.

    Returns ``(segments, warnings)``; warnings are shown in the editor toast.
    """
    warnings: list[str] = []
    cleaned = [normalise_segment(segment, index) for index, segment in enumerate(segments or [])]
    cleaned.sort(key=lambda s: (s["start"], s["end"]))
    if duration > 0:
        for segment in cleaned:
            if segment["start"] > duration:
                segment["start"] = max(0.0, duration - 0.5)
            if segment["end"] > duration + 0.05:
                segment["end"] = duration
    kept: list[dict[str, Any]] = []
    for segment in cleaned:
        if not segment["text"].strip():
            warnings.append("စာသားမရှိသော စာတန်းတစ်ခုကို ဖယ်လိုက်သည်")
            continue
        if segment["end"] <= segment["start"]:
            warnings.append("အချိန်မမှန်သော စာတန်းတစ်ခုကို ဖယ်လိုက်သည်")
            continue
        if kept and segment["start"] < kept[-1]["end"] - 1e-6:
            warnings.append("ထပ်နေသော စာတန်းအချိန်များကို ညှိလိုက်သည်")
            segment["start"] = kept[-1]["end"]
            if segment["end"] <= segment["start"]:
                continue
        kept.append(segment)
    if kept:
        kept[0]["start"] = max(0.0, kept[0]["start"])
        for index in range(len(kept) - 1):
            kept[index]["end"] = round(kept[index + 1]["start"], 3)
        if duration > 0:
            kept[-1]["end"] = min(kept[-1]["end"], duration)
        if kept[-1]["end"] <= kept[-1]["start"]:
            kept[-1]["end"] = round(kept[-1]["start"] + 0.3, 3)
    return kept, warnings


def normalise_project(payload: dict[str, Any], *, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    existing = existing or {}
    project_id = _safe_id(str(payload.get("id") or existing.get("id") or ""))
    video = payload.get("video") if isinstance(payload.get("video"), dict) else existing.get("video", {})
    duration = 0.0
    try:
        duration = float(video.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    segments, warnings = normalise_segments(payload.get("segments") or [], duration)
    logos = [normalise_logo(logo) for logo in (payload.get("logos") or []) if isinstance(logo, dict)]
    blurs = [normalise_blur(box) for box in (payload.get("blurs") or []) if isinstance(box, dict)]
    layer_order = payload.get("layer_order")
    if not isinstance(layer_order, list):
        layer_order = ["subtitle"] + [logo["id"] for logo in logos] + [box["id"] for box in blurs]
    style_defaults = payload.get("style_defaults")
    if not isinstance(style_defaults, dict):
        style_defaults = existing.get("style_defaults") or {}
    created_at = existing.get("created_at") or payload.get("created_at") or now_iso()
    project = {
        "schema_version": SCHEMA_VERSION,
        "id": project_id,
        "name": mt.normalize_text(str(payload.get("name") or existing.get("name") or "အမည်မသိ ပရောဂျက်"))[
            :120
        ]
        or "အမည်မသိ ပရောဂျက်",
        "created_at": created_at,
        "updated_at": now_iso(),
        "video": {
            "id": str(video.get("id") or ""),
            "filename": str(video.get("filename") or "")[:255],
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
            "duration": round(duration, 3),
            "has_audio": bool(video.get("has_audio", True)),
        },
        "segments": segments,
        "logos": logos,
        "blurs": blurs,
        "layer_order": [str(item) for item in layer_order][:200],
        "style_defaults": style_defaults,
        "thumbnail": str(payload.get("thumbnail") or existing.get("thumbnail") or ""),
        "warnings": warnings,
    }
    return project


def save_project(payload: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        requested_id = str(payload.get("id") or "")
        existing: dict[str, Any] | None = None
        if _ID_RE.match(requested_id):
            path = _path_for(requested_id)
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
        project = normalise_project(payload, existing=existing)
        path = _path_for(project["id"])
        tmp = path.with_suffix(".json.tmp")
        data = json.dumps(project, ensure_ascii=False, indent=2)
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
        return project


def list_projects() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(_projects_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.append(
            {
                "id": data.get("id") or path.stem,
                "name": data.get("name") or "အမည်မသိ ပရောဂျက်",
                "updated_at": data.get("updated_at") or "",
                "created_at": data.get("created_at") or "",
                "segment_count": len(data.get("segments") or []),
                "logo_count": len(data.get("logos") or []),
                "blur_count": len(data.get("blurs") or []),
                "duration": (data.get("video") or {}).get("duration") or 0,
                "video_id": (data.get("video") or {}).get("id") or "",
                "thumbnail": data.get("thumbnail") or "",
                "name_preview": (data.get("segments") or [{}])[0].get("text", "")[:60]
                if data.get("segments")
                else "",
            }
        )
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return items


def get_project(project_id: str) -> dict[str, Any] | None:
    if not _ID_RE.match(project_id or ""):
        return None
    path = _path_for(project_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    data.setdefault("id", project_id)
    return data


def delete_project(project_id: str) -> bool:
    if not _ID_RE.match(project_id or ""):
        return False
    path = _path_for(project_id)
    if path.exists():
        with _lock:
            path.unlink()
        return True
    return False

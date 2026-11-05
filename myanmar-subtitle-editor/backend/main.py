"""Myanmar Subtitle Editor — FastAPI backend.

Run locally (from the ``myanmar-subtitle-editor`` folder)::

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Routes
------
* ``/``                 — Dashboard (Glass UI)
* ``/editor``           — Editor (Canvas + Timeline)
* ``/api/upload/*``     — chunked, streamed video upload (500 MB default)
* ``/api/transcribe``   — background Whisper job + progress polling
* ``/api/project/*``    — save / list / get / delete JSON projects
* ``/api/export``       — background FFmpeg render + progress polling
* ``/api/srt``          — SRT download (UTF-8 with BOM)
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config as cfg
from . import project_manager as pm
from . import subtitle_generator as sg
from . import translation as tr
from . import video_renderer as vr

APP_VERSION = "1.0.0"
log = logging.getLogger("mse")

cfg.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
cfg.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.WORK_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS_DIR = cfg.UPLOADS_DIR / "videos"
LOGOS_DIR = cfg.UPLOADS_DIR / "logos"
EXPORTS_DIR = cfg.UPLOADS_DIR / "exports"
TMP_DIR = cfg.UPLOADS_DIR / "_tmp"
for directory in (VIDEOS_DIR, LOGOS_DIR, EXPORTS_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def _log_startup() -> None:
    settings = cfg.get_settings()
    ffmpeg = cfg.get_ffmpeg_info(settings)
    whisper_ok, whisper_reason = cfg.whisper_available(settings)
    log.info("Myanmar Subtitle Editor %s", APP_VERSION)
    log.info("FFmpeg: %s (%s) ok=%s", ffmpeg.path, ffmpeg.source, ffmpeg.ok)
    if not ffmpeg.ok:
        for problem in ffmpeg.problems:
            log.warning("FFmpeg: %s", problem)
    log.info(
        "Whisper: %s (%s) available=%s", settings.stt_backend, settings.model_size, whisper_ok
    )
    if not whisper_ok:
        log.warning("Whisper: %s", whisper_reason)
    if cfg.demo_enabled(settings):
        log.warning("သရုပ်ပြ မုဒ် ဖွင့်ထားသည် — စာတန်းများသည် စမ်းသပ်ရန် သရုပ်ပြ စာသားများ ဖြစ်သည်")
    if not settings.access_password:
        log.warning("EDITOR_PASSWORD မသတ်မှတ်ထားပါ — editor သည် လူတိုင်း ဝင်နိုင်သည်")
    JOBS.prune()


@asynccontextmanager
async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _log_startup()
    yield


app = FastAPI(
    title="Myanmar Subtitle Editor",
    version=APP_VERSION,
    docs_url="/api/docs",
    lifespan=lifespan,
)

if cfg.FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(cfg.FRONTEND_DIR)), name="static")

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(status: int, message: str, detail: str | None = None) -> HTTPException:
    payload: dict[str, Any] = {"detail": message, "message": message}
    if detail:
        payload["error"] = detail[:2000]
    return HTTPException(status_code=status, detail=payload)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _video_meta_path(video_id: str) -> Path:
    return VIDEOS_DIR / f"{video_id}.json"


def _read_video_meta(video_id: str) -> dict[str, Any]:
    if not _ID_RE.match(video_id or ""):
        raise _error(404, "ဗီဒီယို မတွေ့ပါ")
    path = _video_meta_path(video_id)
    if not path.exists():
        raise _error(404, "ဗီဒီယို မတွေ့ပါ (သို့မဟုတ် ဖျက်ထားပြီးပါသည်)")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(500, "ဗီဒီယို အချက်အလက် ဖတ်၍ မရပါ", str(exc))


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_filename(name: str, fallback: str = "video") -> str:
    """Keep Unicode (Myanmar) names; strip path separators and control chars."""
    raw = Path(str(name or "")).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw).strip(" ._")
    if not cleaned:
        return fallback
    return cleaned[:120]


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


# ---------------------------------------------------------------------------
# Optional password protection ("private page")
# ---------------------------------------------------------------------------


def _auth_token(settings: cfg.Settings) -> str:
    return hmac.new(
        settings.access_password.encode("utf-8"), b"myanmar-subtitle-editor", hashlib.sha256
    ).hexdigest()


def _is_authed(request: Request, settings: cfg.Settings) -> bool:
    if not settings.access_password:
        return True
    cookie = request.cookies.get("mse_auth", "")
    return hmac.compare_digest(cookie, _auth_token(settings))


PUBLIC_PATHS = {"/", "/editor", "/healthz", "/favicon.svg", "/api/docs", "/openapi.json"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    settings = cfg.get_settings()
    if not settings.access_password:
        # A public Render deployment must never silently become a public
        # file/project API when its private password is missing.
        if os.environ.get("RENDER") and request.url.path.startswith("/api/"):
            return JSONResponse(status_code=503, content={"detail": "Private editor access is not configured"})
        return await call_next(request)
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(("/static/", "/api/auth/", "/api/docs")):
        return await call_next(request)
    if path.startswith("/api/") and not _is_authed(request, settings):
        return JSONResponse(
            status_code=401,
            content={"detail": "သီးသန့် စကားဝှက် လိုအပ်ပါသည်", "code": "auth_required"},
        )
    return await call_next(request)


@app.get("/api/auth/status")
def auth_status(request: Request) -> dict[str, Any]:
    settings = cfg.get_settings()
    return {
        "required": bool(settings.access_password),
        "authed": _is_authed(request, settings),
    }


@app.post("/api/auth/login")
def auth_login(request: Request, response: Response, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    settings = cfg.get_settings()
    if not settings.access_password:
        return {"ok": True, "required": False}
    supplied = str(payload.get("password") or "")
    if not hmac.compare_digest(supplied, settings.access_password):
        raise _error(401, "စကားဝှက် မှားနေပါသည်")
    response.set_cookie(
        "mse_auth",
        _auth_token(settings),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=60 * 60 * 24 * 30,
    )
    return {"ok": True, "required": True}


@app.post("/api/auth/logout")
def auth_logout(response: Response) -> dict[str, Any]:
    response.delete_cookie("mse_auth")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Capabilities / config / health
# ---------------------------------------------------------------------------


@app.get("/api/capabilities")
def capabilities(force: bool = False) -> dict[str, Any]:
    settings = cfg.get_settings()
    ffmpeg = cfg.get_ffmpeg_info(settings, force=force)
    whisper_ok, whisper_reason = cfg.whisper_available(settings)
    return {
        "version": APP_VERSION,
        "ffmpeg": ffmpeg.to_dict(),
        "whisper": {
            "available": whisper_ok,
            "reason": whisper_reason,
            "backend": settings.stt_backend,
            "model": settings.model_size,
        },
        "demo": cfg.demo_enabled(settings),
        "fonts": {
            "family": cfg.FONT_FAMILY,
            "regular": cfg.FONT_REGULAR.exists(),
            "bold": cfg.FONT_BOLD.exists(),
        },
        "config": settings.to_public_dict(),
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "chunk_size": settings.chunk_size,
            "allowed_video": sorted(cfg.ALLOWED_VIDEO_EXTENSIONS),
            "allowed_image": sorted(cfg.ALLOWED_IMAGE_EXTENSIONS),
        },
        "auth": {"required": bool(settings.access_password)},
        "problems": [
            *ffmpeg.problems,
            *([] if whisper_ok else [whisper_reason]),
            *settings.validate(),
        ],
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    settings = cfg.get_settings()
    ffmpeg = cfg.get_ffmpeg_info(settings)
    return {
        "status": "ok",
        "version": APP_VERSION,
        "ffmpeg": ffmpeg.ok,
        "whisper": cfg.whisper_available(settings)[0],
    }


# ---------------------------------------------------------------------------
# Jobs (transcribe / export): background thread + progress polling
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = "စောင့်ဆိုင်းနေသည်..."
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobCancelled(Exception):
    pass


class JobStore:
    def __init__(self, workers: int = 2) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="mse-job")

    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = _now()

    def progress(self, job: Job, fraction: float | None, stage: str | None = None) -> None:
        with self._lock:
            if job.cancel_requested:
                raise JobCancelled()
            if fraction is not None:
                job.progress = max(job.progress, min(1.0, float(fraction)))
            if stage:
                job.stage = stage
            job.updated_at = _now()

    def submit(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> None:
        def runner() -> None:
            self.update(job, status="running", stage="စတင်နေသည်...", progress=0.01)
            try:
                result = fn(job)
                self.update(
                    job,
                    status="done",
                    progress=1.0,
                    stage="ပြီးစီးပါပြီ",
                    result=result,
                )
            except JobCancelled:
                self.update(job, status="cancelled", stage="ပိတ်လိုက်ပါသည်", error=None)
            except Exception as exc:  # noqa: BLE001 - report to UI
                log.exception("job %s failed", job.id)
                self.update(
                    job,
                    status="error",
                    stage="အမှား ဖြစ်ပွားသည်",
                    error=str(exc) or exc.__class__.__name__,
                )

        self._pool.submit(runner)

    def prune(self, max_age_seconds: float = 60 * 60 * 12) -> None:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            for job_id, job in list(self._jobs.items()):
                try:
                    created = datetime.fromisoformat(job.created_at).timestamp()
                except ValueError:
                    created = time.time()
                if created < cutoff and job.status in {"done", "error", "cancelled"}:
                    self._jobs.pop(job_id, None)


JOBS = JobStore(cfg.get_settings().job_workers)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise _error(404, "အလုပ် မတွေ့ပါ (သို့) သက်တမ်း ကုန်သွားပါပြီ")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise _error(404, "အလုပ် မတွေ့ပါ")
    if job.status in {"done", "error", "cancelled"}:
        return job.to_dict()
    JOBS.update(job, cancel_requested=True, stage="ပိတ်နေသည်...")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Video upload (chunked stream → disk; never reads the whole file into RAM)
# ---------------------------------------------------------------------------

_pending_lock = threading.Lock()
_pending: dict[str, dict[str, Any]] = {}


def _part_path(upload_id: str) -> Path:
    return TMP_DIR / f"{upload_id}.part"


@app.post("/api/upload/start")
def upload_start(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    settings = cfg.get_settings()
    filename = str(payload.get("filename") or "video.mp4")
    extension = Path(filename).suffix.lower()
    if extension not in cfg.ALLOWED_VIDEO_EXTENSIONS:
        raise _error(
            400,
            f"'{extension or '?'}' ဖိုင် အမျိုးအစားကို မပံ့ပိုးပါ။ "
            f"({', '.join(sorted(cfg.ALLOWED_VIDEO_EXTENSIONS))})",
        )
    try:
        size = int(payload.get("size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        raise _error(400, "ဖိုင် အရွယ်အစား မမှန်ပါ")
    if size > settings.max_upload_bytes:
        raise _error(
            413,
            f"ဖိုင် ကြီးလွန်းပါသည် ({_human_size(size)})။ "
            f"အများဆုံး {settings.max_upload_mb} MB သာ ရပါသည်။",
        )
    upload_id = uuid.uuid4().hex
    video_id = uuid.uuid4().hex[:16]
    part = _part_path(upload_id)
    part.write_bytes(b"")
    with _pending_lock:
        _pending[upload_id] = {
            "upload_id": upload_id,
            "video_id": video_id,
            "filename": _safe_filename(filename),
            "extension": extension,
            "size": size,
            "received": 0,
            "created": time.time(),
        }
    return {
        "upload_id": upload_id,
        "video_id": video_id,
        "chunk_size": settings.chunk_size,
        "max_bytes": settings.max_upload_bytes,
    }


@app.put("/api/upload/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = 0) -> dict[str, Any]:
    with _pending_lock:
        state = _pending.get(upload_id)
    if not state:
        raise _error(404, "တင်ပို့မှု မတွေ့ပါ — ပြန်စပါ")
    part = _part_path(upload_id)
    if offset != state["received"]:
        return {"upload_id": upload_id, "received": state["received"], "resync": True}
    written = 0
    with open(part, "ab") as handle:
        async for chunk in request.stream():
            if not chunk:
                continue
            written += len(chunk)
            if state["received"] + written > state["size"] + 1024:
                raise _error(413, "ဖိုင် အရွယ်အစား သတ်မှတ်ချက် ကျော်လွန်နေပါသည်")
            handle.write(chunk)
    with _pending_lock:
        state["received"] += written
        received = state["received"]
    return {"upload_id": upload_id, "received": received, "size": state["size"]}


@app.post("/api/upload/{upload_id}/complete")
def upload_complete(upload_id: str) -> dict[str, Any]:
    with _pending_lock:
        state = _pending.pop(upload_id, None)
    if not state:
        raise _error(404, "တင်ပို့မှု မတွေ့ပါ")
    part = _part_path(upload_id)
    if not part.exists():
        raise _error(400, "တင်ပို့မှု ဖိုင် မတွေ့ပါ")
    actual = part.stat().st_size
    if actual < state["size"] - 1024:
        with _pending_lock:
            state["received"] = actual
            _pending[upload_id] = state
        raise _error(
            400,
            f"ဖိုင် မပြည့်စုံပါ ({_human_size(actual)} / {_human_size(state['size'])})။ "
            "ပြန်တင်ပါ။",
        )
    video_id = state["video_id"]
    target = VIDEOS_DIR / f"{video_id}{state['extension']}"
    part.replace(target)
    try:
        meta = vr.probe_video(target, cfg.get_ffmpeg_info().path)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise _error(400, "ဗီဒီယို ဖတ်၍ မရပါ — ဖိုင် ပျက်စီးနေနိုင်သည်", str(exc))
    if meta.duration <= 0:
        target.unlink(missing_ok=True)
        raise _error(400, "ဗီဒီယို ကြာချိန် မသိနိုင်ပါ — ဖိုင် ပျက်စီးနေနိုင်သည်")
    record = {
        "id": video_id,
        "filename": state["filename"],
        "path": target.name,
        "size": actual,
        "uploaded_at": _now(),
        **meta.to_dict(),
    }
    _write_json_atomic(_video_meta_path(video_id), record)
    return _video_response(record)


def _video_response(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "filename": record.get("filename", ""),
        "size": record.get("size", 0),
        "url": f"/api/video/{record['id']}",
        "thumbnail_url": f"/api/video/{record['id']}/thumbnail",
        "width": record.get("width", 0),
        "height": record.get("height", 0),
        "duration": record.get("duration", 0),
        "fps": record.get("fps", 0),
        "has_audio": record.get("has_audio", False),
        "video_codec": record.get("video_codec", ""),
        "audio_codec": record.get("audio_codec", ""),
        "uploaded_at": record.get("uploaded_at", ""),
    }


@app.post("/api/upload")
def upload_single(file: UploadFile) -> dict[str, Any]:
    """Simple streaming upload (curl/scripts). The dashboard uses chunks instead."""
    settings = cfg.get_settings()
    filename = _safe_filename(file.filename or "video.mp4")
    extension = Path(filename).suffix.lower()
    if extension not in cfg.ALLOWED_VIDEO_EXTENSIONS:
        raise _error(400, f"'{extension}' ဖိုင် အမျိုးအစားကို မပံ့ပိုးပါ")
    video_id = uuid.uuid4().hex[:16]
    target = VIDEOS_DIR / f"{video_id}{extension}"
    size = 0
    with open(target, "wb") as handle:
        while True:
            chunk = file.file.read(settings.chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_bytes:
                handle.close()
                target.unlink(missing_ok=True)
                raise _error(413, f"အများဆုံး {settings.max_upload_mb} MB သာ ရပါသည်")
            handle.write(chunk)
    meta = vr.probe_video(target, cfg.get_ffmpeg_info().path)
    record = {
        "id": video_id,
        "filename": filename,
        "path": target.name,
        "size": size,
        "uploaded_at": _now(),
        **meta.to_dict(),
    }
    _write_json_atomic(_video_meta_path(video_id), record)
    return _video_response(record)


@app.get("/api/video/{video_id}")
def video_stream(video_id: str, request: Request):  # type: ignore[no-untyped-def]
    record = _read_video_meta(video_id)
    path = VIDEOS_DIR / str(record.get("path") or "")
    if not path.exists():
        raise _error(404, "ဗီဒီယိုဖိုင် မတွေ့ပါ")
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return _range_response(path, request, media_type)


def _range_response(path: Path, request: Request, media_type: str) -> Response:
    file_size = path.stat().st_size
    range_header = request.headers.get("range") or ""
    start, end = 0, file_size - 1
    status = 200
    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if match and file_size > 0:
        start_raw, end_raw = match.groups()
        if start_raw:
            start = int(start_raw)
        if end_raw:
            end = min(int(end_raw), file_size - 1)
        elif not start_raw:
            start, end = 0, file_size - 1
        if start >= file_size:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        if end < start:
            end = file_size - 1
        status = 206
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(max(0, end - start + 1)),
        "Cache-Control": "private, max-age=0, must-revalidate",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    if request.method == "HEAD":
        return Response(status_code=status, headers=headers, media_type=media_type)

    def iterator():  # type: ignore[no-untyped-def]
        remaining = end - start + 1
        with open(path, "rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(512 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iterator(), status_code=status, media_type=media_type, headers=headers
    )


@app.get("/api/video/{video_id}/meta")
def video_meta(video_id: str) -> dict[str, Any]:
    return _video_response(_read_video_meta(video_id))


@app.get("/api/video/{video_id}/thumbnail")
def video_thumbnail(video_id: str) -> Response:
    record = _read_video_meta(video_id)
    source = VIDEOS_DIR / str(record.get("path") or "")
    if not source.exists():
        raise _error(404, "ဗီဒီယိုဖိုင် မတွေ့ပါ")
    thumb = VIDEOS_DIR / f"{video_id}.thumb.jpg"
    if not thumb.exists():
        try:
            vr.extract_thumbnail(source, thumb, at=min(2.0, max(0.2, record.get("duration", 1) / 4)))
        except Exception:
            return Response(status_code=204)
    return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@app.delete("/api/video/{video_id}")
def video_delete(video_id: str) -> dict[str, Any]:
    record = _read_video_meta(video_id)
    (VIDEOS_DIR / str(record.get("path") or "")).unlink(missing_ok=True)
    _video_meta_path(video_id).unlink(missing_ok=True)
    (VIDEOS_DIR / f"{video_id}.thumb.jpg").unlink(missing_ok=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Logos
# ---------------------------------------------------------------------------


@app.post("/api/logo")
def logo_upload(file: UploadFile) -> dict[str, Any]:
    settings = cfg.get_settings()
    filename = _safe_filename(file.filename or "logo.png", "logo")
    extension = Path(filename).suffix.lower()
    if extension not in cfg.ALLOWED_IMAGE_EXTENSIONS:
        raise _error(
            400,
            f"'{extension}' ပုံ အမျိုးအစားကို မပံ့ပိုးပါ။ "
            f"({', '.join(sorted(cfg.ALLOWED_IMAGE_EXTENSIONS))})",
        )
    logo_id = uuid.uuid4().hex[:16]
    target = LOGOS_DIR / f"{logo_id}{extension}"
    size = 0
    with open(target, "wb") as handle:
        while True:
            chunk = file.file.read(settings.chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > 12 * 1024 * 1024:
                handle.close()
                target.unlink(missing_ok=True)
                raise _error(413, "Logo ဖိုင် အများဆုံး 12 MB သာ ရပါသည်")
            handle.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise _error(400, "Logo ဖိုင် ဗလာ ဖြစ်နေပါသည်")

    if extension == ".svg":
        converted = _convert_svg(target)
        if converted is None:
            target.unlink(missing_ok=True)
            raise _error(
                400,
                "SVG ဖိုင်ကို PNG အဖြစ် ပြောင်း၍ မရပါ။ PNG/JPG ဖြင့် တင်ပါ "
                "(သို့မဟုတ် librsvg ပါသော FFmpeg သုံးပါ)။",
            )
        target.unlink(missing_ok=True)
        target = converted
        extension = ".png"

    width, height = _image_size(target)
    record = {
        "id": logo_id,
        "name": filename,
        "file": target.name,
        "size": target.stat().st_size,
        "width": width,
        "height": height,
        "uploaded_at": _now(),
    }
    _write_json_atomic(LOGOS_DIR / f"{logo_id}.json", record)
    return {
        "id": logo_id,
        "name": filename,
        "src": f"logos/{target.name}",
        "url": f"/api/media/logos/{target.name}",
        "width": width,
        "height": height,
        "size": record["size"],
    }


def _convert_svg(path: Path) -> Path | None:
    ffmpeg = cfg.get_ffmpeg_info().path
    if ffmpeg:
        out = path.with_suffix(".png")
        result = None
        try:
            import subprocess

            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-y", "-i", str(path), "-frames:v", "1", str(out)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            result = None
        if result and result.returncode == 0 and out.exists():
            return out
    try:  # optional: cairosvg
        import cairosvg  # type: ignore

        out = path.with_suffix(".png")
        cairosvg.svg2png(url=str(path), write_to=str(out), output_width=1024)
        return out if out.exists() else None
    except Exception:
        return None


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return 0, 0


@app.get("/api/media/logos/{name}")
def logo_media(name: str) -> Response:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,120}", name):
        raise _error(404, "ဖိုင် မတွေ့ပါ")
    path = LOGOS_DIR / name
    if not path.exists():
        raise _error(404, "ဖိုင် မတွေ့ပါ")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


# ---------------------------------------------------------------------------
# Demo helpers (no ASR model required) — clearly labelled in the UI
# ---------------------------------------------------------------------------


@app.post("/api/demo/video")
def demo_video() -> dict[str, Any]:
    settings = cfg.get_settings()
    if not cfg.demo_enabled(settings):
        raise _error(403, "သရုပ်ပြ မုဒ် ပိတ်ထားပါသည် (demo_mode=off)")
    video_id = "demovideo01"
    target = VIDEOS_DIR / f"{video_id}.mp4"
    if not target.exists() or target.stat().st_size == 0:
        try:
            vr.make_demo_video(target, duration=22.0)
        except Exception as exc:
            raise _error(500, "သရုပ်ပြ ဗီဒီယို ဖန်တီး၍ မရပါ", str(exc))
    meta = vr.probe_video(target, cfg.get_ffmpeg_info().path)
    record = {
        "id": video_id,
        "filename": "sample-demo.mp4",
        "path": target.name,
        "size": target.stat().st_size,
        "uploaded_at": _now(),
        **meta.to_dict(),
    }
    _write_json_atomic(_video_meta_path(video_id), record)
    return _video_response(record)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _settings_with_overrides(overrides: dict[str, Any]) -> cfg.Settings:
    settings = cfg.get_settings()
    allowed = {
        "model_size": str,
        "language": str,
        "stt_backend": str,
        "gap_threshold": float,
        "max_chars": int,
        "max_duration": float,
        "max_hold_seconds": float,
        "min_duration": float,
        "last_hold_mode": str,
        "vad_filter": bool,
    }
    changes: dict[str, Any] = {}
    for key, caster in allowed.items():
        if key not in overrides or overrides[key] is None:
            continue
        value = overrides[key]
        if caster is bool:
            changes[key] = bool(value)
            continue
        try:
            changes[key] = caster(value)
        except (TypeError, ValueError):
            raise _error(400, f"'{key}' တန်ဖိုး မမှန်ပါ")
    updated = dataclasses.replace(settings, **changes) if changes else settings
    problems = updated.validate()
    if problems:
        raise _error(400, "ဆက်တင် မမှန်ပါ: " + " / ".join(problems))
    return updated


@app.post("/api/transcribe")
def transcribe(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    video_id = str(payload.get("video_id") or "")
    record = _read_video_meta(video_id)
    settings = _settings_with_overrides(payload.get("options") or {})
    demo = bool(payload.get("demo"))
    if demo and not cfg.demo_enabled(settings):
        raise _error(403, "သရုပ်ပြ မုဒ် ပိတ်ထားပါသည်")
    whisper_ok, reason = cfg.whisper_available(settings)
    if not demo and not whisper_ok:
        raise _error(
            503,
            "စကားပြော မှတ်မိ system (Whisper) အသင့် မဖြစ်ပါ။ " + reason,
        )
    path = VIDEOS_DIR / str(record.get("path") or "")
    if not path.exists():
        raise _error(404, "ဗီဒီယိုဖိုင် မတွေ့ပါ")

    job = JOBS.create("transcribe")

    def run(current: Job) -> dict[str, Any]:
        if demo:
            result = sg.demo_transcribe(float(record.get("duration") or 22.0), settings)
            JOBS.progress(current, 1.0, "သရုပ်ပြ စာတန်း ဖန်တီးပြီးပါပြီ")
            result["demo"] = True
            return result
        result = sg.transcribe_video(
            str(path),
            settings,
            duration=float(record.get("duration") or 0.0),
            progress=lambda fraction, stage: JOBS.progress(
                current, (fraction * 0.85) if fraction is not None else None, stage
            ),
        )
        JOBS.progress(current, 0.9, "မြန်မာဘာသာသို့ ပြန်နေသည်...")
        result["segments"] = tr.translate_segments(
            result["segments"], str(result.get("language") or "")
        )
        result["translated_to"] = "my"
        JOBS.progress(current, 1.0, "ပြီးစီးပါပြီ")
        return result

    JOBS.submit(job, run)
    return {"job_id": job.id, "status": "queued"}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@app.post("/api/project/save")
def project_save(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        project = pm.save_project(payload)
    except Exception as exc:
        raise _error(500, "သိမ်းဆည်း၍ မရပါ", str(exc))
    return {"ok": True, "project": project, "message": "သိမ်းဆည်းပြီးပါပြီ"}


@app.get("/api/project/list")
def project_list() -> dict[str, Any]:
    return {"projects": pm.list_projects()}


@app.get("/api/project/{project_id}")
def project_get(project_id: str) -> dict[str, Any]:
    project = pm.get_project(project_id)
    if not project:
        raise _error(404, "ပရောဂျက် မတွေ့ပါ")
    return {"project": project}


@app.delete("/api/project/{project_id}")
def project_delete(project_id: str) -> dict[str, Any]:
    if not pm.delete_project(project_id):
        raise _error(404, "ပရောဂျက် မတွေ့ပါ")
    return {"ok": True, "message": "ပရောဂျက် ဖျက်ပြီးပါပြီ"}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _project_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    project_payload = payload.get("project")
    if not isinstance(project_payload, dict):
        project_id = str(payload.get("project_id") or "")
        existing = pm.get_project(project_id)
        if not existing:
            raise _error(400, "ပရောဂျက် အချက်အလက် မပါပါ")
        project_payload = existing
    return pm.normalise_project(project_payload)


@app.post("/api/export")
def export_video(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    settings = cfg.get_settings()
    project = _project_from_payload(payload)
    if not project["segments"]:
        raise _error(400, "စာတန်းထိုး တစ်ခုမှ မရှိသေးပါ")
    video_id = str(project.get("video", {}).get("id") or "")
    record = _read_video_meta(video_id)
    source = VIDEOS_DIR / str(record.get("path") or "")
    if not source.exists():
        raise _error(404, "ဗီဒီယိုဖိုင် မတွေ့ပါ — ပြန်တင်ပါ")
    ffmpeg = cfg.get_ffmpeg_info(settings)
    if not ffmpeg.ok:
        raise _error(
            503,
            "FFmpeg အသင့် မဖြစ်ပါ — မြန်မာစာတန်း burn-in လုပ်၍ မရပါ။ "
            + " / ".join(ffmpeg.problems),
        )

    duration = float(record.get("duration") or 0.0)
    job = JOBS.create("export")
    output = EXPORTS_DIR / f"{job.id}.mp4"
    filename = _safe_filename(f"{project['name']}-myanmar-subtitles.mp4", "export.mp4")

    def run(current: Job) -> dict[str, Any]:
        result = vr.render_video(
            video_path=source,
            segments=project["segments"],
            output_path=output,
            blur_boxes=[b for b in project["blurs"] if b.get("visible", True)],
            logos=[logo for logo in project["logos"] if logo.get("visible", True)],
            duration=duration,
            style_defaults=project.get("style_defaults") or {},
            settings=settings,
            progress=lambda fraction, stage: JOBS.progress(current, fraction, stage),
            cancelled=lambda: current.cancel_requested,
        )
        size = output.stat().st_size
        return {
            "download_url": f"/api/export/{current.id}/download",
            "filename": filename,
            "size": size,
            "duration": result.duration,
            "elapsed": result.elapsed,
            "segments": len(project["segments"]),
            "message": "ဗီဒီယို သိမ်းဆည်းပြီးပါပြီ",
        }

    JOBS.submit(job, run)
    return {"job_id": job.id, "status": "queued", "filename": filename}


@app.get("/api/export/{job_id}/download")
def export_download(job_id: str) -> Response:
    job = JOBS.get(job_id)
    if not job or job.status != "done" or not job.result:
        raise _error(404, "ရလဒ် မရှိသေးပါ")
    path = EXPORTS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise _error(404, "ရလဒ်ဖိုင် မတွေ့ပါ")
    filename = str(job.result.get("filename") or "export.mp4")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename,
        headers={"Cache-Control": "private, max-age=0"},
    )


@app.post("/api/srt")
def srt_download(payload: dict[str, Any] = Body(...)) -> Response:
    project = _project_from_payload(payload)
    if not project["segments"]:
        raise _error(400, "စာတန်းထိုး တစ်ခုမှ မရှိသေးပါ")
    target = TMP_DIR / f"srt-{uuid.uuid4().hex[:12]}.srt"
    try:
        vr.write_srt(project["segments"], target)
    except Exception as exc:
        raise _error(500, "SRT ဖိုင် ဖန်တီး၍ မရပါ", str(exc))
    filename = _safe_filename(f"{project['name']}.srt", "subtitles.srt")
    return FileResponse(
        target,
        media_type="text/plain; charset=utf-8",
        filename=filename,
        background=None,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/srt/{project_id}")
def srt_download_saved(project_id: str) -> Response:
    project = pm.get_project(project_id)
    if not project:
        raise _error(404, "ပရောဂျက် မတွေ့ပါ")
    return srt_download({"project": project})


# ---------------------------------------------------------------------------
# Frontend pages
# ---------------------------------------------------------------------------


def _page(name: str) -> Response:
    path = cfg.FRONTEND_DIR / name
    if not path.exists():
        raise _error(500, f"Frontend ဖိုင် မတွေ့ပါ: {name}")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/")
def index() -> Response:
    return _page("index.html")


@app.get("/editor")
@app.get("/editor/{project_id}")
def editor_page(project_id: str | None = None) -> Response:
    return _page("editor.html")


@app.get("/favicon.svg")
def favicon() -> Response:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#a855f7"/><stop offset="1" stop-color="#38bdf8"/>'
        "</linearGradient></defs>"
        '<rect width="64" height="64" rx="16" fill="url(#g)"/>'
        '<rect x="12" y="38" width="40" height="10" rx="5" fill="#fff" opacity="0.92"/>'
        '<circle cx="32" cy="24" r="8" fill="#fff"/>'
        "</svg>"
    )
    return Response(svg, media_type="image/svg+xml")


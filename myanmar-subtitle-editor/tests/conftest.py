"""Shared pytest fixtures: isolated storage + the best available FFmpeg."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MSE_UPLOADS_DIR", tempfile.mkdtemp(prefix="mse-test-uploads-"))
os.environ.setdefault("MSE_PROJECTS_DIR", tempfile.mkdtemp(prefix="mse-test-projects-"))
os.environ.pop("EDITOR_PASSWORD", None)

# Prefer a static FFmpeg that ships libass/harfbuzz when the host has none.
if not os.environ.get("FFMPEG_BINARY"):
    try:
        import imageio_ffmpeg

        os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        pass

import pytest  # noqa: E402

from backend import config as cfg  # noqa: E402


@pytest.fixture(scope="session")
def ffmpeg_info():
    info = cfg.get_ffmpeg_info()
    if not info.path:
        pytest.skip("FFmpeg မရှိသဖြင့် ဤစမ်းသပ်မှုကို ကျော်လိုက်သည်")
    return info


@pytest.fixture(scope="session")
def requires_libass(ffmpeg_info):
    if not ffmpeg_info.ok:
        pytest.skip("libass/harfbuzz ပါသော FFmpeg မရှိပါ: " + " / ".join(ffmpeg_info.problems))
    return ffmpeg_info


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory, requires_libass):
    """A tiny 3 second lavfi clip used by the render / API tests."""
    from backend import video_renderer as vr

    target = tmp_path_factory.mktemp("media") / "sample.mp4"
    if not target.exists():
        vr.make_demo_video(target, duration=3.0)
    return target

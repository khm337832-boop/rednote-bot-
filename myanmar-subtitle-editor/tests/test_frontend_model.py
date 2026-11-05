"""Browser-side model tests (run with Node) + static asset checks."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS_TEST = ROOT / "tests" / "js" / "project_model.test.js"


def test_frontend_project_model():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js မရှိသဖြင့် ကျော်လိုက်သည်")
    result = subprocess.run(
        [node, str(JS_TEST)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize(
    "name",
    [
        "frontend/index.html",
        "frontend/editor.html",
        "frontend/css/glass.css",
        "frontend/css/editor.css",
        "frontend/js/api.js",
        "frontend/js/dashboard.js",
        "frontend/js/editor.js",
        "frontend/js/project.js",
        "frontend/js/layers.js",
        "frontend/js/timeline.js",
        "frontend/js/myanmar_text.js",
        "frontend/assets/vendor/konva.min.js",
        "frontend/assets/fonts/NotoSansMyanmar-Regular.ttf",
        "frontend/assets/fonts/NotoSansMyanmar-Bold.ttf",
        "backend/fonts/NotoSansMyanmar-Regular.ttf",
    ],
)
def test_assets_exist(name):
    assert (ROOT / name).exists(), f"{name} မရှိပါ"


def test_html_uses_utf8_and_myanmar_font():
    for name in ("frontend/index.html", "frontend/editor.html"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert '<meta charset="UTF-8"' in text
        assert "/static/css/glass.css" in text
    css = (ROOT / "frontend/css/glass.css").read_text(encoding="utf-8")
    assert "@font-face" in css
    assert "NotoSansMyanmar-Regular.ttf" in css
    assert "backdrop-filter" in css and "-webkit-backdrop-filter" in css
    assert "rgba(255, 255, 255, 0.1)" in css.replace(" ", " ") or "rgba(255, 255, 255, 0.1)" in css
    assert "[data-theme=\"light\"]" in css


def test_editor_page_has_timeline_and_glass_panels():
    html = (ROOT / "frontend/editor.html").read_text(encoding="utf-8")
    for marker in (
        'id="timeline"',
        'id="stage"',
        'id="layer-list"',
        'id="props-body"',
        'id="btn-export"',
        'id="btn-srt"',
        'id="btn-save"',
        'id="btn-logo"',
        'id="btn-blur"',
        'id="btn-transcribe"',
        "konva.min.js",
    ):
        assert marker in html, marker


def test_all_ui_strings_are_burmese():
    """Quick sanity check: the main navigation labels are in Myanmar script."""
    html = (ROOT / "frontend/editor.html").read_text(encoding="utf-8")
    for label in ("သိမ်းရန်", "ဗီဒီယို သိမ်းရန် (Export)", "SRT ရယူရန်", "Logo ထည့်ရန်", "Blur ထည့်ရန်"):
        assert label in html

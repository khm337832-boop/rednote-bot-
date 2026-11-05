"""ASS/SRT ထုတ်လုပ်မှုနှင့် FFmpeg burn-in စမ်းသပ်မှုများ (မြန်မာစာ shaping ပါဝင်)."""
from __future__ import annotations

import statistics
import subprocess
from pathlib import Path

import pytest

from backend import config as cfg
from backend import video_renderer as vr

BURMESE = "မင်္ဂလာပါ ခင်ဗျာ"


def test_build_ass_header_and_dialogues():
    segments = [
        {"start": 0.0, "end": 2.5, "text": BURMESE, "style": {"rotation": 20, "opacity": 0.5}},
        {"start": 2.5, "end": 4.0, "text": "ဒီနေ့"},
    ]
    ass = vr.build_ass(segments, width=1920, height=1080)
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "Noto Sans Myanmar" in ass
    assert ass.count("Dialogue:") == 2
    assert "0:00:00.00,0:00:02.50" in ass
    assert "0:00:02.50,0:00:04.00" in ass
    # default position (0.5, 0.86) of a 1920x1080 frame
    assert "\\pos(960,929)" in ass
    # opacity 0.5 → ASS alpha 0x80
    assert "\\alpha&H80&" in ass
    assert "\\frz20.00" in ass


def test_build_ass_escapes_user_override_tags():
    ass = vr.build_ass(
        [{"start": 0, "end": 1, "text": "{\\pos(0,0)} မင်္ဂလာပါ"}], width=1280, height=720
    )
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue:")][0]
    assert "｛" in dialogue and "｝" in dialogue
    # exactly one override block (ours) — the user's braces were neutralised
    assert dialogue.count("{") == 1 and dialogue.count("}") == 1


def test_build_ass_wraps_long_text_with_explicit_line_breaks():
    text = "မြန်မာစာတန်းထိုး ဖန်တီးနည်းကို အဆင့်ဆင့် ရှင်းပြပါမယ် ကျေးဇူးတင်ပါတယ်"
    ass = vr.build_ass(
        [{"start": 0, "end": 5, "text": text, "style": {"width": 0.3}}],
        width=1280,
        height=720,
    )
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue:")][0]
    assert "\\N" in dialogue.split("}", 1)[1]


def test_build_ass_skips_invalid_segments():
    ass = vr.build_ass(
        [
            {"start": 2, "end": 1, "text": "ပျက်"},
            {"start": 0, "end": 1, "text": "   "},
            {"start": 0, "end": 1, "text": "ကောင်း"},
        ],
        width=640,
        height=360,
    )
    assert ass.count("Dialogue:") == 1


def test_build_ass_clamps_to_video_duration():
    ass = vr.build_ass(
        [{"start": 4.0, "end": 30.0, "text": "က"}], width=640, height=360, duration=10.0
    )
    assert "0:00:04.00,0:00:10.00" in ass


def test_filter_script_blur_and_logo():
    script = vr.build_filter_script(
        width=1280,
        height=720,
        blur_boxes=[{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.25, "intensity": 12}],
        logos=[
            {"x": 0.05, "y": 0.05, "w": 0.2, "h": 0.1, "rotation": 15, "opacity": 0.8}
        ],
        logo_input_index=[1],
    )
    assert "split=2" in script
    assert "boxblur=luma_radius=" in script
    assert "crop=384:180:128:144" in script
    assert "overlay=128:144" in script
    assert "colorchannelmixer=aa=0.800" in script
    assert "rotate=" in script
    assert "ass=subs.ass:fontsdir=fonts" in script
    assert "[vfinal]" in script


def test_filter_script_without_optional_layers():
    script = vr.build_filter_script(
        width=640, height=360, blur_boxes=[], logos=[], logo_input_index=[]
    )
    assert "[0:v]ass=subs.ass" in script
    assert "[vfinal]" in script


def test_blur_radius_scales_with_intensity_and_resolution():
    assert vr.blur_radius(1, 720, 720) < vr.blur_radius(20, 720, 720)
    assert vr.blur_radius(10, 2160, 2160) > vr.blur_radius(10, 720, 720)
    assert 1 <= vr.blur_radius(0, 720, 720) <= 120


def test_build_srt_and_bom(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.5, "text": BURMESE},
        {"start": 2.5, "end": 5.0, "text": "ဒီနေ့"},
    ]
    srt = vr.build_srt(segments)
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert BURMESE in srt
    target = tmp_path / "out.srt"
    vr.write_srt(segments, target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 with BOM
    assert "ဒီနေ့" in raw.decode("utf-8-sig")


def test_probe_video(sample_video):
    meta = vr.probe_video(sample_video)
    assert meta.width == 1280 and meta.height == 720
    assert 2.5 <= meta.duration <= 3.6
    assert meta.has_audio is True


def test_render_burns_myanmar_captions(sample_video, tmp_path):
    output = tmp_path / "rendered.mp4"
    result = vr.render_video(
        video_path=sample_video,
        segments=[
            {"start": 0.0, "end": 3.0, "text": BURMESE, "style": {"y": 0.8, "fontSize": 0.09}},
        ],
        output_path=output,
        duration=3.0,
    )
    assert output.exists() and output.stat().st_size > 1000
    assert result.duration == pytest.approx(3.0, abs=0.2)
    ffmpeg = cfg.get_ffmpeg_info().path
    frame = tmp_path / "frame.png"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-y", "-ss", "1.0", "-i", str(output), "-frames:v", "1", str(frame)],
        check=True,
        capture_output=True,
    )
    from PIL import Image

    with Image.open(frame) as image:
        width, height = image.size
        region = image.convert("RGB").crop((0, int(height * 0.68), width, height))
        white = sum(1 for pixel in region.getdata() if min(pixel) > 225)
    # စာတန်း စာလုံးများ (အဖြူ + အနက်ဘောင်) ပေါ်နေရမည်
    assert white > 400


def test_render_applies_blur_box(sample_video, tmp_path):
    source = sample_video
    ffmpeg = cfg.get_ffmpeg_info().path
    output = tmp_path / "blurred.mp4"
    vr.render_video(
        video_path=source,
        segments=[],
        output_path=output,
        blur_boxes=[{"x": 0.6, "y": 0.55, "w": 0.2, "h": 0.25, "intensity": 20}],
        duration=3.0,
    )

    def variance(path: Path, box: tuple[int, int, int, int]) -> float:
        frame = tmp_path / f"{path.stem}-{box[0]}.png"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-y", "-ss", "1.0", "-i", str(path), "-frames:v", "1", str(frame)],
            check=True,
            capture_output=True,
        )
        from PIL import Image

        with Image.open(frame) as image:
            grey = list(image.convert("L").crop(box).getdata())
        return statistics.pvariance(grey)

    box = (int(0.62 * 1280), int(0.57 * 720), int(0.78 * 1280), int(0.78 * 720))
    control = (int(0.05 * 1280), int(0.57 * 720), int(0.21 * 1280), int(0.78 * 720))
    assert variance(output, box) < variance(source, box)
    assert variance(output, control) == pytest.approx(variance(source, control), rel=0.25)


def test_extract_thumbnail(sample_video, tmp_path):
    target = tmp_path / "thumb.jpg"
    vr.extract_thumbnail(sample_video, target, at=1.0)
    assert target.exists() and target.stat().st_size > 500

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, patch

import pytest
import subtitle_app as app


def fake_ffmpeg(tmp_path, monkeypatch, script):
    executable = tmp_path / 'ffmpeg'
    executable.write_text('#!/usr/bin/env python3\n' + script)
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])


def test_progress_reports_timestamps_and_heartbeats(tmp_path, monkeypatch):
    fake_ffmpeg(tmp_path, monkeypatch,
                "import time\nprint('out_time_us=5000000', flush=True)\n"
                "time.sleep(.20)\nprint('out_time_us=20000000', flush=True)\ntime.sleep(.10)\n")
    edit = AsyncMock()
    with patch.object(app.bot, 'edit_text', edit):
        asyncio.run(app.make_preview('in', 'out', 10, 123, 456, report_interval=.05, timeout=3))
    texts = [call.args[2] for call in edit.await_args_list]
    assert any('50%' in text for text in texts)
    assert any('99%' in text for text in texts)
    assert all('100%' not in text for text in texts)
    assert all('နောက်ဆုံးတိုးခဲ့သည်မှာ' in text for text in texts)


def test_silent_process_still_reports_and_times_out(tmp_path, monkeypatch):
    fake_ffmpeg(tmp_path, monkeypatch, 'import time\ntime.sleep(30)\n')
    edit = AsyncMock()
    with patch.object(app.bot, 'edit_text', edit):
        with pytest.raises(RuntimeError, match='timed out'):
            asyncio.run(app.make_preview('in', 'out', 10, 123, 456, report_interval=.03, timeout=.15))
    assert edit.await_count >= 1
    assert '0%' in edit.await_args.args[2]


def test_ffmpeg_failure_is_not_success(tmp_path, monkeypatch):
    fake_ffmpeg(tmp_path, monkeypatch, "import sys\nsys.stderr.write('failure')\nsys.exit(1)\n")
    with pytest.raises(subprocess.CalledProcessError):
        asyncio.run(app.make_preview('in', 'out', 10, 123, None, timeout=3))


def test_real_ffmpeg_preview(tmp_path, monkeypatch):
    imageio_ffmpeg = pytest.importorskip('imageio_ffmpeg')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    (tmp_path / 'ffmpeg').symlink_to(ffmpeg)
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ['PATH'])
    source = tmp_path / 'source.mp4'
    target = tmp_path / 'preview.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=320x240:rate=24:duration=2', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=2', '-c:v', 'libx264', '-c:a', 'aac',
                    '-shortest', str(source)], check=True)
    asyncio.run(app.make_preview(source, target, 2, 1, None, timeout=30))
    assert target.stat().st_size > 0
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(target), '-map', '0:v:0',
                    '-map', '0:a:0', '-f', 'null', '-'], check=True)

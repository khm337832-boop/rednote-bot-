import asyncio
import json
import math
import subprocess
from unittest.mock import AsyncMock, patch

import pytest
import subtitle_app as app
from subtitle_engine import SubtitleCue, SubtitleStyle, burn_subtitles


def test_text_translation_retains_ids_and_timing_and_batches():
    async def run():
        original = [SubtitleCue(i, i + .5, 'မူရင်းစာ') for i in range(21)]
        calls = []
        async def generate(client, contents, **kwargs):
            assert len(contents) == 1 and isinstance(contents[0], str)
            payload = json.loads(contents[0].split('Source cues: ', 1)[1])
            calls.append(payload)
            response = {'translations': [{'id': x['id'], 'text': 'ပြင်ထားသောစာ'} for x in payload]}
            kwargs['validator'](response)
            return response
        with patch.object(app.bot, '_generate_direct_gemini_json', side_effect=generate):
            result = await app._translate_source_cues(object(), original, [])
        assert [len(c) for c in calls] == [20, 1]
        assert [(x.start, x.end) for x in result] == [(x.start, x.end) for x in original]
        assert all(c.text == 'ပြင်ထားသောစာ' for c in result)
    asyncio.run(run())


def test_review_rejects_missing_duplicate_or_non_burmese():
    for data in [{'translations': []}, {'translations': [{'id': 0, 'text': 'စာ'}, {'id': 0, 'text': 'စာ'}]},
                 {'translations': [{'id': 0, 'text': 'English'}]}, {'translations': [{'id': True, 'text': 'စာ'}]}]:
        with pytest.raises(ValueError):
            app._subtitle_translation_mapping(data, [0])


def test_cover_settings_validate():
    for style in [SubtitleStyle(cover='true'), SubtitleStyle(cover_top=math.nan), SubtitleStyle(cover_top=100)]:
        with pytest.raises(ValueError):
            style.validate()


def test_cover_is_burned_and_audio_preserved(tmp_path, monkeypatch):
    imageio_ffmpeg = pytest.importorskip('imageio_ffmpeg')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    (tmp_path / 'ffmpeg').symlink_to(ffmpeg)
    monkeypatch.setenv('PATH', str(tmp_path) + ':' + __import__('os').environ['PATH'])
    source = tmp_path / 'source.mp4'
    target = tmp_path / 'out.mp4'
    subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i', 'color=white:size=320x240:rate=10:duration=1',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'libx264', '-c:a', 'aac',
                    '-shortest', str(source)], check=True)
    burn_subtitles(str(source), str(target), [SubtitleCue(0, .9, 'test')], 320, 240, SubtitleStyle(50, 30, 4, True, 75), duration=1)
    # Decode a frame and assert the lower band is black, upper area remains white.
    frame = subprocess.run([ffmpeg, '-v', 'error', '-i', str(target), '-frames:v', '1', '-f', 'rawvideo',
                            '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
    def rgb(x, y): return frame[(y * 320 + x) * 3:(y * 320 + x) * 3 + 3]
    assert min(rgb(10, 20)) > 200
    assert max(rgb(10, 220)) < 20
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(target), '-map', '0:a:0', '-f', 'null', '-'], check=True)


def test_translation_overload_keeps_only_completed_windows(tmp_path):
    async def run():
        async def upload(client, path):
            return json.loads(__import__("pathlib").Path(path).read_text())
        def extract(source, path, start, length):
            path.write_text(json.dumps({'start': start}))
        translations = 0
        async def generate(client, contents, **kwargs):
            nonlocal translations
            if kwargs['purpose'] == 'subtitle text translation':
                translations += 1
                assert len(contents) == 1 and isinstance(contents[0], str)
                if translations == 2:
                    raise RuntimeError('503 UNAVAILABLE')
                data = {'translations': [{'id': 0, 'text': 'ပြင်ထားသောစာ'}]}
            else:
                assert 'Do not translate' in contents[1]
                data = {'cues': [{'start': 1, 'end': 3, 'text': 'original words'}]}
            kwargs['validator'](data)
            return data
        with patch.object(app.bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', side_effect=upload), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(app.bot, '_generate_direct_gemini_json', side_effect=generate), \
             patch.object(app.bot, 'send_text', AsyncMock()):
            with pytest.raises(RuntimeError, match='503'):
                await app.transcribe_translate('source', 120, str(tmp_path), 123, True)
        assert app._load_subtitle_checkpoint(tmp_path, 120, 60) == [[SubtitleCue(1, 3, 'ပြင်ထားသောစာ')]]
    asyncio.run(run())

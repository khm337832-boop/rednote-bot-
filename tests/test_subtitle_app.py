import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ['BOT_TOKEN'] = ''
os.environ['GEMINI_API_KEY'] = 'test-key'

import pytest
from fastapi.testclient import TestClient
import subtitle_app as app
from subtitle_engine import SubtitleCue, SubtitleStyle


def test_translation_cues_validate_relative_times_and_burmese():
    cues = app.parse_model_cues({'cues': [{'start': 1, 'end': 3, 'text': 'မင်္ဂလာပါ'}]}, 60, 10)
    assert cues == [SubtitleCue(61, 63, 'မင်္ဂလာပါ')]
    for cue in [{'start': -1, 'end': 3, 'text': 'စာ'},
                {'start': 11, 'end': 12, 'text': 'စာ'},
                {'start': 0, 'end': 1, 'text': 'hello'},
                {'start': float('nan'), 'end': 1, 'text': 'စာ'}]:
        with pytest.raises(ValueError):
            app.parse_model_cues({'cues': [cue]}, 0, 10)


def test_editor_requires_secret_session_and_origin():
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / 'preview.mp4'
        source.write_bytes(b'fake video')
        token = 'test-capability'
        app.SESSIONS[token] = {'created': time.time(), 'dir': temp, 'source': str(source),
                               'preview': str(source), 'chat': 123, 'duration': 3,
                               'width': 320, 'height': 240, 'cues': [SubtitleCue(0, 2, 'စာ')],
                               'rendering': False}
        with TestClient(app.app, base_url='https://testserver') as client:
            assert client.get('/editor/data').status_code == 404
            assert client.post('/editor/session', json={'token': token}, headers={'Origin':'https://evil.example'}).status_code == 403
            assert client.post('/editor/session', json={'token': 'wrong'}, headers={'Origin':'https://testserver'}).status_code == 404
            assert client.post('/editor/session', json={'token': token}, headers={'Origin':'https://testserver'}).status_code == 200
            assert client.get('/editor/data').json()['cues'][0]['text'] == 'စာ'
            assert client.get('/editor/video').content == b'fake video'
            payload = {'x':50, 'y':90, 'size':4, 'cues':[{'start':0,'end':5,'text':'စာ'}]}
            assert client.post('/editor/render', json=payload, headers={'Origin':'https://testserver'}).status_code == 400
            assert client.post('/editor/render', json=payload, headers={'Origin':'https://evil.example'}).status_code == 403
        app.SESSIONS.clear()


def test_partial_delivery_resumes_without_reuploading_parts():
    async def run():
        with tempfile.TemporaryDirectory() as temp:
            parts = [str(Path(temp) / f'{n}.mp4') for n in range(3)]
            for p in parts:
                Path(p).write_bytes(b'video')
            session = {'chat': 123, 'dir': temp, 'delivery_parts': parts, 'delivered_count':1, 'rendering':True}
            upload = AsyncMock(return_value={'ok': True})
            with patch.object(app.bot, 'tg_upload_media', upload), patch.object(app.bot, 'send_text', AsyncMock()):
                await app.deliver(session, [SubtitleCue(0,1,'စာ')], SubtitleStyle())
            assert upload.await_count == 2
            assert [call.args[4] for call in upload.await_args_list] == parts[1:]
            assert session['delivered_count'] == 3 and session['done']
    asyncio.run(run())


def test_local_media_pipeline_with_stubbed_translation_and_telegram():
    import shutil
    import subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe not installed')

    async def run():
        with tempfile.TemporaryDirectory() as temp:
            source = str(Path(temp) / 'input.mp4')
            subprocess.run(['ffmpeg', '-nostdin', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=320x240:rate=24:duration=2', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-shortest', source], check=True, capture_output=True)
            async def download(file_id, target, **kwargs):
                shutil.copy(source, target)
                return 2.0
            upload = AsyncMock(return_value={'ok': True})
            async def subtitles(*args):
                return [SubtitleCue(0, 1.5, 'မင်္ဂလာပါ')]
            with patch.object(app.bot, 'download_telegram_file', download), \
                 patch.object(app, 'transcribe_translate', subtitles), \
                 patch.object(app.bot, 'send_text', AsyncMock()), \
                 patch.object(app.bot, 'tg_upload_media', upload):
                await app.create_job(123, 1, {'file_id':'test'}, '')
                session = next(s for s in app.SESSIONS.values() if s['chat'] == 123)
                assert Path(session['preview']).exists()
                session['rendering'] = True
                await app.deliver(session, session['cues'], SubtitleStyle())
                assert session.get('done') is True
                assert upload.await_count == 1
                duration, width, height = app.ffprobe(str(Path(session['dir']) / 'final.mp4'))
                assert (round(duration), width, height) == (2, 320, 240)
                shutil.rmtree(session['dir'])
                app.SESSIONS.clear()
    asyncio.run(run())

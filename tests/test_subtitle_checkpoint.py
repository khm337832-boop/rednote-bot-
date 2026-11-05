import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import subtitle_app as app
import bot
from subtitle_engine import SubtitleCue


def test_503_then_retry_resumes_completed_windows_without_duplicate(tmp_path):
    async def run():
        calls = []
        async def upload(client, path):
            return json.loads(Path(path).read_text())
        def extract(source, path, start, length):
            path.write_text(json.dumps({'start': start, 'length': length}))
        async def generate(client, contents, **kwargs):
            audio = contents[0]
            calls.append(audio['start'])
            if audio['start'] == 60 and calls.count(60) == 1:
                raise RuntimeError('Gemini subtitle translation failed: 503 UNAVAILABLE high demand')
            data = {'cues': [{'start': 1, 'end': 3, 'text': 'စာတန်း'}]}
            kwargs['validator'](data)
            return data
        with patch.object(bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(bot, '_upload_video_to_gemini', side_effect=upload), \
             patch.object(bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(bot, '_generate_direct_gemini_json', side_effect=generate), \
             patch.object(bot, 'send_text', AsyncMock()):
            with pytest.raises(RuntimeError, match='503'):
                await app.transcribe_translate('source', 120, str(tmp_path), 123, True)
            saved = app._load_subtitle_checkpoint(tmp_path, 120, 60)
            assert saved == [[SubtitleCue(1, 3, 'စာတန်း')]]
            result = await app.transcribe_translate('source', 120, str(tmp_path), 123, True)
        assert calls == [0, 60, 60]
        assert result == [SubtitleCue(1, 3, 'စာတန်း'), SubtitleCue(61, 63, 'စာတန်း')]
        assert len(app._load_subtitle_checkpoint(tmp_path, 120, 60)) == 2
        assert not list(tmp_path.glob('*.mp3'))
    asyncio.run(run())


def test_corrupt_or_wrong_duration_checkpoint_fails_closed(tmp_path):
    path = tmp_path / 'subtitle-checkpoint.json'
    path.write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError):
        app._load_subtitle_checkpoint(tmp_path, 120, 60)
    app._save_subtitle_checkpoint(tmp_path, 120, 60, [[SubtitleCue(1, 2, 'စာ')]])
    with pytest.raises(ValueError):
        app._load_subtitle_checkpoint(tmp_path, 121, 60)
    assert path.stat().st_mode & 0o077 == 0


def test_provider_outage_keeps_source_and_retry_is_private_per_chat(tmp_path):
    async def run():
        chat = 123
        jobdir = tmp_path / 'job'
        jobdir.mkdir()
        (jobdir / 'source.mp4').write_bytes(b'source')
        app.ACTIVE.add(chat)
        outage = RuntimeError('Gemini subtitle translation failed: 503 UNAVAILABLE high demand')
        with patch.object(app.bot, 'send_text', AsyncMock(return_value=1)) as send, \
             patch.object(app, 'ffprobe', return_value=(120, 320, 240)), \
             patch.object(app, 'transcribe_translate', AsyncMock(side_effect=outage)):
            await app.create_job(chat, 1, None, '', uploaded_dir=str(jobdir))
        assert app.RETRY_JOBS[chat]['directory'] == str(jobdir)
        assert (jobdir / 'source.mp4').exists() and chat not in app.ACTIVE
        assert '/retry' in send.await_args_list[-1].args[1]
        update = {'message': {'message_id': 2, 'chat': {'id': chat, 'type': 'private'},
                              'from': {'id': chat}, 'text': '/retry'}}
        restart = AsyncMock()
        with patch.object(bot, 'is_allowed', return_value=True), patch.object(app, 'create_job', restart):
            await app.process_update(update)
        assert restart.await_args.kwargs['uploaded_dir'] == str(jobdir)
        assert chat in app.ACTIVE and chat not in app.RETRY_JOBS
        app.ACTIVE.clear()
        app.RETRY_JOBS.clear()
    asyncio.run(run())


def test_non_provider_failure_does_not_keep_private_source(tmp_path):
    async def run():
        jobdir = tmp_path / 'bad'
        jobdir.mkdir()
        (jobdir / 'source.mp4').write_bytes(b'source')
        app.ACTIVE.add(123)
        with patch.object(app.bot, 'send_text', AsyncMock(return_value=1)), \
             patch.object(app, 'ffprobe', return_value=(10, 320, 240)), \
             patch.object(app, 'transcribe_translate', AsyncMock(side_effect=ValueError('invalid timing'))):
            await app.create_job(123, 1, None, '', uploaded_dir=str(jobdir))
        assert not jobdir.exists() and 123 not in app.RETRY_JOBS
    asyncio.run(run())


def test_expired_or_cancelled_retry_removes_source(tmp_path):
    async def run():
        expired = tmp_path / 'expired'
        expired.mkdir()
        app.RETRY_JOBS[111] = {'created': time.time() - app.RETRY_TTL - 1, 'directory': str(expired)}
        app.clean_expired()
        assert 111 not in app.RETRY_JOBS and not expired.exists()
        active = tmp_path / 'cancel'
        active.mkdir()
        app.RETRY_JOBS[123] = {'created': time.time(), 'directory': str(active)}
        update = {'message': {'message_id': 2, 'chat': {'id': 123, 'type': 'private'},
                              'from': {'id': 123}, 'text': '/cancel'}}
        with patch.object(bot, 'is_allowed', return_value=True), patch.object(bot, 'send_text', AsyncMock()):
            await app.process_update(update)
        assert 123 not in app.RETRY_JOBS and not active.exists()
    asyncio.run(run())

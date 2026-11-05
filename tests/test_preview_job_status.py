import asyncio
from unittest.mock import AsyncMock, patch

import subtitle_app as app
from subtitle_engine import SubtitleCue


def test_job_reports_preview_failure_instead_of_completion(tmp_path):
    async def run():
        directory = tmp_path / 'job'
        directory.mkdir()
        (directory / 'source.mp4').write_bytes(b'video')
        app.ACTIVE.add(123)
        send = AsyncMock(return_value=99)
        edit = AsyncMock()
        with patch.object(app.bot, 'send_text', send), patch.object(app.bot, 'edit_text', edit), \
             patch.object(app, 'ffprobe', return_value=(10, 320, 240)), \
             patch.object(app, 'transcribe_translate', AsyncMock(return_value=[SubtitleCue(0, 1, 'စာ')])), \
             patch.object(app, 'make_preview', AsyncMock(side_effect=RuntimeError('preview error'))):
            await app.create_job(123, 1, None, '', uploaded_dir=str(directory))
        assert 'မအောင်မြင်' in edit.await_args.args[2]
        assert not directory.exists()
        assert 123 not in app.ACTIVE
        assert any('preview error' in call.args[1] for call in send.await_args_list)
    asyncio.run(run())

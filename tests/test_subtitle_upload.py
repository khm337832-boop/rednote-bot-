import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import subtitle_app as app


def test_large_telegram_attachment_offers_browser_upload_without_download():
    async def run():
        update = {'message': {'message_id': 8, 'chat': {'id': 123, 'type': 'private'},
                              'from': {'id': 456}, 'video': {'file_id': 'id', 'file_size': 78283633}}}
        with patch.object(app.bot, 'is_allowed', return_value=True), \
             patch.object(app.bot, 'send_text', AsyncMock(return_value=42)) as send, \
             patch.object(app.bot, 'download_telegram_file', AsyncMock()) as download:
            await app.process_update(update)
            assert download.await_count == 0
            assert send.await_count == 1
            link = send.await_args.kwargs['reply_markup']['inline_keyboard'][0][0]['url']
            assert '/upload#' in link and link.split('#')[1] in app.UPLOAD_SESSIONS
        app.UPLOAD_SESSIONS.clear()
    asyncio.run(run())


def test_browser_upload_authorization_and_large_file_handoff():
    token = 'private-upload-token'
    app.UPLOAD_SESSIONS[token] = {'created': time.time(), 'chat': 123, 'reply': 8, 'uploading': False, 'used': False}
    handed_off = []

    async def fake_job(chat, reply, attachment, url, *, uploaded_dir=None):
        source = Path(uploaded_dir) / 'source.mp4'
        handed_off.append((chat, reply, source.stat().st_size, source.read_bytes()[:4]))
        import shutil
        shutil.rmtree(uploaded_dir)
        app.ACTIVE.discard(chat)

    # The exact reported size: this exercises the large-body path without
    # depending on the public Telegram API or a real provider.
    body = b'video' * 15656726 + b'xx' + b'\0'  # 78,283,633 bytes
    assert len(body) == 78283633
    with patch.object(app, 'create_job', fake_job), patch.object(app, 'UPLOAD_DISK_RESERVE', 0):
        with TestClient(app.app, base_url='https://testserver') as client:
            origin = {'Origin': 'https://testserver'}
            assert client.post('/upload/video', content=b'test', headers=origin).status_code == 404
            assert client.post('/upload/session', json={'token': token}, headers={'Origin':'https://evil.example'}).status_code == 403
            assert client.post('/upload/session', json={'token': 'wrong'}, headers=origin).status_code == 404
            assert client.post('/upload/session', json={'token': token}, headers=origin).status_code == 200
            assert client.post('/upload/video', content=body, headers={'Origin':'https://evil.example'}).status_code == 403
            offset = 0
            while offset < len(body):
                chunk = body[offset:offset + 4 * 1024 * 1024]
                headers = {**origin, 'X-Upload-Offset':str(offset), 'X-Upload-Total':str(len(body))}
                result = client.post('/upload/video', content=chunk, headers=headers)
                assert result.status_code == 200
                offset = result.json()['received']
            assert result.json()['complete'] is True
            client.get('/health')
            assert client.post('/upload/video', content=b'video', headers=origin).status_code == 404
            assert handed_off == [(123, 8, 78283633, b'vide')]
    app.UPLOAD_SESSIONS.clear()
    app.ACTIVE.clear()


def test_failed_upload_releases_lock_and_leaves_link_retryable():
    token = 'retry-upload-token'
    session = {'created': time.time(), 'chat': 123, 'reply': 8, 'uploading': False, 'used': False}
    app.UPLOAD_SESSIONS[token] = session
    with tempfile.TemporaryDirectory() as temp:
        with patch.object(app, 'UPLOAD_DISK_RESERVE', 10**20), \
             patch.object(app.tempfile, 'tempdir', temp):
            with TestClient(app.app, base_url='https://testserver') as client:
                origin = {'Origin': 'https://testserver'}
                assert client.post('/upload/session', json={'token': token}, headers=origin).status_code == 200
                assert client.post('/upload/video', content=b'video', headers={**origin, 'X-Upload-Offset':'0', 'X-Upload-Total':'5'}).status_code == 507
                assert not session['used'] and not session['uploading'] and 123 not in app.ACTIVE
                assert list(Path(temp).iterdir()) == []
    app.UPLOAD_SESSIONS.clear()


def test_expired_upload_link_rejected():
    token = 'expired'
    app.UPLOAD_SESSIONS[token] = {'created': time.time() - app.UPLOAD_TTL - 1, 'chat':123, 'used':False}
    with TestClient(app.app, base_url='https://testserver') as client:
        assert client.post('/upload/session', json={'token': token}, headers={'Origin':'https://testserver'}).status_code == 404
    app.clean_expired()
    assert token not in app.UPLOAD_SESSIONS


def test_chunk_order_status_cancel_and_cleanup():
    token = 'chunks'
    session = {'created': time.time(), 'chat': 999, 'reply': 1, 'uploading': False, 'used': False}
    app.UPLOAD_SESSIONS[token] = session
    with patch.object(app, 'UPLOAD_DISK_RESERVE', 0):
        with TestClient(app.app, base_url='https://testserver') as client:
            origin = {'Origin': 'https://testserver'}
            assert client.post('/upload/session', json={'token': token}, headers=origin).status_code == 200
            hdr = {**origin, 'X-Upload-Offset': '0', 'X-Upload-Total': '10'}
            assert client.post('/upload/video', content=b'abcde', headers=hdr).json()['received'] == 5
            path = Path(session['directory']) / 'source.mp4'
            assert path.read_bytes() == b'abcde'
            # Duplicate offsets are rejected instead of appending twice.
            assert client.post('/upload/video', content=b'abcde', headers=hdr).status_code == 409
            assert client.get('/upload/status').json()['received'] == 5
            assert path.read_bytes() == b'abcde'
            assert client.post('/upload/cancel', headers={'Origin':'https://evil.example'}).status_code == 403
            assert client.post('/upload/cancel', headers=origin).status_code == 200
            assert not path.exists() and 999 not in app.ACTIVE
            assert client.post('/upload/video', content=b'abcde', headers=hdr).status_code == 404
    app.UPLOAD_SESSIONS.clear()


def test_requesting_new_upload_discards_idle_partial():
    async def run():
        with tempfile.TemporaryDirectory() as temp:
            partial = Path(temp) / 'partial'
            partial.mkdir()
            (partial / 'source.mp4').write_bytes(b'partial')
            app.UPLOAD_SESSIONS['old'] = {'created': time.time(), 'chat':123, 'directory':str(partial), 'uploading':False, 'used':False}
            app.ACTIVE.add(123)
            with patch.object(app.bot, 'send_text', AsyncMock(return_value=1)):
                await app.send_upload_link(123)
            assert not partial.exists() and 123 not in app.ACTIVE
            assert 'old' not in app.UPLOAD_SESSIONS
        app.UPLOAD_SESSIONS.clear()
    asyncio.run(run())

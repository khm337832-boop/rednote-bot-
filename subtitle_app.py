"""Telegram Burmese subtitle bot. The browser editor is capability protected."""
import asyncio
import hmac
import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

import bot
from subtitle_engine import SubtitleCue, SubtitleStyle, burn_subtitles, make_ass

log = logging.getLogger('subtitle-bot')
SESSIONS = {}
ACTIVE = set()
SEEN_UPDATES = {}
TTL = 24 * 3600
VERSION = 'subtitle-v1'
WEBHOOK_READY = False
EDITOR = Path(__file__).with_name('subtitle_editor.html')


def session_for(request):
    token = request.cookies.get('subtitle_session', '')
    session = SESSIONS.get(token)
    if not session or time.time() - session['created'] > TTL:
        raise HTTPException(404, 'Editor session expired; send the video again')
    return session


def check_origin(request):
    origin = request.headers.get('origin', '')
    expected = str(bot.PUBLIC_BASE_URL or '').rstrip('/')
    parsed = urlsplit(origin)
    host = request.headers.get('host', '')
    # The preview proxy has its own host. A browser cannot forge Origin or
    # Host, so an exact host match also works safely behind that proxy.
    matches_host = parsed.scheme in ('http', 'https') and parsed.netloc == host and not parsed.path
    if not origin or not (hmac.compare_digest(origin, expected) or matches_host):
        raise HTTPException(403, 'Invalid origin')


def clean_expired():
    for token, session in list(SESSIONS.items()):
        if time.time() - session['created'] > TTL and not session.get('rendering'):
            SESSIONS.pop(token, None)
            shutil.rmtree(session['dir'], ignore_errors=True)


def ffprobe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
        'format=duration:stream=codec_type,width,height:stream_tags=rotate:stream_side_data=rotation',
        '-of', 'json', str(path)], check=True, capture_output=True, text=True, timeout=60)
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    width, height = int(stream['width']), int(stream['height'])
    rotation = int((stream.get('tags') or {}).get('rotate') or 0)
    for side in stream.get('side_data_list', []):
        rotation = int(side.get('rotation', rotation))
    if abs(rotation) % 180 == 90:
        width, height = height, width
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid video')
    if not any(s.get('codec_type') == 'audio' for s in info['streams']):
        raise ValueError('Video has no audio to translate')
    return duration, width, height


def parse_model_cues(data, offset, length):
    if not isinstance(data, dict) or not isinstance(data.get('cues'), list):
        raise ValueError('Translation did not return timed subtitles')
    if len(data['cues']) > 300:
        raise ValueError('Too many subtitles in one audio chunk')
    cues = []
    for item in data['cues']:
        if not isinstance(item, dict):
            raise ValueError('Invalid subtitle')
        start, end = float(item['start']), float(item['end'])
        text = str(item.get('text') or '').strip()
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= length or end <= start or end > length + .05 or not text or len(text) > 500:
            raise ValueError('Invalid subtitle timing or text')
        if not any('\u1000' <= c <= '\u109f' for c in text):
            raise ValueError('Translation is not Burmese')
        cues.append(SubtitleCue(round(offset + start, 3), round(offset + min(end, length), 3), text))
    cues.sort(key=lambda c: c.start)
    return cues


def make_preview(source, target):
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-i', str(source),
                    '-vf', "scale=w='min(960,iw)':h=-2", '-map', '0:v:0', '-map', '0:a:0?',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k',
                    '-movflags', '+faststart', str(target)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)


def extract_audio(source, target, start, length):
    subprocess.run(['ffmpeg', '-nostdin', '-y', '-ss', str(start), '-i', str(source),
                    '-t', str(length), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'libmp3lame', '-b:a', '32k', str(target)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)


async def transcribe_translate(source, duration, directory, chat):
    client = bot._make_gemini_client()
    cues = []
    chunk_seconds = 60
    count = math.ceil(duration / chunk_seconds)
    for index in range(count):
        start = index * chunk_seconds
        length = min(chunk_seconds, duration - start)
        audio = Path(directory) / f'audio-{index}.mp3'
        await asyncio.to_thread(extract_audio, source, audio, start, length)
        uploaded = None
        try:
            uploaded = await bot._upload_video_to_gemini(client, str(audio))
            prompt = ('Listen to the complete audio carefully. Transcribe and translate every audible spoken line into natural, clear Burmese subtitles, not a summary. '
                      'Preserve meaning, names and dialogue; do not invent content. Use Burmese script for names where possible. '
                      f'Return JSON only: {{"cues":[{{"start":0.0,"end":2.0,"text":"မြန်မာစာ"}}]}}. '
                      f'Timestamps are seconds relative to this audio chunk, from 0 to {length:.3f}; use short readable cues synchronized to speech. '
                      'For silence only return an empty cues array. Never omit audible dialogue intentionally.')
            if cues:
                previous = [c.text for c in cues[-6:]]
                prompt += (' Previous subtitle context for consistent names only; do not repeat these lines unless heard in this audio: '
                           + json.dumps(previous, ensure_ascii=False))
            data = await bot._generate_direct_gemini_json(client, [uploaded, prompt], purpose='subtitle translation', max_output_tokens=8192)
            cues.extend(parse_model_cues(data, start, length))
        finally:
            await bot._delete_gemini_file(client, uploaded)
            audio.unlink(missing_ok=True)
        if count > 1:
            await bot.send_text(chat, f'ဘာသာပြန်နေပါသည် ({index + 1}/{count})')
    if not cues:
        raise ValueError('No spoken subtitles were detected; please check the source audio')
    return cues


async def create_job(chat, reply, attachment, url):
    if chat in ACTIVE:
        await bot.send_text(chat, 'လက်ရှိ video တစ်ခု လုပ်နေပါသည်။ ပြီးမှ နောက်တစ်ခု ပို့ပါ။', reply_to=reply)
        return
    ACTIVE.add(chat)
    directory = tempfile.mkdtemp(prefix='subtitle-job-')
    kept = False
    try:
        clean_expired()
        await bot.send_text(chat, '🎬 Video ကို ဒေါင်းပြီး မြန်မာစာတန်း ဘာသာပြန်နေပါသည်။', reply_to=reply)
        source = str(Path(directory) / 'source.mp4')
        if attachment:
            await bot.download_telegram_file(attachment['file_id'], source, expected_size=attachment.get('file_size', 0), raise_on_error=True)
        else:
            await bot.download_full(url, source, raise_on_error=True)
        duration, width, height = await asyncio.to_thread(ffprobe, source)
        cues = await transcribe_translate(source, duration, directory, chat)
        await bot.send_text(chat, '🖼️ Browser preview ပြုလုပ်နေပါသည်။')
        preview = str(Path(directory) / 'preview.mp4')
        await asyncio.to_thread(make_preview, source, preview)
        base = bot.PUBLIC_BASE_URL.rstrip('/')
        if not base:
            raise RuntimeError('Public editor URL is not configured')
        token = secrets.token_urlsafe(32)
        SESSIONS[token] = {'created': time.time(), 'dir': directory, 'source': source,
                           'preview': preview, 'chat': chat, 'duration': duration,
                           'width': width, 'height': height, 'cues': cues, 'rendering': False}
        message_id = await bot.send_text(chat, '✅ စာတန်းပြင်ဆင်ပြီးပါပြီ။ Link ကိုဖွင့်ပြီး preview ကြည့်၊ စာတန်းကို ဆွဲရွှေ့၊ size နှင့် စာသား/အချိန် ပြင်ပါ။ ပြီးလျှင် Final ပို့ရန် နှိပ်ပါ။\n\n🔒 Link ကို အခြားသူနှင့် မမျှဝေပါနှင့်။ Session သည် အများဆုံး ၂၄ နာရီခံပြီး server restart ဖြစ်လျှင် ပျောက်နိုင်ပါသည်။',
            reply_markup={'inline_keyboard': [[{'text': '📝 စာတန်းပြင်ရန်', 'url': f'{base}/editor#{token}'}]]})
        if not message_id:
            SESSIONS.pop(token, None)
            raise RuntimeError('Could not send the editor link to Telegram')
        kept = True
    except Exception as exc:
        log.exception('Subtitle preparation failed')
        await bot.send_text(chat, '❌ စာတန်းပြင်ဆင်၍ မရပါ: ' + bot._user_error_text(exc), reply_to=reply)
    finally:
        ACTIVE.discard(chat)
        if not kept:
            shutil.rmtree(directory, ignore_errors=True)


async def deliver(session, cues, style):
    chat, directory = session['chat'], session['dir']
    try:
        await bot.send_text(chat, '🎞️ စာတန်းထိုးပြီး final video ပြုလုပ်နေပါသည်။ မူရင်းအသံကို ထိန်းထားပါမည်။')
        parts = session.get('delivery_parts')
        if not parts:
            out = str(Path(directory) / 'final.mp4')
            await asyncio.to_thread(burn_subtitles, session['source'], out, cues, session['width'], session['height'], style, duration=session['duration'])
            actual, _, _ = await asyncio.to_thread(ffprobe, out)
            if abs(actual - session['duration']) > max(1.0, session['duration'] * .01):
                raise ValueError('Final video duration does not match source')
            parts = await bot.split_for_telegram(out, directory)
            if not parts or any(os.path.getsize(p) > bot.MAX_TG_BYTES for p in parts):
                raise ValueError('Could not prepare all Telegram parts')
            validation = await asyncio.to_thread(bot._validate_telegram_delivery_parts, parts, actual)
            if not validation.get('ok'):
                raise ValueError('Telegram parts do not cover the final video')
            session['delivery_parts'] = parts
        for i in range(session.get('delivered_count', 0), len(parts)):
            part = parts[i]
            result = await bot.tg_upload_media('sendVideo', {'chat_id': str(chat), 'caption': f'စာတန်းထိုး video {i+1}/{len(parts)}'}, 'video', os.path.basename(part), part, 'video/mp4')
            if not result.get('ok'):
                raise RuntimeError(f'Telegram delivery failed at part {i+1}/{len(parts)}')
            session['delivered_count'] = i + 1
        await bot.send_text(chat, f'✅ Final video {len(parts)}/{len(parts)} ပိုင်း အကုန်ပို့ပြီးပါပြီ။')
        session['done'] = True
    except Exception as exc:
        log.exception('Subtitle rendering or delivery failed')
        await bot.send_text(chat, '❌ Final မပြီးပါ: ' + bot._user_error_text(exc) + '\nEditor မှာ ထပ်ကြိုးစားနိုင်ပါသည်။')
    finally:
        session['rendering'] = False


async def cleanup_loop():
    while True:
        await asyncio.sleep(900)
        clean_expired()


@asynccontextmanager
async def lifespan(app):
    global WEBHOOK_READY
    WEBHOOK_READY = False
    if bot.BOT_TOKEN and bot.PUBLIC_BASE_URL:
        secret = bot._telegram_webhook_secret()
        commands = await bot.tg_api('setMyCommands', {'commands': [{'command':'start', 'description':'မြန်မာစာတန်းထိုးရန်'}]}, timeout=15)
        result = await bot.tg_api('setWebhook', {'url': bot.PUBLIC_BASE_URL.rstrip('/') + '/webhook', 'secret_token': secret, 'allowed_updates':['message']}, timeout=15)
        WEBHOOK_READY = bool(result.get('ok'))
        if not commands.get('ok') or not WEBHOOK_READY:
            log.error('Telegram setup failed; bot may not receive messages')
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        for session in SESSIONS.values():
            shutil.rmtree(session['dir'], ignore_errors=True)


app = FastAPI(lifespan=lifespan)


@app.get('/health')
@app.get('/healthz')
async def health():
    return {'ok': True, 'version': VERSION, 'render_git_commit': os.getenv('RENDER_GIT_COMMIT', ''), 'telegram_webhook_ready': WEBHOOK_READY, 'active_jobs': len(ACTIVE)}


@app.get('/editor')
async def editor():
    return HTMLResponse(EDITOR.read_text(encoding='utf-8'), headers={'Cache-Control':'no-store', 'Referrer-Policy':'no-referrer', 'X-Content-Type-Options':'nosniff'})


@app.post('/editor/session')
async def open_session(request: Request):
    check_origin(request)
    body = await request.json()
    token = str(body.get('token') or '')
    session = SESSIONS.get(token)
    if not session or time.time() - session['created'] > TTL:
        raise HTTPException(404, 'Session expired')
    response = JSONResponse({'ok': True}, headers={'Cache-Control':'no-store'})
    response.set_cookie('subtitle_session', token, httponly=True, secure=request.url.scheme == 'https' or bot.PUBLIC_BASE_URL.startswith('https://'), samesite='strict', max_age=TTL, path='/editor')
    return response


@app.get('/editor/data')
async def editor_data(request: Request):
    session = session_for(request)
    style = session.get('delivery_style') or SubtitleStyle()
    return JSONResponse({'width':session['width'], 'height':session['height'], 'duration':session['duration'], 'cues':[c.__dict__ for c in session['cues']], 'done':session.get('done', False), 'rendering':session.get('rendering', False), 'delivered_count':session.get('delivered_count', 0), 'style':style.__dict__}, headers={'Cache-Control':'no-store'})


@app.get('/editor/video')
async def editor_video(request: Request):
    session = session_for(request)
    return FileResponse(session['preview'], media_type='video/mp4', headers={'Cache-Control':'no-store'})


@app.get('/editor/font')
async def editor_font():
    path = '/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf'
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, media_type='font/ttf')


@app.post('/editor/render')
async def render(request: Request):
    check_origin(request)
    session = session_for(request)
    if session['rendering'] or session.get('done'):
        raise HTTPException(409, 'Already rendering or delivered')
    try:
        data = await request.json()
        items = data['cues']
        if not isinstance(items, list) or not (0 < len(items) <= 10000):
            raise ValueError('Invalid cues')
        cues = [SubtitleCue(float(c['start']), float(c['end']), str(c['text']).strip()) for c in items]
        if any(len(c.text) > 500 for c in cues):
            raise ValueError('Subtitle text too long')
        style = SubtitleStyle(float(data['x']), float(data['y']), float(data['size']))
        make_ass(cues, session['width'], session['height'], style, duration=session['duration'])
    except (ValueError, TypeError, KeyError, OverflowError):
        raise HTTPException(400, 'Invalid subtitle text, timing, or style')
    if session.get('delivered_count'):
        if cues != session['cues'] or style != session.get('delivery_style'):
            raise HTTPException(409, 'Some parts were already sent. Restore the same settings to resume delivery.')
    else:
        session.pop('delivery_parts', None)
        session['cues'] = cues
        session['delivery_style'] = style
    session['rendering'] = True
    asyncio.create_task(deliver(session, cues, style))
    return {'ok': True}


async def process_update(update):
    msg = update.get('message') or {}
    chat = (msg.get('chat') or {}).get('id')
    user = (msg.get('from') or {}).get('id')
    if not chat or not bot.is_allowed(user):
        return
    if (msg.get('chat') or {}).get('type') != 'private':
        await bot.send_text(chat, '🔒 Editor link ကို ကာကွယ်ရန် bot ကို private chat မှာ သုံးပါ။')
        return
    text = (msg.get('text') or msg.get('caption') or '').strip()
    attachment = bot._telegram_video_attachment(msg)
    url = bot._extract_first_url(text)
    if attachment or url:
        await create_job(chat, msg.get('message_id'), attachment, url)
    else:
        await bot.send_text(chat, '🎬 Video သို့မဟုတ် video link ပို့ပါ။ မူရင်းအသံကိုထားပြီး မြန်မာစာတန်း ဘာသာပြန်ပါမယ်။ Browser မှာ စာတန်းကို ဆွဲရွှေ့၊ size နဲ့ စာသားပြင်ပြီး Final ပို့နိုင်ပါတယ်။ Telegram တိုက်ရိုက် upload သည် Bot API ၏ 20MB ကန့်သတ်ချက်ရှိပါသည်။', reply_markup={'remove_keyboard': True})


@app.post('/webhook')
async def webhook(request: Request):
    secret = bot._telegram_webhook_secret()
    if secret and not hmac.compare_digest(request.headers.get('x-telegram-bot-api-secret-token', ''), secret):
        raise HTTPException(403)
    update = await request.json()
    now = time.time()
    for key, seen in list(SEEN_UPDATES.items()):
        if now - seen > 3600:
            SEEN_UPDATES.pop(key, None)
    update_id = update.get('update_id')
    if update_id is not None:
        if update_id in SEEN_UPDATES:
            return {'ok': True}
        SEEN_UPDATES[update_id] = now
    asyncio.create_task(process_update(update))
    return {'ok': True}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '10000')), access_log=False)

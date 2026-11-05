import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import bot
import subtitle_app as app
from subtitle_engine import SubtitleCue


@pytest.mark.parametrize('data', [
    [{'start': 1, 'end': 2, 'text': 'မင်္ဂလာပါ'}],
    {'segments': [{'start_time': '00:01.000', 'end_time': '00:02.000', 'burmese': 'မင်္ဂလာပါ'}]},
    {'result': {'subtitles': [{'start': 1, 'end': 2, 'translation': 'မင်္ဂလာပါ'}]}},
])
def test_known_timed_cue_shapes(data):
    assert app.parse_model_cues(data, 60, 10) == [SubtitleCue(61, 62, 'မင်္ဂလာပါ')]


@pytest.mark.parametrize('data', [
    {'result': 'ရိုးရိုးစာသား'}, {'cues': []}, [],
    {'cues': [{'start': 0, 'end': 1, 'text': 'English'}]},
    {'cues': [{'start': '01:60', 'end': '02:01', 'text': 'စာ'}]},
    {'cues': [{'start': 1, 'end': 11, 'text': 'စာ'}]},
])
def test_invalid_or_ambiguous_data_is_not_silently_accepted(data):
    with pytest.raises(ValueError):
        app.parse_model_cues(data, 0, 10)


def test_explicit_silence_is_accepted():
    assert app.parse_model_cues({'cues': [], 'silence': True}, 0, 10) == []


def test_bad_schema_does_not_win_model_race():
    async def run():
        calls = []
        def generate_content(model, contents, config):
            calls.append(model)
            value = {'wrong_key': 'bad'} if model == 'invalid' else {'cues': [{'start': 0, 'end': 1, 'text': 'စာ'}]}
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        with patch.object(bot, 'PREFERRED_MODELS', ['invalid', 'valid']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models), \
             patch.object(bot, 'GEMINI_HEDGE_INITIAL_PARALLEL', 1):
            result = await bot._generate_direct_gemini_json(
                client, [object(), 'prompt'], purpose='test', strict_json=True,
                validator=lambda data: app.parse_model_cues(data, 0, 10))
        assert calls == ['invalid', 'valid']
        assert app.parse_model_cues(result, 0, 10)[0].text == 'စာ'
    asyncio.run(run())


def test_truncated_json_cannot_be_salvaged_as_complete_cues():
    async def run():
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
            text='{"cues": [{"start":0,"end":1,"text":"စာ"}]', candidates=[])))
        with patch.object(bot, 'PREFERRED_MODELS', ['only']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models):
            with pytest.raises(RuntimeError):
                await bot._generate_direct_gemini_json(
                    client, ['prompt'], purpose='test', strict_json=True,
                    validator=lambda data: app.parse_model_cues(data, 0, 10))
    asyncio.run(run())


def test_subtitle_pipeline_retries_wrong_schema_without_losing_chunk(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        calls = []
        def generate_content(model, contents, config):
            calls.append(model)
            value = {'message': 'bad'} if model == 'invalid' else {'data': {'segments': [
                {'start_time': 0.5, 'end_time': 1.5, 'burmese': 'မင်္ဂလာပါ'}]}}
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        def extract(source, target, start, length):
            target.write_bytes(b'audio')
        with patch.object(app.bot, '_make_gemini_client', return_value=client), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', AsyncMock(return_value=object())), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()) as delete, \
             patch.object(bot, 'PREFERRED_MODELS', ['invalid', 'valid']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models), \
             patch.object(bot, 'GEMINI_HEDGE_INITIAL_PARALLEL', 1):
            result = await app.transcribe_translate('source.mp4', 10, str(tmp_path), 123)
        assert result == [SubtitleCue(.5, 1.5, 'မင်္ဂလာပါ')]
        assert calls == ['invalid', 'valid']
        assert delete.await_count == 1
        assert not list(tmp_path.iterdir())
    asyncio.run(run())

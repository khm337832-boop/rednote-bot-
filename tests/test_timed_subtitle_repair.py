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


def test_translated_field_precedes_source_text():
    data = {'cues': [{'start': 0, 'end': 1, 'text': 'Hello', 'burmese': 'မင်္ဂလာပါ'}]}
    assert app.parse_model_cues(data, 0, 10)[0].text == 'မင်္ဂလာပါ'


def test_foreign_cues_repaired_without_changing_times_or_burmese_cues():
    from unittest.mock import AsyncMock
    async def run():
        cues = [SubtitleCue(1, 2, 'Hello'), SubtitleCue(3, 4, 'ရပါတယ်'), SubtitleCue(5, 6, '你好')]
        generator = AsyncMock(return_value={'translations': [{'id': 0, 'text': 'မင်္ဂလာပါ'}, {'id': 2, 'text': 'နေကောင်းလား'}]})
        with patch.object(bot, '_generate_direct_gemini_json', generator):
            result = await app._translate_foreign_cues(object(), cues)
        assert result == [SubtitleCue(1, 2, 'မင်္ဂလာပါ'), cues[1], SubtitleCue(5, 6, 'နေကောင်းလား')]
        assert generator.await_args.kwargs['attempts_per_model'] == 2
    asyncio.run(run())


@pytest.mark.parametrize('payload', [
    {'translations': [{'id': 0, 'text': 'English'}]},
    {'translations': [{'id': 0, 'text': 'စာ'}, {'id': 0, 'text': 'စာ'}]},
    {'translations': [{'id': 1, 'text': 'စာ'}]},
    {'translations': []},
])
def test_translation_mapping_rejects_missing_duplicate_unknown_or_foreign(payload):
    with pytest.raises(ValueError):
        app._subtitle_translation_mapping(payload, [0])


def test_pipeline_recovers_foreign_cues_with_one_available_model(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        generator = AsyncMock(side_effect=[
            {'cues': [{'start': 0.5, 'end': 2, 'text': 'Hello'}, {'start': 2, 'end': 3, 'text': 'ကောင်းပြီ'}]},
            {'translations': [{'id': 0, 'text': 'မင်္ဂလာပါ'}]},
        ])
        def extract(source, target, start, length):
            target.write_bytes(b'audio')
        with patch.object(app.bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', AsyncMock(return_value=object())), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(app.bot, '_generate_direct_gemini_json', generator), \
             patch.object(app.bot, 'send_text', AsyncMock()):
            result = await app.transcribe_translate('source', 10, str(tmp_path), 123)
        assert result == [SubtitleCue(.5, 2, 'မင်္ဂလာပါ'), SubtitleCue(2, 3, 'ကောင်းပြီ')]
        assert generator.await_count == 2
    asyncio.run(run())


def test_single_model_repairs_foreign_cue_after_invalid_mapping(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        calls = []
        def generate_content(model, contents, config):
            calls.append(model)
            if len(calls) == 1:
                data = {'cues': [{'start': 1, 'end': 2, 'text': 'Hello'}]}
            elif len(calls) == 2:
                data = {'translations': [{'id': 0, 'text': 'Hello'}]}
            else:
                data = {'translations': [{'id': 0, 'text': 'မင်္ဂလာပါ'}]}
            return SimpleNamespace(text=json.dumps(data, ensure_ascii=False))
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        def extract(source, target, start, length):
            target.write_bytes(b'audio')
        with patch.object(app.bot, '_make_gemini_client', return_value=client), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', AsyncMock(return_value=object())), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(app.bot, 'send_text', AsyncMock()), \
             patch.object(bot, 'PREFERRED_MODELS', ['only']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models):
            result = await app.transcribe_translate('source', 10, str(tmp_path), 123)
        assert result == [SubtitleCue(1, 2, 'မင်္ဂလာပါ')]
        assert len(calls) == 3
    asyncio.run(run())


def test_camel_case_timing_fields():
    data = {'cues': [{'startTime': '00:01.0', 'endTime': '00:02.0', 'text': 'မင်္ဂလာပါ'}]}
    assert app.parse_model_cues(data, 60, 10) == [SubtitleCue(61, 62, 'မင်္ဂလာပါ')]


def test_timing_repair_preserves_valid_cues_and_original_text():
    source = {'cues': [
        {'start': 0, 'end': 2, 'text': 'မင်္ဂလာပါ'},
        {'start': 'unknown', 'end': None, 'text': 'ဘယ်သွားမလဲ'},
        {'start': 5, 'end': 7, 'text': 'အိမ်ပြန်မယ်'},
    ]}
    items, bad = app._timing_repair_plan(source, 60, 10)
    assert bad == [1]
    repaired = app._apply_timing_repairs(items, bad, {'timings': [{'id': 1, 'start': 2.5, 'end': 4, 'text': 'မဖြစ်ရ'}]}, 60, 10)
    assert repaired == [SubtitleCue(60, 62, 'မင်္ဂလာပါ'), SubtitleCue(62.5, 64, 'ဘယ်သွားမလဲ'), SubtitleCue(65, 67, 'အိမ်ပြန်မယ်')]


@pytest.mark.parametrize('mapping', [
    {'timings': []},
    {'timings': [{'id': 1, 'start': 3, 'end': 20}]},
    {'timings': [{'id': 1, 'start': 2, 'end': 1}]},
    {'timings': [{'id': 1, 'start': 2, 'end': 3}, {'id': 1, 'start': 4, 'end': 5}]},
    {'timings': [{'id': 0, 'start': 2, 'end': 3}]},
])
def test_timing_repair_never_accepts_incomplete_or_wrong_mappings(mapping):
    items = [{'start': 0, 'end': 1, 'text': 'စာ'}, {'start': None, 'end': None, 'text': 'စာ'}]
    with pytest.raises(ValueError):
        app._apply_timing_repairs(items, [1], mapping, 0, 10)


def test_no_timing_repair_without_a_full_candidate_or_text():
    assert app._timing_repair_plan({'cues': []}, 0, 10) is None
    assert app._timing_repair_plan({'cues': [{'start': None, 'end': None, 'text': ''}]}, 0, 10) is None
    assert app._timing_repair_plan({'cues': [{'start': 1, 'end': 2, 'text': 'စာ'}]}, 0, 10) is None


def test_single_model_invalid_timings_then_audio_grounded_repair(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        calls = []
        def generate_content(model, contents, config):
            calls.append((model, contents))
            if len(calls) == 1:
                data = {'cues': [{'start': 0, 'end': 1, 'text': 'ပထမ'}, {'start': 'unknown', 'end': None, 'text': 'ဒုတိယ'}]}
            else:
                data = {'timings': [{'id': 1, 'start': 2, 'end': 3}]}
            return SimpleNamespace(text=json.dumps(data, ensure_ascii=False))
        media = object()
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        def extract(source, target, start, length):
            target.write_bytes(b'audio')
        with patch.object(app.bot, '_make_gemini_client', return_value=client), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', AsyncMock(return_value=media)), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(app.bot, 'send_text', AsyncMock()), \
             patch.object(bot, 'PREFERRED_MODELS', ['only']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models):
            result = await app.transcribe_translate('source', 10, str(tmp_path), 123)
        assert result == [SubtitleCue(0, 1, 'ပထမ'), SubtitleCue(2, 3, 'ဒုတိယ')]
        assert len(calls) >= 2
        assert calls[1][1][0] is media
    asyncio.run(run())


def test_explicit_timestamp_pair_and_rounded_duration_checks():
    data = {'cues': [{'timestamps': [1, 2], 'text': 'စာ'}]}
    assert app.parse_model_cues(data, 0, 10) == [SubtitleCue(1, 2, 'စာ')]
    with pytest.raises(ValueError):
        app.parse_model_cues({'cues': [{'start': 1, 'end': 1.001, 'text': 'စာ'}]}, 0, 10)


def test_failed_timing_repair_is_not_delivered(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        def generate_content(model, contents, config):
            if 'timings' in contents[1]:
                data = {'timings': [{'id': 0, 'start': 20, 'end': 21}]}
            else:
                data = {'cues': [{'start': None, 'end': None, 'text': 'စာ'}]}
            return SimpleNamespace(text=json.dumps(data, ensure_ascii=False))
        media = object()
        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        def extract(source, target, start, length):
            target.write_bytes(b'audio')
        with patch.object(app.bot, '_make_gemini_client', return_value=client), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(app.bot, '_upload_video_to_gemini', AsyncMock(return_value=media)), \
             patch.object(app.bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(app.bot, 'send_text', AsyncMock()), \
             patch.object(bot, 'PREFERRED_MODELS', ['only']), \
             patch.object(bot, '_live_gemini_models', side_effect=lambda models: models):
            with pytest.raises(RuntimeError):
                await app.transcribe_translate('source', 10, str(tmp_path), 123)
        assert app.LAST_TIMING_REPAIR['status'] == 'failed'
        assert app.LAST_TIMING_REPAIR['invalid_cues'] == 1
        assert not list(tmp_path.iterdir())
    asyncio.run(run())


def test_unrepairable_response_splits_audio_and_keeps_both_halves(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        extracted = []
        async def upload(client, path):
            return json.loads(__import__('pathlib').Path(path).read_text())
        def extract(source, path, start, length):
            extracted.append((start, length))
            path.write_text(json.dumps({'start': start, 'length': length}))
        async def generate(client, contents, **kwargs):
            audio = contents[0]
            if audio['length'] == 60:
                # Timing is invalid and text is missing: v1.4's targeted
                # repair cannot form a plan from this candidate.
                data = {'cues': [{'start': None, 'end': None, 'text': ''}]}
            else:
                data = {'cues': [{'start': 1, 'end': 3, 'text': 'စာတန်း'}]}
            try:
                kwargs['validator'](data)
            except ValueError as exc:
                raise RuntimeError('invalid response') from exc
            return data
        with patch.object(bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(bot, '_upload_video_to_gemini', side_effect=upload), \
             patch.object(bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(bot, '_generate_direct_gemini_json', side_effect=generate), \
             patch.object(bot, 'send_text', AsyncMock()):
            result = await app.transcribe_translate('source', 60, str(tmp_path), 123)
        assert extracted == [(0, 60), (0, 30), (30, 30)]
        assert result == [SubtitleCue(1, 3, 'စာတန်း'), SubtitleCue(31, 33, 'စာတန်း')]
        assert app.LAST_TIMING_REPAIR['status'] == 'recovered_by_split'
        assert not list(tmp_path.iterdir())
    asyncio.run(run())


def test_provider_failure_without_invalid_response_does_not_split(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        extracted = []
        def extract(source, path, start, length):
            extracted.append((start, length))
            path.write_bytes(b'audio')
        with patch.object(bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(bot, '_upload_video_to_gemini', AsyncMock(return_value=object())), \
             patch.object(bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(bot, '_generate_direct_gemini_json', AsyncMock(side_effect=RuntimeError('provider unavailable'))):
            with pytest.raises(RuntimeError, match='provider unavailable'):
                await app.transcribe_translate('source', 60, str(tmp_path), 123)
        assert extracted == [(0, 60)]
        assert not list(tmp_path.iterdir())
    asyncio.run(run())


def test_no_partial_output_if_second_split_cannot_be_repaired(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        async def upload(client, path):
            return json.loads(__import__('pathlib').Path(path).read_text())
        def extract(source, path, start, length):
            path.write_text(json.dumps({'start': start, 'length': length}))
        async def generate(client, contents, **kwargs):
            audio = contents[0]
            data = ({'cues': [{'start': None, 'end': None, 'text': ''}]}
                    if audio['length'] == 60 or audio['start'] >= 30 else
                    {'cues': [{'start': 1, 'end': 3, 'text': 'စာတန်း'}]})
            try:
                kwargs['validator'](data)
            except ValueError as exc:
                raise RuntimeError('invalid response') from exc
            return data
        with patch.object(bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(bot, '_upload_video_to_gemini', side_effect=upload), \
             patch.object(bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(bot, '_generate_direct_gemini_json', side_effect=generate), \
             patch.object(bot, 'send_text', AsyncMock()):
            with pytest.raises(RuntimeError):
                await app.transcribe_translate('source', 60, str(tmp_path), 123)
        assert app.LAST_TIMING_REPAIR['status'] == 'failed'
        assert not list(tmp_path.iterdir())
    asyncio.run(run())


def test_failed_targeted_timing_repair_falls_back_to_shorter_audio(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        extracted = []
        def extract(source, path, start, length):
            extracted.append((start, length))
            path.write_text(json.dumps({'start': start, 'length': length}))
        async def upload(client, path):
            return json.loads(__import__('pathlib').Path(path).read_text())
        async def generate(client, contents, **kwargs):
            audio = contents[0]
            if kwargs['purpose'] == 'subtitle timing repair':
                data = {'timings': []}
            elif audio['length'] == 60:
                data = {'cues': [{'start': None, 'end': None, 'text': 'စာတန်း'}]}
            else:
                data = {'cues': [{'start': 1, 'end': 3, 'text': 'စာတန်း'}]}
            try:
                kwargs['validator'](data)
            except ValueError as exc:
                raise RuntimeError('invalid response') from exc
            return data
        with patch.object(bot, '_make_gemini_client', return_value=object()), \
             patch.object(app, 'extract_audio', side_effect=extract), \
             patch.object(bot, '_upload_video_to_gemini', side_effect=upload), \
             patch.object(bot, '_delete_gemini_file', AsyncMock()), \
             patch.object(bot, '_generate_direct_gemini_json', side_effect=generate), \
             patch.object(bot, 'send_text', AsyncMock()):
            result = await app.transcribe_translate('source', 60, str(tmp_path), 123)
        assert extracted == [(0, 60), (0, 30), (30, 30)]
        assert result == [SubtitleCue(1, 3, 'စာတန်း'), SubtitleCue(31, 33, 'စာတန်း')]
        assert app.LAST_TIMING_REPAIR['status'] == 'recovered_by_split'
    asyncio.run(run())

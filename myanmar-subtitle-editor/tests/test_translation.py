import pytest
from backend.translation import translate_segments, _validate_batch


def test_translation_preserves_capcut_timing_and_order():
    segments = [
        {'start': 0.0, 'end': 2.5, 'speech_end': 1.5, 'text': 'hello'},
        {'start': 2.5, 'end': 5.0, 'speech_end': 3.5, 'text': 'today'},
        {'start': 5.0, 'end': 10.0, 'speech_end': 7.0, 'text': 'good weather'},
    ]
    def translate(texts, context):
        assert texts == ['hello', 'today', 'good weather']
        return ['မင်္ဂလာပါ', 'ဒီနေ့', 'ရာသီဥတု ကောင်းတယ်']
    result = translate_segments(segments, 'en', translate=translate)
    assert [(s['start'], s['end'], s['speech_end']) for s in result] == [(0, 2.5, 1.5), (2.5, 5, 3.5), (5, 10, 7)]
    assert [s['text'] for s in result] == ['မင်္ဂလာပါ', 'ဒီနေ့', 'ရာသီဥတု ကောင်းတယ်']
    assert segments[0]['text'] == 'hello'


def test_myanmar_speech_does_not_need_external_translation():
    segments = [{'start': 0, 'end': 2, 'text': 'မင်္ဂလာပါ'}]
    assert translate_segments(segments, 'my', translate=lambda *_: pytest.fail('unexpected model call')) == segments


@pytest.mark.parametrize('payload', [
    {'translations': [{'id': 0, 'text': 'စာ'}, {'id': 0, 'text': 'စာ'}]},
    {'translations': [{'id': 0, 'text': 'hello'}]},
    {'translations': []},
    {'translations': [{'id': True, 'text': 'စာ'}]},
])
def test_invalid_or_incomplete_translation_fails(payload):
    with pytest.raises(ValueError):
        _validate_batch(payload, 1)


def test_missing_gemini_key_fails_honestly(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(RuntimeError):
        translate_segments([{'start':0, 'end':1, 'text':'hello'}], 'en')

"""CapCut စတိုင် စာတန်းအချိန် (timing) ယုတ္တိ စမ်းသပ်မှုများ.

အရေးအကြီးဆုံး စည်းကမ်း: ``segment[i].end == segment[i+1].start`` — စာတန်း
တစ်ခုသည် နောက်စကား မစပြောမချင်း (silence ရှိရင်တောင်) ဆက်ပေါ်နေရမည်။
"""
from __future__ import annotations

import unicodedata

import pytest

from backend import myanmar_text as mt
from backend.subtitle_generator import (
    CaptionSegment,
    Word,
    build_segments,
    clean_words,
    demo_transcribe,
    enforce_hold_timing,
    group_words,
    Part,
    split_phrase,
    validate_segments,
)

# အစား ("leading") အက္ခရာများ — segment တစ်ခုသည် ဒါတွေနှင့် အစမခံရ၊
# အဆုံးသတ်လျှင်လည်း syllable ကို ဖြတ်ထားသကဲ့သို့ ဖြစ်သည်။
LEADING_MARKS = set("ေျြွှ္")


def _example_words() -> list[Word]:
    return [
        Word(0.0, 1.5, "မင်္ဂလာပါ"),
        Word(2.5, 3.5, "ဒီနေ့"),
        Word(5.0, 7.0, "ရာသီဥတု ကောင်းတယ်"),
    ]


def test_capcut_example_timing():
    """သတ်မှတ်ချက် ဥပမာ — 0-1.5, 2.5-3.5, 5.0-7.0 → 0-2.5, 2.5-5.0, 5.0-8.0."""
    segments = build_segments(_example_words(), duration=8.0)
    assert [round(s.start, 3) for s in segments] == [0.0, 2.5, 5.0]
    assert [round(s.end, 3) for s in segments] == [2.5, 5.0, 8.0]
    assert [s.text for s in segments] == ["မင်္ဂလာပါ", "ဒီနေ့", "ရာသီဥတု ကောင်းတယ်"]


def test_each_caption_ends_when_next_starts():
    segments = build_segments(_example_words(), duration=12.0)
    for current, following in zip(segments, segments[1:]):
        assert current.end == pytest.approx(following.start, abs=1e-3)


def test_captions_do_not_vanish_during_silence():
    """1.0s နှင့် 1.5s ကြားတွင် စကားမရှိသော်လည်း ပထမ စာတန်း ဆက်ပေါ်နေရမည်."""
    segments = build_segments(_example_words(), duration=8.0)
    assert segments[0].start <= 0.0
    assert segments[0].end >= 2.5
    # 1.8s (silence) သည် ပထမ segment အတွင်း ရှိသည်
    assert segments[0].start <= 1.8 <= segments[0].end


def test_first_caption_does_not_show_before_speech():
    words = [Word(4.0, 5.0, "နောက်မှ စပြောသည်")]
    segments = build_segments(words, duration=10.0, max_hold_seconds=3.0)
    assert segments[0].start == pytest.approx(4.0, abs=1e-3)


def test_last_caption_is_capped_at_max_hold():
    """နောက်ဆုံး စာတန်းသည် စကားပြီးချိန်မှ ၃ စက္ကန့်ထက် မကျော်ရ (မူလ config)."""
    segments = build_segments(_example_words(), duration=30.0, max_hold_seconds=3.0)
    assert segments[-1].end == pytest.approx(10.0, abs=1e-3)


def test_last_caption_can_hold_until_video_end():
    segments = build_segments(
        _example_words(), duration=30.0, max_hold_seconds=3.0, last_hold_mode="video_end"
    )
    assert segments[-1].end == pytest.approx(30.0, abs=1e-3)


def test_last_caption_never_exceeds_video_end():
    segments = build_segments(_example_words(), duration=8.0, max_hold_seconds=10.0)
    assert segments[-1].end == pytest.approx(8.0, abs=1e-3)


def test_configured_gap_threshold_starts_new_segment():
    words = [Word(0.0, 1.0, "ပထမ"), Word(1.6, 2.0, "ဒုတိယ")]
    two = build_segments(words, duration=6.0, gap_threshold=0.5)
    assert len(two) == 2
    one = build_segments(words, duration=6.0, gap_threshold=1.0)
    assert len(one) == 1


def test_punctuation_starts_new_segment_even_without_gap():
    words = [Word(0.0, 1.0, "ပထမ။"), Word(1.05, 2.0, "ဒုတိယ")]
    segments = build_segments(words, duration=6.0, gap_threshold=0.5, punctuation_break=True)
    assert len(segments) == 2
    merged = build_segments(words, duration=6.0, gap_threshold=0.5, punctuation_break=False)
    assert len(merged) == 1


def test_whisper_sentence_boundary_forces_break():
    words = [Word(0.0, 0.9, "စာသား"), Word(1.0, 1.8, "ဆက်")]
    assert len(build_segments(words, duration=5.0, gap_threshold=0.5)) == 1
    assert len(build_segments(words, duration=5.0, gap_threshold=0.5, sentence_ends=[0.95])) == 2


def test_tiny_gaps_are_merged_so_every_caption_is_visible():
    words = [Word(0.0, 1.0, "တစ်"), Word(1.1, 1.4, "နှစ်")]
    segments = build_segments(words, duration=5.0, min_duration=0.3)
    assert len(segments) == 1
    assert segments[0].end - segments[0].start >= 0.3


def test_hold_timing_merges_parts_closer_than_min_duration():
    parts = [
        Part(0.0, 0.5, "ပထမ", 0.5),
        Part(0.1, 0.6, "ဒုတိယ", 0.6),
        Part(2.0, 2.5, "တတိယ", 2.5),
    ]
    segments = enforce_hold_timing(
        parts, duration=6.0, max_hold_seconds=3.0, min_duration=0.3
    )
    assert len(segments) == 2
    assert segments[0].end == pytest.approx(segments[1].start)
    assert "ပထမ" in segments[0].text and "ဒုတိယ" in segments[0].text


def test_long_phrase_splits_without_cutting_syllables():
    phrase = (
        "မြန်မာစာတန်းထိုး ဖန်တီးနည်းကို အဆင့်ဆင့် ရှင်းပြပါမယ်။ "
        "ပထမအဆင့်က ဗီဒီယို တင်ပါ။ ဒုတိယအဆင့်က စာတန်း ဖန်တီးပါ။"
    )
    words = [
        Word(0.0, 4.0, phrase),
    ]
    segments = build_segments(words, duration=20.0, max_chars=20, max_duration=3.0)
    assert len(segments) > 1
    for segment in segments:
        assert segment.text.strip()
        assert len(segment.text) > 0
        assert segment.text[0] not in LEADING_MARKS
        assert unicodedata.category(segment.text[0]) not in {"Mn", "Mc"}
        bare = segment.text.replace(" ", "")
        assert "".join(mt.syllables(bare)) == bare
        assert segment.end > segment.start
    for current, following in zip(segments, segments[1:]):
        assert current.end == pytest.approx(following.start, abs=1e-3)
    # စာသား အားလုံး မပျောက်ရ (space များ ချန်လျှင်)
    rebuilt = "".join(segment.text for segment in segments).replace(" ", "")
    assert rebuilt == phrase.replace(" ", "")


def test_long_duration_splits_even_short_text():
    words = [Word(0.0, 9.0, "ရှည်လျားသော စကားတစ်ခွန်း")]
    segments = build_segments(words, duration=20.0, max_duration=4.0, max_chars=100)
    assert len(segments) >= 2
    # နောက်ဆုံး စာတန်းသည် စကားပြီးချိန်အထိ ဆက်ပေါ်သဖြင့် ကြာနိုင်သည် (CapCut rule)
    assert all(segment.end - segment.start <= 4.5 for segment in segments[:-1])
    assert segments[-1].end <= 12.5


def test_split_phrase_respects_char_cap():
    words = [Word(i * 0.2, (i + 1) * 0.2, "က") for i in range(30)]
    parts = split_phrase(words, max_chars=10, max_duration=30.0)
    assert len(parts) >= 3
    for part in parts:
        assert len(part.text.replace(" ", "")) <= 10


def test_clean_words_drops_empty_and_fixes_order():
    words = [
        Word(1.0, 1.5, "  "),
        Word(0.5, 0.9, "ပထမ"),
        Word(0.4, 0.6, "ပထမ"),
    ]
    cleaned = clean_words(words)
    assert [w.text for w in cleaned] == ["ပထမ", "ပထမ"]
    assert all(w.end >= w.start for w in cleaned)


def test_group_words_returns_phrases():
    groups = group_words(_example_words(), gap_threshold=0.5)
    assert [len(group) for group in groups] == [1, 1, 1]


def test_empty_input_returns_no_segments():
    assert build_segments([], duration=10.0) == []


def test_validate_segments_detects_overlap_and_bad_times():
    problems = validate_segments(
        [
            {"start": 0.0, "end": 2.0, "text": "က"},
            {"start": 1.5, "end": 3.0, "text": "ခ"},
            {"start": 4.0, "end": 4.0, "text": "ဂ"},
        ],
        duration=5.0,
    )
    assert any("ထပ်" in problem for problem in problems)
    assert any("end <= start" in problem for problem in problems)


def test_demo_transcribe_matches_capcut_rules():
    result = demo_transcribe(22.0)
    segments = result["segments"]
    assert len(segments) >= 3
    for current, following in zip(segments, segments[1:]):
        assert current["end"] == pytest.approx(following["start"], abs=1e-3)
    assert segments[0]["start"] == pytest.approx(0.0, abs=1e-3)
    assert segments[-1]["end"] <= 22.0
    assert validate_segments(segments, duration=22.0) == []


def test_segments_serialise_round_trip():
    segments = build_segments(_example_words(), duration=8.0)
    payload = [segment.to_dict() for segment in segments]
    assert all({"start", "end", "text"} <= set(item) for item in payload)
    restored = [CaptionSegment(**item) for item in payload]
    assert [s.text for s in restored] == [s.text for s in segments]


def test_tokens_with_spaces_stay_separated_in_captions():
    """Whisper token spacing ကို မဖျက်ရ — ``Blurနဲ့Logo`` ဖြစ်မနေရ။"""
    words = [
        Word(0.0, 1.0, "Blur "),
        Word(1.0, 2.0, "နဲ့ "),
        Word(2.0, 3.0, "Logo "),
        Word(3.0, 4.0, "ကို"),
    ]
    segments = build_segments(words, duration=6.0)
    assert [segment.text for segment in segments] == ["Blur နဲ့ Logo ကို"]


def test_demo_captions_keep_word_spaces():
    segments = demo_transcribe(22.0)["segments"]
    texts = [segment["text"] for segment in segments]
    assert "ရာသီဥတု ကောင်းတယ်" in texts
    assert any("Blur နဲ့ Logo ကို" in text for text in texts)

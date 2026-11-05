"""မြန်မာစာ syllable ခွဲခြင်း၊ NFC normalize နှင့် line wrapping စမ်းသပ်မှုများ."""
from __future__ import annotations

import unicodedata

import pytest

from backend import myanmar_text as mt

COMBINING = {"Mn", "Mc"}
# syllable တစ်ခု မစနိုင်သော / မဆုံးနိုင်သော အက္ခရာများ
LEADING_MARKS = set("ေျြွှ္")


def test_syllables_join_finals_with_previous():
    assert mt.syllables("မင်္ဂလာပါ") == ["မင်္ဂ", "လာ", "ပါ"]
    assert mt.syllables("ဖြစ်တယ်") == ["ဖြစ်", "တယ်"]
    assert mt.syllables("ကျွန်း") == ["ကျွန်း"]
    assert mt.syllables("ရာသီဥတု") == ["ရာ", "သီ", "ဥ", "တု"]


def test_syllables_never_start_with_a_combining_mark():
    for sentence in (
        "မြန်မာစာတန်းထိုး ဖန်တီးနည်း",
        "အင်္ဂလိပ်စာ သင်ယူခြင်း",
        "ဒီနေ့ ရာသီဥတု ကောင်းတယ်",
    ):
        for syllable in mt.syllables(sentence):
            if not syllable.strip():
                continue
            assert unicodedata.category(syllable[0]) not in COMBINING


def test_count_graphemes_uses_syllables_not_codepoints():
    assert mt.count_graphemes("မင်္ဂလာပါ") == 3
    assert mt.count_graphemes("မင်္ဂလာပါ ခင်ဗျာ") == 5
    assert mt.count_graphemes("") == 0


def test_normalize_text_nfc_and_whitespace():
    decomposed = "ကေ" + "\u102c"  # ေ + ာ
    normalized = mt.normalize_text("  " + unicodedata.normalize("NFD", "ကော") + "  \u200b")
    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == "ကော"
    assert mt.normalize_text("က\t\t ခ\n\nဂ") == "က ခ\nဂ"
    assert decomposed  # sanity: source string is non-empty


def test_zawgyi_detection():
    assert mt.looks_like_zawgyi("ေကာင်း") is True
    assert mt.looks_like_zawgyi("ကောင်း") is False


def test_wrap_text_never_breaks_inside_a_syllable():
    text = "မြန်မာစာတန်းထိုး ဖန်တီးနည်းကို အဆင့်ဆင့် ရှင်းပြပါမယ်"
    lines = mt.wrap_text(text, 8, lambda value: float(len(value)))
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= 8
        assert unicodedata.category(line[0]) not in COMBINING
        # စာကြောင်းတိုင်းသည် ပြည့်စုံသော syllable များသာ ဖြစ်ရမည်
        bare = line.replace(" ", "")
        assert "".join(mt.syllables(bare)) == bare
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_wrap_text_preserves_explicit_newlines():
    lines = mt.wrap_text("ပထမကြောင်း\nဒုတိယကြောင်း", 50, len)
    assert lines == ["ပထမကြောင်း", "ဒုတိယကြောင်း"]


def test_wrap_text_rejects_non_positive_width():
    with pytest.raises(ValueError):
        mt.wrap_text("စာ", 0, len)


def test_split_at_syllable_returns_two_parts():
    text = "မြန်မာစာတန်းထိုး ဖန်တီးနည်း"
    left, right = mt.split_at_syllable(text, 0.5)
    assert left and right
    assert " ".join((left + " " + right).split()) == text
    assert unicodedata.category(right[0]) not in COMBINING
    # space ရှိသော နေရာကို ဦးစားပေးသင့်သည်
    assert left == "မြန်မာစာတန်းထိုး"


def test_join_tokens_does_not_insert_spaces_in_myanmar_text():
    assert mt.join_tokens("မင်္ဂလာ", "ပါ") == "မင်္ဂလာပါ"
    assert mt.join_tokens("hello", "world") == "hello world"
    assert mt.join_tokens("မင်္ဂလာပါ ", "ခင်ဗျာ") == "မင်္ဂလာပါ ခင်ဗျာ"


def test_join_tokens_keeps_space_between_latin_and_myanmar():
    # မြန်မာစကားလုံးနှင့် လက်တင်စကားလုံး ကပ်မနေရ
    assert mt.join_tokens("Blur", "နဲ့") == "Blur နဲ့"
    assert mt.join_tokens("နဲ့", "Logo") == "နဲ့ Logo"
    assert mt.join_tokens("ကို", "စိတ်ကြိုက်") == "ကိုစိတ်ကြိုက်"


def test_is_sentence_end():
    assert mt.is_sentence_end("ပြီးပါပြီ။")
    assert mt.is_sentence_end("done.")
    assert not mt.is_sentence_end("ဆက်")


def test_layout_tokens_kinds():
    kinds = [token.kind for token in mt.layout_tokens("မင်္ဂလာပါ VIP ။")]
    assert kinds == ["my", "my", "my", "space", "word", "space", "punct"]


def test_wrap_by_char_budget():
    lines = mt.wrap_by_char_budget("မင်္ဂလာပါ ခင်ဗျာ", 6)
    assert len(lines) >= 2
    assert all(len(line) <= 6 for line in lines)

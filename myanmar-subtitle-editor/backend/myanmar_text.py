"""မြန်မာစာ ကိုင်တွယ်မှု — Unicode normalize, syllable ခွဲခြင်း, line wrap.

မြန်မာစာတွင် စကားလုံးကြား space မရှိပါ။ ထို့ကြောင့် token/word အရေအတွက်ဖြင့်
ခွဲခြင်းသည် မှားတတ်သည်။ ဤ module သည် orthographic cluster (ဗျည်း + ဗျည်းတွဲ +
သရ + အသံ) များကို ပေါင်းပြီး ပြည့်စုံသော " syllable " တစ်ခု ဖန်တီးကာ
ထို syllable နယ်နိမိတ်တွင်သာ စာကြောင်း/segment ခွဲနိုင်စေသည်။
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable

# Unicode Myanmar block building blocks -------------------------------------
VIRAMA = "\u1039"  # ္  (stacked consonant)
ASAT = "\u103A"  # ်  (killed consonant / final)
TONE_MARKS = frozenset({"\u1037", "\u1038"})  # ့ း
FINAL_MARKS = frozenset({ASAT, VIRAMA}) | TONE_MARKS
DIGIT_ZERO, DIGIT_NINE = "\u1040", "\u1049"

MYANMAR_PUNCTUATION = frozenset({"\u104A", "\u104B"})  # ၊ ။
LATIN_PUNCTUATION = frozenset(".,!?;:%…—–")
BREAK_PUNCTUATION = MYANMAR_PUNCTUATION | LATIN_PUNCTUATION
SENTENCE_END = frozenset({"\u104B", ".", "!", "?", "…"})  # ။ ។ . ! ?

_ZW_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
_MULTI_SPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_ZAWGYI_HINT = re.compile("[\u1031][\u1000-\u1021]")

_MYANMAR_RANGES = (
    (0x1000, 0x109F),
    (0xA9E0, 0xA9FF),
    (0xAA60, 0xAA7F),
)


def is_myanmar_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return any(low <= code <= high for low, high in _MYANMAR_RANGES)


def is_myanmar_letter(ch: str) -> bool:
    """Base letters (consonants, independent vowels) — not marks."""
    if not is_myanmar_char(ch):
        return False
    if ch in (VIRAMA, ASAT) or ch in TONE_MARKS:
        return False
    return unicodedata.category(ch) not in {"Mn", "Mc", "Nd"} and ch not in MYANMAR_PUNCTUATION


def _is_mark(ch: str) -> bool:
    """True when `ch` must stay attached to the previous cluster."""
    if ch == VIRAMA:
        return True
    if is_myanmar_char(ch):
        # Mn = combining mark, Mc = spacing combining mark (ျ ွ ာ း ...)
        return unicodedata.category(ch) in {"Mn", "Mc"}
    # Generic combining marks (e.g. variation selectors) also stay attached.
    return unicodedata.category(ch) in {"Mn", "Mc", "Me"}


def normalize_text(text: str) -> str:
    """NFC normalize, strip zero-width junk, collapse spaces, tidy punctuation.

    Zawgyi is *not* converted (the app is Unicode only); the caller is warned
    separately via :func:`looks_like_zawgyi`.
    """
    if not text:
        return ""
    text = text.translate(_ZW_CHARS)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def looks_like_zawgyi(text: str) -> bool:
    """Cheap heuristic: in Unicode the vowel sign ေ (U+1031) follows its consonant."""
    if not text:
        return False
    return bool(_ZAWGYI_HINT.search(text))


def split_clusters(text: str) -> list[str]:
    """Split into orthographic clusters (base letter + attached marks).

    A virama (္) joins the *following* consonant into the same cluster, which
    is how stacked consonants (က္က) and kinzi (င်္က) are handled.
    """
    clusters: list[str] = []
    current = ""
    join_next = False
    for ch in text:
        if not current:
            current = ch
            join_next = ch == VIRAMA
            continue
        if join_next or _is_mark(ch):
            current += ch
            join_next = ch == VIRAMA
            continue
        clusters.append(current)
        current = ch
        join_next = ch == VIRAMA
    if current:
        clusters.append(current)
    return clusters


def _is_final_cluster(cluster: str) -> bool:
    """A bare final such as န် / င် / စ် / က္က belongs to the previous syllable."""
    if not cluster:
        return False
    if VIRAMA in cluster:
        return True
    if not is_myanmar_letter(cluster[0]):
        return False
    return all(ch in FINAL_MARKS for ch in cluster[1:]) and len(cluster) > 1


def syllables(text: str) -> list[str]:
    """Return syllables (non-breaking units) of a Myanmar run."""
    out: list[str] = []
    for cluster in split_clusters(text):
        if out and _is_final_cluster(cluster):
            out[-1] += cluster
        else:
            out.append(cluster)
    return out


def count_graphemes(text: str) -> int:
    """Count visible units — Myanmar syllables, other scripts per character."""
    total = 0
    for token in layout_tokens(text):
        if token.kind in {"my", "word"}:
            total += 1 if token.kind == "word" else 1
        elif token.kind == "punct":
            total += 1
    return total


class Token:
    __slots__ = ("text", "kind")  # kind: my | word | space | punct

    def __init__(self, text: str, kind: str) -> None:
        self.text = text
        self.kind = kind

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Token({self.text!r}, {self.kind})"


def layout_tokens(text: str) -> list[Token]:
    """Split into layout tokens: Myanmar syllables, words, spaces, punctuation."""
    tokens: list[Token] = []
    buf = ""
    mode = ""

    def flush() -> None:
        nonlocal buf, mode
        if not buf:
            return
        if mode == "my":
            for syl in syllables(buf):
                tokens.append(Token(syl, "my"))
        elif mode == "space":
            tokens.append(Token(buf, "space"))
        elif mode == "punct":
            tokens.append(Token(buf, "punct"))
        else:
            tokens.append(Token(buf, "word"))
        buf = ""
        mode = ""

    for ch in text:
        if ch in "\n":
            flush()
            tokens.append(Token("\n", "space"))
            continue
        if ch.isspace():
            new_mode = "space"
        elif ch in BREAK_PUNCTUATION:
            new_mode = "punct"
        elif is_myanmar_char(ch):
            new_mode = "my"
        else:
            new_mode = "word"
        if new_mode != mode:
            flush()
            mode = new_mode
        buf += ch
    flush()
    return tokens


def _layout_units(text: str) -> list[str]:
    """Breakable units for line wrapping (punctuation sticks to the previous unit)."""
    units: list[str] = []
    for token in layout_tokens(text):
        if token.kind == "space":
            if token.text == "\n":
                units.append("\n")
                continue
            if units and units[-1] != "\n":
                units[-1] = units[-1] + " "
            continue
        if token.kind == "punct":
            if units and units[-1] != "\n":
                units[-1] = units[-1] + token.text
            else:
                units.append(token.text)
            continue
        units.append(token.text)
    return [u for u in units if u]


def wrap_text(
    text: str,
    max_width: float,
    measure: Callable[[str], float],
    *,
    max_lines: int = 0,
) -> list[str]:
    """Greedy, syllable-aware line wrapping.

    ``measure`` returns the rendered width of a string in the same unit as
    ``max_width``. Explicit newlines in the caption are preserved.
    """
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    lines: list[str] = []
    for paragraph in text.split("\n"):
        units = _layout_units(paragraph)
        if not units:
            lines.append("")
            continue
        current = ""
        for unit in units:
            candidate = (current + unit) if current else unit
            if current and measure(candidate.rstrip()) > max_width:
                lines.append(current.rstrip())
                current = unit.lstrip()
            else:
                current = candidate
        if current.strip() or not lines:
            lines.append(current.rstrip())
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if max_lines and len(lines) > max_lines:
        head = lines[: max_lines - 1]
        head.append(" ".join(lines[max_lines - 1 :]))
        return head
    return lines


def wrap_by_char_budget(
    text: str, chars_per_line: int, *, max_lines: int = 0
) -> list[str]:
    """Fallback wrapping when no font metrics are available."""
    return wrap_text(text, max(1, chars_per_line), lambda s: float(len(s)), max_lines=max_lines)


def split_at_syllable(text: str, fraction: float = 0.5) -> tuple[str, str]:
    """Split text near ``fraction`` of its length, never inside a syllable."""
    text = text.strip()
    if not text:
        return "", ""
    units: list[str] = []
    for token in layout_tokens(text):
        if token.kind == "space":
            if units:
                units[-1] += " "
            continue
        if token.kind == "punct" and units:
            units[-1] += token.text
            continue
        units.append(token.text)
    if len(units) <= 1:
        return text, ""

    total = sum(len(u) for u in units)
    target = max(1.0, total * min(max(fraction, 0.05), 0.95))
    best_index = 1
    best_score = float("inf")
    consumed = 0
    for index in range(1, len(units)):
        consumed += len(units[index - 1])
        score = abs(consumed - target)
        # Prefer a natural break: after a space first, then after punctuation.
        if units[index - 1].endswith(" "):
            score -= total * 0.08
        elif units[index - 1].rstrip().endswith(tuple(BREAK_PUNCTUATION)):
            score -= total * 0.05
        if score < best_score:
            best_score = score
            best_index = index
    left = "".join(units[:best_index]).strip()
    right = "".join(units[best_index:]).strip()
    return left, right


def join_tokens(previous: str, nxt: str) -> str:
    """Join two Whisper tokens without inserting stray spaces in Myanmar text."""
    if not previous:
        return nxt
    if not nxt:
        return previous
    prev_text = previous.rstrip()
    next_text = nxt.lstrip()
    if not next_text:
        return prev_text
    if previous != prev_text or nxt != next_text:
        return prev_text + " " + next_text
    last, first = prev_text[-1], next_text[0]
    # Myanmar syllables inside one word are glued; a Latin word next to a
    # Myanmar word keeps its space ("Blur နဲ့ Logo ကို").
    if is_myanmar_char(last) and (is_myanmar_char(first) or first in BREAK_PUNCTUATION):
        return prev_text + next_text
    if last in BREAK_PUNCTUATION and is_myanmar_char(first):
        return prev_text + next_text
    return prev_text + " " + next_text


def text_height_estimate(lines: Iterable[str], font_size: float) -> float:
    return len(list(lines)) * font_size * 1.45


def is_sentence_end(text: str) -> bool:
    text = text.rstrip()
    return bool(text) and text[-1] in SENTENCE_END


def clean_caption(text: str) -> str:
    """Normalise a caption while keeping intentional line breaks."""
    text = normalize_text(text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)

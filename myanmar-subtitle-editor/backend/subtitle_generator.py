"""faster-whisper / openai-whisper မှ စာတန်းထိုး segment များ ဖန်တီးခြင်း.

အဓိကသဘောတရား (CapCut စတိုင်)
-----------------------------
* စကားပြောတဲ့ တစ်ကြောင်းချင်းစီကို သီးသန့် segment အဖြစ် ခွဲသည်။
* segment တစ်ခုသည် နောက်စကား မစပြောမချင်း ဆက်ပေါ်နေသည်။
  ထို့ကြောင့် ``segment[i].end == segment[i+1].start`` ဖြစ်သည်။
* နောက်ဆုံး segment သည် ဗီဒီယိုအဆုံးအထိ သို့မဟုတ် စကားပြီးချိန်မှ
  ``max_hold_seconds`` (မူလ ၃ စက္ကန့်) အထိ ဆက်ပေါ်နိုင်သည်။

Whisper ၏ word timestamp သည် မြန်မာစာအတွက် token (စာလုံးအပိုင်းအစ) အလိုက်
ထွက်တတ်သဖြင့် "စကားလုံးအရေအတွက်" ကို မသုံးဘဲ token ကြား gap နှင့်
syllable နယ်နိမိတ်ကို အခြေခံသည်။
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from . import myanmar_text as mt

ProgressFn = Callable[[float | None, str], None]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Word:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class CaptionSegment:
    start: float
    end: float
    text: str
    speech_end: float = 0.0  # when the speech for this caption actually stops
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speech_end": round(self.speech_end or self.end, 3),
        }


@dataclass
class Part:
    """Intermediate caption piece before hold timing is enforced."""

    start: float
    end: float
    text: str
    speech_end: float


# ---------------------------------------------------------------------------
# Word cleaning
# ---------------------------------------------------------------------------


def clean_words(words: Iterable[Word]) -> list[Word]:
    """Normalise text/timestamps and drop empty or broken tokens."""
    cleaned: list[Word] = []
    for word in words:
        raw = word.text or ""
        text = mt.clean_caption(raw)
        if not text:
            continue
        # Keep the original spacing at the token edges: Whisper uses a leading
        # space to mark a new word, and dropping it glues separate Myanmar and
        # Latin words together ("Blurနဲ့Logo" instead of "Blur နဲ့ Logo").
        if raw[:1].isspace():
            text = " " + text
        if raw[-1:].isspace():
            text = text + " "
        try:
            start = float(word.start)
            end = float(word.end)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start = max(0.0, start)
        end = max(start, end)
        cleaned.append(Word(start=start, end=end, text=text))
    cleaned.sort(key=lambda w: (w.start, w.end))
    # Remove duplicated tokens produced by overlapping decoding windows.
    deduped: list[Word] = []
    for word in cleaned:
        if deduped:
            prev = deduped[-1]
            if word.text == prev.text and abs(word.start - prev.start) < 0.02:
                continue
            if word.end <= prev.start:
                continue
        deduped.append(word)
    return deduped


def _join_word_list(words: Sequence[Word]) -> tuple[str, list[tuple[int, int, Word]]]:
    """Join tokens into one string and return character spans per token."""
    text = ""
    spans: list[tuple[int, int, Word]] = []
    for word in words:
        stripped = word.text.strip()
        if not stripped:
            continue
        if not text:
            combined = stripped
        else:
            combined = mt.join_tokens(text, word.text)
        # ``join_tokens`` keeps the token at the end of the string (its own
        # spacing may follow it), so this offset points at the token itself.
        probe = combined.rstrip()
        if probe.endswith(stripped):
            offset = len(probe) - len(stripped)
        else:  # defensive: never lose a token because of odd spacing
            combined = probe + stripped
            offset = len(probe)
        text = combined
        spans.append((offset, offset + len(stripped), word))
    return text, spans


# ---------------------------------------------------------------------------
# Grouping tokens into spoken phrases
# ---------------------------------------------------------------------------


def group_words(
    words: Sequence[Word],
    *,
    gap_threshold: float = 0.5,
    punctuation_break: bool = True,
    sentence_ends: Iterable[float] = (),
) -> list[list[Word]]:
    """Group tokens into spoken phrases.

    A new phrase starts when the silence between tokens exceeds
    ``gap_threshold``, after a sentence-final punctuation mark, or at a
    Whisper sentence boundary.
    """
    if not words:
        return []
    boundaries = sorted({round(float(t), 3) for t in sentence_ends if t is not None})
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            previous = current[-1]
            gap = word.start - previous.end
            ends_sentence = punctuation_break and mt.is_sentence_end(previous.text)
            crossed = any(previous.end - 0.02 <= b <= word.start + 0.02 for b in boundaries)
            if gap > gap_threshold or ends_sentence or crossed:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


# ---------------------------------------------------------------------------
# Splitting long phrases without cutting a Myanmar syllable
# ---------------------------------------------------------------------------


def _token_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) offsets for every layout token."""
    spans: list[tuple[int, int, str]] = []
    position = 0
    for token in mt.layout_tokens(text):
        spans.append((position, position + len(token.text), token.kind))
        position += len(token.text)
    return spans


def _expand_long_words(
    words: Sequence[Word], *, max_chars: int, max_duration: float
) -> list[Word]:
    """Split over-long tokens at syllable boundaries with proportional timing.

    Whisper normally returns many short tokens, but a single long token (or a
    caption pasted by the user) must still respect the 35-character / 6-second
    rules — and must never be cut inside a syllable.
    """
    expanded: list[Word] = []
    for word in words:
        duration = max(0.0, word.end - word.start)
        units = [token.text for token in mt.layout_tokens(word.text) if token.kind != "space"]
        if not units:
            expanded.append(word)
            continue
        parts = max(
            1,
            math.ceil(len(units) / max(1, max_chars)),
            math.ceil(duration / max_duration) if max_duration > 0 else 1,
        )
        if parts <= 1 or len(units) < 2:
            expanded.append(word)
            continue
        chunks: list[str] = []
        for index in range(parts):
            start_index = round(index * len(units) / parts)
            end_index = round((index + 1) * len(units) / parts)
            chunk = "".join(units[start_index:end_index])
            if chunk:
                chunks.append(chunk)
        if len(chunks) <= 1:
            expanded.append(word)
            continue
        if word.text[-1:].isspace():
            chunks[-1] = chunks[-1] + " "
        total_chars = sum(len(chunk) for chunk in chunks) or 1
        cursor = word.start
        for chunk in chunks:
            span = duration * len(chunk) / total_chars
            chunk_end = min(word.end, cursor + span)
            expanded.append(Word(round(cursor, 3), round(max(cursor, chunk_end), 3), chunk))
            cursor = chunk_end
    return expanded


def _time_at_offset(offset: int, spans: Sequence[tuple[int, int, Word]], fallback: float) -> float:
    """Start time of the word covering ``offset`` (nearest word when between)."""
    for start, end, word in spans:
        if offset < end:
            return word.start if offset > start else word.start
    if spans:
        return spans[-1][2].end
    return fallback


def split_phrase(
    words: Sequence[Word],
    *,
    max_chars: int = 35,
    max_duration: float = 6.0,
    min_duration: float = 0.3,
) -> list[Part]:
    """Split one spoken phrase into caption parts respecting char/duration caps.

    Split points always fall on a syllable boundary; the timing of each part
    comes from the word timestamps.
    """
    if not words:
        return []
    words = _expand_long_words(list(words), max_chars=max_chars, max_duration=max_duration)
    text, word_spans = _join_word_list(words)
    if not text:
        return []

    tokens = _token_spans(text)
    if not tokens:
        return [Part(words[0].start, words[-1].end, text, words[-1].end)]

    # Breakable offsets: start of each text token (a syllable or a whole word).
    break_offsets = {
        start for start, end, kind in tokens if kind in {"my", "word"} and end > start
    }
    break_offsets.add(0)

    parts: list[Part] = []
    part_start_offset = 0
    part_syllables = 0
    part_start_time = words[0].start
    last_time = words[0].start

    for index, (start, end, kind) in enumerate(tokens):
        token_text = text[start:end]
        if kind == "space":
            continue
        added = 1 if kind in {"my", "word"} else 0
        token_time_end = _time_at_offset(end, word_spans, last_time)
        token_time_start = _time_at_offset(start, word_spans, part_start_time)
        too_long = part_syllables + added > max_chars
        too_slow = (token_time_end - part_start_time) > max_duration and part_syllables >= 2
        if part_syllables > 0 and (too_long or too_slow) and start in break_offsets:
            part_text = text[part_start_offset:start].strip()
            if part_text:
                parts.append(
                    Part(
                        start=part_start_time,
                        end=token_time_start,
                        text=part_text,
                        speech_end=last_time,
                    )
                )
            part_start_offset = start
            part_syllables = 0
            part_start_time = token_time_start
        part_syllables += added
        last_time = max(last_time, token_time_end)

    tail = text[part_start_offset:].strip()
    if tail:
        parts.append(
            Part(
                start=part_start_time,
                end=max(last_time, words[-1].end),
                text=tail,
                speech_end=max(last_time, words[-1].end),
            )
        )
    return [p for p in parts if p.text] or [
        Part(words[0].start, words[-1].end, text, words[-1].end)
    ]


# ---------------------------------------------------------------------------
# CapCut hold timing
# ---------------------------------------------------------------------------


def enforce_hold_timing(
    parts: Sequence[Part],
    *,
    duration: float,
    max_hold_seconds: float = 3.0,
    min_duration: float = 0.3,
    last_hold_mode: str = "cap",
) -> list[CaptionSegment]:
    """Make captions persist until the next caption starts.

    ``segments[i].end = segments[i+1].start`` — the caption does not disappear
    during silence. The final caption holds until the video ends or for at most
    ``max_hold_seconds`` after the speech stops (``last_hold_mode="cap"``).
    """
    ordered = sorted(
        [p for p in parts if p.text.strip()],
        key=lambda p: (p.start, p.end),
    )
    if not ordered:
        return []
    if last_hold_mode not in {"cap", "video_end"}:
        raise ValueError("last_hold_mode must be 'cap' or 'video_end'")

    # 1. Merge parts that start too close together (cannot be displayed).
    merged: list[Part] = []
    for part in ordered:
        part = Part(max(0.0, part.start), max(part.start, part.end), part.text, part.speech_end)
        if merged and part.start - merged[-1].start < min_duration:
            previous = merged[-1]
            previous.text = mt.join_tokens(previous.text, part.text)
            previous.end = max(previous.end, part.end)
            previous.speech_end = max(previous.speech_end, part.speech_end)
            continue
        merged.append(part)

    # 2. CapCut rule: every caption ends exactly when the next one begins.
    segments: list[CaptionSegment] = []
    for index, part in enumerate(merged):
        if index + 1 < len(merged):
            end = merged[index + 1].start
        else:
            speech_end = part.speech_end or part.end or part.start
            if last_hold_mode == "video_end":
                end = duration if duration > 0 else speech_end + max_hold_seconds
            else:
                end = speech_end + max_hold_seconds
                if duration > 0:
                    end = min(end, duration)
            end = max(end, part.start + min_duration)
            if duration > 0:
                end = min(end, max(duration, part.start + min_duration))
        if end <= part.start:
            continue
        if segments and part.start < segments[-1].start:
            continue
        segments.append(
            CaptionSegment(
                start=part.start,
                end=end,
                text=part.text,
                speech_end=part.speech_end or part.end,
            )
        )

    # 3. Never overlap, always positive duration, minimum length where possible.
    for index, segment in enumerate(segments):
        segment.start = max(0.0, round(segment.start, 3))
        if index + 1 < len(segments):
            segment.end = round(max(segment.start + 0.001, segments[index + 1].start), 3)
            segments[index + 1].start = segment.end
        else:
            segment.end = round(max(segment.start + min_duration, segment.end), 3)
        if duration > 0:
            segment.start = min(segment.start, duration)
            segment.end = min(segment.end, max(duration, segment.start + 0.001))
    return [s for s in segments if s.end > s.start and s.text.strip()]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_segments(
    words: Sequence[Word],
    *,
    duration: float,
    gap_threshold: float = 0.5,
    punctuation_break: bool = True,
    max_chars: int = 35,
    max_duration: float = 6.0,
    min_duration: float = 0.3,
    max_hold_seconds: float = 3.0,
    last_hold_mode: str = "cap",
    sentence_ends: Iterable[float] = (),
) -> list[CaptionSegment]:
    """Tokens → CapCut-style captions (the function under unit test)."""
    cleaned = clean_words(words)
    if not cleaned:
        return []
    if duration <= 0:
        duration = cleaned[-1].end
    prepared = _expand_long_words(cleaned, max_chars=max_chars, max_duration=max_duration)
    parts: list[Part] = []
    for group in group_words(
        prepared,
        gap_threshold=gap_threshold,
        punctuation_break=punctuation_break,
        sentence_ends=sentence_ends,
    ):
        parts.extend(
            split_phrase(
                group,
                max_chars=max_chars,
                max_duration=max_duration,
                min_duration=min_duration,
            )
        )
    return enforce_hold_timing(
        parts,
        duration=duration,
        max_hold_seconds=max_hold_seconds,
        min_duration=min_duration,
        last_hold_mode=last_hold_mode,
    )


def segments_from_words(words: Sequence[Word], duration: float, settings: Any) -> list[CaptionSegment]:
    return build_segments(
        words,
        duration=duration,
        gap_threshold=settings.gap_threshold,
        punctuation_break=settings.punctuation_break,
        max_chars=settings.max_chars,
        max_duration=settings.max_duration,
        min_duration=settings.min_duration,
        max_hold_seconds=settings.max_hold_seconds,
        last_hold_mode=settings.last_hold_mode,
    )


# ---------------------------------------------------------------------------
# Timelines utilities used by the editor API and tests
# ---------------------------------------------------------------------------


def validate_segments(segments: Sequence[dict[str, Any]], duration: float = 0.0) -> list[str]:
    """Return a list of problems with a saved timeline (empty when valid)."""
    problems: list[str] = []
    previous_end = -1.0
    for index, segment in enumerate(segments):
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            problems.append(f"segment {index}: အချိန် ဖတ်၍ မရပါ")
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            problems.append(f"segment {index}: အချိန် မမှန်ပါ")
            continue
        if start < 0:
            problems.append(f"segment {index}: start < 0")
        if end <= start:
            problems.append(f"segment {index}: end <= start")
        if start + 1e-6 < previous_end:
            problems.append(f"segment {index}: ရှေ့ segment နှင့် ထပ်နေသည်")
        if duration > 0 and end > duration + 0.05:
            problems.append(f"segment {index}: ဗီဒီယို ကြာချိန်ထက် ကျော်နေသည်")
        if not str(segment.get("text", "")).strip():
            problems.append(f"segment {index}: စာသား မရှိပါ")
        previous_end = end
    return problems


def segments_to_dicts(segments: Sequence[CaptionSegment]) -> list[dict[str, Any]]:
    return [segment.to_dict() for segment in segments]


# ---------------------------------------------------------------------------
# Demo captions (used when Whisper is not installed, and by the tests)
# ---------------------------------------------------------------------------

DEMO_LINES: list[tuple[float, float, str]] = [
    (0.0, 1.5, "မင်္ဂလာပါ"),
    (2.5, 3.5, "ဒီနေ့"),
    (5.0, 7.0, "ရာသီဥတု ကောင်းတယ်"),
    (8.2, 11.0, "မြန်မာစာတန်းထိုး ဖန်တီးနည်းကို ပြပါမယ်"),
    (12.0, 14.5, "Blur နဲ့ Logo ကို စိတ်ကြိုက် နေရာချပါ"),
    (16.0, 19.0, "ပြီးရင် ဗီဒီယို သိမ်းဆည်းလိုက်ပါ"),
]


def demo_words(duration: float = 22.0) -> list[Word]:
    """Sample Burmese token stream with word-level timings.

    Instead of inventing text on a real job, this is used only for the demo
    button / automated tests when no ASR model is available.
    """
    words: list[Word] = []
    for start, end, line in DEMO_LINES:
        if start >= duration:
            break
        units: list[str] = []
        for token in mt.layout_tokens(line):
            if token.kind == "space":
                if units:
                    units[-1] += " "
                continue
            units.append(token.text)
        units = [unit for unit in units if unit.strip()]
        if not units:
            continue
        clamped_end = min(end, duration)
        span = max(0.2, clamped_end - start)
        step = span / len(units)
        for index, unit in enumerate(units):
            word_start = start + index * step
            words.append(
                Word(
                    start=round(word_start, 3),
                    end=round(min(word_start + step, clamped_end), 3),
                    text=unit,
                )
            )
    return words


def demo_transcribe(duration: float = 22.0, settings: Any = None) -> dict[str, Any]:
    words = demo_words(duration)
    if settings is not None:
        segments = segments_from_words(words, duration, settings)
    else:
        segments = build_segments(words, duration=duration)
    return {
        "words": [w.to_dict() for w in words],
        "segments": segments_to_dicts(segments),
        "language": "my",
        "duration": duration,
        "backend": "demo",
        "model": "demo",
    }


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model_cache: dict[tuple[str, ...], Any] = {}


def _resolve_device(settings: Any) -> tuple[str, str]:
    device = (settings.device or "auto").lower()
    compute = (settings.compute_type or "auto").lower()
    if device == "auto":
        device = "cpu"
        try:  # pragma: no cover - depends on host GPU availability
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _load_faster_whisper(settings: Any) -> Any:
    from faster_whisper import WhisperModel  # type: ignore

    device, compute = _resolve_device(settings)
    key = ("faster-whisper", settings.model_size, device, compute, str(settings.cpu_threads))
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        kwargs: dict[str, Any] = {"device": device, "compute_type": compute}
        if settings.cpu_threads:
            kwargs["cpu_threads"] = int(settings.cpu_threads)
        try:
            model = WhisperModel(settings.model_size, **kwargs)
        except Exception:
            if device == "cpu":
                raise
            # GPU present but unusable → fall back to CPU int8.
            model = WhisperModel(settings.model_size, device="cpu", compute_type="int8")
        _model_cache[key] = model
        return model


def _load_openai_whisper(settings: Any) -> Any:
    import whisper  # type: ignore

    key = ("openai-whisper", settings.model_size)
    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]
        device = "cuda" if (settings.device == "cuda") else "cpu"
        model = whisper.load_model(settings.model_size, device=device)
        _model_cache[key] = model
        return model


def transcribe_video(
    video_path: str,
    settings: Any,
    *,
    duration: float = 0.0,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run speech-to-text and return words + CapCut segments.

    Raises RuntimeError with a clear (Myanmar) message when the configured
    backend cannot be imported.
    """

    def report(fraction: float | None, stage: str) -> None:
        if progress:
            progress(fraction, stage)

    backend = settings.stt_backend
    if backend not in {"faster-whisper", "openai-whisper"}:
        raise RuntimeError(
            f"မသိသော STT backend '{backend}' — 'faster-whisper' သို့မဟုတ် 'openai-whisper' သုံးပါ"
        )

    words: list[Word] = []
    sentence_ends: list[float] = []
    detected_duration = duration
    detected_language = settings.language

    if backend == "faster-whisper":
        report(0.02, "မော်ဒယ် ဖွင့်နေသည်...")
        try:
            model = _load_faster_whisper(settings)
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper မော်ဒယ် ဖွင့်၍ မရပါ။ `pip install -r backend/requirements.txt` "
                f"ဖြင့် ထည့်ပြီး ပထမဆုံး အကြိမ်တွင် မော်ဒယ် download လိုအပ်ပါသည်။ ({exc})"
            ) from exc

        report(0.05, "အသံဖိုင် ဖတ်နေသည်...")
        segments_iter, info = model.transcribe(
            video_path,
            language=settings.language,
            beam_size=int(settings.beam_size),
            vad_filter=bool(settings.vad_filter),
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        detected_language = getattr(info, "language", settings.language) or settings.language
        total = float(getattr(info, "duration", 0.0) or detected_duration or 0.0)
        detected_duration = total or detected_duration
        for segment in segments_iter:
            seg_words = list(getattr(segment, "words", None) or [])
            for word in seg_words:
                words.append(Word(float(word.start), float(word.end), str(word.word)))
            if not seg_words and getattr(segment, "text", ""):
                # Some models/decoders omit word timings — fall back to the
                # segment text spread across its span.
                text = str(segment.text)
                pieces = [s for s in mt.syllables(text) if s.strip()] or [text]
                span = max(0.01, float(segment.end) - float(segment.start))
                step = span / max(1, len(pieces))
                for index, piece in enumerate(pieces):
                    words.append(
                        Word(
                            float(segment.start) + index * step,
                            min(float(segment.start) + (index + 1) * step, float(segment.end)),
                            piece,
                        )
                    )
            sentence_ends.append(float(segment.end))
            fraction = 0.05 + 0.85 * min(1.0, (float(segment.end) / total) if total else 0.5)
            report(fraction, "စကားပြောများ ဖတ်နေသည်...")
    else:  # openai-whisper
        report(0.05, "မော်ဒယ် ဖွင့်နေသည်...")
        try:
            model = _load_openai_whisper(settings)
        except Exception as exc:
            raise RuntimeError(
                "openai-whisper မော်ဒယ် ဖွင့်၍ မရပါ။ openai-whisper နှင့် torch ထည့်သွင်းရန် "
                f"README တွင် ကြည့်ပါ။ ({exc})"
            ) from exc
        report(0.2, "စကားပြောများ ဖတ်နေသည်...")
        result = model.transcribe(
            video_path,
            language=settings.language,
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,
            verbose=False,
        )
        detected_language = result.get("language", settings.language) or settings.language
        for segment in result.get("segments", []):
            for word in segment.get("words") or []:
                words.append(Word(float(word["start"]), float(word["end"]), str(word["word"])))
            if not segment.get("words") and segment.get("text"):
                text = str(segment["text"])
                pieces = [s for s in mt.syllables(text) if s.strip()] or [text]
                span = max(0.01, float(segment["end"]) - float(segment["start"]))
                step = span / max(1, len(pieces))
                for index, piece in enumerate(pieces):
                    words.append(
                        Word(
                            float(segment["start"]) + index * step,
                            min(float(segment["start"]) + (index + 1) * step, float(segment["end"])),
                            piece,
                        )
                    )
            sentence_ends.append(float(segment["end"]))
        report(0.9, "စာတန်း ဖန်တီးနေသည်...")

    words = clean_words(words)
    if not words:
        raise RuntimeError(
            "ဗီဒီယိုထဲတွင် စကားပြော မတွေ့ပါ (သို့မဟုတ် အသံလုံးဝ မရှိပါ)။ ဗီဒီယိုကို စစ်ဆေးပြီး "
            "ထပ်ကြိုးစားပါ။"
        )
    if detected_duration <= 0:
        detected_duration = max(word.end for word in words)
    segments = build_segments(
        words,
        duration=detected_duration,
        gap_threshold=settings.gap_threshold,
        punctuation_break=settings.punctuation_break,
        max_chars=settings.max_chars,
        max_duration=settings.max_duration,
        min_duration=settings.min_duration,
        max_hold_seconds=settings.max_hold_seconds,
        last_hold_mode=settings.last_hold_mode,
        sentence_ends=sentence_ends,
    )
    report(1.0, "ပြီးစီးပါပြီ")
    return {
        "words": [word.to_dict() for word in words],
        "segments": segments_to_dicts(segments),
        "language": detected_language,
        "duration": round(detected_duration, 3),
        "backend": backend,
        "model": settings.model_size,
    }

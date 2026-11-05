/* ============================================================
   myanmar_text.js — မြန်မာစာ syllable ခွဲခြင်း + line wrap (browser)
   backend/myanmar_text.py နှင့် တူညီသော ယုတ္တိ (preview/export ကိုက်ညီရန်)
   ============================================================ */
const MyanmarText = (() => {
  const VIRAMA = "\u1039"; // ္
  const ASAT = "\u103A"; // ်
  const TONES = new Set(["\u1037", "\u1038"]); // ့ း
  const FINAL_MARKS = new Set([ASAT, VIRAMA, ...TONES]);
  const MY_PUNCT = new Set(["\u104A", "\u104B"]); // ၊ ။
  const LATIN_PUNCT = new Set(".,!?;:%…—–".split(""));
  const BREAK_PUNCT = new Set([...MY_PUNCT, ...LATIN_PUNCT]);
  const SENTENCE_END = new Set(["\u104B", ".", "!", "?", "…"]);
  const ZERO_WIDTH = /[\u200b\u200c\u200d\ufeff]/g;
  const MARK_RE = /\p{M}/u;
  const LETTER_RE = /\p{L}/u;
  const MYANMAR_RE = /[\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F]/;

  function isMyanmarChar(ch) {
    return !!ch && MYANMAR_RE.test(ch);
  }

  function isMyanmarLetter(ch) {
    if (!isMyanmarChar(ch)) return false;
    if (ch === VIRAMA || ch === ASAT || TONES.has(ch)) return false;
    if (MY_PUNCT.has(ch)) return false;
    return !MARK_RE.test(ch);
  }

  function isMark(ch) {
    return MARK_RE.test(ch) || ch === VIRAMA;
  }

  function normalize(value) {
    if (!value) return "";
    let text = String(value).replace(ZERO_WIDTH, "");
    text = text.normalize("NFC");
    text = text.replace(/\r\n?/g, "\n");
    text = text.replace(/[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]+/g, " ");
    return text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length)
      .join("\n");
  }

  /* base letter + attached marks; virama pulls the next consonant in */
  function clusters(text) {
    const out = [];
    let current = "";
    let joinNext = false;
    for (const ch of text) {
      if (!current) {
        current = ch;
        joinNext = ch === VIRAMA;
        continue;
      }
      if (joinNext || isMark(ch)) {
        current += ch;
        joinNext = ch === VIRAMA;
        continue;
      }
      out.push(current);
      current = ch;
      joinNext = ch === VIRAMA;
    }
    if (current) out.push(current);
    return out;
  }

  function isFinalCluster(cluster) {
    if (!cluster) return false;
    if (cluster.includes(VIRAMA)) return true;
    if (!isMyanmarLetter(cluster[0])) return false;
    return cluster.length > 1 && [...cluster.slice(1)].every((ch) => FINAL_MARKS.has(ch));
  }

  function syllables(text) {
    const out = [];
    for (const cluster of clusters(text)) {
      if (out.length && isFinalCluster(cluster)) out[out.length - 1] += cluster;
      else out.push(cluster);
    }
    return out;
  }

  function kindOf(ch) {
    if (/\s/.test(ch)) return "space";
    if (BREAK_PUNCT.has(ch)) return "punct";
    if (isMyanmarChar(ch)) return "my";
    return "word";
  }

  function tokens(text) {
    const out = [];
    let buffer = "";
    let mode = "";
    const flush = () => {
      if (!buffer) return;
      if (mode === "my") syllables(buffer).forEach((syl) => out.push({ text: syl, kind: "my" }));
      else out.push({ text: buffer, kind: mode });
      buffer = "";
      mode = "";
    };
    for (const ch of text) {
      if (ch === "\n") {
        flush();
        out.push({ text: "\n", kind: "space" });
        continue;
      }
      const next = kindOf(ch);
      if (next !== mode) {
        flush();
        mode = next;
      }
      buffer += ch;
    }
    flush();
    return out;
  }

  /* breakable units; spaces and punctuation stick to the previous unit */
  function layoutUnits(text) {
    const units = [];
    for (const token of tokens(text)) {
      if (token.kind === "space") {
        if (token.text === "\n") {
          units.push("\n");
          continue;
        }
        if (units.length && units[units.length - 1] !== "\n") {
          units[units.length - 1] += " ";
        }
        continue;
      }
      if (token.kind === "punct") {
        if (units.length && units[units.length - 1] !== "\n") units[units.length - 1] += token.text;
        else units.push(token.text);
        continue;
      }
      units.push(token.text);
    }
    return units.filter((unit) => unit.length);
  }

  function wrap(text, maxWidth, measure, maxLines = 0) {
    if (!(maxWidth > 0)) return [String(text || "")];
    const lines = [];
    for (const paragraph of String(text || "").split("\n")) {
      const units = layoutUnits(paragraph);
      if (!units.length) {
        lines.push("");
        continue;
      }
      let current = "";
      for (const unit of units) {
        const candidate = current ? current + unit : unit;
        if (current && measure(candidate.replace(/\s+$/, "")) > maxWidth) {
          lines.push(current.replace(/\s+$/, ""));
          current = unit.replace(/^\s+/, "");
        } else {
          current = candidate;
        }
      }
      if (current.trim().length || !lines.length) lines.push(current.replace(/\s+$/, ""));
    }
    while (lines.length && !lines[0]) lines.shift();
    while (lines.length && !lines[lines.length - 1]) lines.pop();
    if (maxLines && lines.length > maxLines) {
      const head = lines.slice(0, maxLines - 1);
      head.push(lines.slice(maxLines - 1).join(" "));
      return head;
    }
    return lines;
  }

  function countGraphemes(text) {
    return tokens(text).filter((token) => token.kind === "my" || token.kind === "word" || token.kind === "punct")
      .length;
  }

  function isSentenceEnd(text) {
    const trimmed = String(text || "").replace(/\s+$/, "");
    return trimmed.length > 0 && SENTENCE_END.has(trimmed[trimmed.length - 1]);
  }

  function joinTokens(previous, next) {
    if (!previous) return next;
    if (!next) return previous;
    const prev = previous.replace(/\s+$/, "");
    const nxt = next.replace(/^\s+/, "");
    if (!nxt) return prev;
    if (previous !== prev || next !== nxt) return `${prev} ${nxt}`;
    const last = prev[prev.length - 1];
    const first = nxt[0];
    if (isMyanmarChar(last) || isMyanmarChar(first)) return prev + nxt;
    if (BREAK_PUNCT.has(last)) return prev + nxt;
    return `${prev} ${nxt}`;
  }

  function splitAtSyllable(text, fraction = 0.5) {
    const value = String(text || "").trim();
    if (!value) return ["", ""];
    const units = [];
    for (const token of tokens(value)) {
      if (token.kind === "space") {
        if (units.length) units[units.length - 1] += " ";
        continue;
      }
      if (token.kind === "punct" && units.length) {
        units[units.length - 1] += token.text;
        continue;
      }
      units.push(token.text);
    }
    if (units.length <= 1) return [value, ""];
    const total = units.reduce((sum, unit) => sum + unit.length, 0);
    const target = Math.max(1, total * Math.min(Math.max(fraction, 0.05), 0.95));
    let consumed = 0;
    let bestIndex = 1;
    let bestScore = Infinity;
    for (let index = 1; index < units.length; index += 1) {
      consumed += units[index - 1].length;
      let score = Math.abs(consumed - target);
      if (units[index - 1].endsWith(" ")) score -= total * 0.08;
      else if (BREAK_PUNCT.has(units[index - 1].trim().slice(-1))) score -= total * 0.05;
      if (score < bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }
    return [units.slice(0, bestIndex).join("").trim(), units.slice(bestIndex).join("").trim()];
  }

  return {
    normalize,
    clusters,
    syllables,
    tokens,
    layoutUnits,
    wrap,
    countGraphemes,
    isSentenceEnd,
    isMyanmarChar,
    isMyanmarLetter,
    joinTokens,
    splitAtSyllable,
  };
})();

if (typeof window !== "undefined") window.MyanmarText = MyanmarText;

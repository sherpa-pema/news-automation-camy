"""Shared broadcast text normalization for TTS synthesis and Timeline forced alignment.

Ensures VoiceAgent (TTS input) and TimelineAgent (MMS_FA forced alignment tokens)
use the exact same spoken-form expansion for:
- Pronunciation respellings (config/pronunciations.yaml, longest-first, whole-word)
- Currency (Rs. 5,000, NRs. 1,250.50, NPR 500, $12.50, etc.)
- Percentages (50%, 12.5%)
- Decimals (3.14, 105.92)
- Integers, comma-grouped numbers, ordinals (6th), dates (November 6, 2026), years
- Acronyms (NRB -> N R B, UML -> U M L)
- Optional Devanagari detection and uroman romanization check
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

WEAK_WORDS = {
    "a", "an", "the", "of", "to", "in", "and", "but", "for",
    "on", "at", "by", "with", "from", "as", "or", "into", "onto",
}

NUMBER_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
    40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
    80: "eighty", 90: "ninety",
}

ORDINAL_WORDS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 21: "twenty first", 22: "twenty second",
    23: "twenty third", 24: "twenty fourth", 25: "twenty fifth", 26: "twenty sixth",
    27: "twenty seventh", 28: "twenty eighth", 29: "twenty ninth",
    30: "thirtieth", 31: "thirty first",
}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
MONTHS_PATTERN = "|".join(sorted(MONTHS))

# Pronounceable acronyms / words that should NOT be spelled letter-by-letter
PRONOUNCEABLE_ACRONYMS = {
    "NASA", "NATO", "SAARC", "NEPSE", "NCELL", "COVID",
    "UNESCO", "UNICEF", "RADAR", "LASER", "SCUBA", "AIDS",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

CURRENCY_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:(?P<prefix>NRs|NPR|Rs)\.?\s*|(?P<sym>[$€£])\s*)"
    r"(?P<num>\d+(?:,\d+)*(?:\.\d+)?)"
    r"(?:\s+(?P<scale>hundreds?|thousands?|lakhs?|lacs?|millions?|crores?|arabs?|billions?|kharabs?|kharbas?|trillions?))?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(r"(?<![\d.])(?P<num>\d+(?:,\d+)*(?:\.\d+)?)\s*%")
DATE_DAY_RE = re.compile(
    rf"\b(?:{MONTHS_PATTERN})\s+(?P<day>\d{{1,2}})\b(?!\s*(?:st|nd|rd|th|%|\.\d))",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(r"\b(?P<num>\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
DECIMAL_RE = re.compile(r"(?<![\d.])\d+(?:,\d+)*\.\d+(?!(?:\.\d|\d))")
INTEGER_RE = re.compile(r"(?<![\d.])\d+(?:,\d+)*(?!(?:\.\d|\d))")
ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z]{2,6})(?![A-Za-z0-9_])")


def contains_devanagari(text: str) -> bool:
    """Returns True if text contains any Devanagari Unicode characters."""
    return bool(text and DEVANAGARI_RE.search(text))


def get_optional_uroman() -> Callable[[str], str] | None:
    """Returns a uroman romanizer callable if the uroman package is installed, else None."""
    try:
        import uroman as ur  # type: ignore[import-not-found]

        if hasattr(ur, "Uroman"):
            instance = ur.Uroman()
            return lambda s: str(instance.romanize_string(s))
        if hasattr(ur, "romanize_string"):
            return lambda s: str(ur.romanize_string(s))
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return None


def _number_under_thousand(num: int) -> str:
    if num in NUMBER_WORDS:
        return NUMBER_WORDS[num]
    if num < 100:
        tens = (num // 10) * 10
        units = num % 10
        return f"{NUMBER_WORDS[tens]} {NUMBER_WORDS[units]}"
    hundreds = num // 100
    rem = num % 100
    h_str = f"{NUMBER_WORDS[hundreds]} hundred"
    return f"{h_str} {_number_under_thousand(rem)}" if rem else h_str


def number_to_spoken_words(num: int, year_style: bool = True) -> str:
    """Converts a non-negative integer to spoken English words."""
    if num < 0:
        return f"minus {number_to_spoken_words(-num, year_style=year_style)}"
    if num in NUMBER_WORDS:
        return NUMBER_WORDS[num]
    if num < 1000:
        return _number_under_thousand(num)
    if year_style and 2000 <= num <= 2099:
        century = num // 100
        rem = num % 100
        century_str = _number_under_thousand(century)
        if rem == 0:
            return f"{century_str} hundred"
        if rem < 10:
            return f"{century_str} o {NUMBER_WORDS[rem]}"
        return f"{century_str} {_number_under_thousand(rem)}"

    scales = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ]
    parts: list[str] = []
    rem = num
    for scale_val, scale_name in scales:
        if rem >= scale_val:
            q, rem = divmod(rem, scale_val)
            parts.append(f"{number_to_spoken_words(q, year_style=False)} {scale_name}")
    if rem > 0:
        parts.append(_number_under_thousand(rem))
    return " ".join(parts)


def expand_ordinal(num: int) -> str:
    """Converts an integer to an ordinal spoken string (e.g. 6 -> 'sixth')."""
    if num in ORDINAL_WORDS:
        return ORDINAL_WORDS[num]
    cardinal = number_to_spoken_words(num, year_style=False)
    tokens = cardinal.split()
    if not tokens:
        return f"{num}th"
    last = tokens[-1]
    irregular = {
        "one": "first",
        "two": "second",
        "three": "third",
        "five": "fifth",
        "eight": "eighth",
        "nine": "ninth",
        "twelve": "twelfth",
    }
    if last in irregular:
        tokens[-1] = irregular[last]
    elif last.endswith("y"):
        tokens[-1] = last[:-1] + "ieth"
    else:
        tokens[-1] = last + "th"
    return " ".join(tokens)


def expand_decimal_str(num_str: str, allow_year_style: bool = True) -> str:
    """Converts a numeric string (with optional commas and decimal point) to spoken words."""
    clean = num_str.replace(",", "").strip()
    if "." in clean:
        int_part_str, frac_part_str = clean.split(".", 1)
        int_val = int(int_part_str) if int_part_str else 0
        int_spoken = number_to_spoken_words(int_val, year_style=False)
        digit_words = " ".join(NUMBER_WORDS[int(ch)] for ch in frac_part_str if ch.isdigit())
        return f"{int_spoken} point {digit_words}".strip()
    val = int(clean)
    has_comma = "," in num_str
    return number_to_spoken_words(val, year_style=(allow_year_style and not has_comma))


def expand_digits(word: str, is_date_context: bool = False) -> str:
    """Expands digits, decimals, percentages, or ordinals in a single token to spoken English."""
    core = re.sub(r"^[^\w%]+|[^\w%]+$", "", word)
    if not core:
        return word

    pct_match = re.match(r"^(\d+(?:,\d+)*(?:\.\d+)?)%$", core)
    if pct_match:
        return f"{expand_decimal_str(pct_match.group(1), allow_year_style=False)} percent"

    ord_match = re.match(r"^(\d+)(st|nd|rd|th)$", core, re.IGNORECASE)
    if ord_match:
        return expand_ordinal(int(ord_match.group(1)))

    if re.match(r"^\d+(?:,\d+)*\.\d+$", core):
        return expand_decimal_str(core, allow_year_style=False)

    if re.match(r"^\d+(?:,\d+)*$", core):
        n = int(core.replace(",", ""))
        if is_date_context and 1 <= n <= 31:
            return expand_ordinal(n)
        return expand_decimal_str(core, allow_year_style=True)

    return word


def preclean_text(text: str) -> str:
    """Strips paralinguistic tags and normalizes quotes, dashes, and spacing."""
    if not text:
        return ""
    cleaned = re.sub(r"\[.*?\]", "", text)
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", " - ").replace("–", " - ")
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _overlaps(start: int, end: int, spans: list[tuple[int, int, str]]) -> bool:
    return any(not (end <= s or start >= e) for s, e, _ in spans)


def _collect_normalization_spans(
    cleaned: str,
    pronunciations: dict[str, str] | None = None,
) -> list[tuple[int, int, str]]:
    """Finds non-overlapping normalized spans on pre-cleaned text in deterministic priority order."""
    spans: list[tuple[int, int, str]] = []

    def add_span(start: int, end: int, repl: str) -> None:
        if start < end and not _overlaps(start, end, spans):
            spans.append((start, end, repl))

    # 1. Currency (e.g. Rs. 5,000, NPR 50 crore, $12.50)
    for m in CURRENCY_RE.finditer(cleaned):
        prefix = m.group("prefix")
        sym = m.group("sym")
        num_str = m.group("num")
        scale = m.group("scale")

        spoken_num = expand_decimal_str(num_str, allow_year_style=False)
        is_singular = num_str.replace(",", "") in ("1", "1.0", "1.00") and not scale

        if prefix:
            unit = "rupee" if is_singular else "rupees"
        elif sym == "$":
            unit = "dollar" if is_singular else "dollars"
        elif sym == "€":
            unit = "euro" if is_singular else "euros"
        else:
            unit = "pound" if is_singular else "pounds"

        if scale:
            repl = f"{spoken_num} {scale.lower()} {unit}"
        else:
            repl = f"{spoken_num} {unit}"
        add_span(m.start(), m.end(), repl)

    # 2. Percentages (e.g. 50%, 12.5%)
    for m in PERCENT_RE.finditer(cleaned):
        spoken_num = expand_decimal_str(m.group("num"), allow_year_style=False)
        add_span(m.start(), m.end(), f"{spoken_num} percent")

    # 3. Pronunciation dictionary (longest-first, case-insensitive whole-word)
    if pronunciations:
        for p_word, p_val in sorted(pronunciations.items(), key=lambda kv: len(kv[0]), reverse=True):
            if not p_word:
                continue
            pattern = re.compile(rf"\b{re.escape(p_word)}\b", re.IGNORECASE)
            for m in pattern.finditer(cleaned):
                add_span(m.start(), m.end(), p_val)

    # 4. Date day context (e.g. November 6 -> November sixth)
    for m in DATE_DAY_RE.finditer(cleaned):
        day_val = int(m.group("day"))
        if 1 <= day_val <= 31:
            add_span(m.start("day"), m.end("day"), expand_ordinal(day_val))

    # 5. Explicit ordinals (e.g. 6th -> sixth)
    for m in ORDINAL_RE.finditer(cleaned):
        add_span(m.start(), m.end(), expand_ordinal(int(m.group("num"))))

    # 6. Decimals (e.g. 3.14, 105.92)
    for m in DECIMAL_RE.finditer(cleaned):
        add_span(m.start(), m.end(), expand_decimal_str(m.group(0), allow_year_style=False))

    # 7. Plain integers and years (e.g. 2026 -> twenty twenty six, 5,000 -> five thousand)
    for m in INTEGER_RE.finditer(cleaned):
        add_span(m.start(), m.end(), expand_decimal_str(m.group(0), allow_year_style=True))

    # 8. Unmapped all-caps acronyms (e.g. NRB -> N R B, UML -> U M L)
    for m in ACRONYM_RE.finditer(cleaned):
        acr = m.group(1)
        if acr in PRONOUNCEABLE_ACRONYMS:
            continue
        add_span(m.start(), m.end(), " ".join(acr))

    spans.sort(key=lambda sp: sp[0])
    return spans


def normalize_for_speech(
    text: str,
    pronunciations: dict[str, str] | None = None,
) -> str:
    """Normalizes broadcast script text into spoken form for TTS synthesis."""
    cleaned = preclean_text(text)
    if not cleaned:
        return ""

    spans = _collect_normalization_spans(cleaned, pronunciations)
    if not spans:
        return cleaned

    out_parts: list[str] = []
    cursor = 0
    for start, end, repl in spans:
        if cursor < start:
            out_parts.append(cleaned[cursor:start])
        out_parts.append(repl)
        cursor = end
    if cursor < len(cleaned):
        out_parts.append(cleaned[cursor:])

    result = "".join(out_parts)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return re.sub(r"\s+", " ", result).strip()


def _spoken_to_aligner_tokens(
    spoken_text: str,
    romanizer: Callable[[str], str] | None = None,
) -> tuple[str, list[str], bool]:
    """Converts a spoken string into lowercase [a-z'] aligner sub-tokens.

    Returns (effective_spoken_text, tokens, missing_romanizer_for_devanagari).
    """
    if contains_devanagari(spoken_text):
        if romanizer is None:
            return spoken_text, [], True
        spoken_text = romanizer(spoken_text)

    sub_tokens = [re.sub(r"[^a-z']", "", p.lower()) for p in re.split(r"[\s\-]+", spoken_text)]
    sub_tokens = [p for p in sub_tokens if p]
    return spoken_text, sub_tokens, False


def tokenize_for_alignment(
    text: str,
    pronunciations: dict[str, str] | None = None,
    romanizer: Callable[[str], str] | None = None,
    warnings_out: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tokenizes raw narration into display words paired with TTS spoken form and MMS tokens.

    Uses the exact same normalization rules as normalize_for_speech() so TTS and
    MMS_FA stay synchronized, while preserving original display chunks (like 'Rs. 5,000')
    for captions and timeline.json.
    """
    cleaned = preclean_text(text)
    if not cleaned:
        return []

    has_dev = contains_devanagari(cleaned)
    effective_romanizer = romanizer
    if has_dev and effective_romanizer is None:
        effective_romanizer = get_optional_uroman()
        if effective_romanizer is None and warnings_out is not None:
            warnings_out.append(
                "Devanagari characters detected in narration, but 'uroman' is not installed; "
                "Devanagari words will degrade to weighted interpolation."
            )

    spans = _collect_normalization_spans(cleaned, pronunciations)

    # Group characters into whitespace-delimited chunks while keeping multi-word
    # normalization spans (e.g. 'Rs. 5,000' or 'Sarita Rai') intact as one chunk.
    n = len(cleaned)
    span_idx = 0
    num_spans = len(spans)

    def span_containing(pos: int) -> tuple[int, int, str] | None:
        nonlocal span_idx
        while span_idx < num_spans and spans[span_idx][1] <= pos:
            span_idx += 1
        if span_idx < num_spans and spans[span_idx][0] <= pos < spans[span_idx][1]:
            return spans[span_idx]
        return None

    chunks: list[tuple[int, int]] = []
    pos = 0
    while pos < n:
        if cleaned[pos].isspace():
            pos += 1
            continue
        chunk_start = pos
        while pos < n and not cleaned[pos].isspace():
            sp = span_containing(pos)
            if sp is not None:
                pos = sp[1]
            else:
                pos += 1
        chunks.append((chunk_start, pos))

    words_data: list[dict[str, Any]] = []
    for c_start, c_end in chunks:
        orig_chunk = cleaned[c_start:c_end]

        # Build spoken string for this exact chunk using overlapping spans
        chunk_spans = [sp for sp in spans if sp[0] >= c_start and sp[1] <= c_end]
        if chunk_spans:
            parts: list[str] = []
            cur = c_start
            for s_st, s_en, repl in chunk_spans:
                if cur < s_st:
                    parts.append(cleaned[cur:s_st])
                parts.append(repl)
                cur = s_en
            if cur < c_end:
                parts.append(cleaned[cur:c_end])
            spoken_raw = "".join(parts)
            # Strip outer punctuation from spoken tts form (matching legacy prepare_scene_tokens)
            spoken_text = re.sub(r"^[^\w]+|[^\w]+$", "", spoken_raw).strip()
            if not spoken_text:
                spoken_text = spoken_raw.strip()
        else:
            core_match = re.match(r"^([^\w]*)([\w\-]+)([^\w]*)$", orig_chunk)
            spoken_text = core_match.group(2) if core_match else orig_chunk

        eff_spoken, sub_tokens, _ = _spoken_to_aligner_tokens(
            spoken_text,
            romanizer=effective_romanizer,
        )

        words_data.append({
            "orig": orig_chunk,
            "tts": eff_spoken,
            "tokens": sub_tokens,
        })

    return words_data

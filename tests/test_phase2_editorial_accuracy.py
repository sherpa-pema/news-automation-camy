"""Tests for Phase 2 Editorial Accuracy: Devanagari normalization, font fallbacks, and motion classification."""
import re
from pathlib import Path
import pytest
from src.utils.text_normalizer import (
    preclean_text,
    devanagari_to_roman_fallback,
    number_to_spoken_words,
    normalize_for_speech,
    split_into_sentences,
    _clean_word_for_lookup,
)
from src.utils.motion_graphics import (
    get_font_candidates,
    get_font_path,
    get_font,
)
from src.utils.scene_classifier import classify_scene


def test_devanagari_digit_conversion_in_preclean():
    """Verify Devanagari digits (०-९) are converted to ASCII (0-9) in preclean_text."""
    raw = "सडक मर्मतका लागि रू ५,००० विनियोजन गरिएको छ।"
    cleaned = preclean_text(raw)
    assert "५,०००" not in cleaned
    assert "5,000" in cleaned


def test_devanagari_transliteration_conjuncts_and_schwa():
    """Verify 3-codepoint conjuncts (क्ष, त्र, ज्ञ, श्र) and final-consonant schwa deletion."""
    # Schwa deletion: final consonants do not carry trailing 'a'
    nepal = devanagari_to_roman_fallback("नेपाल")
    assert nepal == "nepaal"

    ram = devanagari_to_roman_fallback("राम")
    assert ram == "raam"

    # 3-codepoint conjuncts:
    # ज्ञ (gy) + ान -> gyaan
    gyan = devanagari_to_roman_fallback("ज्ञान")
    assert "gy" in gyan

    # क्ष (ksh) + ेत्र -> kshetr
    kshetra = devanagari_to_roman_fallback("क्षेत्र")
    assert "ksh" in kshetra

    # त्र (tr) + िशूल -> trishool
    trishul = devanagari_to_roman_fallback("त्रिशूल")
    assert trishul.startswith("tr")


def test_clean_word_for_lookup_preserves_matras():
    """Punctuation stripping must not strip vowel matras (ा, ि, ी, etc.)."""
    word = ' "नेपाल," '
    cleaned = _clean_word_for_lookup(word)
    assert cleaned == "नेपाल"

    word_with_danda = "काठमाडौं।"
    assert _clean_word_for_lookup(word_with_danda) == "काठमाडौं"


def test_spoken_numbers_millennium_and_twentieth_century():
    """Verify year 2000 is 'two thousand' and 1900-1999 expands to year style."""
    assert number_to_spoken_words(2000, year_style=True) == "two thousand"
    assert number_to_spoken_words(2026, year_style=True) == "twenty twenty six"
    assert number_to_spoken_words(2083, year_style=True) == "twenty eighty three"
    assert number_to_spoken_words(1900, year_style=True) == "nineteen hundred"
    assert number_to_spoken_words(1995, year_style=True) == "nineteen ninety five"


def test_nepali_month_dates_expansion():
    """Nepali month names followed by day numbers must expand to spoken ordinals."""
    res = normalize_for_speech("The festival begins on Kartik 22 in Kathmandu.")
    assert "Kartik twenty second" in res

    res2 = normalize_for_speech("New fiscal policies take effect on Baisakh 1 across Nepal.")
    assert "Baisakh first" in res2


def test_scene_classifier_devanagari_danda_splitting():
    """Devanagari dandas (।) must split sentences and trigger headline_split when >= 3 sentences."""
    script_scene = {
        "id": "scene_02",
        "type": "story",
        "headline": "काठमाडौं सडक विस्तार",
        "narration": "काठमाडौं उपत्यकामा सडक विस्तार तीव्र पारिएको छ। मुख्य सडक खण्डहरूमा नयाँ कालोपत्रे सुरु भएको छ। ट्राफिक व्यवस्थापन सुधार गर्न थप प्रहरी परिचालन गरिएको छ।",
        "category": "infrastructure",
        "sources": ["Kantipur"],
    }
    cls = classify_scene(script_scene, has_video=False)
    assert cls.recipe == "headline_split"
    assert len(cls.data.get("bullets", [])) == 2


def test_scene_classifier_nepali_quote_attribution():
    """Nepali speech verbs (भने, बताए, अनुसार) must trigger quote_card with speaker extraction."""
    script_scene = {
        "id": "scene_03",
        "type": "story",
        "headline": "ऊर्जा मन्त्रालयको घोषणा",
        "narration": 'ऊर्जा मन्त्रीले "नेपालले आगामी वर्ष थप बिजुली निर्यात गर्नेछ" भने।',
        "sources": ["Gorkhapatra"],
    }
    cls = classify_scene(script_scene, has_video=False)
    assert cls.recipe == "quote_card"
    assert "थप बिजुली निर्यात" in cls.data.get("quote", "")


def test_font_fallbacks_include_nirmala():
    """Font candidate discovery must prioritize Nirmala UI for Devanagari script rendering."""
    primary = get_font_candidates(regular=False)
    regular = get_font_candidates(regular=True)
    assert any("Nirmala" in f for f in primary)
    assert any("Nirmala" in f for f in regular)

    # get_font must return a valid font object without raising
    font = get_font(primary, 32)
    assert font is not None

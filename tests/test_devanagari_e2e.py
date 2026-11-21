"""End-to-end tests for Devanagari text normalization, scene classification,
motion graphics rendering, and ASS subtitle generation.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import pytest

from src.utils.text_normalizer import (
    contains_devanagari,
    normalize_for_speech,
    split_into_sentences,
)
from src.utils.scene_classifier import classify_scene
from src.utils.motion_graphics import (
    draw_lower_third,
    draw_quote_card,
    generate_ass_subtitles,
    get_font,
    get_font_candidates,
)


def test_devanagari_text_normalizer() -> None:
    """Tests Devanagari script detection, numeral expansion, currency conversion, and sentence splitting."""
    sample = "नेपाली, काठमाडौं, क्षत्रिय, २०८३, Rs 1,50,000"
    assert contains_devanagari(sample) is True

    normalized = normalize_for_speech(sample)
    # Check Nepali terms preserved
    assert "नेपाली" in normalized
    assert "काठमाडौं" in normalized
    assert "क्षत्रिय" in normalized
    # Check numeral 2083 expanded to spoken words
    assert "twenty eighty three" in normalized
    # Check Rs 1,50,000 expanded to rupees
    assert "one hundred fifty thousand rupees" in normalized

    # Purna viram sentence splitting
    nepali_paragraph = "काठमाडौं नेपालको राजधानी हो। पोखरा पर्यटकीय शहर हो। लुम्बिनी बुद्धको जन्मस्थल हो।"
    sentences = split_into_sentences(nepali_paragraph)
    assert len(sentences) == 3
    assert sentences[0].endswith("हो।")
    assert sentences[1].endswith("हो।")
    assert sentences[2].endswith("हो।")


def test_devanagari_scene_classifier() -> None:
    """Tests scene classification heuristics with Nepali sentences and quote attributions."""
    sc_quote = {
        "type": "story",
        "headline": "सडक विस्तार परियोजना तीव्र पारिने",
        "narration": '“काठमाडौं उपत्यकाको सडक विस्तार समयमै सम्पन्न हुनेछ,” मन्त्री राम थापाले बताए।',
        "sources": ["कान्तिपुर"],
    }
    classification = classify_scene(sc_quote)
    assert classification.recipe == "quote_card"
    assert "काठमाडौं उपत्यकाको सडक विस्तार" in classification.data.get("quote", "")
    assert "राम थापा" in classification.data.get("speaker", "")

    sc_stat = {
        "type": "story",
        "headline": "विदेशी मुद्रा सञ्चिति १५ प्रतिशतले वृद्धि",
        "narration": "नेपाल राष्ट्र बैंकका अनुसार चालू आर्थिक वर्षमा विदेशी मुद्रा सञ्चिति १५ प्रतिशतले बढेको छ।",
        "sources": ["नागरिक"],
    }
    classification_stat = classify_scene(sc_stat)
    assert classification_stat.recipe in ("stat_callout", "lower_third", "quote_card", "cinematic_broll_full")


def test_devanagari_motion_graphics_rendering(tmp_path: Path) -> None:
    """Verifies that motion graphics cards render Devanagari characters with non-blank glyph pixels."""
    font_candidates = [
        "C:\\Windows\\Fonts\\NirmalaB.ttf",
        "C:\\Windows\\Fonts\\Nirmala.ttc",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
    ]
    font = get_font(font_candidates, 36)
    assert font is not None

    out_png = tmp_path / "nepali_card.png"
    draw_lower_third(
        headline="काठमाडौंमा नयाँ मेट्रो परियोजनाको शिलान्यास",
        source_text="नेपाल टेलिभिजन",
        output_path=out_png,
        width=1680,
        height=160,
        font_candidates=font_candidates,
    )

    assert out_png.exists()
    img = Image.open(out_png)
    assert img.size == (1680, 160)

    # Verify non-blank glyph pixels were drawn
    has_visible_pixels = False
    for pixel in img.getdata():
        if pixel[3] > 0 and (pixel[0] > 50 or pixel[1] > 50 or pixel[2] > 50):
            has_visible_pixels = True
            break
    assert has_visible_pixels is True


def test_devanagari_generate_ass_subtitles(tmp_path: Path) -> None:
    """Tests ASS subtitle generation with BorderStyle=3 and safe margin configuration."""
    captions = [
        {"id": "1", "start": 0.0, "end": 2.5, "text": "नमस्ते, आजको मुख्य समाचार।"},
        {"id": "2", "start": 2.5, "end": 5.0, "text": "काठमाडौंमा ट्राफिक व्यवस्थापन सुधार।"},
    ]
    ass_path = tmp_path / "captions.ass"
    generate_ass_subtitles(
        captions=captions,
        output_path=ass_path,
        resolution=(1920, 1080),
        font_size=52,
        margin_v=80,
        font_name="Nirmala UI",
        border_style=3,
        box_rgba=[7, 20, 38, 220],
    )

    assert ass_path.exists()
    content = ass_path.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "Style: Default,Nirmala UI,52" in content
    # Verify BorderStyle=3 is present in style definition
    assert ",3,3.5,1.0,2,40,40,80,1" in content
    # Verify dialogue lines with Devanagari text
    assert "Dialogue: 0,0:00:00.00,0:00:02.50,Default,,0,0,0,,नमस्ते, आजको मुख्य समाचार।" in content
    assert "Dialogue: 0,0:00:02.50,0:00:05.00,Default,,0,0,0,,काठमाडौंमा ट्राफिक व्यवस्थापन सुधार।" in content

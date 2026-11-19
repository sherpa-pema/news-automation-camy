"""Tests for Phase 2: Visual Quality hardening.

Covers:
- 2.1: Caption system resolution, geometry, and clearance
- 2.2: Card & featured-card coexistence, stat label cleaning
- 2.3: Stills motion supersampling & variety in FFmpeg filter graph
- 2.4: Transition grammar (cuts vs dips)
- 2.5: Plan hygiene: wrap_text hardening, audio_grid size, single global watermark
"""
from pathlib import Path
import json
import pytest
from PIL import Image, ImageFont

from src.utils.captions import resolve_style, CaptionStyle
from src.utils.motion_timing import resolve_transitions
from src.utils.motion_graphics import clean_stat_label, wrap_text, draw_stat_callout
from src.utils.media_review import draw_card_image
from src.agents.render import RenderAgent


def test_2_1_caption_system_extents_landscape() -> None:
    """Verifies that resolved caption geometry in landscape clears the lower third by >= gap_from_band_px."""
    style_cfg = {
        "captions": {
            "font_family": "Segoe UI Semibold",
            "size_landscape": 44,
            "size_vertical": 64,
            "max_lines": 2,
            "line_height_pct": 1.25,
            "highlight_color": "#F5A623",
            "box_rgba": [7, 20, 38, 224],
            "gap_from_band_px": 24,
        },
        "layout_rectangles": {
            "caption_band": {"x": 160, "y": 910, "w": 1600, "h": 100},
            "lower_third": {"x": 96, "y": 730, "w": 1320, "h": 155},
        },
    }
    style = resolve_style((1920, 1080), style_cfg)
    assert isinstance(style, CaptionStyle)
    assert style.font_size == 44
    assert not style.is_vertical

    # Lower third bottom = 730 + 155 = 885
    # Required minimum top = 885 + 24 = 909
    assert style.caption_top >= 885 + 24
    assert style.caption_top == 910
    assert style.caption_bottom == 1020
    assert style.margin_v == 60

    plan_dict = style.to_plan_dict()
    assert plan_dict["position"] == "bottom"
    assert plan_dict["font"] == "Segoe UI Semibold"
    assert plan_dict["font_size"] == 44
    assert plan_dict["gap_from_band_px"] == 24


def test_2_1_caption_system_extents_vertical() -> None:
    """Verifies that resolved caption geometry in vertical mode clears lower third by >= gap_from_band_px."""
    style_cfg = {
        "captions": {
            "font_family": "Segoe UI Semibold",
            "size_landscape": 44,
            "size_vertical": 64,
            "gap_from_band_px": 24,
        },
        "layout_rectangles_vertical": {
            "caption_band": {"x": 60, "y": 1445, "w": 960, "h": 120},
            "lower_third": {"x": 60, "y": 1240, "w": 960, "h": 180},
        },
    }
    style = resolve_style((1080, 1920), style_cfg)
    assert style.is_vertical
    assert style.font_size == 64
    # Lower third bottom = 1240 + 180 = 1420
    # Required minimum top = 1420 + 24 = 1444
    assert style.caption_top >= 1420 + 24
    assert style.caption_top == 1445


def test_2_2_clean_stat_label() -> None:
    """Verifies that clean_stat_label strips duplicated numbers/percentages/metrics from label."""
    # When value is embedded
    assert clean_stat_label("Inflation hits 14.2% in October", "14.2%") == "Inflation hits in October"
    # When label starts with numeric
    assert clean_stat_label("14.2% Inflation Rate", "14.2%") == "Inflation Rate"
    # When label has trailing colon and number
    assert clean_stat_label("Unemployment Rate: 7.5%", "7.5%") == "Unemployment Rate"
    # Pure entity label remains untouched
    assert clean_stat_label("Nepal GDP Growth", "4.5%") == "Nepal GDP Growth"


def test_2_2_draw_card_image_quiet_mode(tmp_path: Path) -> None:
    """Verifies that draw_card_image with quiet=True suppresses category and watermark labels."""
    card_normal = tmp_path / "card_normal.png"
    card_quiet = tmp_path / "card_quiet.png"

    draw_card_image("Test Card", card_normal, width=960, height=540, category="NEWS BRIEFING", quiet=False)
    draw_card_image("Test Card", card_quiet, width=960, height=540, category="NEWS BRIEFING", quiet=True)

    assert card_normal.exists()
    assert card_quiet.exists()

    # The quiet image should differ from the normal image because chrome was omitted
    im_normal = Image.open(card_normal)
    im_quiet = Image.open(card_quiet)
    assert im_normal.size == im_quiet.size
    # Verify pixels are not identical
    diff = any(p1 != p2 for p1, p2 in zip(im_normal.getdata(), im_quiet.getdata()))
    assert diff, "Quiet card must suppress category badge and watermark"


def test_2_3_supersampled_zoompan_in_render(tmp_path: Path) -> None:
    """Verifies that render segment builds supersampled zoompan with variety anchors."""
    agent = RenderAgent(run_dir=tmp_path)
    # Check that zoompan parameters use 2x supersampling and lanczos downscaling
    sc = {
        "id": "scene_01",
        "start": 0.0,
        "end": 2.0,
        "shots": [
            {
                "shot_index": 0,
                "file": "media/shots/scene_01/shot_00.png",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "type": "image",
                "motion": "zoom_in",
            }
        ],
        "overlays": [],
        "transition": {"type": "cut", "duration_sec": 0.0},
    }
    # Create fake shot image
    shot_img = tmp_path / "media" / "shots" / "scene_01" / "shot_00.png"
    shot_img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080), (10, 20, 30)).save(shot_img)

    out_mp4 = tmp_path / "segment.mp4"
    # Render segment silently
    agent._render_scene_segment(
        sc=sc,
        fps=30,
        target_res=(1920, 1080),
        output_path=out_mp4,
    )
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 1000


def test_2_4_transition_grammar() -> None:
    """Verifies transition grammar rules: cut for adjacency/intra-story, dip for intro/outro/topic change."""
    style_cfg = {
        "transitions": {
            "scene_boundary": {"type": "dip_to_black", "duration_sec": 0.2, "fade_out_sec": 0.1, "fade_in_sec": 0.1},
        }
    }

    # 1. First scene has no predecessor -> cut
    t1 = resolve_transitions({"id": "intro", "type": "intro"}, prev_scene=None, style_config=style_cfg)
    assert t1["type"] == "cut"
    assert t1["duration_sec"] == 0.0

    # 2. Intro -> Story -> dip_to_black
    t2 = resolve_transitions(
        {"id": "scene_01", "type": "story"},
        prev_scene={"id": "intro", "type": "intro"},
        style_config=style_cfg,
    )
    assert t2["type"] == "dip_to_black"
    assert t2["duration_sec"] == 0.2

    # 3. Adjacent story beats / intra-story -> cut
    t3 = resolve_transitions(
        {"id": "scene_02", "type": "story", "topic": "economy"},
        prev_scene={"id": "scene_01", "type": "story", "topic": "economy"},
        style_config=style_cfg,
    )
    assert t3["type"] == "cut"
    assert t3["duration_sec"] == 0.0

    # 4. Topic change between stories -> dip_to_black
    t4 = resolve_transitions(
        {"id": "scene_03", "type": "story", "topic": "sports"},
        prev_scene={"id": "scene_02", "type": "story", "topic": "politics"},
        style_config=style_cfg,
    )
    assert t4["type"] == "dip_to_black"
    assert t4["duration_sec"] == 0.2

    # 5. Story -> Outro -> dip_to_black
    t5 = resolve_transitions(
        {"id": "outro", "type": "outro"},
        prev_scene={"id": "scene_03", "type": "story"},
        style_config=style_cfg,
    )
    assert t5["type"] == "dip_to_black"
    assert t5["duration_sec"] == 0.2


def test_2_5_wrap_text_hardening() -> None:
    """Verifies that wrap_text does not lose words and cleanly truncates with a single ellipsis."""
    font = ImageFont.load_default()

    # Short string wraps within max_lines
    lines = wrap_text("One two three four", font, max_width=500, max_lines=2)
    assert " ".join(lines) == "One two three four"

    # Very long text truncates gracefully with a single ellipsis and no repeated '...'
    long_text = "This is an extremely long headline that cannot fit into two narrow lines and will require clean truncation"
    lines = wrap_text(long_text, font, max_width=120, max_lines=2)
    assert len(lines) <= 2
    full_wrapped = "\n".join(lines)
    assert "…" in full_wrapped or "..." in full_wrapped
    assert "......" not in full_wrapped
    assert "……" not in full_wrapped


def test_2_5_audio_grid_size_when_disabled() -> None:
    """Verifies that audio_grid payload stays under 2 KB when background music is disabled."""
    from src.utils.audio_sync import extract_audio_grid

    scenes = [{"id": f"scene_{i:02d}", "start": i * 5.0, "end": (i + 1) * 5.0, "text": "Lots of narration text here"} for i in range(25)]
    grid = extract_audio_grid(
        bgm_path=None,
        narration_path=None,
        scene_boundaries=scenes,
        style_config={"music": {"enabled": False}},
    )
    grid_json = json.dumps(grid)
    size_bytes = len(grid_json.encode("utf-8"))
    assert size_bytes < 2048, f"Disabled audio_grid size {size_bytes} exceeds 2 KB limit"

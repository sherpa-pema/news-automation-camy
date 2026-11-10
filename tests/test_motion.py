"""Unit tests for Motion Agent, ASS generation, and overlay layout safety."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.agents.motion import MotionAgent
from src.utils.motion_graphics import (
    draw_lower_third,
    draw_outro_card,
    draw_title_card,
    generate_ass_subtitles,
)
from src.utils.schema_validate import validate_contract
from src.utils.tts import create_silence_wav


@pytest.fixture
def style_config() -> dict[str, Any]:
    with open("config/style.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_captions() -> list[dict[str, Any]]:
    return [
        {
            "id": "c1",
            "scene_id": "intro",
            "start": 0.5,
            "end": 2.5,
            "text": "Hello world, welcome to the daily news.",
        },
        {
            "id": "c2",
            "scene_id": "scene_01",
            "start": 3.0,
            "end": 6.0,
            "text": "Kathmandu weather is sunny {clear} & bright.",
        },
    ]


def test_ass_generation_escaping_and_structure(tmp_path: Path, sample_captions: list[dict[str, Any]]) -> None:
    ass_path = tmp_path / "test.ass"
    generate_ass_subtitles(sample_captions, ass_path, font_size=54)

    assert ass_path.exists()
    content = ass_path.read_text(encoding="utf-8")

    assert "[Script Info]" in content
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "[V4+ Styles]" in content
    assert "Style: Default,Arial,54" in content
    assert "[Events]" in content
    assert "Dialogue: 0,0:00:00.50,0:00:02.50,Default" in content
    # Curly braces must be escaped as \( and \)
    assert r"\{clear\}" in content
    assert "{" not in content.replace(r"\{", "").replace(r"\}", "")


def test_overlay_safe_zone_and_non_overlap(style_config: dict[str, Any]) -> None:
    layout = style_config["layout_rectangles"]
    safe = style_config.get("safe_margin_pct", 0.05)
    assert safe == 0.05

    # Safe boundaries in 1920x1080:
    min_x, max_x = 1920 * safe, 1920 * (1 - safe)
    min_y, max_y = 1080 * safe, 1080 * (1 - safe)

    # Lower third
    lt = layout["lower_third"]
    assert lt["x"] >= min_x
    assert lt["x"] + lt["w"] <= max_x
    assert lt["y"] >= min_y
    assert lt["y"] + lt["h"] <= max_y

    # Caption band
    cb = layout["caption_band"]
    assert cb["y"] >= min_y
    assert cb["y"] + cb["h"] <= max_y

    # Strict non-overlap: lower third must sit ABOVE the caption band
    assert (lt["y"] + lt["h"]) <= cb["y"], (
        f"Lower third bottom ({lt['y'] + lt['h']}) collides with caption band top ({cb['y']})"
    )

    # Tags
    for tag_key in ["illustrative_tag", "test_run_tag"]:
        tag = layout[tag_key]
        assert tag["x"] >= min_x
        assert tag["x"] + tag["w"] <= max_x
        assert tag["y"] >= min_y
        assert tag["y"] + tag["h"] <= max_y


def test_pillow_overlay_generation(tmp_path: Path, style_config: dict[str, Any]) -> None:
    lt_path = tmp_path / "lt.png"
    draw_lower_third(
        headline="Short Test Headline",
        source_text="OnlineKhabar",
        output_path=lt_path,
    )
    assert lt_path.exists()
    assert lt_path.stat().st_size > 500

    title_path = tmp_path / "title.png"
    draw_title_card(
        channel_name="Nepal Daily Briefing",
        date_str="Sunday, November 8, 2026",
        output_path=title_path,
    )
    assert title_path.exists()
    assert title_path.stat().st_size > 500

    outro_path = tmp_path / "outro.png"
    draw_outro_card(
        channel_name="Nepal Daily Briefing",
        output_path=outro_path,
        disclose_synthetic_voice=True,
    )
    assert outro_path.exists()
    assert outro_path.stat().st_size > 500


def test_motion_agent_plan_creation_and_watermark(tmp_path: Path) -> None:
    # Set up minimal run fixture
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "run_kind": "test",
        "origin": "fixture",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Historic Win in NPL",
                "source": "MyRepublica",
                "source_urls": ["https://example.com/1"],
            },
            {"id": "outro", "type": "outro", "headline": "Summary"},
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    tl_dir = tmp_path / "timeline"
    tl_dir.mkdir(parents=True)
    tl_data = {
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 4.0, "speech_start": 0.5, "speech_end": 3.5},
            {"id": "scene_01", "type": "story", "start": 4.0, "end": 10.0, "speech_start": 4.5, "speech_end": 9.5},
            {"id": "outro", "type": "outro", "start": 10.0, "end": 14.0, "speech_start": 10.5, "speech_end": 13.5},
        ]
    }
    with open(tl_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(tl_data, f)

    with open(tl_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"id": "c1", "scene_id": "intro", "start": 0.5, "end": 3.0, "text": "Welcome to news"},
                {"id": "c2", "scene_id": "scene_01", "start": 4.5, "end": 9.0, "text": "Historic cricket win"},
            ],
            f,
        )

    # Audio
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    create_silence_wav(audio_dir / "narration.wav", duration_sec=14.0)

    # Dummy media manifest
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    media_data = {
        "resolution": {"width": 1920, "height": 1080},
        "fps": 30,
        "shots": [
            {
                "id": "shot_0",
                "scene_id": "intro",
                "file_path": "media/shots/intro_0.jpg",
                "type": "image",
                "source": "pixabay",
            },
            {
                "id": "shot_1",
                "scene_id": "scene_01",
                "file_path": "media/shots/s1_0.jpg",
                "type": "image",
                "source": "wikimedia_commons",
                "attribution_required": True,
            },
            {
                "id": "shot_2",
                "scene_id": "outro",
                "file_path": "media/shots/outro_0.jpg",
                "type": "image",
                "source": "pixabay",
            },
        ],
    }
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump(media_data, f)

    agent = MotionAgent(run_dir=tmp_path)
    outputs = agent.run()

    plan_path = tmp_path / "motion" / "render_plan.json"
    assert plan_path.exists()

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # Validate against schema contract
    is_valid, errs = validate_contract(plan, "contracts/render_plan.schema.json")
    assert is_valid, f"Render plan failed schema validation: {errs}"

    # Verify watermark scheduled since run_kind is 'test'
    overlays = plan.get("overlays", [])
    has_test_tag = any(ov.get("type") in ("test_tag", "test_run_tag") for ov in overlays)
    assert has_test_tag, "Test run watermark tag should be scheduled when run_kind is 'test'"

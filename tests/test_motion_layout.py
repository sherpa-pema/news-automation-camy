"""Unit tests for pre-render layout validation and safe zone enforcement."""
from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from src.agents.motion import MotionAgent
from src.utils.overlay_plan import validate_layout
from src.utils.tts import create_silence_wav


def test_validate_layout_safe_margin_violation() -> None:
    style_config = {"safe_margin_pct": 0.05}
    # 1920x1080 -> min_x=96, min_y=54, max_x=1824, max_y=1026
    plan_data = {
        "resolution": "1920x1080",
        "scenes": [],
        "overlays": [
            {
                "id": "bad_overlay_left",
                "type": "lower_third",
                "x": 20,  # < 96
                "y": 800,
                "w": 500,
                "h": 100,
            }
        ],
    }
    errors = validate_layout(plan_data, style_config)
    assert len(errors) == 1
    assert "bad_overlay_left" in errors[0]
    assert "safe margin" in errors[0]


def test_validate_layout_caption_band_collision() -> None:
    style_config = {
        "safe_margin_pct": 0.05,
        "layout_rectangles": {
            "caption_band": {"y": 910, "h": 120},
        },
    }
    # lower_third y + h = 780 + 155 = 935 > 910
    plan_data = {
        "resolution": "1920x1080",
        "scenes": [],
        "overlays": [
            {
                "id": "scene_01_lower_third",
                "type": "lower_third",
                "x": 100,
                "y": 780,
                "w": 1720,
                "h": 155,
            }
        ],
    }
    errors = validate_layout(plan_data, style_config)
    assert any("overlaps caption band" in e and "scene_01_lower_third" in e for e in errors)


def test_validate_layout_featured_card_caption_band_collision() -> None:
    style_config = {
        "safe_margin_pct": 0.05,
        "layout_rectangles": {
            "caption_band": {"y": 910, "h": 120},
        },
    }
    # stat_callout extending too far down into caption band: y=500, h=450 -> y+h=950 > 910
    plan_data = {
        "resolution": "1920x1080",
        "scenes": [],
        "overlays": [
            {
                "id": "scene_01_stat_callout",
                "type": "stat_callout",
                "x": 100,
                "y": 500,
                "w": 650,
                "h": 450,
            }
        ],
    }
    errors = validate_layout(plan_data, style_config)
    assert any("overlaps caption band" in e and "scene_01_stat_callout" in e for e in errors)


def test_validate_layout_intra_scene_aabb_collision() -> None:
    style_config = {
        "safe_margin_pct": 0.05,
        "layout_rectangles": {
            "caption_band": {"y": 910, "h": 120},
        },
    }
    # Two co-visible overlays in same scene overlapping in space
    plan_data = {
        "resolution": "1920x1080",
        "scenes": [
            {
                "id": "scene_01",
                "overlays": [
                    {
                        "id": "ov_a",
                        "type": "stat_callout",
                        "start": 1.0,
                        "end": 6.0,
                        "x": 200,
                        "y": 200,
                        "w": 400,
                        "h": 300,
                    },
                    {
                        "id": "ov_b",
                        "type": "quote_card",
                        "start": 2.0,
                        "end": 7.0,
                        "x": 300,
                        "y": 250,
                        "w": 400,
                        "h": 300,
                    },
                ],
            }
        ],
        "overlays": [],
    }
    errors = validate_layout(plan_data, style_config)
    assert any("overlap spatially while co-visible" in e and "ov_a" in e and "ov_b" in e for e in errors)


def test_motion_agent_fails_fast_on_layout_violation(tmp_path: Path) -> None:
    """Asserts MotionAgent.run() raises ValueError in < 1s when layout collision is injected."""
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "run_kind": "test",
        "origin": "fixture",
        "date": "2026-11-15",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {"id": "scene_01", "type": "story", "headline": "Test Story", "source": "Test Source"},
            {"id": "outro", "type": "outro", "headline": "Summary"},
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    tl_dir = tmp_path / "timeline"
    tl_dir.mkdir(parents=True)
    tl_data = {
        "duration_sec": 12.0,
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 3.0, "speech_start": 0.3},
            {"id": "scene_01", "type": "story", "start": 3.0, "end": 9.0, "speech_start": 3.3},
            {"id": "outro", "type": "outro", "start": 9.0, "end": 12.0, "speech_start": 9.3},
        ],
    }
    with open(tl_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(tl_data, f)

    with open(tl_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    create_silence_wav(audio_dir / "narration.wav", duration_sec=12.0)

    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    media_data = {
        "resolution": [1920, 1080],
        "fps": 30,
        "shots": [
            {"id": "s1", "scene_id": "intro", "file": "media/shots/s1.jpg", "type": "image", "start": 0.0, "end": 3.0},
            {"id": "s2", "scene_id": "scene_01", "file": "media/shots/s2.jpg", "type": "image", "start": 3.0, "end": 9.0},
            {"id": "s3", "scene_id": "outro", "file": "media/shots/s3.jpg", "type": "image", "start": 9.0, "end": 12.0},
        ],
    }
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump(media_data, f)

    agent = MotionAgent(run_dir=tmp_path)
    # Inject an invalid lower_third rect that collides with caption band (y=800, h=160 -> y+h=960 > 910)
    agent.style_config["layout_rectangles"] = {
        "lower_third": {"x": 100, "y": 800, "w": 1720, "h": 160},
        "caption_band": {"y": 910, "h": 120},
    }

    t0 = time.perf_counter()
    with pytest.raises(ValueError) as excinfo:
        agent.run()
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.5, f"Validation took too long: {elapsed:.2f}s"
    assert "Motion Agent layout validation failed" in str(excinfo.value)
    assert "scene_01_lower_third" in str(excinfo.value)

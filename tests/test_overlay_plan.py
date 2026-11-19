"""Unit tests for Motion Graphics Composition, Overlay Planning & Semantic Classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from src.agents.motion import MotionAgent
from src.utils.scene_classifier import classify_scene, SceneClassification
from src.utils.overlay_plan import (
    format_date_display,
    build_all_overlays,
    build_scene_overlays,
)
from src.utils.tts import create_silence_wav


def test_format_date_display() -> None:
    formatted = format_date_display("2026-11-15")
    assert "November" in formatted
    assert "2026" in formatted


def test_semantic_classifier_archetypes() -> None:
    # 1. Intro / Outro
    assert classify_scene({"type": "intro"}).recipe == "title_card"
    assert classify_scene({"type": "outro"}).recipe == "outro_card"

    # 2. Stat Callout
    stat_script = {
        "type": "story",
        "headline": "Gold price hits record high",
        "narration": "Gold price increased by Rs 1,200 per tola in the local market.",
    }
    sc = classify_scene(stat_script)
    assert sc.recipe == "stat_callout"
    assert "Rs 1,200" in sc.data.get("value", "")

    # 3. Quote Card
    quote_script = {
        "type": "story",
        "headline": "Energy treaty finalized",
        "narration": 'Managing Director Kul Man Ghising said "This power deal will transform Nepal." at the briefing.',
    }
    qc = classify_scene(quote_script)
    assert qc.recipe == "quote_card"
    assert "This power deal will transform Nepal" in qc.data.get("quote", "")
    assert "Kul Man Ghising" in qc.data.get("speaker", "")

    # 4. Headline Split (3 sentences)
    split_script = {
        "type": "story",
        "headline": "Three new trade corridors approved",
        "narration": "First corridor connects to border. Second corridor expands transit capacity. Third corridor facilitates cargo.",
    }
    hc = classify_scene(split_script, has_video=False)
    assert hc.recipe == "headline_split"
    assert len(hc.data.get("bullets", [])) >= 2

    # 5. Default story
    default_script = {
        "type": "story",
        "headline": "Weather update for Kathmandu",
        "narration": "Sunny and clear skies are expected throughout the valley.",
    }
    dc = classify_scene(default_script)
    assert dc.recipe == "cinematic_broll_full"
    assert dc.template_name == "lower_third.html"


def test_overlay_plan_and_composition_structure(tmp_path: Path) -> None:
    # Set up minimal run fixture
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "run_kind": "test",
        "origin": "fixture",
        "date": "2026-11-15",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Remittance surges past 1 trillion",
                "narration": "Nepal recorded remittance inflow exceeding Rs 1.2 trillion this quarter.",
                "sources": ["Nepal Rastra Bank"],
            },
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

    captions = [
        {"id": "c1", "scene_id": "intro", "start": 0.5, "end": 2.5, "text": "Welcome to daily briefing."},
        {"id": "c2", "scene_id": "scene_01", "start": 3.5, "end": 7.0, "text": "Remittance surges past 1 trillion."},
    ]
    with open(tl_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    create_silence_wav(audio_dir / "narration.wav", duration_sec=12.0)

    media_dir = tmp_path / "media"
    shots_dir = media_dir / "shots"
    shots_dir.mkdir(parents=True)
    dummy_img = shots_dir / "shot1.jpg"
    dummy_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00\xFF\xC0\x00\x11\x08\x00\x0A\x00\x0A\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x1F\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0B\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xBF\x00\xFF\xD9")

    media_data = {
        "resolution": [1920, 1080],
        "fps": 30,
        "shots": [
            {"id": "s1", "scene_id": "intro", "file": "media/shots/shot1.jpg", "type": "image", "start": 0.0, "end": 3.0},
            {"id": "s2", "scene_id": "scene_01", "file": "media/shots/shot1.jpg", "type": "image", "start": 3.0, "end": 9.0},
            {"id": "s3", "scene_id": "outro", "file": "media/shots/shot1.jpg", "type": "image", "start": 9.0, "end": 12.0},
        ],
    }
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump(media_data, f)

    agent = MotionAgent(run_dir=tmp_path)
    outputs = agent.run()

    # 1. Output files verify (HyperFrames retired; outputs are plan, captions.ass, and PNG overlays)
    assert "motion/render_plan.json" in outputs
    assert "motion/captions.ass" in outputs
    assert "motion/hf_project/index.html" not in outputs

    # 2. Render plan contains declarative blueprint
    with open(tmp_path / "motion" / "render_plan.json", "r", encoding="utf-8") as f:
        plan = json.load(f)
    assert "hf_project" not in plan
    assert "master_composition" not in plan
    assert plan["scenes"][1]["shot_recipe"] == "stat_callout"
    assert len(plan["scenes"]) == 3

    # 3. ASS subtitles generated
    ass_path = tmp_path / "motion" / "captions.ass"
    assert ass_path.exists()
    ass_content = ass_path.read_text(encoding="utf-8")
    assert "[Script Info]" in ass_content
    assert "PlayResX: 1920" in ass_content
    assert "Remittance surges past 1 trillion" in ass_content


def test_overlay_vertical_composition(tmp_path: Path) -> None:
    # Set up minimal run fixture for vertical 9:16
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "run_kind": "test",
        "origin": "fixture",
        "date": "2026-11-15",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Remittance surges past 1 trillion",
                "narration": "Nepal recorded remittance inflow exceeding Rs 1.2 trillion this quarter.",
                "sources": ["Nepal Rastra Bank"],
            },
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

    captions = [
        {"id": "c1", "scene_id": "intro", "start": 0.5, "end": 2.5, "text": "Welcome to vertical news."},
    ]
    with open(tl_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    create_silence_wav(audio_dir / "narration.wav", duration_sec=12.0)

    media_dir = tmp_path / "media"
    shots_dir = media_dir / "shots"
    shots_dir.mkdir(parents=True)
    dummy_img = shots_dir / "shot1.jpg"
    dummy_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00\xFF\xC0\x00\x11\x08\x00\x0A\x00\x0A\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01\xFF\xC4\x00\x1F\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0B\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xBF\x00\xFF\xD9")

    # Media manifest with vertical orientation
    media_data = {
        "orientation": "vertical",
        "aspect_ratio": "9:16",
        "resolution": [1080, 1920],
        "fps": 30,
        "shots": [
            {"id": "s1", "scene_id": "intro", "file": "media/shots/shot1.jpg", "type": "image", "start": 0.0, "end": 3.0},
            {"id": "s2", "scene_id": "scene_01", "file": "media/shots/shot1.jpg", "type": "image", "start": 3.0, "end": 9.0},
            {"id": "s3", "scene_id": "outro", "file": "media/shots/shot1.jpg", "type": "image", "start": 9.0, "end": 12.0},
        ],
    }
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump(media_data, f)

    agent = MotionAgent(run_dir=tmp_path)
    outputs = agent.run()

    # 1. Output files verify
    assert "motion/render_plan.json" in outputs
    assert "motion/captions.ass" in outputs
    assert "motion/hf_project/index.html" not in outputs

    # 2. Render plan resolution must be 1080x1920
    with open(tmp_path / "motion" / "render_plan.json", "r", encoding="utf-8") as f:
        plan = json.load(f)
    assert plan["resolution"] == "1080x1920"

    # 3. Captions ASS must have PlayResX: 1080 and PlayResY: 1920
    ass_text = (tmp_path / "motion" / "captions.ass").read_text(encoding="utf-8")
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text

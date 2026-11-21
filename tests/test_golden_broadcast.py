"""Golden broadcast regression gate.

Validates that golden bilingual (Nepali + English) run fixtures adhere strictly
to JSON Schema contracts and invariant quality criteria (audio/video sync, safe margins,
Devanagari text preservation, and loudness compliance).
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils.schema_validate import validate_contract
from src.utils.text_normalizer import contains_devanagari

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden_broadcast"


def test_golden_timeline_schema_and_invariants() -> None:
    tl_path = GOLDEN_DIR / "timeline.json"
    assert tl_path.exists(), "golden timeline.json fixture missing"
    data = json.loads(tl_path.read_text(encoding="utf-8"))

    # 1. Schema Validation
    is_valid, err = validate_contract(data, "contracts/timeline.schema.json")
    assert is_valid, f"Golden timeline failed schema validation: {err}"

    # 2. Structural & Bilingual Invariants
    scenes = data["scenes"]
    assert len(scenes) >= 3, "Golden timeline must contain intro, stories, and outro"
    assert scenes[0]["id"] == "intro"
    assert scenes[-1]["id"] == "outro"

    has_nepali = False
    has_english = False

    last_end = 0.0
    for sc in scenes:
        start = float(sc["start"])
        end = float(sc["end"])
        assert start >= last_end - 0.01, f"Scene {sc['id']} start time overlaps previous scene"
        assert end > start, f"Scene {sc['id']} has non-positive duration"
        last_end = end

        # Speech bounds
        assert sc["speech_start"] >= start
        assert sc["speech_end"] <= end

        # Word timing checks
        words = sc.get("words", [])
        assert len(words) > 0
        w_last = start
        for w in words:
            assert w["start"] >= start - 0.05
            assert w["end"] <= end + 0.05
            assert w["end"] >= w["start"]
            if contains_devanagari(w["w"]):
                has_nepali = True
            else:
                has_english = True

    assert has_nepali is True, "Golden timeline must contain Devanagari text"
    assert has_english is True, "Golden timeline must contain English text"


def test_golden_render_plan_schema_and_invariants() -> None:
    plan_path = GOLDEN_DIR / "render_plan.json"
    assert plan_path.exists(), "golden render_plan.json fixture missing"
    data = json.loads(plan_path.read_text(encoding="utf-8"))

    # 1. Schema Validation
    is_valid, err = validate_contract(data, "contracts/render_plan.schema.json")
    assert is_valid, f"Golden render_plan failed schema validation: {err}"

    # 2. Geometry & Safe Margin Invariants
    res = data["resolution"]
    w_str, h_str = res.split("x")
    w_res, h_res = int(w_str), int(h_str)
    assert (w_res, h_res) in [(1920, 1080), (1080, 1920)]

    for sc in data["scenes"]:
        assert sc["caption_style"]["position"] in ("bottom", "middle", "top")
        assert sc["transition"]["duration_sec"] >= 0.0
        # Overlays safe zone (must reside inside canvas bounds)
        for ov in sc.get("overlays", []):
            assert ov["x"] >= 0
            assert ov["y"] >= 0
            assert ov["x"] + ov["w"] <= w_res
            assert ov["y"] + ov["h"] <= h_res


def test_golden_qa_report_schema_and_invariants() -> None:
    qa_path = GOLDEN_DIR / "qa_report.json"
    assert qa_path.exists(), "golden qa_report.json fixture missing"
    data = json.loads(qa_path.read_text(encoding="utf-8"))

    # 1. Schema Validation
    is_valid, err = validate_contract(data, "contracts/qa_report.schema.json")
    assert is_valid, f"Golden qa_report failed schema validation: {err}"

    # 2. QA Invariants
    assert data["passed"] is True
    assert data["redo_stage"] is None
    checks = {c["name"]: c["passed"] for c in data["checks"]}
    assert checks.get("final_mp4_streams") is True
    assert checks.get("loudness") is True
    assert checks.get("captions") is True

    # 3. Loudness compliance
    metrics = data["metrics"]
    lufs = metrics.get("integrated_loudness_lufs")
    tp = metrics.get("true_peak_db")
    assert lufs is not None and abs(lufs - (-16.0)) <= 1.5
    assert tp is not None and tp <= -1.0

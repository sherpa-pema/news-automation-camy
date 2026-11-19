"""Unit tests for shot motion resolution and deterministic variety."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.utils.overlay_plan import resolve_shot_motion


def test_resolve_shot_motion_featured_card_suppresses_card_zoom() -> None:
    card_shot = {"type": "card", "motion": "zoom_in"}
    # When featured card is scheduled, card motion must be suppressed to avoid zooming text
    m = resolve_shot_motion(card_shot, index=0, featured_card=True)
    assert m == "none"

    # When no featured card, card shot can zoom
    m_no_feat = resolve_shot_motion(card_shot, index=0, featured_card=False)
    assert m_no_feat == "zoom_in"


def test_resolve_shot_motion_stills_variety() -> None:
    still_shot = {"type": "image"}
    # Even index -> zoom_in, Odd index -> zoom_out
    assert resolve_shot_motion(still_shot, index=0) == "zoom_in"
    assert resolve_shot_motion(still_shot, index=1) == "zoom_out"
    assert resolve_shot_motion(still_shot, index=2) == "zoom_in"
    assert resolve_shot_motion(still_shot, index=3) == "zoom_out"


def test_resolve_shot_motion_pass_through_explicit() -> None:
    assert resolve_shot_motion({"type": "image", "motion": "ken_burns"}, index=0) == "ken_burns"
    assert resolve_shot_motion({"type": "image", "motion": "none"}, index=0) == "none"


def test_resolve_shot_motion_video_is_none() -> None:
    assert resolve_shot_motion({"type": "video"}, index=0) is None
    assert resolve_shot_motion({"type": "video"}, index=1) is None


def test_golden_run_has_motion_variety() -> None:
    """Verifies that 2026-11-15 golden run plan contains motion variety and no consecutive > 2."""
    plan_path = Path("tests/fixtures/golden_render_plan.json")
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    all_motions = []
    for sc in plan.get("scenes", []):
        for sh in sc.get("shots", []):
            if sh.get("type") in ("image", "card"):
                all_motions.append(sh.get("motion"))

    # Must contain at least two distinct motion values
    distinct_motions = set(all_motions)
    assert len(distinct_motions) >= 2, f"Expected at least 2 distinct motions, got {distinct_motions}"
    assert "zoom_in" in distinct_motions
    assert "zoom_out" in distinct_motions

    # Verify no run of identical motion > 2 across multi-shot scenes
    for sc in plan.get("scenes", []):
        sc_shots = sc.get("shots", [])
        if len(sc_shots) >= 2:
            sc_motions = [s.get("motion") for s in sc_shots if s.get("type") in ("image", "card")]
            consecutive = 1
            for i in range(1, len(sc_motions)):
                if sc_motions[i] == sc_motions[i - 1]:
                    consecutive += 1
                    assert consecutive <= 2, f"Scene {sc['id']} has >2 consecutive identical motions: {sc_motions}"
                else:
                    consecutive = 1

"""Golden regression tests for Motion Agent using real 2026-11-15 run data.

Validates that real-world pipeline inputs produce a valid render_plan.json:
- All shot files referenced by scenes exist on disk and are non-empty.
- All overlay PNGs referenced in the plan exist on disk.
- Story lower_third source credits match script source and do NOT fall back to channel name.
- Output matches the golden snapshot (normalized).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Any
import pytest

from src.agents.motion import MotionAgent


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Recursively normalizes a render plan for cross-platform snapshot comparison."""
    norm = json.loads(json.dumps(plan))

    def _normalize_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _normalize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_normalize_obj(v) for v in obj]
        elif isinstance(obj, str):
            # Normalize Windows backslashes in paths
            return obj.replace("\\", "/")
        return obj

    return _normalize_obj(norm)


@pytest.fixture
def golden_run(tmp_path: Path) -> dict[str, Any]:
    """Sets up a temp run directory with 2026-11-15 inputs and executes MotionAgent."""
    src_run = Path("data/runs/2026-11-15")
    if not src_run.exists():
        pytest.skip(f"Source golden run dir {src_run} does not exist (gitignored runtime artifact)")

    # Copy input directories
    shutil.copytree(src_run / "script", tmp_path / "script")
    shutil.copytree(src_run / "timeline", tmp_path / "timeline")
    shutil.copytree(src_run / "media", tmp_path / "media")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    shutil.copy(src_run / "audio" / "narration.wav", tmp_path / "audio" / "narration.wav")

    if (src_run / "manifest.json").exists():
        shutil.copy(src_run / "manifest.json", tmp_path / "manifest.json")

    # Clean up file handlers on logger before and after run
    def _cleanup_logger():
        for h in logging.getLogger("motion_agent").handlers[:]:
            if isinstance(h, logging.FileHandler):
                h.close()
                logging.getLogger("motion_agent").removeHandler(h)

    _cleanup_logger()

    agent = MotionAgent(run_dir=tmp_path)
    outputs = agent.run()

    _cleanup_logger()

    plan_path = tmp_path / "motion" / "render_plan.json"
    assert plan_path.exists(), "motion/render_plan.json was not generated"

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    with open(tmp_path / "script" / "script.json", "r", encoding="utf-8") as f:
        script = json.load(f)

    return {
        "run_dir": tmp_path,
        "agent": agent,
        "plan": plan,
        "script": script,
        "outputs": outputs,
    }


def test_golden_shot_and_overlay_files_exist(golden_run: dict[str, Any]) -> None:
    """Asserts that every shot file and overlay PNG exists on disk."""
    run_dir: Path = golden_run["run_dir"]
    plan: dict[str, Any] = golden_run["plan"]

    scenes = plan.get("scenes", [])
    assert len(scenes) == 21, f"Expected 21 scenes, got {len(scenes)}"

    # 1. Assert shot files exist
    total_shots = 0
    for sc in scenes:
        shots = sc.get("shots", [])
        assert len(shots) > 0, f"Scene {sc['id']} has no shots"
        for sh in shots:
            total_shots += 1
            sh_file = sh.get("file")
            assert sh_file, f"Scene {sc['id']} shot has empty file path"
            full_path = run_dir / sh_file
            assert full_path.is_file(), f"Shot file does not exist: {sh_file}"

    assert total_shots == 30, f"Expected 30 total shots, got {total_shots}"

    # 2. Assert overlay PNG files exist
    overlays = plan.get("overlays", [])
    assert len(overlays) > 0, "No overlays found in render_plan"
    for ov in overlays:
        png_path = ov.get("png")
        assert png_path, f"Overlay {ov.get('id')} has empty png field"
        full_png = run_dir / png_path
        assert full_png.is_file(), f"Overlay PNG file does not exist: {png_path}"


def test_golden_story_source_attribution(golden_run: dict[str, Any]) -> None:
    """Asserts that story lower_third source credits match script source and are NOT channel fallback."""
    plan: dict[str, Any] = golden_run["plan"]
    script: dict[str, Any] = golden_run["script"]
    agent: MotionAgent = golden_run["agent"]

    channel_fallback = agent.channel_config.get("channel_name", "Nepal Daily Briefing")
    script_scenes = {s["id"]: s for s in script.get("scenes", [])}

    story_scenes = [sc for sc in plan.get("scenes", []) if sc.get("id", "").startswith("scene_")]
    assert len(story_scenes) == 19, f"Expected 19 story scenes, got {len(story_scenes)}"

    for sc in story_scenes:
        sc_id = sc["id"]
        sc_script = script_scenes.get(sc_id, {})
        script_source = sc_script.get("source")

        if script_source:
            credit = sc.get("lower_third", {}).get("source_credit", "")
            # Must NOT fall back to channel name
            assert credit != channel_fallback, (
                f"Scene {sc_id} lower_third.source_credit fell back to channel '{channel_fallback}' "
                f"instead of using script source '{script_source}'"
            )
            # Must contain or match the script source outlets
            outlets = [o.strip() for o in script_source.split(",")]
            matches = any(outlet in credit for outlet in outlets)
            assert matches, (
                f"Scene {sc_id} lower_third.source_credit '{credit}' does not reflect "
                f"script source '{script_source}'"
            )


def test_golden_render_plan_snapshot(golden_run: dict[str, Any]) -> None:
    """Asserts that generated render_plan.json matches the golden snapshot."""
    plan: dict[str, Any] = golden_run["plan"]
    snapshot_path = Path("tests/fixtures/golden_render_plan.json")
    assert snapshot_path.exists(), "Snapshot fixture tests/fixtures/golden_render_plan.json missing"

    with open(snapshot_path, "r", encoding="utf-8") as f:
        expected_plan = json.load(f)

    norm_actual = normalize_plan(plan)
    norm_expected = normalize_plan(expected_plan)

    assert norm_actual == norm_expected, "Render plan does not match golden snapshot"

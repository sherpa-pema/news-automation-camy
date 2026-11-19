"""Unit tests for canonical render_plan persistence and sync."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import pytest

from src.agents.motion import (
    MotionAgent,
    sync_render_plan_data,
    sync_render_plan_with_media_manifest,
)


def test_sync_render_plan_persists_to_disk_and_matches_in_memory(tmp_path: Path) -> None:
    available_runs = sorted(
        [d for d in Path("data/runs").iterdir() if d.is_dir() and (d / "script").exists() and (d / "timeline").exists()],
        reverse=True,
    )
    if not available_runs:
        pytest.skip("No runs available in data/runs")
    src_run = available_runs[0]

    shutil.copytree(src_run / "script", tmp_path / "script")
    shutil.copytree(src_run / "timeline", tmp_path / "timeline")
    shutil.copytree(src_run / "media", tmp_path / "media")
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    shutil.copy(src_run / "audio" / "narration.wav", tmp_path / "audio" / "narration.wav")

    agent = MotionAgent(run_dir=tmp_path)
    agent.run()

    plan_path = tmp_path / "motion" / "render_plan.json"
    assert plan_path.exists()
    original_plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    # 1. Mutate media_manifest.json (simulate user visual picking in web office)
    manifest_path = tmp_path / "media" / "media_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Change shot in scene_01 to a stock image
    for sh in manifest_data["shots"]:
        if sh["scene_id"] == "scene_01":
            sh["file"] = "media/shots/custom_pick.jpg"
            sh["source"] = "wikimedia_commons"
            sh["author"] = "Contributor"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 2. Run sync
    res = sync_render_plan_with_media_manifest(tmp_path)
    assert res is True

    # 3. Assert persisted plan matches in-memory and changed
    new_plan_bytes = plan_path.read_bytes()
    new_plan_hash = hashlib.sha256(new_plan_bytes).hexdigest()
    assert new_plan_hash != original_plan_hash, "Render plan was not updated after manifest mutation"

    persisted_plan = json.loads(new_plan_bytes.decode("utf-8"))
    sc1 = next(s for s in persisted_plan["scenes"] if s["id"] == "scene_01")
    assert sc1["shots"][0]["file"] == "media/shots/custom_pick.jpg"
    assert sc1["shots"][0]["source"] == "wikimedia_commons"

    # Illustrative tag should have been scheduled because wikimedia_commons is stock
    ov_types = [ov["type"] for ov in sc1.get("overlays", [])]
    assert "illustrative_tag" in ov_types

    # Re-running sync must be idempotent (no drift)
    res_second = sync_render_plan_with_media_manifest(tmp_path)
    assert res_second is True
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == new_plan_hash

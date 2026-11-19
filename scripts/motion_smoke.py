"""Smoke test and diagnostic helper for Motion Agent.

Runs MotionAgent on a specified run directory (or latest) and prints
scene, shot, overlay, and credit counts, along with schema and layout validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.agents.motion import MotionAgent
from src.utils.schema_validate import validate_contract


def run_motion_smoke(run_dir: Path, execute: bool = True) -> int:
    """Runs motion smoke verification on run_dir.

    Args:
        run_dir: Path to the run directory (e.g. data/runs/2026-11-15).
        execute: If True, executes MotionAgent.run() before inspecting.

    Returns:
        0 if validation passes, non-zero on failure.
    """
    print(f"=== Motion Smoke: {run_dir} ===")
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    # Cleanup any existing file handlers on motion_agent logger to avoid file locks
    for h in logging.getLogger("motion_agent").handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logging.getLogger("motion_agent").removeHandler(h)

    agent = MotionAgent(run_dir=run_dir)

    if execute:
        print("Running MotionAgent.run()...")
        try:
            outputs = agent.run()
            print(f"Generated {len(outputs)} output files.")
        except Exception as e:
            print(f"ERROR during MotionAgent.run(): {e}", file=sys.stderr)
            return 2

    # Load plan
    plan_path = run_dir / "motion" / "render_plan.json"
    if not plan_path.exists():
        plan_path = run_dir / "render_plan.json"
    if not plan_path.exists():
        print(f"ERROR: render_plan.json not found in {run_dir}", file=sys.stderr)
        return 1

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    # 1. Schema Validation
    schema_path = "contracts/render_plan.schema.json"
    valid_schema, schema_errs = validate_contract(plan, schema_path)
    print(f"\n--- Contract Validation ({schema_path}) ---")
    if valid_schema:
        print("  Status: PASS")
    else:
        print(f"  Status: FAIL -> {schema_errs}")

    # 2. Plan Counts & Breakdown
    scenes = plan.get("scenes", [])
    overlays = plan.get("overlays", [])
    duration = plan.get("duration_sec", plan.get("duration", 0.0))
    res = plan.get("resolution", "unknown")
    fps = plan.get("fps", 30)

    print(f"\n--- Plan Overview ---")
    print(f"  Resolution: {res} @ {fps} fps")
    print(f"  Duration: {duration:.1f} s")
    print(f"  Total scenes: {len(scenes)}")
    print(f"  Total overlays: {len(overlays)}")

    # Scene recipe breakdown
    recipes = Counter(sc.get("shot_recipe", "unclassified") for sc in scenes)
    print(f"\n--- Scene Recipes ---")
    for r, count in recipes.most_common():
        print(f"  {r}: {count}")

    # Shots & Motion breakdown
    total_shots = 0
    motion_types = Counter()
    shot_types = Counter()
    missing_shot_files = []

    for sc in scenes:
        sc_shots = sc.get("shots", [])
        total_shots += len(sc_shots)
        for sh in sc_shots:
            m = sh.get("motion") or "none"
            motion_types[m] += 1
            t = sh.get("type", "unknown")
            shot_types[t] += 1
            sh_file = sh.get("file", "")
            if not sh_file or not (run_dir / sh_file).exists():
                missing_shot_files.append((sc["id"], sh_file))

    print(f"\n--- Shots Breakdown (Total: {total_shots}) ---")
    print(f"  Types: {dict(shot_types)}")
    print(f"  Motions: {dict(motion_types)}")
    if missing_shot_files:
        print(f"  WARNING: {len(missing_shot_files)} shot files missing on disk:")
        for sc_id, f in missing_shot_files[:5]:
            print(f"    - [{sc_id}] {f}")
    else:
        print("  All shot files exist on disk: YES")

    # Overlays breakdown
    overlay_types = Counter(ov.get("type", "unknown") for ov in overlays)
    missing_overlay_files = []
    for ov in overlays:
        png = ov.get("png", "")
        if not png or not (run_dir / png).exists():
            missing_overlay_files.append(png)

    print(f"\n--- Overlays Breakdown ---")
    for ot, count in overlay_types.most_common():
        print(f"  {ot}: {count}")
    if missing_overlay_files:
        print(f"  WARNING: {len(missing_overlay_files)} overlay PNGs missing on disk:")
        for p in missing_overlay_files[:5]:
            print(f"    - {p}")
    else:
        print("  All overlay PNG files exist on disk: YES")

    # 3. Source Attribution Check
    script_path = run_dir / "script" / "script.json"
    if not script_path.exists():
        script_path = run_dir / "script.json"

    fallback_channel = agent.channel_config.get("channel_name", "Nepal Daily Briefing")
    story_credits = []
    channel_fallback_count = 0
    correct_credit_count = 0

    if script_path.exists():
        with open(script_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)
        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}

        for sc in scenes:
            sc_id = sc["id"]
            sc_script = script_scenes.get(sc_id, {})
            if sc_script.get("type") == "story":
                script_source = sc_script.get("source")
                credit = sc.get("lower_third", {}).get("source_credit", "")
                is_fallback = (credit == fallback_channel) or ("Verified Outlets" in credit)
                if is_fallback:
                    channel_fallback_count += 1
                else:
                    correct_credit_count += 1
                story_credits.append({
                    "id": sc_id,
                    "script_source": script_source,
                    "credit": credit,
                    "is_fallback": is_fallback,
                })

        print(f"\n--- Source Attribution ---")
        print(f"  Stories with script source: {len(story_credits)}")
        print(f"  Channel / fallback credits: {channel_fallback_count}")
        print(f"  Distinct script-derived credits: {correct_credit_count}")
        if channel_fallback_count > 0:
            print("  Sample fallbacks:")
            for item in story_credits[:3]:
                if item["is_fallback"]:
                    print(f"    - [{item['id']}] script.source='{item['script_source']}' -> credit='{item['credit']}'")

    print("\n=== Summary ===")
    status = "PASS" if valid_schema and not missing_shot_files and not missing_overlay_files else "FAIL"
    print(f"Validation: {status}")
    if channel_fallback_count > 0:
        print(f"Attribution Notice: {channel_fallback_count} stories using channel fallback (Target for Phase 1).")

    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion Agent Smoke Test & Verification")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to run dir (defaults to data/runs/2026-11-15 or data/latest)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date of run dir to test (e.g. 2026-11-15)",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="Skip executing MotionAgent.run() and only inspect existing plan",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if args.date:
            run_dir = Path(f"data/runs/{args.date}")
        elif Path("data/runs/2026-11-15").exists():
            run_dir = Path("data/runs/2026-11-15")
        elif Path("data/latest").exists():
            run_dir = Path("data/latest")
        else:
            print("ERROR: No run directory specified or found.", file=sys.stderr)
            return 1

    return run_motion_smoke(run_dir, execute=not args.no_exec)


if __name__ == "__main__":
    sys.exit(main())

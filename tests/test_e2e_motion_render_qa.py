"""End-to-end integration test: MotionAgent -> RenderAgent -> QAAgent
for both landscape (16:9) and vertical (9:16) configurations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from src.agents.motion import MotionAgent
from src.agents.render import RenderAgent
from src.agents.qa import QAAgent
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)


def _setup_pipeline_environment(
    base_dir: Path,
    orientation: str = "landscape",
    duration_sec: float = 6.0,
) -> Path:
    """Sets up a complete 3-scene run directory ready for MotionAgent."""
    is_vert = orientation == "vertical"
    res = (1080, 1920) if is_vert else (1920, 1080)

    # 1. Audio: 48kHz stereo WAV with compliant -18 LUFS loudness
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    narration_wav = audio_dir / "narration.wav"
    cmd_audio = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_sec}",
        "-af", "loudnorm=I=-18:TP=-2:LRA=7",
        "-ar", "48000", "-ac", "2",
        str(narration_wav),
    ]
    subprocess.run(cmd_audio, capture_output=True, check=True)

    voice_report = {
        "engine": "piper",
        "total_duration_sec": duration_sec,
        "loudness_after_lufs": -18.0,
        "true_peak_after_db": -2.0,
        "loudnorm": {"target_lufs": -18.0},
    }
    (audio_dir / "voice_report.json").write_text(json.dumps(voice_report), encoding="utf-8")

    # 2. Script
    script_dir = base_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_data = {
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-20",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Kathmandu Daily News Briefing",
                "narration": "Welcome to today's news.",
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Historic Solar Energy Project Launched in Pokhara",
                "narration": "Solar power expands in Pokhara.",
                "source": "Kathmandu Post",
                "sources": ["Kathmandu Post"],
            },
            {
                "id": "outro",
                "type": "outro",
                "headline": "Summary & Credits",
                "narration": "Thank you for watching.",
            },
        ],
    }
    script_bytes = json.dumps(script_data, indent=2).encode("utf-8")
    script_file = script_dir / "script.json"
    script_file.write_bytes(script_bytes)
    script_hash = hashlib.sha256(script_bytes).hexdigest()

    curr_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
    approval_data = {
        "approved_at": "2026-11-20T08:00:00Z",
        "script_sha256": script_hash,
        "approved_by": "editor",
        "validator_fingerprint": curr_fp,
    }
    (script_dir / "approval.json").write_text(json.dumps(approval_data), encoding="utf-8")
    (script_dir / "validation_report.json").write_text(
        json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": curr_fp}),
        encoding="utf-8",
    )

    # 3. Timeline & Captions
    tl_dir = base_dir / "timeline"
    tl_dir.mkdir(parents=True, exist_ok=True)
    timeline_data = {
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0, "duration": 2.0},
            {"id": "scene_01", "type": "story", "start": 2.0, "end": 4.0, "duration": 2.0},
            {"id": "outro", "type": "outro", "start": 4.0, "end": 6.0, "duration": 2.0},
        ],
        "total_duration": duration_sec,
    }
    (tl_dir / "timeline.json").write_text(json.dumps(timeline_data), encoding="utf-8")

    captions_data = [
        {"id": "cap_01", "start": 0.1, "end": 1.9, "text": "Welcome to today's news."},
        {"id": "cap_02", "start": 2.1, "end": 3.9, "text": "Solar power expands in Pokhara."},
        {"id": "cap_03", "start": 4.1, "end": 5.9, "text": "Thank you for watching."},
    ]
    (tl_dir / "captions.json").write_text(json.dumps(captions_data), encoding="utf-8")

    # 4. Media & Shots
    media_dir = base_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    shots_dir = media_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # Create solid color shot images
    Image.new("RGB", res, color=(20, 45, 85)).save(str(shots_dir / "shot_intro.jpg"))
    Image.new("RGB", res, color=(35, 75, 40)).save(str(shots_dir / "shot_story.jpg"))
    Image.new("RGB", res, color=(80, 40, 20)).save(str(shots_dir / "shot_outro.jpg"))

    media_data = {
        "orientation": orientation,
        "aspect_ratio": "9:16" if is_vert else "16:9",
        "resolution": list(res),
        "fps": 30,
        "shots": [
            {
                "scene_id": "intro",
                "shot_index": 0,
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "type": "image",
                "file": "media/shots/shot_intro.jpg",
                "source": "local_library",
                "attribution_required": False,
            },
            {
                "scene_id": "scene_01",
                "shot_index": 0,
                "start": 2.0,
                "end": 4.0,
                "duration": 2.0,
                "type": "image",
                "file": "media/shots/shot_story.jpg",
                "source": "Kathmandu Post",
                "author": "Pokhara Staff",
                "attribution_required": False,
            },
            {
                "scene_id": "outro",
                "shot_index": 0,
                "start": 4.0,
                "end": 6.0,
                "duration": 2.0,
                "type": "image",
                "file": "media/shots/shot_outro.jpg",
                "source": "local_library",
                "attribution_required": False,
            },
        ],
    }
    (media_dir / "media_manifest.json").write_text(json.dumps(media_data), encoding="utf-8")

    # 5. Manifest
    manifest = {
        "date": "2026-11-20",
        "run_kind": "test",
        "orientation": orientation,
        "stages": {},
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base_dir


def test_e2e_motion_render_qa_landscape(tmp_path: Path) -> None:
    """Runs real MotionAgent -> RenderAgent -> QAAgent on 16:9 landscape format."""
    run_dir = tmp_path / "run_landscape"
    _setup_pipeline_environment(run_dir, orientation="landscape", duration_sec=6.0)

    # 1. MotionAgent
    motion_agent = MotionAgent(run_dir=run_dir)
    motion_outputs = motion_agent.run()
    assert (run_dir / "motion" / "render_plan.json").exists()
    assert (run_dir / "motion" / "captions.ass").exists()

    # 2. RenderAgent
    render_agent = RenderAgent(run_dir=run_dir)
    render_outputs = render_agent.run()
    final_mp4 = run_dir / "render" / "final.mp4"
    assert final_mp4.exists()

    # 3. QAAgent
    qa_agent = QAAgent(run_dir=run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    qa_outputs = qa_agent.run(manifest=manifest)

    # 4. Assertions on QA report
    qa_report_file = run_dir / "qa" / "qa_report.json"
    assert qa_report_file.exists()
    qa_report = json.loads(qa_report_file.read_text(encoding="utf-8"))
    assert qa_report["passed"] is True, f"QA failed on checks: {[c for c in qa_report['checks'] if not c['passed']]}"
    assert qa_report["redo_stage"] is None

    # Verify specific metrics
    metrics = qa_report["metrics"]
    assert metrics.get("duration_diff_ms", 999.0) <= 50.0
    assert metrics.get("subtitle_burn_stddev", 0.0) >= 0.5


def test_e2e_motion_render_qa_vertical(tmp_path: Path) -> None:
    """Runs real MotionAgent -> RenderAgent -> QAAgent on 9:16 vertical format."""
    run_dir = tmp_path / "run_vertical"
    _setup_pipeline_environment(run_dir, orientation="vertical", duration_sec=6.0)

    # 1. MotionAgent
    motion_agent = MotionAgent(run_dir=run_dir)
    motion_outputs = motion_agent.run()
    assert (run_dir / "motion" / "render_plan.json").exists()
    assert (run_dir / "motion" / "captions.ass").exists()

    # 2. RenderAgent
    render_agent = RenderAgent(run_dir=run_dir)
    render_outputs = render_agent.run()
    final_mp4 = run_dir / "render" / "final.mp4"
    assert final_mp4.exists()

    # 3. QAAgent
    qa_agent = QAAgent(run_dir=run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    qa_outputs = qa_agent.run(manifest=manifest)

    # 4. Assertions on QA report
    qa_report_file = run_dir / "qa" / "qa_report.json"
    assert qa_report_file.exists()
    qa_report = json.loads(qa_report_file.read_text(encoding="utf-8"))
    assert qa_report["passed"] is True, f"QA failed on checks: {[c for c in qa_report['checks'] if not c['passed']]}"
    assert qa_report["redo_stage"] is None

    # Verify specific metrics
    metrics = qa_report["metrics"]
    assert metrics.get("duration_diff_ms", 999.0) <= 50.0
    assert metrics.get("subtitle_burn_stddev", 0.0) >= 0.5

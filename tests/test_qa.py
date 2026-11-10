"""Unit tests for QA Agent and deterministic quality gates."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from src.agents.qa import QAAgent
from src.utils.schema_validate import validate_contract
from src.utils.tts import create_silence_wav


def _generate_synthetic_video(
    output_path: Path,
    audio_path: Path | None = None,
    duration_sec: float = 2.0,
    with_audio: bool = True,
    color: str = "blue",
) -> None:
    """Generates a synthetic MP4 video matching broadcast specs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=1920x1080:r=30:d={duration_sec}",
    ]
    if with_audio:
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path)])
        else:
            cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={duration_sec}"])
        cmd.extend(["-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-pix_fmt", "yuv420p"])
    else:
        cmd.extend(["-c:v", "libx264", "-an", "-pix_fmt", "yuv420p"])

    cmd.append(str(output_path))
    subprocess.run(cmd, capture_output=True, check=True)


@pytest.fixture
def base_qa_fixture(tmp_path: Path) -> Path:
    """Sets up a complete passing run fixture for QA."""
    # Script
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "origin": "fixture",
        "date": "2026-11-08",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {"id": "scene_01", "type": "story", "headline": "Story 1", "source": "MyRepublica"},
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    # Audio
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    narration_wav = audio_dir / "narration.wav"
    create_silence_wav(narration_wav, duration_sec=2.0, sample_rate=48000)

    # Timeline & Captions
    tl_dir = tmp_path / "timeline"
    tl_dir.mkdir(parents=True)
    with open(tl_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump({"scenes": [{"id": "intro", "start": 0.0, "end": 1.0}, {"id": "scene_01", "start": 1.0, "end": 2.0}]}, f)

    with open(tl_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump([{"id": "c1", "scene_id": "intro", "start": 0.1, "end": 0.9, "text": "Intro"}], f)

    # Media manifest
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"shots": []}, f)

    # Motion plan & ASS
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir(parents=True)
    plan_data = {
        "resolution": "1920x1080",
        "fps": 30,
        "duration_sec": 2.0,
        "scenes": [
            {"id": "intro", "start": 0.0, "end": 1.0, "overlays": []},
            {
                "id": "scene_01",
                "start": 1.0,
                "end": 2.0,
                "lower_third": {"headline": "Story 1", "source_credit": "MyRepublica"},
                "overlays": [{"id": "lt_1", "type": "lower_third", "x": 96, "y": 730, "w": 1320, "h": 155}],
            },
        ],
        "overlays": [{"id": "lt_1", "type": "lower_third", "x": 96, "y": 730, "w": 1320, "h": 155}],
    }
    with open(motion_dir / "render_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan_data, f)

    (motion_dir / "captions.ass").write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n[Events]\nDialogue: 0,0:00:00.10,0:00:00.90,Default,Intro\n",
        encoding="utf-8",
    )

    # Render final.mp4 and description.md
    render_dir = tmp_path / "render"
    render_dir.mkdir(parents=True)
    _generate_synthetic_video(render_dir / "final.mp4", audio_path=narration_wav, duration_sec=2.0, with_audio=True)

    desc_content = (
        "# Daily News\n\n## Chapters\n00:00 - Intro\n00:01 - Story\n00:02 - Outro\n\n"
        "## Disclaimers & Disclosures\n- Narration audio was synthesized using an AI voice model.\n"
    )
    (render_dir / "description.md").write_text(desc_content, encoding="utf-8")

    return tmp_path


def test_qa_agent_passes_on_valid_fixture(base_qa_fixture: Path) -> None:
    # Set voice_report matching silence/anullsrc (measured around -70 LUFS)
    # Or test with measured loudness from silence
    audio_dir = base_qa_fixture / "audio"
    voice_report = {
        "engine": "fake",
        "total_duration_sec": 2.0,
        "loudness_after_lufs": -70.0,
        "true_peak_after_db": -70.0,
    }
    with open(audio_dir / "voice_report.json", "w", encoding="utf-8") as f:
        json.dump(voice_report, f)

    agent = QAAgent(run_dir=base_qa_fixture)
    agent.run()

    rep_path = base_qa_fixture / "qa" / "qa_report.json"
    assert rep_path.exists()

    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["passed"] is True
    # Validate schema
    is_valid, errs = validate_contract(rep, "contracts/qa_report.schema.json")
    assert is_valid, f"QA report failed schema validation: {errs}"


def test_qa_catches_duration_mismatch(base_qa_fixture: Path) -> None:
    # Audio is 2.0s, but generate video of 2.3s (> 50ms difference)
    render_dir = base_qa_fixture / "render"
    _generate_synthetic_video(render_dir / "final.mp4", duration_sec=2.3, with_audio=True)

    agent = QAAgent(run_dir=base_qa_fixture)
    with pytest.raises(ValueError):
        agent.run()

    with open(base_qa_fixture / "qa" / "qa_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["passed"] is False
    assert rep["redo_stage"] == "render"
    stream_check = next(c for c in rep["checks"] if c["name"] == "final_mp4_streams")
    assert stream_check["passed"] is False
    assert "Duration sync mismatch" in stream_check["detail"] or "diff" in stream_check["detail"]


def test_qa_catches_missing_audio_stream(base_qa_fixture: Path) -> None:
    # Video without audio stream
    render_dir = base_qa_fixture / "render"
    _generate_synthetic_video(render_dir / "final.mp4", duration_sec=2.0, with_audio=False)

    agent = QAAgent(run_dir=base_qa_fixture)
    with pytest.raises(ValueError):
        agent.run()

    with open(base_qa_fixture / "qa" / "qa_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["passed"] is False
    assert rep["redo_stage"] == "render"
    stream_check = next(c for c in rep["checks"] if c["name"] == "final_mp4_streams")
    assert stream_check["passed"] is False
    assert "No audio stream found" in stream_check["detail"]


def test_qa_catches_unintended_black_frame(base_qa_fixture: Path) -> None:
    # 2-second video that is completely black
    render_dir = base_qa_fixture / "render"
    _generate_synthetic_video(render_dir / "final.mp4", duration_sec=2.0, with_audio=True, color="black")

    agent = QAAgent(run_dir=base_qa_fixture)
    with pytest.raises(ValueError):
        agent.run()

    with open(base_qa_fixture / "qa" / "qa_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)

    black_check = next(c for c in rep["checks"] if c["name"] == "blackdetect")
    assert black_check["passed"] is False
    assert "unexpected black stretches" in black_check["detail"].lower()
    assert rep["passed"] is False
    assert rep["redo_stage"] == "render"


def test_qa_handles_missing_voice_report_gracefully(base_qa_fixture: Path) -> None:
    # Ensure voice report is missing
    vr_file = base_qa_fixture / "audio" / "voice_report.json"
    if vr_file.exists():
        vr_file.unlink()

    agent = QAAgent(run_dir=base_qa_fixture)
    agent.run()

    with open(base_qa_fixture / "qa" / "qa_report.json", "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Should record warning or note rather than crashing
    loudness_check = next(c for c in rep["checks"] if c["name"] == "loudness")
    assert "skipped" in loudness_check["detail"].lower() or "missing" in loudness_check["detail"].lower()

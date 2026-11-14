"""Unit tests for audio_sync module (Librosa beat analysis, cut snapping, and synthetic bed)."""
from __future__ import annotations

from pathlib import Path
import pytest

from src.utils.audio_sync import (
    extract_audio_grid,
    generate_synthetic_broadcast_theme,
    load_and_analyze_music,
    snap_cuts_to_beats,
)


def test_generate_synthetic_broadcast_theme(tmp_path: Path) -> None:
    theme_path = tmp_path / "broadcast_theme.wav"
    out = generate_synthetic_broadcast_theme(theme_path, duration_sec=5.0, bpm=120.0)

    assert out.exists()
    assert out.stat().st_size > 1000  # Non-trivial audio file


def test_load_and_analyze_music_synthetic(tmp_path: Path) -> None:
    theme_path = tmp_path / "theme_120bpm.wav"
    generate_synthetic_broadcast_theme(theme_path, duration_sec=8.0, bpm=120.0)

    analysis = load_and_analyze_music(theme_path)
    assert "tempo_bpm" in analysis
    assert "beat_times" in analysis
    assert len(analysis["beat_times"]) > 0
    # 8 seconds at 120 BPM should yield around 14-18 beats
    assert 10 <= len(analysis["beat_times"]) <= 22
    assert analysis["duration_sec"] == pytest.approx(8.0, rel=0.1)


def test_snap_cuts_to_beats() -> None:
    # Beat grid at 0.5s intervals: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # 0.0 should remain 0.0
    # 1.48 should snap to 1.50 (diff 0.02 <= 0.35)
    # 2.89 should snap to 3.00 (diff 0.11 <= 0.35)
    # 3.82 should snap to 4.00 (diff 0.18 <= 0.35)
    raw_cuts = [0.0, 1.48, 2.89, 3.82]
    snapped = snap_cuts_to_beats(raw_cuts, beat_times, tolerance_sec=0.35)

    assert snapped[0] == 0.0
    assert snapped[1] == 1.5
    assert snapped[2] == 3.0
    assert snapped[3] == 4.0


def test_snap_cuts_out_of_tolerance() -> None:
    beat_times = [1.0, 2.0, 3.0]
    # 1.5 is diff 0.5 > tolerance 0.25 -> should NOT snap
    raw_cuts = [0.0, 1.5, 2.05]
    snapped = snap_cuts_to_beats(raw_cuts, beat_times, tolerance_sec=0.25)

    assert snapped[0] == 0.0
    assert snapped[1] == 1.5 # Preserved original
    assert snapped[2] == 2.0 # Snapped to 2.0 from 2.05


def test_extract_audio_grid_disabled() -> None:
    scenes = [{"id": "s1", "start": 0.0, "end": 5.0}]
    grid = extract_audio_grid(None, None, scenes, style_config={"music": {"enabled": False}})
    assert grid["enabled"] is False
    assert grid["snapped_scenes"] == scenes


def test_extract_audio_grid_enabled(tmp_path: Path) -> None:
    theme_path = tmp_path / "theme.wav"
    generate_synthetic_broadcast_theme(theme_path, duration_sec=10.0, bpm=120.0)

    scenes = [
        {"id": "intro", "start": 0.0, "end": 4.95},
        {"id": "scene_01", "start": 4.95, "end": 9.92},
    ]
    style_cfg = {
        "music": {
            "enabled": True,
            "voice_duck_db": -22,
            "music_bed_db": -8,
            "fade_duration_sec": 0.25,
            "snap_tolerance_sec": 0.35,
        }
    }
    grid = extract_audio_grid(theme_path, None, scenes, style_config=style_cfg)
    assert grid["enabled"] is True
    assert len(grid["snapped_scenes"]) == 2
    assert "ducking" in grid
    assert len(grid["ducking"]["intervals"]) == 2

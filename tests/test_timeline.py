"""Unit tests for the Timeline Agent, forced alignment utilities, and caption chunking.

Runs purely with FakeAligner and synthetic fixtures. No GPU or model required.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any

import pytest

from src.agents.timeline import TimelineAgent
from src.utils.schema_validate import validate_contract
from src.utils.timeline import (
    FakeAligner,
    build_srt_content,
    chunk_scene_words_into_captions,
    expand_digits,
    interpolate_unplaced_words,
    prepare_scene_tokens,
    split_chunk_into_lines,
    validate_srt_syntax,
)
from src.utils.tts import create_silence_wav


@pytest.fixture
def sample_script_dict() -> dict[str, Any]:
    with open("tests/fixtures/sample_script.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_timeline_run_dir(tmp_path: Path, sample_script_dict: dict[str, Any]) -> Path:
    """Sets up a complete run directory fixture with audio, durations, and script."""
    # Write script.json
    script_file = tmp_path / "script.json"
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(sample_script_dict, f, indent=2)

    audio_dir = tmp_path / "audio"
    scenes_dir = audio_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy scene WAVs and durations
    scene_durations = []
    current_time = 0.0

    for idx, sc in enumerate(sample_script_dict["scenes"]):
        sc_id = sc["id"]
        sc_dur = 10.0
        wav_path = scenes_dir / f"{sc_id}.wav"
        create_silence_wav(wav_path, duration_sec=sc_dur, sample_rate=48000)

        scene_durations.append({
            "id": sc_id,
            "file": f"audio/scenes/{sc_id}.wav",
            "start_sec": current_time,
            "end_sec": current_time + sc_dur,
            "duration_sec": sc_dur,
        })
        current_time += sc_dur

    # Write scene_durations.json
    with open(audio_dir / "scene_durations.json", "w", encoding="utf-8") as f:
        json.dump(scene_durations, f, indent=2)

    # Master narration.wav matching total scene duration
    narration_wav = audio_dir / "narration.wav"
    create_silence_wav(narration_wav, duration_sec=current_time, sample_rate=48000)

    # Initial manifest
    manifest_data = {
        "run_id": "test_run",
        "date": "2026-11-06",
        "stages": {
            "timeline": {
                "status": "pending",
                "retry_count": 0,
            }
        }
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return tmp_path


def test_respelled_word_mapping():
    """A respelled word (e.g. Birgunj -> Beer-gunj) counts as ONE word with original spelling."""
    pronunciations = {
        "Birgunj": "Beer-gunj",
        "Bagmati": "Bahg-mah-tee",
    }
    raw_text = "Trucks arrived at Birgunj near the Bagmati corridor on November 6, 2026."
    token_items = prepare_scene_tokens(raw_text, pronunciations)

    # Find the Birgunj item
    birgunj_item = next(item for item in token_items if "Birgunj" in item["orig"])
    assert birgunj_item["orig"] == "Birgunj"
    assert birgunj_item["tts"] == "Beer-gunj"
    assert birgunj_item["tokens"] == ["beer", "gunj"]

    # Number expansion check
    item_6 = next(item for item in token_items if "6," in item["orig"])
    assert item_6["orig"] == "6,"
    assert item_6["tts"] == "sixth"
    assert item_6["tokens"] == ["sixth"]

    item_2026 = next(item for item in token_items if "2026." in item["orig"])
    assert item_2026["orig"] == "2026."
    assert item_2026["tts"] == "twenty twenty six"
    assert item_2026["tokens"] == ["twenty", "twenty", "six"]


def test_digit_expansion():
    """Digit expansion handles years and numbers cleanly."""
    assert expand_digits("2026") == "twenty twenty six"
    assert expand_digits("50") == "fifty"
    assert expand_digits("80") == "eighty"
    assert expand_digits("6th") == "sixth"


def test_unplaced_words_interpolation():
    """Unplaced words are interpolated smoothly between neighbors."""
    words = [
        {"w": "Good", "start": 0.5, "end": 0.8, "score": 0.95, "unplaced": False},
        {"w": "morning", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "friends", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "today", "start": 2.0, "end": 2.4, "score": 0.90, "unplaced": False},
    ]

    updated, unplaced_count = interpolate_unplaced_words(words, scene_duration=5.0)
    assert unplaced_count == 2
    assert updated[1]["start"] == 0.8
    assert updated[1]["end"] == 1.4
    assert updated[2]["start"] == 1.4
    assert updated[2]["end"] == 2.0
    assert updated[1]["unplaced"] is True
    assert updated[2]["unplaced"] is True


def test_unplaced_threshold_failure(mock_timeline_run_dir: Path):
    """Exceeding the unplaced threshold (scene >15% or run-wide >5%) raises ValueError."""
    agent = TimelineAgent(mock_timeline_run_dir)

    class BrokenAligner(FakeAligner):
        def align_scenes(self, scene_jobs, settings):
            res = super().align_scenes(scene_jobs, settings)
            # Mark all words in first scene as unplaced
            first_sc = list(res["scenes"].keys())[0]
            for w in res["scenes"][first_sc]["words"]:
                w["unplaced"] = True
                w["start"] = None
                w["end"] = None
            return res

    import src.agents.timeline as t_module
    orig_get_aligner = t_module.get_timeline_aligner
    t_module.get_timeline_aligner = lambda name, cfg: BrokenAligner()

    try:
        with pytest.raises(ValueError, match="failed alignment.*unplaced"):
            agent.run(aligner_override="fake", no_preview=True)
    finally:
        t_module.get_timeline_aligner = orig_get_aligner


def test_caption_chunking_constraints():
    """Caption chunking enforces max line length, line count, and min/max duration."""
    words = [
        {"w": "Good", "tts": "good", "start": 0.4, "end": 0.7, "score": 0.95},
        {"w": "evening.", "tts": "evening", "start": 0.75, "end": 1.2, "score": 0.95},
        {"w": "Welcome", "tts": "welcome", "start": 1.5, "end": 1.9, "score": 0.95},
        {"w": "to", "tts": "to", "start": 1.95, "end": 2.1, "score": 0.95},
        {"w": "the", "tts": "the", "start": 2.15, "end": 2.3, "score": 0.95},
        {"w": "Nepal", "tts": "nepal", "start": 2.35, "end": 2.7, "score": 0.95},
        {"w": "Daily", "tts": "daily", "start": 2.75, "end": 3.1, "score": 0.95},
        {"w": "Briefing.", "tts": "briefing", "start": 3.15, "end": 3.6, "score": 0.95},
    ]

    captions, _ = chunk_scene_words_into_captions(
        scene_words=words,
        scene_id="intro",
        scene_start=0.0,
        scene_end=5.0,
        max_chars_per_line=42,
        min_duration_sec=0.9,
        max_duration_sec=5.0,
    )

    assert len(captions) >= 1
    for c in captions:
        assert c["start"] >= 0.0
        assert c["end"] <= 5.0
        assert c["duration"] >= 0.9
        lines = c["text"].split("\n")
        assert len(lines) <= 2
        for line in lines:
            assert len(line) <= 50  # soft limit tolerance


def test_weak_word_avoidance():
    """Line breaks avoid ending on weak words like 'the', 'of', 'to'."""
    chunk = [
        {"w": "Welcome"}, {"w": "to"}, {"w": "the"},
        {"w": "Nepal"}, {"w": "Daily"}, {"w": "Briefing"},
        {"w": "for"}, {"w": "today."}
    ]
    # Total length: "Welcome to the Nepal Daily Briefing for today." = 46 chars (>42 chars)
    formatted = split_chunk_into_lines(chunk, max_chars=30)
    lines = formatted.split("\n")
    assert len(lines) == 2
    # Line 1 should not end in 'the' or 'to'
    last_word_l1 = lines[0].split()[-1].lower()
    assert last_word_l1 not in ("the", "to", "for")


def test_srt_formatting_and_syntax_validation():
    """Generated SRT format validates against strict SubRip rules and is re-parseable."""
    captions = [
        {
            "id": 1,
            "scene_id": "intro",
            "start": 0.35,
            "end": 2.50,
            "text": "Good evening.\nWelcome to the news.",
        },
        {
            "id": 2,
            "scene_id": "intro",
            "start": 2.55,
            "end": 4.80,
            "text": "Here are today's top stories.",
        },
    ]

    srt_text = build_srt_content(captions)
    is_valid, err = validate_srt_syntax(srt_text)
    assert is_valid is True, f"SRT validation failed: {err}"
    assert "00:00:00,350 --> 00:00:02,500" in srt_text
    assert "00:00:02,550 --> 00:00:04,800" in srt_text


def test_duration_mismatch_raises_error(mock_timeline_run_dir: Path):
    """Mismatch between scene duration sum and narration.wav > 20ms raises ValueError."""
    narration_wav = mock_timeline_run_dir / "audio" / "narration.wav"
    # Create narration with deliberate duration mismatch (+1.5 seconds)
    create_silence_wav(narration_wav, duration_sec=71.5, sample_rate=48000)

    agent = TimelineAgent(mock_timeline_run_dir)
    with pytest.raises(ValueError, match="within 20ms tolerance"):
        agent.run(aligner_override="fake", no_preview=True)


def test_end_to_end_timeline_agent_run(mock_timeline_run_dir: Path):
    """End-to-end execution of TimelineAgent produces schema-compliant timeline and valid subtitles."""
    agent = TimelineAgent(mock_timeline_run_dir)
    output_files = agent.run(aligner_override="fake", no_preview=True)

    # Check outputs exist
    for rel_f in output_files:
        assert (mock_timeline_run_dir / rel_f).exists()

    # Validate schema
    timeline_file = mock_timeline_run_dir / "timeline" / "timeline.json"
    with open(timeline_file, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    is_valid, err = validate_contract(timeline_data, "contracts/timeline.schema.json")
    assert is_valid is True, f"Schema validation error: {err}"

    # Verify scene partitioning
    scenes = timeline_data["scenes"]
    assert scenes[0]["start"] == 0.0
    for idx in range(len(scenes) - 1):
        assert abs(scenes[idx]["end"] - scenes[idx + 1]["start"]) < 0.001

    # Verify word monotonicity
    for sc in scenes:
        words = sc["words"]
        for k in range(len(words) - 1):
            assert words[k]["start"] <= words[k + 1]["start"]
            assert words[k]["start"] <= words[k]["end"]

    # Verify SRT syntax
    srt_file = mock_timeline_run_dir / "timeline" / "captions.srt"
    with open(srt_file, "r", encoding="utf-8") as f:
        srt_content = f.read()
    valid_srt, srt_err = validate_srt_syntax(srt_content)
    assert valid_srt is True, f"SRT syntax error: {srt_err}"

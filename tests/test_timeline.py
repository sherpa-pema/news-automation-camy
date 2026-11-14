"""Unit tests for the Timeline Agent, forced alignment utilities, and caption chunking.

Runs purely with FakeAligner and synthetic fixtures. No GPU or model required.
"""
from __future__ import annotations

import json
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
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_file = script_dir / "script.json"
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
    """Unplaced words are interpolated proportionally by word weight between neighbors."""
    # Equal-weight short words split evenly
    equal_words = [
        {"w": "Good", "tts": "Good", "start": 0.5, "end": 0.8, "score": 0.95, "unplaced": False},
        {"w": "cat", "tts": "cat", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "dog", "tts": "dog", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "today", "tts": "today", "start": 2.0, "end": 2.4, "score": 0.90, "unplaced": False},
    ]

    updated, unplaced_count = interpolate_unplaced_words(equal_words, scene_duration=5.0)
    assert unplaced_count == 2
    assert updated[1]["start"] == 0.8
    assert updated[1]["end"] == 1.4
    assert updated[2]["start"] == 1.4
    assert updated[2]["end"] == 2.0
    assert updated[1]["unplaced"] is True
    assert updated[2]["unplaced"] is True

    # Unequal-weight words allocate proportionally more duration to multi-syllable words
    weighted_words = [
        {"w": "The", "tts": "The", "start": 0.5, "end": 0.8, "score": 0.95, "unplaced": False},
        {"w": "a", "tts": "a", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "international", "tts": "international", "start": None, "end": None, "score": 0.0, "unplaced": True},
        {"w": "summit", "tts": "summit", "start": 2.8, "end": 3.2, "score": 0.90, "unplaced": False},
    ]
    updated_w, cnt_w = interpolate_unplaced_words(weighted_words, scene_duration=5.0)
    assert cnt_w == 2
    dur_short = updated_w[1]["end"] - updated_w[1]["start"]
    dur_long = updated_w[2]["end"] - updated_w[2]["start"]
    assert dur_long > dur_short * 2.5
    assert updated_w[1]["start"] == 0.8
    assert updated_w[1]["end"] == updated_w[2]["start"]
    assert updated_w[2]["end"] == 2.8


def test_unplaced_threshold_degrades_gracefully(mock_timeline_run_dir: Path):
    """Exceeding the unplaced threshold degrades to weighted interpolation and flags manifest.json."""
    from src.utils.manifest import create_initial_manifest

    manifest = create_initial_manifest(
        "test_run",
        "2026-11-06",
        {"stages": [{"id": "timeline", "inputs": [], "outputs": []}]},
    )
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
        output_files = agent.run(manifest=manifest, aligner_override="fake", no_preview=True)
    finally:
        t_module.get_timeline_aligner = orig_get_aligner

    for rel_f in output_files:
        assert (mock_timeline_run_dir / rel_f).exists()

    timeline_data = json.loads((mock_timeline_run_dir / "timeline" / "timeline.json").read_text(encoding="utf-8"))
    is_valid, err = validate_contract(timeline_data, "contracts/timeline.schema.json")
    assert is_valid is True, f"Schema validation error: {err}"
    assert any("degraded alignment" in w for w in timeline_data["warnings"])

    first_scene = timeline_data["scenes"][0]
    for k in range(len(first_scene["words"]) - 1):
        assert first_scene["words"][k]["start"] <= first_scene["words"][k + 1]["start"]
        assert first_scene["words"][k]["start"] <= first_scene["words"][k]["end"]

    saved_manifest = json.loads((mock_timeline_run_dir / "manifest.json").read_text(encoding="utf-8"))
    tl_metrics = saved_manifest["stages"]["timeline"]["metrics"]
    assert tl_metrics["degraded"] is True
    assert tl_metrics["unplaced_ratio"] > 0.05
    assert len(tl_metrics["degraded_scenes"]) == 1
    assert tl_metrics["degraded_scenes"][0]["id"] == first_scene["id"]


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


def test_timeline_schema_validation_negative_cases():
    """Negative test proving timeline.schema.json rejects missing fields and invalid structures."""
    # 1. Missing required field 'scenes'
    bad_timeline = {
        "audio_file": "audio/narration.wav",
        "duration_sec": 120.5,
    }
    is_valid, err = validate_contract(bad_timeline, "contracts/timeline.schema.json")
    assert is_valid is False
    assert "'scenes' is a required property" in err

    # 2. Scene missing required field 'words'
    bad_scene_timeline = {
        "audio_file": "audio/narration.wav",
        "duration_sec": 120.5,
        "scenes": [
            {
                "id": "intro",
                "start": 0.0,
                "end": 10.0,
                # missing 'words'
            }
        ]
    }
    is_valid, err = validate_contract(bad_scene_timeline, "contracts/timeline.schema.json")
    assert is_valid is False
    assert "'words' is a required property" in err


def test_timeline_agent_run_approval_fingerprint_gate(tmp_path):
    """TimelineAgent.run must execute the approval and validator fingerprint gate without NameError."""
    import hashlib
    import wave
    from src.agents.timeline import TimelineAgent
    from src.utils.editor_validator import compute_validator_fingerprint

    run_dir = tmp_path / "2026-11-08"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    script_data = {
        "language": "English",
        "origin": "editor",
        "scenes": [],
    }
    script_path = script_dir / "script.json"
    script_bytes = json.dumps(script_data).encode("utf-8")
    script_path.write_bytes(script_bytes)
    script_hash = hashlib.sha256(script_bytes).hexdigest()

    curr_fp = compute_validator_fingerprint()
    approval_data = {
        "approved_at": "2026-11-08T10:00:00Z",
        "script_sha256": script_hash,
        "approved_by": "user",
        "validator_fingerprint": curr_fp,
    }
    (script_dir / "approval.json").write_text(json.dumps(approval_data), encoding="utf-8")
    (script_dir / "validation_report.json").write_text(
        json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": curr_fp}),
        encoding="utf-8",
    )

    (audio_dir / "scene_durations.json").write_text(json.dumps([]), encoding="utf-8")
    wav_path = audio_dir / "narration.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"")

    manifest = {"stages": {"timeline": {"status": "pending", "retry_count": 0, "input_files": [], "output_files": []}}}

    agent = TimelineAgent(run_dir, config={"stages": [{"id": "timeline", "aligner": "fake"}]})
    # The approval & fingerprint gate executes first (verifying hashes & fingerprint without NameError).
    # Since scenes is empty, it proceeds past alignment to schema validation requiring non-empty scenes.
    with pytest.raises(ValueError, match="should be non-empty"):
        agent.run(manifest)


def test_scene_partition_gap_still_aborts(mock_timeline_run_dir: Path):
    """Audio partition gaps/overlaps between scenes still raise ValueError."""
    dur_path = mock_timeline_run_dir / "audio" / "scene_durations.json"
    durations = json.loads(dur_path.read_text(encoding="utf-8"))
    durations[1]["start_sec"] += 0.5
    durations[1]["end_sec"] += 0.5
    dur_path.write_text(json.dumps(durations, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    with pytest.raises(ValueError, match="partition gap/overlap"):
        agent.run(aligner_override="fake", no_preview=True)


def test_fake_aligner_weights_durations_by_word_length():
    """FakeAligner allocates longer durations to multi-syllable/longer words than short ones."""
    words = prepare_scene_tokens("To international telecommunications infrastructure and a.")
    aligner = FakeAligner()
    res = aligner.align_scenes(
        [{"id": "s1", "audio_path": "dummy.wav", "words": words, "duration_sec": 8.0}],
        {},
    )
    aligned = res["scenes"]["s1"]["words"]
    dur_to = aligned[0]["end"] - aligned[0]["start"]
    dur_intl = aligned[1]["end"] - aligned[1]["start"]
    dur_telecom = aligned[2]["end"] - aligned[2]["start"]
    dur_a = aligned[5]["end"] - aligned[5]["start"]

    assert dur_intl > dur_to * 2.0
    assert dur_telecom > dur_a * 2.0
    for k in range(len(aligned) - 1):
        assert aligned[k]["start"] <= aligned[k]["end"] <= aligned[k + 1]["start"]


def test_shared_normalizer_parity_between_voice_and_timeline(mock_timeline_run_dir: Path):
    """Voice clean_narration_text and Timeline prepare_scene_tokens expand currency, %, decimals, and acronyms identically."""
    from src.agents.voice import clean_narration_text

    pronunciations = {
        "Sarita Rai": "Sa-ree-ta Rye",
        "Birgunj": "Beer-gunj",
    }
    raw = "NRB and UML noted Rs. 5,000, or 12.5%, at 3.14 ratio for Sarita Rai in Birgunj."

    voice_text = clean_narration_text(raw, pronunciations)
    assert "N R B" in voice_text
    assert "U M L" in voice_text
    assert "five thousand rupees" in voice_text
    assert "twelve point five percent" in voice_text
    assert "three point one four" in voice_text
    assert "Sa-ree-ta Rye" in voice_text
    assert "Beer-gunj" in voice_text

    tl_tokens = prepare_scene_tokens(raw, pronunciations)
    flattened_tl = [tok for item in tl_tokens for tok in item["tokens"]]

    import re
    voice_words = [re.sub(r"[^a-z']", "", w.lower()) for w in re.split(r"[\s\-]+", voice_text)]
    voice_words = [w for w in voice_words if w]
    assert flattened_tl == voice_words

    # Verify grouped display tokens preserve original broadcast text
    rs_item = next(item for item in tl_tokens if "Rs." in item["orig"])
    assert rs_item["orig"] == "Rs. 5,000,"
    assert rs_item["tts"] == "five thousand rupees"

    # End-to-end sentence timing stays aligned when currency spans multiple whitespace words
    script_path = mock_timeline_run_dir / "script" / "script.json"
    script_data = json.loads(script_path.read_text(encoding="utf-8"))
    script_data["scenes"][0]["narration"] = (
        "NRB allocated Rs. 5,000 for Birgunj projects today. Inflation rose by 12.5%."
    )
    script_path.write_text(json.dumps(script_data, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    agent.run(aligner_override="fake", no_preview=True)
    tl_out = json.loads((mock_timeline_run_dir / "timeline" / "timeline.json").read_text(encoding="utf-8"))
    s0 = tl_out["scenes"][0]
    assert len(s0["sentences"]) == 2
    assert s0["sentences"][0]["end"] < s0["sentences"][1]["start"]
    assert "Rs. 5,000" in " ".join(w["w"] for w in s0["words"])


def test_devanagari_compatibility_guard(mock_timeline_run_dir: Path):
    """English-only text incurs no Devanagari warnings; Devanagari without uroman degrades gracefully."""
    en_warnings: list[str] = []
    prepare_scene_tokens("Good evening from Kathmandu.", warnings_out=en_warnings)
    assert en_warnings == []

    # Devanagari with an explicit romanizer produces aligner tokens
    rom_warnings: list[str] = []
    rom_map = {"नेपाल": "nepal", "समाचार": "samachar"}
    rom_tokens = prepare_scene_tokens(
        "नेपाल समाचार",
        romanizer=lambda s: rom_map.get(s, s),
        warnings_out=rom_warnings,
    )
    assert rom_warnings == []
    assert [t["tokens"] for t in rom_tokens] == [["nepal"], ["samachar"]]

    # Devanagari without uroman emits compatibility warning and degrades cleanly in TimelineAgent
    import src.utils.text_normalizer as tn
    orig_get_uroman = tn.get_optional_uroman
    tn.get_optional_uroman = lambda: None
    try:
        script_path = mock_timeline_run_dir / "script" / "script.json"
        script_data = json.loads(script_path.read_text(encoding="utf-8"))
        script_data["scenes"][0]["narration"] = "नेपाल समाचार आजको मुख्य खबर।"
        script_path.write_text(json.dumps(script_data, indent=2, ensure_ascii=False), encoding="utf-8")

        agent = TimelineAgent(mock_timeline_run_dir)
        # Simulate MMS-like aligner marking empty-token words unplaced
        class EmptyTokenUnplacedAligner(FakeAligner):
            def align_scenes(self, scene_jobs, settings):
                res = super().align_scenes(scene_jobs, settings)
                for job in scene_jobs:
                    sc_words = res["scenes"][job["id"]]["words"]
                    for tok, w_out in zip(job["words"], sc_words):
                        if not tok["tokens"]:
                            w_out["unplaced"] = True
                            w_out["start"] = None
                            w_out["end"] = None
                return res

        import src.agents.timeline as t_module
        orig_get_aligner = t_module.get_timeline_aligner
        t_module.get_timeline_aligner = lambda name, cfg: EmptyTokenUnplacedAligner()
        try:
            agent.run(aligner_override="fake", no_preview=True)
        finally:
            t_module.get_timeline_aligner = orig_get_aligner
    finally:
        tn.get_optional_uroman = orig_get_uroman

    tl_out = json.loads((mock_timeline_run_dir / "timeline" / "timeline.json").read_text(encoding="utf-8"))
    assert any("uroman" in w for w in tl_out["warnings"])
    assert len(tl_out["scenes"][0]["words"]) == 5



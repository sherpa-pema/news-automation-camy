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
        sc_dur = float(sc.get("est_duration_sec", 15.0))
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
                "speech_start": 0.5,
                "speech_end": 9.5,
                "sentences": [],
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


def test_real_run_caption_quality_regression_2026_11_15():
    """Phase 0 Acceptance: On the 2026-11-15 run, 0 cues late by >50ms, 0 duration/line/CPS violations."""
    run_file = Path("data/runs/2026-11-15/timeline/timeline.json")
    if not run_file.exists():
        pytest.skip("2026-11-15 real run fixture not found")

    tl_data = json.loads(run_file.read_text(encoding="utf-8"))
    all_captions: list[dict[str, Any]] = []
    cap_id = 1

    for sc in tl_data["scenes"]:
        caps, cap_id = chunk_scene_words_into_captions(
            scene_words=sc["words"],
            scene_id=sc["id"],
            scene_start=sc["start"],
            scene_end=sc["end"],
            start_caption_id=cap_id,
            max_chars_per_line=42,
            max_lines=2,
            min_duration_sec=0.9,
            max_duration_sec=5.0,
            lead_in_sec=0.05,
            hold_after_sec=0.15,
            min_gap_sec=0.04,
            target_cps=17.0,
            max_cps=20.0,
        )
        all_captions.extend(caps)

    # 1. 0 cues late by > 50 ms (and 0 cues late by > 0 ms)
    late_cues = [c for c in all_captions if c["start"] > c["words"][0]["start"]]
    assert len(late_cues) == 0, f"Found {len(late_cues)} late cues"

    # 2. Invariants: start < end, end >= last_word.end, start <= first_word.start
    for c in all_captions:
        assert c["start"] < c["end"]
        assert c["start"] <= c["words"][0]["start"]
        assert c["end"] >= c["words"][-1]["end"]

    # 3. Duration limits [0.9s, 5.0s]
    dur_violations = [c for c in all_captions if c["duration"] > 5.05 or c["duration"] < 0.89]
    assert len(dur_violations) == 0, f"Duration violations: {dur_violations}"

    # 4. Line limits: <= 2 lines, each <= 42 chars
    line_violations = []
    for c in all_captions:
        lines = c["text"].split("\n")
        assert len(lines) <= 2, f"More than 2 lines: {c['text']}"
        for line in lines:
            if len(line) > 42:
                line_violations.append((c["id"], line, len(line)))
    assert len(line_violations) == 0, f"Line length violations: {line_violations}"

    # 5. Reading speed (CPS <= 20.0)
    cpss = [len(c["text"].replace("\n", " ")) / c["duration"] for c in all_captions]
    cps_violations = [c for c in all_captions if (len(c["text"].replace("\n", " ")) / c["duration"]) > 20.05]
    assert len(cps_violations) == 0, f"CPS violations: {cps_violations}"
    median_cps = sorted(cpss)[len(cpss) // 2]
    assert 15.0 <= median_cps <= 18.0, f"Median CPS out of expected range: {median_cps}"

    # 6. SRT syntax validation
    srt_text = build_srt_content(all_captions)
    valid_srt, srt_err = validate_srt_syntax(srt_text)
    assert valid_srt is True, f"SRT syntax error: {srt_err}"


def test_caption_chunking_randomized_properties():
    """Property test: randomized word sequences never produce invalid cues, late starts, or overlaps."""
    import random

    rng = random.Random(42)
    sample_vocabulary = [
        "The", "Supreme", "Court", "of", "Nepal", "issued", "a", "landmark",
        "directive", "today,", "instructing", "the", "government", "to", "ensure",
        "transparency", "in", "major", "infrastructure", "spending.", "Meanwhile,",
        "economic", "indicators", "showed", "a", "moderate", "uptick", "in", "exports.",
        "Farmers", "in", "the", "southern", "plains", "celebrated", "favorable", "weather."
    ]

    for iteration in range(30):
        n_words = rng.randint(5, 40)
        curr_t = rng.uniform(0.2, 1.0)
        scene_words = []
        for i in range(n_words):
            w_text = rng.choice(sample_vocabulary)
            dur = rng.uniform(0.1, 0.45)
            scene_words.append({
                "w": w_text,
                "tts": w_text.lower(),
                "start": round(curr_t, 3),
                "end": round(curr_t + dur, 3),
                "score": 0.95,
            })
            curr_t += dur + rng.uniform(0.01, 0.15)

        sc_end = round(curr_t + rng.uniform(0.3, 1.0), 3)
        captions, _ = chunk_scene_words_into_captions(
            scene_words=scene_words,
            scene_id=f"rand_{iteration}",
            scene_start=0.0,
            scene_end=sc_end,
            max_chars_per_line=42,
            max_lines=2,
            min_duration_sec=0.9,
            max_duration_sec=5.0,
        )

        assert len(captions) >= 1
        for idx, c in enumerate(captions):
            # Invariant: cue start never after first word
            assert c["start"] <= c["words"][0]["start"]
            # Invariant: cue end never before last word
            assert c["end"] >= c["words"][-1]["end"]
            # Invariant: cue start strictly < cue end
            assert c["start"] < c["end"]
            # Limits
            lines = c["text"].split("\n")
            assert len(lines) <= 2
            for l in lines:
                assert len(l) <= 42
            # No overlaps with previous cue
            if idx > 0:
                assert captions[idx - 1]["end"] <= c["start"] + 0.001


def test_partial_scene_run_does_not_overwrite_canonical(mock_timeline_run_dir: Path):
    """0.4 Partial-run safety: --scene writes timeline/partial/<scene>.json without touching canonical timeline."""
    agent = TimelineAgent(mock_timeline_run_dir)

    # 1. Run full timeline first to generate canonical artifacts
    full_outs = agent.run(aligner_override="fake", no_preview=True)
    canonical_timeline_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    canonical_bytes = canonical_timeline_path.read_bytes()
    canonical_srt_bytes = (mock_timeline_run_dir / "timeline" / "captions.srt").read_bytes()

    # 2. Run with scene_filter="intro" without allow_partial_overwrite
    partial_outs = agent.run(
        aligner_override="fake",
        scene_filter="intro",
        no_preview=True,
        allow_partial_overwrite=False,
    )

    # Canonical files must remain byte-identical!
    assert canonical_timeline_path.read_bytes() == canonical_bytes
    assert (mock_timeline_run_dir / "timeline" / "captions.srt").read_bytes() == canonical_srt_bytes

    # Partial files must be generated
    assert "timeline/partial/intro.json" in partial_outs
    assert "timeline/partial/intro.srt" in partial_outs
    assert (mock_timeline_run_dir / "timeline" / "partial" / "intro.json").exists()
    assert (mock_timeline_run_dir / "timeline" / "partial" / "intro.srt").exists()

    partial_data = json.loads((mock_timeline_run_dir / "timeline" / "partial" / "intro.json").read_text(encoding="utf-8"))
    assert len(partial_data["scenes"]) == 1
    assert partial_data["scenes"][0]["id"] == "intro"


def test_save_manifest_atomic_and_concurrency(tmp_path: Path):
    """0.5 Manifest concurrency: save_manifest uses atomic rename and prevents corrupt writes."""
    import concurrent.futures
    import threading
    from src.utils.manifest import create_initial_manifest, load_manifest, save_manifest, update_stage_status

    manifest_path = tmp_path / "manifest.json"
    manifest = create_initial_manifest(
        "test_run",
        "2026-11-06",
        {"stages": [{"id": "voice"}, {"id": "timeline"}, {"id": "media"}]},
    )
    save_manifest(manifest, manifest_path)

    lock = threading.Lock()

    def update_timeline():
        for i in range(10):
            with lock:
                curr = load_manifest(manifest_path)
                update_stage_status(curr, "timeline", "running", metrics={"step": i})
                save_manifest(curr, manifest_path)

    def update_media():
        for i in range(10):
            with lock:
                curr = load_manifest(manifest_path)
                update_stage_status(curr, "media", "running", metrics={"candidates": i})
                save_manifest(curr, manifest_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(update_timeline)
        f2 = executor.submit(update_media)
        f1.result()
        f2.result()

    final_m = load_manifest(manifest_path)
    assert final_m["stages"]["timeline"]["metrics"]["step"] == 9
    assert final_m["stages"]["media"]["metrics"]["candidates"] == 9


def test_caption_audit_pure_function():
    """1.1 Pure function test for audit_captions with valid and invalid cues."""
    from src.utils.timeline import audit_captions

    valid_captions = [
        {
            "id": 1,
            "scene_id": "intro",
            "start": 0.05,
            "end": 2.5,
            "duration": 2.45,
            "text": "Welcome to today's news briefing.",
            "words": [
                {"w": "Welcome", "start": 0.1, "end": 0.5, "score": 0.95},
                {"w": "to", "start": 0.6, "end": 0.8, "score": 0.98},
                {"w": "today's", "start": 0.9, "end": 1.3, "score": 0.92},
                {"w": "news", "start": 1.4, "end": 1.7, "score": 0.96},
                {"w": "briefing.", "start": 1.8, "end": 2.3, "score": 0.94},
            ],
        }
    ]
    scenes = [
        {
            "id": "intro",
            "words": [
                {"w": "Welcome", "start": 0.1, "end": 0.5},
                {"w": "to", "start": 0.6, "end": 0.8},
                {"w": "today's", "start": 0.9, "end": 1.3},
                {"w": "news", "start": 1.4, "end": 1.7},
                {"w": "briefing.", "start": 1.8, "end": 2.3},
            ],
        }
    ]

    res = audit_captions(valid_captions, scenes=scenes)
    assert res["summary"]["passed"] is True
    assert res["summary"]["violations_count"] == 0
    assert res["summary"]["late_captions_count"] == 0
    assert res["summary"]["overlap_count"] == 0
    assert "intro" in res["per_scene"]
    assert res["per_scene"]["intro"]["caption_count"] == 1

    # Deliberate violations test:
    # 1. Late start > 50ms (caption starts at 0.3s, word starts at 0.1s -> 200ms late)
    # 2. Line length exceeded (> 42 chars)
    # 3. High CPS
    bad_captions = [
        {
            "id": 1,
            "scene_id": "intro",
            "start": 0.3,  # 200ms after first word
            "end": 0.6,    # duration 0.3s -> too short and high CPS
            "duration": 0.3,
            "text": "This line is intentionally constructed to exceed the forty-two characters threshold easily",
            "words": [
                {"w": "This", "start": 0.1, "end": 0.5, "score": 0.9},
            ],
        },
        {
            "id": 2,
            "scene_id": "intro",
            "start": 0.5,  # Overlaps with caption 1 (end 0.6s)
            "end": 1.0,
            "duration": 0.5,
            "text": "Second line",
            "words": [
                {"w": "Second", "start": 0.55, "end": 0.9, "score": 0.9},
            ],
        },
    ]

    bad_res = audit_captions(bad_captions, scenes=scenes)
    assert bad_res["summary"]["passed"] is False
    assert bad_res["summary"]["violations_count"] > 0
    assert bad_res["summary"]["late_50ms_count"] >= 1
    assert bad_res["summary"]["overlap_count"] >= 1
    assert bad_res["summary"]["line_length_violations"] >= 1
    assert bad_res["summary"]["min_duration_violations"] >= 1
    assert bad_res["summary"]["word_conservation_errors"] >= 1


def test_timeline_agent_emits_caption_audit_and_quality(mock_timeline_run_dir: Path):
    """1.2 & 1.4 Acceptance: TimelineAgent writes caption_audit.json, attaches quality metadata, and validates schema."""
    from src.utils.schema_validate import validate_contract

    agent = TimelineAgent(mock_timeline_run_dir)
    outputs = agent.run(aligner_override="fake", no_preview=True)

    assert "timeline/caption_audit.json" in outputs
    audit_path = mock_timeline_run_dir / "timeline" / "caption_audit.json"
    assert audit_path.exists()

    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert "summary" in audit_data
    assert "per_scene" in audit_data
    assert audit_data["summary"]["passed"] is True, f"Violations: {audit_data['violations']}"
    assert audit_data["summary"]["violations_count"] == 0

    tl_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    tl_data = json.loads(tl_path.read_text(encoding="utf-8"))

    # Top-level degraded and low_confidence_scenes
    assert "degraded" in tl_data
    assert tl_data["degraded"] is False
    assert "low_confidence_scenes" in tl_data
    assert isinstance(tl_data["low_confidence_scenes"], list)

    # Per-scene quality metadata
    for sc in tl_data["scenes"]:
        assert "quality" in sc
        q = sc["quality"]
        assert "words" in q
        assert "low_confidence_words" in q
        assert "min_score" in q
        assert "unplaced" in q
        assert "caption_count" in q
        assert "max_late_ms" in q
        assert "max_cps" in q
        assert q["words"] == len(sc["words"])

    # Schema validation
    is_valid_tl, err_tl = validate_contract(tl_data, "contracts/timeline.schema.json")
    assert is_valid_tl is True, f"timeline.json contract failed: {err_tl}"

    caps_path = mock_timeline_run_dir / "timeline" / "captions.json"
    caps_data = json.loads(caps_path.read_text(encoding="utf-8"))
    is_valid_caps, err_caps = validate_contract(caps_data, "contracts/captions.schema.json")
    assert is_valid_caps is True, f"captions.json contract failed: {err_caps}"

    # Verify agent.last_metrics
    m = agent.last_metrics
    assert m["caption_audit_passed"] is True
    assert m["caption_violations"] == 0
    assert "max_late_ms" in m
    assert "max_cps" in m
    assert "median_cps" in m


def test_low_confidence_scene_flagged(mock_timeline_run_dir: Path):
    """1.2 & DoD: degraded = true when >= 5% words are low-confidence or unplaced."""
    import src.agents.timeline as t_module
    from src.utils.timeline import FakeAligner

    class LowConfidenceFakeAligner(FakeAligner):
        def align_scenes(self, scene_jobs, settings=None):
            res = super().align_scenes(scene_jobs, settings=settings or {})
            # Make words in first scene have low scores
            first_sc = scene_jobs[0]["id"]
            if first_sc in res["scenes"]:
                for w in res["scenes"][first_sc]["words"]:
                    w["score"] = 0.20  # < low_score_threshold (0.35)
            return res

    orig_get_aligner = t_module.get_timeline_aligner
    t_module.get_timeline_aligner = lambda name, cfg: LowConfidenceFakeAligner()
    try:
        agent = TimelineAgent(mock_timeline_run_dir)
        agent.run(aligner_override="fake", no_preview=True)
    finally:
        t_module.get_timeline_aligner = orig_get_aligner

    tl_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    tl_data = json.loads(tl_path.read_text(encoding="utf-8"))

    first_sc_id = tl_data["scenes"][0]["id"]
    assert first_sc_id in tl_data["low_confidence_scenes"]
    # Check words have low_confidence: True
    low_conf_words = [w for w in tl_data["scenes"][0]["words"] if w.get("low_confidence")]
    assert len(low_conf_words) > 0

    # Scene quality reflects low_confidence_words
    q0 = tl_data["scenes"][0]["quality"]
    assert q0["low_confidence_words"] == len(low_conf_words)
    assert q0["min_score"] <= 0.40

    # Run-level degraded check
    assert tl_data["degraded"] is True
    assert agent.last_metrics["degraded"] is True
    assert agent.last_metrics["low_score_words"] >= len(low_conf_words)


def test_captions_schema_negative_cases():
    """1.4 Negative contract tests for contracts/captions.schema.json."""
    from src.utils.schema_validate import validate_contract

    schema_path = "contracts/captions.schema.json"

    # Missing required 'start'
    bad_cues_1 = [{"id": 1, "scene_id": "intro", "end": 1.0, "duration": 1.0, "text": "Hi", "words": []}]
    valid, _ = validate_contract(bad_cues_1, schema_path)
    assert valid is False

    # Negative duration
    bad_cues_2 = [{"id": 1, "scene_id": "intro", "start": 1.0, "end": 0.5, "duration": -0.5, "text": "Hi", "words": []}]
    valid, _ = validate_contract(bad_cues_2, schema_path)
    assert valid is False

    # Missing word 'end'
    bad_cues_3 = [{
        "id": 1,
        "scene_id": "intro",
        "start": 0.0,
        "end": 1.0,
        "duration": 1.0,
        "text": "Hi",
        "words": [{"w": "Hi", "start": 0.0}]
    }]
    valid, _ = validate_contract(bad_cues_3, schema_path)
    assert valid is False


def test_sentence_word_conservation(mock_timeline_run_dir: Path):
    """2.1 Asserts Devanagari terminators, abbreviations, and token conservation mismatch handling."""
    from src.utils.text_normalizer import split_into_sentences

    # 1. Test Devanagari terminators
    devanagari_text = "यो पहिलो वाक्य हो। यो दोस्रो वाक्य हो॥ यो तेस्रो हो।"
    sents = split_into_sentences(devanagari_text)
    assert len(sents) == 3
    assert sents[0] == "यो पहिलो वाक्य हो।"
    assert sents[1] == "यो दोस्रो वाक्य हो॥"
    assert sents[2] == "यो तेस्रो हो।"

    # 2. Test abbreviation / currency boundary
    curr_text = "The budget allocated Rs. 5,000 for the project. Works start tomorrow."
    sents_curr = split_into_sentences(curr_text)
    assert len(sents_curr) == 2
    assert "Rs. 5,000" in sents_curr[0]

    # 3. Test token conservation mismatch warning and proportional fallback in TimelineAgent
    agent = TimelineAgent(mock_timeline_run_dir)
    # Modify script.json to have mismatched sentence structure
    script_file = mock_timeline_run_dir / "script" / "script.json"
    s_data = json.loads(script_file.read_text(encoding="utf-8"))
    # Introduce extra sentence boundaries in narration
    s_data["scenes"][0]["narration"] = "First sentence! Second sentence? Third sentence. Fourth sentence now."
    script_file.write_text(json.dumps(s_data, indent=2), encoding="utf-8")

    out_files = agent.run(aligner_override="fake", no_preview=True)
    assert len(out_files) > 0

    tl_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    tl_data = json.loads(tl_path.read_text(encoding="utf-8"))
    first_sc = tl_data["scenes"][0]
    assert len(first_sc["sentences"]) >= 3
    # Check all sentences have valid timestamps and monotonicity
    for s in first_sc["sentences"]:
        assert s["start"] <= s["end"]
        assert s["start"] >= first_sc["start"]
        assert s["end"] <= first_sc["end"]


def test_scene_set_mismatch_raises(mock_timeline_run_dir: Path):
    """2.2 Input invariant: Scene set mismatch or duplicates raise ValueError."""
    # 1. Duplicate scene ID in script.json
    script_file = mock_timeline_run_dir / "script" / "script.json"
    s_data = json.loads(script_file.read_text(encoding="utf-8"))
    s_data["scenes"].append(dict(s_data["scenes"][0]))
    script_file.write_text(json.dumps(s_data, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    with pytest.raises(ValueError, match="Duplicate scene IDs found in script.json"):
        agent.run(aligner_override="fake", no_preview=True)

    # 2. Missing scene in scene_durations
    s_data["scenes"].pop()
    script_file.write_text(json.dumps(s_data, indent=2), encoding="utf-8")

    dur_file = mock_timeline_run_dir / "audio" / "scene_durations.json"
    dur_data = json.loads(dur_file.read_text(encoding="utf-8"))
    removed_sc = dur_data.pop(0)
    dur_file.write_text(json.dumps(dur_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Scene set mismatch between script.json and scene_durations.json"):
        agent.run(aligner_override="fake", no_preview=True)


def test_scene_order_validated(mock_timeline_run_dir: Path):
    """2.2 Input invariant: Out-of-order or non-monotonic scenes raise ValueError."""
    dur_file = mock_timeline_run_dir / "audio" / "scene_durations.json"
    dur_data = json.loads(dur_file.read_text(encoding="utf-8"))

    # Swap first two scenes in scene_durations to cause order mismatch with script.json
    dur_data[0], dur_data[1] = dur_data[1], dur_data[0]
    dur_file.write_text(json.dumps(dur_data, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    with pytest.raises(ValueError, match="Scene order mismatch between script.json"):
        agent.run(aligner_override="fake", no_preview=True)

    # Test non-monotonic start_sec
    script_file = mock_timeline_run_dir / "script" / "script.json"
    s_data = json.loads(script_file.read_text(encoding="utf-8"))
    # Also swap script so order matches, but set start_sec non-monotonic
    s_data["scenes"][0], s_data["scenes"][1] = s_data["scenes"][1], s_data["scenes"][0]
    script_file.write_text(json.dumps(s_data, indent=2), encoding="utf-8")

    dur_data[1]["start_sec"] = dur_data[0]["start_sec"] - 1.0  # Non-monotonic!
    dur_file.write_text(json.dumps(dur_data, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Non-monotonic scene start_sec"):
        agent.run(aligner_override="fake", no_preview=True)


def test_final_scene_end_reconciled(mock_timeline_run_dir: Path):
    """2.2 Input invariant: Reconciles scenes[-1].end to duration_sec exactly within 20ms residual."""
    dur_file = mock_timeline_run_dir / "audio" / "scene_durations.json"
    dur_data = json.loads(dur_file.read_text(encoding="utf-8"))

    # Introduce a 10ms residual in the final scene
    orig_end = dur_data[-1]["end_sec"]
    dur_data[-1]["end_sec"] = orig_end - 0.010
    dur_data[-1]["duration_sec"] = dur_data[-1]["duration_sec"] - 0.010
    dur_file.write_text(json.dumps(dur_data, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    agent.run(aligner_override="fake", no_preview=True)

    tl_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    tl_data = json.loads(tl_path.read_text(encoding="utf-8"))

    # Master duration equals final scene end exactly
    assert tl_data["scenes"][-1]["end"] == tl_data["duration_sec"]
    assert tl_data["scenes"][-1]["end"] == orig_end


def test_worker_per_scene_error_handled_gracefully(mock_timeline_run_dir: Path):
    """2.4 Worker hardening: Per-scene failure falls back to weighted interpolation without crashing."""
    import src.agents.timeline as t_module
    from src.utils.timeline import FakeAligner

    class ErrorProneFakeAligner(FakeAligner):
        def align_scenes(self, scene_jobs, settings=None):
            res = super().align_scenes(scene_jobs, settings=settings or {})
            # Simulate a worker error on the second scene
            if len(scene_jobs) > 1:
                failing_sc = scene_jobs[1]["id"]
                res["scenes"][failing_sc] = {
                    "duration_sec": 0.0,
                    "error": "CUDA out of memory in worker align_scene",
                    "words": [
                        {
                            "w": w.get("orig", ""),
                            "tts": w.get("tts", ""),
                            "start": None,
                            "end": None,
                            "score": 0.0,
                            "unplaced": True,
                        }
                        for w in scene_jobs[1]["words"]
                    ],
                }
            return res

    orig_get_aligner = t_module.get_timeline_aligner
    t_module.get_timeline_aligner = lambda name, cfg: ErrorProneFakeAligner()
    try:
        agent = TimelineAgent(mock_timeline_run_dir)
        agent.run(aligner_override="fake", no_preview=True)
    finally:
        t_module.get_timeline_aligner = orig_get_aligner

    tl_path = mock_timeline_run_dir / "timeline" / "timeline.json"
    tl_data = json.loads(tl_path.read_text(encoding="utf-8"))

    second_sc = tl_data["scenes"][1]
    # All words in the failed scene should be gracefully interpolated
    for w in second_sc["words"]:
        assert w["start"] is not None
        assert w["end"] is not None
        assert w["unplaced"] is True

    assert tl_data["degraded"] is True


def test_alignment_cache_hit(mock_timeline_run_dir: Path):
    """3.1 Two-level cache: Second run of identical inputs hits cache and skips model alignment."""
    import src.agents.timeline as t_module
    from src.utils.timeline import FakeAligner

    call_count = 0

    class CountingFakeAligner(FakeAligner):
        def align_scenes(self, scene_jobs, settings=None):
            nonlocal call_count
            call_count += 1
            return super().align_scenes(scene_jobs, settings=settings or {})

    orig_get_aligner = t_module.get_timeline_aligner
    t_module.get_timeline_aligner = lambda name, cfg: CountingFakeAligner()

    try:
        agent = TimelineAgent(mock_timeline_run_dir)

        # Run 1: Cold run, cache is populated
        agent.run(aligner_override="fake", no_preview=True)
        assert call_count == 1
        assert agent.last_metrics["cached"] is False
        assert agent.last_metrics["cache_misses"] > 0

        # Verify cache files were created
        cache_dir = mock_timeline_run_dir / "timeline" / "cache" / "alignment"
        assert cache_dir.exists()
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == agent.last_metrics["cache_misses"]

        # Run 2: Warm run, should hit cache completely
        agent.run(aligner_override="fake", no_preview=True)
        # Aligner.align_scenes should NOT have been called again!
        assert call_count == 1
        assert agent.last_metrics["cached"] is True
        assert agent.last_metrics["cache_hits"] == len(cache_files)
        assert agent.last_metrics["cache_misses"] == 0

        # Run 3: Forced no-cache run, should re-align
        agent.run(aligner_override="fake", no_preview=True, no_cache=True)
        assert call_count == 2
        assert agent.last_metrics["cached"] is False

    finally:
        t_module.get_timeline_aligner = orig_get_aligner


def test_audit_cli_mode(mock_timeline_run_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """3.2 CLI mode: --audit prints quality summary of existing run without re-aligning."""
    from src.agents.timeline import main

    agent = TimelineAgent(mock_timeline_run_dir)
    agent.run(aligner_override="fake", no_preview=True)

    # Call audit_existing_run directly
    audit_report = agent.audit_existing_run()
    assert "summary" in audit_report
    assert audit_report["summary"]["passed"] is True
    assert audit_report["summary"]["total_captions"] > 0

    # Test CLI invocation with --audit
    monkeypatch.setattr(
        "sys.argv",
        [
            "timeline.py",
            "--runs-dir", str(mock_timeline_run_dir.parent),
            "--date", mock_timeline_run_dir.name,
            "--audit",
        ],
    )
    exit_code = main()
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "=== TIMELINE CAPTION AUDIT REPORT ===" in captured.out
    assert "Audit Passed:" in captured.out
    assert "Reading Speed (CPS):" in captured.out


def test_preview_video_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """3.2 Preview: generate_timeline_preview_video uses fast preset and 10 fps."""
    import subprocess
    from src.utils.timeline import generate_timeline_preview_video

    recorded_cmd = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        nonlocal recorded_cmd
        recorded_cmd = list(cmd)
        # Create dummy file to simulate ffmpeg output
        out_name = cmd[-1]
        out_path = Path(kwargs.get("cwd", tmp_path)) / out_name
        out_path.write_bytes(b"dummy_mp4_bytes")
        class DummyProc:
            returncode = 0
            stderr = ""
        return DummyProc()

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    dummy_narration = tmp_path / "audio" / "narration.wav"
    dummy_narration.parent.mkdir(parents=True, exist_ok=True)
    dummy_narration.write_bytes(b"dummy")

    dummy_srt = tmp_path / "timeline" / "captions.srt"
    dummy_srt.parent.mkdir(parents=True, exist_ok=True)
    dummy_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n", encoding="utf-8")

    out_mp4 = tmp_path / "timeline" / "timeline_preview.mp4"

    res = generate_timeline_preview_video(
        narration_path=dummy_narration,
        srt_path=dummy_srt,
        output_mp4=out_mp4,
        fps=10,
        preset="veryfast",
    )

    assert res["file"] == str(out_mp4)
    # Check ffmpeg command options
    assert "-preset" in recorded_cmd
    preset_idx = recorded_cmd.index("-preset")
    assert recorded_cmd[preset_idx + 1] == "veryfast"

    # Check fps in color lavfi input
    color_in = [arg for arg in recorded_cmd if "color=" in arg]
    assert len(color_in) > 0
    assert "r=10" in color_in[0]


def test_multilingual_devanagari_transliteration():
    """4.1 Multilingual reality: Devanagari fallback transliterator generates Latin tokens for MMS."""
    from src.utils.text_normalizer import devanagari_to_roman_fallback, tokenize_for_alignment

    nepali_phrase = "काठमाडौं र पोखरामा भारी वर्षा भयो।"
    romanized = devanagari_to_roman_fallback(nepali_phrase)
    # Check that output is ASCII Latin characters
    assert any(c in "abcdefghijklmnopqrstuvwxyz" for c in romanized.lower())

    warnings: list[str] = []
    tokens = tokenize_for_alignment(nepali_phrase, warnings_out=warnings)
    assert len(tokens) >= 4
    # All tokens should have non-empty MMS tokens
    for tok in tokens:
        assert len(tok["tokens"]) > 0


def test_multilingual_audit_and_warnings(mock_timeline_run_dir: Path):
    """4.1 Multilingual reality: Language-driven reporting in caption_audit."""
    script_file = mock_timeline_run_dir / "script" / "script.json"
    s_data = json.loads(script_file.read_text(encoding="utf-8"))
    s_data["language"] = "Nepali"
    s_data["scenes"][0]["narration"] = "काठमाडौंमा नयाँ योजना सुरु भयो।"
    script_file.write_text(json.dumps(s_data, indent=2), encoding="utf-8")

    agent = TimelineAgent(mock_timeline_run_dir)
    agent.run(aligner_override="fake", no_preview=True)

    audit_path = mock_timeline_run_dir / "timeline" / "caption_audit.json"
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))

    m_status = audit_data["summary"].get("multilingual_status", {})
    assert m_status.get("language") == "Nepali"
    assert m_status.get("devanagari_detected") is True


def test_verbatim_caption_check():
    """4.2 Optional: Verbatim word conservation check between scenes and captions."""
    from src.utils.timeline import audit_captions

    scenes = [{
        "id": "intro",
        "words": [
            {"w": "Hello", "start": 0.0, "end": 0.5},
            {"w": "world", "start": 0.5, "end": 1.0},
        ],
    }]

    # Exact matching captions
    good_captions = [{
        "id": 1,
        "scene_id": "intro",
        "start": 0.0,
        "end": 1.0,
        "text": "Hello world",
        "words": [{"w": "Hello", "start": 0.0, "end": 0.5}, {"w": "world", "start": 0.5, "end": 1.0}],
    }]
    audit_good = audit_captions(good_captions, scenes)
    assert audit_good["summary"]["verbatim_conserved"] is True
    assert audit_good["summary"]["word_conservation_errors"] == 0

    # Caption missing a word
    dropped_captions = [{
        "id": 1,
        "scene_id": "intro",
        "start": 0.0,
        "end": 1.0,
        "text": "Hello",
        "words": [{"w": "Hello", "start": 0.0, "end": 0.5}],
    }]
    audit_bad = audit_captions(dropped_captions, scenes)
    assert audit_bad["summary"]["verbatim_conserved"] is False
    assert audit_bad["summary"]["word_conservation_errors"] > 0








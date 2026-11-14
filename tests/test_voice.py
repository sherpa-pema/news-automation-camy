"""Unit tests for the Voice Agent and TTS utilities.

Runs purely with FakeTTS and mock components. No model weights or .venv-tts required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import wave

from src.agents.voice import (
    VoiceAgent,
    clean_narration_text,
    split_into_sentences,
    check_audio_silence_gaps,
)
from src.utils.tts import (
    FakeTTS,
    assemble_scene_audio,
    create_silence_wav,
    read_wav_info,
)


@pytest.fixture
def sample_script_dict() -> dict[str, Any]:
    with open("tests/fixtures/sample_script.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def mock_run_dir(tmp_path: Path, sample_script_dict: dict[str, Any]) -> Path:
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_file = script_dir / "script.json"
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(sample_script_dict, f, indent=2)
    return tmp_path


def test_pronunciation_dictionary_applied():
    """Case-insensitive whole-word pronunciation replacement."""
    pronunciations = {
        "Birgunj": "Beer-gunj",
        "Bagmati": "Bahg-mah-tee",
        "Sindhupalchok": "Sin-dhoo-pal-chowk",
    }
    raw = "The truck departed from birgunj near the bagmati river, crossing through SINDHUPALCHOK safely."
    cleaned = clean_narration_text(raw, pronunciations)

    assert "Beer-gunj" in cleaned
    assert "birgunj" not in cleaned.lower()
    assert "Bahg-mah-tee" in cleaned
    assert "Sin-dhoo-pal-chowk" in cleaned

    # Whole-word check: 'Birgunjport' should NOT be replaced
    not_whole_word = "Birgunjport terminal"
    result = clean_narration_text(not_whole_word, pronunciations)
    assert result == "Birgunjport terminal"


def test_narration_cleaning_and_bracket_tag_stripping():
    """Only narration is read, and paralinguistic tags like [Sigh] are stripped."""
    text = "Good evening [Sigh]. Today in Kathmandu [Laughter], temperatures dropped [Breathing]."
    cleaned = clean_narration_text(text)
    assert "[Sigh]" not in cleaned
    assert "[Laughter]" not in cleaned
    assert "[Breathing]" not in cleaned
    assert cleaned == "Good evening. Today in Kathmandu, temperatures dropped."


def test_sentence_splitting():
    """Sentence splitting handles punctuation and protects abbreviations."""
    text = "Dr. Sharma reported the findings today. The team, including Mr. Karki, agreed! Did it work? Yes."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Dr. Sharma reported the findings today."
    assert sentences[1] == "The team, including Mr. Karki, agreed!"
    assert sentences[2] == "Did it work?"
    assert sentences[3] == "Yes."


def test_padding_and_pause_insertion(tmp_path: Path):
    """Verifies 0.25s pause between sentences and pad_before (0.4s) / pad_after (0.6s)."""
    fake = FakeTTS(sample_rate=44100, words_per_sec=10.0)

    s1_path = tmp_path / "s1.wav"
    s2_path = tmp_path / "s2.wav"
    fake.synthesize_batch(
        [
            {"id": "s1", "text": "Sentence one text.", "out_path": str(s1_path)},
            {"id": "s2", "text": "Sentence two text here.", "out_path": str(s2_path)},
        ],
        {},
    )

    dur1 = read_wav_info(s1_path)["duration_sec"]
    dur2 = read_wav_info(s2_path)["duration_sec"]

    out_scene = tmp_path / "scene.wav"
    scene_dur = assemble_scene_audio(
        [s1_path, s2_path],
        out_scene,
        pause_sec=0.25,
        pad_before=0.4,
        pad_after=0.6,
        target_sample_rate=44100,
    )

    expected = round(0.4 + dur1 + 0.25 + dur2 + 0.6, 3)
    assert abs(scene_dur - expected) < 0.05


def test_scene_durations_json_matches_concatenated_length(mock_run_dir: Path):
    """scene_durations.json sum matches raw concatenated audio length."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={"stages": [{"id": "voice", "engine": "fake"}]},
    )

    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")

    durations_file = mock_run_dir / "audio" / "scene_durations.json"
    assert durations_file.exists()

    with open(durations_file, "r", encoding="utf-8") as f:
        durations = json.load(f)

    sum_scene_durations = sum(d["duration_sec"] for d in durations)

    # Check that segments are contiguous
    for i in range(len(durations) - 1):
        assert abs(durations[i]["end_sec"] - durations[i + 1]["start_sec"]) < 0.001

    assert abs(durations[-1]["end_sec"] - sum_scene_durations) < 0.01


def test_worker_job_result_roundtrip_and_failure_handling(tmp_path: Path):
    """Tests FakeTTS batch handling and failure simulation."""
    fake = FakeTTS()
    items = [
        {"id": "good_1", "text": "Normal text sentence.", "out_path": str(tmp_path / "good1.wav")},
        {"id": "bad_1", "text": "Failing text [FAIL_PERMANENT].", "out_path": str(tmp_path / "bad1.wav")},
        {"id": "good_2", "text": "Another normal sentence.", "out_path": str(tmp_path / "good2.wav")},
    ]

    result = fake.synthesize_batch(items, {})
    results = result["results"]
    assert len(results) == 3
    assert results[0]["status"] == "success"
    assert results[1]["status"] == "failed"
    assert results[1]["error"] == "Simulated permanent failure"
    assert results[2]["status"] == "success"


def test_validation_catches_empty_scene(mock_run_dir: Path):
    """Validation fails if a scene audio file is empty."""
    agent = VoiceAgent(run_dir=mock_run_dir)
    audio_dir = mock_run_dir / "audio"
    scenes_dir = audio_dir / "scenes"
    audio_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Valid narration dummy
    create_silence_wav(audio_dir / "narration.wav", duration_sec=120.0, sample_rate=48000)

    # Empty scene file
    empty_file = scenes_dir / "intro.wav"
    empty_file.write_bytes(b"")

    is_valid, err = agent.validate(["audio/narration.wav", "audio/scenes/intro.wav"])
    assert not is_valid
    assert "empty or too small" in err


def test_validation_catches_silence_gap(tmp_path: Path):
    """Validation catches silence gap > 1.5 seconds inside a scene."""
    wav_path = tmp_path / "silence_gap.wav"
    # Create 44.1kHz mono WAV with 2.0s tone, 2.0s silence, 1.0s tone
    sr = 44100
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        # 2s tone
        import math, struct
        frames = bytearray()
        for i in range(sr * 2):
            val = int(8000 * math.sin(2.0 * math.pi * 440.0 * (i / sr)))
            frames.extend(struct.pack("<h", val))
        # 2s silence (> 1.5s threshold)
        frames.extend(b"\x00\x00" * (sr * 2))
        # 1s tone
        for i in range(sr):
            val = int(8000 * math.sin(2.0 * math.pi * 440.0 * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)

    max_silence = check_audio_silence_gaps(wav_path)
    assert max_silence >= 1.9


def test_speaker_consistency_gate_passes_and_reports(mock_run_dir: Path):
    """VoiceAgent records speaker consistency metrics and validation passes when within bounds."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={"stages": [{"id": "voice", "engine": "fake", "similarity_threshold": 0.72}]},
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation failed: {err}"

    report_path = mock_run_dir / "audio" / "voice_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "speaker_consistency" in data
    assert data["speaker_consistency"]["similarity_threshold"] == 0.72


def test_speaker_consistency_gate_fails_on_low_similarity(mock_run_dir: Path):
    """Validation fails if an item falls below the speaker embedding similarity threshold."""
    script_file = mock_run_dir / "script" / "script.json"
    with open(script_file, "r", encoding="utf-8") as f:
        script = json.load(f)
    # Inject low similarity trigger into first scene
    script["scenes"][0]["narration"] = "Good evening TRIGGER_LOW_SIM. Welcome to the broadcast."
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2)

    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={"stages": [{"id": "voice", "engine": "fake", "similarity_threshold": 0.72}]},
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert not is_valid
    assert "Speaker consistency gate failed" in err
    assert "0.5000 < threshold 0.7200" in err


def test_speaker_consistency_gate_fails_on_pitch_deviation(mock_run_dir: Path):
    """Validation fails if median F0 deviates beyond f0_tolerance_pct."""
    script_file = mock_run_dir / "script" / "script.json"
    with open(script_file, "r", encoding="utf-8") as f:
        script = json.load(f)
    script["scenes"][0]["narration"] = "Good evening TRIGGER_HIGH_PITCH. Welcome to the broadcast."
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2)

    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "similarity_threshold": 0.72,
                    "f0_tolerance_pct": 0.25,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")

    # Update report with reference F0 = 216.0 to trigger comparison
    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    rep["speaker_consistency"]["reference_f0_hz"] = 216.0
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    is_valid, err = agent.validate(output_files)
    assert not is_valid
    assert "Speaker pitch gate failed" in err
    assert "350.0Hz differs from reference 216.0Hz" in err


def test_pronunciation_verification_tags_and_unverified_tracking(tmp_path):
    """Verify pronunciation dictionary supports verified boolean and tracks unverified applications."""
    from src.agents.voice import load_pronunciations, load_pronunciations_with_metadata, get_unverified_pronunciations_applied

    test_yaml = tmp_path / "pronunciations.yaml"
    test_yaml.write_text(
        """
pronunciations:
  Birgunj:
    respelling: "Beer-gunj"
    verified: true
  MBBS:
    respelling: "M B B S"
    verified: false
  Lumbini:
    respelling: "Loom-bee-nee"
    verified: false
""",
        encoding="utf-8",
    )

    # 1. Plain dictionary mapping returns string respellings
    plain = load_pronunciations(test_yaml)
    assert plain["Birgunj"] == "Beer-gunj"
    assert plain["MBBS"] == "M B B S"
    assert plain["Lumbini"] == "Loom-bee-nee"

    # 2. Metadata returns verified boolean
    meta = load_pronunciations_with_metadata(test_yaml)
    assert meta["Birgunj"]["verified"] is True
    assert meta["MBBS"]["verified"] is False
    assert meta["Lumbini"]["verified"] is False

    # 3. get_unverified_pronunciations_applied identifies unverified words in scenes
    scenes = [
        {"id": "scene_01", "narration": "Students registered for the MBBS course in Birgunj."},
        {"id": "scene_02", "narration": "Lumbini hosted thousands of visitors today."},
    ]
    unverified_applied = get_unverified_pronunciations_applied(scenes, test_yaml)
    words_applied = {item["word"] for item in unverified_applied}
    assert "MBBS" in words_applied
    assert "Lumbini" in words_applied
    assert "Birgunj" not in words_applied  # Because Birgunj is verified: true


def test_voice_agent_run_approval_fingerprint_gate(tmp_path):
    """VoiceAgent.run must execute the approval and validator fingerprint gate without NameError."""
    import hashlib
    from src.agents.voice import VoiceAgent
    from src.utils.editor_validator import compute_validator_fingerprint

    run_dir = tmp_path / "2026-11-08"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)

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

    manifest = {"stages": {"voice": {"status": "pending", "retry_count": 0, "input_files": [], "output_files": []}}}

    agent = VoiceAgent(run_dir)
    # The approval & fingerprint gate executes first (verifying hashes & fingerprint without NameError).
    # Since scenes is empty, it proceeds past the gate and raises ValueError for no scenes.
    with pytest.raises(ValueError, match="contains no scenes to synthesize"):
        agent.run(manifest)


def test_longest_first_pronunciation_replacement():
    """Longer multi-word pronunciation entries match before shorter substrings."""
    pronunciations = {
        "Rai": "Rye",
        "Sarita Rai": "Sa-ree-ta Rye",
    }
    cleaned = clean_narration_text("Reporter Sarita Rai spoke with Mr. Rai today.", pronunciations)
    assert "Sa-ree-ta Rye" in cleaned
    assert "Mr. Rye" in cleaned


def test_assemble_scene_audio_resamples_mismatched_rate(tmp_path: Path):
    """assemble_scene_audio converts WAVs whose sample rate differs from target_sample_rate."""
    fake_24k = FakeTTS(sample_rate=24000, words_per_sec=5.0)
    s1 = tmp_path / "s_24k.wav"
    fake_24k.synthesize_batch([{"id": "s1", "text": "Five words for timing check.", "out_path": str(s1)}], {})
    dur_in = read_wav_info(s1)["duration_sec"]

    out = tmp_path / "scene_44k.wav"
    dur_out = assemble_scene_audio([s1], out, pad_before=0.2, pad_after=0.3, target_sample_rate=44100)
    info_out = read_wav_info(out)
    assert info_out["sample_rate"] == 44100
    assert info_out["channels"] == 1
    assert abs(dur_out - (dur_in + 0.5)) < 0.08


def test_scene_content_hash_cache_hit_and_invalidation(mock_run_dir: Path, monkeypatch):
    """Unchanged scenes reuse cached WAVs; modified narration invalidates only that scene."""
    from src.utils import tts as tts_mod

    synth_calls: list[list[str]] = []
    orig_fake = FakeTTS()

    class SpyFakeTTS(FakeTTS):
        def synthesize_batch(self, items, settings):
            synth_calls.append([it["id"] for it in items])
            res = orig_fake.synthesize_batch(items, settings)
            res["reference_f0_hz"] = 216.0
            return res

    monkeypatch.setattr(tts_mod, "get_tts_engine", lambda name, cfg=None: SpyFakeTTS())
    import src.agents.voice as voice_mod
    monkeypatch.setattr(voice_mod, "get_tts_engine", lambda name, cfg=None: SpyFakeTTS())

    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={"stages": [{"id": "voice", "engine": "fake", "target_lufs": -16.0}]},
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    agent.run(manifest, engine_override="fake")
    assert len(synth_calls) == 1
    first_count = len(synth_calls[0])
    assert first_count > 1

    # Verify reference_f0_hz & top-level loudness propagated to voice_report.json
    rep = json.loads((mock_run_dir / "audio" / "voice_report.json").read_text(encoding="utf-8"))
    assert rep["speaker_consistency"]["reference_f0_hz"] == 216.0
    assert rep["loudness_after_lufs"] is not None
    assert rep["true_peak_after_db"] is not None

    # Second run without script changes should hit cache for all scenes (0 synthesis calls)
    agent.run(manifest, engine_override="fake")
    assert len(synth_calls) == 1

    # Edit only one scene -> only that single scene is re-synthesized
    script_file = mock_run_dir / "script" / "script.json"
    script_data = json.loads(script_file.read_text(encoding="utf-8"))
    edited_id = script_data["scenes"][0]["id"]
    script_data["scenes"][0]["narration"] += " Extra breaking update."
    script_file.write_text(json.dumps(script_data, indent=2), encoding="utf-8")

    agent.run(manifest, engine_override="fake")
    assert len(synth_calls) == 2
    assert synth_calls[1] == [edited_id]


def test_ming_engine_daemon_and_fallback(tmp_path: Path):
    """MingEngine uses persistent HTTP daemon when reachable and falls back cleanly if unreachable."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    from src.utils.tts import MingEngine

    received = []

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            received.append(body)
            resp = json.dumps({"status": "success", "model_load_time_sec": 0.0, "results": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        eng = MingEngine(
            python_path=tmp_path / "missing_python.exe",
            daemon_url=f"http://127.0.0.1:{port}",
        )
        res = eng.synthesize_batch([{"id": "s1", "text": "Hello.", "out_path": str(tmp_path / "s1.wav")}], {})
        assert res["status"] == "success"
        assert len(received) == 1
        assert received[0]["crossfade_ms"] == 12.0
    finally:
        server.shutdown()
        server.server_close()

    # When daemon is down, it falls back to subprocess path (which raises FileNotFoundError for missing python)
    eng_down = MingEngine(
        python_path=tmp_path / "missing_python.exe",
        daemon_url=f"http://127.0.0.1:{port}",
    )
    with pytest.raises(FileNotFoundError):
        eng_down.synthesize_batch([{"id": "s1", "text": "Hello.", "out_path": str(tmp_path / "s1.wav")}], {})




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
        def do_GET(self):
            if self.path == "/health":
                resp = json.dumps({
                    "status": "ok",
                    "model_path": str(Path("models/Ming-3.0-tts-0.5b").resolve()),
                    "device": "cuda:0",
                    "dtype": "bfloat16",
                    "text_normalization": False,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_response(404)
                self.end_headers()

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


def test_voice_report_has_quality_and_perf_blocks(mock_run_dir: Path):
    """VoiceAgent creates voice_report.json containing quality (VQI) and perf blocks."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={"stages": [{"id": "voice", "engine": "fake"}]},
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    assert "audio/voice_report.json" in output_files

    report_path = mock_run_dir / "audio" / "voice_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Quality block (VQI)
    assert "quality" in data
    q = data["quality"]
    for key in [
        "cosine_min", "cosine_max", "cosine_mean", "cosine_std",
        "f0_min_hz", "f0_max_hz", "f0_spread_pct", "reference_f0_hz",
        "scene_lufs_min", "scene_lufs_max", "scene_lufs_spread",
        "wps_mean", "wps_variance", "wps_std",
        "dead_air_duration_sec", "dead_air_ratio",
        "artifact_flags", "unverified_pronunciations_applied",
    ]:
        assert key in q, f"Missing VQI key: {key}"

    # 2. Perf block
    assert "perf" in data
    p = data["perf"]
    for key in [
        "model_load_time_sec", "synthesis_wall_time_sec", "assembly_time_sec",
        "mastering_time_sec", "total_wall_time_sec", "real_time_factor",
        "stage_breakdown_sec",
    ]:
        assert key in p, f"Missing perf key: {key}"

    stages = p["stage_breakdown_sec"]
    for s_key in ["model_load", "prefill", "ar_step", "cfm", "vae_decode", "qc", "write"]:
        assert s_key in stages, f"Missing stage breakdown key: {s_key}"


def test_compute_voice_quality_index_matches_table_section_1():
    """VQI reproduces the table in Section 1 on 2026-11-15 run within tolerance."""
    from src.utils.tts import compute_voice_quality_index

    run_dir = Path("data/runs/2026-11-15")
    baseline_path = run_dir / "audio" / "voice_report_baseline.json"
    target_path = baseline_path if baseline_path.exists() else (run_dir / "audio" / "voice_report.json")
    if not target_path.exists():
        pytest.skip("2026-11-15 run not present on disk")

    with open(target_path, encoding="utf-8") as f:
        rep = json.load(f)
    with open(run_dir / "script" / "script.json", encoding="utf-8") as f:
        script = json.load(f)

    words_map = {sc["id"]: len(sc.get("narration", "").split()) for sc in script["scenes"]}
    scenes_for_vqi = [{**s, "word_count": words_map.get(s["id"], 0)} for s in rep["scenes"]]

    q = compute_voice_quality_index(
        consistency_metrics=rep["speaker_consistency"]["metrics"],
        reference_f0_hz=rep["speaker_consistency"]["reference_f0_hz"],
        scenes_info=scenes_for_vqi,
        total_duration_sec=rep["total_audio_duration_sec"],
        pad_per_scene_sec=1.0,
    )

    # Section 1 Table Assertions:
    # 1. Cosine: 0.828 - 0.886 (spread ~0.058, min >= 0.82)
    assert 0.82 <= q["cosine_min"] <= 0.835
    assert 0.88 <= q["cosine_max"] <= 0.895
    assert q["cosine_std"] <= 0.02

    # 2. F0: 204.2 - 242.3 Hz (~18.7% spread ~ 19%)
    assert 200.0 <= q["f0_min_hz"] <= 206.0
    assert 240.0 <= q["f0_max_hz"] <= 245.0
    assert 18.0 <= q["f0_spread_pct"] <= 19.5

    # 3. Mastered scene LUFS spread <= 1.0 LU
    assert q["scene_lufs_spread"] <= 1.0

    # 4. Dead air ratio: ~10% (21.0s / 204.8s)
    assert 0.09 <= q["dead_air_ratio"] <= 0.11


def test_voice_bench_fake_engine_and_compare(tmp_path: Path):
    """voice_bench runs with fake engine and formats side-by-side comparison."""
    from tools.voice_bench import run_benchmark, compare_reports, format_scoreboard

    # Create dummy run_dir with script.json
    run_dir = tmp_path / "mock_run"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_data = {
        "scenes": [
            {"id": "intro", "narration": "Hello from intro scene."},
            {"id": "scene_01", "narration": "First scene breaking news update."},
            {"id": "scene_02", "narration": "Second scene closing news report."},
        ]
    }
    (script_dir / "script.json").write_text(json.dumps(script_data), encoding="utf-8")

    out_a = tmp_path / "bench_a"
    rep_a = run_benchmark(run_dir, ["intro", "scene_01", "scene_02"], out_a, engine_name="fake")
    assert (out_a / "bench_report.json").exists()
    assert "quality" in rep_a
    assert "perf" in rep_a

    out_b = tmp_path / "bench_b"
    rep_b = run_benchmark(run_dir, ["intro", "scene_01", "scene_02"], out_b, engine_name="fake")

    # Format scoreboard
    scoreboard = format_scoreboard(rep_a)
    assert "Voice Quality & Performance Scoreboard" in scoreboard
    assert "Speaker Cosine Min" in scoreboard

    # Compare reports
    cmp_table = compare_reports(rep_a, rep_b, label_a="RunA", label_b="RunB")
    assert "Side-by-Side Comparison: [RunA] vs [RunB]" in cmp_table
    assert "Real-Time Factor (RTF)" in cmp_table


def test_compute_candidate_score_formula():
    """compute_candidate_score correctly computes weighted VQI score across all terms."""
    from src.utils.tts import compute_candidate_score

    # Exact match: cosine=1.0, f0 matches target, duration matches expected, no artifacts
    score, breakdown = compute_candidate_score(
        cosine_sim=1.0,
        cand_f0=216.0,
        target_f0=216.0,
        duration_sec=10.0,
        expected_duration_sec=10.0,
        artifact_score=1.0,
        weights={"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2},
    )
    # 1.0*1.0 + 0.6*1.0 + 0.2*1.0 + 0.2*1.0 = 2.0
    assert abs(score - 2.0) < 1e-4
    assert breakdown["cosine_term"] == 1.0
    assert breakdown["f0_term"] == 1.0
    assert breakdown["duration_term"] == 1.0
    assert breakdown["artifact_term"] == 1.0

    # Half pitch deviation: cand_f0 is 10% off -> f0_term = 0.90
    score2, breakdown2 = compute_candidate_score(
        cosine_sim=0.90,
        cand_f0=237.6,
        target_f0=216.0,
        duration_sec=10.0,
        expected_duration_sec=10.0,
        artifact_score=1.0,
        weights={"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2},
    )
    # 1.0*0.9 + 0.6*0.9 + 0.2*1.0 + 0.2*1.0 = 0.9 + 0.54 + 0.2 + 0.2 = 1.84
    assert abs(score2 - 1.84) < 1e-3
    assert abs(breakdown2["f0_term"] - 0.90) < 1e-3


def test_candidate_ranker_selects_higher_scoring_candidate(mock_run_dir: Path):
    """VoiceAgent with n_candidates=2 selects the higher-scoring candidate and records breakdown."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "n_candidates": 2,
                    "f0_anchor": "running_median",
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation failed: {err}"

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    spk_data = rep["speaker_consistency"]
    assert spk_data["n_candidates"] == 2
    assert spk_data["f0_anchor"] == "running_median"

    # In FakeTTS default simulation for n_candidates=2:
    # candidate 0: cosine 0.82, candidate 1: cosine 0.88 -> candidate 1 scores higher
    metrics = spk_data["metrics"]
    for item_id, m in metrics.items():
        assert m["selected_candidate_idx"] == 1
        assert "ranker_score" in m
        assert "ranker_breakdown" in m
        assert len(m["candidates_evaluated"]) == 2


def test_candidate_ranker_prefers_candidate_0_when_scored_higher(mock_run_dir: Path):
    """VoiceAgent with TRIGGER_PREFER_FIRST selects candidate 0."""
    script_file = mock_run_dir / "script" / "script.json"
    with open(script_file, "r", encoding="utf-8") as f:
        script = json.load(f)
    script["scenes"][0]["narration"] = "TRIGGER_PREFER_FIRST News intro broadcast."
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2)

    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "n_candidates": 2,
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    assert (mock_run_dir / "audio" / "narration.wav").exists()

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # First scene had TRIGGER_PREFER_FIRST, so candidate 0 must win
    first_scene_metric = rep["speaker_consistency"]["metrics"]["intro"]
    assert first_scene_metric["selected_candidate_idx"] == 0


def test_n_candidates_1_preserves_single_attempt_behavior(mock_run_dir: Path):
    """VoiceAgent with n_candidates=1 evaluates a single candidate per scene."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "n_candidates": 1,
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation failed: {err}"

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    spk_data = rep["speaker_consistency"]
    assert spk_data["n_candidates"] == 1
    for item_id, m in spk_data["metrics"].items():
        assert m["selected_candidate_idx"] == 0
        assert len(m["candidates_evaluated"]) == 1


def test_get_scene_padding_broadcast_policy():
    """get_scene_padding implements broadcast padding policy for intro, outro, and interior scenes."""
    from src.utils.tts import get_scene_padding

    # Broadcast policy:
    # 1. Intro
    p_before, p_after = get_scene_padding("intro", scene_idx=0, total_scenes=5, padding_policy="broadcast")
    assert p_before == 0.40
    assert p_after == 0.15

    # 2. Interior scene
    p_before, p_after = get_scene_padding("scene_02", scene_idx=2, total_scenes=5, padding_policy="broadcast")
    assert p_before == 0.05
    assert p_after == 0.15

    # 3. Outro
    p_before, p_after = get_scene_padding("outro", scene_idx=4, total_scenes=5, padding_policy="broadcast")
    assert p_before == 0.05
    assert p_after == 0.60

    # Fixed fallback policy
    p_before, p_after = get_scene_padding("scene_02", scene_idx=2, total_scenes=5, padding_policy="fixed")
    assert p_before == 0.4
    assert p_after == 0.6


def test_equal_power_micro_fade():
    """_apply_micro_fade_pcm16 applies equal-power curve to PCM frames cleanly."""
    from src.utils.tts import _apply_micro_fade_pcm16
    import struct

    sr = 44100
    total_samples = 4410  # 100ms
    # Constant non-zero amplitude
    frames = struct.pack(f"<{total_samples}h", *([10000] * total_samples))

    faded = _apply_micro_fade_pcm16(frames, sr, fade_ms=10.0)
    samples = struct.unpack(f"<{total_samples}h", faded)

    # First sample should be near 0
    assert abs(samples[0]) < 100
    # Center sample should be unchanged
    assert samples[total_samples // 2] == 10000
    # Last sample should be near 0
    assert abs(samples[-1]) < 100


def test_voice_agent_broadcast_padding_policy_dead_air(mock_run_dir: Path):
    """VoiceAgent with broadcast padding policy achieves dead air ratio <= 3.0%."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "padding_policy": "broadcast",
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation failed: {err}"

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    q = rep.get("quality", {})
    # Dead air ratio target <= 3.0%
    assert q["dead_air_ratio"] <= 0.03
    assert rep["speaker_consistency"]["padding_policy"] == "broadcast"


def test_master_audio_loudnorm_linear_strict_success(tmp_path: Path):
    """master_audio_loudnorm achieves -16.0 +/- 0.3 LUFS in linear mode with true-peak <= -1.5 dBFS."""
    from src.utils.tts import master_audio_loudnorm, concatenate_scenes

    source_wav = Path("data/benchmarks/candidate_3/narration_master.wav")
    if not source_wav.exists():
        pytest.skip("Benchmark audio not found")

    out_master = tmp_path / "test_master_linear.wav"
    res = master_audio_loudnorm(
        input_wav=source_wav,
        output_wav=out_master,
        target_lufs=-16.0,
        true_peak=-1.5,
        sample_rate=48000,
        highpass_hz=80.0,
        compressor=True,
        limiter=True,
        limiter_ceiling_dbfs=-1.5,
        loudness_strategy="linear_strict",
    )

    assert res["fallback_used"] is False
    assert res["loudnorm_mode"] == "linear"
    assert res["after_lufs"] == pytest.approx(-16.0, abs=0.3)
    assert res["after_true_peak"] <= -1.45  # True peak ceiling -1.5 dBFS with 0.05 dB margin
    assert res["output_sample_rate"] == 48000


def test_master_audio_loudnorm_linear_strict_refuses_fallback(tmp_path: Path):
    """master_audio_loudnorm under linear_strict raises RuntimeError when linear target cannot be met."""
    from src.utils.tts import master_audio_loudnorm

    source_wav = Path("data/benchmarks/candidate_3/narration_master.wav")
    if not source_wav.exists():
        pytest.skip("Benchmark audio not found")

    out_master = tmp_path / "test_refuse.wav"
    # Unachievable ceiling with compressor disabled and target_lufs pushed to -10 LUFS
    with pytest.raises(RuntimeError, match="Refusing fallback under loudness_strategy='linear_strict'"):
        master_audio_loudnorm(
            input_wav=source_wav,
            output_wav=out_master,
            target_lufs=-10.0,
            true_peak=-6.0,
            sample_rate=48000,
            highpass_hz=80.0,
            compressor=False,
            limiter=False,
            loudness_strategy="linear_strict",
        )


def test_voice_agent_linear_strict_default(mock_run_dir: Path):
    """VoiceAgent defaults to linear_strict mastering and achieves -16 LUFS linear mode."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "target_lufs": -16.0,
                    "target_scene_rms_dbfs": -23.0,
                    "loudness_strategy": "linear_strict",
                    "limiter_ceiling_dbfs": -1.5,
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation failed: {err}"

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    loud = rep["loudnorm"]
    assert loud["loudness_strategy"] == "linear_strict"
    assert loud["fallback_used"] is False
    assert loud["loudnorm_mode"] == "linear"
    assert loud["after_lufs"] == pytest.approx(-16.0, abs=0.5)
    assert loud["after_true_peak"] <= -1.45


def test_unverified_pronunciation_gate_blocks_run(mock_run_dir: Path):
    """VoiceAgent.validate() blocks runs when unverified pronunciations are applied."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "target_lufs": -16.0,
                    "target_scene_rms_dbfs": -23.0,
                    "similarity_threshold": 0.72,
                    "allow_unverified_pronunciations": False,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")

    # Manually inject an unverified applied pronunciation into voice_report.json
    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    rep["unverified_pronunciations_applied"] = [
        {"word": "UnverifiedEntity", "respelling": "Un-ver-i-fied", "scene_id": "intro"}
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    is_valid, err = agent.validate(output_files)
    assert not is_valid
    assert "Unverified pronunciation gate failed" in err
    assert "UnverifiedEntity" in err


def test_unverified_pronunciation_gate_passes_when_allowed(mock_run_dir: Path):
    """VoiceAgent.validate() passes when allow_unverified_pronunciations: true is set."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "target_lufs": -16.0,
                    "target_scene_rms_dbfs": -23.0,
                    "similarity_threshold": 0.72,
                    "allow_unverified_pronunciations": True,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")

    # Manually inject an unverified applied pronunciation into voice_report.json
    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    rep["unverified_pronunciations_applied"] = [
        {"word": "UnverifiedEntity", "respelling": "Un-ver-i-fied", "scene_id": "intro"}
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Validation should pass when allow_unverified_pronunciations is True, got: {err}"


def test_proper_noun_discovery_in_voice_report(mock_run_dir: Path):
    """VoiceAgent populates pending_proper_nouns in voice_report.json."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "target_lufs": -16.0,
                    "target_scene_rms_dbfs": -23.0,
                    "similarity_threshold": 0.72,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")

    report_path = mock_run_dir / "audio" / "voice_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert "pending_proper_nouns" in rep
    assert isinstance(rep["pending_proper_nouns"], list)
    assert "unverified_pronunciations_applied" in rep
    assert isinstance(rep["unverified_pronunciations_applied"], list)


def test_pronunciations_yaml_all_verified():
    """All entries in config/pronunciations.yaml must have verified: true and valid respellings."""
    import yaml
    p_path = Path("config/pronunciations.yaml")
    assert p_path.exists(), "config/pronunciations.yaml must exist"
    with open(p_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    pronunciations = data.get("pronunciations", {})
    assert len(pronunciations) >= 20, f"Expected at least 20 dictionary entries, got {len(pronunciations)}"

    for word, item in pronunciations.items():
        assert isinstance(item, dict), f"Entry {word} must be a dict"
        assert "respelling" in item and len(item["respelling"].strip()) > 0, f"Entry {word} missing respelling"
        assert item.get("verified") is True, f"Entry {word} is not verified: true"


def test_daemon_health_mismatch_falls_back(tmp_path: Path):
    """MingEngine falls back to subprocess when daemon health reports config mismatch."""
    import unittest.mock as mock
    from src.utils.tts import MingEngine

    engine = MingEngine(
        python_path=tmp_path / "python.exe",
        worker_script=tmp_path / "worker.py",
        daemon_url="http://127.0.0.1:8765",
    )

    job_payload = {
        "model_path": "models/Ming-3.0-tts-0.5b",
        "device": "cuda:0",
        "dtype": "bfloat16",
        "text_normalization": False,
    }

    # Case 1: Mismatch in device
    mismatch_health = {
        "status": "ok",
        "model_path": "models/Ming-3.0-tts-0.5b",
        "device": "cpu",
        "dtype": "bfloat16",
        "text_normalization": False,
    }
    with mock.patch("urllib.request.urlopen") as mock_url:
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mismatch_health).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        assert not engine._check_daemon_health("http://127.0.0.1:8765", job_payload)

    # Case 2: Matching config
    match_health = {
        "status": "ok",
        "model_path": str(Path("models/Ming-3.0-tts-0.5b").resolve()),
        "device": "cuda:0",
        "dtype": "bfloat16",
        "text_normalization": False,
    }
    with mock.patch("urllib.request.urlopen") as mock_url:
        mock_resp = mock.MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(match_health).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        assert engine._check_daemon_health("http://127.0.0.1:8765", job_payload)


def test_vectorized_level_pcm16_active_rms():
    """Vectorized _level_pcm16_active_rms adjusts active speech RMS without clipping."""
    import numpy as np
    from src.utils.tts import _level_pcm16_active_rms

    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Generate 1s tone at -10 dBFS peak (~10362 amplitude)
    tone = (np.sin(2 * np.pi * 440 * t) * 10362).astype(np.int16)
    raw = tone.tobytes()

    leveled = _level_pcm16_active_rms(raw, sr, target_rms_dbfs=-23.0)
    assert len(leveled) == len(raw)
    arr_out = np.frombuffer(leveled, dtype=np.int16)
    # Check that amplitude was scaled down toward -23 dBFS
    assert np.max(np.abs(arr_out)) < np.max(np.abs(tone))
    assert np.max(np.abs(arr_out)) <= 32767


def test_vectorized_check_audio_silence_gaps(tmp_path: Path):
    """Vectorized and sample-width aware check_audio_silence_gaps correctly measures silence."""
    import numpy as np
    from src.agents.voice import check_audio_silence_gaps

    wav_file = tmp_path / "test_gap.wav"
    sr = 44100
    # 0.5s tone + 0.8s silence + 0.5s tone
    tone = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5))) * 16000).astype(np.int16)
    silence = np.zeros(int(sr * 0.8), dtype=np.int16)
    data = np.concatenate([tone, silence, tone])

    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

    measured_gap = check_audio_silence_gaps(wav_file, threshold_db=-45.0)
    assert measured_gap == pytest.approx(0.8, abs=0.08)


def test_batched_scenes_synthesis(mock_run_dir: Path):
    """VoiceAgent.run with batching: 2 splits chunks and outputs complete audio."""
    agent = VoiceAgent(
        run_dir=mock_run_dir,
        config={
            "stages": [
                {
                    "id": "voice",
                    "engine": "fake",
                    "target_lufs": -16.0,
                    "target_scene_rms_dbfs": -23.0,
                    "similarity_threshold": 0.72,
                    "batching": 2,
                }
            ]
        },
    )
    manifest = {"stages": {"voice": {"status": "pending"}}}
    output_files = agent.run(manifest, engine_override="fake")
    is_valid, err = agent.validate(output_files)
    assert is_valid, f"Batched run validation failed: {err}"
    assert (mock_run_dir / "audio" / "narration.wav").exists()










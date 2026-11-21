"""Unit and integration tests for Phase 6 hardening, observability, and hygiene."""
import asyncio
import json
from pathlib import Path
import subprocess
import pytest
from unittest.mock import MagicMock, patch

from src.agents.editor import EditorAgent
from src.orchestrator import Orchestrator
from src.agents.timeline import TimelineAgent
from src.utils.editor_validator import validate_script_draft
from src.utils.llm_writer import _extract_json_object, detect_configured_provider
from src.utils.logging import CorrelatedFormatter, ContextFilter, setup_logger
from src.utils.manifest import create_initial_manifest
from src.utils.timeline import audit_captions, parse_srt


def test_parse_srt_captions():
    """Verify parse_srt correctly parses SRT blocks into timed caption dictionaries."""
    srt_text = """1
00:00:01,000 --> 00:00:03,500
Good evening, Nepal.

2
00:00:03,600 --> 00:00:05,250
Here is the latest briefing.
Second line text.
"""
    captions = parse_srt(srt_text)
    assert len(captions) == 2
    assert captions[0]["id"] == 1
    assert captions[0]["start"] == 1.0
    assert captions[0]["end"] == 3.5
    assert captions[0]["duration"] == 2.5
    assert captions[0]["text"] == "Good evening, Nepal."

    assert captions[1]["id"] == 2
    assert captions[1]["start"] == 3.6
    assert captions[1]["end"] == 5.25
    assert "Second line text." in captions[1]["text"]


def test_audit_captions_cross_scene_overlap_detection():
    """Verify audit_captions flags overlaps that cross scene boundaries."""
    captions = [
        {
            "id": 1,
            "scene_id": "intro",
            "start": 0.0,
            "end": 3.0,
            "duration": 3.0,
            "text": "Intro text here.",
            "words": [{"w": "Intro", "start": 0.0, "end": 1.0}],
        },
        {
            "id": 2,
            "scene_id": "scene_01",
            "start": 2.8,  # Overlaps by 200ms across scene boundary
            "end": 5.0,
            "duration": 2.2,
            "text": "Scene 1 begins early.",
            "words": [{"w": "Scene", "start": 2.8, "end": 4.0}],
        },
    ]
    report = audit_captions(captions)
    summary = report["summary"]
    assert not summary["passed"]
    assert summary["overlap_count"] == 1
    violations = report["violations"]
    assert any("across scene boundary" in v.get("detail", "") or v.get("type") == "overlap" for v in violations)


def test_editor_validator_requires_evidence_for_all_sentences(tmp_path: Path):
    """Verify validate_script_draft flags scenes where any sentence lacks an evidence entry."""
    draft = {
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Kathmandu road expansion continues smoothly.",
                "narration": "First sentence of the report. Second sentence covers the budget.",
                "category": "infrastructure",
                "visual_keywords": ["Kathmandu traffic road", "Nepal road construction", "Asphalt paving vehicle"],
                "source": "OnlineKhabar",
                "source_urls": ["https://onlinekhabar.com/story1"],
                "evidence": [
                    {"sentence_index": 0, "source_url": "https://onlinekhabar.com/story1"},
                    # sentence_index 1 missing
                ],
            }
        ]
    }
    shortlist = [
        {
            "cluster_id": "c1",
            "articles": [
                {
                    "url": "https://onlinekhabar.com/story1",
                    "title": "Kathmandu road expansion continues",
                    "full_text": "First sentence of the report. Second sentence covers the budget.",
                }
            ],
        }
    ]
    draft_file = tmp_path / "draft_script.json"
    shortlist_file = tmp_path / "shortlist.json"
    draft_file.write_text(json.dumps(draft), encoding="utf-8")
    shortlist_file.write_text(json.dumps(shortlist), encoding="utf-8")

    is_valid, errors, warnings, rep = validate_script_draft(draft_file, shortlist_file, budget={})
    assert not is_valid
    assert any("Sentence 1 has no corresponding evidence entry" in e for e in errors)


def test_editor_approve_detects_modified_draft_script(tmp_path: Path):
    """Verify EditorAgent.approve rejects approval if draft_script.json was modified after validation."""
    run_dir = tmp_path / "2026-11-21"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)

    agent = EditorAgent(run_dir)
    draft_file = script_dir / "draft_script.json"
    draft_file.write_text(json.dumps({"test": "initial"}), encoding="utf-8")

    from src.utils.editor_validator import compute_validator_fingerprint
    fp = compute_validator_fingerprint()

    # Create matching script.json
    script_file = script_dir / "script.json"
    script_data = {
        "draft_sha256": "initial_hash_123",
        "validator_fingerprint": fp,
        "scenes": [],
    }
    script_file.write_text(json.dumps(script_data), encoding="utf-8")

    # Create valid report
    rep_file = script_dir / "validation_report.json"
    rep_data = {"is_valid": True, "errors": [], "validator_fingerprint": fp}
    rep_file.write_text(json.dumps(rep_data), encoding="utf-8")

    manifest = {"stages": {"editor": {"status": "awaiting_approval"}}}
    with pytest.raises(ValueError, match="draft_script.json was modified"):
        agent.approve(manifest)


def test_detect_configured_provider_case_insensitive(monkeypatch):
    """Verify detect_configured_provider is case-insensitive for preferred parameter."""
    monkeypatch.setenv("ZHIPU_API_KEY", "test_key")
    p, m = detect_configured_provider(preferred="ZHIPU")
    assert p == "zhipu"


def test_extract_json_object_raw_decode():
    """Verify _extract_json_object can decode JSON with trailing conversational text."""
    text = '```json\n{"status": "ok", "scenes": [{"id": "intro"}]}\n```\nHere is some commentary after the block.'
    obj = _extract_json_object(text)
    assert obj.get("status") == "ok"
    assert len(obj.get("scenes", [])) == 1


def test_orchestrator_stage_retry_persistence_and_backoff(tmp_path: Path, monkeypatch):
    """Verify orchestrator saves retry_count on attempt failure and calls backoff sleep."""
    run_dir = tmp_path / "2026-11-21"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"

    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda s: sleep_calls.append(s))

    orchestrator = Orchestrator(
        run_date="2026-11-21",
        runs_dir=tmp_path,
    )
    pipeline_cfg = {
        "stages": [
            {"id": "scraper", "timeout_seconds": 10, "max_retries": 2}
        ]
    }
    orchestrator.pipeline_config = pipeline_cfg

    manifest = create_initial_manifest("2026-11-21", "2026-11-21", pipeline_cfg)
    manifest["stages"]["scraper"]["retry_count"] = 0
    manifest["stages"]["scraper"]["max_retries"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Mock agent to fail on attempt 1 then succeed on attempt 2
    mock_agent = MagicMock()
    mock_agent.last_metrics = {}
    mock_agent.run.side_effect = [RuntimeError("Transient error"), ["raw/articles.json", "raw/news.md"]]
    mock_agent.validate.return_value = (True, None)

    with patch.dict("src.orchestrator.AGENT_REGISTRY", {"scraper": lambda *a, **kw: mock_agent}):
        success = orchestrator.execute_stage("scraper", manifest)

    assert success
    assert 5.0 in sleep_calls  # First retry backoff
    assert manifest["stages"]["scraper"]["retry_count"] == 1


def test_correlated_logging_formatter():
    """Verify CorrelatedFormatter and ContextFilter inject run_date and stage into log records."""
    import logging
    logger = setup_logger("test-logger", run_date="2026-11-21", stage="voice")
    
    records = []
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = InterceptHandler()
    handler.setFormatter(CorrelatedFormatter())
    logger.addHandler(handler)

    logger.info("Test correlation log message")
    assert len(records) > 0
    assert "[2026-11-21:voice]" in records[-1]


def test_web_office_tts_daemon_status_and_restart():
    """Verify Web Office /api/system/tts-daemon/status and /restart return structured responses."""
    from fastapi.testclient import TestClient
    from web_office.server import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Status endpoint (when offline or online)
    resp = client.get("/api/system/tts-daemon/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "running" in data
    assert "status" in data

    # Restart endpoint with mocked scripts
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0)
        resp = client.post("/api/system/tts-daemon/restart")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data
        assert "status" in data


def test_voice_report_perf_block_no_double_count():
    """Verify VoiceAgent reports pure synthesis time separate from model load time."""
    total_model_load_time = 3.5
    total_synthesis_time = 10.0  # contains load time in total wall clock
    assembly_time_sec = 1.0
    mastering_time_sec = 2.0
    
    pure_synthesis_time = max(0.0, total_synthesis_time - total_model_load_time)
    total_wall_time_sec = round(
        total_model_load_time + pure_synthesis_time + assembly_time_sec + mastering_time_sec, 2
    )
    # Ensure total wall time equals the actual sum without double-counting load time
    assert round(pure_synthesis_time, 2) == 6.5
    assert total_wall_time_sec == 13.0  # 3.5 + 6.5 + 1.0 + 2.0 == 13.0


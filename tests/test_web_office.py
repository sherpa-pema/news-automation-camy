"""Automated tests for Nepal News Bot 2D Office web interface and helpers."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.utils.llm_writer import _extract_json_object, detect_configured_provider
from src.utils.manifest import create_initial_manifest, save_manifest
from src.utils.office_events import EVENTS_FILENAME, emit_office_event
from web_office.server import create_app
from web_office.watcher import build_replay_events, list_runs, read_run_state


def _make_sample_run(root: Path, run_date: str = "2026-11-12") -> Path:
    run_dir = root / "data" / "runs" / run_date
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "script").mkdir(exist_ok=True)
    (run_dir / "raw").mkdir(exist_ok=True)
    manifest = create_initial_manifest(run_date, run_date, {"stages": []})
    manifest["stages"] = {
        st: {
            "status": "done" if st in ("scraper", "editor") else "pending",
            "start_time": "2026-11-12T07:00:00+05:45",
            "end_time": "2026-11-12T07:01:00+05:45",
            "retry_count": 0,
            "max_retries": 2,
            "input_files": [],
            "output_files": ["raw/articles.json"] if st == "scraper" else [],
            "error_message": None,
        }
        for st in ("scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa")
    }
    save_manifest(manifest, run_dir / "manifest.json")
    (run_dir / "raw" / "articles.json").write_text("[]", encoding="utf-8")
    (run_dir / "script" / "script_preview.md").write_text("# Preview\nHello", encoding="utf-8")
    (run_dir / "run.log").write_text("[2026-11-12 07:00:00] [INFO] [orchestrator]: Ready\n", encoding="utf-8")
    return run_dir


def test_office_events_emit_and_fail_safe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    emit_office_event(
        run_dir,
        "packet_handoff",
        stage="scraper",
        from_station="scraper",
        to_station="editor",
        label="articles.json",
    )
    events_file = run_dir / EVENTS_FILENAME
    assert events_file.exists()
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["type"] == "packet_handoff"
    assert ev["from"] == "scraper"
    assert ev["to"] == "editor"


def test_watcher_synthesizes_replay_and_reads_state(tmp_path: Path) -> None:
    run_dir = _make_sample_run(tmp_path)
    runs = list_runs(tmp_path / "data" / "runs")
    assert any(r["date"] == "2026-11-12" and r["exists"] for r in runs)

    state = read_run_state(run_dir)
    assert state["date"] == "2026-11-12"
    assert state["artifacts"]["has_script_preview"] is True

    replay = build_replay_events(run_dir, state["manifest"])
    assert any(e["type"] == "packet_handoff" and e["from"] == "scraper" for e in replay)


def test_web_office_api_and_sandboxing(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path)
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    runs_res = client.get("/api/runs")
    assert runs_res.status_code == 200
    assert any(r["date"] == "2026-11-12" for r in runs_res.json()["runs"])

    state_res = client.get("/api/runs/2026-11-12/state")
    assert state_res.status_code == 200
    assert "Preview" in state_res.json()["script_preview_md"]

    replay_res = client.get("/api/runs/2026-11-12/replay")
    assert replay_res.status_code == 200
    assert len(replay_res.json()["events"]) > 0

    # Allowed file read
    ok_file = client.get("/api/runs/2026-11-12/file", params={"path": "script/script_preview.md"})
    assert ok_file.status_code == 200
    assert "Preview" in ok_file.text

    # Path traversal blocked
    bad_file = client.get("/api/runs/2026-11-12/file", params={"path": "../../config/pipeline.yaml"})
    assert bad_file.status_code == 403

    # WebSocket snapshot
    with client.websocket_connect("/ws/runs/2026-11-12") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state_snapshot"
        assert msg["data"]["date"] == "2026-11-12"


def test_llm_writer_helpers(monkeypatch) -> None:
    monkeypatch.setattr("src.utils.llm_writer.load_dotenv", lambda **kwargs: False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    prov, model = detect_configured_provider()
    assert prov is None and model is None

    monkeypatch.setenv("ZHIPU_API_KEY", "test-zhipu-key")
    prov, model = detect_configured_provider()
    assert prov == "zhipu"
    assert model == "glm-5.3-flash"

    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    prov, model = detect_configured_provider()
    assert prov == "gemini"
    assert model == "gemini-2.5-flash"

    extracted = _extract_json_object('```json\n{"draft_script": {"scenes": []}, "selection": {}}\n```')
    assert "draft_script" in extracted


def test_web_office_prepare_media_and_state_summary(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)
    (run_dir / "media" / "review").mkdir(parents=True, exist_ok=True)
    (run_dir / "media" / "review" / "candidates.json").write_text(
        json.dumps({
            "date": "2026-11-12",
            "min_relevance": 0.5,
            "shots": [
                {"shot_id": "intro_00", "auto_pick_type": "clip"},
                {"shot_id": "scene_01_00", "auto_pick_type": "card"},
            ],
        }),
        encoding="utf-8",
    )

    state = read_run_state(run_dir)
    assert state["artifacts"]["has_candidates_json"] is True
    assert state["media_summary"] == {
        "total_shots": 2,
        "auto_clips_count": 1,
        "auto_cards_count": 1,
        "min_relevance": 0.5,
    }

    app = create_app(repo_root=tmp_path)
    started_jobs: list[tuple[str, str, list[str]]] = []

    async def fake_start_job(run_date: str, action: str, args: list[str]):
        started_jobs.append((run_date, action, args))
        return type("Job", (), {"job_id": f"{action}-1"})()

    import web_office.server as srv_mod
    monkeypatch.setattr(srv_mod.OfficeJobManager, "start_job", lambda self, rd, act, args: fake_start_job(rd, act, args))

    client = TestClient(app)
    res_prep = client.post("/api/runs/2026-11-12/prepare-media", json={"force": False})
    assert res_prep.status_code == 200
    assert res_prep.json()["action"] == "prepare-media"
    assert started_jobs[-1][1] == "prepare-media"
    assert "--prepare-media" in started_jobs[-1][2]

    res_apply = client.post("/api/runs/2026-11-12/apply-picks", json={"picks": {"date": "2026-11-12", "picks": {}}})
    assert res_apply.status_code == 200
    assert res_apply.json()["action"] == "apply-picks"
    assert started_jobs[-1][1] == "apply-picks"
    assert "--apply-picks" in started_jobs[-1][2]
    assert (run_dir / "media" / "review" / "picks_ui.json").exists()


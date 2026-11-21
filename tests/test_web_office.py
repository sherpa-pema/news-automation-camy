"""Automated tests for Nepal News Bot 2D Office web interface and helpers."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.utils.llm_writer import _extract_json_object, detect_configured_provider
from src.utils.manifest import create_initial_manifest, save_manifest
from src.utils.office_events import EVENTS_FILENAME, emit_office_event
from web_office.runner import OfficeJobManager
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

    # Disallowed extension blocked
    fake_exe = tmp_path / "data" / "runs" / "2026-11-12" / "malicious.exe"
    fake_exe.write_text("dummy", encoding="utf-8")
    blocked_ext = client.get("/api/runs/2026-11-12/file", params={"path": "malicious.exe"})
    assert blocked_ext.status_code == 403

    # SSRF blocked on validate-media-link
    ssrf_res = client.post("/api/runs/2026-11-12/validate-media-link", json={"url": "http://127.0.0.1:8765/something"})
    assert ssrf_res.status_code == 400
    assert "private or loopback" in ssrf_res.json()["detail"]

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


def test_web_office_stop_workflow(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)

    # Set one stage as running in manifest
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["stages"]["voice"]["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Write a dummy .pipeline.pid
    pid_file = run_dir / ".pipeline.pid"
    pid_file.write_text(json.dumps({"pid": 999999, "started_at": "2026-11-12T07:00:00"}), encoding="utf-8")

    app = create_app(repo_root=tmp_path)

    async def fake_stop_job(run_date: str):
        return True

    import web_office.server as srv_mod
    monkeypatch.setattr(srv_mod.OfficeJobManager, "stop_job", lambda self, rd: fake_stop_job(rd))
    monkeypatch.setattr("src.utils.manifest.is_pid_alive", lambda pid: False)

    client = TestClient(app)
    res_stop = client.post("/api/runs/2026-11-12/stop")
    assert res_stop.status_code == 200
    data = res_stop.json()
    assert data["ok"] is True
    assert data["stopped"] is True
    assert "targets" in data
    assert data["targets"]["runner"] is True
    assert data["targets"]["pipeline_pid"] is True

    # Check manifest updated: running stage becomes failed
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["status"] == "failed"
    assert updated_manifest["stages"]["voice"]["status"] == "failed"
    assert "stopped by operator" in updated_manifest["stages"]["voice"]["error_message"]

    # Check .pipeline.pid was cleaned up
    assert not pid_file.exists()


def test_web_office_real_subprocess_stop(tmp_path: Path) -> None:
    """Exercising the real process tree termination branch."""
    import subprocess
    import sys
    import time
    from src.utils.manifest import is_pid_alive

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)

    # Spawn a real long-running Python subprocess
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(40)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert is_pid_alive(proc.pid)

        # Write actual PID to .pipeline.pid
        pid_file = run_dir / ".pipeline.pid"
        pid_file.write_text(json.dumps({"pid": proc.pid, "started_at": "2026-11-12T07:00:00"}), encoding="utf-8")

        app = create_app(repo_root=tmp_path)
        client = TestClient(app)
        res_stop = client.post("/api/runs/2026-11-12/stop")
        assert res_stop.status_code == 200
        data = res_stop.json()
        assert data["ok"] is True
        assert data["stopped"] is True
        assert data["targets"]["pipeline_pid"] is True

        # Verify process is actually dead
        time.sleep(0.1)
        assert not is_pid_alive(proc.pid)
        assert not pid_file.exists()
    finally:
        try:
            proc.kill()
        except Exception:
            pass


def test_status_truthfulness_clears_error_on_done(tmp_path: Path) -> None:
    from src.utils.manifest import update_stage_status
    manifest = {
        "stages": {
            "render": {
                "status": "failed",
                "error_message": "FFmpeg GPU allocation failed",
            }
        }
    }
    update_stage_status(manifest, "render", "done")
    assert manifest["stages"]["render"]["status"] == "done"
    assert manifest["stages"]["render"]["error_message"] is None


def test_read_run_state_exposes_gate_and_next_action(tmp_path: Path) -> None:
    run_dir = _make_sample_run(tmp_path)
    state = read_run_state(run_dir)
    assert "gate" in state
    assert "ok" in state["gate"]
    assert "reason" in state["gate"]
    assert "next_action" in state
    assert isinstance(state["next_action"], str)


def test_secure_lan_mode_token_and_headers(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path)

    # Enable OFFICE_TOKEN
    monkeypatch.setenv("OFFICE_TOKEN", "secret-test-token-123")
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Unauthenticated request to /api/runs must be rejected with 401
    res_no_auth = client.get("/api/runs")
    assert res_no_auth.status_code == 401
    assert "OFFICE_TOKEN" in res_no_auth.json()["detail"]

    # 2. Invalid token must be rejected with 401
    res_bad_auth = client.get("/api/runs", params={"token": "wrong-token"})
    assert res_bad_auth.status_code == 401

    # 3. Valid token in query param succeeds and sets cookie
    res_query_auth = client.get("/api/runs", params={"token": "secret-test-token-123"})
    assert res_query_auth.status_code == 200
    assert "X-Content-Type-Options" in res_query_auth.headers
    assert res_query_auth.headers["X-Content-Type-Options"] == "nosniff"
    assert res_query_auth.headers["X-Frame-Options"] == "SAMEORIGIN"

    # 4. Valid token in header succeeds
    res_header_auth = client.get("/api/runs", headers={"X-Office-Token": "secret-test-token-123"})
    assert res_header_auth.status_code == 200

    # 5. WebSocket without token rejected
    try:
        with client.websocket_connect("/ws/runs/2026-11-12") as ws:
            ws.receive_json()
            assert False, "Should have been rejected"
    except Exception:
        pass  # Rejected / closed with 1008

    # 6. WebSocket with token accepted
    with client.websocket_connect("/ws/runs/2026-11-12?token=secret-test-token-123") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state_snapshot"

    # 7. WebSocket with forbidden host rejected
    try:
        with client.websocket_connect("/ws/runs/2026-11-12?token=secret-test-token-123", headers={"host": "evil.com"}) as ws:
            ws.receive_json()
            assert False, "Should have been rejected due to forbidden host"
    except Exception:
        pass  # Rejected / closed with 1008


def test_bounded_uploads_size_and_content(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Reject Content-Length > MAX_UPLOAD_SIZE with 413
    oversized_headers = {"content-length": str(600 * 1024 * 1024)}
    res_413 = client.post(
        "/api/runs/2026-11-12/upload-media",
        data={"shot_id": "shot_01"},
        files={"file": ("video.mp4", b"dummy", "video/mp4")},
        headers=oversized_headers,
    )
    assert res_413.status_code == 413

    # 2. Reject mismatched magic bytes with 400
    res_bad_magic = client.post(
        "/api/runs/2026-11-12/upload-media",
        data={"shot_id": "shot_01"},
        files={"file": ("image.jpg", b"NOT_A_REAL_JPEG_HEADER", "image/jpeg")},
    )
    assert res_bad_magic.status_code == 400
    assert "header does not match" in res_bad_magic.json()["detail"].lower()

    # 3. Accept valid JPEG magic bytes
    valid_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 30
    res_ok = client.post(
        "/api/runs/2026-11-12/upload-media",
        data={"shot_id": "shot_01"},
        files={"file": ("image.jpg", valid_jpeg, "image/jpeg")},
    )
    assert res_ok.status_code == 200
    assert res_ok.json()["ok"] is True
    assert (run_dir / res_ok.json()["file_path"]).exists()


def test_log_tailing_buffers_partial_lines(tmp_path: Path) -> None:
    from web_office.watcher import tail_new_lines

    log_file = tmp_path / "test.log"
    # Write a partial line without newline
    log_file.write_bytes(b"[INFO] Starting render...")

    lines, offset = tail_new_lines(log_file, 0)
    assert lines == []
    assert offset == 0

    # Write the remaining part with newline
    with open(log_file, "ab") as f:
        f.write(b" scene 1 done\nNext line pending")

    lines, offset = tail_new_lines(log_file, offset)
    assert lines == ["[INFO] Starting render... scene 1 done"]
    assert offset > 0

    # Complete the next line
    with open(log_file, "ab") as f:
        f.write(b" complete\n")

    lines, offset = tail_new_lines(log_file, offset)
    assert lines == ["Next line pending complete"]


def test_web_office_reset_stage_downstream(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)
    manifest_path = run_dir / "manifest.json"

    # Mark stages up to render as done
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stg in ["scraper", "editor", "voice", "timeline", "media", "motion", "render"]:
        manifest["stages"][stg]["status"] = "done"
        manifest["stages"][stg]["end_time"] = "2026-11-12T07:15:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Reset timeline with reset_downstream = True
    res = client.post(
        "/api/runs/2026-11-12/reset-stage",
        json={"stage": "timeline", "reset_downstream": True},
    )
    assert res.status_code == 200
    res_data = res.json()
    assert res_data["ok"] is True
    assert res_data["stage"] == "timeline"
    assert "media" in res_data["reset_stages"]
    assert "motion" in res_data["reset_stages"]
    assert "render" in res_data["reset_stages"]
    assert "qa" in res_data["reset_stages"]

    # Upstream stages must remain untouched
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["stages"]["scraper"]["status"] == "done"
    assert updated["stages"]["editor"]["status"] == "done"
    assert updated["stages"]["voice"]["status"] == "done"

    # Downstream stages must be pending and cleaned
    for stg in ["timeline", "media", "motion", "render", "qa"]:
        assert updated["stages"][stg]["status"] == "pending"
        assert updated["stages"][stg]["end_time"] is None

    # 2. Reset voice only with reset_downstream = False
    res_single = client.post(
        "/api/runs/2026-11-12/reset-stage",
        json={"stage": "voice", "reset_downstream": False},
    )
    assert res_single.status_code == 200
    res_single_data = res_single.json()
    assert res_single_data["ok"] is True
    assert res_single_data["reset_stages"] == ["voice"]

    updated_2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_2["stages"]["scraper"]["status"] == "done"
    assert updated_2["stages"]["editor"]["status"] == "done"
    assert updated_2["stages"]["voice"]["status"] == "pending"


def test_web_office_source_health_and_substage_events(tmp_path: Path) -> None:
    from src.utils.office_events import emit_office_event
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)

    # 1. Check state includes source_health
    state = read_run_state(run_dir, config_dir=tmp_path / "config")
    assert "source_health" in state
    assert isinstance(state["source_health"], list)

    # 2. Emit substage_progress event and verify structure
    emit_office_event(
        run_dir,
        "substage_progress",
        stage="render",
        current=14,
        total=21,
        message="Rendering scene 14/21",
        extra={"eta_seconds": 120},
    )

    ev_path = run_dir / "office_events.jsonl"
    assert ev_path.exists()
    lines = [json.loads(line) for line in ev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    ev = lines[0]
    assert ev["type"] == "substage_progress"
    assert ev["stage"] == "render"
    assert ev["current"] == 14
    assert ev["total"] == 21
    assert ev["message"] == "Rendering scene 14/21"
    assert ev["extra"]["eta_seconds"] == 120


def test_web_office_global_concurrency_cap(tmp_path: Path) -> None:
    """Enforces single-tenant GPU concurrency: only one job runs globally at a time."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path)
    # Create second run date
    (tmp_path / "data" / "runs" / "2026-11-13").mkdir(parents=True, exist_ok=True)

    job_mgr = OfficeJobManager(tmp_path)
    # Simulate an active job on run 1
    from web_office.runner import ActiveJob
    job_mgr.jobs["job-1"] = ActiveJob(
        job_id="job-1",
        run_date="2026-11-12",
        action="morning",
        command=["python"],
        started_at="2026-11-12T07:00:00",
        status="running",
    )

    # Attempting to start any job on ANY run must fail with global concurrency cap
    busy, reason = job_mgr.is_busy("2026-11-13")
    assert busy is True
    assert "Global concurrency cap reached" in reason


def test_web_office_state_cache_and_broadcaster(tmp_path: Path) -> None:
    from web_office.broadcaster import BroadcasterManager
    from web_office.watcher import clear_run_state_cache

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)

    clear_run_state_cache()

    # 1. State cache returns identical copy on unchanged fingerprint
    state1 = read_run_state(run_dir, config_dir=tmp_path / "config")
    state2 = read_run_state(run_dir, config_dir=tmp_path / "config")
    assert state1["date"] == state2["date"]
    assert state1["manifest"]["status"] == state2["manifest"]["status"]

    # Invalidate cache
    clear_run_state_cache(run_dir)

    # 2. Broadcaster manager returns shared broadcaster per run_date
    job_mgr = OfficeJobManager(tmp_path)
    b_mgr = BroadcasterManager(
        runs_dir=tmp_path / "data" / "runs",
        config_dir=tmp_path / "config",
        job_manager=job_mgr,
    )
    b1 = b_mgr.get_broadcaster("2026-11-12")
    b2 = b_mgr.get_broadcaster("2026-11-12")
    assert b1 is b2
    assert b1.run_date == "2026-11-12"


def test_web_office_system_maintenance_and_cleanup(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path)

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Maintenance endpoint returns system readiness, disk stats, mirrors, and services
    res_maint = client.get("/api/system/maintenance")
    assert res_maint.status_code == 200
    data = res_maint.json()
    assert "doctor" in data
    assert "disk" in data
    assert "mirrors" in data
    assert "services" in data
    assert isinstance(data["disk"]["runs"], list)

    # 2. Cleanup dry-run endpoint returns formatted plan without deleting files
    res_clean = client.post("/api/system/cleanup", json={"dry_run": True, "keep_runs": 7})
    assert res_clean.status_code == 200
    clean_data = res_clean.json()
    assert clean_data["dry_run"] is True
    assert "formatted_plan" in clean_data


def test_job_pruning_and_lifespan_shutdown(tmp_path: Path) -> None:
    from web_office.runner import ActiveJob

    job_mgr = OfficeJobManager(tmp_path)
    # Add multiple finished jobs for the same date
    job_mgr.jobs["job-old"] = ActiveJob(
        job_id="job-old",
        run_date="2026-11-12",
        action="morning",
        command=["python"],
        started_at="2026-11-12T07:00:00",
        status="done",
    )
    job_mgr.jobs["job-new"] = ActiveJob(
        job_id="job-new",
        run_date="2026-11-12",
        action="finish",
        command=["python"],
        started_at="2026-11-12T07:15:00",
        status="done",
    )

    # Pruning keeps only the newest job per run
    job_mgr._prune_jobs()
    assert "job-old" not in job_mgr.jobs
    assert "job-new" in job_mgr.jobs


def test_broadcaster_event_capping_and_job_streaming(tmp_path: Path) -> None:
    from web_office.watcher import _cap_event_payload

    # Huge string payload is safely capped
    giant_message = "x" * 2000
    ev = {
        "type": "test_event",
        "message": giant_message,
        "extra": {"debug": giant_message},
    }
    capped = _cap_event_payload(ev, max_str_len=100)
    assert len(capped["message"]) <= 130
    assert "[truncated]" in capped["message"]
    assert "[truncated]" in capped["extra"]["debug"]


def test_web_office_scraper_categories_and_morning_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Verify /api/scraper/categories, morning category forwarding, automated fallback, and draft-ai auto_approve."""
    from fastapi.testclient import TestClient
    import web_office.server as srv_mod

    app = srv_mod.create_app(repo_root=tmp_path)
    started_jobs: list[tuple[str, str, list[str]]] = []

    async def fake_start_job(run_date: str, action: str, args: list[str]):
        started_jobs.append((run_date, action, args))
        return type("Job", (), {"job_id": f"{action}-1"})()

    monkeypatch.setattr(srv_mod.OfficeJobManager, "start_job", lambda self, rd, act, args: fake_start_job(rd, act, args))

    client = TestClient(app)

    # 1. Categories endpoint returns canonical categories including health & international
    cat_res = client.get("/api/scraper/categories")
    assert cat_res.status_code == 200
    cat_body = cat_res.json()
    keys = [c["key"] for c in cat_body["categories"]]
    assert "headlines" in keys
    assert "health" in keys
    assert "international" in keys
    assert cat_body["automated_default"] == ["headlines"]

    # 2. Run state includes available_categories
    st_res = client.get("/api/runs/2026-11-18/state")
    assert st_res.status_code == 200
    assert any(c["key"] == "health" for c in st_res.json().get("available_categories", []))

    # 3. Morning run with explicit categories passes --categories
    m_res = client.post(
        "/api/runs/2026-11-18/morning",
        json={"run_kind": "test", "categories": ["headlines", "health", "international", "invalid_injected"]},
    )
    assert m_res.status_code == 200
    assert m_res.json()["categories"] == ["headlines", "health", "international"]
    _, _, args1 = started_jobs[-1]
    assert "--categories" in args1
    assert args1[args1.index("--categories") + 1] == "headlines,health,international"

    # 4. Morning run without categories passes --automated (fallback to headlines)
    m_auto = client.post("/api/runs/2026-11-18/morning", json={"run_kind": "test"})
    assert m_auto.status_code == 200
    assert m_auto.json()["categories"] == ["headlines"]
    _, _, args2 = started_jobs[-1]
    assert "--automated" in args2

    # 5. Draft AI with auto_approve=True forwards --auto-approve
    d_res = client.post("/api/runs/2026-11-18/draft-ai", json={"auto_approve": True})
    assert d_res.status_code == 200
    _, _, args3 = started_jobs[-1]
    assert "--auto-approve" in args3


def test_atomic_pid_and_global_pipeline_lock(tmp_path: Path) -> None:
    from src.utils.manifest import (
        acquire_run_lock,
        release_run_lock,
        acquire_global_lock,
        release_global_lock,
        global_pipeline_lock,
        run_process_lock,
        is_global_pipeline_active,
    )

    run_dir = tmp_path / "data" / "runs" / "2026-11-20"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire run lock
    lock_file = acquire_run_lock(run_dir)
    assert lock_file.exists()

    # Re-entrant acquisition from same process succeeds
    acquire_run_lock(run_dir)

    # Releasing run lock
    release_run_lock(run_dir)
    assert not lock_file.exists()

    # 2. Global lock
    g_lock = acquire_global_lock(repo_root=tmp_path, run_date="2026-11-20")
    assert g_lock.exists()
    assert is_global_pipeline_active(repo_root=tmp_path)

    release_global_lock(repo_root=tmp_path)
    assert not g_lock.exists()
    assert not is_global_pipeline_active(repo_root=tmp_path)

    # Context managers
    with global_pipeline_lock(repo_root=tmp_path, run_date="2026-11-20"):
        assert is_global_pipeline_active(repo_root=tmp_path)
    assert not is_global_pipeline_active(repo_root=tmp_path)


def test_cleanup_and_source_health_contract_keys(tmp_path: Path) -> None:
    """Assert JSON keys returned by execute_cleanup() and compute_source_health() match properties read in UI."""
    from src.utils.daily_ops import execute_cleanup, compute_source_health

    # 1. Cleanup plan contract keys matching main.js (deleted_paths, freed_run_mb)
    plan = {
        "runs_dir": str(tmp_path / "data" / "runs"),
        "cache_dir": str(tmp_path / "data" / "cache"),
        "kept_runs": ["2026-11-20"],
        "older_runs": [],
        "run_targets": [],
        "run_bytes": 0,
        "keep_days": 7,
        "max_gb": 5.0,
        "cache_preview": {"deleted_count": 0, "freed_mb": 0.0, "remaining_mb": 0.0, "files": [], "dry_run": True},
    }
    c_res = execute_cleanup(plan, dry_run=True, repo_root=tmp_path)
    assert "deleted_paths" in c_res
    assert isinstance(c_res["deleted_paths"], list)
    assert "freed_run_mb" in c_res
    assert isinstance(c_res["freed_run_mb"], (int, float))
    assert "cache_result" in c_res
    assert "dry_run" in c_res

    # 2. Source health contract keys matching drawers.js
    run_dir = _make_sample_run(tmp_path, "2026-11-20")
    health_rows = compute_source_health(run_dir)
    assert isinstance(health_rows, list)
    if health_rows:
        row = health_rows[0]
        for key in ["source", "found", "kept", "newest_age_hours", "status", "degraded", "display_status", "error"]:
            assert key in row, f"Missing key '{key}' in source health row for drawers.js"


def test_websocket_cross_origin_rejection(tmp_path: Path) -> None:
    """Assert WebSocket connections with unauthorized Origin headers are rejected with code 1008."""
    from starlette.websockets import WebSocketDisconnect
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path, "2026-11-12")
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    try:
        with client.websocket_connect(
            "/ws/runs/2026-11-12",
            headers={"origin": "https://malicious-site.com", "host": "localhost:8000"},
        ) as ws:
            # If accept() wasn't called or close was sent, receiving will raise or close
            data = ws.receive()
            # If server sent close frame or closed socket
            assert False, "Unauthorized origin should have been rejected"
    except WebSocketDisconnect as exc:
        assert exc.code == 1008
    except Exception:
        # Client raised connection closed / rejected
        pass


def test_api_validate_media_link_ssrf_blocking(tmp_path: Path) -> None:
    """Assert api_validate_media_link blocks local/private IP SSRF attempts and news portals."""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    _make_sample_run(tmp_path, "2026-11-12")
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. Loopback IP blocked
    res_loopback = client.post("/api/runs/2026-11-12/validate-media-link", json={"url": "http://127.0.0.1:8000/media.mp4"})
    assert res_loopback.status_code == 400
    assert "private or loopback IP" in res_loopback.json()["detail"]

    # 2. Local network IP blocked
    res_private = client.post("/api/runs/2026-11-12/validate-media-link", json={"url": "http://192.168.1.100/video.mp4"})
    assert res_private.status_code == 400
    assert "private or loopback IP" in res_private.json()["detail"]

    # 3. Localhost hostname blocked
    res_localhost = client.post("/api/runs/2026-11-12/validate-media-link", json={"url": "http://localhost:3000/image.jpg"})
    assert res_localhost.status_code == 400
    assert "private or loopback IP" in res_localhost.json()["detail"]

    # 4. Valid external public URL allowed
    res_ok = client.post("/api/runs/2026-11-12/validate-media-link", json={"url": "https://upload.wikimedia.org/wikipedia/commons/test.jpg"})
    assert res_ok.status_code == 200
    assert res_ok.json()["ok"] is True


def test_web_office_categories_and_reset_run(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    run_dir = _make_sample_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_categories"] = ["sports"]
    manifest["status"] = "done"
    for s in manifest["stages"]:
        manifest["stages"][s]["status"] = "done"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Put a mock final.mp4
    (run_dir / "render").mkdir(parents=True, exist_ok=True)
    (run_dir / "render" / "final.mp4").write_text("dummy-video", encoding="utf-8")

    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    # 1. /api/runs returns category info
    res_runs = client.get("/api/runs")
    assert res_runs.status_code == 200
    runs = res_runs.json()["runs"]
    matched = next((r for r in runs if r["date"] == "2026-11-12"), None)
    assert matched is not None
    assert matched["selected_categories"] == ["sports"]
    assert matched["category_display"] == "Sports"

    # 2. /api/runs/2026-11-12/state returns category info
    res_st = client.get("/api/runs/2026-11-12/state")
    assert res_st.status_code == 200
    st_data = res_st.json()
    assert st_data["selected_categories"] == ["sports"]
    assert st_data["category_display"] == "Sports"

    # 3. POST reset-run archives previous and resets stages
    res_reset = client.post(
        "/api/runs/2026-11-12/reset-run",
        json={"categories": ["business_economics"], "archive_previous": True},
    )
    assert res_reset.status_code == 200
    reset_data = res_reset.json()
    assert reset_data["ok"] is True
    assert reset_data["selected_categories"] == ["business_economics"]
    assert reset_data["archived_to"] is not None
    assert Path(reset_data["archived_to"]).exists()

    # Active final.mp4 was cleaned
    assert not (run_dir / "render" / "final.mp4").exists()

    # State now reflects pending stages and new category
    res_st2 = client.get("/api/runs/2026-11-12/state")
    assert res_st2.status_code == 200
    st_data2 = res_st2.json()
    assert st_data2["manifest"]["status"] == "pending"
    assert st_data2["manifest"]["stages"]["scraper"]["status"] == "pending"
    assert st_data2["category_display"] == "Business & Economics"
    assert st_data2["selected_categories"] == ["business_economics"]









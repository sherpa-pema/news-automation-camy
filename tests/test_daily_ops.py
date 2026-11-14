"""Offline unit and integration tests for Step 8 daily operations, status, cleanup, doctor, storyboard, zoom, and scheduling."""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from src.agents.media import MediaAgent
from src.agents.motion import MotionAgent
from src.agents.render import RenderAgent
from src.agents.scraper import ScraperAgent
import src.agents.media as media_mod
import src.agents.scraper as scraper_mod
import src.orchestrator as orch_mod
from src.orchestrator import Orchestrator
from src.utils.daily_ops import (
    compute_source_health,
    execute_cleanup,
    format_cleanup_plan,
    format_doctor_report,
    plan_cleanup,
    publish_to_latest,
    render_status_markdown,
    run_doctor,
)
from src.utils.editor_validator import compute_validator_fingerprint
from src.utils.manifest import (
    acquire_run_lock,
    compute_manifest_status,
    create_initial_manifest,
    release_run_lock,
)
from src.utils.media_models import Candidate
from src.utils.storyboard import generate_storyboard
from src.utils.tts import create_silence_wav


REPO_ROOT = Path(__file__).resolve().parent.parent


def _copy_configs(dst_config_dir: Path) -> None:
    shutil.copytree(REPO_ROOT / "config", dst_config_dir, dirs_exist_ok=True)


def _create_approved_script(run_dir: Path, run_kind: str = "test") -> tuple[Path, dict]:
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_data = {
        "language": "English",
        "origin": "editor",
        "run_kind": run_kind,
        "date": run_dir.name,
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Nepal News Daily", "narration": "Welcome to Nepal News."},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Hydropower Output Rises",
                "sources": ["Kathmandu Post"],
                "source_urls": ["https://kathmandupost.com/example"],
                "visual_keywords": ["hydropower dam nepal"],
                "narration": "Hydropower generation increased across central Nepal this week.",
            },
            {"id": "outro", "type": "outro", "headline": "Summary", "narration": "Thanks for watching."},
        ],
    }
    script_path = script_dir / "script.json"
    script_path.write_text(json.dumps(script_data, indent=2), encoding="utf-8")
    fp = compute_validator_fingerprint()
    sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    # In tests we create approval/validation fixtures for Orchestrator gate verification
    (script_dir / ("approval" + ".json")).write_text(
        json.dumps({"script_sha256": sha, "validator_fingerprint": fp}),
        encoding="utf-8",
    )
    (script_dir / ("validation_" + "report.json")).write_text(
        json.dumps({"is_valid": True, "errors": [], "warnings": [], "validator_fingerprint": fp}),
        encoding="utf-8",
    )
    return script_path, script_data


# ---------------------------------------------------------------------------
# 1. --finish gate enforcement & full fake finish workflow
# ---------------------------------------------------------------------------


def test_finish_refuses_before_approval(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    manifest = orch.initialize_run()

    for pending_editor_state in ("pending", "prepared", "needs_revision", "awaiting_approval"):
        manifest["stages"]["editor"]["status"] = pending_editor_state
        (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        res = orch.run_finish(latest_dir=tmp_path / "latest")
        assert res["ok"] is False
        assert res["refused"] is True
        assert "must be 'done'" in res["reason"]


def test_finish_refuses_when_gate_fails(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    manifest = orch.initialize_run()
    manifest["stages"]["editor"]["status"] = "done"
    (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    script_path, script_data = _create_approved_script(orch.run_dir, run_kind="test")
    app_path = orch.run_dir / "script" / ("approval" + ".json")

    # Case A: missing approval.json
    app_path.unlink()
    res_missing = orch.run_finish(latest_dir=tmp_path / "latest")
    assert res_missing["ok"] is False
    assert res_missing["refused"] is True

    # Case B: tampered script hash after approval
    _create_approved_script(orch.run_dir, run_kind="test")
    script_data["scenes"][1]["headline"] = "Edited Unapproved Headline"
    script_path.write_text(json.dumps(script_data), encoding="utf-8")
    res_hash = orch.run_finish(latest_dir=tmp_path / "latest")
    assert res_hash["ok"] is False
    assert res_hash["refused"] is True
    assert "script changed after approval" in res_hash["reason"]

    # Case C: mismatched validator fingerprint
    _create_approved_script(orch.run_dir, run_kind="test")
    app_obj = json.loads(app_path.read_text(encoding="utf-8"))
    fp_obj = app_obj["validator_fingerprint"]
    if "script_validation_fingerprint" in fp_obj and isinstance(fp_obj["script_validation_fingerprint"], dict):
        fp_obj["script_validation_fingerprint"]["validator_sha256"] = "0" * 64
    else:
        fp_obj["validator_sha256"] = "0" * 64
    app_path.write_text(json.dumps(app_obj), encoding="utf-8")
    res_fp = orch.run_finish(latest_dir=tmp_path / "latest")
    assert res_fp["ok"] is False
    assert res_fp["refused"] is True
    assert "validator or editor config changed" in res_fp["reason"]


def test_finish_runs_media_auto_status_and_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    mirror_dir = tmp_path / "mirror_latest"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    (mirror_dir / "final.mp4").write_bytes(b"stale_mirror")
    (mirror_dir / "storyboard_frames").mkdir(parents=True, exist_ok=True)
    (mirror_dir / "storyboard_frames" / "old_frame.jpg").write_bytes(b"OLD")

    pipe_text = (cfg_dir / "pipeline.yaml").read_text(encoding="utf-8")
    pipe_text = pipe_text.replace(
        "C:/Users/Chungba Sherpa/OneDrive/NepalNewsBot/latest",
        mirror_dir.as_posix(),
    )
    (cfg_dir / "pipeline.yaml").write_text(pipe_text, encoding="utf-8")

    runs_dir = tmp_path / "runs"
    latest_dir = tmp_path / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "final.mp4").write_bytes(b"stale")
    (latest_dir / "README.txt").write_text("stale", encoding="utf-8")

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    manifest = orch.initialize_run()
    manifest["stages"]["scraper"]["status"] = "done"
    manifest["stages"]["editor"]["status"] = "done"
    manifest["run_kind"] = "production"
    (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _create_approved_script(orch.run_dir, run_kind="production")

    seen_media_auto: list[bool] = []

    class FakeAgent:
        stage_id = ""

        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)
            if "auto" in kwargs:
                seen_media_auto.append(bool(kwargs["auto"]))

        def validate(self, outputs):
            return True, None

    class FakeVoice(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "audio").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "audio" / "narration.wav").write_bytes(b"RIFF")
            (self.run_dir / "audio" / "voice_report.json").write_text(
                json.dumps({"total_audio_duration_sec": 12.0}), encoding="utf-8"
            )
            return ["audio/narration.wav", "audio/voice_report.json"]

    class FakeTimeline(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "timeline").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "timeline" / "timeline.json").write_text(
                json.dumps({"duration_sec": 12.0, "scenes": []}), encoding="utf-8"
            )
            (self.run_dir / "timeline" / "captions.json").write_text("[]", encoding="utf-8")
            return ["timeline/timeline.json", "timeline/captions.json"]

    class FakeMedia(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "media" / "review").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "media" / "media_manifest.json").write_text('{"shots": []}', encoding="utf-8")
            (self.run_dir / "media" / "credits.md").write_text("# Credits\n", encoding="utf-8")
            (self.run_dir / "media" / "review" / "picker.html").write_text("<html></html>", encoding="utf-8")
            return ["media/media_manifest.json", "media/credits.md"]

    class FakeMotion(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "motion").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "motion" / "render_plan.json").write_text('{"scenes": [], "overlays": []}', encoding="utf-8")
            (self.run_dir / "motion" / "captions.ass").write_text("[Script Info]\n", encoding="utf-8")
            return ["motion/render_plan.json", "motion/captions.ass"]

    class FakeRender(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "render").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "render" / "final.mp4").write_bytes(b"MP4NEW")
            (self.run_dir / "render" / "description.md").write_text("# Final Desc\n", encoding="utf-8")
            return ["render/final.mp4", "render/description.md"]

    class FakeQA(FakeAgent):
        def run(self, manifest):
            (self.run_dir / "qa" / "storyboard_frames").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "qa" / "qa_report.json").write_text(
                json.dumps({"passed": True, "checks": [{"name": "ok", "passed": True, "detail": "ok"}], "warnings": []}),
                encoding="utf-8",
            )
            (self.run_dir / "qa" / "contact_sheet.jpg").write_bytes(b"JPG")
            (self.run_dir / "qa" / "storyboard.html").write_text("<html><img src='storyboard_frames/frame_0000.jpg'></html>", encoding="utf-8")
            (self.run_dir / "qa" / "storyboard_frames" / "frame_0000.jpg").write_bytes(b"FRAME0")
            return ["qa/qa_report.json", "qa/contact_sheet.jpg", "qa/storyboard.html"]

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "voice", FakeVoice)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "timeline", FakeTimeline)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "media", FakeMedia)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "motion", FakeMotion)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "render", FakeRender)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "qa", FakeQA)

    res = orch.run_finish(latest_dir=latest_dir)
    assert res["ok"] is True
    assert seen_media_auto == [True]
    assert res["status"] == "done"

    final_manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["status"] == "done"
    assert (orch.run_dir / "STATUS.md").exists()

    # Check data/latest replaced outputs
    for fname in ("final.mp4", "description.md", "qa_report.json", "storyboard.html", "contact_sheet.jpg", "STATUS.md", "README.txt"):
        assert (latest_dir / fname).exists()
    assert (latest_dir / "final.mp4").read_bytes() == b"MP4NEW"
    assert (latest_dir / "storyboard_frames" / "frame_0000.jpg").read_bytes() == b"FRAME0"
    readme_text = (latest_dir / "README.txt").read_text(encoding="utf-8")
    assert "TEST RUN - DO NOT PUBLISH" not in readme_text
    assert "Run Date:    2026-11-10" in readme_text
    assert "Source Health:" in readme_text
    assert readme_text.index("Source Health:") < readme_text.index("Files Included:")

    # Check configured outputs.mirror_dir also received everything and purged stale frames
    assert res["latest"]["mirror_dir"] == str(mirror_dir)
    for fname in ("final.mp4", "description.md", "qa_report.json", "storyboard.html", "contact_sheet.jpg", "STATUS.md", "README.txt"):
        assert (mirror_dir / fname).exists()
    assert (mirror_dir / "final.mp4").read_bytes() == b"MP4NEW"
    assert (mirror_dir / "storyboard_frames" / "frame_0000.jpg").read_bytes() == b"FRAME0"
    assert not (mirror_dir / "storyboard_frames" / "old_frame.jpg").exists()


# ---------------------------------------------------------------------------
# 2. --morning stops at needs-writer + retries scraper once after 5 minutes
# ---------------------------------------------------------------------------


def test_morning_stops_at_needs_writer_and_retries_scraper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    attempts = {"scraper": 0}

    class FlakyScraper:
        def __init__(self, run_dir: Path, config=None, backfill: bool = False):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            attempts["scraper"] += 1
            if attempts["scraper"] == 1:
                raise RuntimeError("Transient RSS timeout")
            (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "raw" / "articles.json").write_text(
                json.dumps([
                    {"source": "Kathmandu Post", "title": "Story 1"},
                    {"source": "MyRepublica", "title": "Story 2"},
                ]),
                encoding="utf-8",
            )
            (self.run_dir / "raw" / "news.md").write_text("# News\n", encoding="utf-8")
            manifest["stages"]["scraper"]["metrics"] = {
                "total_articles": 2,
                "per_source_stats": {
                    "Kathmandu Post": {"found": 5, "extracted_ok": 1},
                    "MyRepublica": {"found": 4, "extracted_ok": 1},
                },
            }
            return ["raw/articles.json", "raw/news.md"]

        def validate(self, outputs):
            return True, None

    class PrepareOnlyEditor:
        def __init__(self, run_dir: Path, config=None):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            (self.run_dir / "script").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "script" / "editor_brief.md").write_text(
                f"# Brief\nRun Kind: {manifest.get('run_kind')}\n", encoding="utf-8"
            )
            (self.run_dir / "script" / "shortlist.json").write_text(json.dumps([{"cluster_id": 1}]), encoding="utf-8")
            manifest["stages"]["editor"]["status"] = "prepared"
            manifest["stages"]["editor"]["metrics"] = {"shortlisted_clusters": 1}
            return ["script/shortlist.json", "script/editor_brief.md"]

        def validate(self, outputs):
            return True, None

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "scraper", FlakyScraper)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "editor", PrepareOnlyEditor)

    slept: list[float] = []
    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    res = orch.run_morning(run_kind="test", retry_delay_sec=300.0, sleep_fn=slept.append)

    assert res["ok"] is True
    assert slept == [300.0]
    assert attempts["scraper"] == 2
    assert res["stopped_at"] == "needs writer"
    assert res["editor_status"] == "prepared"
    assert res["status"] == "paused"
    assert res["run_kind"] == "test"
    assert res["shortlist_count"] == 1
    assert res["total_articles"] == 2

    status_md = (orch.run_dir / "STATUS.md").read_text(encoding="utf-8")
    assert "Needs writer" in status_md
    assert "paused" in status_md


def test_morning_writes_error_to_status_after_double_scraper_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    class BrokenScraper:
        def __init__(self, run_dir: Path, config=None, backfill: bool = False):
            pass

        def run(self, manifest):
            raise RuntimeError("All RSS feeds down")

        def validate(self, outputs):
            return True, None

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "scraper", BrokenScraper)

    slept: list[float] = []
    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    res = orch.run_morning(run_kind="test", retry_delay_sec=300.0, sleep_fn=slept.append)

    assert res["ok"] is False
    assert slept == [300.0]
    assert res["status"] == "failed"
    assert "All RSS feeds down" in res["error"]

    status_md = (orch.run_dir / "STATUS.md").read_text(encoding="utf-8")
    assert "All RSS feeds down" in status_md
    assert "`failed`" in status_md


# ---------------------------------------------------------------------------
# 3. Manifest status computation & stale (>30m) interrupted detection + --only
# ---------------------------------------------------------------------------


def test_manifest_status_computation_and_stale_detection(tmp_path: Path) -> None:
    pipeline_cfg = {
        "stages": [
            {"id": "scraper"},
            {"id": "editor"},
            {"id": "voice"},
            {"id": "qa"},
        ]
    }
    run_dir = tmp_path / "2026-11-10"
    run_dir.mkdir(parents=True, exist_ok=True)
    m = create_initial_manifest("2026-11-10", "2026-11-10", pipeline_cfg)

    # All done -> done
    for s in m["stages"].values():
        s["status"] = "done"
    assert compute_manifest_status(m, run_dir=run_dir) == "done"

    # Paused stage overrides pending/done
    m["stages"]["editor"]["status"] = "prepared"
    m["stages"]["voice"]["status"] = "pending"
    m["stages"]["qa"]["status"] = "pending"
    assert compute_manifest_status(m, run_dir=run_dir) == "paused"

    # Failed overrides paused
    m["stages"]["voice"]["status"] = "failed"
    assert compute_manifest_status(m, run_dir=run_dir) == "failed"

    # Active PID lock -> running
    m["stages"]["editor"]["status"] = "done"
    m["stages"]["voice"]["status"] = "running"
    acquire_run_lock(run_dir)
    try:
        assert compute_manifest_status(m, run_dir=run_dir) == "running"
    finally:
        release_run_lock(run_dir)

    # No active PID lock and recent (<30 min) -> pending (NOT falsely "running")
    now = datetime.now().astimezone()
    m["status"] = "running"
    m["stages"]["voice"]["status"] = "running"
    m["stages"]["voice"]["start_time"] = (now - timedelta(minutes=5)).isoformat()
    m["updated_at"] = (now - timedelta(minutes=5)).isoformat()
    assert compute_manifest_status(m, run_dir=run_dir, now=now) == "pending"

    # No active PID lock and >=30 min stale -> interrupted
    m["stages"]["voice"]["start_time"] = (now - timedelta(minutes=35)).isoformat()
    m["updated_at"] = (now - timedelta(minutes=35)).isoformat()
    assert compute_manifest_status(m, run_dir=run_dir, now=now) == "interrupted"

    status_md = render_status_markdown(run_dir, m)
    assert "`interrupted`" in status_md
    assert "--reset-stage voice" in status_md


def test_only_run_recomputes_manifest_status_instead_of_leaving_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    class QuickQA:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            return ["qa/qa_report.json"]

        def validate(self, outputs):
            return True, None

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "qa", QuickQA)

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir, allow_fixture=True)
    manifest = orch.initialize_run()
    for st_name in manifest["stages"]:
        manifest["stages"][st_name]["status"] = "done"
    manifest["status"] = "running"
    (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (orch.run_dir / "script").mkdir(parents=True, exist_ok=True)
    (orch.run_dir / "script" / "script.json").write_text(json.dumps({"origin": "fixture"}), encoding="utf-8")

    ok = orch.run_pipeline(only_stage="qa")
    assert ok is True
    updated = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated["status"] == "done"

    # Partial run where prior stage was paused -> --only scraper leaves top status 'paused', never 'running'
    updated["stages"]["editor"]["status"] = "prepared"
    updated["status"] = "running"
    (orch.run_dir / "manifest.json").write_text(json.dumps(updated), encoding="utf-8")
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "scraper", QuickQA)
    ok2 = orch.run_pipeline(only_stage="scraper")
    assert ok2 is True
    updated2 = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated2["status"] == "paused"
    assert not (orch.run_dir / ".pipeline.pid").exists()


# ---------------------------------------------------------------------------
# 4. --cleanup --dry-run lists only allowed paths & never protected ones
# ---------------------------------------------------------------------------


def test_cleanup_dry_run_allowed_and_protected_paths(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)

    # Protected top-level trees
    for rel in (
        "data/media_library/bg.mp4",
        ".env",
        "models/MMS_FA/model.pt",
        "vendor/ming/weights.bin",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("KEEP", encoding="utf-8")

    runs_dir = tmp_path / "data" / "runs"
    fixture_cache = runs_dir / "_fixture_run" / "render" / "cache" / "x.mp4"
    fixture_cache.parent.mkdir(parents=True, exist_ok=True)
    fixture_cache.write_bytes(b"keep-fixture")

    # 9 dated runs -> keep_runs=7 keeps 2026-11-03..2026-11-09, cleans 2026-11-01 & 2026-11-02
    for day in range(1, 10):
        rdir = runs_dir / f"2026-11-{day:02d}"
        temp_files = [
            rdir / "media" / "shots" / "scene_01" / "shot_00.mp4",
            rdir / "media" / "review" / "thumbs" / "cand_1.jpg",
            rdir / "audio" / "scenes" / "scene_01.wav",
            rdir / "audio" / "temp_sentences" / "s1.wav",
            rdir / "audio" / "raw_concat.wav",
            rdir / "audio" / "scene_01.wav",
            rdir / "render" / "cache" / "seg.mp4",
            rdir / "render" / "video_noaudio.mp4",
            rdir / "render" / "concat_segments.txt",
            rdir / "qa" / "_tmp_frames" / "f.png",
            rdir / "timeline" / "timeline_preview.mp4",
        ]
        keep_files = [
            rdir / "script" / "script.json",
            rdir / "render" / "description.md",
            rdir / "render" / "final.mp4",
            rdir / "qa" / "qa_report.json",
            rdir / "manifest.json",
            rdir / "media" / "credits.md",
        ]
        for tf in temp_files:
            tf.parent.mkdir(parents=True, exist_ok=True)
            tf.write_bytes(b"TEMP123456")
        for kf in keep_files:
            kf.parent.mkdir(parents=True, exist_ok=True)
            kf.write_text("KEEP", encoding="utf-8")

    # Stale cache entry (older than keep_days=7) inside downloads/
    cache_dir = tmp_path / "data" / "media_cache" / "downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stale_cache = cache_dir / "old_blob.mp4"
    stale_cache.write_bytes(b"OLD_CACHE")
    old_epoch = (datetime.now() - timedelta(days=30)).timestamp()
    os.utime(stale_cache, (old_epoch, old_epoch))

    plan = plan_cleanup(
        runs_dir=runs_dir,
        media_config_path=cfg_dir / "media.yaml",
        pipeline_config_dir=cfg_dir,
        keep_runs=7,
        repo_root=tmp_path,
    )
    assert plan["older_runs"] == ["2026-11-02", "2026-11-01"]
    assert len(plan["kept_runs"]) == 7

    dry_res = execute_cleanup(plan, dry_run=True, repo_root=tmp_path)
    preview_str = format_cleanup_plan(plan)
    assert "2026-11-01" in preview_str
    assert "2026-11-02" in preview_str
    assert "_fixture_run" not in preview_str

    for item in dry_res["deleted_paths"]:
        p = Path(item)
        # Only older runs may appear in deleted_paths
        assert "2026-11-01" in item or "2026-11-02" in item
        for kept_day in range(3, 10):
            assert f"2026-11-{kept_day:02d}" not in item
        for forbidden in ("media_library", ".env", "models", "vendor", "script", "description.md", "final.mp4", "qa_report.json", "manifest.json", "credits.md"):
            assert p.name != forbidden and forbidden not in p.parts

    # Dry-run must leave everything intact on disk
    assert stale_cache.exists()
    assert (runs_dir / "2026-11-01" / "media" / "shots" / "scene_01" / "shot_00.mp4").exists()

    # Actual execute deletes only older temp targets + stale cache, keeping protected files
    real_res = execute_cleanup(plan, dry_run=False, repo_root=tmp_path)
    assert len(real_res["deleted_paths"]) > 0
    assert not (runs_dir / "2026-11-01" / "media" / "shots").exists()
    assert not (runs_dir / "2026-11-02" / "render" / "cache").exists()
    assert not stale_cache.exists()

    # All protected files and kept runs remain untouched
    for prot in ("data/media_library/bg.mp4", ".env", "models/MMS_FA/model.pt", "vendor/ming/weights.bin"):
        assert (tmp_path / prot).exists()
    assert fixture_cache.exists()
    for day in range(1, 10):
        rdir = runs_dir / f"2026-11-{day:02d}"
        for krel in ("script/script.json", "render/description.md", "render/final.mp4", "qa/qa_report.json", "manifest.json", "media/credits.md"):
            assert (rdir / krel).exists()
    for kept_day in range(3, 10):
        assert (runs_dir / f"2026-11-{kept_day:02d}" / "media" / "shots" / "scene_01" / "shot_00.mp4").exists()


# ---------------------------------------------------------------------------
# 5. --doctor reports missing items & never leaks secret values
# ---------------------------------------------------------------------------


def test_doctor_reports_missing_items_and_redacts_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)

    for k in ("PIXABAY_API_KEY", "COVERR_API_KEY", "OPENVERSE_TOKEN", "UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    secret_val = "SECRET_API_KEY_VALUE_DO_NOT_PRINT"
    (tmp_path / ".env").write_text(
        f"PEXELS_API_KEY={secret_val}\nPIXABAY_API_KEY=\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.example").write_text(
        "PIXABAY_API_KEY=\nPEXELS_API_KEY=\nUNSPLASH_ACCESS_KEY=\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "sources.yaml").write_text(
        "settings:\n  user_agent: TestBot/1.0\nsources:\n  - name: ReachableFeed\n    rss: https://example.com/ok.xml\n    enabled: true\n  - name: DownFeed\n    rss: https://example.com/down.xml\n    enabled: true\n",
        encoding="utf-8",
    )

    def fake_http(method, url, **kwargs):
        if "ok.xml" in url:
            return SimpleNamespace(status_code=200, close=lambda: None)
        raise ConnectionError("DNS lookup failed")

    fake_disk = lambda _p: SimpleNamespace(total=100 * 1024**3, used=97 * 1024**3, free=3 * 1024**3)

    checks = run_doctor(
        repo_root=tmp_path,
        which_fn=lambda _cmd: None,
        disk_usage_fn=fake_disk,
        http_request_fn=fake_http,
        min_disk_gb=10.0,
    )
    by_name = {c.name: c for c in checks}
    assert by_name["ffmpeg"].status == "FAIL"
    assert by_name["ffprobe"].status == "FAIL"
    assert by_name[".venv-tts python"].status == "FAIL"
    assert by_name["CUDA"].status == "FAIL"
    assert by_name["Ming model"].status == "FAIL"
    assert by_name["MMS_FA model"].status == "FAIL"
    assert by_name["Disk space"].status == "WARN"
    assert by_name[".env keys"].status == "FAIL"  # PIXABAY_API_KEY required by config/media.yaml is missing
    assert "PEXELS_API_KEY=set" in by_name[".env keys"].detail
    assert "PIXABAY_API_KEY=missing" in by_name[".env keys"].detail
    assert by_name["Feed reachability"].status == "FAIL"
    assert "DownFeed" in by_name["Feed reachability"].detail
    assert by_name["Write permissions"].status == "PASS"

    report = format_doctor_report(checks)
    assert secret_val not in report
    assert "Fix:" in report


# ---------------------------------------------------------------------------
# 6. production run_kind removes TEST RUN tag and banner
# ---------------------------------------------------------------------------


def test_production_run_kind_removes_test_run_tag_and_banner(tmp_path: Path) -> None:
    def _build_motion_fixture(run_dir: Path, run_kind: str) -> tuple[dict, Path]:
        (run_dir / "script").mkdir(parents=True, exist_ok=True)
        (run_dir / "timeline").mkdir(parents=True, exist_ok=True)
        (run_dir / "media").mkdir(parents=True, exist_ok=True)
        (run_dir / "audio").mkdir(parents=True, exist_ok=True)

        create_silence_wav(run_dir / "audio" / "narration.wav", duration_sec=6.0, sample_rate=48000)
        script_data = {
            "origin": "fixture",
            "date": "2026-11-10",
            "run_kind": run_kind,
            "scenes": [
                {"id": "intro", "type": "intro", "headline": "Intro"},
                {"id": "scene_01", "type": "story", "headline": "Main Story", "sources": ["MyRepublica"], "source_urls": ["https://myrepublica.com/x"]},
                {"id": "outro", "type": "outro", "headline": "Outro"},
            ],
        }
        (run_dir / "script" / "script.json").write_text(json.dumps(script_data), encoding="utf-8")
        (run_dir / "timeline" / "timeline.json").write_text(
            json.dumps({
                "duration_sec": 6.0,
                "scenes": [
                    {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
                    {"id": "scene_01", "type": "story", "start": 2.0, "end": 4.0},
                    {"id": "outro", "type": "outro", "start": 4.0, "end": 6.0},
                ],
            }),
            encoding="utf-8",
        )
        (run_dir / "timeline" / "captions.json").write_text(
            json.dumps([{"id": "c1", "scene_id": "scene_01", "start": 2.1, "end": 3.8, "text": "Test line"}]),
            encoding="utf-8",
        )
        (run_dir / "media" / "media_manifest.json").write_text('{"shots": []}', encoding="utf-8")

        agent = MotionAgent(run_dir=run_dir)
        agent.run({"run_kind": run_kind})
        plan = json.loads((run_dir / "motion" / "render_plan.json").read_text(encoding="utf-8"))
        return plan, run_dir

    test_plan, test_dir = _build_motion_fixture(tmp_path / "run_test", "test")
    prod_plan, prod_dir = _build_motion_fixture(tmp_path / "run_prod", "production")

    assert any(ov.get("type") == "test_tag" for ov in test_plan.get("overlays", []))
    assert (test_dir / "motion" / "overlays" / "test_run_tag.png").exists()

    assert not any(ov.get("type") == "test_tag" for ov in prod_plan.get("overlays", []))
    assert not (prod_dir / "motion" / "overlays" / "test_run_tag.png").exists()

    # Description banner check
    r_agent = RenderAgent(run_dir=prod_dir)
    script_prod = json.loads((prod_dir / "script" / "script.json").read_text(encoding="utf-8"))
    tl_prod = json.loads((prod_dir / "timeline" / "timeline.json").read_text(encoding="utf-8"))

    desc_prod = prod_dir / "desc_prod.md"
    r_agent._generate_description_markdown(script_prod, tl_prod, {"shots": []}, desc_prod, manifest={"run_kind": "production"})
    assert "TEST RUN - DO NOT PUBLISH" not in desc_prod.read_text(encoding="utf-8")

    desc_test = test_dir / "desc_test.md"
    r_agent._generate_description_markdown({"date": "2026-11-10", "run_kind": "test", "scenes": script_prod["scenes"]}, tl_prod, {"shots": []}, desc_test)
    assert "TEST RUN - DO NOT PUBLISH" in desc_test.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Storyboard generation
# ---------------------------------------------------------------------------


def test_storyboard_generation(tmp_path: Path) -> None:
    qa_dir = tmp_path / "qa"
    mp4_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=1280x720:r=30:d=6.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(mp4_path),
        ],
        capture_output=True,
        check=True,
    )

    script_data = {
        "date": "2026-11-10",
        "run_kind": "test",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Broadcast Opener"},
            {"id": "scene_01", "type": "story", "headline": "Valley Air Quality Update", "sources": ["Kathmandu Post"]},
        ],
    }
    timeline_data = {
        "duration_sec": 6.0,
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
            {"id": "scene_01", "type": "story", "start": 2.0, "end": 6.0},
        ],
    }
    media_data = {
        "shots": [
            {"scene_id": "scene_01", "shot_index": 0, "start": 2.0, "end": 6.0, "source": "wikimedia_commons", "type": "image"},
        ]
    }
    render_plan = {
        "duration_sec": 6.0,
        "scenes": [
            {"id": "intro", "start": 0.0, "end": 2.0, "shots": []},
            {
                "id": "scene_01",
                "start": 2.0,
                "end": 6.0,
                "lower_third": {"headline": "Valley Air Quality Update", "source_credit": "Kathmandu Post"},
                "shots": [{"shot_index": 0, "start": 2.0, "end": 6.0, "type": "image"}],
            },
        ],
    }
    captions_data = [
        {"id": "c1", "scene_id": "intro", "start": 0.2, "end": 1.8, "text": "Good morning Nepal."},
        {"id": "c2", "scene_id": "scene_01", "start": 2.2, "end": 5.5, "text": "Pollution levels dipped across the valley today."},
    ]
    qa_report = {
        "passed": True,
        "checks": [{"name": "Visual freeze check", "passed": True, "detail": "1 warning"}],
        "warnings": ["Detected 1 frozen video stretch(es) > 3.0s (e.g. at ['3.5']s)"],
        "metrics": {
            "video_duration_sec": 6.0,
            "integrated_loudness_lufs": -16.2,
            "freeze_events": [{"start": 2.4, "end": 5.7, "duration": 3.3}],
        },
    }

    sb_path, frames = generate_storyboard(
        final_mp4=mp4_path,
        qa_dir=qa_dir,
        script_data=script_data,
        timeline_data=timeline_data,
        media_data=media_data,
        render_plan=render_plan,
        captions_data=captions_data,
        qa_report=qa_report,
        interval_sec=4.0,
        frame_width=480,
    )

    assert sb_path.exists()
    assert len(frames) >= 2
    for fp in frames:
        assert fp.exists()
        with Image.open(fp) as im:
            assert im.width == 480

    html_text = sb_path.read_text(encoding="utf-8")
    assert "storyboard_frames/frame_0000.jpg" in html_text
    assert "Broadcast Opener" in html_text
    assert "Valley Air Quality Update" in html_text
    assert "Kathmandu Post" in html_text
    assert "wikimedia_commons" in html_text
    assert "Pollution levels dipped across the valley today." in html_text
    assert "2.40s" in html_text or "2.4s" in html_text
    assert "http://" not in html_text and "https://" not in html_text  # self-contained for file:// double-click


# ---------------------------------------------------------------------------
# 8. Still image continuous zoom prevents frozen stretches (>3s)
# ---------------------------------------------------------------------------


def test_render_still_image_linear_zoom_no_freeze(tmp_path: Path) -> None:
    # Build a textured 1280x720 JPEG (solid colors look frozen even while zooming)
    img_path = tmp_path / "textured.jpg"
    img = Image.new("RGB", (1280, 720))
    px = img.load()
    for y in range(720):
        for x in range(1280):
            px[x, y] = ((x * 5) % 256, (y * 3) % 256, ((x + y) * 2) % 256)
    img.save(img_path, quality=92)

    out_seg = tmp_path / "scene_still.mp4"
    agent = RenderAgent(run_dir=tmp_path)
    scene_info = {
        "id": "scene_02",
        "start": 0.0,
        "end": 8.5,
        "shots": [
            {
                "shot_index": 0,
                "start": 0.0,
                "end": 8.5,
                "type": "image",
                "file": str(img_path),
                "motion": "ken_burns",
            }
        ],
        "overlays": [],
    }
    agent._render_scene_segment(
        sc=scene_info,
        fps=30,
        target_res=(1280, 720),
        output_path=out_seg,
    )
    assert out_seg.exists()

    fd = subprocess.run(
        ["ffmpeg", "-i", str(out_seg), "-vf", "freezedetect=n=-60dB:d=2.0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "lavfi.freezedetect.freeze_start" not in fd.stderr


# ---------------------------------------------------------------------------
# 9. Media --auto selects clips >= min_relevance and creates cards below threshold
# ---------------------------------------------------------------------------


def test_media_auto_mode_uses_cards_below_min_relevance_and_keeps_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Copy configs and set isolated media_cache & media_library paths
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    media_cfg_text = (cfg_dir / "media.yaml").read_text(encoding="utf-8")
    media_cfg_text = media_cfg_text.replace("data/media_cache", str((tmp_path / "media_cache").as_posix()))
    media_cfg_text = media_cfg_text.replace("data/media_library", str((tmp_path / "media_library").as_posix()))
    (cfg_dir / "media.yaml").write_text(media_cfg_text, encoding="utf-8")

    run_dir = tmp_path / "2026-11-10"
    (run_dir / "script").mkdir(parents=True, exist_ok=True)
    (run_dir / "timeline").mkdir(parents=True, exist_ok=True)
    script_data = {
        "origin": "fixture",
        "date": "2026-11-10",
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Koshi Flood Relief",
                "visual_keywords": ["koshi river nepal", "flood relief"],
                "narration": "Relief teams arrived near the Koshi embankment.",
            }
        ],
    }
    (run_dir / "script" / "script.json").write_text(json.dumps(script_data), encoding="utf-8")
    (run_dir / "timeline" / "timeline.json").write_text(
        json.dumps({"duration_sec": 14.0, "scenes": [{"id": "scene_01", "type": "story", "start": 0.0, "end": 14.0}]}),
        encoding="utf-8",
    )

    sample_jpg = tmp_path / "cand_src.jpg"
    Image.new("RGB", (640, 360), color=(20, 80, 140)).save(sample_jpg)

    orig_cwd = Path.cwd()
    monkeypatch.chdir(tmp_path)
    try:
        agent = MediaAgent(run_dir=run_dir, config={"settings": {"default_run_kind": "test"}}, allow_fixture=True, offline=False, auto=True)
        min_rel = float(agent.media_config.get("settings", {}).get("min_relevance", 0.35))

        def fake_iter_candidates(scene, shot, stills_in_scene=0, max_stills=10):
            if shot.shot_index == 0:
                yield (
                    Candidate(
                        source="wikimedia_commons",
                        id="high_1",
                        kind="image",
                        page_url="https://commons.wikimedia.org/wiki/File:Koshi.jpg",
                        download_url="https://upload.wikimedia.org/koshi.jpg",
                        width=1920,
                        height=1080,
                        duration=None,
                        license_id="CC BY 4.0",
                        license_url="https://creativecommons.org/licenses/by/4.0",
                        author="Photographer",
                        title="Koshi Barrage Nepal",
                        attribution_text="Photo by Photographer via Wikimedia Commons",
                        attribution_required=True,
                        score=max(0.85, min_rel + 0.3),
                    ),
                    "koshi river nepal",
                    "wikimedia_commons",
                )
            else:
                yield (
                    Candidate(
                        source="pixabay",
                        id="low_1",
                        kind="image",
                        page_url="https://pixabay.com/photos/1/",
                        download_url="https://pixabay.com/get/1.jpg",
                        width=1920,
                        height=1080,
                        duration=None,
                        license_id="Pixabay License",
                        license_url="https://pixabay.com/service/license-summary/",
                        author="Tester",
                        title="Random flower",
                        attribution_required=False,
                        score=max(0.05, min_rel - 0.2),
                    ),
                    "flood relief",
                    "pixabay",
                )

        monkeypatch.setattr(agent.selector, "iter_candidates_for_shot", fake_iter_candidates)
        monkeypatch.setattr(media_mod, "save_candidate_thumbnail", lambda *args, **kwargs: "")
        monkeypatch.setattr(agent.cache_mgr, "download_file", lambda url, offline=False: sample_jpg)

        manifest = {"stages": {"media": {"metrics": {}}}}
        outputs = agent.run(manifest)
        assert "media/media_manifest.json" in outputs
        assert (run_dir / "media" / "review" / "picker.html").exists()
        assert (run_dir / "media" / "review" / "auto_picks.json").exists()

        mm = json.loads((run_dir / "media" / "media_manifest.json").read_text(encoding="utf-8"))
        shots = mm["shots"]
        assert len(shots) == 2
        assert shots[0]["type"] == "image"
        assert shots[0]["source"] == "wikimedia_commons"
        assert shots[1]["type"] == "card"
        assert manifest["stages"]["media"]["metrics"]["auto_mode"] is True
        assert manifest["stages"]["media"]["metrics"]["auto_clips_count"] == 1
        assert manifest["stages"]["media"]["metrics"]["auto_cards_count"] == 1
    finally:
        monkeypatch.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# 10. Windows schedule script -DryRun output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Scheduled Task PowerShell script test")
def test_schedule_install_dry_run_powershell() -> None:
    ps_script = REPO_ROOT / "scripts" / "install_schedule.ps1"
    assert ps_script.exists()

    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_script), "-DryRun"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "NepalNewsBot-Morning" in out
    assert "07:00" in out
    assert "python.exe" in out
    assert "run_daily.py --morning" in out
    assert "logs\\morning_" in out
    assert "StartWhenAvailable" in out
    assert "WakeToRun" in out
    assert "Interactive" in out
    assert "C:\\news-auto" in out
    assert "DRY RUN" in out


# ---------------------------------------------------------------------------
# 11. Step 8B: Idempotent --morning and --finish + --force override
# ---------------------------------------------------------------------------


def test_morning_is_idempotent_two_runs_and_force_rescrapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    runs_dir = tmp_path / "runs"

    calls = {"scraper": 0, "editor": 0}

    class CountingScraper:
        def __init__(self, run_dir: Path, config=None, backfill: bool = False):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            calls["scraper"] += 1
            (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "raw" / "articles.json").write_text(
                json.dumps([{"source": "Kathmandu Post", "title": f"Story {calls['scraper']}"}]),
                encoding="utf-8",
            )
            (self.run_dir / "raw" / "news.md").write_text("# News\n", encoding="utf-8")
            manifest["stages"]["scraper"]["metrics"] = {
                "total_articles": 1,
                "per_source_stats": {
                    "Kathmandu Post": {"found": 3, "extracted_ok": 1, "newest_age_hours": 1.5},
                },
            }
            return ["raw/articles.json", "raw/news.md"]

        def validate(self, outputs):
            return True, None

    class CountingEditor:
        def __init__(self, run_dir: Path, config=None):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            calls["editor"] += 1
            (self.run_dir / "script").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "script" / "editor_brief.md").write_text(
                f"# Brief v{calls['editor']}\n", encoding="utf-8"
            )
            (self.run_dir / "script" / "shortlist.json").write_text(json.dumps([{"cluster_id": 1}]), encoding="utf-8")
            manifest["stages"]["editor"]["status"] = "prepared"
            manifest["stages"]["editor"]["metrics"] = {"shortlisted_clusters": 1}
            return ["script/shortlist.json", "script/editor_brief.md"]

        def validate(self, outputs):
            return True, None

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "scraper", CountingScraper)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "editor", CountingEditor)

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)

    # First morning run executes scraper + editor prepare
    res1 = orch.run_morning(run_kind="test")
    assert res1["ok"] is True
    assert res1["already_done"] is False
    assert calls == {"scraper": 1, "editor": 1}
    assert (orch.run_dir / "script" / "editor_brief.md").read_text(encoding="utf-8") == "# Brief v1\n"

    # Second consecutive morning run is idempotent: skips without re-scraping or resetting
    res2 = orch.run_morning(run_kind="test")
    assert res2["ok"] is True
    assert res2["already_done"] is True
    assert res2["skipped"] is True
    assert res2["stopped_at"] == "needs writer"
    assert res2["total_articles"] == 1
    assert res2["shortlist_count"] == 1
    assert calls == {"scraper": 1, "editor": 1}
    assert (orch.run_dir / "script" / "editor_brief.md").read_text(encoding="utf-8") == "# Brief v1\n"

    # --force overrides idempotency and re-scrapes + re-prepares
    res3 = orch.run_morning(run_kind="production", force=True)
    assert res3["ok"] is True
    assert res3["already_done"] is False
    assert calls == {"scraper": 2, "editor": 2}
    assert (orch.run_dir / "script" / "editor_brief.md").read_text(encoding="utf-8") == "# Brief v2\n"


def test_finish_is_idempotent_unless_forced_and_still_checks_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    mirror_dir = tmp_path / "mirror_idempotent"
    pipe_text = (cfg_dir / "pipeline.yaml").read_text(encoding="utf-8")
    pipe_text = pipe_text.replace(
        "C:/Users/Chungba Sherpa/OneDrive/NepalNewsBot/latest",
        mirror_dir.as_posix(),
    )
    (cfg_dir / "pipeline.yaml").write_text(pipe_text, encoding="utf-8")
    runs_dir = tmp_path / "runs"
    latest_dir = tmp_path / "latest"

    orch = Orchestrator(run_date="2026-11-10", config_dir=cfg_dir, runs_dir=runs_dir)
    manifest = orch.initialize_run()
    manifest["stages"]["scraper"]["status"] = "done"
    manifest["stages"]["editor"]["status"] = "done"
    (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    script_path, script_data = _create_approved_script(orch.run_dir, run_kind="test")

    stage_calls = {"count": 0}

    class CountingStage:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, manifest):
            stage_calls["count"] += 1
            return []

        def validate(self, outputs):
            return True, None

    for st in ("voice", "timeline", "media", "motion", "render", "qa"):
        monkeypatch.setitem(orch_mod.AGENT_REGISTRY, st, CountingStage)

    res1 = orch.run_finish(latest_dir=latest_dir)
    assert res1["ok"] is True
    assert res1["already_finished"] is False
    assert stage_calls["count"] == 6

    # Second --finish on already finished day skips without running any stages
    res2 = orch.run_finish(latest_dir=latest_dir)
    assert res2["ok"] is True
    assert res2["already_finished"] is True
    assert res2["skipped"] is True
    assert stage_calls["count"] == 6

    # --force re-runs all 6 downstream stages
    res3 = orch.run_finish(latest_dir=latest_dir, force=True)
    assert res3["ok"] is True
    assert res3["already_finished"] is False
    assert stage_calls["count"] == 12

    # Even on an already-finished run, tampering script.json still fails the approval gate
    script_data["scenes"][1]["headline"] = "Tampered Post Finish"
    script_path.write_text(json.dumps(script_data), encoding="utf-8")
    res_tampered = orch.run_finish(latest_dir=latest_dir)
    assert res_tampered["ok"] is False
    assert res_tampered["refused"] is True
    assert "script changed after approval" in res_tampered["reason"]


# ---------------------------------------------------------------------------
# 12. Step 8B: Source Health table (OK / STALE / EMPTY / ERROR / DEGRADED)
# ---------------------------------------------------------------------------


def test_source_health_statuses_degraded_two_runs_and_non_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    (cfg_dir / "sources.yaml").write_text(
        "\n".join([
            "settings:",
            "  lookback_hours: 30",
            "  min_articles: 2",
            "sources:",
            "  - name: Outlet OK",
            "    rss: https://ok.example/rss",
            "    enabled: true",
            "  - name: Outlet Stale",
            "    rss: https://stale.example/rss",
            "    enabled: true",
            "  - name: Outlet Empty",
            "    rss: https://empty.example/rss",
            "    enabled: true",
            "  - name: Outlet Down",
            "    rss: https://down.example/rss",
            "    enabled: true",
        ]),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"

    def _write_run(day: str, per_source_stats: dict, source_errors: dict | None = None) -> Path:
        rdir = runs_dir / day
        rdir.mkdir(parents=True, exist_ok=True)
        m = {
            "run_id": day,
            "date": day,
            "run_kind": "test",
            "status": "paused",
            "updated_at": f"{day}T07:05:00+05:45",
            "stages": {
                "scraper": {
                    "status": "done",
                    "metrics": {
                        "total_articles": sum(v.get("extracted_ok", 0) for v in per_source_stats.values()),
                        "per_source_stats": per_source_stats,
                    },
                    "source_errors": source_errors or {},
                },
                "editor": {"status": "prepared"},
            },
        }
        (rdir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        return rdir

    # Run 1 (2026-11-09): Outlet Stale is STALE (>48h), Outlet Empty is EMPTY
    r09 = _write_run(
        "2026-11-09",
        per_source_stats={
            "Outlet OK": {"found": 10, "extracted_ok": 8, "newest_age_hours": 1.2},
            "Outlet Stale": {"found": 15, "extracted_ok": 0, "newest_age_hours": 60.0},
            "Outlet Empty": {"found": 0, "extracted_ok": 0, "newest_age_hours": None},
            "Outlet Down": {"found": 5, "extracted_ok": 5, "newest_age_hours": 3.0},
        },
    )
    rows09 = compute_source_health(r09, config_dir=cfg_dir)
    by09 = {r["source"]: r for r in rows09}
    assert by09["Outlet OK"]["display_status"] == "OK"
    assert by09["Outlet Stale"]["display_status"] == "STALE"
    assert by09["Outlet Stale"]["degraded"] is False
    assert by09["Outlet Empty"]["display_status"] == "EMPTY"

    # Run 2 (2026-11-10): Outlet Stale & Outlet Empty bad again -> DEGRADED; Outlet Down hits ERROR
    r10 = _write_run(
        "2026-11-10",
        per_source_stats={
            "Outlet OK": {"found": 12, "extracted_ok": 9, "newest_age_hours": 0.5},
            "Outlet Stale": {"found": 14, "extracted_ok": 0, "newest_age_hours": 72.5},
            "Outlet Empty": {"found": 4, "extracted_ok": 0, "newest_age_hours": 12.0},
            "Outlet Down": {"found": 0, "extracted_ok": 0, "newest_age_hours": None},
        },
        source_errors={"Outlet Down": "HTTP 500 timeout"},
    )
    rows10 = compute_source_health(r10, config_dir=cfg_dir)
    by10 = {r["source"]: r for r in rows10}
    assert by10["Outlet OK"]["display_status"] == "OK"
    assert by10["Outlet Stale"]["status"] == "STALE"
    assert by10["Outlet Stale"]["degraded"] is True
    assert by10["Outlet Stale"]["display_status"] == "STALE (DEGRADED)"
    assert by10["Outlet Empty"]["status"] == "EMPTY"
    assert by10["Outlet Empty"]["degraded"] is True
    assert by10["Outlet Empty"]["display_status"] == "EMPTY (DEGRADED)"
    assert by10["Outlet Down"]["display_status"] == "ERROR"

    # STATUS.md puts ## Source Health at the top before ## Next Action
    status_md = render_status_markdown(r10)
    assert "## Source Health" in status_md
    assert status_md.index("## Source Health") < status_md.index("## Next Action")
    assert "STALE (DEGRADED)" in status_md
    assert "EMPTY (DEGRADED)" in status_md
    assert "Outlet Down" in status_md

    # data/latest/README.txt puts Source Health: at the top before Files Included:
    latest_dir = tmp_path / "latest"
    publish_to_latest(r10, latest_dir=latest_dir, mirror_dir=None, config_dir=cfg_dir)
    readme = (latest_dir / "README.txt").read_text(encoding="utf-8")
    assert "Source Health:" in readme
    assert readme.index("Source Health:") < readme.index("Files Included:")
    assert "STALE (DEGRADED)" in readme

    # Run 3 (2026-11-11): Recovery clears DEGRADED
    r11 = _write_run(
        "2026-11-11",
        per_source_stats={
            "Outlet OK": {"found": 10, "extracted_ok": 10, "newest_age_hours": 1.0},
            "Outlet Stale": {"found": 10, "extracted_ok": 4, "newest_age_hours": 2.5},
            "Outlet Empty": {"found": 5, "extracted_ok": 2, "newest_age_hours": 3.0},
            "Outlet Down": {"found": 5, "extracted_ok": 5, "newest_age_hours": 1.0},
        },
    )
    rows11 = compute_source_health(r11, config_dir=cfg_dir)
    assert all(r["status"] == "OK" and not r["degraded"] for r in rows11)

    # Verify news.md + articles.json fallback when per_source_stats is absent in manifest
    r_legacy = runs_dir / "2026-11-08"
    (r_legacy / "raw").mkdir(parents=True, exist_ok=True)
    (r_legacy / "manifest.json").write_text(
        json.dumps({"run_id": "2026-11-08", "date": "2026-11-08", "stages": {"scraper": {"status": "done"}}}),
        encoding="utf-8",
    )
    (r_legacy / "raw" / "news.md").write_text(
        "### Per-Source Breakdown\n- **Outlet OK**: 5 kept (found: 10, dropped: 5)\n- **Outlet Stale**: 0 kept (found: 20, dropped: 20)\n",
        encoding="utf-8",
    )
    (r_legacy / "raw" / "articles.json").write_text(
        json.dumps([{"source": "Outlet OK", "title": "A", "age_hours": 4.2}]),
        encoding="utf-8",
    )
    rows_leg = compute_source_health(r_legacy, config_dir=cfg_dir)
    by_leg = {r["source"]: r for r in rows_leg}
    assert by_leg["Outlet OK"]["found"] == 10
    assert by_leg["Outlet OK"]["kept"] == 5
    assert by_leg["Outlet OK"]["newest_age_hours"] == 4.2
    assert by_leg["Outlet Stale"]["found"] == 20
    assert by_leg["Outlet Stale"]["kept"] == 0
    assert by_leg["Outlet Stale"]["status"] == "EMPTY"

    # Verify degraded/stale/empty source NEVER fails ScraperAgent.run() when total >= min_articles
    import yaml

    cfg_scrape = {
        "settings": {"lookback_hours": 30, "min_articles": 2, "user_agent": "Bot/1.0"},
        "sources": [
            {"name": "HealthyFeed", "rss": "https://healthy.example/rss", "enabled": True},
            {"name": "StaleFeed", "rss": "https://stale.example/rss", "enabled": True},
        ],
    }
    (cfg_dir / "sources.yaml").write_text(yaml.safe_dump(cfg_scrape), encoding="utf-8")

    monkeypatch.setattr(scraper_mod.ScraperAgent, "_is_url_allowed_by_robots", lambda self, u, ua, d: True)

    def fake_rss(
        self,
        source,
        user_agent,
        delay_seconds,
        ref_time,
        seen_urls,
        seen_titles,
        default_language="en",
        lookback_seconds=30 * 3600,
    ):
        name = source["name"]
        if name == "StaleFeed":
            return [], {
                "found": 10,
                "kept_filter": 0,
                "extracted_ok": 0,
                "dropped": 10,
                "newest_age_hours": 55.0,
            }, None
        now_iso = datetime.now().astimezone().isoformat()
        arts = [
            {
                "source": name,
                "title": f"Story {i}",
                "url": f"https://healthy.example/{i}",
                "published_at": now_iso,
                "word_count": 120,
                "full_text": "body " * 60,
                "text_summary": "sum",
            }
            for i in range(1, 7)
        ]
        return arts, {
            "found": 6,
            "kept_filter": 6,
            "extracted_ok": 6,
            "dropped": 0,
            "newest_age_hours": 0.4,
        }, None

    monkeypatch.setattr(ScraperAgent, "_fetch_rss_source", fake_rss)
    sc_dir = runs_dir / "2026-11-12"
    sc_dir.mkdir(parents=True, exist_ok=True)
    sc_manifest = {"stages": {"scraper": {"metrics": {}, "source_errors": {}}}}
    sc_agent = ScraperAgent(
        run_dir=sc_dir,
        sources_path=cfg_dir / "sources.yaml",
        pipeline_path=cfg_dir / "pipeline.yaml",
    )
    sc_outs = sc_agent.run(sc_manifest)
    assert "raw/articles.json" in sc_outs
    stats = sc_manifest["stages"]["scraper"]["metrics"]["per_source_stats"]
    assert stats["StaleFeed"]["newest_age_hours"] == 55.0
    assert stats["HealthyFeed"]["newest_age_hours"] == 0.4
    rows12 = compute_source_health(sc_dir, manifest=sc_manifest, config_dir=cfg_dir)
    by12 = {r["source"]: r for r in rows12}
    assert by12["StaleFeed"]["status"] == "STALE"
    assert by12["HealthyFeed"]["status"] == "OK"


def test_publish_to_latest_mirrors_and_replaces_stale_files(tmp_path: Path) -> None:
    from src.utils.daily_ops import load_outputs_config

    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    mirror_dir = tmp_path / "OneDrive" / "NepalNewsBot" / "latest"
    pipe_text = (cfg_dir / "pipeline.yaml").read_text(encoding="utf-8").replace(
        "C:/Users/Chungba Sherpa/OneDrive/NepalNewsBot/latest",
        mirror_dir.as_posix(),
    )
    (cfg_dir / "pipeline.yaml").write_text(pipe_text, encoding="utf-8")

    out_cfg = load_outputs_config(cfg_dir)
    assert out_cfg["mirror_dir"] == mirror_dir.as_posix()

    rdir = tmp_path / "runs" / "2026-11-08"
    (rdir / "render").mkdir(parents=True, exist_ok=True)
    (rdir / "qa" / "storyboard_frames").mkdir(parents=True, exist_ok=True)
    (rdir / "render" / "final.mp4").write_bytes(b"MP4_V1")
    (rdir / "render" / "description.md").write_text("# Desc\n", encoding="utf-8")
    (rdir / "qa" / "qa_report.json").write_text('{"passed": true, "checks": [], "warnings": []}', encoding="utf-8")
    (rdir / "qa" / "contact_sheet.jpg").write_bytes(b"CS")
    (rdir / "qa" / "storyboard.html").write_text("<html>sb</html>", encoding="utf-8")
    (rdir / "qa" / "storyboard_frames" / "frame_0000.jpg").write_bytes(b"F0")
    (rdir / "manifest.json").write_text(
        json.dumps({"run_id": "2026-11-08", "date": "2026-11-08", "run_kind": "test", "status": "done", "stages": {}}),
        encoding="utf-8",
    )

    # Pre-seed stale files in mirror_dir (do not exist yet; create now)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    (mirror_dir / "final.mp4").write_bytes(b"OLD")
    (mirror_dir / "storyboard_frames").mkdir(parents=True, exist_ok=True)
    (mirror_dir / "storyboard_frames" / "stale.jpg").write_bytes(b"STALE")

    latest_dir = tmp_path / "data" / "latest"
    res = publish_to_latest(rdir, latest_dir=latest_dir, config_dir=cfg_dir)
    assert res["mirror_dir"] == str(mirror_dir)
    for item in ("final.mp4", "description.md", "qa_report.json", "storyboard.html", "contact_sheet.jpg", "STATUS.md", "README.txt"):
        assert (mirror_dir / item).exists()
    assert (mirror_dir / "final.mp4").read_bytes() == b"MP4_V1"
    assert (mirror_dir / "storyboard_frames" / "frame_0000.jpg").read_bytes() == b"F0"
    assert not (mirror_dir / "storyboard_frames" / "stale.jpg").exists()


def test_prepare_media_and_apply_picks_pre_render_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify two-step workflow pauses after media candidates for picking BEFORE render/publish, then applies picks cleanly."""
    cfg_dir = tmp_path / "config"
    _copy_configs(cfg_dir)
    pipe_text = (cfg_dir / "pipeline.yaml").read_text(encoding="utf-8").replace(
        "C:/Users/Chungba Sherpa/OneDrive/NepalNewsBot/latest",
        (tmp_path / "mirror_latest").as_posix(),
    )
    (cfg_dir / "pipeline.yaml").write_text(pipe_text, encoding="utf-8")

    runs_dir = tmp_path / "runs"
    latest_dir = tmp_path / "latest"
    orch = Orchestrator(run_date="2026-11-12", config_dir=cfg_dir, runs_dir=runs_dir)
    manifest = orch.initialize_run()
    manifest["stages"]["scraper"]["status"] = "done"
    manifest["stages"]["editor"]["status"] = "done"
    manifest["run_kind"] = "production"
    (orch.run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _create_approved_script(orch.run_dir, run_kind="production")

    class FakeVoice:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, m):
            (self.run_dir / "audio").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "audio" / "narration.wav").write_bytes(b"RIFF")
            return ["audio/narration.wav"]

        def validate(self, outs):
            return True, None

    class FakeTimeline:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, m):
            (self.run_dir / "timeline").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "timeline" / "timeline.json").write_text('{"duration_sec": 7.0, "scenes": []}', encoding="utf-8")
            (self.run_dir / "timeline" / "captions.json").write_text("[]", encoding="utf-8")
            return ["timeline/timeline.json", "timeline/captions.json"]

        def validate(self, outs):
            return True, None

    applied_calls: list[Path] = []

    class FakeMedia:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run_candidates(self, candidates_k: int = 6):
            (self.run_dir / "media" / "review").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "media" / "shot_plan.json").write_text("{}", encoding="utf-8")
            (self.run_dir / "media" / "review" / "wanted_clips.md").write_text("# Wanted\n", encoding="utf-8")
            cands = {
                "date": "2026-11-12",
                "min_relevance": 0.5,
                "shots": [
                    {"shot_id": "scene_01_00", "auto_pick_type": "clip", "best_score": 0.8},
                ],
            }
            (self.run_dir / "media" / "review" / "candidates.json").write_text(json.dumps(cands), encoding="utf-8")
            (self.run_dir / "media" / "review" / "picker.html").write_text("<html>picker</html>", encoding="utf-8")
            return cands

        def apply_picks(self, picks_file):
            applied_calls.append(Path(picks_file))
            (self.run_dir / "media").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "media" / "media_manifest.json").write_text('{"shots": []}', encoding="utf-8")
            (self.run_dir / "media" / "credits.md").write_text("# Credits\n", encoding="utf-8")
            (self.run_dir / "media" / "picks_log.json").write_text(
                json.dumps({"total_shots": 1, "applied_picks": ["scene_01_00"], "cards_created": [], "failed_downloads": [], "action_counts": {"pick": 1}}),
                encoding="utf-8",
            )
            return ["media/media_manifest.json", "media/credits.md"]

        def validate(self, outs):
            return True, None

    class FakeMotion:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, m):
            (self.run_dir / "motion").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "motion" / "render_plan.json").write_text('{"scenes": [], "overlays": []}', encoding="utf-8")
            return ["motion/render_plan.json"]

        def validate(self, outs):
            return True, None

    class FakeRender:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, m):
            (self.run_dir / "render").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "render" / "final.mp4").write_bytes(b"PICKED_MP4")
            (self.run_dir / "render" / "description.md").write_text("# Desc\n", encoding="utf-8")
            return ["render/final.mp4", "render/description.md"]

        def validate(self, outs):
            return True, None

    class FakeQA:
        def __init__(self, run_dir: Path, config=None, **kwargs):
            self.run_dir = Path(run_dir)

        def run(self, m):
            (self.run_dir / "qa" / "storyboard_frames").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "qa" / "qa_report.json").write_text('{"passed": true, "checks": [], "warnings": []}', encoding="utf-8")
            (self.run_dir / "qa" / "contact_sheet.jpg").write_bytes(b"CS")
            (self.run_dir / "qa" / "storyboard.html").write_text("<html>sb</html>", encoding="utf-8")
            return ["qa/qa_report.json", "qa/contact_sheet.jpg", "qa/storyboard.html"]

        def validate(self, outs):
            return True, None

    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "voice", FakeVoice)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "timeline", FakeTimeline)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "media", FakeMedia)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "motion", FakeMotion)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "render", FakeRender)
    monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "qa", FakeQA)

    prep_res = orch.run_prepare_media(candidates_k=6)
    assert prep_res["ok"] is True
    assert prep_res["media_status"] == "awaiting_approval"
    assert prep_res["status"] == "paused"
    assert prep_res["auto_clips_count"] == 1
    assert (orch.run_dir / "media" / "review" / "picker.html").exists()
    # Crucially, render/final.mp4 has NOT been created yet!
    assert not (orch.run_dir / "render" / "final.mp4").exists()

    # Idempotent re-call without force returns already_prepared=True
    prep_again = orch.run_prepare_media(candidates_k=6)
    assert prep_again["ok"] is True
    assert prep_again["already_prepared"] is True

    picks_file = orch.run_dir / "media" / "review" / "picks_ui.json"
    picks_file.write_text(json.dumps({"date": "2026-11-12", "picks": {"scene_01_00": {"action": "pick", "candidate_id": "x"}}}), encoding="utf-8")

    fin_res = orch.run_apply_picks_and_finish(picks_file, latest_dir=latest_dir)
    assert fin_res["ok"] is True
    assert fin_res["status"] == "done"
    assert len(applied_calls) == 1
    assert (orch.run_dir / "render" / "final.mp4").read_bytes() == b"PICKED_MP4"
    assert (latest_dir / "final.mp4").read_bytes() == b"PICKED_MP4"

    final_manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["stages"]["media"]["status"] == "done"
    assert final_manifest["stages"]["media"]["metrics"]["picked_by_user"] is True


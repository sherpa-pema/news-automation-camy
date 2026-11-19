"""Orchestrator implementation for Nepal News Bot."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable
import yaml

from src.agents.base import BaseAgent
from src.agents.scraper import ScraperAgent
from src.agents.editor import EditorAgent
from src.agents.voice import VoiceAgent
from src.agents.media import MediaAgent
from src.agents.timeline import TimelineAgent
from src.agents.motion import MotionAgent
from src.agents.render import RenderAgent
from src.agents.qa import QAAgent
from src.utils.daily_ops import (
    get_kathmandu_today,
    get_next_action_text,
    publish_to_latest,
    verify_editor_gate,
    write_status_md,
)
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.logging import setup_logger
from src.utils.manifest import (
    PAUSE_STATES,
    compute_manifest_status,
    create_initial_manifest,
    load_manifest,
    reconcile_manifest_status,
    run_process_lock,
    save_manifest,
    update_stage_status,
)
from src.utils.office_events import emit_office_event


STAGE_HANDOFFS: dict[str, tuple[str, str, str]] = {
    "scraper": ("editor", "raw/articles.json", "articles.json"),
    "editor": ("voice", "script/script.json", "script.json"),
    "voice": ("timeline", "audio/narration.wav", "narration.wav"),
    "timeline": ("media", "timeline.json", "timeline.json"),
    "media": ("motion", "media/media_manifest.json", "media_manifest.json"),
    "motion": ("render", "motion/render_plan.json", "render_plan.json"),
    "render": ("qa", "render/final.mp4", "final.mp4"),
}


class PipelinePausedException(Exception):
    """Raised when the pipeline pauses for human or agent intervention."""

    def __init__(self, status: str, next_action: str = "") -> None:
        self.status = status
        self.next_action = next_action
        super().__init__(f"Pipeline paused in stage state '{status}'. {next_action}")


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "scraper": ScraperAgent,
    "editor": EditorAgent,
    "voice": VoiceAgent,
    "media": MediaAgent,
    "timeline": TimelineAgent,
    "motion": MotionAgent,
    "render": RenderAgent,
    "qa": QAAgent,
}


class Orchestrator:
    """Coordinates specialist agents, tracks manifest state, and enforces contracts."""

    def __init__(
        self,
        run_date: str | None = None,
        config_dir: str | Path = "config",
        runs_dir: str | Path = "data/runs",
        allow_fixture: bool = False,
        backfill: bool = False,
        media_auto: bool = False,
        run_kind: str | None = None,
        orientation: str | None = None,
        allow_robots_fallback: bool = False,
        force_source: str | None = None,
        categories: list[str] | str | None = None,
        automated: bool = False,
    ) -> None:
        self.run_date = run_date or get_kathmandu_today()
        self.config_dir = Path(config_dir)
        self.runs_dir = Path(runs_dir)
        self.run_dir = self.runs_dir / self.run_date
        self.manifest_path = self.run_dir / "manifest.json"
        self.allow_fixture = allow_fixture
        self.backfill = backfill
        self.media_auto = media_auto
        self.run_kind = run_kind
        self.orientation = orientation
        self.allow_robots_fallback = allow_robots_fallback
        self.force_source = force_source
        self.categories = categories
        self.automated = automated
        self._manifest_lock = threading.Lock()

        # Setup logger
        self.logger = setup_logger("orchestrator", self.run_dir / "run.log")

        # Load configs
        self.pipeline_config = self._load_yaml(self.config_dir / "pipeline.yaml")
        self.sources_config = self._load_yaml(self.config_dir / "sources.yaml")

        if self.manifest_path.exists():
            try:
                m = load_manifest(self.manifest_path)
                if not self.orientation and m.get("orientation"):
                    self.orientation = m.get("orientation")
            except Exception:
                pass

        if self.orientation:
            for stg in self.pipeline_config.get("stages", []):
                if stg.get("id") == "media":
                    stg["orientation"] = self.orientation
                    stg["resolution"] = [1080, 1920] if self.orientation == "vertical" else [1920, 1080]

    def _save_manifest_locked(self, manifest: dict[str, Any], active_process: bool = True) -> None:
        """Saves manifest and status.md under the thread lock."""
        with self._manifest_lock:
            save_manifest(manifest, self.manifest_path)
            write_status_md(self.run_dir, manifest, active_process=active_process)

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _default_run_kind(self) -> str:
        rk = self.pipeline_config.get("execution", {}).get("default_run_kind")
        return str(rk) if rk in ("test", "production") else "test"

    def initialize_run(self) -> dict[str, Any]:
        """Creates the run directory and initial manifest if not already existing."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "raw").mkdir(exist_ok=True)
        (self.run_dir / "audio").mkdir(exist_ok=True)
        (self.run_dir / "audio" / "scenes").mkdir(exist_ok=True)
        (self.run_dir / "media").mkdir(exist_ok=True)

        if self.manifest_path.exists():
            self.logger.info(f"Resuming existing run from {self.manifest_path}")
            manifest = load_manifest(self.manifest_path)
            changed = False
            if self.run_kind and not manifest.get("run_kind"):
                manifest["run_kind"] = self.run_kind
                changed = True
            if self.orientation and manifest.get("orientation") != self.orientation:
                manifest["orientation"] = self.orientation
                manifest["resolution"] = [1080, 1920] if self.orientation == "vertical" else [1920, 1080]
                changed = True
            if changed:
                save_manifest(manifest, self.manifest_path)
        else:
            self.logger.info(f"Initializing new run at {self.run_dir}")
            manifest = create_initial_manifest(self.run_date, self.run_date, self.pipeline_config)
            manifest["run_kind"] = self.run_kind or self._default_run_kind()
            if self.orientation:
                manifest["orientation"] = self.orientation
                manifest["resolution"] = [1080, 1920] if self.orientation == "vertical" else [1920, 1080]
            save_manifest(manifest, self.manifest_path)

        return manifest

    def reset_stage(self, stage_id: str, reason: str = "") -> bool:
        """Resets a stage back to pending, resets retries, and logs to manual_edits."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for run '{self.run_date}' at {self.manifest_path}")

        manifest = load_manifest(self.manifest_path)
        if manifest.get("status") == "abandoned":
            self.logger.error(f"Run '{self.run_date}' is abandoned. No stage may be reset.")
            raise RuntimeError(f"Run '{self.run_date}' is abandoned. No stage may be reset.")

        if stage_id not in manifest.get("stages", {}):
            raise ValueError(f"Unknown stage: {stage_id}")

        stage_meta = manifest["stages"][stage_id]
        old_status = stage_meta.get("status", "unknown")
        old_retry = stage_meta.get("retry_count", 0)

        stage_meta["status"] = "pending"
        stage_meta["retry_count"] = 0
        stage_meta["error_message"] = None
        if manifest.get("status") in ("failed", "interrupted"):
            manifest["status"] = "pending"

        entry = {
            "file": str(self.manifest_path.as_posix()),
            "what": f"Reset stages.{stage_id} status from '{old_status}' to 'pending', retry_count from {old_retry} to 0, error_message to null",
            "why": reason or f"Reset stage '{stage_id}' via --reset-stage CLI",
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        manifest.setdefault("manual_edits", []).append(entry)

        save_manifest(manifest, self.manifest_path)
        write_status_md(self.run_dir, manifest)
        self.logger.info(f"Stage '{stage_id}' successfully reset to pending. Logged to manual_edits.")
        return True

    def execute_stage(self, stage_id: str, manifest: dict[str, Any], force: bool = False) -> bool:
        """Executes a single agent stage with contract validation and retry logic."""
        if manifest.get("status") == "abandoned":
            self.logger.error(f"Run '{self.run_date}' is abandoned. No stage may run on it.")
            raise RuntimeError(f"Run '{self.run_date}' is abandoned. No stage may run on it.")

        stage_meta = manifest["stages"].get(stage_id)
        if not stage_meta:
            raise ValueError(f"Unknown stage: {stage_id}")

        if not force and stage_meta["status"] == "done":
            self.logger.info(f"Stage '{stage_id}' already completed. Skipping.")
            return True

        if not force and stage_meta["status"] in PAUSE_STATES and not (stage_id == "media" and self.media_auto):
            current_stage_status = stage_meta["status"]
            manifest["status"] = "paused"
            save_manifest(manifest, self.manifest_path)
            write_status_md(self.run_dir, manifest, active_process=False)
            next_act = ""
            if stage_id == "media":
                next_act = (
                    f"Next action: Open data/runs/{self.run_date}/media/review/picker.html to select visuals, "
                    f"apply picks with:\n"
                    f"  python run_daily.py --apply-picks data/runs/{self.run_date}/media/picks.json --date {self.run_date}"
                )
            elif current_stage_status == "prepared":
                next_act = (
                    f"Next action: Read data/runs/{self.run_date}/script/editor_brief.md, "
                    f"write draft_script.json and selection.json, then run:\n"
                    f"  python -m src.agents.editor validate --date {self.run_date}"
                )
            elif current_stage_status == "needs_revision":
                next_act = (
                    f"Next action: Review errors in data/runs/{self.run_date}/script/validation_report.json, "
                    f"fix draft_script.json, then run:\n"
                    f"  python -m src.agents.editor validate --date {self.run_date}"
                )
            elif current_stage_status == "awaiting_approval":
                next_act = (
                    f"Next action: Review data/runs/{self.run_date}/script/script_preview.md and run:\n"
                    f"  python -m src.agents.editor approve --date {self.run_date}"
                )
            raise PipelinePausedException(current_stage_status, next_act)

        if force:
            stage_meta["retry_count"] = 0
            stage_meta["error_message"] = None

        agent_cls = AGENT_REGISTRY.get(stage_id)
        if not agent_cls:
            raise ValueError(f"No agent registered for stage '{stage_id}'")

        # Provenance and editor gate check for downstream stages
        if stage_id in ("voice", "timeline", "media", "motion", "render", "qa"):
            script_path = self.run_dir / "script" / "script.json"

            origin = "fixture"
            if script_path.exists():
                try:
                    with open(script_path, "r", encoding="utf-8") as f:
                        s_data = json.load(f)
                    origin = s_data.get("origin", "fixture")
                except json.JSONDecodeError:
                    pass

            if origin == "fixture":
                if not self.allow_fixture:
                    raise ValueError(
                        f"Refusing to execute stage '{stage_id}' on fixture script without --allow-fixture flag."
                    )
                self.logger.warning(
                    f"[WARNING] FIXTURE DATA: Running stage '{stage_id}' on fixture script data."
                )
            else:
                editor_status = manifest.get("stages", {}).get("editor", {}).get("status", "pending")
                if editor_status != "done":
                    raise ValueError(
                        f"Refusing to execute downstream stage '{stage_id}': Editor stage status is '{editor_status}', but must be 'done'."
                    )
                approval_file = self.run_dir / "script" / "approval.json"
                if not approval_file.exists():
                    raise ValueError("script changed after approval, re-run validate and approve")
                with open(approval_file, "r", encoding="utf-8") as f:
                    app_data = json.load(f)
                curr_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
                if app_data.get("script_sha256") != curr_hash:
                    raise ValueError("script changed after approval, re-run validate and approve")

                # Verify validator fingerprint hasn't changed
                curr_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
                app_fp = extract_script_validation_fingerprint(app_data.get("validator_fingerprint"))
                if app_fp and app_fp != curr_fp:
                    raise ValueError("validator or editor config changed after approval, re-run validate and approve")

                rep_file = self.run_dir / "script" / "validation_report.json"
                if rep_file.exists():
                    with open(rep_file, "r", encoding="utf-8") as f:
                        rep_data = json.load(f)
                    rep_fp = extract_script_validation_fingerprint(rep_data.get("validator_fingerprint"))
                    if rep_fp and rep_fp != curr_fp:
                        raise ValueError("validator or editor config changed since validation, re-run validate and approve")

        if stage_id == "scraper":
            scraper_kwargs: dict[str, Any] = {
                "config": self.pipeline_config,
                "backfill": self.backfill,
            }
            try:
                import inspect
                sig = inspect.signature(agent_cls.__init__)
                params = sig.parameters
                has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                if has_var_kw or "allow_robots_fallback" in params:
                    scraper_kwargs["allow_robots_fallback"] = self.allow_robots_fallback
                if has_var_kw or "force_source" in params:
                    scraper_kwargs["force_source"] = self.force_source
                if has_var_kw or "categories" in params:
                    scraper_kwargs["categories"] = self.categories
                if has_var_kw or "interactive" in params:
                    scraper_kwargs["interactive"] = False if self.automated else None
            except Exception:
                pass
            agent = agent_cls(self.run_dir, **scraper_kwargs)
        elif stage_id == "media":
            agent = agent_cls(
                self.run_dir,
                config=self.pipeline_config,
                allow_fixture=self.allow_fixture,
                auto=self.media_auto,
                orientation=self.orientation,
            )
        else:
            agent = agent_cls(self.run_dir, config=self.pipeline_config)
        max_retries = stage_meta.get("max_retries", 2)

        while stage_meta["retry_count"] <= max_retries:
            attempt = stage_meta["retry_count"] + 1
            self.logger.info(f"Starting stage '{stage_id}' (attempt {attempt}/{max_retries + 1})...")

            with self._manifest_lock:
                update_stage_status(manifest, stage_id, "running")
                save_manifest(manifest, self.manifest_path)
            emit_office_event(
                self.run_dir,
                "stage_status",
                stage=stage_id,
                status="running",
                message=f"Orchestrator dispatched {stage_id} (attempt {attempt})",
            )

            try:
                # Execute agent
                try:
                    output_files = agent.run(manifest, save_manifest_to_disk=False)
                except TypeError:
                    output_files = agent.run(manifest)

                # Check if agent entered a pause state (prepared, needs_revision, awaiting_approval)
                current_stage_status = manifest.get("stages", {}).get(stage_id, {}).get("status")
                if current_stage_status in PAUSE_STATES:
                    with self._manifest_lock:
                        manifest["status"] = "paused"
                        save_manifest(manifest, self.manifest_path)
                        write_status_md(self.run_dir, manifest, active_process=False)
                    emit_office_event(
                        self.run_dir,
                        "stage_status",
                        stage=stage_id,
                        status=current_stage_status,
                        message=f"Paused at {current_stage_status}",
                    )
                    next_act = ""
                    if current_stage_status == "prepared":
                        next_act = (
                            f"Next action: Read data/runs/{self.run_date}/script/editor_brief.md, "
                            f"write draft_script.json and selection.json, then run:\n"
                            f"  python -m src.agents.editor validate --date {self.run_date}"
                        )
                    elif current_stage_status == "needs_revision":
                        next_act = (
                            f"Next action: Review errors in data/runs/{self.run_date}/script/validation_report.json, "
                            f"fix draft_script.json, then run:\n"
                            f"  python -m src.agents.editor validate --date {self.run_date}"
                        )
                    elif current_stage_status == "awaiting_approval":
                        next_act = (
                            f"Next action: Review data/runs/{self.run_date}/script/script_preview.md and run:\n"
                            f"  python -m src.agents.editor approve --date {self.run_date}"
                        )
                    raise PipelinePausedException(current_stage_status, next_act)

                # Validate outputs against contract schema
                is_valid, validation_err = agent.validate(output_files)
                if not is_valid:
                    raise ValueError(f"Contract validation failed: {validation_err}")

                # Mark stage done
                agent_metrics = getattr(agent, "last_metrics", None)
                with self._manifest_lock:
                    update_stage_status(
                        manifest,
                        stage_id,
                        "done",
                        output_files=output_files,
                        metrics=agent_metrics,
                    )
                    save_manifest(manifest, self.manifest_path)
                    write_status_md(self.run_dir, manifest)
                emit_office_event(
                    self.run_dir,
                    "stage_status",
                    stage=stage_id,
                    status="done",
                    message=f"Completed {stage_id}",
                )
                if stage_id == "editor":
                    # Twin courier handoffs to Sound Booth and Archive Studio
                    emit_office_event(
                        self.run_dir,
                        "packet_handoff",
                        stage="editor",
                        from_station="editor",
                        to_station="voice",
                        artifact="script/script.json",
                        label="script.json -> Sound Booth",
                    )
                    emit_office_event(
                        self.run_dir,
                        "packet_handoff",
                        stage="editor",
                        from_station="editor",
                        to_station="media",
                        artifact="script/script.json",
                        label="script.json -> Archive Studio",
                    )
                elif stage_id in STAGE_HANDOFFS:
                    to_st, art_path, lbl = STAGE_HANDOFFS[stage_id]
                    emit_office_event(
                        self.run_dir,
                        "packet_handoff",
                        stage=stage_id,
                        from_station=stage_id,
                        to_station=to_st,
                        artifact=art_path,
                        label=lbl,
                    )
                self.logger.info(f"Stage '{stage_id}' finished successfully.")
                return True

            except PipelinePausedException:
                # Pause state: re-raise immediately without retrying or treating as error
                raise

            except NotImplementedError as e:
                # Expected in Step 1 stubs
                self.logger.warning(f"Stage '{stage_id}' stub invoked: {e}")
                update_stage_status(manifest, stage_id, "pending", error_message=str(e))
                save_manifest(manifest, self.manifest_path)
                write_status_md(self.run_dir, manifest)
                return False

            except Exception as e:
                self.logger.error(f"Error in stage '{stage_id}' attempt {attempt}: {e}")
                stage_meta["retry_count"] += 1
                if stage_meta["retry_count"] > max_retries:
                    update_stage_status(
                        manifest,
                        stage_id,
                        "failed",
                        error_message=f"Exhausted {max_retries} retries: {str(e)}",
                    )
                    save_manifest(manifest, self.manifest_path)
                    write_status_md(self.run_dir, manifest)
                    emit_office_event(
                        self.run_dir,
                        "stage_status",
                        stage=stage_id,
                        status="failed",
                        message=str(e),
                    )
                    if stage_id == "qa":
                        emit_office_event(
                            self.run_dir,
                            "packet_handoff",
                            stage="qa",
                            from_station="qa",
                            to_station="orchestrator",
                            artifact="qa/qa_report.json",
                            label="redo_stage",
                            status="alert",
                        )
                    return False
                else:
                    self.logger.info(f"Retrying stage '{stage_id}'...")

        return False

    def run_pipeline(self, only_stage: str | None = None) -> bool:
        """Executes the pipeline (full DAG or a single isolated stage if only_stage is specified)."""
        manifest = self.initialize_run()
        if manifest.get("status") == "abandoned":
            self.logger.error(f"Run '{self.run_date}' is abandoned. No stage may run on it.")
            raise RuntimeError(f"Run '{self.run_date}' is abandoned. No stage may run on it.")

        with run_process_lock(self.run_dir):
            manifest["status"] = "running"
            save_manifest(manifest, self.manifest_path)

            try:
                if only_stage:
                    self.logger.info(f"Executing ONLY stage '{only_stage}' for {self.run_date}")
                    if only_stage not in manifest["stages"]:
                        self.logger.error(f"Unknown stage '{only_stage}' specified in --only.")
                        reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                        save_manifest(manifest, self.manifest_path)
                        write_status_md(self.run_dir, manifest, active_process=False)
                        return False

                    success = self.execute_stage(only_stage, manifest, force=True)
                    if not success:
                        self._mark_run_failed(manifest, f"Stage '{only_stage}' failed")
                        return False

                    reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                    save_manifest(manifest, self.manifest_path)
                    write_status_md(self.run_dir, manifest, active_process=False)
                    self.logger.info(f"Target stage '{only_stage}' completed successfully (status={manifest['status']}).")
                    return True

                self.logger.info(f"Starting full pipeline execution for {self.run_date}")

                for st_id in ("scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa"):
                    if not self.execute_stage(st_id, manifest):
                        self._mark_run_failed(manifest, f"{st_id.capitalize()} stage failed")
                        return False

                reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                save_manifest(manifest, self.manifest_path)
                write_status_md(self.run_dir, manifest, active_process=False)
                self.logger.info("Pipeline execution completed successfully!")
                return True

            except PipelinePausedException:
                reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                save_manifest(manifest, self.manifest_path)
                write_status_md(self.run_dir, manifest, active_process=False)
                raise

            except Exception:
                reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                save_manifest(manifest, self.manifest_path)
                write_status_md(self.run_dir, manifest, active_process=False)
                raise

    def _collect_morning_metrics(
        self, manifest: dict[str, Any]
    ) -> tuple[int, dict[str, Any], int, Path]:
        shortlist_path = self.run_dir / "script" / "shortlist.json"
        articles_path = self.run_dir / "raw" / "articles.json"
        brief_path = self.run_dir / "script" / "editor_brief.md"
        shortlist_count = 0
        if shortlist_path.exists():
            try:
                shortlist_count = len(json.loads(shortlist_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        scraper_metrics = manifest.get("stages", {}).get("scraper", {}).get("metrics") or {}
        per_source_stats = scraper_metrics.get("per_source_stats", {})
        total_articles = scraper_metrics.get("total_articles", 0)
        if not per_source_stats and articles_path.exists():
            try:
                arts = json.loads(articles_path.read_text(encoding="utf-8"))
                total_articles = len(arts)
                counts: dict[str, dict[str, int]] = {}
                for a in arts:
                    src = a.get("source", "Unknown")
                    counts.setdefault(src, {"extracted_ok": 0})
                    counts[src]["extracted_ok"] += 1
                per_source_stats = counts
            except Exception:
                pass
        return total_articles, per_source_stats, shortlist_count, brief_path

    def run_morning(
        self,
        run_kind: str | None = None,
        retry_delay_sec: float = 300.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        force: bool = False,
    ) -> dict[str, Any]:
        """Morning daily command: runs scraper (retrying once after 5 min on failure),
        runs editor prepare, writes STATUS.md, and stops at 'needs writer'.

        Idempotent: if scraper is already 'done' and editor is already 'prepared' or later,
        does not re-scrape or reset anything unless force=True.
        """
        if self.manifest_path.exists():
            existing = load_manifest(self.manifest_path)
            if existing.get("status") == "abandoned":
                raise RuntimeError(f"Run '{self.run_date}' is abandoned. No stage may run on it.")
            sc_status = existing.get("stages", {}).get("scraper", {}).get("status", "pending")
            ed_status = existing.get("stages", {}).get("editor", {}).get("status", "pending")
            if not force and sc_status == "done" and ed_status in (
                "prepared",
                "needs_revision",
                "awaiting_approval",
                "done",
            ):
                status_val = compute_manifest_status(existing, run_dir=self.run_dir, active_process=False)
                next_action = get_next_action_text(self.run_dir, existing, active_process=False)
                status_path = write_status_md(self.run_dir, existing, active_process=False)
                total_articles, per_source_stats, shortlist_count, brief_path = self._collect_morning_metrics(
                    existing
                )
                return {
                    "ok": True,
                    "already_done": True,
                    "skipped": True,
                    "status": status_val,
                    "editor_status": ed_status,
                    "stopped_at": "needs writer" if ed_status in ("prepared", "needs_revision") else ed_status,
                    "next_action": next_action,
                    "run_kind": existing.get("run_kind") or self._default_run_kind(),
                    "total_articles": total_articles,
                    "per_source_stats": per_source_stats,
                    "shortlist_count": shortlist_count,
                    "brief_path": str(brief_path),
                    "status_path": str(status_path),
                }

        manifest = self.initialize_run()
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_date}' is abandoned. No stage may run on it.")

        effective_run_kind = run_kind or self.run_kind or manifest.get("run_kind") or self._default_run_kind()
        if force and (run_kind or self.run_kind):
            effective_run_kind = run_kind or self.run_kind or effective_run_kind
        manifest["run_kind"] = effective_run_kind
        if force:
            for st_reset in ("scraper", "editor"):
                if st_reset in manifest.get("stages", {}):
                    manifest["stages"][st_reset]["status"] = "pending"
                    manifest["stages"][st_reset]["retry_count"] = 0
                    manifest["stages"][st_reset]["error_message"] = None
        save_manifest(manifest, self.manifest_path)

        with run_process_lock(self.run_dir):
            manifest["status"] = "running"
            save_manifest(manifest, self.manifest_path)

            scraper_meta = manifest["stages"]["scraper"]
            if force or scraper_meta.get("status") != "done":
                orig_max_retries = scraper_meta.get("max_retries", 2)
                # Attempt 1 (single pass so failure immediately triggers 5-min wait + single retry)
                scraper_meta["retry_count"] = 0
                scraper_meta["max_retries"] = 0
                scraper_meta["error_message"] = None
                ok_scrape = False
                try:
                    ok_scrape = self.execute_stage("scraper", manifest, force=force)
                except Exception as exc:
                    update_stage_status(manifest, "scraper", "failed", error_message=str(exc))
                    ok_scrape = False

                if not ok_scrape:
                    self.logger.warning(
                        f"Scraper attempt 1 failed ({scraper_meta.get('error_message')}). "
                        f"Waiting {retry_delay_sec:.0f}s before retrying once..."
                    )
                    print(
                        f"[WARN] Scraper failed; waiting {int(retry_delay_sec)}s before retrying once..."
                    )
                    sleep_fn(retry_delay_sec)
                    scraper_meta["status"] = "pending"
                    scraper_meta["retry_count"] = 0
                    scraper_meta["max_retries"] = 0
                    scraper_meta["error_message"] = None
                    try:
                        ok_scrape = self.execute_stage("scraper", manifest, force=False)
                    except Exception as exc:
                        update_stage_status(manifest, "scraper", "failed", error_message=str(exc))
                        ok_scrape = False

                scraper_meta["max_retries"] = orig_max_retries
                if not ok_scrape:
                    err_detail = scraper_meta.get("error_message") or "Scraper failed after retry."
                    update_stage_status(manifest, "scraper", "failed", error_message=err_detail)
                    self._mark_run_failed(manifest, f"Morning scraper failed: {err_detail}")
                    status_path = write_status_md(self.run_dir, manifest, active_process=False)
                    return {
                        "ok": False,
                        "already_done": False,
                        "skipped": False,
                        "status": "failed",
                        "run_kind": effective_run_kind,
                        "error": err_detail,
                        "status_path": str(status_path),
                    }

            # Editor prepare step -> pauses at 'prepared' ("needs writer")
            paused_status = manifest["stages"]["editor"].get("status", "pending")
            try:
                self.execute_stage("editor", manifest, force=force)
                paused_status = manifest["stages"]["editor"].get("status", "done")
            except PipelinePausedException as pause_exc:
                paused_status = pause_exc.status

            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, self.manifest_path)
            status_path = write_status_md(self.run_dir, manifest, active_process=False)

        next_action = get_next_action_text(self.run_dir, manifest, active_process=False)
        total_articles, per_source_stats, shortlist_count, brief_path = self._collect_morning_metrics(manifest)

        return {
            "ok": True,
            "already_done": False,
            "skipped": False,
            "status": manifest["status"],
            "editor_status": paused_status,
            "stopped_at": "needs writer" if paused_status in ("prepared", "needs_revision") else paused_status,
            "next_action": next_action,
            "run_kind": effective_run_kind,
            "total_articles": total_articles,
            "per_source_stats": per_source_stats,
            "shortlist_count": shortlist_count,
            "brief_path": str(brief_path),
            "status_path": str(status_path),
        }

    def run_finish(
        self,
        latest_dir: Path | str = "data/latest",
        force: bool = False,
    ) -> dict[str, Any]:
        """Finish daily command: verifies editor approval gate, then runs voice, timeline,
        media (AUTO mode), motion, render, qa, updates STATUS.md, and copies outputs to data/latest/.

        Idempotent: if all downstream stages (voice..qa) are already 'done', does not re-run
        or reset anything unless force=True.
        """
        if not self.manifest_path.exists():
            reason = f"Manifest not found at {self.manifest_path}. Run --morning first."
            return {"ok": False, "refused": True, "reason": reason}

        manifest = load_manifest(self.manifest_path)
        gate_ok, gate_reason = verify_editor_gate(self.run_dir, allow_fixture=self.allow_fixture)
        if not gate_ok:
            return {"ok": False, "refused": True, "reason": gate_reason}

        downstream_stages = ("voice", "timeline", "media", "motion", "render", "qa")
        if not force and all(
            manifest.get("stages", {}).get(s, {}).get("status") == "done"
            for s in downstream_stages
        ):
            status_val = compute_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            next_action = get_next_action_text(self.run_dir, manifest, active_process=False)
            status_path = write_status_md(self.run_dir, manifest, active_process=False)
            return {
                "ok": True,
                "refused": False,
                "already_finished": True,
                "skipped": True,
                "status": status_val,
                "run_kind": manifest.get("run_kind", "test"),
                "next_action": next_action,
                "status_path": str(status_path),
                "storyboard_path": str(self.run_dir / "qa" / "storyboard.html"),
                "final_mp4": str(self.run_dir / "render" / "final.mp4"),
                "description_md": str(self.run_dir / "render" / "description.md"),
            }

        prev_media_auto = self.media_auto
        self.media_auto = True
        try:
            with run_process_lock(self.run_dir):
                manifest["status"] = "running"
                self._save_manifest_locked(manifest)

                for st_id in downstream_stages:
                    if not self.execute_stage(st_id, manifest, force=force):
                        self._mark_run_failed(manifest, f"Finish stage '{st_id}' failed")
                        status_path = write_status_md(self.run_dir, manifest, active_process=False)
                        err_msg = manifest["stages"][st_id].get("error_message") or f"Stage '{st_id}' failed"
                        return {
                            "ok": False,
                            "refused": False,
                            "already_finished": False,
                            "skipped": False,
                            "failed_stage": st_id,
                            "reason": err_msg,
                            "status_path": str(status_path),
                        }

                reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                self._save_manifest_locked(manifest, active_process=False)
                status_path = write_status_md(self.run_dir, manifest, active_process=False)
        finally:
            self.media_auto = prev_media_auto

        latest_info = publish_to_latest(
            self.run_dir,
            latest_dir=latest_dir,
            manifest=manifest,
            config_dir=self.config_dir,
        )
        emit_office_event(
            self.run_dir,
            "packet_handoff",
            stage="qa",
            from_station="qa",
            to_station="publish",
            artifact="render/final.mp4",
            label="final.mp4 → Latest & OneDrive",
        )
        emit_office_event(
            self.run_dir,
            "publish_complete",
            stage="qa",
            status="done",
            message="Published to data/latest & OneDrive mirror",
            extra={
                "latest_dir": str(latest_info.get("latest_dir", "")),
                "mirror_dir": str(latest_info.get("mirror_dir", "")),
            },
        )
        next_action = get_next_action_text(self.run_dir, manifest, active_process=False)
        return {
            "ok": True,
            "refused": False,
            "already_finished": False,
            "skipped": False,
            "status": manifest["status"],
            "run_kind": manifest.get("run_kind", "test"),
            "next_action": next_action,
            "status_path": str(status_path),
            "storyboard_path": str(self.run_dir / "qa" / "storyboard.html"),
            "final_mp4": str(self.run_dir / "render" / "final.mp4"),
            "description_md": str(self.run_dir / "render" / "description.md"),
            "latest": latest_info,
        }

    def run_prepare_media(
        self,
        candidates_k: int = 6,
        force: bool = False,
    ) -> dict[str, Any]:
        """Two-step workflow checkpoint: runs voice + timeline + media candidates discovery,
        pauses with stages.media.status = 'awaiting_approval' so the user can pick visuals
        in picker.html BEFORE motion, render, qa, or publish run.
        """
        if not self.manifest_path.exists():
            reason = f"Manifest not found at {self.manifest_path}. Run --morning first."
            return {"ok": False, "refused": True, "reason": reason}

        manifest = load_manifest(self.manifest_path)
        gate_ok, gate_reason = verify_editor_gate(self.run_dir, allow_fixture=self.allow_fixture)
        if not gate_ok:
            return {"ok": False, "refused": True, "reason": gate_reason}

        media_stage = manifest.get("stages", {}).get("media", {})
        picker_path = self.run_dir / "media" / "review" / "picker.html"
        candidates_path = self.run_dir / "media" / "review" / "candidates.json"

        if (
            not force
            and media_stage.get("status") == "awaiting_approval"
            and picker_path.exists()
            and candidates_path.exists()
        ):
            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, self.manifest_path)
            status_path = write_status_md(self.run_dir, manifest, active_process=False)
            c_data = json.loads(candidates_path.read_text(encoding="utf-8"))
            shots = c_data.get("shots", [])
            clips_cnt = sum(1 for sh in shots if sh.get("auto_pick_type") == "clip")
            cards_cnt = len(shots) - clips_cnt
            return {
                "ok": True,
                "refused": False,
                "already_prepared": True,
                "status": manifest["status"],
                "media_status": "awaiting_approval",
                "total_shots": len(shots),
                "auto_clips_count": clips_cnt,
                "auto_cards_count": cards_cnt,
                "picker_html": str(picker_path),
                "candidates_json": str(candidates_path),
                "status_path": str(status_path),
                "next_action": get_next_action_text(self.run_dir, manifest, active_process=False),
            }

        with run_process_lock(self.run_dir):
            manifest["status"] = "running"
            self._save_manifest_locked(manifest)

            # Emit twin packet handoffs from editor to voice and media
            emit_office_event(
                self.run_dir,
                "packet_handoff",
                stage="editor",
                from_station="editor",
                to_station="voice",
                artifact="script/script.json",
                label="script.json -> Sound Booth",
            )
            emit_office_event(
                self.run_dir,
                "packet_handoff",
                stage="editor",
                from_station="editor",
                to_station="media",
                artifact="script/script.json",
                label="script.json -> Archive Studio",
            )

            import concurrent.futures

            audio_err_info: dict[str, Any] = {"failed_stage": None, "reason": None}

            def _prep_audio() -> bool:
                for st_id in ("voice", "timeline"):
                    if not self.execute_stage(st_id, manifest, force=force):
                        err_msg = manifest["stages"][st_id].get("error_message") or f"Stage '{st_id}' failed"
                        audio_err_info["failed_stage"] = st_id
                        audio_err_info["reason"] = err_msg
                        return False
                return True

            visual_err_info: dict[str, Any] = {"reason": None}
            candidates_holder: dict[str, Any] = {}

            def _prep_visuals() -> bool:
                with self._manifest_lock:
                    update_stage_status(manifest, "media", "running")
                    save_manifest(manifest, self.manifest_path)
                emit_office_event(
                    self.run_dir,
                    "stage_status",
                    stage="media",
                    status="running",
                    message="Discovering visual candidates for picker...",
                )
                try:
                    media_cls = AGENT_REGISTRY.get("media", MediaAgent)
                    agent = media_cls(
                        self.run_dir,
                        config=self.pipeline_config,
                        allow_fixture=self.allow_fixture,
                        auto=False,
                        orientation=self.orientation,
                    )
                    c_data = agent.run_candidates(candidates_k=candidates_k)
                    candidates_holder["data"] = c_data
                    return True
                except Exception as exc:
                    visual_err_info["reason"] = str(exc)
                    with self._manifest_lock:
                        update_stage_status(manifest, "media", "failed", error_message=str(exc))
                        save_manifest(manifest, self.manifest_path)
                    emit_office_event(
                        self.run_dir,
                        "stage_status",
                        stage="media",
                        status="failed",
                        message=str(exc),
                    )
                    return False

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fut_visuals = executor.submit(_prep_visuals)
                fut_audio = executor.submit(_prep_audio)

                vis_ok = fut_visuals.result()
                audio_ok = fut_audio.result()

            if not audio_ok:
                st_id = audio_err_info["failed_stage"]
                self._mark_run_failed(manifest, f"Prepare-media stage '{st_id}' failed")
                status_path = write_status_md(self.run_dir, manifest, active_process=False)
                return {
                    "ok": False,
                    "refused": False,
                    "failed_stage": st_id,
                    "reason": audio_err_info["reason"],
                    "status_path": str(status_path),
                }

            if not vis_ok:
                self._mark_run_failed(manifest, f"Media candidates discovery failed: {visual_err_info['reason']}")
                status_path = write_status_md(self.run_dir, manifest, active_process=False)
                return {
                    "ok": False,
                    "refused": False,
                    "failed_stage": "media",
                    "reason": visual_err_info["reason"],
                    "status_path": str(status_path),
                }

            candidates_data = candidates_holder.get("data", {})

            shots = candidates_data.get("shots", [])
            clips_cnt = sum(1 for sh in shots if sh.get("auto_pick_type") == "clip")
            cards_cnt = len(shots) - clips_cnt

            update_stage_status(
                manifest,
                "media",
                "awaiting_approval",
                output_files=[
                    "media/shot_plan.json",
                    "media/review/candidates.json",
                    "media/review/picker.html",
                    "media/review/wanted_clips.md",
                ],
                error_message=None,
            )
            manifest["stages"]["media"]["metrics"] = {
                "auto_mode": False,
                "prepared_for_picking": True,
                "total_shots": len(shots),
                "auto_clips_count": clips_cnt,
                "auto_cards_count": cards_cnt,
                "min_relevance": candidates_data.get("min_relevance"),
            }
            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, self.manifest_path)
            status_path = write_status_md(self.run_dir, manifest, active_process=False)

            emit_office_event(
                self.run_dir,
                "stage_status",
                stage="media",
                status="awaiting_approval",
                message=f"Picker ready: {clips_cnt} clips, {cards_cnt} cards proposed",
            )
            emit_office_event(
                self.run_dir,
                "packet_handoff",
                stage="media",
                from_station="media",
                to_station="boss",
                artifact="media/review/picker.html",
                label="Visual Picker Ready",
            )
            emit_office_event(
                self.run_dir,
                "boss_alert",
                stage="media",
                message=f"Visual candidates ready ({clips_cnt} clips, {cards_cnt} cards)! Open Visual Picker before rendering.",
            )

        return {
            "ok": True,
            "refused": False,
            "already_prepared": False,
            "status": manifest["status"],
            "media_status": "awaiting_approval",
            "total_shots": len(shots),
            "auto_clips_count": clips_cnt,
            "auto_cards_count": cards_cnt,
            "picker_html": str(picker_path),
            "candidates_json": str(candidates_path),
            "status_path": str(status_path),
            "next_action": get_next_action_text(self.run_dir, manifest, active_process=False),
        }

    def run_apply_picks_and_finish(
        self,
        picks_file: Path | str,
        latest_dir: Path | str = "data/latest",
    ) -> dict[str, Any]:
        """Applies human visual picks from picks_file, marks media done, runs motion -> render -> qa,
        and publishes deliverables to data/latest and OneDrive mirror.
        """
        if not self.manifest_path.exists():
            reason = f"Manifest not found at {self.manifest_path}. Run --morning first."
            return {"ok": False, "refused": True, "reason": reason}

        manifest = load_manifest(self.manifest_path)
        gate_ok, gate_reason = verify_editor_gate(self.run_dir, allow_fixture=self.allow_fixture)
        if not gate_ok:
            return {"ok": False, "refused": True, "reason": gate_reason}

        with run_process_lock(self.run_dir):
            manifest["status"] = "running"
            save_manifest(manifest, self.manifest_path)

            for st_id in ("voice", "timeline"):
                if not self.execute_stage(st_id, manifest, force=False):
                    self._mark_run_failed(manifest, f"Prerequisite stage '{st_id}' failed")
                    status_path = write_status_md(self.run_dir, manifest, active_process=False)
                    err_msg = manifest["stages"][st_id].get("error_message") or f"Stage '{st_id}' failed"
                    return {
                        "ok": False,
                        "refused": False,
                        "failed_stage": st_id,
                        "reason": err_msg,
                        "status_path": str(status_path),
                    }

            update_stage_status(manifest, "media", "running", error_message=None)
            save_manifest(manifest, self.manifest_path)
            emit_office_event(
                self.run_dir,
                "stage_status",
                stage="media",
                status="running",
                message="Applying Boss Visual Picker selections...",
            )

            try:
                media_cls = AGENT_REGISTRY.get("media", MediaAgent)
                agent = media_cls(
                    self.run_dir,
                    config=self.pipeline_config,
                    allow_fixture=self.allow_fixture,
                    auto=False,
                    orientation=self.orientation,
                )
                applied_outputs = agent.apply_picks(picks_file)
                is_valid, val_err = agent.validate(applied_outputs)
                if not is_valid:
                    raise ValueError(f"Media contract validation failed after apply_picks: {val_err}")
            except Exception as exc:
                update_stage_status(manifest, "media", "failed", error_message=str(exc))
                self._mark_run_failed(manifest, f"Applying visual picks failed: {exc}")
                status_path = write_status_md(self.run_dir, manifest, active_process=False)
                emit_office_event(
                    self.run_dir,
                    "stage_status",
                    stage="media",
                    status="failed",
                    message=str(exc),
                )
                return {
                    "ok": False,
                    "refused": False,
                    "failed_stage": "media",
                    "reason": str(exc),
                    "status_path": str(status_path),
                }

            all_media_outputs = list(
                dict.fromkeys(
                    list(applied_outputs)
                    + (["media/shot_plan.json"] if (self.run_dir / "media" / "shot_plan.json").exists() else [])
                    + (["media/review/picker.html"] if (self.run_dir / "media" / "review" / "picker.html").exists() else [])
                )
            )
            media_metrics: dict[str, Any] = {"auto_mode": False, "picked_by_user": True}
            picks_log_path = self.run_dir / "media" / "picks_log.json"
            if picks_log_path.exists():
                try:
                    plog = json.loads(picks_log_path.read_text(encoding="utf-8"))
                    ac = plog.get("action_counts", {})
                    media_metrics.update({
                        "total_shots": plog.get("total_shots", 0),
                        "applied_clips_count": len(plog.get("applied_picks", [])),
                        "cards_created_count": len(plog.get("cards_created", [])),
                        "failed_downloads_count": len(plog.get("failed_downloads", [])),
                        "action_counts": ac,
                    })
                except Exception:
                    pass

            update_stage_status(manifest, "media", "done", output_files=all_media_outputs)
            manifest["stages"]["media"]["metrics"] = media_metrics
            save_manifest(manifest, self.manifest_path)
            emit_office_event(
                self.run_dir,
                "stage_status",
                stage="media",
                status="done",
                message="Applied visual selections",
            )
            emit_office_event(
                self.run_dir,
                "packet_handoff",
                stage="media",
                from_station="media",
                to_station="motion",
                artifact="media/media_manifest.json",
                label="media_manifest.json",
            )

            for st_id in ("motion", "render", "qa"):
                if not self.execute_stage(st_id, manifest, force=True):
                    self._mark_run_failed(manifest, f"Post-picks stage '{st_id}' failed")
                    status_path = write_status_md(self.run_dir, manifest, active_process=False)
                    err_msg = manifest["stages"][st_id].get("error_message") or f"Stage '{st_id}' failed"
                    return {
                        "ok": False,
                        "refused": False,
                        "failed_stage": st_id,
                        "reason": err_msg,
                        "status_path": str(status_path),
                    }

            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, self.manifest_path)
            status_path = write_status_md(self.run_dir, manifest, active_process=False)

        latest_info = publish_to_latest(
            self.run_dir,
            latest_dir=latest_dir,
            manifest=manifest,
            config_dir=self.config_dir,
        )
        emit_office_event(
            self.run_dir,
            "packet_handoff",
            stage="qa",
            from_station="qa",
            to_station="publish",
            artifact="render/final.mp4",
            label="final.mp4 → Latest & OneDrive",
        )
        emit_office_event(
            self.run_dir,
            "publish_complete",
            stage="qa",
            status="done",
            message="Published visual picks broadcast to data/latest & OneDrive mirror",
            extra={
                "latest_dir": str(latest_info.get("latest_dir", "")),
                "mirror_dir": str(latest_info.get("mirror_dir", "")),
            },
        )
        return {
            "ok": True,
            "refused": False,
            "status": manifest["status"],
            "run_kind": manifest.get("run_kind", "test"),
            "next_action": get_next_action_text(self.run_dir, manifest, active_process=False),
            "status_path": str(status_path),
            "storyboard_path": str(self.run_dir / "qa" / "storyboard.html"),
            "final_mp4": str(self.run_dir / "render" / "final.mp4"),
            "description_md": str(self.run_dir / "render" / "description.md"),
            "latest": latest_info,
        }

    def _mark_run_failed(self, manifest: dict[str, Any], reason: str) -> None:
        self.logger.error(f"Run failed: {reason}")
        manifest["status"] = "failed"
        reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
        save_manifest(manifest, self.manifest_path)
        write_status_md(self.run_dir, manifest, active_process=False)

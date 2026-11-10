"""Orchestrator implementation for Nepal News Bot."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
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

        # Setup logger
        self.logger = setup_logger("orchestrator", self.run_dir / "run.log")

        # Load configs
        self.pipeline_config = self._load_yaml(self.config_dir / "pipeline.yaml")
        self.sources_config = self._load_yaml(self.config_dir / "sources.yaml")

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
            if self.run_kind and not manifest.get("run_kind"):
                manifest["run_kind"] = self.run_kind
                save_manifest(manifest, self.manifest_path)
        else:
            self.logger.info(f"Initializing new run at {self.run_dir}")
            manifest = create_initial_manifest(self.run_date, self.run_date, self.pipeline_config)
            manifest["run_kind"] = self.run_kind or self._default_run_kind()
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

        if not force and stage_meta["status"] in PAUSE_STATES:
            current_stage_status = stage_meta["status"]
            manifest["status"] = "paused"
            save_manifest(manifest, self.manifest_path)
            write_status_md(self.run_dir, manifest, active_process=False)
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
            agent = agent_cls(self.run_dir, config=self.pipeline_config, backfill=self.backfill)
        elif stage_id == "media":
            agent = agent_cls(
                self.run_dir,
                config=self.pipeline_config,
                allow_fixture=self.allow_fixture,
                auto=self.media_auto,
            )
        else:
            agent = agent_cls(self.run_dir, config=self.pipeline_config)
        max_retries = stage_meta.get("max_retries", 2)

        while stage_meta["retry_count"] <= max_retries:
            attempt = stage_meta["retry_count"] + 1
            self.logger.info(f"Starting stage '{stage_id}' (attempt {attempt}/{max_retries + 1})...")

            update_stage_status(manifest, stage_id, "running")
            save_manifest(manifest, self.manifest_path)

            try:
                # Execute agent
                output_files = agent.run(manifest)

                # Check if agent entered a pause state (prepared, needs_revision, awaiting_approval)
                current_stage_status = manifest.get("stages", {}).get(stage_id, {}).get("status")
                if current_stage_status in PAUSE_STATES:
                    manifest["status"] = "paused"
                    save_manifest(manifest, self.manifest_path)
                    write_status_md(self.run_dir, manifest, active_process=False)
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
                update_stage_status(manifest, stage_id, "done", output_files=output_files)
                save_manifest(manifest, self.manifest_path)
                write_status_md(self.run_dir, manifest)
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
                save_manifest(manifest, self.manifest_path)

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
                save_manifest(manifest, self.manifest_path)
                status_path = write_status_md(self.run_dir, manifest, active_process=False)
        finally:
            self.media_auto = prev_media_auto

        latest_info = publish_to_latest(self.run_dir, latest_dir=latest_dir, manifest=manifest)
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

    def _mark_run_failed(self, manifest: dict[str, Any], reason: str) -> None:
        self.logger.error(f"Run failed: {reason}")
        manifest["status"] = "failed"
        save_manifest(manifest, self.manifest_path)
        write_status_md(self.run_dir, manifest, active_process=False)

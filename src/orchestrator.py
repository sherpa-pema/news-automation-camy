"""Orchestrator implementation stub for Nepal News Bot."""
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

PAUSE_STATES = {"prepared", "needs_revision", "awaiting_approval"}


class PipelinePausedException(Exception):
    """Raised when the pipeline pauses for human or agent intervention."""

    def __init__(self, status: str, next_action: str = "") -> None:
        self.status = status
        self.next_action = next_action
        super().__init__(f"Pipeline paused in stage state '{status}'. {next_action}")

from src.agents.base import BaseAgent
from src.agents.scraper import ScraperAgent
from src.agents.editor import EditorAgent
from src.agents.voice import VoiceAgent
from src.agents.media import MediaAgent
from src.agents.timeline import TimelineAgent
from src.agents.motion import MotionAgent
from src.agents.render import RenderAgent
from src.agents.qa import QAAgent
from src.utils.editor_validator import compute_validator_fingerprint
from src.utils.logging import setup_logger
from src.utils.manifest import (
    create_initial_manifest,
    load_manifest,
    save_manifest,
    update_stage_status,
)


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
    ) -> None:
        self.run_date = run_date or date.today().isoformat()
        self.config_dir = Path(config_dir)
        self.runs_dir = Path(runs_dir)
        self.run_dir = self.runs_dir / self.run_date
        self.manifest_path = self.run_dir / "manifest.json"
        self.allow_fixture = allow_fixture
        self.backfill = backfill

        # Setup logger
        self.logger = setup_logger("orchestrator", self.run_dir / "run.log")

        # Load configs
        self.pipeline_config = self._load_yaml(self.config_dir / "pipeline.yaml")
        self.sources_config = self._load_yaml(self.config_dir / "sources.yaml")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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
        else:
            self.logger.info(f"Initializing new run at {self.run_dir}")
            manifest = create_initial_manifest(self.run_date, self.run_date, self.pipeline_config)
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
        if manifest.get("status") == "failed":
            manifest["status"] = "pending"

        from datetime import datetime
        entry = {
            "file": str(self.manifest_path.as_posix()),
            "what": f"Reset stages.{stage_id} status from '{old_status}' to 'pending', retry_count from {old_retry} to 0, error_message to null",
            "why": reason or f"Reset stage '{stage_id}' via --reset-stage CLI",
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        manifest.setdefault("manual_edits", []).append(entry)

        save_manifest(manifest, self.manifest_path)
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
                curr_fp = compute_validator_fingerprint()
                app_fp = app_data.get("validator_fingerprint")
                if app_fp and app_fp != curr_fp:
                    raise ValueError("validator or editor config changed after approval, re-run validate and approve")

                rep_file = self.run_dir / "script" / "validation_report.json"
                if rep_file.exists():
                    with open(rep_file, "r", encoding="utf-8") as f:
                        rep_data = json.load(f)
                    rep_fp = rep_data.get("validator_fingerprint")
                    if rep_fp and rep_fp != curr_fp:
                        raise ValueError("validator or editor config changed since validation, re-run validate and approve")

        if stage_id == "scraper":
            agent = agent_cls(self.run_dir, config=self.pipeline_config, backfill=self.backfill)
        elif stage_id == "media":
            agent = agent_cls(self.run_dir, config=self.pipeline_config, allow_fixture=self.allow_fixture)
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
        manifest["status"] = "running"
        save_manifest(manifest, self.manifest_path)

        if only_stage:
            self.logger.info(f"Executing ONLY stage '{only_stage}' for {self.run_date}")
            if only_stage not in manifest["stages"]:
                self.logger.error(f"Unknown stage '{only_stage}' specified in --only.")
                return False

            success = self.execute_stage(only_stage, manifest, force=True)
            if not success:
                self._mark_run_failed(manifest, f"Stage '{only_stage}' failed")
                return False

            save_manifest(manifest, self.manifest_path)
            self.logger.info(f"Target stage '{only_stage}' completed successfully.")
            return True

        self.logger.info(f"Starting full pipeline execution for {self.run_date}")

        # 1. Scraper
        if not self.execute_stage("scraper", manifest):
            self._mark_run_failed(manifest, "Scraper stage failed")
            return False

        # 2. Editor
        if not self.execute_stage("editor", manifest):
            self._mark_run_failed(manifest, "Editor stage failed")
            return False

        # 3. Voice (synthesizes audio)
        if not self.execute_stage("voice", manifest):
            self._mark_run_failed(manifest, "Voice stage failed")
            return False

        # 4. Timeline (depends on voice)
        if not self.execute_stage("timeline", manifest):
            self._mark_run_failed(manifest, "Timeline stage failed")
            return False

        # 5. Media (depends on timeline + editor)
        if not self.execute_stage("media", manifest):
            self._mark_run_failed(manifest, "Media stage failed")
            return False

        # 5. Motion (depends on timeline + media)
        if not self.execute_stage("motion", manifest):
            self._mark_run_failed(manifest, "Motion stage failed")
            return False

        # 6. Render (depends on motion + media + voice + timeline)
        if not self.execute_stage("render", manifest):
            self._mark_run_failed(manifest, "Render stage failed")
            return False

        # 7. QA
        if not self.execute_stage("qa", manifest):
            self._mark_run_failed(manifest, "QA stage failed")
            return False

        manifest["status"] = "done"
        save_manifest(manifest, self.manifest_path)
        self.logger.info("Pipeline execution completed successfully!")
        return True

    def _mark_run_failed(self, manifest: dict[str, Any], reason: str) -> None:
        self.logger.error(f"Run failed: {reason}")
        manifest["status"] = "failed"
        save_manifest(manifest, self.manifest_path)

"""Orchestrator implementation stub for Nepal News Bot."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any
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
from src.utils.logging import setup_logger
from src.utils.manifest import (
    create_initial_manifest,
    load_manifest,
    save_manifest,
    update_stage_status,
)
from src.utils.schema_validate import validate_contract, validate_file


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
    ) -> None:
        self.run_date = run_date or date.today().isoformat()
        self.config_dir = Path(config_dir)
        self.runs_dir = Path(runs_dir)
        self.run_dir = self.runs_dir / self.run_date
        self.manifest_path = self.run_dir / "manifest.json"

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

    def execute_stage(self, stage_id: str, manifest: dict[str, Any], force: bool = False) -> bool:
        """Executes a single agent stage with contract validation and retry logic."""
        stage_meta = manifest["stages"].get(stage_id)
        if not stage_meta:
            raise ValueError(f"Unknown stage: {stage_id}")

        if not force and stage_meta["status"] == "done":
            self.logger.info(f"Stage '{stage_id}' already completed. Skipping.")
            return True

        if force:
            stage_meta["retry_count"] = 0
            stage_meta["error_message"] = None

        agent_cls = AGENT_REGISTRY.get(stage_id)
        if not agent_cls:
            raise ValueError(f"No agent registered for stage '{stage_id}'")

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

                # Validate outputs against contract schema
                is_valid, validation_err = agent.validate(output_files)
                if not is_valid:
                    raise ValueError(f"Contract validation failed: {validation_err}")

                # Mark stage done
                update_stage_status(manifest, stage_id, "done", output_files=output_files)
                save_manifest(manifest, self.manifest_path)
                self.logger.info(f"Stage '{stage_id}' finished successfully.")
                return True

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

        # 3. Parallel Fork: Voice and Media
        self.logger.info("Forking parallel execution: [Voice Agent, Media Agent]...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_voice = executor.submit(self.execute_stage, "voice", manifest)
            future_media = executor.submit(self.execute_stage, "media", manifest)

            voice_ok = future_voice.result()
            media_ok = future_media.result()

        if not voice_ok or not media_ok:
            self._mark_run_failed(manifest, "Parallel branch (Voice/Media) failed")
            return False

        # 4. Timeline (depends on voice)
        if not self.execute_stage("timeline", manifest):
            self._mark_run_failed(manifest, "Timeline stage failed")
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

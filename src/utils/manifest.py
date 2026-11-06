"""Manifest operations utility stub."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from src.utils.schema_validate import validate_contract


def create_initial_manifest(run_id: str, date_str: str, pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """Generates an initial manifest structure with all pipeline stages in 'pending' status."""
    now_iso = datetime.now().astimezone().isoformat()
    stages = {}

    for stage_cfg in pipeline_config.get("stages", []):
        stage_id = stage_cfg["id"]
        stages[stage_id] = {
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "retry_count": 0,
            "max_retries": stage_cfg.get("retries", 2),
            "input_files": stage_cfg.get("inputs", []),
            "output_files": stage_cfg.get("outputs", []),
            "error_message": None,
        }

    manifest = {
        "run_id": run_id,
        "date": date_str,
        "status": "pending",
        "created_at": now_iso,
        "updated_at": now_iso,
        "stages": stages,
    }
    return manifest


def load_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Loads and returns the manifest dictionary from disk.

    TODO: Implement schema validation against manifest.schema.json.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any], manifest_path: Path | str) -> None:
    """Atomically writes the manifest to disk and updates updated_at timestamp.

    TODO: Use tempfile + atomic rename to prevent corruption during concurrent reads.
    """
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def update_stage_status(
    manifest: dict[str, Any],
    stage_id: str,
    status: str,
    error_message: str | None = None,
    output_files: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    source_errors: dict[str, str] | None = None,
) -> None:
    """Updates the state of a single stage within the in-memory manifest."""
    if stage_id not in manifest["stages"]:
        raise KeyError(f"Stage '{stage_id}' not found in manifest")

    stage = manifest["stages"][stage_id]
    stage["status"] = status
    now_iso = datetime.now().astimezone().isoformat()

    if status == "running":
        stage["start_time"] = now_iso
    elif status in ("done", "failed"):
        stage["end_time"] = now_iso

    if error_message is not None:
        stage["error_message"] = error_message
    if output_files is not None:
        stage["output_files"] = output_files
    if metrics is not None:
        stage["metrics"] = metrics
    if source_errors is not None:
        stage["source_errors"] = source_errors

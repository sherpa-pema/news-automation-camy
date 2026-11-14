"""Safe, non-blocking JSONL telemetry emitter for the 2D Office web interface.

Writes append-only structured events to `<run_dir>/office_events.jsonl`.
All operations are fail-safe so telemetry errors can never affect pipeline execution
or deterministic gates.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

EVENTS_FILENAME = "office_events.jsonl"


def emit_office_event(
    run_dir: Path | str,
    event_type: str,
    *,
    stage: str | None = None,
    from_station: str | None = None,
    to_station: str | None = None,
    artifact: str | None = None,
    label: str | None = None,
    message: str | None = None,
    status: str | None = None,
    current: int | None = None,
    total: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Appends a single structured telemetry event to `<run_dir>/office_events.jsonl`.

    Silently ignores any filesystem or serialization errors.
    """
    try:
        rdir = Path(run_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "ts": datetime.now().astimezone().isoformat(),
            "type": event_type,
        }
        if stage is not None:
            payload["stage"] = stage
        if from_station is not None:
            payload["from"] = from_station
        if to_station is not None:
            payload["to"] = to_station
        if artifact is not None:
            payload["artifact"] = artifact
            art_path = rdir / artifact
            if art_path.exists() and art_path.is_file():
                payload["size_bytes"] = art_path.stat().st_size
        if label is not None:
            payload["label"] = label
        if message is not None:
            payload["message"] = message
        if status is not None:
            payload["status"] = status
        if current is not None:
            payload["current"] = current
        if total is not None:
            payload["total"] = total
        if extra:
            payload["extra"] = extra

        events_path = rdir / EVENTS_FILENAME
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never allow UI telemetry to interfere with pipeline execution
        pass

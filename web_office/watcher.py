"""Run state reader, log/event tailer, and historical replay synthesizer for 2D Office."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from src.utils.daily_ops import (
    compute_source_health,
    get_kathmandu_today,
    get_next_action_text,
    load_outputs_config,
    verify_editor_gate,
)
from src.utils.llm_writer import detect_configured_provider
from src.utils.manifest import (
    compute_manifest_status,
    is_run_process_active,
    load_manifest,
)
from src.utils.office_events import EVENTS_FILENAME

DATED_RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

STAGE_ORDER = ("scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa")

HANDOFF_MAP: dict[str, tuple[str, str, str]] = {
    "scraper": ("editor", "raw/articles.json", "articles.json"),
    "editor": ("voice", "script/script.json", "script.json"),
    "voice": ("timeline", "audio/narration.wav", "narration.wav"),
    "timeline": ("media", "timeline.json", "timeline.json"),
    "media": ("motion", "media/media_manifest.json", "media_manifest.json"),
    "motion": ("render", "motion/render_plan.json", "render_plan.json"),
    "render": ("qa", "render/final.mp4", "final.mp4"),
}

# --- Caches with mtime invalidation ---
_PROVIDER_CACHE: dict[str, tuple[float, tuple[str | None, str | None]]] = {}
_OUTPUTS_CONFIG_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_RUN_STATE_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def clear_run_state_cache(run_dir: Path | None = None) -> None:
    """Clears the run state cache globally or for a specific run directory."""
    if run_dir is not None:
        _RUN_STATE_CACHE.pop(str(Path(run_dir).resolve()), None)
    else:
        _RUN_STATE_CACHE.clear()
        _PROVIDER_CACHE.clear()
        _OUTPUTS_CONFIG_CACHE.clear()


def _get_cached_provider(repo_root: Path = Path(".")) -> tuple[str | None, str | None]:
    env_path = repo_root / ".env"
    mtime = env_path.stat().st_mtime if env_path.exists() else 0.0
    key = str(repo_root.resolve())
    cached = _PROVIDER_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    res = detect_configured_provider()
    _PROVIDER_CACHE[key] = (mtime, res)
    return res


def _get_cached_outputs_config(config_dir: Path) -> dict[str, Any]:
    cfg_file = config_dir / "outputs.yaml"
    mtime = cfg_file.stat().st_mtime if cfg_file.exists() else 0.0
    key = str(config_dir.resolve())
    cached = _OUTPUTS_CONFIG_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    cfg = load_outputs_config(config_dir)
    _OUTPUTS_CONFIG_CACHE[key] = (mtime, cfg)
    return cfg


def _read_recent_log_lines(path: Path, max_lines: int = 250, max_bytes: int = 40_000) -> list[str]:
    """Reads the last lines from a log file efficiently from the end."""
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _cap_event_payload(ev: dict[str, Any], max_str_len: int = 600) -> dict[str, Any]:
    """Caps large fields in event payloads to prevent unbounded memory usage."""
    capped = dict(ev)
    if "extra" in capped and isinstance(capped["extra"], dict):
        extra = dict(capped["extra"])
        for k, v in extra.items():
            if isinstance(v, str) and len(v) > max_str_len:
                extra[k] = v[:max_str_len] + "... [truncated]"
        capped["extra"] = extra
    if "message" in capped and isinstance(capped["message"], str) and len(capped["message"]) > max_str_len:
        capped["message"] = capped["message"][:max_str_len] + "... [truncated]"
    return capped


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_text(path: Path, max_chars: int = 120_000) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except Exception:
        return None


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Lists all dated runs in data/runs sorted newest-first."""
    today = get_kathmandu_today()
    items: list[dict[str, Any]] = []
    seen_dates: set[str] = set()

    if runs_dir.exists():
        for entry in sorted(runs_dir.iterdir(), reverse=True):
            if not entry.is_dir() or not DATED_RUN_RE.match(entry.name):
                continue
            seen_dates.add(entry.name)
            manifest_path = entry / "manifest.json"
            manifest = _safe_read_json(manifest_path) or {}
            active = is_run_process_active(entry)
            status = (
                compute_manifest_status(manifest, run_dir=entry, active_process=active)
                if manifest
                else "pending"
            )
            stages_summary = {
                s: manifest.get("stages", {}).get(s, {}).get("status", "pending")
                for s in STAGE_ORDER
            }
            items.append(
                {
                    "date": entry.name,
                    "exists": True,
                    "status": status,
                    "run_kind": manifest.get("run_kind", "test"),
                    "active_process": active,
                    "updated_at": manifest.get("updated_at"),
                    "has_final_mp4": (entry / "render" / "final.mp4").exists(),
                    "stages": stages_summary,
                }
            )

    if today not in seen_dates:
        items.insert(
            0,
            {
                "date": today,
                "exists": False,
                "status": "uninitialized",
                "run_kind": "test",
                "active_process": False,
                "updated_at": None,
                "has_final_mp4": False,
                "stages": {s: "pending" for s in STAGE_ORDER},
            },
        )

    return items


def build_replay_events(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns real office_events.jsonl events, or synthesizes a replay timeline from manifest."""
    events_path = run_dir / EVENTS_FILENAME
    if events_path.exists():
        events: list[dict[str, Any]] = []
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev_obj = json.loads(line)
                events.append(_cap_event_payload(ev_obj))
            except Exception:
                pass
        if events:
            return events[-300:]

    # Synthesize from manifest.json so historical runs replay smoothly
    stages = manifest.get("stages", {})
    synth: list[dict[str, Any]] = []
    for st in STAGE_ORDER:
        meta = stages.get(st, {})
        st_status = meta.get("status", "pending")
        if st_status == "pending":
            continue
        ts_start = meta.get("start_time") or manifest.get("created_at") or ""
        ts_end = meta.get("end_time") or ts_start

        synth.append(
            {
                "ts": ts_start,
                "type": "stage_status",
                "stage": st,
                "status": "running",
                "message": f"Orchestrator dispatched {st}",
            }
        )

        if st == "editor":
            if (run_dir / "script" / "editor_brief.md").exists():
                synth.append(
                    {
                        "ts": ts_start,
                        "type": "packet_handoff",
                        "stage": "editor",
                        "from": "editor",
                        "to": "boss",
                        "artifact": "script/editor_brief.md",
                        "label": "editor_brief.md",
                    }
                )
            if (run_dir / "script" / "script_preview.md").exists():
                synth.append(
                    {
                        "ts": ts_end,
                        "type": "packet_handoff",
                        "stage": "editor",
                        "from": "editor",
                        "to": "boss",
                        "artifact": "script/script_preview.md",
                        "label": "script_preview.md",
                    }
                )
            if st_status == "done":
                synth.append(
                    {
                        "ts": ts_end,
                        "type": "packet_handoff",
                        "stage": "editor",
                        "from": "boss",
                        "to": "editor",
                        "artifact": "script/approval.json",
                        "label": "approval.json (SIGNED)",
                    }
                )

        if st_status == "done":
            synth.append(
                {
                    "ts": ts_end,
                    "type": "stage_status",
                    "stage": st,
                    "status": "done",
                    "message": f"Completed {st}",
                }
            )
            if st in HANDOFF_MAP:
                to_st, art, lbl = HANDOFF_MAP[st]
                art_path = run_dir / art
                ev: dict[str, Any] = {
                    "ts": ts_end,
                    "type": "packet_handoff",
                    "stage": st,
                    "from": st,
                    "to": to_st,
                    "artifact": art,
                    "label": lbl,
                }
                if art_path.exists() and art_path.is_file():
                    ev["size_bytes"] = art_path.stat().st_size
                synth.append(ev)
        elif st_status in ("prepared", "needs_revision", "awaiting_approval", "failed"):
            synth.append(
                {
                    "ts": ts_end,
                    "type": "stage_status",
                    "stage": st,
                    "status": st_status,
                    "message": meta.get("error_message") or f"State: {st_status}",
                }
            )

    if stages.get("qa", {}).get("status") == "done" and (run_dir / "render" / "final.mp4").exists():
        synth.append(
            {
                "ts": stages["qa"].get("end_time") or "",
                "type": "packet_handoff",
                "stage": "qa",
                "from": "qa",
                "to": "publish",
                "artifact": "render/final.mp4",
                "label": "final.mp4 -> Latest & OneDrive",
            }
        )

    return synth


def read_run_state(
    run_dir: Path,
    config_dir: Path = Path("config"),
    include_events: bool = False,
) -> dict[str, Any]:
    """Builds the full snapshot JSON for a target date run with fingerprint-based state caching."""
    run_date = run_dir.name
    manifest_path = run_dir / "manifest.json"
    pid_file = run_dir / ".pipeline.pid"
    log_file = run_dir / "run.log"
    status_file = run_dir / "STATUS.md"
    cand_file = run_dir / "media" / "review" / "candidates.json"
    qa_file = run_dir / "qa" / "qa_report.json"
    script_file = run_dir / "script" / "script.json"
    final_file = run_dir / "render" / "final.mp4"

    # Compute fast filesystem mtime fingerprint
    fp = (
        manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0,
        manifest_path.stat().st_size if manifest_path.exists() else 0,
        pid_file.stat().st_mtime_ns if pid_file.exists() else 0,
        log_file.stat().st_mtime_ns if log_file.exists() else 0,
        log_file.stat().st_size if log_file.exists() else 0,
        status_file.stat().st_mtime_ns if status_file.exists() else 0,
        cand_file.stat().st_mtime_ns if cand_file.exists() else 0,
        qa_file.stat().st_mtime_ns if qa_file.exists() else 0,
        script_file.stat().st_mtime_ns if script_file.exists() else 0,
        final_file.exists(),
        include_events,
    )

    cache_key = str(run_dir.resolve())
    cached_entry = _RUN_STATE_CACHE.get(cache_key)
    if cached_entry and cached_entry[0] == fp:
        # Fast cache hit: update only live active process check and return copy
        st = dict(cached_entry[1])
        st["active_process"] = is_run_process_active(run_dir)
        return copy.deepcopy(st)

    exists = manifest_path.exists()
    active = is_run_process_active(run_dir)

    if exists:
        manifest = load_manifest(manifest_path)
        computed_status = compute_manifest_status(manifest, run_dir=run_dir, active_process=active)
        manifest["status"] = computed_status
    else:
        manifest = {
            "run_id": run_date,
            "date": run_date,
            "run_kind": "test",
            "status": "uninitialized",
            "created_at": None,
            "updated_at": None,
            "stages": {
                s: {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "retry_count": 0,
                    "max_retries": 2,
                    "input_files": [],
                    "output_files": [],
                    "error_message": None,
                }
                for s in STAGE_ORDER
            },
        }

    log_lines = _read_recent_log_lines(log_file, max_lines=250, max_bytes=40_000)

    repo_root = config_dir.parent if config_dir.name == "config" else Path(".")
    provider, model = _get_cached_provider(repo_root)
    outputs_cfg = _get_cached_outputs_config(config_dir)

    artifacts = {
        "has_news_md": (run_dir / "raw" / "news.md").exists(),
        "has_editor_brief": (run_dir / "script" / "editor_brief.md").exists(),
        "has_draft_script": (run_dir / "script" / "draft_script.json").exists(),
        "has_script_preview": (run_dir / "script" / "script_preview.md").exists(),
        "has_script_json": (run_dir / "script" / "script.json").exists(),
        "has_approval": (run_dir / "script" / "approval.json").exists(),
        "has_narration_wav": (run_dir / "audio" / "narration.wav").exists(),
        "has_timeline_preview": (run_dir / "timeline" / "timeline_preview.mp4").exists(),
        "has_candidates_json": cand_file.exists(),
        "has_picker_html": (run_dir / "media" / "review" / "picker.html").exists(),
        "has_media_review": (run_dir / "media" / "review" / "index.html").exists(),
        "has_final_mp4": final_file.exists(),
        "has_description_md": (run_dir / "render" / "description.md").exists(),
        "has_storyboard": (run_dir / "qa" / "storyboard.html").exists(),
        "has_contact_sheet": (run_dir / "qa" / "contact_sheet.jpg").exists(),
    }

    media_summary: dict[str, Any] | None = None
    cands_obj = _safe_read_json(cand_file)
    if isinstance(cands_obj, dict) and isinstance(cands_obj.get("shots"), list):
        shots_list = cands_obj["shots"]
        clips_cnt = sum(1 for sh in shots_list if sh.get("auto_pick_type") == "clip")
        media_summary = {
            "total_shots": len(shots_list),
            "auto_clips_count": clips_cnt,
            "auto_cards_count": len(shots_list) - clips_cnt,
            "min_relevance": cands_obj.get("min_relevance"),
        }

    gate_ok, gate_reason = verify_editor_gate(run_dir)
    next_action = get_next_action_text(run_dir, manifest, active_process=active)
    source_health = compute_source_health(run_dir, manifest=manifest, config_dir=config_dir)

    # Full events dropped from default snapshot payload for performance
    events_payload = build_replay_events(run_dir, manifest)[-200:] if include_events else []

    state_obj = {
        "date": run_date,
        "exists": exists,
        "active_process": active,
        "manifest": manifest,
        "gate": {"ok": gate_ok, "reason": gate_reason},
        "next_action": next_action,
        "source_health": source_health,
        "status_md": _safe_read_text(status_file) or "",
        "qa_report": _safe_read_json(qa_file),
        "script_preview_md": _safe_read_text(run_dir / "script" / "script_preview.md") or "",
        "validation_report": _safe_read_json(run_dir / "script" / "validation_report.json"),
        "budget": _safe_read_json(run_dir / "script" / "budget.json"),
        "approval": _safe_read_json(run_dir / "script" / "approval.json"),
        "description_md": _safe_read_text(run_dir / "render" / "description.md") or "",
        "artifacts": artifacts,
        "media_summary": media_summary,
        "recent_logs": log_lines,
        "events": events_payload,
        "llm_writer": {
            "configured": bool(provider),
            "provider": provider,
            "model": model,
        },
        "mirror_dir": outputs_cfg.get("mirror_dir"),
    }

    _RUN_STATE_CACHE[cache_key] = (fp, state_obj)
    return copy.deepcopy(state_obj)


def tail_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Reads newly appended lines from `path` starting at byte `offset`.

    Buffers partial lines so that incomplete lines are not split across polls.
    """
    if not path.exists():
        return [], 0
    try:
        size = path.stat().st_size
        if size < offset:
            offset = 0
        if size == offset:
            return [], offset
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
        if not chunk:
            return [], offset

        last_nl = chunk.rfind(b"\n")
        if last_nl == -1:
            # Chunk has no newline yet; wait until line completion
            return [], offset

        complete_chunk = chunk[: last_nl + 1]
        new_offset = offset + len(complete_chunk)
        lines = complete_chunk.decode("utf-8", errors="replace").splitlines()
        return lines, new_offset
    except Exception:
        return [], offset

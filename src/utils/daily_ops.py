"""Operational helpers for daily runs: STATUS.md, data/latest publishing, cleanup, doctor, and gate checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests
import yaml

from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.manifest import (
    compute_manifest_status,
    is_run_process_active,
    load_manifest,
)
from src.utils.media_cache import MediaCacheManager

logger = logging.getLogger("daily_ops")

KATHMANDU_TZ = ZoneInfo("Asia/Kathmandu")
PROTECTED_TOP_DIRS = ("data/media_library", ".env", "models", "vendor")
DATED_RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

STAGE_ORDER = ["scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa"]
_UNSET = object()


def get_default_mirror_dir() -> str:
    """Returns the default OneDrive viewer mirror path for finished latest deliverables."""
    return (Path.home() / "OneDrive" / "NepalNewsBot" / "latest").as_posix()


def get_kathmandu_today() -> str:
    """Returns today's YYYY-MM-DD date string in Asia/Kathmandu timezone."""
    return datetime.now(KATHMANDU_TZ).date().isoformat()


def load_execution_config(config_dir: Path | str = "config") -> dict[str, Any]:
    """Loads execution settings from config/pipeline.yaml."""
    cfg_path = Path(config_dir) / "pipeline.yaml"
    defaults = {
        "runs_dir": "data/runs",
        "default_retries": 2,
        "allow_parallel": True,
        "default_run_kind": "test",
        "keep_runs": 7,
    }
    if not cfg_path.exists():
        return defaults
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        exec_cfg = dict(defaults)
        exec_cfg.update(data.get("execution", {}) or {})
        return exec_cfg
    except Exception:
        return defaults


def load_outputs_config(config_dir: Path | str = "config") -> dict[str, Any]:
    """Loads outputs settings (such as mirror_dir) from config/pipeline.yaml."""
    cfg_path = Path(config_dir) / "pipeline.yaml"
    defaults: dict[str, Any] = {
        "mirror_dir": get_default_mirror_dir(),
    }
    if not cfg_path.exists():
        return defaults
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if "outputs" in data:
            out_cfg = dict(data.get("outputs") or {})
            raw_mirror = out_cfg.get("mirror_dir")
            if raw_mirror:
                expanded = Path(os.path.expanduser(os.path.expandvars(str(raw_mirror).strip()))).as_posix()
                out_cfg["mirror_dir"] = expanded
            else:
                out_cfg["mirror_dir"] = None
            return out_cfg
        return defaults
    except Exception:
        return defaults


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    rem = seconds - mins * 60
    return f"{mins}m {rem:.0f}s"


def _compute_stage_elapsed(stage_meta: dict[str, Any]) -> str:
    t0 = _parse_iso(stage_meta.get("start_time"))
    t1 = _parse_iso(stage_meta.get("end_time"))
    if t0 and t1:
        if t0.tzinfo is None and t1.tzinfo is not None:
            t0 = t0.replace(tzinfo=t1.tzinfo)
        elif t1.tzinfo is None and t0.tzinfo is not None:
            t1 = t1.replace(tzinfo=t0.tzinfo)
        return _format_duration((t1 - t0).total_seconds())
    wall = (stage_meta.get("metrics") or {}).get("wall_time_sec")
    if isinstance(wall, (int, float)):
        return _format_duration(float(wall))
    return "-"


def verify_editor_gate(
    run_dir: Path | str,
    allow_fixture: bool = False,
) -> tuple[bool, str]:
    """Verifies that editor stage is done, approved, and the validator fingerprint gate passes."""
    rdir = Path(run_dir)
    manifest_path = rdir / "manifest.json"
    if not manifest_path.exists():
        return False, f"Manifest not found at {manifest_path}."

    manifest = load_manifest(manifest_path)
    if manifest.get("status") == "abandoned":
        return False, f"Run '{rdir.name}' is abandoned. No stage may run on it."

    editor_status = manifest.get("stages", {}).get("editor", {}).get("status", "pending")
    if editor_status != "done":
        return False, (
            f"Refusing to run downstream stages: editor stage status is '{editor_status}', "
            f"must be 'done' (run validate and approve first)."
        )

    script_path = rdir / "script" / "script.json"
    if not script_path.exists():
        return False, f"Missing validated script at {script_path}."

    try:
        script_data = json.loads(script_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to read {script_path}: {exc}"

    origin = script_data.get("origin", "fixture")
    if origin == "fixture":
        if not allow_fixture:
            return False, "Refusing to run downstream stages on fixture script without --allow-fixture."
        return True, "Fixture script allowed via --allow-fixture."

    # Read approval file without performing any write operations here
    sig_file = rdir / "script" / "approval.json"
    if not sig_file.exists():
        return False, "Missing script sign-off file; run validate and approve first."

    try:
        app_data = json.loads(sig_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Unable to parse sign-off file: {exc}"

    curr_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    if app_data.get("script_sha256") != curr_hash:
        return False, "script changed after approval, re-run validate and approve"

    curr_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
    app_fp = extract_script_validation_fingerprint(app_data.get("validator_fingerprint"))
    is_fixture = bool(origin == "fixture" or manifest.get("run_kind") == "fixture")
    if not is_fixture:
        if not app_fp or app_fp != curr_fp:
            return False, "validator or editor config changed after approval, re-run validate and approve"

        rep_file = rdir / "script" / "validation_report.json"
        if not rep_file.exists():
            return False, "validation_report.json missing; re-run validate and approve"
        try:
            rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"Unable to read validation report: {exc}"
        if not rep_data.get("is_valid", True) or len(rep_data.get("errors", [])) > 0:
            return False, "Latest validation report contains errors; re-run validate and approve."
        rep_fp = extract_script_validation_fingerprint(rep_data.get("validator_fingerprint"))
        if not rep_fp or rep_fp != curr_fp:
            return False, "validator or editor config changed since validation, re-run validate and approve"
    else:
        if app_fp and app_fp != curr_fp:
            return False, "validator or editor config changed after approval, re-run validate and approve"
        rep_file = rdir / "script" / "validation_report.json"
        if rep_file.exists():
            try:
                rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
            except Exception as exc:
                return False, f"Unable to read validation report: {exc}"
            if not rep_data.get("is_valid", True) or len(rep_data.get("errors", [])) > 0:
                return False, "Latest validation report contains errors; re-run validate and approve."
            rep_fp = extract_script_validation_fingerprint(rep_data.get("validator_fingerprint"))
            if rep_fp and rep_fp != curr_fp:
                return False, "validator or editor config changed since validation, re-run validate and approve"

    return True, "Editor stage done, script hash verified, validator fingerprint matches."


def _load_qa_report(run_dir: Path) -> dict[str, Any] | None:
    for rel in ("qa/qa_report.json", "qa_report.json"):
        p = run_dir / rel
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def _resolve_run_kind(run_dir: Path, manifest: dict[str, Any]) -> str:
    if manifest.get("run_kind"):
        return str(manifest["run_kind"])
    script_path = run_dir / "script" / "script.json"
    if script_path.exists():
        try:
            s_data = json.loads(script_path.read_text(encoding="utf-8"))
            if s_data.get("run_kind"):
                return str(s_data["run_kind"])
        except Exception:
            pass
    return "test"


def get_next_action_text(
    run_dir: Path,
    manifest: dict[str, Any],
    active_process: bool | None = None,
) -> str:
    """Returns plain-English next step based on current run status and stages."""
    top_status = manifest.get("status", "pending")
    stages = manifest.get("stages", {})
    run_date = manifest.get("date", run_dir.name)
    is_active = is_run_process_active(run_dir) if active_process is None else active_process

    if top_status == "abandoned":
        return "This run is marked abandoned. Start a new run date or switch to an active run."

    if top_status == "running" and is_active:
        running_stages = [k for k, v in stages.items() if v.get("status") == "running"]
        stage_lbl = ", ".join(running_stages) if running_stages else "pipeline"
        return f"Pipeline process is currently running ({stage_lbl}). Wait for completion or re-check --status later."

    if top_status == "interrupted":
        running_stages = [k for k, v in stages.items() if v.get("status") == "running"]
        target_st = running_stages[0] if running_stages else "voice"
        return (
            f"Run appears interrupted (no active process after timeout while '{target_st}' was running). "
            f"Reset with `python run_daily.py --date {run_date} --reset-stage {target_st}` and re-run `python run_daily.py --finish --date {run_date}`."
        )

    failed = [(k, v) for k, v in stages.items() if v.get("status") == "failed"]
    if failed:
        st_id, st_meta = failed[0]
        err = st_meta.get("error_message") or "Unknown error"
        return (
            f"Stage '{st_id}' failed ({err}). Inspect logs, fix the issue, run "
            f"`python run_daily.py --date {run_date} --reset-stage {st_id}`, and re-run."
        )

    editor_status = stages.get("editor", {}).get("status", "pending")
    scraper_status = stages.get("scraper", {}).get("status", "pending")

    if editor_status == "prepared":
        return (
            f"Needs writer: read data/runs/{run_date}/script/editor_brief.md, write "
            f"draft_script.json and selection.json, then run `python -m src.agents.editor validate --date {run_date}`."
        )
    if editor_status == "needs_revision":
        return (
            f"Needs writer revision: fix validation errors in draft_script.json (see "
            f"data/runs/{run_date}/script/validation_report.json) and re-run `python -m src.agents.editor validate --date {run_date}`."
        )
    if editor_status == "awaiting_approval":
        return (
            f"Awaiting your approval: review data/runs/{run_date}/script/script_preview.md, then run "
            f"`python -m src.agents.editor approve --date {run_date}` followed by `python run_daily.py --finish --date {run_date}`."
        )
    if scraper_status != "done":
        return f"Run morning prep: `python run_daily.py --morning --date {run_date}`."
    if editor_status == "pending":
        return f"Run morning prep (editor prepare): `python run_daily.py --morning --date {run_date}`."

    media_status = stages.get("media", {}).get("status", "pending")
    if media_status in ("prepared", "awaiting_approval"):
        return (
            f"Visual candidates ready: open `data/runs/{run_date}/media/review/picker.html` (or Boss Office Visual Picker), "
            f"choose visuals, and run `python run_daily.py --apply-picks data/runs/{run_date}/media/picks.json --date {run_date}` "
            f"(or `python run_daily.py --finish --date {run_date}` for auto selection)."
        )

    downstream = ["voice", "timeline", "media", "motion", "render", "qa"]
    if all(stages.get(s, {}).get("status") == "done" for s in STAGE_ORDER if s in stages):
        return (
            f"All stages complete. Inspect `data/runs/{run_date}/qa/storyboard.html` and "
            f"`data/runs/{run_date}/render/description.md`. To swap visuals, edit choices in "
            f"`data/runs/{run_date}/media/review/picker.html`, apply picks, and re-run `--only render` & `--only qa`."
        )

    pending_ds = [s for s in downstream if s in stages and stages[s].get("status") != "done"]
    if editor_status == "done" and pending_ds:
        return f"Script approved. Run `python run_daily.py --finish --date {run_date}` to complete {', '.join(pending_ds)}."

    return f"Run `python run_daily.py --finish --date {run_date}` to resume remaining stages."


def _stage_detail_note(stage_id: str, st: dict[str, Any]) -> str:
    if st.get("error_message"):
        return str(st["error_message"]).replace("|", "/")
    m = st.get("metrics") or {}
    if stage_id == "scraper" and "total_articles" in m:
        return f"{m['total_articles']} articles"
    if stage_id == "editor":
        if st.get("status") == "prepared" and "shortlisted_clusters" in m:
            return f"needs writer ({m['shortlisted_clusters']} clusters)"
        if "total_words" in m:
            return f"{m['total_words']} words"
    if stage_id == "voice" and "total_duration_sec" in m:
        return f"{m['total_duration_sec']}s audio"
    if stage_id == "media":
        if m.get("prepared_for_picking") and st.get("status") in ("prepared", "awaiting_approval"):
            return f"picker ready: {m.get('auto_clips_count', 0)} clips, {m.get('auto_cards_count', 0)} cards"
        if m.get("picked_by_user"):
            return f"picked: {m.get('applied_clips_count', 0)} clips, {m.get('cards_created_count', 0)} cards"
        if m.get("auto_mode"):
            return f"auto: {m.get('auto_clips_count', 0)} clips, {m.get('auto_cards_count', 0)} cards"
        if "total_shots" in m:
            return f"{m['total_shots']} shots"
    if stage_id == "qa" and "checks_count" in m:
        return f"passed={m.get('passed')} ({m.get('warnings_count', 0)} warnings)"
    return "-"


def _load_sources_list(
    run_dir: Path,
    config_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Loads source entries from config/sources.yaml (preferring adjacent test config if present)."""
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(Path(config_dir) / "sources.yaml")
    candidates.append(run_dir.parent.parent / "config" / "sources.yaml")
    candidates.append(run_dir.parent.parent.parent / "config" / "sources.yaml")
    candidates.append(Path("config") / "sources.yaml")
    candidates.append(Path(__file__).resolve().parents[2] / "config" / "sources.yaml")

    for cfg_path in candidates:
        if cfg_path.is_file():
            try:
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                srcs = data.get("sources") or []
                if isinstance(srcs, list):
                    return srcs
            except Exception:
                pass
    return []


NEWS_MD_BREAKDOWN_RE = re.compile(
    r"^\s*-\s+\*\*(?P<source>.+?)\*\*:\s*(?P<kept>\d+)\s+kept\s*\(found:\s*(?P<found>\d+)"
)


def _extract_run_source_stats(
    target_rdir: Path,
    run_manifest: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, dict[str, Any]], dict[str, str]]:
    """Extracts per-source stats and errors from manifest, news.md, and articles.json."""
    if run_manifest is None and (target_rdir / "manifest.json").is_file():
        try:
            run_manifest = load_manifest(target_rdir / "manifest.json")
        except Exception:
            run_manifest = {}
    run_manifest = run_manifest or {}

    scraper_stage = (run_manifest.get("stages") or {}).get("scraper") or {}
    raw_stats = (scraper_stage.get("metrics") or {}).get("per_source_stats") or {}
    stats_map: dict[str, dict[str, Any]] = {}
    for k, v in raw_stats.items():
        if isinstance(v, dict):
            stats_map[str(k)] = dict(v)
    errors_map: dict[str, str] = {
        str(k): str(v) for k, v in (scraper_stage.get("source_errors") or {}).items()
    }

    news_md_path = target_rdir / "raw" / "news.md"
    if not stats_map and news_md_path.is_file():
        try:
            for line in news_md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = NEWS_MD_BREAKDOWN_RE.match(line)
                if m:
                    src = m.group("source").strip()
                    stats_map[src] = {
                        "found": int(m.group("found")),
                        "extracted_ok": int(m.group("kept")),
                    }
        except Exception:
            pass

    articles_path = target_rdir / "raw" / "articles.json"
    if articles_path.is_file():
        try:
            arts = json.loads(articles_path.read_text(encoding="utf-8"))
            if isinstance(arts, list):
                stage_ref = _parse_iso(scraper_stage.get("end_time")) or _parse_iso(
                    scraper_stage.get("start_time")
                )
                grouped_ages: dict[str, list[float]] = {}
                grouped_counts: dict[str, int] = {}
                for art in arts:
                    if not isinstance(art, dict):
                        continue
                    src = str(art.get("source") or "Unknown")
                    grouped_counts[src] = grouped_counts.get(src, 0) + 1
                    age_val = art.get("age_hours")
                    if isinstance(age_val, (int, float)):
                        grouped_ages.setdefault(src, []).append(max(0.0, float(age_val)))
                    else:
                        pub_dt = _parse_iso(art.get("published_at"))
                        ref_dt = _parse_iso(art.get("scraped_at")) or stage_ref
                        if pub_dt is not None and ref_dt is not None:
                            if pub_dt.tzinfo is None and ref_dt.tzinfo is not None:
                                pub_dt = pub_dt.replace(tzinfo=KATHMANDU_TZ)
                            if ref_dt.tzinfo is None and pub_dt.tzinfo is not None:
                                ref_dt = ref_dt.replace(tzinfo=KATHMANDU_TZ)
                            diff_sec = (ref_dt - pub_dt).total_seconds()
                            if diff_sec >= -3600:
                                grouped_ages.setdefault(src, []).append(max(0.0, round(diff_sec / 3600.0, 2)))

                for src, cnt in grouped_counts.items():
                    entry = stats_map.setdefault(src, {"found": cnt, "extracted_ok": cnt})
                    if entry.get("extracted_ok") is None:
                        entry["extracted_ok"] = cnt
                    if entry.get("newest_age_hours") is None and src in grouped_ages and grouped_ages[src]:
                        entry["newest_age_hours"] = min(grouped_ages[src])
        except Exception:
            pass

    scraper_status = scraper_stage.get("status", "pending")
    has_data = bool(
        stats_map
        or errors_map
        or news_md_path.is_file()
        or articles_path.is_file()
        or scraper_status in ("done", "failed")
    )
    return has_data, stats_map, errors_map


def _classify_source_health_status(
    kept: int,
    newest_age_hours: float | None,
    error: str | None,
    stale_threshold_hours: float = 48.0,
) -> str:
    if error and "ROBOTS_UNKNOWN" in str(error):
        return "ROBOTS_UNKNOWN"
    if error:
        return "ERROR"
    if newest_age_hours is not None and newest_age_hours > stale_threshold_hours:
        return "STALE"
    if kept <= 0:
        return "EMPTY"
    return "OK"


def compute_source_health(
    run_dir: Path | str,
    manifest: dict[str, Any] | None = None,
    config_dir: Path | str | None = None,
    stale_threshold_hours: float = 48.0,
) -> list[dict[str, Any]]:
    """Computes per-source health rows (OK, STALE, EMPTY, ERROR, DEGRADED)."""
    rdir = Path(run_dir)
    if manifest is None and (rdir / "manifest.json").is_file():
        try:
            manifest = load_manifest(rdir / "manifest.json")
        except Exception:
            manifest = {}
    manifest = manifest or {}

    has_data, stats_map, errors_map = _extract_run_source_stats(rdir, manifest)

    sources_cfg = _load_sources_list(rdir, config_dir=config_dir)
    ordered_names: list[str] = []
    seen: set[str] = set()
    for src in sources_cfg:
        if src.get("enabled", True) and src.get("name"):
            sname = str(src["name"])
            if sname not in seen:
                ordered_names.append(sname)
                seen.add(sname)
    for extra in list(stats_map.keys()) + list(errors_map.keys()):
        if extra not in seen:
            ordered_names.append(extra)
            seen.add(extra)

    if not ordered_names:
        return []

    # Find immediately preceding dated run that has scraper data for 2-run DEGRADED detection
    prev_stats: dict[str, dict[str, Any]] = {}
    prev_errors: dict[str, str] = {}
    has_prev = False
    runs_parent = rdir.parent
    if runs_parent.is_dir():
        prior_dirs = sorted(
            [
                d
                for d in runs_parent.iterdir()
                if _is_valid_dated_dir(d) and (not _is_valid_dated_dir(rdir) or d.name < rdir.name)
            ],
            key=lambda p: p.name,
            reverse=True,
        )
        for pdir in prior_dirs:
            if pdir.resolve() == rdir.resolve():
                continue
            p_has, p_stats, p_errs = _extract_run_source_stats(pdir, None)
            if p_has:
                has_prev = True
                prev_stats = p_stats
                prev_errors = p_errs
                break

    rows: list[dict[str, Any]] = []
    for src_name in ordered_names:
        if not has_data:
            rows.append({
                "source": src_name,
                "found": 0,
                "kept": 0,
                "newest_age_hours": None,
                "status": "PENDING",
                "degraded": False,
                "display_status": "PENDING",
                "error": None,
            })
            continue

        cur = stats_map.get(src_name, {})
        found = int(cur.get("found", 0) or 0)
        kept = int(cur.get("extracted_ok", cur.get("kept", 0)) or 0)
        age_raw = cur.get("newest_age_hours")
        newest_age = round(max(0.0, float(age_raw)), 2) if isinstance(age_raw, (int, float)) else None
        err_msg = errors_map.get(src_name) or cur.get("error")

        base_status = _classify_source_health_status(
            kept=kept,
            newest_age_hours=newest_age,
            error=err_msg,
            stale_threshold_hours=stale_threshold_hours,
        )

        degraded = False
        if has_prev and base_status in ("EMPTY", "STALE") and (src_name in prev_stats or src_name in prev_errors):
            pcur = prev_stats.get(src_name, {})
            p_kept = int(pcur.get("extracted_ok", pcur.get("kept", 0)) or 0)
            p_age_raw = pcur.get("newest_age_hours")
            p_age = round(max(0.0, float(p_age_raw)), 2) if isinstance(p_age_raw, (int, float)) else None
            p_err = prev_errors.get(src_name) or pcur.get("error")
            prev_status = _classify_source_health_status(
                kept=p_kept,
                newest_age_hours=p_age,
                error=p_err,
                stale_threshold_hours=stale_threshold_hours,
            )
            if prev_status in ("EMPTY", "STALE"):
                degraded = True

        display_status = f"{base_status} (DEGRADED)" if degraded else base_status
        rows.append({
            "source": src_name,
            "found": found,
            "kept": kept,
            "newest_age_hours": newest_age,
            "status": base_status,
            "degraded": degraded,
            "display_status": display_status,
            "error": err_msg,
        })

    return rows


def format_source_health_markdown(rows: list[dict[str, Any]]) -> list[str]:
    """Formats the Source Health section for STATUS.md."""
    lines = ["## Source Health"]
    if not rows:
        lines.append("- No sources configured.")
        return lines

    lines.extend([
        "| Source | Found | Kept | Newest Age (h) | Status |",
        "| --- | --- | --- | --- | --- |",
    ])
    degraded_names: list[str] = []
    err_notes: list[str] = []
    for r in rows:
        age_str = f"{r['newest_age_hours']:.2f}" if r["newest_age_hours"] is not None else "-"
        lines.append(
            f"| `{r['source']}` | {r['found']} | {r['kept']} | {age_str} | `{r['display_status']}` |"
        )
        if r.get("degraded"):
            degraded_names.append(r["source"])
        if r.get("error"):
            err_notes.append(f"`{r['source']}`: {r['error']}")

    if degraded_names:
        lines.append(
            f"- **DEGRADED** (EMPTY or STALE in last 2 runs; non-blocking): {', '.join(degraded_names)}"
        )
    if err_notes:
        lines.append(f"- **Source Errors**: {'; '.join(err_notes)}")
    return lines


def format_source_health_text(rows: list[dict[str, Any]]) -> list[str]:
    """Formats the Source Health section for plain-text README.txt."""
    lines = ["Source Health:"]
    if not rows:
        lines.append("- (no sources recorded)")
        return lines
    for r in rows:
        age_str = f"{r['newest_age_hours']:.2f}h" if r["newest_age_hours"] is not None else "n/a"
        line = (
            f"- {r['source']}: found={r['found']}, kept={r['kept']}, "
            f"newest_age={age_str}, status={r['display_status']}"
        )
        if r.get("error"):
            line += f" ({r['error']})"
        lines.append(line)
    return lines


def render_status_markdown(
    run_dir: Path | str,
    manifest: dict[str, Any] | None = None,
    active_process: bool | None = None,
) -> str:
    """Formats STATUS.md content from manifest and on-disk run artifacts."""
    rdir = Path(run_dir)
    if manifest is None:
        m_path = rdir / "manifest.json"
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found at {m_path}")
        manifest = load_manifest(m_path)

    effective_manifest = dict(manifest)
    effective_manifest["status"] = compute_manifest_status(
        manifest,
        run_dir=rdir,
        active_process=active_process,
    )
    run_date = str(effective_manifest.get("date", rdir.name))
    run_kind = _resolve_run_kind(rdir, effective_manifest)
    top_status = effective_manifest["status"]
    updated_at = effective_manifest.get("updated_at", "-")
    next_action = get_next_action_text(rdir, effective_manifest, active_process=active_process)

    source_health_rows = compute_source_health(rdir, manifest=effective_manifest)

    lines = [
        f"# Daily Run Status - {run_date}",
        "",
        f"- **Date**: `{run_date}`",
        f"- **Run Kind**: `{run_kind}`",
        f"- **Overall Status**: `{top_status}`",
        f"- **Last Updated**: `{updated_at}`",
        "",
        *format_source_health_markdown(source_health_rows),
        "",
        "## Next Action",
        f"{next_action}",
        "",
        "## Stages",
        "| Stage | Status | Start | End | Elapsed | Details |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    stages = effective_manifest.get("stages", {})
    seen: set[str] = set()
    ordered_ids = [s for s in STAGE_ORDER if s in stages] + [s for s in stages if s not in STAGE_ORDER]
    for st_id in ordered_ids:
        if st_id in seen:
            continue
        seen.add(st_id)
        st = stages[st_id]
        st_status = st.get("status", "pending")
        t_start = st.get("start_time") or "-"
        t_end = st.get("end_time") or "-"
        elapsed = _compute_stage_elapsed(st)
        note = _stage_detail_note(st_id, st)
        lines.append(f"| `{st_id}` | `{st_status}` | `{t_start}` | `{t_end}` | `{elapsed}` | {note} |")

    lines.extend(["", "## QA Summary"])
    qa_rep = _load_qa_report(rdir)
    if qa_rep is None:
        lines.append("- QA has not run yet.")
    else:
        checks = qa_rep.get("checks", [])
        passed_cnt = sum(1 for c in checks if c.get("passed"))
        metrics = qa_rep.get("metrics", {})
        qa_warnings = qa_rep.get("warnings", [])
        lines.append(f"- **Result**: `{'PASS' if qa_rep.get('passed') else 'FAIL'}` ({passed_cnt}/{len(checks)} checks passed)")
        if "video_duration_sec" in metrics:
            lines.append(
                f"- **Duration**: {metrics.get('video_duration_sec')}s video / "
                f"{metrics.get('audio_duration_sec', '-')}s audio "
                f"(diff: {metrics.get('duration_diff_ms', '-')} ms)"
            )
        if "integrated_loudness_lufs" in metrics:
            lines.append(
                f"- **Loudness**: {metrics.get('integrated_loudness_lufs')} LUFS, "
                f"True Peak: {metrics.get('true_peak_db')} dBTP"
            )
        if qa_warnings:
            lines.append(f"- **Warnings ({len(qa_warnings)})**:")
            for w in qa_warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("- **Warnings**: None")
        failed_checks = [c for c in checks if not c.get("passed")]
        if failed_checks:
            lines.append("- **Failed Checks**:")
            for fc in failed_checks:
                lines.append(f"  - `{fc.get('name')}`: {fc.get('detail')}")

    lines.extend(["", "## Key Paths"])
    candidate_paths = [
        ("Status File", "STATUS.md"),
        ("Manifest", "manifest.json"),
        ("News Digest", "raw/news.md"),
        ("Editor Brief", "script/editor_brief.md"),
        ("Script Preview", "script/script_preview.md"),
        ("Approved Script", "script/script.json"),
        ("Visual Picker", "media/review/picker.html"),
        ("Final Video", "render/final.mp4"),
        ("Publish Description", "render/description.md"),
        ("QA Report", "qa/qa_report.json"),
        ("Storyboard", "qa/storyboard.html"),
        ("Contact Sheet", "qa/contact_sheet.jpg"),
    ]
    for label, rel in candidate_paths:
        p = rdir / rel
        state = "exists" if p.exists() else "pending"
        lines.append(f"- **{label}**: `{p.as_posix()}` ({state})")

    lines.append("")
    return "\n".join(lines)


def write_status_md(
    run_dir: Path | str,
    manifest: dict[str, Any] | None = None,
    active_process: bool | None = None,
) -> Path:
    """Writes data/runs/<D>/STATUS.md atomically and returns its Path."""
    rdir = Path(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    content = render_status_markdown(rdir, manifest=manifest, active_process=active_process)
    out_path = rdir / "STATUS.md"
    temp_path = out_path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, out_path)
    return out_path


def _copy_latest_deliverables_to_dir(src_dir: Path, dst_dir: Path) -> list[str]:
    """Atomically replaces deliverables inside dst_dir with everything from src_dir.

    Writes to temporary files (<name>.tmp) first, then uses os.replace() so that syncing
    daemons (OneDrive, Dropbox) never lock, upload, or sync half-written video or metadata.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Clean up any leftover temporary files from interrupted syncs
    for tmp_item in dst_dir.glob("*.tmp"):
        try:
            if tmp_item.is_dir():
                shutil.rmtree(tmp_item, ignore_errors=True)
            else:
                tmp_item.unlink(missing_ok=True)
        except Exception:
            pass

    for old_name in (
        "final.mp4",
        "description.md",
        "shorts_caption.txt",
        "qa_report.json",
        "storyboard.html",
        "contact_sheet.jpg",
        "thumb_01.jpg",
        "thumb_02.jpg",
        "thumb_03.jpg",
        "STATUS.md",
        "README.txt",
    ):
        old_p = dst_dir / old_name
        if old_p.exists():
            old_p.unlink()
    old_frames = dst_dir / "storyboard_frames"
    if old_frames.exists():
        shutil.rmtree(old_frames, ignore_errors=True)

    mirrored: list[str] = []
    for item in sorted(src_dir.iterdir(), key=lambda p: p.name):
        if item.name.endswith(".tmp"):
            continue
        target = dst_dir / item.name
        if item.is_dir():
            tmp_target = dst_dir / f"{item.name}.tmp"
            if tmp_target.exists():
                shutil.rmtree(tmp_target, ignore_errors=True)
            shutil.copytree(item, tmp_target, dirs_exist_ok=True)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp_target, target)
            mirrored.append(f"{item.name}/")
        elif item.is_file():
            tmp_target = dst_dir / f"{item.name}.tmp"
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            shutil.copy2(item, tmp_target)
            os.replace(tmp_target, target)
            mirrored.append(item.name)
    return mirrored


def publish_to_latest(
    run_dir: Path | str,
    latest_dir: Path | str = "data/latest",
    manifest: dict[str, Any] | None = None,
    mirror_dir: Path | str | None | Any = _UNSET,
    config_dir: Path | str = "config",
) -> dict[str, Any]:
    """Copies final deliverables into data/latest/ (replacing previous outputs), writes README.txt,
    and mirrors everything in data/latest/ to outputs.mirror_dir if configured."""
    rdir = Path(run_dir)
    ldir = Path(latest_dir)
    ldir.mkdir(parents=True, exist_ok=True)

    if manifest is None and (rdir / "manifest.json").exists():
        manifest = load_manifest(rdir / "manifest.json")
    manifest = manifest or {}

    # Clear previous deliverables cleanly before writing new ones
    for old_name in (
        "final.mp4",
        "description.md",
        "shorts_caption.txt",
        "qa_report.json",
        "storyboard.html",
        "contact_sheet.jpg",
        "thumb_01.jpg",
        "thumb_02.jpg",
        "thumb_03.jpg",
        "STATUS.md",
        "README.txt",
    ):
        old_p = ldir / old_name
        if old_p.exists():
            old_p.unlink()
    old_frames = ldir / "storyboard_frames"
    if old_frames.exists():
        shutil.rmtree(old_frames, ignore_errors=True)

    copied: list[str] = []
    # 1. final.mp4
    src_mp4 = rdir / "render" / "final.mp4"
    if not src_mp4.exists():
        src_mp4 = rdir / "final.mp4"
    if src_mp4.exists():
        shutil.copy2(src_mp4, ldir / "final.mp4")
        copied.append("final.mp4")

    # 2. description.md
    src_desc = rdir / "render" / "description.md"
    if src_desc.exists():
        shutil.copy2(src_desc, ldir / "description.md")
        copied.append("description.md")

    # 2b. shorts_caption.txt (for vertical/shorts runs)
    src_shorts = rdir / "render" / "shorts_caption.txt"
    if src_shorts.exists():
        shutil.copy2(src_shorts, ldir / "shorts_caption.txt")
        copied.append("shorts_caption.txt")

    # 3. qa_report.json
    src_qa = rdir / "qa" / "qa_report.json"
    if not src_qa.exists():
        src_qa = rdir / "qa_report.json"
    if src_qa.exists():
        shutil.copy2(src_qa, ldir / "qa_report.json")
        copied.append("qa_report.json")

    # 4. storyboard.html + frames
    src_sb = rdir / "qa" / "storyboard.html"
    if src_sb.exists():
        shutil.copy2(src_sb, ldir / "storyboard.html")
        copied.append("storyboard.html")
    src_sb_frames = rdir / "qa" / "storyboard_frames"
    if src_sb_frames.exists():
        shutil.copytree(src_sb_frames, ldir / "storyboard_frames", dirs_exist_ok=True)
        copied.append("storyboard_frames/")

    # 5. contact_sheet.jpg + thumbnail candidates
    src_cs = rdir / "qa" / "contact_sheet.jpg"
    if src_cs.exists():
        shutil.copy2(src_cs, ldir / "contact_sheet.jpg")
        copied.append("contact_sheet.jpg")

    for t_idx in (1, 2, 3):
        thumb_name = f"thumb_{t_idx:02d}.jpg"
        src_thumb = rdir / "qa" / thumb_name
        if src_thumb.exists():
            shutil.copy2(src_thumb, ldir / thumb_name)
            copied.append(thumb_name)

    # 6. STATUS.md
    src_status = rdir / "STATUS.md"
    if not src_status.exists() and manifest:
        write_status_md(rdir, manifest=manifest, active_process=False)
    if src_status.exists():
        shutil.copy2(src_status, ldir / "STATUS.md")
        copied.append("STATUS.md")

    # 7. README.txt
    run_date = str(manifest.get("date", rdir.name))
    run_kind = _resolve_run_kind(rdir, manifest)
    qa_rep = _load_qa_report(rdir) or {}
    qa_passed = qa_rep.get("passed", False)
    qa_warns = qa_rep.get("warnings", [])
    now_ktm = datetime.now(KATHMANDU_TZ).isoformat()
    source_health_rows = compute_source_health(rdir, manifest=manifest, config_dir=config_dir)

    readme_lines = [
        f"Nepal News Bot - Latest Output Package",
        f"======================================",
        f"Run Date:    {run_date}",
        f"Run Kind:    {run_kind}",
        f"Updated At:  {now_ktm}",
        f"Source Run:  {rdir.as_posix()}",
        f"QA Passed:   {qa_passed}",
        f"QA Warnings: {len(qa_warns)}",
    ]
    if run_kind == "test":
        readme_lines.extend([
            "",
            "*** TEST RUN - DO NOT PUBLISH ***",
        ])
    readme_lines.extend([
        "",
        *format_source_health_text(source_health_rows),
        "",
        "Files Included:",
        "- final.mp4            : Rendered broadcast video",
        "- description.md       : YouTube/publish title, chapters, sources, and credits",
        "- qa_report.json       : Automated QA verification report",
        "- storyboard.html      : Self-contained visual storyboard (double-click to view)",
        "- storyboard_frames/   : Frame JPEGs referenced by storyboard.html",
        "- contact_sheet.jpg    : Scene contact sheet overview",
        "- STATUS.md            : Run status and source health summary",
        "",
        "Warnings:",
    ])
    if qa_warns:
        for w in qa_warns:
            readme_lines.append(f"- {w}")
    else:
        readme_lines.append("- None")

    readme_path = ldir / "README.txt"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    copied.append("README.txt")

    if mirror_dir is _UNSET:
        out_cfg = load_outputs_config(config_dir=config_dir)
        resolved_mirror = out_cfg.get("mirror_dir")
    else:
        resolved_mirror = str(mirror_dir).strip() if mirror_dir else None

    mirrored: list[str] = []
    mirrored_to: str | None = None
    if resolved_mirror:
        mdir = Path(resolved_mirror)
        mirrored = _copy_latest_deliverables_to_dir(ldir, mdir)
        mirrored_to = str(mdir)

    return {
        "latest_dir": str(ldir),
        "copied": copied,
        "readme": str(readme_path),
        "mirror_dir": mirrored_to,
        "mirrored": mirrored,
    }


def _is_valid_dated_dir(p: Path) -> bool:
    if not p.is_dir() or not DATED_RUN_RE.match(p.name):
        return False
    try:
        datetime.strptime(p.name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _dir_or_file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for sub in path.rglob("*"):
            if sub.is_file():
                try:
                    total += sub.stat().st_size
                except OSError:
                    pass
        return total
    return 0


def _is_protected_path(target: Path, repo_root: Path, kept_run_dirs: set[Path]) -> bool:
    """Hard safety barrier preventing cleanup from touching protected paths or kept files."""
    resolved_target = target.resolve()
    for prot_rel in PROTECTED_TOP_DIRS:
        prot_abs = (repo_root / prot_rel).resolve()
        if resolved_target == prot_abs or prot_abs in resolved_target.parents:
            return True

    for kr in kept_run_dirs:
        kr_res = kr.resolve()
        if resolved_target == kr_res or kr_res in resolved_target.parents:
            return True

    # Never delete key retained run artifacts
    if target.name in (
        "script",
        "description.md",
        "final.mp4",
        "qa_report.json",
        "manifest.json",
        "credits.md",
        "media_manifest.json",
        "narration.wav",
    ):
        return True

    return False


def plan_cleanup(
    runs_dir: Path | str = "data/runs",
    media_config_path: Path | str = "config/media.yaml",
    pipeline_config_dir: Path | str = "config",
    keep_runs: int | None = None,
    repo_root: Path | str = ".",
) -> dict[str, Any]:
    """Plans deletion targets for runs older than keep_runs and stale media cache."""
    root = Path(repo_root).resolve()
    runs_path = Path(runs_dir)
    if not runs_path.is_absolute():
        runs_path = (root / runs_path).resolve()

    exec_cfg = load_execution_config(
        Path(pipeline_config_dir) if Path(pipeline_config_dir).is_absolute() else (root / pipeline_config_dir)
    )
    retention = int(keep_runs if keep_runs is not None else exec_cfg.get("keep_runs", 7))

    media_cfg_file = Path(media_config_path)
    if not media_cfg_file.is_absolute():
        media_cfg_file = root / media_cfg_file
    media_cfg: dict[str, Any] = {}
    if media_cfg_file.exists():
        media_cfg = yaml.safe_load(media_cfg_file.read_text(encoding="utf-8")) or {}
    m_settings = media_cfg.get("settings", {})

    cache_dir_raw = Path(m_settings.get("cache_dir", "data/media_cache"))
    cache_dir = cache_dir_raw if cache_dir_raw.is_absolute() else (root / cache_dir_raw)
    max_gb = float(m_settings.get("cache_max_gb", 5.0))
    keep_days = int(m_settings.get("cache_keep_days", 7))

    dated_dirs = sorted(
        [d for d in runs_path.iterdir() if _is_valid_dated_dir(d)] if runs_path.exists() else [],
        key=lambda p: p.name,
        reverse=True,
    )
    kept_dirs = dated_dirs[: max(0, retention)]
    older_dirs = dated_dirs[max(0, retention) :]
    kept_set = {d.resolve() for d in kept_dirs}

    run_delete_targets: list[dict[str, Any]] = []
    for old_run in older_dirs:
        candidates: list[Path] = [
            old_run / "media" / "shots",
            old_run / "media" / "review" / "thumbs",
            old_run / "audio" / "scenes",
            old_run / "audio" / "temp_sentences",
            old_run / "audio" / "raw_concat.wav",
            old_run / "render" / "cache",
            old_run / "render" / "video_noaudio.mp4",
            old_run / "render" / "concat_segments.txt",
            old_run / "qa" / "_tmp_frames",
            old_run / "timeline" / "timeline_preview.mp4",
        ]
        audio_dir = old_run / "audio"
        if audio_dir.exists():
            candidates.extend(sorted(audio_dir.glob("scene_*.wav")))

        for cand in candidates:
            if not cand.exists():
                continue
            if _is_protected_path(cand, root, kept_set):
                continue
            size_bytes = _dir_or_file_size(cand)
            run_delete_targets.append({
                "run_date": old_run.name,
                "path": str(cand),
                "kind": "dir" if cand.is_dir() else "file",
                "size_bytes": size_bytes,
            })

    # Evict superseded/stale render/cache/*.mp4 files within kept runs
    for kept_run in kept_dirs:
        rcache = kept_run / "render" / "cache"
        if not rcache.exists():
            continue
        scene_files: dict[str, list[Path]] = {}
        for mp4 in sorted(rcache.glob("scene_*.mp4")):
            parts = mp4.stem.split("_")
            prefix = "_".join(parts[:3]) if len(parts) >= 3 else mp4.stem
            scene_files.setdefault(prefix, []).append(mp4)
        for prefix, flist in scene_files.items():
            if len(flist) > 1:
                flist.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for stale_mp4 in flist[1:]:
                    if _is_protected_path(stale_mp4, root, kept_set):
                        continue
                    run_delete_targets.append({
                        "run_date": kept_run.name,
                        "path": str(stale_mp4),
                        "kind": "file",
                        "size_bytes": stale_mp4.stat().st_size,
                    })

    cache_mgr = MediaCacheManager(cache_dir=cache_dir)
    cache_preview = cache_mgr.prune_cache(
        runs_dir=runs_path,
        max_gb=max_gb,
        keep_days=keep_days,
        dry_run=True,
    )

    total_run_bytes = sum(t["size_bytes"] for t in run_delete_targets)
    return {
        "keep_runs": retention,
        "kept_runs": [d.name for d in kept_dirs],
        "older_runs": [d.name for d in older_dirs],
        "run_targets": run_delete_targets,
        "run_bytes": total_run_bytes,
        "cache_dir": str(cache_dir),
        "cache_preview": cache_preview,
        "runs_dir": str(runs_path),
        "max_gb": max_gb,
        "keep_days": keep_days,
    }


def execute_cleanup(
    plan: dict[str, Any],
    dry_run: bool = False,
    repo_root: Path | str = ".",
) -> dict[str, Any]:
    """Deletes planned targets (unless dry_run=True) and prunes media cache."""
    root = Path(repo_root).resolve()
    runs_dir = Path(plan["runs_dir"])
    kept_set = {(runs_dir / name).resolve() for name in plan.get("kept_runs", [])}

    deleted_paths: list[str] = []
    freed_bytes = 0

    if not dry_run:
        for item in plan.get("run_targets", []):
            target = Path(item["path"])
            if _is_protected_path(target, root, kept_set):
                continue
            if not target.exists():
                continue
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted_paths.append(str(target))
            freed_bytes += int(item.get("size_bytes", 0))

        cache_mgr = MediaCacheManager(cache_dir=plan["cache_dir"])
        cache_result = cache_mgr.prune_cache(
            runs_dir=runs_dir,
            max_gb=float(plan.get("max_gb", 5.0)),
            keep_days=int(plan.get("keep_days", 7)),
            dry_run=False,
        )
    else:
        cache_result = plan["cache_preview"]

    return {
        "dry_run": dry_run,
        "kept_runs": plan.get("kept_runs", []),
        "older_runs": plan.get("older_runs", []),
        "deleted_paths": deleted_paths if not dry_run else [t["path"] for t in plan.get("run_targets", [])],
        "freed_run_mb": round((freed_bytes if not dry_run else plan.get("run_bytes", 0)) / (1024 * 1024), 2),
        "cache_result": cache_result,
    }


def format_cleanup_plan(plan: dict[str, Any]) -> str:
    """Formats cleanup preview showing exactly what will be deleted first."""
    lines = [
        f"Cleanup Retention Window: keep last {plan['keep_runs']} dated run(s)",
        f"Kept runs in full ({len(plan['kept_runs'])}): {', '.join(plan['kept_runs']) or '(none)'}",
        f"Older runs eligible ({len(plan['older_runs'])}): {', '.join(plan['older_runs']) or '(none)'}",
        "",
        "Run artifacts to delete:",
    ]
    if not plan["run_targets"]:
        lines.append("  (none)")
    else:
        for t in plan["run_targets"]:
            mb = t["size_bytes"] / (1024 * 1024)
            lines.append(f"  - [{t['kind']}] {t['path']} ({mb:.2f} MB)")

    cprev = plan["cache_preview"]
    lines.extend([
        "",
        f"Media cache prune preview ({plan['cache_dir']}): {cprev['deleted_count']} file(s), {cprev['freed_mb']} MB to free (remaining {cprev['remaining_mb']} MB)",
    ])
    for cf in cprev.get("files", []):
        lines.append(f"  - [cache] {cf}")
    return "\n".join(lines)


@dataclass
class DoctorCheck:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str
    fix_hint: str = ""


def _read_dotenv_keys(dotenv_path: Path) -> dict[str, bool]:
    """Parses .env returning ONLY presence booleans (never storing secret values)."""
    present: dict[str, bool] = {}
    if not dotenv_path.exists():
        return present
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            present[key] = bool(val)
    return present


def run_doctor(
    repo_root: Path | str = ".",
    which_fn: Callable[[str], str | None] = shutil.which,
    subprocess_run: Callable[..., Any] = subprocess.run,
    disk_usage_fn: Callable[[Path | str], Any] = shutil.disk_usage,
    http_request_fn: Callable[..., Any] | None = None,
    min_disk_gb: float = 10.0,
) -> list[DoctorCheck]:
    """Runs system readiness checks and returns structured PASS/WARN/FAIL results."""
    root = Path(repo_root).resolve()
    checks: list[DoctorCheck] = []

    # 1 & 2. ffmpeg and ffprobe
    for tool_name in ("ffmpeg", "ffprobe"):
        exe_path = which_fn(tool_name)
        if not exe_path:
            checks.append(DoctorCheck(
                name=tool_name,
                status="FAIL",
                detail=f"{tool_name} executable not found on PATH",
                fix_hint=f"Install FFmpeg (with {tool_name}) and add its bin folder to Windows PATH.",
            ))
            continue
        try:
            proc = subprocess_run([exe_path, "-version"], capture_output=True, text=True, timeout=10)
            if proc.returncode == 0:
                first_line = (proc.stdout or "").splitlines()[0] if (proc.stdout or "").splitlines() else exe_path
                checks.append(DoctorCheck(name=tool_name, status="PASS", detail=first_line))
            else:
                checks.append(DoctorCheck(
                    name=tool_name,
                    status="FAIL",
                    detail=f"{tool_name} exited with code {proc.returncode}",
                    fix_hint=f"Reinstall FFmpeg so `{tool_name} -version` succeeds.",
                ))
        except Exception as exc:
            checks.append(DoctorCheck(
                name=tool_name,
                status="FAIL",
                detail=f"Failed running {tool_name}: {exc}",
                fix_hint=f"Verify {tool_name} works from PowerShell.",
            ))

    # 3. .venv-tts python
    tts_py = root / ".venv-tts" / "Scripts" / "python.exe"
    if not tts_py.exists():
        alt_py = root / ".venv-tts" / "bin" / "python"
        if alt_py.exists():
            tts_py = alt_py

    tts_ok = False
    if not tts_py.exists():
        checks.append(DoctorCheck(
            name=".venv-tts python",
            status="FAIL",
            detail=f"Missing {tts_py}",
            fix_hint="Create `.venv-tts` with Python 3.10+ and install PyTorch/CUDA, Ming TTS, and torchaudio.",
        ))
    else:
        try:
            proc = subprocess_run([str(tts_py), "--version"], capture_output=True, text=True, timeout=15)
            ver = (proc.stdout or proc.stderr or "").strip()
            if proc.returncode == 0:
                tts_ok = True
                checks.append(DoctorCheck(name=".venv-tts python", status="PASS", detail=f"{ver} ({tts_py})"))
            else:
                checks.append(DoctorCheck(
                    name=".venv-tts python",
                    status="FAIL",
                    detail=f".venv-tts exited {proc.returncode}: {ver}",
                    fix_hint="Rebuild `.venv-tts` virtual environment.",
                ))
        except Exception as exc:
            checks.append(DoctorCheck(
                name=".venv-tts python",
                status="FAIL",
                detail=f"Error invoking .venv-tts: {exc}",
                fix_hint="Rebuild `.venv-tts` virtual environment.",
            ))

    # 4. CUDA (via .venv-tts)
    if not tts_ok:
        checks.append(DoctorCheck(
            name="CUDA",
            status="FAIL",
            detail="Skipped because .venv-tts python is unavailable",
            fix_hint="Fix `.venv-tts` first, then ensure CUDA-enabled PyTorch is installed.",
        ))
    else:
        try:
            proc = subprocess_run(
                [
                    str(tts_py),
                    "-c",
                    "import torch; avail=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if avail else ''; print(f'{int(avail)}|{torch.version.cuda}|{name}')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else ""
            parts = out.split("|")
            if proc.returncode == 0 and len(parts) >= 1 and parts[0] == "1":
                cuda_ver = parts[1] if len(parts) > 1 else ""
                dev_name = parts[2] if len(parts) > 2 else "GPU"
                checks.append(DoctorCheck(
                    name="CUDA",
                    status="PASS",
                    detail=f"Available (CUDA {cuda_ver}, device: {dev_name})",
                ))
            else:
                checks.append(DoctorCheck(
                    name="CUDA",
                    status="FAIL",
                    detail=f"torch.cuda.is_available() is False ({out or proc.stderr.strip()})",
                    fix_hint="Install NVIDIA driver and CUDA-enabled PyTorch wheel in `.venv-tts`.",
                ))
        except Exception as exc:
            checks.append(DoctorCheck(
                name="CUDA",
                status="FAIL",
                detail=f"CUDA probe error: {exc}",
                fix_hint="Verify PyTorch + CUDA inside `.venv-tts`.",
            ))

    # 5. Ming model files
    ming_dir = root / "models" / "Ming-3.0-tts-0.5b"
    ming_missing: list[str] = []
    if not ming_dir.is_dir():
        ming_missing.append("directory missing")
    else:
        if not (ming_dir / "config.json").is_file():
            ming_missing.append("config.json")
        if not (ming_dir / "campplus.onnx").is_file():
            ming_missing.append("campplus.onnx")
        if not list(ming_dir.glob("*.safetensors")):
            ming_missing.append("*.safetensors weights")
    if not ming_missing:
        checks.append(DoctorCheck(
            name="Ming model",
            status="PASS",
            detail=f"Verified files in {ming_dir}",
        ))
    else:
        checks.append(DoctorCheck(
            name="Ming model",
            status="FAIL",
            detail=f"Incomplete or missing at {ming_dir}: {', '.join(ming_missing)}",
            fix_hint="Download Ming-3.0-tts-0.5b model files (config.json, *.safetensors, campplus.onnx) into models/Ming-3.0-tts-0.5b/.",
        ))

    # 6. MMS_FA model file
    mms_model = root / "models" / "MMS_FA" / "model.pt"
    if mms_model.is_file() and mms_model.stat().st_size > 0:
        size_mb = mms_model.stat().st_size / (1024 * 1024)
        checks.append(DoctorCheck(
            name="MMS_FA model",
            status="PASS",
            detail=f"Present at {mms_model} ({size_mb:.1f} MB)",
        ))
    else:
        checks.append(DoctorCheck(
            name="MMS_FA model",
            status="FAIL",
            detail=f"Missing or empty model file: {mms_model}",
            fix_hint="Download torchaudio MMS_FA model weights to models/MMS_FA/model.pt.",
        ))

    # 7. Free disk space (warn under 10 GB)
    try:
        usage = disk_usage_fn(root)
        free_gb = float(usage.free) / (1024 ** 3)
        if free_gb < min_disk_gb:
            checks.append(DoctorCheck(
                name="Disk space",
                status="WARN",
                detail=f"{free_gb:.2f} GB free (below {min_disk_gb:.0f} GB threshold)",
                fix_hint="Free disk space or run `python run_daily.py --cleanup`.",
            ))
        else:
            checks.append(DoctorCheck(
                name="Disk space",
                status="PASS",
                detail=f"{free_gb:.2f} GB free (>= {min_disk_gb:.0f} GB)",
            ))
    except Exception as exc:
        checks.append(DoctorCheck(
            name="Disk space",
            status="FAIL",
            detail=f"Disk check failed: {exc}",
            fix_hint="Ensure project drive is accessible.",
        ))

    # 8. .env keys (ONLY print 'set' or 'missing', NEVER print values!)
    expected_keys = ["PIXABAY_API_KEY", "COVERR_API_KEY", "OPENVERSE_TOKEN", "UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY"]
    env_example = root / ".env.example"
    if env_example.exists():
        parsed_example = list(_read_dotenv_keys(env_example).keys())
        if parsed_example:
            expected_keys = parsed_example

    required_keys: set[str] = set()
    media_yaml = root / "config" / "media.yaml"
    if media_yaml.exists():
        try:
            m_cfg = yaml.safe_load(media_yaml.read_text(encoding="utf-8")) or {}
            for _src_name, src_cfg in (m_cfg.get("sources") or {}).items():
                if isinstance(src_cfg, dict) and src_cfg.get("enabled") and src_cfg.get("requires_key"):
                    kname = src_cfg.get("api_key_env") or src_cfg.get("token_env")
                    if kname:
                        required_keys.add(str(kname))
        except Exception:
            pass

    dotenv_presence = _read_dotenv_keys(root / ".env")
    key_states: dict[str, str] = {}
    missing_required: list[str] = []
    for key in expected_keys:
        is_set = bool(os.environ.get(key) or dotenv_presence.get(key, False))
        key_states[key] = "set" if is_set else "missing"
        if not is_set and key in required_keys:
            missing_required.append(key)

    key_summary = ", ".join(f"{k}={v}" for k, v in key_states.items())
    if missing_required:
        checks.append(DoctorCheck(
            name=".env keys",
            status="FAIL",
            detail=f"{key_summary} (missing required: {', '.join(missing_required)})",
            fix_hint="Copy .env.example to .env and populate required API keys.",
        ))
    elif not (root / ".env").exists() and not any(v == "set" for v in key_states.values()):
        checks.append(DoctorCheck(
            name=".env keys",
            status="WARN",
            detail=f".env file missing ({key_summary})",
            fix_hint="Create .env from .env.example for API access.",
        ))
    else:
        checks.append(DoctorCheck(
            name=".env keys",
            status="PASS",
            detail=key_summary,
        ))

    # 9. Feed reachability per enabled source
    sources_yaml = root / "config" / "sources.yaml"
    req_fn = http_request_fn or requests.request
    if not sources_yaml.exists():
        checks.append(DoctorCheck(
            name="Feed reachability",
            status="FAIL",
            detail=f"Missing {sources_yaml}",
            fix_hint="Restore config/sources.yaml.",
        ))
    else:
        try:
            s_cfg = yaml.safe_load(sources_yaml.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            s_cfg = {}
            checks.append(DoctorCheck(
                name="Feed reachability",
                status="FAIL",
                detail=f"Invalid YAML in {sources_yaml}: {exc}",
                fix_hint="Fix YAML syntax in config/sources.yaml.",
            ))
        user_agent = (s_cfg.get("settings") or {}).get("user_agent", "NepalNewsBot/0.1")
        headers = {"User-Agent": user_agent}
        enabled_sources = [
            src for src in (s_cfg.get("sources") or [])
            if src.get("enabled", True) and src.get("rss")
        ]
        if enabled_sources:
            src_statuses: list[str] = []
            unreachable: list[str] = []
            for src in enabled_sources:
                src_name = src.get("name", src["rss"])
                url = src["rss"]
                ok = False
                info = ""
                try:
                    resp = req_fn("HEAD", url, headers=headers, timeout=10, allow_redirects=True)
                    code = int(getattr(resp, "status_code", 0))
                    if code >= 400 or code == 0:
                        resp = req_fn("GET", url, headers=headers, timeout=10, stream=True, allow_redirects=True)
                        code = int(getattr(resp, "status_code", 0))
                    ok = 200 <= code < 400
                    info = f"HTTP {code}"
                    close_fn = getattr(resp, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception as exc:
                    info = str(exc)
                src_statuses.append(f"{src_name}: {'OK' if ok else 'FAIL'} ({info})")
                if not ok:
                    unreachable.append(src_name)

            if unreachable:
                checks.append(DoctorCheck(
                    name="Feed reachability",
                    status="FAIL",
                    detail="; ".join(src_statuses),
                    fix_hint=f"Check network/DNS or feed URLs for unreachable source(s): {', '.join(unreachable)}.",
                ))
            else:
                checks.append(DoctorCheck(
                    name="Feed reachability",
                    status="PASS",
                    detail="; ".join(src_statuses),
                ))

    # 10. Write permissions (data/ and logs/)
    write_errs: list[str] = []
    for dname in ("data", "logs"):
        target_dir = root / dname
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            probe_file = target_dir / f".doctor_write_probe_{os.getpid()}.tmp"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
        except Exception as exc:
            write_errs.append(f"{dname}/ ({exc})")
    if write_errs:
        checks.append(DoctorCheck(
            name="Write permissions",
            status="FAIL",
            detail=f"Cannot write to: {', '.join(write_errs)}",
            fix_hint="Check folder permissions or Windows Defender/antivirus locks on data/ and logs/.",
        ))
    else:
        checks.append(DoctorCheck(
            name="Write permissions",
            status="PASS",
            detail="data/ and logs/ are writable",
        ))

    # 11. Timezone (Asia/Kathmandu)
    try:
        tz = ZoneInfo("Asia/Kathmandu")
        now_ktm = datetime.now(tz)
        local_now = datetime.now().astimezone()
        ktm_off = now_ktm.strftime("%z")
        local_off = local_now.strftime("%z")
        local_name = local_now.tzname() or "local"
        if ktm_off != "+0545":
            checks.append(DoctorCheck(
                name="Timezone",
                status="FAIL",
                detail=f"Asia/Kathmandu offset resolved to {ktm_off} instead of +0545",
                fix_hint="Update tzdata package: pip install --upgrade tzdata",
            ))
        elif local_off != ktm_off:
            checks.append(DoctorCheck(
                name="Timezone",
                status="WARN",
                detail=f"Asia/Kathmandu={now_ktm.date().isoformat()} ({ktm_off}), OS local={local_name} ({local_off}); Task Scheduler uses local PC clock",
                fix_hint="Set Windows system timezone to (UTC+05:45) Kathmandu so 07:00 schedule matches Nepal time.",
            ))
        else:
            checks.append(DoctorCheck(
                name="Timezone",
                status="PASS",
                detail=f"Asia/Kathmandu ({ktm_off}), OS local={local_name} ({local_off}), today={now_ktm.date().isoformat()}",
            ))
    except Exception as exc:
        checks.append(DoctorCheck(
            name="Timezone",
            status="FAIL",
            detail=f"Unable to load Asia/Kathmandu timezone: {exc}",
            fix_hint="Install tzdata: .venv\\Scripts\\python.exe -m pip install tzdata",
        ))

    return checks


def format_doctor_report(checks: list[DoctorCheck]) -> str:
    """Formats doctor checks as plain-English PASS/WARN/FAIL lines with fix hints."""
    lines = ["Nepal News Bot - System Doctor", "=" * 32]
    for chk in checks:
        lines.append(f"[{chk.status}] {chk.name}: {chk.detail}")
        if chk.fix_hint and chk.status != "PASS":
            lines.append(f"       Fix: {chk.fix_hint}")

    fails = sum(1 for c in checks if c.status == "FAIL")
    warns = sum(1 for c in checks if c.status == "WARN")
    passes = sum(1 for c in checks if c.status == "PASS")
    lines.extend([
        "",
        f"Summary: {passes} PASS, {warns} WARN, {fails} FAIL",
    ])
    return "\n".join(lines)

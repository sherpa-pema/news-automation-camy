"""Editor Agent implementation for Nepal News Bot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from src.agents.base import BaseAgent
from src.utils.editor_brief import generate_editor_brief
from src.utils.editor_budget import (
    calculate_words_per_second,
    calculate_words_per_second_details,
    compute_word_budget,
    estimate_scene_duration_sec,
)
from src.utils.editor_cluster import (
    cluster_articles,
    extract_spine_numbers,
    parse_timestamp,
)
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
    load_editor_config,
    validate_script_draft,
    write_script_preview_markdown,
)
from src.utils.daily_ops import write_status_md
from src.utils.logging import setup_logger
from src.utils.manifest import (
    load_manifest,
    reconcile_manifest_status,
    save_manifest,
    update_stage_status,
)
from src.utils.office_events import emit_office_event

logger = setup_logger("editor")


class EditorAgent(BaseAgent):
    """Editor Agent handles clustering, briefing, validation, and approval of broadcast scripts."""

    def __init__(
        self,
        run_dir: Path | str,
        config: dict[str, Any] | None = None,
        editor_mode: str = "ide_agent",
        categories: list[str] | str | None = None,
    ) -> None:
        super().__init__(run_dir, config)
        self.script_dir = self.run_dir / "script"
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.editor_cfg, self.whitelist = load_editor_config()
        self.editor_mode = editor_mode
        self.categories = categories


    def _get_default_run_kind(self) -> str:
        """Reads default_run_kind from config/pipeline.yaml (defaults to 'test')."""
        cfg_path = Path("config/pipeline.yaml")
        if cfg_path.exists():
            try:
                import yaml
                data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                rk = data.get("execution", {}).get("default_run_kind")
                if rk in ("test", "production"):
                    return str(rk)
            except Exception:
                pass
        return "test"

    @property
    def name(self) -> str:
        return "editor"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/script.schema.json"

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Coordinates prepare, validate, or status reporting based on manifest state."""
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_dir.name}' is abandoned. No stage may run on it.")

        stage_meta = manifest.get("stages", {}).get("editor", {})
        status = stage_meta.get("status", "pending")
        draft_path = self.script_dir / "draft_script.json"

        if status == "pending":
            return self.prepare(manifest)

        if draft_path.exists() and status in ("prepared", "needs_revision"):
            logger.info("Draft script detected. Running deterministic validation...")
            return self.validate_draft(manifest)

        if status == "prepared":
            next_action = (
                f"Next action: Read {self.script_dir / 'editor_brief.md'}, "
                f"write draft_script.json and selection.json, then run:\n"
                f"  python -m src.agents.editor validate --date {self.run_dir.name}"
            )
            print(f"\n[PAUSED: PREPARED]\n{next_action}\n")
            return [str(self.script_dir / "editor_brief.md"), str(self.script_dir / "shortlist.json")]

        if status == "needs_revision":
            next_action = (
                f"Next action: Review errors in {self.script_dir / 'validation_report.json'}, "
                f"fix draft_script.json, then run:\n"
                f"  python -m src.agents.editor validate --date {self.run_dir.name}"
            )
            print(f"\n[PAUSED: NEEDS REVISION]\n{next_action}\n")
            return [str(self.script_dir / "validation_report.json")]

        if status == "awaiting_approval":
            next_action = (
                f"Next action: Review {self.script_dir / 'script_preview.md'} and run:\n"
                f"  python -m src.agents.editor approve --date {self.run_dir.name}"
            )
            print(f"\n[PAUSED: AWAITING APPROVAL]\n{next_action}\n")
            return [str(self.script_dir / "script.json"), str(self.script_dir / "script_preview.md")]

        if status == "done":
            return [str(self.script_dir / "script.json")]

        return self.prepare(manifest)

    def _load_prior_stories(self) -> list[dict[str, Any]]:
        """Loads aired story scenes from recent prior run directories for cross-day deduplication."""
        try:
            target_dt = date.fromisoformat(self.run_dir.name)
        except ValueError:
            return []

        dedup_cfg = self.editor_cfg.get("deduplication", {})
        lookback_days = int(dedup_cfg.get("cross_day_lookback_days", 2))
        if lookback_days <= 0:
            return []

        prior_stories: list[dict[str, Any]] = []
        runs_root = self.run_dir.parent
        for delta in range(1, lookback_days + 1):
            prev_date = (target_dt - timedelta(days=delta)).isoformat()
            prev_script_dir = runs_root / prev_date / "script"
            prev_script_path = prev_script_dir / "script.json"
            if not prev_script_path.exists():
                continue

            try:
                prev_script = json.loads(prev_script_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            approved_ts = 0.0
            prev_approval_path = prev_script_dir / "approval.json"
            if prev_approval_path.exists():
                try:
                    app_data = json.loads(prev_approval_path.read_text(encoding="utf-8"))
                    approved_ts = parse_timestamp(app_data.get("approved_at"))
                except Exception:
                    pass
            if approved_ts == 0.0:
                approved_ts = parse_timestamp(prev_script.get("validated_at")) or prev_script_path.stat().st_mtime

            # Also read selection.json if present for completeness
            prev_selection_path = prev_script_dir / "selection.json"
            selected_cluster_ids: list[str] = []
            if prev_selection_path.exists():
                try:
                    sel_data = json.loads(prev_selection_path.read_text(encoding="utf-8"))
                    selected_cluster_ids = [
                        str(item.get("cluster_id", ""))
                        for item in sel_data.get("selected_clusters", [])
                        if isinstance(item, dict)
                    ]
                except Exception:
                    pass

            for sc in prev_script.get("scenes", []):
                if sc.get("type") != "story":
                    continue
                headline = sc.get("headline", "")
                narration = sc.get("narration", "")
                prior_stories.append({
                    "run_date": prev_date,
                    "title": headline,
                    "narration": narration,
                    "source_urls": sc.get("source_urls", []),
                    "approved_ts": approved_ts,
                    "selected_cluster_ids": selected_cluster_ids,
                    "numbers": extract_spine_numbers(
                        f"{headline} {narration}",
                        exclude_years={target_dt.year},
                    ),
                    "pseudo_article": {
                        "title": headline,
                        "text_summary": narration,
                        "full_text": narration,
                    },
                })
        return prior_stories

    def prepare(self, manifest: dict[str, Any]) -> list[str]:
        """Step 1 (prepare): Clusters raw articles, calculates budget, generates shortlist and editor_brief.md."""
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_dir.name}' is abandoned. No stage may run on it.")

        logger.info(f"Running Editor Agent prepare for run {self.run_dir.name}...")
        raw_articles_path = self.run_dir / "raw" / "articles.json"
        if not raw_articles_path.exists():
            raise FileNotFoundError(f"Missing raw articles input: {raw_articles_path}")

        with open(raw_articles_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        logger.info(f"Read {len(articles)} raw articles from {raw_articles_path}.")

        # 1 & 2. Cluster/rank articles and compute speech-rate budget in parallel when budget isn't frozen
        prior_stories = self._load_prior_stories()
        budget_path = self.script_dir / "budget.json"
        budget_details: dict[str, Any] | None = None

        if budget_path.exists():
            shortlist, opinion_clusters, stale_clusters = cluster_articles(
                articles,
                self.editor_cfg,
                return_opinion=True,
                return_stale=True,
                prior_stories=prior_stories,
            )
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                cluster_fut = pool.submit(
                    cluster_articles,
                    articles,
                    self.editor_cfg,
                    True,
                    True,
                    None,
                    prior_stories,
                )
                budget_fut = pool.submit(calculate_words_per_second_details, self.run_dir.parent)
                shortlist, opinion_clusters, stale_clusters = cluster_fut.result()
                budget_details = budget_fut.result()

        # Defense-in-depth: drop off-topic clusters when the run targets specific topic categories
        raw_selected_cats = (
            self.categories
            or manifest.get("selected_categories")
            or manifest.get("stages", {}).get("scraper", {}).get("metrics", {}).get("selected_categories")
            or []
        )
        if isinstance(raw_selected_cats, str):
            raw_selected_cats = [c.strip() for c in raw_selected_cats.split(",") if c.strip()]
        selected_cats_norm = [str(c).strip().lower() for c in raw_selected_cats if str(c).strip()]
        if selected_cats_norm and "all" not in selected_cats_norm and "headlines" not in selected_cats_norm:
            from src.agents.scraper import CATEGORY_ALIAS_MAP

            allowed_aliases: set[str] = set()
            for sel in selected_cats_norm:
                allowed_aliases.update(CATEGORY_ALIAS_MAP.get(sel, {sel}))

            filtered_shortlist = []
            dropped_off_topic = 0
            for cl in shortlist:
                c_cat = str(cl.get("category_guess") or "other").strip().lower()
                if c_cat in allowed_aliases:
                    filtered_shortlist.append(cl)
                elif c_cat == "other":
                    # Keep uncategorized clusters if any member article matched the requested category
                    member_cats = {
                        str(a.get("category_hint") or "").strip().lower()
                        for a in cl.get("articles", [])
                    }
                    if member_cats & allowed_aliases:
                        filtered_shortlist.append(cl)
                    else:
                        dropped_off_topic += 1
                else:
                    dropped_off_topic += 1

            if dropped_off_topic > 0 and filtered_shortlist:
                logger.info(
                    f"Filtered out {dropped_off_topic} off-topic cluster(s) outside requested categories "
                    f"({', '.join(selected_cats_norm)})."
                )
                shortlist = filtered_shortlist

        logger.info(
            f"Clustering complete. Formed {len(shortlist)} shortlisted news clusters, "
            f"{len(opinion_clusters)} excluded opinion clusters, and "
            f"{len(stale_clusters)} excluded stale preview clusters."
        )

        shortlist_path = self.script_dir / "shortlist.json"
        with open(shortlist_path, "w", encoding="utf-8") as f:
            json.dump(shortlist, f, indent=2)

        # Freeze speech-rate budget (write script/budget.json once)
        forced_format = manifest.get("format") or manifest.get("mode")
        if forced_format in ("headlines", "focused"):
            bulletin_mode = forced_format
        elif selected_cats_norm and "all" not in selected_cats_norm and "headlines" not in selected_cats_norm:
            bulletin_mode = "focused"
        elif len(shortlist) < 10:
            bulletin_mode = "focused"
        else:
            bulletin_mode = "headlines"

        if budget_path.exists():
            with open(budget_path, "r", encoding="utf-8") as bf:
                frozen = json.load(bf)
            wps = float(frozen["words_per_second"])
            sps = float(frozen.get("syllables_per_second", 3.3))
            calib_msg = frozen.get("calculation") or f"Frozen budget: {wps:.2f} words/sec"
            saved_mode = frozen.get("format", bulletin_mode)
            budget = compute_word_budget(
                wps,
                sps=sps,
                format=saved_mode,
                cluster_count=len(shortlist),
            )
            for k in (
                "syllables_per_second",
                "comma_pause_sec",
                "period_pause_sec",
                "target_duration_sec",
                "duration_range_sec",
                "target_total_words",
                "min_total_words",
                "max_total_words",
                "tolerance_pct",
                "scene_targets",
                "format",
            ):
                if k in frozen:
                    budget[k] = frozen[k]
        else:
            details = budget_details if budget_details is not None else calculate_words_per_second_details(self.run_dir.parent)
            wps = float(details["words_per_second"])
            sps = float(details.get("syllables_per_second", 3.3))
            comma_pause = float(details.get("comma_pause_sec", 0.25))
            period_pause = float(details.get("period_pause_sec", 0.55))
            calib_msg = str(details["message"])
            budget = compute_word_budget(
                wps,
                sps=sps,
                comma_pause_sec=comma_pause,
                period_pause_sec=period_pause,
                format=bulletin_mode,
                cluster_count=len(shortlist),
            )
            frozen_payload = {
                "words_per_second": wps,
                "syllables_per_second": sps,
                "comma_pause_sec": comma_pause,
                "period_pause_sec": period_pause,
                "source": details.get("source"),
                "source_file": details.get("source_file"),
                "calculation": details.get("calculation"),
                "format": bulletin_mode,
                **{
                    k: v
                    for k, v in budget.items()
                    if k not in ("words_per_second", "syllables_per_second", "comma_pause_sec", "period_pause_sec")
                },
            }
            with open(budget_path, "w", encoding="utf-8") as bf:
                json.dump(frozen_payload, bf, indent=2)

        logger.info(f"Speech rate calibration: {calib_msg}")
        logger.info(
            f"Word budget ({budget.get('format', bulletin_mode)}): {budget['target_total_words']} words "
            f"({budget['min_total_words']} - {budget['max_total_words']}) for ~{budget['target_duration_sec']}s audio."
        )

        run_kind = manifest.get("run_kind") or self._get_default_run_kind()
        manifest["run_kind"] = run_kind

        # 3. Write editor_brief.md
        brief_content = generate_editor_brief(
            run_date=self.run_dir.name,
            budget=budget,
            calib_msg=calib_msg,
            shortlist=shortlist,
            opinion_clusters=opinion_clusters,
            stale_preview_clusters=stale_clusters,
            run_kind=run_kind,
            mode=budget.get("format", bulletin_mode),
            selected_categories=selected_cats_norm,
            custom_templates=self.editor_cfg.get("intro_templates"),
        )
        brief_path = self.script_dir / "editor_brief.md"
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_content)

        output_files = ["script/shortlist.json", "script/editor_brief.md", "script/budget.json"]
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            manifest["run_kind"] = run_kind
            if selected_cats_norm and not manifest.get("selected_categories"):
                manifest["selected_categories"] = selected_cats_norm
            update_stage_status(
                manifest,
                "editor",
                "prepared",
                output_files=output_files,
                metrics={
                    "total_raw_articles": len(articles),
                    "shortlisted_clusters": len(shortlist),
                    "excluded_opinion_clusters": len(opinion_clusters),
                    "excluded_stale_preview_clusters": len(stale_clusters),
                    "words_per_second": wps,
                    "target_total_words": budget["target_total_words"],
                    "word_budget_range": [budget["min_total_words"], budget["max_total_words"]],
                },
            )
            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, manifest_path)
            write_status_md(self.run_dir, manifest)

        emit_office_event(
            self.run_dir,
            "packet_handoff",
            stage="editor",
            from_station="editor",
            to_station="boss",
            artifact="script/editor_brief.md",
            label="editor_brief.md (Needs Writer)",
        )
        emit_office_event(
            self.run_dir,
            "boss_alert",
            stage="editor",
            status="prepared",
            message="Editor brief ready — draft script needed.",
        )

        next_action = (
            f"Next action: Read {brief_path}, write draft_script.json and selection.json, then run:\n"
            f"  python -m src.agents.editor validate --date {self.run_dir.name}"
        )
        print(f"\n[OK] Prepared shortlist and editor brief.\n{next_action}\n")
        return [str(shortlist_path), str(brief_path), str(budget_path)]

    def validate_draft(self, manifest: dict[str, Any], auto_approve: bool | None = None) -> list[str]:
        """Step 2 (validate): Validates draft_script.json. On pass, writes script.json and script_preview.md."""
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_dir.name}' is abandoned. No stage may run on it.")

        logger.info(f"Validating draft script for run {self.run_dir.name}...")
        draft_path = self.script_dir / "draft_script.json"
        shortlist_path = self.script_dir / "shortlist.json"
        report_path = self.script_dir / "validation_report.json"

        budget_path = self.script_dir / "budget.json"
        if budget_path.exists():
            with open(budget_path, "r", encoding="utf-8") as bf:
                frozen = json.load(bf)
            wps = float(frozen["words_per_second"])
            sps = float(frozen.get("syllables_per_second", 3.3))
            budget = compute_word_budget(wps, sps=sps)
            for k in (
                "syllables_per_second",
                "comma_pause_sec",
                "period_pause_sec",
                "target_duration_sec",
                "duration_range_sec",
                "target_total_words",
                "min_total_words",
                "max_total_words",
                "tolerance_pct",
                "scene_targets",
            ):
                if k in frozen:
                    budget[k] = frozen[k]
        else:
            details = calculate_words_per_second_details(self.run_dir.parent)
            wps = float(details["words_per_second"])
            budget = compute_word_budget(
                wps,
                sps=float(details.get("syllables_per_second", 3.3)),
                comma_pause_sec=float(details.get("comma_pause_sec", 0.25)),
                period_pause_sec=float(details.get("period_pause_sec", 0.55)),
            )

        is_valid, errors, warnings, report = validate_script_draft(
            draft_path=draft_path,
            shortlist_path=shortlist_path,
            budget=budget,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        manifest_path = self.run_dir / "manifest.json"

        if not is_valid:
            logger.warning(f"Validation FAILED with {len(errors)} errors and {len(warnings)} warnings.")

            # Move script.json and script_preview.md into script/_stale/
            stale_dir = self.script_dir / "_stale"
            stale_dir.mkdir(parents=True, exist_ok=True)

            curr_script = self.script_dir / "script.json"
            if curr_script.exists():
                shutil.move(str(curr_script), str(stale_dir / "script.json"))

            curr_preview = self.script_dir / "script_preview.md"
            if curr_preview.exists():
                shutil.move(str(curr_preview), str(stale_dir / "script_preview.md"))

            # Write script_preview.md as "VALIDATION FAILED" with the errors
            failed_preview = self.script_dir / "script_preview.md"
            failed_lines = [
                "# VALIDATION FAILED",
                "",
                f"Validation failed with {len(errors)} error(s):",
            ]
            for err in errors:
                failed_lines.append(f"- [FAIL] {err}")
            if warnings:
                failed_lines.append("")
                failed_lines.append("## Warnings:")
                for w in warnings:
                    failed_lines.append(f"- [WARN] {w}")
            failed_preview.write_text("\n".join(failed_lines), encoding="utf-8")

            output_files = ["script/validation_report.json", "script/script_preview.md"]
            if manifest_path.exists():
                update_stage_status(
                    manifest,
                    "editor",
                    "needs_revision",
                    output_files=output_files,
                    error_message=f"Validation failed: {len(errors)} errors found.",
                )
                reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
                save_manifest(manifest, manifest_path)
                write_status_md(self.run_dir, manifest)

            print(f"\n[VALIDATION FAILED] {len(errors)} error(s) must be resolved:")
            for err in errors:
                print(f"  [FAIL] {err}")
            next_action = (
                f"Next action: Fix errors in {draft_path} and re-run:\n"
                f"  python -m src.agents.editor validate --date {self.run_dir.name}"
            )
            print(f"\n{next_action}\n")
            return [str(report_path), str(failed_preview)]

        # On success: write script.json and script_preview.md
        with open(draft_path, "r", encoding="utf-8") as f:
            draft_data = json.load(f)

        draft_sha256 = hashlib.sha256(draft_path.read_bytes()).hexdigest()
        validated_at = datetime.now(timezone.utc).isoformat()
        total_words = report["metrics"]["total_words"]

        # Enrich scenes with syllable + pause blended est_duration_sec
        for sc in draft_data.get("scenes", []):
            sc["est_duration_sec"] = estimate_scene_duration_sec(sc.get("narration", ""), budget)
        total_duration = round(sum(float(sc.get("est_duration_sec", 0.0)) for sc in draft_data.get("scenes", [])), 1)
        report["warnings"] = warnings

        effective_run_kind = draft_data.get("run_kind") or manifest.get("run_kind") or self._get_default_run_kind()
        final_script = {
            "language": draft_data.get("language", "English"),
            "origin": "editor",
            "run_kind": effective_run_kind,
            "editor_mode": self.editor_mode,
            "draft_sha256": draft_sha256,
            "validator_fingerprint": report.get("validator_fingerprint"),
            "validated_at": validated_at,
            "date": self.run_dir.name,
            "total_estimated_duration_sec": total_duration,
            "warnings": warnings,
            "scenes": draft_data.get("scenes", []),
        }

        script_path = self.script_dir / "script.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(final_script, f, indent=2)

        # Also write preview
        preview_path = self.script_dir / "script_preview.md"
        write_script_preview_markdown(final_script, report, preview_path)

        # Ensure run_kind appears in script_preview.md
        preview_content = preview_path.read_text(encoding="utf-8")
        if "- **Run Kind**:" not in preview_content:
            target_str = f"- **Origin**: `{final_script.get('origin', 'editor')}` (Mode: `{final_script.get('editor_mode', 'ide_agent')}`)"
            replacement_str = f"{target_str}\n- **Run Kind**: `{final_script.get('run_kind', 'test')}`"
            preview_content = preview_content.replace(target_str, replacement_str)
            preview_path.write_text(preview_content, encoding="utf-8")

        output_files = ["script/script.json", "script/script_preview.md", "script/validation_report.json"]
        if manifest_path.exists():
            manifest["run_kind"] = final_script.get("run_kind", "test")
            update_stage_status(
                manifest,
                "editor",
                "awaiting_approval",
                output_files=output_files,
                error_message=None,
                metrics={
                    "total_words": total_words,
                    "estimated_duration_sec": total_duration,
                    "warnings_count": len(warnings),
                    "draft_sha256": draft_sha256,
                },
            )
            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, manifest_path)
            write_status_md(self.run_dir, manifest)

        emit_office_event(
            self.run_dir,
            "packet_handoff",
            stage="editor",
            from_station="editor",
            to_station="boss",
            artifact="script/script_preview.md",
            label="script_preview.md (Sign-Off)",
        )
        emit_office_event(
            self.run_dir,
            "boss_alert",
            stage="editor",
            status="awaiting_approval",
            message="Script validated — awaiting Boss approval.",
        )

        logger.info(f"Validation PASSED. Wrote {script_path} and {preview_path}.")

        should_auto_approve = (
            auto_approve is True
            or (
                auto_approve is None
                and effective_run_kind == "test"
                and bool(self.editor_cfg.get("auto_approve_test_runs", False))
            )
        )
        if should_auto_approve:
            logger.info("Auto-approving validated draft (auto_approve enabled).")
            self.approve(manifest)
            return [str(script_path), str(preview_path), str(report_path), str(self.script_dir / "approval.json")]

        next_action = (
            f"Next action: Review {preview_path} and run:\n"
            f"  python -m src.agents.editor approve --date {self.run_dir.name}"
        )
        print(f"\n[VALIDATION PASSED] Status: awaiting_approval.\n{next_action}\n")
        return [str(script_path), str(preview_path), str(report_path)]

    def approve(self, manifest: dict[str, Any]) -> list[str]:
        """Step 3 (approve): Writes approval.json and sets manifest status to done."""
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_dir.name}' is abandoned. No stage may run on it.")

        logger.info(f"Approving script for run {self.run_dir.name}...")
        script_path = self.script_dir / "script.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing validated script.json at {script_path}. Run validate first.")

        report_path = self.script_dir / "validation_report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing validation_report.json at {report_path}. Run validate first.")

        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        if not report_data.get("is_valid", False) or len(report_data.get("errors", [])) > 0:
            raise ValueError(
                f"Cannot approve script: latest validation failed with {len(report_data.get('errors', []))} error(s)."
            )

        stage_status = manifest.get("stages", {}).get("editor", {}).get("status")
        if stage_status == "needs_revision":
            raise ValueError("Cannot approve script: latest validation failed and stage is in 'needs_revision' state.")

        recorded_fp = extract_script_validation_fingerprint(report_data.get("validator_fingerprint"))
        current_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
        if recorded_fp != current_fp:
            raise ValueError(
                f"Validator fingerprint mismatch! The editor validator or config has changed since validation.\n"
                f"Recorded: {recorded_fp}\n"
                f"Current:  {current_fp}\n"
                f"Re-run validate before approving."
            )

        # Verify draft_sha256 against current draft_script.json if it exists
        draft_script_path = self.script_dir / "draft_script.json"
        if draft_script_path.exists():
            current_draft_sha = hashlib.sha256(draft_script_path.read_bytes()).hexdigest()
            with open(script_path, "r", encoding="utf-8") as sf:
                script_data = json.load(sf)
            recorded_draft_sha = script_data.get("draft_sha256")
            if recorded_draft_sha and recorded_draft_sha != current_draft_sha:
                raise ValueError(
                    f"draft_script.json was modified after script.json was generated.\n"
                    f"Recorded draft_sha256: {recorded_draft_sha}\n"
                    f"Current draft_sha256:  {current_draft_sha}\n"
                    f"Re-run validate before approving."
                )

        script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
        approval_data = {
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "script_sha256": script_sha256,
            "approved_by": "user",
            "validator_fingerprint": current_fp,
        }

        approval_path = self.script_dir / "approval.json"
        with open(approval_path, "w", encoding="utf-8") as f:
            json.dump(approval_data, f, indent=2)

        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            update_stage_status(
                manifest,
                "editor",
                "done",
                output_files=["script/script.json", "script/approval.json"],
                metrics={"approved_script_sha256": script_sha256},
            )
            reconcile_manifest_status(manifest, run_dir=self.run_dir, active_process=False)
            save_manifest(manifest, manifest_path)
            write_status_md(self.run_dir, manifest)

        logger.info(f"Script approved successfully. Hash: {script_sha256}")
        emit_office_event(
            self.run_dir,
            "packet_handoff",
            stage="editor",
            from_station="boss",
            to_station="editor",
            artifact="script/approval.json",
            label="approval.json (SIGNED)",
        )
        emit_office_event(
            self.run_dir,
            "stage_status",
            stage="editor",
            status="done",
            message="Boss approved script.",
        )
        print(f"\n[OK] Script approved. Stage 'editor' marked as done.")
        return [str(script_path), str(approval_path)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Editor Agent CLI.")
    parser.add_argument("action", choices=["prepare", "validate", "approve", "write"], help="Action to execute")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date (YYYY-MM-DD)")
    parser.add_argument("--provider", default=None, help="Optional LLM provider (gemini, anthropic, openai, ollama)")
    parser.add_argument("--categories", default=None, help="Optional comma-separated categories (e.g. sports, headlines)")
    parser.add_argument("--auto-approve", action="store_true", help="Automatically approve validated draft if validation passes cleanly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path("data/runs") / args.date
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return 1

    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
    agent = EditorAgent(run_dir, categories=args.categories)

    if args.action == "prepare":
        agent.prepare(manifest)
        return 0
    elif args.action == "write":
        from src.utils.llm_writer import generate_and_validate_draft

        res = generate_and_validate_draft(run_dir, provider=args.provider, categories=args.categories)
        if res.get("ok") and args.auto_approve:
            manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
            agent.approve(manifest)
            res["auto_approved"] = True
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    elif args.action == "validate":
        agent.validate_draft(manifest, auto_approve=True if args.auto_approve else None)
        stage_status = manifest.get("stages", {}).get("editor", {}).get("status")
        return 0 if stage_status in ("awaiting_approval", "done") else 1
    elif args.action == "approve":
        agent.approve(manifest)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

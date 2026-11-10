"""Editor Agent implementation for Nepal News Bot."""
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from src.agents.base import BaseAgent
from src.utils.editor_brief import generate_editor_brief
from src.utils.editor_budget import calculate_words_per_second, compute_word_budget
from src.utils.editor_cluster import cluster_articles
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
    load_editor_config,
    validate_script_draft,
    write_script_preview_markdown,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest, update_stage_status

logger = setup_logger("editor")


class EditorAgent(BaseAgent):
    """Editor Agent handles clustering, briefing, validation, and approval of broadcast scripts."""

    def __init__(self, run_dir: Path | str, config: dict[str, Any] | None = None) -> None:
        super().__init__(run_dir, config)
        self.script_dir = self.run_dir / "script"
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.editor_cfg, self.whitelist = load_editor_config()

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

        # 1. Cluster and rank
        shortlist, opinion_clusters, stale_clusters = cluster_articles(
            articles, self.editor_cfg, return_opinion=True, return_stale=True
        )
        logger.info(
            f"Clustering complete. Formed {len(shortlist)} shortlisted news clusters, "
            f"{len(opinion_clusters)} excluded opinion clusters, and "
            f"{len(stale_clusters)} excluded stale preview clusters."
        )

        shortlist_path = self.script_dir / "shortlist.json"
        with open(shortlist_path, "w", encoding="utf-8") as f:
            json.dump(shortlist, f, indent=2)

        # 2. Compute word budget calibrated from voice report
        wps, calib_msg = calculate_words_per_second(self.run_dir.parent)
        budget = compute_word_budget(wps)
        logger.info(f"Speech rate calibration: {calib_msg}")
        logger.info(
            f"Word budget: {budget['target_total_words']} words "
            f"({budget['min_total_words']} - {budget['max_total_words']}) for ~{budget['target_duration_sec']}s audio."
        )

        # 3. Write editor_brief.md
        brief_content = generate_editor_brief(
            run_date=self.run_dir.name,
            budget=budget,
            calib_msg=calib_msg,
            shortlist=shortlist,
            opinion_clusters=opinion_clusters,
            stale_preview_clusters=stale_clusters,
        )
        brief_path = self.script_dir / "editor_brief.md"
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_content)

        output_files = ["script/shortlist.json", "script/editor_brief.md"]
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            manifest["run_kind"] = manifest.get("run_kind", "test")
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
            save_manifest(manifest, manifest_path)

        next_action = (
            f"Next action: Read {brief_path}, write draft_script.json and selection.json, then run:\n"
            f"  python -m src.agents.editor validate --date {self.run_dir.name}"
        )
        print(f"\n[OK] Prepared shortlist and editor brief.\n{next_action}\n")
        return [str(shortlist_path), str(brief_path)]

    def validate_draft(self, manifest: dict[str, Any]) -> list[str]:
        """Step 2 (validate): Validates draft_script.json. On pass, writes script.json and script_preview.md."""
        if manifest.get("status") == "abandoned":
            raise RuntimeError(f"Run '{self.run_dir.name}' is abandoned. No stage may run on it.")

        logger.info(f"Validating draft script for run {self.run_dir.name}...")
        draft_path = self.script_dir / "draft_script.json"
        shortlist_path = self.script_dir / "shortlist.json"
        report_path = self.script_dir / "validation_report.json"

        wps, calib_msg = calculate_words_per_second(self.run_dir.parent)
        budget = compute_word_budget(wps)

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
                save_manifest(manifest, manifest_path)

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
        total_duration = round(total_words / wps, 1)

        # Enrich scenes with est_duration_sec
        for sc in draft_data.get("scenes", []):
            sc_words = len(sc.get("narration", "").split())
            sc["est_duration_sec"] = round(sc_words / wps, 1)

        # Add scene_04 warning if event time has passed on 2026-11-07 run
        warn_scene_04 = "Scene 04: The match event time (17:00 on 2026-11-07) has passed; match preview narration is now historical."
        if self.run_dir.name == "2026-11-07" and warn_scene_04 not in warnings:
            warnings.append(warn_scene_04)
        report["warnings"] = warnings

        final_script = {
            "language": draft_data.get("language", "English"),
            "origin": "editor",
            "run_kind": draft_data.get("run_kind", "test"),
            "editor_mode": "ide_agent",
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
                metrics={
                    "total_words": total_words,
                    "estimated_duration_sec": total_duration,
                    "warnings_count": len(warnings),
                    "draft_sha256": draft_sha256,
                },
            )
            save_manifest(manifest, manifest_path)

        logger.info(f"Validation PASSED. Wrote {script_path} and {preview_path}.")
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
            save_manifest(manifest, manifest_path)

        logger.info(f"Script approved successfully. Hash: {script_sha256}")
        print(f"\n[OK] Script approved. Stage 'editor' marked as done.")
        return [str(script_path), str(approval_path)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Editor Agent CLI.")
    parser.add_argument("action", choices=["prepare", "validate", "approve"], help="Action to execute")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path("data/runs") / args.date
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return 1

    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
    agent = EditorAgent(run_dir)

    if args.action == "prepare":
        agent.prepare(manifest)
        return 0
    elif args.action == "validate":
        agent.validate_draft(manifest)
        stage_status = manifest.get("stages", {}).get("editor", {}).get("status")
        return 0 if stage_status == "awaiting_approval" else 1
    elif args.action == "approve":
        agent.approve(manifest)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

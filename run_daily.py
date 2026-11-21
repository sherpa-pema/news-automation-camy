#!/usr/bin/env python3
"""Nepal News Bot - Daily Entry Point.

Starts and manages daily automated news runs:
  --morning : scrape + editor prepare -> stops at needs writer
  --finish  : verify approval gate -> run voice..qa + publish to data/latest
  --status  : print and write STATUS.md
  --cleanup : delete temporary artifacts from runs older than keep_runs + prune cache
  --doctor  : verify tools, venv, CUDA, models, disk space, env keys, feeds, permissions, timezone
"""
import argparse
from datetime import datetime
import re
import sys

from src.orchestrator import Orchestrator, PipelinePausedException
from src.utils.daily_ops import (
    execute_cleanup,
    format_cleanup_plan,
    format_doctor_report,
    get_kathmandu_today,
    load_execution_config,
    plan_cleanup,
    render_status_markdown,
    run_doctor,
    write_status_md,
)
from src.utils.manifest import load_manifest, reconcile_manifest_status, save_manifest


def _valid_date(val: str) -> str:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        raise argparse.ArgumentTypeError(f"Invalid date format: '{val}'. Expected YYYY-MM-DD.")
    try:
        datetime.strptime(val, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid calendar date: '{val}'.") from exc
    return val


def build_parser() -> argparse.ArgumentParser:
    exec_cfg = load_execution_config()
    default_rk = exec_cfg.get("default_run_kind", "test")
    parser = argparse.ArgumentParser(description="Run daily Nepal News Bot pipeline.")
    parser.add_argument(
        "--date",
        dest="run_date",
        type=_valid_date,
        default=get_kathmandu_today(),
        help="Date of the run in YYYY-MM-DD format (default: today in Asia/Kathmandu).",
    )
    parser.add_argument(
        "--morning",
        action="store_true",
        help="Run morning prep (scraper with 5-min single retry + editor prepare) and stop at needs writer.",
    )
    parser.add_argument(
        "--prepare-media",
        action="store_true",
        help="Verify approval gate, run voice + timeline + media candidates discovery, and pause for visual picking.",
    )
    parser.add_argument(
        "--apply-picks",
        dest="apply_picks_file",
        default=None,
        help="Apply visual review picks JSON, run motion -> render -> qa, and update data/latest.",
    )
    parser.add_argument(
        "--finish",
        action="store_true",
        help="Verify editor approval gate, then run voice, timeline, media (--auto), motion, render, qa, and update data/latest.",
    )
    parser.add_argument(
        "--run-kind",
        dest="run_kind",
        choices=["test", "production"],
        default=None,
        help=f"Run kind written to manifest/script/description (config default: {default_rk}).",
    )
    parser.add_argument(
        "--orientation",
        dest="orientation",
        choices=["landscape", "vertical"],
        default=None,
        help="Export video orientation: 'landscape' (16:9, 1920x1080) or 'vertical' (9:16, 1080x1920 for TikTok/Reels).",
    )
    parser.add_argument(
        "--aspect-ratio",
        "--aspect",
        dest="aspect_ratio",
        choices=["16:9", "9:16"],
        default=None,
        help="Export video aspect ratio alias: '16:9' (landscape) or '9:16' (vertical).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status summary and write data/runs/<D>/STATUS.md.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove temporary media/audio/render caches from runs older than retention window and prune media cache.",
    )
    parser.add_argument(
        "--keep-runs",
        type=int,
        default=None,
        help="Number of recent dated runs to keep intact during --cleanup (overrides execution.keep_runs in config/pipeline.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview --cleanup deletions without modifying any files.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run system readiness diagnostics (ffmpeg, .venv-tts, CUDA, models, disk, .env, feeds, permissions, timezone).",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize the run directory and manifest without executing stages.",
    )
    parser.add_argument(
        "--only",
        dest="only_stage",
        default=None,
        help="Run only the specified pipeline stage (e.g. scraper).",
    )
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Allow downstream stages to execute on fixture/mock script data.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Anchor lookback window to 23:59:59 of target date instead of current scrape time.",
    )
    parser.add_argument(
        "--reset-stage",
        dest="reset_stage",
        default=None,
        help="Reset a failed or paused stage back to pending, reset retries, and log the action to manual_edits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-running --morning or --finish even if already prepared or finished.",
    )
    parser.add_argument(
        "--allow-robots-fallback",
        action="store_true",
        help="Explicitly allow scraper to crawl sources whose robots.txt failed to fetch/parse.",
    )
    parser.add_argument(
        "--force-source",
        dest="force_source",
        default=None,
        help="Bypass circuit breaker and force scraping a specific source by name.",
    )
    parser.add_argument(
        "--categories",
        dest="categories",
        default=None,
        help="Comma-separated categories to scrape (e.g. 'headlines,politics,health').",
    )
    parser.add_argument(
        "--automated",
        action="store_true",
        help="Run in automated mode without interactive category prompt; fallback to headlines & breaking news.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. --doctor
    if args.doctor:
        checks = run_doctor()
        print(format_doctor_report(checks))
        fails = [c for c in checks if c.status == "FAIL"]
        return 1 if fails else 0

    # 2. --cleanup [--dry-run]
    if args.cleanup:
        plan = plan_cleanup(keep_runs=args.keep_runs)
        print("=== Cleanup Preview (will be deleted) ===")
        print(format_cleanup_plan(plan))
        if args.dry_run:
            print("\n[DRY RUN] No files or cache entries were deleted.")
            return 0
        res = execute_cleanup(plan, dry_run=False)
        print(
            f"\n[OK] Cleanup finished: deleted {len(res['deleted_paths'])} run path(s) "
            f"({res['freed_run_mb']} MB), pruned {res['cache_result']['deleted_count']} cache file(s) "
            f"({res['cache_result']['freed_mb']} MB)."
        )
        return 0

    exec_cfg = load_execution_config()
    effective_run_kind = args.run_kind or exec_cfg.get("default_run_kind", "test")

    effective_orientation = args.orientation
    if not effective_orientation and args.aspect_ratio:
        effective_orientation = "vertical" if args.aspect_ratio == "9:16" else "landscape"

    orchestrator = Orchestrator(
        run_date=args.run_date,
        allow_fixture=args.allow_fixture,
        backfill=args.backfill,
        run_kind=args.run_kind,
        orientation=effective_orientation,
        allow_robots_fallback=args.allow_robots_fallback,
        force_source=args.force_source,
        categories=args.categories,
        automated=args.automated,
    )

    # 3. --status
    if args.status:
        if not orchestrator.manifest_path.exists():
            print(f"[ERROR] Run manifest not found for {args.run_date} at {orchestrator.manifest_path}")
            return 1
        manifest = load_manifest(orchestrator.manifest_path)
        prev_status = manifest.get("status")
        new_status = reconcile_manifest_status(manifest, run_dir=orchestrator.run_dir)
        if new_status != prev_status and prev_status != "abandoned":
            save_manifest(manifest, orchestrator.manifest_path)
        status_path = write_status_md(orchestrator.run_dir, manifest)
        md_text = render_status_markdown(orchestrator.run_dir, manifest)
        print(md_text)
        print(f"[OK] Wrote status report: {status_path}")
        return 0

    print("==================================================")
    print(" Nepal News Bot - Multi-Agent Automated Pipeline")
    print(f" Target Run Date: {args.run_date}")
    if args.morning:
        print(f" Mode:            --morning (run_kind={effective_run_kind})")
    elif args.prepare_media:
        print(" Mode:            --prepare-media (voice + timeline + picker candidates)")
    elif args.apply_picks_file:
        print(f" Mode:            --apply-picks ({args.apply_picks_file})")
    elif args.finish:
        print(" Mode:            --finish (media=auto + publish latest)")
    if effective_orientation:
        res_label = "1080x1920 (9:16 Vertical)" if effective_orientation == "vertical" else "1920x1080 (16:9 Landscape)"
        print(f" Orientation:     {effective_orientation} [{res_label}]")
    if args.only_stage:
        print(f" Stage Filter:    {args.only_stage}")
    if args.allow_fixture:
        print(" Provenance:      --allow-fixture (ALLOWING FIXTURE DATA)")
    if args.backfill:
        print(" Window Anchor:   --backfill (END-OF-DATE)")
    if args.categories:
        print(f" Categories:      {args.categories}")
    elif args.automated:
        print(" Categories:      headlines (Automated Fallback)")
    print("==================================================")

    if args.init_only:
        manifest = orchestrator.initialize_run()
        write_status_md(orchestrator.run_dir, manifest)
        print(f"[OK] Initialized run folder and manifest at: {orchestrator.run_dir}")
        print(f"Run ID: {manifest['run_id']} | Run Kind: {manifest.get('run_kind')} | Status: {manifest['status']}")
        return 0

    if args.reset_stage:
        orchestrator.reset_stage(
            args.reset_stage,
            reason=f"CLI reset via python run_daily.py --date {args.run_date} --reset-stage {args.reset_stage}",
        )
        print(f"[OK] Successfully reset stage '{args.reset_stage}' to pending for run {args.run_date}.")
        print(f"Logged action to {orchestrator.manifest_path} manual_edits.")
        return 0

    # 4. --morning
    if args.morning:
        res = orchestrator.run_morning(run_kind=effective_run_kind, force=args.force)
        if not res.get("ok"):
            print(f"\n[FAILED] Morning run failed for {args.run_date}: {res.get('error')}")
            print(f"Status report written to: {res.get('status_path')}")
            return 1

        if res.get("already_done"):
            print("\n[MORNING ALREADY PREPARED - SKIPPED (pass --force to re-scrape)]")
            print(f"Date:           {args.run_date}")
            print(f"Run Kind:       {res['run_kind']}")
            print(f"Overall Status: {res['status']} (editor: {res['editor_status']})")
            print(f"Total Articles: {res['total_articles']} | Shortlist: {res['shortlist_count']} clusters")
            print(f"Status Report:  {res['status_path']}")
            print(f"\nNext action: {res['next_action']}")
            return 0

        print("\n[MORNING COMPLETE - NEEDS WRITER]")
        print(f"Date:           {args.run_date}")
        print(f"Run Kind:       {res['run_kind']}")
        print(f"Stopped At:     {res['stopped_at']}")
        print(f"Total Articles: {res['total_articles']}")
        if res.get("per_source_stats"):
            print("Per-Source Breakdown:")
            for src, st in res["per_source_stats"].items():
                kept = st.get("extracted_ok", 0)
                found = st.get("found", kept)
                print(f"  - {src}: {kept} kept (found: {found})")
        print(f"Shortlist:      {res['shortlist_count']} clusters")
        print(f"Editor Brief:   {res['brief_path']}")
        print(f"Status Report:  {res['status_path']}")
        print(
            "\nNext action: Have Antigravity read editor_brief.md, write draft_script.json "
            f"and selection.json, run validate, then review script_preview.md and run:\n"
            f"  python -m src.agents.editor approve --date {args.run_date}\n"
            f"  python run_daily.py --finish --date {args.run_date}"
        )
        return 0

    # 5a. --prepare-media (two-step pre-render visual picker pause)
    if args.prepare_media:
        res = orchestrator.run_prepare_media(force=args.force)
        if res.get("refused"):
            print(f"\n[REFUSED] Cannot run --prepare-media for {args.run_date}:")
            print(f"  {res.get('reason')}")
            return 1
        if not res.get("ok"):
            print(f"\n[FAILED] --prepare-media stopped at stage '{res.get('failed_stage')}': {res.get('reason')}")
            print(f"Status report: {res.get('status_path')}")
            return 1

        print("\n[MEDIA CANDIDATES READY FOR VISUAL PICKER]")
        print(f"Date:           {args.run_date}")
        print(f"Total Shots:    {res.get('total_shots')} (auto-matched clips: {res.get('auto_clips_count')}, cards: {res.get('auto_cards_count')})")
        print(f"Picker Page:    {res.get('picker_html')}")
        print(f"Status Report:  {res.get('status_path')}")
        print(f"\nNext action: {res.get('next_action')}")
        return 0

    # 5b. --apply-picks (apply user visual choices -> render -> qa -> publish latest)
    if args.apply_picks_file:
        res = orchestrator.run_apply_picks_and_finish(args.apply_picks_file)
        if res.get("refused"):
            print(f"\n[REFUSED] Cannot apply picks for {args.run_date}:")
            print(f"  {res.get('reason')}")
            return 1
        if not res.get("ok"):
            print(f"\n[FAILED] --apply-picks stopped at stage '{res.get('failed_stage')}': {res.get('reason')}")
            print(f"Status report: {res.get('status_path')}")
            return 1

        print("\n[APPLY PICKS & PUBLISH COMPLETE]")
        print(f"Date:          {args.run_date}")
        print(f"Run Kind:      {res['run_kind']}")
        print(f"Final Video:   {res['final_mp4']}")
        print(f"Description:   {res['description_md']}")
        print(f"Storyboard:    {res['storyboard_path']}")
        print(f"Status Report: {res['status_path']}")
        print(f"Latest Pack:   {res['latest']['latest_dir']} ({', '.join(res['latest']['copied'])})")
        if res["latest"].get("mirror_dir"):
            print(f"Mirror Pack:   {res['latest']['mirror_dir']} ({', '.join(res['latest'].get('mirrored', []))})")
        return 0

    # 5. --finish
    if args.finish:
        res = orchestrator.run_finish(force=args.force)
        if res.get("refused"):
            print(f"\n[REFUSED] Cannot run --finish for {args.run_date}:")
            print(f"  {res.get('reason')}")
            return 1
        if not res.get("ok"):
            print(f"\n[FAILED] --finish stopped at stage '{res.get('failed_stage')}': {res.get('reason')}")
            print(f"Status report: {res.get('status_path')}")
            return 1

        if res.get("already_finished"):
            print("\n[FINISH ALREADY COMPLETE - SKIPPED (pass --force to re-run)]")
            print(f"Date:           {args.run_date}")
            print(f"Run Kind:       {res['run_kind']}")
            print(f"Overall Status: {res['status']}")
            print(f"Final Video:    {res['final_mp4']}")
            print(f"Description:    {res['description_md']}")
            print(f"Storyboard:     {res['storyboard_path']}")
            print(f"Status Report:  {res['status_path']}")
            print(f"\nNext action: {res['next_action']}")
            return 0

        print("\n[FINISH COMPLETE]")
        print(f"Date:          {args.run_date}")
        print(f"Run Kind:      {res['run_kind']}")
        print(f"Final Video:   {res['final_mp4']}")
        print(f"Description:   {res['description_md']}")
        print(f"Storyboard:    {res['storyboard_path']}")
        print(f"Status Report: {res['status_path']}")
        print(f"Latest Pack:   {res['latest']['latest_dir']} ({', '.join(res['latest']['copied'])})")
        if res["latest"].get("mirror_dir"):
            print(f"Mirror Pack:   {res['latest']['mirror_dir']} ({', '.join(res['latest'].get('mirrored', []))})")
        return 0

    # 6. Standard orchestrator run (--only or full DAG)
    try:
        success = orchestrator.run_pipeline(only_stage=args.only_stage)
        return 0 if success else 1
    except PipelinePausedException as e:
        print(f"\n[PIPELINE PAUSED: {e.status.upper()}]")
        print(f"{e.next_action}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())

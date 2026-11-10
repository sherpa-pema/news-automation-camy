#!/usr/bin/env python3
"""Nepal News Bot - Daily Entry Point.

Starts the orchestrator for today's automated news run.
"""
import argparse
import sys
from datetime import date
from src.orchestrator import Orchestrator, PipelinePausedException


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily Nepal News Bot pipeline.")
    parser.add_argument(
        "--date",
        dest="run_date",
        default=date.today().isoformat(),
        help="Date of the run in YYYY-MM-DD format (default: today).",
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

    args = parser.parse_args()

    print(f"==================================================")
    print(f" Nepal News Bot - Multi-Agent Automated Pipeline")
    print(f" Target Run Date: {args.run_date}")
    if args.only_stage:
        print(f" Stage Filter:    {args.only_stage}")
    if args.allow_fixture:
        print(f" Provenance:      --allow-fixture (ALLOWING FIXTURE DATA)")
    if args.backfill:
        print(f" Window Anchor:   --backfill (END-OF-DATE)")
    print(f"==================================================")

    orchestrator = Orchestrator(
        run_date=args.run_date,
        allow_fixture=args.allow_fixture,
        backfill=args.backfill,
    )

    if args.init_only:
        manifest = orchestrator.initialize_run()
        print(f"[OK] Initialized run folder and manifest at: {orchestrator.run_dir}")
        print(f"Run ID: {manifest['run_id']} | Status: {manifest['status']}")
        return 0

    if args.reset_stage:
        orchestrator.reset_stage(
            args.reset_stage,
            reason=f"CLI reset via python run_daily.py --date {args.run_date} --reset-stage {args.reset_stage}",
        )
        print(f"[OK] Successfully reset stage '{args.reset_stage}' to pending for run {args.run_date}.")
        print(f"Logged action to {orchestrator.manifest_path} manual_edits.")
        return 0

    try:
        success = orchestrator.run_pipeline(only_stage=args.only_stage)
        return 0 if success else 1
    except PipelinePausedException as e:
        print(f"\n[PIPELINE PAUSED: {e.status.upper()}]")
        print(f"{e.next_action}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())

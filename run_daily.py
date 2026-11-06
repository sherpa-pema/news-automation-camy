#!/usr/bin/env python3
"""Nepal News Bot - Daily Entry Point.

Starts the orchestrator for today's automated news run.
"""
import argparse
import sys
from datetime import date
from src.orchestrator import Orchestrator


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

    args = parser.parse_args()

    print(f"==================================================")
    print(f" Nepal News Bot - Multi-Agent Automated Pipeline")
    print(f" Target Run Date: {args.run_date}")
    if args.only_stage:
        print(f" Stage Filter:    {args.only_stage}")
    print(f"==================================================")

    orchestrator = Orchestrator(run_date=args.run_date)

    if args.init_only:
        manifest = orchestrator.initialize_run()
        print(f"[OK] Initialized run folder and manifest at: {orchestrator.run_dir}")
        print(f"Run ID: {manifest['run_id']} | Status: {manifest['status']}")
        return 0

    success = orchestrator.run_pipeline(only_stage=args.only_stage)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

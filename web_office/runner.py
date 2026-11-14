"""Subprocess job runner for Nepal News Bot 2D Office actions."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

from src.utils.manifest import is_run_process_active


@dataclass
class ActiveJob:
    job_id: str
    run_date: str
    action: str
    command: list[str]
    started_at: str
    status: str = "running"
    returncode: int | None = None
    output_tail: list[str] = field(default_factory=list)


class OfficeJobManager:
    """Runs safe CLI subprocesses in the repo root using the active Python interpreter."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.python_exe = self._resolve_python()
        self.jobs: dict[str, ActiveJob] = {}
        self._lock = asyncio.Lock()

    def _resolve_python(self) -> str:
        venv_py = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable

    def is_busy(self, run_date: str) -> tuple[bool, str | None]:
        run_dir = self.repo_root / "data" / "runs" / run_date
        if is_run_process_active(run_dir):
            return True, f"Pipeline process lock (.pipeline.pid) is active for {run_date}."
        for job in self.jobs.values():
            if job.run_date == run_date and job.status == "running":
                return True, f"Office job '{job.action}' is currently running for {run_date}."
        return False, None

    async def start_job(self, run_date: str, action: str, args: list[str]) -> ActiveJob:
        async with self._lock:
            busy, reason = self.is_busy(run_date)
            if busy:
                raise RuntimeError(reason or "Pipeline is busy")

            cmd = [self.python_exe, *args]
            job_id = f"{run_date}:{action}:{int(datetime.now().timestamp())}"
            job = ActiveJob(
                job_id=job_id,
                run_date=run_date,
                action=action,
                command=cmd,
                started_at=datetime.now().astimezone().isoformat(),
            )
            self.jobs[job_id] = job

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )

            asyncio.create_task(self._monitor_process(job, proc))
            return job

    async def _monitor_process(self, job: ActiveJob, proc: asyncio.subprocess.Process) -> None:
        try:
            if proc.stdout is not None:
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    job.output_tail.append(line)
                    if len(job.output_tail) > 200:
                        job.output_tail = job.output_tail[-200:]
            rc = await proc.wait()
            job.returncode = rc
            job.status = "done" if rc == 0 else "failed"
        except Exception as exc:
            job.output_tail.append(f"[ERROR] Job runner exception: {exc}")
            job.status = "failed"
            job.returncode = -1

    def get_active_job(self, run_date: str) -> dict[str, Any] | None:
        latest: ActiveJob | None = None
        for job in self.jobs.values():
            if job.run_date == run_date:
                latest = job
        if not latest:
            return None
        return {
            "job_id": latest.job_id,
            "action": latest.action,
            "status": latest.status,
            "returncode": latest.returncode,
            "started_at": latest.started_at,
            "output_tail": latest.output_tail[-30:],
        }

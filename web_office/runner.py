"""Subprocess job runner for Nepal News Bot 2D Office actions."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any, Callable

from src.utils.manifest import is_pid_alive, is_run_process_active


def terminate_pid_tree(pid: int) -> bool:
    """Terminates process and all descendant processes recursively."""
    if not is_pid_alive(pid):
        return True
    try:
        if sys.platform == "win32":
            import subprocess
            res = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                text=True,
            )
            import time
            for _ in range(12):
                if not is_pid_alive(pid):
                    return True
                time.sleep(0.05)
            return not is_pid_alive(pid)
        else:
            import os
            import signal
            import time
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
            for _ in range(12):
                if not is_pid_alive(pid):
                    return True
                time.sleep(0.05)
            return not is_pid_alive(pid)
    except Exception:
        return not is_pid_alive(pid)


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
    proc: Any = None


class OfficeJobManager:
    """Runs safe CLI subprocesses in the repo root using the active Python interpreter.

    Enforces a global single-tenant concurrency cap (GPU & TTS resources are single-tenant).
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.python_exe = self._resolve_python()
        self.jobs: dict[str, ActiveJob] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()
        self.output_callbacks: list[Callable[[str, str, str], Any]] = []

    def _resolve_python(self) -> str:
        venv_py = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if venv_py.exists():
            return str(venv_py)
        return sys.executable

    def check_global_concurrency(self, requesting_run_date: str) -> tuple[bool, str | None]:
        """Ensures only ONE pipeline subprocess runs globally at any time."""
        # 1. Check runner jobs
        for job in self.jobs.values():
            if job.status == "running":
                return True, (
                    f"Global concurrency cap reached: job '{job.action}' for {job.run_date} "
                    f"is currently executing on the pipeline. GPU/TTS resources are single-tenant."
                )

        # 2. Check external pipeline lock on requesting run date
        run_dir = self.repo_root / "data" / "runs" / requesting_run_date
        if is_run_process_active(run_dir):
            return True, f"Pipeline process lock (.pipeline.pid) is active for {requesting_run_date}."

        return False, None

    def is_busy(self, run_date: str) -> tuple[bool, str | None]:
        return self.check_global_concurrency(run_date)

    def register_output_callback(self, cb: Callable[[str, str, str], Any]) -> None:
        """Registers a callback (run_date, action, line) -> None for real-time stdout streaming."""
        if cb not in self.output_callbacks:
            self.output_callbacks.append(cb)

    def unregister_output_callback(self, cb: Callable[[str, str, str], Any]) -> None:
        if cb in self.output_callbacks:
            self.output_callbacks.remove(cb)

    async def start_job(self, run_date: str, action: str, args: list[str]) -> ActiveJob:
        async with self._lock:
            busy, reason = self.check_global_concurrency(run_date)
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
            job.proc = proc

            task = asyncio.create_task(self._monitor_process(job, proc))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
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
                    # Real-time stdout streaming to subscribers
                    for cb in list(self.output_callbacks):
                        try:
                            res = cb(job.run_date, job.action, line)
                            if asyncio.iscoroutine(res):
                                asyncio.create_task(res)
                        except Exception:
                            pass
            rc = await proc.wait()
            if job.status != "stopped":
                job.returncode = rc
                job.status = "done" if rc == 0 else "failed"
        except Exception as exc:
            err_line = f"[ERROR] Job runner exception: {exc}"
            job.output_tail.append(err_line)
            for cb in list(self.output_callbacks):
                try:
                    cb(job.run_date, job.action, err_line)
                except Exception:
                    pass
            if job.status != "stopped":
                job.status = "failed"
                job.returncode = -1
        finally:
            self._prune_jobs()

    def _prune_jobs(self) -> None:
        """Prunes finished jobs to retain only the latest job per run, capping total jobs."""
        latest_per_run: dict[str, ActiveJob] = {}
        for j in self.jobs.values():
            if j.status == "running":
                latest_per_run[j.job_id] = j
            else:
                existing = latest_per_run.get(j.run_date)
                if not existing or j.started_at > existing.started_at:
                    latest_per_run[j.run_date] = j

        # Keep only the retained jobs
        retained_ids = {j.job_id for j in latest_per_run.values()}
        for j_id in list(self.jobs.keys()):
            if j_id not in retained_ids:
                del self.jobs[j_id]

    async def stop_job(self, run_date: str) -> bool:
        """Terminates any active runner subprocess for the given run_date."""
        async with self._lock:
            stopped = False
            for job in self.jobs.values():
                if job.run_date == run_date and job.status == "running":
                    proc = getattr(job, "proc", None)
                    if proc and proc.returncode is None:
                        try:
                            terminate_pid_tree(proc.pid)
                        except Exception as exc:
                            job.output_tail.append(f"[WARN] Failed killing process: {exc}")
                    job.status = "stopped"
                    job.returncode = -15
                    job.output_tail.append("[STOPPED] Workflow interrupted by operator via Web Office HQ.")
                    stopped = True
            return stopped

    async def shutdown(self) -> None:
        """Gracefully shuts down all running subprocesses and monitor tasks on server shutdown."""
        async with self._lock:
            for job in list(self.jobs.values()):
                if job.status == "running":
                    proc = getattr(job, "proc", None)
                    if proc and proc.returncode is None:
                        try:
                            terminate_pid_tree(proc.pid)
                        except Exception:
                            pass
                    job.status = "stopped"
                    job.returncode = -15
            for task in list(self._tasks):
                task.cancel()

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

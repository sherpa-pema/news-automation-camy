"""Manifest operations, PID lock tracking, and stage-derived status computation."""
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

from src.utils.schema_validate import validate_contract

PAUSE_STATES = {"prepared", "needs_revision", "awaiting_approval"}
LOCK_FILENAME = ".pipeline.pid"
GLOBAL_LOCK_FILENAME = ".pipeline_global.lock"
DEFAULT_STALE_MINUTES = 30.0


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Parses an ISO-8601 timestamp string safely."""
    if not value or not isinstance(value, str):
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def is_pid_alive(pid: int) -> bool:
    """Checks whether an OS process with the given PID is currently active."""
    if not isinstance(pid, int) or pid <= 0:
        return False

    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            err = kernel32.GetLastError()
            return err == ERROR_ACCESS_DENIED
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def is_run_process_active(run_dir: Path | str | None) -> bool:
    """Returns True only if a live pipeline process holds .pipeline.pid in run_dir."""
    if run_dir is None:
        return False
    lock_path = Path(run_dir) / LOCK_FILENAME
    if not lock_path.exists():
        return False
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        if raw.startswith("{"):
            data = json.loads(raw)
            pid = int(data.get("pid", 0))
        else:
            pid = int(raw)
    except (ValueError, OSError, json.JSONDecodeError):
        return False
    return is_pid_alive(pid)


def acquire_run_lock(run_dir: Path | str) -> Path:
    """Writes .pipeline.pid for the current active process atomically."""
    lock_path = Path(run_dir) / LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "started_at": datetime.now().astimezone().isoformat(),
    }
    content = json.dumps(payload).encode("utf-8")

    if lock_path.exists():
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            existing_pid = int(data.get("pid", 0))
            if existing_pid == os.getpid():
                return lock_path
            if existing_pid > 0 and is_pid_alive(existing_pid):
                raise RuntimeError(
                    f"Pipeline process lock ({LOCK_FILENAME}) already held by active PID {existing_pid}."
                )
            try:
                lock_path.unlink()
            except OSError:
                pass
        except (ValueError, OSError, json.JSONDecodeError):
            try:
                lock_path.unlink()
            except OSError:
                pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except FileExistsError:
        raw = lock_path.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        existing_pid = int(data.get("pid", 0))
        if existing_pid > 0 and is_pid_alive(existing_pid) and existing_pid != os.getpid():
            raise RuntimeError(
                f"Pipeline process lock ({LOCK_FILENAME}) acquired by concurrent PID {existing_pid}."
            )
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
    return lock_path


def release_run_lock(run_dir: Path | str) -> None:
    """Removes .pipeline.pid if present and owned by current process or dead."""
    lock_path = Path(run_dir) / LOCK_FILENAME
    try:
        if lock_path.exists():
            raw = lock_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            owner_pid = int(data.get("pid", 0))
            if owner_pid == os.getpid() or not is_pid_alive(owner_pid):
                lock_path.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        try:
            lock_path.unlink()
        except OSError:
            pass


@contextmanager
def run_process_lock(run_dir: Path | str) -> Iterator[Path]:
    """Context manager marking run_dir active while a pipeline process executes."""
    lock_path = Path(run_dir) / LOCK_FILENAME
    owned = True
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            if int(data.get("pid", 0)) == os.getpid():
                owned = False
        except Exception:
            pass
    if owned:
        acquire_run_lock(run_dir)
    try:
        yield lock_path
    finally:
        if owned:
            release_run_lock(run_dir)


def get_global_lock_path(repo_root: Path | str | None = None) -> Path:
    if repo_root is None:
        root = Path(__file__).resolve().parent.parent.parent
    else:
        root = Path(repo_root).resolve()
    return root / "data" / GLOBAL_LOCK_FILENAME


def is_global_pipeline_active(repo_root: Path | str | None = None) -> bool:
    lock_path = get_global_lock_path(repo_root)
    if not lock_path.exists():
        return False
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data.get("pid", 0))
        return is_pid_alive(pid)
    except Exception:
        return False


def acquire_global_lock(repo_root: Path | str | None = None, run_date: str = "") -> Path:
    """Atomically acquires the single global lock for hardware / GPU resources."""
    lock_path = get_global_lock_path(repo_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "run_date": run_date,
        "started_at": datetime.now().astimezone().isoformat(),
    }
    content = json.dumps(payload).encode("utf-8")

    if lock_path.exists():
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            existing_pid = int(data.get("pid", 0))
            if existing_pid == os.getpid():
                return lock_path
            if existing_pid > 0 and is_pid_alive(existing_pid):
                locked_date = data.get("run_date", "unknown")
                raise RuntimeError(
                    f"Global pipeline lock ({GLOBAL_LOCK_FILENAME}) is held by active PID {existing_pid} (run date: {locked_date})."
                )
            try:
                lock_path.unlink()
            except OSError:
                pass
        except (ValueError, OSError, json.JSONDecodeError):
            try:
                lock_path.unlink()
            except OSError:
                pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except FileExistsError:
        raw = lock_path.read_text(encoding="utf-8").strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        existing_pid = int(data.get("pid", 0))
        if existing_pid > 0 and is_pid_alive(existing_pid) and existing_pid != os.getpid():
            locked_date = data.get("run_date", "unknown")
            raise RuntimeError(
                f"Global pipeline lock ({GLOBAL_LOCK_FILENAME}) acquired by concurrent PID {existing_pid} (run date: {locked_date})."
            )
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
    return lock_path


def release_global_lock(repo_root: Path | str | None = None) -> None:
    """Removes .pipeline_global.lock if owned by current process or dead."""
    lock_path = get_global_lock_path(repo_root)
    try:
        if lock_path.exists():
            raw = lock_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            owner_pid = int(data.get("pid", 0))
            if owner_pid == os.getpid() or not is_pid_alive(owner_pid):
                lock_path.unlink()
    except Exception:
        try:
            lock_path.unlink()
        except OSError:
            pass


@contextmanager
def global_pipeline_lock(repo_root: Path | str | None = None, run_date: str = "") -> Iterator[Path]:
    """Context manager acquiring the single-tenant global pipeline lock."""
    lock_path = get_global_lock_path(repo_root)
    owned = True
    if lock_path.exists():
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
            if int(data.get("pid", 0)) == os.getpid():
                owned = False
        except Exception:
            pass
    if owned:
        acquire_global_lock(repo_root=repo_root, run_date=run_date)
    try:
        yield lock_path
    finally:
        if owned:
            release_global_lock(repo_root=repo_root)


def compute_manifest_status(
    manifest: dict[str, Any],
    run_dir: Path | str | None = None,
    active_process: bool | None = None,
    now: datetime | None = None,
    stale_minutes: float = DEFAULT_STALE_MINUTES,
) -> str:
    """Computes top-level manifest status from stages, active PID lock, and stale detection."""
    current_top = manifest.get("status", "pending")
    if current_top == "abandoned":
        return "abandoned"

    stages = manifest.get("stages", {})
    statuses = [st.get("status", "pending") for st in stages.values()]

    if active_process is None:
        active_process = is_run_process_active(run_dir) if run_dir is not None else False

    if active_process:
        return "running"

    if statuses and all(s == "done" for s in statuses):
        return "done"

    if any(s == "failed" for s in statuses):
        return "failed"

    if any(s in PAUSE_STATES for s in statuses):
        return "paused"

    has_running_stage = any(s == "running" for s in statuses)
    if current_top == "interrupted":
        return "interrupted"

    if has_running_stage or current_top == "running":
        ref_time = _parse_iso_timestamp(manifest.get("updated_at") or manifest.get("created_at"))
        if has_running_stage:
            # Prefer running stage start_time if more specific or updated_at missing
            for st in stages.values():
                if st.get("status") == "running" and st.get("start_time"):
                    st_time = _parse_iso_timestamp(st.get("start_time"))
                    if st_time and (ref_time is None or st_time > ref_time):
                        ref_time = st_time
                    break

        curr_now = now or datetime.now().astimezone()
        if ref_time is not None:
            if ref_time.tzinfo is None and curr_now.tzinfo is not None:
                ref_time = ref_time.replace(tzinfo=curr_now.tzinfo)
            age_seconds = (curr_now - ref_time).total_seconds()
            if age_seconds >= stale_minutes * 60.0:
                return "interrupted"

    return "pending"


def reconcile_manifest_status(
    manifest: dict[str, Any],
    run_dir: Path | str | None = None,
    active_process: bool | None = None,
    now: datetime | None = None,
    stale_minutes: float = DEFAULT_STALE_MINUTES,
) -> str:
    """Updates and returns manifest['status'] derived cleanly from stages and liveness."""
    new_status = compute_manifest_status(
        manifest=manifest,
        run_dir=run_dir,
        active_process=active_process,
        now=now,
        stale_minutes=stale_minutes,
    )
    manifest["status"] = new_status
    return new_status


def create_initial_manifest(run_id: str, date_str: str, pipeline_config: dict[str, Any]) -> dict[str, Any]:
    """Generates an initial manifest structure with all pipeline stages in 'pending' status."""
    now_iso = datetime.now().astimezone().isoformat()
    stages = {}

    for stage_cfg in pipeline_config.get("stages", []):
        stage_id = stage_cfg["id"]
        stages[stage_id] = {
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "retry_count": 0,
            "max_retries": stage_cfg.get("retries", 2),
            "input_files": stage_cfg.get("inputs", []),
            "output_files": stage_cfg.get("outputs", []),
            "error_message": None,
        }

    manifest = {
        "run_id": run_id,
        "date": date_str,
        "status": "pending",
        "created_at": now_iso,
        "updated_at": now_iso,
        "stages": stages,
    }
    return manifest


def load_manifest(manifest_path: Path | str, validate: bool = True) -> dict[str, Any]:
    """Loads and returns the manifest dictionary from disk, validating against schema."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if validate:
        now_iso = datetime.now().astimezone().isoformat()
        if "created_at" not in manifest:
            manifest["created_at"] = now_iso
        if "updated_at" not in manifest:
            manifest["updated_at"] = now_iso
        schema_path = Path(__file__).resolve().parents[2] / "contracts" / "manifest.schema.json"
        if schema_path.exists():
            is_valid, err = validate_contract(manifest, schema_path)
            if not is_valid:
                raise ValueError(f"manifest.json schema validation failed: {err}")
    return manifest


def save_manifest(manifest: dict[str, Any], manifest_path: Path | str) -> None:
    """Atomically writes the manifest to disk and updates updated_at timestamp."""
    now_iso = datetime.now().astimezone().isoformat()
    manifest["updated_at"] = now_iso
    if "created_at" not in manifest:
        manifest["created_at"] = now_iso
    schema_path = Path(__file__).resolve().parents[2] / "contracts" / "manifest.schema.json"
    if schema_path.exists():
        is_valid, err = validate_contract(manifest, schema_path)
        if not is_valid:
            raise ValueError(f"manifest.json schema validation failed: {err}")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_path, path)


def update_stage_status(
    manifest: dict[str, Any],
    stage_id: str,
    status: str,
    error_message: str | None = None,
    output_files: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    source_errors: dict[str, str] | None = None,
) -> None:
    """Updates the state of a single stage within the in-memory manifest."""
    if stage_id not in manifest["stages"]:
        raise KeyError(f"Stage '{stage_id}' not found in manifest")

    stage = manifest["stages"][stage_id]
    stage["status"] = status
    now_iso = datetime.now().astimezone().isoformat()

    if status == "running":
        stage["start_time"] = now_iso
        stage["error_message"] = None
    elif status == "done":
        stage["end_time"] = now_iso
        stage["error_message"] = None
    elif status in ("awaiting_approval", "prepared"):
        stage["error_message"] = None
    elif status == "failed":
        stage["end_time"] = now_iso

    if error_message is not None:
        stage["error_message"] = error_message
    if output_files is not None:
        stage["output_files"] = output_files
    if metrics is not None:
        stage["metrics"] = metrics
    if source_errors is not None:
        stage["source_errors"] = source_errors


def record_manual_edit(
    manifest: dict[str, Any],
    file: str,
    what: str,
    why: str,
    timestamp: str | None = None,
) -> None:
    """Records a manual edit entry to the manifest's manual_edits list."""
    if "manual_edits" not in manifest:
        manifest["manual_edits"] = []
    manifest["manual_edits"].append({
        "file": file,
        "what": what,
        "why": why,
        "timestamp": timestamp or datetime.now().astimezone().isoformat(),
    })


"""FastAPI + WebSocket server for Nepal News Bot 2D Office."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal
from urllib.parse import urlparse
import uuid

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.editor import EditorAgent
from src.agents.scraper import AVAILABLE_CATEGORIES
from src.utils.daily_ops import (
    execute_cleanup,
    format_cleanup_plan,
    load_outputs_config,
    plan_cleanup,
    run_doctor,
)
from src.utils.manifest import (
    is_pid_alive,
    load_manifest,
    reconcile_manifest_status,
    record_manual_edit,
    release_run_lock,
    save_manifest,
    update_stage_status,
)
from src.utils.office_events import emit_office_event
from web_office.broadcaster import BroadcasterManager
from web_office.runner import OfficeJobManager, terminate_pid_tree
from web_office.watcher import (
    build_replay_events,
    list_runs,
    read_run_state,
    tail_new_lines,
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "runs"
LATEST_DIR = REPO_ROOT / "data" / "latest"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB


def _terminate_pid_tree(pid: int) -> bool:
    """Terminates process and all descendant processes recursively."""
    return terminate_pid_tree(pid)


def _verify_media_content(path: Path, ext: str) -> bool:
    """Verifies file header/magic bytes for uploaded images or videos."""
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with open(path, "rb") as f:
            header = f.read(32)
        if ext in (".jpg", ".jpeg"):
            return header.startswith(b"\xff\xd8\xff")
        if ext == ".png":
            return header.startswith(b"\x89PNG\r\n\x1a\n")
        if ext == ".webp":
            return header.startswith(b"RIFF") and header[8:12] == b"WEBP"
        if ext in (".mp4", ".mov"):
            return header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip")
        if ext in (".mkv", ".webm"):
            return header.startswith(b"\x1a\x45\xdf\xa3")
        if ext == ".avi":
            return header.startswith(b"RIFF") and header[8:12] == b"AVI "
        return False
    except Exception:
        return False


def _is_host_allowed(host_header: str, allowed_hosts: list[str]) -> bool:
    """Validates Host or Origin against localhost, private LAN ranges, and allowed hosts."""
    if not host_header:
        return False
    host = host_header.split(":")[0].strip().lower()
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1", "testserver", "testclient"):
        return True
    if host in allowed_hosts:
        return True
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private:
            return True
    except ValueError:
        pass
    return False


def _is_ip_private_or_loopback(host_or_ip: str) -> bool:
    """Checks whether a hostname or IP address resolves to private, loopback, or link-local ranges."""
    import socket
    clean_host = host_or_ip.split(":")[0].strip().lower()
    if not clean_host:
        return True
    if clean_host in ("localhost", "testserver", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(clean_host)
        return (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        pass
    try:
        addr_info = socket.getaddrinfo(clean_host, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_loopback
                or ip.is_private
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return True
    except Exception:
        return False
    return False


def _get_request_token(request: Request) -> str | None:
    """Extracts authentication token from query, header, or cookie."""
    token = request.query_params.get("token")
    if token:
        return token
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    x_tok = request.headers.get("x-office-token")
    if x_tok:
        return x_tok
    cookie_tok = request.cookies.get("office_token")
    if cookie_tok:
        return cookie_tok
    return None


def _is_token_valid(token: str | None, configured_token: str) -> bool:
    """Constant-time token validation against configured token."""
    if not configured_token:
        return True
    if not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), configured_token.encode("utf-8"))


def _validate_run_date(run_date: str) -> str:
    if not DATE_RE.match(run_date):
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    return run_date


def _resolve_sandboxed_file(run_date: str, rel_path: str) -> Path:
    _validate_run_date(run_date)
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path and Path(rel_path).is_absolute():
        raise HTTPException(status_code=403, detail="Absolute paths are forbidden.")

    allowed_root = (RUNS_DIR / run_date).resolve()
    candidate = (allowed_root / rel_path).resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Path traversal outside run directory is forbidden.") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {rel_path}")
    return candidate


class MorningRequest(BaseModel):
    run_kind: str = "test"
    force: bool = False
    categories: list[str] | None = None


class DraftAIRequest(BaseModel):
    provider: Literal["zhipu", "bigmodel", "gemini", "anthropic", "openai", "ollama"] | None = None
    auto_approve: bool = False


class DraftManualRequest(BaseModel):
    draft_script: dict[str, Any]
    selection: dict[str, Any] | None = None


class FinishRequest(BaseModel):
    force: bool = False
    orientation: Literal["landscape", "vertical"] | None = None


class PrepareMediaRequest(BaseModel):
    force: bool = False
    orientation: Literal["landscape", "vertical"] | None = None


class ResetStageRequest(BaseModel):
    stage: str
    reset_downstream: bool = False


class ApplyPicksRequest(BaseModel):
    picks: dict[str, Any]


class ValidateMediaLinkRequest(BaseModel):
    url: str


class CleanupRequest(BaseModel):
    dry_run: bool = True
    keep_runs: int = 7


def _compute_run_items(runs_dir: Path) -> list[dict[str, Any]]:
    run_items = []
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if d.is_dir() and DATE_RE.match(d.name):
                try:
                    sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                except Exception:
                    sz = 0
                st_md = d / "manifest.json"
                st = "pending"
                if st_md.exists():
                    try:
                        st = json.loads(st_md.read_text(encoding="utf-8")).get("status", "pending")
                    except Exception:
                        pass
                run_items.append(
                    {
                        "date": d.name,
                        "size_mb": round(sz / (1024 * 1024), 2),
                        "status": st,
                        "has_final_mp4": (d / "render" / "final.mp4").exists(),
                    }
                )
    return run_items


def create_app(repo_root: Path | None = None) -> FastAPI:
    root = (repo_root or REPO_ROOT).resolve()
    runs_dir = root / "data" / "runs"
    config_dir = root / "config"
    job_manager = OfficeJobManager(root)
    broadcaster_manager = BroadcasterManager(
        runs_dir=runs_dir,
        config_dir=config_dir,
        job_manager=job_manager,
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        yield
        await job_manager.shutdown()
        await broadcaster_manager.shutdown()

    app = FastAPI(title="Nepal News Bot 2D Office", version="1.0.0", lifespan=lifespan)
    app.state.job_manager = job_manager
    app.state.broadcaster_manager = broadcaster_manager

    # Cached Doctor runner with 30s TTL
    cached_doctor: dict[str, Any] = {"checks": [], "ts": 0.0, "ok": True}

    async def get_cached_doctor(ttl_seconds: float = 30.0) -> dict[str, Any]:
        now = asyncio.get_event_loop().time()
        if cached_doctor["checks"] and (now - cached_doctor["ts"] < ttl_seconds):
            return cached_doctor
        checks = await asyncio.to_thread(run_doctor)
        res = {
            "ok": all(c.status != "FAIL" for c in checks),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "fix_hint": c.fix_hint,
                }
                for c in checks
            ],
            "ts": now,
        }
        cached_doctor.clear()
        cached_doctor.update(res)
        return res

    @app.middleware("http")
    async def security_and_auth_middleware(request: Request, call_next):
        configured_token = os.getenv("OFFICE_TOKEN", "").strip()
        allowed_hosts = [
            h.strip().lower()
            for h in os.getenv("OFFICE_ALLOWED_HOSTS", "").split(",")
            if h.strip()
        ]

        host = request.headers.get("host", "")
        if not host or not _is_host_allowed(host, allowed_hosts):
            return JSONResponse(status_code=403, content={"detail": f"Host '{host}' is forbidden."})

        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if not parsed.netloc or not _is_host_allowed(parsed.netloc, allowed_hosts):
                return JSONResponse(status_code=403, content={"detail": f"Origin '{origin}' is forbidden."})

        # Token check on /api
        path = request.url.path
        if configured_token and path.startswith("/api"):
            req_token = _get_request_token(request)
            if not _is_token_valid(req_token, configured_token):
                return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid or missing OFFICE_TOKEN"})

        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Set cookie if valid token in query param
        token_in_query = request.query_params.get("token")
        if token_in_query and _is_token_valid(token_in_query, configured_token):
            response.set_cookie(
                key="office_token",
                value=token_in_query,
                httponly=True,
                samesite="strict",
            )

        return response

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        return {"ok": True, "repo_root": str(root)}

    @app.get("/api/runs")
    async def api_list_runs() -> dict[str, Any]:
        runs = await asyncio.to_thread(list_runs, runs_dir)
        return {"runs": runs}

    @app.get("/api/doctor")
    async def api_doctor() -> dict[str, Any]:
        return await get_cached_doctor()

    @app.get("/api/system/maintenance")
    async def api_system_maintenance() -> dict[str, Any]:
        doctor_res = await get_cached_doctor()

        disk_total, disk_used, disk_free = shutil.disk_usage(root)
        run_items = await asyncio.to_thread(_compute_run_items, runs_dir)

        outputs_cfg = load_outputs_config(config_dir)
        mirror_dir_raw = outputs_cfg.get("mirror_dir")
        mirror_status = {
            "configured_path": mirror_dir_raw,
            "exists": False,
            "writable": False,
        }
        if mirror_dir_raw:
            mp = Path(mirror_dir_raw)
            mirror_status["exists"] = mp.exists()
            if mp.exists():
                try:
                    test_f = mp / ".office_write_test"
                    test_f.write_text("ok", encoding="utf-8")
                    test_f.unlink()
                    mirror_status["writable"] = True
                except Exception:
                    mirror_status["writable"] = False

        latest_dir = root / "data" / "latest"
        latest_files = (
            [f.name for f in latest_dir.iterdir() if f.is_file()] if latest_dir.exists() else []
        )

        pipeline_cfg_path = config_dir / "pipeline.yaml"
        daemon_url = "http://127.0.0.1:8766"
        if pipeline_cfg_path.exists():
            try:
                import yaml
                p_cfg = yaml.safe_load(pipeline_cfg_path.read_text(encoding="utf-8")) or {}
                for st in p_cfg.get("stages", []):
                    if st.get("id") == "voice" and st.get("daemon_url"):
                        daemon_url = st["daemon_url"]
                        break
            except Exception:
                pass

        def _check_daemon_health(url: str) -> bool:
            try:
                import urllib.request
                target = url.rstrip("/") + "/health"
                req = urllib.request.Request(target, headers={"User-Agent": "NewsAuto-Office/1.0"})
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    return resp.status == 200
            except Exception:
                return False

        tts_daemon_ok = await asyncio.to_thread(_check_daemon_health, daemon_url)

        scheduler_status = "unknown"
        if sys.platform == "win32":
            try:
                res = await asyncio.to_thread(
                    subprocess.run,
                    ["schtasks", "/query", "/tn", "NepalNewsBot-Morning", "/fo", "csv"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                scheduler_status = "scheduled" if res.returncode == 0 else "not_configured"
            except Exception:
                scheduler_status = "unavailable"

        return {
            "doctor": doctor_res,
            "disk": {
                "total_gb": round(disk_total / (1024**3), 2),
                "free_gb": round(disk_free / (1024**3), 2),
                "used_gb": round(disk_used / (1024**3), 2),
                "runs": run_items,
            },
            "mirrors": {
                "onedrive": mirror_status,
                "latest": {
                    "exists": latest_dir.exists(),
                    "file_count": len(latest_files),
                    "has_final_mp4": (latest_dir / "final.mp4").exists(),
                },
            },
            "services": {
                "tts_daemon": tts_daemon_ok,
                "kokoro_tts": tts_daemon_ok,
                "edge_tts": tts_daemon_ok,
                "windows_scheduler": scheduler_status,
            },
        }

    @app.post("/api/system/cleanup")
    async def api_system_cleanup(body: CleanupRequest) -> dict[str, Any]:
        plan = await asyncio.to_thread(
            plan_cleanup,
            runs_dir=runs_dir,
            media_config_path=config_dir / "media.yaml",
            pipeline_config_dir=config_dir,
            keep_runs=body.keep_runs,
            repo_root=root,
        )
        result = await asyncio.to_thread(
            execute_cleanup,
            plan=plan,
            dry_run=body.dry_run,
            repo_root=root,
        )
        result["dry_run"] = body.dry_run
        result["formatted_plan"] = format_cleanup_plan(plan)
        return result

    @app.get("/api/system/tts-daemon/status")
    async def api_tts_daemon_status() -> dict[str, Any]:
        def _check_health() -> dict[str, Any]:
            import urllib.request
            try:
                with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=1.5) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        return {"running": True, "status": "online", "details": data}
            except Exception as e:
                return {"running": False, "status": "offline", "error": str(e)}
            return {"running": False, "status": "offline"}

        return await asyncio.to_thread(_check_health)

    @app.post("/api/system/tts-daemon/restart")
    async def api_tts_daemon_restart() -> dict[str, Any]:
        def _restart() -> dict[str, Any]:
            import subprocess
            import time
            stop_script = root / "scripts" / "stop_tts_daemon.ps1"
            start_script = root / "scripts" / "start_tts_daemon.ps1"
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(stop_script)],
                    capture_output=True,
                    check=False,
                )
                subprocess.run(
                    ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", str(start_script)],
                    capture_output=True,
                    check=False,
                )
            time.sleep(1.0)
            import urllib.request
            try:
                with urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        return {"ok": True, "status": "online", "details": data}
            except Exception as e:
                return {"ok": False, "status": "offline", "error": str(e)}
            return {"ok": False, "status": "offline"}

        return await asyncio.to_thread(_restart)

    def _serialize_categories() -> list[dict[str, Any]]:
        return [
            {"id": num, "key": slug, "label": label, "keywords": list(kws)}
            for num, (slug, label, kws) in AVAILABLE_CATEGORIES.items()
        ]

    @app.get("/api/scraper/categories")
    async def api_scraper_categories() -> dict[str, Any]:
        return {
            "categories": _serialize_categories(),
            "automated_default": ["headlines"],
        }

    @app.get("/api/runs/{run_date}/state")
    async def api_run_state(run_date: str) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        state = await asyncio.to_thread(read_run_state, run_dir, config_dir)
        state["active_job"] = job_manager.get_active_job(run_date)
        state["available_categories"] = _serialize_categories()
        return state

    @app.get("/api/runs/{run_date}/replay")
    async def api_run_replay(run_date: str) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            return {"date": run_date, "events": []}
        manifest = load_manifest(manifest_path)
        events = await asyncio.to_thread(build_replay_events, run_dir, manifest)
        return {"date": run_date, "events": events}

    ALLOWED_RUN_FILE_EXTENSIONS = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi",
        ".wav", ".mp3", ".ogg",
        ".jpg", ".jpeg", ".png", ".webp",
        ".json", ".md", ".txt", ".csv", ".log", ".html", ".srt",
    }

    @app.get("/api/runs/{run_date}/file")
    async def api_run_file(run_date: str, path: str = Query(..., min_length=1)):
        _validate_run_date(run_date)
        allowed_root = (runs_dir / run_date).resolve()
        candidate = (allowed_root / path).resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Path outside run directory is forbidden.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if candidate.suffix.lower() not in ALLOWED_RUN_FILE_EXTENSIONS:
            raise HTTPException(status_code=403, detail=f"File extension '{candidate.suffix}' is not permitted.")
        return FileResponse(path=str(candidate))

    @app.post("/api/runs/{run_date}/morning")
    async def api_start_morning(run_date: str, body: MorningRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        rk = body.run_kind if body.run_kind in ("test", "production") else "test"
        args = ["run_daily.py", "--morning", "--date", run_date, "--run-kind", rk]
        valid_keys = {slug for slug, _, _ in AVAILABLE_CATEGORIES.values()} | {"all"}
        cleaned_cats = [
            str(c).strip().lower()
            for c in (body.categories or [])
            if str(c).strip().lower() in valid_keys
        ]
        if cleaned_cats:
            args.extend(["--categories", ",".join(dict.fromkeys(cleaned_cats))])
        else:
            args.append("--automated")
        if body.force:
            args.append("--force")
        try:
            job = await job_manager.start_job(run_date, "morning", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "job_id": job.job_id,
            "action": "morning",
            "categories": cleaned_cats or ["headlines"],
        }

    @app.post("/api/runs/{run_date}/draft-ai")
    async def api_draft_ai(run_date: str, body: DraftAIRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["-m", "src.agents.editor", "write", "--date", run_date]
        if body.provider:
            args.extend(["--provider", body.provider])
        if body.auto_approve:
            args.append("--auto-approve")
        try:
            job = await job_manager.start_job(run_date, "draft-ai", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "draft-ai"}

    @app.post("/api/runs/{run_date}/draft-manual")
    async def api_draft_manual(run_date: str, body: DraftManualRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        busy, reason = job_manager.is_busy(run_date)
        if busy:
            raise HTTPException(status_code=409, detail=reason)

        run_dir = runs_dir / run_date
        script_dir = run_dir / "script"
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run directory for {run_date} not found.")
        script_dir.mkdir(parents=True, exist_ok=True)
        (script_dir / "draft_script.json").write_text(
            json.dumps(body.draft_script, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if body.selection is not None:
            (script_dir / "selection.json").write_text(
                json.dumps(body.selection, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        manifest_path = run_dir / "manifest.json"
        manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
        agent = EditorAgent(run_dir)
        await asyncio.to_thread(agent.validate_draft, manifest)
        rep = json.loads((script_dir / "validation_report.json").read_text(encoding="utf-8"))
        return {
            "ok": bool(rep.get("is_valid")),
            "validation_report": rep,
        }

    @app.post("/api/runs/{run_date}/approve")
    async def api_approve_script(run_date: str) -> dict[str, Any]:
        _validate_run_date(run_date)
        busy, reason = job_manager.is_busy(run_date)
        if busy:
            raise HTTPException(status_code=409, detail=reason)

        run_dir = runs_dir / run_date
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Manifest not found for {run_date}.")
        manifest = load_manifest(manifest_path)
        agent = EditorAgent(run_dir)
        try:
            outputs = await asyncio.to_thread(agent.approve, manifest)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Reconcile status after approve
        reconciled_manifest = load_manifest(manifest_path)
        reconcile_manifest_status(reconciled_manifest, run_dir=run_dir, active_process=False)
        save_manifest(reconciled_manifest, manifest_path)
        return {"ok": True, "outputs": outputs}

    @app.post("/api/runs/{run_date}/prepare-media")
    async def api_prepare_media(run_date: str, body: PrepareMediaRequest | None = None) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["run_daily.py", "--prepare-media", "--date", run_date]
        if body and body.force:
            args.append("--force")
        if body and body.orientation:
            args.extend(["--orientation", body.orientation])
        try:
            job = await job_manager.start_job(run_date, "prepare-media", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "prepare-media"}

    @app.post("/api/runs/{run_date}/finish")
    async def api_start_finish(run_date: str, body: FinishRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["run_daily.py", "--finish", "--date", run_date]
        if body.force:
            args.append("--force")
        if body.orientation:
            args.extend(["--orientation", body.orientation])
        try:
            job = await job_manager.start_job(run_date, "finish", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "finish"}

    @app.post("/api/runs/{run_date}/reset-stage")
    async def api_reset_stage(run_date: str, body: ResetStageRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        busy, reason = job_manager.is_busy(run_date)
        if busy:
            raise HTTPException(status_code=409, detail=reason)

        run_dir = runs_dir / run_date
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"Manifest not found for {run_date}.")

        manifest = load_manifest(manifest_path)
        if manifest.get("status") == "abandoned":
            raise HTTPException(status_code=400, detail=f"Cannot reset stages on abandoned run {run_date}.")

        stage_order = ("scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa")
        if body.stage not in stage_order:
            raise HTTPException(status_code=400, detail=f"Unknown stage '{body.stage}'.")

        stages_to_reset = list(stage_order[stage_order.index(body.stage):]) if body.reset_downstream else [body.stage]
        for st in stages_to_reset:
            if st in manifest.get("stages", {}):
                update_stage_status(manifest, st, "pending", error_message=None)
                manifest["stages"][st]["retry_count"] = 0
                manifest["stages"][st]["start_time"] = None
                manifest["stages"][st]["end_time"] = None
        if manifest.get("status") in ("failed", "interrupted", "paused"):
            manifest["status"] = "pending"

        record_manual_edit(
            manifest,
            file="manifest.json",
            what=f"Reset stage '{body.stage}' (downstream={body.reset_downstream})",
            why="Manual reset via Web Office HQ",
        )
        reconcile_manifest_status(manifest, run_dir=run_dir, active_process=False)
        save_manifest(manifest, manifest_path)
        try:
            from src.orchestrator import write_status_md
            write_status_md(run_dir, manifest, active_process=False)
        except Exception:
            pass
        return {"ok": True, "action": "reset-stage", "stage": body.stage, "reset_stages": stages_to_reset}

    @app.post("/api/runs/{run_date}/stop")
    async def api_stop_run(run_date: str) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        runner_stopped = False
        pid_stopped = False

        # 1. Stop managed runner job
        runner_stopped = await job_manager.stop_job(run_date)

        # 2. Terminate any external pipeline process holding .pipeline.pid
        lock_file = run_dir / ".pipeline.pid"
        if lock_file.exists():
            try:
                raw = lock_file.read_text(encoding="utf-8").strip()
                pid = None
                if raw.startswith("{"):
                    pid = int(json.loads(raw).get("pid", 0))
                elif raw:
                    pid = int(raw)
                if pid and is_pid_alive(pid):
                    pid_stopped = await asyncio.to_thread(_terminate_pid_tree, pid)
                elif pid:
                    pid_stopped = True
            except Exception:
                pass
            finally:
                release_run_lock(run_dir)

        # 3. Update manifest if any stage was active
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = load_manifest(manifest_path)
                changed = False
                for s_key, s_data in manifest.get("stages", {}).items():
                    if isinstance(s_data, dict) and s_data.get("status") == "running":
                        update_stage_status(
                            manifest,
                            s_key,
                            "failed",
                            error_message="Execution stopped by operator via Web Office HQ.",
                        )
                        changed = True
                if manifest.get("status") == "running":
                    manifest["status"] = "failed"
                    changed = True
                if changed:
                    reconcile_manifest_status(manifest, run_dir=run_dir, active_process=False)
                    save_manifest(manifest, manifest_path)
            except Exception:
                pass

        # 4. Record event to office_events.jsonl
        emit_office_event(
            run_dir,
            "pipeline_stopped",
            stage="orchestrator",
            message="Workflow execution stopped by operator via Web Office HQ.",
        )

        stopped_any = bool(runner_stopped or pid_stopped)
        return {
            "ok": True,
            "stopped": stopped_any,
            "targets": {
                "runner": bool(runner_stopped),
                "pipeline_pid": bool(pid_stopped),
            },
            "message": "Workflow stopped by operator.",
        }

    @app.post("/api/runs/{run_date}/apply-picks")
    async def api_apply_picks(run_date: str, body: ApplyPicksRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        busy, reason = job_manager.is_busy(run_date)
        if busy:
            raise HTTPException(status_code=409, detail=reason)

        run_dir = runs_dir / run_date
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run directory for {run_date} not found.")
        picks_dir = run_dir / "media" / "review"
        picks_dir.mkdir(parents=True, exist_ok=True)
        picks_file = picks_dir / "picks_ui.json"
        picks_file.write_text(json.dumps(body.picks, indent=2), encoding="utf-8")

        args = ["run_daily.py", "--apply-picks", str(picks_file), "--date", run_date]
        try:
            job = await job_manager.start_job(run_date, "apply-picks", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "apply-picks"}

    @app.post("/api/runs/{run_date}/upload-media")
    async def api_upload_media(
        request: Request,
        run_date: str,
        shot_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run directory for {run_date} not found.")

        # Early rejection based on Content-Length header
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                if int(content_length_header) > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
                    )
            except ValueError:
                pass

        safe_shot_id = re.sub(r"[^\w\-]", "_", shot_id)
        raw_filename = Path(file.filename or "media_file").name
        ext = Path(raw_filename).suffix.lower()
        stem = Path(raw_filename).stem[:100]
        orig_filename = f"{stem}{ext}" if len(raw_filename) > 100 else raw_filename
        allowed_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp"}
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(allowed_exts))}",
            )

        uploads_dir = run_dir / "media" / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"{safe_shot_id}_{re.sub(r'[^\w\-.]', '_', orig_filename)}"
        dest_path = uploads_dir / dest_filename
        temp_dest = uploads_dir / f"{dest_filename}.upload_tmp.{uuid.uuid4().hex}"

        total_bytes = 0
        chunk_size = 64 * 1024
        try:
            with open(temp_dest, "wb") as out_f:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
                        )
                    out_f.write(chunk)

            if total_bytes == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            if not _verify_media_content(temp_dest, ext):
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file header does not match expected format for extension '{ext}'.",
                )

            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            temp_dest.replace(dest_path)
        except Exception:
            if temp_dest.exists():
                try:
                    temp_dest.unlink()
                except OSError:
                    pass
            raise

        # Generate thumbnail
        thumbs_dir = run_dir / "media" / "review" / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        thumb_name = f"{safe_shot_id}_upload.jpg"
        thumb_path = thumbs_dir / thumb_name

        is_video = ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        if is_video:
            async def _run_ffmpeg_thumb(args: list[str]) -> bool:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *args,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=30)
                        return proc.returncode == 0
                    except (asyncio.TimeoutError, TimeoutError):
                        try:
                            proc.kill()
                            await proc.wait()
                        except Exception:
                            pass
                        return False
                except Exception:
                    return False

            cmd = [
                "ffmpeg", "-y", "-ss", "00:00:00.5", "-i", str(dest_path),
                "-vframes", "1", "-vf", "scale=480:-1", str(thumb_path)
            ]
            ok = await _run_ffmpeg_thumb(cmd)
            if not ok or not thumb_path.exists():
                cmd_fallback = [
                    "ffmpeg", "-y", "-i", str(dest_path),
                    "-vframes", "1", "-vf", "scale=480:-1", str(thumb_path)
                ]
                await _run_ffmpeg_thumb(cmd_fallback)
        else:
            def _resize_image() -> None:
                try:
                    from PIL import Image
                    with Image.open(dest_path) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        new_h = max(1, int(h * 480 / w))
                        im_resized = im.resize((480, new_h), Image.Resampling.LANCZOS)
                        im_resized.save(thumb_path, "JPEG", quality=85)
                except Exception:
                    pass

            await asyncio.to_thread(_resize_image)

        rel_file = str(dest_path.relative_to(run_dir).as_posix())
        rel_thumb = f"media/review/thumbs/{thumb_name}"
        thumb_url = f"/api/runs/{run_date}/file?path={rel_thumb}" if thumb_path.exists() else None

        return {
            "ok": True,
            "shot_id": safe_shot_id,
            "filename": orig_filename,
            "file_path": rel_file,
            "thumb_url": thumb_url,
            "kind": "video" if is_video else "image",
            "size_bytes": total_bytes,
        }

    @app.post("/api/runs/{run_date}/validate-media-link")
    async def api_validate_media_link(run_date: str, body: ValidateMediaLinkRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        url = body.url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            raise HTTPException(status_code=400, detail="Invalid URL: Must begin with http:// or https://")

        media_cfg_path = config_dir / "media.yaml"
        if not media_cfg_path.exists():
            media_cfg_path = REPO_ROOT / "config" / "media.yaml"
        blocklist = []
        if media_cfg_path.exists():
            import yaml
            m_cfg = yaml.safe_load(media_cfg_path.read_text(encoding="utf-8")) or {}
            blocklist = [d.lower() for d in m_cfg.get("blocklisted_domains", [])]

        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            host = parsed.hostname or parsed.netloc.split(":")[0]
            domain = parsed.netloc.lower()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid URL format.")

        if not host:
            raise HTTPException(status_code=400, detail="Invalid URL: Missing host.")

        if _is_ip_private_or_loopback(host):
            raise HTTPException(
                status_code=400,
                detail=f"URL host '{host}' resolves to a private or loopback IP. Local network addresses are forbidden.",
            )

        if any(b in domain for b in blocklist):
            raise HTTPException(
                status_code=400,
                detail=f"URL domain '{domain}' is in the copyright news blocklist. Downloading from news portals is prohibited.",
            )

        return {"ok": True, "url": url, "domain": domain}

    @app.websocket("/ws/runs/{run_date}")
    async def ws_run_stream(websocket: WebSocket, run_date: str) -> None:
        if not DATE_RE.match(run_date):
            await websocket.close(code=1008)
            return

        allowed_hosts = [
            h.strip().lower()
            for h in os.getenv("OFFICE_ALLOWED_HOSTS", "").split(",")
            if h.strip()
        ]
        host = websocket.headers.get("host", "")
        if not host or not _is_host_allowed(host, allowed_hosts):
            await websocket.close(code=1008)
            return

        origin = websocket.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if not parsed.netloc or not _is_host_allowed(parsed.netloc, allowed_hosts):
                await websocket.close(code=1008)
                return

        configured_token = os.getenv("OFFICE_TOKEN", "").strip()
        if configured_token:
            token = websocket.query_params.get("token") or websocket.cookies.get("office_token")
            if not _is_token_valid(token, configured_token):
                await websocket.close(code=1008)
                return

        await websocket.accept()
        broadcaster = broadcaster_manager.get_broadcaster(run_date)
        await broadcaster.subscribe(websocket)

        try:
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                except asyncio.TimeoutError:
                    # Keepalive probe: check if client is still responsive
                    try:
                        await websocket.send_json({"type": "ping"})
                        continue
                    except Exception:
                        break
                msg_type = data.get("type")
                if msg_type in ("ping", "pong"):
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                elif msg_type == "request_sync":
                    await broadcaster.send_current_state(websocket)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await broadcaster.unsubscribe(websocket)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()

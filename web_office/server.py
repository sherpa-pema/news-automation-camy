"""FastAPI + WebSocket server for Nepal News Bot 2D Office."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.editor import EditorAgent
from src.utils.daily_ops import run_doctor
from src.utils.manifest import load_manifest
from web_office.runner import OfficeJobManager
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


class DraftAIRequest(BaseModel):
    provider: str | None = None


class DraftManualRequest(BaseModel):
    draft_script: dict[str, Any]
    selection: dict[str, Any] | None = None


class FinishRequest(BaseModel):
    force: bool = False


class PrepareMediaRequest(BaseModel):
    force: bool = False


class ResetStageRequest(BaseModel):
    stage: str


class ApplyPicksRequest(BaseModel):
    picks: dict[str, Any]


class ValidateMediaLinkRequest(BaseModel):
    url: str


def create_app(repo_root: Path | None = None) -> FastAPI:
    root = (repo_root or REPO_ROOT).resolve()
    runs_dir = root / "data" / "runs"
    config_dir = root / "config"
    job_manager = OfficeJobManager(root)

    app = FastAPI(title="Nepal News Bot 2D Office", version="1.0.0")

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        return {"ok": True, "repo_root": str(root)}

    @app.get("/api/runs")
    async def api_list_runs() -> dict[str, Any]:
        return {"runs": list_runs(runs_dir)}

    @app.get("/api/doctor")
    async def api_doctor() -> dict[str, Any]:
        checks = await asyncio.to_thread(run_doctor)
        return {
            "ok": all(c.status != "FAIL" for c in checks),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in checks
            ],
        }

    @app.get("/api/runs/{run_date}/state")
    async def api_run_state(run_date: str) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        state = await asyncio.to_thread(read_run_state, run_dir, config_dir)
        state["active_job"] = job_manager.get_active_job(run_date)
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
        return FileResponse(path=str(candidate))

    @app.post("/api/runs/{run_date}/morning")
    async def api_start_morning(run_date: str, body: MorningRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        rk = body.run_kind if body.run_kind in ("test", "production") else "test"
        args = ["run_daily.py", "--morning", "--date", run_date, "--run-kind", rk]
        if body.force:
            args.append("--force")
        try:
            job = await job_manager.start_job(run_date, "morning", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "morning"}

    @app.post("/api/runs/{run_date}/draft-ai")
    async def api_draft_ai(run_date: str, body: DraftAIRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["-m", "src.agents.editor", "write", "--date", run_date]
        if body.provider:
            args.extend(["--provider", body.provider])
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
        return {"ok": True, "outputs": outputs}

    @app.post("/api/runs/{run_date}/prepare-media")
    async def api_prepare_media(run_date: str, body: PrepareMediaRequest | None = None) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["run_daily.py", "--prepare-media", "--date", run_date]
        if body and body.force:
            args.append("--force")
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
        try:
            job = await job_manager.start_job(run_date, "finish", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "finish"}

    @app.post("/api/runs/{run_date}/reset-stage")
    async def api_reset_stage(run_date: str, body: ResetStageRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
        args = ["run_daily.py", "--date", run_date, "--reset-stage", body.stage]
        try:
            job = await job_manager.start_job(run_date, f"reset:{body.stage}", args)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "job_id": job.job_id, "action": "reset-stage"}

    @app.post("/api/runs/{run_date}/apply-picks")
    async def api_apply_picks(run_date: str, body: ApplyPicksRequest) -> dict[str, Any]:
        _validate_run_date(run_date)
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
        run_date: str,
        shot_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        _validate_run_date(run_date)
        run_dir = runs_dir / run_date
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run directory for {run_date} not found.")

        safe_shot_id = re.sub(r"[^\w\-]", "_", shot_id)
        orig_filename = Path(file.filename or "media_file").name
        ext = Path(orig_filename).suffix.lower()
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

        content = await file.read()
        dest_path.write_bytes(content)

        # Generate thumbnail
        thumbs_dir = run_dir / "media" / "review" / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)
        thumb_name = f"{safe_shot_id}_upload.jpg"
        thumb_path = thumbs_dir / thumb_name

        is_video = ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        if is_video:
            cmd = [
                "ffmpeg", "-y", "-ss", "00:00:00.5", "-i", str(dest_path),
                "-vframes", "1", "-vf", "scale=480:-1", str(thumb_path)
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0 or not thumb_path.exists():
                cmd_fallback = [
                    "ffmpeg", "-y", "-i", str(dest_path),
                    "-vframes", "1", "-vf", "scale=480:-1", str(thumb_path)
                ]
                subprocess.run(cmd_fallback, capture_output=True)
        else:
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
            "size_bytes": len(content),
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
            domain = urlparse(url).netloc.lower()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid URL format.")

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

        await websocket.accept()
        run_dir = runs_dir / run_date
        manifest_path = run_dir / "manifest.json"
        events_path = run_dir / "office_events.jsonl"
        log_path = run_dir / "run.log"

        last_manifest_mtime = -1.0
        last_active: bool | None = None
        events_offset = events_path.stat().st_size if events_path.exists() else 0
        log_offset = log_path.stat().st_size if log_path.exists() else 0

        try:
            initial_state = await asyncio.to_thread(read_run_state, run_dir, config_dir)
            initial_state["active_job"] = job_manager.get_active_job(run_date)
            await websocket.send_json({"type": "state_snapshot", "data": initial_state})

            while True:
                await asyncio.sleep(0.35)
                cur_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else -1.0
                state_now = await asyncio.to_thread(read_run_state, run_dir, config_dir)
                cur_active = state_now["active_process"] or bool(
                    (job_manager.get_active_job(run_date) or {}).get("status") == "running"
                )

                if cur_mtime != last_manifest_mtime or cur_active != last_active:
                    last_manifest_mtime = cur_mtime
                    last_active = cur_active
                    state_now["active_job"] = job_manager.get_active_job(run_date)
                    await websocket.send_json({"type": "state_snapshot", "data": state_now})

                new_ev_lines, events_offset = tail_new_lines(events_path, events_offset)
                for raw_line in new_ev_lines:
                    if not raw_line.strip():
                        continue
                    try:
                        ev_obj = json.loads(raw_line)
                        await websocket.send_json({"type": "office_event", "data": ev_obj})
                    except Exception:
                        pass

                new_log_lines, log_offset = tail_new_lines(log_path, log_offset)
                if new_log_lines:
                    await websocket.send_json({"type": "log_lines", "data": new_log_lines})
        except WebSocketDisconnect:
            return
        except Exception:
            try:
                await websocket.close()
            except Exception:
                pass

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()

"""Shared background broadcaster per run for Web Office WebSocket clients.

Guarantees:
1. Exactly one disk poller per active run, not one poller per connected socket.
2. File-mtime fingerprinting avoids redundant JSON parsing and status recalculation.
3. Event payloads are capped and bounded.
4. Real-time subprocess stdout is forwarded as `job_output` messages.
5. Heartbeat pings detect dropped connections and keep WebSockets alive.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from web_office.watcher import _cap_event_payload, read_run_state, tail_new_lines

logger = logging.getLogger("web_office.broadcaster")


class RunBroadcaster:
    """Manages all active WebSocket connections for a single run_date."""

    def __init__(self, run_date: str, runs_dir: Path, config_dir: Path, job_manager: Any) -> None:
        self.run_date = run_date
        self.runs_dir = runs_dir
        self.config_dir = config_dir
        self.job_manager = job_manager
        self.run_dir = runs_dir / run_date
        self.subscribers: set[WebSocket] = set()

        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._stopped = False

        self.last_manifest_mtime = -1.0
        self.last_active: bool | None = None
        self.events_offset = 0
        self.log_offset = 0
        self._last_ping_ts = 0.0

    @property
    def subscriber_count(self) -> int:
        return len(self.subscribers)

    async def subscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.subscribers.add(websocket)
            # Send immediate initial state snapshot to newly connected client
            try:
                initial_state = await asyncio.to_thread(read_run_state, self.run_dir, self.config_dir)
                initial_state["active_job"] = self.job_manager.get_active_job(self.run_date)
                await websocket.send_json({"type": "state_snapshot", "data": initial_state})
            except Exception as exc:
                logger.debug(f"Failed sending initial snapshot to client: {exc}")

            if self._task is None or self._task.done():
                self._stopped = False
                # Initialize file offsets to current end of files so clients only receive new events/logs
                events_path = self.run_dir / "office_events.jsonl"
                log_path = self.run_dir / "run.log"
                self.events_offset = events_path.stat().st_size if events_path.exists() else 0
                self.log_offset = log_path.stat().st_size if log_path.exists() else 0
                self._task = asyncio.create_task(self._run_loop())

    async def unsubscribe(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.subscribers.discard(websocket)
            if not self.subscribers and self._task and not self._task.done():
                self._stopped = True
                self._task.cancel()
                self._task = None

    async def send_current_state(self, websocket: WebSocket) -> None:
        """Sends a fresh state snapshot directly to a specific socket (e.g. after sync request)."""
        try:
            state = await asyncio.to_thread(read_run_state, self.run_dir, self.config_dir)
            state["active_job"] = self.job_manager.get_active_job(self.run_date)
            await websocket.send_json({"type": "state_snapshot", "data": state})
        except Exception:
            pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Dispatches a JSON message to all subscribed clients, pruning dead sockets."""
        if not self.subscribers:
            return
        dead: list[WebSocket] = []
        for ws in list(self.subscribers):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self.subscribers.discard(ws)

    def on_job_output(self, action: str, line: str) -> None:
        """Called synchronously by JobManager when stdout line is emitted."""
        if not self.subscribers:
            return
        asyncio.create_task(
            self.broadcast(
                {
                    "type": "job_output",
                    "data": {
                        "run_date": self.run_date,
                        "action": action,
                        "line": line,
                    },
                }
            )
        )

    async def _run_loop(self) -> None:
        """Single background polling loop per run."""
        manifest_path = self.run_dir / "manifest.json"
        events_path = self.run_dir / "office_events.jsonl"
        log_path = self.run_dir / "run.log"

        while not self._stopped and self.subscribers:
            try:
                await asyncio.sleep(0.35)
                now_ts = asyncio.get_event_loop().time()

                # 1. State change check (using fast mtime)
                cur_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else -1.0
                state_now = await asyncio.to_thread(read_run_state, self.run_dir, self.config_dir)
                cur_active = state_now["active_process"] or bool(
                    (self.job_manager.get_active_job(self.run_date) or {}).get("status") == "running"
                )

                if cur_mtime != self.last_manifest_mtime or cur_active != self.last_active:
                    self.last_manifest_mtime = cur_mtime
                    self.last_active = cur_active
                    state_now["active_job"] = self.job_manager.get_active_job(self.run_date)
                    await self.broadcast({"type": "state_snapshot", "data": state_now})

                # 2. Tail office events (single read for all clients)
                new_ev_lines, self.events_offset = tail_new_lines(events_path, self.events_offset)
                for raw_line in new_ev_lines:
                    if not raw_line.strip():
                        continue
                    try:
                        ev_obj = json.loads(raw_line)
                        capped_ev = _cap_event_payload(ev_obj)
                        await self.broadcast({"type": "office_event", "data": capped_ev})
                    except Exception:
                        pass

                # 3. Tail run logs (single read for all clients)
                new_log_lines, self.log_offset = tail_new_lines(log_path, self.log_offset)
                if new_log_lines:
                    await self.broadcast({"type": "log_lines", "data": new_log_lines})

                # 4. Periodic heartbeat ping (every 10s)
                if now_ts - self._last_ping_ts >= 10.0:
                    self._last_ping_ts = now_ts
                    await self.broadcast({"type": "ping", "ts": datetime.now().astimezone().isoformat()})

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"Broadcaster loop exception for {self.run_date}: {exc}")
                await asyncio.sleep(0.5)

    async def close(self) -> None:
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
        for ws in list(self.subscribers):
            try:
                await ws.close()
            except Exception:
                pass
        self.subscribers.clear()


class BroadcasterManager:
    """Registry of RunBroadcasters keyed by run_date."""

    def __init__(self, runs_dir: Path, config_dir: Path, job_manager: Any) -> None:
        self.runs_dir = runs_dir
        self.config_dir = config_dir
        self.job_manager = job_manager
        self.broadcasters: dict[str, RunBroadcaster] = {}
        self._lock = asyncio.Lock()

        # Connect JobManager stdout to broadaster
        self.job_manager.register_output_callback(self.on_job_output)

    def get_broadcaster(self, run_date: str) -> RunBroadcaster:
        if run_date not in self.broadcasters:
            self.broadcasters[run_date] = RunBroadcaster(
                run_date=run_date,
                runs_dir=self.runs_dir,
                config_dir=self.config_dir,
                job_manager=self.job_manager,
            )
        return self.broadcasters[run_date]

    def on_job_output(self, run_date: str, action: str, line: str) -> None:
        b = self.broadcasters.get(run_date)
        if b:
            b.on_job_output(action, line)

    async def shutdown(self) -> None:
        self.job_manager.unregister_output_callback(self.on_job_output)
        for b in list(self.broadcasters.values()):
            await b.close()
        self.broadcasters.clear()

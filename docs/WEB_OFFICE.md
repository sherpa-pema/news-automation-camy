# Nepal News Bot — Automation HQ (Web Office)

The **Web Office** (`web_office/`) is the central real-time monitoring and control dashboard for Nepal News Bot. It visualizes the automated daily pipeline as a virtual 2D floorplan with animated agent stations, interactive inspection drawers, live telemetry streaming, safety gates, and maintenance utilities.

---

## 1. Architecture Overview

```
                                 [ Web Clients ]
                        Desktop Browser  /  Mobile Phone (LAN)
                                |                  |
                                +--------+---------+
                                         | (WebSocket / HTTP)
                                         v
                         +-------------------------------+
                         |   FastAPI + Security Guard    |
                         |  (Token Auth, Security Headers)|
                         +---------------+---------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
    +----------------------+                             +--------------------+
    | BroadcasterManager   |                             |  OfficeJobManager  |
    | (1 Broadcaster / run)|                             | (Global GPU Cap: 1)|
    +----------+-----------+                             +---------+----------+
               |                                                   |
               | (350ms Poller + Heartbeats)                       | Subprocess Exec
               v                                                   v
   +-----------------------+                             +--------------------+
   |  Watcher State Cache  |                             | run_daily.py CLI / |
   | (mtime Fingerprint)   |                             | Agent Subprocesses |
   +-----------+-----------+                             +--------------------+
               |
               v
     data/runs/YYYY-MM-DD/
     (manifest, events, logs)
```

### Key Architectural Pillars:
- **Singleton Broadcaster per Run**: Regardless of how many browser tabs or mobile devices connect to watch a run, exactly **one** lightweight background poller monitors the run on disk. When subscriber count drops to zero, the background task stops automatically.
- **Single-Tenant Concurrency Guard**: Because GPU, Kokoro TTS, Whisper alignment, and FFmpeg rendering are hardware-constrained resources, `OfficeJobManager` enforces a global single-tenant concurrency cap across all runs. Any concurrent execution request is rejected with `409 Conflict`.
- **Fast Fingerprint-based State Caching**: `read_run_state()` computes an $O(1)$ timestamp/size fingerprint of run artifacts. Repeated requests return cached state snapshots instantly without disk re-reading or JSON parsing.
- **Real-Time Job Output Streaming**: Subprocess stdout lines are streamed live as `job_output` WebSocket messages and buffered in the station drawers.
- **FastAPI Lifespan Cleanup**: On server exit or restart, all running subprocess trees are terminated cleanly (`taskkill /F /T` on Windows) to prevent orphan background processes.

---

## 2. API Endpoints Reference

All `/api/` endpoints require valid authentication if `OFFICE_TOKEN` is configured.

### Run Inspection & Control
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Returns service status and repository root. |
| `GET` | `/api/runs` | Lists all dated runs sorted newest-first. |
| `GET` | `/api/runs/{date}/state` | Returns the full snapshot for a run (manifest, gates, next action, source health). |
| `GET` | `/api/runs/{date}/replay` | Returns structured timeline events for historical replay mode. |
| `GET` | `/api/runs/{date}/file?path=...` | Safely serves a sandboxed run artifact (JSON, Markdown, audio, video). |
| `POST` | `/api/runs/{date}/morning` | Dispatches the morning scraper & editor pipeline (`--morning`). |
| `POST` | `/api/runs/{date}/draft-ai` | Triggers AI script draft generation with configured LLM provider. |
| `POST` | `/api/runs/{date}/draft-manual` | Submits and validates a manual JSON script draft. |
| `POST` | `/api/runs/{date}/approve` | Formal sign-off on the editor draft, writing `approval.json`. |
| `POST` | `/api/runs/{date}/prepare-media` | Executes voice, timeline alignment, and media candidate scoring. |
| `POST` | `/api/runs/{date}/apply-picks` | Applies visual selections from the interactive picker and initiates render. |
| `POST` | `/api/runs/{date}/finish` | Executes remaining pipeline stages through final render and QA (`--finish`). |
| `POST` | `/api/runs/{date}/reset-stage` | Resets a stage back to `pending`. Supports `reset_downstream: true` to reset all subsequent pipeline stages. |
| `POST` | `/api/runs/{date}/stop` | Terminates running pipeline subprocesses and clears `.pipeline.pid`. |
| `POST` | `/api/runs/{date}/upload-media` | Uploads user-supplied video/image for a specific shot (bounded at 500 MB with magic byte verification). |
| `POST` | `/api/runs/{date}/validate-media-link` | Verifies a proposed media URL against copyright news blocklists. |

### System Health & Maintenance
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/doctor` | Returns system readiness checks (cached with 30s TTL). |
| `GET` | `/api/system/maintenance` | Comprehensive system report: Doctor checks, disk usage per run, mirror mount status, TTS daemon status, and Windows Task Scheduler status. |
| `POST` | `/api/system/cleanup` | Storage cleanup utility: plans or executes deletion of runs older than `keep_runs` days and prunes media cache. Accepts `{ "dry_run": bool, "keep_runs": int }`. |

---

## 3. WebSocket Streaming & Event Protocol

Connect to `ws://<host>:<port>/ws/runs/{date}` (or `?token=<OFFICE_TOKEN>`).

### Server-to-Client Messages
- **`state_snapshot`**: Full state payload broadcast when state fingerprint changes or upon client sync request.
- **`office_event`**: Append-only telemetry event from `office_events.jsonl`:
  - `substage_progress`: Determinate progress for long stages (`current`, `total`, `eta_seconds`, `message`).
  - `packet_handoff`: Visual animation trigger for document packets traveling between stations.
  - `stage_status`: Dispatched when a stage begins, succeeds (`done`), or fails.
  - `scraper_source_health`: Live report of article counts, newest age, and status per news source.
  - `publish_complete`: Emitted when deliverables are copied to `data/latest` and OneDrive.
  - `pipeline_stopped`: Operator interrupted the run.
  - `boss_alert`: High-priority alert requiring user action.
- **`job_output`**: Real-time stdout lines streamed directly from active subprocesses.
- **`log_lines`**: New log lines appended to `run.log`.
- **`ping`**: Heartbeat emitted every 10 seconds.

### Client-to-Server Messages
- **`{"type": "pong"}`**: Heartbeat response.
- **`{"type": "request_sync"}`**: Requests an immediate fresh state snapshot.

---

## 4. LAN Mode & Security Doctrine

The Web Office is designed for safe monitoring from phones or tablets on your local network.

### Security Defenses:
1. **Token Authentication (`OFFICE_TOKEN`)**:
   - When set, all HTTP `/api/` endpoints and `/ws/` connections require the token.
   - Passed via query parameter `?token=...`, HTTP header `X-Office-Token`, or HttpOnly cookie.
   - Compared in constant time (`hmac.compare_digest`) to prevent timing attacks.
2. **Host & Origin Allowlists**:
   - Restricts incoming HTTP and WebSocket requests to localhost and private LAN IP subnets (`192.168.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`).
3. **Bounded Uploads**:
   - File uploads capped at 500 MB with early `Content-Length` rejection.
   - Content streamed in 64 KB chunks with magic byte validation (JPEG, PNG, WebP, MP4, QuickTime).
4. **Sandboxed Previews**:
   - All iframe previews enforce `sandbox="allow-scripts allow-same-origin allow-forms allow-popups"`.
5. **Security Headers**:
   - All responses include `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, and strict referrer policies.

### Starting in LAN Mode:
Run `Start_Office_LAN.bat`. It will:
- Automatically generate a secure 16-character `OFFICE_TOKEN` if not set in your environment.
- Detect your local LAN IP address.
- Launch the server on `0.0.0.0:8765`.
- Open your browser with the authenticated token URL.
- Print a QR code / mobile URL in the console for your phone.

---

## 5. UI Features & Keyboard Shortcuts

- **DOM Stage Rail**: Accessible chips under the HUD displaying status dot, elapsed clock (`m:ss`), retry count (`↺ n`), and determinate progress bars with live scene counts and ETAs.
- **Next-Action Banner**: Context-aware guidance banner offering immediate action buttons (e.g. *Review Script*, *Open Visual Picker*, *Fix & Retry QA*, *Stop Pipeline*).
- **Log Viewer**:
  - Live log streaming with syntax coloring (red errors, amber warnings, green completions, blue info).
  - Search filter input to isolate specific keywords.
  - Auto-scroll pause toggle (`⏸ Pause` / `▶ Follow`).
  - One-click **Copy to Clipboard** and **Download .log** buttons.
- **Keyboard Shortcuts**:
  - <kbd>1</kbd> - <kbd>8</kbd>: Open station drawer (1=`scraper`, 2=`editor`, 3=`voice`, 4=`timeline`, 5=`media`, 6=`motion`, 7=`render`, 8=`qa`).
  - <kbd>Esc</kbd>: Close active drawer or modal.
  - <kbd>Space</kbd>: Toggle office audio chimes on/off.
  - <kbd>?</kbd>: Open keyboard shortcuts help.
- **Tab Inactivity Throttling**:
  - When the browser tab is hidden, the 2D canvas animation loop automatically halts (dropping CPU to 0%).
  - If a stage completes or fails in the background, a desktop notification and tab title badge (`🔔 Action Needed`) notify the operator.

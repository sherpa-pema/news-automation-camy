# Daily Run Operational Runbook

This guide covers the semi-automatic daily routine for **Nepal News Bot** (`nepal-news-bot`).

> **Important safety notes**
> - Default `execution.default_run_kind` in `config/pipeline.yaml` is `test`, which stamps a `TEST RUN` tag on the video and a `TEST RUN - DO NOT PUBLISH` banner in `description.md`. Only pass `--run-kind production` (or update `default_run_kind`) when intentionally producing a publish-ready broadcast.
> - **Never auto-approve scripts.** You are always the human sign-off before rendering.
> - **PC scheduling requirement:** The Windows Scheduled Task (`NepalNewsBot-Morning`) runs under `LogonType = Interactive`, so your PC must be **powered on** and you must be **signed into Windows** for the 07:00 job (or a missed-start catch-up) to run.

---

## Quick Daily Sequence

1. **System check (optional / first run of week)**
   ```powershell
   .\.venv\Scripts\python.exe run_daily.py --doctor
   ```
2. **Morning prep** (runs automatically at `07:00` if scheduled, or double-click `Run_Morning.bat`):
   ```powershell
   .\.venv\Scripts\python.exe run_daily.py --morning
   ```
   - Defaults `--date` to today in `Asia/Kathmandu` and `--run-kind` to `config/pipeline.yaml` (`test`).
   - Runs `scraper` (if it fails, waits 5 minutes and retries once; if it still fails, writes the error to `STATUS.md`).
   - Runs `editor` (`prepare`) and stops at **`needs writer`** (`status: paused`).
3. **Draft & validate script** — paste Message **(A)** below into Antigravity.
4. **Review** `data/runs/<today>/script/script_preview.md` and the supporting source sentences shown in chat.
5. **Approve & finish** — paste Message **(B)** below into Antigravity (or run `--approve` + `Run_Finish.bat`).
   - `--finish` verifies `editor == done` and the approval + validator fingerprint gate, then runs `voice`, `timeline`, `media --auto`, `motion`, `render`, and `qa`.
   - Copies final outputs into `data/latest/` (`final.mp4`, `description.md`, `qa_report.json`, `storyboard.html` + `storyboard_frames/`, `contact_sheet.jpg`, `README.txt`).
6. **Inspect output without playing video**:
   - Open `data/runs/<today>/STATUS.md` (or `data/runs/<today>/qa/storyboard.html` by double-clicking it, also copied to `data/latest/storyboard.html`).
7. **Tweak visuals (optional)** — use `picker.html` + `--apply-picks` if you want to swap stock clips or cards (see Message **C** below).

---

## Ready-to-Paste Antigravity Messages

### (A) "Write today's script"

Paste this into Antigravity after `--morning` completes (`needs writer`):

```text
Write today's script: read data/runs/<today>/script/editor_brief.md, write draft_script.json and selection.json, run `python -m src.agents.editor validate --date <today>` (maximum 3 revision rounds), then stop and show script_preview.md together with the supporting source sentences per scene. Do not approve.
```

*(Replace `<today>` with the run date `YYYY-MM-DD`, or leave `<today>` if obvious from `--status`.)*

---

### (B) "Approved"

After reading `script_preview.md` and confirming every fact matches the source sentences, paste this into Antigravity:

```text
Approved: run `python -m src.agents.editor approve --date <today>`, then run `python run_daily.py --finish --date <today>` and report data/runs/<today>/STATUS.md.
```

---

### (C) "Tweak visuals" (Optional, after `--finish`)

Every `--finish` run executes `media --auto` (picking scored candidates at or above `min_relevance`, creating headline cards below threshold, and keeping `media/review/picker.html`). To replace any shot visually:

1. Double-click `data/runs/<today>/media/review/picker.html` (or save your updated `picks.json` from the picker UI).
2. Paste this short message into Antigravity (or run the commands yourself):

```text
Tweak visuals for <today>: apply my picks from <path-to-picks.json> using `python -m src.agents.media --date <today> --apply-picks <path-to-picks.json>`, then re-run `motion`, `render`, and `qa` for <today> (and report STATUS.md).
```

*(Manually running `--only motion`, `--only render`, and `--only qa` updates `final.mp4` and `qa/storyboard.html`. To also refresh `data/latest/`, reset `qa` and run `--finish --date <today>` or inspect `data/runs/<today>/qa/storyboard.html` directly.)*

---

## Double-Click Launchers (Project Root)

- `Run_Morning.bat` — runs `.venv\Scripts\python.exe run_daily.py --morning` and pauses at the end.
- `Run_Finish.bat` — runs `.venv\Scripts\python.exe run_daily.py --finish` and pauses at the end.
- `Status.bat` — runs `.venv\Scripts\python.exe run_daily.py --status` and pauses at the end.

---

## Windows Task Scheduler (`NepalNewsBot-Morning`)

The script `scripts/install_schedule.ps1` sets up a daily task named `NepalNewsBot-Morning`:
- **Schedule:** Daily at `07:00` local time (`StartWhenAvailable` enabled so missed starts run as soon as possible).
- **Command:** `.venv\Scripts\python.exe run_daily.py --morning` (working directory = repo root).
- **Log file:** `logs/morning_<YYYY-MM-DD>.log`.
- **Logon mode:** `Interactive` (runs **only while you are signed in**; PC must be turned on and unlocked/signed in).

Preview without registering:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_schedule.ps1 -DryRun
```

Install (when ready):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_schedule.ps1
```

Remove later:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/uninstall_schedule.ps1
```

---

## Maintenance Commands

- **Check status and rewrite `STATUS.md`**:
  ```powershell
  .\.venv\Scripts\python.exe run_daily.py --status --date <YYYY-MM-DD>
  ```
- **Preview storage cleanup** (keeps newest 7 runs in full; deletes temporary downloaded shots, thumbs, temp audio, and render caches from older runs while preserving `script/`, `description.md`, `final.mp4`, `qa_report.json`, `manifest.json`, and `credits.md`, and prunes stale `data/media_cache`):
  ```powershell
  .\.venv\Scripts\python.exe run_daily.py --cleanup --dry-run
  ```
- **Execute cleanup** (only after reviewing dry-run list):
  ```powershell
  .\.venv\Scripts\python.exe run_daily.py --cleanup
  ```
  *(Never touches `data/media_library`, `.env`, `models/`, `vendor/`, or any run within `keep_runs`.)*
- **Run a publish-ready build** (omits `TEST RUN` tag & `TEST RUN - DO NOT PUBLISH` banner):
  ```powershell
  .\.venv\Scripts\python.exe run_daily.py --morning --run-kind production
  ```

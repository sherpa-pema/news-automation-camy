# Daily Run Operational Runbook

This guide covers the semi-automatic daily routine for **Nepal News Bot** (`nepal-news-bot`).

- **Active Project Folder:** `C:\news-auto` *(stored outside OneDrive to prevent file-sync locks during TTS, alignment, and FFmpeg rendering)*
- **OneDrive Viewer Mirror:** `C:\Users\Chungba Sherpa\OneDrive\NepalNewsBot\latest` *(configured via `outputs.mirror_dir` in `config/pipeline.yaml`; automatically receives finished outputs after `--finish`)*

> **Important safety notes**
> - Default `execution.default_run_kind` in `config/pipeline.yaml` is `test`, which stamps a `TEST RUN` tag on the video and a `TEST RUN - DO NOT PUBLISH` banner in `description.md`. Only pass `--run-kind production` (or update `default_run_kind`) when intentionally producing a publish-ready broadcast.
> - **Never auto-approve scripts.** You are always the human sign-off before rendering.
> - **PC scheduling requirement:** The Windows Scheduled Task (`NepalNewsBot-Morning`) runs under `LogonType = Interactive` with **`WakeToRun` enabled**. It can wake your PC from Windows Sleep (if Power Options *"Allow wake timers"* and BIOS wake timers are enabled), but your PC must **not be powered off** and you must be **signed into your Windows account** for the 07:00 job (or missed-start catch-up) to run.

---

## Quick Daily Sequence

All CLI commands below run from `C:\news-auto`:

1. **System check (optional / first run of week)**
   ```powershell
   cd C:\news-auto
   .\.venv\Scripts\python.exe run_daily.py --doctor
   ```
2. **Morning prep** (runs automatically at `07:00` if scheduled, or double-click `C:\news-auto\Run_Morning.bat`):
   ```powershell
   .\.venv\Scripts\python.exe run_daily.py --morning
   ```
   - Defaults `--date` to today in `Asia/Kathmandu` and `--run-kind` to `config/pipeline.yaml` (`test`).
   - Idempotent: if today's `scraper` is already `done` and `editor` is `prepared` (or later), re-running `--morning` prints the current status and next action without re-scraping (pass `--force` to override).
   - Runs `scraper` (if it fails, waits 5 minutes and retries once; if it still fails, writes the error to `STATUS.md`).
   - Runs `editor` (`prepare`) and stops at **`needs writer`** (`status: paused`).
3. **Draft & validate script** — open `C:\news-auto` in Antigravity and paste Message **(A)** below.
4. **Review** `data/runs/<today>/script/script_preview.md` and the supporting source sentences shown in chat.
5. **Approve & finish** — paste Message **(B)** below into Antigravity (or run `approve` + `C:\news-auto\Run_Finish.bat`).
   - `--finish` refuses unless `editor == done` and the approval + validator fingerprint gate passes, then runs `voice`, `timeline`, `media --auto`, `motion`, `render`, and `qa` (also idempotent once `voice..qa` are `done` unless `--force` is passed).
   - Copies final outputs (`final.mp4`, `description.md`, `qa_report.json`, `storyboard.html` + `storyboard_frames/`, `contact_sheet.jpg`, `STATUS.md`, `README.txt`) into `C:\news-auto\data\latest\` **and** mirrors them to `outputs.mirror_dir` (`C:\Users\Chungba Sherpa\OneDrive\NepalNewsBot\latest`).
6. **Inspect output without playing video**:
   - Open `C:\news-auto\data\runs\<today>\STATUS.md` (shows `Source Health` at the top) or double-click `C:\Users\Chungba Sherpa\OneDrive\NepalNewsBot\latest\storyboard.html` (or `C:\news-auto\data\latest\storyboard.html`).
7. **Tweak visuals (optional)** — use `picker.html` + `--apply-picks` if you want to swap stock clips or cards (see Message **C** below).

---

## Ready-to-Paste Antigravity Messages

### (A) "Write today's script"

Paste this into Antigravity (with `C:\news-auto` open) after `--morning` completes (`needs writer`):

```text
Write today's script: read data/runs/<today>/script/editor_brief.md, write draft_script.json and selection.json, run `.\.venv\Scripts\python.exe -m src.agents.editor validate --date <today>` (maximum 3 revision rounds), then stop and show script_preview.md together with the supporting source sentences per scene. Do not approve.
```

*(Replace `<today>` with the run date `YYYY-MM-DD`, or leave `<today>` if obvious from `--status`.)*

---

### (B) "Approved"

After reading `script_preview.md` and confirming every fact matches the source sentences, paste this into Antigravity:

```text
Approved: run `.\.venv\Scripts\python.exe -m src.agents.editor approve --date <today>`, then run `.\.venv\Scripts\python.exe run_daily.py --finish --date <today>` and report data/runs/<today>/STATUS.md.
```

---

### (C) "Tweak visuals" (Optional, after `--finish`)

Every `--finish` run executes `media --auto` (picking scored candidates at or above `min_relevance`, creating headline cards below threshold, and keeping `media/review/picker.html`). To replace any shot visually:

1. Double-click `C:\news-auto\data\runs\<today>\media\review\picker.html` (or save your updated `picks.json` from the picker UI).
2. Paste this short message into Antigravity (or run the commands yourself):

```text
Tweak visuals for <today>: apply my picks from <path-to-picks.json> using `.\.venv\Scripts\python.exe -m src.agents.media --date <today> --apply-picks <path-to-picks.json>`, then re-run `motion`, `render`, and `qa` for <today> (and report STATUS.md).
```

*(Manually running `--only motion`, `--only render`, and `--only qa` updates `final.mp4` and `qa/storyboard.html`. To also refresh `data/latest/` and the OneDrive viewer mirror, reset `qa` and run `--finish --date <today>`.)*

---

## Double-Click Launchers (`C:\news-auto`)

All three batch launchers `cd` directly to `C:\news-auto` and pause at the end so the terminal stays open:

- `C:\news-auto\Run_Morning.bat` — runs `C:\news-auto\.venv\Scripts\python.exe run_daily.py --morning`
- `C:\news-auto\Run_Finish.bat` — runs `C:\news-auto\.venv\Scripts\python.exe run_daily.py --finish`
- `C:\news-auto\Status.bat` — runs `C:\news-auto\.venv\Scripts\python.exe run_daily.py --status`

---

## Windows Task Scheduler (`NepalNewsBot-Morning`)

The script `C:\news-auto\scripts\install_schedule.ps1` sets up a daily task named `NepalNewsBot-Morning`:
- **Project Root / Working Directory:** `C:\news-auto`
- **Schedule:** Daily at `07:00` local time (`StartWhenAvailable` enabled so missed starts run as soon as possible).
- **Wake PC:** **`WakeToRun` enabled** (`-WakeToRun` wakes the PC from Windows Sleep when wake timers are allowed in Power Options & BIOS).
- **Command:** `C:\news-auto\.venv\Scripts\python.exe run_daily.py --morning`
- **Log file:** `C:\news-auto\logs\morning_<YYYY-MM-DD>.log`
- **Logon mode:** `Interactive` (PC cannot be completely shut down, and your Windows user session must remain signed in).

Preview without registering:
```powershell
powershell -ExecutionPolicy Bypass -File C:\news-auto\scripts\install_schedule.ps1 -DryRun
```

Install (when ready):
```powershell
powershell -ExecutionPolicy Bypass -File C:\news-auto\scripts\install_schedule.ps1
```

Remove later:
```powershell
powershell -ExecutionPolicy Bypass -File C:\news-auto\scripts\uninstall_schedule.ps1
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

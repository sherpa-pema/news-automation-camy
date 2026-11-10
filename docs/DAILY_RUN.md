# Daily Run Operational Runbook

This document details the standard daily sequence for generating the automated Nepal news broadcast.

---

## Daily Operational Steps

### Step 1: Scrape Fresh News
Fetch RSS feeds and clean raw articles from verified news outlets:
```bash
python run_daily.py --date <YYYY-MM-DD> --only scraper
```

### Step 2: Prepare Editorial Brief
Cluster stories across outlets, calculate the speech rate word budget, and generate the editor brief:
```bash
python run_daily.py --date <YYYY-MM-DD> --only editor
```
*(The orchestrator will pause with exit code 3 and prompt for script drafting).*

---

### Step 3: Script Drafting by Antigravity
Prompt the Antigravity assistant in the chat with the following prompt:

```text
Read data/runs/<today>/script/editor_brief.md, write draft_script.json and selection.json, then run the validate command and fix failures until it passes (maximum 3 revision rounds)
```

The Antigravity assistant will:
1. Select 5 top news clusters from `shortlist.json`.
2. Write `data/runs/<YYYY-MM-DD>/script/selection.json`.
3. Draft `data/runs/<YYYY-MM-DD>/script/draft_script.json` obeying all TTS, grounding, and attribution rules.
4. Run `python -m src.agents.editor validate --date <YYYY-MM-DD>`.
5. Fix any validation errors until the validator produces `script.json` and `script_preview.md`.

---

### Step 4: Editorial Review & Human Approval
1. Inspect the human-readable summary in:
   `data/runs/<YYYY-MM-DD>/script/script_preview.md`
2. If satisfied, approve the script:
   ```bash
   python -m src.agents.editor approve --date <YYYY-MM-DD>
   ```
   *(This generates `approval.json` recording the SHA-256 hash and marks the editor stage as `done`).*

---

### Step 5: Downstream Synthesis & Assembly
Execute the remaining audio, alignment, and media stages:
```bash
python run_daily.py --date <YYYY-MM-DD>
```
Downstream stages verify the approved SHA-256 hash before running to guarantee no unapproved changes were introduced.

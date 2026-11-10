# Editor Agent Specification

## 1. Purpose
The Editor Agent acts as the broadcast news editor. It reviews the raw ingested articles, eliminates duplicate reporting across outlets, clusters articles about the same event, ranks clusters, generates a writer brief with calibrated speech budget, validates broadcast narration scripts deterministically, and records approvals.

In this architecture, the Antigravity agent acts as the writer (no external LLM calls, no API keys, no LLM client library). Python code prepares the brief and deterministically validates; the human/agent writes the script.

## 2. Stage States & Manifest Lifecycle
Stage states in the manifest (extending `contracts/manifest.schema.json`):
- `pending`: Unprocessed run awaiting preparation.
- `prepared`: Shortlist clustered, budget calculated, and `editor_brief.md` generated. *(Pause state)*
- `needs_revision`: Draft script failed one or more validation checks. *(Pause state)*
- `awaiting_approval`: Validation passed cleanly; `script.json` and `script_preview.md` generated, awaiting human approval. *(Pause state)*
- `done`: Approved via `approval.json` with matching SHA-256 hash.
- `failed`: Unrecoverable error.

`prepared`, `needs_revision`, and `awaiting_approval` are **PAUSE states**: the orchestrator stops cleanly with exit code 3 and a printed next-action instruction. Downstream stages (`voice`, `timeline`, `media`, `render`, `qa`) are strictly blocked until the editor stage is `done`.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/raw/articles.json`: Ingested articles from the Scraper Agent.
- **Outputs** in `data/runs/<YYYY-MM-DD>/script/`:
  - `shortlist.json`: Top 12 ranked story clusters with all member outlets and URLs.
  - `editor_brief.md`: Writer brief containing task rules, schema examples, date in words, budget, and cluster texts.
  - `draft_script.json`: Structured broadcast draft written by the writer.
  - `selection.json`: Editorial rationale for chosen and skipped clusters written by the writer.
  - `validation_report.json`: Deterministic check results and metrics.
  - `script.json`: Validated broadcast script written **only** by the validator.
  - `script_preview.md`: Formatted preview for editorial review.
  - `approval.json`: Approval ledger `{approved_at, script_sha256, approved_by: "user"}`.

## 4. CLI Commands
```bash
# 1. Cluster articles and generate writer brief
python -m src.agents.editor prepare --date <YYYY-MM-DD>

# 2. Deterministically validate draft script
python -m src.agents.editor validate --date <YYYY-MM-DD>

# 3. Approve validated script and advance stage to done
python -m src.agents.editor approve --date <YYYY-MM-DD>

# Or via daily orchestrator (runs prepare if pending, validate if draft exists):
python run_daily.py --date <YYYY-MM-DD> --only editor
```

## 5. Ground Rules & Deterministic Validation
- **Opinion & Commentary Filter**: URL patterns, title patterns, author patterns, and first-person pronouns (>= 5 occurrences) identify opinion pieces. Opinion pieces are marked with `is_opinion: true` and excluded from the news shortlist by default (`config/editor.yaml` `allow_opinion: false`), and listed separately under Excluded Opinion Pieces in `editor_brief.md`.
- **Temporal & Event Timing**: Never write preview wording (e.g. "is scheduled to face", "will begin at five") for an event whose start time has already passed at writing time. A source that says an event is scheduled does not show it happened, so never rewrite a preview as past tense; skip it. In prepare, articles that announce a future event whose time has passed (will, scheduled, set to, to face, to host, plus a time) are flagged as `stale_preview`, excluded from the shortlist by default (`config/editor.yaml` `allow_stale_preview: false`), and listed separately under Excluded Stale Previews in `editor_brief.md`.
- **No Verbatim Text**: Sliding window check; any 8 consecutive words shared between narration and cluster text fails validation.
- **Number Grounding**: Digits and spelled-out words are normalized, including currency prefixes (e.g. "Rs105.92") and spoken decimals ("point"). Every numeric figure mentioned in narration must exist in the cluster's article text. A number within 2% is a warning only when preceded by approximation modifiers (about, around, over, more than, nearly); otherwise it fails. Invented figures fail.
- **TTS-Friendly Rules**: Digits (0-9), %, $, &, /, parentheses (), brackets [], and emojis are strictly forbidden in narration text. All numbers must be spelled out (e.g., "fifty thousand rupees", "eighty percent").
- **Headline Constraints**: Max 10 words, sentence case, no ending period. Headlines may use digits where appropriate (e.g., "COP31", "NPL Season 3") as they are visual screen titles, not read aloud.
- **Visual Keywords**: Exactly 3 to 5 generic stock terms (no names of real people or brands). Visual keywords must strictly match the subject matter of the story (e.g., wild cliff honey hunting is not beekeeping; use "mountain cliffs", "wild honeycombs").
- **Mandatory Attribution & Evidence**: Story scenes must include all cluster outlet names and member URLs. Every narration sentence must be mapped to supporting evidence URLs.
- **Sensitivity Gate**: Crime, fatal accidents, deaths, abuse, or stories involving minors MUST set `sensitive: true`.
- **Word Budget**: Calibrated against measured words-per-second from real voice reports to ensure total audio duration between 2:00 and 3:00 (120 to 180 seconds).

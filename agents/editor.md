# Editor Agent Specification

## 1. Purpose
The Editor Agent acts as the broadcast news editor. It reviews raw ingested articles, eliminates duplicate reporting across outlets (including English and Devanagari coverage of the same event), clusters and chronologically tracks developing updates, applies 48-hour cross-day deduplication memory, generates a writer brief with a calibrated word/syllable/pause speech budget, optionally invokes `llm_writer` (`write`) with surgical scene-level self-correction, deterministically validates broadcast narration scripts, and records approvals.

Both agent-assisted writing and standalone LLM writing (`python -m src.agents.editor write --date YYYY-MM-DD`, using configured API keys via `src/utils/llm_writer.py`) produce `draft_script.json` + `selection.json`, which are then verified deterministically by `EditorAgent.validate_draft()`.

## 2. Stage States & Manifest Lifecycle
Stage states in the manifest (extending `contracts/manifest.schema.json`):
- `pending`: Unprocessed run awaiting preparation.
- `prepared`: Shortlist clustered, budget calculated, and `editor_brief.md` generated. *(Pause state)*
- `needs_revision`: Draft script failed one or more validation checks. *(Pause state)*
- `awaiting_approval`: Validation passed cleanly; `script.json` and `script_preview.md` generated, awaiting human approval. *(Pause state)*
- `done`: Approved via `approval.json` with matching SHA-256 hash and `validator_fingerprint`.
- `failed`: Unrecoverable error.

`prepared`, `needs_revision`, and `awaiting_approval` are **PAUSE states**: the orchestrator stops cleanly with exit code 3 and a printed next-action instruction. Downstream stages (`voice`, `timeline`, `media`, `render`, `qa`) are strictly blocked until the editor stage is `done`.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/raw/articles.json`: Ingested articles from the Scraper Agent.
  - Up to `deduplication.cross_day_lookback_days` prior run directories (`script/script.json`, `script/approval.json`, `script/selection.json`) for cross-day deduplication.
  - Historical `voice/voice_report.json` + `script/script.json` files for calibrating `words_per_second` and `syllables_per_second`.
- **Outputs** in `data/runs/<YYYY-MM-DD>/script/`:
  - `shortlist.json`: Top 12 ranked story clusters with all member outlets, URLs, chronological `superseded` flags, `has_superseded_updates`, and `covered_recently` / `covered_recently_reason` metadata.
  - `budget.json`: Frozen speech-rate calibration (`words_per_second`, `syllables_per_second`, `comma_pause_sec`, `period_pause_sec`, scene and total word limits).
  - `editor_brief.md`: Writer brief containing task rules, schema examples, date in words, budget, cluster texts, `[OLDER UPDATE — DO NOT USE FOR NUMBERS]`, and `[COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]` tags.
  - `draft_script.json`: Structured broadcast draft written by the writer or `llm_writer`.
  - `selection.json`: Editorial rationale for chosen and skipped clusters.
  - `validation_report.json`: Deterministic check results (`errors`, `warnings`, `structured_errors` / `structured_warnings` with `scene_index` and `scene_id`, `scene_results`, `validator_fingerprint`, and metrics).
  - `script.json`: Validated broadcast script written **only** by the validator.
  - `script_preview.md`: Formatted preview for editorial review.
  - `approval.json`: Approval ledger `{approved_at, script_sha256, validator_fingerprint, approved_by: "user"}`.

## 4. CLI Commands
```bash
# 1. Cluster articles and generate writer brief + frozen budget
python -m src.agents.editor prepare --date <YYYY-MM-DD>

# 2. (Optional) Standalone LLM script generation with up to 5 rounds of self-correction
python -m src.agents.editor write --date <YYYY-MM-DD>

# 3. Deterministically validate draft script
python -m src.agents.editor validate --date <YYYY-MM-DD>

# 4. Approve validated script and advance stage to done
python -m src.agents.editor approve --date <YYYY-MM-DD>

# Or via daily orchestrator (runs prepare if pending, validate if draft exists):
python run_daily.py --date <YYYY-MM-DD> --only editor
```

## 5. Clustering, Supersession & Stale Previews (`src/utils/editor_cluster.py`, `src/utils/llm_curator.py`)
- **Hybrid LLM Semantic Curation & Multi-Source Synthesis**: When enabled (`config/editor.yaml` `llm_curator.enabled: true`), `curate_and_cluster_with_llm()` pre-filters raw news into an efficient compact digest and leverages the configured LLM (`zhipu`, `gemini`, `anthropic`, `openai`, `ollama`) to semantically group articles covering the same event across English and Nepali outlets, filter out irrelevant foreign wire articles, assign true contextual categories, rate national editorial priority (1–10), and generate explicit `synthesis_notes` detailing what complementary facts each outlet provides. If LLM curation is unconfigured, disabled, or fails, the agent seamlessly falls back to deterministic heuristic clustering without crashing.
- **Cross-Script Entity & Numeric Spine Matching**: In heuristic clustering, `articles_are_similar()` normalizes entity tokens via `config/editor.yaml` `entity_aliases` (parties, institutions, locations, leaders across English & Devanagari) and converts Devanagari digits (`०-९`) to ASCII (`0-9`). Two articles merge if they pass `SequenceMatcher` / token overlap thresholds OR share `>= 2` canonical entities AND `>= 1` non-trivial numeric token (excluding `0`, `1`, and the publication year) within a 12-hour publication window.
- **Intra-Cluster Temporal Supersession**: Articles inside each cluster are ordered by `published_at` descending (newest/primary first, breaking ties by `source_trust`). When articles report differing numeric figures, older articles are marked `"superseded": true` in `shortlist.json` and labeled `[OLDER UPDATE — DO NOT USE FOR NUMBERS]` in `editor_brief.md`; the newest article becomes `primary_title`.
- **48-Hour Cross-Day Deduplication**: Up to `deduplication.cross_day_lookback_days` (default `2`) prior runs are inspected. Matching candidate clusters receive `deduplication.cross_day_repeat_penalty` (default `-600`) unless they include an article published after the prior run's `approved_at` with new numeric data. Surviving matches are flagged `covered_recently: true` and `[COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]`.
- **Opinion & Commentary Filter**: URL, title, author, and first-person pronoun heuristics exclude opinion pieces (`allow_opinion: false`).
- **Nepal Time-of-Day & Explicit Stale Previews**: `detect_stale_preview()` evaluates preview language against `+05:45` NPT both for explicit clock times and Nepal time-of-day anchors (`"this morning"` / `"today morning"` / `"आज बिहान"` stale when `hour >= 12`; `"this afternoon"` / `"आज दिउँसो"` stale when `hour >= 17`; `"nepse to open"` / `"market set to open"` stale when `hour >= 15`). Excluded by default (`allow_stale_preview: false`).

## 6. Deterministic Validation & Speech Budgeting (`src/utils/editor_validator.py`, `src/utils/editor_budget.py`)
- **No Verbatim Text**: Sliding window check; any 8 consecutive words shared between narration and cluster text fails validation.
- **Unit-Scale Number Grounding & Bikram Sambat Tolerance**:
  - Recognizes South Asian and Western unit multipliers across English and Devanagari (`thousand`/`हजार` = 1e3, `lakh`/`लाख`/`lac` = 1e5, `million` = 1e6, `crore`/`करोड` = 1e7, `arab`/`अर्ब`/`billion` = 1e9, `kharba`/`खर्ब` = 1e11), compound pairs (e.g., `"१ करोड ५० लाख"` -> `15000000.0`), currency prefixes, and spoken decimals (`"one point five crore"`).
  - Compares both raw and scale-normalized canonical values, applying the 2% approximation-modifier warning rule (`about`, `around`, `over`, `more than`, `nearly`) to scale-normalized values as well.
  - Whitelists valid Bikram Sambat years (`run_year + 56` through `run_year + 58`, e.g., `2082`/`2083`/`2084` for `2026`, including spoken `"twenty eighty three"` / `"two thousand eighty three"`) when used in fiscal/date contexts.
- **TTS-Friendly Rules & Acronym Evasion**:
  - Digits (`0-9`), `%`, `$`, `&`, `/`, `()`, `[]`, and emojis are forbidden in narration.
  - `acronym_blocklist` in `config/editor.yaml` is checked case-insensitively across all narration tokens (catching `"Nea"`, `"Nrb"`, `"Nepse"`, etc.) unless mapped in `config/pronunciations.yaml`.
  - Capitalized non-sentence-initial narration tokens absent from `config/editor_whitelist.yaml` and `config/pronunciations.yaml` emit structured TTS pronunciation warnings in `validation_report.json`.
- **Culturally Safe `visual_keywords`**: Strictly 3–5 generic stock terms; fails if any term matches `visual_keyword_rules.blocked_western_terms` (e.g., `"dollar bills"`, `"capitol dome"`, `"nyse"`).
- **Headline Constraints**: Max 10 words, sentence case, no ending period (digits permitted on screen).
- **Mandatory Attribution & Sensitivity Gate**: Story scenes must include all cluster outlet names/URLs and per-sentence evidence URLs; crime/fatalities/abuse/minors require `sensitive: true`.
- **Syllable + Pause Duration Budget & Surgical Self-Correction**:
  - `editor_budget.py` calibrates `words_per_second` and `syllables_per_second` (default `3.3` sps) with `comma_pause_sec: 0.25` and `period_pause_sec: 0.55` in `budget.json`, estimating per-scene `est_duration_sec` during `validate`.
  - `validation_report.json` attaches `"scene_index"` (`0–6` or `null`) and `"scene_id"` (`"intro"`, `"story_1"`, etc.) to every `structured_errors` / `structured_warnings` entry while preserving human-readable `errors` / `warnings` lists.
  - `llm_writer.py` runs up to 5 validation rounds: when global schema/word-budget/diversity checks pass and errors are isolated to specific scene indices, passing scenes are locked and only the failing scene(s) are repaired and spliced back into `draft_script.json`.


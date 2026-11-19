# Pipeline Gate & Safety Map (`nepal-news-bot`)

This document maps every deterministic gate and blocking check across `nepal-news-bot`: where it is implemented (file and function), the exact refusal/error message emitted, and the pytest test cases verifying it.

---

## 1. Run-Level & Pre-Stage Execution Gates

### 1.1 Abandoned Run Refusal
A run marked `"status": "abandoned"` cannot execute or reset any stage.

| Location (File & Function) | Exact Refusal Message | Covering Tests |
| --- | --- | --- |
| `src/agents/editor.py` -> `EditorAgent.run`, `prepare`, `validate_draft`, `approve` | `Run '<date>' is abandoned. No stage may run on it.` | `tests/test_editor.py::test_abandoned_run_refuses_all_stages` |
| `src/orchestrator.py` -> `Orchestrator.execute_stage`, `run_pipeline`, `run_morning` | `Run '<date>' is abandoned. No stage may run on it.` | `tests/test_editor.py::test_abandoned_run_refuses_all_stages` |
| `src/orchestrator.py` -> `Orchestrator.reset_stage` | `Run '<date>' is abandoned. No stage may be reset.` | `tests/test_editor.py::test_reset_stage_logs_to_manual_edits_and_validates_schema` |
| `src/utils/daily_ops.py` -> `verify_editor_gate` | `Run '<date>' is abandoned. No stage may run on it.` | `tests/test_editor.py::test_abandoned_run_refuses_all_stages` |

### 1.2 Pause States & Editor-Done Requirement Before `--finish` / Downstream Stages
The pipeline pauses between human-in-the-loop editor checkpoints (`prepared`, `needs_revision`, `awaiting_approval`) and refuses all downstream execution until `editor == "done"` and `script/script.json` exists.

| Check | File & Function | Exact Refusal Message | Covering Tests |
| --- | --- | --- | --- |
| Pipeline pause checkpoint (`prepared`, `needs_revision`, `awaiting_approval`) | `src/orchestrator.py` -> `Orchestrator.execute_stage`; CLI `run_daily.py` -> `main` | Raises `PipelinePausedException(status, next_action)`; `run_daily.py` prints `[PIPELINE PAUSED: <STATUS>]` and exits with code `3`. | `tests/test_editor.py::test_pause_states_exit_with_code_3_and_block_downstream`; `tests/test_daily_ops.py::test_morning_stops_at_needs_writer_and_retries_scraper` |
| Downstream stage requires `editor == "done"` | `src/orchestrator.py` -> `Orchestrator.execute_stage` | `Refusing to execute downstream stage '<stage_id>': Editor stage status is '<editor_status>', but must be 'done'.` | `tests/test_editor.py::test_pause_states_exit_with_code_3_and_block_downstream` |
| Missing manifest before `--finish` | `src/utils/daily_ops.py` -> `verify_editor_gate`; `src/orchestrator.py` -> `Orchestrator.run_finish` | `Manifest not found at <path>.` (or `Manifest not found at <path>. Run --morning first.`) | `tests/test_daily_ops.py::test_finish_refuses_before_approval` |
| `--finish` refuses unless `editor == "done"` | `src/utils/daily_ops.py` -> `verify_editor_gate`; `src/orchestrator.py` -> `Orchestrator.run_finish` | `Refusing to run downstream stages: editor stage status is '<editor_status>', must be 'done' (run validate and approve first).` | `tests/test_daily_ops.py::test_finish_refuses_before_approval` |
| Missing validated script before `--finish` | `src/utils/daily_ops.py` -> `verify_editor_gate` | `Missing validated script at <run_dir>/script/script.json.` | `tests/test_daily_ops.py::test_finish_refuses_before_approval` |
| Legacy root `script.json` rejected | `src/agents/voice.py` -> `VoiceAgent.run`; `src/agents/timeline.py` -> `TimelineAgent.run`; `src/agents/media.py` -> `MediaAgent.run` | `Missing required script file: <run_dir>/script/script.json` (Voice/Timeline); `Missing required script.json at <path>` (Media) | `tests/test_editor.py::test_legacy_script_path_rejected_and_script_subfolder_required` |

### 1.3 Fixture Script Refusal (`origin == "fixture"` without `--allow-fixture`)

| File & Function | Exact Refusal Message | Covering Tests |
| --- | --- | --- |
| `src/orchestrator.py` -> `Orchestrator.execute_stage` | `Refusing to execute stage '<stage_id>' on fixture script without --allow-fixture flag.` | `tests/test_media.py::test_orchestrator_refuses_fixture_script_without_flag` |
| `src/utils/daily_ops.py` -> `verify_editor_gate` | `Refusing to run downstream stages on fixture script without --allow-fixture.` | `tests/test_daily_ops.py::test_finish_refuses_when_gate_fails` |
| `src/agents/media.py` -> `MediaAgent.run`, `run_candidates`, `apply_picks` | `Refusing to run on fixture script data without --allow-fixture.` | `tests/test_media.py::test_orchestrator_refuses_fixture_script_without_flag` |

---

## 2. Script Validation, Approval Hash & Validator Fingerprint Gates

### 2.1 Deterministic Script Validation Gate (`validate_script_draft`)
Only `EditorAgent.validate_draft` (`src/agents/editor.py`) writes `script/validation_report.json` and promotes a passing draft to `script/script.json` (`awaiting_approval`). Failures move stale `script.json`/`script_preview.md` to `script/_stale/` and set `editor` to `needs_revision`.

| Check | File & Function | Exact Error / Refusal Message Pattern | Covering Tests |
| --- | --- | --- | --- |
| Missing/unreadable draft or schema mismatch (`contracts/script.schema.json`) | `src/utils/editor_validator.py` -> `validate_script_draft` | `Draft script file not found at <path>`; `Failed to parse draft JSON: <err>`; `Schema validation failed: <err>` | `tests/test_editor.py::test_schema_validation_rejects_missing_field_and_wrong_type` |
| Exact 7-scene structure | `src/utils/editor_validator.py` -> `validate_script_draft` | `Expected exactly 7 scenes (intro + 5 stories + outro), got <N>` | `tests/test_editor.py::test_schema_validation_rejects_missing_field_and_wrong_type` |
| Headline & keyword rules | `src/utils/editor_validator.py` -> `validate_script_draft` | `Headline exceeds 10 words (<N> words): '<headline>'`; `Headline must not end with a period: '<headline>'`; `Visual keywords count must be 3 to 5 (got <N>)` | `tests/test_editor.py::test_prepare_step_writes_shortlist_and_brief_with_budget` |
| Word count & sentence count budgets | `src/utils/editor_validator.py` -> `validate_script_draft` | `Intro word count <N> outside 15-40 words`; `Outro word count <N> outside 15-40 words`; `Story scene word count <N> outside 20-60 words`; `Total word count <N> outside budget range [<min>, <max>]`; `Story narration must have 2 to 4 sentences (got <N>)` | `tests/test_editor.py::test_frozen_budget_ignores_newer_voice_report_after_prepare` |
| TTS text hygiene (no digits, symbols, brackets, non-ASCII) | `src/utils/editor_validator.py` -> `validate_script_draft` | `Digits found in narration (spell numbers out): <list>`; `Banned symbols in narration: <set>`; `Parentheses or brackets found in narration`; `Non-ASCII characters or emojis found: <set>` | `tests/test_editor.py::test_tts_rule_validator_rejects_digits_symbols_and_parens` |
| Grounding & evidence URL binding to shortlist | `src/utils/editor_validator.py` -> `validate_script_draft` | `Missing source outlet`; `Missing source_urls`; `Missing evidence mappings`; `Evidence sentence_index <idx> out of range [0, <max>]`; `Evidence source_url '<url>' does not belong to any shortlisted cluster`; `source_urls <list> do not match any shortlisted cluster`; `Evidence source_url '<url>' not part of cluster URLs` | `tests/test_editor.py::test_evidence_from_unshortlisted_cluster_fails_validation` |
| Sensitive topic flag enforcement | `src/utils/editor_validator.py` -> `validate_script_draft` | `Cluster text contains sensitive trigger word '<trig>', but sensitive is false` | `tests/test_editor.py::test_sensitive_trigger_check_fails_if_sensitive_false` |
| Verbatim 8-word copy ban | `src/utils/editor_validator.py` -> `validate_script_draft` | `Verbatim 8-word overlap with source text: '<phrase>'` | `tests/test_editor.py::test_verbatim_check_catches_8_word_copy` |
| Grounded numbers & 2% approximation modifier rule | `src/utils/editor_validator.py` -> `validate_script_draft` | `Number <n> is within 2% of source number <c> but lacks approximation modifier (about, around, over, more than, nearly)`; `Invented number <n> in narration does not appear in source cluster` | `tests/test_editor.py::test_number_normalization_and_invented_figure`; `test_currency_prefix_and_spoken_decimals`; `test_number_tolerance_and_approximation_modifier` |
| Stale artifact quarantine on validation failure | `src/agents/editor.py` -> `EditorAgent.validate_draft` | Moves prior `script.json`/`script_preview.md` to `script/_stale/`, writes `# VALIDATION FAILED` preview, sets `editor="needs_revision"` | `tests/test_editor.py::test_validation_failure_moves_stale_files_and_writes_failed_preview` |

### 2.2 Approval Sign-Off Gate (`EditorAgent.approve`)
Only `EditorAgent.approve` (`src/agents/editor.py`) writes `script/approval.json` with `approved_by: "user"`, `script_sha256`, and the 7-key `script_validation_fingerprint`.

| Check | File & Function | Exact Refusal Message | Covering Tests |
| --- | --- | --- | --- |
| Missing `script.json` | `src/agents/editor.py` -> `EditorAgent.approve` | `Missing validated script.json at <path>. Run validate first.` | `tests/test_editor.py::test_approve_refuses_when_validation_failed_or_no_script` |
| Missing `validation_report.json` | `src/agents/editor.py` -> `EditorAgent.approve` | `Missing validation_report.json at <path>. Run validate first.` | `tests/test_editor.py::test_approve_refuses_when_validation_failed_or_no_script` |
| Latest validation failed or stage in `needs_revision` | `src/agents/editor.py` -> `EditorAgent.approve` | `Cannot approve script: latest validation failed with <N> error(s).`; `Cannot approve script: latest validation failed and stage is in 'needs_revision' state.` | `tests/test_editor.py::test_approve_refuses_when_validation_failed_or_no_script` |
| Validator or config changed since validation (`extract_script_validation_fingerprint`) | `src/agents/editor.py` -> `EditorAgent.approve` | `Validator fingerprint mismatch! The editor validator or config has changed since validation.\nRecorded: ...\nCurrent:  ...\nRe-run validate before approving.` | `tests/test_editor.py::test_validator_fingerprint_mismatch_refuses_approval`; `test_only_editor_approve_and_validate_write_approval_or_validation_report` |

### 2.3 Post-Approval Integrity Gate (`script_sha256` & 7-Key `script_validation_fingerprint`)
Enforced independently at the orchestrator boundary AND inside every downstream agent (`VoiceAgent`, `TimelineAgent`, `MediaAgent`, `MotionAgent`, `RenderAgent`, `QAAgent`). Only the 7 script-validation hashes (`validator_sha256`, `editor_config_sha256`, `editor_whitelist_sha256`, `sensitive_triggers_sha256`, `pronunciations_sha256`, `imported_modules_sha256`, `script_schema_sha256`) block execution; downstream contract schemas (`contracts_hash`) are informational and never block approval.

| Check | File & Function | Exact Refusal Message | Covering Tests |
| --- | --- | --- | --- |
| Missing `approval.json` | `src/orchestrator.py` -> `Orchestrator.execute_stage`; `src/utils/daily_ops.py` -> `verify_editor_gate`; `VoiceAgent.run`; `TimelineAgent.run`; `MediaAgent.run` / `run_candidates` / `apply_picks`; `MotionAgent.run`; `RenderAgent._verify_gate`; `QAAgent.run` | `script changed after approval, re-run validate and approve` (`verify_editor_gate`: `Missing script sign-off file; run validate and approve first.`; QA Check 1: `Missing approval.json for editor script`) | `tests/test_daily_ops.py::test_finish_refuses_when_gate_fails`; `tests/test_editor.py::test_hand_editing_script_detected_after_approval` |
| `script_sha256` mismatch (post-approval edit) | Same 7 call sites above | `script changed after approval, re-run validate and approve` (QA Check 1: `Script SHA-256 mismatch (approval: ..., current: ...)`) | `tests/test_daily_ops.py::test_finish_refuses_when_gate_fails`; `test_finish_is_idempotent_unless_forced_and_still_checks_gate`; `tests/test_editor.py::test_hand_editing_script_detected_after_approval` |
| `approval.json` 7-key validator fingerprint mismatch | Same 7 call sites + `src/utils/editor_validator.py` -> `verify_script_approval_gate` | `validator or editor config changed after approval, re-run validate and approve` (`verify_script_approval_gate`: `Fingerprint mismatch on <key>: ...`; QA Check 1: `Validator fingerprint mismatch between approval and environment`) | `tests/test_daily_ops.py::test_finish_refuses_when_gate_fails`; `tests/test_editor.py::test_downstream_stage_refuses_on_fingerprint_mismatch`; `test_editing_whitelist_invalidates_approval`; `tests/test_voice.py::test_voice_agent_run_approval_fingerprint_gate`; `tests/test_timeline.py::test_timeline_agent_run_approval_fingerprint_gate`; `tests/test_media.py::test_media_agent_run_approval_fingerprint_gate` |
| `validation_report.json` invalid or fingerprint mismatch | `src/utils/daily_ops.py` -> `verify_editor_gate`; `Orchestrator.execute_stage`; `VoiceAgent.run`; `TimelineAgent.run`; `MediaAgent.run` / `apply_picks`; `MotionAgent.run`; `RenderAgent._verify_gate` | `validator or editor config changed since validation, re-run validate and approve` (`verify_editor_gate` invalid report: `Latest validation report contains errors; re-run validate and approve.`) | `tests/test_editor.py::test_editing_media_manifest_schema_does_not_invalidate_approval`; `test_orchestrator_execute_stage_ignores_downstream_schema_changes` |

---

## 3. Downstream Stage, Schema, QA & Cleanup Gates

### 3.1 Contract & Stage-Specific Gates

| Stage | Check & File/Function | Exact Refusal / Error Message | Covering Tests |
| --- | --- | --- | --- |
| Schema Contract (`BaseAgent`) | `src/agents/base.py` -> `BaseAgent.validate` + `src/orchestrator.py` -> `execute_stage` | `Expected output file not found on disk: <rel_path>`; `Schema validation failed for <json>: <err>`; Orchestrator raises `Contract validation failed: <err>` | `tests/test_editor.py::test_manifest_schema_validation_negative_cases`; `tests/test_scraper.py::test_articles_schema_validation_negative_cases`; `tests/test_timeline.py::test_timeline_schema_validation_negative_cases`; `tests/test_media.py::test_media_manifest_schema_validation_negative_cases` |
| Scraper | `src/agents/scraper.py` -> `ScraperAgent.run` (`min_articles` threshold), `_is_url_allowed_by_robots` | `INSUFFICIENT_SOURCES: Scraped <N> articles, below minimum threshold (<min>).`; `articles.json schema validation failed: <err>` | `tests/test_scraper.py::test_broken_source_does_not_fail_whole_run`; `test_robots_disallowed_paths_skipped`; `tests/test_daily_ops.py::test_source_health_statuses_degraded_two_runs_and_non_blocking` |
| Voice | `src/agents/voice.py` -> `VoiceAgent.validate` (48kHz, mono, 90-220s duration, silence gap <=1.5s, peak <= -0.05 dBFS, `speaker_consistency` cosine >= 0.72 & F0 <= 10%, unverified pronunciation gate, linear_strict loudness) | `Master sample rate must be 48000 Hz...`; `Total narration duration ... outside valid window (90s - 220s)`; `Scene audio ... is empty or too small`; `Scene audio ... contains internal silence gap of ...s > 1.5s`; `Digital clipping detected in narration.wav...`; `Speaker consistency gate failed for item '<id>': similarity ... < threshold ...`; `Speaker pitch gate failed for item '<id>': F0 ... differs from reference ...`; `Unverified pronunciation gate failed: unverified entries applied <words>...` | `tests/test_voice.py::test_validation_catches_empty_scene`; `test_validation_catches_silence_gap`; `test_speaker_consistency_gate_fails_on_low_similarity`; `test_speaker_consistency_gate_fails_on_pitch_deviation`; `test_unverified_pronunciation_gate_blocks_run`; `test_unverified_pronunciation_gate_passes_when_allowed`; `test_voice_agent_linear_strict_default` |
| Voice Note | Configuration note: `voice`, `speed`, and `instruction_schema` keys are inert on the TTS zero-shot cloning path (voice identity is anchored entirely by `config/voice/reference.wav`). Daemon compatibility handshake checks `model_path`, `device`, `dtype`, `text_normalization` on `/health` and falls back cleanly to subprocess worker on mismatch. | N/A (Documentation & Warning) | `tests/test_voice.py::test_daemon_health_mismatch_falls_back` |
| Timeline | `src/agents/timeline.py` -> `TimelineAgent.run`, `validate` (scene set equality, 20ms sum sync, 1ms partition continuity, monotonic start_sec, unplaced ratios <=15% scene / <=5% run with weighted interpolation fallback, monotonicity, SRT syntax, captions schema, caption audit 0 late > 50ms) | `Scene duration sum (...) does not match narration.wav (...) within 20ms tolerance ...`; `Scene partition gap/overlap between ...`; `Scene set mismatch between script.json and scene_durations.json...`; `Monotonicity violation in scene ...`; `captions.json contract validation failed: ...`; `Generated timeline does not conform to schema contract: ...` | `tests/test_timeline.py::test_duration_mismatch_raises_error`; `test_scene_set_mismatch_raises`; `test_scene_order_validated`; `test_unplaced_words_ratio_warning_and_degradation`; `test_srt_formatting_and_syntax_validation`; `test_audit_captions_detects_late_starts_and_violations` |
| Media | `src/agents/media.py` -> `MediaAgent._validate_manifest_shots`, `apply_picks` (tiling <=0.05s, SHA-256, blocklisted domain, license allowlist) | `Scene <id> tiling mismatch: sum of shots ... != scene duration ...`; `SHA-256 mismatch for <file>...`; `Shot uses blocklisted domain: <url>` / `Shot <id> uses blocklisted domain: ...`; `Shot has disallowed license: <lic>` / `Shot <id> candidate <cid> has disallowed license: '<lic>'` | `tests/test_media.py::test_shot_planning_tiling_and_snapping`; `test_blocklisted_news_domains`; `test_license_policy_filtering`; `tests/test_media_review.py::test_license_policy_enforcement_rejects_cc_by_sa`; `test_apply_picks_download_or_norm_failure_creates_card_and_syncs_render_plan` |
| Motion & Render | `src/agents/motion.py` -> `MotionAgent.run`; `src/agents/render.py` -> `RenderAgent._verify_gate`, `run` | Provenance/approval/fingerprint gate errors (Section 2.3) + missing `timeline.json`/`render_plan.json`/`narration.wav` `FileNotFoundError` | `tests/test_motion.py::test_motion_agent_plan_creation_and_watermark`; `tests/test_render.py::test_render_agent_cache_invalidation_and_card_concat` |
| QA | `src/agents/qa.py` -> `QAAgent.run` (Checks 1-9: `provenance`, `final_mp4_streams`, `loudness`, `blackdetect`, [`freezedetect` warning], `story_lower_thirds`, `overlay_layout`, `captions`, `description_md`) | `QA inspection failed on checks: <failed_checks>. Recommended redo stage: <redo_stage>` | `tests/test_qa.py::test_qa_agent_passes_on_valid_fixture`; `test_qa_catches_duration_mismatch`; `test_qa_catches_missing_audio_stream`; `test_qa_catches_unintended_black_frame` |
| Cleanup Barrier | `src/utils/daily_ops.py` -> `_is_protected_path`, `plan_cleanup`, `execute_cleanup` | Never plans/deletes `data/media_library`, `.env`, `models`, `vendor`, any run inside `keep_runs` (7), or `script/`, `description.md`, `final.mp4`, `qa_report.json`, `manifest.json`, `credits.md`, `media_manifest.json`, `narration.wav`. | `tests/test_daily_ops.py::test_cleanup_dry_run_allowed_and_protected_paths` |

---

## 4. Gate-Related Changes Since Baseline Commit `678d867`

### `git diff 678d867 --stat`
```text
 .gitignore                     |   6 +
 config/pipeline.yaml           |   3 +
 contracts/manifest.schema.json |   2 +-
 docs/DAILY_RUN.md              | 136 ++++++++++---
 run_daily.py                   | 222 ++++++++++++++++++--
 src/agents/editor.py           | 108 +++++++++-
 src/agents/media.py            | 388 ++++++++++++++++++++++++++++-------
 src/agents/motion.py           | 190 +++++++++++++++++-
 src/agents/qa.py               |  45 ++++-
 src/agents/render.py           |  66 +++++-
 src/agents/scraper.py           |  59 ++++--
 src/orchestrator.py            | 445 +++++++++++++++++++++++++++++++++--------
 src/utils/editor_brief.py      |   4 +
 src/utils/editor_budget.py      |  61 ++++--
 src/utils/manifest.py          | 199 +++++++++++++++++-
 src/utils/media_cache.py       |  22 +-
 src/utils/media_review.py       |   2 +-
 tests/test_editor.py           | 222 +++++++++++++++++++-
 tests/test_media_review.py     | 229 +++++++++++++++++++++
 tests/test_render.py           |  95 +++++++++
 20 files changed, 2248 insertions(+), 256 deletions(-)
```
*(Plus untracked Step 8 / 8B files added after `678d867`: `src/utils/daily_ops.py`, `src/utils/storyboard.py`, `tests/test_daily_ops.py`, `docs/GATES.md`, `scripts/install_schedule.ps1`, `scripts/uninstall_schedule.ps1`, `Run_Morning.bat`, `Run_Finish.bat`, `Status.bat`.)*

### Gate-Related Functions Changed Since `678d867`
- `src/utils/editor_validator.py` -> *(unchanged since `678d867` — validator logic and 7-key `extract_script_validation_fingerprint` were already established at baseline).*
- `src/agents/editor.py` -> `EditorAgent.prepare`: Freezes speech-rate calibration in `script/budget.json` once, propagates `run_kind`, reconciles manifest status, and refreshes `STATUS.md` so `--morning` leaves a stable budget across days.
- `src/agents/editor.py` -> `EditorAgent.validate_draft`: Enforces frozen `script/budget.json` instead of recomputing from newer runs' `voice_report.json`, stamps `run_kind`, reconciles manifest state, and updates `STATUS.md` so post-prepare renders cannot break validation.
- `src/agents/editor.py` -> `EditorAgent.approve`: Checks `extract_script_validation_fingerprint` on `validation_report.json` against the environment (ignoring downstream schema hashes), reconciles manifest status, and updates `STATUS.md`.
- `src/orchestrator.py` -> `Orchestrator.execute_stage`: Compares the 7-key `extract_script_validation_fingerprint` for both `approval.json` and `validation_report.json`, routes `--media-auto`, and writes `STATUS.md` at pause states so downstream contract edits never falsely block approved scripts.
- `src/orchestrator.py` -> `Orchestrator.run_pipeline`: Wraps execution in `run_process_lock` and calls `reconcile_manifest_status` on finish or exception so `--only` or crashed runs never remain stuck in `running`.
- `src/orchestrator.py` -> `Orchestrator.reset_stage`: Allows recovering `interrupted` runs alongside `failed`/`done`/`paused`, logs reset entries to `manual_edits`, and refreshes `STATUS.md`.
- `src/orchestrator.py` -> `Orchestrator.run_morning` & `Orchestrator.run_finish`: Added gated daily commands (single 5-min scraper retry, pause at `needs writer`, read-only `verify_editor_gate` before `--finish`, idempotency with `--force`, and `publish_to_latest`).
- `src/utils/manifest.py` -> `compute_manifest_status`, `reconcile_manifest_status`, `is_run_process_active`, `run_process_lock`: Computes top-level manifest status from stage states plus PID lock liveness, turning >30-minute lockless `running` runs into `interrupted`.
- `src/utils/daily_ops.py` -> `verify_editor_gate`, `_is_protected_path`, `compute_source_health`: Added standalone read-only approval/hash/fingerprint/fixture gate verification for `--finish`, hard path protection for `--cleanup`, and non-blocking 2-run `Source Health` reporting.
- `src/agents/media.py` -> `MediaAgent.run`, `run_candidates`, `apply_picks`, `run_auto`: Switched post-approval fingerprint checks to `extract_script_validation_fingerprint`, kept blocklist/license refusals strict in `apply_picks`, fell back to headline cards on download/normalization failure, and resynced `render_plan.json` after picks.
- `src/agents/motion.py` -> `MotionAgent.run`, `sync_render_plan_data`, `sync_render_plan_with_media_manifest`: Resolves `run_kind` from script or manifest, verifies script gate, and re-validates synced `render_plan.json` against `contracts/render_plan.schema.json`.
- `src/agents/render.py` -> `RenderAgent.run`, `_render_scene_segment`, `compute_scene_cache_key`: Re-syncs `render_plan.json` with `media_manifest.json` before rendering, honors `manifest["run_kind"]` for publish banners, and fixes >3s frozen still stretches via continuous linear zoom with cache key invalidation.
- `src/agents/qa.py` -> `QAAgent.run`: Records structured `freezedetect` intervals in metrics while generating self-contained `qa/storyboard.html` alongside Check 1–9 enforcement.

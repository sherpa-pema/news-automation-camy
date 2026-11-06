# Orchestrator Agent Specification

## 1. Purpose
The Orchestrator Agent is the central coordinator of the multi-agent pipeline. It manages execution lifecycle, maintains the run directory and `manifest.json`, delegates work to specialist agents strictly via file paths, validates outputs against JSON Schemas, handles retries, and enables parallel execution where dependencies permit.

## 2. Ground Rules
- **File-Only Communication**: Specialist agents never share memory or hidden state; all communication is mediated strictly through files referenced in `manifest.json`.
- **Reproducibility**: Any single stage must be re-runnable independently using only its specified inputs.
- **Strict Validation**: Never advance to the next stage if an agent's output fails schema validation or contract requirements.
- **Fail-Fast with Accountability**: Retry failed stages up to 2 times. If retries are exhausted, halt pipeline, log explicit diagnostic details, and never silently skip stages.

## 3. Inputs & Outputs
- **Inputs**:
  - `config/pipeline.yaml`: Execution DAG, retries, timeouts, and agent configs.
  - `config/sources.yaml`: Global settings and news source configurations.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/manifest.json`: Single source of truth for stage statuses, timestamps, file paths, retries, and errors.
  - `data/runs/<YYYY-MM-DD>/run.log`: Central execution log.

## 4. JSON Schema Contract
- Must satisfy: [manifest.schema.json](../contracts/manifest.schema.json)

## 5. Tools & Modules
- `src/orchestrator.py`
- `src/utils/manifest.py`: Manifest reading, updating, and atomic writing.
- `src/utils/schema_validate.py`: Validation wrapper using `jsonschema`.
- `concurrent.futures.ThreadPoolExecutor`: For running independent stages in parallel (e.g., Voice and Media).

## 6. Execution Flow & Parallelism
1. Initialize date-stamped folder `data/runs/<YYYY-MM-DD>/`.
2. Generate initial `manifest.json` with all stages set to `pending`.
3. Sequentially trigger `scraper` -> `editor`.
4. Fork parallel execution:
   - Branch A: `voice` -> `timeline`
   - Branch B: `media`
5. Join when Branch A and Branch B complete.
6. Trigger `motion` (requires `timeline.json` and `media_manifest.json`).
7. Trigger `render` (requires motion plan, audio, media, captions).
8. Trigger `qa` (validates final MP4 and sync). If QA fails, re-run indicated `redo_stage`.

## 7. Failure Modes & Recovery
- **Stage Timeout**: Mark stage failed, increment `retry_count`, retry up to 2 times.
- **Schema Validation Failure**: Reject output, treat as stage failure, trigger retry.
- **QA Rejection**: Read `qa_report.json` `redo_stage` field, reset indicated stage and downstream stages to `pending`, and re-execute.
- **Max Retries Exceeded**: Mark run `failed`, write failure summary to `manifest.json`, and exit non-zero.

## 8. Definition of Done Checklist
- [ ] Run directory `data/runs/<YYYY-MM-DD>/` created.
- [ ] `manifest.json` tracks every stage from `pending` to `done` or `failed`.
- [ ] All generated outputs validate against their respective JSON schemas.
- [ ] Independent stages (`voice` and `media`) execute concurrently.
- [ ] QA verification passed and final MP4 is produced.

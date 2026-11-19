# Timeline Agent Specification

## 1. Purpose
The Timeline Agent establishes the temporal backbone of the video. By performing per-scene forced phonetic alignment using torchaudio's MMS_FA aligner on scene audio WAVs against the known text from `script/script.json`, it calculates precise word-level, sentence-level, and scene-level start/end timestamps, produces structured caption cues and standard SRT subtitle files, and audits caption quality.

## 2. Ground Rules
- **Temporal Alignment**: Timestamps must precisely synchronize spoken words with screen display; caption cues must never start late by > 50 ms.
- **Scene Boundaries**: The sum of scene durations must equal the total audio duration in `narration.wav` within 20 ms tolerance, and final scene end is reconciled exactly.
- **Verbatim Captions**: Subtitle words in `captions.srt` and `captions.json` must conserve the spoken narration words without omissions or reordering.
- **Reading Speed Limits**: Caption cues target 17.0 chars/sec and must not exceed 20.0 chars/sec.
- **Display Constraints**: Cues are restricted to 1–2 lines, at most 42 characters per line, and duration between 0.9s and 5.0s.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/script/script.json`: Scene script text and metadata.
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Master narration audio.
  - `data/runs/<YYYY-MM-DD>/audio/scene_durations.json`: Scene audio partition offsets and durations.
  - `data/runs/<YYYY-MM-DD>/audio/scenes/<scene_id>.wav`: Per-scene audio files.
  - `config/pronunciations.yaml`: Phonetic respellings and pronunciations.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/timeline/timeline.json`: Scene boundaries, speech bounds, sentence timings, and word-level timestamps (satisfies `contracts/timeline.schema.json`).
  - `data/runs/<YYYY-MM-DD>/timeline/captions.json`: Structured caption phrases with word-level intervals (satisfies `contracts/captions.schema.json`).
  - `data/runs/<YYYY-MM-DD>/timeline/captions.srt`: Formatted SubRip subtitle track for video rendering.
  - `data/runs/<YYYY-MM-DD>/timeline/caption_audit.json`: Sync, duration, line length, CPS, and word conservation audit report.
  - `data/runs/<YYYY-MM-DD>/timeline/timeline_preview.mp4`: Lightweight preview video with burned-in subtitles.
  - Root backward-compatibility copies: `timeline.json` and `captions.srt`.

## 4. JSON Schema Contracts
- Timeline: [timeline.schema.json](../contracts/timeline.schema.json)
- Captions: [captions.schema.json](../contracts/captions.schema.json)

## 5. Tools & Modules
- Forced Alignment: `torchaudio` MMS_FA forced aligner running in an isolated subprocess (`.venv-tts`).
- Mock Aligner: `FakeAligner` for unit tests and offline dry runs.
- Subtitle & Preview: Custom seek-safe chunker and FFmpeg for preview video generation.

## 6. Failure Modes & Degradation Policies
- **Alignment Drift**: Mitigated by per-scene alignment anchored to absolute audio partition offsets.
- **Unplaced Words**: Words unplaced by MMS_FA gracefully fall back to weighted interpolation; marked `unplaced: true`.
- **Degraded Status**: When $\ge 5\%$ of words are unplaced or $\ge 5\%$ are low-confidence ($< 0.35$), `degraded: true` is set, triggering QA re-route to timeline if audit checks fail.
- **Scene Set Mismatch**: Strict equality check between `script.json` and `scene_durations.json` aborts immediately on duplicate or missing scene IDs.

## 7. Definition of Done Checklist
- [x] Word-level start/end timestamps, scores, and confidence flags generated for every spoken word.
- [x] Per-scene `quality` metrics, `speech_start`, `speech_end`, and `sentences` bounds computed.
- [x] `captions.srt` generated and validated against standard SRT syntax.
- [x] Canonical artifacts emitted in `timeline/` alongside backward-compatible root copies.
- [x] Both `timeline.json` and `captions.json` validate against their respective JSON schema contracts.
- [x] Total timeline duration matches `narration.wav` duration within 20 ms tolerance (reconciled exactly).
- [x] `caption_audit.json` verifies 0 late starts > 50 ms, 0 overlaps, and reading speed $\le 20$ CPS.

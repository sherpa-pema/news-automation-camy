# Timeline Agent Specification

## 1. Purpose
The Timeline Agent establishes the temporal backbone of the video. By performing forced phonetic alignment or timestamped automatic speech recognition (ASR) on `narration.wav` against the known text from `script.json`, it calculates precise word-level and scene-level start/end timestamps and generates standard SRT subtitle files.

## 2. Ground Rules
- **Temporal Alignment**: Timestamps must precisely synchronize spoken words with screen display; drift must not exceed 100 milliseconds.
- **Scene Boundaries**: The sum of scene durations must exactly equal the total audio duration in `narration.wav`.
- **Verbatim Captions**: Subtitle words in `captions.srt` must match the spoken narration without omissions.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Master narration audio.
  - `data/runs/<YYYY-MM-DD>/script.json`: Scene texts and order.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/timeline.json`: Scene boundaries and word-level timestamp intervals.
  - `data/runs/<YYYY-MM-DD>/captions.srt`: Formatted SubRip subtitle track for video rendering.

## 4. JSON Schema Contract
- Must satisfy: [timeline.schema.json](../contracts/timeline.schema.json)
- Structure: `{ audio_file, duration_sec, scenes: [{ id, start, end, words: [{ w, start, end }] }] }`

## 5. Tools & Modules
- Forced Alignment / Timestamped ASR: `whisper-timestamped`, `aeneas`, `gentle`, or `stable-ts`.
- Subtitle generation library: `pysrt` or custom SRT formatter.
- `ffprobe`: Audio stream duration extraction for ground-truth validation.

## 6. Failure Modes
- **Alignment Drift**: Cumulative timing error causes later scenes to start out-of-sync.
- **Missing Words**: Words dropped or merged during ASR alignment.
- **Duration Discrepancy**: Last scene timestamp does not match audio length within 0.25s tolerance.

## 7. Definition of Done Checklist
- [ ] Word-level start/end timestamps generated for every spoken word.
- [ ] Scene start and end boundaries calculated for all scenes.
- [ ] `captions.srt` generated and syntactically valid.
- [ ] Output `timeline.json` validates against `timeline.schema.json`.
- [ ] Total timeline duration matches `narration.wav` duration within ±0.2 seconds.

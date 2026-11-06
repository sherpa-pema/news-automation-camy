# QA Agent Specification

## 1. Purpose
The QA (Quality Assurance) Agent performs automated pre-publish inspection on `final.mp4`. It verifies audio/video synchronization, validates audio loudness levels, detects unwanted black frames or frozen video, verifies that every news story displays an on-screen source credit, and confirms caption presence. It outputs `qa_report.json` with a definitive pass/fail verdict and specifies which pipeline stage to redo if any check fails.

## 2. Ground Rules
- **No Compromise Quality Gate**: The video must never be published if any critical check fails.
- **Actionable Remediation**: If a failure occurs, the QA Agent must pinpoint the root cause and explicitly indicate which stage needs to be redone in the `redo_stage` field.
- **Attribution Enforcement**: Every single story scene in the video must have an verified source credit corresponding to `script.json`.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/final.mp4`: Final rendered video.
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Reference audio track.
  - `data/runs/<YYYY-MM-DD>/timeline.json`: Scene boundaries and reference duration.
  - `data/runs/<YYYY-MM-DD>/captions.srt`: Reference subtitles.
  - `data/runs/<YYYY-MM-DD>/media_manifest.json`: Asset licensing and attribution manifest.
  - `data/runs/<YYYY-MM-DD>/script.json`: Reference story source credits.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/qa_report.json`: Comprehensive QA test results and redo recommendation.

## 4. JSON Schema Contract
- Must satisfy: [qa_report.schema.json](../contracts/qa_report.schema.json)
- Structure: `{ passed, checks: [{ name, passed, detail }], redo_stage }`

## 5. Tools & Modules
- `ffprobe`: Container and stream metadata extraction (exact duration, fps, resolution).
- `ffmpeg -vf blackdetect`: Automated detection of accidental black screen dropouts.
- `ffmpeg -af ebur128`: Audio integrated loudness and true peak analysis (EBU R128 compliance).
- Automated script / manifest cross-checker.

## 6. Verification Checks Matrix
| Check Name | Target Criteria | Failure Action (`redo_stage`) |
| :--- | :--- | :--- |
| `duration_sync` | Video duration matches audio within ±0.25s | `render` |
| `loudness_compliance` | Integrated loudness between -14 and -18 LUFS; peak < -1 dBTP | `voice` |
| `no_black_frames` | No continuous black frames > 0.5s outside of intentional scene cuts | `render` |
| `source_attribution_complete` | Every story scene has verified source credit in render plan / overlays | `motion` |
| `media_licensing_verified` | Every media asset in manifest has valid license and origin URL | `media` |
| `captions_present` | Subtitles exist and are non-empty for the full video duration | `timeline` |

## 7. Definition of Done Checklist
- [ ] All verification checks executed and logged.
- [ ] Individual pass/fail status and detail recorded per check.
- [ ] Overall `passed` boolean assigned correctly.
- [ ] `redo_stage` is `null` if passed, or populated with the exact faulty stage ID if failed.
- [ ] Output `qa_report.json` validates against `qa_report.schema.json`.

# Render Agent Specification

## 1. Purpose
The Render Agent executes the rendering plan defined in `render_plan.json`. It combines narration audio, background media, motion overlays, lower-thirds, source credits, and subtitle captions into a high-definition final MP4 video file using `ffmpeg` (or a supported motion graphics engine).

## 2. Ground Rules
- **Deterministic Rendering**: The rendering pipeline must produce identical visual output given the same input files and random seeds.
- **Standards Compliance**: Output must encode to standard H.264 video (YUV420p) and AAC audio in an MP4 container for universal platform compatibility.
- **Frame-Accurate Synchronization**: Video frames must align exactly with audio timestamps from `narration.wav`. No visual drift or truncated audio.
- **Clean Execution**: Temporary render artifacts and intermediate frame buffers must be cleanly isolated and purged upon completion.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/render_plan.json`: Composition blueprints and styling rules.
  - `data/runs/<YYYY-MM-DD>/timeline.json`: Scene timings.
  - `data/runs/<YYYY-MM-DD>/media_manifest.json`: Verified local visual files.
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Master narration audio.
  - `data/runs/<YYYY-MM-DD>/captions.srt`: Subtitle stream.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/final.mp4`: Final broadcast-ready news video.

## 4. Output Validation & Contract
- Output is verified via `ffprobe`:
  - Format: MP4 container (`mov,mp4,m4a,3gp,3g2,mj2`).
  - Video Codec: `h264`, pixel format `yuv420p`.
  - Audio Codec: `aac`, stereo, 44.1kHz or 48kHz.
  - Resolution: Exactly matching `render_plan.json` (e.g., 1920x1080).
  - Duration: Matches `narration.wav` duration within ±0.1 seconds.

## 5. Tools & Modules
- `ffmpeg`: Video encoding, complex filter graph execution, overlay compositing, subtitle burn-in.
- `ffprobe`: Video stream inspection and duration verification.
- Python subprocess wrapper with execution timeout and stderr progress parsing.

## 6. Failure Modes
- **FFmpeg Filtergraph Syntax Error**: Log full stderr, dump generated filter command for debugging, fail stage.
- **Missing Input Media**: Pre-flight verification fails early if any media file referenced in the plan is inaccessible.
- **Process Timeout**: If rendering exceeds configured timeout (default 600s), terminate subprocess and record error.

## 7. Definition of Done Checklist
- [ ] `final.mp4` successfully generated in run directory.
- [ ] Video streams encoded in H.264 / AAC.
- [ ] Captions and lower-third credits clearly burned in / composited.
- [ ] Video duration exactly matches master audio duration.
- [ ] File integrity verified with `ffprobe` (no corrupted frames or truncated streams).

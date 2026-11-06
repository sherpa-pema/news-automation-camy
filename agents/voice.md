# Voice Agent Specification

## 1. Purpose
The Voice Agent synthesizes natural, articulate, broadcast-quality speech from the narration text provided in `script.json` using an open-source Text-to-Speech (TTS) model. It generates both a master narration audio track and individual per-scene audio segments.

## 2. Ground Rules
- **Open-Source Engine**: Use local, open-source TTS models (e.g., Kokoro-82M, Piper, Bark, or Coqui XTTS-v2).
- **Acoustic Consistency**: Maintain identical voice timbre, pitch, pacing, and sample rate across all scenes.
- **Audio Cleanliness**: Ensure audio has no digital clipping, excessive trailing silence, or unnatural robotic glitches.
- **Independent Execution**: Operates asynchronously in parallel with the Media Agent.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/script.json`: Narration scenes to synthesize.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/audio/narration.wav`: Concatenated master audio track (16-bit 48kHz WAV).
  - `data/runs/<YYYY-MM-DD>/audio/scenes/<scene_id>.wav`: Isolated audio files for each scene.

## 4. Output Validation & Contract
- Outputs are verified by file existence and audio validation checks:
  - Format: WAV, PCM 16-bit / 24-bit, 44.1kHz or 48kHz, mono/stereo.
  - Duration: Between 100 and 200 seconds (targeting 2–3 minutes).
  - No empty audio (RMS amplitude > -50 dBFS).

## 5. Tools & Modules
- Open-source TTS inference runtime (Kokoro / Piper-TTS / PyTorch).
- `soundfile` / `scipy.io.wavfile` / `pydub`: Audio concatenation and normalization.
- `ffmpeg`: Audio resample, loudness normalization (EBU R128 to -16 LUFS).

## 6. Failure Modes
- **GPU/CUDA Out of Memory**: Fallback to CPU inference or smaller batch size.
- **Audio Clipping / Glitches**: Audio normalization pass with peak limiter.
- **Duration Mismatch**: Narration audio significantly under (<90s) or over (>210s) duration threshold.

## 7. Definition of Done Checklist
- [ ] Synthesized audio for every scene in `script.json`.
- [ ] Generated individual scene audio files in `audio/scenes/`.
- [ ] Produced stitched `audio/narration.wav`.
- [ ] Total audio duration is within target window (120–180 seconds).
- [ ] Audio volume normalized to standard broadcast loudness.

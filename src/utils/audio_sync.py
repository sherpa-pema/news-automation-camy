"""Audio synchronization and rhythmic beat analysis utilities using Librosa.

Provides musical beat tracking, transition cut snapping, and automated voice ducking
curves for broadcast news video production.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import wavfile

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

logger = logging.getLogger("audio_sync")


def generate_synthetic_broadcast_theme(
    output_path: Path | str,
    duration_sec: float = 60.0,
    bpm: float = 110.0,
    sr: int = 44100,
) -> Path:
    """Generates a clean, rhythmic, low-key broadcast news audio bed.

    Used as a guaranteed fallback or test track when no external music file is provided.
    Features a soft electronic pulse, subtle bass pad, and rhythmic downbeat transients.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    beat_interval = 60.0 / bpm
    total_beats = int(duration_sec / beat_interval)

    # Base harmonic pad (Navy broadcast tone: D-minor chord / 146.83Hz D3, 220Hz A3, 349.23Hz F4)
    pad = (
        0.18 * np.sin(2 * np.pi * 146.83 * t) +
        0.12 * np.sin(2 * np.pi * 220.00 * t) +
        0.08 * np.sin(2 * np.pi * 349.23 * t)
    )

    # Add rhythmic pulses at each beat
    pulse = np.zeros_like(t)
    decay_samples = int(0.12 * sr)
    decay_env = np.exp(-np.linspace(0, 5, decay_samples))

    for b in range(total_beats):
        beat_idx = int(b * beat_interval * sr)
        if beat_idx + decay_samples < len(pulse):
            is_downbeat = (b % 4 == 0)
            freq = 440.0 if is_downbeat else 261.63
            amplitude = 0.25 if is_downbeat else 0.12
            pulse_chunk = amplitude * np.sin(2 * np.pi * freq * t[:decay_samples]) * decay_env
            pulse[beat_idx : beat_idx + decay_samples] += pulse_chunk

    signal = pad + pulse
    # Normalize to prevent clipping (-3dB)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.707

    # Convert to 16-bit PCM stereo
    int16_signal = (signal * 32767).astype(np.int16)
    stereo_signal = np.column_stack([int16_signal, int16_signal])

    wavfile.write(str(out), sr, stereo_signal)
    logger.info(f"Generated synthetic broadcast theme at {out} (BPM={bpm}, duration={duration_sec}s)")
    return out


def load_and_analyze_music(
    bgm_path: str | Path,
    sr: int = 22050,
    bulletin_duration_sec: float | None = None,
) -> dict[str, Any]:
    """Analyzes background music to extract tempo, beat timestamps, and downbeats.

    Returns:
        dict containing:
            - tempo_bpm: float
            - beat_times: list[float]
            - downbeat_times: list[float]
            - duration_sec: float
    """
    p = Path(bgm_path)
    if not p.exists():
        logger.warning(f"Music file not found: {p}. Returning empty analysis.")
        return {"tempo_bpm": 120.0, "beat_times": [], "downbeat_times": [], "duration_sec": 0.0}

    if not HAS_LIBROSA:
        logger.warning("Librosa not installed. Using synthetic 120 BPM grid.")
        bpm = 120.0
        beat_interval = 60.0 / bpm
        dur = max(60.0, float(bulletin_duration_sec or 60.0))
        num_beats = int(dur / beat_interval) + 2
        beats = [round(i * beat_interval, 3) for i in range(num_beats)]
        return {
            "tempo_bpm": bpm,
            "beat_times": beats,
            "downbeat_times": beats[::4],
            "duration_sec": dur,
        }

    try:
        y, sr_actual = librosa.load(str(p), sr=sr)
        duration_sec = float(librosa.get_duration(y=y, sr=sr_actual))
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_actual)

        # Handle scalar vs array tempo in newer librosa versions
        if isinstance(tempo, np.ndarray):
            tempo_val = float(tempo.item(0)) if tempo.size > 0 else 120.0
        else:
            tempo_val = float(tempo)

        beat_times = [float(bt) for bt in librosa.frames_to_time(beat_frames, sr=sr_actual)]
        # Downbeats estimate: every 4th beat
        downbeat_times = beat_times[::4] if beat_times else []

        logger.info(f"Analyzed {p.name}: Tempo={tempo_val:.1f} BPM, {len(beat_times)} beats detected.")
        return {
            "tempo_bpm": round(tempo_val, 2),
            "beat_times": [round(b, 3) for b in beat_times],
            "downbeat_times": [round(b, 3) for b in downbeat_times],
            "duration_sec": round(duration_sec, 3),
        }
    except Exception as e:
        logger.error(f"Failed to analyze music with librosa: {e}. Falling back to default.")
        bpm = 120.0
        beat_interval = 60.0 / bpm
        dur = max(60.0, float(bulletin_duration_sec or 60.0))
        num_beats = int(dur / beat_interval) + 2
        beats = [round(i * beat_interval, 3) for i in range(num_beats)]
        return {
            "tempo_bpm": bpm,
            "beat_times": beats,
            "downbeat_times": beats[::4],
            "duration_sec": dur,
        }


def snap_cuts_to_beats(
    raw_cuts: list[float],
    beat_times: list[float],
    tolerance_sec: float = 0.35,
    min_gap_sec: float = 0.4,
) -> list[float]:
    """Gently snaps raw scene transition cut times to the closest musical beat.

    If the closest musical beat is within +/- tolerance_sec, the transition snaps to that beat.
    Ensures monotonic increasing order and non-negative timestamps.
    """
    if not beat_times or not raw_cuts:
        out: list[float] = []
        for c in raw_cuts:
            if not out:
                out.append(round(c, 3))
            else:
                out.append(round(max(c, out[-1] + min_gap_sec), 3))
        return out

    beat_arr = np.array(beat_times)
    snapped: list[float] = []

    for idx, cut in enumerate(raw_cuts):
        # Don't snap the absolute start (0.0)
        if cut <= 0.05:
            snapped.append(0.0)
            continue

        prev = snapped[-1] if snapped else 0.0

        # Find closest beat
        diffs = np.abs(beat_arr - cut)
        closest_idx = int(np.argmin(diffs))
        min_diff = diffs[closest_idx]

        if min_diff <= tolerance_sec:
            snapped_cut = float(beat_arr[closest_idx])
            if snapped_cut >= prev + min_gap_sec:
                val = snapped_cut
            else:
                val = max(cut, prev + min_gap_sec)
        else:
            val = max(cut, prev + min_gap_sec)

        snapped.append(round(val, 3))

    return snapped


def extract_audio_grid(
    bgm_path: str | Path | None,
    narration_path: str | Path | None,
    scene_boundaries: list[dict[str, Any]],
    style_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrates musical analysis, cut snapping, and sidechain ducking configuration.

    Args:
        bgm_path: Path to background music track (or None if disabled).
        narration_path: Path to master narration audio.
        scene_boundaries: List of scene dicts with "id", "start", "end".
        style_config: Optional style configuration dict.

    Returns:
        dict structured as audio_grid handoff.
    """
    cfg = style_config or {}
    music_cfg = cfg.get("music", {})
    enabled = bool(music_cfg.get("enabled", False))
    voice_duck_db = float(music_cfg.get("voice_duck_db", -22.0))
    music_bed_db = float(music_cfg.get("music_bed_db", -8.0))
    fade_duration = float(music_cfg.get("fade_duration_sec", 0.25))
    snap_tolerance = float(music_cfg.get("snap_tolerance_sec", 0.35))

    if not enabled or not bgm_path or not Path(bgm_path).exists():
        # Disabled or missing audio bed: store slim scene records (id and times only, target < 2 KB)
        slim_scenes = [
            {
                "id": s.get("id"),
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", 0.0)), 3),
            }
            for s in scene_boundaries
        ]
        return {
            "enabled": False,
            "tempo_bpm": 120.0,
            "beat_times": [],
            "downbeat_times": [],
            "snapped_scenes": slim_scenes,
            "ducking": {
                "voice_duck_db": voice_duck_db,
                "music_bed_db": music_bed_db,
                "fade_duration_sec": fade_duration,
                "intervals": [],
            },
        }

    bulletin_dur = float(scene_boundaries[-1]["end"]) if scene_boundaries else 60.0
    analysis = load_and_analyze_music(bgm_path, bulletin_duration_sec=bulletin_dur)
    beats = analysis.get("beat_times", [])

    # Extract cut points: scene starts and ends
    snapped_scenes: list[dict[str, Any]] = []
    cuts = [float(s["start"]) for s in scene_boundaries]
    if scene_boundaries:
        cuts.append(float(scene_boundaries[-1]["end"]))

    snapped_cuts = snap_cuts_to_beats(cuts, beats, tolerance_sec=snap_tolerance)

    for i, s in enumerate(scene_boundaries):
        sc_copy = dict(s)
        sc_copy["start"] = snapped_cuts[i]
        sc_copy["end"] = snapped_cuts[i + 1]
        sc_copy["duration"] = round(sc_copy["end"] - sc_copy["start"], 3)
        snapped_scenes.append(sc_copy)

    # Compute ducking intervals where speech is active
    ducking_intervals: list[dict[str, float]] = []
    for sc in snapped_scenes:
        sp_start = sc.get("speech_start")
        sp_end = sc.get("speech_end")
        if sp_start is not None and sp_end is not None:
            sp_s = float(sp_start)
            sp_e = float(sp_end)
            if sp_e > sp_s:
                d_start = round(max(0.0, sp_s - 0.05), 3)
                d_end = round(sp_e + 0.1, 3)
                ducking_intervals.append({"start": d_start, "end": d_end})
                continue

        # Fallback to scene duration bounds
        sc_dur = float(sc["end"]) - float(sc["start"])
        if sc_dur > 0.8:
            # Leave a small musical swell at start (0.15s) and end (0.2s)
            d_start = round(float(sc["start"]) + 0.15, 3)
            d_end = round(max(d_start + 0.1, float(sc["end"]) - 0.20), 3)
            ducking_intervals.append({"start": d_start, "end": d_end})

    return {
        "enabled": True,
        "bgm_file": str(bgm_path),
        "tempo_bpm": analysis.get("tempo_bpm", 120.0),
        "beat_times": beats,
        "downbeat_times": analysis.get("downbeat_times", []),
        "snapped_scenes": snapped_scenes,
        "ducking": {
            "voice_duck_db": voice_duck_db,
            "music_bed_db": music_bed_db,
            "fade_duration_sec": fade_duration,
            "intervals": ducking_intervals,
        },
    }

"""Generates candidate reference audio clips using Ming-3.0-tts.

Runs in .venv-tts.
Synthesizes a 10-15s neutral broadcast passage using task IP and voice US_James
across multiple fixed seeds, measures audio quality and speaker consistency,
and selects the best candidate for config/voice/reference.wav.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from ming_tts import MingTTS
from spkemb_extractor import SpkembExtractor

REFERENCE_PASSAGE = (
    "Good evening and welcome to the daily news briefing. "
    "Today we bring you comprehensive coverage of economic developments, regional trade, and community initiatives across the country. "
    "Stay tuned as we present the top stories and essential updates for tonight."
)

CANDIDATE_SEEDS = [1895, 2026, 42, 7777]


def measure_silence_gaps(waveform: torch.Tensor, sample_rate: int, threshold_db: float = -45.0) -> list[float]:
    """Finds silent segments longer than 100ms and returns their durations in seconds."""
    arr = waveform.abs().cpu().numpy().flatten()
    eps = 1e-7
    db = 20.0 * np.log10(np.maximum(arr, eps))
    silent = db < threshold_db

    gaps = []
    gap_len = 0
    for is_sil in silent:
        if is_sil:
            gap_len += 1
        else:
            if gap_len > 0:
                dur = gap_len / sample_rate
                if dur >= 0.1:
                    gaps.append(dur)
                gap_len = 0
    if gap_len > 0:
        dur = gap_len / sample_rate
        if dur >= 0.1:
            gaps.append(dur)
    return gaps


def main() -> int:
    out_dir = Path("data/tts_tests/reference_candidates")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_voice_dir = Path("config/voice")
    cfg_voice_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gen_ref] Initializing MingTTS on cuda:0...")
    model = MingTTS("models/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
    extractor = SpkembExtractor("models/Ming-3.0-tts-0.5b/campplus.onnx")

    candidates_meta = []

    for seed in CANDIDATE_SEEDS:
        print(f"\n[gen_ref] Synthesizing candidate with seed={seed}...")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        wav_path = out_dir / f"ref_candidate_seed_{seed}.wav"
        t0 = time.time()
        waveform = model.synthesize(
            task="IP",
            text=REFERENCE_PASSAGE,
            language_id="en",
            controls={"ip": "US_James"},
            instruction_schema="en",
            use_zero_spk_emb=True,
            output_path=str(wav_path),
        )
        synth_sec = round(time.time() - t0, 2)

        # Reload with soundfile to ensure valid PCM
        data, sr = sf.read(str(wav_path))
        # Ensure mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration_sec = len(data) / sr
        max_amp = float(np.max(np.abs(data)))
        peak_db = 20.0 * math.log10(max_amp) if max_amp > 1e-9 else -999.0
        rms = float(np.sqrt(np.mean(data**2)))
        rms_db = 20.0 * math.log10(rms) if rms > 1e-9 else -999.0

        # Pitch
        wave_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        pitch = torchaudio.functional.detect_pitch_frequency(wave_tensor, sr)
        voiced = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
        f0_med = round(float(voiced.median()), 1) if voiced.numel() > 0 else 0.0

        # Silence gaps
        gaps = measure_silence_gaps(wave_tensor, sr, threshold_db=-40.0)
        max_silence = max(gaps) if gaps else 0.0

        # Speaker embedding
        if sr != 16000:
            wave16k = torchaudio.transforms.Resample(sr, 16000)(wave_tensor)
        else:
            wave16k = wave_tensor
        emb = extractor(wave16k)
        emb_norm = emb / torch.norm(emb, p=2, dim=-1, keepdim=True)

        meta = {
            "seed": seed,
            "file": str(wav_path),
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sr,
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "median_f0_hz": f0_med,
            "max_silence_sec": round(max_silence, 2),
            "synth_time_sec": synth_sec,
            "is_clipped": peak_db >= -0.05,
        }
        candidates_meta.append((meta, emb_norm, data, sr))
        print(f"[gen_ref] Candidate seed={seed}: dur={duration_sec:.2f}s, peak={peak_db:.2f}dBFS, F0={f0_med}Hz, max_gap={max_silence:.2f}s")

    # Score and select the best candidate:
    # Criteria:
    # 1. Duration between 10s and 16s
    # 2. No silence gaps over 0.5s
    # 3. Peak below -0.5 dBFS (no clipping)
    # 4. Healthy F0 (male: 100-160Hz)
    best_candidate = None
    best_penalty = float("inf")

    for meta, emb, data, sr in candidates_meta:
        penalty = 0.0
        # Duration preference: 12-14s
        dur = meta["duration_sec"]
        if dur < 10.0 or dur > 16.0:
            penalty += 100.0
        else:
            penalty += abs(dur - 13.0) * 2.0

        # Silence gap penalty (> 0.5s is bad)
        if meta["max_silence_sec"] > 0.5:
            penalty += (meta["max_silence_sec"] - 0.5) * 50.0

        # Clipping penalty
        if meta["is_clipped"] or meta["peak_db"] > -0.5:
            penalty += 50.0

        if penalty < best_penalty:
            best_penalty = penalty
            best_candidate = (meta, emb, data, sr)

    if not best_candidate:
        best_candidate = candidates_meta[0]

    sel_meta, sel_emb, sel_data, sel_sr = best_candidate
    print(f"\n[gen_ref] Selected best candidate: seed={sel_meta['seed']} (penalty={best_penalty:.2f})")

    # Resample to 24 kHz mono
    wav_tensor = torch.tensor(sel_data, dtype=torch.float32).unsqueeze(0)
    if sel_sr != 24000:
        resampler = torchaudio.transforms.Resample(sel_sr, 24000)
        wav_24k = resampler(wav_tensor)
    else:
        wav_24k = wav_tensor

    # If peak is above -3 dBFS, scale it cleanly so peak is at -3.0 dBFS
    target_peak_linear = 10.0 ** (-3.0 / 20.0)  # ~0.7079
    max_val = torch.max(torch.abs(wav_24k)).item()
    if max_val > target_peak_linear:
        wav_24k = wav_24k * (target_peak_linear / max_val)

    # Save to config/voice/reference.wav (24 kHz mono)
    ref_wav_path = cfg_voice_dir / "reference.wav"
    ref_txt_path = cfg_voice_dir / "reference.txt"

    torchaudio.save(str(ref_wav_path), wav_24k, 24000, encoding="PCM_S", bits_per_sample=16)
    with open(ref_txt_path, "w", encoding="utf-8") as f:
        f.write(REFERENCE_PASSAGE.strip())

    # Write candidates report
    report = {
        "passage": REFERENCE_PASSAGE,
        "selected_seed": sel_meta["seed"],
        "reference_wav": str(ref_wav_path),
        "reference_txt": str(ref_txt_path),
        "candidates": [m[0] for m in candidates_meta],
    }
    with open(out_dir / "candidates_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[gen_ref] Saved reference audio to: {ref_wav_path}")
    print(f"[gen_ref] Saved reference transcript to: {ref_txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

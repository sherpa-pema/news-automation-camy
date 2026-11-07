#!/usr/bin/env python3
"""Ming-3.0-TTS Worker Process with Speaker Consistency Enforceability.

Runs inside the isolated .venv-tts environment.
Loads the model once, processes synthesis jobs (supporting task="IP" and task="TTS" zero-shot cloning),
evaluates speaker embedding similarity via CamPP+ ONNX, checks pitch contour (F0),
pre-normalizes item loudness to -3.0 dBFS, retries outlier chunks up to 3 seeds,
and outputs structured execution metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Ensure vendor directory is on sys.path
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from ming_tts import MingTTS
    from spkemb_extractor import SpkembExtractor
except ImportError as err:
    print(f"FATAL: Failed to import MingTTS/SpkembExtractor from {VENDOR_DIR}: {err}", file=sys.stderr)
    sys.exit(2)


def compute_f0(audio_tensor: torch.Tensor, sample_rate: int) -> float:
    """Computes median fundamental frequency (F0) in Hz."""
    if audio_tensor.numel() == 0 or sample_rate <= 0:
        return 0.0
    if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
    elif audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    try:
        pitch = torchaudio.functional.detect_pitch_frequency(audio_tensor, sample_rate)
        voiced = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
        if voiced.numel() > 0:
            return round(float(voiced.median().item()), 1)
    except Exception:
        pass
    return 0.0


def measure_audio_health(file_path: Path | str) -> dict[str, Any]:
    """Inspects generated audio for duration, peak, RMS, and silence."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"healthy": False, "reason": "File does not exist or is empty"}

    try:
        data, sr = sf.read(str(path))
    except Exception as exc:
        return {"healthy": False, "reason": f"Corrupt audio file: {exc}"}

    duration_sec = len(data) / sr if sr > 0 else 0.0
    if duration_sec < 0.1:
        return {"healthy": False, "reason": f"Audio duration too short: {duration_sec:.2f}s"}

    max_amp = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    peak_db = 20.0 * np.log10(max_amp) if max_amp > 1e-9 else -999.0
    rms_db = 20.0 * np.log10(rms) if rms > 1e-9 else -999.0

    if rms_db < -55.0:
        return {
            "healthy": False,
            "reason": f"Near-silent audio (RMS = {rms_db:.1f} dBFS < -55.0 dBFS)",
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "peak_db": peak_db,
            "rms_db": rms_db,
        }

    return {
        "healthy": True,
        "duration_sec": duration_sec,
        "sample_rate": sr,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "is_clipped": peak_db >= -0.05,
    }


def pre_normalize_audio(audio_data: np.ndarray, target_peak_db: float = -3.0) -> np.ndarray:
    """Scales audio cleanly so its maximum peak matches target_peak_db."""
    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp <= 1e-9:
        return audio_data
    target_amp = 10.0 ** (target_peak_db / 20.0)
    scaling_factor = target_amp / max_amp
    return (audio_data * scaling_factor).astype(np.float32)


def run_worker(job_path: Path, result_path: Path) -> int:
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    model_path = job.get("model_path", "models/Ming-3.0-tts-0.5b")
    campplus_path = job.get("campplus_path", "models/Ming-3.0-tts-0.5b/campplus.onnx")
    device = job.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    task = job.get("task", "TTS")
    voice = job.get("voice", "US_Emma")
    raw_lang = job.get("language_id", "en")
    language_id = "en" if str(raw_lang).strip().lower() in ("english", "en") else raw_lang
    instruction_schema = job.get("instruction_schema", "en")
    base_seed = int(job.get("seed", 1895))
    text_normalization = bool(job.get("text_normalization", False))
    speed = job.get("speed")
    items = job.get("items", [])

    # Consistency settings
    reference_wav = job.get("reference_wav", "config/voice/reference.wav")
    reference_txt = job.get("reference_txt", "config/voice/reference.txt")
    similarity_threshold = float(job.get("similarity_threshold", 0.72))
    f0_tolerance_pct = float(job.get("f0_tolerance_pct", 0.25))
    max_retries = int(job.get("max_retries", 2))

    ref_prompt_text = None
    ref_emb_norm = None
    ref_f0 = 0.0

    print(f"[ming_worker] Initializing MingTTS on device={device} (task={task})...")
    t_load_start = time.time()
    try:
        model = MingTTS(
            model_path=model_path,
            device=device,
            text_normalization=text_normalization,
        )
        extractor = SpkembExtractor(campplus_path)

        # Load reference audio and text if task is TTS or reference provided
        ref_path = Path(reference_wav)
        if ref_path.exists():
            ref_audio, ref_sr = torchaudio.load(str(ref_path))
            if ref_audio.shape[0] > 1:
                ref_audio = ref_audio.mean(dim=0, keepdim=True)
            ref_f0 = compute_f0(ref_audio, ref_sr)
            ref_audio_16k = (
                torchaudio.transforms.Resample(ref_sr, 16000)(ref_audio)
                if ref_sr != 16000
                else ref_audio
            )
            ref_emb = extractor(ref_audio_16k)
            ref_emb_norm = ref_emb / torch.norm(ref_emb, p=2, dim=-1, keepdim=True)

            ref_txt_path = Path(reference_txt)
            if ref_txt_path.exists():
                with open(ref_txt_path, "r", encoding="utf-8") as f:
                    ref_prompt_text = f.read().strip()

            print(
                f"[ming_worker] Loaded reference: {ref_path.name} (F0={ref_f0:.1f}Hz, prompt_len={len(ref_prompt_text or '')} chars)"
            )

        model_load_time_sec = time.time() - t_load_start
        print(f"[ming_worker] Model loaded in {model_load_time_sec:.2f}s")
    except Exception as exc:
        print(f"[ming_worker] FATAL: Initialization failed: {exc}", file=sys.stderr)
        err_report = {
            "status": "fatal_error",
            "error": f"Failed to initialize: {exc}",
            "model_load_time_sec": 0.0,
            "results": [],
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(err_report, f, indent=2)
        return 1

    results = []
    for item in items:
        item_id = item["id"]
        text = item["text"]
        out_path = Path(item["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)

        controls = {"ip": voice}
        if speed:
            controls["speed"] = speed

        t_synth_start = time.time()
        best_candidate: dict[str, Any] | None = None
        best_score = -999.0

        for attempt in range(max_retries + 1):
            seed = base_seed + attempt
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            temp_candidate_wav = out_path.with_name(f"{out_path.stem}_att{attempt}.wav")

            try:
                if task == "TTS" and ref_prompt_text:
                    model.synthesize(
                        task="TTS",
                        text=text,
                        language_id=language_id,
                        prompt_wav_path=str(ref_path),
                        prompt_text=ref_prompt_text,
                        use_spk_emb=True,
                        use_zero_spk_emb=False,
                        output_path=str(temp_candidate_wav),
                    )
                else:
                    model.synthesize(
                        task=task,
                        text=text,
                        language_id=language_id,
                        controls=controls,
                        instruction_schema=instruction_schema,
                        use_zero_spk_emb=True,
                        output_path=str(temp_candidate_wav),
                    )

                audio_data, audio_sr = sf.read(str(temp_candidate_wav))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                # Pre-normalize loudness to -3.0 dBFS
                norm_data = pre_normalize_audio(audio_data, target_peak_db=-3.0)
                sf.write(str(temp_candidate_wav), norm_data, audio_sr, subtype="PCM_16")

                health = measure_audio_health(temp_candidate_wav)
                if not health["healthy"]:
                    raise ValueError(health.get("reason", "Unhealthy audio"))

                # Measure F0 and speaker embedding similarity
                tensor_wave = torch.tensor(norm_data, dtype=torch.float32).unsqueeze(0)
                cand_f0 = compute_f0(tensor_wave, audio_sr)

                if audio_sr != 16000:
                    cand_16k = torchaudio.transforms.Resample(audio_sr, 16000)(tensor_wave)
                else:
                    cand_16k = tensor_wave

                cand_emb = extractor(cand_16k)
                cand_emb_norm = cand_emb / torch.norm(cand_emb, p=2, dim=-1, keepdim=True)

                if ref_emb_norm is not None:
                    sim = float(torch.matmul(ref_emb_norm, cand_emb_norm.T).item())
                else:
                    sim = 1.0

                # Check gate criteria
                f0_ok = True
                if ref_f0 > 0:
                    f0_diff = abs(cand_f0 - ref_f0) / ref_f0
                    if f0_diff > f0_tolerance_pct:
                        f0_ok = False

                sim_ok = sim >= similarity_threshold
                score = sim - (0.5 if not f0_ok else 0.0)

                cand_info = {
                    "attempt": attempt,
                    "wav_file": temp_candidate_wav,
                    "duration_sec": health["duration_sec"],
                    "sample_rate": health["sample_rate"],
                    "peak_db": health["peak_db"],
                    "rms_db": health["rms_db"],
                    "f0_median_hz": cand_f0,
                    "cosine_sim": round(sim, 4),
                    "passes_gate": sim_ok and f0_ok,
                    "score": score,
                }

                if score > best_score:
                    best_score = score
                    best_candidate = cand_info

                if sim_ok and f0_ok:
                    # Candidate passed cleanly on this attempt
                    break
                else:
                    print(
                        f"[ming_worker] {item_id} attempt {attempt + 1}: sim={sim:.4f} (gate={similarity_threshold}), F0={cand_f0:.1f}Hz (gate={ref_f0:.1f}Hz +-25%). Retrying..."
                    )
            except Exception as e:
                print(f"[ming_worker] {item_id} attempt {attempt + 1} error: {e}")
                if temp_candidate_wav.exists():
                    temp_candidate_wav.unlink(missing_ok=True)

        seconds_taken = time.time() - t_synth_start

        if best_candidate and best_candidate.get("wav_file") and best_candidate["wav_file"].exists():
            # Promote best candidate to target out_path
            best_wav = best_candidate["wav_file"]
            if out_path.exists():
                out_path.unlink()
            best_wav.replace(out_path)

            # Clean up other attempt files
            for att in range(max_retries + 1):
                f_att = out_path.with_name(f"{out_path.stem}_att{att}.wav")
                f_att.unlink(missing_ok=True)

            results.append({
                "id": item_id,
                "status": "success",
                "out_path": str(out_path),
                "duration_sec": round(best_candidate["duration_sec"], 3),
                "sample_rate": best_candidate["sample_rate"],
                "seconds_taken": round(seconds_taken, 3),
                "peak_db": round(best_candidate["peak_db"], 2),
                "rms_db": round(best_candidate["rms_db"], 2),
                "f0_median_hz": best_candidate["f0_median_hz"],
                "cosine_sim": best_candidate["cosine_sim"],
                "passes_gate": best_candidate["passes_gate"],
                "retries": best_candidate["attempt"],
                "error": None,
            })
            print(
                f"[ming_worker] [OK] {item_id}: dur={best_candidate['duration_sec']:.2f}s, sim={best_candidate['cosine_sim']}, F0={best_candidate['f0_median_hz']}Hz (retries={best_candidate['attempt']})"
            )
        else:
            # Clean up attempt files
            for att in range(max_retries + 1):
                f_att = out_path.with_name(f"{out_path.stem}_att{att}.wav")
                f_att.unlink(missing_ok=True)

            results.append({
                "id": item_id,
                "status": "failed",
                "out_path": str(out_path),
                "duration_sec": 0.0,
                "sample_rate": 0,
                "seconds_taken": round(seconds_taken, 3),
                "peak_db": -999.0,
                "rms_db": -999.0,
                "f0_median_hz": 0.0,
                "cosine_sim": 0.0,
                "passes_gate": False,
                "retries": max_retries,
                "error": "Failed all retry attempts",
            })
            print(f"[ming_worker] [FAIL] {item_id}: Failed all retry attempts")

    has_failures = any(r["status"] == "failed" for r in results)
    worker_report = {
        "status": "error" if has_failures else "success",
        "model_load_time_sec": round(model_load_time_sec, 2),
        "reference_wav": reference_wav,
        "reference_f0_hz": ref_f0,
        "similarity_threshold": similarity_threshold,
        "results": results,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(worker_report, f, indent=2)

    return 1 if has_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ming-3.0-TTS Subprocess Worker")
    parser.add_argument("--job-file", required=True, help="Path to job JSON file")
    parser.add_argument("--result-file", required=True, help="Path to result JSON output")
    args = parser.parse_args()

    job_path = Path(args.job_file)
    result_path = Path(args.result_file)

    if not job_path.exists():
        print(f"FATAL: Job file not found: {job_path}", file=sys.stderr)
        return 1

    return run_worker(job_path, result_path)


if __name__ == "__main__":
    sys.exit(main())

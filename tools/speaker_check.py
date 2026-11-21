"""Speaker consistency and acoustic check utility.

Runs in .venv-tts.
Uses the model's own SpkembExtractor (models/Ming-3.0-tts-0.5b/campplus.onnx)
and torchaudio pitch detection to measure speaker embedding similarity, median F0,
and loudness across audio segments or sentences cut from master narration.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

# Add vendor/Ming-3.0-tts to sys.path
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from spkemb_extractor import SpkembExtractor


def compute_median_f0(waveform: torch.Tensor, sample_rate: int) -> float:
    """Computes median fundamental frequency (F0) of voiced frames in Hz."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    voiced = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
    if voiced.numel() == 0:
        return 0.0
    return round(float(voiced.median()), 1)


def compute_rms_dbfs(waveform: torch.Tensor) -> float:
    """Computes RMS loudness in dBFS."""
    arr = waveform.cpu().numpy()
    rms = float(np.sqrt(np.mean(arr**2)))
    if rms < 1e-9:
        return -999.0
    return round(20.0 * math.log10(rms), 2)


_RESAMPLERS: dict[tuple[int, int], Any] = {}


def _get_resampler(orig_sr: int, target_sr: int) -> Any:
    if orig_sr == target_sr:
        return None
    key = (orig_sr, target_sr)
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
    return _RESAMPLERS[key]


def extract_embedding(extractor: SpkembExtractor, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Extracts a normalized 192-dim speaker embedding tensor [1, 192]."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = _get_resampler(sample_rate, 16000)
        waveform = resampler(waveform) if resampler is not None else waveform

    # Kaldi fbank expects at least 400 samples (25ms)
    if waveform.shape[-1] < 400:
        pad_len = 400 - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    emb = extractor(waveform)
    norm = torch.norm(emb, p=2, dim=-1, keepdim=True)
    if norm > 0:
        emb = emb / norm
    return emb


def analyze_segments(
    segments: list[dict[str, Any]],
    extractor_path: str | Path = "models/Ming-3.0-tts-0.5b/campplus.onnx",
    reference_wav: str | Path | None = None,
) -> dict[str, Any]:
    """Analyzes a list of audio segments [{id, scene, text, waveform, sample_rate, start, end}]."""
    extractor = SpkembExtractor(str(extractor_path))

    embeddings: list[torch.Tensor] = []
    f0_list: list[float] = []
    rms_list: list[float] = []

    for seg in segments:
        w = seg["waveform"]
        sr = seg["sample_rate"]
        emb = extract_embedding(extractor, w, sr)
        f0 = compute_median_f0(w, sr)
        rms = compute_rms_dbfs(w)

        embeddings.append(emb)
        f0_list.append(f0)
        rms_list.append(rms)

    if not embeddings:
        return {"status": "empty", "segments": []}

    all_embs = torch.cat(embeddings, dim=0)  # [N, 192]

    # Centroid
    centroid = all_embs.mean(dim=0, keepdim=True)
    centroid_norm = torch.norm(centroid, p=2)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    sim_to_centroid = torch.mm(all_embs, centroid.t()).squeeze(-1).cpu().numpy()

    ref_sims: list[float] | None = None
    if reference_wav and Path(reference_wav).exists():
        ref_wave, ref_sr = torchaudio.load(str(reference_wav))
        ref_emb = extract_embedding(extractor, ref_wave, ref_sr)
        ref_sim_arr = torch.mm(all_embs, ref_emb.t()).squeeze(-1).cpu().numpy()
        ref_sims = [round(float(s), 4) for s in ref_sim_arr]

    # Build per-segment output records
    results = []
    for i, seg in enumerate(segments):
        rec = {
            "id": seg.get("id", str(i)),
            "scene": seg.get("scene", ""),
            "index": i,
            "start_sec": seg.get("start", 0.0),
            "end_sec": seg.get("end", 0.0),
            "duration_sec": seg.get("duration", 0.0),
            "text": seg.get("text", ""),
            "similarity_to_centroid": round(float(sim_to_centroid[i]), 4),
            "median_f0_hz": f0_list[i],
            "rms_dbfs": rms_list[i],
        }
        if ref_sims is not None:
            rec["similarity_to_reference"] = ref_sims[i]
        results.append(rec)

    # Summary statistics
    sims = [r["similarity_to_centroid"] for r in results]
    valid_f0 = [f for f in f0_list if f > 0]
    valid_rms = [r for r in rms_list if r > -100]

    return {
        "status": "success",
        "count": len(results),
        "similarity_min": round(float(np.min(sims)), 4) if sims else 0.0,
        "similarity_mean": round(float(np.mean(sims)), 4) if sims else 0.0,
        "similarity_max": round(float(np.max(sims)), 4) if sims else 0.0,
        "f0_min": round(float(np.min(valid_f0)), 1) if valid_f0 else 0.0,
        "f0_median": round(float(np.median(valid_f0)), 1) if valid_f0 else 0.0,
        "f0_max": round(float(np.max(valid_f0)), 1) if valid_f0 else 0.0,
        "rms_min": round(float(np.min(valid_rms)), 2) if valid_rms else 0.0,
        "rms_max": round(float(np.max(valid_rms)), 2) if valid_rms else 0.0,
        "rms_spread": round(float(np.max(valid_rms) - np.min(valid_rms)), 2) if valid_rms else 0.0,
        "segments": results,
    }


def analyze_timeline_narration(
    narration_path: Path | str,
    timeline_path: Path | str,
    extractor_path: Path | str = "models/Ming-3.0-tts-0.5b/campplus.onnx",
    reference_wav: str | Path | None = None,
) -> dict[str, Any]:
    """Cuts narration.wav into sentence segments using timeline.json sentence timestamps."""
    narration_path = Path(narration_path)
    timeline_path = Path(timeline_path)

    if not narration_path.exists():
        raise FileNotFoundError(f"Narration audio file not found: {narration_path}")
    if not timeline_path.exists():
        raise FileNotFoundError(f"Timeline JSON not found: {timeline_path}")

    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    waveform, sr = torchaudio.load(str(narration_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    segments = []
    sent_counter = 0

    for sc in timeline_data.get("scenes", []):
        sc_id = sc["id"]
        for s_idx, sent in enumerate(sc.get("sentences", [])):
            s_start = sent["start"]
            s_end = sent["end"]
            start_frame = max(0, int(s_start * sr))
            end_frame = min(waveform.shape[-1], int(s_end * sr))

            if end_frame <= start_frame:
                continue

            seg_wave = waveform[:, start_frame:end_frame]

            segments.append({
                "id": f"{sc_id}_s{s_idx}",
                "scene": sc_id,
                "text": sent.get("text", ""),
                "waveform": seg_wave,
                "sample_rate": sr,
                "start": round(s_start, 3),
                "end": round(s_end, 3),
                "duration": round(s_end - s_start, 3),
            })
            sent_counter += 1

    return analyze_segments(segments, extractor_path=extractor_path, reference_wav=reference_wav)


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker embedding and F0 consistency analyzer")
    parser.add_argument("--narration", default=None, help="Path to narration.wav")
    parser.add_argument("--timeline", default=None, help="Path to timeline.json")
    parser.add_argument("--reference-wav", default=None, help="Optional reference WAV to compute similarity against")
    parser.add_argument("--extractor", default="models/Ming-3.0-tts-0.5b/campplus.onnx", help="Path to campplus.onnx")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    if args.narration and args.timeline:
        report = analyze_timeline_narration(
            args.narration,
            args.timeline,
            extractor_path=args.extractor,
            reference_wav=args.reference_wav,
        )
    else:
        print("Usage: python tools/speaker_check.py --narration <path> --timeline <path>")
        return 1

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # Print summary table
    print("=" * 80)
    print(f" SPEAKER CONSISTENCY AND PITCH REPORT ({report['count']} segments)")
    print("=" * 80)
    print(f"Similarity to Centroid: min={report['similarity_min']:.4f}, mean={report['similarity_mean']:.4f}, max={report['similarity_max']:.4f}")
    print(f"Fundamental Freq (F0):  min={report['f0_min']:.1f} Hz, median={report['f0_median']:.1f} Hz, max={report['f0_max']:.1f} Hz")
    print(f"Loudness (RMS dBFS):    min={report['rms_min']:.2f} dBFS, max={report['rms_max']:.2f} dBFS, spread={report['rms_spread']:.2f} dB")
    print("-" * 80)
    print(f"{'ID':<15} {'Scene':<10} {'Start':<8} {'Dur':<6} {'Sim Centroid':<14} {'Median F0':<12} {'RMS dBFS':<10}")
    print("-" * 80)
    for seg in report["segments"]:
        print(
            f"{seg['id']:<15} {seg['scene']:<10} {seg['start_sec']:<8.2f} {seg['duration_sec']:<6.2f} "
            f"{seg['similarity_to_centroid']:<14.4f} {seg['median_f0_hz']:<12.1f} {seg['rms_dbfs']:<10.2f}"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

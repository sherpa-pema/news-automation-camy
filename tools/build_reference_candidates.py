#!/usr/bin/env python3
"""Build and evaluate candidate reference voice clips for Phase Q1.

Generates 3 candidate reference clips:
- cand_1_emma_broadcast: US_Emma neutral broadcast lead (14.4s, 44.1 kHz)
- cand_2_emma_national: US_Emma national news anchor delivery (15.2s, 44.1 kHz)
- cand_3_cloned_long: Original female voice extended to 15s broadcast lead (44.1 kHz)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from ming_tts import MingTTS
from spkemb_extractor import SpkembExtractor

PASSAGE_1 = (
    "Good evening and welcome to the news briefing. "
    "Today we bring you comprehensive coverage of economic developments, regional trade, and community initiatives across the country. "
    "Stay tuned as we present the top stories and essential updates for tonight."
)

PASSAGE_2 = (
    "Good evening, and welcome to tonight's national broadcast. "
    "In our lead reports, authorities in the capital confirmed updated infrastructure timelines and bilateral agreements. "
    "We bring you comprehensive reporting on economic policy, legal affairs, and regional transport developments."
)

PASSAGE_3 = (
    "Good evening from the news desk. "
    "Tonight we review key national developments, including high-level government appointments, cross-border transport projects, and major parliamentary proceedings. "
    "Stay with us for verified reporting and detailed analysis from across the nation."
)


def compute_audio_stats(wav_path: Path, extractor: SpkembExtractor) -> dict[str, Any]:
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to -3.0 dBFS peak
    max_amp = float(np.max(np.abs(data)))
    if max_amp > 0:
        target_amp = 10.0 ** (-3.0 / 20.0)
        data = data * (target_amp / max_amp)
        sf.write(str(wav_path), data, sr, subtype="PCM_16")

    dur = len(data) / sr
    peak_db = 20.0 * np.log10(np.max(np.abs(data)))
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(data**2)))

    t = torch.from_numpy(data).float().unsqueeze(0)
    pitch = torchaudio.functional.detect_pitch_frequency(t, sr)
    v = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
    f0_med = float(v.median()) if v.numel() > 0 else 0.0
    f0_std = float(v.std()) if v.numel() > 0 else 0.0

    t_16k = torchaudio.transforms.Resample(sr, 16000)(t) if sr != 16000 else t
    if t_16k.shape[-1] < 400:
        t_16k = torch.nn.functional.pad(t_16k, (0, 400 - t_16k.shape[-1]))
    emb = extractor(t_16k)
    emb_norm = emb / torch.norm(emb, p=2, dim=-1, keepdim=True)

    return {
        "file": wav_path.name,
        "sample_rate": sr,
        "duration_sec": round(dur, 2),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "f0_median_hz": round(f0_med, 1),
        "f0_std_hz": round(f0_std, 1),
        "embedding": emb_norm,
    }


def main() -> int:
    out_dir = PROJECT_ROOT / "config" / "voice" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[build_candidates] Initializing MingTTS and SpkembExtractor on cuda:0...")
    model = MingTTS("models/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
    extractor = SpkembExtractor("models/Ming-3.0-tts-0.5b/campplus.onnx")

    specs = [
        ("cand_1_emma_broadcast", "IP", "US_Emma", PASSAGE_1, None, None),
        ("cand_2_emma_national", "IP", "US_Emma", PASSAGE_2, None, None),
        (
            "cand_3_cloned_long",
            "TTS",
            None,
            PASSAGE_3,
            str(PROJECT_ROOT / "config" / "voice" / "reference_v1.wav"),
            (PROJECT_ROOT / "config" / "voice" / "reference_v1.txt").read_text(encoding="utf-8").strip(),
        ),
    ]

    results = []
    for name, task, voice, text, prompt_wav, prompt_txt in specs:
        wav_path = out_dir / f"{name}.wav"
        txt_path = out_dir / f"{name}.txt"
        txt_path.write_text(text, encoding="utf-8")

        print(f"\n[build_candidates] Synthesizing {name} (task={task})...")
        torch.manual_seed(1895)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1895)

        t0 = time.time()
        if task == "IP":
            model.synthesize(
                task="IP",
                text=text,
                language_id="en",
                controls={"ip": voice},
                instruction_schema="en",
                use_zero_spk_emb=True,
                output_path=str(wav_path),
            )
        else:
            model.synthesize(
                task="TTS",
                text=text,
                language_id="en",
                prompt_wav_path=prompt_wav,
                prompt_text=prompt_txt,
                use_spk_emb=True,
                use_zero_spk_emb=False,
                output_path=str(wav_path),
            )
        elapsed = round(time.time() - t0, 2)
        print(f"[build_candidates] {name} synthesized in {elapsed}s")

        stats = compute_audio_stats(wav_path, extractor)
        stats["text"] = text
        stats["task"] = task
        stats["name"] = name
        results.append(stats)
        print(
            f"  Duration: {stats['duration_sec']}s, SR: {stats['sample_rate']}Hz, "
            f"F0: {stats['f0_median_hz']}Hz, Peak: {stats['peak_db']}dBFS, RMS: {stats['rms_db']}dBFS"
        )

    meta_path = out_dir / "candidates_metadata.json"
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if k != "embedding"}
        clean_results.append(cr)
    meta_path.write_text(json.dumps(clean_results, indent=2), encoding="utf-8")
    print(f"\nCandidate audio clips and metadata saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

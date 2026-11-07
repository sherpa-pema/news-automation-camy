"""Saves and formats a reference voice clip for Ming-3.0-tts zero-shot cloning."""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio

def main():
    src_mp3 = Path(r"C:\Users\Chungba Sherpa\Downloads\NP_lemi.mp3")
    if not src_mp3.exists():
        print(f"Error: {src_mp3} does not exist.")
        return 1

    wav, sr = torchaudio.load(str(src_mp3))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 24000 Hz
    wav_24k = torchaudio.transforms.Resample(sr, 24000)(wav)

    # Scale peak to -3.0 dBFS
    max_val = torch.max(torch.abs(wav_24k)).item()
    target_peak = 10.0 ** (-3.0 / 20.0)
    if max_val > target_peak:
        wav_24k = wav_24k * (target_peak / max_val)

    ref_text = (
        "Hi guys, I don't even know where to begin, but one of my biggest dreams "
        "of my career is finally coming true. I'm in Paris. I am here for Paris "
        "Fashion Week as a backstage makeup artist."
    )

    # Save to config/voice
    cfg_dir = Path("config/voice")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(cfg_dir / "reference.wav"), wav_24k, 24000, encoding="PCM_S", bits_per_sample=16)
    with open(cfg_dir / "reference.txt", "w", encoding="utf-8") as f:
        f.write(ref_text)

    # Also save to data/tts_tests
    tests_dir = Path("data/tts_tests")
    torchaudio.save(str(tests_dir / "NP_lemi_ref.wav"), wav_24k, 24000, encoding="PCM_S", bits_per_sample=16)
    with open(tests_dir / "NP_lemi_ref.txt", "w", encoding="utf-8") as f:
        f.write(ref_text)

    peak_db = 20 * np.log10(torch.max(torch.abs(wav_24k)).item())
    rms_db = 20 * np.log10(torch.sqrt(torch.mean(wav_24k**2)).item())
    dur = wav_24k.shape[1] / 24000
    print(f"Saved reference: {dur:.2f}s, peak={peak_db:.2f} dBFS, rms={rms_db:.2f} dBFS")
    return 0

if __name__ == "__main__":
    sys.exit(main())

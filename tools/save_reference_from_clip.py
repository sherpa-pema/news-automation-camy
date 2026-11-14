"""Saves and formats a reference voice clip for Ming-3.0-tts zero-shot cloning."""
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio

def main():
    src_wav = Path(r"C:\Users\Chungba Sherpa\Downloads\female_voice.wav")
    src_mp3 = Path(r"C:\Users\Chungba Sherpa\Downloads\NP_lemi.mp3")

    target_src = src_wav if src_wav.exists() else src_mp3
    if not target_src.exists():
        print(f"Error: Neither {src_wav} nor {src_mp3} exists.")
        return 1

    wav, sr = torchaudio.load(str(target_src))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # If processing female_voice.wav, trim leading/trailing digital silence
    if target_src == src_wav:
        threshold = 10.0 ** (-45.0 / 20.0)
        abs_sig = np.abs(wav.squeeze(0).numpy())
        idx = np.where(abs_sig > threshold)[0]
        start_idx = max(0, idx[0] - int(sr * 0.10))
        end_idx = min(wav.shape[1], idx[-1] + int(sr * 0.15))
        wav = wav[:, start_idx:end_idx]
        ref_text = "In the Game of Football, every player is a hero embarking on their quest."
    else:
        ref_text = (
            "Hi guys, I don't even know where to begin, but one of my biggest dreams "
            "of my career is finally coming true. I'm in Paris. I am here for Paris "
            "Fashion Week as a backstage makeup artist."
        )

    # Resample to 24000 Hz
    wav_24k = torchaudio.transforms.Resample(sr, 24000)(wav) if sr != 24000 else wav

    # Scale peak to -3.0 dBFS
    max_val = torch.max(torch.abs(wav_24k)).item()
    target_peak = 10.0 ** (-3.0 / 20.0)
    if max_val > 0:
        wav_24k = wav_24k * (target_peak / max_val)

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

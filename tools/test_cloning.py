"""Tests Ming-3.0-tts zero-shot voice cloning using config/voice/reference.wav."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torchaudio

VENDOR_DIR = Path("vendor/Ming-3.0-tts").resolve()
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from ming_tts import MingTTS
from spkemb_extractor import SpkembExtractor

def main():
    ref_wav = Path("config/voice/reference.wav")
    ref_txt = Path("config/voice/reference.txt")
    out_dir = Path("data/tts_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ref_txt, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    test_sentence = "Good evening. Welcome to the Nepal Daily Briefing for Thursday, November 6, 2026."

    print("[test_cloning] Loading MingTTS...")
    model = MingTTS("models/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
    extractor = SpkembExtractor("models/Ming-3.0-tts-0.5b/campplus.onnx")

    # Measure reference embedding
    ref_audio, sr = torchaudio.load(str(ref_wav))
    if sr != 16000:
        ref_audio_16k = torchaudio.transforms.Resample(sr, 16000)(ref_audio)
    else:
        ref_audio_16k = ref_audio
    ref_emb = extractor(ref_audio_16k)
    ref_emb_norm = ref_emb / torch.norm(ref_emb, p=2, dim=-1, keepdim=True)

    out_wav = out_dir / "test_cloned_np_lemi.wav"
    print("[test_cloning] Synthesizing cloned sentence...")
    torch.manual_seed(1895)
    t0 = time.time()
    waveform = model.synthesize(
        task="TTS",
        text=test_sentence,
        language_id="en",
        prompt_wav_path=str(ref_wav),
        prompt_text=prompt_text,
        use_spk_emb=True,
        use_zero_spk_emb=False,
        output_path=str(out_wav),
    )
    synth_sec = round(time.time() - t0, 2)
    print(f"[test_cloning] Synthesis completed in {synth_sec}s -> {out_wav}")

    # Inspect generated wav
    gen_audio, gen_sr = torchaudio.load(str(out_wav))
    dur = gen_audio.shape[1] / gen_sr
    peak = 20 * np.log10(torch.max(torch.abs(gen_audio)).item())
    rms = 20 * np.log10(torch.sqrt(torch.mean(gen_audio**2)).item())

    # Pitch
    pitch = torchaudio.functional.detect_pitch_frequency(gen_audio, gen_sr)
    v = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
    f0 = float(v.median()) if v.numel() > 0 else 0

    # Similarity to reference
    if gen_sr != 16000:
        gen_16k = torchaudio.transforms.Resample(gen_sr, 16000)(gen_audio)
    else:
        gen_16k = gen_audio
    gen_emb = extractor(gen_16k)
    gen_emb_norm = gen_emb / torch.norm(gen_emb, p=2, dim=-1, keepdim=True)
    sim = float(torch.matmul(ref_emb_norm, gen_emb_norm.T).item())

    print(f"Metrics: dur={dur:.2f}s, peak={peak:.2f} dBFS, rms={rms:.2f} dBFS, F0={f0:.1f} Hz, CosineSim={sim:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

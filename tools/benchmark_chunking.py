"""Benchmark Chunking A (sentence-by-sentence) vs Chunking B (scene-by-scene)."""
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio

VENDOR_DIR = Path("vendor/Ming-3.0-tts").resolve()
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from ming_tts import MingTTS
from spkemb_extractor import SpkembExtractor
from transformers import pipeline

SCENE_01_SENTENCES = [
    "In Kathmandu, environmental authorities announced the completion of a major riverbank restoration phase along the Bagmati corridor.",
    "Officials confirmed that advanced biological treatment facilities are now operational, noticeably reducing pollutants near historic heritage sites.",
    "Community volunteers gathered in large numbers to plant native shrubs along the embankments.",
]
SCENE_01_FULL = " ".join(SCENE_01_SENTENCES)

def measure_silence_gaps(waveform: torch.Tensor, sample_rate: int, threshold_db: float = -40.0) -> list[float]:
    arr = waveform.abs().cpu().numpy().flatten()
    db = 20.0 * np.log10(np.maximum(arr, 1e-7))
    silent = db < threshold_db
    gaps, gap_len = [], 0
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

def evaluate_wav(wav_path: Path, extractor: SpkembExtractor, ref_emb_norm: torch.Tensor, asr_pipe):
    audio, sr = torchaudio.load(str(wav_path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    dur = audio.shape[1] / sr
    peak = 20 * np.log10(torch.max(torch.abs(audio)).item())
    rms = 20 * np.log10(torch.sqrt(torch.mean(audio**2)).item())

    # Pitch
    pitch = torchaudio.functional.detect_pitch_frequency(audio, sr)
    v = pitch[(pitch >= 60.0) & (pitch <= 600.0)]
    f0_med = float(v.median()) if v.numel() > 0 else 0.0
    f0_std = float(v.std()) if v.numel() > 0 else 0.0

    # Speaker embedding similarity
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio) if sr != 16000 else audio
    emb = extractor(audio_16k)
    emb_norm = emb / torch.norm(emb, p=2, dim=-1, keepdim=True)
    sim = float(torch.matmul(ref_emb_norm, emb_norm.T).item())

    # Silence gaps
    gaps = measure_silence_gaps(audio, sr)
    max_gap = max(gaps) if gaps else 0.0

    # ASR transcription check
    asr_res = asr_pipe(str(wav_path))
    transcribed_text = asr_res.get("text", "").strip()

    return {
        "duration_sec": round(dur, 2),
        "peak_db": round(peak, 2),
        "rms_db": round(rms, 2),
        "f0_median_hz": round(f0_med, 1),
        "f0_std_hz": round(f0_std, 1),
        "cosine_sim_to_ref": round(sim, 4),
        "max_silence_sec": round(max_gap, 2),
        "transcribed_text": transcribed_text,
    }

def main():
    out_dir = Path("data/tts_tests/chunking_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_wav = Path("config/voice/reference.wav")
    ref_txt = Path("config/voice/reference.txt")
    with open(ref_txt, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    print("[benchmark] Initializing models...")
    model = MingTTS("models/Ming-3.0-tts-0.5b", device="cuda:0", text_normalization=False)
    extractor = SpkembExtractor("models/Ming-3.0-tts-0.5b/campplus.onnx")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cuda:0")

    # Reference embedding
    ref_audio, sr = torchaudio.load(str(ref_wav))
    ref_audio_16k = torchaudio.transforms.Resample(sr, 16000)(ref_audio) if sr != 16000 else ref_audio
    ref_emb = extractor(ref_audio_16k)
    ref_emb_norm = ref_emb / torch.norm(ref_emb, p=2, dim=-1, keepdim=True)

    # --- Method A: Sentence-by-sentence ---
    print("\n--- Running Method A: Sentence-by-sentence ---")
    t0 = time.time()
    sent_wavs = []
    sent_metrics = []
    for idx, s in enumerate(SCENE_01_SENTENCES):
        s_wav = out_dir / f"method_a_sent_{idx}.wav"
        torch.manual_seed(1895 + idx)
        model.synthesize(
            task="TTS",
            text=s,
            language_id="en",
            prompt_wav_path=str(ref_wav),
            prompt_text=prompt_text,
            use_spk_emb=True,
            use_zero_spk_emb=False,
            output_path=str(s_wav),
        )
        data, s_sr = sf.read(str(s_wav))
        sf.write(str(s_wav), data, s_sr, subtype="PCM_16")
        sent_wavs.append((data, s_sr))

    # Concat with 0.25s pause
    target_sr = sent_wavs[0][1]
    silence = np.zeros(int(0.25 * target_sr), dtype=np.float32)
    concatenated = []
    for i, (d, s_sr) in enumerate(sent_wavs):
        concatenated.append(d)
        if i < len(sent_wavs) - 1:
            concatenated.append(silence)
    method_a_data = np.concatenate(concatenated)
    method_a_wav = out_dir / "method_a_scene_01.wav"
    sf.write(str(method_a_wav), method_a_data, target_sr, subtype="PCM_16")
    time_a = round(time.time() - t0, 2)
    metrics_a = evaluate_wav(method_a_wav, extractor, ref_emb_norm, asr_pipe)
    metrics_a["wall_time_sec"] = time_a

    # --- Method B: Scene-by-scene ---
    print("\n--- Running Method B: Scene-by-scene ---")
    t0 = time.time()
    method_b_wav = out_dir / "method_b_scene_01.wav"
    torch.manual_seed(1895)
    model.synthesize(
        task="TTS",
        text=SCENE_01_FULL,
        language_id="en",
        prompt_wav_path=str(ref_wav),
        prompt_text=prompt_text,
        use_spk_emb=True,
        use_zero_spk_emb=False,
        output_path=str(method_b_wav),
    )
    data_b, b_sr = sf.read(str(method_b_wav))
    sf.write(str(method_b_wav), data_b, b_sr, subtype="PCM_16")
    time_b = round(time.time() - t0, 2)
    metrics_b = evaluate_wav(method_b_wav, extractor, ref_emb_norm, asr_pipe)
    metrics_b["wall_time_sec"] = time_b

    print("\n=== Benchmark Results ===")
    print("Method A (Sentence-by-sentence):")
    for k, v in metrics_a.items():
        print(f"  {k}: {v}")

    print("\nMethod B (Scene-by-scene):")
    for k, v in metrics_b.items():
        print(f"  {k}: {v}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

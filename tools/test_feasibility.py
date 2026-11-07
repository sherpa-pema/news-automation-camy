import sys
import os
import time
import psutil
import torch
import soundfile as sf

# Add vendor repo to path
sys.path.insert(0, os.path.abspath("vendor/Ming-3.0-tts"))

from ming_tts import MingTTS

PARAGRAPH = (
    "In Kathmandu, environmental cleanup crews achieved a major milestone along the "
    "Bagmati river corridor today, while transport authorities in Sindhupalchok reopened "
    "essential trade links to the north. Meanwhile, autumn tourism numbers surged in "
    "Pokhara with bustling lakeside hotels, and customs officials down in Birgunj reported "
    "record commercial cargo clearance through the southern integrated check post."
)

def test_inference(device="cuda:0"):
    print(f"--- Testing Device: {device} ---")
    model_path = "models/Ming-3.0-tts-0.5b"
    
    t0 = time.time()
    try:
        model = MingTTS(
            model_path=model_path,
            device=device,
            text_normalization=False, # pynini skipped
        )
        load_time = time.time() - t0
        print(f"Model loaded successfully in {load_time:.2f}s on {device}")
    except Exception as e:
        print(f"Error loading model on {device}: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() and "cuda" in device else 0
    print(f"Memory after load: RAM = {ram_mb:.1f} MB, VRAM = {vram_mb:.1f} MB")

    tests = [
        ("test_us_james", "US_James", 1895),
        ("test_us_stella", "US_Stella", 1895),
        ("test_us_james_run1", "US_James", 1895),
        ("test_us_james_run2", "US_James", 1895),
        ("test_us_james_run3", "US_James", 1895),
        # Also test official English IP voices for comparison
        ("test_us_emma", "US_Emma", 1895),
        ("test_uk_george", "UK_George", 1895),
    ]

    results = []
    for test_id, voice, seed in tests:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        out_path = f"data/tts_tests/{test_id}.wav"
        print(f"\nRunning test: {test_id} (voice={voice}, seed={seed})...")

        t_start = time.time()
        try:
            waveform = model.synthesize(
                task="IP",
                text=PARAGRAPH,
                language_id="English",
                controls={"ip": voice},
                use_zero_spk_emb=True,
                output_path=out_path,
            )
            wall_time = time.time() - t_start

            # Read back file to check audio stats
            data, sr = sf.read(out_path)
            duration_sec = len(data) / sr
            rtf = wall_time / duration_sec if duration_sec > 0 else 0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() and "cuda" in device else 0
            peak_ram = process.memory_info().rss / (1024 * 1024)

            max_amp = float(abs(data).max()) if len(data) > 0 else 0
            is_clipped = max_amp >= 0.999

            res = {
                "id": test_id,
                "voice": voice,
                "path": out_path,
                "wall_time_sec": round(wall_time, 2),
                "audio_duration_sec": round(duration_sec, 2),
                "rtf": round(rtf, 2),
                "sample_rate": sr,
                "max_amp": round(max_amp, 4),
                "is_clipped": is_clipped,
                "peak_vram_mb": round(peak_vram, 1),
                "peak_ram_mb": round(peak_ram, 1),
                "status": "success",
            }
            print(f"Success! Wall: {res['wall_time_sec']}s, Dur: {res['audio_duration_sec']}s, RTF: {res['rtf']}, Peak VRAM: {res['peak_vram_mb']}MB, Max Amp: {res['max_amp']}")
            results.append(res)
        except Exception as e:
            print(f"Failed test {test_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": test_id,
                "voice": voice,
                "status": "failed",
                "error": str(e),
            })

    return True, results

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    success, res = test_inference(device)
    if not success and device != "cpu":
        print("\nCUDA failed, falling back to CPU testing...")
        test_inference("cpu")

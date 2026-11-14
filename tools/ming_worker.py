#!/usr/bin/env python3
"""Ming-3.0-TTS Worker Process with Speaker Consistency Enforceability.

Runs inside the isolated .venv-tts environment.
Loads the model once, processes synthesis jobs (supporting task="IP" and task="TTS" zero-shot cloning),
evaluates speaker embedding similarity via CamPP+ ONNX, checks pitch contour (F0),
pre-normalizes item loudness to -3.0 dBFS in memory, retries outlier chunks,
and outputs structured execution metrics.

Also supports an optional persistent HTTP server mode (--serve --port 8765)
so repeated scene edits or pipeline retries avoid cold model reload overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def measure_audio_health_array(data: np.ndarray, sr: int) -> dict[str, Any]:
    """Inspects an in-memory float waveform for duration, peak, RMS, and silence."""
    if data.size == 0 or sr <= 0:
        return {"healthy": False, "reason": "Generated waveform is empty"}

    duration_sec = len(data) / sr
    if duration_sec < 0.1:
        return {"healthy": False, "reason": f"Audio duration too short: {duration_sec:.2f}s"}

    max_amp = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
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


def measure_audio_health(file_path: Path | str) -> dict[str, Any]:
    """Inspects generated audio file on disk for duration, peak, RMS, and silence."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"healthy": False, "reason": "File does not exist or is empty"}

    try:
        data, sr = sf.read(str(path))
    except Exception as exc:
        return {"healthy": False, "reason": f"Corrupt audio file: {exc}"}

    return measure_audio_health_array(np.asarray(data, dtype=np.float32), sr)


def pre_normalize_audio(audio_data: np.ndarray, target_peak_db: float = -3.0) -> np.ndarray:
    """Scales audio cleanly so its maximum peak matches target_peak_db."""
    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp <= 1e-9:
        return audio_data.astype(np.float32, copy=False)
    target_amp = 10.0 ** (target_peak_db / 20.0)
    scaling_factor = target_amp / max_amp
    return (audio_data * scaling_factor).astype(np.float32)


class MingWorkerRuntime:
    """Holds a loaded MingTTS model and handles batched synthesis with consistency gating."""

    def __init__(
        self,
        model_path: str = "models/Ming-3.0-tts-0.5b",
        campplus_path: str = "models/Ming-3.0-tts-0.5b/campplus.onnx",
        device: str | None = None,
        dtype: str = "bfloat16",
        text_normalization: bool = False,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.dtype = dtype
        self.text_normalization = text_normalization
        self._lock = threading.Lock()
        self._ref_cache: dict[str, tuple[torch.Tensor | None, float, str | None]] = {}

        t_load_start = time.time()
        print(f"[ming_worker] Initializing MingTTS on device={self.device} dtype={self.dtype}...")
        self.model = MingTTS(
            model_path=model_path,
            device=self.device,
            text_normalization=text_normalization,
            torch_dtype=self.dtype,
        )
        # Reuse MingTTS's already-initialized SpkembExtractor instead of loading ONNX twice
        self.extractor = getattr(self.model, "spkemb_extractor", None) or SpkembExtractor(campplus_path)
        self.sample_rate = int(getattr(self.model, "sample_rate", 44100))
        self._resampler_16k = (
            torchaudio.transforms.Resample(self.sample_rate, 16000)
            if self.sample_rate != 16000
            else None
        )
        self.model_load_time_sec = time.time() - t_load_start
        print(f"[ming_worker] Model loaded in {self.model_load_time_sec:.2f}s (sr={self.sample_rate}Hz)")

    def _get_reference(self, reference_wav: str, reference_txt: str) -> tuple[torch.Tensor | None, float, str | None]:
        cache_key = f"{reference_wav}|{reference_txt}"
        if cache_key in self._ref_cache:
            return self._ref_cache[cache_key]

        ref_emb_norm: torch.Tensor | None = None
        ref_f0 = 0.0
        ref_prompt_text: str | None = None

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
            ref_emb = self.extractor(ref_audio_16k)
            ref_emb_norm = ref_emb / torch.norm(ref_emb, p=2, dim=-1, keepdim=True)

            ref_txt_path = Path(reference_txt)
            if ref_txt_path.exists():
                with open(ref_txt_path, "r", encoding="utf-8") as f:
                    ref_prompt_text = f.read().strip()

            print(
                f"[ming_worker] Loaded reference: {ref_path.name} "
                f"(F0={ref_f0:.1f}Hz, prompt_len={len(ref_prompt_text or '')} chars)"
            )

        self._ref_cache[cache_key] = (ref_emb_norm, ref_f0, ref_prompt_text)
        return ref_emb_norm, ref_f0, ref_prompt_text

    def process_job(self, job: dict[str, Any], include_load_time: bool = True) -> dict[str, Any]:
        with self._lock:
            task = job.get("task", "TTS")
            voice = job.get("voice", "US_Emma")
            raw_lang = job.get("language_id", "en")
            language_id = "en" if str(raw_lang).strip().lower() in ("english", "en") else raw_lang
            instruction_schema = job.get("instruction_schema", "en")
            base_seed = int(job.get("seed", 1895))
            speed = job.get("speed")
            cfg = float(job.get("cfg", 2.0))
            sigma = float(job.get("sigma", 0.25))
            temperature = float(job.get("temperature", 0.0))
            max_length = int(job.get("max_length", 24))
            crossfade_ms = float(job.get("crossfade_ms", 12.0))
            items = job.get("items", [])

            reference_wav = job.get("reference_wav", "config/voice/reference.wav")
            reference_txt = job.get("reference_txt", "config/voice/reference.txt")
            similarity_threshold = float(job.get("similarity_threshold", 0.72))
            f0_tolerance_pct = float(job.get("f0_tolerance_pct", 0.25))
            max_retries = int(job.get("max_retries", 2))

            ref_path = Path(reference_wav)
            ref_emb_norm, ref_f0, ref_prompt_text = self._get_reference(reference_wav, reference_txt)

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

                    try:
                        if task == "TTS" and ref_prompt_text:
                            waveform_t = self.model.synthesize(
                                task="TTS",
                                text=text,
                                language_id=language_id,
                                prompt_wav_path=str(ref_path),
                                prompt_text=ref_prompt_text,
                                use_spk_emb=True,
                                use_zero_spk_emb=False,
                                cfg=cfg,
                                sigma=sigma,
                                temperature=temperature,
                                max_length=max_length,
                                crossfade_ms=crossfade_ms,
                                output_path=None,
                            )
                        else:
                            waveform_t = self.model.synthesize(
                                task=task,
                                text=text,
                                language_id=language_id,
                                controls=controls,
                                instruction_schema=instruction_schema,
                                use_zero_spk_emb=True,
                                cfg=cfg,
                                sigma=sigma,
                                temperature=temperature,
                                max_length=max_length,
                                crossfade_ms=crossfade_ms,
                                output_path=None,
                            )

                        # Evaluate completely in-memory before writing winning candidate once
                        w_cpu = waveform_t.detach().cpu().float()
                        if w_cpu.ndim > 1 and w_cpu.shape[0] > 1:
                            w_cpu = w_cpu.mean(dim=0, keepdim=True)
                        elif w_cpu.ndim == 1:
                            w_cpu = w_cpu.unsqueeze(0)

                        audio_np = w_cpu.squeeze(0).numpy()
                        norm_data = pre_normalize_audio(audio_np, target_peak_db=-3.0)
                        audio_sr = self.sample_rate

                        health = measure_audio_health_array(norm_data, audio_sr)
                        if not health["healthy"]:
                            raise ValueError(health.get("reason", "Unhealthy audio"))

                        tensor_wave = torch.from_numpy(norm_data).unsqueeze(0)
                        cand_f0 = compute_f0(tensor_wave, audio_sr)

                        cand_16k = self._resampler_16k(tensor_wave) if self._resampler_16k is not None else tensor_wave
                        if cand_16k.shape[-1] < 400:
                            cand_16k = torch.nn.functional.pad(cand_16k, (0, 400 - cand_16k.shape[-1]))

                        cand_emb = self.extractor(cand_16k)
                        cand_emb_norm = cand_emb / torch.norm(cand_emb, p=2, dim=-1, keepdim=True)

                        if ref_emb_norm is not None:
                            sim = float(torch.matmul(ref_emb_norm, cand_emb_norm.T).item())
                        else:
                            sim = 1.0

                        f0_ok = True
                        if ref_f0 > 0 and cand_f0 > 0:
                            f0_diff = abs(cand_f0 - ref_f0) / ref_f0
                            if f0_diff > f0_tolerance_pct:
                                f0_ok = False

                        sim_ok = sim >= similarity_threshold
                        score = sim - (0.5 if not f0_ok else 0.0)

                        cand_info = {
                            "attempt": attempt,
                            "norm_data": norm_data,
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
                            break
                        else:
                            print(
                                f"[ming_worker] {item_id} attempt {attempt + 1}: sim={sim:.4f} "
                                f"(gate={similarity_threshold}), F0={cand_f0:.1f}Hz "
                                f"(gate={ref_f0:.1f}Hz +-{int(f0_tolerance_pct*100)}%). Retrying..."
                            )
                    except Exception as e:
                        print(f"[ming_worker] {item_id} attempt {attempt + 1} error: {e}")

                seconds_taken = time.time() - t_synth_start

                if best_candidate is not None:
                    sf.write(str(out_path), best_candidate["norm_data"], best_candidate["sample_rate"], subtype="PCM_16")
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
                        f"[ming_worker] [OK] {item_id}: dur={best_candidate['duration_sec']:.2f}s, "
                        f"sim={best_candidate['cosine_sim']}, F0={best_candidate['f0_median_hz']}Hz "
                        f"(retries={best_candidate['attempt']})"
                    )
                else:
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
            return {
                "status": "error" if has_failures else "success",
                "model_load_time_sec": round(self.model_load_time_sec, 2) if include_load_time else 0.0,
                "reference_wav": reference_wav,
                "reference_f0_hz": ref_f0,
                "similarity_threshold": similarity_threshold,
                "results": results,
            }


def run_worker(job_path: Path, result_path: Path) -> int:
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    try:
        runtime = MingWorkerRuntime(
            model_path=job.get("model_path", "models/Ming-3.0-tts-0.5b"),
            campplus_path=job.get("campplus_path", "models/Ming-3.0-tts-0.5b/campplus.onnx"),
            device=job.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype=job.get("dtype", "bfloat16"),
            text_normalization=bool(job.get("text_normalization", False)),
        )
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

    worker_report = runtime.process_job(job, include_load_time=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(worker_report, f, indent=2)

    return 1 if worker_report["status"] == "error" else 0


def serve_worker(
    host: str,
    port: int,
    model_path: str,
    campplus_path: str,
    device: str | None,
    dtype: str,
    text_normalization: bool,
) -> int:
    runtime = MingWorkerRuntime(
        model_path=model_path,
        campplus_path=campplus_path,
        device=device,
        dtype=dtype,
        text_normalization=text_normalization,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "model_path": runtime.model_path,
                    "device": runtime.device,
                    "sample_rate": runtime.sample_rate,
                })
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/synthesize":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                job = json.loads(raw.decode("utf-8"))
                report = runtime.process_job(job, include_load_time=False)
                self._send_json(200, report)
            except Exception as exc:
                self._send_json(500, {"status": "fatal_error", "error": str(exc), "results": []})

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress noisy per-request HTTP stderr logs
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[ming_worker] Persistent TTS daemon listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[ming_worker] Shutting down TTS daemon.")
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ming-3.0-TTS Subprocess Worker / Daemon")
    parser.add_argument("--job-file", help="Path to job JSON file (one-shot mode)")
    parser.add_argument("--result-file", help="Path to result JSON output (one-shot mode)")
    parser.add_argument("--serve", action="store_true", help="Run as a persistent local HTTP daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Daemon bind host")
    parser.add_argument("--port", type=int, default=8765, help="Daemon bind port")
    parser.add_argument("--model-path", default="models/Ming-3.0-tts-0.5b")
    parser.add_argument("--campplus-path", default="models/Ming-3.0-tts-0.5b/campplus.onnx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--text-normalization", action="store_true")
    args = parser.parse_args()

    if args.serve:
        return serve_worker(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            campplus_path=args.campplus_path,
            device=args.device,
            dtype=args.dtype,
            text_normalization=args.text_normalization,
        )

    if not args.job_file or not args.result_file:
        parser.error("--job-file and --result-file are required unless --serve is specified")

    job_path = Path(args.job_file)
    result_path = Path(args.result_file)

    if not job_path.exists():
        print(f"FATAL: Job file not found: {job_path}", file=sys.stderr)
        return 1

    return run_worker(job_path, result_path)


if __name__ == "__main__":
    sys.exit(main())

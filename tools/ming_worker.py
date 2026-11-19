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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "Ming-3.0-tts"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ming_tts import MingTTS
    from spkemb_extractor import SpkembExtractor
except ImportError as err:
    print(f"FATAL: Failed to import MingTTS/SpkembExtractor from {VENDOR_DIR}: {err}", file=sys.stderr)
    sys.exit(2)

from src.utils.tts import compute_candidate_score


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


def pre_normalize_audio(audio_data: np.ndarray, target_peak_db: float = -6.0) -> np.ndarray:
    """Scales audio cleanly so its maximum peak matches target_peak_db."""
    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp <= 1e-9:
        return audio_data.astype(np.float32, copy=False)
    target_amp = 10.0 ** (target_peak_db / 20.0)
    scaling_factor = target_amp / max_amp
    return (audio_data * scaling_factor).astype(np.float32)


class ModelStageTimer:
    """Accumulates fine-grained stage timings across prefill, AR decoding, CFM, VAE, QC, and write."""

    def __init__(self) -> None:
        self.prefill_sec = 0.0
        self.ar_sec = 0.0
        self.cfm_sec = 0.0
        self.vae_sec = 0.0
        self.qc_sec = 0.0
        self.write_sec = 0.0

    def reset(self) -> None:
        self.prefill_sec = 0.0
        self.ar_sec = 0.0
        self.cfm_sec = 0.0
        self.vae_sec = 0.0
        self.qc_sec = 0.0
        self.write_sec = 0.0

    def get_breakdown(self) -> dict[str, float]:
        return {
            "prefill": round(self.prefill_sec, 3),
            "ar_step": round(self.ar_sec, 3),
            "cfm": round(self.cfm_sec, 3),
            "vae_decode": round(self.vae_sec, 3),
            "qc": round(self.qc_sec, 3),
            "write": round(self.write_sec, 3),
        }


def instrument_ming_model(ming_tts_instance: Any, timer: ModelStageTimer) -> None:
    """Hooks timing wrappers around prefill, CFM, AR forward, and VAE decode."""
    inner = getattr(ming_tts_instance, "model", None)
    if inner is None:
        return

    # 1. Prefill
    if hasattr(inner, "prepare_input_embed"):
        orig_prepare = inner.prepare_input_embed

        def timed_prepare(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_prepare(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.prefill_sec += time.perf_counter() - t0
            return res

        inner.prepare_input_embed = timed_prepare

    # 2. CFM
    if hasattr(inner, "flowloss") and hasattr(inner.flowloss, "sample"):
        orig_cfm = inner.flowloss.sample

        def timed_cfm(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_cfm(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.cfm_sec += time.perf_counter() - t0
            return res

        inner.flowloss.sample = timed_cfm

    # 3. AR Step
    if hasattr(inner, "model") and hasattr(inner.model, "forward"):
        orig_forward = inner.model.forward

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_forward(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.ar_sec += time.perf_counter() - t0
            return res

        inner.model.forward = timed_forward

    # 4. VAE decode
    if hasattr(inner, "audio") and hasattr(inner.audio, "decode"):
        orig_decode = inner.audio.decode

        def timed_decode(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            res = orig_decode(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.vae_sec += time.perf_counter() - t0
            return res

        inner.audio.decode = timed_decode


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
        self.timer = ModelStageTimer()

        t_load_start = time.time()
        print(f"[ming_worker] Initializing MingTTS on device={self.device} dtype={self.dtype}...")
        self.model = MingTTS(
            model_path=model_path,
            device=self.device,
            text_normalization=text_normalization,
            torch_dtype=self.dtype,
        )
        instrument_ming_model(self.model, self.timer)
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
            f0_tolerance_pct = float(job.get("f0_tolerance_pct", 0.10))
            n_candidates = int(job.get("n_candidates", 1))
            f0_anchor = str(job.get("f0_anchor", "running_median")).strip().lower()
            ranker_weights = job.get("ranker_weights", {"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2})
            max_retries = int(job.get("max_retries", 2))

            ref_path = Path(reference_wav)
            ref_emb_norm, ref_f0, ref_prompt_text = self._get_reference(reference_wav, reference_txt)

            self.timer.reset()
            results = []
            accepted_f0s: list[float] = [ref_f0] if ref_f0 > 0 else []

            for item_idx, item in enumerate(items):
                item_id = item["id"]
                text = item["text"]
                out_path = Path(item["out_path"])
                out_path.parent.mkdir(parents=True, exist_ok=True)

                words = len(text.split())
                expected_dur = max(1.0, words / 2.4)

                if f0_anchor == "running_median" and accepted_f0s:
                    target_f0 = float(np.median(accepted_f0s))
                else:
                    target_f0 = ref_f0

                controls = {"ip": voice}
                if speed:
                    controls["speed"] = speed

                t_synth_start = time.time()
                candidates_evaluated = []
                best_candidate: dict[str, Any] | None = None
                best_score = -999999.0

                num_attempts = n_candidates if n_candidates > 1 else (max_retries + 1)

                for attempt in range(num_attempts):
                    seed = base_seed + item_idx * 100 + attempt
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
                        worker_peak_db = float(job.get("target_peak_db", -6.0))
                        norm_data = pre_normalize_audio(audio_np, target_peak_db=worker_peak_db)
                        audio_sr = self.sample_rate

                        health = measure_audio_health_array(norm_data, audio_sr)
                        if not health["healthy"]:
                            raise ValueError(health.get("reason", "Unhealthy audio"))

                        t_qc = time.perf_counter()
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
                        self.timer.qc_sec += time.perf_counter() - t_qc

                        # Artifact score
                        art_score = 1.0
                        if health.get("has_excess_silence"):
                            art_score -= 0.3
                        if health.get("clipping_margin_db", 1.0) < 0.1:
                            art_score -= 0.4
                        art_score = max(0.0, art_score)

                        # Continuous Ranker Score
                        score, breakdown = compute_candidate_score(
                            cosine_sim=sim,
                            cand_f0=cand_f0,
                            target_f0=target_f0,
                            duration_sec=health["duration_sec"],
                            expected_duration_sec=expected_dur,
                            artifact_score=art_score,
                            weights=ranker_weights,
                        )

                        sim_ok = sim >= similarity_threshold
                        f0_ok = True
                        if ref_f0 > 0 and cand_f0 > 0:
                            f0_diff = abs(cand_f0 - ref_f0) / ref_f0
                            if f0_diff > f0_tolerance_pct:
                                f0_ok = False

                        passes_gate = sim_ok and f0_ok

                        cand_info = {
                            "candidate_idx": attempt,
                            "seed": seed,
                            "norm_data": norm_data,
                            "duration_sec": health["duration_sec"],
                            "sample_rate": health["sample_rate"],
                            "peak_db": health["peak_db"],
                            "rms_db": health["rms_db"],
                            "f0_median_hz": cand_f0,
                            "target_f0_hz": target_f0,
                            "cosine_sim": round(sim, 4),
                            "passes_gate": passes_gate,
                            "ranker_score": round(score, 4),
                            "ranker_breakdown": breakdown,
                        }
                        candidates_evaluated.append(cand_info)

                        cand_passes = cand_info.get("passes_gate", False)
                        best_passes = best_candidate.get("passes_gate", False) if best_candidate else False
                        if best_candidate is None or (cand_passes and not best_passes) or (cand_passes == best_passes and score > best_score):
                            best_score = score
                            best_candidate = cand_info

                        # When n_candidates == 1, early break if passes gate
                        if n_candidates == 1 and passes_gate:
                            break
                        elif n_candidates == 1:
                            print(
                                f"[ming_worker] {item_id} attempt {attempt + 1}: sim={sim:.4f} "
                                f"(gate={similarity_threshold}), F0={cand_f0:.1f}Hz "
                                f"(gate={ref_f0:.1f}Hz +-{int(f0_tolerance_pct*100)}%). Retrying..."
                            )
                    except Exception as e:
                        print(f"[ming_worker] {item_id} attempt {attempt + 1} error: {e}")

                seconds_taken = time.time() - t_synth_start

                if best_candidate is not None:
                    t_write = time.perf_counter()
                    sf.write(str(out_path), best_candidate["norm_data"], best_candidate["sample_rate"], subtype="PCM_16")
                    self.timer.write_sec += time.perf_counter() - t_write

                    if best_candidate["f0_median_hz"] > 0:
                        accepted_f0s.append(best_candidate["f0_median_hz"])

                    clean_cands = [
                        {k: v for k, v in c.items() if k != "norm_data"}
                        for c in candidates_evaluated
                    ]
                    results.append({
                        "id": item_id,
                        "status": "success",
                        "out_path": str(out_path),
                        "duration_sec": round(best_candidate["duration_sec"], 3),
                        "sample_rate": best_candidate["sample_rate"],
                        "seconds_taken": round(seconds_taken, 3),
                        "peak_db": round(best_candidate["peak_db"], 2),
                        "rms_db": round(best_candidate["rms_db"], 2),
                        "f0_median_hz": round(best_candidate["f0_median_hz"], 1),
                        "target_f0_hz": round(best_candidate["target_f0_hz"], 1),
                        "cosine_sim": best_candidate["cosine_sim"],
                        "passes_gate": best_candidate["passes_gate"],
                        "retries": len(candidates_evaluated) - 1,
                        "selected_candidate_idx": best_candidate["candidate_idx"],
                        "seed_used": best_candidate["seed"],
                        "ranker_score": best_candidate["ranker_score"],
                        "ranker_breakdown": best_candidate["ranker_breakdown"],
                        "candidates_evaluated": clean_cands,
                        "error": None,
                    })
                    print(
                        f"[ming_worker] [OK] {item_id}: dur={best_candidate['duration_sec']:.2f}s, "
                        f"sim={best_candidate['cosine_sim']}, F0={best_candidate['f0_median_hz']:.1f}Hz "
                        f"(cand={best_candidate['candidate_idx']}/{len(candidates_evaluated)}, score={best_candidate['ranker_score']})"
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
                "stage_breakdown_sec": {
                    "model_load": round(self.model_load_time_sec, 2) if include_load_time else 0.0,
                    **self.timer.get_breakdown(),
                },
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
    idle_ttl_sec: int = 1800,
) -> int:
    import os

    runtime = MingWorkerRuntime(
        model_path=model_path,
        campplus_path=campplus_path,
        device=device,
        dtype=dtype,
        text_normalization=text_normalization,
    )

    last_active_time = time.time()
    active_lock = threading.Lock()
    shutdown_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            nonlocal last_active_time
            if self.path == "/health":
                with active_lock:
                    idle_sec = round(time.time() - last_active_time, 1)
                self._send_json(200, {
                    "status": "ok",
                    "model_path": runtime.model_path,
                    "device": runtime.device,
                    "dtype": runtime.dtype,
                    "text_normalization": runtime.text_normalization,
                    "sample_rate": runtime.sample_rate,
                    "idle_seconds": idle_sec,
                    "pid": os.getpid(),
                })
            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            nonlocal last_active_time
            if self.path == "/shutdown":
                self._send_json(200, {"status": "shutting_down", "pid": os.getpid()})
                print("[ming_worker] Received shutdown request via /shutdown")
                shutdown_event.set()
                threading.Thread(target=server.shutdown).start()
                return

            if self.path != "/synthesize":
                self._send_json(404, {"error": "Not found"})
                return

            with active_lock:
                last_active_time = time.time()

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                job = json.loads(raw.decode("utf-8"))
                report = runtime.process_job(job, include_load_time=False)
                with active_lock:
                    last_active_time = time.time()
                self._send_json(200, report)
            except Exception as exc:
                with active_lock:
                    last_active_time = time.time()
                self._send_json(500, {"status": "fatal_error", "error": str(exc), "results": []})

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress noisy per-request HTTP stderr logs
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"[ming_worker] Persistent TTS daemon listening on http://{host}:{port} "
        f"(PID={os.getpid()}, idle_ttl={idle_ttl_sec}s)"
    )

    def idle_checker() -> None:
        while not shutdown_event.is_set():
            time.sleep(5)
            if idle_ttl_sec > 0:
                with active_lock:
                    idle = time.time() - last_active_time
                if idle >= idle_ttl_sec:
                    print(f"[ming_worker] Daemon idle for {idle:.0f}s >= {idle_ttl_sec}s. Exiting to release VRAM...")
                    shutdown_event.set()
                    server.shutdown()
                    break

    checker_thread = threading.Thread(target=idle_checker, daemon=True)
    checker_thread.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[ming_worker] Shutting down TTS daemon.")
    finally:
        shutdown_event.set()
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ming-3.0-TTS Subprocess Worker / Daemon")
    parser.add_argument("--job-file", help="Path to job JSON file (one-shot mode)")
    parser.add_argument("--result-file", help="Path to result JSON output (one-shot mode)")
    parser.add_argument("--serve", action="store_true", help="Run as a persistent local HTTP daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Daemon bind host")
    parser.add_argument("--port", type=int, default=8766, help="Daemon bind port")
    parser.add_argument("--model-path", default="models/Ming-3.0-tts-0.5b")
    parser.add_argument("--campplus-path", default="models/Ming-3.0-tts-0.5b/campplus.onnx")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--text-normalization", action="store_true")
    parser.add_argument("--idle-ttl-sec", type=int, default=1800, help="Idle timeout before auto-exit (sec)")
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
            idle_ttl_sec=args.idle_ttl_sec,
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

"""TTS Engine abstraction and audio processing utilities.

Supports swappable TTS engines:
- MingEngine: Invokes tools/ming_worker.py inside isolated .venv-tts via subprocess.
- FakeTTS: Pure Python test double producing valid WAV audio without dependencies.
"""
from __future__ import annotations

import json
import logging
import math
import re
import struct
import subprocess
import tempfile
import urllib.error
import urllib.request
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class TTSEngine(ABC):
    """Abstract TTS engine interface."""

    @abstractmethod
    def synthesize_batch(
        self,
        items: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """Synthesizes a list of items [{id, text, out_path}].

        Returns a dictionary containing:
            - status: "success" or "error"
            - model_load_time_sec: float
            - results: list of item result dicts
        """
        pass


class MingEngine(TTSEngine):
    """Ming-3.0-TTS engine running via persistent daemon or subprocess in .venv-tts."""

    def __init__(
        self,
        python_path: str | Path = ".venv-tts/Scripts/python.exe",
        worker_script: str | Path = "tools/ming_worker.py",
        timeout_sec: int = 300,
        daemon_url: str | None = None,
    ) -> None:
        self.python_path = Path(python_path)
        self.worker_script = Path(worker_script)
        self.timeout_sec = timeout_sec
        self.daemon_url = daemon_url

    def _check_daemon_health(self, daemon_url: str, job_payload: dict[str, Any]) -> bool:
        """Handshakes with persistent TTS daemon via GET /health and validates compatibility."""
        health_url = f"{daemon_url.rstrip('/')}/health"
        req = urllib.request.Request(health_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status != 200:
                    return False
                health_data = json.loads(resp.read().decode("utf-8"))
                if health_data.get("status") != "ok":
                    return False

                mismatches = []
                job_model = Path(job_payload.get("model_path", "")).resolve()
                daemon_model = Path(health_data.get("model_path", "")).resolve()
                if job_model != daemon_model and str(job_payload.get("model_path")) != str(health_data.get("model_path")):
                    mismatches.append(f"model_path ({daemon_model} vs {job_model})")

                if str(health_data.get("device")) != str(job_payload.get("device", "cuda:0")):
                    mismatches.append(f"device ({health_data.get('device')} vs {job_payload.get('device')})")

                if str(health_data.get("dtype")) != str(job_payload.get("dtype", "bfloat16")):
                    mismatches.append(f"dtype ({health_data.get('dtype')} vs {job_payload.get('dtype')})")

                if bool(health_data.get("text_normalization")) != bool(job_payload.get("text_normalization", False)):
                    mismatches.append(f"text_normalization ({health_data.get('text_normalization')} vs {job_payload.get('text_normalization')})")

                if mismatches:
                    logger = logging.getLogger("MingEngine")
                    logger.warning(
                        f"TTS daemon at {daemon_url} configuration mismatch: {', '.join(mismatches)}. "
                        "Falling back to subprocess worker."
                    )
                    return False
                return True
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            return False

    def _try_daemon(self, job_payload: dict[str, Any], daemon_url: str) -> dict[str, Any] | None:
        if not self._check_daemon_health(daemon_url, job_payload):
            return None

        url = f"{daemon_url.rstrip('/')}/synthesize"
        req = urllib.request.Request(
            url,
            data=json.dumps(job_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                result_data = json.loads(resp.read().decode("utf-8"))
                if result_data.get("status") == "fatal_error":
                    raise RuntimeError(
                        f"TTS daemon encountered fatal error: {result_data.get('error')}"
                    )
                return result_data
        except (urllib.error.URLError, ConnectionError, OSError):
            return None

    def synthesize_batch(
        self,
        items: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        if not items:
            return {"status": "success", "model_load_time_sec": 0.0, "results": []}

        job_payload = {
            "model_path": settings.get("model_path", "models/Ming-3.0-tts-0.5b"),
            "task": settings.get("task", "TTS"),
            "voice": settings.get("voice", "US_Emma"),
            "language_id": settings.get("language_id", "en"),
            "instruction_schema": settings.get("instruction_schema", "en"),
            "device": settings.get("device", "cuda:0"),
            "dtype": settings.get("dtype", "bfloat16"),
            "seed": settings.get("seed", 1895),
            "speed": settings.get("speed"),
            "cfg": settings.get("cfg", 2.0),
            "sigma": settings.get("sigma", 0.25),
            "temperature": settings.get("temperature", 0.0),
            "max_length": settings.get("max_length", 24),
            "crossfade_ms": settings.get("crossfade_ms", 12.0),
            "text_normalization": settings.get("text_normalization", False),
            "reference_wav": settings.get("reference_wav", "config/voice/reference.wav"),
            "reference_txt": settings.get("reference_txt", "config/voice/reference.txt"),
            "similarity_threshold": settings.get("similarity_threshold", 0.72),
            "f0_tolerance_pct": settings.get("f0_tolerance_pct", 0.10),
            "n_candidates": settings.get("n_candidates", 1),
            "f0_anchor": settings.get("f0_anchor", "running_median"),
            "ranker_weights": settings.get("ranker_weights", {"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2}),
            "max_retries": settings.get("max_retries", 2),
            "items": items,
        }

        effective_daemon_url = settings.get("daemon_url") or self.daemon_url
        if effective_daemon_url:
            daemon_res = self._try_daemon(job_payload, str(effective_daemon_url))
            if daemon_res is not None:
                return daemon_res

        if not self.python_path.exists():
            raise FileNotFoundError(
                f"Subprocess Python interpreter not found at: {self.python_path}. "
                "Ensure .venv-tts is set up."
            )
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"Worker script not found at: {self.worker_script}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            job_file = Path(tmpdir) / "job.json"
            result_file = Path(tmpdir) / "result.json"

            with open(job_file, "w", encoding="utf-8") as f:
                json.dump(job_payload, f, indent=2)

            cmd = [
                str(self.python_path),
                str(self.worker_script),
                "--job-file",
                str(job_file),
                "--result-file",
                str(result_file),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"TTS worker process timed out after {self.timeout_sec}s. "
                    f"Output so far: {exc.stdout}"
                ) from exc

            if proc.returncode != 0 and not result_file.exists():
                raise RuntimeError(
                    f"TTS worker process exited with code {proc.returncode}.\n"
                    f"STDERR: {proc.stderr}\nSTDOUT: {proc.stdout}"
                )

            if not result_file.exists():
                raise FileNotFoundError(
                    f"TTS worker did not generate result file.\nSTDERR: {proc.stderr}"
                )

            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)

            if result_data.get("status") == "fatal_error":
                raise RuntimeError(
                    f"TTS worker encountered fatal error: {result_data.get('error')}"
                )

            return result_data


def compute_candidate_score(
    cosine_sim: float,
    cand_f0: float,
    target_f0: float,
    duration_sec: float,
    expected_duration_sec: float,
    artifact_score: float = 1.0,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Computes continuous Voice Quality Ranker score (Phase Q2).

    score = w1*cosine + w2*(1 - |f0 - target_f0|/target_f0) + w3*duration_plausibility + w4*artifact_score
    """
    if weights is None:
        weights = {"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2}

    w_cos = float(weights.get("cosine", 1.0))
    w_f0 = float(weights.get("f0", 0.6))
    w_dur = float(weights.get("duration", 0.2))
    w_art = float(weights.get("artifacts", 0.2))

    t_cos = max(-1.0, min(1.0, float(cosine_sim)))

    if target_f0 > 0.0 and cand_f0 > 0.0:
        f0_dev = abs(cand_f0 - target_f0) / target_f0
        t_f0 = max(0.0, 1.0 - f0_dev)
    else:
        t_f0 = 1.0

    if expected_duration_sec > 0.0 and duration_sec > 0.0:
        dur_dev = abs(duration_sec - expected_duration_sec) / expected_duration_sec
        t_dur = max(0.0, 1.0 - dur_dev)
    else:
        t_dur = 1.0

    t_art = max(0.0, min(1.0, float(artifact_score)))

    total = (w_cos * t_cos) + (w_f0 * t_f0) + (w_dur * t_dur) + (w_art * t_art)
    breakdown = {
        "cosine_term": round(t_cos, 4),
        "f0_term": round(t_f0, 4),
        "duration_term": round(t_dur, 4),
        "artifact_term": round(t_art, 4),
        "total_score": round(total, 4),
    }
    return total, breakdown


class FakeTTS(TTSEngine):
    """In-memory test double generating pure Python WAV files."""

    def __init__(self, sample_rate: int = 44100, words_per_sec: float = 3.0) -> None:
        self.sample_rate = sample_rate
        self.words_per_sec = words_per_sec

    def synthesize_batch(
        self,
        items: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        n_candidates = int(settings.get("n_candidates", 1))
        f0_anchor = str(settings.get("f0_anchor", "running_median")).strip().lower()
        ranker_weights = settings.get("ranker_weights", {"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2})
        base_seed = int(settings.get("seed", 1895))
        ref_f0 = float(settings.get("reference_f0_hz", 216.0))
        accepted_f0s: list[float] = [ref_f0] if ref_f0 > 0 else []

        results = []
        for item_idx, item in enumerate(items):
            item_id = item["id"]
            text = item.get("text", "")
            out_path = Path(item["out_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if "[FAIL_PERMANENT]" in text:
                results.append({
                    "id": item_id,
                    "status": "failed",
                    "out_path": str(out_path),
                    "duration_sec": 0.0,
                    "sample_rate": 0,
                    "seconds_taken": 0.01,
                    "peak_db": -999.0,
                    "rms_db": -999.0,
                    "retries": 1,
                    "error": "Simulated permanent failure",
                })
                continue

            words = text.split()
            word_count = max(1, len(words))
            duration_sec = max(0.4, round(word_count / self.words_per_sec, 2))
            expected_dur = max(1.0, word_count / 2.4)

            if f0_anchor == "running_median" and accepted_f0s:
                sorted_f0s = sorted(accepted_f0s)
                mid = len(sorted_f0s) // 2
                target_f0 = (sorted_f0s[mid] if len(sorted_f0s) % 2 == 1 else (sorted_f0s[mid - 1] + sorted_f0s[mid]) / 2.0)
            else:
                target_f0 = ref_f0

            # Generate synthetic 440 Hz tone (or near-silence if [EMPTY])
            num_samples = int(duration_sec * self.sample_rate)
            amplitude = 8000  # -12 dBFS
            if "[EMPTY]" in text:
                amplitude = 0
            elif "[CLIP]" in text:
                amplitude = 32767

            with wave.open(str(out_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                frames = bytearray()
                for i in range(num_samples):
                    if amplitude == 0:
                        val = 0
                    else:
                        val = int(amplitude * math.sin(2.0 * math.pi * 440.0 * (i / self.sample_rate)))
                        val = max(-32768, min(32767, val))
                    frames.extend(struct.pack("<h", val))
                wav_file.writeframes(frames)

            # Evaluate candidate(s)
            num_attempts = n_candidates if n_candidates > 1 else 1
            candidates_eval = []
            for attempt in range(num_attempts):
                seed = base_seed + item_idx * 100 + attempt
                cand_cos = 0.85
                cand_f0 = 216.0
                cand_art = 1.0
                p_gate = True

                if "TRIGGER_LOW_SIM" in text or "[LOW_SIM]" in text:
                    cand_cos = 0.50
                    p_gate = False
                elif "TRIGGER_HIGH_PITCH" in text or "[HIGH_PITCH]" in text:
                    cand_f0 = 350.0
                    p_gate = False
                elif n_candidates > 1:
                    if "TRIGGER_PREFER_FIRST" in text or "TRIGGER_PREFER_CAND_0" in text or "TRIGGER_PREFER_CAND_zero" in text:
                        cand_cos = 0.95 if attempt == 0 else 0.70
                        cand_f0 = target_f0 if attempt == 0 else (target_f0 + 30.0)
                    else:
                        cand_cos = 0.82 if attempt == 0 else 0.88
                        cand_f0 = (target_f0 - 15.0) if attempt == 0 else target_f0

                c_score, c_breakdown = compute_candidate_score(
                    cosine_sim=cand_cos,
                    cand_f0=cand_f0,
                    target_f0=target_f0,
                    duration_sec=duration_sec,
                    expected_duration_sec=expected_dur,
                    artifact_score=cand_art,
                    weights=ranker_weights,
                )
                candidates_eval.append({
                    "candidate_idx": attempt,
                    "seed": seed,
                    "duration_sec": duration_sec,
                    "sample_rate": self.sample_rate,
                    "peak_db": -12.0 if amplitude > 0 else -999.0,
                    "rms_db": -15.0 if amplitude > 0 else -999.0,
                    "f0_median_hz": cand_f0,
                    "target_f0_hz": round(target_f0, 1),
                    "cosine_sim": cand_cos,
                    "passes_gate": p_gate,
                    "ranker_score": round(c_score, 4),
                    "ranker_breakdown": c_breakdown,
                })

            best = max(candidates_eval, key=lambda c: c["ranker_score"])
            if best["f0_median_hz"] > 0:
                accepted_f0s.append(best["f0_median_hz"])

            retries = 1 if ("TRIGGER_RETRY" in text or "[RETRY_THEN_PASS]" in text) else (len(candidates_eval) - 1)
            results.append({
                "id": item_id,
                "status": "success",
                "out_path": str(out_path),
                "duration_sec": best["duration_sec"],
                "sample_rate": best["sample_rate"],
                "seconds_taken": 0.05,
                "peak_db": best["peak_db"],
                "rms_db": best["rms_db"],
                "f0_median_hz": best["f0_median_hz"],
                "target_f0_hz": round(best["target_f0_hz"], 1),
                "cosine_sim": best["cosine_sim"],
                "passes_gate": best["passes_gate"],
                "retries": retries,
                "selected_candidate_idx": best["candidate_idx"],
                "seed_used": best["seed"],
                "ranker_score": best["ranker_score"],
                "ranker_breakdown": best["ranker_breakdown"],
                "candidates_evaluated": candidates_eval,
                "error": None,
            })

        return {
            "status": "success",
            "model_load_time_sec": 0.01,
            "device": "cpu",
            "dtype": "float32",
            "stage_breakdown_sec": {
                "model_load": 0.01,
                "prefill": 0.005,
                "ar_step": 0.02,
                "cfm": 0.015,
                "vae_decode": 0.005,
                "qc": 0.005,
                "write": 0.005,
            },
            "results": results,
        }


def get_tts_engine(engine_name: str, config: dict[str, Any] | None = None) -> TTSEngine:
    """Factory creating the appropriate TTS engine instance."""
    cfg = config or {}
    engine_lower = engine_name.lower().strip()
    if engine_lower == "fake":
        return FakeTTS()
    elif engine_lower in ("ming", "ming-3.0-tts", "ming-0.5b"):
        python_path = cfg.get("subprocess_python", ".venv-tts/Scripts/python.exe")
        timeout_sec = cfg.get("worker_timeout_sec", 300)
        daemon_url = cfg.get("daemon_url")
        return MingEngine(python_path=python_path, timeout_sec=timeout_sec, daemon_url=daemon_url)
    else:
        raise ValueError(f"Unsupported TTS engine: {engine_name}")


def read_wav_info(file_path: Path | str) -> dict[str, Any]:
    """Reads basic WAV metadata without third-party dependencies, with ffprobe fallback."""
    try:
        with wave.open(str(file_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_sec = n_frames / framerate if framerate > 0 else 0.0

        return {
            "channels": n_channels,
            "sample_width": sampwidth,
            "sample_rate": framerate,
            "frames": n_frames,
            "duration_sec": round(duration_sec, 3),
        }
    except Exception:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=channels,sample_rate,duration",
            "-of", "json",
            str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        stream = data["streams"][0]
        return {
            "channels": int(stream.get("channels", 1)),
            "sample_width": 2,
            "sample_rate": int(stream.get("sample_rate", 44100)),
            "frames": 0,
            "duration_sec": round(float(stream.get("duration", 0.0)), 3),
        }


def create_silence_wav(output_path: Path | str, duration_sec: float, sample_rate: int = 44100) -> None:
    """Generates a mono 16-bit silent WAV file."""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)


def _read_pcm16_mono_frames(wav_path: Path, expected_sample_rate: int | None = None) -> tuple[bytes, int]:
    """Reads mono 16-bit PCM frames from wav_path, converting via FFmpeg if format or rate differs."""
    needs_convert = False
    sr = expected_sample_rate or 44100
    try:
        with wave.open(str(wav_path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            if ch != 1 or sw != 2 or (expected_sample_rate is not None and sr != expected_sample_rate):
                needs_convert = True
            else:
                return wf.readframes(wf.getnframes()), sr
    except wave.Error:
        needs_convert = True

    if needs_convert:
        target_sr = expected_sample_rate or sr
        temp_pcm = wav_path.with_suffix(".temp_pcm16.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-ac", "1", "-ar", str(target_sr), "-c:a", "pcm_s16le",
                str(temp_pcm),
            ],
            capture_output=True,
            check=True,
        )
        try:
            with wave.open(str(temp_pcm), "rb") as wf:
                return wf.readframes(wf.getnframes()), wf.getframerate()
        finally:
            temp_pcm.unlink(missing_ok=True)
    return b"", sr


def get_scene_padding(
    scene_id: str,
    scene_idx: int = 0,
    total_scenes: int = 1,
    padding_policy: str = "broadcast",
    pad_before_fallback: float = 0.4,
    pad_after_fallback: float = 0.6,
) -> tuple[float, float]:
    """Returns (pad_before, pad_after) in seconds according to padding policy (Phase Q3).

    Policies:
      - 'broadcast':
          intro: 0.40s before, 0.15s after
          outro: 0.05s before, 0.60s after
          interior scenes: 0.05s before, 0.15s after (inter-scene gap = 0.20s)
      - 'fixed' / any other string:
          uses pad_before_fallback, pad_after_fallback
    """
    pol = str(padding_policy).strip().lower()
    if pol != "broadcast":
        return pad_before_fallback, pad_after_fallback

    s_lower = scene_id.lower()
    is_intro = s_lower == "intro" or scene_idx == 0
    is_outro = s_lower == "outro" or (total_scenes > 1 and scene_idx == total_scenes - 1)

    if is_intro:
        return 0.40, 0.15
    elif is_outro:
        return 0.05, 0.60
    else:
        return 0.05, 0.15


def _apply_micro_fade_pcm16(frames: bytes, sample_rate: int, fade_ms: float = 5.0) -> bytes:
    """Applies a short equal-power fade-in/out to 16-bit mono PCM bytes to eliminate join clicks."""
    fade_samples = int(sample_rate * max(0.0, fade_ms) / 1000.0)
    total_samples = len(frames) // 2
    if fade_samples <= 1 or total_samples < fade_samples * 2:
        return frames
    try:
        import numpy as np
        arr = np.frombuffer(frames, dtype=np.int16).copy()
        t = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        fade_in = np.sin(0.5 * np.pi * t)
        fade_out = np.cos(0.5 * np.pi * t)
        arr[:fade_samples] = np.clip(np.round(arr[:fade_samples] * fade_in), -32768, 32767).astype(np.int16)
        arr[-fade_samples:] = np.clip(np.round(arr[-fade_samples:] * fade_out), -32768, 32767).astype(np.int16)
        return arr.tobytes()
    except Exception:
        samples = list(struct.unpack(f"<{total_samples}h", frames))
        for i in range(fade_samples):
            t = i / fade_samples
            fade_in = math.sin(0.5 * math.pi * t)
            fade_out = math.cos(0.5 * math.pi * t)
            samples[i] = int(samples[i] * fade_in)
            samples[total_samples - 1 - i] = int(samples[total_samples - 1 - i] * fade_out)
        return struct.pack(f"<{total_samples}h", *samples)


def _level_pcm16_active_rms(
    frames: bytes,
    sample_rate: int,
    target_rms_dbfs: float = -23.0,
    max_peak_dbfs: float = -4.0,
) -> bytes:
    """Levels active speech RMS of 16-bit mono PCM bytes without changing duration or clipping."""
    if not frames:
        return frames

    total_samples = len(frames) // 2
    if total_samples == 0:
        return frames

    try:
        import numpy as np
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        win = max(1, int(sample_rate * 0.05))
        n_wins = total_samples // win
        if n_wins == 0:
            return frames

        win_blocks = arr[: n_wins * win].reshape(n_wins, win)
        mean_sq = np.mean(win_blocks ** 2, axis=1)

        silence_thresh_sq = ((10.0 ** (-45.0 / 20.0)) * 32768.0) ** 2
        active_mask = mean_sq >= silence_thresh_sq
        if not np.any(active_mask):
            return frames

        active_rms = float(np.sqrt(np.mean(mean_sq[active_mask])))
        if active_rms < 1.0:
            return frames

        curr_rms_dbfs = 20.0 * math.log10(active_rms / 32768.0)
        gain_db = max(-12.0, min(12.0, target_rms_dbfs - curr_rms_dbfs))

        max_sample = float(np.max(np.abs(arr)))
        if max_sample > 0:
            headroom_db = max_peak_dbfs - (20.0 * math.log10(max_sample / 32768.0))
            gain_db = min(gain_db, headroom_db)

        if abs(gain_db) < 0.25:
            return frames

        gain_lin = 10.0 ** (gain_db / 20.0)
        scaled_arr = np.clip(np.round(arr * gain_lin), -32768, 32767).astype(np.int16)
        return scaled_arr.tobytes()
    except Exception:
        samples = list(struct.unpack(f"<{total_samples}h", frames))
        win = max(1, int(sample_rate * 0.05))
        silence_thresh_sq = ((10.0 ** (-45.0 / 20.0)) * 32768.0) ** 2
        active_sq: list[float] = []
        for start in range(0, total_samples - win + 1, win):
            chunk = samples[start : start + win]
            mean_sq = sum(s * s for s in chunk) / win
            if mean_sq >= silence_thresh_sq:
                active_sq.append(mean_sq)
        if not active_sq:
            return frames

        active_rms = math.sqrt(sum(active_sq) / len(active_sq))
        if active_rms < 1.0:
            return frames
        curr_rms_dbfs = 20.0 * math.log10(active_rms / 32768.0)
        gain_db = max(-12.0, min(12.0, target_rms_dbfs - curr_rms_dbfs))

        max_sample = max(abs(s) for s in samples)
        if max_sample > 0:
            headroom_db = max_peak_dbfs - (20.0 * math.log10(max_sample / 32768.0))
            gain_db = min(gain_db, headroom_db)

        if abs(gain_db) < 0.25:
            return frames
        gain_lin = 10.0 ** (gain_db / 20.0)
        scaled = [max(-32768, min(32767, int(round(s * gain_lin)))) for s in samples]
        return struct.pack(f"<{total_samples}h", *scaled)


def assemble_scene_audio(
    sentence_wavs: list[Path],
    output_path: Path,
    pause_sec: float = 0.25,
    pad_before: float = 0.4,
    pad_after: float = 0.6,
    target_sample_rate: int = 44100,
    micro_fade_ms: float = 5.0,
    target_rms_dbfs: float | None = None,
) -> float:
    """Combines sentence WAVs with pause insertion, micro-fades, optional RMS leveling, and scene padding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames = bytearray()

    def add_silence(seconds: float) -> None:
        n = int(seconds * target_sample_rate)
        all_frames.extend(b"\x00\x00" * n)

    if pad_before > 0:
        add_silence(pad_before)

    for idx, s_wav in enumerate(sentence_wavs):
        frames, _ = _read_pcm16_mono_frames(s_wav, expected_sample_rate=target_sample_rate)
        if micro_fade_ms > 0:
            frames = _apply_micro_fade_pcm16(frames, target_sample_rate, fade_ms=micro_fade_ms)
        all_frames.extend(frames)

        if idx < len(sentence_wavs) - 1 and pause_sec > 0:
            add_silence(pause_sec)

    if pad_after > 0:
        add_silence(pad_after)

    final_frames = bytes(all_frames)
    if target_rms_dbfs is not None:
        final_frames = _level_pcm16_active_rms(final_frames, target_sample_rate, target_rms_dbfs=target_rms_dbfs)

    total_samples = len(final_frames) // 2
    duration_sec = total_samples / target_sample_rate if target_sample_rate > 0 else 0.0

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sample_rate)
        wf.writeframes(final_frames)

    return round(duration_sec, 3)


def concatenate_scenes(scene_wavs: list[Path], output_path: Path) -> float:
    """Concatenates multiple scene WAV files in exact order, ensuring consistent format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames = bytearray()
    sample_rate: int | None = None

    for s_wav in scene_wavs:
        frames, sr = _read_pcm16_mono_frames(s_wav, expected_sample_rate=sample_rate)
        if sample_rate is None:
            sample_rate = sr
        all_frames.extend(frames)

    effective_sr = sample_rate or 44100
    total_samples = len(all_frames) // 2
    duration_sec = total_samples / effective_sr if effective_sr > 0 else 0.0

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(effective_sr)
        wf.writeframes(all_frames)

    return round(duration_sec, 3)


def master_audio_loudnorm(
    input_wav: Path,
    output_wav: Path,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
    sample_rate: int = 48000,
    highpass_hz: float = 80.0,
    compressor: bool = True,
    limiter: bool = True,
    limiter_ceiling_dbfs: float = -1.5,
    loudness_strategy: str = "linear_strict",
) -> dict[str, Any]:
    """Applies high-pass filtering + compressor + two-pass EBU R128 loudness normalization and true-peak limiter via FFmpeg.

    Under loudness_strategy='linear_strict', raises RuntimeError if linear mode cannot be satisfied.
    Under loudness_strategy='fallback', falls back to -18 LUFS if target cannot be achieved cleanly in linear mode.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    pre_filters: list[str] = []
    if highpass_hz and highpass_hz > 0:
        pre_filters.append(f"highpass=f={highpass_hz}")
    if compressor:
        # Pre-conditioning: gentle broadcast compressor (2:1, 20ms attack, 250ms release, 2x makeup) + lookahead limiter
        pre_filters.append("acompressor=threshold=-24dB:ratio=2:attack=20:release=250:makeup=2")
        pre_filters.append("alimiter=limit=0.4:level=false")

    pre_filter_str = (",".join(pre_filters) + ",") if pre_filters else ""

    effective_tp = min(true_peak, limiter_ceiling_dbfs)
    limit_linear = round(10.0 ** (effective_tp / 20.0), 4)
    post_filter = f",alimiter=limit={limit_linear}:level=false" if limiter else ""

    def run_two_pass(target: float) -> tuple[dict[str, Any], str, float, float]:
        # Pass 1: Measure
        p1_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_wav),
            "-af", f"{pre_filter_str}loudnorm=I={target}:TP={effective_tp}:LRA=11:print_format=json",
            "-f", "null",
            "-",
        ]
        p1 = subprocess.run(p1_cmd, capture_output=True, text=True, check=True)
        # Parse JSON from stderr
        json_match_p1 = re.search(r"\{\s*\"input_i\".*?\}", p1.stderr, re.DOTALL)
        if not json_match_p1:
            raise RuntimeError(f"Could not parse loudnorm Pass 1 JSON from FFmpeg:\n{p1.stderr}")
        stats_p1 = json.loads(json_match_p1.group(0))

        input_i = float(stats_p1["input_i"])
        input_tp = float(stats_p1["input_tp"])
        input_lra = float(stats_p1["input_lra"])
        input_thresh = float(stats_p1["input_thresh"])
        target_offset = float(stats_p1["target_offset"])

        # Pass 2: Render with linear=true, output stats JSON, and apply master true-peak limiter
        p2_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_wav),
            "-af", (
                f"{pre_filter_str}loudnorm=I={target}:TP={effective_tp}:LRA=11:"
                f"measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
                f"measured_thresh={input_thresh}:offset={target_offset}:linear=true:print_format=json"
                f"{post_filter}"
            ),
            "-ar", str(sample_rate),
            "-ac", "1",
            str(output_wav),
        ]
        p2 = subprocess.run(p2_cmd, capture_output=True, text=True, check=True)

        json_match_p2 = re.search(r"\{\s*\"input_i\".*?\}", p2.stderr, re.DOTALL)
        stats_p2 = json.loads(json_match_p2.group(0)) if json_match_p2 else stats_p1
        norm_type = stats_p2.get("normalization_type", "linear")

        # Measure output
        meas_cmd = [
            "ffmpeg",
            "-i", str(output_wav),
            "-filter_complex", "ebur128=peak=true",
            "-f", "null",
            "-",
        ]
        meas = subprocess.run(meas_cmd, capture_output=True, text=True, check=True)
        summary_i = re.findall(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d\.]+)\s*LUFS", meas.stderr)
        summary_tp = re.findall(r"True peak:\s*\n\s*Peak:\s*(-?[\d\.]+)\s*dBFS", meas.stderr)
        if summary_i:
            final_lufs = float(summary_i[-1])
        else:
            all_i = re.findall(r"I:\s*(-?[\d\.]+)\s*LUFS", meas.stderr)
            final_lufs = float(all_i[-1]) if all_i else target

        if summary_tp:
            final_tp = float(summary_tp[-1])
        else:
            all_tp = re.findall(r"Peak:\s*(-?[\d\.]+)\s*dBFS", meas.stderr)
            final_tp = float(all_tp[-1]) if all_tp else effective_tp

        return stats_p2, norm_type, final_lufs, final_tp

    # Attempt primary target (-16 LUFS)
    stats, mode, final_lufs, final_tp = run_two_pass(target_lufs)
    selected_target = target_lufs
    fallback_used = False

    is_failed_linear = (mode == "dynamic" or final_tp > (effective_tp + 0.1))

    if is_failed_linear:
        if loudness_strategy == "linear_strict":
            raise RuntimeError(
                f"Loudnorm failed to achieve linear mode (mode='{mode}', target={target_lufs} LUFS, "
                f"measured={final_lufs} LUFS, true_peak={final_tp} dBFS, ceiling={effective_tp} dBFS). "
                f"Refusing fallback under loudness_strategy='linear_strict'."
            )
        else:
            # Fallback to -18 LUFS under fallback strategy
            stats_18, mode_18, final_lufs_18, final_tp_18 = run_two_pass(-18.0)
            selected_target = -18.0
            fallback_used = True
            stats, mode, final_lufs, final_tp = stats_18, mode_18, final_lufs_18, final_tp_18

    return {
        "target_lufs": selected_target,
        "fallback_used": fallback_used,
        "loudnorm_mode": mode,
        "before_lufs": float(stats["input_i"]),
        "before_true_peak": float(stats["input_tp"]),
        "after_lufs": final_lufs,
        "after_true_peak": final_tp,
        "output_sample_rate": sample_rate,
        "loudness_strategy": loudness_strategy,
        "limiter_ceiling_dbfs": effective_tp,
    }


def measure_wav_ebur128(wav_path: Path | str) -> dict[str, float]:
    """Measures integrated loudness (LUFS) and true peak (dBFS) using ffmpeg ebur128."""
    wav = Path(wav_path)
    if not wav.exists() or wav.stat().st_size == 0:
        return {"lufs": -70.0, "true_peak_db": -99.0}
    cmd = [
        "ffmpeg",
        "-nostats",
        "-i", str(wav),
        "-filter_complex", "ebur128=peak=true",
        "-f", "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary_i = re.findall(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d\.]+)\s*LUFS", proc.stderr)
        summary_tp = re.findall(r"True peak:\s*\n\s*Peak:\s*(-?[\d\.]+)\s*dBFS", proc.stderr)
        lufs = float(summary_i[-1]) if summary_i else -70.0
        tp = float(summary_tp[-1]) if summary_tp else -99.0
        return {"lufs": round(lufs, 2), "true_peak_db": round(tp, 2)}
    except Exception:
        return {"lufs": -70.0, "true_peak_db": -99.0}


def measure_wav_ebur128_segment(
    wav_path: Path | str,
    start_sec: float,
    duration_sec: float,
) -> dict[str, float]:
    """Measures integrated loudness (LUFS) and true peak (dBFS) of a segment in a WAV file."""
    wav = Path(wav_path)
    if not wav.exists() or wav.stat().st_size == 0 or duration_sec <= 0:
        return {"lufs": -70.0, "true_peak_db": -99.0}
    cmd = [
        "ffmpeg",
        "-nostats",
        "-ss", str(max(0.0, start_sec)),
        "-t", str(max(0.01, duration_sec)),
        "-i", str(wav),
        "-filter_complex", "ebur128=peak=true",
        "-f", "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        summary_i = re.findall(r"Integrated loudness:\s*\n\s*I:\s*(-?[\d\.]+)\s*LUFS", proc.stderr)
        summary_tp = re.findall(r"True peak:\s*\n\s*Peak:\s*(-?[\d\.]+)\s*dBFS", proc.stderr)
        lufs = float(summary_i[-1]) if summary_i else -70.0
        tp = float(summary_tp[-1]) if summary_tp else -99.0
        return {"lufs": round(lufs, 2), "true_peak_db": round(tp, 2)}
    except Exception:
        return {"lufs": -70.0, "true_peak_db": -99.0}


def measure_wav_artifacts(wav_path: Path | str) -> dict[str, Any]:
    """Measures audio artifacts: clipping margin, DC offset, intra-scene silence, and level jumps."""
    wav = Path(wav_path)
    if not wav.exists() or wav.stat().st_size == 0:
        return {
            "clipping_margin_db": 0.0,
            "max_intra_scene_silence_sec": 0.0,
            "dc_offset": 0.0,
            "level_jumps_detected": False,
            "has_clipping": True,
            "has_excess_silence": False,
        }

    try:
        with wave.open(str(wav), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
    except Exception:
        return {
            "clipping_margin_db": 0.0,
            "max_intra_scene_silence_sec": 0.0,
            "dc_offset": 0.0,
            "level_jumps_detected": False,
            "has_clipping": True,
            "has_excess_silence": False,
        }

    if sr == 0 or len(raw) == 0:
        return {
            "clipping_margin_db": 0.0,
            "max_intra_scene_silence_sec": 0.0,
            "dc_offset": 0.0,
            "level_jumps_detected": False,
            "has_clipping": True,
            "has_excess_silence": False,
        }

    import numpy as np

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.zeros(1, dtype=np.float32)

    dc_offset = float(np.mean(samples))
    max_amp = float(np.max(np.abs(samples)))
    peak_db = 20.0 * np.log10(max_amp) if max_amp > 1e-9 else -99.0
    clipping_margin_db = round(max(0.0, -peak_db), 2)
    has_clipping = bool(peak_db >= -0.05)

    # Level jump check across 50ms windows
    win_len = int(sr * 0.05)
    level_jumps = False
    if len(samples) >= win_len:
        num_wins = len(samples) // win_len
        win_samples = samples[: num_wins * win_len].reshape(num_wins, win_len)
        win_rms = np.sqrt(np.mean(win_samples ** 2, axis=1))
        active = win_rms > 1e-3
        for i in range(len(active) - 1):
            if active[i] and active[i + 1]:
                diff_db = abs(20.0 * np.log10(win_rms[i + 1] / win_rms[i]))
                if diff_db > 25.0:
                    level_jumps = True
                    break

    # Silence gap check (20ms frames, -45 dBFS threshold)
    frame_ms = 20
    frame_len = int(sr * (frame_ms / 1000.0))
    threshold_amp = 10.0 ** (-45.0 / 20.0)
    max_silence_frames = 0
    curr_silence_frames = 0

    if len(samples) >= frame_len:
        n_sub = len(samples) // frame_len
        sub = samples[: n_sub * frame_len].reshape(n_sub, frame_len)
        sub_rms = np.sqrt(np.mean(sub ** 2, axis=1))
        for rms_val in sub_rms:
            if rms_val < threshold_amp:
                curr_silence_frames += 1
                if curr_silence_frames > max_silence_frames:
                    max_silence_frames = curr_silence_frames
            else:
                curr_silence_frames = 0

    max_silence_sec = round(max_silence_frames * (frame_ms / 1000.0), 2)
    has_excess_silence = max_silence_sec > 1.5

    return {
        "clipping_margin_db": clipping_margin_db,
        "max_intra_scene_silence_sec": max_silence_sec,
        "dc_offset": round(dc_offset, 5),
        "level_jumps_detected": level_jumps,
        "has_clipping": has_clipping,
        "has_excess_silence": has_excess_silence,
    }


def compute_voice_quality_index(
    consistency_metrics: dict[str, Any],
    reference_f0_hz: float,
    scenes_info: list[dict[str, Any]],
    total_duration_sec: float,
    pad_per_scene_sec: float = 1.0,
    artifact_flags: dict[str, Any] | None = None,
    unverified_pronunciations: list[Any] | None = None,
) -> dict[str, Any]:
    """Computes the 7-metric Voice Quality Index (VQI)."""
    import numpy as np

    # 1. Cosine similarity
    cosines = [float(m.get("cosine_sim", 1.0)) for m in consistency_metrics.values()]
    if cosines:
        cos_min = round(float(np.min(cosines)), 4)
        cos_max = round(float(np.max(cosines)), 4)
        cos_mean = round(float(np.mean(cosines)), 4)
        cos_std = round(float(np.std(cosines)), 4)
    else:
        cos_min = cos_max = cos_mean = 1.0
        cos_std = 0.0

    # 2. F0
    f0s = [float(m.get("f0_median_hz", 0.0)) for m in consistency_metrics.values() if float(m.get("f0_median_hz", 0.0)) > 0]
    if f0s:
        f0_min = round(float(np.min(f0s)), 1)
        f0_max = round(float(np.max(f0s)), 1)
        f0_spread_pct = round(((f0_max - f0_min) / f0_min) * 100.0, 1) if f0_min > 0 else 0.0
    else:
        f0_min = f0_max = round(reference_f0_hz, 1) if reference_f0_hz > 0 else 220.0
        f0_spread_pct = 0.0

    # 3. Scene LUFS spread
    scene_lufs = []
    for s in scenes_info:
        if "mastered_lufs" in s and s["mastered_lufs"] is not None and s["mastered_lufs"] > -60.0:
            scene_lufs.append(float(s["mastered_lufs"]))
        elif "lufs" in s and s["lufs"] is not None and s["lufs"] > -60.0:
            scene_lufs.append(float(s["lufs"]))

    if scene_lufs:
        lufs_min = round(float(np.min(scene_lufs)), 2)
        lufs_max = round(float(np.max(scene_lufs)), 2)
        lufs_spread = round(lufs_max - lufs_min, 2)
    else:
        lufs_min = lufs_max = -18.0
        lufs_spread = 0.0

    # 4. Words per second (WPS)
    wps_vals = []
    for s in scenes_info:
        dur = float(s.get("actual_duration_sec", 0.0))
        wc = float(s.get("word_count", 0))
        if dur > 0 and wc > 0:
            wps_vals.append(wc / dur)

    if wps_vals:
        wps_mean = round(float(np.mean(wps_vals)), 2)
        wps_var = round(float(np.var(wps_vals)), 4)
        wps_std = round(float(np.std(wps_vals)), 4)
    else:
        wps_mean = 2.0
        wps_var = 0.0
        wps_std = 0.0

    # 5. Dead air ratio
    n_scenes = len(scenes_info)
    dead_air_sec = round(n_scenes * pad_per_scene_sec, 2)
    dead_air_ratio = round(dead_air_sec / total_duration_sec, 4) if total_duration_sec > 0 else 0.0

    # 6. Artifact flags
    artifacts = artifact_flags or {
        "clipping_margin_db": 1.5,
        "max_intra_scene_silence_sec": 0.0,
        "dc_offset": 0.0,
        "level_jumps_detected": False,
        "has_clipping": False,
        "has_excess_silence": False,
    }

    # 7. Unverified pronunciations
    unverified = unverified_pronunciations or []

    return {
        "cosine_min": cos_min,
        "cosine_max": cos_max,
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "f0_spread_pct": f0_spread_pct,
        "reference_f0_hz": round(reference_f0_hz, 1),
        "scene_lufs_min": lufs_min,
        "scene_lufs_max": lufs_max,
        "scene_lufs_spread": lufs_spread,
        "wps_mean": wps_mean,
        "wps_variance": wps_var,
        "wps_std": wps_std,
        "dead_air_duration_sec": dead_air_sec,
        "dead_air_ratio": dead_air_ratio,
        "artifact_flags": artifacts,
        "unverified_pronunciations_applied": unverified,
    }

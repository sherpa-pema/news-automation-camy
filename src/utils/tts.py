"""TTS Engine abstraction and audio processing utilities.

Supports swappable TTS engines:
- MingEngine: Invokes tools/ming_worker.py inside isolated .venv-tts via subprocess.
- FakeTTS: Pure Python test double producing valid WAV audio without dependencies.
"""
from __future__ import annotations

import json
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

    def _try_daemon(self, job_payload: dict[str, Any], daemon_url: str) -> dict[str, Any] | None:
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
            "f0_tolerance_pct": settings.get("f0_tolerance_pct", 0.25),
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
        results = []
        for item in items:
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

            cosine_sim = 0.85
            f0_med = 216.0
            passes_gate = True
            retries = 0

            if "TRIGGER_LOW_SIM" in text or "[LOW_SIM]" in text:
                cosine_sim = 0.50
                passes_gate = False
                retries = 2
            elif "TRIGGER_HIGH_PITCH" in text or "[HIGH_PITCH]" in text:
                f0_med = 350.0
                passes_gate = False
                retries = 2
            elif "TRIGGER_RETRY" in text or "[RETRY_THEN_PASS]" in text:
                retries = 1

            results.append({
                "id": item_id,
                "status": "success",
                "out_path": str(out_path),
                "duration_sec": duration_sec,
                "sample_rate": self.sample_rate,
                "seconds_taken": 0.05,
                "peak_db": -12.0 if amplitude > 0 else -999.0,
                "rms_db": -15.0 if amplitude > 0 else -999.0,
                "f0_median_hz": f0_med,
                "cosine_sim": cosine_sim,
                "passes_gate": passes_gate,
                "retries": retries,
                "error": None,
            })

        return {
            "status": "success",
            "model_load_time_sec": 0.01,
            "device": "cpu",
            "dtype": "float32",
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


def _apply_micro_fade_pcm16(frames: bytes, sample_rate: int, fade_ms: float = 5.0) -> bytes:
    """Applies a short linear fade-in/out to 16-bit mono PCM bytes to eliminate join clicks."""
    fade_samples = int(sample_rate * max(0.0, fade_ms) / 1000.0)
    total_samples = len(frames) // 2
    if fade_samples <= 1 or total_samples < fade_samples * 2:
        return frames
    samples = list(struct.unpack(f"<{total_samples}h", frames))
    for i in range(fade_samples):
        gain = i / fade_samples
        samples[i] = int(samples[i] * gain)
        samples[total_samples - 1 - i] = int(samples[total_samples - 1 - i] * gain)
    return struct.pack(f"<{total_samples}h", *samples)


def _level_pcm16_active_rms(
    frames: bytes,
    sample_rate: int,
    target_rms_dbfs: float = -20.0,
    max_peak_dbfs: float = -2.0,
) -> bytes:
    """Levels active speech RMS of 16-bit mono PCM bytes without changing duration or clipping."""
    total_samples = len(frames) // 2
    if total_samples == 0:
        return frames
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
) -> dict[str, Any]:
    """Applies high-pass filtering + two-pass EBU R128 loudness normalization and resampling via FFmpeg.

    Falls back to -18 LUFS if target cannot be achieved cleanly in linear mode.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    pre_filter = f"highpass=f={highpass_hz}," if highpass_hz and highpass_hz > 0 else ""

    def run_two_pass(target: float) -> tuple[dict[str, Any], str, float, float]:
        # Pass 1: Measure
        p1_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_wav),
            "-af", f"{pre_filter}loudnorm=I={target}:TP={true_peak}:LRA=11:print_format=json",
            "-f", "null",
            "-",
        ]
        p1 = subprocess.run(p1_cmd, capture_output=True, text=True, check=True)
        # Parse JSON from stderr
        json_match = re.search(r"\{\s*\"input_i\".*?\}", p1.stderr, re.DOTALL)
        if not json_match:
            raise RuntimeError(f"Could not parse loudnorm Pass 1 JSON from FFmpeg:\n{p1.stderr}")
        stats = json.loads(json_match.group(0))

        # Check normalization type: loudnorm outputs normalization_type in stats
        norm_type = stats.get("normalization_type", "linear")
        input_i = float(stats["input_i"])
        input_tp = float(stats["input_tp"])
        input_lra = float(stats["input_lra"])
        input_thresh = float(stats["input_thresh"])
        target_offset = float(stats["target_offset"])

        # Pass 2: Render with linear=true and resample to 48kHz
        p2_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_wav),
            "-af", (
                f"{pre_filter}loudnorm=I={target}:TP={true_peak}:LRA=11:"
                f"measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
                f"measured_thresh={input_thresh}:offset={target_offset}:linear=true"
            ),
            "-ar", str(sample_rate),
            "-ac", "1",
            str(output_wav),
        ]
        subprocess.run(p2_cmd, capture_output=True, text=True, check=True)

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
            final_tp = float(all_tp[-1]) if all_tp else true_peak

        return stats, norm_type, final_lufs, final_tp

    # Attempt primary target (-16 LUFS)
    stats, mode, final_lufs, final_tp = run_two_pass(target_lufs)
    selected_target = target_lufs
    fallback_used = False

    # If mode fell back to dynamic or clipping occurred, attempt -18 LUFS fallback
    if mode == "dynamic" or final_tp > true_peak + 0.1:
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
    }

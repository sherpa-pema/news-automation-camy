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
    """Ming-3.0-TTS engine running via subprocess in .venv-tts."""

    def __init__(
        self,
        python_path: str | Path = ".venv-tts/Scripts/python.exe",
        worker_script: str | Path = "tools/ming_worker.py",
        timeout_sec: int = 300,
    ) -> None:
        self.python_path = Path(python_path)
        self.worker_script = Path(worker_script)
        self.timeout_sec = timeout_sec

    def synthesize_batch(
        self,
        items: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        if not items:
            return {"status": "success", "model_load_time_sec": 0.0, "results": []}

        if not self.python_path.exists():
            raise FileNotFoundError(
                f"Subprocess Python interpreter not found at: {self.python_path}. "
                "Ensure .venv-tts is set up."
            )
        if not self.worker_script.exists():
            raise FileNotFoundError(
                f"Worker script not found at: {self.worker_script}"
            )

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
            "text_normalization": settings.get("text_normalization", False),
            "reference_wav": settings.get("reference_wav", "config/voice/reference.wav"),
            "reference_txt": settings.get("reference_txt", "config/voice/reference.txt"),
            "similarity_threshold": settings.get("similarity_threshold", 0.72),
            "f0_tolerance_pct": settings.get("f0_tolerance_pct", 0.25),
            "max_retries": settings.get("max_retries", 2),
            "items": items,
        }

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
        return MingEngine(python_path=python_path, timeout_sec=timeout_sec)
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


def assemble_scene_audio(
    sentence_wavs: list[Path],
    output_path: Path,
    pause_sec: float = 0.25,
    pad_before: float = 0.4,
    pad_after: float = 0.6,
    target_sample_rate: int = 44100,
) -> float:
    """Combines sentence WAVs with pause insertion and scene padding."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames = bytearray()

    def add_silence(seconds: float) -> None:
        n = int(seconds * target_sample_rate)
        all_frames.extend(b"\x00\x00" * n)

    if pad_before > 0:
        add_silence(pad_before)

    for idx, s_wav in enumerate(sentence_wavs):
        try:
            with wave.open(str(s_wav), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
        except wave.Error:
            temp_pcm = s_wav.with_suffix(".temp_pcm16.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(s_wav), "-c:a", "pcm_s16le", str(temp_pcm)],
                capture_output=True,
                check=True,
            )
            with wave.open(str(temp_pcm), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
            temp_pcm.unlink(missing_ok=True)

        all_frames.extend(frames)

        if idx < len(sentence_wavs) - 1 and pause_sec > 0:
            add_silence(pause_sec)

    if pad_after > 0:
        add_silence(pad_after)

    total_samples = len(all_frames) // 2
    duration_sec = total_samples / target_sample_rate if target_sample_rate > 0 else 0.0

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sample_rate)
        wf.writeframes(all_frames)

    return round(duration_sec, 3)


def concatenate_scenes(scene_wavs: list[Path], output_path: Path) -> float:
    """Concatenates multiple scene WAV files in exact order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_frames = bytearray()
    sample_rate = 44100

    for s_wav in scene_wavs:
        try:
            with wave.open(str(s_wav), "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        except wave.Error:
            temp_pcm = s_wav.with_suffix(".temp_pcm16_c.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(s_wav), "-c:a", "pcm_s16le", str(temp_pcm)],
                capture_output=True,
                check=True,
            )
            with wave.open(str(temp_pcm), "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            temp_pcm.unlink(missing_ok=True)
        all_frames.extend(frames)

    total_samples = len(all_frames) // 2
    duration_sec = total_samples / sample_rate if sample_rate > 0 else 0.0

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(all_frames)

    return round(duration_sec, 3)


def master_audio_loudnorm(
    input_wav: Path,
    output_wav: Path,
    target_lufs: float = -16.0,
    true_peak: float = -1.5,
    sample_rate: int = 48000,
) -> dict[str, Any]:
    """Applies two-pass EBU R128 loudness normalization and resampling using FFmpeg.

    Falls back to -18 LUFS if target cannot be achieved cleanly in linear mode.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    def run_two_pass(target: float) -> tuple[dict[str, Any], str, float, float]:
        # Pass 1: Measure
        p1_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_wav),
            "-af", f"loudnorm=I={target}:TP={true_peak}:LRA=11:print_format=json",
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
                f"loudnorm=I={target}:TP={true_peak}:LRA=11:"
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

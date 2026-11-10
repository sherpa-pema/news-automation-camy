"""Voice Agent implementation.

Synthesizes natural broadcast speech from script.json using an open-source TTS
engine (Ming-3.0-tts-0.5b via isolated worker, or FakeTTS for tests).
Masters the master narration to broadcast loudness (-16 LUFS / -1.5 dB True Peak).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.agents.base import BaseAgent
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest, update_stage_status
from src.utils.tts import (
    FakeTTS as FakeTTS,
    MingEngine as MingEngine,
    assemble_scene_audio,
    concatenate_scenes,
    get_tts_engine,
    master_audio_loudnorm,
    read_wav_info,
)


def load_pronunciations(pronunciations_file: Path | str = "config/pronunciations.yaml") -> dict[str, str]:
    """Loads the phonetic respelling dictionary from config/pronunciations.yaml."""
    path = Path(pronunciations_file)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    if "pronunciations" in data and isinstance(data["pronunciations"], dict):
        data = data["pronunciations"]
    res = {}
    for word, val in data.items():
        if isinstance(val, dict):
            res[word] = val.get("respelling", "")
        else:
            res[word] = str(val)
    return res


def load_pronunciations_with_metadata(pronunciations_file: Path | str = "config/pronunciations.yaml") -> dict[str, dict[str, Any]]:
    """Loads phonetic respellings with verification metadata from config/pronunciations.yaml."""
    path = Path(pronunciations_file)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    if "pronunciations" in data and isinstance(data["pronunciations"], dict):
        data = data["pronunciations"]
    res = {}
    for word, val in data.items():
        if isinstance(val, dict):
            res[word] = {
                "respelling": val.get("respelling", ""),
                "verified": bool(val.get("verified", False)),
            }
        else:
            res[word] = {
                "respelling": str(val),
                "verified": False,
            }
    return res


def get_unverified_pronunciations_applied(
    scenes: list[dict[str, Any]],
    pronunciations_file: Path | str = "config/pronunciations.yaml",
) -> list[dict[str, Any]]:
    """Identifies which unverified pronunciation respellings were matched and applied across scenes."""
    meta = load_pronunciations_with_metadata(pronunciations_file)
    unverified_applied = []
    for word, info in meta.items():
        if not info.get("verified", False):
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            for sc in scenes:
                raw_text = sc.get("narration", "")
                if pattern.search(raw_text):
                    unverified_applied.append({
                        "word": word,
                        "respelling": info["respelling"],
                        "scene_id": sc.get("id"),
                    })
                    break
    return unverified_applied


def clean_narration_text(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Cleans script text for broadcast narration.

    1. Strips any bracketed paralinguistic tags like [Sigh] or [Laughter].
    2. Normalizes smart quotes and whitespace.
    3. Applies whole-word, case-insensitive phonetic replacements.
    """
    if not text:
        return ""

    # Strip bracketed tags
    cleaned = re.sub(r"\[.*?\]", "", text)

    # Normalize unicode quotes and dashes
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", " - ").replace("–", " - ")

    # Normalize spaces around punctuation
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Apply pronunciation dictionary (case-insensitive whole-word replacement)
    if pronunciations:
        for word, respelling in pronunciations.items():
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            cleaned = pattern.sub(respelling, cleaned)

    return cleaned


def split_into_sentences(text: str) -> list[str]:
    """Splits text into clean sentences for prosody-controlled sentence-level synthesis."""
    if not text:
        return []

    # Protect common abbreviations before splitting
    protected = text
    abbrevs = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "e.g.", "i.e.", "vs.", "No."]
    for idx, abb in enumerate(abbrevs):
        protected = protected.replace(abb, f"__ABB_{idx}__")

    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        # Restore abbreviations
        for idx, abb in enumerate(abbrevs):
            s_clean = s_clean.replace(f"__ABB_{idx}__", abb)
        sentences.append(s_clean)

    return sentences


def check_audio_silence_gaps(wav_path: Path | str, threshold_db: float = -45.0) -> float:
    """Measures the maximum internal silence gap in seconds inside a WAV file without third-party deps."""
    import struct
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sr == 0 or len(raw) == 0:
        return 0.0

    threshold_amp = (10.0 ** (threshold_db / 20.0)) * 32768.0
    threshold_sq = threshold_amp * threshold_amp

    win_size = int(sr * 0.05)  # 50ms window
    if win_size == 0:
        return 0.0

    bytes_per_sample = 2
    win_bytes = win_size * bytes_per_sample
    total_wins = len(raw) // win_bytes

    max_streak = 0
    curr_streak = 0

    for i in range(total_wins):
        chunk = raw[i * win_bytes : (i + 1) * win_bytes]
        samples = struct.unpack(f"<{win_size}h", chunk)
        sum_sq = sum(s * s for s in samples)
        mean_sq = sum_sq / win_size
        if mean_sq < threshold_sq:
            curr_streak += 1
        else:
            if curr_streak > max_streak:
                max_streak = curr_streak
            curr_streak = 0

    if curr_streak > max_streak:
        max_streak = curr_streak

    return round(max_streak * 0.05, 3)


class VoiceAgent(BaseAgent):
    """Voice Agent synthesizes broadcast speech from script.json narration."""

    def __init__(self, run_dir: Path | str, config: dict[str, Any] | None = None) -> None:
        super().__init__(Path(run_dir), config=config)
        self.logger = setup_logger("VoiceAgent", log_file=self.run_dir / "run.log")

    @property
    def name(self) -> str:
        return "voice"

    @property
    def schema_contract_path(self) -> str | None:
        return None  # Audio contract verified by signal and file validation checks

    def _get_voice_stage_config(self) -> dict[str, Any]:
        """Extracts the voice stage configuration block from the pipeline config."""
        stages = self.config.get("stages", [])
        voice_cfg = next((s for s in stages if s.get("id") == "voice"), {})
        return voice_cfg

    def run(
        self,
        manifest: dict[str, Any],
        only_scene: str | None = None,
        engine_override: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Synthesizes broadcast speech from script.json narration."""
        self.logger.info(f"Starting Voice Agent run in {self.run_dir}...")
        voice_cfg = self._get_voice_stage_config()

        # Engine selection
        engine_name = engine_override or voice_cfg.get("engine", "ming")
        if dry_run:
            engine_name = "fake"

        # Parameters
        voice = voice_cfg.get("voice", "NP_lemi")
        task = voice_cfg.get("task", "TTS")
        model_path = voice_cfg.get("model_path", "models/Ming-3.0-tts-0.5b")
        device = voice_cfg.get("device", "cuda:0")
        dtype = voice_cfg.get("dtype", "bfloat16")
        language_id = voice_cfg.get("language_id", "en")
        instruction_schema = voice_cfg.get("instruction_schema", "en")
        speed = voice_cfg.get("speed")
        text_normalization = voice_cfg.get("text_normalization", False)
        chunk_mode = voice_cfg.get("chunk_mode", "scene").lower().strip()
        reference_wav = voice_cfg.get("reference_wav", "config/voice/reference.wav")
        reference_txt = voice_cfg.get("reference_txt", "config/voice/reference.txt")
        similarity_threshold = float(voice_cfg.get("similarity_threshold", 0.72))
        f0_tolerance_pct = float(voice_cfg.get("f0_tolerance_pct", 0.25))
        max_retries = int(voice_cfg.get("max_retries", 2))
        pause_between_sentences = float(voice_cfg.get("pause_between_sentences", 0.25))
        pad_before = float(voice_cfg.get("pad_before", 0.4))
        pad_after = float(voice_cfg.get("pad_after", 0.6))
        target_lufs = float(voice_cfg.get("target_lufs", -16.0))
        output_sample_rate = int(voice_cfg.get("output_sample_rate", 48000))
        disclose_synthetic_voice = bool(voice_cfg.get("disclose_synthetic_voice", True))

        # Check for script/script.json input
        script_path = self.run_dir / "script" / "script.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing required script file: {script_path}")

        with open(script_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        # Verify approval if origin is editor
        origin = script_data.get("origin", "editor")
        if origin != "fixture":
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            import hashlib
            curr_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
            if app_data.get("script_sha256") != curr_hash:
                raise ValueError("script changed after approval, re-run validate and approve")
            from src.utils.editor_validator import compute_validator_fingerprint, extract_script_validation_fingerprint
            curr_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
            app_fp = extract_script_validation_fingerprint(app_data.get("validator_fingerprint"))
            if app_fp and app_fp != curr_fp:
                raise ValueError("validator or editor config changed after approval, re-run validate and approve")
            rep_file = self.run_dir / "script" / "validation_report.json"
            if rep_file.exists():
                with open(rep_file, "r", encoding="utf-8") as rf:
                    rep_data = json.load(rf)
                rep_fp = extract_script_validation_fingerprint(rep_data.get("validator_fingerprint"))
                if rep_fp and rep_fp != curr_fp:
                    raise ValueError("validator or editor config changed since validation, re-run validate and approve")

        scenes = script_data.get("scenes", [])
        if not scenes:
            raise ValueError("script.json contains no scenes to synthesize")

        pronunciations = load_pronunciations()
        tts_engine = get_tts_engine(engine_name, voice_cfg)

        audio_dir = self.run_dir / "audio"
        scenes_dir = audio_dir / "scenes"
        temp_sentences_dir = audio_dir / "temp_sentences"
        audio_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        temp_sentences_dir.mkdir(parents=True, exist_ok=True)

        target_scenes = scenes
        if only_scene:
            target_scenes = [s for s in scenes if s["id"] == only_scene]
            if not target_scenes:
                raise ValueError(f"Specified scene '{only_scene}' not found in script.json")
            self.logger.info(f"Partial synthesis requested for single scene: {only_scene}")

        total_retries = 0
        all_scene_durations: list[dict[str, Any]] = []
        scene_consistency_map: dict[str, Any] = {}

        all_items = []
        scene_items_map: dict[str, list[Path]] = {}

        for scene in target_scenes:
            scene_id = scene["id"]
            # READ ONLY narration text
            raw_narration = scene.get("narration", "")
            cleaned_text = clean_narration_text(raw_narration, pronunciations)

            if chunk_mode == "scene":
                s_id = scene_id
                s_path = temp_sentences_dir / f"{s_id}_raw.wav"
                all_items.append({"id": s_id, "text": cleaned_text, "out_path": str(s_path)})
                scene_items_map[scene_id] = [s_path]
            else:
                sentences = split_into_sentences(cleaned_text)
                scene_items_map[scene_id] = []
                for idx, sent in enumerate(sentences):
                    s_id = f"{scene_id}_s{idx}"
                    s_path = temp_sentences_dir / f"{s_id}.wav"
                    all_items.append({"id": s_id, "text": sent, "out_path": str(s_path)})
                    scene_items_map[scene_id].append(s_path)

        # Determine which items actually need synthesis
        items_to_synthesize = []
        for item in all_items:
            s_path = Path(item["out_path"])
            scene_id = item["id"].split("_s")[0]
            force_regen = (only_scene is not None and scene_id == only_scene)

            is_valid = False
            if s_path.exists() and s_path.stat().st_size > 1000 and not force_regen:
                try:
                    info = read_wav_info(s_path)
                    if info["duration_sec"] > 0.1:
                        is_valid = True
                except Exception:
                    is_valid = False

            if not is_valid:
                if s_path.exists():
                    s_path.unlink()
                items_to_synthesize.append(item)
            else:
                self.logger.info(f"Using existing cached audio: {item['id']}")

        engine_settings = {
            "model_path": model_path,
            "task": task,
            "voice": voice,
            "language_id": language_id,
            "instruction_schema": instruction_schema,
            "device": device,
            "dtype": dtype,
            "speed": speed,
            "text_normalization": text_normalization,
            "reference_wav": reference_wav,
            "reference_txt": reference_txt,
            "similarity_threshold": similarity_threshold,
            "f0_tolerance_pct": f0_tolerance_pct,
            "max_retries": max_retries,
        }

        total_synthesis_time = 0.0
        total_model_load_time = 0.0

        if items_to_synthesize:
            self.logger.info(
                f"Synthesizing {len(items_to_synthesize)} {chunk_mode}(s) "
                f"({len(all_items) - len(items_to_synthesize)} already cached) "
                f"across {len(target_scenes)} scene(s)..."
            )
            t0 = time.time()
            batch_result = tts_engine.synthesize_batch(items_to_synthesize, engine_settings)
            total_synthesis_time = time.time() - t0
            total_model_load_time = batch_result.get("model_load_time_sec", 0.0)

            # Inspect per-item result for failures and retries
            for res_item in batch_result.get("results", []):
                total_retries += res_item.get("retries", 0)
                r_id = res_item.get("id", "")
                scene_consistency_map[r_id] = {
                    "cosine_sim": res_item.get("cosine_sim", 1.0),
                    "f0_median_hz": res_item.get("f0_median_hz", 0.0),
                    "retries": res_item.get("retries", 0),
                    "passes_gate": res_item.get("passes_gate", True),
                }
                if res_item.get("status") == "failed":
                    err_msg = res_item.get("error", "Unknown error")
                    raise RuntimeError(f"Failed to synthesize {chunk_mode} {r_id}: {err_msg}")
        else:
            self.logger.info(
                f"All {len(all_items)} {chunk_mode}(s) already synthesized and valid on disk."
            )

        # Assemble each scene with padding and pauses
        pause_to_use = 0.0 if chunk_mode == "scene" else pause_between_sentences
        for scene in target_scenes:
            scene_id = scene["id"]
            wav_paths = scene_items_map[scene_id]
            scene_wav_path = scenes_dir / f"{scene_id}.wav"
            scene_dur = assemble_scene_audio(
                wav_paths,
                scene_wav_path,
                pause_sec=pause_to_use,
                pad_before=pad_before,
                pad_after=pad_after,
                target_sample_rate=44100 if engine_name == "ming" else 44100,
            )

            # Also create aliases audio/scene_<id>.wav and audio/<id>.wav for convenience
            alias_path = audio_dir / f"scene_{scene_id}.wav"
            shutil.copyfile(scene_wav_path, alias_path)
            if scene_id.startswith("scene_"):
                shutil.copyfile(scene_wav_path, audio_dir / f"{scene_id}.wav")
            self.logger.info(f"Scene '{scene_id}' assembled: duration = {scene_dur:.2f}s")
            self.logger.info(f"Scene '{scene_id}' assembled: duration = {scene_dur:.2f}s")

        # Clean up temporary sentence wavs
        shutil.rmtree(temp_sentences_dir, ignore_errors=True)

        # Assemble full list of all scenes in script order
        scene_files_in_order = []
        scene_metrics = []
        current_time = 0.0

        for scene in scenes:
            s_id = scene["id"]
            s_file = scenes_dir / f"{s_id}.wav"
            if not s_file.exists():
                raise FileNotFoundError(f"Missing assembled scene audio: {s_file}")

            info = read_wav_info(s_file)
            dur = info["duration_sec"]
            start_t = current_time
            end_t = round(start_t + dur, 3)
            current_time = end_t

            scene_files_in_order.append(s_file)
            all_scene_durations.append({
                "id": s_id,
                "file": f"audio/scenes/{s_id}.wav",
                "start_sec": start_t,
                "end_sec": end_t,
                "duration_sec": dur,
            })
            scene_metrics.append({
                "id": s_id,
                "estimated_duration_sec": scene.get("estimated_duration_sec", 0.0),
                "actual_duration_sec": dur,
                "file": f"audio/scenes/{s_id}.wav",
            })

        # Save scene_durations.json
        durations_json_path = audio_dir / "scene_durations.json"
        with open(durations_json_path, "w", encoding="utf-8") as f:
            json.dump(all_scene_durations, f, indent=2)

        # Concatenate scenes into intermediate raw track
        raw_concat_path = audio_dir / "raw_concat.wav"
        raw_total_dur = concatenate_scenes(scene_files_in_order, raw_concat_path)

        # Loudness normalization and mastering (two-pass loudnorm to 48kHz)
        master_narration_path = audio_dir / "narration.wav"
        self.logger.info(f"Mastering narration with FFmpeg two-pass loudnorm (target={target_lufs} LUFS)...")
        master_stats = master_audio_loudnorm(
            raw_concat_path,
            master_narration_path,
            target_lufs=target_lufs,
            true_peak=-1.5,
            sample_rate=output_sample_rate,
        )
        # Clean up intermediate raw concat
        raw_concat_path.unlink(missing_ok=True)

        final_master_info = read_wav_info(master_narration_path)
        final_duration_sec = final_master_info["duration_sec"]
        rtf = round(total_synthesis_time / final_duration_sec, 2) if final_duration_sec > 0 else 0.0

        # Build voice_report.json
        unverified_applied = get_unverified_pronunciations_applied(scenes)
        voice_report = {
            "engine": engine_name,
            "model": model_path,
            "voice": voice,
            "task": task,
            "chunk_mode": chunk_mode,
            "device": device,
            "dtype": dtype,
            "disclose_synthetic_voice": disclose_synthetic_voice,
            "model_load_time_sec": round(total_model_load_time, 2),
            "synthesis_wall_time_sec": round(total_synthesis_time, 2),
            "total_audio_duration_sec": final_duration_sec,
            "real_time_factor": rtf,
            "retries": total_retries,
            "speaker_consistency": {
                "reference_wav": str(reference_wav),
                "similarity_threshold": similarity_threshold,
                "f0_tolerance_pct": f0_tolerance_pct,
                "metrics": scene_consistency_map,
            },
            "loudnorm": master_stats,
            "scenes": scene_metrics,
            "unverified_pronunciations_applied": unverified_applied,
        }
        report_json_path = audio_dir / "voice_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(voice_report, f, indent=2)

        output_files = [
            "audio/narration.wav",
            "audio/scene_durations.json",
            "audio/voice_report.json",
        ]
        for s in scenes:
            output_files.append(f"audio/scenes/{s['id']}.wav")

        # Update manifest metrics
        metrics = {
            "engine": engine_name,
            "model": model_path,
            "voice": voice,
            "total_duration_sec": final_duration_sec,
            "real_time_factor": rtf,
            "model_load_time_sec": round(total_model_load_time, 2),
            "retries": total_retries,
            "disclose_synthetic_voice": disclose_synthetic_voice,
            "loudnorm_mode": master_stats.get("loudnorm_mode"),
            "loudness_after_lufs": master_stats.get("after_lufs"),
            "true_peak_after_db": master_stats.get("after_true_peak"),
            "warnings": [],
        }
        if master_stats.get("fallback_used"):
            metrics["warnings"].append("Loudnorm fell back to -18 LUFS to avoid dynamic compression")

        update_stage_status(
            manifest,
            "voice",
            "done",
            output_files=output_files,
            metrics=metrics,
        )

        self.logger.info(
            f"Voice Agent completed: {final_duration_sec:.2f}s audio synthesized at RTF={rtf}. "
            f"Loudness: {master_stats['after_lufs']} LUFS, True Peak: {master_stats['after_true_peak']} dBFS"
        )
        return output_files

    def validate(self, output_files: list[str]) -> tuple[bool, str | None]:
        """Validates all voice outputs against acoustic and duration requirements."""
        # Check basic file existence
        base_valid, base_err = super().validate(output_files)
        if not base_valid:
            return False, base_err

        master_rel = "audio/narration.wav"
        master_path = self.run_dir / master_rel
        if not master_path.exists():
            return False, f"Master narration file not found: {master_rel}"

        if master_path.stat().st_size < 1000:
            return False, f"Master narration file is suspiciously small or empty: {master_rel}"

        # Check durations and audio properties of master
        try:
            info = read_wav_info(master_path)
        except Exception as exc:
            return False, f"Failed to read master narration WAV: {exc}"

        if info["sample_rate"] != 48000:
            return False, f"Master sample rate must be 48000 Hz, got {info['sample_rate']} Hz"

        if info["channels"] != 1:
            return False, f"Master audio must be mono, got {info['channels']} channels"

        # Duration check: between 90s and 220s (target 1.5 to 3.5 minutes)
        total_dur = info["duration_sec"]
        if total_dur < 90.0 or total_dur > 220.0:
            return False, f"Total narration duration {total_dur:.1f}s outside valid window (90s - 220s)"

        # Check scene files and internal silence gaps
        scenes_dir = self.run_dir / "audio" / "scenes"
        if not scenes_dir.exists():
            return False, "audio/scenes directory does not exist"

        scene_files = list(scenes_dir.glob("*.wav"))
        if not scene_files:
            return False, "No scene WAV files found in audio/scenes"

        for sf in scene_files:
            if sf.stat().st_size < 1000:
                return False, f"Scene audio {sf.name} is empty or too small"
            max_silence = check_audio_silence_gaps(sf)
            if max_silence > 1.5:
                return False, f"Scene audio {sf.name} contains internal silence gap of {max_silence:.2f}s > 1.5s"

        # Verify loudness and true peak with ffmpeg ebur128
        meas_cmd = [
            "ffmpeg",
            "-i", str(master_path),
            "-filter_complex", "ebur128=peak=true",
            "-f", "null",
            "-",
        ]
        try:
            meas = subprocess.run(meas_cmd, capture_output=True, text=True, check=True)
            out_tp_match = re.search(r"Peak:\s*(-?[\d\.]+)\s*dBFS", meas.stderr)
            if out_tp_match:
                tp = float(out_tp_match.group(1))
                if tp > -0.05:
                    return False, f"Digital clipping detected in narration.wav: True Peak = {tp:.2f} dBFS >= 0 dBFS"
        except Exception as exc:
            return False, f"Failed to run audio loudness validation: {exc}"

        # Speaker consistency gate check from voice_report.json
        report_path = self.run_dir / "audio" / "voice_report.json"
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as rf:
                    rep_data = json.load(rf)
                spk_data = rep_data.get("speaker_consistency", {})
                threshold = float(spk_data.get("similarity_threshold", 0.72))
                tol_pct = float(spk_data.get("f0_tolerance_pct", 0.25))
                metrics_map = spk_data.get("metrics", {})

                for item_id, m in metrics_map.items():
                    sim = float(m.get("cosine_sim", 1.0))
                    f0 = float(m.get("f0_median_hz", 0.0))
                    passes_gate = m.get("passes_gate", True)

                    if sim < threshold:
                        return (
                            False,
                            f"Speaker consistency gate failed for item '{item_id}': "
                            f"similarity {sim:.4f} < threshold {threshold:.4f}",
                        )
                    # Check pitch tolerance against reference if provided
                    ref_f0 = float(spk_data.get("reference_f0_hz", 0.0))
                    if ref_f0 > 0.0 and f0 > 0.0:
                        diff = abs(f0 - ref_f0) / ref_f0
                        if diff > tol_pct:
                            return (
                                False,
                                f"Speaker pitch gate failed for item '{item_id}': "
                                f"F0 {f0:.1f}Hz differs from reference {ref_f0:.1f}Hz by {diff*100:.1f}% > {tol_pct*100:.1f}%",
                            )
                    elif not passes_gate:
                        return (
                            False,
                            f"Speaker consistency gate failed for item '{item_id}': failed gate checks",
                        )
            except Exception as exc:
                return False, f"Failed to validate speaker consistency: {exc}"

        return True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nepal News Bot Voice Agent")
    parser.add_argument("--date", default="2026-11-06", help="Date of run in YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Run with FakeTTS engine")
    parser.add_argument("--engine", default=None, help="TTS engine override (e.g. fake, ming)")
    parser.add_argument("--scene", default=None, help="Re-synthesize only a single scene")
    args = parser.parse_args()

    run_dir = Path("data/runs") / args.date
    manifest_path = run_dir / "manifest.json"

    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cfg_path = Path("config/pipeline.yaml")
    pipeline_cfg = {}
    if pipeline_cfg_path.exists():
        with open(pipeline_cfg_path, "r", encoding="utf-8") as f:
            pipeline_cfg = yaml.safe_load(f)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        from src.utils.manifest import create_initial_manifest
        manifest = create_initial_manifest(f"run_{args.date}", args.date, pipeline_cfg)

    agent = VoiceAgent(run_dir=run_dir, config=pipeline_cfg)
    output_files = agent.run(
        manifest,
        only_scene=args.scene,
        engine_override=args.engine,
        dry_run=args.dry_run,
    )
    is_valid, err = agent.validate(output_files)
    if not is_valid:
        print(f"ERROR: Voice Agent validation failed: {err}", file=sys.stderr)
        save_manifest(manifest, manifest_path)
        sys.exit(1)

    save_manifest(manifest, manifest_path)
    print(f"SUCCESS: Voice Agent finished successfully. Output files: {output_files}")


if __name__ == "__main__":
    main()

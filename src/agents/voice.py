"""Voice Agent implementation.

Synthesizes natural broadcast speech from script.json using an open-source TTS
engine (Ming-3.0-tts-0.5b via isolated worker, or FakeTTS for tests).
Masters the master narration to broadcast loudness (-16 LUFS / -1.5 dB True Peak).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.agents.base import BaseAgent
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest, update_stage_status
from src.utils.office_events import emit_office_event
from src.utils.text_normalizer import normalize_for_speech
from src.utils.tts import (
    FakeTTS as FakeTTS,
    MingEngine as MingEngine,
    assemble_scene_audio,
    compute_voice_quality_index,
    concatenate_scenes,
    get_scene_padding,
    get_tts_engine,
    master_audio_loudnorm,
    measure_wav_artifacts,
    measure_wav_ebur128,
    measure_wav_ebur128_segment,
    read_wav_info,
    unload_tts_daemon,
)


from src.utils.text_normalizer import (
    get_unverified_pronunciations_applied as get_unverified_pronunciations_applied,
    load_pronunciations as load_pronunciations,
    load_pronunciations_with_metadata as load_pronunciations_with_metadata,
    split_into_sentences as split_into_sentences,
)


def extract_pending_proper_nouns(
    scenes: list[dict[str, Any]],
    pronunciations_file: Path | str = "config/pronunciations.yaml",
    whitelist_file: Path | str = "config/editor_whitelist.yaml",
) -> list[str]:
    """Identifies unmapped proper nouns across scenes that are absent from pronunciations and whitelist."""
    known_words: set[str] = set()
    p_path = Path(pronunciations_file)
    if p_path.exists():
        try:
            with open(p_path, "r", encoding="utf-8") as f:
                p_data = yaml.safe_load(f) or {}
            for w in p_data.get("pronunciations", {}).keys():
                for tok in re.findall(r"\b[a-zA-Z]+\b", w):
                    known_words.add(tok.lower())
        except Exception:
            pass

    w_path = Path(whitelist_file)
    if w_path.exists():
        try:
            with open(w_path, "r", encoding="utf-8") as f:
                w_data = yaml.safe_load(f) or {}
            for item in w_data.get("proper_nouns", []):
                for tok in re.findall(r"\b[a-zA-Z]+\b", item):
                    known_words.add(tok.lower())
        except Exception:
            pass

    pending: set[str] = set()
    for sc in scenes:
        narration = sc.get("narration", "")
        sentences = re.split(r"(?<=[.!?])\s+", narration)
        for sent in sentences:
            tokens = sent.strip().split()
            for idx, tok in enumerate(tokens):
                if idx == 0:
                    continue
                clean = re.sub(r"[^a-zA-Z]", "", tok)
                if len(clean) >= 2 and clean[0].isupper() and clean.lower() not in known_words:
                    pending.add(clean)

    return sorted(pending)


def clean_narration_text(text: str, pronunciations: dict[str, str] | None = None) -> str:
    """Cleans and normalizes script text for broadcast narration.

    Delegates to the shared normalizer (src.utils.text_normalizer.normalize_for_speech)
    so VoiceAgent and TimelineAgent expand pronunciations, currency, percentages,
    decimals, numbers, and acronyms identically.
    """
    return normalize_for_speech(text, pronunciations)


def _compute_file_sha256(path: Path) -> str:
    """Returns SHA-256 hex digest of a file, or 'missing' if absent."""
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_scene_cache_key(
    cleaned_text: str,
    engine_name: str,
    voice: str,
    task: str,
    language_id: str,
    chunk_mode: str,
    ref_sha256: str,
    ref_txt_sha256: str,
    engine_settings: dict[str, Any],
    padding_policy: str = "broadcast",
    pad_before: float = 0.0,
    pad_after: float = 0.0,
    target_scene_rms_dbfs: float | None = None,
) -> str:
    """Computes deterministic cache key for a scene's synthesized audio."""
    payload = {
        "v": 3,
        "text": cleaned_text,
        "engine": engine_name,
        "voice": voice,
        "task": task,
        "language_id": language_id,
        "chunk_mode": chunk_mode,
        "ref_sha256": ref_sha256,
        "ref_txt_sha256": ref_txt_sha256,
        "model_path": engine_settings.get("model_path"),
        "base_seed": engine_settings.get("seed"),
        "padding_policy": padding_policy,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "target_scene_rms_dbfs": target_scene_rms_dbfs,
        "n_candidates": engine_settings.get("n_candidates"),
        "f0_anchor": engine_settings.get("f0_anchor"),
        "speed": engine_settings.get("speed"),
        "cfg": engine_settings.get("cfg"),
        "sigma": engine_settings.get("sigma"),
        "temperature": engine_settings.get("temperature"),
        "max_length": engine_settings.get("max_length"),
        "crossfade_ms": engine_settings.get("crossfade_ms"),
        "text_normalization": engine_settings.get("text_normalization"),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _create_scene_aliases(scene_id: str, scene_wav_path: Path, audio_dir: Path) -> None:
    """Creates audio/scene_<id>.wav and audio/<id>.wav aliases using hardlinks (fallback to copyfile)."""
    alias_path = audio_dir / f"scene_{scene_id}.wav"
    try:
        if alias_path.exists():
            alias_path.unlink()
        os.link(scene_wav_path, alias_path)
    except Exception:
        shutil.copyfile(scene_wav_path, alias_path)

    if scene_id.startswith("scene_"):
        id_alias = audio_dir / f"{scene_id}.wav"
        try:
            if id_alias.exists():
                id_alias.unlink()
            os.link(scene_wav_path, id_alias)
        except Exception:
            shutil.copyfile(scene_wav_path, id_alias)


def check_audio_silence_gaps(wav_path: Path | str, threshold_db: float = -45.0) -> float:
    """Measures maximum internal silence gap in seconds inside a WAV file (format-aware, vectorized)."""
    p = Path(wav_path)
    if not p.exists() or p.stat().st_size == 0:
        return 0.0

    try:
        import soundfile as sf
        arr, sr = sf.read(str(p), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr == 0 or len(arr) == 0:
            return 0.0

        win_size = int(sr * 0.05)  # 50ms window
        if win_size == 0:
            return 0.0

        thresh_lin = 10.0 ** (threshold_db / 20.0)
        thresh_sq = thresh_lin * thresh_lin

        total_wins = len(arr) // win_size
        if total_wins == 0:
            return 0.0

        win_blocks = arr[: total_wins * win_size].reshape(total_wins, win_size)
        mean_sq = np.mean(win_blocks ** 2, axis=1)
        is_silent = mean_sq < thresh_sq

        if not np.any(is_silent):
            return 0.0

        padded = np.pad(is_silent.astype(np.int8), (1, 1), mode="constant")
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_streak = int(np.max(ends - starts)) if len(starts) > 0 else 0
        return round(max_streak * 0.05, 3)
    except Exception:
        pass

    import wave

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        channels = max(1, wf.getnchannels())
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sr == 0 or len(raw) == 0:
        return 0.0

    thresh_lin = 10.0 ** (threshold_db / 20.0)
    thresh_sq = thresh_lin * thresh_lin
    win_size = int(sr * 0.05)  # 50ms window
    if win_size == 0:
        return 0.0

    try:
        import numpy as np

        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            try:
                arr = np.frombuffer(raw, dtype=np.float32).copy()
            except Exception:
                arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            # Fallback for 24-bit PCM
            raw_bytes = np.frombuffer(raw, dtype=np.uint8)
            samples_24 = raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 3)].reshape(-1, 3)
            padded_24 = np.pad(samples_24, ((0, 0), (1, 0)), mode="constant")
            arr = padded_24.view(dtype="<i4").squeeze(-1).astype(np.float32) / 2147483648.0

        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)

        total_wins = len(arr) // win_size
        if total_wins == 0:
            return 0.0

        win_blocks = arr[: total_wins * win_size].reshape(total_wins, win_size)
        mean_sq = np.mean(win_blocks ** 2, axis=1)
        is_silent = mean_sq < thresh_sq

        if not np.any(is_silent):
            return 0.0

        padded = np.pad(is_silent.astype(np.int8), (1, 1), mode="constant")
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_streak = int(np.max(ends - starts)) if len(starts) > 0 else 0
        return round(max_streak * 0.05, 3)

    except Exception:
        # Robust fallback
        import struct

        bytes_per_sample = max(1, sampwidth)
        frame_bytes = bytes_per_sample * channels
        win_bytes = win_size * frame_bytes
        total_wins = len(raw) // win_bytes

        max_val = float(1 << (8 * bytes_per_sample - 1)) if bytes_per_sample > 1 else 128.0
        threshold_amp = thresh_lin * max_val
        threshold_sq = threshold_amp * threshold_amp

        max_streak = 0
        curr_streak = 0

        for i in range(total_wins):
            chunk = raw[i * win_bytes : (i + 1) * win_bytes]
            if bytes_per_sample == 2:
                samples = struct.unpack(f"<{win_size * channels}h", chunk)
            else:
                samples = [0]
            sum_sq = sum(float(s * s) for s in samples)
            mean_sq = sum_sq / (win_size * channels)
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
        cfg = self.config
        if not cfg:
            cfg_path = Path("config/pipeline.yaml")
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                except Exception:
                    cfg = {}
        stages = cfg.get("stages", [])
        if isinstance(stages, dict):
            return stages.get("voice", {})
        voice_cfg = next((s for s in stages if isinstance(s, dict) and s.get("id") == "voice"), {})
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
        cfg = float(voice_cfg.get("cfg", 2.0))
        sigma = float(voice_cfg.get("sigma", 0.25))
        temperature = float(voice_cfg.get("temperature", 0.0))
        max_length = int(voice_cfg.get("max_length", 24))
        crossfade_ms = float(voice_cfg.get("crossfade_ms", 12.0))
        daemon_url = voice_cfg.get("daemon_url")
        raw_batching = voice_cfg.get("batching")
        batching = int(raw_batching) if raw_batching is not None else 0
        text_normalization = voice_cfg.get("text_normalization", False)
        chunk_mode = voice_cfg.get("chunk_mode", "scene").lower().strip()
        reference_wav = voice_cfg.get("reference_wav", "config/voice/reference.wav")
        reference_txt = voice_cfg.get("reference_txt", "config/voice/reference.txt")
        similarity_threshold = float(voice_cfg.get("similarity_threshold", 0.72))
        f0_tolerance_pct = float(voice_cfg.get("f0_tolerance_pct", 0.15))
        n_candidates = int(voice_cfg.get("n_candidates", 1))
        f0_anchor = str(voice_cfg.get("f0_anchor", "running_median")).strip().lower()
        ranker_weights = voice_cfg.get(
            "ranker_weights",
            {"cosine": 1.0, "f0": 0.6, "duration": 0.2, "artifacts": 0.2},
        )
        base_seed = int(voice_cfg.get("base_seed", 1895))
        max_retries = int(voice_cfg.get("max_retries", 2))
        padding_policy = str(voice_cfg.get("padding_policy", "fixed")).strip().lower()
        pause_between_sentences = float(voice_cfg.get("pause_between_sentences", 0.25))
        pad_before = float(voice_cfg.get("pad_before", 0.4))
        pad_after = float(voice_cfg.get("pad_after", 0.6))
        target_lufs = float(voice_cfg.get("target_lufs", -16.0))
        raw_scene_rms = voice_cfg.get("target_scene_rms_dbfs", -23.0)
        target_scene_rms_dbfs = float(raw_scene_rms) if raw_scene_rms is not None else None
        loudness_strategy = str(voice_cfg.get("loudness_strategy", "linear_strict")).strip().lower()
        limiter_ceiling_dbfs = float(voice_cfg.get("limiter_ceiling_dbfs", -1.5))
        compressor = bool(voice_cfg.get("compressor", True))
        limiter = bool(voice_cfg.get("limiter", True))
        highpass_hz = float(voice_cfg.get("highpass_hz", 80.0))
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
        reference_f0_hz = 0.0

        engine_settings = {
            "model_path": model_path,
            "task": task,
            "voice": voice,
            "language_id": language_id,
            "instruction_schema": instruction_schema,
            "device": device,
            "dtype": dtype,
            "speed": speed,
            "cfg": cfg,
            "sigma": sigma,
            "temperature": temperature,
            "max_length": max_length,
            "crossfade_ms": crossfade_ms,
            "daemon_url": daemon_url,
            "text_normalization": text_normalization,
            "reference_wav": reference_wav,
            "reference_txt": reference_txt,
            "similarity_threshold": similarity_threshold,
            "f0_tolerance_pct": f0_tolerance_pct,
            "n_candidates": n_candidates,
            "f0_anchor": f0_anchor,
            "ranker_weights": ranker_weights,
            "seed": base_seed,
            "max_retries": max_retries,
        }

        ref_sha256 = _compute_file_sha256(Path(reference_wav)) if engine_name != "fake" else "fake"
        ref_txt_sha256 = _compute_file_sha256(Path(reference_txt)) if engine_name != "fake" and reference_txt else "fake"
        all_items: list[dict[str, Any]] = []
        scene_items_map: dict[str, list[Path]] = {}
        scene_cache_keys: dict[str, str] = {}
        cached_scenes: set[str] = set()

        for idx, scene in enumerate(target_scenes):
            scene_id = scene["id"]
            # READ ONLY narration text
            raw_narration = scene.get("narration", "")
            cleaned_text = clean_narration_text(raw_narration, pronunciations)
            s_pad_before, s_pad_after = get_scene_padding(
                scene_id=scene_id,
                scene_idx=idx,
                total_scenes=len(target_scenes),
                padding_policy=padding_policy,
                pad_before_fallback=pad_before,
                pad_after_fallback=pad_after,
            )
            cache_key = _compute_scene_cache_key(
                cleaned_text=cleaned_text,
                engine_name=engine_name,
                voice=voice,
                task=task,
                language_id=language_id,
                chunk_mode=chunk_mode,
                ref_sha256=ref_sha256,
                ref_txt_sha256=ref_txt_sha256,
                engine_settings=engine_settings,
                padding_policy=padding_policy,
                pad_before=s_pad_before,
                pad_after=s_pad_after,
                target_scene_rms_dbfs=target_scene_rms_dbfs,
            )
            scene_cache_keys[scene_id] = cache_key

            scene_wav_path = scenes_dir / f"{scene_id}.wav"
            scene_meta_path = scenes_dir / f"{scene_id}.meta.json"
            force_regen = (only_scene is not None and scene_id == only_scene)

            # Check persistent global cache (data/cache/voice/) for intro scenes across runs
            persistent_voice_cache_dir = Path("data/cache/voice")
            is_semantic_intro_hit = False
            if not force_regen and scene_id == "intro" and (not scene_wav_path.exists() or scene_wav_path.stat().st_size <= 1000):
                from src.utils.category_intro import get_intro_cache_filenames, identify_category_from_intro
                cat_key = identify_category_from_intro(scenes[idx], manifest)
                if cat_key:
                    sem_wav, sem_meta = get_intro_cache_filenames(cat_key)
                    p_wav = persistent_voice_cache_dir / sem_wav
                    p_meta = persistent_voice_cache_dir / sem_meta

                    # Fallback to legacy hashed intro filename if semantic filename is not present
                    if not (p_wav.exists() and p_meta.exists() and p_wav.stat().st_size > 1000):
                        p_wav_h = persistent_voice_cache_dir / f"intro_{cache_key}.wav"
                        p_meta_h = persistent_voice_cache_dir / f"intro_{cache_key}.meta.json"
                        if p_wav_h.exists() and p_meta_h.exists() and p_wav_h.stat().st_size > 1000:
                            p_wav, p_meta = p_wav_h, p_meta_h

                    if p_wav.exists() and p_meta.exists() and p_wav.stat().st_size > 1000:
                        try:
                            p_meta_data = json.loads(p_meta.read_text(encoding="utf-8"))
                            if p_meta_data.get("cache_key") == cache_key:
                                shutil.copyfile(p_wav, scene_wav_path)
                                shutil.copyfile(p_meta, scene_meta_path)
                                is_semantic_intro_hit = True
                                self.logger.info(f"[PERSISTENT CACHE HIT] Reusing cached intro audio across runs: {p_wav.name}")
                            else:
                                self.logger.info(f"[PERSISTENT CACHE STALE] Narration or settings changed for {p_wav.name}, re-synthesizing...")
                        except Exception as exc:
                            self.logger.warning(f"Failed copying persistent cached intro: {exc}")

            if not force_regen and scene_wav_path.exists() and scene_meta_path.exists() and scene_wav_path.stat().st_size > 1000:
                try:
                    meta_data = json.loads(scene_meta_path.read_text(encoding="utf-8"))
                    info = read_wav_info(scene_wav_path)
                    passes_consistency = is_semantic_intro_hit or all(m.get("passes_gate", True) for m in meta_data.get("consistency", {}).values())
                    if (
                        (meta_data.get("cache_key") == cache_key or is_semantic_intro_hit)
                        and info["duration_sec"] > 0.1
                        and passes_consistency
                    ):
                        cached_scenes.add(scene_id)
                        for k_id, k_metrics in meta_data.get("consistency", {}).items():
                            c_copy = dict(k_metrics)
                            c_copy["cached"] = True
                            scene_consistency_map[k_id] = c_copy
                        if not reference_f0_hz and meta_data.get("reference_f0_hz"):
                            reference_f0_hz = float(meta_data["reference_f0_hz"])
                        self.logger.info(f"Reusing cached scene audio (hash match): {scene_id}")
                        continue
                except Exception:
                    pass

            if chunk_mode == "scene":
                s_id = scene_id
                s_path = temp_sentences_dir / f"{s_id}_raw.wav"
                all_items.append({"id": s_id, "text": cleaned_text, "out_path": str(s_path), "scene_id": scene_id})
                scene_items_map[scene_id] = [s_path]
            else:
                sentences = split_into_sentences(cleaned_text)
                scene_items_map[scene_id] = []
                for s_idx, sent in enumerate(sentences):
                    s_id = f"{scene_id}_s{s_idx}"
                    s_path = temp_sentences_dir / f"{s_id}.wav"
                    all_items.append({"id": s_id, "text": sent, "out_path": str(s_path), "scene_id": scene_id})
                    scene_items_map[scene_id].append(s_path)

        # Determine which items actually need synthesis
        items_to_synthesize = []
        for item in all_items:
            s_path = Path(item["out_path"])
            scene_id = item.get("scene_id") or item["id"].split("_s")[0]
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
                self.logger.info(f"Using existing cached chunk audio: {item['id']}")

        total_synthesis_time = 0.0
        total_model_load_time = 0.0
        stage_breakdown = {
            "model_load": 0.0,
            "prefill": 0.0,
            "ar_step": 0.0,
            "cfm": 0.0,
            "vae_decode": 0.0,
            "qc": 0.0,
            "write": 0.0,
        }

        if items_to_synthesize:
            step_size = batching if batching > 0 else len(items_to_synthesize)
            self.logger.info(
                f"Synthesizing {len(items_to_synthesize)} {chunk_mode}(s) "
                f"({len(cached_scenes)} scene(s) cached, {len(all_items) - len(items_to_synthesize)} chunk(s) cached) "
                f"across {len(target_scenes)} scene(s)..."
            )
            t0 = time.time()
            for b_idx in range(0, len(items_to_synthesize), step_size):
                b_items = items_to_synthesize[b_idx : b_idx + step_size]
                cur_done = min(b_idx + len(b_items), len(items_to_synthesize))
                emit_office_event(
                    self.run_dir,
                    "substage_progress",
                    stage="voice",
                    current=cur_done,
                    total=len(items_to_synthesize),
                    message=f"Synthesizing voice chunk {cur_done}/{len(items_to_synthesize)}",
                )
                b_settings = dict(engine_settings)
                b_settings["seed"] = base_seed + (b_idx * 100)
                batch_result = tts_engine.synthesize_batch(b_items, b_settings)
                total_model_load_time += batch_result.get("model_load_time_sec", 0.0)
                if "stage_breakdown_sec" in batch_result:
                    for k, v in batch_result["stage_breakdown_sec"].items():
                        stage_breakdown[k] = round(stage_breakdown.get(k, 0.0) + v, 3)
                if batch_result.get("reference_f0_hz"):
                    reference_f0_hz = float(batch_result["reference_f0_hz"])

                # Inspect per-item result for failures and retries
                for res_item in batch_result.get("results", []):
                    total_retries += res_item.get("retries", 0)
                    r_id = res_item.get("id", "")
                    scene_consistency_map[r_id] = {
                        "cosine_sim": res_item.get("cosine_sim", 1.0),
                        "f0_median_hz": res_item.get("f0_median_hz", 0.0),
                        "target_f0_hz": res_item.get("target_f0_hz", 0.0),
                        "retries": res_item.get("retries", 0),
                        "passes_gate": res_item.get("passes_gate", True),
                        "selected_candidate_idx": res_item.get("selected_candidate_idx", 0),
                        "seed_used": res_item.get("seed_used", base_seed),
                        "ranker_score": res_item.get("ranker_score"),
                        "ranker_breakdown": res_item.get("ranker_breakdown"),
                        "candidates_evaluated": res_item.get("candidates_evaluated"),
                    }
                    if res_item.get("status") == "failed":
                        err_msg = res_item.get("error", "Unknown error")
                        raise RuntimeError(f"Failed to synthesize {chunk_mode} {r_id}: {err_msg}")

            total_synthesis_time = time.time() - t0
            stage_breakdown["model_load"] = round(total_model_load_time, 2)
        else:
            self.logger.info(
                f"All {len(target_scenes)} scene(s) already synthesized and valid on disk."
            )

        # Assemble each newly synthesized scene with padding, pauses, and active RMS leveling
        t_asm_start = time.time()
        pause_to_use = 0.0 if chunk_mode == "scene" else pause_between_sentences
        for idx, scene in enumerate(target_scenes):
            scene_id = scene["id"]
            scene_wav_path = scenes_dir / f"{scene_id}.wav"
            emit_office_event(
                self.run_dir,
                "substage_progress",
                stage="voice",
                current=idx + 1,
                total=len(target_scenes),
                message=f"Voice scene {idx + 1}/{len(target_scenes)} ({scene_id})",
                extra={"scene_id": scene_id},
            )
            if scene_id in cached_scenes:
                _create_scene_aliases(scene_id, scene_wav_path, audio_dir)
                continue
            wav_paths = scene_items_map[scene_id]
            s_pad_before, s_pad_after = get_scene_padding(
                scene_id=scene_id,
                scene_idx=idx,
                total_scenes=len(target_scenes),
                padding_policy=padding_policy,
                pad_before_fallback=pad_before,
                pad_after_fallback=pad_after,
            )
            scene_dur = assemble_scene_audio(
                wav_paths,
                scene_wav_path,
                pause_sec=pause_to_use,
                pad_before=s_pad_before,
                pad_after=s_pad_after,
                target_sample_rate=44100,
                target_rms_dbfs=target_scene_rms_dbfs,
            )

            # Save per-scene content-hash metadata for future cache hits
            scene_meta_path = scenes_dir / f"{scene_id}.meta.json"
            scene_cons = {
                k: v for k, v in scene_consistency_map.items()
                if k == scene_id or k.startswith(f"{scene_id}_s")
            }
            scene_meta_path.write_text(
                json.dumps(
                    {
                        "cache_key": scene_cache_keys[scene_id],
                        "reference_f0_hz": reference_f0_hz,
                        "consistency": scene_cons,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            # If intro scene, also save persistent copy in data/cache/voice/ for future runs
            if scene_id == "intro":
                try:
                    from src.utils.category_intro import get_intro_cache_filenames, identify_category_from_intro
                    cat_key = identify_category_from_intro(scenes[idx], manifest)
                    if cat_key:
                        sem_wav, sem_meta = get_intro_cache_filenames(cat_key)
                        persistent_voice_cache_dir = Path("data/cache/voice")
                        persistent_voice_cache_dir.mkdir(parents=True, exist_ok=True)
                        p_wav = persistent_voice_cache_dir / sem_wav
                        p_meta = persistent_voice_cache_dir / sem_meta
                        shutil.copyfile(scene_wav_path, p_wav)
                        shutil.copyfile(scene_meta_path, p_meta)
                        self.logger.info(f"[PERSISTENT CACHE SAVED] Cached intro audio for future runs: {p_wav.name}")
                except Exception as exc:
                    self.logger.warning(f"Failed saving persistent cached intro audio: {exc}")

            # Create aliases audio/scene_<id>.wav and audio/<id>.wav using hardlinks (fallback to copyfile)
            _create_scene_aliases(scene_id, scene_wav_path, audio_dir)
            self.logger.info(f"Scene '{scene_id}' assembled: duration = {scene_dur:.2f}s")

        assembly_time_sec = time.time() - t_asm_start

        # Clean up temporary sentence wavs
        shutil.rmtree(temp_sentences_dir, ignore_errors=True)

        # Assemble full list of all scenes in script order
        scene_files_in_order = []
        scene_frames_list = []

        for idx, scene in enumerate(scenes):
            s_id = scene["id"]
            s_file = scenes_dir / f"{s_id}.wav"
            if not s_file.exists():
                raise FileNotFoundError(f"Missing assembled scene audio: {s_file}")

            info = read_wav_info(s_file)
            scene_files_in_order.append(s_file)
            scene_frames_list.append(info.get("frames", 0))

        # Concatenate scenes into intermediate raw track
        raw_concat_path = audio_dir / "raw_concat.wav"
        raw_total_dur = concatenate_scenes(scene_files_in_order, raw_concat_path)

        # Loudness normalization and mastering (high-pass + two-pass loudnorm to 48kHz)
        master_narration_path = audio_dir / "narration.wav"
        self.logger.info(f"Mastering narration with FFmpeg two-pass loudnorm (target={target_lufs} LUFS)...")
        t_master_start = time.time()
        master_stats = master_audio_loudnorm(
            raw_concat_path,
            master_narration_path,
            target_lufs=target_lufs,
            true_peak=limiter_ceiling_dbfs,
            sample_rate=output_sample_rate,
            highpass_hz=highpass_hz,
            compressor=compressor,
            limiter=limiter,
            limiter_ceiling_dbfs=limiter_ceiling_dbfs,
            loudness_strategy=loudness_strategy,
        )
        mastering_time_sec = time.time() - t_master_start
        # Clean up intermediate raw concat
        raw_concat_path.unlink(missing_ok=True)

        final_master_info = read_wav_info(master_narration_path)
        final_duration_sec = final_master_info["duration_sec"]
        total_master_samples = final_master_info.get("frames", 0)
        master_sr = final_master_info.get("sample_rate", output_sample_rate) or 48000
        if total_master_samples == 0 and final_duration_sec > 0:
            total_master_samples = int(round(final_duration_sec * master_sr))

        rtf = round(total_synthesis_time / final_duration_sec, 2) if final_duration_sec > 0 else 0.0

        # Compute sample-exact scene durations anchored to master narration audio
        total_raw_frames = sum(scene_frames_list)
        all_scene_durations = []
        scene_metrics = []
        current_sample = 0

        for idx, scene in enumerate(scenes):
            s_id = scene["id"]
            s_frames = scene_frames_list[idx]

            start_sample = current_sample
            if idx == len(scenes) - 1:
                end_sample = total_master_samples
            else:
                share = (s_frames / total_raw_frames) if total_raw_frames > 0 else (1.0 / len(scenes))
                end_sample = min(total_master_samples, current_sample + int(round(total_master_samples * share)))
            current_sample = end_sample

            start_sec = round(start_sample / master_sr, 4)
            end_sec = round(end_sample / master_sr, 4)
            dur_sec = round((end_sample - start_sample) / master_sr, 4)

            s_pad_before, s_pad_after = get_scene_padding(
                scene_id=s_id,
                scene_idx=idx,
                total_scenes=len(scenes),
                padding_policy=padding_policy,
                pad_before_fallback=pad_before,
                pad_after_fallback=pad_after,
            )

            all_scene_durations.append({
                "id": s_id,
                "file": f"audio/scenes/{s_id}.wav",
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur_sec,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "mastered_16k_path": f"audio/scenes_mastered_16k/{s_id}.wav",
                "pad_before": s_pad_before,
                "pad_after": s_pad_after,
            })
            scene_metrics.append({
                "id": s_id,
                "estimated_duration_sec": scene.get("estimated_duration_sec", 0.0),
                "actual_duration_sec": dur_sec,
                "pad_before": s_pad_before,
                "pad_after": s_pad_after,
                "file": f"audio/scenes/{s_id}.wav",
            })

        # Resample master narration to 16 kHz and slice per-scene mastered audio
        mastered_16k_dir = audio_dir / "scenes_mastered_16k"
        mastered_16k_dir.mkdir(parents=True, exist_ok=True)
        narration_16k_path = audio_dir / "narration_16k.wav"

        cmd_16k = [
            "ffmpeg", "-y",
            "-i", str(master_narration_path),
            "-af", "aresample=resampler=soxr:osr=16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(narration_16k_path),
        ]
        try:
            subprocess.run(cmd_16k, capture_output=True, check=True)
            with wave.open(str(narration_16k_path), "rb") as wf_16k:
                n_16k_frames = wf_16k.getnframes()
                bytes_16k = wf_16k.readframes(n_16k_frames)

            for dur_item in all_scene_durations:
                s_id = dur_item["id"]
                st_16k = dur_item["start_sample"] // 3
                end_16k = dur_item["end_sample"] // 3
                if dur_item["end_sample"] == total_master_samples:
                    end_16k = n_16k_frames
                sc_bytes = bytes_16k[st_16k * 2 : end_16k * 2]
                sc_16k_path = mastered_16k_dir / f"{s_id}.wav"
                with wave.open(str(sc_16k_path), "wb") as sc_wf:
                    sc_wf.setnchannels(1)
                    sc_wf.setsampwidth(2)
                    sc_wf.setframerate(16000)
                    sc_wf.writeframes(sc_bytes)
        except Exception as e:
            self.logger.warning(f"Could not produce 16k mastered audio: {e}")

        # Save scene_durations.json
        durations_json_path = audio_dir / "scene_durations.json"
        with open(durations_json_path, "w", encoding="utf-8") as f:
            json.dump(all_scene_durations, f, indent=2)

        # Measure per-scene loudness and True Peak after mastering
        words_map = {sc["id"]: len(sc.get("narration", "").split()) for sc in scenes}
        master_data, master_sr = None, 48000
        try:
            import soundfile as sf
            master_data, master_sr = sf.read(str(master_narration_path), dtype="float32")
            if master_data.ndim > 1:
                master_data = master_data.mean(axis=1)
        except Exception:
            pass

        for sm in scene_metrics:
            s_file = audio_dir / f"scenes/{sm['id']}.wav"
            dur_info = next((d for d in all_scene_durations if d["id"] == sm["id"]), None)
            start_t = dur_info["start_sec"] if dur_info else 0.0
            dur_t = dur_info["duration_sec"] if dur_info else sm["actual_duration_sec"]

            sc_lufs = -70.0
            sc_tp = -99.0
            if s_file.exists() and s_file.stat().st_size > 0:
                try:
                    s_arr, _ = sf.read(str(s_file), dtype="float32")
                    if s_arr.ndim > 1:
                        s_arr = s_arr.mean(axis=1)
                    if len(s_arr) > 0:
                        s_rms = float(np.sqrt(np.mean(s_arr ** 2)))
                        s_peak = float(np.max(np.abs(s_arr)))
                        sc_lufs = round(20.0 * np.log10(max(1e-9, s_rms)) - 0.691, 2)
                        sc_tp = round(20.0 * np.log10(max(1e-9, s_peak)), 2)
                except Exception:
                    pass

            m_lufs = -70.0
            m_tp = -99.0
            if master_data is not None and len(master_data) > 0 and master_sr > 0:
                start_samp = int(max(0.0, start_t) * master_sr)
                end_samp = int(min(len(master_data), (start_t + max(0.01, dur_t)) * master_sr))
                if end_samp > start_samp:
                    seg = master_data[start_samp:end_samp]
                    seg_rms = float(np.sqrt(np.mean(seg ** 2)))
                    seg_peak = float(np.max(np.abs(seg)))
                    m_lufs = round(20.0 * np.log10(max(1e-9, seg_rms)) - 0.691, 2)
                    m_tp = round(20.0 * np.log10(max(1e-9, seg_peak)), 2)

            sm["lufs"] = sc_lufs
            sm["true_peak_db"] = sc_tp
            sm["mastered_lufs"] = m_lufs
            sm["mastered_true_peak_db"] = m_tp
            sm["word_count"] = words_map.get(sm["id"], 0)

        # Measure audio artifacts & build Voice Quality Index (VQI)
        artifact_flags = measure_wav_artifacts(master_narration_path)
        unverified_applied = get_unverified_pronunciations_applied(scenes)
        pending_proper_nouns = extract_pending_proper_nouns(scenes)
        total_pad_sec = sum(
            sm.get("pad_before", 0.0) + sm.get("pad_after", 0.0) for sm in scene_metrics
        )
        avg_pad_sec = total_pad_sec / max(1, len(scene_metrics))
        quality_block = compute_voice_quality_index(
            consistency_metrics=scene_consistency_map,
            reference_f0_hz=reference_f0_hz,
            scenes_info=scene_metrics,
            total_duration_sec=final_duration_sec,
            pad_per_scene_sec=round(avg_pad_sec, 2),
            artifact_flags=artifact_flags,
            unverified_pronunciations=unverified_applied,
        )
        pure_synthesis_time = max(0.0, total_synthesis_time - total_model_load_time)
        perf_block = {
            "model_load_time_sec": round(total_model_load_time, 2),
            "synthesis_wall_time_sec": round(pure_synthesis_time, 2),
            "assembly_time_sec": round(assembly_time_sec, 2),
            "mastering_time_sec": round(mastering_time_sec, 2),
            "total_wall_time_sec": round(
                total_model_load_time + pure_synthesis_time + assembly_time_sec + mastering_time_sec, 2
            ),
            "real_time_factor": rtf,
            "stage_breakdown_sec": stage_breakdown,
        }

        # Build voice_report.json
        speaker_consistency_payload: dict[str, Any] = {
            "reference_wav": str(reference_wav),
            "similarity_threshold": similarity_threshold,
            "f0_tolerance_pct": f0_tolerance_pct,
            "n_candidates": n_candidates,
            "f0_anchor": f0_anchor,
            "padding_policy": padding_policy,
            "ranker_weights": ranker_weights,
            "base_seed": base_seed,
            "metrics": scene_consistency_map,
        }
        if reference_f0_hz > 0.0:
            speaker_consistency_payload["reference_f0_hz"] = reference_f0_hz

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
            "loudness_after_lufs": master_stats.get("after_lufs"),
            "true_peak_after_db": master_stats.get("after_true_peak"),
            "speaker_consistency": speaker_consistency_payload,
            "loudnorm": master_stats,
            "scenes": scene_metrics,
            "unverified_pronunciations_applied": unverified_applied,
            "pending_proper_nouns": pending_proper_nouns,
            "quality": quality_block,
            "perf": perf_block,
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
            "quality": quality_block,
            "perf": perf_block,
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

        # Unload TTS daemon to free GPU VRAM for downstream processes (TimelineAgent, RenderAgent)
        daemon_url = voice_cfg.get("daemon_url", "http://127.0.0.1:8766")
        if daemon_url and engine_name == "ming":
            try:
                unloaded = unload_tts_daemon(daemon_url=daemon_url)
                if unloaded:
                    self.logger.info("TTS daemon VRAM unloaded successfully.")
            except Exception as e:
                self.logger.warning(f"Could not unload TTS daemon: {e}")

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

        # Duration check: between min_duration_sec and max_duration_sec from config
        voice_cfg = self._get_voice_stage_config()
        only_scene = voice_cfg.get("only_scene")
        if not only_scene:
            min_dur = float(voice_cfg.get("min_duration_sec", 60.0))
            max_dur = float(voice_cfg.get("max_duration_sec", 240.0))
            total_dur = info["duration_sec"]
            if total_dur < min_dur or total_dur > max_dur:
                return False, f"Total narration duration {total_dur:.1f}s outside valid window ({min_dur}s - {max_dur}s)"

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
                tol_pct = float(spk_data.get("f0_tolerance_pct", 0.15))
                metrics_map = spk_data.get("metrics", {})

                for item_id, m in metrics_map.items():
                    if m.get("cached"):
                        continue
                    sim = float(m.get("cosine_sim", 1.0))
                    f0 = float(m.get("f0_median_hz", 0.0))
                    passes_gate = m.get("passes_gate", True)

                    if sim < threshold:
                        return (
                            False,
                            f"Speaker consistency gate failed for item '{item_id}': "
                            f"similarity {sim:.4f} < threshold {threshold:.4f}",
                        )
                    # Check pitch tolerance against target or reference if provided
                    target_f0 = float(m.get("target_f0_hz") or spk_data.get("reference_f0_hz", 0.0))
                    if target_f0 > 0.0 and f0 > 0.0:
                        diff = abs(f0 - target_f0) / target_f0
                        if diff > tol_pct:
                            return (
                                False,
                                f"Speaker pitch gate failed for item '{item_id}': "
                                f"F0 {f0:.1f}Hz differs from reference {target_f0:.1f}Hz by {diff*100:.1f}% > {tol_pct*100:.1f}%",
                            )
                    elif not passes_gate:
                        return (
                            False,
                            f"Speaker consistency gate failed for item '{item_id}': failed gate checks",
                        )
                # Pronunciation verification gate check
                voice_cfg = self._get_voice_stage_config()
                allow_unverified = bool(voice_cfg.get("allow_unverified_pronunciations", False))
                unverified_applied = rep_data.get("unverified_pronunciations_applied", [])
                if unverified_applied and not allow_unverified:
                    unverified_words = [item.get("word") for item in unverified_applied]
                    return (
                        False,
                        f"Unverified pronunciation gate failed: unverified entries applied {unverified_words}. "
                        f"All applied pronunciations must have verified: true in config/pronunciations.yaml "
                        f"or allow_unverified_pronunciations enabled in pipeline config.",
                    )
            except Exception as exc:
                return False, f"Failed to validate speaker consistency or pronunciation gate: {exc}"

        return True, None

    def prewarm_intro_cache(self, engine_override: str | None = None, dry_run: bool = False) -> list[str]:
        """Pre-synthesizes and caches intro voice audio for all canonical categories into data/cache/voice/."""
        import tempfile
        from src.utils.category_intro import DEFAULT_CATEGORY_INTROS, resolve_category_intro

        prewarmed: list[str] = []
        for cat_key in DEFAULT_CATEGORY_INTROS:
            intro_data = resolve_category_intro(cat_key)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_run = Path(tmp_dir)
                (tmp_run / "script").mkdir(parents=True, exist_ok=True)
                dummy_script = {
                    "language": "English",
                    "origin": "fixture",
                    "date": "2026-11-21",
                    "scenes": [
                        {
                            "id": "intro",
                            "type": "intro",
                            "headline": intro_data["headline"],
                            "narration": intro_data["narration"],
                            "category": cat_key,
                            "sensitive": False,
                            "source": "Nepal News Desk",
                            "source_urls": [],
                            "visual_keywords": intro_data["visual_keywords"],
                            "evidence": [],
                        }
                    ],
                }
                (tmp_run / "script" / "script.json").write_text(json.dumps(dummy_script), encoding="utf-8")
                dummy_manifest = {
                    "run_id": f"prewarm_{cat_key}",
                    "date": "2026-11-21",
                    "status": "running",
                    "selected_categories": [cat_key],
                    "stages": {"voice": {"status": "pending"}},
                }
                tmp_agent = VoiceAgent(run_dir=tmp_run, config=self.config)
                try:
                    tmp_agent.run(dummy_manifest, engine_override=engine_override, dry_run=dry_run)
                    prewarmed.append(cat_key)
                finally:
                    # Close file handlers to release Windows lock on run.log before tmp_dir cleanup
                    for h in list(tmp_agent.logger.handlers):
                        if isinstance(h, logging.FileHandler):
                            h.close()
                            tmp_agent.logger.removeHandler(h)
        return prewarmed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nepal News Bot Voice Agent")
    parser.add_argument("--date", default="2026-11-06", help="Date of run in YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Run with FakeTTS engine")
    parser.add_argument("--engine", default=None, help="TTS engine override (e.g. fake, ming)")
    parser.add_argument("--scene", default=None, help="Re-synthesize only a single scene")
    parser.add_argument("--prewarm-intros", action="store_true", help="Pre-synthesize and cache intro voice audio for all canonical categories")
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
    if args.prewarm_intros:
        cats = agent.prewarm_intro_cache(engine_override=args.engine, dry_run=args.dry_run)
        print(f"SUCCESS: Pre-warmed intro voice cache for {len(cats)} categories: {cats}")
        return

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

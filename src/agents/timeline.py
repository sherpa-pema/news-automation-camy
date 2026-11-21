"""Timeline Agent implementation.

Performs forced phonetic alignment between narration audio (per scene and master)
and script.json, producing word-level timestamps, caption chunks, and preview video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.agents.base import BaseAgent
from src.utils.text_normalizer import (
    contains_devanagari,
    get_optional_uroman,
    load_pronunciations,
    split_into_sentences,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest
from src.utils.office_events import emit_office_event
from src.utils.schema_validate import validate_contract, validate_file
from src.utils.timeline import (
    CaptionCue,
    FakeAligner as FakeAligner,
    MMSAligner as MMSAligner,
    SceneJob,
    WordTiming,
    audit_captions,
    build_srt_content,
    chunk_scene_words_into_captions,
    generate_timeline_preview_video,
    get_timeline_aligner,
    interpolate_unplaced_words,
    parse_srt,
    prepare_scene_tokens,
    validate_srt_syntax,
)
from src.utils.tts import read_wav_info


def compute_scene_alignment_cache_key(
    job: SceneJob,
    aligner_name: str,
    model_path: str,
    pronunciations: dict[str, str],
    device: str,
    dtype: str,
) -> str:
    """Computes a deterministic hash key for a scene's forced alignment inputs.

    Keyed by:
    - scene audio sha256 + mtime + size
    - scene narration token hash
    - aligner name and model path
    - pronunciations dictionary hash
    - dtype and device
    """
    audio_path = Path(job.get("audio_path", ""))
    audio_sig = ""
    if audio_path.exists():
        stat = audio_path.stat()
        h_audio = hashlib.sha256()
        with open(audio_path, "rb") as f:
            while chunk := f.read(65536):
                h_audio.update(chunk)
        audio_sig = f"{h_audio.hexdigest()}_{stat.st_size}_{stat.st_mtime}"
    else:
        audio_sig = f"missing:{job.get('audio_path')}"

    words_repr = json.dumps(
        [(w.get("orig", ""), w.get("tts", ""), w.get("tokens", [])) for w in job.get("words", [])],
        sort_keys=True,
    )
    narration_hash = hashlib.sha256(words_repr.encode("utf-8")).hexdigest()
    pron_hash = hashlib.sha256(json.dumps(pronunciations, sort_keys=True).encode("utf-8")).hexdigest()

    raw_key = f"{audio_sig}|{narration_hash}|{aligner_name}|{model_path}|{pron_hash}|{dtype}|{device}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]


class TimelineAgent(BaseAgent):
    """Timeline Agent produces word-level and scene-level timestamps from audio and script.

    Outputs:
        - timeline/timeline.json (validates against contracts/timeline.schema.json)
        - timeline/captions.srt  (SubRip subtitles)
        - timeline/captions.json (structured caption phrases with word timings)
        - timeline/timeline_preview.mp4 (black 1280x720 video with burned-in subtitles)
    """

    def __init__(self, run_dir: Path | str, config: dict[str, Any] | None = None) -> None:
        super().__init__(Path(run_dir), config=config)
        self.logger = setup_logger("timeline_agent", self.run_dir / "run.log")

    @property
    def name(self) -> str:
        return "timeline"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/timeline.schema.json"

    def validate(self, output_files: list[str]) -> tuple[bool, str | None]:
        """Validates outputs exist and conform to timeline and captions schemas."""
        base_valid, base_err = super().validate(output_files)
        if not base_valid:
            return False, base_err

        for f in output_files:
            if f.endswith("captions.json"):
                caps_file = self.run_dir / f
                if caps_file.exists():
                    is_valid, err = validate_file(caps_file, "contracts/captions.schema.json")
                    if not is_valid:
                        return False, f"captions.json contract validation failed: {err}"
        return True, None

    def _get_stage_config(self) -> dict[str, Any]:
        """Extracts timeline stage configuration from pipeline.yaml."""
        if not self.config:
            return {}
        stages = self.config.get("stages", [])
        for st in stages:
            if st.get("id") == "timeline":
                return st
        return {}

    def get_artifact_paths(self, scene_filter: str | None = None) -> dict[str, Path]:
        """Returns standard artifact paths for canonical, legacy root, or partial outputs."""
        timeline_dir = self.run_dir / "timeline"
        if scene_filter:
            partial_dir = timeline_dir / "partial"
            return {
                "partial_dir": partial_dir,
                "partial_json": partial_dir / f"{scene_filter}.json",
                "partial_srt": partial_dir / f"{scene_filter}.srt",
                "partial_audit": partial_dir / f"{scene_filter}_audit.json",
            }
        return {
            "timeline_dir": timeline_dir,
            "timeline_json": timeline_dir / "timeline.json",
            "captions_srt": timeline_dir / "captions.srt",
            "captions_json": timeline_dir / "captions.json",
            "caption_audit_json": timeline_dir / "caption_audit.json",
            "preview_mp4": timeline_dir / "timeline_preview.mp4",
            "root_timeline_json": self.run_dir / "timeline.json",
            "root_captions_srt": self.run_dir / "captions.srt",
        }

    def load_inputs(
        self,
        scene_filter: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], float, int, list[SceneJob], list[str], Path, dict[str, str]]:
        """Loads and verifies input files, validating strict scene invariants and duration tiling."""
        script_file = self.run_dir / "script" / "script.json"
        if not script_file.exists():
            raise FileNotFoundError(f"Missing required script file: {script_file}")

        narration_file = self.run_dir / "audio" / "narration.wav"
        if not narration_file.exists():
            raise FileNotFoundError(f"Missing master narration audio: {narration_file}")

        scene_durations_file = self.run_dir / "audio" / "scene_durations.json"
        if not scene_durations_file.exists():
            raise FileNotFoundError(f"Missing scene durations metadata: {scene_durations_file}")

        # 1. Read input metadata
        with open(script_file, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        origin = script_data.get("origin", "editor")
        if origin != "fixture":
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            import hashlib
            curr_hash = hashlib.sha256(script_file.read_bytes()).hexdigest()
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

        with open(scene_durations_file, "r", encoding="utf-8") as f:
            scene_durations: list[dict[str, Any]] = json.load(f)

        narration_info = read_wav_info(narration_file)
        narration_duration = narration_info["duration_sec"]
        narration_sr = narration_info["sample_rate"]

        self.logger.info(
            f"Narration master duration: {narration_duration:.3f}s, "
            f"sample rate: {narration_sr} Hz, total scenes: {len(scene_durations)}"
        )

        # 2. Check scene set and ordering invariants
        script_scenes = script_data.get("scenes", [])
        script_scene_ids = [s["id"] for s in script_scenes]
        audio_scene_ids = [s["id"] for s in scene_durations]

        # Duplicate ID check
        if len(script_scene_ids) != len(set(script_scene_ids)):
            seen_s: set[str] = set()
            dupes_s = [x for x in script_scene_ids if x in seen_s or seen_s.add(x)]
            raise ValueError(f"Duplicate scene IDs found in script.json: {dupes_s}")

        if len(audio_scene_ids) != len(set(audio_scene_ids)):
            seen_a: set[str] = set()
            dupes_a = [x for x in audio_scene_ids if x in seen_a or seen_a.add(x)]
            raise ValueError(f"Duplicate scene IDs found in scene_durations.json: {dupes_a}")

        # Scene set equality check
        script_set = set(script_scene_ids)
        audio_set = set(audio_scene_ids)
        if script_set != audio_set:
            missing_in_audio = sorted(script_set - audio_set)
            missing_in_script = sorted(audio_set - script_set)
            raise ValueError(
                f"Scene set mismatch between script.json and scene_durations.json: "
                f"missing in audio: {missing_in_audio}, missing in script: {missing_in_script}"
            )

        # Scene ordering check
        if script_scene_ids != audio_scene_ids:
            raise ValueError(
                f"Scene order mismatch between script.json ({script_scene_ids}) "
                f"and scene_durations.json ({audio_scene_ids})"
            )

        # 3. Monotonicity and duration validation across scene_durations
        for idx in range(len(scene_durations)):
            s_curr = scene_durations[idx]
            if s_curr.get("start_sec", 0.0) < 0:
                raise ValueError(f"Scene '{s_curr['id']}' has negative start_sec: {s_curr['start_sec']}")
            if s_curr.get("duration_sec", 0.0) <= 0:
                raise ValueError(f"Scene '{s_curr['id']}' has non-positive duration_sec: {s_curr['duration_sec']}")
            if s_curr.get("end_sec", 0.0) <= s_curr.get("start_sec", 0.0):
                raise ValueError(
                    f"Scene '{s_curr['id']}' end_sec <= start_sec ({s_curr['end_sec']} <= {s_curr['start_sec']})"
                )

            if idx < len(scene_durations) - 1:
                s_next = scene_durations[idx + 1]
                if s_curr["start_sec"] >= s_next["start_sec"]:
                    raise ValueError(
                        f"Non-monotonic scene start_sec: '{s_curr['id']}' ({s_curr['start_sec']}s) "
                        f">= next '{s_next['id']}' ({s_next['start_sec']}s)"
                    )
                if abs(s_curr["end_sec"] - s_next["start_sec"]) > 0.001:
                    raise ValueError(
                        f"Scene partition gap/overlap between '{s_curr['id']}' ({s_curr['end_sec']}s) "
                        f"and '{s_next['id']}' ({s_next['start_sec']}s)."
                    )

        # Verify scene duration sum equals master narration duration within 20ms
        sum_scene_durations = sum(s["duration_sec"] for s in scene_durations)
        diff_ms = abs(sum_scene_durations - narration_duration) * 1000.0
        if diff_ms > 20.0:
            raise ValueError(
                f"Scene duration sum ({sum_scene_durations:.3f}s) does not match narration.wav "
                f"({narration_duration:.3f}s) within 20ms tolerance (diff: {diff_ms:.1f}ms)."
            )

        # Reconcile final scene end to narration_duration exactly if within 20ms residual
        if scene_durations:
            final_sc = scene_durations[-1]
            residual = round(narration_duration - final_sc["end_sec"], 4)
            if abs(residual) > 0.0001 and abs(residual) <= 0.020:
                old_end = final_sc["end_sec"]
                final_sc["end_sec"] = round(narration_duration, 3)
                final_sc["duration_sec"] = round(final_sc["end_sec"] - final_sc["start_sec"], 3)
                self.logger.info(
                    f"Reconciled final scene '{final_sc['id']}' end from {old_end:.3f}s to "
                    f"narration duration {narration_duration:.3f}s (residual: {residual*1000:.1f}ms adjusted)."
                )

        # 4. Load pronunciations
        pronunciations = load_pronunciations("config/pronunciations.yaml")

        # 5. Prepare scene alignment jobs
        scene_jobs: list[SceneJob] = []
        script_scenes_map = {s["id"]: s for s in script_scenes}
        prep_warnings: list[str] = []

        for s_meta in scene_durations:
            sc_id = s_meta["id"]
            if scene_filter and sc_id != scene_filter:
                continue

            sc_script = script_scenes_map[sc_id]
            audio_path: Path | None = None
            mastered_rel = s_meta.get("mastered_16k_path")
            if mastered_rel:
                cand = self.run_dir / mastered_rel
                if cand.exists():
                    audio_path = cand
            if not audio_path:
                cand_conv = self.run_dir / "audio" / "scenes_mastered_16k" / f"{sc_id}.wav"
                if cand_conv.exists():
                    audio_path = cand_conv
                else:
                    audio_path = self.run_dir / s_meta["file"]

            if not audio_path.exists():
                raise FileNotFoundError(f"Scene audio file not found: {audio_path}")

            sc_token_warnings: list[str] = []
            token_words = prepare_scene_tokens(
                sc_script["narration"],
                pronunciations,
                warnings_out=sc_token_warnings,
            )
            for tw in sc_token_warnings:
                msg = f"Scene '{sc_id}': {tw}"
                self.logger.warning(msg)
                prep_warnings.append(msg)

            scene_jobs.append(SceneJob(
                id=sc_id,
                audio_path=str(audio_path),
                words=token_words,
                duration_sec=s_meta["duration_sec"],
                start_sec=s_meta["start_sec"],
                end_sec=s_meta["end_sec"],
                pad_before=float(s_meta.get("pad_before", 0.4)),
                pad_after=float(s_meta.get("pad_after", 0.6)),
                headline=sc_script.get("headline", ""),
                narration=sc_script.get("narration", ""),
            ))

        return (
            script_data,
            scene_durations,
            narration_duration,
            narration_sr,
            scene_jobs,
            prep_warnings,
            narration_file,
            pronunciations,
        )

    def align_scenes(
        self,
        scene_jobs: list[SceneJob],
        aligner_name: str,
        stage_cfg: dict[str, Any],
        pronunciations: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> tuple[dict[str, Any], float]:
        """Executes forced alignment across scene jobs with a two-level cache to skip model load."""
        model_path = stage_cfg.get("model_path", "models/MMS_FA/model.pt")
        device = stage_cfg.get("device", "cuda:0")
        dtype = stage_cfg.get("dtype", "float32")
        prons = pronunciations or {}

        cache_dir = self.run_dir / "timeline" / "cache" / "alignment"
        cached_scenes: dict[str, Any] = {}
        uncached_jobs: list[SceneJob] = []
        uncached_cache_paths: dict[str, Path] = {}

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for job in scene_jobs:
                sc_id = job["id"]
                cache_key = compute_scene_alignment_cache_key(
                    job=job,
                    aligner_name=aligner_name,
                    model_path=model_path,
                    pronunciations=prons,
                    device=device,
                    dtype=dtype,
                )
                c_file = cache_dir / f"{cache_key}.json"
                if c_file.exists():
                    try:
                        cached_sc = json.loads(c_file.read_text(encoding="utf-8"))
                        if isinstance(cached_sc, dict) and "words" in cached_sc:
                            cached_scenes[sc_id] = cached_sc
                            continue
                    except Exception as e:
                        self.logger.warning(f"Failed reading cache file {c_file}: {e}")
                uncached_jobs.append(job)
                uncached_cache_paths[sc_id] = c_file
        else:
            uncached_jobs = list(scene_jobs)

        if not uncached_jobs:
            self.logger.info(
                f"Alignment cache hit for all {len(scene_jobs)} scenes; skipped aligner model loading."
            )
            return {
                "status": "success",
                "model_load_time_sec": 0.0,
                "device": device,
                "cached": True,
                "cache_hits": len(scene_jobs),
                "cache_misses": 0,
                "scenes": cached_scenes,
            }, 0.0

        aligner_config = {
            "model_path": model_path,
            "device": device,
            "dtype": dtype,
            "subprocess_python": stage_cfg.get("subprocess_python", ".venv-tts/Scripts/python.exe"),
            "worker_timeout_sec": stage_cfg.get("worker_timeout_sec", 300),
        }
        aligner = get_timeline_aligner(aligner_name, aligner_config)

        self.logger.info(
            f"Running alignment using {aligner_name} on {len(uncached_jobs)}/{len(scene_jobs)} scenes "
            f"({len(cached_scenes)} cached)..."
        )
        align_start = time.perf_counter()
        raw_res = aligner.align_scenes(uncached_jobs, aligner_config)
        align_wall_time = round(time.perf_counter() - align_start, 3)

        newly_aligned = raw_res.get("scenes", {})
        if use_cache:
            for sc_id, sc_data in newly_aligned.items():
                if sc_id in uncached_cache_paths:
                    if not isinstance(sc_data, dict) or "error" in sc_data:
                        continue
                    sc_words = sc_data.get("words", [])
                    if not sc_words or all(w.get("unplaced") for w in sc_words):
                        continue
                    try:
                        uncached_cache_paths[sc_id].write_text(
                            json.dumps(sc_data, indent=2), encoding="utf-8"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to write cache for scene '{sc_id}': {e}")

        all_scenes = dict(cached_scenes)
        all_scenes.update(newly_aligned)

        model_load_time_sec = raw_res.get("model_load_time_sec", 0.0)
        self.logger.info(
            f"Alignment finished in {align_wall_time:.2f}s "
            f"(model load: {model_load_time_sec:.2f}s, device: {raw_res.get('device')}, "
            f"cache hits: {len(cached_scenes)}, misses: {len(uncached_jobs)})"
        )
        if raw_res.get("device_fallback"):
            self.logger.warning("Aligner worker reported GPU device fallback to CPU.")

        merged_result = {
            "status": raw_res.get("status", "success"),
            "model_load_time_sec": model_load_time_sec,
            "device": raw_res.get("device", device),
            "device_fallback": raw_res.get("device_fallback", False),
            "cached": (len(cached_scenes) > 0 and len(uncached_jobs) == 0),
            "cache_hits": len(cached_scenes),
            "cache_misses": len(uncached_jobs),
            "scenes": all_scenes,
        }
        return merged_result, align_wall_time

    def build_scene_timeline(
        self,
        scene_jobs: list[SceneJob],
        align_result: dict[str, Any],
        pronunciations: dict[str, str],
        stage_cfg: dict[str, Any],
        prep_warnings: list[str],
        narration_duration: float,
        scene_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str], dict[str, Any], bool]:
        """Builds scenes output from alignment results, validating speech rates, word monotonicity, and sentence bounds."""
        low_score_threshold = float(stage_cfg.get("low_score_threshold", 0.35))
        max_unplaced_scene_ratio = float(stage_cfg.get("max_unplaced_scene_ratio", 0.15))
        max_unplaced_total_ratio = float(stage_cfg.get("max_unplaced_total_ratio", 0.05))
        min_wps = float(stage_cfg.get("min_wps", 1.2))
        max_wps = float(stage_cfg.get("max_wps", 6.0))

        scenes_output: list[dict[str, Any]] = []
        warnings: list[str] = list(prep_warnings)
        degraded_scenes: list[dict[str, Any]] = []
        low_confidence_scenes: list[str] = []

        total_words_count = 0
        total_unplaced_count = 0
        total_low_score_count = 0
        total_jobs = len(scene_jobs)

        for idx, job in enumerate(scene_jobs, 1):
            sc_id = job["id"]
            emit_office_event(
                self.run_dir,
                "substage_progress",
                stage="timeline",
                current=idx,
                total=total_jobs,
                message=f"Aligning phonemes for scene {idx}/{total_jobs} ({sc_id})",
                extra={"scene_id": sc_id},
            )
            sc_duration = job["duration_sec"]
            sc_start_offset = job["start_sec"]
            sc_end_offset = job["end_sec"]
            pad_before = job.get("pad_before", 0.4)
            pad_after = job.get("pad_after", 0.6)

            raw_scene_res = align_result.get("scenes", {}).get(sc_id, {})
            scene_words_rel = raw_scene_res.get("words", [])
            if not scene_words_rel and job.get("words"):
                scene_words_rel = [
                    {
                        "w": w.get("orig", ""),
                        "tts": w.get("tts", ""),
                        "tokens": w.get("tokens", []),
                        "start": None,
                        "end": None,
                        "score": 0.0,
                        "unplaced": True,
                    }
                    for w in job["words"]
                ]

            # Interpolate unplaced words proportionally by word weight with per-scene padding
            scene_words_rel, unplaced_in_scene = interpolate_unplaced_words(
                scene_words_rel,
                scene_duration=sc_duration,
                pad_before=pad_before,
                pad_after=pad_after,
            )

            n_words = len(scene_words_rel)
            total_words_count += n_words
            total_unplaced_count += unplaced_in_scene

            if n_words > 0:
                scene_unplaced_ratio = unplaced_in_scene / n_words
                if scene_unplaced_ratio > max_unplaced_scene_ratio:
                    warn_msg = (
                        f"Scene '{sc_id}': degraded alignment ({unplaced_in_scene}/{n_words} "
                        f"words unplaced, {scene_unplaced_ratio*100:.1f}% > {max_unplaced_scene_ratio*100:.1f}% limit); "
                        "fell back to weighted interpolation."
                    )
                    self.logger.warning(warn_msg)
                    warnings.append(warn_msg)
                    degraded_scenes.append({
                        "id": sc_id,
                        "unplaced_words": unplaced_in_scene,
                        "total_words": n_words,
                        "unplaced_ratio": round(scene_unplaced_ratio, 4),
                        "reason": (
                            f"unplaced_ratio {scene_unplaced_ratio:.3f} > "
                            f"{max_unplaced_scene_ratio:.3f}; fell back to weighted interpolation"
                        ),
                    })
                elif unplaced_in_scene > 0:
                    warn_msg = f"Scene '{sc_id}': {unplaced_in_scene}/{n_words} words unplaced and interpolated."
                    self.logger.warning(warn_msg)
                    warnings.append(warn_msg)

            # Speech rate check (words per second)
            if sc_duration > 0:
                wps = n_words / sc_duration
                if wps < min_wps:
                    warn_msg = f"Scene '{sc_id}': low speech rate {wps:.2f} wps (< {min_wps} wps)."
                    self.logger.warning(warn_msg)
                    warnings.append(warn_msg)
                elif wps > max_wps:
                    warn_msg = f"Scene '{sc_id}': high speech rate {wps:.2f} wps (> {max_wps} wps)."
                    self.logger.warning(warn_msg)
                    warnings.append(warn_msg)

            # Convert relative timestamps to absolute narration timestamps
            scene_words_abs: list[WordTiming] = []
            sc_low_score_words: list[tuple[str, float]] = []

            for w in scene_words_rel:
                abs_start = round(sc_start_offset + w["start"], 3)
                abs_end = round(sc_start_offset + w["end"], 3)
                score = round(float(w.get("score", 1.0)), 3)
                is_unplaced = bool(w.get("unplaced", False))
                is_low_conf = (not is_unplaced) and (score < low_score_threshold)

                if is_low_conf:
                    total_low_score_count += 1
                    sc_low_score_words.append((w["w"], score))

                scene_words_abs.append(WordTiming(
                    w=w["w"],
                    tts=w.get("tts", ""),
                    start=abs_start,
                    end=abs_end,
                    score=score,
                    unplaced=is_unplaced,
                    low_confidence=is_low_conf,
                ))

            if sc_low_score_words:
                low_confidence_scenes.append(sc_id)
                sc_scores = [s for _, s in sc_low_score_words]
                ex_words = ", ".join(repr(w) for w, _ in sc_low_score_words[:5])
                warn_msg = (
                    f"Scene '{sc_id}': {len(sc_low_score_words)} low-confidence words "
                    f"(min score: {min(sc_scores):.2f}, mean: {sum(sc_scores)/len(sc_scores):.2f}; "
                    f"examples: {ex_words})."
                )
                self.logger.warning(warn_msg)
                warnings.append(warn_msg)

            # Check word monotonicity
            for k in range(len(scene_words_abs) - 1):
                if scene_words_abs[k]["start"] > scene_words_abs[k + 1]["start"]:
                    raise ValueError(
                        f"Monotonicity violation in scene '{sc_id}': word '{scene_words_abs[k]['w']}' "
                        f"starts at {scene_words_abs[k]['start']}s > next '{scene_words_abs[k+1]['w']}' "
                        f"starts at {scene_words_abs[k+1]['start']}s."
                    )

            # Speech start and end bounds
            speech_start = scene_words_abs[0]["start"] if scene_words_abs else sc_start_offset
            speech_end = scene_words_abs[-1]["end"] if scene_words_abs else sc_end_offset

            # Split sentences and compute timestamps with token conservation check
            raw_sentences = split_into_sentences(job["narration"])
            per_sent_tokens = [prepare_scene_tokens(s, pronunciations) for s in raw_sentences]
            total_sent_tokens = sum(len(toks) for toks in per_sent_tokens)

            sentences_meta: list[dict[str, Any]] = []

            if not scene_words_abs:
                for sent_text in raw_sentences:
                    sentences_meta.append({
                        "text": sent_text,
                        "start": sc_start_offset,
                        "end": sc_end_offset,
                    })
            elif total_sent_tokens == len(scene_words_abs):
                # Exact 1-to-1 sequential mapping
                w_idx = 0
                for sent_text, sent_toks in zip(raw_sentences, per_sent_tokens):
                    n_sw = len(sent_toks)
                    if n_sw > 0 and w_idx < len(scene_words_abs):
                        s_first = scene_words_abs[w_idx]
                        s_last = scene_words_abs[min(len(scene_words_abs) - 1, w_idx + n_sw - 1)]
                        sentences_meta.append({
                            "text": sent_text,
                            "start": s_first["start"],
                            "end": s_last["end"],
                        })
                        w_idx += n_sw
                    else:
                        sentences_meta.append({
                            "text": sent_text,
                            "start": scene_words_abs[0]["start"],
                            "end": scene_words_abs[-1]["end"],
                        })
            else:
                # Token conservation mismatch: log warning and distribute timestamps proportionally
                warn_msg = (
                    f"Scene '{sc_id}': token conservation mismatch between scene words "
                    f"({len(scene_words_abs)}) and sentence tokens ({total_sent_tokens}); "
                    "distributing timestamps proportionally."
                )
                self.logger.warning(warn_msg)
                warnings.append(warn_msg)

                sent_weights = [max(1, len(toks)) for toks in per_sent_tokens]
                total_w = sum(sent_weights)
                cum_w = 0.0
                n_abs = len(scene_words_abs)

                for sent_text, sw in zip(raw_sentences, sent_weights):
                    start_frac = cum_w / total_w
                    cum_w += sw
                    end_frac = cum_w / total_w

                    w_first_idx = min(n_abs - 1, int(start_frac * n_abs))
                    w_last_idx = min(n_abs - 1, max(w_first_idx, int(end_frac * n_abs - 1e-4)))

                    s_first = scene_words_abs[w_first_idx]
                    s_last = scene_words_abs[w_last_idx]
                    sentences_meta.append({
                        "text": sent_text,
                        "start": s_first["start"],
                        "end": s_last["end"],
                    })

            # Determine scene type
            scene_type = "story"
            if sc_id in ("intro", "outro"):
                scene_type = sc_id

            scenes_output.append({
                "id": sc_id,
                "type": scene_type,
                "start": round(sc_start_offset, 3),
                "end": round(sc_end_offset, 3),
                "speech_start": round(speech_start, 3),
                "speech_end": round(speech_end, 3),
                "sentences": sentences_meta,
                "words": scene_words_abs,
            })

        # Reconcile final scene end to narration_duration exactly if not single-scene run
        if scenes_output and not scene_filter:
            scenes_output[-1]["end"] = round(narration_duration, 3)

        # Run-wide unplaced and low-score ratio check
        total_unplaced_ratio = (total_unplaced_count / total_words_count) if total_words_count > 0 else 0.0
        total_low_score_ratio = (total_low_score_count / total_words_count) if total_words_count > 0 else 0.0

        if total_words_count > 0 and total_unplaced_ratio > max_unplaced_total_ratio:
            warn_msg = (
                f"Run-wide unplaced words exceeded limit: {total_unplaced_count}/{total_words_count} "
                f"({total_unplaced_ratio*100:.1f}% > {max_unplaced_total_ratio*100:.1f}% limit); "
                "degraded to weighted interpolation."
            )
            self.logger.warning(warn_msg)
            warnings.append(warn_msg)

        max_low_score_total_ratio = float(stage_cfg.get("max_low_score_total_ratio", 0.05))
        if total_words_count > 0 and total_low_score_ratio >= max_low_score_total_ratio:
            warn_msg = (
                f"Run-wide low-confidence words exceeded limit: {total_low_score_count}/{total_words_count} "
                f"({total_low_score_ratio*100:.1f}% >= {max_low_score_total_ratio*100:.1f}% limit); alignment marked degraded."
            )
            self.logger.warning(warn_msg)
            warnings.append(warn_msg)

        is_degraded = (
            bool(degraded_scenes)
            or (total_words_count > 0 and total_unplaced_ratio > max_unplaced_total_ratio)
            or (total_words_count > 0 and total_low_score_ratio >= max_low_score_total_ratio)
        )

        counts = {
            "total_words": total_words_count,
            "total_unplaced": total_unplaced_count,
            "total_low_score": total_low_score_count,
            "unplaced_ratio": total_unplaced_ratio,
            "low_score_ratio": total_low_score_ratio,
        }

        return scenes_output, warnings, degraded_scenes, low_confidence_scenes, counts, is_degraded

    def build_captions(
        self,
        scenes_output: list[dict[str, Any]],
        stage_cfg: dict[str, Any],
    ) -> tuple[list[CaptionCue], dict[str, Any]]:
        """Chunks scene words into captions and audits their sync, reading speed, and constraints."""
        max_chars_per_line = int(stage_cfg.get("max_chars_per_line", 42))
        max_lines = int(stage_cfg.get("max_lines", 2))
        min_duration_sec = float(stage_cfg.get("min_duration_sec", 0.9))
        max_duration_sec = float(stage_cfg.get("max_duration_sec", 5.0))
        lead_in_sec = float(stage_cfg.get("lead_in_sec", 0.05))
        hold_after_sec = float(stage_cfg.get("hold_after_sec", 0.15))
        min_gap_sec = float(stage_cfg.get("min_gap_sec", 0.04))
        target_cps = float(stage_cfg.get("target_cps", 17.0))
        max_cps = float(stage_cfg.get("max_cps", 20.0))

        global_captions: list[CaptionCue] = []
        caption_id_counter = 1

        for sc in scenes_output:
            sc_captions, caption_id_counter = chunk_scene_words_into_captions(
                scene_words=sc.get("words", []),
                scene_id=sc["id"],
                scene_start=sc["start"],
                scene_end=sc["end"],
                start_caption_id=caption_id_counter,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                lead_in_sec=lead_in_sec,
                hold_after_sec=hold_after_sec,
                min_gap_sec=min_gap_sec,
                target_cps=target_cps,
                max_cps=max_cps,
            )
            global_captions.extend(sc_captions)

        caption_audit = audit_captions(
            captions=global_captions,
            scenes=scenes_output,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            target_cps=target_cps,
            max_cps=max_cps,
            late_threshold_ms=50.0,
        )

        # Attach per-scene quality metrics to scenes_output
        per_scene_audit = caption_audit.get("per_scene", {})
        for sc in scenes_output:
            sc_id = sc["id"]
            sc_audit = per_scene_audit.get(sc_id, {})
            sc_words = sc.get("words", [])
            sc_low_conf = sum(1 for w in sc_words if w.get("low_confidence"))
            sc_unplaced = sum(1 for w in sc_words if w.get("unplaced"))
            sc_min_score = min((float(w.get("score", 1.0)) for w in sc_words), default=1.0)

            sc["quality"] = {
                "words": len(sc_words),
                "low_confidence_words": sc_low_conf,
                "min_score": round(float(sc_min_score), 3),
                "unplaced": sc_unplaced,
                "caption_count": sc_audit.get("caption_count", 0),
                "max_late_ms": round(float(sc_audit.get("max_late_ms", 0.0)), 2),
                "max_cps": round(float(sc_audit.get("max_cps", 0.0)), 2),
            }

        return global_captions, caption_audit

    def write_artifacts(
        self,
        timeline_data: dict[str, Any],
        global_captions: list[CaptionCue],
        caption_audit: dict[str, Any],
        scene_filter: str | None,
        allow_partial_overwrite: bool,
        no_preview: bool,
        narration_file: Path,
        stage_cfg: dict[str, Any] | None = None,
    ) -> tuple[list[str], Path | None, list[str]]:
        """Validates contracts and writes canonical, partial, and legacy artifact files."""
        write_warnings: list[str] = []

        # Validate against schema contracts
        is_valid, val_err = validate_contract(timeline_data, self.schema_contract_path)
        if not is_valid:
            raise ValueError(f"Generated timeline does not conform to schema contract: {val_err}")

        captions_schema_path = Path("contracts/captions.schema.json")
        if captions_schema_path.exists():
            c_valid, c_err = validate_contract(global_captions, captions_schema_path)
            if not c_valid:
                raise ValueError(f"Generated captions do not conform to schema contract: {c_err}")

        # Format SRT content and validate syntax
        srt_content = build_srt_content(global_captions)
        srt_valid, srt_err = validate_srt_syntax(srt_content)
        if not srt_valid:
            raise ValueError(f"Generated captions.srt syntax validation failed: {srt_err}")

        paths = self.get_artifact_paths(scene_filter if not allow_partial_overwrite else None)

        if scene_filter and not allow_partial_overwrite:
            paths["partial_dir"].mkdir(parents=True, exist_ok=True)

            with open(paths["partial_json"], "w", encoding="utf-8") as f:
                json.dump(timeline_data, f, indent=2)

            with open(paths["partial_srt"], "w", encoding="utf-8") as f:
                f.write(srt_content)

            with open(paths["partial_audit"], "w", encoding="utf-8") as f:
                json.dump(caption_audit, f, indent=2)

            self.logger.info(
                f"Partial run for scene '{scene_filter}' written to {paths['partial_dir']} "
                "(canonical artifacts preserved)."
            )
            return [
                f"timeline/partial/{scene_filter}.json",
                f"timeline/partial/{scene_filter}.srt",
            ], None, write_warnings

        paths["timeline_dir"].mkdir(parents=True, exist_ok=True)

        with open(paths["timeline_json"], "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)

        with open(paths["captions_srt"], "w", encoding="utf-8") as f:
            f.write(srt_content)

        with open(paths["captions_json"], "w", encoding="utf-8") as f:
            json.dump(global_captions, f, indent=2)

        with open(paths["caption_audit_json"], "w", encoding="utf-8") as f:
            json.dump(caption_audit, f, indent=2)

        # Legacy copies at root run_dir for backwards compatibility
        shutil.copyfile(paths["timeline_json"], paths["root_timeline_json"])
        shutil.copyfile(paths["captions_srt"], paths["root_captions_srt"])

        output_files = [
            "timeline/timeline.json",
            "timeline/captions.srt",
            "timeline/captions.json",
            "timeline/caption_audit.json",
            "timeline.json",
            "captions.srt",
        ]

        preview_enabled = bool(stage_cfg.get("preview_enabled", True)) if stage_cfg else True
        preview_fps = int(stage_cfg.get("preview_fps", 10)) if stage_cfg else 10
        preview_preset = str(stage_cfg.get("preview_preset", "veryfast")) if stage_cfg else "veryfast"

        preview_mp4_path = paths["preview_mp4"]
        if not no_preview and preview_enabled:
            self.logger.info(
                f"Generating timeline preview video ({preview_fps} fps, preset {preview_preset})..."
            )
            try:
                render_stats = generate_timeline_preview_video(
                    narration_path=narration_file,
                    srt_path=paths["captions_srt"],
                    output_mp4=preview_mp4_path,
                    fps=preview_fps,
                    preset=preview_preset,
                )
                output_files.append("timeline/timeline_preview.mp4")
                self.logger.info(
                    f"Rendered timeline_preview.mp4 in {render_stats['render_time_sec']}s "
                    f"({render_stats['size_bytes'] / (1024*1024):.2f} MB)"
                )
            except Exception as e:
                self.logger.error(f"Failed to render preview video: {e}")
                write_warnings.append(f"Preview video rendering failed: {e}")

        return output_files, preview_mp4_path, write_warnings

    def collect_metrics(
        self,
        aligner_name: str,
        model_path: str,
        align_result: dict[str, Any],
        device: str,
        align_wall_time: float,
        counts: dict[str, Any],
        global_captions: list[CaptionCue],
        caption_audit: dict[str, Any],
        preview_rendered: bool,
        is_degraded: bool,
        low_confidence_scenes: list[str],
        degraded_scenes: list[dict[str, Any]],
        warnings: list[str],
        manifest: dict[str, Any] | None = None,
        save_manifest_to_disk: bool = True,
    ) -> dict[str, Any]:
        """Collects metrics, registers on self.last_metrics, and merges into manifest under lock."""
        audit_summary = caption_audit.get("summary", {})
        metrics = {
            "aligner": aligner_name,
            "model": model_path,
            "device": align_result.get("device", device),
            "model_load_time_sec": align_result.get("model_load_time_sec", 0.0),
            "alignment_wall_time_sec": align_wall_time,
            "cached": align_result.get("cached", False),
            "cache_hits": align_result.get("cache_hits", 0),
            "cache_misses": align_result.get("cache_misses", 0),
            "total_words": counts["total_words"],
            "unplaced_words": counts["total_unplaced"],
            "unplaced_ratio": round(counts["unplaced_ratio"], 4),
            "low_score_words": counts["total_low_score"],
            "low_score_ratio": round(counts["low_score_ratio"], 4),
            "caption_count": len(global_captions),
            "caption_audit_passed": audit_summary.get("passed", True),
            "caption_violations": audit_summary.get("violations_count", 0),
            "max_late_ms": audit_summary.get("max_late_ms", 0.0),
            "max_cps": audit_summary.get("max_cps", 0.0),
            "median_cps": audit_summary.get("median_cps", 0.0),
            "preview_video_rendered": preview_rendered,
            "degraded": is_degraded,
            "low_confidence_scenes": low_confidence_scenes,
            "degraded_scenes": degraded_scenes,
            "warnings": warnings,
        }
        self.last_metrics = metrics

        if manifest:
            manifest.setdefault("stages", {}).setdefault("timeline", {})["metrics"] = metrics
            if save_manifest_to_disk and (self.run_dir / "manifest.json").exists():
                save_manifest(manifest, self.run_dir / "manifest.json")

        return metrics

    def audit_existing_run(self) -> dict[str, Any]:
        """Audits captions of an existing run without re-aligning audio."""
        paths = self.get_artifact_paths()
        timeline_json_path = paths["timeline_json"]
        captions_json_path = paths["captions_json"]

        if not timeline_json_path.exists():
            raise FileNotFoundError(
                f"Cannot audit: {timeline_json_path} does not exist. Run timeline stage first."
            )

        timeline_data = json.loads(timeline_json_path.read_text(encoding="utf-8"))
        scenes = timeline_data.get("scenes", [])

        if captions_json_path.exists():
            captions = json.loads(captions_json_path.read_text(encoding="utf-8"))
        else:
            captions_srt_path = paths["captions_srt"]
            if not captions_srt_path.exists():
                raise FileNotFoundError(
                    f"Neither captions.json nor captions.srt exists in {paths['timeline_dir']}"
                )
            captions = parse_srt(captions_srt_path.read_text(encoding="utf-8"))

        stage_cfg = self._get_stage_config()
        caption_audit = audit_captions(
            captions=captions,
            scenes=scenes,
            max_chars_per_line=int(stage_cfg.get("max_chars_per_line", 42)),
            max_lines=int(stage_cfg.get("max_lines", 2)),
            min_duration_sec=float(stage_cfg.get("min_duration_sec", 0.9)),
            max_duration_sec=float(stage_cfg.get("max_duration_sec", 5.0)),
            target_cps=float(stage_cfg.get("target_cps", 17.0)),
            max_cps=float(stage_cfg.get("max_cps", 20.0)),
            late_threshold_ms=50.0,
        )
        return caption_audit

    def run(
        self,
        manifest: dict[str, Any] | None = None,
        aligner_override: str | None = None,
        scene_filter: str | None = None,
        no_preview: bool = False,
        allow_partial_overwrite: bool = False,
        save_manifest_to_disk: bool = True,
        no_cache: bool = False,
    ) -> list[str]:
        """Aligns narration audio with script text to produce timeline and subtitles."""
        self.logger.info(f"Starting Timeline Agent run in: {self.run_dir}")
        stage_cfg = self._get_stage_config()
        aligner_name = aligner_override or stage_cfg.get("aligner", "mms_fa")
        model_path = stage_cfg.get("model_path", "models/MMS_FA/model.pt")
        device = stage_cfg.get("device", "cuda:0")
        use_cache = bool(stage_cfg.get("use_cache", True)) and not no_cache

        # 1. Load inputs and check invariants
        (
            script_data,
            scene_durations,
            narration_duration,
            narration_sr,
            scene_jobs,
            prep_warnings,
            narration_file,
            pronunciations,
        ) = self.load_inputs(scene_filter=scene_filter)

        # 1.1 Multilingual check and language-driven aligner reporting
        language = script_data.get("language", "English")
        has_devanagari = any(contains_devanagari(job.get("narration", "")) for job in scene_jobs)
        uroman_avail = get_optional_uroman() is not None
        if has_devanagari or language.lower() not in ("en", "english"):
            self.logger.info(
                f"Language '{language}' detected (Devanagari: {has_devanagari}, uroman: {uroman_avail})."
            )
            if has_devanagari and not uroman_avail:
                msg = (
                    f"Language '{language}' has Devanagari characters but 'uroman' is not installed; "
                    "using fallback transliteration for MMS forced alignment."
                )
                self.logger.warning(msg)
                prep_warnings.append(msg)

        # 2. Execute forced alignment with two-level caching
        align_result, align_wall_time = self.align_scenes(
            scene_jobs=scene_jobs,
            aligner_name=aligner_name,
            stage_cfg=stage_cfg,
            pronunciations=pronunciations,
            use_cache=use_cache,
        )

        # 3. Post-process alignment: interpolation, offset conversion, and speech boundaries
        (
            scenes_output,
            warnings,
            degraded_scenes,
            low_confidence_scenes,
            counts,
            is_degraded,
        ) = self.build_scene_timeline(
            scene_jobs=scene_jobs,
            align_result=align_result,
            pronunciations=pronunciations,
            stage_cfg=stage_cfg,
            prep_warnings=prep_warnings,
            narration_duration=narration_duration,
            scene_filter=scene_filter,
        )

        # 4. Chunk words into captions and audit quality
        global_captions, caption_audit = self.build_captions(
            scenes_output=scenes_output,
            stage_cfg=stage_cfg,
        )
        caption_audit.setdefault("summary", {})["multilingual_status"] = {
            "language": language,
            "devanagari_detected": has_devanagari,
            "romanizer_available": uroman_avail,
        }

        # 5. Build Timeline Data payload
        timeline_data = {
            "audio_file": "audio/narration.wav",
            "duration_sec": round(narration_duration, 3),
            "sample_rate": narration_sr,
            "language": script_data.get("language", "English"),
            "aligner": {
                "name": aligner_name,
                "model": model_path,
                "device": align_result.get("device", device),
            },
            "warnings": warnings,
            "degraded": is_degraded,
            "low_confidence_scenes": low_confidence_scenes,
            "scenes": scenes_output,
        }

        # 6. Write outputs and optional preview video
        output_files, preview_mp4_path, write_warnings = self.write_artifacts(
            timeline_data=timeline_data,
            global_captions=global_captions,
            caption_audit=caption_audit,
            scene_filter=scene_filter,
            allow_partial_overwrite=allow_partial_overwrite,
            no_preview=no_preview,
            narration_file=narration_file,
            stage_cfg=stage_cfg,
        )
        warnings.extend(write_warnings)

        # If partial run, return immediately without overwriting canonical metrics
        if scene_filter and not allow_partial_overwrite:
            return output_files

        # 7. Collect and record metrics
        self.collect_metrics(
            aligner_name=aligner_name,
            model_path=model_path,
            align_result=align_result,
            device=device,
            align_wall_time=align_wall_time,
            counts=counts,
            global_captions=global_captions,
            caption_audit=caption_audit,
            preview_rendered=(preview_mp4_path.exists() if preview_mp4_path else False),
            is_degraded=is_degraded,
            low_confidence_scenes=low_confidence_scenes,
            degraded_scenes=degraded_scenes,
            warnings=warnings,
            manifest=manifest,
            save_manifest_to_disk=save_manifest_to_disk,
        )

        self.logger.info(
            f"Timeline Agent successfully generated {len(scenes_output)} scenes and "
            f"{len(global_captions)} captions."
        )
        return output_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Timeline Agent on a news run directory.")
    parser.add_argument("--date", default="2026-11-06", help="Run date folder in data/runs/YYYY-MM-DD")
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs directory")
    parser.add_argument("--config", default="config/pipeline.yaml", help="Path to pipeline.yaml")
    parser.add_argument("--aligner", default=None, choices=["mms_fa", "fake"], help="Override aligner")
    parser.add_argument("--dry-run", action="store_true", help="Run with FakeAligner and skip preview video")
    parser.add_argument("--scene", default=None, help="Align a single scene id only")
    parser.add_argument("--allow-partial-overwrite", action="store_true", help="Allow partial single-scene run to overwrite canonical timeline files")
    parser.add_argument("--no-preview", action="store_true", help="Skip timeline_preview.mp4 rendering")
    parser.add_argument("--audit", action="store_true", help="Audit caption quality of an existing run without re-aligning")
    parser.add_argument("--no-cache", action="store_true", help="Bypass alignment cache and force re-alignment")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.date
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    pipeline_cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            pipeline_cfg = yaml.safe_load(f)

    aligner_choice = "fake" if args.dry_run else args.aligner
    no_preview = args.no_preview or args.dry_run

    agent = TimelineAgent(run_dir, config=pipeline_cfg)

    if args.audit:
        try:
            audit_report = agent.audit_existing_run()
            summary = audit_report.get("summary", {})
            print("\n=== TIMELINE CAPTION AUDIT REPORT ===")
            print(f"Audit Passed:        {summary.get('passed')}")
            print(f"Total Captions:      {summary.get('total_captions')}")
            print(f"Total Violations:    {summary.get('violations_count')}")
            print(f"Late Starts (>50ms): {summary.get('late_50ms_count', summary.get('late_starts_count', 0))} (Max late: {summary.get('max_late_ms', 0.0)} ms)")
            print(f"Overlaps:            {summary.get('overlap_count', summary.get('overlaps_count', 0))}")
            print(f"Reading Speed (CPS): Median {summary.get('median_cps')}, Max {summary.get('max_cps')}")
            print(f"Duration Bounds:     Min Violations {summary.get('min_duration_violations', 0)}, Max Violations {summary.get('max_duration_violations', 0)}")
            print(f"Line Violations:     Length {summary.get('line_length_violations')}, Count {summary.get('line_count_violations')}")

            per_scene = audit_report.get("per_scene", {})
            if per_scene:
                print("\nPer-Scene Quality Breakdown:")
                for sc_id, sc_stat in per_scene.items():
                    print(
                        f" - {sc_id:12s}: {sc_stat.get('caption_count', 0):2d} cues, "
                        f"max CPS: {sc_stat.get('max_cps', 0.0):4.1f}, "
                        f"max late: {sc_stat.get('max_late_ms', 0.0):5.1f}ms, "
                        f"violations: {sc_stat.get('violations', 0)}"
                    )
            print("=====================================\n")
            return 0 if summary.get("passed", True) else 1
        except Exception as e:
            print(f"[FATAL] Caption audit failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else None

    try:
        out_files = agent.run(
            manifest=manifest,
            aligner_override=aligner_choice,
            scene_filter=args.scene,
            no_preview=no_preview,
            allow_partial_overwrite=args.allow_partial_overwrite,
            no_cache=args.no_cache,
        )
        is_valid, err = agent.validate(out_files)
        if not is_valid:
            print(f"[ERROR] Validation failed: {err}")
            return 1

        if manifest and manifest_path.exists():
            save_manifest(manifest, manifest_path)

        print(f"[SUCCESS] Timeline Agent completed successfully.")
        print(f"Artifacts generated:")
        for f in out_files:
            print(f" - {run_dir / f}")
        return 0
    except Exception as e:
        print(f"[FATAL] Timeline Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

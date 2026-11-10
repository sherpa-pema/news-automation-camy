"""Timeline Agent implementation.

Performs forced phonetic alignment between narration audio (per scene and master)
and script.json, producing word-level timestamps, caption chunks, and preview video.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.agents.base import BaseAgent
from src.agents.voice import load_pronunciations, split_into_sentences
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest
from src.utils.schema_validate import validate_contract
from src.utils.timeline import (
    FakeAligner as FakeAligner,
    MMSAligner as MMSAligner,
    build_srt_content,
    chunk_scene_words_into_captions,
    generate_timeline_preview_video,
    get_timeline_aligner,
    interpolate_unplaced_words,
    prepare_scene_tokens,
    validate_srt_syntax,
)
from src.utils.tts import read_wav_info


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

    def _get_stage_config(self) -> dict[str, Any]:
        """Extracts timeline stage configuration from pipeline.yaml."""
        if not self.config:
            return {}
        stages = self.config.get("stages", [])
        for st in stages:
            if st.get("id") == "timeline":
                return st
        return {}

    def run(
        self,
        manifest: dict[str, Any] | None = None,
        aligner_override: str | None = None,
        scene_filter: str | None = None,
        no_preview: bool = False,
    ) -> list[str]:
        """Aligns narration audio with script text to produce timeline and subtitles."""
        self.logger.info(f"Starting Timeline Agent run in: {self.run_dir}")

        stage_cfg = self._get_stage_config()
        aligner_name = aligner_override or stage_cfg.get("aligner", "mms_fa")
        model_path = stage_cfg.get("model_path", "models/MMS_FA/model.pt")
        device = stage_cfg.get("device", "cuda:0")
        dtype = stage_cfg.get("dtype", "float32")
        subprocess_python = stage_cfg.get("subprocess_python", ".venv-tts/Scripts/python.exe")
        worker_timeout_sec = stage_cfg.get("worker_timeout_sec", 300)
        low_score_threshold = float(stage_cfg.get("low_score_threshold", 0.35))
        max_unplaced_scene_ratio = float(stage_cfg.get("max_unplaced_scene_ratio", 0.15))
        max_unplaced_total_ratio = float(stage_cfg.get("max_unplaced_total_ratio", 0.05))
        min_wps = float(stage_cfg.get("min_wps", 1.2))
        max_wps = float(stage_cfg.get("max_wps", 6.0))

        # Captioning parameters
        max_chars_per_line = int(stage_cfg.get("max_chars_per_line", 42))
        min_duration_sec = float(stage_cfg.get("min_duration_sec", 0.9))
        max_duration_sec = float(stage_cfg.get("max_duration_sec", 5.0))
        lead_in_sec = float(stage_cfg.get("lead_in_sec", 0.05))
        hold_after_sec = float(stage_cfg.get("hold_after_sec", 0.15))
        min_gap_sec = float(stage_cfg.get("min_gap_sec", 0.04))

        # Paths
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

        # 2. Verify scene duration sum equals master narration duration within 20ms
        sum_scene_durations = sum(s["duration_sec"] for s in scene_durations)
        diff_ms = abs(sum_scene_durations - narration_duration) * 1000.0
        if diff_ms > 20.0:
            raise ValueError(
                f"Scene duration sum ({sum_scene_durations:.3f}s) does not match narration.wav "
                f"({narration_duration:.3f}s) within 20ms tolerance (diff: {diff_ms:.1f}ms)."
            )

        # Verify scene offsets tile audio with no gaps and no overlaps
        for idx in range(len(scene_durations) - 1):
            s_curr = scene_durations[idx]
            s_next = scene_durations[idx + 1]
            if abs(s_curr["end_sec"] - s_next["start_sec"]) > 0.001:
                raise ValueError(
                    f"Scene partition gap/overlap between '{s_curr['id']}' ({s_curr['end_sec']}s) "
                    f"and '{s_next['id']}' ({s_next['start_sec']}s)."
                )

        # 3. Load pronunciation dictionary
        pronunciations = load_pronunciations("config/pronunciations.yaml")

        # 4. Prepare scene alignment jobs
        scene_jobs: list[dict[str, Any]] = []
        script_scenes_map = {s["id"]: s for s in script_data.get("scenes", [])}

        for s_meta in scene_durations:
            sc_id = s_meta["id"]
            if scene_filter and sc_id != scene_filter:
                continue

            sc_script = script_scenes_map.get(sc_id)
            if not sc_script:
                raise ValueError(f"Scene '{sc_id}' found in audio metadata but missing in script.json")

            audio_path = self.run_dir / s_meta["file"]
            if not audio_path.exists():
                raise FileNotFoundError(f"Scene audio file not found: {audio_path}")

            token_words = prepare_scene_tokens(sc_script["narration"], pronunciations)
            scene_jobs.append({
                "id": sc_id,
                "audio_path": str(audio_path),
                "words": token_words,
                "duration_sec": s_meta["duration_sec"],
                "start_sec": s_meta["start_sec"],
                "end_sec": s_meta["end_sec"],
                "headline": sc_script.get("headline", ""),
                "narration": sc_script.get("narration", ""),
            })

        # 5. Execute forced alignment
        aligner_config = {
            "model_path": model_path,
            "device": device,
            "dtype": dtype,
            "subprocess_python": subprocess_python,
            "worker_timeout_sec": worker_timeout_sec,
        }
        aligner = get_timeline_aligner(aligner_name, aligner_config)

        self.logger.info(f"Running alignment using {aligner_name} on {len(scene_jobs)} scenes...")
        align_start = time.perf_counter()
        align_result = aligner.align_scenes(scene_jobs, aligner_config)
        align_wall_time = round(time.perf_counter() - align_start, 3)

        model_load_time_sec = align_result.get("model_load_time_sec", 0.0)
        self.logger.info(
            f"Alignment finished in {align_wall_time:.2f}s "
            f"(model load: {model_load_time_sec:.2f}s, device: {align_result.get('device')})"
        )

        # 6. Post-process alignment: interpolation, offset conversion, and sanity checks
        scenes_output: list[dict[str, Any]] = []
        global_captions: list[dict[str, Any]] = []
        caption_id_counter = 1
        warnings: list[str] = []

        total_words_count = 0
        total_unplaced_count = 0
        total_low_score_count = 0

        for job in scene_jobs:
            sc_id = job["id"]
            sc_duration = job["duration_sec"]
            sc_start_offset = job["start_sec"]
            sc_end_offset = job["end_sec"]

            raw_scene_res = align_result["scenes"].get(sc_id, {})
            scene_words_rel = raw_scene_res.get("words", [])

            # Interpolate unplaced words
            scene_words_rel, unplaced_in_scene = interpolate_unplaced_words(
                scene_words_rel,
                scene_duration=sc_duration,
            )

            n_words = len(scene_words_rel)
            total_words_count += n_words
            total_unplaced_count += unplaced_in_scene

            if n_words > 0:
                scene_unplaced_ratio = unplaced_in_scene / n_words
                if scene_unplaced_ratio > max_unplaced_scene_ratio:
                    raise ValueError(
                        f"Scene '{sc_id}' failed alignment: {unplaced_in_scene}/{n_words} "
                        f"words unplaced ({scene_unplaced_ratio*100:.1f}% > {max_unplaced_scene_ratio*100:.1f}% limit)."
                    )
                if unplaced_in_scene > 0:
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
            scene_words_abs: list[dict[str, Any]] = []
            for w in scene_words_rel:
                abs_start = round(sc_start_offset + w["start"], 3)
                abs_end = round(sc_start_offset + w["end"], 3)
                score = round(float(w.get("score", 1.0)), 3)

                if score < low_score_threshold and not w.get("unplaced"):
                    total_low_score_count += 1
                    warn_msg = (
                        f"Scene '{sc_id}': low alignment score {score:.2f} (< {low_score_threshold}) "
                        f"for word '{w['w']}'."
                    )
                    warnings.append(warn_msg)

                scene_words_abs.append({
                    "w": w["w"],
                    "tts": w.get("tts", ""),
                    "start": abs_start,
                    "end": abs_end,
                    "score": score,
                })

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

            # Split sentences and compute timestamps
            raw_sentences = split_into_sentences(job["narration"])
            sentences_meta: list[dict[str, Any]] = []
            word_idx = 0
            for sent_text in raw_sentences:
                sent_words = sent_text.strip().split()
                n_sw = len(sent_words)
                if word_idx < len(scene_words_abs):
                    s_first = scene_words_abs[word_idx]
                    s_last = scene_words_abs[min(len(scene_words_abs) - 1, word_idx + n_sw - 1)]
                    sentences_meta.append({
                        "text": sent_text,
                        "start": s_first["start"],
                        "end": s_last["end"],
                    })
                word_idx += n_sw

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

            # Chunk words into captions
            scene_captions, caption_id_counter = chunk_scene_words_into_captions(
                scene_words=scene_words_abs,
                scene_id=sc_id,
                scene_start=sc_start_offset,
                scene_end=sc_end_offset,
                start_caption_id=caption_id_counter,
                max_chars_per_line=max_chars_per_line,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                lead_in_sec=lead_in_sec,
                hold_after_sec=hold_after_sec,
                min_gap_sec=min_gap_sec,
            )
            global_captions.extend(scene_captions)

        # Run-wide unplaced ratio check
        if total_words_count > 0:
            total_unplaced_ratio = total_unplaced_count / total_words_count
            if total_unplaced_ratio > max_unplaced_total_ratio:
                raise ValueError(
                    f"Run-wide unplaced words limit exceeded: {total_unplaced_count}/{total_words_count} "
                    f"({total_unplaced_ratio*100:.1f}% > {max_unplaced_total_ratio*100:.1f}% limit)."
                )

        # 7. Build Timeline and Subtitle Data
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
            "scenes": scenes_output,
        }

        # Validate against schema contract
        is_valid, val_err = validate_contract(timeline_data, self.schema_contract_path)
        if not is_valid:
            raise ValueError(f"Generated timeline does not conform to schema contract: {val_err}")

        # Format SRT content and validate syntax
        srt_content = build_srt_content(global_captions)
        srt_valid, srt_err = validate_srt_syntax(srt_content)
        if not srt_valid:
            raise ValueError(f"Generated captions.srt syntax validation failed: {srt_err}")

        # 8. Write outputs
        timeline_dir = self.run_dir / "timeline"
        timeline_dir.mkdir(parents=True, exist_ok=True)

        timeline_json_path = timeline_dir / "timeline.json"
        captions_srt_path = timeline_dir / "captions.srt"
        captions_json_path = timeline_dir / "captions.json"
        preview_mp4_path = timeline_dir / "timeline_preview.mp4"

        with open(timeline_json_path, "w", encoding="utf-8") as f:
            json.dump(timeline_data, f, indent=2)

        with open(captions_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        with open(captions_json_path, "w", encoding="utf-8") as f:
            json.dump(global_captions, f, indent=2)

        # Also copy to root run_dir for backwards compatibility
        shutil.copyfile(timeline_json_path, self.run_dir / "timeline.json")
        shutil.copyfile(captions_srt_path, self.run_dir / "captions.srt")

        output_files = [
            "timeline/timeline.json",
            "timeline/captions.srt",
            "timeline/captions.json",
            "timeline.json",
            "captions.srt",
        ]

        # 9. Verification video generation
        render_stats = None
        if not no_preview:
            self.logger.info("Generating timeline preview video with burned-in subtitles...")
            try:
                render_stats = generate_timeline_preview_video(
                    narration_path=narration_file,
                    srt_path=captions_srt_path,
                    output_mp4=preview_mp4_path,
                )
                output_files.append("timeline/timeline_preview.mp4")
                self.logger.info(
                    f"Rendered timeline_preview.mp4 in {render_stats['render_time_sec']}s "
                    f"({render_stats['size_bytes'] / (1024*1024):.2f} MB)"
                )
            except Exception as e:
                self.logger.error(f"Failed to render preview video: {e}")
                warnings.append(f"Preview video rendering failed: {e}")

        # 10. Update manifest metrics
        if manifest:
            metrics = {
                "aligner": aligner_name,
                "model": model_path,
                "device": align_result.get("device", device),
                "model_load_time_sec": model_load_time_sec,
                "alignment_wall_time_sec": align_wall_time,
                "total_words": total_words_count,
                "unplaced_words": total_unplaced_count,
                "low_score_words": total_low_score_count,
                "caption_count": len(global_captions),
                "preview_video_rendered": preview_mp4_path.exists(),
                "warnings": warnings,
            }
            manifest.setdefault("stages", {}).setdefault("timeline", {})["metrics"] = metrics
            save_manifest(manifest, self.run_dir / "manifest.json")

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
    parser.add_argument("--no-preview", action="store_true", help="Skip timeline_preview.mp4 rendering")
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
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else None

    try:
        out_files = agent.run(
            manifest=manifest,
            aligner_override=aligner_choice,
            scene_filter=args.scene,
            no_preview=no_preview,
        )
        is_valid, err = agent.validate(out_files)
        if not is_valid:
            print(f"[ERROR] Validation failed: {err}")
            return 1

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

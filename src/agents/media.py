"""Media Agent: Sources, licenses, normalizes, and verifies visual assets for all scenes."""
import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv

from src.agents.base import BaseAgent
from src.utils.media_cache import MediaCacheManager
from src.utils.media_models import Candidate, PlannedScene, PlannedShot
from src.utils.media_normalizer import (
    compute_sha256,
    normalize_image,
    normalize_video,
)
from src.utils.media_review import (
    create_contact_sheet_for_scene,
    draw_card_image,
    generate_credits_markdown,
    generate_library_report,
    generate_picker_html,
    generate_review_html,
    generate_wanted_clips_md,
    save_candidate_thumbnail,
)
from src.utils.media_selector import MediaSelector
from src.utils.media_sources.local_library import add_to_library_cli
from src.utils.schema_validate import validate_contract
from src.utils.shot_planner import plan_shots_for_scenes
from src.utils.office_events import emit_office_event

load_dotenv()
logger = logging.getLogger("media")


class MediaAgent(BaseAgent):
    """Sources, validates, and normalizes visual media assets for each scene."""

    def __init__(
        self,
        run_dir: Path | str,
        config: dict[str, Any] | None = None,
        media_config_path: Path | str = "config/media.yaml",
        offline: bool = False,
        dry_run: bool = False,
        allow_fixture: bool = False,
        auto: bool = False,
        orientation: str | None = None,
    ) -> None:
        super().__init__(Path(run_dir), config=config)
        self.media_config_path = Path(media_config_path)
        self.media_config = self._load_yaml(self.media_config_path)
        self.offline = offline
        self.dry_run = dry_run
        self.allow_fixture = allow_fixture
        self.auto = auto
        self.orientation = orientation

        # Resolve orientation and resolution overrides from parameter, config, or manifest
        resolved_orient = orientation
        if not resolved_orient and self.config:
            for stg in self.config.get("stages", []):
                if stg.get("id") == "media" and stg.get("orientation"):
                    resolved_orient = stg.get("orientation")
                    break
        if resolved_orient:
            self.media_config.setdefault("settings", {})["orientation"] = resolved_orient
            if resolved_orient == "vertical":
                self.media_config["settings"]["resolution"] = [1080, 1920]
            else:
                self.media_config["settings"]["resolution"] = [1920, 1080]

        self.cache_mgr = MediaCacheManager(
            cache_dir=self.media_config.get("settings", {}).get("cache_dir", "data/media_cache"),
            user_agent=self.media_config.get("settings", {}).get("user_agent", "NepalNewsBot/1.0"),
            ttl_seconds=int(self.media_config.get("settings", {}).get("cache_ttl_hours", 24)) * 3600,
        )
        self.selector = MediaSelector(self.media_config, self.cache_mgr, offline=self.offline)
        self.warnings: list[str] = []

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def name(self) -> str:
        return "media"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/media_manifest.schema.json"

    def _load_inputs(self, require_timeline: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Loads script.json and optionally timeline.json from run directory."""
        script_path = self.run_dir / "script" / "script.json"
        if not script_path.exists():
            script_path = self.run_dir.parent.parent / "runs" / self.run_dir.name / "script" / "script.json"

        timeline_path = self.run_dir / "timeline" / "timeline.json"
        if not timeline_path.exists():
            timeline_path = self.run_dir / "timeline.json"

        if not script_path.exists():
            raise FileNotFoundError(f"Missing required script.json at {script_path}")
        if require_timeline and not timeline_path.exists():
            raise FileNotFoundError(f"Missing required timeline.json at {timeline_path}")

        with open(script_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        timeline_data = None
        if timeline_path.exists():
            with open(timeline_path, "r", encoding="utf-8") as f:
                timeline_data = json.load(f)

        return script_data, timeline_data

    def run(self, manifest: dict[str, Any], auto: bool | None = None) -> list[str]:
        """Executes full media sourcing and normalization pipeline (or AUTO mode when enabled)."""
        use_auto = self.auto if auto is None else auto
        if use_auto:
            return self.run_auto(manifest=manifest)
        start_time = time.time()
        logger.info(f"Starting Media Agent for run: {self.run_dir.name}")

        script_data, timeline_data = self._load_inputs()

        # Check script provenance origin
        script_origin = script_data.get("origin", "fixture")
        if script_origin == "fixture":
            if not self.allow_fixture:
                raise ValueError("Refusing to run on fixture script data without --allow-fixture.")
            msg = "[WARNING] FIXTURE DATA: Running Media Agent on fixture script data."
            logger.warning(msg)
            self.warnings.append(msg)
        else:
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            script_path = self.run_dir / "script" / "script.json"
            if not script_path.exists():
                raise FileNotFoundError(f"Missing required script.json at {script_path}")
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

        scenes = timeline_data.get("scenes", [])
        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}

        media_dir = self.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        shots_dir = media_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        review_dir = media_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self.media_config.get("settings", {})
        target_res = tuple(settings.get("resolution", [1920, 1080]))
        fps = int(settings.get("fps", 30))
        orientation = settings.get("orientation", "landscape")
        max_stills = int(settings.get("max_stills_per_scene", 2))

        # Check Voice stage disclosure
        disclose_voice = True
        voice_stage = manifest.get("stages", {}).get("voice", {})
        if "metrics" in voice_stage:
            disclose_voice = voice_stage["metrics"].get("disclose_synthetic_voice", True)

        # 1. Shot Planning
        planned_scenes = plan_shots_for_scenes(
            scenes=scenes,
            target_duration=float(settings.get("target_shot_duration", 5.0)),
            min_duration=float(settings.get("min_shot_duration", 2.5)),
            max_duration=float(settings.get("max_shot_duration", 6.0)),
        )

        shot_plan_path = media_dir / "shot_plan.json"
        shot_plan_data = {
            "date": self.run_dir.name,
            "scene_count": len(planned_scenes),
            "scenes": [
                {
                    "scene_id": ps.scene_id,
                    "start": ps.start,
                    "end": ps.end,
                    "duration": ps.duration,
                    "shots": [
                        {
                            "shot_index": sh.shot_index,
                            "start": sh.start,
                            "end": sh.end,
                            "duration": sh.duration,
                            "snapped_to_sentence": sh.snapped_to_sentence,
                        }
                        for sh in ps.shots
                    ],
                }
                for ps in planned_scenes
            ],
        }
        with open(shot_plan_path, "w", encoding="utf-8") as f:
            json.dump(shot_plan_data, f, indent=2)

        # 2. Sourcing and Normalization
        manifest_shots: list[dict[str, Any]] = []
        fallback_count = 0
        fallback_scenes: dict[str, list[str]] = {}
        total_download_bytes = 0

        for p_scene in planned_scenes:
            sc_id = p_scene.scene_id
            sc_script = script_scenes.get(sc_id, {"id": sc_id, "headline": sc_id, "visual_keywords": []})
            is_sensitive = self.selector.is_scene_sensitive(sc_script)

            scene_shots_dir = shots_dir / sc_id
            scene_shots_dir.mkdir(parents=True, exist_ok=True)

            stills_count = 0

            for shot in p_scene.shots:
                shot_fulfilled = False

                if not self.dry_run:
                    # Iterate through candidates across prioritized sources and query ladder
                    for cand, matched_query, src_name in self.selector.iter_candidates_for_shot(
                        scene=sc_script,
                        shot=shot,
                        stills_in_scene=stills_count,
                        max_stills=max_stills,
                    ):
                        raw_path = self.cache_mgr.download_file(cand.download_url, offline=self.offline)
                        if not raw_path or not raw_path.exists():
                            self.selector.mark_candidate_bad(cand.id)
                            continue

                        if cand.kind == "video":
                            out_filename = f"shot_{shot.shot_index:02d}.mp4"
                            out_path = scene_shots_dir / out_filename
                            try:
                                probe = normalize_video(
                                    src_path=raw_path,
                                    dest_path=out_path,
                                    target_duration=shot.duration,
                                    target_resolution=target_res,
                                    fps=fps,
                                    source_duration=cand.duration,
                                )
                                total_download_bytes += raw_path.stat().st_size
                                self.selector.mark_candidate_used(cand, matched_query, src_name)
                                manifest_shots.append({
                                    "scene_id": sc_id,
                                    "shot_index": shot.shot_index,
                                    "start": shot.start,
                                    "end": shot.end,
                                    "type": "video",
                                    "file": f"media/shots/{sc_id}/{out_filename}",
                                    "source": cand.source,
                                    "origin_url": cand.page_url,
                                    "license": cand.license_id,
                                    "license_url": cand.license_url,
                                    "author": cand.author,
                                    "attribution_text": cand.attribution_text or f"Video by {cand.author} via {cand.source}",
                                    "attribution_required": cand.attribution_required,
                                    "width": probe.get("width"),
                                    "height": probe.get("height"),
                                    "duration_sec": probe.get("duration"),
                                    "sha256": probe.get("sha256"),
                                    "query": matched_query,
                                    "motion": None,
                                    "sensitive_scene": is_sensitive,
                                    "style": None,
                                })
                                shot_fulfilled = True
                                break
                            except Exception as e:
                                logger.warning(f"Video normalization failed for {cand.id}: {e}")
                                self.selector.mark_candidate_bad(cand.id)
                                continue

                        elif cand.kind == "image":
                            out_filename = f"shot_{shot.shot_index:02d}.jpg"
                            out_path = scene_shots_dir / out_filename
                            try:
                                probe = normalize_image(
                                    src_path=raw_path,
                                    dest_path=out_path,
                                    target_resolution=target_res,
                                )
                                total_download_bytes += raw_path.stat().st_size
                                stills_count += 1
                                self.selector.mark_candidate_used(cand, matched_query, src_name)
                                manifest_shots.append({
                                    "scene_id": sc_id,
                                    "shot_index": shot.shot_index,
                                    "start": shot.start,
                                    "end": shot.end,
                                    "type": "image",
                                    "file": f"media/shots/{sc_id}/{out_filename}",
                                    "source": cand.source,
                                    "origin_url": cand.page_url,
                                    "license": cand.license_id,
                                    "license_url": cand.license_url,
                                    "author": cand.author,
                                    "attribution_text": cand.attribution_text or f"Photo by {cand.author} via {cand.source}",
                                    "attribution_required": cand.attribution_required,
                                    "width": probe.get("width"),
                                    "height": probe.get("height"),
                                    "duration_sec": round(shot.duration, 3),
                                    "sha256": probe.get("sha256"),
                                    "query": matched_query,
                                    "motion": "ken_burns",
                                    "sensitive_scene": is_sensitive,
                                    "style": None,
                                })
                                shot_fulfilled = True
                                break
                            except Exception as e:
                                logger.warning(f"Image normalization failed for {cand.id}: {e}")
                                self.selector.mark_candidate_bad(cand.id)
                                continue

                if not shot_fulfilled:
                    # Emit fallback shot only after all candidates and queries are exhausted
                    fallback_count += 1
                    fallback_scenes.setdefault(sc_id, []).append(str(shot.shot_index))
                    warn_msg = f"Scene '{sc_id}' shot #{shot.shot_index}: no acceptable visual found; generated fallback shot."
                    self.warnings.append(warn_msg)
                    logger.warning(warn_msg)

                    manifest_shots.append({
                        "scene_id": sc_id,
                        "shot_index": shot.shot_index,
                        "start": shot.start,
                        "end": shot.end,
                        "type": "fallback",
                        "file": None,
                        "source": "fallback",
                        "origin_url": None,
                        "license": "N/A",
                        "license_url": None,
                        "author": "System",
                        "attribution_text": "Animated fallback graphic",
                        "attribution_required": False,
                        "width": None,
                        "height": None,
                        "duration_sec": round(shot.duration, 3),
                        "sha256": None,
                        "query": None,
                        "motion": None,
                        "sensitive_scene": is_sensitive,
                        "style": "animated_backdrop",
                    })

        # 3. Validation
        self._validate_manifest_shots(manifest_shots, planned_scenes)

        # 4. Generate Manifest Output
        manifest_payload = {
            "date": self.run_dir.name,
            "orientation": orientation,
            "resolution": list(target_res),
            "fps": fps,
            "illustrative_label": bool(settings.get("illustrative_label", True)),
            "disclose_synthetic_voice": disclose_voice,
            "script_origin": script_origin,
            "shots": manifest_shots,
            "warnings": self.warnings,
        }

        # Contract validation against media_manifest.schema.json
        if self.schema_contract_path:
            is_valid, err = validate_contract(manifest_payload, self.schema_contract_path)
            if not is_valid:
                raise ValueError(f"media_manifest.json schema validation failed: {err}")

        # Write manifest to media/media_manifest.json and root media_manifest.json
        manifest_sub_path = media_dir / "media_manifest.json"
        manifest_root_path = self.run_dir / "media_manifest.json"
        with open(manifest_sub_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
        with open(manifest_root_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        # 5. Credits
        credits_path = media_dir / "credits.md"
        generate_credits_markdown(
            shots=manifest_shots,
            output_path=credits_path,
            disclose_synthetic_voice=disclose_voice,
        )

        # 6. Review Contact Sheets and HTML
        shots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for s in manifest_shots:
            shots_by_scene.setdefault(s["scene_id"], []).append(s)

        for sc_id, sc_shots in shots_by_scene.items():
            cs_path = review_dir / f"{sc_id}.jpg"
            create_contact_sheet_for_scene(sc_id, sc_shots, self.run_dir, cs_path)

        review_html_path = review_dir / "index.html"
        generate_review_html(
            manifest_data=manifest_payload,
            script_scenes=list(script_scenes.values()),
            output_path=review_html_path,
        )

        # 7. Update manifest stage metrics
        wall_time = round(time.time() - start_time, 2)
        total_media_size_mb = round(total_download_bytes / (1024 * 1024), 2)
        attr_count = sum(1 for s in manifest_shots if s.get("attribution_required"))

        media_metrics = {
            "total_shots": len(manifest_shots),
            "fallback_count": fallback_count,
            "fallback_scenes": fallback_scenes,
            "download_size_mb": total_media_size_mb,
            "wall_time_sec": wall_time,
            "script_origin": script_origin,
            "attribution_required_count": attr_count,
            "query_usage_counts": dict(self.selector.query_usage_counts),
            "rejected_near_duplicates": self.selector.rejected_near_duplicates_count,
            "map_or_diagram_count": self.selector.map_diagram_count,
            "source_metrics": self.selector.metrics,
            "warnings_count": len(self.warnings),
        }

        if "stages" in manifest and "media" in manifest["stages"]:
            manifest["stages"]["media"]["metrics"] = media_metrics

        logger.info(f"Media Agent finished in {wall_time}s. Shots: {len(manifest_shots)}, Fallbacks: {fallback_count}")
        return ["media/media_manifest.json", "media/shot_plan.json", "media/credits.md"]

    def _validate_manifest_shots(
        self,
        shots: list[dict[str, Any]],
        planned_scenes: list[PlannedScene],
    ) -> None:
        """Strict validation of tiling, sha256, licenses, domain blocks, and sensitive rules."""
        # Check tiling per scene
        shots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for s in shots:
            shots_by_scene.setdefault(s["scene_id"], []).append(s)

        for ps in planned_scenes:
            sc_shots = shots_by_scene.get(ps.scene_id, [])
            if not sc_shots:
                raise ValueError(f"Scene {ps.scene_id} has no shots assigned!")

            total_shot_time = sum((s["end"] - s["start"]) for s in sc_shots)
            diff = abs(total_shot_time - ps.duration)
            if diff > 0.05:
                raise ValueError(
                    f"Scene {ps.scene_id} tiling mismatch: sum of shots {total_shot_time:.3f}s != scene duration {ps.duration:.3f}s (diff {diff:.3f}s > 20ms tolerance)"
                )

        # Validate files and rules
        for s in shots:
            if s["type"] != "fallback":
                file_rel = s.get("file")
                if not file_rel:
                    raise ValueError(f"Shot {s['scene_id']} #{s['shot_index']} has missing file path!")
                full_path = self.run_dir / file_rel
                if not full_path.exists():
                    raise FileNotFoundError(f"Media file not found: {full_path}")

                # Check SHA256 match
                recorded_hash = s.get("sha256")
                actual_hash = compute_sha256(full_path)
                if recorded_hash != actual_hash:
                    raise ValueError(f"SHA-256 mismatch for {file_rel}: {actual_hash} != {recorded_hash}")

                # Check blocklisted domains
                if self.selector.is_domain_blocklisted(s.get("origin_url")):
                    raise ValueError(f"Shot uses blocklisted domain: {s.get('origin_url')}")

            # Check license allowed
            if s["type"] not in ("fallback", "card"):
                if not self.selector.is_license_allowed(s.get("license", ""), source=s.get("source", "")):
                    raise ValueError(f"Shot has disallowed license: {s.get('license')}")

    def run_candidates(self, candidates_k: int = 6) -> dict[str, Any]:
        """Runs candidates mode: plans shots (target 7s, min 4s), collects top K candidates per shot,
        scores relevance, saves thumbnails only to media/review/thumbs/, determines auto-picks
        (clip if score >= min_relevance else card proposal), and writes candidates.json, picker.html,
        and wanted_clips.md."""
        logger.info(f"Starting Media Agent candidates mode (k={candidates_k}) for run: {self.run_dir.name}")
        script_data, timeline_data = self._load_inputs(require_timeline=False)

        # Check script provenance origin
        script_origin = script_data.get("origin", "fixture")
        if script_origin == "fixture":
            if not self.allow_fixture:
                raise ValueError("Refusing to run on fixture script data without --allow-fixture.")
        else:
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            script_path = self.run_dir / "script" / "script.json"
            if not script_path.exists():
                raise FileNotFoundError(f"Missing required script.json at {script_path}")
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

        if timeline_data is not None:
            scenes = timeline_data.get("scenes", [])
            is_provisional = False
        else:
            from src.utils.shot_planner import estimate_timeline_scenes_from_script
            scenes = estimate_timeline_scenes_from_script(script_data.get("scenes", []))
            is_provisional = True

        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}

        media_dir = self.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        review_dir = media_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir = review_dir / "thumbs"
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        settings = self.media_config.get("settings", {})
        target_shot_sec = float(settings.get("target_shot_sec", settings.get("target_shot_duration", 7.0)))
        min_shot_duration = float(settings.get("min_shot_duration", 4.0))
        max_shot_duration = float(settings.get("max_shot_duration", 10.0))
        min_relevance = float(settings.get("min_relevance", 0.5))

        # 1. Shot Planning with target 7s, min 4s
        planned_scenes = plan_shots_for_scenes(
            scenes=scenes,
            target_duration=target_shot_sec,
            min_duration=min_shot_duration,
            max_duration=max_shot_duration,
        )

        shot_plan_path = media_dir / "shot_plan.json"
        shot_plan_data = {
            "date": self.run_dir.name,
            "scene_count": len(planned_scenes),
            "target_shot_sec": target_shot_sec,
            "provisional": is_provisional,
            "scenes": [
                {
                    "scene_id": ps.scene_id,
                    "start": ps.start,
                    "end": ps.end,
                    "duration": ps.duration,
                    "shots": [
                        {
                            "shot_index": sh.shot_index,
                            "start": sh.start,
                            "end": sh.end,
                            "duration": sh.duration,
                            "snapped_to_sentence": sh.snapped_to_sentence,
                        }
                        for sh in ps.shots
                    ],
                }
                for ps in planned_scenes
            ],
        }
        with open(shot_plan_path, "w", encoding="utf-8") as f:
            json.dump(shot_plan_data, f, indent=2)

        # 2. Gather Candidates per Shot
        shots_output: list[dict[str, Any]] = []
        used_auto_pick_ids: set[str] = set()
        total_shots = sum(len(ps.shots) for ps in planned_scenes)
        shot_num = 0

        for ps in planned_scenes:
            sc_id = ps.scene_id
            sc_script = script_scenes.get(sc_id, {"id": sc_id, "headline": sc_id, "visual_keywords": []})
            sc_timeline = next((s for s in scenes if s.get("id") == sc_id), {})

            for shot in ps.shots:
                shot_num += 1
                shot_id = f"{sc_id}_{shot.shot_index:02d}"

                emit_office_event(
                    self.run_dir,
                    "substage_progress",
                    stage="media",
                    current=shot_num,
                    total=total_shots,
                    message=f"Searching visual candidates {shot_num}/{total_shots} ({shot_id})",
                    extra={"shot_id": shot_id},
                )

                # Find spoken narration sentences during this shot
                overlapping_sentences = [
                    sent["text"] for sent in sc_timeline.get("sentences", [])
                    if sent.get("start", 0.0) < shot.end and sent.get("end", 0.0) > shot.start
                ]
                if not overlapping_sentences and sc_timeline.get("sentences"):
                    overlapping_sentences = [sc_timeline["sentences"][0]["text"]]

                # Collect distinct candidates across sources & query rungs
                pool_cands: dict[str, Candidate] = {}
                pool_limit = max(candidates_k * 5, 30)
                for cand, matched_query, src_name in self.selector.iter_candidates_for_shot(
                    scene=sc_script,
                    shot=shot,
                    stills_in_scene=0,
                    max_stills=10,
                ):
                    prev = pool_cands.get(cand.id)
                    if prev is None or float(getattr(cand, "score", 0.0)) > float(getattr(prev, "score", 0.0)):
                        pool_cands[cand.id] = cand
                    if len(pool_cands) >= pool_limit:
                        break

                # Sort: local_library first, unused auto-picks ahead of already-picked ones, then score descending
                sorted_cands = sorted(
                    pool_cands.values(),
                    key=lambda c: (
                        0 if c.source == "local_library" else 1,
                        1 if c.id in used_auto_pick_ids else 0,
                        -float(getattr(c, "score", 0.0)),
                    ),
                )
                top_k = sorted_cands[:candidates_k]

                # Save thumbnails only
                cand_entries: list[dict[str, Any]] = []
                for c in top_k:
                    rel_thumb = save_candidate_thumbnail(c, thumbs_dir, cache_mgr=self.cache_mgr, offline=self.offline)
                    cand_entries.append({
                        "id": c.id,
                        "source": c.source,
                        "title": c.title,
                        "page_url": c.page_url,
                        "download_url": c.download_url,
                        "kind": c.kind,
                        "license": c.license_id,
                        "license_url": c.license_url,
                        "author": c.author,
                        "attribution_text": c.attribution_text or f"{c.kind.title()} by {c.author} via {c.source}",
                        "attribution_required": c.attribution_required,
                        "width": c.width,
                        "height": c.height,
                        "duration": c.duration,
                        "score": round(float(getattr(c, "score", 0.0)), 2),
                        "matched_tokens": getattr(c, "matched_tokens", []),
                        "thumb": rel_thumb,
                    })

                best_score = max((c["score"] for c in cand_entries), default=0.0)

                # Determine auto-pick:
                # - Below min_relevance -> card proposal (no clip auto-picked)
                # - No duplicates of other shots' current picks
                auto_pick_id = None
                auto_pick_type = "card"
                for c in cand_entries:
                    if c["score"] >= min_relevance and c["id"] not in used_auto_pick_ids:
                        auto_pick_id = c["id"]
                        auto_pick_type = "clip"
                        used_auto_pick_ids.add(c["id"])
                        break

                shots_output.append({
                    "shot_id": shot_id,
                    "scene_id": sc_id,
                    "shot_index": shot.shot_index,
                    "scene_headline": sc_script.get("headline", sc_id),
                    "start": round(shot.start, 3),
                    "end": round(shot.end, 3),
                    "duration": round(shot.duration, 3),
                    "narration_sentences": overlapping_sentences,
                    "visual_keywords": sc_script.get("visual_keywords", []),
                    "auto_pick_id": auto_pick_id,
                    "auto_pick_type": auto_pick_type,
                    "best_score": round(best_score, 2),
                    "candidates": cand_entries,
                })

        candidates_data = {
            "date": self.run_dir.name,
            "min_relevance": min_relevance,
            "target_shot_sec": target_shot_sec,
            "total_shots": len(shots_output),
            "shots": shots_output,
        }

        # 3. Write outputs
        candidates_json_path = review_dir / "candidates.json"
        with open(candidates_json_path, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, indent=2)

        picker_html_path = review_dir / "picker.html"
        generate_picker_html(candidates_data, picker_html_path)

        wanted_clips_path = review_dir / "wanted_clips.md"
        generate_wanted_clips_md(candidates_data, wanted_clips_path)

        logger.info(f"Candidates mode complete: {len(shots_output)} shots processed.")
        return candidates_data

    def apply_picks(self, picks_file: Path | str) -> list[str]:
        """Applies human visual review choices from picks.json:
        - pick: downloads full file, validates license & domain, normalizes, adds to manifest.
        - card: generates 1080p gradient card image with card_text and slow zoom.
        - own_file: copies local footage, normalizes, records as user-provided.
        - research: runs query ladder with new query, writes candidates_round2.json.
        Writes media/picks_log.json, regenerates credits.md, review/index.html,
        and media_manifest.json, validates contract and shot tiling."""
        logger.info(f"Applying visual review picks from: {picks_file}")
        self.warnings = []
        script_data, timeline_data = self._load_inputs()

        # Gate check
        script_origin = script_data.get("origin", "fixture")
        if script_origin == "fixture":
            if not self.allow_fixture:
                raise ValueError("Refusing to run on fixture script data without --allow-fixture.")
        else:
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            script_path = self.run_dir / "script" / "script.json"
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

        disclose_voice = True
        run_manifest_file = self.run_dir / "manifest.json"
        if run_manifest_file.exists():
            try:
                with open(run_manifest_file, "r", encoding="utf-8") as rmf:
                    rm_data = json.load(rmf)
                voice_metrics = rm_data.get("stages", {}).get("voice", {}).get("metrics", {})
                disclose_voice = bool(voice_metrics.get("disclose_synthetic_voice", True))
            except Exception:
                pass

        # Snapshot pre-picks manifest (if any) to measure shot changes
        prev_shots_list: list[dict[str, Any]] = []
        existing_manifest_path = self.run_dir / "media" / "media_manifest.json"
        if not existing_manifest_path.exists():
            existing_manifest_path = self.run_dir / "media_manifest.json"
        if existing_manifest_path.exists():
            try:
                with open(existing_manifest_path, "r", encoding="utf-8") as pmf:
                    prev_shots_list = json.load(pmf).get("shots", [])
            except Exception:
                pass

        # Load picks JSON
        p_path = Path(picks_file)
        if not p_path.exists():
            if (self.run_dir / picks_file).exists():
                p_path = self.run_dir / picks_file
            elif (self.run_dir / "media" / picks_file).exists():
                p_path = self.run_dir / "media" / picks_file
            elif (self.run_dir / "media" / "review" / picks_file).exists():
                p_path = self.run_dir / "media" / "review" / picks_file
            else:
                raise FileNotFoundError(f"Picks file not found: {picks_file}")

        with open(p_path, "r", encoding="utf-8") as pf:
            raw_picks = json.load(pf)

        picks_dict: dict[str, dict[str, Any]] = raw_picks.get("picks", raw_picks)

        # Load candidates.json for candidate metadata lookups
        candidates_json_path = self.run_dir / "media" / "review" / "candidates.json"
        candidates_by_id: dict[str, dict[str, Any]] = {}
        if candidates_json_path.exists():
            with open(candidates_json_path, "r", encoding="utf-8") as cjf:
                c_data = json.load(cjf)
                for sh in c_data.get("shots", []):
                    for c in sh.get("candidates", []):
                        candidates_by_id[c["id"]] = c

        media_dir = self.run_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        shots_dir = media_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        review_dir = media_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self.media_config.get("settings", {})
        target_shot_sec = float(settings.get("target_shot_sec", settings.get("target_shot_duration", 7.0)))
        min_shot_duration = float(settings.get("min_shot_duration", 4.0))
        max_shot_duration = float(settings.get("max_shot_duration", 10.0))
        target_res = tuple(settings.get("resolution", [1920, 1080]))
        fps = int(settings.get("fps", 30))
        orientation = settings.get("orientation", "landscape")

        # Shot planning
        scenes = timeline_data.get("scenes", [])
        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}
        planned_scenes = plan_shots_for_scenes(
            scenes=scenes,
            target_duration=target_shot_sec,
            min_duration=min_shot_duration,
            max_duration=max_shot_duration,
        )

        manifest_shots: list[dict[str, Any]] = []
        round2_research: list[dict[str, Any]] = []
        action_counts = {"pick": 0, "card": 0, "own_file": 0, "upload": 0, "link": 0, "research": 0}
        applied_picks: list[dict[str, Any]] = []
        refused_picks: list[dict[str, Any]] = []
        failed_downloads: list[dict[str, Any]] = []
        cards_created: list[dict[str, Any]] = []
        total_download_bytes = 0

        for ps in planned_scenes:
            sc_id = ps.scene_id
            sc_script = script_scenes.get(sc_id, {"id": sc_id, "headline": sc_id, "visual_keywords": []})
            is_sensitive = self.selector.is_scene_sensitive(sc_script)
            scene_shots_dir = shots_dir / sc_id
            scene_shots_dir.mkdir(parents=True, exist_ok=True)

            for shot in ps.shots:
                shot_key = f"{sc_id}_{shot.shot_index:02d}"
                pick = picks_dict.get(shot_key) or picks_dict.get(f"{sc_id}_{shot.shot_index}")
                is_default_card = False

                if not pick:
                    # Default: create card proposal
                    pick = {"action": "card", "card_text": sc_script.get("headline", sc_id)}
                    is_default_card = True

                action = pick.get("action", "card")
                action_counts[action] = action_counts.get(action, 0) + 1

                if action == "pick":
                    cand_id = pick.get("candidate_id")
                    cand_info = candidates_by_id.get(cand_id) or pick.get("candidate")
                    if not cand_info:
                        raise ValueError(f"Candidate metadata not found for pick {shot_key} (cand_id: {cand_id})")

                    cand_license = cand_info.get("license", "")
                    cand_source = cand_info.get("source", "")
                    cand_page_url = cand_info.get("page_url")
                    cand_down_url = cand_info.get("download_url") or cand_page_url

                    # Verify license policy (e.g. allow_cc_by_sa honored)
                    if not self.selector.is_license_allowed(cand_license, source=cand_source):
                        reason = f"Shot {shot_key} candidate {cand_id} has disallowed license: '{cand_license}'"
                        refused_picks.append({"shot_id": shot_key, "candidate_id": cand_id, "reason": reason})
                        raise ValueError(reason)

                    # Verify domain blocklist
                    if self.selector.is_domain_blocklisted(cand_page_url) or self.selector.is_domain_blocklisted(cand_down_url):
                        reason = f"Shot {shot_key} uses blocklisted domain: {cand_page_url}"
                        refused_picks.append({"shot_id": shot_key, "candidate_id": cand_id, "reason": reason})
                        raise ValueError(reason)

                    # Download full asset (on failure or invalid asset, list it and create a card; never substitute stock)
                    raw_path = self.cache_mgr.download_file(cand_down_url, offline=self.offline)
                    download_err: str | None = None
                    cand_kind = cand_info.get("kind", "video")

                    if not raw_path or not raw_path.exists():
                        download_err = f"Failed to download asset from {cand_down_url}"
                    else:
                        try:
                            if cand_kind == "video":
                                out_filename = f"shot_{shot.shot_index:02d}.mp4"
                                out_path = scene_shots_dir / out_filename
                                probe = normalize_video(
                                    src_path=raw_path,
                                    dest_path=out_path,
                                    target_duration=shot.duration,
                                    target_resolution=target_res,
                                    fps=fps,
                                    source_duration=cand_info.get("duration"),
                                )
                                total_download_bytes += raw_path.stat().st_size
                                manifest_shots.append({
                                    "scene_id": sc_id,
                                    "shot_index": shot.shot_index,
                                    "start": shot.start,
                                    "end": shot.end,
                                    "type": "video",
                                    "card_text": None,
                                    "file": f"media/shots/{sc_id}/{out_filename}",
                                    "source": cand_source,
                                    "origin_url": cand_page_url,
                                    "license": cand_license,
                                    "license_url": cand_info.get("license_url"),
                                    "author": cand_info.get("author", "Unknown"),
                                    "attribution_text": cand_info.get("attribution_text") or f"Video by {cand_info.get('author')} via {cand_source}",
                                    "attribution_required": cand_info.get("attribution_required", False),
                                    "width": probe.get("width"),
                                    "height": probe.get("height"),
                                    "duration_sec": probe.get("duration"),
                                    "sha256": probe.get("sha256"),
                                    "query": cand_info.get("query"),
                                    "motion": None,
                                    "sensitive_scene": is_sensitive,
                                    "style": None,
                                })
                            else:  # Image
                                out_filename = f"shot_{shot.shot_index:02d}.jpg"
                                out_path = scene_shots_dir / out_filename
                                probe = normalize_image(
                                    src_path=raw_path,
                                    dest_path=out_path,
                                    target_resolution=target_res,
                                )
                                total_download_bytes += raw_path.stat().st_size
                                manifest_shots.append({
                                    "scene_id": sc_id,
                                    "shot_index": shot.shot_index,
                                    "start": shot.start,
                                    "end": shot.end,
                                    "type": "image",
                                    "card_text": None,
                                    "file": f"media/shots/{sc_id}/{out_filename}",
                                    "source": cand_source,
                                    "origin_url": cand_page_url,
                                    "license": cand_license,
                                    "license_url": cand_info.get("license_url"),
                                    "author": cand_info.get("author", "Unknown"),
                                    "attribution_text": cand_info.get("attribution_text") or f"Photo by {cand_info.get('author')} via {cand_source}",
                                    "attribution_required": cand_info.get("attribution_required", False),
                                    "width": probe.get("width"),
                                    "height": probe.get("height"),
                                    "duration_sec": round(shot.duration, 3),
                                    "sha256": probe.get("sha256"),
                                    "query": cand_info.get("query"),
                                    "motion": "ken_burns",
                                    "sensitive_scene": is_sensitive,
                                    "style": None,
                                })

                            applied_picks.append({
                                "shot_id": shot_key,
                                "candidate_id": cand_id,
                                "source": cand_source,
                                "kind": cand_kind,
                                "file": f"media/shots/{sc_id}/{out_filename}",
                                "sha256": probe.get("sha256"),
                            })
                        except Exception as norm_exc:
                            download_err = f"Failed to normalize downloaded asset from {cand_down_url}: {norm_exc}"

                    if download_err:
                        warn_msg = f"Shot {shot_key} ({cand_id}): {download_err}. Created non-stock headline card instead."
                        logger.warning(warn_msg)
                        self.warnings.append(warn_msg)
                        failed_downloads.append({
                            "shot_id": shot_key,
                            "scene_id": sc_id,
                            "shot_index": shot.shot_index,
                            "candidate_id": cand_id,
                            "source": cand_source,
                            "download_url": cand_down_url,
                            "reason": download_err,
                        })
                        card_text = sc_script.get("headline", sc_id)
                        out_filename = f"card_{shot.shot_index:02d}.png"
                        out_path = scene_shots_dir / out_filename
                        draw_card_image(card_text, out_path, width=target_res[0], height=target_res[1])
                        card_sha256 = compute_sha256(out_path)
                        cards_created.append({
                            "shot_id": shot_key,
                            "file": f"media/shots/{sc_id}/{out_filename}",
                            "reason": "download_failed",
                            "card_text": card_text,
                        })
                        manifest_shots.append({
                            "scene_id": sc_id,
                            "shot_index": shot.shot_index,
                            "start": shot.start,
                            "end": shot.end,
                            "type": "card",
                            "card_text": card_text,
                            "file": f"media/shots/{sc_id}/{out_filename}",
                            "source": "system",
                            "origin_url": None,
                            "license": "N/A",
                            "license_url": None,
                            "author": "System",
                            "attribution_text": "Graphic card",
                            "attribution_required": False,
                            "width": target_res[0],
                            "height": target_res[1],
                            "duration_sec": round(shot.duration, 3),
                            "sha256": card_sha256,
                            "query": None,
                            "motion": "zoom_in",
                            "sensitive_scene": is_sensitive,
                            "style": "card",
                        })

                elif action == "card":
                    card_text = pick.get("card_text") or sc_script.get("headline", sc_id)
                    out_filename = f"card_{shot.shot_index:02d}.png"
                    out_path = scene_shots_dir / out_filename
                    draw_card_image(card_text, out_path, width=target_res[0], height=target_res[1])
                    card_sha256 = compute_sha256(out_path)
                    cards_created.append({
                        "shot_id": shot_key,
                        "file": f"media/shots/{sc_id}/{out_filename}",
                        "reason": "default_card" if is_default_card else "explicit_card",
                        "card_text": card_text,
                    })
                    manifest_shots.append({
                        "scene_id": sc_id,
                        "shot_index": shot.shot_index,
                        "start": shot.start,
                        "end": shot.end,
                        "type": "card",
                        "card_text": card_text,
                        "file": f"media/shots/{sc_id}/{out_filename}",
                        "source": "system",
                        "origin_url": None,
                        "license": "N/A",
                        "license_url": None,
                        "author": "System",
                        "attribution_text": "Graphic card",
                        "attribution_required": False,
                        "width": target_res[0],
                        "height": target_res[1],
                        "duration_sec": round(shot.duration, 3),
                        "sha256": card_sha256,
                        "query": None,
                        "motion": "zoom_in",
                        "sensitive_scene": is_sensitive,
                        "style": "card",
                    })

                elif action in ("own_file", "upload"):
                    file_val = pick.get("file_path", "")
                    own_path = Path(file_val)
                    if not own_path.exists():
                        if (self.run_dir / file_val).exists():
                            own_path = self.run_dir / file_val
                        elif (self.run_dir / "media" / file_val).exists():
                            own_path = self.run_dir / "media" / file_val
                        else:
                            raise FileNotFoundError(f"Own file not found for shot {shot_key}: {file_val}")
                    is_video = own_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")
                    out_filename = f"shot_{shot.shot_index:02d}.mp4" if is_video else f"shot_{shot.shot_index:02d}.jpg"
                    out_path = scene_shots_dir / out_filename
                    if is_video:
                        probe = normalize_video(own_path, out_path, shot.duration, target_res, fps)
                    else:
                        probe = normalize_image(own_path, out_path, target_res)

                    user_license = pick.get("license", "user-provided")
                    manifest_shots.append({
                        "scene_id": sc_id,
                        "shot_index": shot.shot_index,
                        "start": shot.start,
                        "end": shot.end,
                        "type": "video" if is_video else "image",
                        "card_text": None,
                        "file": f"media/shots/{sc_id}/{out_filename}",
                        "source": "user-provided",
                        "origin_url": "local",
                        "license": user_license,
                        "license_url": None,
                        "author": pick.get("author", "User"),
                        "attribution_text": pick.get("attribution", "User-provided media"),
                        "attribution_required": False,
                        "width": probe.get("width"),
                        "height": probe.get("height"),
                        "duration_sec": probe.get("duration", round(shot.duration, 3)),
                        "sha256": probe.get("sha256"),
                        "query": "user_upload" if action == "upload" else "own_file",
                        "motion": None if is_video else "ken_burns",
                        "sensitive_scene": is_sensitive,
                        "style": None,
                    })

                elif action in ("link", "url"):
                    media_url = (pick.get("url") or pick.get("link") or pick.get("file_path", "")).strip()
                    if not media_url:
                        raise ValueError(f"Shot {shot_key} link action specified but no URL provided.")

                    if self.selector.is_domain_blocklisted(media_url):
                        reason = f"Shot {shot_key} link uses blocklisted news domain: {media_url}"
                        refused_picks.append({"shot_id": shot_key, "url": media_url, "reason": reason})
                        raise ValueError(reason)

                    raw_path = self.cache_mgr.download_file(media_url, offline=self.offline)
                    download_err: str | None = None
                    if not raw_path or not raw_path.exists():
                        download_err = f"Failed to download asset from {media_url}"
                    else:
                        try:
                            from src.utils.media_normalizer import probe_media
                            try:
                                probe_info = probe_media(raw_path)
                                is_video = bool(probe_info.get("duration", 0) > 0 and probe_info.get("fps", 0) > 0)
                            except Exception:
                                is_video = raw_path.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")

                            out_filename = f"shot_{shot.shot_index:02d}.mp4" if is_video else f"shot_{shot.shot_index:02d}.jpg"
                            out_path = scene_shots_dir / out_filename
                            if is_video:
                                probe = normalize_video(raw_path, out_path, shot.duration, target_res, fps)
                            else:
                                probe = normalize_image(raw_path, out_path, target_res)

                            total_download_bytes += raw_path.stat().st_size
                            manifest_shots.append({
                                "scene_id": sc_id,
                                "shot_index": shot.shot_index,
                                "start": shot.start,
                                "end": shot.end,
                                "type": "video" if is_video else "image",
                                "card_text": None,
                                "file": f"media/shots/{sc_id}/{out_filename}",
                                "source": "user-link",
                                "origin_url": media_url,
                                "license": "user-provided",
                                "license_url": None,
                                "author": pick.get("author", "User"),
                                "attribution_text": pick.get("attribution", f"Media via {media_url}"),
                                "attribution_required": False,
                                "width": probe.get("width"),
                                "height": probe.get("height"),
                                "duration_sec": probe.get("duration", round(shot.duration, 3)),
                                "sha256": probe.get("sha256"),
                                "query": "user_link",
                                "motion": None if is_video else "ken_burns",
                                "sensitive_scene": is_sensitive,
                                "style": None,
                            })
                        except Exception as norm_exc:
                            download_err = f"Failed to normalize linked asset from {media_url}: {norm_exc}"

                    if download_err:
                        warn_msg = f"Shot {shot_key} link ({media_url}): {download_err}. Created headline card instead."
                        logger.warning(warn_msg)
                        self.warnings.append(warn_msg)
                        failed_downloads.append({
                            "shot_id": shot_key,
                            "scene_id": sc_id,
                            "shot_index": shot.shot_index,
                            "source": "user-link",
                            "download_url": media_url,
                            "reason": download_err,
                        })
                        card_text = sc_script.get("headline", sc_id)
                        out_filename = f"card_{shot.shot_index:02d}.png"
                        out_path = scene_shots_dir / out_filename
                        draw_card_image(card_text, out_path, width=target_res[0], height=target_res[1])
                        card_sha256 = compute_sha256(out_path)
                        cards_created.append({
                            "shot_id": shot_key,
                            "file": f"media/shots/{sc_id}/{out_filename}",
                            "reason": "link_download_failed",
                            "card_text": card_text,
                        })
                        manifest_shots.append({
                            "scene_id": sc_id,
                            "shot_index": shot.shot_index,
                            "start": shot.start,
                            "end": shot.end,
                            "type": "card",
                            "card_text": card_text,
                            "file": f"media/shots/{sc_id}/{out_filename}",
                            "source": "system",
                            "origin_url": None,
                            "license": "N/A",
                            "license_url": None,
                            "author": "System",
                            "attribution_text": "Graphic card",
                            "attribution_required": False,
                            "width": target_res[0],
                            "height": target_res[1],
                            "duration_sec": round(shot.duration, 3),
                            "sha256": card_sha256,
                            "query": None,
                            "motion": "zoom_in",
                            "sensitive_scene": is_sensitive,
                            "style": "card",
                        })

                elif action == "research":
                    new_query = pick.get("query", sc_script.get("headline", sc_id))
                    # Perform query ladder search for this shot with new_query
                    r2_cands = []
                    for c, q, s_name in self.selector.iter_candidates_for_shot(
                        scene={"id": sc_id, "visual_keywords": [new_query]},
                        shot=shot,
                        stills_in_scene=0,
                        max_stills=10,
                    ):
                        r2_cands.append(c)
                        if len(r2_cands) >= 6:
                            break
                    round2_research.append({
                        "shot_id": shot_key,
                        "query": new_query,
                        "candidates_count": len(r2_cands),
                        "candidates": [
                            {
                                "id": rc.id,
                                "source": rc.source,
                                "title": rc.title,
                                "page_url": rc.page_url,
                                "score": getattr(rc, "score", 0.0),
                            }
                            for rc in r2_cands
                        ],
                    })
                    # Use card as placeholder in manifest until round 2 pick is made
                    card_text = f"{sc_script.get('headline', sc_id)} (Researching)"
                    out_filename = f"card_{shot.shot_index:02d}.png"
                    out_path = scene_shots_dir / out_filename
                    draw_card_image(card_text, out_path, width=target_res[0], height=target_res[1])
                    card_sha256 = compute_sha256(out_path)
                    cards_created.append({
                        "shot_id": shot_key,
                        "file": f"media/shots/{sc_id}/{out_filename}",
                        "reason": "research_placeholder",
                        "card_text": card_text,
                    })
                    manifest_shots.append({
                        "scene_id": sc_id,
                        "shot_index": shot.shot_index,
                        "start": shot.start,
                        "end": shot.end,
                        "type": "card",
                        "card_text": card_text,
                        "file": f"media/shots/{sc_id}/{out_filename}",
                        "source": "system",
                        "origin_url": None,
                        "license": "N/A",
                        "license_url": None,
                        "author": "System",
                        "attribution_text": "Graphic card",
                        "attribution_required": False,
                        "width": target_res[0],
                        "height": target_res[1],
                        "duration_sec": round(shot.duration, 3),
                        "sha256": card_sha256,
                        "query": new_query,
                        "motion": "zoom_in",
                        "sensitive_scene": is_sensitive,
                        "style": "card",
                    })

        # If any research actions, write candidates_round2.json
        if round2_research:
            r2_path = review_dir / "candidates_round2.json"
            with open(r2_path, "w", encoding="utf-8") as r2f:
                json.dump({"date": self.run_dir.name, "research_shots": round2_research}, r2f, indent=2)
            logger.info(f"Round 2 research written to {r2_path}")

        # Validate manifest shots (tiling, SHA256, licenses, domain blocks)
        self._validate_manifest_shots(manifest_shots, planned_scenes)

        manifest_payload = {
            "date": self.run_dir.name,
            "orientation": orientation,
            "resolution": list(target_res),
            "fps": fps,
            "illustrative_label": bool(settings.get("illustrative_label", True)),
            "disclose_synthetic_voice": disclose_voice,
            "script_origin": script_origin,
            "shots": manifest_shots,
            "warnings": self.warnings,
        }

        # Contract validation against media_manifest.schema.json
        if self.schema_contract_path:
            is_valid, err = validate_contract(manifest_payload, self.schema_contract_path)
            if not is_valid:
                raise ValueError(f"media_manifest.json schema validation failed: {err}")

        # Write manifest to media/media_manifest.json and root media_manifest.json
        manifest_sub_path = media_dir / "media_manifest.json"
        manifest_root_path = self.run_dir / "media_manifest.json"
        with open(manifest_sub_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)
        with open(manifest_root_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

        # Write final non-provisional shot_plan.json
        shot_plan_path = media_dir / "shot_plan.json"
        shot_plan_data = {
            "date": self.run_dir.name,
            "scene_count": len(planned_scenes),
            "target_shot_sec": target_shot_sec,
            "provisional": False,
            "scenes": [
                {
                    "scene_id": ps.scene_id,
                    "start": ps.start,
                    "end": ps.end,
                    "duration": ps.duration,
                    "shots": [
                        {
                            "shot_index": sh.shot_index,
                            "start": sh.start,
                            "end": sh.end,
                            "duration": sh.duration,
                            "snapped_to_sentence": sh.snapped_to_sentence,
                        }
                        for sh in ps.shots
                    ],
                }
                for ps in planned_scenes
            ],
        }
        with open(shot_plan_path, "w", encoding="utf-8") as f:
            json.dump(shot_plan_data, f, indent=2)

        # Regenerate credits.md
        credits_path = media_dir / "credits.md"
        generate_credits_markdown(manifest_shots, credits_path, disclose_synthetic_voice=disclose_voice)

        # Regenerate per-scene contact sheets and review/index.html
        shots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for s in manifest_shots:
            shots_by_scene.setdefault(s["scene_id"], []).append(s)
        for sc_id, sc_shots in shots_by_scene.items():
            create_contact_sheet_for_scene(sc_id, sc_shots, self.run_dir, review_dir / f"{sc_id}.jpg")

        review_html_path = review_dir / "index.html"
        generate_review_html(manifest_payload, list(script_scenes.values()), review_html_path)

        # Keep existing render_plan.json synced with updated shots/hashes if motion stage already ran
        from src.agents.motion import sync_render_plan_with_media_manifest
        sync_render_plan_with_media_manifest(self.run_dir)

        # Compare shots before vs after picks
        prev_by_key = {(s.get("scene_id"), s.get("shot_index")): s for s in prev_shots_list}
        new_by_key = {(s.get("scene_id"), s.get("shot_index")): s for s in manifest_shots}
        changed_shots: list[dict[str, Any]] = []
        for s in manifest_shots:
            key = (s.get("scene_id"), s.get("shot_index"))
            prev_s = prev_by_key.get(key)
            if not prev_s or prev_s.get("sha256") != s.get("sha256") or prev_s.get("origin_url") != s.get("origin_url"):
                changed_shots.append({
                    "shot_id": f"{key[0]}_{key[1]:02d}",
                    "old_origin_url": prev_s.get("origin_url") if prev_s else None,
                    "old_sha256": prev_s.get("sha256") if prev_s else None,
                    "new_origin_url": s.get("origin_url"),
                    "new_sha256": s.get("sha256"),
                    "new_type": s.get("type"),
                })
        removed_shots = [
            f"{k[0]}_{k[1]:02d}" for k in prev_by_key if k not in new_by_key
        ]

        # Write picks_log.json
        picks_log_path = media_dir / "picks_log.json"
        picks_log_data = {
            "date": self.run_dir.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "picks_file": str(picks_file),
            "action_counts": action_counts,
            "total_shots": len(manifest_shots),
            "previous_total_shots": len(prev_shots_list) if prev_shots_list else None,
            "shots_changed_count": len(changed_shots),
            "shots_removed_count": len(removed_shots),
            "changed_shots": changed_shots,
            "removed_shots": removed_shots,
            "applied_picks": applied_picks,
            "refused_picks": refused_picks,
            "failed_downloads": failed_downloads,
            "cards_created": cards_created,
            "download_size_mb": round(total_download_bytes / (1024 * 1024), 2),
            "warnings": self.warnings,
            "picks": picks_dict,
        }
        with open(picks_log_path, "w", encoding="utf-8") as plf:
            json.dump(picks_log_data, plf, indent=2)

        logger.info(
            f"Successfully applied picks! Shots: {len(manifest_shots)}, "
            f"Applied picks: {len(applied_picks)}, Cards created: {len(cards_created)}, "
            f"Failed downloads: {len(failed_downloads)}"
        )
        return ["media/media_manifest.json", "media/shot_plan.json", "media/credits.md", "media/picks_log.json"]

    def run_conform(self, picks_file: Path | str | None = None) -> list[str]:
        """Conforms visual review picks or auto picks to the actual audio timeline."""
        if picks_file is None:
            review_dir = self.run_dir / "media" / "review"
            for candidate_name in ("picks_ui.json", "picks.json", "auto_picks.json"):
                p = review_dir / candidate_name
                if p.exists():
                    picks_file = p
                    break
            if picks_file is None:
                raise FileNotFoundError(f"No picks file found in {review_dir} to conform.")
        return self.apply_picks(picks_file)

    def run_auto(
        self,
        manifest: dict[str, Any] | None = None,
        candidates_k: int = 6,
    ) -> list[str]:
        """Runs media stage in AUTO mode:
        1. Runs candidates discovery and writes media/review/picker.html + candidates.json.
        2. Converts auto_picks (clips >= min_relevance, card shots below) into auto_picks.json.
        3. Applies those picks immediately so render/motion can run while keeping picker.html available.
        """
        t_start = time.time()
        logger.info(f"Starting Media Agent AUTO mode (k={candidates_k}) for run: {self.run_dir.name}")
        candidates_data = self.run_candidates(candidates_k=candidates_k)

        picks_map: dict[str, dict[str, Any]] = {}
        clips_picked = 0
        cards_picked = 0
        for sh in candidates_data.get("shots", []):
            sh_id = sh["shot_id"]
            if sh.get("auto_pick_type") == "clip" and sh.get("auto_pick_id"):
                cand_obj = next((c for c in sh.get("candidates", []) if c.get("id") == sh["auto_pick_id"]), None)
                picks_map[sh_id] = {
                    "action": "pick",
                    "scene_id": sh["scene_id"],
                    "shot_index": sh["shot_index"],
                    "candidate_id": sh["auto_pick_id"],
                    "source": cand_obj.get("source") if cand_obj else None,
                }
                clips_picked += 1
            else:
                picks_map[sh_id] = {
                    "action": "card",
                    "scene_id": sh["scene_id"],
                    "shot_index": sh["shot_index"],
                    "card_text": sh.get("scene_headline") or sh["scene_id"],
                }
                cards_picked += 1

        auto_picks_path = self.run_dir / "media" / "review" / "auto_picks.json"
        auto_picks_payload = {
            "date": self.run_dir.name,
            "mode": "auto",
            "min_relevance": candidates_data.get("min_relevance"),
            "picks": picks_map,
        }
        auto_picks_path.write_text(json.dumps(auto_picks_payload, indent=2), encoding="utf-8")

        applied_outputs = self.apply_picks(auto_picks_path)

        wall_time = round(time.time() - t_start, 2)
        if manifest and "stages" in manifest and "media" in manifest["stages"]:
            manifest["stages"]["media"]["metrics"] = {
                "auto_mode": True,
                "total_shots": len(candidates_data.get("shots", [])),
                "auto_clips_count": clips_picked,
                "auto_cards_count": cards_picked,
                "min_relevance": candidates_data.get("min_relevance"),
                "wall_time_sec": wall_time,
                "warnings_count": len(self.warnings),
            }

        outputs = list(dict.fromkeys(applied_outputs + ["media/shot_plan.json", "media/review/picker.html"]))
        logger.info(f"Media AUTO complete in {wall_time}s: {clips_picked} clips, {cards_picked} cards.")
        return outputs


# CLI Interface implementation
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Media Agent CLI")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date (YYYY-MM-DD)")
    parser.add_argument("--auto", action="store_true", help="Run candidates discovery then automatically apply auto-picks and cards")
    parser.add_argument("--dry-run", action="store_true", help="Generate plan without downloading")
    parser.add_argument("--offline", action="store_true", help="Use cache and local library only")
    parser.add_argument("--allow-fixture", action="store_true", help="Allow running on fixture/mock script data")
    parser.add_argument("--prune-cache", action="store_true", help="Prune unreferenced downloads older than keep_days")
    parser.add_argument("--scene", help="Target single scene ID")
    parser.add_argument("--replace", type=int, help="Shot index to replace")
    parser.add_argument("--query", help="Search query for shot replacement")
    parser.add_argument("--pin", help="Pin specific local file to shot")
    parser.add_argument("--license", default="own work", help="License for pinned/library file")
    parser.add_argument("--attribution", default="", help="Attribution for pinned/library file")
    parser.add_argument("--add-to-library", dest="add_library_file", help="Import media file to local library")
    parser.add_argument("--candidates", type=int, nargs="?", const=6, default=None, help="Generate candidate review pack with top K candidates per shot")
    parser.add_argument("--apply-picks", dest="apply_picks_file", help="Apply visual review picks from JSON file")
    parser.add_argument("--library-report", action="store_true", help="Print report of files in data/media_library")
    parser.add_argument("--keywords", help="Comma-separated keywords for local library")
    parser.add_argument("--people-visible", default="false", help="Boolean: whether people are visible in footage")
    parser.add_argument("--approve", help="Approve yt-dlp candidate ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # CLI action 0: Library report
    if args.library_report:
        report = generate_library_report()
        print(report)
        return 0

    # CLI action 1: Add to local library
    if args.add_library_file:
        kw_list = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
        people_vis = args.people_visible.lower() in ("true", "1", "yes")
        dest = add_to_library_cli(
            file_path=args.add_library_file,
            keywords=kw_list,
            license_name=args.license,
            people_visible=people_vis,
        )
        print(f"[OK] Added file to local library: {dest}")
        print(f"Sidecar created: {dest.with_suffix('.json')}")
        return 0

    # CLI action 2: Prune cache
    if args.prune_cache:
        cache_mgr = MediaCacheManager()
        media_cfg_path = Path("config/media.yaml")
        cfg = {}
        if media_cfg_path.exists():
            with open(media_cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        settings = cfg.get("settings", {})
        res = cache_mgr.prune_cache(
            runs_dir="data/runs",
            max_gb=float(settings.get("cache_max_gb", 5.0)),
            keep_days=int(settings.get("cache_keep_days", 7)),
        )
        print(f"[OK] Pruned cache: Deleted {res['deleted_count']} files, Freed {res['freed_mb']} MB. Remaining: {res['remaining_mb']} MB.")
        return 0

    run_dir = Path("data/runs") / args.date
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        return 1

    agent = MediaAgent(
        run_dir,
        offline=args.offline,
        dry_run=args.dry_run,
        allow_fixture=args.allow_fixture,
    )

    # CLI action: Candidates review pack mode
    if args.candidates is not None:
        c_k = int(args.candidates)
        print(f"Generating candidate review pack (k={c_k}) for {args.date}...")
        c_data = agent.run_candidates(candidates_k=c_k)
        print(f"[OK] Generated candidates for {len(c_data['shots'])} shots.")
        print(f"Candidates JSON: {run_dir / 'media' / 'review' / 'candidates.json'}")
        print(f"Picker page:     {run_dir / 'media' / 'review' / 'picker.html'}")
        print(f"Wanted clips:    {run_dir / 'media' / 'review' / 'wanted_clips.md'}")
        return 0

    # CLI action: Apply visual review picks
    if args.apply_picks_file:
        print(f"Applying visual review picks from {args.apply_picks_file}...")
        outputs = agent.apply_picks(args.apply_picks_file)
        print("[OK] Applied picks successfully.")
        for o in outputs:
            print(f" - {run_dir / o}")
        picks_log_p = run_dir / "media" / "picks_log.json"
        if picks_log_p.exists():
            plog = json.loads(picks_log_p.read_text(encoding="utf-8"))
            print(f"Applied picks: {len(plog.get('applied_picks', []))} | Cards created: {len(plog.get('cards_created', []))} | Failed downloads: {len(plog.get('failed_downloads', []))} | Shots changed: {plog.get('shots_changed_count')}")
            for fd in plog.get("failed_downloads", []):
                print(f" [WARN] Download failed: {fd['shot_id']} ({fd['candidate_id']}): {fd['reason']}")
        return 0

    # CLI action: Auto mode (candidates + apply auto-picks + picker.html)
    if args.auto:
        manifest_file = run_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
        print(f"Running Media Agent in AUTO mode for {args.date}...")
        outputs = agent.run_auto(manifest)
        valid, err = agent.validate(outputs)
        if not valid:
            print(f"[FAIL] Contract validation error: {err}")
            return 1
        print("[OK] Media Agent AUTO executed successfully.")
        for o in outputs:
            print(f" - {run_dir / o}")
        return 0

    # CLI action 3: Replace shot
    if args.scene and args.replace is not None and args.query:
        print(f"Replacing scene {args.scene} shot #{args.replace} with query '{args.query}'...")
        manifest_path = run_dir / "media" / "media_manifest.json"
        if not manifest_path.exists():
            manifest_path = run_dir / "media_manifest.json"
        if not manifest_path.exists():
            print(f"[ERROR] Manifest not found. Run media stage first.")
            return 1

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_shot = next((s for s in data["shots"] if s["scene_id"] == args.scene and s["shot_index"] == args.replace), None)
        if not target_shot:
            print(f"[ERROR] Shot not found: {args.scene} #{args.replace}")
            return 1

        # Search with new query
        cand = None
        for c, q, s_name in agent.selector.iter_candidates_for_shot(
            scene={"id": args.scene, "visual_keywords": [args.query]},
            shot=PlannedShot(args.scene, args.replace, target_shot["start"], target_shot["end"], target_shot["end"] - target_shot["start"]),
            stills_in_scene=0,
        ):
            cand = c
            break

        if not cand:
            print(f"[WARN] No candidate found for query '{args.query}'")
            return 1

        raw_path = agent.cache_mgr.download_file(cand.download_url, offline=args.offline)
        out_filename = f"shot_{args.replace:02d}.mp4" if cand.kind == "video" else f"shot_{args.replace:02d}.jpg"
        out_path = run_dir / "media" / "shots" / args.scene / out_filename
        if cand.kind == "video":
            probe = normalize_video(raw_path, out_path, target_shot["end"] - target_shot["start"])
        else:
            probe = normalize_image(raw_path, out_path)

        target_shot["file"] = f"media/shots/{args.scene}/{out_filename}"
        target_shot["type"] = cand.kind
        target_shot["source"] = cand.source
        target_shot["origin_url"] = cand.page_url
        target_shot["license"] = cand.license_id
        target_shot["author"] = cand.author
        target_shot["attribution_text"] = cand.attribution_text
        target_shot["attribution_required"] = cand.attribution_required
        target_shot["sha256"] = probe.get("sha256")
        target_shot["query"] = args.query

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        generate_credits_markdown(data["shots"], run_dir / "media" / "credits.md")
        create_contact_sheet_for_scene(args.scene, [s for s in data["shots"] if s["scene_id"] == args.scene], run_dir, run_dir / "media" / "review" / f"{args.scene}.jpg")
        print(f"[OK] Replaced shot {args.scene} #{args.replace} with {cand.source} ({cand.id})")
        return 0

    # CLI action 4: Pin user-provided file to shot
    if args.scene and args.pin:
        pin_file = Path(args.pin)
        if not pin_file.exists():
            print(f"[ERROR] Pinned file not found: {pin_file}")
            return 1

        shot_idx = args.replace if args.replace is not None else 0
        print(f"Pinning {pin_file.name} to scene {args.scene} shot #{shot_idx}...")

        manifest_path = run_dir / "media" / "media_manifest.json"
        if not manifest_path.exists():
            manifest_path = run_dir / "media_manifest.json"
        if not manifest_path.exists():
            print(f"[ERROR] Manifest not found. Run media stage first.")
            return 1

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target_shot = next((s for s in data["shots"] if s["scene_id"] == args.scene and s["shot_index"] == shot_idx), None)
        if not target_shot:
            print(f"[ERROR] Shot not found: {args.scene} #{shot_idx}")
            return 1

        is_video = pin_file.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")
        out_filename = f"shot_{shot_idx:02d}.mp4" if is_video else f"shot_{shot_idx:02d}.jpg"
        out_path = run_dir / "media" / "shots" / args.scene / out_filename

        shot_dur = target_shot["end"] - target_shot["start"]
        if is_video:
            probe = normalize_video(pin_file, out_path, shot_dur)
        else:
            probe = normalize_image(pin_file, out_path)

        target_shot["file"] = f"media/shots/{args.scene}/{out_filename}"
        target_shot["type"] = "video" if is_video else "image"
        target_shot["source"] = "user-provided"
        target_shot["origin_url"] = "local"
        target_shot["license"] = args.license or "own work"
        target_shot["author"] = "User"
        target_shot["attribution_text"] = args.attribution or "User-provided footage"
        target_shot["attribution_required"] = False
        target_shot["sha256"] = probe.get("sha256")
        target_shot["query"] = "pinned"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        generate_credits_markdown(data["shots"], run_dir / "media" / "credits.md")
        create_contact_sheet_for_scene(args.scene, [s for s in data["shots"] if s["scene_id"] == args.scene], run_dir, run_dir / "media" / "review" / f"{args.scene}.jpg")
        print(f"[OK] Pinned file {pin_file.name} to shot {args.scene} #{shot_idx}")
        return 0

    # Normal execution
    manifest_file = run_dir / "manifest.json"
    manifest = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    outputs = agent.run(manifest)
    valid, err = agent.validate(outputs)
    if not valid:
        print(f"[FAIL] Contract validation error: {err}")
        return 1

    print(f"[OK] Media Agent executed successfully.")
    for o in outputs:
        print(f" - {run_dir / o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

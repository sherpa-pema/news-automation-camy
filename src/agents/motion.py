"""Motion Agent: generates render_plan.json, captions.ass, and overlay graphics."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any
import yaml

from src.agents.base import BaseAgent
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.audio_sync import (
    extract_audio_grid,
    generate_synthetic_broadcast_theme,
    snap_cuts_to_beats,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest
from src.utils.scene_classifier import classify_scene, SceneClassification
from src.utils.motion_graphics import generate_ass_subtitles, draw_illustrative_tag
from src.utils.schema_validate import validate_contract
from src.utils.captions import resolve_style
from src.utils.motion_timing import resolve_transitions
from src.utils.overlay_plan import (
    STOCK_SOURCES,
    sanitize_credit,
    resolve_shot_motion,
    validate_layout,
    build_scene_overlays,
    build_all_overlays,
    format_date_display,
)

logger = logging.getLogger("motion")


def classify_shot_recipe(sc_script: dict[str, Any], has_video: bool = False) -> tuple[str, dict[str, Any]]:
    """Semantic scene classifier mapping news stories into motion graphics templates.

    Replaces legacy Opus Video Prompts / Shotcraft heuristics with clean semantic analysis.
    Maintained as a backward-compatible adapter.
    """
    classification = classify_scene(sc_script, has_video=has_video)
    return classification.recipe, classification.data


class MotionAgent(BaseAgent):
    """Motion Agent orchestrates motion graphics compositions and render plans.

    Outputs:
        - motion/render_plan.json (validates against contracts/render_plan.schema.json)
        - motion/captions.ass
        - motion/overlays/*.png
    """

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        super().__init__(run_dir, config)
        self.logger = setup_logger("motion_agent", self.run_dir / "run.log")
        self.style_config = self._load_style_config()
        self.channel_config = self._load_channel_config()

    @property
    def name(self) -> str:
        return "motion"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/render_plan.schema.json"

    def _load_style_config(self) -> dict[str, Any]:
        p = Path("config/style.yaml")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_channel_config(self) -> dict[str, Any]:
        p = Path("config/channel.yaml")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def run(self, manifest: dict[str, Any] | None = None) -> list[str]:
        """Executes motion planning: builds render_plan.json, captions.ass, and overlay assets."""
        self.logger.info(f"Starting Motion Agent run in: {self.run_dir}")

        # 1. Input paths
        script_file = self.run_dir / "script" / "script.json"
        if not script_file.exists():
            script_file = self.run_dir / "script.json"
        if not script_file.exists():
            raise FileNotFoundError(f"Missing required script.json in {self.run_dir}")

        timeline_file = self.run_dir / "timeline" / "timeline.json"
        if not timeline_file.exists():
            timeline_file = self.run_dir / "timeline.json"
        if not timeline_file.exists():
            raise FileNotFoundError(f"Missing required timeline.json in {self.run_dir}")

        captions_file = self.run_dir / "timeline" / "captions.json"
        if not captions_file.exists():
            captions_file = self.run_dir / "captions.json"
        if not captions_file.exists():
            raise FileNotFoundError(f"Missing required captions.json in {self.run_dir}")

        media_manifest_file = self.run_dir / "media" / "media_manifest.json"
        if not media_manifest_file.exists():
            media_manifest_file = self.run_dir / "media_manifest.json"
        if not media_manifest_file.exists():
            raise FileNotFoundError(f"Missing required media_manifest.json in {self.run_dir}")

        audio_file = self.run_dir / "audio" / "narration.wav"
        if not audio_file.exists():
            raise FileNotFoundError(f"Missing required audio/narration.wav in {self.run_dir}")

        # 2. Provenance & Editor Gate Check
        with open(script_file, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        origin = script_data.get("origin", "editor")
        if origin != "fixture":
            approval_file = self.run_dir / "script" / "approval.json"
            if not approval_file.exists():
                raise ValueError("script changed after approval, re-run validate and approve")
            with open(approval_file, "r", encoding="utf-8") as f:
                app_data = json.load(f)
            curr_hash = hashlib.sha256(script_file.read_bytes()).hexdigest()
            if app_data.get("script_sha256") != curr_hash:
                raise ValueError("script changed after approval, re-run validate and approve")
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

        # 3. Load remaining input data
        with open(timeline_file, "r", encoding="utf-8") as f:
            timeline_data = json.load(f)
        with open(captions_file, "r", encoding="utf-8") as f:
            captions_data = json.load(f)
        with open(media_manifest_file, "r", encoding="utf-8") as f:
            media_data = json.load(f)

        motion_dir = self.run_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)

        res = media_data.get("resolution", [1920, 1080])
        if isinstance(res, dict):
            res = [res.get("width", 1920), res.get("height", 1080)]
        res_str = f"{res[0]}x{res[1]}"
        fps = int(media_data.get("fps", 30))
        channel_name = self.channel_config.get("channel_name", "Nepal Daily Briefing")
        run_kind = script_data.get("run_kind")
        if not run_kind and manifest:
            run_kind = manifest.get("run_kind")
        if not run_kind:
            m_path = self.run_dir / "manifest.json"
            if m_path.exists():
                try:
                    run_kind = load_manifest(m_path).get("run_kind")
                except Exception:
                    pass
        run_kind = run_kind or "production"

        # 4. Audio Grid Extraction (preserves audio ducking & theme config)
        bgm_path = self.style_config.get("music", {}).get("default_track")
        if not bgm_path or not Path(bgm_path).exists():
            candidates = [
                Path("config/audio/broadcast_theme.wav"),
                Path("config/audio/broadcast_theme.mp3"),
                self.run_dir / "audio" / "bg_track.mp3",
                self.run_dir / "audio" / "bg_track.wav",
            ]
            for cand in candidates:
                if cand.exists():
                    bgm_path = str(cand)
                    break
        if self.style_config.get("music", {}).get("enabled", False) and (not bgm_path or not Path(bgm_path).exists()):
            self.logger.info("Background music enabled in style.yaml but no audio file found; proceeding with narration.")
            bgm_path = None

        audio_grid = extract_audio_grid(
            bgm_path=bgm_path,
            narration_path=audio_file,
            scene_boundaries=timeline_data.get("scenes", []),
            style_config=self.style_config,
        )
        with open(motion_dir / "audio_grid.json", "w", encoding="utf-8") as agf:
            json.dump(audio_grid, agf, indent=2)

        # 5. Semantic Classification for all scenes
        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}
        shots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for shot in media_data.get("shots", []):
            shots_by_scene.setdefault(shot["scene_id"], []).append(shot)

        classifications: dict[str, SceneClassification] = {}
        for sc in timeline_data.get("scenes", []):
            sc_id = sc["id"]
            sc_script = script_scenes.get(sc_id, {})
            sc_shots = shots_by_scene.get(sc_id, [])
            has_video = any(s.get("type") == "video" for s in sc_shots)
            classifications[sc_id] = classify_scene(sc_script, has_video=has_video)

        # 6. Build Overlays across all scenes and global overlays (watermark)
        total_duration_sec = float(timeline_data.get("duration_sec", 0.0))
        date_str = script_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        date_display = format_date_display(date_str)
        overlays_dir = motion_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        all_overlays, scene_overlays_map = build_all_overlays(
            timeline_scenes=timeline_data.get("scenes", []),
            script_scenes=script_scenes,
            shots_by_scene=shots_by_scene,
            scene_classifications=classifications,
            style_config=self.style_config,
            channel_name=channel_name,
            resolution=res,
            overlays_dir=overlays_dir,
            run_kind=run_kind,
            date_display_str=date_display,
            total_duration=total_duration_sec,
        )

        # 7. Assemble Scene Records for render_plan.json
        planned_scenes: list[dict[str, Any]] = []

        for sc in timeline_data.get("scenes", []):
            sc_id = sc["id"]
            sc_start = float(sc["start"])
            sc_end = float(sc["end"])
            sc_script = script_scenes.get(sc_id, {})
            sc_type = sc_script.get("type", sc.get("type", "story"))
            sc_shots = shots_by_scene.get(sc_id, [])

            cls = classifications.get(
                sc_id,
                SceneClassification(recipe="cinematic_broll_full", template_name="lower_third.html")
            )
            shot_recipe = cls.recipe
            recipe_data = cls.data
            has_featured = shot_recipe in ("stat_callout", "quote_card", "headline_split")

            planned_shots: list[dict[str, Any]] = []
            for sh_idx, s in enumerate(sc_shots):
                motion_type = resolve_shot_motion(s, index=sh_idx, featured_card=has_featured)
                s_start = round(float(s.get("start", sc_start)), 3)
                s_end = round(float(s.get("end", sc_end)), 3)
                planned_shots.append({
                    "shot_index": s.get("shot_index", 0),
                    "file": s.get("file", ""),
                    "start": s_start,
                    "end": s_end,
                    "duration": round(s_end - s_start, 3),
                    "type": s.get("type", "video"),
                    "motion": motion_type,
                    "source": s.get("source"),
                    "author": s.get("author"),
                    "sha256": s.get("sha256"),
                })

            primary_media = planned_shots[0]["file"] if planned_shots else ""
            scene_overlays = scene_overlays_map.get(sc_id, [])

            credit_label = sanitize_credit(sc_script, channel_name=channel_name)
            prev_sc_dict = planned_scenes[-1] if planned_scenes else None
            transition_data = resolve_transitions(sc_script, prev_sc_dict, self.style_config)
            caption_style_obj = resolve_style(resolution=res, style_config=self.style_config)

            sc_dict: dict[str, Any] = {
                "id": sc_id,
                "start": sc_start,
                "end": sc_end,
                "media": primary_media,
                "shots": planned_shots,
                "overlays": scene_overlays,
                "shot_recipe": shot_recipe,
                "beat_aligned": bool(audio_grid.get("enabled")),
                "lower_third": {
                    "headline": sc_script.get("headline", channel_name),
                    "source_credit": credit_label,
                    "style": "navy_amber",
                },
                "caption_style": caption_style_obj.to_plan_dict(),
                "transition": transition_data,
            }
            if sc_type == "story":
                if shot_recipe == "stat_callout":
                    sc_dict["stat_data"] = recipe_data
                elif shot_recipe == "quote_card":
                    sc_dict["quote_data"] = recipe_data
                elif shot_recipe == "headline_split":
                    sc_dict["split_data"] = recipe_data

            planned_scenes.append(sc_dict)

        # 8. Render Plan Payload & ASS Subtitles
        ass_path = motion_dir / "captions.ass"
        cap_style = resolve_style(resolution=res, style_config=self.style_config)
        generate_ass_subtitles(
            captions=captions_data,
            output_path=ass_path,
            resolution=tuple(res),
            font_size=cap_style.font_size,
            margin_v=cap_style.margin_v,
            font_name=cap_style.font_family,
            highlight_color=cap_style.highlight_color,
            box_rgba=cap_style.box_rgba,
        )

        render_plan = {
            "resolution": res_str,
            "fps": fps,
            "audio_file": "audio/narration.wav",
            "duration_sec": total_duration_sec,
            "duration": total_duration_sec,
            "captions_ass": "motion/captions.ass",
            "overlays": all_overlays,
            "audio_grid": audio_grid,
            "transitions": self.style_config.get("transitions", {
                "scene_boundary": {
                    "type": "dip_to_black",
                    "duration_sec": 0.2,
                    "fade_out_sec": 0.1,
                    "fade_in_sec": 0.1,
                },
                "intra_scene": {"type": "cut"},
            }),
            "scenes": planned_scenes,
        }

        # 9. Pre-render Layout Validation
        layout_errors = validate_layout(render_plan, self.style_config)
        if layout_errors:
            raise ValueError(f"Motion Agent layout validation failed: {'; '.join(layout_errors)}")

        # 10. Write and Validate Plan
        plan_path = motion_dir / "render_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(render_plan, f, indent=2)

        root_plan = self.run_dir / "render_plan.json"
        with open(root_plan, "w", encoding="utf-8") as f:
            json.dump(render_plan, f, indent=2)

        output_files = [
            "motion/render_plan.json",
            "motion/captions.ass",
        ]
        for ov in all_overlays:
            if ov["png"] not in output_files:
                output_files.append(ov["png"])

        valid, err = self.validate(output_files)
        if not valid:
            raise ValueError(f"Motion Agent validation failed: {err}")

        self.logger.info(
            f"Motion Agent successfully generated composition and render plan "
            f"({len(planned_scenes)} scenes, {len(all_overlays)} overlays)."
        )
        return output_files


STOCK_ILLUSTRATIVE_SOURCES = STOCK_SOURCES


def sync_render_plan_data(
    plan_data: dict[str, Any],
    media_data: dict[str, Any],
    run_dir: Path | None = None,
    script_data: dict[str, Any] | None = None,
    style_config: dict[str, Any] | None = None,
) -> bool:
    """Updates an existing render_plan payload in-place from media_manifest.json shots."""
    media_shots = media_data.get("shots", [])
    if not media_shots or not all(
        "scene_id" in sh and "start" in sh and "end" in sh and "type" in sh
        for sh in media_shots
    ):
        return False

    shots_by_scene: dict[str, list[dict[str, Any]]] = {}
    for sh in media_shots:
        shots_by_scene.setdefault(sh["scene_id"], []).append(sh)

    script_scenes = {s["id"]: s for s in (script_data or {}).get("scenes", [])}
    style_cfg = style_config or {}

    res = media_data.get("resolution", [1920, 1080])
    if isinstance(res, dict):
        res = [res.get("width", 1920), res.get("height", 1080)]
    is_vertical = media_data.get("orientation") == "vertical" or (len(res) >= 2 and res[1] > res[0])
    rects_key = "layout_rectangles_vertical" if is_vertical else "layout_rectangles"
    rects = style_cfg.get(rects_key) or style_cfg.get("layout_rectangles", {})

    default_tag = {"x": 742, "y": 120, "w": 284, "h": 44} if is_vertical else {"x": 1540, "y": 54, "w": 284, "h": 44}
    tag_rect = rects.get("illustrative_tag", default_tag)
    font_candidates = style_cfg.get("fonts", {}).get("primary_candidates", [])

    plan_scenes = plan_data.get("scenes", [])
    plan_scene_ids = {sc["id"] for sc in plan_scenes}

    rebuilt_overlays: list[dict[str, Any]] = [
        ov for ov in plan_data.get("overlays", [])
        if not ov.get("scene_id") or ov.get("scene_id") not in plan_scene_ids
    ]

    for sc in plan_scenes:
        sc_id = sc["id"]
        sc_start = float(sc.get("start", 0.0))
        sc_end = float(sc.get("end", 0.0))
        sc_shots = sorted(shots_by_scene.get(sc_id, []), key=lambda s: s.get("shot_index", 0))

        has_featured = sc.get("shot_recipe") in ("stat_callout", "quote_card", "headline_split")
        planned_shots: list[dict[str, Any]] = []
        for sh_idx, s in enumerate(sc_shots):
            s_start = round(float(s.get("start", sc_start)), 3)
            s_end = round(float(s.get("end", sc_end)), 3)
            motion_type = resolve_shot_motion(s, index=sh_idx, featured_card=has_featured)
            planned_shots.append({
                "shot_index": s.get("shot_index", 0),
                "file": s.get("file") or "",
                "start": s_start,
                "end": s_end,
                "duration": round(s_end - s_start, 3),
                "type": s.get("type", "video"),
                "motion": motion_type,
                "source": s.get("source"),
                "author": s.get("author"),
                "sha256": s.get("sha256"),
            })

        sc["shots"] = planned_shots
        sc["media"] = planned_shots[0]["file"] if planned_shots else ""

        sc_script = script_scenes.get(sc_id, {})
        sc_type = sc_script.get("type") or ("intro" if sc_id == "intro" else ("outro" if sc_id == "outro" else "story"))
        if sc_type not in ("intro", "outro"):
            existing_overlays = list(sc.get("overlays", []))
            existing_tag = next((ov for ov in existing_overlays if ov.get("type") == "illustrative_tag"), None)
            sc_overlays = [ov for ov in existing_overlays if ov.get("type") != "illustrative_tag"]

            has_illustrative = any(s.get("source") in STOCK_SOURCES for s in sc_shots)
            if has_illustrative:
                if existing_tag:
                    sc_overlays.append(existing_tag)
                else:
                    if run_dir is not None:
                        ill_png = run_dir / "motion" / "overlays" / "illustrative_tag.png"
                        if not ill_png.exists():
                            ill_png.parent.mkdir(parents=True, exist_ok=True)
                            draw_illustrative_tag(
                                ill_png,
                                width=tag_rect["w"],
                                height=tag_rect["h"],
                                font_candidates=font_candidates,
                            )
                    sc_overlays.append({
                        "id": f"{sc_id}_illustrative_tag",
                        "scene_id": sc_id,
                        "type": "illustrative_tag",
                        "png": "motion/overlays/illustrative_tag.png",
                        "start": round(sc_start + 0.2, 3),
                        "end": round(max(sc_start + 0.3, sc_end - 0.2), 3),
                        "x": tag_rect["x"],
                        "y": tag_rect["y"],
                        "w": tag_rect["w"],
                        "h": tag_rect["h"],
                        "animation": "fade",
                        "fade_in": 0.2,
                        "fade_out": 0.2,
                    })
            sc["overlays"] = sc_overlays

        rebuilt_overlays.extend(sc.get("overlays", []))

    plan_data["overlays"] = rebuilt_overlays

    # Persist updated plan to canonical file path(s)
    if run_dir is not None:
        motion_plan_file = run_dir / "motion" / "render_plan.json"
        if motion_plan_file.exists():
            with open(motion_plan_file, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2)
        root_plan_file = run_dir / "render_plan.json"
        if root_plan_file.exists():
            with open(root_plan_file, "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2)

    return True


def sync_render_plan_with_media_manifest(run_dir: Path) -> bool:
    """Syncs existing motion/render_plan.json with media/media_manifest.json."""
    media_file = run_dir / "media" / "media_manifest.json"
    if not media_file.exists():
        media_file = run_dir / "media_manifest.json"
    if not media_file.exists():
        return False

    motion_plan_file = run_dir / "motion" / "render_plan.json"
    root_plan_file = run_dir / "render_plan.json"
    primary_plan_file = motion_plan_file if motion_plan_file.exists() else root_plan_file
    if not primary_plan_file.exists():
        return False

    with open(media_file, "r", encoding="utf-8") as mf:
        media_data = json.load(mf)
    with open(primary_plan_file, "r", encoding="utf-8") as pf:
        plan_data = json.load(pf)

    script_data = None
    script_file = run_dir / "script" / "script.json"
    if not script_file.exists():
        script_file = run_dir / "script.json"
    if script_file.exists():
        with open(script_file, "r", encoding="utf-8") as sf:
            script_data = json.load(sf)

    style_config = {}
    style_path = Path("config/style.yaml")
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as stf:
            style_config = yaml.safe_load(stf) or {}

    if not sync_render_plan_data(
        plan_data=plan_data,
        media_data=media_data,
        run_dir=run_dir,
        script_data=script_data,
        style_config=style_config,
    ):
        return False

    is_valid, err = validate_contract(plan_data, "contracts/render_plan.schema.json")
    if not is_valid:
        raise ValueError(f"Synced render_plan.json schema validation failed: {err}")

    if motion_plan_file.exists():
        with open(motion_plan_file, "w", encoding="utf-8") as pf:
            json.dump(plan_data, pf, indent=2)
    if root_plan_file.exists():
        with open(root_plan_file, "w", encoding="utf-8") as rpf:
            json.dump(plan_data, rpf, indent=2)

    logger.info(f"Synced render_plan.json with media_manifest.json for {run_dir.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nepal News Bot Motion Agent.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Run date in YYYY-MM-DD")
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs directory")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.date
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    agent = MotionAgent(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {}

    outputs = agent.run(manifest)
    print(f"Motion Agent completed successfully. Outputs: {outputs}")


if __name__ == "__main__":
    main()

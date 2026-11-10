"""Motion Agent: generates render_plan.json, overlay graphics, and ASS subtitles."""
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
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest
from src.utils.motion_graphics import (
    draw_illustrative_tag,
    draw_lower_third,
    draw_outro_card,
    draw_test_run_tag,
    draw_title_card,
    generate_ass_subtitles,
)
from src.utils.schema_validate import validate_contract

logger = logging.getLogger("motion")


class MotionAgent(BaseAgent):
    """Motion Agent designs the visual composition and motion graphics plan per scene.

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
        """Executes motion planning: builds render_plan.json, overlay PNGs, and captions.ass."""
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

        # Output folders
        motion_dir = self.run_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir = motion_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        # Style & dimensions
        res = media_data.get("resolution", [1920, 1080])
        if isinstance(res, dict):
            res = [res.get("width", 1920), res.get("height", 1080)]
        res_str = f"{res[0]}x{res[1]}"
        fps = int(media_data.get("fps", 30))
        channel_name = self.channel_config.get("channel_name", "Nepal Daily Briefing")
        date_str = script_data.get("date", datetime.now().strftime("%Y-%m-%d"))
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
        disclose_synthetic_voice = bool(media_data.get("disclose_synthetic_voice", True))

        rects = self.style_config.get("layout_rectangles", {})
        lt_rect = rects.get("lower_third", {"x": 96, "y": 730, "w": 1320, "h": 155})
        title_rect = rects.get("title_card", {"x": 260, "y": 280, "w": 1400, "h": 460})
        outro_rect = rects.get("outro_card", {"x": 260, "y": 280, "w": 1400, "h": 460})
        tag_rect = rects.get("illustrative_tag", {"x": 1540, "y": 54, "w": 284, "h": 44})
        test_rect = rects.get("test_run_tag", {"x": 96, "y": 54, "w": 180, "h": 44})

        font_candidates = self.style_config.get("fonts", {}).get("primary_candidates", [])

        # 4. Generate ASS Subtitles
        ass_path = motion_dir / "captions.ass"
        margin_v = int(self.style_config.get("layout_rectangles", {}).get("caption_margin_v", 70))
        generate_ass_subtitles(
            captions=captions_data,
            output_path=ass_path,
            resolution=(res[0], res[1]),
            font_size=54,
            margin_v=margin_v,
        )
        self.logger.info(f"Generated ASS subtitles at {ass_path}")

        # Draw reusable test run tag if test run
        test_tag_rel_path: str | None = None
        if run_kind == "test" and self.style_config.get("watermark_test_runs", True):
            test_tag_path = overlays_dir / "test_run_tag.png"
            draw_test_run_tag(test_tag_path, width=test_rect["w"], height=test_rect["h"], font_candidates=font_candidates)
            test_tag_rel_path = f"motion/overlays/{test_tag_path.name}"

        # Group shots by scene
        shots_by_scene: dict[str, list[dict[str, Any]]] = {}
        for shot in media_data.get("shots", []):
            shots_by_scene.setdefault(shot["scene_id"], []).append(shot)

        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}
        planned_scenes: list[dict[str, Any]] = []
        all_overlays: list[dict[str, Any]] = []

        total_duration_sec = float(timeline_data.get("duration_sec", 0.0))

        # 5. Build scene motion plans
        for sc in timeline_data.get("scenes", []):
            sc_id = sc["id"]
            sc_start = float(sc["start"])
            sc_end = float(sc["end"])
            sc_dur = sc_end - sc_start
            speech_start = float(sc.get("speech_start", sc_start))

            sc_script = script_scenes.get(sc_id, {})
            sc_type = sc_script.get("type", sc.get("type", "story"))
            sc_shots = shots_by_scene.get(sc_id, [])

            # Format shots for plan
            planned_shots: list[dict[str, Any]] = []
            for s in sc_shots:
                is_still = s.get("type") in ("image", "card")
                motion_type = "zoom_in" if is_still else None
                planned_shots.append({
                    "shot_index": s.get("shot_index", 0),
                    "file": s.get("file", ""),
                    "start": round(float(s.get("start", sc_start)), 3),
                    "end": round(float(s.get("end", sc_end)), 3),
                    "duration": round(float(s.get("end", sc_end)) - float(s.get("start", sc_start)), 3),
                    "type": s.get("type", "video"),
                    "motion": motion_type,
                    "source": s.get("source"),
                    "author": s.get("author"),
                    "sha256": s.get("sha256"),
                })

            primary_media = planned_shots[0]["file"] if planned_shots else ""

            # Overlays per scene
            scene_overlays: list[dict[str, Any]] = []

            # A. Test run overlay (if test run)
            if test_tag_rel_path:
                t_overlay = {
                    "id": f"{sc_id}_test_tag",
                    "scene_id": sc_id,
                    "type": "test_tag",
                    "png": test_tag_rel_path,
                    "start": sc_start,
                    "end": sc_end,
                    "x": test_rect["x"],
                    "y": test_rect["y"],
                    "w": test_rect["w"],
                    "h": test_rect["h"],
                    "animation": None,
                    "fade_in": 0.0,
                    "fade_out": 0.0,
                }
                scene_overlays.append(t_overlay)
                all_overlays.append(t_overlay)

            # B. Scene-type specific overlays
            if sc_type == "intro":
                # Title card: visible from 0.3s to min(sc_end - 0.5s, 6.0s)
                title_png = overlays_dir / f"{sc_id}_title.png"
                draw_title_card(
                    channel_name=channel_name,
                    date_str=date_str,
                    output_path=title_png,
                    width=title_rect["w"],
                    height=title_rect["h"],
                    font_candidates=font_candidates,
                )
                card_start = round(sc_start + 0.3, 3)
                card_end = round(min(sc_start + 6.0, sc_end - 0.5), 3)
                if card_end <= card_start:
                    card_end = round(sc_end - 0.2, 3)

                ov = {
                    "id": f"{sc_id}_title_card",
                    "scene_id": sc_id,
                    "type": "title_card",
                    "png": f"motion/overlays/{title_png.name}",
                    "start": card_start,
                    "end": card_end,
                    "x": title_rect["x"],
                    "y": title_rect["y"],
                    "w": title_rect["w"],
                    "h": title_rect["h"],
                    "animation": "fade",
                    "fade_in": 0.3,
                    "fade_out": 0.3,
                }
                scene_overlays.append(ov)
                all_overlays.append(ov)

            elif sc_type == "outro":
                # Outro card: visible from 0.3s to sc_end - 0.5s
                outro_png = overlays_dir / f"{sc_id}_outro.png"
                draw_outro_card(
                    channel_name=channel_name,
                    output_path=outro_png,
                    width=outro_rect["w"],
                    height=outro_rect["h"],
                    disclose_synthetic_voice=disclose_synthetic_voice,
                    font_candidates=font_candidates,
                )
                card_start = round(sc_start + 0.3, 3)
                card_end = round(sc_end - 0.5, 3)
                if card_end <= card_start:
                    card_end = round(sc_end - 0.2, 3)

                ov = {
                    "id": f"{sc_id}_outro_card",
                    "scene_id": sc_id,
                    "type": "outro_card",
                    "png": f"motion/overlays/{outro_png.name}",
                    "start": card_start,
                    "end": card_end,
                    "x": outro_rect["x"],
                    "y": outro_rect["y"],
                    "w": outro_rect["w"],
                    "h": outro_rect["h"],
                    "animation": "fade",
                    "fade_in": 0.3,
                    "fade_out": 0.3,
                }
                scene_overlays.append(ov)
                all_overlays.append(ov)

            else:
                # Story scene: Lower third with headline & source
                headline = sc_script.get("headline", "")
                sources = sc_script.get("sources", [])
                source_str = ", ".join(sources) if isinstance(sources, list) else str(sources)
                if not source_str:
                    source_str = "Verified Outlets"

                lt_png = overlays_dir / f"{sc_id}_lower_third.png"
                draw_lower_third(
                    headline=headline,
                    source_text=source_str,
                    output_path=lt_png,
                    width=lt_rect["w"],
                    height=lt_rect["h"],
                    font_candidates=font_candidates,
                )

                # Show from speech_start + 0.3s for up to 7.0s, not past sc_end - 0.5s
                lt_start = round(speech_start + 0.3, 3)
                lt_end = round(min(lt_start + 7.0, sc_end - 0.5), 3)
                if lt_end <= lt_start:
                    lt_end = round(sc_end - 0.2, 3)

                lt_ov = {
                    "id": f"{sc_id}_lower_third",
                    "scene_id": sc_id,
                    "type": "lower_third",
                    "png": f"motion/overlays/{lt_png.name}",
                    "start": lt_start,
                    "end": lt_end,
                    "x": lt_rect["x"],
                    "y": lt_rect["y"],
                    "w": lt_rect["w"],
                    "h": lt_rect["h"],
                    "animation": "slide_up",
                    "fade_in": 0.2,
                    "fade_out": 0.2,
                }
                scene_overlays.append(lt_ov)
                all_overlays.append(lt_ov)

                # Illustrative visuals tag (shown if stock or archive shots used)
                has_illustrative = any(
                    s.get("source") in ("wikimedia_commons", "pixabay", "coverr", "openverse", "nasa", "internet_archive")
                    for s in sc_shots
                )
                if has_illustrative:
                    ill_png = overlays_dir / "illustrative_tag.png"
                    if not ill_png.exists():
                        draw_illustrative_tag(ill_png, width=tag_rect["w"], height=tag_rect["h"], font_candidates=font_candidates)

                    tag_start = round(sc_start + 0.2, 3)
                    tag_end = round(sc_end - 0.2, 3)
                    tag_ov = {
                        "id": f"{sc_id}_illustrative_tag",
                        "scene_id": sc_id,
                        "type": "illustrative_tag",
                        "png": f"motion/overlays/{ill_png.name}",
                        "start": tag_start,
                        "end": tag_end,
                        "x": tag_rect["x"],
                        "y": tag_rect["y"],
                        "w": tag_rect["w"],
                        "h": tag_rect["h"],
                        "animation": "fade",
                        "fade_in": 0.2,
                        "fade_out": 0.2,
                    }
                    scene_overlays.append(tag_ov)
                    all_overlays.append(tag_ov)

            # Scene transition: dip to black at scene boundary
            dip_dur = float(self.style_config.get("transitions", {}).get("scene_boundary", {}).get("duration_sec", 0.2))
            planned_scenes.append({
                "id": sc_id,
                "start": sc_start,
                "end": sc_end,
                "media": primary_media,
                "shots": planned_shots,
                "overlays": scene_overlays,
                "lower_third": {
                    "headline": sc_script.get("headline", channel_name),
                    "source_credit": ", ".join(sc_script.get("sources", [])) if sc_script.get("sources") else channel_name,
                    "style": "navy_amber",
                },
                "caption_style": {
                    "font": "Arial",
                    "font_size": 54,
                    "position": "bottom",
                    "highlight_color": "#F5A623",
                },
                "transition": {
                    "type": "dip_to_black",
                    "duration_sec": dip_dur,
                },
            })

        # 6. Render Plan Payload
        render_plan = {
            "resolution": res_str,
            "fps": fps,
            "audio_file": "audio/narration.wav",
            "duration_sec": total_duration_sec,
            "duration": total_duration_sec,
            "captions_ass": "motion/captions.ass",
            "overlays": all_overlays,
            "transitions": {
                "scene_boundary": {
                    "type": "dip_to_black",
                    "duration_sec": 0.2,
                    "fade_out_sec": 0.1,
                    "fade_in_sec": 0.1,
                },
                "intra_scene": {"type": "cut"},
            },
            "scenes": planned_scenes,
        }

        # 7. Write plan and validate schema contract
        plan_path = motion_dir / "render_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(render_plan, f, indent=2)

        # Also write root render_plan.json for backward compatibility if needed
        root_plan = self.run_dir / "render_plan.json"
        with open(root_plan, "w", encoding="utf-8") as f:
            json.dump(render_plan, f, indent=2)

        output_files = [
            "motion/render_plan.json",
            "motion/captions.ass",
        ]
        # Include overlay files
        for ov in all_overlays:
            if ov["png"] not in output_files:
                output_files.append(ov["png"])

        valid, err = self.validate(output_files)
        if not valid:
            raise ValueError(f"Motion Agent validation failed: {err}")

        self.logger.info(f"Motion Agent successfully generated render plan with {len(planned_scenes)} scenes and {len(all_overlays)} overlays.")
        return output_files


STOCK_ILLUSTRATIVE_SOURCES = (
    "wikimedia_commons",
    "pixabay",
    "coverr",
    "openverse",
    "nasa",
    "internet_archive",
)


def sync_render_plan_data(
    plan_data: dict[str, Any],
    media_data: dict[str, Any],
    run_dir: Path | None = None,
    script_data: dict[str, Any] | None = None,
    style_config: dict[str, Any] | None = None,
) -> bool:
    """Updates an existing render_plan payload in-place from media_manifest.json shots.

    Returns True if plan_data had compatible contract shots and was synced, False otherwise.
    """
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
    rects = style_cfg.get("layout_rectangles", {})
    tag_rect = rects.get("illustrative_tag", {"x": 1540, "y": 54, "w": 284, "h": 44})
    font_candidates = style_cfg.get("fonts", {}).get("primary_candidates", [])

    plan_scenes = plan_data.get("scenes", [])
    plan_scene_ids = {sc["id"] for sc in plan_scenes}

    # Preserve any global / non-scene overlays cleanly
    rebuilt_overlays: list[dict[str, Any]] = [
        ov for ov in plan_data.get("overlays", [])
        if not ov.get("scene_id") or ov.get("scene_id") not in plan_scene_ids
    ]

    for sc in plan_scenes:
        sc_id = sc["id"]
        sc_start = float(sc.get("start", 0.0))
        sc_end = float(sc.get("end", 0.0))
        sc_shots = sorted(shots_by_scene.get(sc_id, []), key=lambda s: s.get("shot_index", 0))

        planned_shots: list[dict[str, Any]] = []
        for s in sc_shots:
            s_start = round(float(s.get("start", sc_start)), 3)
            s_end = round(float(s.get("end", sc_end)), 3)
            is_still = s.get("type") in ("image", "card")
            planned_shots.append({
                "shot_index": s.get("shot_index", 0),
                "file": s.get("file") or "",
                "start": s_start,
                "end": s_end,
                "duration": round(s_end - s_start, 3),
                "type": s.get("type", "video"),
                "motion": "zoom_in" if is_still else None,
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

            has_illustrative = any(s.get("source") in STOCK_ILLUSTRATIVE_SOURCES for s in sc_shots)
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
    return True


def sync_render_plan_with_media_manifest(run_dir: Path) -> bool:
    """Syncs existing motion/render_plan.json (and root render_plan.json) with media/media_manifest.json."""
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

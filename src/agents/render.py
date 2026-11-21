"""Render Agent: stitches media, motion overlays, narration audio, and subtitles into final MP4."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import yaml

from src.agents.base import BaseAgent
from src.agents.motion import sync_render_plan_data
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest
from src.utils.office_events import emit_office_event

logger = logging.getLogger("render")


def format_chapter_time(sec: float) -> str:
    """Formats seconds into YouTube chapter timestamp MM:SS or HH:MM:SS."""
    total_s = int(round(sec))
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


_HAS_NVENC: bool | None = None


def check_nvenc_available() -> bool:
    """Probes whether h264_nvenc encoder is available and supported by the GPU driver."""
    global _HAS_NVENC
    if _HAS_NVENC is not None:
        return _HAS_NVENC
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.04",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        _HAS_NVENC = (res.returncode == 0)
    except Exception:
        _HAS_NVENC = False
    return _HAS_NVENC


def slice_ass_for_scene(master_ass_path: Path, scene_start: float, scene_end: float, out_scene_ass: Path) -> str:
    """Slices master ASS file to dialogue lines belonging to scene_start..scene_end, shifted to 0..sc_dur."""
    out_scene_ass.parent.mkdir(parents=True, exist_ok=True)
    if not master_ass_path.exists() or master_ass_path.stat().st_size == 0:
        empty_ass = (
            "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,54,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.0,2,40,40,70,1\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        out_scene_ass.write_text(empty_ass, encoding="utf-8")
        return "empty"

    def ass_time_to_sec(t_str: str) -> float:
        parts = t_str.strip().split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        return 0.0

    def sec_to_ass_time(sec: float) -> str:
        s = max(0.0, sec)
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        rem = s % 60
        return f"{h}:{m:02d}:{rem:05.2f}"

    lines = master_ass_path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    dialogue_lines: list[str] = []
    in_events = False

    for line in lines:
        if line.strip().startswith("[Events]"):
            in_events = True
            header_lines.append(line)
            continue
        if not in_events:
            header_lines.append(line)
            continue

        if in_events:
            if line.strip().startswith("Format:"):
                header_lines.append(line)
                continue
            if line.strip().startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) >= 10:
                    start_sec = ass_time_to_sec(parts[1])
                    end_sec = ass_time_to_sec(parts[2])
                    if end_sec > scene_start and start_sec < scene_end:
                        shifted_start = max(0.0, start_sec - scene_start)
                        shifted_end = min(scene_end - scene_start, max(shifted_start + 0.05, end_sec - scene_start))
                        parts[1] = sec_to_ass_time(shifted_start)
                        parts[2] = sec_to_ass_time(shifted_end)
                        dialogue_lines.append(",".join(parts))

    out_content = "\n".join(header_lines) + "\n" + "\n".join(dialogue_lines) + "\n"
    out_scene_ass.write_text(out_content, encoding="utf-8")
    return hashlib.sha256(out_content.encode("utf-8")).hexdigest()[:16]


def compute_scene_cache_key(sc: dict[str, Any], style_hash: str, extra_hash: str = "") -> str:
    """Computes a deterministic hash for a scene's render plan inputs."""
    payload = {
        "id": sc["id"],
        "start": sc["start"],
        "end": sc["end"],
        "shots": [
            {
                "file": s.get("file"),
                "duration": s.get("duration"),
                "type": s.get("type"),
                "motion": s.get("motion"),
                "sha256": s.get("sha256"),
            }
            for s in sc.get("shots", [])
        ],
        "overlays": [
            {
                "id": ov.get("id"),
                "png": ov.get("png"),
                "start": ov.get("start"),
                "end": ov.get("end"),
                "x": ov.get("x"),
                "y": ov.get("y"),
                "w": ov.get("w"),
                "h": ov.get("h"),
                "animation": ov.get("animation"),
                "fade_in": ov.get("fade_in"),
                "fade_out": ov.get("fade_out"),
            }
            for ov in sc.get("overlays", [])
        ],
        "transition": sc.get("transition"),
        "style_hash": style_hash,
        "extra_hash": extra_hash,
        "zoom_mode": "linear_v2",
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class RenderAgent(BaseAgent):
    """Render Agent stitches media, narration, overlays, and subtitles into the final MP4."""

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        super().__init__(run_dir, config)
        self.logger = setup_logger("render_agent", self.run_dir / "run.log")
        self.style_config = self._load_style_config()
        self.channel_config = self._load_channel_config()

    @property
    def name(self) -> str:
        return "render"

    @property
    def schema_contract_path(self) -> str | None:
        return None  # Final MP4 stream integrity validated via ffprobe

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

    def _verify_gate(self) -> None:
        """Verifies script approval and validator fingerprint gate."""
        script_file = self.run_dir / "script" / "script.json"
        if not script_file.exists():
            script_file = self.run_dir / "script.json"
        if not script_file.exists():
            raise FileNotFoundError(f"Missing required script.json in {self.run_dir}")

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

    def _render_scene_segment(
        self,
        sc: dict[str, Any],
        fps: int,
        target_res: tuple[int, int],
        output_path: Path,
        crf: int = 20,
        preset: str = "veryfast",
        global_overlays: list[dict[str, Any]] | None = None,
        scene_ass_path: Path | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        """Builds a frame-exact silent MP4 segment for a single scene with shots, dips, and overlays."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tw, th = target_res
        shots = sc.get("shots", [])
        overlays = list(sc.get("overlays", []))
        if global_overlays:
            sc_start_f = float(sc.get("start", 0.0))
            sc_end_f = float(sc.get("end", 0.0))
            for g_ov in global_overlays:
                if any(ov.get("id") == g_ov.get("id") for ov in overlays):
                    continue
                if max(float(g_ov.get("start", 0.0)), sc_start_f) < min(float(g_ov.get("end", 0.0)), sc_end_f):
                    overlays.append(g_ov)
        sc_dur = float(sc["end"]) - float(sc["start"])
        sc_frames = int(round(sc_dur * fps))

        inputs: list[str] = []
        filter_parts: list[str] = []

        # 1. Inputs and filters for shots
        if not shots:
            # Fallback black canvas if no shots
            filter_parts.append(f"color=c=navy:s={tw}x{th}:d={sc_dur:.3f}:r={fps}[sc_bg];")
        else:
            for idx, shot in enumerate(shots):
                shot_rel = shot.get("file")
                shot_path = (self.run_dir / shot_rel) if shot_rel else None
                has_file = shot_path is not None and shot_path.exists()
                is_image = has_file and (
                    shot.get("type") in ("image", "card")
                    or str(shot_path).lower().endswith((".jpg", ".jpeg", ".png"))
                )
                if not has_file:
                    # Generate solid color fallback for missing shot
                    shot_path_str = f"color=c=navy:s={tw}x{th}:d={shot.get('duration', 5.0):.3f}:r={fps}"
                    inputs.extend(["-f", "lavfi", "-i", shot_path_str])
                else:
                    if is_image:
                        inputs.extend(["-loop", "1", "-i", str(shot_path)])
                    else:
                        inputs.extend(["-stream_loop", "-1", "-i", str(shot_path)])

                shot_dur = float(shot.get("duration", 5.0))
                shot_frames = max(1, int(round(shot_dur * fps)))

                if is_image:
                    still_zoom_cfg = self.style_config.get("transitions", {}).get("still_zoom", {})
                    slow_zoom_enabled = bool(still_zoom_cfg.get("slow_zoom", True))
                    start_scale = float(still_zoom_cfg.get("start_scale", 1.00))
                    end_scale = float(still_zoom_cfg.get("end_scale", 1.08))
                    motion_name = shot.get("motion")

                    # Supersampled canvas (1.25x, guaranteed even) to eliminate sub-pixel jitter
                    stw = int(round(tw * 1.25 / 2.0)) * 2
                    sth = int(round(th * 1.25 / 2.0)) * 2

                    if slow_zoom_enabled and shot_frames > 1 and motion_name != "none":
                        scale_delta = end_scale - start_scale
                        denom = max(1, shot_frames - 1)
                        if motion_name == "zoom_out":
                            z_expr = f"{end_scale:.4f}-{scale_delta:.4f}*(on/{denom})"
                        else:
                            z_expr = f"{start_scale:.4f}+{scale_delta:.4f}*(on/{denom})"
                    else:
                        z_expr = f"{start_scale:.4f}"

                    # Variety pan anchors
                    if idx % 3 == 1:
                        y_anchor = "ih/2.3-(ih/zoom/2)"
                        x_anchor = "iw/2-(iw/zoom/2)"
                    elif idx % 3 == 2:
                        y_anchor = "ih/1.7-(ih/zoom/2)"
                        x_anchor = "iw/2-(iw/zoom/2)"
                    else:
                        y_anchor = "ih/2-(ih/zoom/2)"
                        x_anchor = "iw/2-(iw/zoom/2)"

                    filter_parts.append(
                        f"[{idx}:v]scale={stw}:{sth}:force_original_aspect_ratio=decrease,pad=max(iw\\,{stw}):max(ih\\,{sth}):(ow-iw)/2:(oh-ih)/2,setsar=1,"
                        f"zoompan=z='{z_expr}':x='{x_anchor}':y='{y_anchor}':d={shot_frames}:s={stw}x{sth}:fps={fps},"
                        f"scale={tw}:{th}:flags=lanczos,"
                        f"trim=duration={shot_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[shot_{idx}];"
                    )
                else:
                    filter_parts.append(
                        f"[{idx}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,pad=max(iw\\,{tw}):max(ih\\,{th}):(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                        f"trim=duration={shot_dur:.3f},setpts=PTS-STARTPTS,format=yuv420p[shot_{idx}];"
                    )

            concat_inputs = "".join(f"[shot_{i}]" for i in range(len(shots)))
            filter_parts.append(f"{concat_inputs}concat=n={len(shots)}:v=1:a=0[sc_bg];")

        # 2. Overlay passes with optional preview scale factor support
        current_stream = "[sc_bg]"
        next_in_idx = len(shots) if shots else 0
        scale_x = float(scale_factor) if isinstance(scale_factor, (int, float)) else float(scale_factor[0])
        scale_y = float(scale_factor) if isinstance(scale_factor, (int, float)) else float(scale_factor[1])
        needs_scale = (abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3)

        for ov_idx, ov in enumerate(overlays):
            ov_path = self.run_dir / ov["png"]
            if not ov_path.exists():
                continue
            inputs.extend(["-loop", "1", "-i", str(ov_path)])
            in_idx = next_in_idx
            next_in_idx += 1

            rel_start = max(0.0, float(ov["start"]) - float(sc["start"]))
            rel_end = max(rel_start + 0.1, float(ov["end"]) - float(sc["start"]))
            ov_dur = rel_end - rel_start

            fade_in = max(0.05, float(ov.get("fade_in", 0.2)))
            fade_out = max(0.05, float(ov.get("fade_out", 0.2)))
            fade_in = min(fade_in, ov_dur / 2.0)
            fade_out = min(fade_out, ov_dur / 2.0)
            fade_out_st = max(0.0, ov_dur - fade_out)

            animation = ov.get("animation", "fade")
            ox = int(ov.get("x", 0) * scale_x)
            oy = int(ov.get("y", 0) * scale_y)

            # 1. Prepare RGBA stream with alpha fades, optional scaling, and PTS shift
            anim_stream = f"[anim_ov_{ov_idx}]"
            scale_filter = f"scale=iw*{scale_x:.4f}:ih*{scale_y:.4f}," if needs_scale else ""
            filter_parts.append(
                f"[{in_idx}:v]format=rgba,{scale_filter}trim=duration={ov_dur:.3f},fps={fps},setpts=PTS-STARTPTS,"
                f"fade=t=in:st=0:d={fade_in:.3f}:alpha=1,"
                f"fade=t=out:st={fade_out_st:.3f}:d={fade_out:.3f}:alpha=1,"
                f"setpts=PTS+{rel_start:.3f}/TB{anim_stream};"
            )

            # 2. Clamped cubic ease-out position interpolations
            if animation == "slide_up":
                slide_dist = int(40 * scale_y)
                y_expr = f"{oy}+{slide_dist}*pow(1-min(max((t-{rel_start:.3f})/{fade_in:.3f},0),1),3)"
                x_expr = str(ox)
            elif animation == "slide_right":
                slide_dist = int(50 * scale_x)
                x_expr = f"{ox}-{slide_dist}*pow(1-min(max((t-{rel_start:.3f})/{fade_in:.3f},0),1),3)"
                y_expr = str(oy)
            else:
                x_expr = str(ox)
                y_expr = str(oy)

            next_stream = f"[ov_{ov_idx}]"
            filter_parts.append(
                f"{current_stream}{anim_stream}overlay=x='{x_expr}':y='{y_expr}':eof_action=pass:"
                f"enable='between(t,{rel_start:.3f},{rel_end:.3f})'{next_stream};"
            )
            current_stream = next_stream

        # 2.5 Burn scene-local ASS subtitles if present
        if scene_ass_path and scene_ass_path.exists():
            ass_escaped = scene_ass_path.resolve().as_posix().replace(":", "\\:")
            sub_stream = f"[subbed_{sc.get('id', 'sc')}]"
            filter_parts.append(f"{current_stream}ass='{ass_escaped}'{sub_stream};")
            current_stream = sub_stream

        # 3. Transitions (reads resolved transition numbers from plan)
        trans_in = sc.get("transition_in")
        trans_out = sc.get("transition_out")

        if not trans_in and not trans_out:
            trans = sc.get("transition", {})
            trans_type = trans.get("type", "cut")
            trans_dur = float(trans.get("duration_sec", 0.0))
            fade_out = float(trans.get("fade_out_sec", trans_dur / 2.0))
            fade_in = float(trans.get("fade_in_sec", trans_dur / 2.0))
            tin_type = trans_type if trans_dur > 0 else "cut"
            tin_dur = fade_in
            tout_type = trans_type if trans_dur > 0 else "cut"
            tout_dur = fade_out
        else:
            tin_type = trans_in.get("type", "cut") if trans_in else "cut"
            tin_dur = float(trans_in.get("duration_sec", 0.0)) if trans_in else 0.0
            tout_type = trans_out.get("type", "cut") if trans_out else "cut"
            tout_dur = float(trans_out.get("duration_sec", 0.0)) if trans_out else 0.0

        f_filters: list[str] = []
        if tin_type == "dip_to_black" and tin_dur > 0:
            f_filters.append(f"fade=t=in:st=0:d={tin_dur:.3f}")
        elif tin_type == "fade_white" and tin_dur > 0:
            f_filters.append(f"fade=t=in:st=0:d={tin_dur:.3f}:color=white")

        if tout_type == "dip_to_black" and tout_dur > 0:
            st_out = max(0.0, sc_dur - tout_dur)
            f_filters.append(f"fade=t=out:st={st_out:.3f}:d={tout_dur:.3f}")
        elif tout_type == "fade_white" and tout_dur > 0:
            st_out = max(0.0, sc_dur - tout_dur)
            f_filters.append(f"fade=t=out:st={st_out:.3f}:d={tout_dur:.3f}:color=white")

        if f_filters:
            filter_parts.append(f"{current_stream}{','.join(f_filters)},scale=in_range=auto:out_range=limited,format=yuv420p[outv]")
        else:
            filter_parts.append(f"{current_stream}scale=in_range=auto:out_range=limited,format=yuv420p[outv]")

        filter_graph = "".join(filter_parts)

        if check_nvenc_available():
            nv_preset = "p1" if (preset in ("ultrafast", "superfast", "veryfast") or crf >= 28) else "p4"
            nv_cq = 28 if crf >= 28 else 20
            v_codec_args = ["-c:v", "h264_nvenc", "-preset", nv_preset, "-rc", "vbr", "-cq", str(nv_cq), "-b:v", "0", "-pix_fmt", "yuv420p"]
        else:
            v_codec_args = ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex",
                filter_graph,
                "-map",
                "[outv]",
            ]
            + v_codec_args
            + [
                "-r",
                str(fps),
                "-vframes",
                str(sc_frames),
                str(output_path),
            ]
        )

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            self.logger.error(f"FFmpeg error rendering scene '{sc['id']}': {res.stderr[-1000:]}")
            raise RuntimeError(f"FFmpeg scene render failed for '{sc['id']}': {res.stderr[-500:]}")

    def _generate_description_markdown(
        self,
        script_data: dict[str, Any],
        media_data: dict[str, Any],
        timeline_data: dict[str, Any],
        output_path: Path,
        manifest: dict[str, Any] | None = None,
        render_plan: dict[str, Any] | None = None,
    ) -> None:
        """Generates publish pack render/description.md with chapters, sources, and credits."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
        date_str = script_data.get("date", datetime.now().strftime("%Y-%m-%d"))

        try:
            dt = datetime.fromisoformat(date_str)
            date_display = dt.strftime("%A, %B ") + str(dt.day) + dt.strftime(", %Y")
        except Exception:
            date_display = date_str

        lines: list[str] = []

        # 1. Test run banner if test run
        if run_kind == "test":
            lines.append("TEST RUN - DO NOT PUBLISH")
            lines.append("")

        # 2. Suggested Title
        lines.append(f"# {channel_name} | {date_display}")
        lines.append("")

        # 3. YouTube Chapters
        lines.append("## Chapters")
        scenes = timeline_data.get("scenes", [])
        script_scenes = {s["id"]: s for s in script_data.get("scenes", [])}

        chapters: list[tuple[float, str]] = []
        last_ch_time = -100.0

        for sc in scenes:
            sc_id = sc["id"]
            sc_start = float(sc["start"])
            sc_script = script_scenes.get(sc_id, {})
            title = sc_script.get("headline", "")
            if sc_id == "intro":
                title = "Introduction"
            elif sc_id == "outro":
                title = "Summary & Credits"
            elif not title:
                title = f"Story {sc_id}"

            min_ch_gap = 1.0 if (len(scenes) <= 4 or float(timeline_data.get("total_duration", 0.0)) < 30.0) else 10.0
            if sc_id == "intro" or (sc_start - last_ch_time >= min_ch_gap):
                chapters.append((sc_start, title))
                last_ch_time = sc_start

        # Ensure first is exactly 00:00
        if not chapters or chapters[0][0] > 0.0:
            chapters.insert(0, (0.0, "Introduction"))

        for ch_start, ch_title in chapters:
            lines.append(f"{format_chapter_time(ch_start)} - {ch_title}")
        lines.append("")

        # 4. News Sources (only for scenes actually present in timeline)
        lines.append("## News Sources")
        tl_scene_ids = {sc["id"] for sc in scenes}
        for sc in script_data.get("scenes", []):
            if sc.get("type") == "story":
                if tl_scene_ids and sc.get("id") not in tl_scene_ids:
                    continue
                headline = sc.get("headline", "News Story")
                sources = sc.get("sources") or sc.get("source") or []
                src_label = ", ".join(sources) if isinstance(sources, list) else str(sources)
                paren = f" ({src_label})" if src_label else ""
                lines.append(f"- **{headline}**{paren}")
                for u in sc.get("source_urls", []):
                    lines.append(f"  - [{u}]({u})")
        lines.append("")

        # 5. Visual Credits (only for actually-used media)
        lines.append("## Visual Credits")
        shots = media_data.get("shots", [])

        # Filter shots to only actually-used media files if render_plan is provided
        used_file_names = set()
        if render_plan:
            for r_sc in render_plan.get("scenes", []):
                if r_sc.get("media"):
                    used_file_names.add(Path(r_sc["media"]).name)
                for sh in r_sc.get("shots", []):
                    if sh.get("file"):
                        used_file_names.add(Path(sh["file"]).name)

        if used_file_names:
            shots = [
                s for s in shots
                if Path(s.get("file", "")).name in used_file_names
                or Path(s.get("media_path", "")).name in used_file_names
                or str(s.get("id")) in used_file_names
            ]

        cc_shots = [s for s in shots if s.get("attribution_required")]
        if cc_shots:
            seen_credits: set[tuple[str, str, str]] = set()
            for s in cc_shots:
                attr = s.get("attribution_text") or f"File by {s.get('author')} via {s.get('source')}"
                lic = s.get("license", "CC")
                lic_url = s.get("license_url", "")
                pg_url = s.get("origin_url", "")
                credit_key = (attr, lic, pg_url or "")
                if credit_key in seen_credits:
                    continue
                seen_credits.add(credit_key)
                lines.append(f"- **{attr}** ([{lic}]({lic_url})) - [Source Page]({pg_url})")

        used_sources = {str(s.get("source", "")).lower().replace(" ", "_") for s in shots if s.get("source")}
        if "pixabay" in used_sources:
            lines.append("- Stock video and imagery courtesy of [Pixabay](https://pixabay.com).")
        if "pexels" in used_sources:
            lines.append("- Stock video and imagery courtesy of [Pexels](https://www.pexels.com).")
        if "unsplash" in used_sources:
            lines.append("- Photography courtesy of [Unsplash](https://unsplash.com).")
        if "nasa" in used_sources:
            lines.append("- Media and imagery courtesy of [NASA](https://images.nasa.gov).")
        if "coverr" in used_sources:
            lines.append("- Footage courtesy of [Coverr](https://coverr.co).")
        if "openverse" in used_sources:
            lines.append("- Open-licensed imagery sourced via [Openverse](https://openverse.org).")
        if "internet_archive" in used_sources:
            lines.append("- Historical and archive footage courtesy of [Internet Archive](https://archive.org).")
        if "wikimedia_commons" in used_sources:
            lines.append("- Media and imagery sourced via [Wikimedia Commons](https://commons.wikimedia.org).")
        if "user-provided" in used_sources or "local_library" in used_sources:
            lines.append("- Field footage and photography provided directly by the production team.")
        lines.append("")

        # 6. Disclaimers & Disclosures
        lines.append("## Disclaimers & Disclosures")
        lines.append("- Narration audio was synthesized using an AI voice model.")
        lines.append("- Stock and archive footage is illustrative.")
        lines.append("- Script is AI-assisted and based on the linked reports.")
        lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

        # 7. Short-form caption block for 9:16 vertical runs
        is_vertical = (
            getattr(self, "orientation", None) == "vertical"
            or (render_plan and render_plan.get("settings", {}).get("resolution") == [1080, 1920])
            or (render_plan and render_plan.get("aspect_ratio") == "9:16")
        )
        if is_vertical:
            hook_line = ""
            for sc in script_data.get("scenes", []):
                if sc.get("type") == "story" and sc.get("headline"):
                    hook_line = sc["headline"].strip()
                    break
            if not hook_line:
                hook_line = script_data.get("scenes", [{}])[0].get("headline", "Today's Nepal News")

            shorts_tags = [
                "#NepalNews", "#Kathmandu", "#NepalUpdates", "#Shorts",
                "#नेपालसमाचार", "#काठमाडौं", "#ताजासमाचार", "#नेपाल"
            ]
            shorts_content = f"{hook_line}\n\n{' '.join(shorts_tags)}\n"
            shorts_path = output_path.parent / "shorts_caption.txt"
            shorts_path.write_text(shorts_content, encoding="utf-8")

    def run(
        self,
        manifest: dict[str, Any] | None = None,
        preview: bool = False,
        only_scene: str | None = None,
    ) -> list[str]:
        """Renders final MP4 video and publish pack."""
        self.logger.info(f"Starting Render Agent run in: {self.run_dir} (preview={preview}, only_scene={only_scene})")
        t_start = time.perf_counter()

        # Gate check
        self._verify_gate()

        # 1. Inputs
        plan_file = self.run_dir / "motion" / "render_plan.json"
        if not plan_file.exists():
            plan_file = self.run_dir / "render_plan.json"
        if not plan_file.exists():
            raise FileNotFoundError(f"Missing required render_plan.json in {self.run_dir}")

        audio_file = self.run_dir / "audio" / "narration.wav"
        if not audio_file.exists():
            raise FileNotFoundError(f"Missing required audio/narration.wav in {self.run_dir}")

        script_file = self.run_dir / "script" / "script.json"
        timeline_file = self.run_dir / "timeline" / "timeline.json"
        media_file = self.run_dir / "media" / "media_manifest.json"

        with open(plan_file, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        with open(script_file, "r", encoding="utf-8") as f:
            script_data = json.load(f)
        with open(timeline_file, "r", encoding="utf-8") as f:
            timeline_data = json.load(f)
        with open(media_file, "r", encoding="utf-8") as f:
            media_data = json.load(f)

        # Ensure render_plan scene shots/sha256 match current media_manifest.json (e.g., after --apply-picks)
        sync_render_plan_data(
            plan_data=plan_data,
            media_data=media_data,
            run_dir=self.run_dir,
            script_data=script_data,
            style_config=self.style_config,
        )
        with open(plan_file, "w", encoding="utf-8") as pf:
            json.dump(plan_data, pf, indent=2)

        # Output folders
        render_dir = self.run_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = render_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Parameters
        fps = int(plan_data.get("fps", 30))
        plan_res = plan_data.get("resolution")
        if plan_res and isinstance(plan_res, str) and "x" in plan_res:
            parts = [int(p) for p in plan_res.split("x")]
            res_cfg = [parts[0], parts[1]]
        elif isinstance(plan_res, (list, tuple)) and len(plan_res) >= 2:
            res_cfg = [int(plan_res[0]), int(plan_res[1])]
        else:
            res_cfg = self.style_config.get("video", {}).get("resolution", [1920, 1080])

        crf = int(self.style_config.get("video", {}).get("crf", 20))
        preset = str(self.style_config.get("video", {}).get("preset", "veryfast"))

        if preview:
            prev_cfg = self.style_config.get("preview", {})
            crf = int(prev_cfg.get("crf", 30))
            preset = str(prev_cfg.get("preset", "ultrafast"))
            target_res = (int(res_cfg[0]) // 2, int(res_cfg[1]) // 2)
        else:
            target_res = (int(res_cfg[0]), int(res_cfg[1]))
        style_hash = hashlib.sha256(json.dumps(self.style_config, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        final_mp4_name = "final_preview.mp4" if preview else "final.mp4"
        final_mp4_path = render_dir / final_mp4_name

        # Master ASS path for scene subtitle slicing
        ass_rel_path = "motion/captions.ass"
        if not (self.run_dir / ass_rel_path).exists():
            ass_rel_path = plan_data.get("captions_ass", "motion/captions.ass")
        master_ass_path = self.run_dir / ass_rel_path

        # Step 1: Render or reuse cached scene segments
        scenes = plan_data.get("scenes", [])
        global_overlays = [ov for ov in plan_data.get("overlays", []) if ov.get("scene_id") is None]
        cache_hits = 0

        total_scenes = len(scenes)
        scene_segment_paths: list[Path] = [None] * total_scenes  # type: ignore[list-item]
        tasks_to_render = []

        for i, sc in enumerate(scenes, 1):
            sc_id = sc["id"]
            sc_start = float(sc.get("start", 0.0))
            sc_end = float(sc.get("end", 0.0))
            sc_ass_path = cache_dir / f"scene_{sc_id}_{target_res[0]}x{target_res[1]}.ass"
            ass_hash = slice_ass_for_scene(master_ass_path, sc_start, sc_end, sc_ass_path)
            sc_key = compute_scene_cache_key(sc, style_hash, extra_hash=ass_hash)
            cached_segment = cache_dir / f"scene_{sc_id}_{sc_key}_{target_res[0]}x{target_res[1]}.mp4"
            scene_segment_paths[i - 1] = cached_segment

            need_render = True
            if only_scene and sc_id != only_scene and cached_segment.exists() and cached_segment.stat().st_size > 1000:
                need_render = False
            elif not only_scene and cached_segment.exists() and cached_segment.stat().st_size > 1000:
                need_render = False

            if not need_render:
                self.logger.info(f"Reusing cached segment for scene '{sc_id}' ({cached_segment.name})")
                cache_hits += 1
            else:
                tasks_to_render.append((i, sc, cached_segment, sc_ass_path))

        if tasks_to_render:
            from concurrent.futures import ThreadPoolExecutor

            def _worker_render(item: tuple[int, dict[str, Any], Path, Path]) -> None:
                idx, sc_item, out_path, ass_path = item
                msg = f"Rendering scene {idx}/{total_scenes} ({sc_item['id']})"
                emit_office_event(
                    self.run_dir,
                    "substage_progress",
                    stage="render",
                    current=idx,
                    total=total_scenes,
                    message=msg,
                    extra={"scene_id": sc_item["id"]},
                )
                self.logger.info(f"Rendering segment for scene '{sc_item['id']}'...")
                self._render_scene_segment(
                    sc=sc_item,
                    fps=fps,
                    target_res=target_res,
                    output_path=out_path,
                    crf=crf,
                    preset=preset,
                    global_overlays=global_overlays,
                    scene_ass_path=ass_path,
                    scale_factor=0.5 if preview else 1.0,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(_worker_render, tasks_to_render))

        # Step 2: Concatenate segments into video_noaudio.mp4
        emit_office_event(
            self.run_dir,
            "substage_progress",
            stage="render",
            current=total_scenes,
            total=total_scenes,
            message=f"Concatenating {total_scenes} scene segments and muxing audio...",
            extra={"scene_id": "concat", "eta_seconds": 5},
        )
        concat_list_file = render_dir / "concat_segments.txt"
        concat_lines = [f"file '{p.resolve().as_posix()}'" for p in scene_segment_paths]
        concat_list_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        video_noaudio_path = render_dir / "video_noaudio.mp4"
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_file),
            "-c",
            "copy",
            str(video_noaudio_path),
        ]
        res = subprocess.run(concat_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg segment concatenation failed: {res.stderr[-500:]}")

        # Step 3: Final pass: Mux narration audio and BGM using single-pass copy (subtitles already burned)
        final_mp4_name = "final_preview.mp4" if preview else "final.mp4"
        final_mp4_path = render_dir / final_mp4_name

        # Check for background music and audio grid configuration
        audio_grid_cfg = plan_data.get("audio_grid", {})
        bgm_cand = audio_grid_cfg.get("bgm_file")
        if not bgm_cand or not Path(bgm_cand).exists():
            bgm_cand = self.style_config.get("music", {}).get("default_track")
        if not bgm_cand or not Path(bgm_cand).exists():
            for cand in [
                Path("config/audio/broadcast_theme.wav"),
                Path("config/audio/broadcast_theme.mp3"),
                self.run_dir / "audio" / "bg_track.mp3",
                self.run_dir / "audio" / "bg_track.wav",
            ]:
                if cand.exists():
                    bgm_cand = str(cand)
                    break

        music_enabled = bool(self.style_config.get("music", {}).get("enabled", False))
        has_bgm = music_enabled and (bgm_cand is not None) and Path(bgm_cand).exists()

        if has_bgm:
            bgm_vol = float(self.style_config.get("music", {}).get("volume", 0.15))
            final_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_noaudio_path.relative_to(self.run_dir)),
                "-i",
                str(audio_file.relative_to(self.run_dir)),
                "-stream_loop",
                "-1",
                "-i",
                str(Path(bgm_cand).resolve()),
                "-filter_complex",
                (
                    f"[2:a]volume={bgm_vol:.3f}[bgm_base];"
                    f"[bgm_base][1:a]sidechaincompress=threshold=0.03:ratio=4:attack=50:release=350[ducked_bgm];"
                    f"[1:a][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]"
                ),
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(final_mp4_path.relative_to(self.run_dir)),
            ]
        else:
            final_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_noaudio_path.relative_to(self.run_dir)),
                "-i",
                str(audio_file.relative_to(self.run_dir)),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(final_mp4_path.relative_to(self.run_dir)),
            ]

        t_encode_start = time.perf_counter()
        res = subprocess.run(final_cmd, cwd=self.run_dir, capture_output=True, text=True)
        t_encode_end = time.perf_counter()

        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg final assembly failed: {res.stderr[-500:]}")

        # Also place root final.mp4 for convenience / contract compatibility
        root_final = self.run_dir / final_mp4_name
        try:
            if root_final.exists():
                root_final.unlink()
            os.link(final_mp4_path, root_final)
        except Exception:
            shutil.copy2(final_mp4_path, root_final)

        # Step 4: Generate publish pack description.md
        desc_path = render_dir / "description.md"
        self._generate_description_markdown(
            script_data=script_data,
            media_data=media_data,
            timeline_data=timeline_data,
            output_path=desc_path,
            manifest=manifest,
            render_plan=plan_data,
        )

        t_total = time.perf_counter() - t_start
        total_audio_dur = float(timeline_data.get("duration_sec", 1.0))
        encode_speed = round(total_audio_dur / max(0.001, t_total), 2)

        self.logger.info(
            f"Render Agent finished: {final_mp4_name} ({final_mp4_path.stat().st_size / (1024*1024):.2f} MB) "
            f"in {t_total:.2f}s (speed: {encode_speed}x, cache hits: {cache_hits}/{len(scenes)})"
        )

        outputs = [
            f"render/{final_mp4_name}",
            "render/description.md",
            final_mp4_name,
        ]
        shorts_path = render_dir / "shorts_caption.txt"
        if shorts_path.exists():
            outputs.append("render/shorts_caption.txt")

        # Update manifest metrics if manifest provided
        if manifest:
            stage_meta = manifest.setdefault("stages", {}).setdefault("render", {})
            stage_meta.setdefault("metrics", {})
            stage_meta["metrics"]["wall_time_sec"] = round(t_total, 2)
            stage_meta["metrics"]["encode_speed"] = encode_speed
            stage_meta["metrics"]["cache_hits"] = cache_hits
            stage_meta["metrics"]["total_scenes"] = len(scenes)
            stage_meta["metrics"]["preview_mode"] = preview

        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nepal News Bot Render Agent.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Run date in YYYY-MM-DD")
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs directory")
    parser.add_argument("--preview", action="store_true", help="Fast low-res preview render (960x540, CRF 30)")
    parser.add_argument("--scene", default=None, help="Re-render only the specified scene id")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.date
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    agent = RenderAgent(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {}

    outputs = agent.run(manifest, preview=args.preview, only_scene=args.scene)
    if manifest_path.exists():
        save_manifest(manifest, manifest_path)

    print(f"[OK] Render Agent completed successfully.")
    for o in outputs:
        print(f" - {run_dir / o}")


if __name__ == "__main__":
    main()

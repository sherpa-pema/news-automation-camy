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
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest

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


def compute_scene_cache_key(sc: dict[str, Any], style_hash: str) -> str:
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
            }
            for ov in sc.get("overlays", [])
        ],
        "transition": sc.get("transition"),
        "style_hash": style_hash,
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
    ) -> None:
        """Builds a frame-exact silent MP4 segment for a single scene with shots, dips, and overlays."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tw, th = target_res
        shots = sc.get("shots", [])
        overlays = sc.get("overlays", [])
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
                shot_path = self.run_dir / shot["file"]
                if not shot_path.exists():
                    # Generate solid color fallback for missing shot
                    shot_path_str = f"color=c=navy:s={tw}x{th}:d={shot.get('duration', 5.0):.3f}:r={fps}"
                    inputs.extend(["-f", "lavfi", "-i", shot_path_str])
                else:
                    inputs.extend(["-i", str(shot_path)])

                shot_dur = float(shot.get("duration", 5.0))
                shot_frames = int(round(shot_dur * fps))
                is_image = shot.get("type") == "image" or str(shot_path).lower().endswith((".jpg", ".jpeg", ".png"))

                if is_image:
                    filter_parts.append(
                        f"[{idx}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                        f"zoompan=z='min(zoom+0.0006,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={shot_frames}:s={tw}x{th}:fps={fps},"
                        f"trim=duration={shot_dur:.3f},setpts=PTS-STARTPTS[shot_{idx}];"
                    )
                else:
                    filter_parts.append(
                        f"[{idx}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
                        f"trim=duration={shot_dur:.3f},setpts=PTS-STARTPTS[shot_{idx}];"
                    )

            concat_inputs = "".join(f"[shot_{i}]" for i in range(len(shots)))
            filter_parts.append(f"{concat_inputs}concat=n={len(shots)}:v=1:a=0[sc_bg];")

        # 2. Overlay passes
        current_stream = "[sc_bg]"
        shot_input_count = len(shots) if shots else 1

        for ov_idx, ov in enumerate(overlays):
            ov_path = self.run_dir / ov["png"]
            if not ov_path.exists():
                continue
            inputs.extend(["-i", str(ov_path)])
            in_idx = shot_input_count + ov_idx

            rel_start = max(0.0, float(ov["start"]) - float(sc["start"]))
            rel_end = max(rel_start + 0.1, float(ov["end"]) - float(sc["start"]))
            ox = int(ov.get("x", 0))
            oy = int(ov.get("y", 0))

            next_stream = f"[ov_{ov_idx}]"
            filter_parts.append(
                f"{current_stream}[{in_idx}:v]overlay=x={ox}:y={oy}:enable='between(t,{rel_start:.3f},{rel_end:.3f})'{next_stream};"
            )
            current_stream = next_stream

        # 3. Dip to black (0.1s in, 0.1s out)
        filter_parts.append(f"{current_stream}fade=t=in:st=0:d=0.1,fade=t=out:st={max(0.1, sc_dur - 0.1):.3f}:d=0.1[outv]")

        filter_graph = "".join(filter_parts)

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex",
                filter_graph,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
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
    ) -> None:
        """Generates publish pack render/description.md with chapters, sources, and credits."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        channel_name = self.channel_config.get("channel_name", "Nepal Daily Briefing")
        run_kind = script_data.get("run_kind", "production")
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

            if sc_id == "intro" or (sc_start - last_ch_time >= 10.0):
                chapters.append((sc_start, title))
                last_ch_time = sc_start

        # Ensure first is exactly 00:00
        if not chapters or chapters[0][0] > 0.0:
            chapters.insert(0, (0.0, "Introduction"))

        for ch_start, ch_title in chapters:
            lines.append(f"{format_chapter_time(ch_start)} - {ch_title}")
        lines.append("")

        # 4. News Sources
        lines.append("## News Sources")
        for sc in script_data.get("scenes", []):
            if sc.get("type") == "story":
                headline = sc.get("headline", "News Story")
                sources = sc.get("sources") or sc.get("source") or []
                src_label = ", ".join(sources) if isinstance(sources, list) else str(sources)
                paren = f" ({src_label})" if src_label else ""
                lines.append(f"- **{headline}**{paren}")
                for u in sc.get("source_urls", []):
                    lines.append(f"  - [{u}]({u})")
        lines.append("")

        # 5. Visual Credits
        lines.append("## Visual Credits")
        shots = media_data.get("shots", [])
        cc_shots = [s for s in shots if s.get("attribution_required")]
        if cc_shots:
            for s in cc_shots:
                attr = s.get("attribution_text") or f"File by {s.get('author')} via {s.get('source')}"
                lic = s.get("license", "CC")
                lic_url = s.get("license_url", "")
                pg_url = s.get("origin_url", "")
                lines.append(f"- **{attr}** ([{lic}]({lic_url})) - [Source Page]({pg_url})")
        lines.append("- Stock video and imagery courtesy of [Pixabay](https://pixabay.com).")
        lines.append("")

        # 6. Disclaimers & Disclosures
        lines.append("## Disclaimers & Disclosures")
        lines.append("- Narration audio was synthesized using an AI voice model.")
        lines.append("- Stock and archive footage is illustrative.")
        lines.append("- Script is AI-assisted and based on the linked reports.")
        lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

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

        # Output folders
        render_dir = self.run_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = render_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Parameters
        fps = int(plan_data.get("fps", 30))
        res_cfg = self.style_config.get("video", {}).get("resolution", [1920, 1080])
        crf = int(self.style_config.get("video", {}).get("crf", 20))
        preset = str(self.style_config.get("video", {}).get("preset", "veryfast"))

        if preview:
            prev_cfg = self.style_config.get("preview", {})
            res_cfg = prev_cfg.get("resolution", [960, 540])
            crf = int(prev_cfg.get("crf", 30))
            preset = str(prev_cfg.get("preset", "ultrafast"))

        target_res = (int(res_cfg[0]), int(res_cfg[1]))
        style_hash = hashlib.sha256(json.dumps(self.style_config, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        # Step 1: Render or reuse cached scene segments
        scene_segment_paths: list[Path] = []
        scenes = plan_data.get("scenes", [])
        cache_hits = 0

        for sc in scenes:
            sc_id = sc["id"]
            sc_key = compute_scene_cache_key(sc, style_hash)
            cached_segment = cache_dir / f"scene_{sc_id}_{sc_key}_{target_res[0]}x{target_res[1]}.mp4"

            need_render = True
            if only_scene and sc_id != only_scene and cached_segment.exists() and cached_segment.stat().st_size > 1000:
                need_render = False
            elif not only_scene and cached_segment.exists() and cached_segment.stat().st_size > 1000:
                need_render = False

            if not need_render:
                self.logger.info(f"Reusing cached segment for scene '{sc_id}' ({cached_segment.name})")
                cache_hits += 1
            else:
                self.logger.info(f"Rendering segment for scene '{sc_id}'...")
                self._render_scene_segment(
                    sc=sc,
                    fps=fps,
                    target_res=target_res,
                    output_path=cached_segment,
                    crf=crf,
                    preset=preset,
                )

            scene_segment_paths.append(cached_segment)

        # Step 2: Concatenate segments into video_noaudio.mp4
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

        # Step 3: Final pass: Burn captions.ass and mux narration audio
        final_mp4_name = "final_preview.mp4" if preview else "final.mp4"
        final_mp4_path = render_dir / final_mp4_name

        ass_rel_path = "motion/captions.ass"
        if not (self.run_dir / ass_rel_path).exists():
            ass_rel_path = plan_data.get("captions_ass", "motion/captions.ass")

        # Run ffmpeg with cwd set to self.run_dir to prevent Windows libass escaping issues
        final_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_noaudio_path.relative_to(self.run_dir)),
            "-i",
            str(audio_file.relative_to(self.run_dir)),
            "-vf",
            f"ass={ass_rel_path}",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
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
        shutil.copy2(final_mp4_path, root_final)

        # Step 4: Generate publish pack description.md
        desc_path = render_dir / "description.md"
        self._generate_description_markdown(
            script_data=script_data,
            media_data=media_data,
            timeline_data=timeline_data,
            output_path=desc_path,
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

"""QA Agent: automated pre-publish inspection of final.mp4 and pipeline artifacts."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from PIL import Image, ImageDraw, ImageFont

from src.agents.base import BaseAgent
from src.utils.editor_validator import (
    compute_validator_fingerprint,
    extract_script_validation_fingerprint,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest
from src.utils.storyboard import generate_storyboard

logger = logging.getLogger("qa")


def probe_file(file_path: Path) -> dict[str, Any]:
    """Inspects media file container and stream properties using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}: {res.stderr}")
    return json.loads(res.stdout)


class QAAgent(BaseAgent):
    """QA Agent validates final video quality, synchronization, loudness, attribution, and generates contact sheet."""

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        super().__init__(run_dir, config)
        self.logger = setup_logger("qa_agent", self.run_dir / "run.log")

    @property
    def name(self) -> str:
        return "qa"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/qa_report.schema.json"

    def _generate_contact_sheet(
        self,
        final_mp4: Path,
        timeline_scenes: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Extracts representative frames from each scene and stitches into qa/contact_sheet.jpg."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = self.run_dir / "qa" / "_tmp_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[tuple[str, Path]] = []
        for sc in timeline_scenes:
            sc_id = sc["id"]
            sc_start = float(sc["start"])
            sc_end = float(sc["end"])

            # 1. Frame ~1s into scene
            t1 = round(sc_start + 1.0, 2)
            if t1 >= sc_end:
                t1 = round((sc_start + sc_end) / 2.0, 2)

            # 2. Frame near end of scene
            t2 = round(max(t1 + 0.5, sc_end - 0.5), 2)

            for label, t in [(f"{sc_id} @ {t1}s", t1), (f"{sc_id} @ {t2}s", t2)]:
                f_path = tmp_dir / f"frame_{sc_id}_{int(t*100):06d}.jpg"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(t),
                    "-i",
                    str(final_mp4),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(f_path),
                ]
                subprocess.run(cmd, capture_output=True)
                if f_path.exists():
                    frame_paths.append((label, f_path))

        if not frame_paths:
            return

        # Stitch into grid (2 columns wide, N rows)
        thumb_w, thumb_h = 640, 360
        cols = 2
        rows = (len(frame_paths) + cols - 1) // cols
        sheet_w = cols * thumb_w + (cols + 1) * 20
        sheet_h = rows * (thumb_h + 30) + 40

        sheet_img = Image.new("RGB", (sheet_w, sheet_h), (11, 31, 58))
        draw = ImageDraw.Draw(sheet_img)
        font = ImageFont.load_default()

        for idx, (label, f_path) in enumerate(frame_paths):
            c = idx % cols
            r = idx // cols
            x = 20 + c * (thumb_w + 20)
            y = 20 + r * (thumb_h + 30)

            try:
                with Image.open(f_path) as im:
                    im_thumb = im.resize((thumb_w, thumb_h))
                    sheet_img.paste(im_thumb, (x, y))
            except Exception:
                pass

            draw.text((x + 6, y + thumb_h + 6), label, font=font, fill=(245, 166, 35))

        sheet_img.save(str(output_path), "JPEG", quality=85)

        # Clean up temporary frames
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def run(self, manifest: dict[str, Any] | None = None) -> list[str]:
        """Runs the deterministic QA check battery on final video and outputs qa_report.json."""
        self.logger.info(f"Starting QA Agent run in: {self.run_dir}")

        qa_dir = self.run_dir / "qa"
        qa_dir.mkdir(parents=True, exist_ok=True)

        final_mp4 = self.run_dir / "render" / "final.mp4"
        if not final_mp4.exists():
            final_mp4 = self.run_dir / "final.mp4"
        if not final_mp4.exists():
            raise FileNotFoundError(f"Missing required video file: {final_mp4}")

        narration_wav = self.run_dir / "audio" / "narration.wav"
        if not narration_wav.exists():
            raise FileNotFoundError(f"Missing required reference audio: {narration_wav}")

        render_plan_file = self.run_dir / "motion" / "render_plan.json"
        if not render_plan_file.exists():
            render_plan_file = self.run_dir / "render_plan.json"

        script_file = self.run_dir / "script" / "script.json"
        timeline_file = self.run_dir / "timeline" / "timeline.json"
        if not timeline_file.exists():
            timeline_file = self.run_dir / "timeline.json"
        captions_file = self.run_dir / "timeline" / "captions.json"
        if not captions_file.exists():
            captions_file = self.run_dir / "captions.json"
        ass_file = self.run_dir / "motion" / "captions.ass"
        media_file = self.run_dir / "media" / "media_manifest.json"
        desc_file = self.run_dir / "render" / "description.md"
        voice_report_file = self.run_dir / "audio" / "voice_report.json"

        checks: list[dict[str, Any]] = []
        warnings: list[str] = []
        redo_stage: str | None = None
        metrics: dict[str, Any] = {}

        # ----------------------------------------------------------------------
        # Check 1: Provenance and approval fingerprint
        # ----------------------------------------------------------------------
        provenance_passed = True
        prov_detail = "Script origin editor, approval SHA-256 and validator fingerprint match."
        try:
            with open(script_file, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            if s_data.get("origin") != "fixture":
                app_file = self.run_dir / "script" / "approval.json"
                if not app_file.exists():
                    provenance_passed = False
                    prov_detail = "Missing approval.json for editor script"
                else:
                    with open(app_file, "r", encoding="utf-8") as af:
                        app_data = json.load(af)
                    cur_h = hashlib.sha256(script_file.read_bytes()).hexdigest()
                    if app_data.get("script_sha256") != cur_h:
                        provenance_passed = False
                        prov_detail = f"Script SHA-256 mismatch (approval: {app_data.get('script_sha256')[:8]}, current: {cur_h[:8]})"
                    cur_fp = extract_script_validation_fingerprint(compute_validator_fingerprint())
                    app_fp = extract_script_validation_fingerprint(app_data.get("validator_fingerprint"))
                    if app_fp != cur_fp:
                        provenance_passed = False
                        prov_detail = "Validator fingerprint mismatch between approval and environment"
        except Exception as e:
            provenance_passed = False
            prov_detail = f"Provenance check exception: {e}"

        checks.append({"name": "provenance", "passed": provenance_passed, "detail": prov_detail})
        if not provenance_passed and not redo_stage:
            redo_stage = "editor"

        # ----------------------------------------------------------------------
        # Check 2: Final MP4 stream metadata and duration sync
        # ----------------------------------------------------------------------
        stream_passed = True
        stream_details: list[str] = []
        try:
            probe_vid = probe_file(final_mp4)
            probe_aud = probe_file(narration_wav)

            v_streams = [s for s in probe_vid.get("streams", []) if s.get("codec_type") == "video"]
            a_streams = [s for s in probe_vid.get("streams", []) if s.get("codec_type") == "audio"]

            if not v_streams:
                stream_passed = False
                stream_details.append("No video stream found in final.mp4")
            else:
                vs = v_streams[0]
                codec_v = vs.get("codec_name")
                w = int(vs.get("width", 0))
                h = int(vs.get("height", 0))
                if codec_v != "h264":
                    stream_passed = False
                    stream_details.append(f"Expected H.264 video, got {codec_v}")
                if w != 1920 or h != 1080:
                    stream_passed = False
                    stream_details.append(f"Expected 1920x1080 resolution, got {w}x{h}")

            if not a_streams:
                stream_passed = False
                stream_details.append("No audio stream found in final.mp4")
            else:
                as_ = a_streams[0]
                codec_a = as_.get("codec_name")
                ar = int(as_.get("sample_rate", 0))
                ch = int(as_.get("channels", 0))
                if codec_a != "aac":
                    stream_passed = False
                    stream_details.append(f"Expected AAC audio, got {codec_a}")
                if ar != 48000:
                    stream_passed = False
                    stream_details.append(f"Expected 48000 Hz sample rate, got {ar}")
                if ch != 2:
                    stream_passed = False
                    stream_details.append(f"Expected 2 audio channels (stereo), got {ch}")

            vid_dur = float(probe_vid.get("format", {}).get("duration", 0.0))
            aud_dur = float(probe_aud.get("format", {}).get("duration", 0.0))
            diff_ms = abs(vid_dur - aud_dur) * 1000.0

            metrics["video_duration_sec"] = round(vid_dur, 3)
            metrics["audio_duration_sec"] = round(aud_dur, 3)
            metrics["duration_diff_ms"] = round(diff_ms, 1)

            if diff_ms > 50.0:
                stream_passed = False
                stream_details.append(f"Duration mismatch: video {vid_dur:.3f}s vs audio {aud_dur:.3f}s (diff: {diff_ms:.1f}ms > 50ms limit)")
            else:
                stream_details.append(f"Duration sync within 50ms (diff: {diff_ms:.1f}ms, video: {vid_dur:.3f}s, audio: {aud_dur:.3f}s)")

        except Exception as e:
            stream_passed = False
            stream_details.append(f"Stream inspection error: {e}")

        checks.append({
            "name": "final_mp4_streams",
            "passed": stream_passed,
            "detail": "; ".join(stream_details) if stream_details else "All stream checks passed",
        })
        if not stream_passed and not redo_stage:
            redo_stage = "render"

        # ----------------------------------------------------------------------
        # Check 3: Audio loudness compliance (EBU R128)
        # ----------------------------------------------------------------------
        loudness_passed = True
        loudness_detail = ""
        try:
            ebur_cmd = ["ffmpeg", "-i", str(final_mp4), "-af", "ebur128=peak=true", "-f", "null", "-"]
            ebur_res = subprocess.run(ebur_cmd, capture_output=True, text=True)

            m_i = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+|-\s*inf|\+?\w+)\s+LUFS", ebur_res.stderr)
            m_tp = re.search(r"True peak:\s+Peak:\s+([-\d.]+|-\s*inf|\+?\w+)\s+dBFS", ebur_res.stderr)

            if m_i and m_tp:
                val_i = m_i.group(1).strip()
                val_tp = m_tp.group(1).strip()
                meas_lufs = -70.0 if "inf" in val_i else float(val_i)
                meas_tp = -99.0 if "inf" in val_tp else float(val_tp)
                metrics["integrated_loudness_lufs"] = meas_lufs
                metrics["true_peak_db"] = meas_tp

                # Target comparison with voice report
                if voice_report_file.exists():
                    target_lufs = -18.0
                    try:
                        with open(voice_report_file, "r", encoding="utf-8") as vf:
                            v_rep = json.load(vf)
                        target_lufs = float(v_rep.get("loudness_after_lufs", -18.0))
                    except Exception:
                        pass

                    lufs_diff = abs(meas_lufs - target_lufs)
                    if lufs_diff > 1.5:
                        loudness_passed = False
                        loudness_detail = f"Integrated loudness {meas_lufs:.1f} LUFS exceeds target {target_lufs:.1f} by {lufs_diff:.1f} LU (> 1.5 LU limit)"
                    elif meas_tp > -1.0:
                        loudness_passed = False
                        loudness_detail = f"True peak {meas_tp:.1f} dBTP exceeds ceiling -1.0 dBTP"
                    else:
                        loudness_detail = f"Integrated loudness {meas_lufs:.1f} LUFS (target: {target_lufs:.1f}), True peak {meas_tp:.1f} dBTP (<= -1.0 dBTP)"
                else:
                    warnings.append("Voice report audio/voice_report.json missing; skipped loudness target comparison")
                    if meas_tp > -1.0:
                        loudness_passed = False
                        loudness_detail = f"True peak {meas_tp:.1f} dBTP exceeds ceiling -1.0 dBTP (voice_report.json missing)"
                    else:
                        loudness_passed = True
                        loudness_detail = f"Measured {meas_lufs:.1f} LUFS, True peak {meas_tp:.1f} dBTP (skipped target check: missing voice_report.json)"
            else:
                loudness_passed = False
                loudness_detail = "Failed to parse EBU R128 loudness metrics from ffmpeg output"
        except Exception as e:
            loudness_passed = False
            loudness_detail = f"Loudness inspection error: {e}"

        checks.append({"name": "loudness", "passed": loudness_passed, "detail": loudness_detail})
        if not loudness_passed and not redo_stage:
            # If voice synthesis itself was already compliant, the attenuation/gain shift occurred in render stage
            voice_was_ok = False
            if voice_report_file.exists():
                try:
                    with open(voice_report_file, "r", encoding="utf-8") as vf:
                        v_rep = json.load(vf)
                    v_lufs = float(v_rep.get("loudness_after_lufs", -99.0))
                    if abs(v_lufs - (-18.0)) <= 1.5:
                        voice_was_ok = True
                except Exception:
                    pass
            redo_stage = "render" if voice_was_ok else "voice"

        # ----------------------------------------------------------------------
        # Check 4: Black frame detection
        # ----------------------------------------------------------------------
        black_passed = True
        black_detail = ""
        try:
            b_cmd = ["ffmpeg", "-i", str(final_mp4), "-vf", "blackdetect=d=0.5:pic_th=0.98", "-f", "null", "-"]
            b_res = subprocess.run(b_cmd, capture_output=True, text=True)

            black_hits = re.findall(r"black_start:([-\d.]+)\s+black_end:([-\d.]+)\s+black_duration:([-\d.]+)", b_res.stderr)
            violating_black = []
            vid_end = metrics.get("video_duration_sec", 120.0)

            for b_start_s, b_end_s, b_dur_s in black_hits:
                b_start = float(b_start_s)
                b_end = float(b_end_s)
                b_dur = float(b_dur_s)
                # Unintended black segment if:
                # 1. Total duration of black segment exceeds 0.5s inside video
                # 2. Or video has black stretch > 1.0s (dips are only 0.2s)
                if (b_start > 0.5 and b_end < (vid_end - 0.5)) or b_dur > 1.0:
                    violating_black.append((b_start, b_end, b_dur))

            if violating_black:
                black_passed = False
                black_detail = f"Detected unexpected black stretches: {violating_black}"
            else:
                black_detail = f"No unintended black segments > 0.5s detected ({len(black_hits)} intentional cuts/boundaries)"
        except Exception as e:
            black_passed = False
            black_detail = f"Blackdetect error: {e}"

        checks.append({"name": "blackdetect", "passed": black_passed, "detail": black_detail})
        if not black_passed and not redo_stage:
            redo_stage = "render"

        # ----------------------------------------------------------------------
        # Check 5: Freeze detection (reported as warnings)
        # ----------------------------------------------------------------------
        freeze_events: list[dict[str, float]] = []
        try:
            fz_cmd = ["ffmpeg", "-i", str(final_mp4), "-vf", "freezedetect=n=0.003:d=3.0", "-f", "null", "-"]
            fz_res = subprocess.run(fz_cmd, capture_output=True, text=True)
            freeze_starts = re.findall(r"lavfi\.freezedetect\.freeze_start:\s+([-\d.]+)", fz_res.stderr)
            freeze_ends = re.findall(r"lavfi\.freezedetect\.freeze_end:\s+([-\d.]+)", fz_res.stderr)
            freeze_durs = re.findall(r"lavfi\.freezedetect\.freeze_duration:\s+([-\d.]+)", fz_res.stderr)
            vid_dur_val = float(metrics.get("video_duration_sec", 0.0))
            for i, f_start_s in enumerate(freeze_starts):
                f_start = float(f_start_s)
                f_end = float(freeze_ends[i]) if i < len(freeze_ends) else vid_dur_val
                f_dur = float(freeze_durs[i]) if i < len(freeze_durs) else max(0.0, f_end - f_start)
                freeze_events.append({
                    "start": round(f_start, 3),
                    "end": round(f_end, 3),
                    "duration": round(f_dur, 3),
                })
            if freeze_starts:
                warnings.append(f"Detected {len(freeze_starts)} frozen video stretch(es) > 3.0s (e.g. at {freeze_starts[:3]}s)")
        except Exception:
            pass
        metrics["freeze_events"] = freeze_events

        # ----------------------------------------------------------------------
        # Check 6: Story Lower Thirds and Illustrative Tags
        # ----------------------------------------------------------------------
        attr_passed = True
        attr_details: list[str] = []
        try:
            with open(render_plan_file, "r", encoding="utf-8") as pf:
                plan_data = json.load(pf)
            with open(script_file, "r", encoding="utf-8") as sf:
                script_data = json.load(sf)

            plan_scenes = {sc["id"]: sc for sc in plan_data.get("scenes", [])}

            for sc_script in script_data.get("scenes", []):
                sc_id = sc_script["id"]
                if sc_script.get("type") == "story":
                    sc_plan = plan_scenes.get(sc_id)
                    if not sc_plan:
                        attr_passed = False
                        attr_details.append(f"Scene '{sc_id}' missing in render plan")
                        continue

                    # Verify lower third exists
                    lt = sc_plan.get("lower_third")
                    if not lt or not lt.get("headline") or not lt.get("source_credit"):
                        attr_passed = False
                        attr_details.append(f"Scene '{sc_id}' lower-third missing headline or source credit")

                    # Verify lower third overlay is scheduled
                    sc_overlays = sc_plan.get("overlays", [])
                    has_lt_ov = any(ov.get("type") == "lower_third" for ov in sc_overlays)
                    if not has_lt_ov:
                        attr_passed = False
                        attr_details.append(f"Scene '{sc_id}' missing scheduled lower_third overlay")

                    # Check illustrative tag if stock shots used
                    has_stock = any(
                        s.get("source") in ("wikimedia_commons", "pixabay", "coverr", "openverse", "nasa")
                        for s in sc_plan.get("shots", [])
                    )
                    if has_stock:
                        has_tag_ov = any(ov.get("type") == "illustrative_tag" for ov in sc_overlays)
                        if not has_tag_ov:
                            attr_passed = False
                            attr_details.append(f"Scene '{sc_id}' uses stock media but lacks illustrative tag")

        except Exception as e:
            attr_passed = False
            attr_details.append(f"Lower-third / attribution check error: {e}")

        checks.append({
            "name": "story_lower_thirds",
            "passed": attr_passed,
            "detail": "; ".join(attr_details) if attr_details else "All story scenes contain lower-third attribution and required illustrative tags",
        })
        if not attr_passed and not redo_stage:
            redo_stage = "motion"

        # ----------------------------------------------------------------------
        # Check 7: Overlay layout safe zone compliance
        # ----------------------------------------------------------------------
        layout_passed = True
        layout_details: list[str] = []
        try:
            with open(render_plan_file, "r", encoding="utf-8") as pf:
                plan_data = json.load(pf)

            overlays = plan_data.get("overlays", [])
            # 5% safe margins in 1920x1080: x: 96..1824, y: 54..1026
            for ov in overlays:
                ox = ov.get("x", 0)
                oy = ov.get("y", 0)
                ow = ov.get("w", 0)
                oh = ov.get("h", 0)
                ov_id = ov.get("id")

                if ox < 96 or oy < 54 or (ox + ow) > 1824 or (oy + oh) > 1026:
                    layout_passed = False
                    layout_details.append(f"Overlay '{ov_id}' exceeds 5% safe margin bounds (rect: [{ox},{oy},{ow},{oh}])")

                # Caption band collision check: captions sit in y: 910..1020
                if ov.get("type") == "lower_third" and (oy + oh) > 900:
                    layout_passed = False
                    layout_details.append(f"Lower-third '{ov_id}' overlaps caption band (bottom y={oy+oh} > 900)")

        except Exception as e:
            layout_passed = False
            layout_details.append(f"Overlay layout check error: {e}")

        checks.append({
            "name": "overlay_layout",
            "passed": layout_passed,
            "detail": "; ".join(layout_details) if layout_details else "All overlays reside strictly within safe zones and clear caption bands",
        })
        if not layout_passed and not redo_stage:
            redo_stage = "motion"

        # ----------------------------------------------------------------------
        # Check 8: Subtitle captions presence & timing
        # ----------------------------------------------------------------------
        captions_passed = True
        cap_details: list[str] = []
        try:
            if not ass_file.exists():
                captions_passed = False
                cap_details.append(f"Missing {ass_file}")
            else:
                with open(captions_file, "r", encoding="utf-8") as cf:
                    json_caps = json.load(cf)
                ass_content = ass_file.read_text(encoding="utf-8")
                dialogue_lines = [l for l in ass_content.splitlines() if l.startswith("Dialogue:")]

                if len(dialogue_lines) != len(json_caps):
                    captions_passed = False
                    cap_details.append(f"ASS dialogue count ({len(dialogue_lines)}) != captions.json count ({len(json_caps)})")

                # Check all subtitle times are within video duration
                vid_dur = metrics.get("video_duration_sec", 120.0)
                for cap in json_caps:
                    if float(cap.get("end", 0.0)) > vid_dur + 0.1:
                        captions_passed = False
                        cap_details.append(f"Caption '{cap.get('id')}' ends past video duration ({cap.get('end')}s > {vid_dur}s)")
                        break

        except Exception as e:
            captions_passed = False
            cap_details.append(f"Captions check error: {e}")

        checks.append({
            "name": "captions",
            "passed": captions_passed,
            "detail": "; ".join(cap_details) if cap_details else f"Captions synchronized ({len(json_caps)} events within duration)",
        })
        if not captions_passed and not redo_stage:
            redo_stage = "timeline"

        # ----------------------------------------------------------------------
        # Check 9: Publish pack description.md
        # ----------------------------------------------------------------------
        desc_passed = True
        desc_details: list[str] = []
        try:
            if not desc_file.exists():
                desc_passed = False
                desc_details.append("Missing render/description.md")
            else:
                desc_text = desc_file.read_text(encoding="utf-8")

                # Test banner
                with open(script_file, "r", encoding="utf-8") as sf:
                    s_data = json.load(sf)
                if s_data.get("run_kind") == "test":
                    if "TEST RUN - DO NOT PUBLISH" not in desc_text:
                        desc_passed = False
                        desc_details.append("Missing 'TEST RUN - DO NOT PUBLISH' banner in test run description")

                # Chapters: >=3 chapters, starts at 00:00
                chapter_matches = re.findall(r"^(\d{2}:\d{2}(?::\d{2})?)\s+-\s+(.+)$", desc_text, re.MULTILINE)
                if len(chapter_matches) < 3:
                    desc_passed = False
                    desc_details.append(f"Description contains {len(chapter_matches)} chapters (minimum 3 required)")
                elif chapter_matches[0][0] != "00:00":
                    desc_passed = False
                    desc_details.append(f"First chapter starts at {chapter_matches[0][0]}, expected 00:00")

                # Disclosures
                if "Narration audio was synthesized using an AI voice model." not in desc_text:
                    desc_passed = False
                    desc_details.append("Missing AI-voice disclosure in description")

                # CC attribution completeness
                with open(media_file, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                for shot in m_data.get("shots", []):
                    if shot.get("attribution_required"):
                        auth = shot.get("author", "")
                        if auth and auth not in desc_text:
                            desc_passed = False
                            desc_details.append(f"Required attribution author '{auth}' missing in description.md")
                            break

        except Exception as e:
            desc_passed = False
            desc_details.append(f"Description check error: {e}")

        checks.append({
            "name": "description_md",
            "passed": desc_passed,
            "detail": "; ".join(desc_details) if desc_details else "Publish pack description.md complete with valid chapters, disclosures, and credits",
        })
        if not desc_passed and not redo_stage:
            redo_stage = "render"

        # ----------------------------------------------------------------------
        # Contact sheet & Storyboard generation
        # ----------------------------------------------------------------------
        contact_sheet_path = qa_dir / "contact_sheet.jpg"
        tl_data_loaded: dict[str, Any] = {}
        if timeline_file.exists():
            with open(timeline_file, "r", encoding="utf-8") as tf:
                tl_data_loaded = json.load(tf)
        tl_scenes = tl_data_loaded.get("scenes", [])
        self._generate_contact_sheet(final_mp4, tl_scenes, contact_sheet_path)

        # ----------------------------------------------------------------------
        # Summary & Report
        # ----------------------------------------------------------------------
        overall_passed = all(c["passed"] for c in checks)
        if overall_passed:
            redo_stage = None

        qa_report = {
            "passed": overall_passed,
            "checks": checks,
            "warnings": warnings,
            "redo_stage": redo_stage,
            "metrics": metrics,
            "visual_quality_note": "Visual quality needs human watching",
        }

        report_path = qa_dir / "qa_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(qa_report, f, indent=2)

        # Also write root qa_report.json
        root_report = self.run_dir / "qa_report.json"
        with open(root_report, "w", encoding="utf-8") as f:
            json.dump(qa_report, f, indent=2)

        # Generate self-contained HTML storyboard
        def _safe_load_json(p: Path, default: Any) -> Any:
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return default
            return default

        storyboard_path, _storyboard_frames = generate_storyboard(
            final_mp4=final_mp4,
            qa_dir=qa_dir,
            script_data=_safe_load_json(script_file, {}),
            timeline_data=tl_data_loaded,
            media_data=_safe_load_json(media_file, {}),
            render_plan=_safe_load_json(render_plan_file, {}),
            captions_data=_safe_load_json(captions_file, []),
            qa_report=qa_report,
        )

        outputs = [
            "qa/qa_report.json",
            "qa/contact_sheet.jpg",
            "qa/storyboard.html",
            "qa_report.json",
        ]

        # Update manifest metrics
        if manifest:
            stage_meta = manifest.setdefault("stages", {}).setdefault("qa", {})
            stage_meta.setdefault("metrics", {})
            stage_meta["metrics"]["passed"] = overall_passed
            stage_meta["metrics"]["checks_count"] = len(checks)
            stage_meta["metrics"]["warnings_count"] = len(warnings)
            stage_meta["metrics"]["redo_stage"] = redo_stage

        self.logger.info(f"QA Agent completed: passed={overall_passed}, checks={len(checks)}, warnings={len(warnings)}, redo_stage={redo_stage}")

        if not overall_passed:
            failed_checks = [c["name"] for c in checks if not c["passed"]]
            raise ValueError(f"QA inspection failed on checks: {failed_checks}. Recommended redo stage: {redo_stage}")

        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nepal News Bot QA Agent.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Run date in YYYY-MM-DD")
    parser.add_argument("--runs-dir", default="data/runs", help="Base runs directory")
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.date
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    agent = QAAgent(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {}

    outputs = agent.run(manifest)
    print(f"QA Agent completed successfully. Passed. Outputs: {outputs}")


if __name__ == "__main__":
    main()

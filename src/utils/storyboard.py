"""Self-contained HTML storyboard generator for QA visual inspection without playing video."""
from __future__ import annotations

import html
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("qa.storyboard")


def _fmt_ts(sec: float) -> str:
    """Formats seconds as M:SS.s (or H:MM:SS.s)."""
    sec = max(0.0, float(sec))
    mins = int(sec // 60)
    rem = sec - mins * 60
    if mins >= 60:
        hrs = mins // 60
        mins = mins % 60
        return f"{hrs}:{mins:02d}:{rem:04.1f}"
    return f"{mins}:{rem:04.1f}"


def _extract_timestamps_from_warning(warning_text: str) -> list[float]:
    """Best-effort extraction of numeric timestamps mentioned in a QA warning."""
    matches = re.findall(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*s?\b", warning_text)
    vals: list[float] = []
    for m in matches:
        try:
            v = float(m)
            # Ignore obvious threshold constants like 3.0 if paired with '>' or '<'
            vals.append(v)
        except ValueError:
            pass
    return vals


def build_storyboard_timestamps(
    duration_sec: float,
    timeline_scenes: list[dict[str, Any]],
    interval_sec: float = 4.0,
) -> list[float]:
    """Selects representative frame timestamps roughly every interval_sec across duration_sec,
    ensuring every scene has at least one sampled frame."""
    if duration_sec <= 0:
        return []

    sample_set: set[float] = set()
    first_t = 0.5 if duration_sec > 0.6 else round(duration_sec / 2.0, 2)
    sample_set.add(round(first_t, 2))

    cur = float(interval_sec)
    while cur < duration_sec - 0.25:
        if abs(cur - first_t) >= 1.0:
            sample_set.add(round(cur, 2))
        cur += float(interval_sec)

    # Ensure every scene boundary region has at least one frame
    for sc in timeline_scenes:
        sc_start = float(sc.get("start", 0.0))
        sc_end = float(sc.get("end", 0.0))
        if sc_end <= sc_start:
            continue
        has_inside = any(sc_start <= t <= sc_end for t in sample_set)
        if not has_inside:
            mid = round(min(duration_sec - 0.05, max(0.05, (sc_start + sc_end) / 2.0)), 2)
            sample_set.add(mid)

    return sorted(t for t in sample_set if 0.0 <= t <= max(0.0, duration_sec - 0.02) or duration_sec < 0.2)


def _find_covering_scene(
    t: float,
    timeline_scenes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not timeline_scenes:
        return None
    for sc in timeline_scenes:
        if float(sc.get("start", 0.0)) <= t <= float(sc.get("end", 0.0)):
            return sc
    return min(
        timeline_scenes,
        key=lambda sc: min(abs(t - float(sc.get("start", 0.0))), abs(t - float(sc.get("end", 0.0)))),
    )


def _find_covering_shot(
    t: float,
    scene_id: str | None,
    plan_shots_by_scene: dict[str, list[dict[str, Any]]],
    media_shots_by_scene: dict[str, list[dict[str, Any]]],
    all_shots_flat: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = plan_shots_by_scene.get(scene_id or "", []) or media_shots_by_scene.get(scene_id or "", [])
    if not candidates:
        candidates = all_shots_flat

    # Enrich plan shot with source/type from media_manifest if missing
    media_lookup = {
        (s.get("scene_id"), s.get("shot_index")): s
        for s in all_shots_flat
    }

    chosen: dict[str, Any] | None = None
    for sh in candidates:
        if float(sh.get("start", 0.0)) - 1e-3 <= t <= float(sh.get("end", 0.0)) + 1e-3:
            chosen = dict(sh)
            break
    if chosen is None and candidates:
        chosen = dict(
            min(
                candidates,
                key=lambda sh: min(abs(t - float(sh.get("start", 0.0))), abs(t - float(sh.get("end", 0.0)))),
            )
        )
    if chosen is None:
        return {"shot_index": "-", "source": "unknown", "type": "unknown"}

    m_match = media_lookup.get((scene_id or chosen.get("scene_id"), chosen.get("shot_index")))
    if m_match:
        chosen.setdefault("source", m_match.get("source"))
        if not chosen.get("source"):
            chosen["source"] = m_match.get("source")
        if not chosen.get("type"):
            chosen["type"] = m_match.get("type")
    return chosen


def _find_caption_text(t: float, captions_data: list[dict[str, Any]]) -> str:
    if not captions_data:
        return ""
    exact = [
        str(c.get("text", "")).strip()
        for c in captions_data
        if float(c.get("start", 0.0)) <= t <= float(c.get("end", 0.0))
    ]
    if exact:
        return " ".join(x for x in exact if x)

    near = [
        str(c.get("text", "")).strip()
        for c in captions_data
        if float(c.get("start", 0.0)) - 0.5 <= t <= float(c.get("end", 0.0)) + 0.5
    ]
    return " ".join(x for x in near if x)


def generate_storyboard(
    final_mp4: Path,
    qa_dir: Path,
    script_data: dict[str, Any],
    timeline_data: dict[str, Any],
    media_data: dict[str, Any],
    render_plan: dict[str, Any],
    captions_data: list[dict[str, Any]],
    qa_report: dict[str, Any],
    interval_sec: float = 4.0,
    frame_width: int = 480,
) -> tuple[Path, list[Path]]:
    """Extracts ~480px JPEG frames every ~4s from final_mp4 and builds self-contained qa/storyboard.html."""
    qa_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = qa_dir / "storyboard_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale frames from previous runs
    for old_f in frames_dir.glob("frame_*.jpg"):
        try:
            old_f.unlink()
        except OSError:
            pass

    metrics = qa_report.get("metrics", {})
    duration_sec = float(
        metrics.get("video_duration_sec")
        or render_plan.get("duration_sec")
        or timeline_data.get("duration_sec")
        or 0.0
    )
    timeline_scenes = timeline_data.get("scenes") or render_plan.get("scenes") or []
    timestamps = build_storyboard_timestamps(duration_sec, timeline_scenes, interval_sec=interval_sec)

    script_scenes = {s.get("id"): s for s in script_data.get("scenes", []) if s.get("id")}
    plan_shots_by_scene: dict[str, list[dict[str, Any]]] = {
        sc.get("id"): list(sc.get("shots", []))
        for sc in render_plan.get("scenes", [])
        if sc.get("id")
    }
    media_shots_by_scene: dict[str, list[dict[str, Any]]] = {}
    all_media_shots = list(media_data.get("shots", []))
    for ms in all_media_shots:
        sc_id = ms.get("scene_id")
        if sc_id:
            media_shots_by_scene.setdefault(sc_id, []).append(ms)

    warnings_list: list[str] = list(qa_report.get("warnings", []))
    freeze_events: list[dict[str, Any]] = list(metrics.get("freeze_events", []))
    failed_checks = [c for c in qa_report.get("checks", []) if not c.get("passed", True)]

    extracted_frames: list[Path] = []
    entries: list[dict[str, Any]] = []

    for idx, t in enumerate(timestamps):
        fname = f"frame_{idx:04d}.jpg"
        fpath = frames_dir / fname
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(final_mp4),
            "-vframes",
            "1",
            "-vf",
            f"scale={frame_width}:-2",
            "-q:v",
            "3",
            str(fpath),
        ]
        subprocess.run(cmd, capture_output=True)
        if not fpath.exists():
            continue
        extracted_frames.append(fpath)

        sc_tl = _find_covering_scene(t, timeline_scenes) or {"id": "unknown", "start": t, "end": t}
        sc_id = str(sc_tl.get("id", "unknown"))
        sc_script = script_scenes.get(sc_id, {})
        plan_sc = next((p for p in render_plan.get("scenes", []) if p.get("id") == sc_id), {})
        lt = plan_sc.get("lower_third") or {}

        headline = sc_script.get("headline") or lt.get("headline") or sc_id
        raw_sources = sc_script.get("source") or sc_script.get("sources") or lt.get("source_credit") or "Nepal News Desk"
        if isinstance(raw_sources, list):
            scene_source = ", ".join(str(x) for x in raw_sources)
        else:
            scene_source = str(raw_sources)

        shot_info = _find_covering_shot(
            t=t,
            scene_id=sc_id,
            plan_shots_by_scene=plan_shots_by_scene,
            media_shots_by_scene=media_shots_by_scene,
            all_shots_flat=all_media_shots,
        )
        caption_txt = _find_caption_text(t, captions_data)

        # Match QA warnings or freeze events relevant near timestamp t
        frame_warnings: list[str] = []
        for fe in freeze_events:
            f_start = float(fe.get("start", -1.0))
            f_end = float(fe.get("end", f_start))
            f_dur = float(fe.get("duration", max(0.0, f_end - f_start)))
            if (f_start - 1.5) <= t <= (f_end + 1.5):
                frame_warnings.append(
                    f"Freeze warning at {f_start:.1f}s - {f_end:.1f}s ({f_dur:.1f}s)"
                )
        for w in warnings_list:
            ts_nums = _extract_timestamps_from_warning(w)
            if any(abs(wt - t) <= interval_sec * 0.65 for wt in ts_nums):
                if w not in frame_warnings:
                    frame_warnings.append(w)

        entries.append({
            "t": t,
            "rel_img": f"storyboard_frames/{fname}",
            "scene_id": sc_id,
            "scene_start": float(sc_tl.get("start", 0.0)),
            "scene_end": float(sc_tl.get("end", 0.0)),
            "headline": headline,
            "scene_source": scene_source,
            "shot_index": shot_info.get("shot_index", 0),
            "shot_source": shot_info.get("source") or "unknown",
            "shot_type": shot_info.get("type") or "unknown",
            "caption": caption_txt,
            "warnings": frame_warnings,
        })

    run_date = str(script_data.get("date") or timeline_data.get("date") or "unknown")
    run_kind = str(script_data.get("run_kind") or "test")
    passed = bool(qa_report.get("passed", False))
    status_label = "PASSED" if passed else "FAILED"
    status_cls = "status-pass" if passed else "status-fail"

    # Build top QA warnings section
    warn_items_html = []
    if freeze_events:
        for fe in freeze_events:
            warn_items_html.append(
                f"<li><strong>Freeze</strong> at <code>{float(fe['start']):.2f}s</code> - "
                f"<code>{float(fe['end']):.2f}s</code> ({float(fe['duration']):.2f}s)</li>"
            )
    for w in warnings_list:
        warn_items_html.append(f"<li>{html.escape(w)}</li>")
    for fc in failed_checks:
        warn_items_html.append(
            f"<li><strong>FAILED [{html.escape(str(fc.get('name')))}]</strong>: "
            f"{html.escape(str(fc.get('detail', '')))}</li>"
        )
    if not warn_items_html:
        warn_html = "<div class='qa-clean'>No QA warnings or failed checks reported.</div>"
    else:
        warn_html = "<ul class='qa-warn-list'>" + "\n".join(warn_items_html) + "</ul>"

    # Group frames cleanly under scene boundary headers
    sections_html: list[str] = []
    current_scene_id: str | None = None
    in_grid = False

    for entry in entries:
        if entry["scene_id"] != current_scene_id:
            if in_grid:
                sections_html.append("</div>")
            current_scene_id = entry["scene_id"]
            sections_html.append(
                f"""
  <section class="scene-section" id="scene-{html.escape(entry['scene_id'])}">
    <div class="scene-boundary">
      <div class="scene-tag">{html.escape(entry['scene_id'])}</div>
      <div class="scene-main">
        <h2>{html.escape(str(entry['headline']))}</h2>
        <div class="scene-sub">
          <span>Source: <strong>{html.escape(str(entry['scene_source']))}</strong></span>
          <span>Time: <code>{_fmt_ts(entry['scene_start'])}</code> - <code>{_fmt_ts(entry['scene_end'])}</code> ({entry['scene_start']:.1f}s - {entry['scene_end']:.1f}s)</span>
        </div>
      </div>
    </div>
    <div class="frames-grid">
"""
            )
            in_grid = True

        w_badge = ""
        if entry["warnings"]:
            w_badge = "<div class='frame-warn'>" + "<br>".join(html.escape(w) for w in entry["warnings"]) + "</div>"

        cap_display = html.escape(entry["caption"]) if entry["caption"] else "<span class='cap-empty'>(No spoken caption at timestamp)</span>"

        sections_html.append(
            f"""      <article class="frame-card {'has-warn' if entry['warnings'] else ''}">
        <div class="frame-head">
          <span class="ts-pill">{entry['t']:.1f}s ({_fmt_ts(entry['t'])})</span>
          <span class="shot-pill">Shot #{html.escape(str(entry['shot_index']))} | {html.escape(str(entry['shot_type']))} | {html.escape(str(entry['shot_source']))}</span>
        </div>
        <img src="{html.escape(entry['rel_img'])}" alt="{html.escape(entry['scene_id'])} frame at {entry['t']:.1f}s" width="{frame_width}" loading="lazy">
        <div class="caption-box">{cap_display}</div>
        {w_badge}
      </article>"""
        )

    if in_grid:
        sections_html.append("    </div>\n  </section>")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Storyboard - {html.escape(run_date)} ({html.escape(run_kind)})</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      background: #0d1117;
      color: #e6edf3;
      line-height: 1.45;
    }}
    header {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 18px 22px;
      margin-bottom: 20px;
    }}
    .header-top {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    h1 {{ margin: 0; font-size: 22px; color: #58a6ff; }}
    .meta-line {{ margin-top: 8px; color: #8b949e; font-size: 14px; }}
    .status-badge {{
      padding: 4px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
    }}
    .status-pass {{ background: #238636; color: #ffffff; }}
    .status-fail {{ background: #b62324; color: #ffffff; }}
    .qa-panel {{
      margin-top: 14px;
      padding: 12px 14px;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      font-size: 13px;
    }}
    .qa-clean {{ color: #3fb950; }}
    .qa-warn-list {{ margin: 6px 0 0 18px; padding: 0; color: #f5a623; }}
    .scene-section {{ margin-bottom: 28px; }}
    .scene-boundary {{
      display: flex;
      align-items: flex-start;
      gap: 14px;
      background: #161b22;
      border-left: 5px solid #f5a623;
      border-top: 1px solid #30363d;
      border-right: 1px solid #30363d;
      border-bottom: 1px solid #30363d;
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 14px;
    }}
    .scene-tag {{
      background: #1f6feb;
      color: #ffffff;
      padding: 4px 10px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 13px;
      white-space: nowrap;
    }}
    .scene-main h2 {{ margin: 0 0 4px 0; font-size: 17px; color: #f0f6fc; }}
    .scene-sub {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #8b949e; }}
    .frames-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
      gap: 16px;
    }}
    .frame-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .frame-card.has-warn {{ border-color: #f5a623; }}
    .frame-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      background: #0d1117;
      border-bottom: 1px solid #21262d;
      font-size: 12px;
    }}
    .ts-pill {{ font-weight: 700; color: #58a6ff; }}
    .shot-pill {{ color: #8b949e; }}
    .frame-card img {{
      width: 100%;
      height: auto;
      display: block;
      background: #000000;
    }}
    .caption-box {{
      padding: 10px 12px;
      font-size: 14px;
      color: #e6edf3;
      border-top: 1px solid #21262d;
      flex-grow: 1;
    }}
    .cap-empty {{ color: #6e7681; font-style: italic; }}
    .frame-warn {{
      padding: 8px 12px;
      background: #332409;
      border-top: 1px solid #f5a623;
      color: #f5c363;
      font-size: 12px;
      font-weight: 600;
    }}
    code {{ background: #21262d; padding: 1px 5px; border-radius: 3px; }}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>Nepal News Bot — Visual Storyboard</h1>
      <span class="status-badge {status_cls}">QA {status_label}</span>
    </div>
    <div class="meta-line">
      Run Date: <strong>{html.escape(run_date)}</strong> |
      Run Kind: <strong>{html.escape(run_kind)}</strong> |
      Duration: <strong>{duration_sec:.2f}s</strong> |
      Frames Sampled: <strong>{len(entries)}</strong> (every ~{interval_sec:.0f}s) |
      Loudness: <strong>{metrics.get('integrated_loudness_lufs', 'N/A')} LUFS</strong>
    </div>
    <div class="qa-panel">
      <strong>QA Warnings &amp; Diagnostics:</strong>
      {warn_html}
    </div>
  </header>
  {"".join(sections_html)}
</body>
</html>
"""

    storyboard_path = qa_dir / "storyboard.html"
    storyboard_path.write_text(html_doc, encoding="utf-8")
    logger.info(f"Generated storyboard at {storyboard_path} with {len(extracted_frames)} frames.")
    return storyboard_path, extracted_frames

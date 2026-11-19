"""Generates review contact sheets, review HTML, and visual credits."""
import html
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("media.review")


def generate_credits_markdown(
    shots: list[dict[str, Any]],
    output_path: Path,
    disclose_synthetic_voice: bool = True,
    stock_sources_used: set[str] | None = None,
) -> None:
    """Generates credits.md containing attributions, stock credits, and legal disclaimers."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_sources = stock_sources_used or set()

    lines = [
        "# Visual and Audio Credits",
        "",
        "## Creative Commons Attributions (CC BY)",
    ]

    cc_by_count = 0
    seen_attributions: set[str] = set()
    for s in shots:
        source = s.get("source", "")
        used_sources.add(source)
        attr = s.get("attribution_text")
        lic = s.get("license", "")
        is_cc_by = "by" in lic.lower() or s.get("attribution_required", False)

        if is_cc_by and attr and attr not in seen_attributions and s.get("type") != "fallback":
            seen_attributions.add(attr)
            cc_by_count += 1
            lic_url = s.get("license_url")
            origin = s.get("origin_url")
            entry = f"- **{attr}**"
            if lic_url:
                entry += f" ([License: {lic}]({lic_url}))"
            if origin and origin != "local":
                entry += f" - [Source Page]({origin})"
            lines.append(entry)

    if cc_by_count == 0:
        lines.append("- No Creative Commons BY assets required attribution in this broadcast.")

    lines.append("")
    lines.append("## Stock & Archive Providers")
    if "pixabay" in used_sources:
        lines.append("- Stock video and imagery courtesy of [Pixabay](https://pixabay.com).")
    if "nasa" in used_sources:
        lines.append("- Media and imagery courtesy of [NASA](https://images.nasa.gov).")
    if "coverr" in used_sources:
        lines.append("- Footage courtesy of [Coverr](https://coverr.co).")
    if "openverse" in used_sources:
        lines.append("- Open-licensed imagery sourced via [Openverse](https://openverse.org).")
    if "user-provided" in used_sources:
        lines.append("- Field footage and photography provided directly by the production team.")

    lines.append("")
    lines.append("## Editorial Disclaimers")
    lines.append("- Stock and archive footage is illustrative.")
    if disclose_synthetic_voice:
        lines.append("- Narration audio was synthesized using an AI voice model.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Credits generated at {output_path}")


def create_contact_sheet_for_scene(
    scene_id: str,
    scene_shots: list[dict[str, Any]],
    run_dir: Path,
    output_path: Path,
) -> None:
    """Creates a contact sheet JPEG showing the first frame of each shot in the scene with captions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 480, 270
    n_shots = max(1, len(scene_shots))

    # Single row layout or multi-row if > 4 shots
    cols = min(n_shots, 4)
    rows = (n_shots + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(20, 24, 30))
    draw = ImageDraw.Draw(sheet)

    for idx, shot in enumerate(scene_shots):
        row = idx // cols
        col = idx % cols
        x_offset = col * tile_w
        y_offset = row * tile_h

        shot_type = shot.get("type")
        file_rel = shot.get("file")
        tile_img = None

        if shot_type == "fallback" or not file_rel:
            # Fallback tile
            tile_img = Image.new("RGB", (tile_w, tile_h), color=(40, 44, 52))
            tdraw = ImageDraw.Draw(tile_img)
            tdraw.rectangle([10, 10, tile_w - 10, tile_h - 10], outline=(80, 90, 100), width=2)
            tdraw.text((tile_w // 2, tile_h // 2 - 10), "FALLBACK SHOT", fill=(200, 200, 200), anchor="mm")
            tdraw.text((tile_w // 2, tile_h // 2 + 15), "(Animated Graphic)", fill=(150, 150, 150), anchor="mm")
        else:
            file_path = run_dir / file_rel
            if file_path.exists():
                if shot_type == "video":
                    # Extract frame at 0.5s via ffmpeg
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-ss", "0.5",
                        "-i", str(file_path),
                        "-frames:v", "1",
                        "-f", "image2pipe",
                        "-vcodec", "png",
                        "-",
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
                    if proc.returncode == 0 and proc.stdout:
                        import io
                        tile_img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
                elif shot_type in ("image", "card"):
                    tile_img = Image.open(file_path).convert("RGB")

            if not tile_img:
                tile_img = Image.new("RGB", (tile_w, tile_h), color=(30, 30, 30))

        # Resize tile to exact size
        tile_resized = tile_img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(tile_resized, (x_offset, y_offset))

        # Draw caption overlay box at bottom of tile
        box_h = 44
        caption_box = [x_offset, y_offset + tile_h - box_h, x_offset + tile_w, y_offset + tile_h]
        draw.rectangle(caption_box, fill=(0, 0, 0, 200))

        sh_idx = shot.get("shot_index", idx)
        src = shot.get("source", "unknown")
        lic = shot.get("license", "unknown")
        dur = shot.get("end", 0) - shot.get("start", 0)

        line1 = f"Shot #{sh_idx} ({dur:.1f}s) | Source: {src}"
        line2 = f"License: {lic}"
        draw.text((x_offset + 8, y_offset + tile_h - 38), line1, fill=(255, 255, 255))
        draw.text((x_offset + 8, y_offset + tile_h - 20), line2, fill=(200, 200, 200))

    sheet.save(output_path, "JPEG", quality=88)
    logger.info(f"Contact sheet saved for {scene_id} at {output_path}")


def generate_review_html(
    manifest_data: dict[str, Any],
    script_scenes: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generates review/index.html linking contact sheets and per-shot breakdowns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_str = manifest_data.get("date", "Unknown")
    shots = manifest_data.get("shots", [])

    shots_by_scene: dict[str, list[dict[str, Any]]] = {}
    for s in shots:
        shots_by_scene.setdefault(s.get("scene_id", "unknown"), []).append(s)

    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>Media Agent Review - " + html.escape(date_str) + "</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f141c; color: #e1e4e8; margin: 0; padding: 24px; }",
        "    h1 { color: #58a6ff; margin-bottom: 8px; }",
        "    .meta { color: #8b949e; margin-bottom: 24px; font-size: 14px; }",
        "    .scene-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 24px; padding: 18px; }",
        "    .scene-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 14px; }",
        "    .scene-title { font-size: 18px; font-weight: 600; color: #f0f6fc; }",
        "    .badge { padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }",
        "    .badge-sensitive { background: #b62324; color: white; }",
        "    .badge-normal { background: #238636; color: white; }",
        "    .keywords { color: #79c0ff; font-size: 13px; margin-bottom: 12px; }",
        "    .contact-sheet { width: 100%; max-width: 960px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 14px; display: block; }",
        "    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }",
        "    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #21262d; }",
        "    th { background: #0d1117; color: #8b949e; }",
        "    tr:hover { background: #1f242c; }",
        "    a { color: #58a6ff; text-decoration: none; }",
        "    a:hover { text-decoration: underline; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>Nepal News Bot - Media Stage Review</h1>",
        f"  <div class='meta'>Run Date: <strong>{date_str}</strong> | Orientation: <strong>{manifest_data.get('orientation')}</strong> | FPS: <strong>{manifest_data.get('fps')}</strong> | Total Shots: <strong>{len(shots)}</strong></div>",
    ]

    for scene in script_scenes:
        sc_id = scene["id"]
        sc_shots = shots_by_scene.get(sc_id, [])
        is_sens = any(s.get("sensitive_scene") for s in sc_shots)
        badge_html = "<span class='badge badge-sensitive'>SENSITIVE</span>" if is_sens else "<span class='badge badge-normal'>STANDARD</span>"
        kw_str = ", ".join(scene.get("visual_keywords", []))

        html_lines.append(f"  <div class='scene-card'>")
        html_lines.append(f"    <div class='scene-header'>")
        html_lines.append(f"      <div class='scene-title'>{sc_id}: {html.escape(scene.get('headline', ''))}</div>")
        html_lines.append(f"      <div>{badge_html}</div>")
        html_lines.append(f"    </div>")
        html_lines.append(f"    <div class='keywords'>Keywords: {html.escape(kw_str)}</div>")
        html_lines.append(f"    <img class='contact-sheet' src='{sc_id}.jpg' alt='Contact Sheet for {sc_id}'>")

        html_lines.append("    <table>")
        html_lines.append("      <thead><tr><th>#</th><th>Time (Narration)</th><th>Type</th><th>Source</th><th>License</th><th>Author</th><th>Origin</th></tr></thead>")
        html_lines.append("      <tbody>")
        for s in sc_shots:
            sh_idx = s.get("shot_index")
            t_span = f"{s.get('start', 0.0):.1f}s - {s.get('end', 0.0):.1f}s"
            typ = s.get("type")
            src = s.get("source")
            lic = s.get("license")
            author = s.get("author")
            orig = s.get("origin_url")
            orig_html = f"<a href='{orig}' target='_blank'>Link</a>" if orig and orig != "local" else (orig or "-")

            html_lines.append(f"        <tr><td>{sh_idx}</td><td>{t_span}</td><td>{typ}</td><td>{src}</td><td>{lic}</td><td>{html.escape(author or '')}</td><td>{orig_html}</td></tr>")
        html_lines.append("      </tbody>")
        html_lines.append("    </table>")
        html_lines.append("  </div>")

    html_lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(html_lines), encoding="utf-8")
    logger.info(f"Review HTML generated at {output_path}")


def draw_card_image(
    card_text: str,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    category: str = "NEWS BRIEFING",
    quiet: bool = False,
) -> None:
    """Renders a 1920x1080 gradient card with large centered wrapped text and accents.

    If quiet is True, suppresses top-left category badge and bottom-right watermark
    so background cards do not visually collide with featured foreground cards.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (11, 31, 58, 255))
    draw = ImageDraw.Draw(img)

    # Vertical gradient background
    for y in range(height):
        factor = y / height
        r = int(11 * (1 - factor) + 20 * factor)
        g = int(31 * (1 - factor) + 52 * factor)
        b = int(58 * (1 - factor) + 90 * factor)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Amber decorative frame / accents
    margin = 80
    draw.rounded_rectangle(
        [margin, margin, width - margin, height - margin],
        radius=24,
        outline=(245, 166, 35, 200),
        width=4,
    )
    # Inner border
    draw.rounded_rectangle(
        [margin + 16, margin + 16, width - margin - 16, height - margin - 16],
        radius=16,
        outline=(255, 255, 255, 40),
        width=2,
    )

    if not quiet:
        # Accent top strip
        draw.rounded_rectangle(
            [margin + 40, margin + 40, width - margin - 40, margin + 50],
            radius=4,
            fill=(245, 166, 35, 255),
        )

    # Fonts
    font_paths = ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 56)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    font_tag = None
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font_tag = ImageFont.truetype(fp, 32)
                break
            except Exception:
                pass
    if font_tag is None:
        font_tag = ImageFont.load_default()

    if not quiet:
        # Small category tag
        draw.text((margin + 50, margin + 75), category.upper(), font=font_tag, fill=(245, 166, 35, 255))

        # Subtle category watermark in lower right corner (does not compete with motion overlays)
        font_watermark = None
        for fp in font_paths:
            if Path(fp).exists():
                try:
                    font_watermark = ImageFont.truetype(fp, 24)
                    break
                except Exception:
                    pass
        if font_watermark is None:
            font_watermark = ImageFont.load_default()

        wm_text = category.upper()
        wm_bbox = draw.textbbox((0, 0), wm_text, font=font_watermark)
        wm_w = wm_bbox[2] - wm_bbox[0]
        draw.text(
            (width - margin - 60 - wm_w, height - margin - 50),
            wm_text,
            font=font_watermark,
            fill=(255, 255, 255, 50),
        )

    img.convert("RGB").save(output_path, "PNG")


def save_candidate_thumbnail(
    cand: Any,
    thumbs_dir: Path,
    cache_mgr: Any,
    offline: bool = False,
) -> str:
    """Saves a ~480px wide thumbnail for a candidate into thumbs_dir and returns relative path."""
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    clean_id = re.sub(r"[^\w\-.]", "_", cand.id)
    thumb_file = thumbs_dir / f"{clean_id}.jpg"
    rel_path = f"thumbs/{thumb_file.name}"

    if thumb_file.exists() and thumb_file.stat().st_size > 500:
        return rel_path

    # 1. Local file candidate
    if cand.source in ("local_library", "user-provided") or (cand.download_url and Path(cand.download_url).exists()):
        local_p = Path(cand.download_url)
        if local_p.exists():
            if cand.kind == "video":
                cmd = [
                    "ffmpeg", "-y", "-ss", "00:00:00.5", "-i", str(local_p),
                    "-vframes", "1", "-vf", "scale=480:-1", str(thumb_file),
                ]
                subprocess.run(cmd, capture_output=True)
            else:
                try:
                    with Image.open(local_p) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        new_h = max(1, int(h * 480 / w))
                        im_resized = im.resize((480, new_h), Image.Resampling.LANCZOS)
                        im_resized.save(thumb_file, "JPEG", quality=85)
                except Exception as e:
                    logger.debug(f"Pillow resize error for {local_p}: {e}")

    # 2. Remote candidate with thumbnail URL
    if not (thumb_file.exists() and thumb_file.stat().st_size > 500):
        thumb_url = getattr(cand, "thumb_url", "") or cand.extra.get("thumb_url") or cand.download_url
        if thumb_url and thumb_url.startswith("http"):
            try:
                cached_thumb = cache_mgr.download_file(thumb_url, offline=offline)
                try:
                    with Image.open(cached_thumb) as im:
                        im = im.convert("RGB")
                        w, h = im.size
                        new_h = max(1, int(h * 480 / w))
                        im_resized = im.resize((480, new_h), Image.Resampling.LANCZOS)
                        im_resized.save(thumb_file, "JPEG", quality=85)
                except Exception:
                    cmd = [
                        "ffmpeg", "-y", "-i", str(cached_thumb),
                        "-vframes", "1", "-vf", "scale=480:-1", str(thumb_file),
                    ]
                    subprocess.run(cmd, capture_output=True)
            except Exception as e:
                logger.debug(f"Failed to fetch thumb from {thumb_url}: {e}")

    # 3. Clean fallback thumbnail if download unavailable
    if not (thumb_file.exists() and thumb_file.stat().st_size > 500):
        im = Image.new("RGB", (480, 270), color=(11, 31, 58))
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, 480, 270], outline=(245, 166, 35), width=3)
        draw.text((20, 20), cand.source.upper(), fill=(245, 166, 35))
        title_snippet = (cand.title[:60] + "...") if len(cand.title) > 60 else cand.title
        draw.text((20, 60), title_snippet, fill=(255, 255, 255))
        draw.text((20, 220), f"License: {cand.license_id}", fill=(180, 195, 215))
        im.save(thumb_file, "JPEG", quality=85)

    return rel_path


def generate_wanted_clips_md(candidates_data: dict[str, Any], output_path: Path) -> None:
    """Writes media/review/wanted_clips.md listing shots below threshold with library suggestions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_rel = float(candidates_data.get("min_relevance", 0.5))
    shots = candidates_data.get("shots", [])

    lines = [
        "# Wanted Clips & Library Suggestions",
        "",
        f"The following shots had their best candidates score below the relevance threshold ({min_rel:.2f}).",
        "Consider filming or collecting media for these subjects and adding them to `data/media_library/`.",
        "",
    ]

    below_count = 0
    for sh in shots:
        best_score = float(sh.get("best_score", 0.0))
        if best_score < min_rel:
            below_count += 1
            sh_id = sh.get("shot_id")
            scene_id = sh.get("scene_id")
            sh_idx = sh.get("shot_index", 0)
            headline = sh.get("scene_headline", "")
            t_span = f"{sh.get('start', 0.0):.1f}s - {sh.get('end', 0.0):.1f}s ({sh.get('duration', 0.0):.1f}s)"
            sentences = sh.get("narration_sentences", [])
            sent_str = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            kws = sh.get("visual_keywords", [])
            kw_str = ", ".join(kws)

            lines.append(f"## {scene_id} Shot #{sh_idx}: {headline}")
            lines.append(f"- **Shot ID**: `{sh_id}`")
            lines.append(f"- **Timing**: {t_span}")
            lines.append(f"- **Narration**: \"{sent_str}\"")
            lines.append(f"- **Visual Keywords**: {kw_str}")
            lines.append(f"- **Best Candidate Score**: {best_score:.2f} (Below threshold {min_rel:.2f})")
            lines.append("- **Suggested Library Addition**:")
            lines.append(f"  - Subject: Footage or photo illustrating {headline}")
            lines.append(f"  - Recommended Keywords: {kw_str}")
            lines.append(f"  - Format: 1920x1080 Landscape MP4 (>= {sh.get('duration', 5.0):.1f}s) or high-res JPG")
            lines.append(f"  - Destination: `data/media_library/` with matching `.json` sidecar")
            lines.append("")

    if below_count == 0:
        lines.append("All shots found high-scoring candidates matching or exceeding the relevance threshold.")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wanted clips report written to {output_path}")


def generate_library_report(library_dir: Path | str = "data/media_library") -> str:
    """Scans data/media_library/ and returns a formatted report string."""
    library_dir = Path(library_dir)
    if not library_dir.exists():
        return f"Media Library ({library_dir}): Directory does not exist."

    media_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in library_dir.iterdir() if f.is_file() and f.suffix.lower() in media_exts]

    if not files:
        return f"Media Library ({library_dir}): 0 media files found. (Directory is empty or contains no supported media)."

    lines = [
        f"Media Library Report: {library_dir}",
        "=" * 80,
        f"{'Filename':<32} | {'Kind':<6} | {'Duration':<8} | {'Keywords'}",
        "-" * 80,
    ]

    for f in sorted(files, key=lambda x: x.name):
        sidecar = f.with_suffix(".json")
        meta = {}
        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as sf:
                    meta = json.load(sf)
            except Exception:
                pass

        is_video = f.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi")
        kind_str = "Video" if is_video else "Image"
        dur_str = f"{meta.get('duration', '-')}s" if meta.get("duration") else ("Video" if is_video else "N/A")
        kws = ", ".join(meta.get("keywords", [])) if meta.get("keywords") else "No sidecar"

        lines.append(f"{f.name:<32} | {kind_str:<6} | {dur_str:<8} | {kws}")

    lines.append("=" * 80)
    lines.append(f"Total media items: {len(files)}")
    return "\n".join(lines)


def generate_picker_html(candidates_data: dict[str, Any], output_path: Path) -> None:
    """Generates self-contained interactive picker.html for visual review without external URLs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    date_str = candidates_data.get("date", "")
    shots = candidates_data.get("shots", [])

    # Group shots by scene
    shots_by_scene: dict[str, list[dict[str, Any]]] = {}
    for sh in shots:
        shots_by_scene.setdefault(sh.get("scene_id", "scene"), []).append(sh)

    # Convert data to safe embedded JSON
    embedded_json = json.dumps(candidates_data).replace("<", "\\u003c")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Media Review & Shot Picker - {html.escape(date_str)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0B1F3A;
      color: #FFFFFF;
      margin: 0;
      padding: 24px;
      line-height: 1.5;
    }}
    .top-bar {{
      position: sticky;
      top: 12px;
      z-index: 1000;
      background: #08162B;
      border: 2px solid #F5A623;
      border-radius: 12px;
      padding: 16px 24px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7);
      margin-bottom: 28px;
    }}
    .top-bar-inner {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .title-area h1 {{
      margin: 0;
      font-size: 22px;
      color: #F5A623;
      font-weight: 700;
    }}
    .title-area .sub {{
      font-size: 13px;
      color: #A0B2C6;
      margin-top: 4px;
    }}
    .actions-area {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    button.btn {{
      background: #F5A623;
      color: #0B1F3A;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      font-size: 14px;
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
    }}
    button.btn:hover {{
      background: #FFB84D;
      transform: translateY(-1px);
    }}
    button.btn:active {{
      transform: translateY(1px);
    }}
    button.btn-sec {{
      background: #1C3559;
      color: #FFFFFF;
      border: 1px solid #3B5B88;
    }}
    button.btn-sec:hover {{
      background: #254470;
    }}
    #summary-status {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid #1C3559;
      font-size: 14px;
      color: #E2E8F0;
      font-weight: 500;
    }}
    .scene-block {{
      background: #102646;
      border: 1px solid #1C3D6E;
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 32px;
    }}
    .scene-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #F5A623;
      padding-bottom: 8px;
      margin-bottom: 16px;
    }}
    .scene-title {{
      font-size: 18px;
      font-weight: bold;
      color: #FFFFFF;
    }}
    .shot-card {{
      background: #08162B;
      border: 1px solid #204172;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .shot-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 13px;
    }}
    .shot-id {{
      font-weight: bold;
      color: #F5A623;
    }}
    .shot-narration {{
      background: #142848;
      border-left: 4px solid #F5A623;
      padding: 10px 14px;
      border-radius: 4px;
      font-size: 14px;
      font-style: italic;
      color: #E2E8F0;
      margin-bottom: 16px;
    }}
    .candidates-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .cand-item {{
      background: #122849;
      border: 2px solid #224373;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      position: relative;
    }}
    .cand-item.auto-picked {{
      border-color: #28A745;
      box-shadow: 0 0 12px rgba(40, 167, 69, 0.4);
    }}
    .cand-item.selected {{
      border-color: #F5A623;
      background: #183765;
    }}
    .cand-thumb-wrap {{
      width: 100%;
      height: 160px;
      background: #050E1C;
      position: relative;
      overflow: hidden;
    }}
    .cand-thumb-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .cand-score {{
      position: absolute;
      top: 8px;
      right: 8px;
      background: rgba(8, 22, 43, 0.9);
      border: 1px solid #F5A623;
      color: #F5A623;
      font-size: 12px;
      font-weight: bold;
      padding: 3px 8px;
      border-radius: 4px;
    }}
    .cand-score.high {{
      border-color: #28A745;
      color: #28A745;
    }}
    .cand-info {{
      padding: 12px;
      display: flex;
      flex-direction: column;
      flex: 1;
    }}
    .cand-title {{
      font-size: 13px;
      font-weight: bold;
      color: #FFFFFF;
      margin-bottom: 6px;
      line-height: 1.3;
      max-height: 34px;
      overflow: hidden;
    }}
    .cand-details {{
      font-size: 11px;
      color: #A0B2C6;
      margin-bottom: 8px;
    }}
    .cand-tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-bottom: 12px;
    }}
    .token-pill {{
      background: #204172;
      color: #DDE7F3;
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 3px;
    }}
    .cand-radio-label {{
      margin-top: auto;
      display: flex;
      align-items: center;
      gap: 8px;
      background: #08162B;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      font-weight: bold;
    }}
    .cand-radio-label input {{
      cursor: pointer;
    }}
    .manual-options {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      margin-top: 14px;
    }}
    .manual-card {{
      background: #0D1F38;
      border: 1px solid #1C3D6E;
      border-radius: 8px;
      padding: 12px 16px;
      transition: border-color 0.2s, background 0.2s;
    }}
    .manual-card:hover {{
      border-color: #3867A6;
    }}
    .manual-card.active {{
      border-color: #F5A623;
      background: #11284A;
    }}
    .manual-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    .manual-card-header label {{
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
      color: #FFFFFF;
    }}
    .opt-tag {{
      font-size: 11px;
      color: #A0B2C6;
    }}
    .dropzone-box {{
      border: 2px dashed #2E568F;
      border-radius: 6px;
      padding: 14px;
      text-align: center;
      cursor: pointer;
      background: #091629;
      transition: all 0.2s ease;
      color: #DDE7F3;
      font-size: 13px;
    }}
    .dropzone-box:hover, .dropzone-box.drag-over {{
      border-color: #F5A623;
      background: #122849;
    }}
    .dropzone-box .icon {{
      font-size: 20px;
      margin-right: 8px;
    }}
    .upload-preview-box {{
      display: flex;
      align-items: center;
      gap: 12px;
      background: #08162B;
      border: 1px solid #28A745;
      border-radius: 6px;
      padding: 8px 12px;
      margin-top: 8px;
    }}
    .upload-preview-box img {{
      width: 80px;
      height: 45px;
      object-fit: cover;
      border-radius: 4px;
      background: #000;
    }}
    .upload-file-details {{
      flex: 1;
      font-size: 12px;
    }}
    .upload-file-name {{
      color: #FFFFFF;
      font-weight: bold;
      word-break: break-all;
    }}
    .upload-file-status {{
      color: #28A745;
      font-size: 11px;
      font-weight: 600;
      margin-top: 2px;
    }}
    .text-row {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .text-row input[type="text"] {{
      flex: 1;
      background: #142848;
      border: 1px solid #2B5086;
      color: #FFFFFF;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 13px;
    }}
    .text-row input[type="text"]:focus {{
      outline: none;
      border-color: #F5A623;
    }}
    .btn-sm {{
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge-default {{
      background: #1C3559;
      color: #F5A623;
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 3px;
      border: 1px solid #F5A623;
    }}
    .badge-auto {{
      background: #28A745;
      color: #FFFFFF;
      font-size: 10px;
      font-weight: bold;
      padding: 2px 6px;
      border-radius: 3px;
      margin-left: 6px;
    }}
  </style>
</head>
<body>

  <!-- Sticky Top Bar -->
  <div class="top-bar">
    <div class="top-bar-inner">
      <div class="title-area">
        <h1>Media Review & Shot Picker</h1>
        <div class="sub">Run Date: {html.escape(date_str)} | Manual Visual Assignment Dock</div>
      </div>
      <div class="actions-area">
        <button class="btn" id="btnApplyOffice" onclick="applyPicksToOffice()" style="background:#28A745;color:#FFFFFF">✅ Apply Picks & Render Video</button>
        <button class="btn btn-sec" onclick="fillCardsWithTopClips()">✨ Auto-Pick Top Clips</button>
        <button class="btn btn-sec" onclick="setAllGraphicCards()">🎨 Set All to Graphic Cards</button>
        <button class="btn btn-sec" onclick="copyChoices()">📋 Copy Choices</button>
        <button class="btn btn-sec" onclick="downloadPicks()">💾 Download picks.json</button>
      </div>
    </div>
    <div id="summary-status">Loading candidate selections...</div>
  </div>

  <div id="scenes-container">
"""

    for sc_id, sc_shots in shots_by_scene.items():
        first_sh = sc_shots[0]
        headline = first_sh.get("scene_headline", sc_id)
        html_content += f"""
    <div class="scene-block" id="scene_{sc_id}">
      <div class="scene-header">
        <div class="scene-title">{sc_id}: {html.escape(headline)}</div>
        <div style="font-size: 13px; color: #A0B2C6;">{len(sc_shots)} shots</div>
      </div>
"""
        for sh in sc_shots:
            sh_id = sh.get("shot_id")
            sh_idx = sh.get("shot_index", 0)
            t_span = f"{sh.get('start', 0.0):.1f}s – {sh.get('end', 0.0):.1f}s ({sh.get('duration', 0.0):.1f}s)"
            sentences = sh.get("narration_sentences", [])
            sent_str = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            auto_id = sh.get("auto_pick_id")
            auto_type = sh.get("auto_pick_type", "clip")

            html_content += f"""
      <div class="shot-card" id="card_{sh_id}">
        <div class="shot-meta">
          <span class="shot-id">Shot #{sh_idx} ({sh_id})</span>
          <span style="color: #A0B2C6;">Timing: {t_span}</span>
        </div>
        <div class="shot-narration">
          🗣️ "{html.escape(sent_str)}"
        </div>

        <div class="candidates-grid">
"""
            candidates = sh.get("candidates", [])
            if candidates:
                html_content += """
        <div class="candidates-grid">
"""
                for cand in candidates:
                    c_id = cand.get("id")
                    score = float(cand.get("score", 0.0))
                    is_auto = (c_id == auto_id and auto_type == "clip")
                    auto_badge = '<span class="badge-auto">AUTO PICK</span>' if is_auto else ""
                    score_cls = "high" if score >= 0.5 else ""
                    tokens = cand.get("matched_tokens", [])
                    tokens_html = "".join(f'<span class="token-pill">{html.escape(t)}</span>' for t in tokens)
                    dur_text = f"{cand.get('duration'):.1f}s" if cand.get("duration") else "Still"
                    dim_text = f"{cand.get('width', 0)}x{cand.get('height', 0)}"

                    html_content += f"""
          <div class="cand-item {'auto-picked' if is_auto else ''}" id="cand_{sh_id}_{c_id}">
            <div class="cand-thumb-wrap">
              <img src="{cand.get('thumb', '')}" alt="{html.escape(cand.get('title', ''))}" onerror="this.style.display='none'">
              <div class="cand-score {score_cls}">{score:.2f}</div>
            </div>
            <div class="cand-info">
              <div class="cand-title">{html.escape(cand.get('title', ''))}</div>
              <div class="cand-details">
                {cand.get('source')} | {cand.get('license')} | {dur_text} ({dim_text})
              </div>
              <div class="cand-tokens">
                {tokens_html}
              </div>
              <label class="cand-radio-label">
                <input type="radio" name="pick_{sh_id}" value="pick" data-cand-id="{c_id}" data-source="{cand.get('source')}" onchange="updateChoices()">
                <span>Select clip {auto_badge}</span>
              </label>
            </div>
          </div>
"""
                html_content += """
        </div> <!-- end candidates-grid -->
"""

            default_card_text = headline
            html_content += f"""
        <div class="manual-options">
          <!-- 1. Upload Video or Image -->
          <div class="manual-card" id="card_opt_upload_{sh_id}">
            <div class="manual-card-header">
              <label>
                <input type="radio" name="pick_{sh_id}" value="upload" id="radio_upload_{sh_id}" onchange="updateChoices()">
                <span>📁 Upload Video or Image</span>
              </label>
              <span class="opt-tag">MP4, MOV, MKV, JPG, PNG, WEBP</span>
            </div>
            <div class="dropzone-box" id="dropzone_{sh_id}" onclick="document.getElementById('file_input_{sh_id}').click()">
              <input type="file" id="file_input_{sh_id}" accept="video/*,image/*" style="display:none;" onchange="handleFileUpload('{sh_id}', this.files[0])">
              <span class="icon">📤</span>
              <span id="drop_text_{sh_id}">Click to browse file or drag & drop here</span>
            </div>
            <div id="upload_preview_{sh_id}" class="upload-preview-box" style="display:none;">
              <img id="upload_img_{sh_id}" src="" alt="Thumbnail" onerror="this.style.display='none'">
              <div class="upload-file-details">
                <div class="upload-file-name" id="upload_name_{sh_id}"></div>
                <div class="upload-file-status" id="upload_status_{sh_id}">✅ Selected for this shot</div>
              </div>
              <button type="button" class="btn btn-sec btn-sm" onclick="clearUpload('{sh_id}')" title="Clear upload">✕</button>
            </div>
            <input type="hidden" id="upload_path_{sh_id}" value="">
          </div>

          <!-- 2. Direct Web Link (URL) -->
          <div class="manual-card" id="card_opt_link_{sh_id}">
            <div class="manual-card-header">
              <label>
                <input type="radio" name="pick_{sh_id}" value="link" id="radio_link_{sh_id}" onchange="updateChoices()">
                <span>🔗 Direct Web Link (URL)</span>
              </label>
              <span class="opt-tag">Direct link to MP4 or Image</span>
            </div>
            <div class="text-row">
              <input type="text" id="link_url_{sh_id}" placeholder="https://example.com/footage.mp4" oninput="handleLinkInput('{sh_id}')" onchange="updateChoices()">
              <button type="button" class="btn btn-sec btn-sm" onclick="validateLink('{sh_id}')">Verify</button>
              <span id="link_status_{sh_id}" style="font-size:12px;"></span>
            </div>
          </div>

          <!-- 3. Branded Graphic Card -->
          <div class="manual-card" id="card_opt_card_{sh_id}">
            <div class="manual-card-header">
              <label>
                <input type="radio" name="pick_{sh_id}" value="card" id="radio_card_{sh_id}" onchange="updateChoices()">
                <span>🎨 Branded 1080p Headline Card</span>
              </label>
              <span class="badge-default">DEFAULT</span>
            </div>
            <div class="text-row">
              <input type="text" id="card_text_{sh_id}" value="{html.escape(default_card_text)}" placeholder="Headline text to display on graphic card..." oninput="handleCardInput('{sh_id}')" onchange="updateChoices()">
            </div>
          </div>

          <!-- 4. Local File Path on Disk -->
          <div class="manual-card" id="card_opt_own_{sh_id}">
            <div class="manual-card-header">
              <label>
                <input type="radio" name="pick_{sh_id}" value="own_file" id="radio_own_{sh_id}" onchange="updateChoices()">
                <span>💻 File Path on Disk</span>
              </label>
              <span class="opt-tag">Local file path</span>
            </div>
            <div class="text-row">
              <input type="text" id="own_file_{sh_id}" placeholder="C:/path/to/local_footage.mp4" oninput="handleOwnFileInput('{sh_id}')" onchange="updateChoices()">
            </div>
          </div>
        </div>

      </div> <!-- end shot-card -->
"""
        html_content += "    </div> <!-- end scene-block -->\n"

    # Embedded JavaScript
    html_content += f"""
  </div> <!-- end scenes-container -->

  <script>
    const CANDIDATES_DATA = {embedded_json};
    const RUN_DATE = CANDIDATES_DATA.date || "{date_str}";
    const STORAGE_KEY = "media_picks_" + RUN_DATE;

    function initPage() {{
      initDropzones();

      let saved = null;
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) saved = JSON.parse(raw);
      }} catch (e) {{
        console.warn("localStorage read blocked or unavailable", e);
      }}

      CANDIDATES_DATA.shots.forEach(sh => {{
        const shId = sh.shot_id;
        const radios = document.getElementsByName("pick_" + shId);
        let applied = false;

        if (saved && saved.picks && saved.picks[shId]) {{
          const p = saved.picks[shId];
          if (p.action === "upload") {{
            for (let r of radios) if (r.value === "upload") r.checked = true;
            const input = document.getElementById("upload_path_" + shId);
            if (input && p.file_path) {{
              input.value = p.file_path;
              const prevBox = document.getElementById("upload_preview_" + shId);
              const uploadName = document.getElementById("upload_name_" + shId);
              const dropText = document.getElementById("drop_text_" + shId);
              if (prevBox) prevBox.style.display = "flex";
              if (uploadName) uploadName.textContent = p.file_path.split("/").pop();
              if (dropText) dropText.textContent = "✅ File selected. Click to replace.";
            }}
            applied = true;
          }} else if (p.action === "link") {{
            for (let r of radios) if (r.value === "link") r.checked = true;
            const input = document.getElementById("link_url_" + shId);
            if (input && p.url) input.value = p.url;
            applied = true;
          }} else if (p.action === "pick") {{
            for (let r of radios) {{
              if (r.value === "pick" && r.dataset.candId === p.candidate_id) {{
                r.checked = true;
                applied = true;
                break;
              }}
            }}
          }} else if (p.action === "card") {{
            for (let r of radios) if (r.value === "card") r.checked = true;
            const input = document.getElementById("card_text_" + shId);
            if (input && p.card_text) input.value = p.card_text;
            applied = true;
          }} else if (p.action === "own_file") {{
            for (let r of radios) if (r.value === "own_file") r.checked = true;
            const input = document.getElementById("own_file_" + shId);
            if (input && p.file_path) input.value = p.file_path;
            applied = true;
          }}
        }}

        if (!applied) {{
          // Default: if clip candidate is auto-selected, use it; otherwise default to card
          if (sh.auto_pick_type === "clip" && sh.auto_pick_id) {{
            for (let r of radios) {{
              if (r.value === "pick" && r.dataset.candId === sh.auto_pick_id) {{
                r.checked = true;
                applied = true;
                break;
              }}
            }}
          }}
          if (!applied) {{
            for (let r of radios) if (r.value === "card") r.checked = true;
          }}
        }}
      }});

      updateChoices();
    }}

    function initDropzones() {{
      CANDIDATES_DATA.shots.forEach(sh => {{
        const shId = sh.shot_id;
        const dz = document.getElementById("dropzone_" + shId);
        if (!dz) return;
        dz.addEventListener("dragover", (e) => {{
          e.preventDefault();
          dz.classList.add("drag-over");
        }});
        dz.addEventListener("dragleave", (e) => {{
          e.preventDefault();
          dz.classList.remove("drag-over");
        }});
        dz.addEventListener("drop", (e) => {{
          e.preventDefault();
          dz.classList.remove("drag-over");
          if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {{
            handleFileUpload(shId, e.dataTransfer.files[0]);
          }}
        }});
      }});
    }}

    async function handleFileUpload(shId, file) {{
      if (!file) return;
      const dropText = document.getElementById("drop_text_" + shId);
      const prevBox = document.getElementById("upload_preview_" + shId);
      const uploadImg = document.getElementById("upload_img_" + shId);
      const uploadName = document.getElementById("upload_name_" + shId);
      const uploadPath = document.getElementById("upload_path_" + shId);
      const radioUpload = document.getElementById("radio_upload_" + shId);

      if (window.location.protocol.startsWith("http")) {{
        if (dropText) dropText.textContent = "⏳ Uploading " + file.name + "...";
        try {{
          const fd = new FormData();
          fd.append("shot_id", shId);
          fd.append("file", file);
          const res = await fetch("/api/runs/" + encodeURIComponent(RUN_DATE) + "/upload-media", {{
            method: "POST",
            body: fd
          }});
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));

          uploadPath.value = data.file_path;
          uploadName.textContent = file.name + (data.kind ? " (" + data.kind.toUpperCase() + ")" : "");
          if (data.thumb_url) {{
            uploadImg.src = data.thumb_url;
            uploadImg.style.display = "block";
          }} else {{
            uploadImg.style.display = "none";
          }}
          if (prevBox) prevBox.style.display = "flex";
          if (dropText) dropText.textContent = "✅ File uploaded! Click to replace.";
          if (radioUpload) radioUpload.checked = true;
          updateChoices();
        }} catch (err) {{
          alert("Upload failed: " + err.message);
          if (dropText) dropText.textContent = "❌ Upload failed. Click to try again.";
        }}
      }} else {{
        // Local file:/// mode
        uploadPath.value = file.name;
        uploadName.textContent = file.name;
        if (prevBox) prevBox.style.display = "flex";
        if (dropText) dropText.textContent = "✅ File selected (" + file.name + "). Ensure it is in run folder.";
        if (radioUpload) radioUpload.checked = true;
        updateChoices();
      }}
    }}

    function clearUpload(shId) {{
      const uploadPath = document.getElementById("upload_path_" + shId);
      const prevBox = document.getElementById("upload_preview_" + shId);
      const dropText = document.getElementById("drop_text_" + shId);
      const fileInput = document.getElementById("file_input_" + shId);
      if (uploadPath) uploadPath.value = "";
      if (prevBox) prevBox.style.display = "none";
      if (dropText) dropText.textContent = "Click to browse file or drag & drop here";
      if (fileInput) fileInput.value = "";
      const radioCard = document.getElementById("radio_card_" + shId);
      if (radioCard) radioCard.checked = true;
      updateChoices();
    }}

    async function validateLink(shId) {{
      const linkInput = document.getElementById("link_url_" + shId);
      const statusEl = document.getElementById("link_status_" + shId);
      const url = (linkInput ? linkInput.value : "").trim();
      if (!url) {{
        if (statusEl) statusEl.innerHTML = "<span style='color:#E06C75;'>Please enter a URL.</span>";
        return;
      }}
      if (statusEl) statusEl.innerHTML = "⏳ Validating...";
      if (window.location.protocol.startsWith("http")) {{
        try {{
          const res = await fetch("/api/runs/" + encodeURIComponent(RUN_DATE) + "/validate-media-link", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }},
            body: JSON.stringify({{ url: url }})
          }});
          const data = await res.json();
          if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));
          statusEl.innerHTML = "<span style='color:#28A745;'>✅ Domain allowed (" + data.domain + ")</span>";
          const radioLink = document.getElementById("radio_link_" + shId);
          if (radioLink) radioLink.checked = true;
          updateChoices();
        }} catch (e) {{
          statusEl.innerHTML = "<span style='color:#E06C75;'>❌ " + e.message + "</span>";
        }}
      }} else {{
        if (url.startsWith("http://") || url.startsWith("https://")) {{
          statusEl.innerHTML = "<span style='color:#28A745;'>✅ URL format OK</span>";
          const radioLink = document.getElementById("radio_link_" + shId);
          if (radioLink) radioLink.checked = true;
          updateChoices();
        }} else {{
          statusEl.innerHTML = "<span style='color:#E06C75;'>❌ Must start with http:// or https://</span>";
        }}
      }}
    }}

    function handleLinkInput(shId) {{
      const input = document.getElementById("link_url_" + shId);
      if (input && input.value.trim().length > 0) {{
        const r = document.getElementById("radio_link_" + shId);
        if (r) r.checked = true;
      }}
      updateChoices();
    }}

    function handleCardInput(shId) {{
      const r = document.getElementById("radio_card_" + shId);
      if (r) r.checked = true;
      updateChoices();
    }}

    function handleOwnFileInput(shId) {{
      const input = document.getElementById("own_file_" + shId);
      if (input && input.value.trim().length > 0) {{
        const r = document.getElementById("radio_own_" + shId);
        if (r) r.checked = true;
      }}
      updateChoices();
    }}

    function fillCardsWithTopClips() {{
      let switched = 0;
      CANDIDATES_DATA.shots.forEach(sh => {{
        const shId = sh.shot_id;
        const radios = document.getElementsByName("pick_" + shId);
        let curAction = "card";
        for (let r of radios) if (r.checked) curAction = r.value;
        if (curAction === "card") {{
          for (let r of radios) {{
            if (r.value === "pick") {{
              r.checked = true;
              switched++;
              break;
            }}
          }}
        }}
      }});
      updateChoices();
      const statusEl = document.getElementById("summary-status");
      statusEl.innerHTML += ` — Switched ${{switched}} card(s) to top candidate clip.`;
    }}

    function setAllGraphicCards() {{
      CANDIDATES_DATA.shots.forEach(sh => {{
        const shId = sh.shot_id;
        const radioCard = document.getElementById("radio_card_" + shId);
        if (radioCard) radioCard.checked = true;
      }});
      updateChoices();
      const statusEl = document.getElementById("summary-status");
      statusEl.innerHTML += " — Set all shots to 1080p Graphic Cards.";
    }}

    function getPicksPayload() {{
      const picks = {{}};
      CANDIDATES_DATA.shots.forEach(sh => {{
        const shId = sh.shot_id;
        const radios = document.getElementsByName("pick_" + shId);
        let selectedAction = "card";
        let chosenRadio = null;

        for (let r of radios) {{
          if (r.checked) {{
            selectedAction = r.value;
            chosenRadio = r;
            break;
          }}
        }}

        if (selectedAction === "upload") {{
          const path = document.getElementById("upload_path_" + shId)?.value || "";
          picks[shId] = {{
            action: "upload",
            scene_id: sh.scene_id,
            shot_index: sh.shot_index,
            file_path: path
          }};
        }} else if (selectedAction === "link") {{
          const url = document.getElementById("link_url_" + shId)?.value || "";
          picks[shId] = {{
            action: "link",
            scene_id: sh.scene_id,
            shot_index: sh.shot_index,
            url: url
          }};
        }} else if (selectedAction === "card") {{
          const txt = document.getElementById("card_text_" + shId)?.value || sh.scene_headline;
          picks[shId] = {{
            action: "card",
            scene_id: sh.scene_id,
            shot_index: sh.shot_index,
            card_text: txt
          }};
        }} else if (selectedAction === "own_file") {{
          const path = document.getElementById("own_file_" + shId)?.value || "";
          picks[shId] = {{
            action: "own_file",
            scene_id: sh.scene_id,
            shot_index: sh.shot_index,
            file_path: path
          }};
        }} else if (selectedAction === "pick" && chosenRadio) {{
          picks[shId] = {{
            action: "pick",
            scene_id: sh.scene_id,
            shot_index: sh.shot_index,
            candidate_id: chosenRadio.dataset.candId,
            source: chosenRadio.dataset.source
          }};
        }}
      }});

      return {{
        date: RUN_DATE,
        timestamp: new Date().toISOString(),
        picks: picks
      }};
    }}

    function updateChoices() {{
      const payload = getPicksPayload();
      let uploads = 0, links = 0, cards = 0, own = 0, clips = 0;

      for (let shId in payload.picks) {{
        const act = payload.picks[shId].action;
        if (act === "upload") uploads++;
        else if (act === "link") links++;
        else if (act === "card") cards++;
        else if (act === "own_file") own++;
        else if (act === "pick") clips++;

        // Highlight selected manual option card
        ["upload", "link", "card", "own"].forEach(opt => {{
          const optEl = document.getElementById("card_opt_" + opt + "_" + shId);
          if (optEl) {{
            if (act === opt || (opt === "own" && act === "own_file")) {{
              optEl.classList.add("active");
            }} else {{
              optEl.classList.remove("active");
            }}
          }}
        }});

        // Highlight candidate item if picked
        const items = document.querySelectorAll(`[id^="cand_${{shId}}_"]`);
        items.forEach(it => it.classList.remove("selected"));
        if (act === "pick") {{
          const chosenId = payload.picks[shId].candidate_id;
          const chosenEl = document.getElementById(`cand_${{shId}}_${{chosenId}}`);
          if (chosenEl) chosenEl.classList.add("selected");
        }}
      }}

      try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      }} catch (e) {{}}

      const statusEl = document.getElementById("summary-status");
      statusEl.innerHTML = `Current choices: <strong>${{uploads}}</strong> uploads, <strong>${{links}}</strong> web links, <strong>${{cards}}</strong> graphic cards, <strong>${{own}}</strong> local paths${{clips > 0 ? ', <strong>' + clips + '</strong> clips' : ''}}. (Saved locally)`;
    }}

    function copyChoices() {{
      const payload = getPicksPayload();
      const compactJson = JSON.stringify(payload, null, 2);

      let uploads = 0, links = 0, cards = 0, own = 0, clips = 0;
      for (let k in payload.picks) {{
        const act = payload.picks[k].action;
        if (act === "upload") uploads++;
        else if (act === "link") links++;
        else if (act === "card") cards++;
        else if (act === "own_file") own++;
        else if (act === "pick") clips++;
      }}

      const summaryText = `[Picks Summary] Date: ${{RUN_DATE}} | ${{uploads}} uploads, ${{links}} links, ${{cards}} cards, ${{own}} local files.`;

      navigator.clipboard.writeText(compactJson).then(() => {{
        const statusEl = document.getElementById("summary-status");
        statusEl.innerHTML = `✅ <strong>Copied to clipboard!</strong> ${{summaryText}} Paste into your reply or save to picks.json.`;
      }}).catch(err => {{
        prompt("Copy failed automatically. Please copy the JSON below:", compactJson);
      }});
    }}

    function downloadPicks() {{
      const payload = getPicksPayload();
      const str = JSON.stringify(payload, null, 2);
      const blob = new Blob([str], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "picks.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    async function applyPicksToOffice() {{
      const statusEl = document.getElementById("summary-status");
      if (!window.location.protocol.startsWith("http")) {{
        statusEl.innerHTML = "⚠ Opened as local file — use <strong>Open Visual Picker</strong> inside Boss Office, or click <strong>Download picks.json</strong>.";
        return;
      }}
      const btn = document.getElementById("btnApplyOffice");
      const origText = btn ? btn.textContent : "";
      if (btn) {{
        btn.disabled = true;
        btn.textContent = "⏳ Applying & Starting Render...";
      }}
      statusEl.innerHTML = "⏳ Submitting visual picks to 2D Office & queuing render...";
      try {{
        const payload = getPicksPayload();
        const res = await fetch("/api/runs/" + encodeURIComponent(RUN_DATE) + "/apply-picks", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ picks: payload }})
        }});
        const data = await res.json();
        if (!res.ok) {{
          throw new Error(data.detail || ("HTTP " + res.status));
        }}
        statusEl.innerHTML = "✅ <strong>Visual picks submitted!</strong> Rendering & publishing started — watch the 2D Office floorplan.";
        if (window.parent && window.parent !== window) {{
          window.parent.postMessage({{ type: "office_picks_submitted", date: RUN_DATE, job_id: data.job_id }}, "*");
        }}
      }} catch (err) {{
        statusEl.innerHTML = "❌ <strong>Failed to apply picks:</strong> " + err.message;
        if (btn) {{
          btn.disabled = false;
          btn.textContent = origText;
        }}
      }}
    }}

    window.addEventListener("DOMContentLoaded", () => {{
      const btn = document.getElementById("btnApplyOffice");
      if (btn && !window.location.protocol.startsWith("http")) {{
        btn.title = "Open via 2D Office server to apply directly";
      }}
      initPage();
    }});
  </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Picker HTML generated at {output_path}")



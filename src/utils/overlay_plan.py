"""Single source of truth for overlay planning, attribution sanitization, motion resolution, and layout validation."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from src.utils.motion_graphics import (
    draw_headline_split,
    draw_illustrative_tag,
    draw_lower_third,
    draw_news_ticker,
    draw_outro_card,
    draw_quote_card,
    draw_stat_callout,
    draw_test_run_tag,
    draw_title_card,
)

# 1. Stock / Illustrative Sources
STOCK_SOURCES: tuple[str, ...] = (
    "wikimedia_commons",
    "pixabay",
    "coverr",
    "openverse",
    "nasa",
    "internet_archive",
    "unsplash",
    "pexels",
)

# 2. Timing Formulas & Constants (in seconds)
STAT_CALLOUT_START_OFFSET = 0.3
STAT_CALLOUT_MAX_DURATION = 6.5
STAT_CALLOUT_END_MARGIN = 0.4

QUOTE_CARD_START_OFFSET = 0.3
QUOTE_CARD_MAX_DURATION = 7.0
QUOTE_CARD_END_MARGIN = 0.4

SPLIT_START_OFFSET = 0.2
SPLIT_MAX_DURATION = 7.5
SPLIT_END_MARGIN = 0.4

LOWER_THIRD_SPEECH_OFFSET = 0.3
LOWER_THIRD_MAX_DURATION = 7.0
LOWER_THIRD_END_MARGIN = 0.5

ILLUSTRATIVE_TAG_OFFSET = 0.2
ILLUSTRATIVE_TAG_END_MARGIN = 0.2

MAX_CREDIT_CHARS = 48


def sanitize_credit(scene_script: dict[str, Any], channel_name: str = "Nepal Daily Briefing") -> str:
    """Sanitizes and formats source credit attribution.

    Rules:
    - Accepts `source` (string, comma-split on `,` or ` and `), falls back to `sources` (list), then channel_name.
    - Strips trailing punctuation, dedupes, caps at 3 outlets.
    - Prefixes `Source: ` once.
    - Truncates to fit credit line width with `…`.
    """
    raw_source = scene_script.get("source")
    raw_sources = scene_script.get("sources")

    outlets: list[str] = []

    if raw_source and isinstance(raw_source, str) and raw_source.strip():
        # Split on comma or " and "
        parts = re.split(r",|\band\b", raw_source)
        for p in parts:
            clean = p.strip().strip(". ,;:")
            # Remove redundant leading "Source:" if already in string
            clean = re.sub(r"^source:\s*", "", clean, flags=re.IGNORECASE).strip()
            if clean and clean not in outlets:
                outlets.append(clean)
    elif raw_sources and isinstance(raw_sources, list):
        for s in raw_sources:
            if isinstance(s, str):
                parts = re.split(r",|\band\b", s)
                for p in parts:
                    clean = p.strip().strip(". ,;:")
                    clean = re.sub(r"^source:\s*", "", clean, flags=re.IGNORECASE).strip()
                    if clean and clean not in outlets:
                        outlets.append(clean)

    if not outlets:
        # Fall back to channel name
        fallback = channel_name.strip().strip(". ,;:")
        return fallback

    # Cap at 3 outlets
    outlets = outlets[:3]
    joined = ", ".join(outlets)

    # Prefix Source:
    full_credit = f"Source: {joined}"

    # Truncate with ellipsis if needed
    if len(full_credit) > MAX_CREDIT_CHARS:
        full_credit = full_credit[: MAX_CREDIT_CHARS - 1].rstrip() + "…"

    return full_credit


def resolve_shot_motion(shot: dict[str, Any], index: int = 0, featured_card: bool = False) -> str | None:
    """Resolves camera motion type for a shot.

    Rules:
    - Pass through `zoom_in | zoom_out | none | ken_burns` if explicitly configured on non-still or user-specified.
    - For stills (`type in ("image", "card")`):
      - Force `none` for `type == "card"` if a featured card is scheduled (avoid zooming text).
      - If manifest is silent (or default `zoom_in`), assign `zoom_out` for odd indices to ensure variety.
    - For video: returns None.
    """
    shot_type = shot.get("type", "image")
    is_still = shot_type in ("image", "card")
    if not is_still:
        return None

    if shot_type == "card" and featured_card:
        return "none"

    explicit_motion = shot.get("motion")
    if explicit_motion in ("zoom_in", "zoom_out", "none", "ken_burns"):
        if explicit_motion != "zoom_in":
            return explicit_motion

    # Deterministic variety for stills
    return "zoom_out" if (index % 2 == 1) else "zoom_in"


def build_scene_overlays(
    scene_id: str,
    scene_script: dict[str, Any],
    shots: list[dict[str, Any]],
    classification: Any,  # SceneClassification
    style_config: dict[str, Any],
    channel_name: str,
    rects: dict[str, Any],
    scene_start: float,
    scene_end: float,
    speech_start: float | None = None,
    font_candidates: list[str] | None = None,
    overlays_dir: Path | None = None,
    run_kind: str = "production",
    date_display_str: str = "",
) -> list[dict[str, Any]]:
    """Builds the complete overlay records for a single scene and generates PNG assets.

    Single source of truth used across motion planning, sync_render_plan_data, and composer.
    """
    sc_type = scene_script.get("type") or ("intro" if scene_id == "intro" else ("outro" if scene_id == "outro" else "story"))
    sp_start = scene_start if speech_start is None else speech_start
    font_cands = font_candidates or style_config.get("fonts", {}).get("primary_candidates", [])
    recipe = getattr(classification, "recipe", "cinematic_broll_full")
    recipe_data = getattr(classification, "data", {})

    overlays: list[dict[str, Any]] = []

    # Safe rect lookups
    title_rect = rects.get("title_card", {"x": 200, "y": 300, "w": 1520, "h": 480})
    outro_rect = rects.get("outro_card", {"x": 200, "y": 300, "w": 1520, "h": 480})
    stat_rect = rects.get("stat_callout", {"x": 100, "y": 200, "w": 650, "h": 420})
    quote_rect = rects.get("quote_card", {"x": 100, "y": 200, "w": 900, "h": 460})
    split_rect = rects.get("headline_split", {"x": 100, "y": 200, "w": 900, "h": 480})
    lt_rect = rects.get("lower_third", {"x": 100, "y": 730, "w": 1720, "h": 155})
    tag_rect = rects.get("illustrative_tag", {"x": 1540, "y": 54, "w": 284, "h": 44})
    test_rect = rects.get("test_run_tag", {"x": 48, "y": 48, "w": 280, "h": 48})

    if sc_type == "intro":
        card_start = round(scene_start, 3)
        card_end = round(scene_end, 3)
        png_name = f"{scene_id}_title.png"
        png_rel = f"motion/overlays/{png_name}"

        if overlays_dir is not None:
            png_path = overlays_dir / png_name
            png_path.parent.mkdir(parents=True, exist_ok=True)
            draw_title_card(
                channel_name=channel_name,
                date_str=date_display_str or "Daily Briefing",
                output_path=png_path,
                width=title_rect["w"],
                height=title_rect["h"],
                font_candidates=font_cands,
            )

        overlays.append({
            "id": f"{scene_id}_title_card",
            "scene_id": scene_id,
            "type": "title_card",
            "png": png_rel,
            "start": card_start,
            "end": card_end,
            "x": title_rect["x"],
            "y": title_rect["y"],
            "w": title_rect["w"],
            "h": title_rect["h"],
            "animation": "fade",
            "fade_in": 0.3,
            "fade_out": 0.3,
        })

    elif sc_type == "outro":
        card_start = round(scene_start, 3)
        card_end = round(scene_end, 3)
        png_name = f"{scene_id}_outro.png"
        png_rel = f"motion/overlays/{png_name}"

        if overlays_dir is not None:
            png_path = overlays_dir / png_name
            png_path.parent.mkdir(parents=True, exist_ok=True)
            draw_outro_card(
                channel_name=channel_name,
                output_path=png_path,
                width=outro_rect["w"],
                height=outro_rect["h"],
                font_candidates=font_cands,
            )

        overlays.append({
            "id": f"{scene_id}_outro_card",
            "scene_id": scene_id,
            "type": "outro_card",
            "png": png_rel,
            "start": card_start,
            "end": card_end,
            "x": outro_rect["x"],
            "y": outro_rect["y"],
            "w": outro_rect["w"],
            "h": outro_rect["h"],
            "animation": "fade",
            "fade_in": 0.3,
            "fade_out": 0.3,
        })

    else:
        # Story scene
        headline = scene_script.get("headline", "")
        credit_label = sanitize_credit(scene_script, channel_name=channel_name)

        if recipe == "stat_callout":
            stat_start = round(scene_start + STAT_CALLOUT_START_OFFSET, 3)
            stat_end = round(min(stat_start + STAT_CALLOUT_MAX_DURATION, scene_end - STAT_CALLOUT_END_MARGIN), 3)
            if stat_end <= stat_start:
                stat_end = round(scene_end - 0.2, 3)

            png_name = f"{scene_id}_stat_callout.png"
            png_rel = f"motion/overlays/{png_name}"

            if overlays_dir is not None:
                png_path = overlays_dir / png_name
                png_path.parent.mkdir(parents=True, exist_ok=True)
                draw_stat_callout(
                    stat_value=recipe_data.get("value", ""),
                    stat_label=recipe_data.get("label", headline),
                    stat_delta=recipe_data.get("delta"),
                    source_credit=credit_label,
                    output_path=png_path,
                    width=stat_rect["w"],
                    height=stat_rect["h"],
                    font_candidates=font_cands,
                )

            overlays.append({
                "id": f"{scene_id}_stat_callout",
                "scene_id": scene_id,
                "type": "stat_callout",
                "png": png_rel,
                "start": stat_start,
                "end": stat_end,
                "x": stat_rect["x"],
                "y": stat_rect["y"],
                "w": stat_rect["w"],
                "h": stat_rect["h"],
                "animation": "slide_up",
                "fade_in": 0.25,
                "fade_out": 0.25,
            })

        elif recipe == "quote_card":
            quote_start = round(scene_start + QUOTE_CARD_START_OFFSET, 3)
            quote_end = round(min(quote_start + QUOTE_CARD_MAX_DURATION, scene_end - QUOTE_CARD_END_MARGIN), 3)
            if quote_end <= quote_start:
                quote_end = round(scene_end - 0.2, 3)

            png_name = f"{scene_id}_quote_card.png"
            png_rel = f"motion/overlays/{png_name}"

            if overlays_dir is not None:
                png_path = overlays_dir / png_name
                png_path.parent.mkdir(parents=True, exist_ok=True)
                draw_quote_card(
                    quote_text=recipe_data.get("quote", ""),
                    speaker_name=recipe_data.get("speaker", "Official Statement"),
                    speaker_title=recipe_data.get("title"),
                    source_credit=credit_label,
                    output_path=png_path,
                    width=quote_rect["w"],
                    height=quote_rect["h"],
                    font_candidates=font_cands,
                )

            overlays.append({
                "id": f"{scene_id}_quote_card",
                "scene_id": scene_id,
                "type": "quote_card",
                "png": png_rel,
                "start": quote_start,
                "end": quote_end,
                "x": quote_rect["x"],
                "y": quote_rect["y"],
                "w": quote_rect["w"],
                "h": quote_rect["h"],
                "animation": "fade",
                "fade_in": 0.25,
                "fade_out": 0.25,
            })

        elif recipe == "headline_split":
            split_start = round(scene_start + SPLIT_START_OFFSET, 3)
            split_end = round(min(split_start + SPLIT_MAX_DURATION, scene_end - SPLIT_END_MARGIN), 3)
            if split_end <= split_start:
                split_end = round(scene_end - 0.2, 3)

            png_name = f"{scene_id}_headline_split.png"
            png_rel = f"motion/overlays/{png_name}"

            if overlays_dir is not None:
                png_path = overlays_dir / png_name
                png_path.parent.mkdir(parents=True, exist_ok=True)
                draw_headline_split(
                    kicker=recipe_data.get("kicker", "DEVELOPING"),
                    headline=headline,
                    bullets=recipe_data.get("bullets", []),
                    source_credit=credit_label,
                    output_path=png_path,
                    width=split_rect["w"],
                    height=split_rect["h"],
                    font_candidates=font_cands,
                )

            overlays.append({
                "id": f"{scene_id}_headline_split",
                "scene_id": scene_id,
                "type": "headline_split",
                "png": png_rel,
                "start": split_start,
                "end": split_end,
                "x": split_rect["x"],
                "y": split_rect["y"],
                "w": split_rect["w"],
                "h": split_rect["h"],
                "animation": "slide_right",
                "fade_in": 0.25,
                "fade_out": 0.25,
            })

        else:
            # Default: lower third
            lt_start = round(sp_start + LOWER_THIRD_SPEECH_OFFSET, 3)
            lt_end = round(min(lt_start + LOWER_THIRD_MAX_DURATION, scene_end - LOWER_THIRD_END_MARGIN), 3)
            if lt_end <= lt_start:
                lt_end = round(scene_end - 0.2, 3)

            png_name = f"{scene_id}_lower_third.png"
            png_rel = f"motion/overlays/{png_name}"

            if overlays_dir is not None:
                png_path = overlays_dir / png_name
                png_path.parent.mkdir(parents=True, exist_ok=True)
                draw_lower_third(
                    headline=headline,
                    source_text=credit_label,
                    output_path=png_path,
                    width=lt_rect["w"],
                    height=lt_rect["h"],
                    font_candidates=font_cands,
                )

            overlays.append({
                "id": f"{scene_id}_lower_third",
                "scene_id": scene_id,
                "type": "lower_third",
                "png": png_rel,
                "start": lt_start,
                "end": lt_end,
                "x": lt_rect["x"],
                "y": lt_rect["y"],
                "w": lt_rect["w"],
                "h": lt_rect["h"],
                "animation": "slide_up",
                "fade_in": 0.2,
                "fade_out": 0.2,
            })

        # Illustrative tag if stock shots used
        has_stock = any(s.get("source") in STOCK_SOURCES for s in shots)
        if has_stock:
            tag_start = round(scene_start + ILLUSTRATIVE_TAG_OFFSET, 3)
            tag_end = round(max(scene_start + 0.3, scene_end - ILLUSTRATIVE_TAG_END_MARGIN), 3)
            png_name = "illustrative_tag.png"
            png_rel = f"motion/overlays/{png_name}"

            if overlays_dir is not None:
                png_path = overlays_dir / png_name
                png_path.parent.mkdir(parents=True, exist_ok=True)
                draw_illustrative_tag(
                    output_path=png_path,
                    width=tag_rect["w"],
                    height=tag_rect["h"],
                    font_candidates=font_cands,
                )

            overlays.append({
                "id": f"{scene_id}_illustrative_tag",
                "scene_id": scene_id,
                "type": "illustrative_tag",
                "png": png_rel,
                "start": tag_start,
                "end": tag_end,
                "x": tag_rect["x"],
                "y": tag_rect["y"],
                "w": tag_rect["w"],
                "h": tag_rect["h"],
                "animation": "fade",
                "fade_in": 0.2,
                "fade_out": 0.2,
            })

    return overlays


def format_date_display(date_str: str) -> str:
    """Formats YYYY-MM-DD into a human-readable broadcast date."""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%A, %B ") + str(dt.day) + dt.strftime(", %Y")
    except Exception:
        return date_str


def build_all_overlays(
    timeline_scenes: list[dict[str, Any]],
    script_scenes: dict[str, Any],
    shots_by_scene: dict[str, list[dict[str, Any]]],
    scene_classifications: dict[str, Any],
    style_config: dict[str, Any],
    channel_name: str,
    resolution: tuple[int, int] | list[int] | str,
    overlays_dir: Path,
    run_kind: str = "production",
    date_display_str: str = "",
    total_duration: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Builds overlays across all scenes and global overlays (watermark).

    Returns:
        (all_overlays, scene_overlays_map)
    """
    if isinstance(resolution, str) and "x" in resolution:
        parts = [int(p) for p in resolution.split("x")]
        w, h = parts[0], parts[1]
    elif isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        w, h = int(resolution[0]), int(resolution[1])
    else:
        w, h = 1920, 1080

    is_vertical = h > w
    rects_key = "layout_rectangles_vertical" if is_vertical else "layout_rectangles"
    rects = style_config.get(rects_key) or style_config.get("layout_rectangles", {})
    font_candidates = style_config.get("fonts", {}).get("primary_candidates", [])

    all_overlays: list[dict[str, Any]] = []
    scene_overlays_map: dict[str, list[dict[str, Any]]] = {}

    # Global watermark if enabled and not production (single global overlay, scene_id: None)
    watermark_enabled = style_config.get("watermark_test_runs", True)
    if watermark_enabled and run_kind != "production":
        test_rect = rects.get("test_run_tag", {"x": 48, "y": 48, "w": 280, "h": 48})
        test_png = overlays_dir / "test_run_tag.png"
        test_png.parent.mkdir(parents=True, exist_ok=True)
        draw_test_run_tag(test_png, width=test_rect["w"], height=test_rect["h"], font_candidates=font_candidates)
        all_overlays.append({
            "id": "global_test_tag",
            "scene_id": None,
            "type": "test_tag",
            "png": f"motion/overlays/{test_png.name}",
            "start": 0.0,
            "end": total_duration,
            "x": test_rect["x"],
            "y": test_rect["y"],
            "w": test_rect["w"],
            "h": test_rect["h"],
            "animation": None,
            "fade_in": 0.0,
            "fade_out": 0.0,
        })

    for sc in timeline_scenes:
        sc_id = sc["id"]
        sc_script = script_scenes.get(sc_id, {})
        sc_shots = shots_by_scene.get(sc_id, [])
        cls = scene_classifications.get(sc_id)
        sc_start = float(sc["start"])
        sc_end = float(sc["end"])
        sp_start = float(sc.get("speech_start", sc_start))

        sc_overlays = build_scene_overlays(
            scene_id=sc_id,
            scene_script=sc_script,
            shots=sc_shots,
            classification=cls,
            style_config=style_config,
            channel_name=channel_name,
            rects=rects,
            scene_start=sc_start,
            scene_end=sc_end,
            speech_start=sp_start,
            font_candidates=font_candidates,
            overlays_dir=overlays_dir,
            run_kind=run_kind,
            date_display_str=date_display_str,
        )
        scene_overlays_map[sc_id] = sc_overlays
        all_overlays.extend(sc_overlays)

    return all_overlays, scene_overlays_map


def validate_layout(plan_data: dict[str, Any], style_config: dict[str, Any]) -> list[str]:
    """Validates overlay layout geometry and safe zone compliance.

    Rules checked:
    - 5% safe margin bounds.
    - Caption band clearance for lower_third AND featured cards.
    - Intra-scene temporal co-visibility and spatial AABB collision.

    Returns:
        List of violation error strings (empty if all checks pass).
    """
    errors: list[str] = []

    res_str = plan_data.get("resolution", "1920x1080")
    if isinstance(res_str, str) and "x" in res_str:
        parts = [int(p) for p in res_str.split("x")]
        w, h = parts[0], parts[1]
    elif isinstance(res_str, (list, tuple)) and len(res_str) >= 2:
        w, h = int(res_str[0]), int(res_str[1])
    else:
        w, h = 1920, 1080

    safe_margin_pct = float(style_config.get("safe_margin_pct", 0.05))
    min_x = round(w * safe_margin_pct)
    max_x = round(w * (1.0 - safe_margin_pct))
    min_y = round(h * safe_margin_pct)
    max_y = round(h * (1.0 - safe_margin_pct))

    is_vertical = h > w
    rects_key = "layout_rectangles_vertical" if is_vertical else "layout_rectangles"
    rects = style_config.get(rects_key) or style_config.get("layout_rectangles", {})
    cb_y = int(rects.get("caption_band", {}).get("y", 1440 if is_vertical else 910))

    # All overlays
    overlays = plan_data.get("overlays", [])
    for ov in overlays:
        ox = int(ov.get("x", 0))
        oy = int(ov.get("y", 0))
        ow = int(ov.get("w", 0))
        oh = int(ov.get("h", 0))
        ov_id = ov.get("id", "unknown")
        ov_type = ov.get("type", "")

        # 1. 5% safe bounds
        if ox < min_x or oy < min_y or (ox + ow) > max_x or (oy + oh) > max_y:
            errors.append(
                f"Overlay '{ov_id}' exceeds 5% safe margin bounds (rect: [{ox},{oy},{ow},{oh}], bounds: [{min_x}..{max_x}, {min_y}..{max_y}])"
            )

        # 2. Caption band clearance for lower_third AND featured cards
        if ov_type in ("lower_third", "quote_card", "stat_callout", "headline_split"):
            if (oy + oh) > cb_y:
                errors.append(
                    f"Overlay '{ov_id}' ({ov_type}) overlaps caption band (bottom y={oy+oh} > {cb_y})"
                )

    # 3. Intra-scene spatial overlap detection (AABB)
    skip_types = {"test_tag", "test_run_tag", "illustrative_tag"}
    for sc in plan_data.get("scenes", []):
        sc_overlays = sc.get("overlays", [])
        for i, ov_a in enumerate(sc_overlays):
            if ov_a.get("type") in skip_types:
                continue
            for ov_b in sc_overlays[i + 1 :]:
                if ov_b.get("type") in skip_types:
                    continue

                # Temporal co-visibility
                a_start = float(ov_a.get("start", 0))
                a_end = float(ov_a.get("end", 0))
                b_start = float(ov_b.get("start", 0))
                b_end = float(ov_b.get("end", 0))
                if a_end <= b_start or b_end <= a_start:
                    continue

                # Spatial AABB overlap
                ax, ay = int(ov_a.get("x", 0)), int(ov_a.get("y", 0))
                aw, ah = int(ov_a.get("w", 0)), int(ov_a.get("h", 0))
                bx, by = int(ov_b.get("x", 0)), int(ov_b.get("y", 0))
                bw, bh = int(ov_b.get("w", 0)), int(ov_b.get("h", 0))

                if (ax < bx + bw) and (ax + aw > bx) and (ay < by + bh) and (ay + ah > by):
                    errors.append(
                        f"Overlays '{ov_a.get('id')}' and '{ov_b.get('id')}' overlap spatially while co-visible in scene '{sc.get('id')}'"
                    )

    return errors

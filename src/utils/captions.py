"""Caption styling and geometry resolution for subtitle rendering and plan contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CaptionStyle:
    font_family: str
    font_size: int
    max_lines: int
    line_height_pct: float
    highlight_color: str
    box_rgba: list[int]
    gap_from_band_px: int
    caption_height: int
    margin_v: int
    caption_top: int
    caption_bottom: int
    is_vertical: bool

    def to_plan_dict(self) -> dict[str, Any]:
        """Returns caption_style dictionary conforming to render_plan contract."""
        return {
            "font": self.font_family,
            "font_size": self.font_size,
            "position": "bottom",
            "highlight_color": self.highlight_color,
            "margin_v": self.margin_v,
            "max_lines": self.max_lines,
            "line_height_pct": self.line_height_pct,
            "box_rgba": self.box_rgba,
            "gap_from_band_px": self.gap_from_band_px,
            "caption_top": self.caption_top,
            "caption_bottom": self.caption_bottom,
            "caption_height": self.caption_height,
        }


def parse_resolution(resolution: tuple[int, int] | list[int] | str) -> tuple[int, int]:
    """Parses resolution representation into (width, height)."""
    if isinstance(resolution, str):
        parts = resolution.lower().split("x")
        return int(parts[0]), int(parts[1])
    return int(resolution[0]), int(resolution[1])


def resolve_style(
    resolution: tuple[int, int] | list[int] | str = (1920, 1080),
    style_config: dict[str, Any] | None = None,
) -> CaptionStyle:
    """Resolves font size, margins, and geometric extents for captions.

    Guarantees caption_top >= overlay_bottom + gap_from_band_px so captions
    never collide with lower thirds.
    """
    w, h = parse_resolution(resolution)
    is_vertical = h > w
    cfg = style_config or {}

    captions_cfg = cfg.get("captions", {})
    font_family = str(captions_cfg.get("font_family", "Segoe UI Semibold"))
    size_landscape = int(captions_cfg.get("size_landscape", 44))
    size_vertical = int(captions_cfg.get("size_vertical", 64))
    font_size = size_vertical if is_vertical else size_landscape

    max_lines = int(captions_cfg.get("max_lines", 2))
    line_height_pct = float(captions_cfg.get("line_height_pct", 1.25))
    highlight_color = str(captions_cfg.get("highlight_color", "#F5A623"))
    box_rgba = list(captions_cfg.get("box_rgba", [7, 20, 38, 224]))
    gap_from_band_px = int(captions_cfg.get("gap_from_band_px", 24))

    caption_height = int(max_lines * font_size * line_height_pct)

    rect_key = "layout_rectangles_vertical" if is_vertical else "layout_rectangles"
    rects = cfg.get(rect_key, {})
    lower_third = rects.get("lower_third", {})
    lt_y = int(lower_third.get("y", 1240 if is_vertical else 730))
    lt_h = int(lower_third.get("h", 180 if is_vertical else 155))
    overlay_bottom = lt_y + lt_h

    caption_band = rects.get("caption_band", {})
    nominal_top = int(caption_band.get("y", overlay_bottom + gap_from_band_px))

    # Assert and enforce caption_top >= overlay_bottom + gap
    min_caption_top = overlay_bottom + gap_from_band_px
    caption_top = max(nominal_top, min_caption_top)
    caption_bottom = caption_top + caption_height

    # Honor caption_margin_v as a floor, shrinking font_size if needed to avoid pushing into lower_third
    target_margin_v = rects.get("caption_margin_v")
    if target_margin_v is not None:
        target_margin_v = int(target_margin_v)
        max_allowed_bottom = h - target_margin_v
        avail_height = max_allowed_bottom - min_caption_top
        if avail_height > 20 and caption_height > avail_height:
            font_size = max(20, int(avail_height / (max_lines * line_height_pct)))
            caption_height = int(max_lines * font_size * line_height_pct)
        caption_top = min_caption_top
        caption_bottom = caption_top + caption_height
        margin_v = max(20, h - caption_bottom)
    else:
        # MarginV in ASS (Alignment=2) is distance from bottom of frame to bottom of text
        margin_v = max(20, h - caption_bottom)

    return CaptionStyle(
        font_family=font_family,
        font_size=font_size,
        max_lines=max_lines,
        line_height_pct=line_height_pct,
        highlight_color=highlight_color,
        box_rgba=box_rgba,
        gap_from_band_px=gap_from_band_px,
        caption_height=caption_height,
        margin_v=margin_v,
        caption_top=caption_top,
        caption_bottom=caption_bottom,
        is_vertical=is_vertical,
    )

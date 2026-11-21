"""Motion graphics generation utilities: Pillow overlay generation and ASS subtitle creation."""
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Any
from PIL import Image, ImageDraw, ImageFont
import PIL.features
import yaml

logger = logging.getLogger(__name__)

# Check Raqm / HarfBuzz text shaping feature at startup
try:
    if not PIL.features.check("raqm"):
        logger.warning(
            "Pillow 'raqm' (HarfBuzz text shaping) is not available; "
            "Devanagari text and complex conjuncts will be rendered with basic freetype without advanced shaping."
        )
except Exception as _e:
    logger.warning("Could not check Pillow 'raqm' feature: %s", _e)

DEFAULT_PRIMARY_FONTS = [
    "C:\\Windows\\Fonts\\NirmalaB.ttf",
    "C:\\Windows\\Fonts\\Nirmala.ttc",
    "C:\\Windows\\Fonts\\segoeuib.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]
DEFAULT_REGULAR_FONTS = [
    "C:\\Windows\\Fonts\\Nirmala.ttf",
    "C:\\Windows\\Fonts\\Nirmala.ttc",
    "C:\\Windows\\Fonts\\segoeui.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


def get_font_candidates(regular: bool = False) -> list[str]:
    """Loads font candidates from config/style.yaml or returns defaults with Nirmala UI Devanagari support."""
    try:
        style_path = Path("config/style.yaml")
        if style_path.exists():
            cfg = yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
            key = "regular_candidates" if regular else "primary_candidates"
            fonts_cfg = cfg.get("fonts", {}).get(key)
            if fonts_cfg and isinstance(fonts_cfg, list):
                return [str(c) for c in fonts_cfg]
    except Exception:
        pass
    return DEFAULT_REGULAR_FONTS if regular else DEFAULT_PRIMARY_FONTS


def get_font_path(font_path_candidates: list[str] | None = None, regular: bool = False) -> str | None:
    """Finds the first existing font file path from candidates or config/style.yaml."""
    candidates = font_path_candidates or get_font_candidates(regular=regular)
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            return str(p)
    return None


def get_font(font_path_candidates: list[str] | None, size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load the first available font candidate, falling back to default."""
    candidates = font_path_candidates or get_font_candidates(regular=False)
    for path_str in candidates:
        p = Path(path_str)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    # Fallback to default
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int, max_lines: int = 2) -> list[str]:
    """Wraps text into up to max_lines lines respecting max_width in pixels.

    Guarantees no word loss across allowed lines, cleanly truncates the final line
    with a single ellipsis if it overflows, and handles long unbreakable tokens.
    """
    words = text.strip().split()
    if not words or max_lines < 1:
        return []

    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word]) if current_line else word
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]

        if w <= max_width:
            current_line.append(word)
        else:
            if len(lines) == max_lines - 1:
                # We are at the final allowed line, break to pack remaining words
                break
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                trunc = word
                while font.getbbox(trunc + "…")[2] - font.getbbox(trunc + "…")[0] > max_width and len(trunc) > 1:
                    trunc = trunc[:-1]
                lines.append(trunc + "…")
                current_line = []

    if len(lines) < max_lines:
        consumed_count = sum(len(ln.replace("…", "").split()) for ln in lines)
        remaining_words = words[consumed_count:]
        if remaining_words:
            final_line = " ".join(remaining_words)
            if font.getbbox(final_line)[2] - font.getbbox(final_line)[0] > max_width:
                while font.getbbox(final_line + "…")[2] - font.getbbox(final_line + "…")[0] > max_width and len(final_line) > 1:
                    final_line = final_line[:-1].rstrip()
                final_line = final_line.rstrip(". ,;:") + "…"
            lines.append(final_line)

    return lines

    return lines[:max_lines]


def draw_rounded_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    border_color: tuple[int, int, int, int] | None = None,
    border_width: int = 0,
) -> None:
    """Draws a rounded rectangle with optional border."""
    draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=border_color, width=border_width)


def draw_title_card(
    channel_name: str,
    date_str: str,
    output_path: Path,
    width: int = 1400,
    height: int = 460,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws intro title card PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded panel background (Navy #0B1F3A with ~92% opacity)
    bg_color = (11, 31, 58, 235)
    border_color = (245, 166, 35, 200) # Amber accent border
    draw_rounded_panel(draw, (0, 0, width, height), radius=24, fill=bg_color, border_color=border_color, border_width=3)

    # Accent top strip
    draw.rounded_rectangle((24, 16, width - 24, 24), radius=4, fill=(245, 166, 35, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    font_channel = get_font(fonts, 50 if width >= 1100 else 40)
    font_title = get_font(fonts, 40 if width >= 1100 else 32)
    font_date = get_font(fonts, 32 if width >= 1100 else 26)
    font_note = get_font(fonts, 24 if width >= 1100 else 20)
    max_w = width - 120

    # Format date into plain words (e.g. Sunday, November 8, 2026)
    try:
        dt = datetime.fromisoformat(date_str)
        date_display = dt.strftime("%A, %B ") + str(dt.day) + dt.strftime(", %Y")
    except Exception:
        date_display = date_str

    # Texts
    # 1. Channel name (Amber accent)
    chan_lines = wrap_text(channel_name.upper(), font_channel, max_width=max_w, max_lines=1)
    draw.text((60, 60), chan_lines[0] if chan_lines else channel_name.upper(), font=font_channel, fill=(245, 166, 35, 255))

    # 2. Headline Tagline
    draw.text((60, 155), "TODAY'S TOP STORIES", font=font_title, fill=(255, 255, 255, 255))

    # 3. Separator line
    draw.line((60, 235, width - 60, 235), fill=(255, 255, 255, 60), width=2)

    # 4. Date in plain words
    date_lines = wrap_text(date_display, font_date, max_width=max_w, max_lines=1)
    draw.text((60, 275), date_lines[0] if date_lines else date_display, font=font_date, fill=(220, 230, 242, 255))

    # 5. Briefing note
    note_text = "Independent daily coverage powered by verified news sources."
    note_lines = wrap_text(note_text, font_note, max_width=max_w, max_lines=2)
    y_note = 360 if len(note_lines) == 1 else 345
    for n_line in note_lines:
        draw.text((60, y_note), n_line, font=font_note, fill=(180, 195, 215, 220))
        y_note += 28

    img.save(str(output_path), "PNG")


def draw_lower_third(
    headline: str,
    source_text: str,
    output_path: Path,
    width: int = 1320,
    height: int = 155,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws story scene lower third with headline and mandatory news source attribution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Main panel background
    bg_color = (11, 31, 58, 235)
    draw_rounded_panel(draw, (0, 0, width, height), radius=16, fill=bg_color)

    # Amber vertical accent bar on left edge
    accent_bar = (0, 0, 12, height)
    draw.rounded_rectangle(accent_bar, radius=6, fill=(245, 166, 35, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    font_headline = get_font(fonts, 34)
    font_source = get_font(fonts, 23)

    # Wrap headline up to 2 lines
    max_text_w = width - 60
    headline_lines = wrap_text(headline, font_headline, max_width=max_text_w, max_lines=2)

    # Render headline
    y_pos = 16
    for line in headline_lines:
        draw.text((36, y_pos), line, font=font_headline, fill=(255, 255, 255, 255))
        y_pos += 42

    # Render Source credit
    credit_line = source_text if source_text.lower().startswith("source:") else f"Source: {source_text}"
    wrapped_credit = wrap_text(credit_line, font_source, max_width=max_text_w, max_lines=1)
    credit_display = wrapped_credit[0] if wrapped_credit else credit_line
    draw.text((36, height - 38), credit_display, font=font_source, fill=(245, 166, 35, 255))

    img.save(str(output_path), "PNG")


def draw_illustrative_tag(
    output_path: Path,
    width: int = 284,
    height: int = 44,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws small 'Illustrative visuals' corner tag."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_color = (11, 31, 58, 220)
    draw_rounded_panel(draw, (0, 0, width, height), radius=10, fill=bg_color)

    fonts = font_candidates or get_font_candidates(regular=False)
    font = get_font(fonts, 20)

    # Dot
    draw.ellipse((14, 18, 22, 26), fill=(245, 166, 35, 255))
    # Text
    draw.text((32, 10), "Illustrative visuals", font=font, fill=(240, 245, 255, 240))

    img.save(str(output_path), "PNG")


def draw_test_run_tag(
    output_path: Path,
    width: int = 180,
    height: int = 44,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws small 'TEST RUN' tag for test pipeline executions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Soft red panel
    bg_color = (217, 83, 79, 235)
    draw_rounded_panel(draw, (0, 0, width, height), radius=10, fill=bg_color)

    fonts = font_candidates or get_font_candidates(regular=False)
    font = get_font(fonts, 22)

    bbox = font.getbbox("TEST RUN")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = (height - th) // 2 - 2

    draw.text((tx, ty), "TEST RUN", font=font, fill=(255, 255, 255, 255))
    img.save(str(output_path), "PNG")


def draw_outro_card(
    channel_name: str,
    output_path: Path,
    width: int = 1400,
    height: int = 460,
    disclose_synthetic_voice: bool = True,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws outro card PNG with mandatory AI-voice disclosure and stock footage notice."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_color = (11, 31, 58, 235)
    border_color = (245, 166, 35, 200)
    draw_rounded_panel(draw, (0, 0, width, height), radius=24, fill=bg_color, border_color=border_color, border_width=3)

    # Accent top strip
    draw.rounded_rectangle((24, 16, width - 24, 24), radius=4, fill=(245, 166, 35, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    font_channel = get_font(fonts, 50 if width >= 1100 else 40)
    font_title = get_font(fonts, 40 if width >= 1100 else 32)
    font_desc = get_font(fonts, 24 if width >= 1100 else 20)
    max_w = width - 120

    # Channel name
    chan_lines = wrap_text(channel_name.upper(), font_channel, max_width=max_w, max_lines=1)
    draw.text((60, 60), chan_lines[0] if chan_lines else channel_name.upper(), font=font_channel, fill=(245, 166, 35, 255))

    # Thanks for watching
    draw.text((60, 155), "THANKS FOR WATCHING", font=font_title, fill=(255, 255, 255, 255))

    draw.line((60, 235, width - 60, 235), fill=(255, 255, 255, 60), width=2)

    # Disclosure lines wrapped to max_w
    if disclose_synthetic_voice:
        line1 = "Narration: AI-generated voice. Footage: stock and archive, illustrative."
        line2 = "See description for full journalist credits and source article links."
        all_desc_lines = wrap_text(line1, font_desc, max_width=max_w, max_lines=2) + wrap_text(line2, font_desc, max_width=max_w, max_lines=2)
    else:
        line = "See description for full journalist credits and source article links."
        all_desc_lines = wrap_text(line, font_desc, max_width=max_w, max_lines=2)

    y_desc = 275
    for d_line in all_desc_lines[:3]:
        draw.text((60, y_desc), d_line, font=font_desc, fill=(220, 230, 242, 235))
        y_desc += 34

    img.save(str(output_path), "PNG")


def format_ass_time(sec: float) -> str:
    """Formats seconds into ASS timestamp H:MM:SS.cc (centiseconds)."""
    total_cs = max(0, int(round(sec * 100.0)))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_ass_text(text: str) -> str:
    """Escapes braces and backslashes for ASS subtitle format and converts newlines to \\N."""
    # Replace backslashes first
    t = text.replace("\\", "")
    # Remove or escape curly braces (override tags in ASS)
    t = t.replace("{", "\\{").replace("}", "\\}")
    # Replace newlines with ASS \N
    t = re.sub(r"\r?\n", r"\\N", t)
    return t


def generate_ass_subtitles(
    captions: list[dict[str, Any]],
    output_path: Path,
    resolution: tuple[int, int] = (1920, 1080),
    font_size: int = 54,
    margin_v: int = 70,
    font_name: str = "Arial",
    highlight_color: str = "#F5A623",
    box_rgba: list[int] | None = None,
    border_style: int = 1,
) -> None:
    """Generates an Advanced SubStation Alpha (.ass) subtitle file positioned in the caption band."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rx, ry = resolution

    back_colour = "&H80000000"
    if box_rgba and len(box_rgba) >= 4:
        r, g, b, a = box_rgba[:4]
        ass_a = max(0, min(255, 255 - int(a)))
        back_colour = f"&H{ass_a:02X}{int(b):02X}{int(g):02X}{int(r):02X}"

    sec_colour = "&H000000FF"
    if highlight_color and highlight_color.startswith("#") and len(highlight_color) == 7:
        hr, hg, hb = highlight_color[1:3], highlight_color[3:5], highlight_color[5:7]
        sec_colour = f"&H00{hb}{hg}{hr}".upper()

    header = f"""[Script Info]
Title: Nepal Daily Briefing Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {rx}
PlayResY: {ry}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,{sec_colour},&H00000000,{back_colour},-1,0,0,0,100,100,0,0,{border_style},3.5,1.0,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogues: list[str] = []
    for cap in captions:
        start_str = format_ass_time(float(cap["start"]))
        end_str = format_ass_time(float(cap["end"]))
        escaped_txt = escape_ass_text(cap["text"])
        dialogues.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{escaped_txt}")

    full_ass = header + "\n".join(dialogues) + "\n"
    output_path.write_text(full_ass, encoding="utf-8")


def clean_stat_label(label: str, value: str | None = None) -> str:
    """Cleans stat card label to entity/topic, stripping duplicate numeric phrases."""
    cleaned = label.strip()
    if value:
        val_clean = re.sub(r"[^\w\.\%\$]", "", value).strip()
        if val_clean:
            cleaned = re.sub(re.escape(val_clean), "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[\$\€\£\¥]?\d+(?:\.\d+)?%?\s*[:\-\–]?\s*", "", cleaned)
    cleaned = re.sub(r"\s*[:\-\–]?\s*[\$\€\£\¥]?\d+(?:\.\d+)?%?\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(". ,;:-– ")
    return cleaned if cleaned else label.strip()


def draw_stat_callout(
    stat_value: str,
    stat_label: str,
    stat_delta: str | None,
    source_credit: str,
    output_path: Path,
    width: int = 1400,
    height: int = 520,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws a high-impact broadcast statistic card (Shotcraft stat_callout recipe)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded panel background (Deep Navy with 94% opacity)
    bg_color = (11, 31, 58, 240)
    border_color = (245, 166, 35, 220)  # Amber border
    draw_rounded_panel(draw, (0, 0, width, height), radius=24, fill=bg_color, border_color=border_color, border_width=3)

    # Accent top strip
    draw.rounded_rectangle((24, 16, width - 24, 24), radius=4, fill=(245, 166, 35, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    regular_fonts = get_font_candidates(regular=True)
    font_val = get_font(fonts, 80 if width >= 1100 else 64)
    font_delta = get_font(fonts, 32 if width >= 1100 else 26)
    font_label = get_font(fonts, 38 if width >= 1100 else 30)
    font_credit = get_font(regular_fonts, 24 if width >= 1100 else 20)

    val_bbox = font_val.getbbox(stat_value)
    val_w = val_bbox[2] - val_bbox[0]

    # Check if delta badge fits on the right or needs to be stacked below
    is_stacked = (width < 1100) or (64 + val_w + 32 + 240 > width - 64)
    if not is_stacked:
        draw.text((64, 60), stat_value, font=font_val, fill=(255, 255, 255, 255))
        badge_x = min(width - 340, 64 + val_w + 32)
        if stat_delta:
            delta_bg = (40, 167, 69, 220) if ("+" in stat_delta or "▲" in stat_delta) else (245, 166, 35, 220)
            draw.rounded_rectangle((badge_x, 90, badge_x + 240, 142), radius=12, fill=delta_bg)
            draw.text((badge_x + 18, 98), stat_delta, font=font_delta, fill=(255, 255, 255, 255))
        y_div = 175
    else:
        draw.text((64, 50), stat_value, font=font_val, fill=(255, 255, 255, 255))
        if stat_delta:
            delta_bg = (40, 167, 69, 220) if ("+" in stat_delta or "▲" in stat_delta) else (245, 166, 35, 220)
            badge_w = min(240, width - 128)
            draw.rounded_rectangle((64, 130, 64 + badge_w, 175), radius=10, fill=delta_bg)
            draw.text((78, 137), stat_delta, font=font_delta, fill=(255, 255, 255, 255))
            y_div = 195
        else:
            y_div = 150

    # Divider line
    draw.line((64, y_div, width - 64, y_div), fill=(255, 255, 255, 50), width=2)

    # Clean label text (up to 2 lines, vertically balanced)
    cleaned_label = clean_stat_label(stat_label, stat_value)
    label_lines = wrap_text(cleaned_label.upper(), font_label, max_width=width - 128, max_lines=2)
    y_text = y_div + 20
    for line in label_lines:
        draw.text((64, y_text), line, font=font_label, fill=(245, 166, 35, 255))
        y_text += 42

    # Source credit at bottom
    credit_str = source_credit if source_credit.lower().startswith("source:") else f"Source: {source_credit}"
    credit_wrapped = wrap_text(credit_str, font_credit, max_width=width - 128, max_lines=1)
    draw.text((64, height - 50), credit_wrapped[0] if credit_wrapped else credit_str, font=font_credit, fill=(180, 200, 220, 240))

    img.save(str(output_path), "PNG")


def draw_headline_split(
    kicker: str,
    headline: str,
    bullets: list[str],
    source_credit: str,
    output_path: Path,
    width: int = 760,
    height: int = 760,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws a vertical split-screen briefing panel (Shotcraft headline_split recipe)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Deep Navy background
    bg_color = (11, 31, 58, 245)
    border_color = (245, 166, 35, 200)
    draw_rounded_panel(draw, (0, 0, width, height), radius=24, fill=bg_color, border_color=border_color, border_width=3)

    # Left accent indicator bar
    draw.rounded_rectangle((16, 24, 24, height - 24), radius=4, fill=(245, 166, 35, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    regular_fonts = get_font_candidates(regular=True)
    font_kicker = get_font(fonts, 28)
    font_hl = get_font(fonts, 38)
    font_bullet = get_font(regular_fonts, 26)
    font_credit = get_font(regular_fonts, 22)

    # Kicker tag
    draw.text((56, 40), kicker.upper(), font=font_kicker, fill=(245, 166, 35, 255))

    # Headline (up to 3 lines)
    hl_lines = wrap_text(headline, font_hl, max_width=width - 100, max_lines=3)
    y_pos = 90
    for line in hl_lines:
        draw.text((56, y_pos), line, font=font_hl, fill=(255, 255, 255, 255))
        y_pos += 48

    # Separator
    y_pos += 16
    draw.line((56, y_pos, width - 56, y_pos), fill=(255, 255, 255, 50), width=2)
    y_pos += 30

    # Bullets
    for b in bullets[:3]:
        bullet_lines = wrap_text(b, font_bullet, max_width=width - 120, max_lines=2)
        # Bullet diamond
        draw.polygon([(60, y_pos + 12), (68, y_pos + 6), (76, y_pos + 12), (68, y_pos + 18)], fill=(245, 166, 35, 255))
        for bl in bullet_lines:
            draw.text((92, y_pos), bl, font=font_bullet, fill=(225, 235, 245, 255))
            y_pos += 36
        y_pos += 14

    # Bottom source credit
    credit_str = source_credit if source_credit.lower().startswith("source:") else f"Source: {source_credit}"
    draw.text((56, height - 52), credit_str, font=font_credit, fill=(180, 200, 220, 240))

    img.save(str(output_path), "PNG")


def draw_quote_card(
    quote_text: str,
    speaker_name: str,
    speaker_title: str | None,
    source_credit: str,
    output_path: Path,
    width: int = 1400,
    height: int = 520,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws an authoritative quote card (Shotcraft quote_focus recipe)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Deep Navy background
    bg_color = (11, 31, 58, 240)
    border_color = (70, 130, 180, 220)  # Steel blue border
    draw_rounded_panel(draw, (0, 0, width, height), radius=24, fill=bg_color, border_color=border_color, border_width=3)

    # Top accent bar
    draw.rounded_rectangle((24, 16, width - 24, 24), radius=4, fill=(70, 130, 180, 255))

    fonts = font_candidates or get_font_candidates(regular=False)
    regular_fonts = get_font_candidates(regular=True)
    font_glyph = get_font(fonts, 100)
    font_quote = get_font(regular_fonts, 36)
    font_speaker = get_font(fonts, 34)
    font_title = get_font(regular_fonts, 26)
    font_credit = get_font(regular_fonts, 22)

    # Large quotation mark glyph
    draw.text((54, 30), "\u201C", font=font_glyph, fill=(245, 166, 35, 160))

    # Quote body (up to 3 lines)
    clean_quote = quote_text.strip().strip('"').strip("'")
    q_lines = wrap_text(f'"{clean_quote}"', font_quote, max_width=width - 180, max_lines=3)
    y_pos = 110
    for line in q_lines:
        draw.text((120, y_pos), line, font=font_quote, fill=(255, 255, 255, 255))
        y_pos += 48

    # Attribution divider
    y_pos = max(y_pos + 16, 280)
    draw.line((120, y_pos, width - 120, y_pos), fill=(255, 255, 255, 50), width=2)
    y_pos += 24

    # Speaker Name and Title
    draw.text((120, y_pos), f"\u2014 {speaker_name}", font=font_speaker, fill=(245, 166, 35, 255))
    if speaker_title:
        draw.text((120, y_pos + 46), speaker_title, font=font_title, fill=(180, 205, 230, 240))

    # Bottom source credit
    credit_str = source_credit if source_credit.lower().startswith("source:") else f"Source: {source_credit}"
    draw.text((120, height - 52), credit_str, font=font_credit, fill=(160, 185, 210, 200))

    img.save(str(output_path), "PNG")


def draw_news_ticker(
    ticker_text: str,
    output_path: Path,
    width: int = 1728,
    height: int = 44,
    font_candidates: list[str] | None = None,
) -> None:
    """Draws a breaking news / bulletin bottom ticker strap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Navy ticker background
    bg_color = (11, 31, 58, 245)
    draw.rectangle((0, 0, width, height), fill=bg_color)

    # Amber tag on left
    draw.rectangle((0, 0, 180, height), fill=(245, 166, 35, 255))
    fonts = font_candidates or get_font_candidates(regular=False)
    regular_fonts = get_font_candidates(regular=True)
    font_tag = get_font(fonts, 22)
    font_txt = get_font(regular_fonts, 24)

    draw.text((20, 8), "LATEST UPDATE", font=font_tag, fill=(11, 31, 58, 255))
    draw.text((204, 8), ticker_text, font=font_txt, fill=(255, 255, 255, 255))

    img.save(str(output_path), "PNG")


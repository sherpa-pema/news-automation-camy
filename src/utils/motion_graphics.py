"""Motion graphics generation utilities: Pillow overlay generation and ASS subtitle creation."""
from datetime import datetime
from pathlib import Path
import re
from typing import Any
from PIL import Image, ImageDraw, ImageFont


def get_font(font_path_candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load the first available font candidate, falling back to default."""
    for path_str in font_path_candidates:
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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    font_channel = get_font(fonts, 50)
    font_title = get_font(fonts, 40)
    font_date = get_font(fonts, 32)

    # Format date into plain words (e.g. Sunday, November 8, 2026)
    try:
        dt = datetime.fromisoformat(date_str)
        date_display = dt.strftime("%A, %B ") + str(dt.day) + dt.strftime(", %Y")
    except Exception:
        date_display = date_str

    # Texts
    # 1. Channel name (Amber accent)
    draw.text((60, 60), channel_name.upper(), font=font_channel, fill=(245, 166, 35, 255))

    # 2. Headline Tagline
    draw.text((60, 155), "TODAY'S TOP STORIES", font=font_title, fill=(255, 255, 255, 255))

    # 3. Separator line
    draw.line((60, 235, width - 60, 235), fill=(255, 255, 255, 60), width=2)

    # 4. Date in plain words
    draw.text((60, 275), date_display, font=font_date, fill=(220, 230, 242, 255))

    # 5. Briefing note
    font_note = get_font(fonts, 24)
    draw.text((60, 360), "Independent daily coverage powered by verified news sources.", font=font_note, fill=(180, 195, 215, 220))

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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
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
    draw.text((36, height - 38), credit_line, font=font_source, fill=(245, 166, 35, 255))

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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    font_channel = get_font(fonts, 50)
    font_title = get_font(fonts, 40)
    font_desc = get_font(fonts, 24)

    # Channel name
    draw.text((60, 60), channel_name.upper(), font=font_channel, fill=(245, 166, 35, 255))

    # Thanks for watching
    draw.text((60, 155), "THANKS FOR WATCHING", font=font_title, fill=(255, 255, 255, 255))

    draw.line((60, 235, width - 60, 235), fill=(255, 255, 255, 60), width=2)

    # Disclosure lines
    if disclose_synthetic_voice:
        line1 = "Narration: AI-generated voice. Footage: stock and archive, illustrative."
        line2 = "See description for full journalist credits and source article links."
        draw.text((60, 275), line1, font=font_desc, fill=(220, 230, 242, 255))
        draw.text((60, 320), line2, font=font_desc, fill=(220, 230, 242, 220))
    else:
        line = "See description for full journalist credits and source article links."
        draw.text((60, 280), line, font=font_desc, fill=(220, 230, 242, 255))

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
) -> None:
    """Generates an Advanced SubStation Alpha (.ass) subtitle file positioned in the caption band."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rx, ry = resolution

    back_colour = "&H80000000"
    if box_rgba and len(box_rgba) >= 4:
        r, g, b, a = box_rgba[:4]
        ass_a = max(0, min(255, 255 - int(a)))
        back_colour = f"&H{ass_a:02X}{int(b):02X}{int(g):02X}{int(r):02X}"

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
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},-1,0,0,0,100,100,0,0,1,3.5,1.0,2,40,40,{margin_v},1

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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    regular_fonts = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
    font_val = get_font(fonts, 80)
    font_delta = get_font(fonts, 32)
    font_label = get_font(fonts, 38)
    font_credit = get_font(regular_fonts, 24)

    # Giant stat value
    draw.text((64, 60), stat_value, font=font_val, fill=(255, 255, 255, 255))

    # Delta badge (e.g. +14.2% or RECORD HIGH) if present
    curr_x = 64
    val_bbox = font_val.getbbox(stat_value)
    val_w = val_bbox[2] - val_bbox[0]
    badge_x = min(width - 340, 64 + val_w + 32)
    if stat_delta:
        # Green tint for positive, amber for neutral
        delta_bg = (40, 167, 69, 220) if ("+" in stat_delta or "▲" in stat_delta) else (245, 166, 35, 220)
        draw.rounded_rectangle((badge_x, 90, badge_x + 240, 142), radius=12, fill=delta_bg)
        draw.text((badge_x + 18, 98), stat_delta, font=font_delta, fill=(255, 255, 255, 255))

    # Divider line
    draw.line((64, 175, width - 64, 175), fill=(255, 255, 255, 50), width=2)

    # Clean label text (up to 2 lines, vertically balanced)
    cleaned_label = clean_stat_label(stat_label, stat_value)
    label_lines = wrap_text(cleaned_label.upper(), font_label, max_width=width - 128, max_lines=2)
    y_text = 230 if len(label_lines) == 1 else 205
    for line in label_lines:
        draw.text((64, y_text), line, font=font_label, fill=(245, 166, 35, 255))
        y_text += 50

    # Source credit at bottom
    credit_str = source_credit if source_credit.lower().startswith("source:") else f"Source: {source_credit}"
    draw.text((64, height - 58), credit_str, font=font_credit, fill=(180, 200, 220, 240))

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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    regular_fonts = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
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

    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    regular_fonts = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
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
    fonts = font_candidates or ["C:\\Windows\\Fonts\\segoeuib.ttf", "C:\\Windows\\Fonts\\arialbd.ttf"]
    regular_fonts = ["C:\\Windows\\Fonts\\segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]
    font_tag = get_font(fonts, 22)
    font_txt = get_font(regular_fonts, 24)

    draw.text((20, 8), "LATEST UPDATE", font=font_tag, fill=(11, 31, 58, 255))
    draw.text((204, 8), ticker_text, font=font_txt, fill=(255, 255, 255, 255))

    img.save(str(output_path), "PNG")


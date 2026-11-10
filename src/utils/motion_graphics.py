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
    """Wraps text into up to max_lines lines respecting max_width in pixels."""
    words = text.strip().split()
    if not words:
        return []

    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        # Measure text width
        bbox = font.getbbox(test_line)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                if len(lines) == max_lines - 1:
                    break
            else:
                lines.append(word)
                if len(lines) == max_lines - 1:
                    break

    # Add remaining words to the final line
    idx = len(" ".join(lines).split()) if lines else 0
    remaining_words = words[idx:]
    if remaining_words:
        final_line = " ".join(remaining_words)
        # If final line is still too long, truncate with ellipsis
        while font.getbbox(final_line)[2] - font.getbbox(final_line)[0] > max_width and len(final_line) > 4:
            final_line = final_line[:-4].rstrip() + "..."
        lines.append(final_line)

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
) -> None:
    """Generates an Advanced SubStation Alpha (.ass) subtitle file positioned in the caption band."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rx, ry = resolution

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
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.0,2,40,40,{margin_v},1

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

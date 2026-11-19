"""Automated frame-diff tests for overlay animation (alpha fade in/out and positioning) in FFmpeg."""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from PIL import Image
import pytest

from src.agents.render import RenderAgent


def test_overlay_alpha_fade_animation(tmp_path: Path) -> None:
    """Verifies that an overlay with fade_in/fade_out animates with partial alpha during mid-fade."""
    # 1. Base black image (640x360)
    bg_path = tmp_path / "bg.jpg"
    Image.new("RGB", (640, 360), (0, 0, 0)).save(str(bg_path))

    # 2. Bright red overlay (200x100)
    ov_path = tmp_path / "red.png"
    Image.new("RGBA", (200, 100), (255, 0, 0, 255)).save(str(ov_path))

    # Scene duration 4.0s; overlay active from 1.0s to 3.0s, fade_in=0.6s, fade_out=0.6s
    sc = {
        "id": "scene_anim",
        "start": 0.0,
        "end": 4.0,
        "duration": 4.0,
        "shots": [{"file": "bg.jpg", "duration": 4.0, "motion": None}],
        "overlays": [
            {
                "id": "ov_red",
                "png": "red.png",
                "start": 1.0,
                "end": 3.0,
                "x": 100,
                "y": 100,
                "fade_in": 0.6,
                "fade_out": 0.6,
                "animation": "fade",
            }
        ],
    }

    agent = RenderAgent(run_dir=tmp_path)
    out_mp4 = tmp_path / "rendered_scene.mp4"
    agent._render_scene_segment(
        sc=sc,
        fps=30,
        target_res=(640, 360),
        output_path=out_mp4,
        crf=18,
        preset="ultrafast",
    )

    # Clean up file handlers on Windows
    for h in logging.getLogger("render_agent").handlers[:]:
        h.close()
        logging.getLogger("render_agent").removeHandler(h)

    assert out_mp4.exists(), "Rendered MP4 segment missing"

    def get_pixel_at(t_sec: float, x: int = 150, y: int = 150) -> tuple[int, int, int]:
        frame_png = tmp_path / f"frame_{t_sec:.2f}.png"
        cmd = ["ffmpeg", "-y", "-ss", f"{t_sec:.2f}", "-i", str(out_mp4), "-vframes", "1", str(frame_png)]
        subprocess.run(cmd, capture_output=True, check=True)
        im = Image.open(frame_png).convert("RGB")
        return im.getpixel((x, y))

    # 1. Before overlay starts (t = 0.5s): overlay not visible (pure black background)
    p_before = get_pixel_at(0.5)
    assert p_before[0] < 15, f"Expected near-zero red before overlay starts, got {p_before}"

    # 2. Mid-fade (t = 1.2s): ~33% alpha (red should be partially transparent: 40 < R < 160)
    p_mid = get_pixel_at(1.2)
    assert 40 <= p_mid[0] <= 160, f"Expected partial alpha at mid-fade, got {p_mid}"

    # 3. Fully in (t = 2.0s): 100% alpha (red should be fully bright: R > 220)
    p_full = get_pixel_at(2.0)
    assert p_full[0] >= 220, f"Expected full opacity at t=2.0s, got {p_full}"

    # 4. After overlay ends (t = 3.5s): overlay gone (pure black background)
    p_after = get_pixel_at(3.5)
    assert p_after[0] < 15, f"Expected near-zero red after overlay ends, got {p_after}"


def test_overlay_slide_up_animation(tmp_path: Path) -> None:
    """Verifies that an overlay with animation='slide_up' renders and interpolates without error."""
    bg_path = tmp_path / "bg.jpg"
    Image.new("RGB", (640, 360), (0, 0, 0)).save(str(bg_path))

    ov_path = tmp_path / "blue.png"
    Image.new("RGBA", (150, 60), (0, 0, 255, 255)).save(str(ov_path))

    sc = {
        "id": "scene_slide",
        "start": 0.0,
        "end": 3.0,
        "duration": 3.0,
        "shots": [{"file": "bg.jpg", "duration": 3.0, "motion": None}],
        "overlays": [
            {
                "id": "ov_slide",
                "png": "blue.png",
                "start": 0.5,
                "end": 2.5,
                "x": 100,
                "y": 200,
                "fade_in": 0.4,
                "fade_out": 0.4,
                "animation": "slide_up",
            }
        ],
    }

    agent = RenderAgent(run_dir=tmp_path)
    out_mp4 = tmp_path / "rendered_slide.mp4"
    agent._render_scene_segment(
        sc=sc,
        fps=30,
        target_res=(640, 360),
        output_path=out_mp4,
        crf=18,
        preset="ultrafast",
    )

    for h in logging.getLogger("render_agent").handlers[:]:
        h.close()
        logging.getLogger("render_agent").removeHandler(h)

    assert out_mp4.exists(), "Slide-up MP4 segment missing"

"""Tests for Step 7.4: publishing metadata, auto-chapters, shorts caption, and thumbnail candidates."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from src.agents.render import RenderAgent
from src.agents.qa import QAAgent


def test_youtube_chapters_and_used_media_filtering(tmp_path: Path) -> None:
    agent = RenderAgent(run_dir=tmp_path)
    output_desc = tmp_path / "description.md"

    script_data = {
        "date": "2026-11-21",
        "run_kind": "test",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Briefing Intro"},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Kathmandu Ring Road Expansion Approved",
                "sources": ["Kantipur Daily"],
                "source_urls": ["https://kantipur.com/story1"],
            },
            {
                "id": "scene_02",
                "type": "story",
                "headline": "Pruned Story Should Not Appear",
                "sources": ["Unused Source"],
                "source_urls": ["https://unused.com/pruned"],
            },
            {"id": "outro", "type": "outro", "headline": "Summary & Signoff"},
        ],
    }

    # Only intro, scene_01, outro in timeline (scene_02 was pruned)
    timeline_data = {
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 10.0},
            {"id": "scene_01", "type": "story", "start": 10.0, "end": 35.0},
            {"id": "outro", "type": "outro", "start": 35.0, "end": 45.0},
        ]
    }

    media_data = {
        "shots": [
            {
                "id": "used_shot_1",
                "file": "media/shots/ring_road.jpg",
                "author": "Pramod Shrestha",
                "source": "Wikimedia Commons",
                "attribution_required": True,
                "license": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
                "origin_url": "https://commons.wikimedia.org/wiki/File:RingRoad.jpg",
                "attribution_text": "File:RingRoad.jpg by Pramod Shrestha",
            },
            {
                "id": "unused_shot_2",
                "file": "media/shots/unused_mountain.jpg",
                "author": "Random Photographer",
                "source": "Unsplash",
                "attribution_required": True,
                "license": "Unsplash License",
                "license_url": "",
                "origin_url": "https://unsplash.com/unused",
                "attribution_text": "Unused Mountain Photo",
            },
        ]
    }

    render_plan = {
        "scenes": [
            {"id": "intro", "media": "media/shots/card_00.png"},
            {
                "id": "scene_01",
                "shots": [{"file": "media/shots/ring_road.jpg"}],
            },
            {"id": "outro", "media": "media/shots/outro_card.png"},
        ]
    }

    agent._generate_description_markdown(
        script_data=script_data,
        media_data=media_data,
        timeline_data=timeline_data,
        output_path=output_desc,
        render_plan=render_plan,
    )

    assert output_desc.exists()
    content = output_desc.read_text(encoding="utf-8")

    # Chapters
    assert "00:00 - Introduction" in content
    assert "00:10 - Kathmandu Ring Road Expansion Approved" in content
    assert "00:35 - Summary & Credits" in content

    # Sources: included scene_01, excluded pruned scene_02
    assert "Kathmandu Ring Road Expansion Approved" in content
    assert "Kantipur Daily" in content
    assert "Pruned Story Should Not Appear" not in content
    assert "Unused Source" not in content

    # Media credits: only used_shot_1 appears, unused_shot_2 is filtered out
    assert "Pramod Shrestha" in content
    assert "Wikimedia Commons" in content
    assert "Random Photographer" not in content
    assert "Unused Mountain Photo" not in content


def test_vertical_shorts_caption_generation(tmp_path: Path) -> None:
    agent = RenderAgent(run_dir=tmp_path)
    agent.orientation = "vertical"
    output_desc = tmp_path / "description.md"

    script_data = {
        "date": "2026-11-21",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Morning News"},
            {"id": "scene_01", "type": "story", "headline": "Electric Bus Fleet Expands Across Valley"},
        ],
    }
    timeline_data = {
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 8.0},
            {"id": "scene_01", "type": "story", "start": 8.0, "end": 30.0},
        ]
    }
    media_data = {"shots": []}
    render_plan = {
        "aspect_ratio": "9:16",
        "settings": {"resolution": [1080, 1920]},
        "scenes": [],
    }

    agent._generate_description_markdown(
        script_data=script_data,
        media_data=media_data,
        timeline_data=timeline_data,
        output_path=output_desc,
        render_plan=render_plan,
    )

    shorts_path = tmp_path / "shorts_caption.txt"
    assert shorts_path.exists()
    caption_text = shorts_path.read_text(encoding="utf-8")

    # Verify hook line
    assert "Electric Bus Fleet Expands Across Valley" in caption_text
    # Verify bilingual hashtags
    assert "#NepalNews" in caption_text
    assert "#नेपालसमाचार" in caption_text
    assert "#काठमाडौं" in caption_text


def test_qa_thumbnail_candidates_generation(tmp_path: Path) -> None:
    # Generate a lightweight dummy mp4 (3 seconds)
    import subprocess
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir(parents=True)
    video_path = tmp_path / "final.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    agent = QAAgent(run_dir=tmp_path)
    tl_scenes = [
        {"id": "intro", "type": "intro", "start": 0.0, "end": 1.5},
        {"id": "scene_01", "type": "story", "start": 1.5, "end": 3.0},
        {"id": "scene_02", "type": "story", "start": 3.0, "end": 4.0},
    ]

    thumbs = agent._generate_thumbnail_candidates(video_path, tl_scenes, qa_dir)
    assert len(thumbs) == 3
    assert (qa_dir / "thumb_01.jpg").exists()
    assert (qa_dir / "thumb_02.jpg").exists()
    assert (qa_dir / "thumb_03.jpg").exists()

    with Image.open(qa_dir / "thumb_01.jpg") as im:
        assert im.size[0] == 320 and im.size[1] == 240

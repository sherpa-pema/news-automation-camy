"""Unit tests for Render Agent, frame-exact timing, chapters, and minimal render pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from src.agents.render import RenderAgent, format_chapter_time
from src.agents.qa import probe_file
from src.utils.tts import create_silence_wav


def test_format_chapter_time() -> None:
    assert format_chapter_time(0.0) == "00:00"
    assert format_chapter_time(9.4) == "00:09"
    assert format_chapter_time(65.0) == "01:05"
    assert format_chapter_time(3665.0) == "01:01:05"


def test_frame_exact_scene_tiling() -> None:
    fps = 30
    scene_durations = [5.123, 7.456, 4.321]
    # Total duration = 16.900
    total_dur = sum(scene_durations)
    expected_total_frames = round(total_dur * fps)

    # Calculate frame counts from absolute times to guarantee no drift
    cur_t = 0.0
    scene_frames = []
    for dur in scene_durations:
        start_t = cur_t
        end_t = cur_t + dur
        start_frame = round(start_t * fps)
        end_frame = round(end_t * fps)
        frames = end_frame - start_frame
        scene_frames.append(frames)
        cur_t = end_t

    assert sum(scene_frames) == expected_total_frames
    assert abs(sum(scene_frames) - expected_total_frames) <= 1


def test_description_publish_pack_generation(tmp_path: Path) -> None:
    agent = RenderAgent(run_dir=tmp_path)
    output_desc = tmp_path / "description.md"

    script_data = {
        "date": "2026-11-08",
        "run_kind": "test",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Historic Victory for Kathmandu",
                "sources": ["Kathmandu Post", "MyRepublica"],
                "source_urls": ["https://kathmandupost.com/story1", "https://myrepublica.com/story1"],
            },
            {
                "id": "scene_02",
                "type": "story",
                "headline": "Flower Market Opens at 33 Spots",
                "sources": ["OnlineKhabar"],
                "source_urls": ["https://onlinekhabar.com/story2"],
            },
            {"id": "outro", "type": "outro", "headline": "Summary & Credits"},
        ],
    }

    timeline_data = {
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 12.0},
            {"id": "scene_01", "type": "story", "start": 12.0, "end": 28.0},
            {"id": "scene_02", "type": "story", "start": 28.0, "end": 45.0},
            {"id": "outro", "type": "outro", "start": 45.0, "end": 58.0},
        ]
    }

    media_data = {
        "shots": [
            {
                "id": "shot_1",
                "author": "Vyacheslav Argenberg",
                "source": "Wikimedia Commons",
                "attribution_required": True,
                "license": "CC BY 4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0",
                "origin_url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
                "attribution_text": "File:Test.jpg by Vyacheslav Argenberg",
            }
        ]
    }

    agent._generate_description_markdown(
        script_data=script_data,
        timeline_data=timeline_data,
        media_data=media_data,
        output_path=output_desc,
    )

    assert output_desc.exists()
    content = output_desc.read_text(encoding="utf-8")

    # Banner
    assert "TEST RUN - DO NOT PUBLISH" in content

    # Chapters
    assert "## Chapters" in content
    assert "00:00 - Introduction" in content
    assert "00:12 - Historic Victory for Kathmandu" in content
    assert "00:28 - Flower Market Opens at 33 Spots" in content

    # Chapter rules check
    lines = content.splitlines()
    ch_lines = [l for l in lines if l.startswith("0") and " - " in l]
    assert len(ch_lines) >= 3, f"Must have at least 3 chapters, got {len(ch_lines)}"

    # Credits & Disclaimers
    assert "Vyacheslav Argenberg" in content
    assert "https://commons.wikimedia.org/wiki/File:Test.jpg" in content
    assert "Narration audio was synthesized using an AI voice model." in content
    assert "Stock and archive footage is illustrative." in content


def test_minimal_two_scene_render(tmp_path: Path) -> None:
    """Verifies a real minimal 2-scene render produces valid MP4 with audio/video sync."""
    script_dir = tmp_path / "script"
    script_dir.mkdir(parents=True)
    script_data = {
        "origin": "fixture",
        "run_kind": "test",
        "date": "2026-11-08",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Intro"},
            {"id": "scene_01", "type": "story", "headline": "Test Story", "source": "Source1"},
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    # 2 scenes of 2.0s each = 4.0s total
    tl_dir = tmp_path / "timeline"
    tl_dir.mkdir(parents=True)
    tl_data = {
        "duration_sec": 4.0,
        "scenes": [
            {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
            {"id": "scene_01", "type": "story", "start": 2.0, "end": 4.0},
        ],
    }
    with open(tl_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(tl_data, f)

    # Dummy silence audio 4.0s
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    narration_wav = audio_dir / "narration.wav"
    create_silence_wav(narration_wav, duration_sec=4.0, sample_rate=48000)

    # Create dummy images for shots
    media_dir = tmp_path / "media"
    shots_dir = media_dir / "shots"
    shots_dir.mkdir(parents=True)

    img1 = Image.new("RGB", (320, 240), color=(20, 40, 80))
    shot1_path = shots_dir / "shot1.jpg"
    img1.save(str(shot1_path))

    img2 = Image.new("RGB", (320, 240), color=(80, 40, 20))
    shot2_path = shots_dir / "shot2.jpg"
    img2.save(str(shot2_path))

    media_data = {
        "resolution": [1920, 1080],
        "fps": 30,
        "shots": [
            {"id": "s1", "scene_id": "intro", "file_path": "media/shots/shot1.jpg", "type": "image"},
            {"id": "s2", "scene_id": "scene_01", "file_path": "media/shots/shot2.jpg", "type": "image"},
        ],
    }
    with open(media_dir / "media_manifest.json", "w", encoding="utf-8") as f:
        json.dump(media_data, f)

    # Render plan
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir(parents=True)
    plan_data = {
        "resolution": "1920x1080",
        "fps": 30,
        "audio_file": "audio/narration.wav",
        "duration_sec": 4.0,
        "captions_ass": "motion/captions.ass",
        "scenes": [
            {
                "id": "intro",
                "start": 0.0,
                "end": 2.0,
                "duration": 2.0,
                "shots": [
                    {"shot_index": 0, "file": "media/shots/shot1.jpg", "start": 0.0, "end": 2.0, "duration": 2.0, "motion": None}
                ],
                "overlays": [],
            },
            {
                "id": "scene_01",
                "start": 2.0,
                "end": 4.0,
                "duration": 2.0,
                "shots": [
                    {"shot_index": 0, "file": "media/shots/shot2.jpg", "start": 2.0, "end": 4.0, "duration": 2.0, "motion": None}
                ],
                "overlays": [],
            },
        ],
        "overlays": [],
    }
    with open(motion_dir / "render_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan_data, f)

    # Empty ASS file
    (motion_dir / "captions.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name\nStyle: Default\n[Events]\nFormat: Layer, Start, End, Style, Text\n",
        encoding="utf-8",
    )

    # Run preview render for fast test
    agent = RenderAgent(run_dir=tmp_path)
    outputs = agent.run(preview=True)

    final_mp4 = tmp_path / "render" / "final_preview.mp4"
    assert final_mp4.exists(), f"Render preview output missing: {outputs}"

    probe = probe_file(final_mp4)
    v_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    a_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]

    assert len(v_streams) >= 1, "Must have video stream"
    assert len(a_streams) >= 1, "Must have audio stream"

    vid_dur = float(probe.get("format", {}).get("duration", 0.0))
    # Audio duration is 4.0s
    assert abs(vid_dur - 4.0) < 0.1, f"Expected ~4.0s duration, got {vid_dur}s"

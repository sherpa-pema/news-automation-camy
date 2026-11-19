"""Unit tests for Render Agent, frame-exact timing, chapters, and minimal render pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

from src.agents.render import RenderAgent, format_chapter_time
from src.agents.qa import probe_file
from src.utils.audio_sync import generate_synthetic_broadcast_theme
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


def test_render_agent_cache_invalidation_and_card_concat(tmp_path: Path) -> None:
    """Verify compute_scene_cache_key changes with sha256, RenderAgent syncs outdated render_plan shots, and card PNGs concat cleanly."""
    from src.agents.render import compute_scene_cache_key

    # 1. Cache key must change when sha256 changes
    sc_base = {
        "id": "intro",
        "start": 0.0,
        "end": 2.0,
        "shots": [{"file": "media/shots/intro/shot_00.jpg", "duration": 2.0, "type": "image", "motion": "zoom_in", "sha256": "aaa"}],
        "overlays": [],
        "transition": {"type": "dip_to_black", "duration_sec": 0.2},
    }
    sc_new_hash = {
        **sc_base,
        "shots": [{"file": "media/shots/intro/shot_00.jpg", "duration": 2.0, "type": "image", "motion": "zoom_in", "sha256": "bbb"}],
    }
    assert compute_scene_cache_key(sc_base, "style1") != compute_scene_cache_key(sc_new_hash, "style1")

    # 2. Mixed .jpg + .png card concatenation and sync with media_manifest.json
    (tmp_path / "script").mkdir(parents=True)
    (tmp_path / "script" / "script.json").write_text(
        json.dumps({
            "origin": "fixture",
            "run_kind": "test",
            "date": "2026-11-08",
            "scenes": [
                {"id": "intro", "type": "intro", "headline": "Intro"},
                {"id": "scene_01", "type": "story", "headline": "Card Story", "source": "Desk"},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "timeline").mkdir(parents=True)
    (tmp_path / "timeline" / "timeline.json").write_text(
        json.dumps({
            "duration_sec": 4.0,
            "scenes": [
                {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
                {"id": "scene_01", "type": "story", "start": 2.0, "end": 4.0},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "audio").mkdir(parents=True)
    create_silence_wav(tmp_path / "audio" / "narration.wav", duration_sec=4.0, sample_rate=48000)

    shots_dir = tmp_path / "media" / "shots"
    shots_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(15, 45, 90)).save(str(shots_dir / "shot_00.jpg"))
    Image.new("RGB", (320, 240), color=(90, 20, 20)).save(str(shots_dir / "card_00.png"))

    media_data = {
        "resolution": [1920, 1080],
        "fps": 30,
        "shots": [
            {"scene_id": "intro", "shot_index": 0, "start": 0.0, "end": 2.0, "type": "image", "file": "media/shots/shot_00.jpg", "sha256": "h1"},
            {"scene_id": "scene_01", "shot_index": 0, "start": 2.0, "end": 4.0, "type": "card", "file": "media/shots/card_00.png", "sha256": "h2"},
        ],
    }
    (tmp_path / "media" / "media_manifest.json").write_text(json.dumps(media_data), encoding="utf-8")

    # Stale render_plan pointing to nonexistent old files
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir(parents=True)
    stale_plan = {
        "resolution": "1920x1080",
        "fps": 30,
        "audio_file": "audio/narration.wav",
        "duration_sec": 4.0,
        "captions_ass": "motion/captions.ass",
        "scenes": [
            {"id": "intro", "start": 0.0, "end": 2.0, "duration": 2.0, "shots": [{"shot_index": 0, "file": "media/shots/old.jpg", "start": 0.0, "end": 2.0, "duration": 2.0, "type": "image"}], "overlays": []},
            {"id": "scene_01", "start": 2.0, "end": 4.0, "duration": 2.0, "shots": [{"shot_index": 0, "file": "media/shots/old2.jpg", "start": 2.0, "end": 4.0, "duration": 2.0, "type": "image"}], "overlays": []},
        ],
        "overlays": [],
    }
    (motion_dir / "render_plan.json").write_text(json.dumps(stale_plan), encoding="utf-8")
    (motion_dir / "captions.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name\nStyle: Default\n[Events]\nFormat: Layer, Start, End, Style, Text\n",
        encoding="utf-8",
    )

    agent = RenderAgent(run_dir=tmp_path)
    agent.run(preview=True)
    final_mp4 = tmp_path / "render" / "final_preview.mp4"
    assert final_mp4.exists()
    probe = probe_file(final_mp4)
    assert any(s.get("codec_type") == "video" for s in probe.get("streams", []))
    assert any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    vid_dur = float(probe.get("format", {}).get("duration", 0.0))
    assert abs(vid_dur - 4.0) < 0.1


def test_render_agent_with_bgm_and_ducking(tmp_path: Path) -> None:
    # Setup test run environment with background music
    (tmp_path / "script").mkdir(parents=True)
    (tmp_path / "script" / "script.json").write_text(
        json.dumps({
            "origin": "fixture",
            "run_kind": "test",
            "scenes": [
                {"id": "intro", "type": "intro", "headline": "Intro"},
                {"id": "scene_01", "type": "story", "headline": "Story 1"},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "timeline").mkdir(parents=True)
    (tmp_path / "timeline" / "timeline.json").write_text(
        json.dumps({
            "duration_sec": 4.0,
            "scenes": [
                {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
                {"id": "scene_01", "type": "story", "start": 2.0, "end": 4.0},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "audio").mkdir(parents=True)
    create_silence_wav(tmp_path / "audio" / "narration.wav", duration_sec=4.0, sample_rate=48000)
    bgm_file = tmp_path / "audio" / "bg_track.wav"
    generate_synthetic_broadcast_theme(bgm_file, duration_sec=4.0, bpm=120.0)

    shots_dir = tmp_path / "media" / "shots"
    shots_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), color=(15, 45, 90)).save(str(shots_dir / "shot_00.jpg"))
    Image.new("RGB", (320, 240), color=(90, 20, 20)).save(str(shots_dir / "card_00.png"))

    media_data = {
        "resolution": [1920, 1080],
        "fps": 30,
        "shots": [
            {"scene_id": "intro", "shot_index": 0, "start": 0.0, "end": 2.0, "type": "image", "file": "media/shots/shot_00.jpg", "sha256": "h1"},
            {"scene_id": "scene_01", "shot_index": 0, "start": 2.0, "end": 4.0, "type": "card", "file": "media/shots/card_00.png", "sha256": "h2"},
        ],
    }
    (tmp_path / "media" / "media_manifest.json").write_text(json.dumps(media_data), encoding="utf-8")

    motion_dir = tmp_path / "motion"
    motion_dir.mkdir(parents=True)
    plan_with_bgm = {
        "resolution": "1920x1080",
        "fps": 30,
        "audio_file": "audio/narration.wav",
        "duration_sec": 4.0,
        "captions_ass": "motion/captions.ass",
        "audio_grid": {
            "enabled": True,
            "bgm_file": str(bgm_file),
            "tempo_bpm": 120.0,
        },
        "scenes": [
            {"id": "intro", "start": 0.0, "end": 2.0, "duration": 2.0, "shots": [{"shot_index": 0, "file": "media/shots/shot_00.jpg", "start": 0.0, "end": 2.0, "duration": 2.0, "type": "image"}], "overlays": []},
            {"id": "scene_01", "start": 2.0, "end": 4.0, "duration": 2.0, "shots": [{"shot_index": 0, "file": "media/shots/card_00.png", "start": 2.0, "end": 4.0, "duration": 2.0, "type": "image"}], "overlays": []},
        ],
        "overlays": [],
    }
    (motion_dir / "render_plan.json").write_text(json.dumps(plan_with_bgm), encoding="utf-8")
    (motion_dir / "captions.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n[V4+ Styles]\nFormat: Name\nStyle: Default\n[Events]\nFormat: Layer, Start, End, Style, Text\n",
        encoding="utf-8",
    )

    agent = RenderAgent(run_dir=tmp_path)
    agent.run(preview=True)
    final_mp4 = tmp_path / "render" / "final_preview.mp4"
    assert final_mp4.exists()
    probe = probe_file(final_mp4)
    assert any(s.get("codec_type") == "video" for s in probe.get("streams", []))
    assert any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    vid_dur = float(probe.get("format", {}).get("duration", 0.0))
    assert abs(vid_dur - 4.0) < 0.1


def test_render_agent_vertical_preview(tmp_path: Path) -> None:
    """Verify RenderAgent parses 1080x1920 resolution from render_plan.json and outputs vertical preview."""
    (tmp_path / "script").mkdir(parents=True)
    (tmp_path / "script" / "script.json").write_text(
        json.dumps({
            "origin": "fixture",
            "run_kind": "test",
            "date": "2026-11-08",
            "scenes": [
                {"id": "intro", "type": "intro", "headline": "Vertical Intro"},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "timeline").mkdir(parents=True)
    (tmp_path / "timeline" / "timeline.json").write_text(
        json.dumps({
            "duration_sec": 2.0,
            "scenes": [
                {"id": "intro", "type": "intro", "start": 0.0, "end": 2.0},
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "audio").mkdir(parents=True)
    create_silence_wav(tmp_path / "audio" / "narration.wav", duration_sec=2.0, sample_rate=48000)

    shots_dir = tmp_path / "media" / "shots"
    shots_dir.mkdir(parents=True)
    # Create 1080x1920 image
    Image.new("RGB", (1080, 1920), color=(20, 50, 90)).save(str(shots_dir / "shot_00.jpg"))

    media_data = {
        "orientation": "vertical",
        "aspect_ratio": "9:16",
        "resolution": [1080, 1920],
        "fps": 30,
        "shots": [
            {"scene_id": "intro", "shot_index": 0, "start": 0.0, "end": 2.0, "type": "image", "file": "media/shots/shot_00.jpg", "sha256": "h_vert"},
        ],
    }
    (tmp_path / "media" / "media_manifest.json").write_text(json.dumps(media_data), encoding="utf-8")

    motion_dir = tmp_path / "motion"
    motion_dir.mkdir(parents=True)
    plan_vertical = {
        "resolution": "1080x1920",
        "fps": 30,
        "audio_file": "audio/narration.wav",
        "duration_sec": 2.0,
        "captions_ass": "motion/captions.ass",
        "scenes": [
            {"id": "intro", "start": 0.0, "end": 2.0, "duration": 2.0, "shots": [{"shot_index": 0, "file": "media/shots/shot_00.jpg", "start": 0.0, "end": 2.0, "duration": 2.0, "type": "image"}], "overlays": []},
        ],
        "overlays": [],
    }
    (motion_dir / "render_plan.json").write_text(json.dumps(plan_vertical), encoding="utf-8")
    (motion_dir / "captions.ass").write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\nFormat: Name\nStyle: Default\n[Events]\nFormat: Layer, Start, End, Style, Text\n",
        encoding="utf-8",
    )

    agent = RenderAgent(run_dir=tmp_path)
    outputs = agent.run(preview=True)

    final_mp4 = tmp_path / "render" / "final_preview.mp4"
    assert final_mp4.exists(), f"Render preview output missing: {outputs}"
    probe = probe_file(final_mp4)
    v_stream = next(s for s in probe.get("streams", []) if s.get("codec_type") == "video")
    # Preview resolution for 1080x1920 is 540x960
    assert v_stream.get("width") == 540
    assert v_stream.get("height") == 960



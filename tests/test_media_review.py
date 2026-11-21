"""Unit and integration tests for the Visual Review Workflow in the Media Agent."""
import json
import re
import subprocess
from pathlib import Path
from typing import Any
import pytest
from PIL import Image

from src.agents.media import MediaAgent
from src.utils.media_models import Candidate
from src.utils.media_review import (
    generate_library_report,
    generate_picker_html,
)
from src.utils.media_selector import (
    DEFAULT_FOREIGN_PLACES,
    score_candidate_relevance,
)
from src.utils.schema_validate import validate_contract


@pytest.fixture
def review_config() -> dict[str, Any]:
    return {
        "settings": {
            "orientation": "landscape",
            "fps": 30,
            "resolution": [1920, 1080],
            "target_shot_sec": 7,
            "target_shot_duration": 7.0,
            "min_shot_duration": 4.0,
            "max_shot_duration": 10.0,
            "min_relevance": 0.5,
            "max_stills_per_scene": 2,
            "allow_cc_by_sa": False,
            "illustrative_label": True,
            "user_agent": "NepalNewsBotTest/1.0",
            "cache_dir": "data/test_cache",
            "cache_ttl_hours": 24,
        },
        "foreign_places": DEFAULT_FOREIGN_PLACES,
        "sources": {
            "local_library": {"enabled": True, "path": "data/test_media_library"},
            "wikimedia_commons": {"enabled": True},
            "pixabay": {"enabled": True, "api_key_env": "TEST_PIXABAY_KEY"},
        },
        "allowed_licenses": [
            "Pixabay Content License",
            "Pixabay License",
            "CC0",
            "Public Domain",
            "CC BY",
            "CC-BY-4.0",
            "own work",
            "user-provided",
        ],
        "blocklisted_domains": [
            "kathmandupost.com",
            "onlinekhabar.com",
            "thehimalayantimes.com",
        ],
        "sensitive_keywords": ["crime", "accident", "fatal", "murder", "casualty", "death"],
        "sensitive_scene_person_tags": ["person", "people", "man", "woman", "child", "crowd"],
        "watermark_markers": ["watermark", "logo", "brand"],
    }


def test_relevance_scoring_token_match_phrase_and_place_bonus():
    """Verify relevance scoring: token overlap, phrase bonus (+0.2), place bonus (+0.2)."""
    scene = {
        "id": "scene_01",
        "headline": "Kathmandu Valley Solar Energy Expansion",
        "visual_keywords": ["Kathmandu solar panels rooftop"],
        "narration": "Kathmandu is adding solar capacity to city buildings.",
    }

    cand = Candidate(
        source="pixabay",
        id="c1",
        kind="video",
        page_url="https://pixabay.com/v/1",
        download_url="https://pixabay.com/v1.mp4",
        width=1920,
        height=1080,
        duration=8.0,
        license_id="Pixabay Content License",
        license_url="",
        author="SolarPhotog",
        title="Kathmandu solar panels rooftop view",
        tags=["kathmandu", "solar", "panels", "nepal"],
    )

    score, tokens = score_candidate_relevance(
        cand=cand,
        query="Kathmandu solar panels rooftop",
        visual_keywords=["Kathmandu solar panels rooftop"],
        scene=scene,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=False,
    )

    # All 4 content tokens match ('kathmandu', 'solar', 'panel', 'rooftop')
    assert "kathmandu" in tokens
    assert "solar" in tokens
    # Exact phrase bonus (+0.2) + place bonus for Kathmandu (+0.2) + token match (1.0) -> clamped to 1.0
    assert score == 1.0


def test_kinnaur_foreign_place_penalty():
    """Verify foreign place penalty: candidate mentioning Kinnaur/India subtracts 0.5 when scene does not."""
    scene = {
        "id": "scene_03",
        "headline": "Western Nepal Monsoon Weather Update",
        "visual_keywords": ["monsoon clouds rain mountain"],
        "narration": "Rainfall continues across the hills of Nepal.",
    }

    # Candidate from Kinnaur, India
    cand_kinnaur = Candidate(
        source="pixabay",
        id="c_kinnaur",
        kind="video",
        page_url="https://pixabay.com/v/2",
        download_url="https://pixabay.com/v2.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="TimelapseMaker",
        title="Kinnaur monsoon timelapse heavy rain Himachal India",
        tags=["kinnaur", "himachal", "india", "monsoon", "clouds", "rain"],
    )

    score, tokens = score_candidate_relevance(
        cand=cand_kinnaur,
        query="monsoon clouds rain mountain",
        visual_keywords=["monsoon clouds rain mountain"],
        scene=scene,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=False,
    )

    # Tokens match rain/monsoon/clouds, but penalty -0.5 is applied
    assert "kinnaur" in [t.lower() for t in cand_kinnaur.tags]
    assert score < 0.5  # Drops below min_relevance threshold!


def test_sensitive_scene_person_tag_rule():
    """Verify sensitive scene rejects candidates with person tags or visible people."""
    scene = {
        "id": "scene_accident",
        "headline": "Highway Crash in Dhading",
        "visual_keywords": ["highway road ambulance"],
        "narration": "Authorities responded to a fatal vehicle accident on the highway.",
    }

    cand_people = Candidate(
        source="pixabay",
        id="c_people",
        kind="image",
        page_url="https://pixabay.com/i/1",
        download_url="https://pixabay.com/i1.jpg",
        width=1920,
        height=1080,
        duration=None,
        license_id="Pixabay Content License",
        license_url="",
        author="Reporter",
        title="Highway accident scene with crowd",
        tags=["highway", "accident", "people", "crowd"],
        people_visible=True,
    )

    score, tokens = score_candidate_relevance(
        cand=cand_people,
        query="highway road ambulance",
        visual_keywords=["highway road ambulance"],
        scene=scene,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=True,
    )
    assert score == 0.0


def test_candidates_mode_below_threshold_card_proposal(tmp_path, monkeypatch, review_config):
    """Verify candidates mode proposes card when all candidates score below min_relevance."""
    run_dir = tmp_path / "runs" / "2026-11-08"
    run_dir.mkdir(parents=True)
    script_dir = run_dir / "script"
    script_dir.mkdir()

    script_data = {
        "date": "2026-11-08",
        "run_kind": "test",
        "origin": "fixture",
        "scenes": [
            {
                "id": "scene_01",
                "headline": "Rare Obscure Event Without Media",
                "visual_keywords": ["obscure xyz nonexistent 98765"],
                "narration": "An obscure event with no existing visuals took place.",
            }
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    timeline_data = {
        "duration_sec": 7.0,
        "scenes": [
            {
                "id": "scene_01",
                "start": 0.0,
                "end": 7.0,
                "sentences": [
                    {"text": "An obscure event with no existing visuals took place.", "start": 0.5, "end": 6.5}
                ],
            }
        ],
    }
    with open(run_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline_data, f)

    agent = MediaAgent(run_dir, config=review_config, allow_fixture=True)

    # Mock selector to return a candidate with low relevance (< 0.5)
    low_cand = Candidate(
        source="pixabay",
        id="c_low",
        kind="video",
        page_url="https://pixabay.com/v/low",
        download_url="https://pixabay.com/v_low.mp4",
        width=1920,
        height=1080,
        duration=7.0,
        license_id="Pixabay Content License",
        license_url="",
        author="Creator",
        title="Completely unrelated abstract shapes",
        tags=["shapes", "abstract"],
        score=0.1,
    )

    monkeypatch.setattr(
        agent.selector,
        "iter_candidates_for_shot",
        lambda *args, **kwargs: [(low_cand, "obscure xyz", "pixabay")],
    )

    candidates_data = agent.run_candidates(candidates_k=6)
    assert len(candidates_data["shots"]) == 1
    shot_entry = candidates_data["shots"][0]

    # Because candidate score is 0.1 (< 0.5), auto pick must be card!
    assert shot_entry["auto_pick_type"] == "card"
    assert shot_entry["auto_pick_id"] is None

    # Wanted clips markdown must contain this shot
    wanted_clips_path = run_dir / "media" / "review" / "wanted_clips.md"
    assert wanted_clips_path.exists()
    content = wanted_clips_path.read_text(encoding="utf-8")
    assert "Rare Obscure Event Without Media" in content


def test_picker_html_integrity_no_external_urls(tmp_path):
    """Verify picker.html lists every shot, has NO external network URLs, and inlines valid JSON."""
    candidates_data = {
        "date": "2026-11-08",
        "min_relevance": 0.5,
        "total_shots": 2,
        "shots": [
            {
                "shot_id": "intro_00",
                "scene_id": "intro",
                "shot_index": 0,
                "scene_headline": "Welcome to Nepal Daily Briefing",
                "start": 0.0,
                "end": 7.0,
                "duration": 7.0,
                "narration_sentences": ["Welcome to the Nepal Daily Briefing."],
                "visual_keywords": ["Kathmandu skyline"],
                "auto_pick_id": "pix_01",
                "auto_pick_type": "clip",
                "best_score": 0.85,
                "candidates": [
                    {
                        "id": "pix_01",
                        "source": "pixabay",
                        "title": "Kathmandu Valley Panorama",
                        "page_url": "https://pixabay.com/v/1",
                        "download_url": "https://pixabay.com/v1.mp4",
                        "kind": "video",
                        "license": "Pixabay Content License",
                        "license_url": "",
                        "author": "Videographer",
                        "attribution_text": "Video by Videographer",
                        "attribution_required": False,
                        "width": 1920,
                        "height": 1080,
                        "duration": 10.0,
                        "score": 0.85,
                        "matched_tokens": ["kathmandu", "skyline"],
                        "thumb": "thumbs/pix_01.jpg",
                    }
                ],
            },
            {
                "shot_id": "intro_01",
                "scene_id": "intro",
                "shot_index": 1,
                "scene_headline": "Welcome to Nepal Daily Briefing",
                "start": 7.0,
                "end": 14.0,
                "duration": 7.0,
                "narration_sentences": ["Here are today's top stories."],
                "visual_keywords": ["Nepal news desk"],
                "auto_pick_id": None,
                "auto_pick_type": "card",
                "best_score": 0.3,
                "candidates": [],
            },
        ],
    }

    out_html = tmp_path / "picker.html"
    generate_picker_html(candidates_data, out_html)
    assert out_html.exists()

    html_text = out_html.read_text(encoding="utf-8")

    # 1. No external stylesheets or scripts
    assert "<link rel=\"stylesheet\" href=\"http" not in html_text
    assert "<script src=\"http" not in html_text
    assert "@import url('http" not in html_text
    assert "@import url(\"http" not in html_text

    # 2. Both shots are present
    assert "intro_00" in html_text
    assert "intro_01" in html_text

    # 3. Inlined JSON parses cleanly
    match = re.search(r"const CANDIDATES_DATA = ({.*?});", html_text, re.DOTALL)
    assert match is not None
    parsed_json = json.loads(match.group(1))
    assert parsed_json["date"] == "2026-11-08"
    assert len(parsed_json["shots"]) == 2


def test_apply_picks_all_actions_and_contract(tmp_path, review_config, monkeypatch):
    """Verify apply_picks handles pick, card, own_file, and research, and passes contract validation."""
    run_dir = tmp_path / "runs" / "2026-11-08"
    run_dir.mkdir(parents=True)
    script_dir = run_dir / "script"
    script_dir.mkdir()
    media_dir = run_dir / "media"
    review_dir = media_dir / "review"
    review_dir.mkdir(parents=True)

    script_data = {
        "date": "2026-11-08",
        "run_kind": "test",
        "origin": "fixture",
        "scenes": [
            {
                "id": "scene_01",
                "headline": "First Story Headline",
                "visual_keywords": ["nepal parliament"],
                "narration": "Narration text for scene 1.",
            },
            {
                "id": "scene_02",
                "headline": "Second Story Headline",
                "visual_keywords": ["festival celebration"],
                "narration": "Narration text for scene 2.",
            },
        ],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    timeline_data = {
        "duration_sec": 28.0,
        "scenes": [
            {
                "id": "scene_01",
                "start": 0.0,
                "end": 14.0,
                "sentences": [{"text": "Narration text for scene 1.", "start": 0.5, "end": 13.5}],
            },
            {
                "id": "scene_02",
                "start": 14.0,
                "end": 28.0,
                "sentences": [{"text": "Narration text for scene 2.", "start": 14.5, "end": 27.5}],
            },
        ],
    }
    with open(run_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline_data, f)

    # Create dummy local user video
    own_file_path = tmp_path / "my_local_footage.mp4"
    cmd_gen = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=320x240:rate=30",
        "-c:v", "libx264", str(own_file_path),
    ]
    subprocess.run(cmd_gen, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Create dummy downloadable video in cache
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    cached_download_file = cache_dir / "test_download.mp4"
    cmd_dl = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=8:size=640x360:rate=30",
        "-c:v", "libx264", str(cached_download_file),
    ]
    subprocess.run(cmd_dl, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Create candidates.json
    candidates_json_data = {
        "date": "2026-11-08",
        "shots": [
            {
                "shot_id": "scene_01_00",
                "candidates": [
                    {
                        "id": "pix_video_1",
                        "source": "pixabay",
                        "title": "Parliament Building Kathmandu",
                        "page_url": "https://pixabay.com/v/1",
                        "download_url": "https://pixabay.com/v1.mp4",
                        "kind": "video",
                        "license": "Pixabay Content License",
                        "author": "Filmmaker",
                        "width": 1920,
                        "height": 1080,
                        "duration": 8.0,
                        "score": 0.9,
                    }
                ],
            }
        ],
    }
    with open(review_dir / "candidates.json", "w", encoding="utf-8") as cjf:
        json.dump(candidates_json_data, cjf)

    # Create picks.json testing:
    # 1. pick action
    # 2. card action
    # 3. own_file action
    # 4. research action
    picks_payload = {
        "date": "2026-11-08",
        "picks": {
            "scene_01_00": {
                "action": "pick",
                "scene_id": "scene_01",
                "shot_index": 0,
                "candidate_id": "pix_video_1",
            },
            "scene_01_01": {
                "action": "card",
                "scene_id": "scene_01",
                "shot_index": 1,
                "card_text": "Nepal Parliament Approves Clean Energy Bill",
            },
            "scene_02_00": {
                "action": "own_file",
                "scene_id": "scene_02",
                "shot_index": 0,
                "file_path": str(own_file_path),
                "license": "own work",
                "attribution": "Field Video by User",
            },
            "scene_02_01": {
                "action": "research",
                "scene_id": "scene_02",
                "shot_index": 1,
                "query": "nepal traditional oil lamps tihar",
            },
        },
    }
    picks_file = media_dir / "picks.json"
    with open(picks_file, "w", encoding="utf-8") as pf:
        json.dump(picks_payload, pf)

    agent = MediaAgent(run_dir, config=review_config, allow_fixture=True)

    # Mock download_file to return cached file
    monkeypatch.setattr(agent.cache_mgr, "download_file", lambda url, offline=False: cached_download_file)

    outputs = agent.apply_picks(picks_file)
    assert len(outputs) >= 3

    # Verify manifest exists and validates against contract
    manifest_path = media_dir / "media_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest_data = json.load(mf)

    is_valid, err = validate_contract(manifest_data, agent.schema_contract_path)
    assert is_valid, f"Contract validation failed: {err}"

    shots = manifest_data["shots"]
    assert len(shots) == 4

    # Shot 0: pick (video)
    assert shots[0]["type"] == "video"
    assert shots[0]["source"] == "pixabay"
    assert shots[0]["file"].endswith(".mp4")

    # Shot 1: card
    assert shots[1]["type"] == "card"
    assert shots[1]["card_text"] == "Nepal Parliament Approves Clean Energy Bill"
    assert shots[1]["motion"] == "zoom_in"
    assert Path(run_dir / shots[1]["file"]).exists()

    # Shot 2: own_file (video)
    assert shots[2]["type"] == "video"
    assert shots[2]["source"] == "user-provided"
    assert shots[2]["author"] == "User"

    # Shot 3: research (represented with card placeholder)
    assert shots[3]["type"] == "card"

    # Verify candidates_round2.json created for research action
    r2_path = review_dir / "candidates_round2.json"
    assert r2_path.exists()
    r2_data = json.loads(r2_path.read_text(encoding="utf-8"))
    assert len(r2_data["research_shots"]) == 1
    assert r2_data["research_shots"][0]["shot_id"] == "scene_02_01"


def test_license_policy_enforcement_rejects_cc_by_sa(tmp_path, review_config, monkeypatch):
    """Verify license policy is enforced: CC BY-SA pick is refused when allow_cc_by_sa is false."""
    run_dir = tmp_path / "runs" / "2026-11-08"
    run_dir.mkdir(parents=True)
    script_dir = run_dir / "script"
    script_dir.mkdir()
    media_dir = run_dir / "media"
    review_dir = media_dir / "review"
    review_dir.mkdir(parents=True)

    script_data = {
        "date": "2026-11-08",
        "run_kind": "test",
        "origin": "fixture",
        "scenes": [{"id": "scene_01", "headline": "Headline", "visual_keywords": [], "narration": "Text."}],
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f)

    timeline_data = {
        "duration_sec": 7.0,
        "scenes": [
            {"id": "scene_01", "start": 0.0, "end": 7.0, "sentences": [{"text": "Text.", "start": 0.5, "end": 6.5}]}
        ],
    }
    with open(run_dir / "timeline.json", "w", encoding="utf-8") as f:
        json.dump(timeline_data, f)

    candidates_json_data = {
        "date": "2026-11-08",
        "shots": [
            {
                "shot_id": "scene_01_00",
                "candidates": [
                    {
                        "id": "cc_sa_cand",
                        "source": "wikimedia_commons",
                        "title": "CC BY-SA Media File",
                        "page_url": "https://commons.wikimedia.org/wiki/File:test.jpg",
                        "download_url": "https://upload.wikimedia.org/test.jpg",
                        "kind": "image",
                        "license": "CC BY-SA 4.0",
                        "author": "Uploader",
                        "width": 1920,
                        "height": 1080,
                        "duration": None,
                        "score": 0.9,
                    }
                ],
            }
        ],
    }
    with open(review_dir / "candidates.json", "w", encoding="utf-8") as cjf:
        json.dump(candidates_json_data, cjf)

    picks_payload = {
        "date": "2026-11-08",
        "picks": {
            "scene_01_00": {
                "action": "pick",
                "scene_id": "scene_01",
                "shot_index": 0,
                "candidate_id": "cc_sa_cand",
            }
        },
    }
    picks_file = media_dir / "picks.json"
    with open(picks_file, "w", encoding="utf-8") as pf:
        json.dump(picks_payload, pf)

    review_config["settings"]["allow_cc_by_sa"] = False
    agent = MediaAgent(run_dir, config=review_config, allow_fixture=True)

    with pytest.raises(ValueError, match="disallowed license: 'CC BY-SA 4.0'"):
        agent.apply_picks(picks_file)


def test_library_report_formatting(tmp_path):
    """Verify generate_library_report outputs formatted table and handles empty/non-empty folders."""
    lib_dir = tmp_path / "media_library"
    lib_dir.mkdir()

    # Empty library
    rep_empty = generate_library_report(lib_dir)
    assert "0 media files found" in rep_empty

    # Create dummy media and sidecar
    test_img = lib_dir / "kathmandu_skyline.jpg"
    im = Image.new("RGB", (320, 240), color=(50, 100, 150))
    im.save(test_img)

    sidecar = lib_dir / "kathmandu_skyline.json"
    sidecar_data = {
        "keywords": ["kathmandu", "skyline", "city"],
        "license": "own work",
        "people_visible": False,
    }
    sidecar.write_text(json.dumps(sidecar_data), encoding="utf-8")

    rep_full = generate_library_report(lib_dir)
    assert "kathmandu_skyline.jpg" in rep_full
    assert "Image" in rep_full
    assert "kathmandu, skyline, city" in rep_full
    assert "Total media items: 1" in rep_full


def test_apply_picks_download_or_norm_failure_creates_card_and_syncs_render_plan(tmp_path, review_config, monkeypatch):
    """Verify apply_picks creates non-stock headline cards on download/normalization failure without substituting stock, and syncs render_plan.json."""
    run_dir = tmp_path / "runs" / "2026-11-08"
    (run_dir / "script").mkdir(parents=True)
    (run_dir / "media" / "review").mkdir(parents=True)
    (run_dir / "motion" / "overlays").mkdir(parents=True)

    script_data = {
        "date": "2026-11-08",
        "run_kind": "test",
        "origin": "fixture",
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Hydro Project Progress",
                "visual_keywords": ["hydropower dam"],
                "narration": "Engineers inspected the hydropower site today. Generation remains steady.",
            }
        ],
    }
    (run_dir / "script" / "script.json").write_text(json.dumps(script_data), encoding="utf-8")

    timeline_data = {
        "duration_sec": 12.0,
        "scenes": [
            {
                "id": "scene_01",
                "start": 0.0,
                "end": 12.0,
                "sentences": [{"text": "Engineers inspected the site.", "start": 0.5, "end": 11.5}],
            }
        ],
    }
    (run_dir / "timeline.json").write_text(json.dumps(timeline_data), encoding="utf-8")

    # Pre-existing render_plan with 3 old shots and an illustrative_tag overlay
    old_plan = {
        "resolution": "1920x1080",
        "fps": 30,
        "audio_file": "audio/narration.wav",
        "duration_sec": 12.0,
        "captions_ass": "motion/captions.ass",
        "overlays": [
            {
                "id": "scene_01_lower_third",
                "scene_id": "scene_01",
                "type": "lower_third",
                "png": "motion/overlays/scene_01_lower_third.png",
                "start": 1.0,
                "end": 7.0,
                "x": 96,
                "y": 730,
                "w": 1320,
                "h": 155,
            },
            {
                "id": "scene_01_illustrative_tag",
                "scene_id": "scene_01",
                "type": "illustrative_tag",
                "png": "motion/overlays/illustrative_tag.png",
                "start": 0.2,
                "end": 11.8,
                "x": 1540,
                "y": 54,
                "w": 284,
                "h": 44,
            },
        ],
        "scenes": [
            {
                "id": "scene_01",
                "start": 0.0,
                "end": 12.0,
                "media": "media/shots/scene_01/old_00.mp4",
                "lower_third": {"headline": "Hydro Project Progress", "source_credit": "Desk"},
                "caption_style": {"position": "bottom"},
                "transition": {"type": "dip_to_black", "duration_sec": 0.2},
                "shots": [
                    {"shot_index": 0, "file": "media/shots/scene_01/old_00.mp4", "start": 0.0, "end": 4.0, "type": "video"},
                    {"shot_index": 1, "file": "media/shots/scene_01/old_01.mp4", "start": 4.0, "end": 8.0, "type": "video"},
                    {"shot_index": 2, "file": "media/shots/scene_01/old_02.mp4", "start": 8.0, "end": 12.0, "type": "video"},
                ],
                "overlays": [
                    {
                        "id": "scene_01_lower_third",
                        "scene_id": "scene_01",
                        "type": "lower_third",
                        "png": "motion/overlays/scene_01_lower_third.png",
                        "start": 1.0,
                        "end": 7.0,
                        "x": 96,
                        "y": 730,
                        "w": 1320,
                        "h": 155,
                    },
                    {
                        "id": "scene_01_illustrative_tag",
                        "scene_id": "scene_01",
                        "type": "illustrative_tag",
                        "png": "motion/overlays/illustrative_tag.png",
                        "start": 0.2,
                        "end": 11.8,
                        "x": 1540,
                        "y": 54,
                        "w": 284,
                        "h": 44,
                    },
                ],
            }
        ],
    }
    (run_dir / "motion" / "render_plan.json").write_text(json.dumps(old_plan), encoding="utf-8")
    (run_dir / "render_plan.json").write_text(json.dumps(old_plan), encoding="utf-8")

    # Pre-existing media_manifest.json with 3 old shots so picks_log records shrinkage/changes
    (run_dir / "media" / "media_manifest.json").write_text(
        json.dumps({
            "date": "2026-11-08",
            "orientation": "landscape",
            "resolution": [1920, 1080],
            "fps": 30,
            "shots": [
                {"scene_id": "scene_01", "shot_index": 0, "start": 0.0, "end": 4.0, "type": "video", "file": "media/shots/scene_01/old_00.mp4", "origin_url": "https://old/0", "sha256": "old0"},
                {"scene_id": "scene_01", "shot_index": 1, "start": 4.0, "end": 8.0, "type": "video", "file": "media/shots/scene_01/old_01.mp4", "origin_url": "https://old/1", "sha256": "old1"},
                {"scene_id": "scene_01", "shot_index": 2, "start": 8.0, "end": 12.0, "type": "video", "file": "media/shots/scene_01/old_02.mp4", "origin_url": "https://old/2", "sha256": "old2"},
            ],
        }),
        encoding="utf-8",
    )

    # Corrupt file returned for second pick so normalize_video fails
    corrupt_mp4 = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"NOT_A_VALID_MP4_STREAM")

    candidates_json_data = {
        "date": "2026-11-08",
        "shots": [
            {
                "shot_id": "scene_01_00",
                "candidates": [
                    {
                        "id": "cand_missing",
                        "source": "pixabay",
                        "title": "Expired Video Link",
                        "page_url": "https://pixabay.com/v/expired",
                        "download_url": "https://cdn.pixabay.com/expired.mp4",
                        "kind": "video",
                        "license": "Pixabay Content License",
                        "author": "AuthorA",
                        "width": 1920,
                        "height": 1080,
                        "duration": 10.0,
                    }
                ],
            },
            {
                "shot_id": "scene_01_01",
                "candidates": [
                    {
                        "id": "cand_corrupt",
                        "source": "pixabay",
                        "title": "Corrupt Video Stream",
                        "page_url": "https://pixabay.com/v/corrupt",
                        "download_url": "https://cdn.pixabay.com/corrupt.mp4",
                        "kind": "video",
                        "license": "Pixabay Content License",
                        "author": "AuthorB",
                        "width": 1920,
                        "height": 1080,
                        "duration": 10.0,
                    }
                ],
            },
        ],
    }
    (run_dir / "media" / "review" / "candidates.json").write_text(json.dumps(candidates_json_data), encoding="utf-8")

    picks_file = run_dir / "media" / "picks.json"
    picks_file.write_text(
        json.dumps({
            "date": "2026-11-08",
            "picks": {
                "scene_01_00": {"action": "pick", "scene_id": "scene_01", "shot_index": 0, "candidate_id": "cand_missing"},
                "scene_01_01": {"action": "pick", "scene_id": "scene_01", "shot_index": 1, "candidate_id": "cand_corrupt"},
            },
        }),
        encoding="utf-8",
    )

    agent = MediaAgent(run_dir, config=review_config, allow_fixture=True)

    def _fake_download(url: str, offline: bool = False):
        if "expired" in url:
            return None
        return corrupt_mp4

    monkeypatch.setattr(agent.cache_mgr, "download_file", _fake_download)

    agent.apply_picks(picks_file)

    manifest_data = json.loads((run_dir / "media" / "media_manifest.json").read_text(encoding="utf-8"))
    is_valid, err = validate_contract(manifest_data, agent.schema_contract_path)
    assert is_valid, f"media_manifest validation failed: {err}"

    shots = manifest_data["shots"]
    assert len(shots) == 2
    assert all(s["type"] == "card" and s["source"] == "system" for s in shots)
    assert all((run_dir / s["file"]).exists() for s in shots)

    plog = json.loads((run_dir / "media" / "picks_log.json").read_text(encoding="utf-8"))
    assert len(plog["failed_downloads"]) == 2
    assert len(plog["cards_created"]) == 2
    assert all(c["reason"] == "download_failed" for c in plog["cards_created"])
    assert len(plog["applied_picks"]) == 0
    assert plog["previous_total_shots"] == 3

    # Verify render_plan.json synced cleanly without stock illustrative tag
    synced_plan = json.loads((run_dir / "motion" / "render_plan.json").read_text(encoding="utf-8"))
    ok_plan, plan_err = validate_contract(synced_plan, "contracts/render_plan.schema.json")
    assert ok_plan, f"render_plan validation failed: {plan_err}"
    assert len(synced_plan["scenes"][0]["shots"]) == 2
    assert synced_plan["scenes"][0]["shots"][0]["sha256"] == shots[0]["sha256"]
    assert synced_plan["scenes"][0]["shots"][0]["type"] == "card"
    assert not any(ov.get("type") == "illustrative_tag" for ov in synced_plan["overlays"])
    assert any(ov.get("type") == "lower_third" for ov in synced_plan["overlays"])


def test_score_relevance_no_dilution_multi_keyword_and_sensitive():
    """Verify scoring against active query does not dilute across unsearched visual_keywords or cap sensitive queries."""
    scene_multi = {
        "id": "intro",
        "headline": "Morning Nepal News",
        "visual_keywords": ["Kathmandu valley", "news studio", "map of Nepal"],
        "narration": "Good morning from Nepal.",
    }
    cand_valley = Candidate(
        source="wikimedia_commons",
        id="wc_1",
        kind="video",
        page_url="https://commons.wikimedia.org/wiki/File:Kathmandu.webm",
        download_url="https://upload.wikimedia.org/Kathmandu.webm",
        width=1920,
        height=1080,
        duration=8.0,
        license_id="CC0",
        license_url="",
        author="Author",
        title="Kathmandu valley morning view",
        tags=["kathmandu", "valley", "nepal"],
    )
    score_multi, tokens_multi = score_candidate_relevance(
        cand=cand_valley,
        query="Kathmandu valley",
        visual_keywords=scene_multi["visual_keywords"],
        scene=scene_multi,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=False,
    )
    assert score_multi >= 0.5
    assert "kathmandu" in tokens_multi

    # Sensitive scene atmospheric query should exceed min_relevance (0.5) when matching atmospheric tokens
    scene_sens = {
        "id": "scene_01",
        "headline": "Court Remands Suspect After Highway Accident",
        "visual_keywords": ["courtroom exterior", "police car", "nepal flag"],
        "narration": "Police arrested a suspect following a fatal crash.",
    }
    cand_atmos = Candidate(
        source="pixabay",
        id="pix_road",
        kind="video",
        page_url="https://pixabay.com/v/999",
        download_url="https://pixabay.com/999.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="Artist",
        title="Empty road landscape morning mist",
        tags=["empty", "road", "landscape"],
    )
    score_sens, _ = score_candidate_relevance(
        cand=cand_atmos,
        query="empty road landscape",
        visual_keywords=scene_sens["visual_keywords"],
        scene=scene_sens,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=True,
    )
    assert score_sens >= 0.5

    # Sensitive scene with person tag still rejected at 0.0
    cand_atmos_people = Candidate(
        source="pixabay",
        id="pix_road_people",
        kind="video",
        page_url="https://pixabay.com/v/1000",
        download_url="https://pixabay.com/1000.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="Artist",
        title="Empty road landscape with walking person",
        tags=["empty", "road", "landscape", "person"],
    )
    score_people, _ = score_candidate_relevance(
        cand=cand_atmos_people,
        query="empty road landscape",
        visual_keywords=scene_sens["visual_keywords"],
        scene=scene_sens,
        foreign_places=DEFAULT_FOREIGN_PLACES,
        is_sensitive=True,
    )
    assert score_people == 0.0


def test_picker_html_has_direct_apply_picks_controls(tmp_path):
    """Verify picker.html includes direct Apply Picks button and auto-pick fallback control without external links."""
    candidates_data = {
        "date": "2026-11-12",
        "min_relevance": 0.5,
        "total_shots": 1,
        "shots": [
            {
                "shot_id": "scene_01_00",
                "scene_id": "scene_01",
                "shot_index": 0,
                "scene_headline": "Test Headline",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "narration_sentences": ["Test sentence."],
                "visual_keywords": ["nepal"],
                "auto_pick_id": None,
                "auto_pick_type": "card",
                "best_score": 0.4,
                "candidates": [],
            }
        ],
    }
    out_html = tmp_path / "picker.html"
    generate_picker_html(candidates_data, out_html)
    html_text = out_html.read_text(encoding="utf-8")
    assert "applyPicksToOffice()" in html_text
    assert "fillCardsWithTopClips()" in html_text
    assert "/apply-picks" in html_text
    assert "<script src=\"http" not in html_text


def test_draw_card_image_visible_center_text_landscape_and_vertical(tmp_path: Path) -> None:
    """Assert draw_card_image() renders visible text pixels in the center region for 16:9 and 9:16."""
    from src.utils.media_review import draw_card_image

    # Landscape 16:9 (1920x1080)
    card_16_9 = tmp_path / "card_16_9.png"
    draw_card_image("Nepal Economic Growth Exceeds Target", card_16_9, width=1920, height=1080)
    assert card_16_9.exists()
    img_l = Image.open(card_16_9)
    assert img_l.size == (1920, 1080)
    # Center text box crop
    center_l = img_l.crop((300, 300, 1620, 780))
    # Count bright white text pixels in center
    has_text_l = any(
        p[0] > 230 and p[1] > 230 and p[2] > 230
        for p in list(center_l.getdata())
    )
    assert has_text_l is True

    # Vertical 9:16 (1080x1920)
    card_9_16 = tmp_path / "card_9_16.png"
    draw_card_image("Nepal Economic Growth Exceeds Target", card_9_16, width=1080, height=1920)
    assert card_9_16.exists()
    img_v = Image.open(card_9_16)
    assert img_v.size == (1080, 1920)
    # Center text box crop
    center_v = img_v.crop((150, 600, 930, 1320))
    has_text_v = any(
        p[0] > 230 and p[1] > 230 and p[2] > 230
        for p in list(center_v.getdata())
    )
    assert has_text_v is True


def test_picker_html_balanced_div_tags(tmp_path: Path) -> None:
    """Assert picker.html has balanced <div> tags both with and without candidate shots."""
    from html.parser import HTMLParser

    class DivBalanceParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.div_open_count = 0
            self.div_close_count = 0

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag.lower() == "div":
                self.div_open_count += 1

        def handle_endtag(self, tag: str) -> None:
            if tag.lower() == "div":
                self.div_close_count += 1

    # Case 1: Populated candidates
    data_with_candidates = {
        "date": "2026-11-20",
        "shots": [
            {
                "shot_id": "sc_01_00",
                "scene_id": "sc_01",
                "shot_index": 0,
                "scene_headline": "Test Story",
                "duration": 5.0,
                "candidates": [
                    {
                        "id": "cand_1",
                        "source": "wikimedia_commons",
                        "kind": "image",
                        "thumbnail_url": "https://example.com/thumb.jpg",
                        "relevance_score": 0.85,
                    }
                ],
            }
        ],
    }
    html_file1 = tmp_path / "picker_with_cands.html"
    generate_picker_html(data_with_candidates, html_file1)
    parser1 = DivBalanceParser()
    parser1.feed(html_file1.read_text(encoding="utf-8"))
    assert parser1.div_open_count > 0
    assert parser1.div_open_count == parser1.div_close_count, (
        f"Unbalanced <div> tags in populated picker: {parser1.div_open_count} open vs {parser1.div_close_count} close"
    )

    # Case 2: Zero candidates (shots: [])
    data_empty = {"date": "2026-11-20", "shots": []}
    html_file2 = tmp_path / "picker_empty.html"
    generate_picker_html(data_empty, html_file2)
    parser2 = DivBalanceParser()
    parser2.feed(html_file2.read_text(encoding="utf-8"))
    assert parser2.div_open_count > 0
    assert parser2.div_open_count == parser2.div_close_count, (
        f"Unbalanced <div> tags in empty picker: {parser2.div_open_count} open vs {parser2.div_close_count} close"
    )




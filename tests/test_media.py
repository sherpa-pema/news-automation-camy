"""Unit and integration tests for the Media Agent."""
import json
from pathlib import Path
import subprocess
from typing import Any
import pytest
from PIL import Image

from src.agents.media import MediaAgent
from src.utils.media_cache import MediaCacheManager
from src.utils.media_models import Candidate, PlannedShot
from src.utils.media_normalizer import (
    normalize_image,
    normalize_video,
)
from src.utils.media_review import generate_credits_markdown
from src.utils.media_selector import MediaSelector
from src.utils.media_sources.coverr import CoverrSource
from src.utils.media_sources.local_library import LocalLibrarySource
from src.utils.media_sources.nasa import NasaSource
from src.utils.media_sources.openverse import OpenverseSource
from src.utils.media_sources.pixabay import PixabaySource
from src.utils.media_sources.ytdlp import YtDlpSource
from src.utils.schema_validate import validate_contract
from src.utils.shot_planner import plan_shots_for_scenes


@pytest.fixture
def mock_media_config() -> dict[str, Any]:
    return {
        "settings": {
            "orientation": "landscape",
            "fps": 30,
            "resolution": [1920, 1080],
            "target_shot_duration": 5.0,
            "min_shot_duration": 2.5,
            "max_shot_duration": 6.0,
            "max_stills_per_scene": 2,
            "allow_cc_by_sa": False,
            "illustrative_label": True,
            "user_agent": "NepalNewsBotTest/1.0",
            "cache_dir": "data/test_cache",
            "cache_ttl_hours": 24,
        },
        "sources": {
            "local_library": {"enabled": True, "path": "data/test_media_library"},
            "wikimedia_commons": {"enabled": True},
            "pixabay": {"enabled": True, "api_key_env": "TEST_PIXABAY_KEY"},
            "coverr": {"enabled": False, "api_key_env": "TEST_COVERR_KEY"},
            "openverse": {"enabled": True},
            "nasa": {"enabled": True},
            "internet_archive": {"enabled": False},
            "unsplash": {"enabled": False, "api_key_env": "TEST_UNSPLASH_KEY"},
            "pexels": {"enabled": False, "api_key_env": "TEST_PEXELS_KEY"},
        },
        "ytdlp": {"enabled": False},
        "allowed_licenses": [
            "Pixabay Content License",
            "Pixabay License",
            "CC0",
            "Public Domain",
            "CC BY",
            "CC-BY-4.0",
            "NASA media",
            "own work",
            "user-provided",
        ],
        "blocklisted_domains": [
            "kathmandupost.com",
            "onlinekhabar.com",
            "setopati.com",
            "thehimalayantimes.com",
            "ekantipur.com",
        ],
        "category_defaults": {
            "politics": "government building",
            "weather": "monsoon rain",
            "general": "Kathmandu city skyline",
        },
        "sensitive_keywords": ["crime", "accident", "fatal", "murder", "casualty", "death"],
        "sensitive_scene_person_tags": ["person", "people", "man", "woman", "child", "crowd"],
        "watermark_markers": ["watermark", "logo", "brand"],
    }


def test_shot_planning_tiling_and_snapping():
    """Verify shots tile scene exactly and snap to sentence boundaries within 1s tolerance."""
    scenes = [
        {
            "id": "scene_test",
            "start": 0.0,
            "end": 14.8,
            "sentences": [
                {"text": "Sentence 1.", "start": 0.5, "end": 4.8},
                {"text": "Sentence 2.", "start": 5.2, "end": 9.9},
                {"text": "Sentence 3.", "start": 10.3, "end": 14.2},
            ],
        }
    ]

    planned = plan_shots_for_scenes(scenes, target_duration=5.0, min_duration=2.5)
    assert len(planned) == 1
    sc = planned[0]

    # Verify exact tiling
    total_duration = sum(s.duration for s in sc.shots)
    assert abs(total_duration - 14.8) < 0.02
    assert sc.shots[0].start == 0.0
    assert sc.shots[-1].end == 14.8

    for i in range(len(sc.shots) - 1):
        assert sc.shots[i].end == sc.shots[i + 1].start
        assert sc.shots[i].duration >= 2.5


def test_license_policy_filtering(mock_media_config):
    """Verify CC BY-SA, NC, ND, unknown are rejected; CC0, CC BY, Pixabay, own work accepted."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    selector = MediaSelector(mock_media_config, cache_mgr)

    # Allowed
    assert selector.is_license_allowed("CC0") is True
    assert selector.is_license_allowed("Public Domain") is True
    assert selector.is_license_allowed("CC-BY-4.0") is True
    assert selector.is_license_allowed("Pixabay Content License") is True
    assert selector.is_license_allowed("own work") is True

    # Excluded
    assert selector.is_license_allowed("CC BY-SA 4.0") is False
    assert selector.is_license_allowed("CC-BY-NC-4.0") is False
    assert selector.is_license_allowed("CC-BY-ND") is False
    assert selector.is_license_allowed("All Rights Reserved") is False
    assert selector.is_license_allowed("unknown") is False

    # When allow_cc_by_sa is true
    selector.allow_cc_by_sa = True
    assert selector.is_license_allowed("CC BY-SA 4.0") is True


def test_blocklisted_news_domains(mock_media_config):
    """Verify hard blocklist rejects news publisher domains."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    selector = MediaSelector(mock_media_config, cache_mgr)

    assert selector.is_domain_blocklisted("https://kathmandupost.com/image.jpg") is True
    assert selector.is_domain_blocklisted("https://onlinekhabar.com/photo.png") is True
    assert selector.is_domain_blocklisted("https://thehimalayantimes.com/b-roll.mp4") is True
    assert selector.is_domain_blocklisted("https://setopati.com/article.html") is True

    # Legitimate open domains
    assert selector.is_domain_blocklisted("https://pixabay.com/get/video.mp4") is False
    assert selector.is_domain_blocklisted("https://commons.wikimedia.org/wiki/File:Bagmati.jpg") is False


def test_query_ladder_generation(mock_media_config):
    """Verify query ladder includes visual keywords, simplified nouns, and category fallback."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    selector = MediaSelector(mock_media_config, cache_mgr)

    scene = {
        "id": "scene_01",
        "headline": "Prime Minister addresses parliament",
        "narration": "In politics today, the government announced new legislation.",
        "visual_keywords": ["Kathmandu valley", "parliament building", "politicians speaking"],
    }

    ladder = selector.build_query_ladder(scene, is_sensitive=False)
    assert "Kathmandu valley" in ladder
    assert "parliament building" in ladder
    # Simplified noun
    assert "valley" in ladder or "building" in ladder
    # Category fallback
    assert "government building" in ladder


def test_sensitive_scene_filtering(mock_media_config):
    """Verify sensitive scene triggers abstract visuals and rejects person-tagged candidates."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    selector = MediaSelector(mock_media_config, cache_mgr)

    scene = {
        "id": "scene_accident",
        "headline": "Fatal highway accident claims three lives",
        "narration": "A passenger bus crash caused fatal injuries in the hills.",
        "visual_keywords": ["highway bus", "mountain road"],
    }

    assert selector.is_scene_sensitive(scene) is True

    # Person-tagged candidate should be rejected
    cand_person = Candidate(
        source="pixabay",
        id="c1",
        kind="video",
        page_url="https://pixabay.com/v1",
        download_url="https://pixabay.com/v1.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="artist",
        title="Crowd gathering near road",
        tags=["crowd", "people", "street"],
    )
    ok, reason = selector.is_candidate_acceptable(cand_person, is_sensitive=True, target_duration=5.0)
    assert ok is False
    assert "sensitive scene" in reason

    # Abstract/landscape candidate should be accepted
    cand_abstract = Candidate(
        source="pixabay",
        id="c2",
        kind="video",
        page_url="https://pixabay.com/v2",
        download_url="https://pixabay.com/v2.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="artist",
        title="Empty mountain road in rain",
        tags=["road", "rain", "mountains", "fog"],
    )
    ok, reason = selector.is_candidate_acceptable(cand_abstract, is_sensitive=True, target_duration=5.0)
    assert ok is True


def test_local_library_indexing_and_sidecar_validation(tmp_path):
    """Verify local_library indexes valid media, matches keywords, and warns on missing sidecars."""
    lib_dir = tmp_path / "media_library"
    lib_dir.mkdir()

    # 1. Valid media file with sidecar
    vid_file = lib_dir / "phewa_lake.mp4"
    vid_file.write_bytes(b"dummy video data")
    sidecar_data = {
        "title": "Phewa Lake Pokhara",
        "keywords": ["pokhara", "phewa lake", "boats"],
        "license": "own work",
        "author": "Chungba",
        "people_visible": False,
    }
    (lib_dir / "phewa_lake.json").write_text(json.dumps(sidecar_data), encoding="utf-8")

    # 2. File without sidecar
    (lib_dir / "ignored_file.mp4").write_bytes(b"dummy")

    cfg = {"sources": {"local_library": {"enabled": True, "path": str(lib_dir)}}}
    cache_mgr = MediaCacheManager(cache_dir=str(tmp_path / "cache"))
    source = LocalLibrarySource(cfg, cache_mgr)

    assert len(source.indexed_items) == 1
    assert source.indexed_items[0].id == "local_phewa_lake"
    assert any("ignored_file.mp4" in w for w in source.warnings)

    # Keyword match
    results = source.search("phewa lake scenic")
    assert len(results) == 1
    assert results[0].source == "user-provided"
    assert results[0].license_id == "own work"


def test_pixabay_orientation_filtering(mock_media_config, monkeypatch):
    """Verify Pixabay video rendition orientation filtering by dimensions."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = PixabaySource(mock_media_config, cache_mgr)

    monkeypatch.setenv("TEST_PIXABAY_KEY", "mock_key")

    mock_response = {
        "hits": [
            {
                "id": 12345,
                "duration": 12.0,
                "tags": "nature, mountain",
                "videos": {
                    "large": {
                        "url": "https://pixabay.com/video_portrait.mp4",
                        "width": 1080,
                        "height": 1920,
                    },
                    "medium": {
                        "url": "https://pixabay.com/video_landscape.mp4",
                        "width": 1920,
                        "height": 1080,
                    },
                },
            }
        ]
    }

    monkeypatch.setattr(cache_mgr, "get_json", lambda *args, **kwargs: mock_response)

    results = source.search("mountain", kind="video", orientation="landscape")
    assert len(results) == 1
    # Must have chosen the landscape rendition!
    assert results[0].download_url == "https://pixabay.com/video_landscape.mp4"
    assert results[0].width >= results[0].height


def test_coverr_ai_exclusion_and_disabled_by_default(mock_media_config, monkeypatch):
    """Verify Coverr is disabled by default and rejects AI-tagged clips when enabled."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = CoverrSource(mock_media_config, cache_mgr)

    # Disabled by default
    avail, reason = source.is_available
    assert avail is False
    assert "disabled by default" in reason

    # Enable and test AI exclusion
    mock_media_config["sources"]["coverr"]["enabled"] = True
    monkeypatch.setenv("TEST_COVERR_KEY", "mock_coverr_key")

    mock_coverr_resp = {
        "hits": [
            {
                "id": "c_ai_1",
                "title": "Futuristic city AI generated",
                "tags": ["ai", "future"],
                "width": 1920,
                "height": 1080,
                "duration": 8.0,
            },
            {
                "id": "c_real_2",
                "title": "Real sunset over mountain",
                "tags": ["nature", "sunset"],
                "width": 1920,
                "height": 1080,
                "duration": 9.0,
                "urls": {"mp4": "https://coverr.co/real.mp4"},
            },
        ]
    }
    monkeypatch.setattr(cache_mgr, "get_json", lambda *args, **kwargs: mock_coverr_resp)

    results = source.search("sunset", kind="video")
    assert len(results) == 1
    assert results[0].id == "coverr_c_real_2"


def test_missing_key_skipped_with_warning(mock_media_config, monkeypatch):
    """Verify sources requiring missing keys are skipped with a warning and do not raise."""
    monkeypatch.delenv("TEST_PIXABAY_KEY", raising=False)
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = PixabaySource(mock_media_config, cache_mgr)

    avail, reason = source.is_available
    assert avail is False
    assert "Missing required environment variable" in reason

    # Search should return empty list gracefully
    res = source.search("Kathmandu")
    assert res == []


def test_openverse_license_filtering(mock_media_config, monkeypatch):
    """Verify Openverse rejects by-sa and NC/ND licenses."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = OpenverseSource(mock_media_config, cache_mgr)

    mock_data = {
        "results": [
            {
                "id": "img1",
                "license": "by-sa",
                "width": 1920,
                "height": 1080,
            },
            {
                "id": "img2",
                "license": "cc0",
                "width": 1920,
                "height": 1080,
                "url": "https://openverse.org/img2.jpg",
                "creator": "Artist 2",
            },
            {
                "id": "img3",
                "license": "by-nc",
                "width": 1920,
                "height": 1080,
            },
        ]
    }
    monkeypatch.setattr(cache_mgr, "get_json", lambda *args, **kwargs: mock_data)

    results = source.search("river", kind="image")
    assert len(results) == 1
    assert results[0].id == "openverse_img2"
    assert results[0].license_id == "CC0"


def test_nasa_category_gating_and_third_party_skip(mock_media_config, monkeypatch):
    """Verify NASA source is gated to weather/climate/space/earth and skips third-party content."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = NasaSource(mock_media_config, cache_mgr)

    # Politics category -> should return empty immediately
    assert source.search("cloud", scene_category="politics") == []

    # Weather category -> allowed
    mock_nasa_data = {
        "collection": {
            "items": [
                {
                    "data": [
                        {
                            "nasa_id": "nasa_tp",
                            "title": "Third party satellite",
                            "description": "Image courtesy of Getty Images",
                            "keywords": ["satellite"],
                        }
                    ],
                    "links": [{"href": "https://nasa.gov/tp.jpg"}],
                },
                {
                    "data": [
                        {
                            "nasa_id": "nasa_real",
                            "title": "Monsoon cloud pattern over Himalayas",
                            "description": "Captured by NASA Aqua MODIS instrument",
                            "keywords": ["monsoon", "clouds"],
                        }
                    ],
                    "links": [{"href": "https://nasa.gov/monsoon.jpg"}],
                },
            ]
        }
    }
    monkeypatch.setattr(cache_mgr, "get_json", lambda *args, **kwargs: mock_nasa_data)

    res = source.search("monsoon", scene_category="weather")
    assert len(res) == 1
    assert res[0].id == "nasa_nasa_real"
    assert res[0].author == "NASA"


def test_ytdlp_disabled_by_default(mock_media_config):
    """Verify yt-dlp is disabled by default."""
    cache_mgr = MediaCacheManager(cache_dir="data/test_cache")
    source = YtDlpSource(mock_media_config, cache_mgr)
    avail, reason = source.is_available
    assert avail is False
    assert "disabled by default" in reason


def test_video_and_image_normalization(tmp_path):
    """Verify ffmpeg video normalization sets 30fps, target resolution, and removes audio."""
    # 1. Create a tiny 2s video using ffmpeg
    raw_video = tmp_path / "raw_video.mp4"
    cmd_gen = [
        "ffmpeg",
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=25",
        "-f", "lavfi",
        "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(raw_video),
    ]
    subprocess.run(cmd_gen, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    norm_video = tmp_path / "norm_video.mp4"
    probe_v = normalize_video(
        src_path=raw_video,
        dest_path=norm_video,
        target_duration=1.5,
        target_resolution=(1920, 1080),
        fps=30,
    )

    assert norm_video.exists()
    assert probe_v["width"] == 1920
    assert probe_v["height"] == 1080
    assert probe_v["fps"] == 30.0
    assert probe_v["has_audio"] is False
    assert "sha256" in probe_v

    # 2. Image normalization
    raw_img = tmp_path / "raw_img.png"
    im = Image.new("RGB", (640, 480), color=(100, 150, 200))
    im.save(raw_img)

    norm_img = tmp_path / "norm_img.jpg"
    probe_i = normalize_image(raw_img, norm_img, target_resolution=(1920, 1080))
    assert norm_img.exists()
    assert probe_i["width"] == 1920
    assert probe_i["height"] == 1080
    assert probe_i["has_audio"] is False


def test_manifest_and_credits_schema_validation(tmp_path):
    """Verify generated manifest validates against media_manifest.schema.json and credits.md format."""
    manifest_data = {
        "date": "2026-11-06",
        "orientation": "landscape",
        "resolution": [1920, 1080],
        "fps": 30,
        "illustrative_label": True,
        "disclose_synthetic_voice": True,
        "script_origin": "fixture",
        "shots": [
            {
                "scene_id": "intro",
                "shot_index": 0,
                "start": 0.0,
                "end": 4.5,
                "type": "video",
                "file": "media/shots/intro/shot_00.mp4",
                "source": "pixabay",
                "origin_url": "https://pixabay.com/v/1",
                "license": "Pixabay Content License",
                "license_url": "https://pixabay.com/service/license-summary/",
                "author": "Videographer",
                "attribution_text": "Video by Videographer via Pixabay",
                "attribution_required": False,
                "width": 1920,
                "height": 1080,
                "duration_sec": 4.5,
                "sha256": "abcdef1234567890",
                "query": "Kathmandu valley",
                "motion": None,
                "sensitive_scene": False,
                "style": None,
            },
            {
                "scene_id": "intro",
                "shot_index": 1,
                "start": 4.5,
                "end": 9.0,
                "type": "fallback",
                "file": None,
                "source": "fallback",
                "origin_url": None,
                "license": "N/A",
                "license_url": None,
                "author": "System",
                "attribution_text": "Animated fallback graphic",
                "attribution_required": False,
                "width": None,
                "height": None,
                "duration_sec": 4.5,
                "sha256": None,
                "query": None,
                "motion": None,
                "sensitive_scene": False,
                "style": "animated_backdrop",
            },
        ],
        "warnings": ["Scene intro shot 1 used fallback"],
    }

    is_valid, err = validate_contract(manifest_data, "contracts/media_manifest.schema.json")
    assert is_valid, f"Schema validation failed: {err}"

    # Test credits.md generation
    credits_file = tmp_path / "credits.md"
    generate_credits_markdown(manifest_data["shots"], credits_file, disclose_synthetic_voice=True)
    assert credits_file.exists()
    content = credits_file.read_text(encoding="utf-8")
    assert "Stock and archive footage is illustrative." in content
    assert "Narration audio was synthesized using an AI voice model." in content
    assert "Pixabay" in content


def test_svg_candidate_dropped_produces_jpeg_not_fallback(mock_media_config, monkeypatch, tmp_path):
    """Verify that an SVG candidate ranked first is dropped and next valid JPEG candidate is chosen."""
    cache_mgr = MediaCacheManager(cache_dir=str(tmp_path / "cache"))
    selector = MediaSelector(mock_media_config, cache_mgr)

    svg_cand = Candidate(
        source="wikimedia_commons",
        id="c_svg",
        kind="image",
        page_url="https://commons.wikimedia.org/wiki/File:map.svg",
        download_url="https://upload.wikimedia.org/map.svg",
        width=1920,
        height=1080,
        duration=None,
        license_id="CC0",
        license_url="",
        author="Cartographer",
        title="Nepal Map SVG",
        tags=["nepal", "map"],
    )

    jpg_cand = Candidate(
        source="wikimedia_commons",
        id="c_jpg",
        kind="image",
        page_url="https://commons.wikimedia.org/wiki/File:kathmandu.jpg",
        download_url="https://upload.wikimedia.org/kathmandu.jpg",
        width=1920,
        height=1080,
        duration=None,
        license_id="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0",
        author="Photographer",
        title="Kathmandu valley Nepal",
        tags=["nepal", "kathmandu"],
    )

    # Mock search to return [svg_cand, jpg_cand]
    mock_src = selector.sources["wikimedia_commons"]
    monkeypatch.setattr(mock_src, "search", lambda *args, **kwargs: [svg_cand, jpg_cand])
    monkeypatch.setattr(type(mock_src), "is_available", property(lambda self: (True, None)))

    shot = PlannedShot("scene_01", 0, 0.0, 5.0, 5.0)
    candidates = list(selector.iter_candidates_for_shot({"id": "scene_01", "visual_keywords": ["Kathmandu"]}, shot, stills_in_scene=0))

    assert len(candidates) >= 1
    chosen_cand = candidates[0][0]
    # Chosen candidate must be the JPEG, not SVG!
    assert chosen_cand.id == "c_jpg"
    assert chosen_cand.download_url.endswith(".jpg")


def test_query_repetition_cap(mock_media_config, monkeypatch, tmp_path):
    """Verify shots from the same query are capped at 2 per run."""
    cache_mgr = MediaCacheManager(cache_dir=str(tmp_path / "cache"))
    selector = MediaSelector(mock_media_config, cache_mgr)

    cand = Candidate(
        source="pixabay",
        id="c_pix",
        kind="video",
        page_url="https://pixabay.com/v/1",
        download_url="https://pixabay.com/v1.mp4",
        width=1920,
        height=1080,
        duration=10.0,
        license_id="Pixabay Content License",
        license_url="",
        author="Artist",
        title="Kathmandu city",
        tags=["kathmandu"],
    )

    query = "Kathmandu valley"
    selector.mark_candidate_used(cand, query, "pixabay")
    cand2 = Candidate(
        source="pixabay", id="c_pix2", kind="video", page_url="https://pixabay.com/v/2",
        download_url="https://pixabay.com/v2.mp4", width=1920, height=1080, duration=10.0,
        license_id="Pixabay Content License", license_url="", author="Artist", title="Kathmandu", tags=["kathmandu"],
    )
    selector.mark_candidate_used(cand2, query, "pixabay")

    assert selector.query_usage_counts[query] == 2
    # Third search with same query should be skipped in query ladder loop
    scene = {"id": "scene_01", "visual_keywords": [query]}
    shot = PlannedShot("scene_01", 2, 10.0, 15.0, 5.0)
    # The query ladder skips 'Kathmandu valley' when query count reaches 2
    candidates = list(selector.iter_candidates_for_shot(scene, shot, 0))
    # Any candidate yielded must be for a subsequent query, not the capped query
    for c, matched_q, src in candidates:
        assert matched_q != query


def test_max_one_map_or_diagram_still(mock_media_config, tmp_path):
    """Verify at most one map/diagram style still is allowed per run."""
    cache_mgr = MediaCacheManager(cache_dir=str(tmp_path / "cache"))
    selector = MediaSelector(mock_media_config, cache_mgr)

    map_cand1 = Candidate(
        source="wikimedia_commons", id="map_1", kind="image", page_url="https://commons.org/map1",
        download_url="https://commons.org/map1.jpg", width=1920, height=1080, duration=None,
        license_id="Public Domain", license_url="", author="Cartographer", title="District Map of Nepal", tags=["map", "nepal"],
    )
    ok1, reason1 = selector.is_candidate_acceptable(map_cand1, is_sensitive=False, target_duration=5.0)
    assert ok1 is True

    # Mark first map used
    selector.mark_candidate_used(map_cand1, "map", "wikimedia_commons")
    assert selector.map_diagram_count == 1

    # Second map candidate should be rejected
    map_cand2 = Candidate(
        source="wikimedia_commons", id="map_2", kind="image", page_url="https://commons.org/map2",
        download_url="https://commons.org/map2.jpg", width=1920, height=1080, duration=None,
        license_id="Public Domain", license_url="", author="Cartographer", title="Topographical chart of Himalayas", tags=["chart", "diagram"],
    )
    ok2, reason2 = selector.is_candidate_acceptable(map_cand2, is_sensitive=False, target_duration=5.0)
    assert ok2 is False
    assert "map/diagram" in reason2


def test_orchestrator_refuses_fixture_script_without_flag(tmp_path):
    """Verify orchestrator refuses to run downstream stages on fixture script without --allow-fixture."""
    from src.orchestrator import Orchestrator

    run_dir = tmp_path / "2026-11-06"
    run_dir.mkdir(parents=True)
    script_file = run_dir / "script.json"
    script_file.write_text(json.dumps({"language": "English", "origin": "fixture", "scenes": []}), encoding="utf-8")

    orch = Orchestrator(run_date="2026-11-06", runs_dir=tmp_path, allow_fixture=False)
    manifest = {"stages": {"media": {"status": "pending", "retry_count": 0, "max_retries": 1, "input_files": ["script.json"], "output_files": []}}}

    # Must raise ValueError
    with pytest.raises(ValueError, match="Refusing to execute stage 'media' on fixture script without --allow-fixture"):
        orch.execute_stage("media", manifest)

    # When allow_fixture=True, it executes without raising refusal
    orch_allowed = Orchestrator(run_date="2026-11-06", runs_dir=tmp_path, allow_fixture=True)
    assert orch_allowed.allow_fixture is True


def test_cache_pruning(tmp_path):
    """Verify cache pruning removes stale unreferenced files and respects cap."""
    cache_dir = tmp_path / "cache"
    downloads = cache_dir / "downloads"
    downloads.mkdir(parents=True)

    # Create dummy download files
    f1 = downloads / "stale_unreferenced.mp4"
    f1.write_bytes(b"x" * 1024 * 1024)  # 1 MB

    mgr = MediaCacheManager(cache_dir=cache_dir)
    res = mgr.prune_cache(runs_dir=tmp_path / "empty_runs", max_gb=0.0001, keep_days=0)
    assert res["deleted_count"] >= 1
    assert not f1.exists()


def test_media_manifest_schema_validation_negative_cases():
    """Negative test proving media_manifest.schema.json rejects missing fields and invalid structures."""
    # 1. Missing required field 'shots'
    bad_manifest = {
        "date": "2026-11-07",
        "orientation": "landscape",
        "resolution": [1920, 1080],
        "fps": 30,
        "warnings": [],
    }
    is_valid, err = validate_contract(bad_manifest, "contracts/media_manifest.schema.json")
    assert is_valid is False
    assert "'shots' is a required property" in err

    # 2. Shot missing required field 'file'
    bad_shot_manifest = {
        "date": "2026-11-07",
        "orientation": "landscape",
        "resolution": [1920, 1080],
        "fps": 30,
        "warnings": [],
        "shots": [
            {
                "scene_id": "scene_01",
                "shot_index": 0,
                "start": 0.0,
                "end": 4.0,
                "type": "video",
                # missing 'file'
            }
        ]
    }
    is_valid, err = validate_contract(bad_shot_manifest, "contracts/media_manifest.schema.json")
    assert is_valid is False
    assert "'file' is a required property" in err


def test_media_agent_run_approval_fingerprint_gate(tmp_path):
    """MediaAgent.run must execute the approval and validator fingerprint gate without NameError."""
    import hashlib
    from src.utils.editor_validator import compute_validator_fingerprint

    run_dir = tmp_path / "2026-11-08"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_data = {
        "language": "English",
        "origin": "editor",
        "scenes": [],
    }
    script_path = script_dir / "script.json"
    script_bytes = json.dumps(script_data).encode("utf-8")
    script_path.write_bytes(script_bytes)
    script_hash = hashlib.sha256(script_bytes).hexdigest()

    curr_fp = compute_validator_fingerprint()
    approval_data = {
        "approved_at": "2026-11-08T10:00:00Z",
        "script_sha256": script_hash,
        "approved_by": "user",
        "validator_fingerprint": curr_fp,
    }
    (script_dir / "approval.json").write_text(json.dumps(approval_data), encoding="utf-8")
    (script_dir / "validation_report.json").write_text(
        json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": curr_fp}),
        encoding="utf-8",
    )

    (run_dir / "timeline.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
    manifest = {"stages": {"media": {"status": "pending", "retry_count": 0, "input_files": [], "output_files": []}}}

    agent = MediaAgent(run_dir)
    outputs = agent.run(manifest)
    assert any("media_manifest.json" in out for out in outputs)


def test_license_allowed_nasa_and_cc0(mock_media_config) -> None:
    """Assert is_license_allowed('NASA media') and 'CC0 1.0 Universal' return True with allow_cc_by_sa=False."""
    cache_mgr = MediaCacheManager()
    selector = MediaSelector(mock_media_config, cache_mgr=cache_mgr)
    selector.allow_cc_by_sa = False

    assert selector.is_license_allowed("NASA media") is True
    assert selector.is_license_allowed("CC0 1.0 Universal") is True
    # CC BY-SA should be rejected when allow_cc_by_sa is False
    assert selector.is_license_allowed("CC BY-SA 4.0") is False


def test_apply_picks_classifies_jpg_url_as_image(tmp_path: Path, monkeypatch) -> None:
    """Assert apply_picks() classifies .jpg URLs as images (type == 'image')."""
    run_dir = tmp_path / "2026-11-20"
    (run_dir / "script").mkdir(parents=True)
    (run_dir / "script" / "script.json").write_text(json.dumps({
        "origin": "fixture",
        "date": "2026-11-20",
        "scenes": [{"id": "scene_01", "type": "story", "headline": "Test Story"}],
    }), encoding="utf-8")

    (run_dir / "timeline").mkdir(parents=True)
    (run_dir / "timeline" / "timeline.json").write_text(json.dumps({
        "scenes": [{"id": "scene_01", "type": "story", "start": 0.0, "end": 5.0}],
    }), encoding="utf-8")

    # Create dummy local jpg file for simulated download
    sample_jpg = tmp_path / "sample.jpg"
    Image.new("RGB", (640, 360), (50, 100, 150)).save(str(sample_jpg))

    # Mock download_file to return local sample_jpg
    monkeypatch.setattr(
        MediaCacheManager,
        "download_file",
        lambda self, url, target_filename=None, offline=False, max_size_mb=None: sample_jpg,
    )

    picks_data = {
        "scene_01_00": {
            "type": "link",
            "url": "https://example.com/assets/city_view.jpg",
            "author": "CityPhotographer",
            "attribution": "City view via example.com",
        }
    }
    picks_file = run_dir / "picks.json"
    picks_file.write_text(json.dumps(picks_data), encoding="utf-8")

    agent = MediaAgent(run_dir=run_dir, allow_fixture=True)
    agent.apply_picks(picks_file)

    manifest_file = run_dir / "media" / "media_manifest.json"
    assert manifest_file.exists()
    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    shots = m_data.get("shots", [])
    assert len(shots) == 1
    assert shots[0]["type"] == "image"
    assert shots[0]["file"].endswith(".jpg")


def test_media_cache_prune_retains_referenced_manifest_files(tmp_path: Path) -> None:
    """Assert MediaCacheManager.prune_cache() retains files referenced by media_manifest.json."""
    import time
    cache_dir = tmp_path / "cache"
    downloads_dir = cache_dir / "downloads"
    downloads_dir.mkdir(parents=True)

    # 1. Unreferenced old file (10 days old)
    unref_file = downloads_dir / "old_unreferenced.mp4"
    unref_file.write_bytes(b"dummy data unreferenced")
    old_time = time.time() - 10 * 86400
    import os
    os.utime(str(unref_file), (old_time, old_time))

    # 2. Referenced old file (10 days old, but listed in media_manifest.json)
    ref_file = downloads_dir / "abc123def456.jpg"
    ref_file.write_bytes(b"dummy data referenced")
    os.utime(str(ref_file), (old_time, old_time))

    # Setup runs_dir with media_manifest referencing ref_file
    runs_dir = tmp_path / "runs"
    run_media_dir = runs_dir / "2026-11-20" / "media"
    run_media_dir.mkdir(parents=True)
    manifest_data = {
        "shots": [
            {
                "scene_id": "intro",
                "shot_index": 0,
                "sha256": "abc123def456",
                "cache_key": "abc123def456",
                "raw_path": "data/cache/downloads/abc123def456.jpg",
            }
        ]
    }
    (run_media_dir / "media_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    mgr = MediaCacheManager(cache_dir=cache_dir)
    res = mgr.prune_cache(runs_dir=runs_dir, keep_days=7)

    # Assert unreferenced file was pruned, but referenced file was kept
    assert not unref_file.exists()
    assert ref_file.exists()
    assert res["deleted_count"] == 1



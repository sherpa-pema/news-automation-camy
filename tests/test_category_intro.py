"""Unit tests for category-aware intro generation and persistent voice caching."""
import json
from pathlib import Path
import re
import tempfile

import pytest

from src.agents.editor import EditorAgent
from src.agents.voice import VoiceAgent
from src.utils.category_intro import (
    DEFAULT_CATEGORY_INTROS,
    get_canonical_category_keys,
    normalize_selected_categories,
    resolve_category_intro,
)
from src.utils.editor_brief import generate_editor_brief


def test_normalize_selected_categories():
    assert normalize_selected_categories(None) == []
    assert normalize_selected_categories("") == []
    assert normalize_selected_categories(["sports"]) == ["sports"]
    assert normalize_selected_categories("cricket, football") == ["sports"]
    assert normalize_selected_categories("breaking, headlines") == ["headlines"]
    assert normalize_selected_categories(["national_politics"]) == ["politics"]
    assert normalize_selected_categories("economy") == ["business_economics"]
    assert normalize_selected_categories("all") == ["all"]
    assert normalize_selected_categories(["politics", "sports"]) == ["politics", "sports"]


def test_resolve_category_intro_sports():
    res = resolve_category_intro("sports", spoken_date="Saturday, November 21, 2026")
    assert res["category_key"] == "sports"
    assert "Sports" in res["category_label"]
    assert "Sports Roundup" in res["headline"]
    assert "Saturday, November 21, 2026" in res["headline"]

    narr = res["narration"]
    assert "top sports news" in narr.lower()
    assert "match updates" in narr.lower()
    # TTS & budget compliance:
    assert not re.search(r"\d", narr), "Narration must contain 0 digits"
    assert not re.search(r"[%$&/\\()[\]{}]", narr), "Narration must contain no banned symbols"
    wc = len(narr.split())
    assert 10 <= wc <= 25, f"Word count {wc} outside compliant intro range"
    # Date-agnostic: narration must NOT contain calendar date
    assert "2026" not in narr
    assert "november" not in narr.lower()


def test_resolve_category_intro_headlines():
    res = resolve_category_intro("headlines", spoken_date="Saturday, November 21, 2026")
    assert res["category_key"] == "headlines"
    assert "Headlines" in res["category_label"]
    assert "breaking stories" in res["narration"].lower()
    assert not re.search(r"\d", res["narration"])
    assert 10 <= len(res["narration"].split()) <= 25
    assert "2026" not in res["narration"]


def test_all_canonical_intros_comply_with_tts_and_budget():
    keys = get_canonical_category_keys()
    assert "sports" in keys
    assert "headlines" in keys
    assert "politics" in keys
    assert "business_economics" in keys
    assert "health" in keys

    for k in keys + ["default"]:
        res = resolve_category_intro(k, spoken_date="November 21, 2026")
        narr = res["narration"]
        wc = len(narr.split())
        assert 10 <= wc <= 25, f"Category '{k}' word count {wc} outside 10-25 range"
        assert not re.search(r"\d", narr), f"Category '{k}' contains digits"
        assert not re.search(r"[%$&/\\()[\]{}]", narr), f"Category '{k}' contains banned symbols"
        assert 3 <= len(res["visual_keywords"]) <= 5, f"Category '{k}' keywords count wrong"
        # Date agnostic
        assert "2026" not in narr
        assert "november" not in narr.lower()


def test_generate_editor_brief_reflects_category(tmp_path):
    budget = {
        "words_per_second": 2.15,
        "syllables_per_second": 3.3,
        "target_duration_sec": 150.0,
        "target_total_words": 320,
        "min_total_words": 280,
        "max_total_words": 360,
    }
    shortlist = [{"cluster_id": "c1", "primary_title": "Game Won", "articles": []}]

    brief_sports = generate_editor_brief(
        run_date="2026-11-21",
        budget=budget,
        calib_msg="calibrated",
        shortlist=shortlist,
        selected_categories=["sports"],
    )
    assert "Selected Category Focus**: `Sports News`" in brief_sports
    assert "Nepal Sports Roundup" in brief_sports
    assert "match updates" in brief_sports

    brief_headlines = generate_editor_brief(
        run_date="2026-11-21",
        budget=budget,
        calib_msg="calibrated",
        shortlist=shortlist,
        selected_categories=["headlines"],
    )
    assert "Selected Category Focus**: `Headlines & Breaking News`" in brief_headlines
    assert "breaking stories" in brief_headlines


def test_editor_agent_prepare_passes_category(tmp_path):
    run_dir = tmp_path / "2026-11-21"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    articles = [
        {
            "id": "art_1",
            "url": "https://example.com/sports1",
            "title": "Nepal cricket team wins exciting match in Kathmandu",
            "full_text": "Nepal cricket team secured a major victory today in Kathmandu stadium before thousands of cheering fans.",
            "text_summary": "Nepal cricket victory.",
            "category_hint": "sports",
            "word_count": 80,
        }
    ]
    (raw_dir / "articles.json").write_text(json.dumps(articles), encoding="utf-8")

    manifest = {
        "run_id": "test_sports",
        "date": "2026-11-21",
        "selected_categories": ["sports"],
        "stages": {"editor": {"status": "pending"}},
    }

    agent = EditorAgent(run_dir=run_dir)
    agent.prepare(manifest)

    brief_path = run_dir / "script" / "editor_brief.md"
    assert brief_path.exists()
    content = brief_path.read_text(encoding="utf-8")
    assert "Sports News" in content
    assert "Nepal Sports Roundup" in content


def test_voice_persistent_intro_cache_and_reuse(tmp_path, monkeypatch):
    # Route persistent cache to a temporary test directory
    test_cache_dir = tmp_path / "cache_voice"
    monkeypatch.setattr("src.agents.voice.Path", lambda *args: test_cache_dir if len(args) == 1 and args[0] == "data/cache/voice" else Path(*args))

    intro_data = resolve_category_intro("sports")
    script_obj = {
        "language": "English",
        "origin": "fixture",
        "date": "2026-11-21",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": intro_data["headline"],
                "narration": intro_data["narration"],
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": intro_data["visual_keywords"],
                "evidence": [],
            }
        ],
    }

    # Run 1: First day (2026-11-21) -> Should synthesize and write to persistent cache
    run_dir_1 = tmp_path / "run_day1"
    (run_dir_1 / "script").mkdir(parents=True)
    (run_dir_1 / "script" / "script.json").write_text(json.dumps(script_obj), encoding="utf-8")
    manifest_1 = {"run_id": "run_1", "date": "2026-11-21", "stages": {"voice": {"status": "pending"}}}

    agent_1 = VoiceAgent(run_dir=run_dir_1)
    agent_1.run(manifest_1, dry_run=True)

    wav_1 = run_dir_1 / "audio" / "scenes" / "intro.wav"
    assert wav_1.exists()
    wav_1_bytes = wav_1.read_bytes()

    # Verify persistent cache file was created with semantic naming
    cached_wav = test_cache_dir / "intro_sports.wav"
    cached_meta = test_cache_dir / "intro_sports.meta.json"
    assert cached_wav.exists(), f"Expected semantic cache file {cached_wav.name} to exist"
    assert cached_meta.exists(), f"Expected semantic meta file {cached_meta.name} to exist"
    assert cached_wav.read_bytes() == wav_1_bytes

    # Run 2: Second day (2026-11-22) -> Should reuse from persistent cache without re-synthesis
    run_dir_2 = tmp_path / "run_day2"
    (run_dir_2 / "script").mkdir(parents=True)
    (run_dir_2 / "script" / "script.json").write_text(json.dumps(script_obj), encoding="utf-8")
    manifest_2 = {"run_id": "run_2", "date": "2026-11-22", "stages": {"voice": {"status": "pending"}}}

    agent_2 = VoiceAgent(run_dir=run_dir_2)
    agent_2.run(manifest_2, dry_run=True)

    wav_2 = run_dir_2 / "audio" / "scenes" / "intro.wav"
    assert wav_2.exists()
    assert wav_2.read_bytes() == wav_1_bytes, "Audio bytes from day 2 must be identical to cached day 1"


def test_identify_category_from_intro_and_cache_filenames():
    from src.utils.category_intro import (
        DEFAULT_CATEGORY_INTROS,
        get_intro_cache_filenames,
        identify_category_from_intro,
        resolve_category_intro,
    )

    for cat in DEFAULT_CATEGORY_INTROS:
        data = resolve_category_intro(cat)
        # From scene dictionary
        scene = {"id": "intro", "type": "intro", "narration": data["narration"], "headline": data["headline"]}
        assert identify_category_from_intro(scene) == cat

        # From plain narration text
        assert identify_category_from_intro(data["narration"]) == cat

        # From manifest selected_categories
        manifest = {"selected_categories": [cat]}
        assert identify_category_from_intro(None, manifest=manifest) == cat

        # Filename helper
        wav_name, meta_name = get_intro_cache_filenames(cat)
        assert wav_name == f"intro_{cat}.wav"
        assert meta_name == f"intro_{cat}.meta.json"


def test_intro_greeting_fails_validation(tmp_path):
    from src.utils.editor_validator import validate_script_draft

    draft_file = tmp_path / "draft_script.json"
    shortlist_file = tmp_path / "shortlist.json"
    shortlist_file.write_text("[]", encoding="utf-8")

    draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-21",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Nepal Health Briefing for Saturday",
                "narration": "Good evening and welcome to our broadcast. Critical healthcare updates from across the nation.",
                "visual_keywords": ["hospital building", "news studio", "medical clinic"],
                "source": "Nepal News Desk",
                "source_urls": [],
                "evidence": [],
            }
        ],
    }
    draft_file.write_text(json.dumps(draft), encoding="utf-8")

    budget = {
        "words_per_second": 2.15,
        "syllables_per_second": 3.3,
        "min_total_words": 10,
        "max_total_words": 50,
        "target_total_words": 20,
    }

    is_valid, errors, warnings, report = validate_script_draft(draft_file, shortlist_file, budget)
    assert not is_valid
    assert any("passive greetings" in e.lower() for e in errors), f"Expected greeting error, got {errors}"


def test_single_sentence_intro_passes_7_scene_bulletin(tmp_path):
    from src.utils.editor_validator import validate_script_draft

    draft_file = tmp_path / "draft_script.json"
    shortlist_file = tmp_path / "shortlist.json"

    health_intro = resolve_category_intro("health")
    # 7 scenes: intro + 5 stories + outro
    scenes = [
        {
            "id": "intro",
            "type": "intro",
            "headline": health_intro["headline"],
            "narration": health_intro["narration"],  # 12 words
            "visual_keywords": health_intro["visual_keywords"],
            "source": "Nepal News Desk",
            "source_urls": [],
            "evidence": [],
        }
    ]
    clusters = []
    for i in range(1, 6):
        c_id = f"c_{i}"
        art_url = f"https://example.com/health_{i}"
        clusters.append({
            "cluster_id": c_id,
            "primary_title": f"Health Story {i}",
            "articles": [{"url": art_url, "title": f"Health Story {i}", "full_text": "Hospital facility update in Kathmandu with doctors and nurses."}],
        })
        scenes.append({
            "id": f"scene_{i:02d}",
            "type": "story",
            "headline": f"Health development story number {i}",
            "narration": f"Medical specialists in Kathmandu opened another healthcare facility today to treat rural patients and expand service delivery. Officials said twenty health workers will staff the center daily.",
            "category": "health",
            "visual_keywords": ["medical clinic", "hospital ward", "doctors working"],
            "source": "The Himalayan Times",
            "source_urls": [art_url],
            "evidence": [{"sentence_index": 0, "source_url": art_url}, {"sentence_index": 1, "source_url": art_url}],
        })
    scenes.append({
        "id": "outro",
        "type": "outro",
        "headline": "That concludes our daily health update",
        "narration": "That concludes our daily health update. Stay tuned for more medical news and health developments tomorrow.",
        "visual_keywords": ["Kathmandu valley", "news studio", "broadcast desk"],
        "source": "Nepal News Desk",
        "source_urls": [],
        "evidence": [],
    })

    draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-21",
        "scenes": scenes,
    }
    draft_file.write_text(json.dumps(draft), encoding="utf-8")
    shortlist_file.write_text(json.dumps(clusters), encoding="utf-8")

    budget = {
        "words_per_second": 2.15,
        "syllables_per_second": 3.3,
        "min_total_words": 150,
        "max_total_words": 350,
        "target_total_words": 250,
    }

    is_valid, errors, warnings, report = validate_script_draft(draft_file, shortlist_file, budget)
    # Check that intro scene itself had 0 errors
    intro_errors = [e for e in errors if "intro" in e.lower() or "word count 12" in e.lower()]
    assert len(intro_errors) == 0, f"Expected 0 intro errors for 12-word health intro, got: {intro_errors}"


def test_llm_writer_enforce_canonical_intro_scene():
    from src.utils.llm_writer import _enforce_canonical_intro_scene

    intro_info = resolve_category_intro("health")
    corrupted_draft = {
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Wrong headline",
                "narration": "Good evening and welcome to our broadcast. Critical healthcare updates, medical breakthroughs, and wellness news from across the nation.",
                "visual_keywords": ["random keyword"],
                "category": "health",
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Story 1",
                "narration": "Story narration here.",
            },
        ]
    }

    _enforce_canonical_intro_scene(corrupted_draft, intro_info)

    locked_intro = corrupted_draft["scenes"][0]
    assert locked_intro["narration"] == intro_info["narration"]
    assert "Good evening" not in locked_intro["narration"]
    assert "welcome" not in locked_intro["narration"].lower()
    assert locked_intro["headline"] == intro_info["headline"]
    assert locked_intro["visual_keywords"] == intro_info["visual_keywords"]
    assert locked_intro["category"] is None

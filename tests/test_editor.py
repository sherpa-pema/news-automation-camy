"""Unit and integration tests for Editor Agent and deterministic validation."""
import hashlib
import json
from pathlib import Path
import pytest

from src.agents.editor import EditorAgent
from src.orchestrator import Orchestrator
from src.utils.editor_budget import compute_word_budget
from src.utils.editor_cluster import cluster_articles, is_opinion_or_ad, is_thin_story
from src.utils.editor_validator import (
    extract_numbers_from_text,
    validate_script_draft,
)


@pytest.fixture
def mock_editor_config():
    return {
        "clustering": {
            "min_full_text_words": 60,
            "title_similarity_threshold": 0.55,
            "token_overlap_threshold": 0.35,
            "top_clusters_limit": 12,
        },
        "budget": {
            "min_duration_sec": 120.0,
            "max_duration_sec": 180.0,
            "target_duration_sec": 150.0,
            "default_words_per_sec": 2.15,
            "tolerance_pct": 0.10,
        },
        "sensitive_triggers": ["death", "killed", "arrested", "accident", "crash"],
        "categories": {
            "economy": ["bank", "inflation", "rupee", "tax"],
            "politics": ["parliament", "minister", "election"],
            "environment": ["river", "cleanup", "pollution"],
        },
    }


def test_clustering_keeps_all_outlets_and_drops_opinion_and_thin(mock_editor_config):
    """Verify that clustering drops opinion and thin stories, and preserves all outlets for duplicates."""
    articles = [
        # Thin story (< 60 words)
        {
            "source": "Outlet A",
            "title": "Short Alert",
            "url": "https://a.com/short",
            "text_summary": "Too short.",
            "full_text": "Only ten words here.",
            "published_at": "2026-11-07T10:00:00Z",
        },
        # Opinion piece
        {
            "source": "Outlet B",
            "title": "Opinion: Future of Nepal Politics",
            "url": "https://b.com/opinion/future",
            "text_summary": "My personal view on Nepal politics.",
            "full_text": "Word " * 70,
            "published_at": "2026-11-07T10:00:00Z",
        },
        # Story 1 from Outlet C
        {
            "source": "Outlet C",
            "title": "Bagmati River Cleanup Drive Yields Clean Water",
            "url": "https://c.com/cleanup",
            "text_summary": "Major cleanup in Kathmandu along Bagmati corridor.",
            "full_text": "In Kathmandu, environmental volunteers gathered to clean the Bagmati river. " * 10,
            "published_at": "2026-11-07T11:00:00Z",
        },
        # Story 1 duplicate from Outlet D
        {
            "source": "Outlet D",
            "title": "Bagmati River Cleanup Drive Shows Great Success",
            "url": "https://d.com/cleanup",
            "text_summary": "Major cleanup in Kathmandu along Bagmati corridor reported.",
            "full_text": "In Kathmandu, environmental volunteers gathered to clean the Bagmati river. " * 10,
            "published_at": "2026-11-07T11:30:00Z",
        },
    ]

    assert is_thin_story(articles[0], min_words=60) is True
    assert is_opinion_or_ad(articles[1]) is True

    clusters = cluster_articles(articles, mock_editor_config)
    assert len(clusters) == 1
    c = clusters[0]
    assert set(c["sources"]) == {"Outlet C", "Outlet D"}
    assert len(c["urls"]) == 2
    assert c["category_guess"] == "environment"
    assert c["is_opinion"] is False


def test_detect_opinion_heuristics():
    """Verify opinion detection across URL, title, author, and first-person pronouns."""
    from src.utils.editor_cluster import detect_opinion

    # 1. URL pattern
    is_op, reason = detect_opinion({"url": "https://example.com/opinion/reform-now", "title": "News", "full_text": "Story text"})
    assert is_op is True
    assert "URL pattern" in reason

    # 2. Title pattern
    is_op, reason = detect_opinion({"url": "https://example.com/news/123", "title": "Editorial: The Path Forward", "full_text": "Story text"})
    assert is_op is True
    assert "Title pattern" in reason

    # 3. Author pattern
    is_op, reason = detect_opinion({"url": "https://example.com/news/123", "title": "Economic Note", "author": "Guest Columnist", "full_text": "Story text"})
    assert is_op is True
    assert "Author designation" in reason

    # 4. First-person pronouns (>= 5 occurrences, excluding uppercase 'US')
    first_person_text = (
        "In our view, we must recognize what the data tells us. I have traveled across Nepal, "
        "and our team observed these patterns firsthand while the US delegation held meetings."
    )
    is_op, reason = detect_opinion({"url": "https://example.com/news/123", "title": "Economic Note", "full_text": first_person_text})
    assert is_op is True
    assert "First-person pronoun count" in reason

    # 5. Objective reporting (< 5 first-person pronouns)
    objective_text = (
        "Officials announced new infrastructure guidelines today. The ministry spokesperson confirmed "
        "that thirty engineers completed safety audits across five districts."
    )
    is_op, reason = detect_opinion({"url": "https://example.com/news/123", "title": "Infrastructure Update", "full_text": objective_text})
    assert is_op is False
    assert reason == ""


def test_opinion_cluster_exclusion_and_brief_section(tmp_path, mock_editor_config):
    """Verify opinion clusters are excluded from shortlist and listed in brief."""
    from src.utils.editor_brief import generate_editor_brief

    articles = [
        # News story
        {
            "source": "News Source",
            "title": "Hospital Wing Inaugurated in Kailali",
            "url": "https://example.com/hospital",
            "text_summary": "Fifty beds added to regional hospital in Kailali.",
            "full_text": "Health officials inaugurated fifty new hospital beds in Kailali today. " * 8,
            "published_at": "2026-11-07T10:00:00Z",
        },
        # Opinion piece
        {
            "source": "Opinion Desk",
            "title": "Why Our Healthcare System Is Failing Us",
            "url": "https://example.com/opinion/healthcare",
            "text_summary": "My perspective on hospitals.",
            "full_text": "I believe we must fix our medical schools. We can see our mistakes and fix them ourselves. " * 5,
            "published_at": "2026-11-07T11:00:00Z",
        },
    ]

    shortlist, opinion_clusters = cluster_articles(articles, mock_editor_config, return_opinion=True)
    assert len(shortlist) == 1
    assert shortlist[0]["is_opinion"] is False
    assert shortlist[0]["primary_title"] == "Hospital Wing Inaugurated in Kailali"

    assert len(opinion_clusters) == 1
    assert opinion_clusters[0]["is_opinion"] is True

    budget = compute_word_budget(2.15)
    brief = generate_editor_brief("2026-11-07", budget, "Calibrated", shortlist, opinion_clusters)
    assert "## 5. Excluded Opinion & Commentary Pieces" in brief
    assert "Why Our Healthcare System Is Failing Us" in brief


def test_prepare_step_writes_shortlist_and_brief_with_budget(tmp_path, mock_editor_config):
    """Verify prepare creates shortlist.json and editor_brief.md containing budget."""
    run_dir = tmp_path / "2026-11-07"
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)

    articles = [
        {
            "source": "Outlet 1",
            "title": "Major Hydro Project Online",
            "url": "https://example.com/hydro",
            "text_summary": "Fifty megawatts synchronized into national grid.",
            "full_text": "Engineers synchronized fifty megawatts into national power grid today. " * 6,
            "published_at": "2026-11-07T10:00:00Z",
        }
    ]
    (raw_dir / "articles.json").write_text(json.dumps(articles), encoding="utf-8")

    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "running",
        "stages": {"editor": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": []}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    agent = EditorAgent(run_dir)
    agent.prepare(manifest)

    shortlist_p = run_dir / "script" / "shortlist.json"
    brief_p = run_dir / "script" / "editor_brief.md"

    assert shortlist_p.exists()
    assert brief_p.exists()

    brief_content = brief_p.read_text(encoding="utf-8")
    assert "Word Budget" in brief_content
    assert "words per second" in brief_content


def test_number_normalization_and_invented_figure():
    """Verify '10' matches 'ten', 'Rs 50,000' matches 'fifty thousand rupees', and invented figure fails."""
    # Text with digits and currency
    article_text = "The government distributed Rs 50,000 and 10 vehicles to the rural district."
    nums_in_article = extract_numbers_from_text(article_text)
    assert 50000.0 in nums_in_article
    assert 10.0 in nums_in_article

    # Narration with spelled-out equivalents
    narr_grounded = "Officials granted ten vehicles and fifty thousand rupees to local workers."
    nums_in_narr = extract_numbers_from_text(narr_grounded)
    assert 10.0 in nums_in_narr
    assert 50000.0 in nums_in_narr

    # Grounded numbers match
    for num in nums_in_narr:
        assert num in nums_in_article

    # Invented number
    narr_invented = "Officials granted twenty vehicles and fifty thousand rupees to local workers."
    nums_invented = extract_numbers_from_text(narr_invented)
    assert 20.0 in nums_invented
    assert 20.0 not in nums_in_article


def test_currency_prefix_and_spoken_decimals():
    """Verify 'Rs105.92' is extracted and matches 'one hundred five point nine two'."""
    article_text = "The retail cost of honey was established at Rs105.92 per container."
    nums_in_article = extract_numbers_from_text(article_text)
    assert 105.92 in nums_in_article

    narr_text = "Officials priced each container at one hundred five point nine two rupees."
    nums_in_narr = extract_numbers_from_text(narr_text)
    assert 105.92 in nums_in_narr


def test_number_tolerance_and_approximation_modifier(tmp_path):
    """Verify number within 2% is a warning with approximation modifier and an error without."""
    cluster = [{
        "cluster_id": "cluster_01",
        "urls": ["https://example.com/story"],
        "articles": [{
            "url": "https://example.com/story",
            "title": "Title",
            "full_text": "The department deployed 100 inspectors across Kathmandu Valley to monitor public markets."
        }]
    }]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    budget = compute_word_budget(2.15)

    base_draft = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Briefing", "narration": "Welcome to today's news briefing from Kathmandu Valley and the provinces.", "source": "Desk", "source_urls": [], "visual_keywords": ["Kathmandu", "mountains", "studio"], "sensitive": False, "evidence": []},
            {"id": "scene_01", "type": "story", "headline": "Market Monitoring", "narration": "", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["market", "inspectors", "Kathmandu city"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_02", "type": "story", "headline": "Story Two", "narration": "Here is another update from western districts. Officials completed routine inspection work.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["road", "highway", "transit"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_03", "type": "story", "headline": "Story Three", "narration": "Tourism sector operators noted steady autumn customer traffic. Many lodges report good bookings.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["hotel", "mountains", "trekking"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_04", "type": "story", "headline": "Story Four", "narration": "Commerce representatives finalized trade protocols this morning. Digital customs passes accelerated transit.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["customs", "port", "cargo"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_05", "type": "story", "headline": "Story Five", "narration": "Energy regulators approved fresh transmission power capacity. Clean power supplies continue expanding.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["dam", "power", "grid"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "outro", "type": "outro", "headline": "Thank You", "narration": "That concludes our daily update. Stay safe and have a good evening.", "source": "Desk", "source_urls": [], "visual_keywords": ["skyline", "city", "sunset"], "sensitive": False, "evidence": []},
        ]
    }

    draft_path = tmp_path / "draft.json"

    # 1. Within 2% WITH 'about' modifier -> WARN and no number errors
    base_draft["scenes"][1]["narration"] = (
        "In Kathmandu, authorities deployed about one hundred one inspectors today. "
        "The teams visited grocery vendors across multiple municipal wards."
    )
    draft_path.write_text(json.dumps(base_draft), encoding="utf-8")
    is_valid, errors, warnings, _ = validate_script_draft(draft_path, shortlist_path, budget)
    assert not any("number" in e.lower() or "invented" in e.lower() for e in errors)
    assert any("Approximated number 101.0 is within 2%" in w for w in warnings)

    # 2. Within 2% WITHOUT modifier -> FAIL with specific modifier error
    base_draft["scenes"][1]["narration"] = (
        "In Kathmandu, authorities deployed one hundred one inspectors today. "
        "The teams visited grocery vendors across multiple municipal wards."
    )
    draft_path.write_text(json.dumps(base_draft), encoding="utf-8")
    is_valid, errors, warnings, _ = validate_script_draft(draft_path, shortlist_path, budget)
    assert any("Number 101.0 is within 2% of source number 100.0 but lacks approximation modifier" in e for e in errors)

    # 3. Beyond 2% (e.g. 105, which is 5% away) -> FAIL with invented number error
    base_draft["scenes"][1]["narration"] = (
        "In Kathmandu, authorities deployed about one hundred five inspectors today. "
        "The teams visited grocery vendors across multiple municipal wards."
    )
    draft_path.write_text(json.dumps(base_draft), encoding="utf-8")
    is_valid, errors, warnings, _ = validate_script_draft(draft_path, shortlist_path, budget)
    assert any("Invented number 105.0 in narration does not appear in source cluster" in e for e in errors)


def test_verbatim_check_catches_8_word_copy(tmp_path):
    """Verify verbatim checker catches an 8-word verbatim excerpt and passes clean rewrite."""
    source_sentence = "the environmental authorities confirmed that advanced biological treatment facilities are now operational today"
    cluster = [{
        "cluster_id": "cluster_01",
        "urls": ["https://example.com/story"],
        "articles": [{"url": "https://example.com/story", "title": "Title", "full_text": source_sentence}]
    }]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")

    budget = compute_word_budget(2.15)

    # 1. Draft with exact 8-word copy
    bad_draft = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Briefing", "narration": "Welcome to today's news briefing from Kathmandu Valley and the provinces.", "source": "Desk", "source_urls": [], "visual_keywords": ["Kathmandu", "mountains", "studio"], "sensitive": False, "evidence": []},
            {"id": "scene_01", "type": "story", "headline": "River Treatment", "narration": "In Kathmandu, the environmental authorities confirmed that advanced biological treatment facilities are now operational today across multiple areas.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["river", "cleanup", "Kathmandu city"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_02", "type": "story", "headline": "Story Two", "narration": "Here is another update from western districts. Officials completed routine inspection work.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["road", "highway", "transit"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_03", "type": "story", "headline": "Story Three", "narration": "Tourism sector operators noted steady autumn customer traffic. Many lodges report good bookings.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["hotel", "mountains", "trekking"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_04", "type": "story", "headline": "Story Four", "narration": "Commerce representatives finalized trade protocols this morning. Digital customs passes accelerated transit.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["customs", "port", "cargo"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "scene_05", "type": "story", "headline": "Story Five", "narration": "Energy regulators approved fresh transmission power capacity. Clean power supplies continue expanding.", "source": "Outlet", "source_urls": ["https://example.com/story"], "visual_keywords": ["dam", "power", "grid"], "sensitive": False, "evidence": [{"sentence_index": 0, "source_url": "https://example.com/story"}]},
            {"id": "outro", "type": "outro", "headline": "Thank You", "narration": "That concludes our daily update. Stay safe and have a good evening.", "source": "Desk", "source_urls": [], "visual_keywords": ["skyline", "city", "sunset"], "sensitive": False, "evidence": []},
        ]
    }
    draft_p = tmp_path / "draft_script.json"
    draft_p.write_text(json.dumps(bad_draft), encoding="utf-8")

    is_valid, errors, warnings, report = validate_script_draft(draft_p, shortlist_path, budget)
    assert is_valid is False
    assert any("Verbatim 8-word overlap" in err for err in errors)


def test_tts_rule_validator_rejects_digits_symbols_and_parens(tmp_path):
    """Verify TTS rules reject digits, symbols (%, $), and parentheses."""
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps([]), encoding="utf-8")
    budget = compute_word_budget(2.15)

    draft = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Testing TTS Violations",
                "narration": "Inflation reached 10% (according to officials) and cost $50 per citizen.",
                "source": "Outlet",
                "source_urls": [],
                "visual_keywords": ["bank", "money", "market"],
                "sensitive": False,
                "evidence": [{"sentence_index": 0, "source_url": "https://example.com"}],
            }
        ]
    }
    draft_p = tmp_path / "draft.json"
    draft_p.write_text(json.dumps(draft), encoding="utf-8")

    is_valid, errors, warnings, report = validate_script_draft(draft_p, shortlist_path, budget)
    assert is_valid is False
    err_str = " ".join(errors)
    assert "Digits found" in err_str
    assert "Banned symbols" in err_str
    assert "Parentheses" in err_str


def test_sensitive_trigger_check_fails_if_sensitive_false(tmp_path):
    """Verify that a cluster containing sensitive words fails if scene.sensitive is False."""
    cluster = [{
        "cluster_id": "cluster_01",
        "urls": ["https://example.com/crash"],
        "articles": [{"url": "https://example.com/crash", "title": "Bus Crash", "full_text": "A passenger bus had a fatal accident and three people were killed in the crash."}]
    }]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    budget = compute_word_budget(2.15)

    draft = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Transit Mishap in Hills",
                "narration": "Authorities attended a highway incident today. Emergency teams secured the location.",
                "source": "Outlet",
                "source_urls": ["https://example.com/crash"],
                "visual_keywords": ["highway", "hills", "road"],
                "sensitive": False,  # Should be True due to accident/killed in source!
                "evidence": [{"sentence_index": 0, "source_url": "https://example.com/crash"}],
            }
        ]
    }
    draft_p = tmp_path / "draft.json"
    draft_p.write_text(json.dumps(draft), encoding="utf-8")

    is_valid, errors, warnings, report = validate_script_draft(draft_p, shortlist_path, budget)
    assert is_valid is False
    assert any("sensitive trigger word" in err for err in errors)


def test_pause_states_exit_with_code_3_and_block_downstream(tmp_path):
    """Verify pause states return exit code 3 and block downstream stages."""
    run_dir = tmp_path / "2026-11-07"
    run_dir.mkdir(parents=True)
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "raw" / "articles.json").write_text("[]", encoding="utf-8")
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)
    (script_dir / "script.json").write_text(json.dumps({"origin": "editor", "language": "English", "scenes": []}), encoding="utf-8")

    orch = Orchestrator(run_date="2026-11-07", runs_dir=tmp_path)
    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "pending",
        "stages": {
            "editor": {"status": "prepared", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": []},
            "voice": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["script/script.json"], "output_files": []},
        }
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Downstream stage 'voice' must be blocked because editor is not 'done'
    with pytest.raises(ValueError, match="Editor stage status is 'prepared', but must be 'done'"):
        orch.execute_stage("voice", manifest)


def test_hand_editing_script_detected_after_approval(tmp_path):
    """Verify script.json SHA-256 mismatch against approval.json blocks downstream execution."""
    run_dir = tmp_path / "2026-11-07"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)

    script_file = script_dir / "script.json"
    script_file.write_text(json.dumps({"origin": "editor", "language": "English", "scenes": []}), encoding="utf-8")
    original_hash = hashlib.sha256(script_file.read_bytes()).hexdigest()

    approval_file = script_dir / "approval.json"
    approval_file.write_text(json.dumps({"approved_at": "2026-11-07T12:00:00Z", "script_sha256": original_hash, "approved_by": "user"}), encoding="utf-8")

    orch = Orchestrator(run_date="2026-11-07", runs_dir=tmp_path)
    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "running",
        "stages": {
            "editor": {"status": "done", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": ["script/script.json"]},
            "voice": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["script/script.json"], "output_files": []},
        }
    }

    # Now tamper with script.json
    script_file.write_text(json.dumps({"origin": "editor", "language": "English", "scenes": [{"tampered": True}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="script changed after approval, re-run validate and approve"):
        orch.execute_stage("voice", manifest)


def test_legacy_script_path_rejected_and_script_subfolder_required(tmp_path):
    """Verify that downstream agents (Voice, Timeline, Media) reject a run folder that lacks script/script.json even if script.json exists at root."""
    run_dir = tmp_path / "2026-11-07"
    run_dir.mkdir(parents=True)
    # Place script.json in root (old legacy path)
    legacy_script = run_dir / "script.json"
    legacy_script.write_text(json.dumps({"origin": "editor", "language": "English", "scenes": []}), encoding="utf-8")

    from src.agents.voice import VoiceAgent
    from src.agents.timeline import TimelineAgent
    from src.agents.media import MediaAgent

    # 1. VoiceAgent must reject
    voice_agent = VoiceAgent(run_dir)
    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "running",
        "stages": {"voice": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["script/script.json"], "output_files": []}}
    }
    with pytest.raises(FileNotFoundError, match="Missing required script file"):
        voice_agent.run(manifest)

    # 2. TimelineAgent must reject
    timeline_agent = TimelineAgent(run_dir)
    with pytest.raises(FileNotFoundError, match="Missing required script file"):
        timeline_agent.run(manifest)

    # 3. MediaAgent must reject
    media_agent = MediaAgent(run_dir)
    with pytest.raises(FileNotFoundError, match="Missing required script.json"):
        media_agent.run(manifest)


def test_schema_validation_rejects_missing_field_and_wrong_type(tmp_path):
    """Negative test proving draft schema validation rejects missing fields and wrong types."""
    from src.utils.schema_validate import validate_contract

    # 1. Missing mandatory field ('language')
    draft_missing = {
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Missing Language Test",
                "narration": "Hello and welcome.",
                "source": "Nepal Daily Briefing",
                "source_urls": [],
                "visual_keywords": ["studio"],
                "sensitive": False,
                "evidence": [],
            }
        ]
    }
    is_valid, err = validate_contract(draft_missing, "contracts/script.schema.json")
    assert is_valid is False
    assert "'language' is a required property" in err

    # 2. Wrong type ('scenes' should be an array, passed as string)
    draft_wrong_type = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": "not_an_array"
    }
    is_valid, err = validate_contract(draft_wrong_type, "contracts/script.schema.json")
    assert is_valid is False
    assert "'not_an_array' is not of type 'array'" in err

    # 3. Wrong type in scene property ('sensitive' should be bool, passed as int/string)
    draft_wrong_scene_type = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Wrong Type Test",
                "narration": "Testing wrong property types.",
                "source": "Outlet",
                "source_urls": [],
                "visual_keywords": ["test"],
                "sensitive": "not_a_boolean",
                "evidence": [],
            }
        ]
    }
    is_valid, err = validate_contract(draft_wrong_scene_type, "contracts/script.schema.json")
    assert is_valid is False
    assert "'not_a_boolean' is not of type 'boolean'" in err


def test_manifest_schema_validation_negative_cases():
    """Negative test proving manifest.schema.json rejects missing fields and invalid types."""
    from src.utils.schema_validate import validate_contract

    # 1. Missing required field 'run_id'
    bad_manifest = {
        "date": "2026-11-07",
        "status": "pending",
        "stages": {},
    }
    is_valid, err = validate_contract(bad_manifest, "contracts/manifest.schema.json")
    assert is_valid is False
    assert "'run_id' is a required property" in err

    # 2. Invalid status enum
    bad_status_manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "created_at": "2026-11-07T12:00:00Z",
        "updated_at": "2026-11-07T12:00:00Z",
        "status": "invalid_state",
        "stages": {},
    }
    is_valid, err = validate_contract(bad_status_manifest, "contracts/manifest.schema.json")
    assert is_valid is False
    assert "'invalid_state' is not one of" in err


def test_evidence_from_unshortlisted_cluster_fails_validation(tmp_path):
    """Validator must strictly fail any evidence whose source_url is not in the shortlist."""
    budget = compute_word_budget(2.15)
    shortlist_path = tmp_path / "shortlist.json"
    draft_path = tmp_path / "draft_script.json"

    shortlist_data = [
        {
            "cluster_id": "cluster_01",
            "primary_title": "Shortlisted Story",
            "urls": ["https://example.com/shortlisted"],
            "articles": [
                {
                    "title": "Shortlisted Story",
                    "url": "https://example.com/shortlisted",
                    "full_text": "This is a full article text with fifty words about education reform in Kathmandu. " * 5,
                }
            ],
        }
    ]
    shortlist_path.write_text(json.dumps(shortlist_data), encoding="utf-8")

    # Draft has evidence pointing to unshortlisted URL
    draft_data = {
        "language": "English",
        "origin": "editor",
        "date": "2026-11-07",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Intro Scene",
                "narration": "Hello and welcome to the daily news bulletin for today. Here are the latest updates from across Nepal.",
                "source": "Nepal Desk",
                "source_urls": [],
                "visual_keywords": ["kathmandu", "studio", "title"],
                "sensitive": False,
                "evidence": [],
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Valid Story With Bad Evidence",
                "narration": "Education officials announced ten new scholarships in Kathmandu today. Community schools welcomed the reform initiative enthusiastically.",
                "source": "Outlet",
                "source_urls": ["https://example.com/shortlisted"],
                "visual_keywords": ["school", "kathmandu", "students"],
                "sensitive": False,
                "evidence": [
                    {
                        "sentence_index": 0,
                        "source_url": "https://example.com/unshortlisted-article",
                    },
                    {
                        "sentence_index": 1,
                        "source_url": "https://example.com/shortlisted",
                    },
                ],
            },
            {
                "id": "scene_02",
                "type": "story",
                "headline": "Unmatched Source URLs",
                "narration": "Another story about city development projects across the valley. Workers completed road paving ahead of schedule.",
                "source": "Outlet",
                "source_urls": ["https://example.com/other-unshortlisted"],
                "visual_keywords": ["road", "kathmandu", "workers"],
                "sensitive": False,
                "evidence": [
                    {
                        "sentence_index": 0,
                        "source_url": "https://example.com/other-unshortlisted",
                    }
                ],
            },
            {
                "id": "scene_03",
                "type": "story",
                "headline": "Dummy Story Three",
                "narration": "A third story about trade and commerce policies announced today. Financial analysts reviewed the quarterly projections carefully.",
                "source": "Outlet",
                "source_urls": ["https://example.com/shortlisted"],
                "visual_keywords": ["market", "finance", "bank"],
                "sensitive": False,
                "evidence": [{"sentence_index": 0, "source_url": "https://example.com/shortlisted"}],
            },
            {
                "id": "scene_04",
                "type": "story",
                "headline": "Dummy Story Four",
                "narration": "A fourth story discussing healthcare improvements in rural districts. Health workers distributed essential medical supplies across the province.",
                "source": "Outlet",
                "source_urls": ["https://example.com/shortlisted"],
                "visual_keywords": ["clinic", "medicine", "doctor"],
                "sensitive": False,
                "evidence": [{"sentence_index": 0, "source_url": "https://example.com/shortlisted"}],
            },
            {
                "id": "scene_05",
                "type": "story",
                "headline": "Dummy Story Five",
                "narration": "A fifth story covering tourism infrastructure developments in the northern region. Hotel operators reported higher bookings for the upcoming trekking season.",
                "source": "Outlet",
                "source_urls": ["https://example.com/shortlisted"],
                "visual_keywords": ["mountains", "hotel", "tourism"],
                "sensitive": False,
                "evidence": [{"sentence_index": 0, "source_url": "https://example.com/shortlisted"}],
            },
            {
                "id": "outro",
                "type": "outro",
                "headline": "Outro Scene",
                "narration": "That concludes our daily news briefing. Thank you for tuning in and stay safe.",
                "source": "Nepal Desk",
                "source_urls": [],
                "visual_keywords": ["kathmandu", "studio", "title"],
                "sensitive": False,
                "evidence": [],
            },
        ],
    }
    draft_path.write_text(json.dumps(draft_data), encoding="utf-8")

    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is False
    # Check that unshortlisted evidence URL is rejected with explicit error message
    assert any("Evidence source_url 'https://example.com/unshortlisted-article' does not belong to any shortlisted cluster" in e for e in errors)
    assert any("Evidence source_url 'https://example.com/other-unshortlisted' does not belong to any shortlisted cluster" in e for e in errors)


def test_detect_stale_preview_heuristics():
    """Verify that detect_stale_preview flags future event previews whose time has passed."""
    from datetime import datetime, timezone, timedelta
    from src.utils.editor_cluster import detect_stale_preview

    ref_time = datetime(2026, 11, 7, 20, 53, tzinfo=timezone(timedelta(hours=5, minutes=45)))

    # 1. Past event preview: 5 pm match on Nov 7
    article_past = {
        "title": "Kathmandu look to break into top three against leaders Lumbini",
        "published_at": "2026-11-07T12:00:00+05:45",
        "full_text": "Post Report\nKathmandu Gorkhas take on Lumbini Lions on Saturday, with the match set to begin at 5pm at TU Ground.",
    }
    is_stale, reason = detect_stale_preview(article_past, ref_time=ref_time)
    assert is_stale is True
    assert "set to" in reason
    assert "5pm" in reason

    # 2. Future event: Dec 10 at 8:15 am
    article_future = {
        "title": "Candidates shortlisted for interview",
        "published_at": "2026-11-07T12:00:00+05:45",
        "full_text": "Candidates will appear at the board's secretariat at 8:15 am on December 10 with original documents.",
    }
    is_stale, reason = detect_stale_preview(article_future, ref_time=ref_time)
    assert is_stale is False

    # 3. Regular news article without future preview phrasing
    article_regular = {
        "title": "Hospital Inaugurates New Cardiac Care Facility",
        "published_at": "2026-11-07T12:00:00+05:45",
        "full_text": "Health ministers inaugurated the new facility today following five years of planning.",
    }
    is_stale, reason = detect_stale_preview(article_regular, ref_time=ref_time)
    assert is_stale is False


def test_editor_brief_contains_skip_it_rule_and_stale_previews():
    """Verify editor brief uses 'skip it' rule and renders excluded stale preview section."""
    from src.utils.editor_brief import generate_editor_brief

    budget = compute_word_budget(2.15)
    shortlist = [
        {
            "cluster_id": "cluster_01",
            "primary_title": "Primary Story",
            "category_guess": "politics",
            "source_count": 1,
            "sources": ["Outlet A"],
            "articles": [
                {
                    "source": "Outlet A",
                    "title": "Story Title",
                    "url": "https://a.com/story",
                    "published_at": "2026-11-07T10:00:00Z",
                    "text_summary": "Summary",
                    "full_text": "Full text body",
                }
            ],
        }
    ]
    stale_clusters = [
        {
            "cluster_id": "stale_01",
            "primary_title": "Past Cricket Match Preview",
            "stale_preview_reason": "Event time (5pm) on 2026-11-07 has passed",
            "source_count": 1,
            "sources": ["Outlet B"],
            "urls": ["https://b.com/match"],
            "articles": [
                {
                    "source": "Outlet B",
                    "title": "Match Preview",
                    "url": "https://b.com/match",
                    "stale_preview_reason": "Event time (5pm) on 2026-11-07 has passed",
                    "published_at": "2026-11-07T12:00:00Z",
                    "text_summary": "Match preview summary",
                }
            ],
        }
    ]

    brief_text = generate_editor_brief(
        run_date="2026-11-07",
        budget=budget,
        calib_msg="calibrated",
        shortlist=shortlist,
        stale_preview_clusters=stale_clusters,
    )

    # 1. Rule check: never rewrite preview as past tense; skip it
    assert "never rewrite a preview as past tense; skip it" in brief_text
    assert "use past tense (e.g. 'faced', 'met') or skip" not in brief_text

    # 2. Excluded stale preview section
    assert "## 6. Excluded Stale Previews" in brief_text
    assert "Past Cricket Match Preview" in brief_text


def test_validator_fingerprint_mismatch_refuses_approval(tmp_path):
    """EditorAgent.approve must refuse to run if the validator fingerprint has changed since validation."""

    run_dir = tmp_path / "2026-11-07"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)

    script_path = script_dir / "script.json"
    script_path.write_text(json.dumps({"language": "English", "origin": "editor", "scenes": []}), encoding="utf-8")

    # Write validation_report.json with an outdated / fake fingerprint
    report_path = script_dir / "validation_report.json"
    fake_report = {
        "validator_fingerprint": {
            "validator_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "editor_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        },
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }
    report_path.write_text(json.dumps(fake_report), encoding="utf-8")

    agent = EditorAgent(run_dir)
    manifest = {"stages": {"editor": {"status": "awaiting_approval"}}}

    with pytest.raises(ValueError, match="Validator fingerprint mismatch"):
        agent.approve(manifest)


def test_downstream_stage_refuses_on_fingerprint_mismatch(tmp_path):
    """Downstream agents must refuse execution if validator fingerprint has changed since approval."""
    from src.agents.voice import VoiceAgent

    run_dir = tmp_path / "2026-11-07"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)

    dummy_script = {
        "language": "English",
        "origin": "editor",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Test Scene",
                "narration": "Hello from testing narration.",
                "source": "Outlet",
                "source_urls": [],
                "visual_keywords": ["test"],
                "sensitive": False,
                "evidence": [],
            }
        ],
    }
    script_path = script_dir / "script.json"
    script_path.write_text(json.dumps(dummy_script), encoding="utf-8")
    script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()

    # Approval file with mismatched validator fingerprint
    approval_path = script_dir / "approval.json"
    approval_data = {
        "approved_at": "2026-11-07T12:00:00Z",
        "script_sha256": script_hash,
        "approved_by": "user",
        "validator_fingerprint": {
            "validator_sha256": "mismatched_sha",
            "editor_config_sha256": "mismatched_sha",
        },
    }
    approval_path.write_text(json.dumps(approval_data), encoding="utf-8")

    # Also write validation report with mismatched fingerprint
    report_path = script_dir / "validation_report.json"
    report_path.write_text(json.dumps({"validator_fingerprint": approval_data["validator_fingerprint"]}), encoding="utf-8")

    voice_agent = VoiceAgent(run_dir)
    manifest = {"stages": {"voice": {"status": "pending"}}}

    with pytest.raises(ValueError, match="validator or editor config changed"):
        voice_agent.run(manifest)


def test_manifest_schema_with_manual_edits():
    """Verify contracts/manifest.schema.json validates manual_edits property correctly."""
    from src.utils.schema_validate import validate_contract

    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "paused",
        "created_at": "2026-11-07T12:00:00Z",
        "updated_at": "2026-11-07T12:00:00Z",
        "stages": {},
        "manual_edits": [
            {
                "file": "data/runs/2026-11-07/script/shortlist.json",
                "what": "Appended cluster_13 to shortlist",
                "why": "Testing manual edit tracking",
                "timestamp": "2026-11-07T12:00:00Z",
            }
        ],
    }
    is_valid, err = validate_contract(manifest, "contracts/manifest.schema.json")
    assert is_valid is True

    # Negative case: missing required field 'what' in manual_edits item
    manifest_bad = dict(manifest)
    manifest_bad["manual_edits"] = [{"file": "f.txt", "why": "w", "timestamp": "2026-11-07T12:00:00Z"}]
    is_valid, err = validate_contract(manifest_bad, "contracts/manifest.schema.json")
    assert is_valid is False
    assert "'what' is a required property" in err


def test_validation_failure_moves_stale_files_and_writes_failed_preview(tmp_path):
    """When validation fails, move script.json and script_preview.md into script/_stale/ and write VALIDATION FAILED preview."""
    run_dir = tmp_path / "2026-11-08"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate previous script and preview
    old_script = script_dir / "script.json"
    old_script.write_text(json.dumps({"language": "English", "scenes": []}), encoding="utf-8")
    old_preview = script_dir / "script_preview.md"
    old_preview.write_text("# Old Valid Preview", encoding="utf-8")

    # Write a shortlist
    shortlist_path = script_dir / "shortlist.json"
    shortlist_path.write_text(json.dumps([]), encoding="utf-8")

    # Write an invalid draft (missing required fields, 0 scenes)
    draft_path = script_dir / "draft_script.json"
    draft_path.write_text(json.dumps({"language": "English", "scenes": []}), encoding="utf-8")

    manifest = {"stages": {"editor": {"status": "prepared"}}}
    agent = EditorAgent(run_dir)
    res = agent.validate_draft(manifest)

    # script.json and script_preview.md should be in _stale/
    stale_dir = script_dir / "_stale"
    assert (stale_dir / "script.json").exists()
    assert (stale_dir / "script_preview.md").exists()
    assert not old_script.exists()

    # New script_preview.md should be VALIDATION FAILED with errors
    new_preview_content = (script_dir / "script_preview.md").read_text(encoding="utf-8")
    assert "# VALIDATION FAILED" in new_preview_content
    assert "Expected exactly 7 scenes" in new_preview_content


def test_approve_refuses_when_validation_failed_or_no_script(tmp_path):
    """Approve must refuse when the latest validation failed or when no validated script.json exists."""
    run_dir = tmp_path / "2026-11-08"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"stages": {"editor": {"status": "needs_revision"}}}
    agent = EditorAgent(run_dir)

    # Case 1: no script.json exists
    with pytest.raises(FileNotFoundError, match="Missing validated script.json"):
        agent.approve(manifest)

    # Case 2: script.json exists, but validation_report shows failure
    script_path = script_dir / "script.json"
    script_path.write_text(json.dumps({"language": "English"}), encoding="utf-8")
    report_path = script_dir / "validation_report.json"
    report_path.write_text(
        json.dumps({
            "is_valid": False,
            "errors": ["Bad budget"],
            "validator_fingerprint": {"validator_sha256": "x"},
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="latest validation failed"):
        agent.approve(manifest)


def test_editing_whitelist_invalidates_approval(tmp_path):
    """Editing editor_whitelist.yaml must change the fingerprint and invalidate an approval."""
    from src.agents.voice import VoiceAgent
    from src.utils.editor_validator import compute_validator_fingerprint

    whitelist_path = Path("config/editor_whitelist.yaml")
    orig_bytes = whitelist_path.read_bytes()

    try:
        run_dir = tmp_path / "2026-11-08"
        script_dir = run_dir / "script"
        script_dir.mkdir(parents=True, exist_ok=True)

        current_fp = compute_validator_fingerprint()
        dummy_script = {
            "language": "English",
            "origin": "editor",
            "validator_fingerprint": current_fp,
            "scenes": [{"id": "scene_01", "type": "story", "headline": "H", "narration": "N"}],
        }
        script_path = script_dir / "script.json"
        script_path.write_text(json.dumps(dummy_script), encoding="utf-8")

        script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        approval_data = {
            "approved_at": "2026-11-08T12:00:00Z",
            "script_sha256": script_hash,
            "approved_by": "user",
            "validator_fingerprint": current_fp,
        }
        (script_dir / "approval.json").write_text(json.dumps(approval_data), encoding="utf-8")
        (script_dir / "validation_report.json").write_text(
            json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": current_fp}),
            encoding="utf-8",
        )

        manifest = {"stages": {"editor": {"status": "done"}, "voice": {"status": "pending"}}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Now edit the whitelist by appending a dummy entry
        whitelist_path.write_bytes(orig_bytes + b"\n  - dummy_test_entry_whitelist\n")

        # Downstream VoiceAgent check must fail because fingerprint changed
        voice_agent = VoiceAgent(run_dir)
        with pytest.raises(ValueError, match="validator or editor config changed"):
            voice_agent.run(manifest)

    finally:
        whitelist_path.write_bytes(orig_bytes)


def test_abandoned_run_refuses_all_stages(tmp_path):
    """When a run manifest status is 'abandoned', no stage may run on it."""
    run_dir = tmp_path / "2026-11-07"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "2026-11-07",
        "date": "2026-11-07",
        "status": "abandoned",
        "created_at": "2026-11-07T12:00:00Z",
        "updated_at": "2026-11-08T10:00:00Z",
        "stages": {"editor": {"status": "needs_revision", "retry_count": 0, "input_files": [], "output_files": []}},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    agent = EditorAgent(run_dir)
    with pytest.raises(RuntimeError, match="abandoned"):
        agent.prepare(manifest)
    with pytest.raises(RuntimeError, match="abandoned"):
        agent.validate_draft(manifest)
    with pytest.raises(RuntimeError, match="abandoned"):
        agent.approve(manifest)
    with pytest.raises(RuntimeError, match="abandoned"):
        agent.run(manifest)

    orch = Orchestrator(run_date="2026-11-07", runs_dir=tmp_path)
    with pytest.raises(RuntimeError, match="abandoned"):
        orch.execute_stage("scraper", manifest)
    with pytest.raises(RuntimeError, match="abandoned"):
        orch.run_pipeline()


def test_reset_stage_logs_to_manual_edits_and_validates_schema(tmp_path):
    """Orchestrator.reset_stage must reset stage state to pending, reset retries, log to manual_edits, and validate schema."""
    from src.orchestrator import Orchestrator
    from src.utils.schema_validate import validate_file

    run_dir = tmp_path / "2026-11-08"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "2026-11-08",
        "date": "2026-11-08",
        "status": "failed",
        "created_at": "2026-11-08T10:00:00Z",
        "updated_at": "2026-11-08T10:30:00Z",
        "stages": {
            "media": {
                "status": "failed",
                "retry_count": 3,
                "max_retries": 2,
                "input_files": ["script/script.json"],
                "output_files": ["media/media_manifest.json"],
                "error_message": "NameError: name 'hashlib' is not defined",
            }
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    orch = Orchestrator(run_date="2026-11-08", runs_dir=tmp_path)
    res = orch.reset_stage("media", reason="Testing manual media reset")
    assert res is True

    # Reload saved manifest
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["stages"]["media"]["status"] == "pending"
    assert updated["stages"]["media"]["retry_count"] == 0
    assert updated["stages"]["media"]["error_message"] is None
    assert updated["status"] == "pending"

    # Verify manual_edits entry
    assert "manual_edits" in updated
    assert len(updated["manual_edits"]) == 1
    edit = updated["manual_edits"][0]
    assert edit["file"] == str(manifest_path.as_posix())
    assert "media" in edit["what"]
    assert edit["why"] == "Testing manual media reset"
    assert "timestamp" in edit

    # Verify manifest contract schema
    ok, err = validate_file(manifest_path, "contracts/manifest.schema.json")
    assert ok is True, f"Contract validation failed: {err}"

    # Abandoned run must refuse reset
    updated["status"] = "abandoned"
    manifest_path.write_text(json.dumps(updated), encoding="utf-8")
    with pytest.raises(RuntimeError, match="abandoned"):
        orch.reset_stage("media")


def test_editing_media_manifest_schema_does_not_invalidate_approval(tmp_path):
    """Editing downstream schema (contracts/media_manifest.schema.json) must NOT invalidate approval."""
    from src.agents.voice import VoiceAgent
    from src.utils.editor_validator import compute_validator_fingerprint

    schema_path = Path("contracts/media_manifest.schema.json")
    orig_bytes = schema_path.read_bytes()

    try:
        run_dir = tmp_path / "2026-11-08"
        script_dir = run_dir / "script"
        script_dir.mkdir(parents=True, exist_ok=True)

        current_fp = compute_validator_fingerprint()
        dummy_script = {
            "language": "English",
            "origin": "editor",
            "validator_fingerprint": current_fp,
            "scenes": [{"id": "scene_01", "type": "story", "headline": "H", "narration": "N"}],
        }
        script_path = script_dir / "script.json"
        script_path.write_text(json.dumps(dummy_script), encoding="utf-8")

        script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        approval_data = {
            "approved_at": "2026-11-08T12:00:00Z",
            "script_sha256": script_hash,
            "approved_by": "user",
            "validator_fingerprint": current_fp,
        }
        (script_dir / "approval.json").write_text(json.dumps(approval_data), encoding="utf-8")
        (script_dir / "validation_report.json").write_text(
            json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": current_fp}),
            encoding="utf-8",
        )

        manifest = {"stages": {"editor": {"status": "done"}, "voice": {"status": "pending"}}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Edit media_manifest.schema.json by appending a whitespace
        schema_path.write_bytes(orig_bytes + b" ")

        # VoiceAgent check must STILL PASS without raising fingerprint mismatch
        voice_agent = VoiceAgent(run_dir)
        try:
            voice_agent.run(manifest)
        except ValueError as e:
            assert "validator or editor config changed" not in str(e)
            assert "script changed after approval" not in str(e)
    finally:
        schema_path.write_bytes(orig_bytes)


def test_only_editor_approve_and_validate_write_approval_or_validation_report():
    """Verify that NO code path outside EditorAgent (approve and validate) writes approval.json or validation_report.json."""
    import ast
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    tools_dir = repo_root / "tools"

    py_files = list(src_dir.rglob("*.py")) + list(tools_dir.rglob("*.py")) + [repo_root / "run_daily.py"]

    disallowed_writers = []

    for pf in py_files:
        if not pf.exists():
            continue
        # Only src/agents/editor.py is allowed to write approval.json or validation_report.json
        if pf.resolve() == (src_dir / "agents" / "editor.py").resolve():
            continue

        text = pf.read_text(encoding="utf-8")
        if "approval.json" not in text and "validation_report.json" not in text:
            continue

        # Parse AST to check if any file write operations target approval.json or validation_report.json
        tree = ast.parse(text, filename=str(pf))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("write_text", "write_bytes"):
                    line_repr = text.splitlines()[node.lineno - 1] if node.lineno <= len(text.splitlines()) else ""
                    if "approval" in line_repr or "validation_report" in line_repr:
                        disallowed_writers.append(f"{pf.name}:{node.lineno} -> {line_repr.strip()}")

                if func_name == "open" and len(node.args) >= 2:
                    mode_arg = node.args[1]
                    if isinstance(mode_arg, ast.Constant) and any(m in str(mode_arg.value) for m in ("w", "a", "+")):
                        line_repr = text.splitlines()[node.lineno - 1] if node.lineno <= len(text.splitlines()) else ""
                        if "approval" in line_repr or "validation_report" in line_repr:
                            disallowed_writers.append(f"{pf.name}:{node.lineno} -> {line_repr.strip()}")

    assert not disallowed_writers, f"Disallowed write operations found: {disallowed_writers}"






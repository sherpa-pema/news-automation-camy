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
            voice_agent.run(manifest, engine_override="fake")
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


def test_frozen_budget_ignores_newer_voice_report_after_prepare(tmp_path):
    """prepare writes script/budget.json once; a newer voice_report.json created after prepare must NOT change validation output."""
    import os
    import time

    runs_root = tmp_path / "runs"
    prior_run = runs_root / "2026-11-08"
    (prior_run / "script").mkdir(parents=True, exist_ok=True)
    (prior_run / "audio").mkdir(parents=True, exist_ok=True)

    # Prior run has 200 words in 100.0s -> 2.00 wps
    prior_words = " ".join(["word"] * 200)
    (prior_run / "script" / "script.json").write_text(
        json.dumps({"scenes": [{"narration": prior_words}]}), encoding="utf-8"
    )
    (prior_run / "audio" / "voice_report.json").write_text(
        json.dumps({"total_audio_duration_sec": 100.0}), encoding="utf-8"
    )

    run_dir = runs_root / "2026-11-09"
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    cluster_text = "Officials opened a new municipal library in Pokhara today. " * 10
    articles = [
        {
            "source": "Outlet A",
            "title": "New Municipal Library Opened in Pokhara",
            "url": "https://a.com/library",
            "text_summary": "Municipal library opened in Pokhara today.",
            "full_text": cluster_text,
            "published_at": "2026-11-09T08:00:00Z",
        }
    ]
    (run_dir / "raw" / "articles.json").write_text(json.dumps(articles), encoding="utf-8")
    manifest = {
        "run_id": "2026-11-09",
        "date": "2026-11-09",
        "status": "running",
        "stages": {"editor": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": []}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    agent = EditorAgent(run_dir)
    agent.prepare(manifest)

    budget_path = run_dir / "script" / "budget.json"
    assert budget_path.exists()
    frozen_budget = json.loads(budget_path.read_text(encoding="utf-8"))
    assert frozen_budget["words_per_second"] == 2.0
    assert frozen_budget["source_file"]
    assert frozen_budget["calculation"]

    # Create valid 7-scene draft (~254 words -> valid inside [216, 396] at wps=2.0)
    story_narr = (
        "Authorities inaugurated a modern public facility across Pokhara earlier today after months of careful planning. "
        "Local residents warmly welcomed improved community reading rooms, digital archives, and quiet study spaces. "
        "Municipal planners noted steady attendance from local students and dedicated teachers throughout the busy afternoon."
    )
    intro_narr = (
        "The top stories, breaking updates, and major community developments shaping the nation right now."
    )
    outro_narr = (
        "That concludes our daily news briefing for tonight. "
        "Thank you for watching our update and please stay safe this evening."
    )
    draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-09",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Daily Briefing", "narration": intro_narr, "category": None, "sensitive": False, "source": "Desk", "source_urls": [], "visual_keywords": ["studio", "city", "map"], "evidence": []},
        ] + [
            {
                "id": f"scene_{i:02d}",
                "type": "story",
                "headline": f"Story Headline Number {i}",
                "narration": story_narr,
                "category": "society",
                "sensitive": False,
                "source": "Outlet A",
                "source_urls": ["https://a.com/library"],
                "visual_keywords": ["library", "books", "city"],
                "evidence": [
                    {"sentence_index": 0, "source_url": "https://a.com/library"},
                    {"sentence_index": 1, "source_url": "https://a.com/library"},
                    {"sentence_index": 2, "source_url": "https://a.com/library"},
                ],
            }
            for i in range(1, 6)
        ] + [
            {"id": "outro", "type": "outro", "headline": "Thank You", "narration": outro_narr, "category": None, "sensitive": False, "source": "Desk", "source_urls": [], "visual_keywords": ["sunset", "city", "studio"], "evidence": []},
        ],
    }
    (run_dir / "script" / "draft_script.json").write_text(json.dumps(draft), encoding="utf-8")

    # 1st validation
    manifest["stages"]["editor"]["status"] = "prepared"
    agent.validate_draft(manifest)
    report1 = json.loads((run_dir / "script" / "validation_report.json").read_text(encoding="utf-8"))
    script1 = json.loads((run_dir / "script" / "script.json").read_text(encoding="utf-8"))
    assert report1["is_valid"] is True
    assert report1["metrics"]["words_per_second"] == 2.0

    # Create a NEWER voice_report.json after prepare with radically different speech rate (4.0 wps)
    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    newer_vr = run_dir / "audio" / "voice_report.json"
    newer_vr.write_text(json.dumps({"total_audio_duration_sec": 50.0}), encoding="utf-8")
    future_t = time.time() + 60.0
    os.utime(newer_vr, (future_t, future_t))

    # 2nd validation must ignore the newer voice_report.json and use frozen budget.json
    manifest["stages"]["editor"]["status"] = "awaiting_approval"
    agent.validate_draft(manifest)
    report2 = json.loads((run_dir / "script" / "validation_report.json").read_text(encoding="utf-8"))
    script2 = json.loads((run_dir / "script" / "script.json").read_text(encoding="utf-8"))

    assert report2["metrics"] == report1["metrics"]
    assert report2["scene_results"] == report1["scene_results"]
    assert report2["errors"] == report1["errors"]
    assert report2["warnings"] == report1["warnings"]
    assert script2["total_estimated_duration_sec"] == script1["total_estimated_duration_sec"]
    assert script2["scenes"] == script1["scenes"]
    assert json.loads(budget_path.read_text(encoding="utf-8")) == frozen_budget


def test_orchestrator_execute_stage_ignores_downstream_schema_changes(tmp_path, monkeypatch):
    """Verify Orchestrator.execute_stage uses script_validation_fingerprint only and ignores downstream schema edits."""
    import src.orchestrator as orch_mod
    from src.agents.base import BaseAgent
    from src.utils.editor_validator import (
        compute_validator_fingerprint,
        extract_script_validation_fingerprint,
    )

    schema_path = Path("contracts/render_plan.schema.json")
    orig_bytes = schema_path.read_bytes()

    class _DummyRenderAgent(BaseAgent):
        @property
        def name(self) -> str:
            return "render"

        @property
        def schema_contract_path(self) -> str | None:
            return None

        def run(self, manifest: dict) -> list[str]:
            return []

    try:
        run_dir = tmp_path / "2026-11-08"
        script_dir = run_dir / "script"
        script_dir.mkdir(parents=True, exist_ok=True)

        full_fp = compute_validator_fingerprint()
        flat_fp = extract_script_validation_fingerprint(full_fp)
        script_data = {
            "language": "English",
            "origin": "editor",
            "validator_fingerprint": flat_fp,
            "scenes": [{"id": "scene_01", "type": "story", "headline": "H", "narration": "N"}],
        }
        script_path = script_dir / "script.json"
        script_path.write_text(json.dumps(script_data), encoding="utf-8")
        script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()

        # Simulate split format: flat 7-key dict in approval.json + nested validation_report.json
        (script_dir / "approval.json").write_text(
            json.dumps({
                "approved_at": "2026-11-08T12:00:00Z",
                "script_sha256": script_hash,
                "approved_by": "user",
                "validator_fingerprint": flat_fp,
            }),
            encoding="utf-8",
        )
        (script_dir / "validation_report.json").write_text(
            json.dumps({"is_valid": True, "errors": [], "validator_fingerprint": full_fp}),
            encoding="utf-8",
        )

        manifest = {
            "run_id": "2026-11-08",
            "date": "2026-11-08",
            "status": "running",
            "stages": {
                "editor": {"status": "done", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": ["script/script.json"]},
                "render": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["render_plan.json"], "output_files": []},
            },
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        monkeypatch.setitem(orch_mod.AGENT_REGISTRY, "render", _DummyRenderAgent)

        # Mutate downstream schema: Orchestrator.execute_stage must still pass gate!
        schema_path.write_bytes(orig_bytes + b" ")
        orch = Orchestrator(run_date="2026-11-08", runs_dir=tmp_path)
        assert orch.execute_stage("render", manifest, force=True) is True

        # Mutate script_validation_fingerprint key in approval.json -> must refuse!
        bad_fp = dict(flat_fp)
        bad_fp["editor_config_sha256"] = "0" * 64
        (script_dir / "approval.json").write_text(
            json.dumps({
                "approved_at": "2026-11-08T12:00:00Z",
                "script_sha256": script_hash,
                "approved_by": "user",
                "validator_fingerprint": bad_fp,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="validator or editor config changed after approval"):
            orch.execute_stage("render", manifest, force=True)
    finally:
        schema_path.write_bytes(orig_bytes)


def _make_7_scene_draft(
    story_overrides: dict[int, dict] | None = None,
    run_date: str = "2026-11-12",
    source_url: str = "https://example.com/story",
) -> dict:
    """Helper building a valid 7-scene draft for validator unit tests."""
    intro_narr = (
        "The top headlines, breaking news updates, and major developments from across the nation right now."
    )
    outro_narr = (
        "That concludes our daily news briefing for tonight. "
        "Thank you for watching our broadcast and please stay safe this evening."
    )
    default_story = (
        "Officials in Pokhara inaugurated a new community water supply facility this afternoon after years of careful municipal planning and steady local engineering effort. "
        "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
    )
    scenes = [
        {
            "id": "intro",
            "type": "intro",
            "headline": "Daily Briefing",
            "narration": intro_narr,
            "category": None,
            "sensitive": False,
            "source": "Nepal News Desk",
            "source_urls": [],
            "visual_keywords": ["kathmandu valley", "news studio", "map of nepal"],
            "evidence": [],
        }
    ]
    for i in range(1, 6):
        sc = {
            "id": f"story_{i}",
            "type": "story",
            "headline": f"Community Project Update Number {i}",
            "narration": default_story,
            "category": "society",
            "sensitive": False,
            "source": "Outlet A",
            "source_urls": [source_url],
            "visual_keywords": ["mountain village", "water pipes", "community workers"],
            "evidence": [
                {"sentence_index": 0, "source_url": source_url},
                {"sentence_index": 1, "source_url": source_url},
            ],
        }
        if story_overrides and i in story_overrides:
            sc.update(story_overrides[i])
        scenes.append(sc)
    scenes.append(
        {
            "id": "outro",
            "type": "outro",
            "headline": "Thank You for Watching",
            "narration": outro_narr,
            "category": None,
            "sensitive": False,
            "source": "Nepal News Desk",
            "source_urls": [],
            "visual_keywords": ["kathmandu skyline", "sunset mountains", "news studio"],
            "evidence": [],
        }
    )
    return {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": run_date,
        "scenes": scenes,
    }


def test_scale_normalized_number_grounding_english_and_devanagari(tmp_path):
    """Verify 'Rs 1.50 crore' and '१ करोड ५० लाख' ground 'fifteen million', 'one point five crore', and 2% approx."""
    # Direct extractor check
    assert 15000000.0 in extract_numbers_from_text("The ministry allocated Rs 1.50 crore for road repairs.")
    assert 15000000.0 in extract_numbers_from_text("मन्त्रालयले १ करोड ५० लाख रुपैयाँ विनियोजन गरेको छ।")
    assert 15000000.0 in extract_numbers_from_text("Officials spent fifteen million rupees.")
    assert 15000000.0 in extract_numbers_from_text("Officials spent one point five crore rupees.")

    cluster = [
        {
            "cluster_id": "cluster_01",
            "urls": ["https://example.com/story"],
            "articles": [
                {
                    "url": "https://example.com/story",
                    "title": "Budget Allocation",
                    "full_text": "सडक मर्मतका लागि १ करोड ५० लाख रुपैयाँ बजेट छुट्याइएको छ।",
                }
            ],
        }
    ]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    budget = compute_word_budget(2.15)
    draft_path = tmp_path / "draft.json"

    # 1. "fifteen million" grounded by "१ करोड ५० लाख"
    draft = _make_7_scene_draft(
        story_overrides={
            1: {
                "narration": (
                    "Transport authorities allocated fifteen million rupees for rural highway maintenance work across the mountain district this week. "
                    "Senior engineers confirmed that repair crews will prioritize heavily damaged road stretches and drainage culverts before winter."
                )
            }
        }
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is True, f"Unexpected errors: {errors}"

    # 2. "about fifteen point two million" within 2% of 15,000,000 -> warning, no error
    draft["scenes"][1]["narration"] = (
        "Transport authorities allocated about fifteen point two million rupees for rural highway maintenance work across the district. "
        "Senior engineers confirmed that repair crews will prioritize heavily damaged mountain stretches and drainage culverts before winter."
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is True, f"Unexpected errors: {errors}"
    assert any("within 2%" in w for w in warnings)

    # 3. "one point five crore" grounded by "Rs 1.50 crore"
    cluster[0]["articles"][0]["full_text"] = "The transport ministry allocated Rs 1.50 crore for rural highway maintenance."
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    draft["scenes"][1]["narration"] = (
        "Transport authorities allocated one point five crore rupees for rural highway maintenance work across the mountain district. "
        "Senior engineers confirmed that repair crews will prioritize heavily damaged road stretches and drainage culverts before winter."
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is True, f"Unexpected errors: {errors}"


def test_bikram_sambat_year_whitelisting_in_fiscal_and_date_context(tmp_path):
    """Valid BS years (2082/2083/2084 for 2026) in fiscal/date context must not fail as ungrounded."""
    cluster = [
        {
            "cluster_id": "cluster_01",
            "urls": ["https://example.com/story"],
            "articles": [
                {
                    "url": "https://example.com/story",
                    "title": "Budget Plan",
                    "full_text": "The finance ministry released its annual infrastructure budget plan in Kathmandu today.",
                }
            ],
        }
    ]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    budget = compute_word_budget(2.15)
    draft_path = tmp_path / "draft.json"

    # Both "two thousand eighty three" and "twenty eighty three" in fiscal/date context pass
    draft = _make_7_scene_draft(
        run_date="2026-11-12",
        story_overrides={
            1: {
                "narration": (
                    "The finance ministry presented its infrastructure spending plan for fiscal year two thousand eighty three in Kathmandu today. "
                    "Senior budget officials confirmed that public project implementation targets will begin in Baisakh twenty eighty three across the federal provinces."
                )
            }
        },
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is True, f"Unexpected errors: {errors}"

    # An unrelated non-BS year outside source (e.g. 2075) still fails
    draft["scenes"][1]["narration"] = (
        "The finance ministry presented its infrastructure spending plan for fiscal year two thousand seventy five today in Kathmandu. "
        "Senior budget officials confirmed that implementation targets will begin soon across all federal provinces."
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is False
    assert any("Invented number 2075.0" in e for e in errors)


def test_acronym_blocklist_and_western_visual_keywords_and_structured_errors(tmp_path):
    """Verify Title-Case blocked acronyms fail, expanded forms pass, Western keywords fail, and structured errors carry scene_index/id."""
    cluster = [
        {
            "cluster_id": "cluster_01",
            "urls": ["https://example.com/story"],
            "articles": [
                {
                    "url": "https://example.com/story",
                    "title": "Power Grid Update",
                    "full_text": "The Nepal Electricity Authority expanded substation capacity in Lalitpur today.",
                }
            ],
        }
    ]
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps(cluster), encoding="utf-8")
    budget = compute_word_budget(2.15)
    draft_path = tmp_path / "draft.json"

    # 1. Title-Case blocked acronym ("Nea", "Nepse") + blocked Western visual keyword ("dollar bills")
    draft = _make_7_scene_draft(
        story_overrides={
            2: {
                "narration": (
                    "Officials at Nea and Nepse announced substation grid upgrades across Lalitpur today. "
                    "Local distribution teams confirmed steady voltage levels for residential consumers in Dhapakhel."
                ),
                "visual_keywords": ["dollar bills", "power substation", "kathmandu valley"],
            }
        }
    )
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is False
    assert any("Blocked acronym 'Nea'" in e for e in errors)
    assert any("Blocked acronym 'Nepse'" in e for e in errors)
    assert any("blocked Western term 'dollar bills'" in e for e in errors)
    assert any("Dhapakhel" in w for w in warnings)

    # Check structured_errors & structured_warnings carry scene_index and scene_id
    struct_errs = report.get("structured_errors", [])
    assert struct_errs
    assert all(e["scene_index"] == 2 and e["scene_id"] == "story_2" for e in struct_errs)
    assert any(w["scene_index"] == 2 and "Dhapakhel" in w["message"] for w in report.get("structured_warnings", []))

    # 2. Expanded form + Nepal-appropriate visual keywords pass
    draft["scenes"][2]["narration"] = (
        "The Nepal Electricity Authority announced substation grid upgrades across Lalitpur today. "
        "Local distribution teams confirmed steady voltage levels for residential consumers throughout the district."
    )
    draft["scenes"][2]["visual_keywords"] = ["power substation", "electric grid", "kathmandu valley"]
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, warnings, report = validate_script_draft(draft_path, shortlist_path, budget)
    assert is_valid is True, f"Unexpected errors: {errors}"


def test_cross_script_clustering_and_intra_cluster_supersession(tmp_path):
    """Verify English+Devanagari entity+number spine clusters within 12h, and differing numbers mark older as superseded."""
    from src.utils.editor_brief import generate_editor_brief

    en_older = {
        "source": "English Daily",
        "title": "NRB fines bank over NEPSE compliance breach",
        "url": "https://en.example.com/1",
        "published_at": "2026-11-12T08:00:00+05:45",
        "text_summary": "Nepal Rastra Bank penalized a brokerage over 250 million rupees transaction.",
        "full_text": ("Nepal Rastra Bank and NEPSE reviewed the 250 million compliance case involving 45 accounts. " * 6),
        "language": "en",
    }
    ne_newer = {
        "source": "Nepali Patrika",
        "title": "नेपाल राष्ट्र बैंक र नेप्सेको संयुक्त कारबाही, ५२ खातामा छानबिन",
        "url": "https://ne.example.com/1",
        "published_at": "2026-11-12T14:00:00+05:45",
        "text_summary": "राष्ट्र बैंकले २५० करोड तथा नेप्सेको ४५ खातामा कारबाही अघि बढायो।",
        "full_text": ("नेपाल राष्ट्र बैंक र नेप्सेको संयुक्त छानबिनमा ४५ खाता र ५२ उजुरी फेला परेका छन्। " * 6),
        "language": "ne",
    }
    en_too_far = {
        "source": "Late Outlet",
        "title": "Unrelated NRB report",
        "url": "https://en.example.com/2",
        "published_at": "2026-11-14T12:00:00+05:45",  # > 12h away
        "text_summary": "Nepal Rastra Bank and NEPSE issued annual 45 day notice.",
        "full_text": ("Nepal Rastra Bank and NEPSE issued an annual 45 day compliance reminder for members. " * 6),
        "language": "en",
    }

    cfg = {
        "clustering": {"min_full_text_words": 20, "title_similarity_threshold": 0.55, "token_overlap_threshold": 0.35, "top_clusters_limit": 12},
        "categories": {"economy": ["bank", "nepse", "राष्ट्र बैंक"]},
    }
    shortlist = cluster_articles([en_older, ne_newer, en_too_far], cfg)
    assert len(shortlist) == 2

    merged_cluster = next(c for c in shortlist if c["article_count"] == 2)
    assert set(merged_cluster["sources"]) == {"English Daily", "Nepali Patrika"}
    assert merged_cluster["primary_title"] == ne_newer["title"]
    assert merged_cluster["has_superseded_updates"] is True

    # Chronological order (newest/primary first): newer (superseded=False), older (superseded=True)
    arts = merged_cluster["articles"]
    assert arts[0]["url"] == ne_newer["url"] and arts[0]["superseded"] is False
    assert arts[1]["url"] == en_older["url"] and arts[1]["superseded"] is True

    brief = generate_editor_brief("2026-11-12", compute_word_budget(2.15), "Default", shortlist)
    assert "[OLDER UPDATE — DO NOT USE FOR NUMBERS]" in brief


def test_nepal_time_of_day_stale_previews():
    """Verify Nepal time-of-day anchors ('this morning', 'this afternoon', 'NEPSE to open') relative to +05:45 NPT."""
    from datetime import datetime, timedelta, timezone
    from src.utils.editor_cluster import detect_stale_preview

    npt = timezone(timedelta(hours=5, minutes=45))

    # 1. Morning anchor: stale at 14:00 NPT, valid at 09:00 NPT
    art_morning = {
        "title": "Parliament panel scheduled to meet this morning",
        "published_at": "2026-11-12T07:00:00+05:45",
        "full_text": "Lawmakers will convene this morning to review the energy bill.",
    }
    assert detect_stale_preview(art_morning, ref_time=datetime(2026, 11, 12, 14, 0, tzinfo=npt))[0] is True
    assert detect_stale_preview(art_morning, ref_time=datetime(2026, 11, 12, 9, 0, tzinfo=npt))[0] is False

    # 2. Afternoon anchor: stale at 18:00 NPT, valid at 14:00 NPT
    art_afternoon = {
        "title": "Cabinet meeting set to begin आज दिउँसो",
        "published_at": "2026-11-12T10:00:00+05:45",
        "full_text": "आज दिउँसो मन्त्रिपरिषद् बैठक बस्ने तय भएको छ।",
    }
    assert detect_stale_preview(art_afternoon, ref_time=datetime(2026, 11, 12, 18, 0, tzinfo=npt))[0] is True
    assert detect_stale_preview(art_afternoon, ref_time=datetime(2026, 11, 12, 14, 0, tzinfo=npt))[0] is False

    # 3. NEPSE pre-market anchor: stale at 16:00 NPT, valid at 10:00 NPT
    art_nepse = {
        "title": "NEPSE to open following positive banking outlook",
        "published_at": "2026-11-12T10:00:00+05:45",
        "full_text": "Analysts expect the benchmark index ahead of opening bell.",
    }
    assert detect_stale_preview(art_nepse, ref_time=datetime(2026, 11, 12, 16, 0, tzinfo=npt))[0] is True
    assert detect_stale_preview(art_nepse, ref_time=datetime(2026, 11, 12, 10, 0, tzinfo=npt))[0] is False


def test_cross_day_deduplication_penalty_and_brief_flag(tmp_path):
    """Verify 48-hour cross-day deduplication penalizes repeated stories and flags them in editor_brief.md."""
    runs_root = tmp_path / "runs"
    prev_dir = runs_root / "2026-11-11" / "script"
    prev_dir.mkdir(parents=True)
    (prev_dir / "script.json").write_text(
        json.dumps({
            "language": "English",
            "origin": "editor",
            "scenes": [
                {
                    "id": "story_1",
                    "type": "story",
                    "headline": "Highway Expansion Begins in Muglin",
                    "narration": "Crews started expanding the Muglin road section today with twenty workers.",
                    "source": "Outlet A",
                    "source_urls": ["https://a.com/muglin-day1"],
                    "visual_keywords": ["highway", "road", "workers"],
                    "sensitive": False,
                    "evidence": [],
                }
            ],
        }),
        encoding="utf-8",
    )
    (prev_dir / "approval.json").write_text(
        json.dumps({"approved_at": "2026-11-11T18:00:00+05:45", "script_sha256": "abc"}),
        encoding="utf-8",
    )

    today_dir = runs_root / "2026-11-12"
    (today_dir / "raw").mkdir(parents=True)
    articles = [
        {
            "source": "Outlet A",
            "title": "Highway Expansion Begins in Muglin",
            "url": "https://a.com/muglin-day1",
            "published_at": "2026-11-11T12:00:00+05:45",
            "text_summary": "Crews started expanding Muglin road.",
            "full_text": ("Crews started expanding the Muglin road section with twenty workers across the highway corridor. " * 6),
        },
        {
            "source": "Outlet B",
            "title": "New Hydropower Tunnel Completed in Dolakha",
            "url": "https://b.com/hydro",
            "published_at": "2026-11-12T09:00:00+05:45",
            "text_summary": "Engineers completed breakthrough of Dolakha tunnel.",
            "full_text": ("Engineers completed the breakthrough of a major hydropower tunnel in Dolakha district today. " * 6),
        },
    ]
    (today_dir / "raw" / "articles.json").write_text(json.dumps(articles), encoding="utf-8")
    manifest = {
        "run_id": "2026-11-12",
        "date": "2026-11-12",
        "status": "running",
        "stages": {"editor": {"status": "pending", "retry_count": 0, "max_retries": 2, "input_files": ["raw/articles.json"], "output_files": []}},
    }
    (today_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    agent = EditorAgent(today_dir)
    agent.prepare(manifest)

    shortlist = json.loads((today_dir / "script" / "shortlist.json").read_text(encoding="utf-8"))
    assert shortlist[0]["primary_title"] == "New Hydropower Tunnel Completed in Dolakha"
    assert shortlist[1]["covered_recently"] is True

    brief_text = (today_dir / "script" / "editor_brief.md").read_text(encoding="utf-8")
    assert "[COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]" in brief_text

    budget_data = json.loads((today_dir / "script" / "budget.json").read_text(encoding="utf-8"))
    assert "syllables_per_second" in budget_data
    assert budget_data["comma_pause_sec"] == 0.25
    assert budget_data["period_pause_sec"] == 0.55


def test_surgical_single_scene_retry_in_llm_writer(tmp_path, monkeypatch):
    """Verify llm_writer locks passing scenes and surgically repairs only the failing scene."""
    import src.utils.llm_writer as llm_writer_mod

    run_dir = tmp_path / "2026-11-12"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "2026-11-12",
            "date": "2026-11-12",
            "status": "running",
            "stages": {
                "editor": {
                    "status": "prepared",
                    "retry_count": 0,
                    "max_retries": 2,
                    "input_files": ["raw/articles.json"],
                    "output_files": [],
                }
            },
        }),
        encoding="utf-8",
    )

    shortlist = [
        {
            "cluster_id": f"cluster_0{i}",
            "primary_title": f"Cluster {i}",
            "urls": [f"https://example.com/story{i}"],
            "articles": [
                {
                    "source": "Outlet A",
                    "title": f"Cluster {i}",
                    "url": f"https://example.com/story{i}",
                    "full_text": f"Officials in Pokhara announced {10 + i} infrastructure projects today across municipal wards.",
                    "superseded": False,
                }
            ],
        }
        for i in range(1, 6)
    ]
    (script_dir / "shortlist.json").write_text(json.dumps(shortlist), encoding="utf-8")
    budget = compute_word_budget(2.15)
    (script_dir / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    (script_dir / "editor_brief.md").write_text("# Brief\nWrite 7 scenes.", encoding="utf-8")

    # Build round-1 full draft where ONLY story_2 (index 2) fails (blocked acronym 'Nea')
    draft_round1 = _make_7_scene_draft(run_date="2026-11-12")
    for i in range(1, 6):
        u = f"https://example.com/story{i}"
        draft_round1["scenes"][i]["source_urls"] = [u]
        draft_round1["scenes"][i]["evidence"] = [
            {"sentence_index": 0, "source_url": u},
            {"sentence_index": 1, "source_url": u},
        ]
        # ~46 words per story scene -> total ~272 words inside [232, 426]
        draft_round1["scenes"][i]["narration"] = (
            "Officials in Pokhara inaugurated a new community water supply facility this afternoon after careful municipal planning and steady engineering work. "
            "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
        )
    # Break ONLY story_2 (scene_index 2) with blocked acronym 'Nea'
    draft_round1["scenes"][2]["narration"] = (
        "Officials at Nea inaugurated a new community water supply facility this afternoon after careful municipal planning and steady engineering work. "
        "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
    )

    repaired_story_2 = dict(draft_round1["scenes"][2])
    repaired_story_2["narration"] = (
        "The Nepal Electricity Authority supported a new community water supply facility this afternoon after careful municipal planning and steady engineering work. "
        "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
    )

    captured_prompts: list[str] = []

    def fake_call_llm(provider, model, messages, timeout=180):
        prompt_text = messages[-1]["content"]
        captured_prompts.append(prompt_text)
        if len(captured_prompts) == 1:
            return json.dumps({
                "draft_script": draft_round1,
                "selection": {"selected_clusters": [{"cluster_id": "cluster_01", "reason": "Top"}], "skipped_clusters": []},
            })
        return json.dumps({"scenes": [repaired_story_2]})

    monkeypatch.setattr(llm_writer_mod, "detect_configured_provider", lambda preferred=None: ("gemini", "fake-model"))
    monkeypatch.setattr(llm_writer_mod, "_call_llm", fake_call_llm)

    res = llm_writer_mod.generate_and_validate_draft(run_dir, max_rounds=5)
    assert res["ok"] is True
    assert res["rounds"] == 2
    assert res["surgical_rounds"] == 1
    assert len(captured_prompts) == 2

    # Second prompt must be surgical (mentioning ONLY story_2, not story_1)
    assert "SURGICAL SCENE REPAIR" in captured_prompts[1]
    assert "id='story_2'" in captured_prompts[1]
    assert "id='story_1'" not in captured_prompts[1]
    assert "REMAINING GLOBAL WORD BUDGET" in captured_prompts[1]

    # Locked scene_1 must remain byte-for-byte identical, and scene_2 must be spliced
    final_draft = json.loads((script_dir / "draft_script.json").read_text(encoding="utf-8"))
    assert final_draft["scenes"][1] == draft_round1["scenes"][1]
    assert final_draft["scenes"][2]["narration"] == repaired_story_2["narration"]


def test_syllable_pause_duration_estimation_and_calibration(tmp_path):
    """Verify deterministic syllable counting, pause weights, historical SPS calibration, and blended duration."""
    from src.utils.editor_budget import (
        count_syllables_in_text,
        count_syllables_in_word,
        count_punctuation_pauses,
        calculate_words_per_second_details,
        estimate_scene_duration_sec,
    )

    assert count_syllables_in_word("infrastructure") >= 4
    assert count_syllables_in_word("authority") >= 4
    assert count_syllables_in_text("Nepal Electricity Authority, Kathmandu.") >= 12
    commas, periods = count_punctuation_pauses("First, second. Third! Fourth?")
    assert commas == 1
    assert periods == 3

    # Long multisyllabic text should yield longer duration than simple monosyllabic words of same count
    short_words_text = " ".join(["cat dog sun run box"] * 6) + "."
    long_words_text = " ".join(["infrastructure municipality electricity administration telecommunication"] * 6) + "."
    budget_cfg = {
        "words_per_second": 2.15,
        "syllables_per_second": 3.3,
        "comma_pause_sec": 0.25,
        "period_pause_sec": 0.55,
    }
    dur_short = estimate_scene_duration_sec(short_words_text, budget_cfg)
    dur_long = estimate_scene_duration_sec(long_words_text, budget_cfg)
    assert dur_long > dur_short

    # Historical voice_report calibration computes syllables_per_second alongside words_per_second
    runs_dir = tmp_path / "runs"
    run_folder = runs_dir / "2026-11-01"
    (run_folder / "voice").mkdir(parents=True)
    (run_folder / "script").mkdir(parents=True)
    (run_folder / "script" / "script.json").write_text(
        json.dumps({
            "scenes": [
                {
                    "narration": "The Nepal Electricity Authority expanded substation capacity in Lalitpur today, officials confirmed."
                }
            ]
        }),
        encoding="utf-8",
    )
    (run_folder / "voice" / "voice_report.json").write_text(
        json.dumps({"total_audio_duration_sec": 12.0}),
        encoding="utf-8",
    )
    details = calculate_words_per_second_details(runs_dir)
    assert details["source"] != "default"
    assert details["syllables_per_second"] > 0
    assert details["comma_pause_sec"] == 0.25
    assert details["period_pause_sec"] == 0.55


def test_eight_scene_broadcast_sequence_validation_pass(tmp_path):
    """8-scene broadcast draft passes validation when following strict category sequence."""
    from src.utils.editor_validator import validate_script_draft

    run_dir = tmp_path / "2026-11-15"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)

    shortlist = [
        {"cluster_id": "cluster_01", "primary_title": "Parliament passes bill", "urls": ["https://news.com/politics"], "articles": [{"url": "https://news.com/politics", "full_text": "Parliament passed the bill."}]},
        {"cluster_id": "cluster_02", "primary_title": "New school opens in Lalitpur", "urls": ["https://news.com/society"], "articles": [{"url": "https://news.com/society", "full_text": "A new school opened in Lalitpur."}]},
        {"cluster_id": "cluster_03", "primary_title": "Dashain celebrations begin", "urls": ["https://news.com/culture"], "articles": [{"url": "https://news.com/culture", "full_text": "Dashain festival celebrations have begun."}]},
        {"cluster_id": "cluster_04", "primary_title": "NEPSE gains twenty points", "urls": ["https://news.com/economy"], "articles": [{"url": "https://news.com/economy", "full_text": "Nepal Stock Exchange gained twenty points today."}]},
        {"cluster_id": "cluster_05", "primary_title": "Telecom expands five G trials", "urls": ["https://news.com/tech"], "articles": [{"url": "https://news.com/tech", "full_text": "Telecom expanded trials in Kathmandu valley."}]},
        {"cluster_id": "cluster_06", "primary_title": "Nepal cricket team wins match", "urls": ["https://news.com/sports"], "articles": [{"url": "https://news.com/sports", "full_text": "Nepal won the international match today."}]},
    ]
    (script_dir / "shortlist.json").write_text(json.dumps(shortlist), encoding="utf-8")

    draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-15",
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": "Nepal Daily Briefing for Sunday, November 15",
                "narration": "The top stories and major developments shaping the nation right now.",
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": ["Kathmandu valley", "news studio", "map of Nepal"],
                "evidence": [],
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Parliament passes major legislative bill in capital",
                "narration": "In national politics, lawmakers in parliament approved the new legislative bill during an afternoon session in Kathmandu. Leaders across major parties supported the decision following extensive committee deliberations.",
                "category": "national_politics",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/politics"],
                "visual_keywords": ["parliament building", "politicians meeting", "Kathmandu city"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/politics"}, {"sentence_index": 1, "source_url": "https://news.com/politics"}],
            },
            {
                "id": "scene_02",
                "type": "story",
                "headline": "New community school opens in Lalitpur district",
                "narration": "In society and local developments, authorities inaugurated a newly upgraded secondary school in Lalitpur today. The facility features modern classrooms and expanded learning laboratories for area students.",
                "category": "society_local",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/society"],
                "visual_keywords": ["school building", "students in classroom", "Lalitpur city"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/society"}, {"sentence_index": 1, "source_url": "https://news.com/society"}],
            },
            {
                "id": "scene_03",
                "type": "story",
                "headline": "Cultural festival celebrations begin across nation",
                "narration": "In culture and entertainment, communities across Nepal commenced vibrant festive observances with traditional musical performances and culinary gatherings. Cultural heritage centres hosted ceremonial gatherings throughout the capital valley.",
                "category": "culture_entertainment",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/culture"],
                "visual_keywords": ["temple square", "festival lights", "traditional dancers"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/culture"}, {"sentence_index": 1, "source_url": "https://news.com/culture"}],
            },
            {
                "id": "scene_04",
                "type": "story",
                "headline": "Stock exchange benchmark advances twenty points",
                "narration": "In business and economics, trading on the Nepal Stock Exchange advanced twenty points as turnover expanded substantially across banking and hydropower counters. Market analysts attributed the rise to strong institutional liquidity.",
                "category": "business_economics",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/economy"],
                "visual_keywords": ["stock market screen", "office district", "currency exchange"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/economy"}, {"sentence_index": 1, "source_url": "https://news.com/economy"}],
            },
            {
                "id": "scene_05",
                "type": "story",
                "headline": "Telecom expands high speed wireless infrastructure",
                "narration": "In science and technology, telecom providers inaugurated expanded high speed wireless coverage networks across urban hubs in the central region. Engineers confirmed improved digital reliability for enterprise consumers.",
                "category": "science_technology",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/tech"],
                "visual_keywords": ["telecom tower", "data centre servers", "digital network"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/tech"}, {"sentence_index": 1, "source_url": "https://news.com/tech"}],
            },
            {
                "id": "scene_06",
                "type": "story",
                "headline": "National cricket squad secures decisive victory",
                "narration": "In sports, the national cricket lineup completed a memorable tournament victory before an enthusiastic home crowd at the central stadium. Star performers executed disciplined bowling to defend a challenging total.",
                "category": "sports",
                "sensitive": False,
                "source": "Nepal News",
                "source_urls": ["https://news.com/sports"],
                "visual_keywords": ["cricket stadium", "crowd cheering", "trophy celebration"],
                "evidence": [{"sentence_index": 0, "source_url": "https://news.com/sports"}, {"sentence_index": 1, "source_url": "https://news.com/sports"}],
            },
            {
                "id": "outro",
                "type": "outro",
                "headline": "Thank You for Tuning In",
                "narration": "That wraps up today's Nepal Daily Briefing. Follow us for updates throughout the day. Thank you for watching, and stay safe.",
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": ["Kathmandu night skyline", "sunset mountains", "news studio lights"],
                "evidence": [],
            },
        ],
    }
    draft_file = script_dir / "draft_script.json"
    draft_file.write_text(json.dumps(draft), encoding="utf-8")

    budget = {
        "words_per_second": 2.15,
        "syllables_per_second": 3.3,
        "min_total_words": 150,
        "max_total_words": 500,
    }

    ok, errors, warnings, report = validate_script_draft(
        draft_path=draft_file,
        shortlist_path=script_dir / "shortlist.json",
        budget=budget,
    )
    assert ok, f"Expected 8-scene draft to pass, got errors: {errors}"
    assert report["is_valid"]
    assert report["metrics"]["scene_count"] == 8


def test_eight_scene_broadcast_sequence_validation_fail(tmp_path):
    """8-scene broadcast draft fails validation when categories are out of sequence."""
    from src.utils.editor_validator import validate_script_draft

    run_dir = tmp_path / "2026-11-15"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)

    # Swap scene_03 (culture) with sports so scene_03 has category 'sports'
    shortlist = [
        {"cluster_id": "cluster_01", "primary_title": "Parliament passes bill", "urls": ["https://news.com/politics"], "articles": [{"url": "https://news.com/politics", "full_text": "Parliament passed the bill."}]},
        {"cluster_id": "cluster_02", "primary_title": "New school opens in Lalitpur", "urls": ["https://news.com/society"], "articles": [{"url": "https://news.com/society", "full_text": "A new school opened in Lalitpur."}]},
        {"cluster_id": "cluster_03", "primary_title": "Sports match", "urls": ["https://news.com/sports"], "articles": [{"url": "https://news.com/sports", "full_text": "Sports match played."}]},
        {"cluster_id": "cluster_04", "primary_title": "Economy update", "urls": ["https://news.com/economy"], "articles": [{"url": "https://news.com/economy", "full_text": "Economy update."}]},
        {"cluster_id": "cluster_05", "primary_title": "Tech update", "urls": ["https://news.com/tech"], "articles": [{"url": "https://news.com/tech", "full_text": "Tech update."}]},
        {"cluster_id": "cluster_06", "primary_title": "Culture update", "urls": ["https://news.com/culture"], "articles": [{"url": "https://news.com/culture", "full_text": "Culture update."}]},
    ]
    (script_dir / "shortlist.json").write_text(json.dumps(shortlist), encoding="utf-8")

    draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": "test",
        "date": "2026-11-15",
        "scenes": [
            {"id": "intro", "type": "intro", "headline": "Intro", "narration": "The top stories, breaking updates, and major developments shaping the nation right now.", "category": None, "sensitive": False, "source": "Desk", "source_urls": [], "visual_keywords": ["Kathmandu", "studio", "map"], "evidence": []},
            {"id": "scene_01", "type": "story", "headline": "Politics", "narration": "Lawmakers approved the bill today during parliament session.", "category": "national_politics", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/politics"], "visual_keywords": ["parliament", "meeting", "Kathmandu"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/politics"}]},
            {"id": "scene_02", "type": "story", "headline": "Society", "narration": "Authorities inaugurated a new school in Lalitpur today.", "category": "society_local", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/society"], "visual_keywords": ["school", "classroom", "Lalitpur"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/society"}]},
            {"id": "scene_03", "type": "story", "headline": "Sports in Slot 3", "narration": "The national team won their tournament match today.", "category": "sports", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/sports"], "visual_keywords": ["stadium", "crowd", "celebration"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/sports"}]},
            {"id": "scene_04", "type": "story", "headline": "Economy", "narration": "Stock market advanced twenty points in trading today.", "category": "business_economics", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/economy"], "visual_keywords": ["stock", "screen", "office"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/economy"}]},
            {"id": "scene_05", "type": "story", "headline": "Tech", "narration": "Engineers expanded wireless network coverage today.", "category": "science_technology", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/tech"], "visual_keywords": ["tower", "servers", "network"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/tech"}]},
            {"id": "scene_06", "type": "story", "headline": "Culture in Slot 6", "narration": "Communities commenced vibrant festive observances today.", "category": "culture_entertainment", "sensitive": False, "source": "Desk", "source_urls": ["https://news.com/culture"], "visual_keywords": ["temple", "festival", "lights"], "evidence": [{"sentence_index": 0, "source_url": "https://news.com/culture"}]},
            {"id": "outro", "type": "outro", "headline": "Outro", "narration": "That wraps up today's briefing. Follow us for more updates. Thank you for watching.", "category": None, "sensitive": False, "source": "Desk", "source_urls": [], "visual_keywords": ["skyline", "sunset", "studio"], "evidence": []},
        ],
    }
    draft_file = script_dir / "draft_script.json"
    draft_file.write_text(json.dumps(draft), encoding="utf-8")

    budget = {"words_per_second": 2.15, "syllables_per_second": 3.3, "min_total_words": 50, "max_total_words": 500}

    ok, errors, warnings, report = validate_script_draft(draft_path=draft_file, shortlist_path=script_dir / "shortlist.json", budget=budget)
    assert not ok
    assert any("violates broadcast sequence" in e for e in errors)
    assert any("expected 'culture_entertainment'" in e for e in errors)


def test_compact_brief_for_llm_strips_excluded_and_redundant_details():
    """Verify _compact_brief_for_llm strips sections 5/6 and collapses redundant <details>."""
    from src.utils.llm_writer import _compact_brief_for_llm

    raw_brief = (
        "# Brief\n\n## 4. Today's Shortlist\n"
        "### [cluster_01] Major policy update\n"
        "  - **Outlet A**: [Title](https://a.com/1)\n"
        "    *Published*: 2026-11-17\n"
        "    *Summary*: Officials unveiled a comprehensive municipal infrastructure plan across Kathmandu wards today.\n"
        "    <details><summary>Click to view extracted full text</summary>\n\n"
        "    ```text\n"
        "    Very long duplicate full text body that wastes tokens.\n"
        "    ```\n"
        "    </details>\n\n"
        "  - **Outlet B**: [Short Summary Story](https://b.com/2)\n"
        "    *Published*: 2026-11-17\n"
        "    *Summary*: Brief.\n"
        "    <details><summary>Click to view extracted full text</summary>\n\n"
        "    ```text\n"
        "    Essential fallback details needed when summary is too short.\n"
        "    ```\n"
        "    </details>\n\n"
        "---\n"
        "## 5. Excluded Opinion & Commentary Pieces\n"
        "Lots of excluded opinion text that should never reach the writer LLM.\n"
    )
    compact = _compact_brief_for_llm(raw_brief)
    assert "## 4. Today's Shortlist" in compact
    assert "## 5. Excluded Opinion" not in compact
    assert "<details>" not in compact
    assert "Very long duplicate full text body" not in compact
    assert "Essential fallback details needed" in compact


def test_llm_writer_deterministic_sensitive_prepatch_avoids_extra_round(tmp_path, monkeypatch):
    """Verify missing sensitive=true flag is patched deterministically in round 1 with zero extra LLM calls."""
    import src.utils.llm_writer as llm_writer_mod

    run_dir = tmp_path / "2026-11-18"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "2026-11-18",
            "date": "2026-11-18",
            "status": "running",
            "stages": {"editor": {"status": "prepared", "retry_count": 0, "max_retries": 2, "input_files": [], "output_files": []}},
        }),
        encoding="utf-8",
    )

    shortlist = [
        {
            "cluster_id": f"cluster_0{i}",
            "primary_title": f"Cluster {i}",
            "urls": [f"https://example.com/s{i}"],
            "articles": [
                {
                    "source": "Outlet A",
                    "title": f"Cluster {i}",
                    "url": f"https://example.com/s{i}",
                    "full_text": (
                        "Fourteen passengers died in a fatal bus crash along the highway in Makwanpur today."
                        if i == 1
                        else f"Officials in Pokhara announced {10 + i} infrastructure projects today across municipal wards."
                    ),
                    "superseded": False,
                }
            ],
        }
        for i in range(1, 6)
    ]
    (script_dir / "shortlist.json").write_text(json.dumps(shortlist), encoding="utf-8")
    budget = compute_word_budget(2.15)
    (script_dir / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    (script_dir / "editor_brief.md").write_text("# Brief\nWrite 7 scenes.", encoding="utf-8")

    draft = _make_7_scene_draft(run_date="2026-11-18")
    for i in range(1, 6):
        u = f"https://example.com/s{i}"
        draft["scenes"][i]["source_urls"] = [u]
        draft["scenes"][i]["evidence"] = [
            {"sentence_index": 0, "source_url": u},
            {"sentence_index": 1, "source_url": u},
        ]
        draft["scenes"][i]["narration"] = (
            "Officials in Pokhara inaugurated a new community water supply facility this afternoon after careful municipal planning and steady engineering work. "
            "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
        )
    # Scene 1 matches fatal crash cluster, deliberately set sensitive=False
    draft["scenes"][1]["sensitive"] = False

    call_count = 0

    def fake_call_llm(provider, model, messages, timeout=180):
        nonlocal call_count
        call_count += 1
        return json.dumps({
            "draft_script": draft,
            "selection": {"selected_clusters": [{"cluster_id": "cluster_01", "reason": "Top"}], "skipped_clusters": []},
        })

    monkeypatch.setattr(llm_writer_mod, "detect_configured_provider", lambda preferred=None: ("gemini", "fake-model"))
    monkeypatch.setattr(llm_writer_mod, "_call_llm", fake_call_llm)

    res = llm_writer_mod.generate_and_validate_draft(run_dir, max_rounds=5)
    assert res["ok"] is True
    assert call_count == 1
    final_draft = json.loads((script_dir / "draft_script.json").read_text(encoding="utf-8"))
    assert final_draft["scenes"][1]["sensitive"] is True


def test_editor_validate_auto_approve_option(tmp_path):
    """Verify validate_draft(auto_approve=True) transitions stage to done and writes approval.json."""
    run_dir = tmp_path / "2026-11-18"
    script_dir = run_dir / "script"
    script_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": "2026-11-18",
        "date": "2026-11-18",
        "run_kind": "test",
        "status": "running",
        "stages": {"editor": {"status": "prepared", "retry_count": 0, "max_retries": 2, "input_files": [], "output_files": []}},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    shortlist = [
        {
            "cluster_id": f"cluster_0{i}",
            "primary_title": f"Cluster {i}",
            "urls": [f"https://example.com/a{i}"],
            "articles": [{"source": "Outlet A", "title": f"Cluster {i}", "url": f"https://example.com/a{i}", "full_text": "Officials opened a municipal center today in Pokhara.", "superseded": False}],
        }
        for i in range(1, 6)
    ]
    (script_dir / "shortlist.json").write_text(json.dumps(shortlist), encoding="utf-8")
    (script_dir / "budget.json").write_text(json.dumps(compute_word_budget(2.15)), encoding="utf-8")

    draft = _make_7_scene_draft(run_date="2026-11-18")
    for i in range(1, 6):
        u = f"https://example.com/a{i}"
        draft["scenes"][i]["source_urls"] = [u]
        draft["scenes"][i]["evidence"] = [
            {"sentence_index": 0, "source_url": u},
            {"sentence_index": 1, "source_url": u},
        ]
        draft["scenes"][i]["narration"] = (
            "Officials in Pokhara inaugurated a new community water supply facility this afternoon after careful municipal planning and steady engineering work. "
            "Municipal engineers confirmed that local households across multiple rural wards now receive clean drinking water daily from upgraded local treatment stations."
        )
    (script_dir / "draft_script.json").write_text(json.dumps(draft), encoding="utf-8")

    agent = EditorAgent(run_dir)
    outputs = agent.validate_draft(manifest, auto_approve=True)
    assert (script_dir / "approval.json").exists()
    assert any("approval.json" in o for o in outputs)
    assert manifest["stages"]["editor"]["status"] == "done"


def test_rank_clusters_with_diversity_matches_naive_greedy():
    """Verify O(n log n) heap-based diversity ranking matches naive O(n^2) selection order exactly."""
    from src.utils.editor_cluster import rank_clusters_with_diversity

    items = [
        {"id": "p1", "category_guess": "politics", "score": 200.0},
        {"id": "p2", "category_guess": "politics", "score": 190.0},
        {"id": "p3", "category_guess": "politics", "score": 180.0},
        {"id": "e1", "category_guess": "economy", "score": 185.0},
        {"id": "e2", "category_guess": "economy", "score": 120.0},
        {"id": "s1", "category_guess": "sports", "score": 180.0},
        {"id": "h1", "category_guess": "health", "score": 50.0},
    ]

    # Naive O(n^2) reference
    pending = list(items)
    cat_counts: dict[str, int] = {}
    naive_order = []
    while pending:
        best_i = max(range(len(pending)), key=lambda i: (pending[i]["score"] - cat_counts.get(pending[i]["category_guess"], 0) * 15.0, -i))
        chosen = pending.pop(best_i)
        cat_counts[chosen["category_guess"]] = cat_counts.get(chosen["category_guess"], 0) + 1
        naive_order.append(chosen["id"])

    fast_ranked = rank_clusters_with_diversity(items, base_score_fn=lambda c: c["score"], category_penalty=15.0)
    assert [c["id"] for c in fast_ranked] == naive_order


def test_spoken_text_allows_digits_in_narration_display(tmp_path):
    """When spoken_text is provided, narration may contain digits/symbols; spoken_text must not."""
    cluster = [
        {"cluster_id": "c1", "urls": ["https://example.com/story"], "articles": [{"url": "https://example.com/story", "full_text": "Road project cost 1,50,000 rupees."}]}
    ]
    shortlist_p = tmp_path / "shortlist.json"
    shortlist_p.write_text(json.dumps(cluster), encoding="utf-8")

    draft = _make_7_scene_draft(
        story_overrides={
            1: {
                "narration": "Officials allocated 1,50,000 rupees ($1,100) for road repair work in Pokhara today. Engineers confirmed that construction will begin promptly.",
                "spoken_text": "Officials allocated one lakh fifty thousand rupees for road repair work in Pokhara today. Engineers confirmed that construction will begin promptly.",
            }
        }
    )
    draft_p = tmp_path / "draft.json"
    draft_p.write_text(json.dumps(draft), encoding="utf-8")
    budget = compute_word_budget(2.15)

    is_valid, errors, warnings, report = validate_script_draft(draft_p, shortlist_p, budget)
    assert is_valid is True, f"Expected pass with valid spoken_text, got errors: {errors}"

    # If spoken_text itself has digits, it must fail
    draft["scenes"][1]["spoken_text"] = "Officials allocated 150000 rupees for road repair work in Pokhara today."
    draft_p.write_text(json.dumps(draft), encoding="utf-8")
    is_valid, errors, _, _ = validate_script_draft(draft_p, shortlist_p, budget)
    assert is_valid is False
    assert any("Digits found in spoken_text" in e for e in errors)


def test_duplicate_cluster_rejection_when_shortlist_sufficient(tmp_path):
    """When shortlist has sufficient clusters (len >= story_count), duplicate cluster_id is rejected."""
    cluster = [
        {"cluster_id": f"c_{i}", "urls": [f"https://news.com/{i}"], "articles": [{"url": f"https://news.com/{i}", "full_text": "Content here."}]}
        for i in range(1, 7)
    ]
    shortlist_p = tmp_path / "shortlist.json"
    shortlist_p.write_text(json.dumps(cluster), encoding="utf-8")

    # Override scene 2 to use same URL/cluster as scene 1
    draft = _make_7_scene_draft(
        story_overrides={
            1: {"source_urls": ["https://news.com/1"]},
            2: {"source_urls": ["https://news.com/1"]},
            3: {"source_urls": ["https://news.com/3"]},
            4: {"source_urls": ["https://news.com/4"]},
            5: {"source_urls": ["https://news.com/5"]},
        }
    )
    draft_p = tmp_path / "draft.json"
    draft_p.write_text(json.dumps(draft), encoding="utf-8")
    budget = compute_word_budget(2.15)

    is_valid, errors, _, _ = validate_script_draft(draft_p, shortlist_p, budget)
    assert is_valid is False
    assert any("Duplicate story scene for cluster 'c_1'" in e for e in errors)


def test_south_asian_comma_number_extraction():
    """Verify numbers with South Asian grouping (1,50,000) are extracted accurately."""
    from src.utils.editor_validator import extract_numbers_from_text
    nums = extract_numbers_from_text("The total cost was 1,50,000 rupees and 25,00,000 for phase two.")
    assert 150000.0 in nums
    assert 2500000.0 in nums












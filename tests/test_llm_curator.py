"""Unit tests for LLM-powered news curation, semantic clustering, and multi-source synthesis."""
import json
from pathlib import Path
import pytest

from src.utils.llm_curator import (
    build_candidate_digest,
    build_curator_prompt,
    curate_and_cluster_with_llm,
)
from src.utils.editor_cluster import cluster_articles
from src.utils.editor_brief import generate_editor_brief
from src.utils.editor_budget import compute_word_budget
import src.utils.llm_curator as llm_curator_mod


def test_build_candidate_digest():
    """Verify build_candidate_digest filters ads/thin/opinion/stale and constructs compact entries."""
    articles = [
        {
            "source": "Outlet A",
            "title": "Ad piece",
            "url": "https://a.com/ad",
            "full_text": "Buy products now at the lowest prices available anywhere.",
            "is_ad": True,
        },
        {
            "source": "Outlet B",
            "title": "Thin snippet",
            "url": "https://b.com/thin",
            "full_text": "Very brief note here.",
        },
        {
            "source": "Outlet C",
            "title": "My opinion on fiscal reforms",
            "url": "https://c.com/opinion/fiscal",
            "full_text": "I think that my personal opinion on this matter is clear because I have said so many times.",
        },
        {
            "source": "Outlet D",
            "title": "Substation upgrade inaugurated in Lalitpur",
            "url": "https://d.com/substation",
            "published_at": "2026-11-12T10:00:00+05:45",
            "text_summary": "Energy officials inaugurated an upgraded substation in Lalitpur today.",
            "full_text": ("Energy officials inaugurated an upgraded substation in Lalitpur today with twenty engineers present. " * 5),
            "importance_hint": 0.85,
            "source_trust": 0.90,
        },
    ]

    eligible, digest, opinions, stale = build_candidate_digest(articles, min_words=20)
    assert len(eligible) == 1
    assert len(digest) == 1
    assert len(opinions) == 1
    assert digest[0]["source"] == "Outlet D"
    assert digest[0]["title"] == "Substation upgrade inaugurated in Lalitpur"
    assert "inaugurated an upgraded substation" in digest[0]["summary"]


def test_build_curator_prompt():
    """Verify prompt formatting and instructions."""
    digest = [
        {"id": 0, "source": "Outlet A", "title": "Headline 1", "published_at": "2026-11-12", "summary": "Summary 1"}
    ]
    prompt = build_curator_prompt(digest)
    assert "SEMANTIC EVENT CLUSTERING" in prompt
    assert "MULTI-SOURCE INFORMATION SYNTHESIS DIRECTIVES" in prompt
    assert "NATIONAL NEWS RELEVANCE & EDITORIAL PRIORITY" in prompt
    assert '"clusters":' in prompt
    assert "Headline 1" in prompt


def test_curate_and_cluster_with_llm_success(monkeypatch):
    """Verify semantic clustering merges outlets, preserves synthesis notes, and filters foreign wires."""
    articles = [
        {
            "source": "Republica",
            "title": "Death toll in Bhimphedi passenger van accident reaches 14",
            "url": "https://republica.com/van-accident",
            "published_at": "2026-11-12T14:00:00+05:45",
            "text_summary": "Fourteen passengers died and six were injured in Makwanpur today.",
            "full_text": ("Fourteen passengers died and six were injured when an EV van plunged off the road in Bhimphedi. " * 6),
            "source_trust": 0.85,
        },
        {
            "source": "The Kathmandu Post",
            "title": "Makwanpur EV crash: Van was en route from Kathmandu to Sarlahi",
            "url": "https://kathmandupost.com/van-crash",
            "published_at": "2026-11-12T12:00:00+05:45",
            "text_summary": "Police said the vehicle plunged five hundred metres down a cliff.",
            "full_text": ("Police confirmed the electric van fell 500 metres down a steep slope while traveling from Kathmandu to Sarlahi. " * 6),
            "source_trust": 0.90,
        },
        {
            "source": "Foreign Wire",
            "title": "Street protests erupt across Valencia over public transport fees",
            "url": "https://foreign.com/valencia",
            "published_at": "2026-11-12T11:00:00+05:45",
            "text_summary": "Protesters gathered in Valencia demanding transit subsidies.",
            "full_text": ("Protesters gathered in Valencia demanding municipal transit subsidies across central districts. " * 6),
            "source_trust": 0.60,
        },
        {
            "source": "Nagarik News",
            "title": "Nepal Rastra Bank issues new monetary circular for commercial banks",
            "url": "https://nagarik.com/nrb-circular",
            "published_at": "2026-11-12T09:00:00+05:45",
            "text_summary": "The central bank issued liquidity directives for commercial banks.",
            "full_text": ("Nepal Rastra Bank issued updated liquidity directives for commercial banks operating across provinces. " * 6),
            "source_trust": 0.80,
        },
    ]

    fake_llm_response = {
        "clusters": [
            {
                "article_ids": [0, 1],
                "primary_title": "Fourteen killed in Makwanpur electric passenger van accident",
                "category": "disaster",
                "editorial_priority": 9.5,
                "synthesis_notes": "Combine 14 fatalities from Republica with vehicle route and 500m plunge details from Kathmandu Post.",
                "conflicting_figures_resolution": "Earlier reports noted 6 dead; use verified 14 from latest update.",
            },
            {
                "article_ids": [3],
                "primary_title": "Nepal Rastra Bank issues new monetary circular for banks",
                "category": "economy",
                "editorial_priority": 8.0,
                "synthesis_notes": "Single-outlet report on central bank liquidity directive.",
                "conflicting_figures_resolution": None,
            },
        ],
        "filtered_out_ids": [
            {"id": 2, "reason": "Foreign wire story without domestic impact on Nepal."}
        ],
    }

    monkeypatch.setattr(llm_curator_mod, "detect_configured_provider", lambda pref=None: ("gemini", "fake-model"))
    monkeypatch.setattr(llm_curator_mod, "_call_llm", lambda p, m, msgs, timeout=60: json.dumps(fake_llm_response))

    cfg = {
        "llm_curator": {"enabled": True, "enabled_in_tests": True, "timeout_sec": 30},
        "clustering": {"min_full_text_words": 20, "top_clusters_limit": 10},
    }

    res = curate_and_cluster_with_llm(articles, cfg)
    assert res is not None
    shortlist, opinions, stale = res

    assert len(shortlist) == 2
    # First cluster combines Republica and Kathmandu Post
    c0 = shortlist[0]
    assert c0["primary_title"] == "Fourteen killed in Makwanpur electric passenger van accident"
    assert c0["category_guess"] == "disaster"
    assert c0["editorial_priority"] == 9.5
    assert c0["source_count"] == 2
    assert set(c0["sources"]) == {"Republica", "The Kathmandu Post"}
    assert c0["llm_curated"] is True
    assert "Combine 14 fatalities" in c0["synthesis_notes"]
    assert "Earlier reports noted 6" in c0["conflicting_figures_resolution"]

    # Foreign wire story (Valencia protest) was filtered out
    all_cluster_urls = {u for c in shortlist for u in c["urls"]}
    assert "https://foreign.com/valencia" not in all_cluster_urls


def test_curator_fallback_to_heuristic_on_llm_failure(monkeypatch):
    """Verify cluster_articles seamlessly falls back to heuristic clustering if LLM raises an error."""
    articles = [
        {
            "source": "Outlet A",
            "title": "Government announces annual hydro project budget",
            "url": "https://a.com/story1",
            "published_at": "2026-11-12T10:00:00+05:45",
            "text_summary": "Ministry allocated funding for transmission lines.",
            "full_text": ("Ministry allocated funding for transmission lines across seven provinces today. " * 6),
        },
        {
            "source": "Outlet B",
            "title": "Government announces annual hydro project budget",
            "url": "https://b.com/story2",
            "published_at": "2026-11-12T10:30:00+05:45",
            "text_summary": "Ministry allocated funding for transmission lines.",
            "full_text": ("Ministry allocated funding for transmission lines across seven provinces today. " * 6),
        },
    ]

    # Force LLM error
    def failing_call(*args, **kwargs):
        raise RuntimeError("API Gateway 503 Service Unavailable")

    monkeypatch.setattr(llm_curator_mod, "detect_configured_provider", lambda pref=None: ("gemini", "fake-model"))
    monkeypatch.setattr(llm_curator_mod, "_call_llm", failing_call)

    cfg = {
        "llm_curator": {"enabled": True, "enabled_in_tests": True, "timeout_sec": 30},
        "clustering": {"min_full_text_words": 20, "title_similarity_threshold": 0.55, "token_overlap_threshold": 0.35, "top_clusters_limit": 10},
        "categories": {"economy": ["budget", "funding"]},
    }

    # Must NOT raise exception; must gracefully fall back to heuristic clustering
    shortlist = cluster_articles(articles, cfg)
    assert len(shortlist) == 1
    assert shortlist[0]["source_count"] == 2
    assert shortlist[0].get("llm_curated") is not True


def test_editor_brief_displays_multi_source_synthesis():
    """Verify editor_brief.md displays multi-source synthesis and numbers resolution directives."""
    shortlist = [
        {
            "cluster_id": "cluster_01",
            "primary_title": "Makwanpur EV crash death toll reaches 14",
            "category_guess": "disaster",
            "source_count": 2,
            "sources": ["Republica", "The Kathmandu Post"],
            "synthesis_notes": "Combine 14 fatalities from Republica with vehicle route details from Kathmandu Post.",
            "conflicting_figures_resolution": "Earlier update said 11 dead; use 14 from latest article.",
            "articles": [
                {
                    "source": "Republica",
                    "title": "Makwanpur EV crash death toll reaches 14",
                    "url": "https://republica.com/van",
                    "published_at": "2026-11-12T14:00:00+05:45",
                    "text_summary": "Fourteen dead in crash.",
                    "full_text": "Fourteen dead in crash.",
                    "superseded": False,
                }
            ],
        }
    ]

    budget = compute_word_budget(2.15)
    brief = generate_editor_brief("2026-11-12", budget, "Calibrated", shortlist)

    assert "Multi-Source Synthesis" in brief
    assert "Combine 14 fatalities from Republica" in brief
    assert "Numbers Resolution" in brief
    assert "Earlier update said 11 dead" in brief
    assert "9. **Multi-Source Synthesis**:" in brief


def test_curator_stratified_slot_allocation(monkeypatch):
    """Verify that shortlisted clusters are stratified across the 6 broadcast slots in exact sequence."""
    articles = [
        {"source": "Out 1", "title": "Parliament news", "url": "https://a.com/pol", "full_text": "Parliament meeting held today. " * 5},
        {"source": "Out 2", "title": "Local community school", "url": "https://a.com/soc", "full_text": "School in Lalitpur opens today. " * 5},
        {"source": "Out 3", "title": "Dashain cultural festival", "url": "https://a.com/cul", "full_text": "Dashain festival observed today. " * 5},
        {"source": "Out 4", "title": "NEPSE stock market", "url": "https://a.com/biz", "full_text": "Stock market index gained points. " * 5},
        {"source": "Out 5", "title": "Telecom AI tech", "url": "https://a.com/tech", "full_text": "New software and technology deployed. " * 5},
        {"source": "Out 6", "title": "Cricket match championship", "url": "https://a.com/spt", "full_text": "Cricket team won the tournament. " * 5},
    ]

    fake_response = {
        "clusters": [
            {"article_ids": [5], "primary_title": "Cricket victory", "category": "sports", "editorial_priority": 9.0},
            {"article_ids": [0], "primary_title": "Parliament meeting", "category": "national_politics", "editorial_priority": 9.5},
            {"article_ids": [3], "primary_title": "NEPSE gains", "category": "business_economics", "editorial_priority": 7.0},
            {"article_ids": [1], "primary_title": "School opened", "category": "society_local", "editorial_priority": 8.0},
            {"article_ids": [4], "primary_title": "Tech deployed", "category": "science_technology", "editorial_priority": 6.5},
            {"article_ids": [2], "primary_title": "Festival observed", "category": "culture_entertainment", "editorial_priority": 7.5},
        ],
        "filtered_out_ids": [],
    }

    monkeypatch.setattr(llm_curator_mod, "detect_configured_provider", lambda pref=None: ("gemini", "fake-model"))
    monkeypatch.setattr(llm_curator_mod, "_call_llm", lambda p, m, msgs, timeout=60: json.dumps(fake_response))

    cfg = {
        "llm_curator": {"enabled": True, "enabled_in_tests": True, "timeout_sec": 30},
        "clustering": {"min_full_text_words": 10, "top_clusters_limit": 6},
    }

    res = curate_and_cluster_with_llm(articles, cfg)
    assert res is not None
    shortlist, opinions, stale = res

    assert len(shortlist) == 6
    # Cluster 01 must be slot_01 (national_politics)
    assert shortlist[0]["broadcast_slot_id"] == "slot_01"
    assert shortlist[0]["broadcast_slot"] == "national_politics"
    # Cluster 02 must be slot_02 (society_local)
    assert shortlist[1]["broadcast_slot_id"] == "slot_02"
    assert shortlist[1]["broadcast_slot"] == "society_local"
    # Cluster 03 must be slot_03 (culture_entertainment)
    assert shortlist[2]["broadcast_slot_id"] == "slot_03"
    assert shortlist[2]["broadcast_slot"] == "culture_entertainment"
    # Cluster 04 must be slot_04 (business_economics)
    assert shortlist[3]["broadcast_slot_id"] == "slot_04"
    assert shortlist[3]["broadcast_slot"] == "business_economics"
    # Cluster 05 must be slot_05 (science_technology)
    assert shortlist[4]["broadcast_slot_id"] == "slot_05"
    assert shortlist[4]["broadcast_slot"] == "science_technology"
    # Cluster 06 must be slot_06 (sports)
    assert shortlist[5]["broadcast_slot_id"] == "slot_06"
    assert shortlist[5]["broadcast_slot"] == "sports"

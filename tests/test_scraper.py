"""Unit and integration tests for the Scraper Agent using local fixtures."""
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import requests

from src.agents.scraper import (
    ScraperAgent,
    build_text_summary,
    canonical_url_key,
    compute_importance_hint,
    compute_priority_score,
    extract_jsonld_article,
    infer_category_hint,
    is_duplicate_title,
    normalize_language_code,
    normalize_url,
    parse_article_date,
    truncate_at_sentence_boundary,
    KATHMANDU_TZ,
)
from src.utils.editor_cluster import cluster_articles
from src.utils.editor_validator import extract_numbers_from_text
from src.utils.schema_validate import validate_contract

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_feed_xml() -> bytes:
    with open(FIXTURES_DIR / "sample_feed.xml", "rb") as f:
        return f.read()


@pytest.fixture
def sample_article_html() -> str:
    with open(FIXTURES_DIR / "sample_article.html", "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def short_article_html() -> str:
    with open(FIXTURES_DIR / "short_article.html", "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def sample_robots_txt() -> str:
    with open(FIXTURES_DIR / "sample_robots.txt", "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------------------------------------------------------
# 1. URL Normalization and Title Deduplication
# -----------------------------------------------------------------------------
def test_url_normalization():
    raw_url = "https://sample.nepalnews.com/news/hydropower-project-gandaki/?utm_source=rss&utm_medium=feed&ref=newsbot#comments"
    norm = normalize_url(raw_url)
    assert "utm_source" not in norm
    assert "utm_medium" not in norm
    assert "ref" not in norm
    assert "#comments" not in norm
    assert not norm.endswith("/")
    assert norm == "https://sample.nepalnews.com/news/hydropower-project-gandaki"


def test_title_deduplication():
    title1 = "Government Approves New Hydropower Project in Gandaki"
    title2 = "Government Approves New Hydropower Project in Gandaki!"
    title3 = "Kathmandu Valley Air Quality Index Improves Following Rain"

    existing = [title1]
    # Near identical title should be detected as duplicate
    assert is_duplicate_title(title2, existing, threshold=0.9) is True
    # Different title should not be detected as duplicate
    assert is_duplicate_title(title3, existing, threshold=0.9) is False


def test_language_normalization():
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("en") == "en"
    assert normalize_language_code("en-US") == "en"
    assert normalize_language_code("Nepali") == "ne"
    assert normalize_language_code("ne") == "ne"
    assert normalize_language_code(None) == "en"


# -----------------------------------------------------------------------------
# 2. 24-Hour Filter with Asia/Kathmandu Timezone Edge Cases
# -----------------------------------------------------------------------------
def test_24_hour_filter_edge_cases():
    ref_time = datetime(2026, 11, 6, 12, 0, 0, tzinfo=KATHMANDU_TZ)

    # 1. Story from 2 hours ago -> KEEP
    entry_recent = {"published": "Fri, 06 Nov 2026 10:00:00 +0545"}
    dt_recent = parse_article_date(entry_recent, "")
    assert dt_recent is not None
    assert (ref_time - dt_recent).total_seconds() == 7200
    assert 0 <= (ref_time - dt_recent).total_seconds() <= 86400

    # 2. Story from 26 hours ago -> DROP (> 86400s)
    entry_old = {"published": "Thu, 05 Nov 2026 10:00:00 +0545"}
    dt_old = parse_article_date(entry_old, "")
    assert dt_old is not None
    assert (ref_time - dt_old).total_seconds() > 86400

    # 3. UTC timestamp converted to Asia/Kathmandu (+05:45)
    entry_utc = {"published": "Fri, 06 Nov 2026 04:15:00 +0000"}
    dt_utc = parse_article_date(entry_utc, "")
    assert dt_utc is not None
    assert dt_utc.tzinfo == KATHMANDU_TZ
    # 04:15 UTC is 10:00 NPT
    assert dt_utc.hour == 10
    assert dt_utc.minute == 0

    # 4. Undated story -> returns None, scraper will preserve with published_at = null
    entry_undated = {}
    assert parse_article_date(entry_undated, "https://sample.com/story-without-date") is None

    # 5. Date embedded in URL structure (e.g. Kathmandu Post /YYYY/MM/DD/)
    entry_url_date = {}
    dt_url = parse_article_date(entry_url_date, "https://kathmandupost.com/national/2026/11/06/sample-story")
    assert dt_url is not None
    assert dt_url.year == 2026
    assert dt_url.month == 11
    assert dt_url.day == 6


# -----------------------------------------------------------------------------
# 3. Fallback to RSS Summary when Article Extraction Fails
# -----------------------------------------------------------------------------
def test_fallback_to_rss_summary_when_extraction_fails(short_article_html, tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)

    # Mock response for article URL that returns short text (<200 chars)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = short_article_html

    with patch.object(scraper, "fetch_with_retry", return_value=mock_resp):
        rss_entry = {
            "summary": "<p>This is a sufficiently long RSS summary that should be used as fallback when page extraction fails. It contains valuable context about the ongoing conference in Pokhara.</p>"
        }
        article = scraper._extract_article(
            source_name="Sample Outlet",
            title="Pokhara Tourism Summit Concludes",
            url="https://sample.nepalnews.com/news/pokhara-summit",
            published_at="2026-11-06T11:00:00+05:45",
            rss_entry=rss_entry,
            user_agent="NepalNewsBot/0.1",
            delay_seconds=0,
        )

    assert article is not None
    assert "Pokhara Tourism Summit Concludes" in article["title"]
    assert "Pokhara" in article["text_summary"]
    assert "Pokhara" in article["full_text"]
    assert article["word_count"] > 10
    assert article["language"] == "en"


# -----------------------------------------------------------------------------
# 4. Broken Source Does Not Fail the Whole Run
# -----------------------------------------------------------------------------
def test_broken_source_does_not_fail_whole_run(sample_feed_xml, sample_article_html, tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)

    # Mock config with 2 sources: Source A (offline 500 error), Source B (healthy RSS)
    mock_sources_cfg = {
        "settings": {
            "user_agent": "NepalNewsBot/0.1",
            "request_delay_seconds": 0,
            "timezone": "Asia/Kathmandu",
        },
        "sources": [
            {
                "name": "Broken Source",
                "homepage": "https://broken.news.com",
                "rss": "https://broken.news.com/feed",
                "method": "rss",
                "robots_ok": True,
                "enabled": True,
            },
            {
                "name": "Healthy Source",
                "homepage": "https://sample.nepalnews.com",
                "rss": "https://sample.nepalnews.com/feed",
                "method": "rss",
                "robots_ok": True,
                "enabled": True,
            },
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        if "robots.txt" in url:
            resp.status_code = 200
            resp.content = b"User-agent: *\nAllow: /\n"
            resp.text = "User-agent: *\nAllow: /\n"
            return resp
        elif "broken.news.com" in url:
            resp.status_code = 500
            resp.content = b"Internal Server Error"
            resp.text = "Internal Server Error"
            return resp
        elif "sample.nepalnews.com/feed" in url:
            resp.status_code = 200
            resp.content = sample_feed_xml
            resp.text = sample_feed_xml.decode("utf-8")
            return resp
        else:
            resp.status_code = 200
            resp.content = sample_article_html.encode("utf-8")
            resp.text = sample_article_html
            return resp

    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(target_date="2026-11-06", backfill=True)

    # Broken source recorded error
    assert "Broken Source" in errors
    assert "Failed to fetch RSS feed" in errors["Broken Source"]

    # Healthy source succeeded
    assert "Healthy Source" in stats
    assert len(articles) > 0
    assert all(a["source"] == "Healthy Source" for a in articles)


# -----------------------------------------------------------------------------
# 5. articles.json Validates Against Schema
# -----------------------------------------------------------------------------
def test_articles_schema_validation(sample_feed_xml, sample_article_html, tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)

    mock_sources_cfg = {
        "settings": {"user_agent": "NepalNewsBot/0.1", "request_delay_seconds": 0},
        "sources": [
            {
                "name": "Sample News",
                "homepage": "https://sample.nepalnews.com",
                "rss": "https://sample.nepalnews.com/feed",
                "method": "rss",
                "robots_ok": True,
                "enabled": True,
            }
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        if "/feed" in url:
            resp.status_code = 200
            resp.content = sample_feed_xml
            resp.text = sample_feed_xml.decode("utf-8")
        else:
            resp.status_code = 200
            resp.content = sample_article_html.encode("utf-8")
            resp.text = sample_article_html
        return resp

    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, _, _ = scraper.scrape_all(target_date="2026-11-06", backfill=True)

    assert len(articles) >= 2
    assert all(a["language"] == "en" for a in articles)
    # Verify each article satisfies contract schema
    is_valid, err = validate_contract(articles, scraper.schema_contract_path)
    assert is_valid is True, f"Schema validation error: {err}"


# -----------------------------------------------------------------------------
# 6. Robots-Disallowed Paths Are Skipped
# -----------------------------------------------------------------------------
def test_robots_disallowed_paths_skipped(sample_robots_txt, tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)

    # Provide robots parser with mock robots.txt
    mock_rp = MagicMock()
    mock_rp.can_fetch.side_effect = lambda ua, url: False if "disallowed-article" in url or "admin" in url else True
    scraper.robot_parsers["sample.nepalnews.com"] = mock_rp

    assert scraper._is_url_allowed_by_robots("https://sample.nepalnews.com/disallowed-article/123", "NepalNewsBot/0.1", 0) is False
    assert scraper._is_url_allowed_by_robots("https://sample.nepalnews.com/news/allowed-article", "NepalNewsBot/0.1", 0) is True


# -----------------------------------------------------------------------------
# 7. Morning and Evening Live Scrape Window & Backfill Anchoring
# -----------------------------------------------------------------------------
def test_morning_and_evening_scrape_window(tmp_path):
    """Verify morning and evening runs correctly apply the 30-hour lookback window."""
    scraper = ScraperAgent(run_dir=tmp_path)

    # 1. Morning run at 08:00 on 2026-11-07 (+05:45)
    morning_ref = datetime(2026, 11, 7, 8, 0, 0, tzinfo=KATHMANDU_TZ)

    # Article from 22:00 previous evening (10 hours ago) -> KEEP
    entry_evening_before = {"published": "Fri, 06 Nov 2026 22:00:00 +0545"}
    dt_10h = parse_article_date(entry_evening_before, "")
    assert dt_10h is not None
    diff_morning_keep = (morning_ref - dt_10h).total_seconds()
    assert diff_morning_keep == 10 * 3600
    assert 0 <= diff_morning_keep <= 30 * 3600

    # Article from 32 hours ago (00:00 on Nov 6) -> DROP (> 30 hours)
    entry_old_32h = {"published": "Fri, 06 Nov 2026 00:00:00 +0545"}
    dt_32h = parse_article_date(entry_old_32h, "")
    assert dt_32h is not None
    diff_morning_drop = (morning_ref - dt_32h).total_seconds()
    assert diff_morning_drop == 32 * 3600
    assert diff_morning_drop > 30 * 3600

    # 2. Evening run at 20:00 on 2026-11-07 (+05:45)
    evening_ref = datetime(2026, 11, 7, 20, 0, 0, tzinfo=KATHMANDU_TZ)

    # Article from 14:00 today (6 hours ago) -> KEEP
    entry_afternoon = {"published": "Sat, 07 Nov 2026 14:00:00 +0545"}
    dt_6h = parse_article_date(entry_afternoon, "")
    assert dt_6h is not None
    diff_evening_keep = (evening_ref - dt_6h).total_seconds()
    assert diff_evening_keep == 6 * 3600
    assert 0 <= diff_evening_keep <= 30 * 3600

    # Article from yesterday 16:00 (28 hours ago) -> KEEP (within 30h)
    entry_yesterday_16 = {"published": "Fri, 06 Nov 2026 16:00:00 +0545"}
    dt_28h = parse_article_date(entry_yesterday_16, "")
    assert dt_28h is not None
    diff_28h = (evening_ref - dt_28h).total_seconds()
    assert diff_28h == 28 * 3600
    assert 0 <= diff_28h <= 30 * 3600

    # Article from yesterday 12:00 (32 hours ago) -> DROP (> 30h)
    entry_yesterday_12 = {"published": "Fri, 06 Nov 2026 12:00:00 +0545"}
    dt_evening_old = parse_article_date(entry_yesterday_12, "")
    assert dt_evening_old is not None
    diff_evening_old = (evening_ref - dt_evening_old).total_seconds()
    assert diff_evening_old == 32 * 3600
    assert diff_evening_old > 30 * 3600


def test_backfill_flag_anchoring(tmp_path):
    """Verify that backfill=True anchors to 23:59:59 of target date, while backfill=False uses live scrape time."""
    # With backfill=True
    scraper_backfill = ScraperAgent(run_dir=tmp_path, backfill=True)
    assert scraper_backfill.backfill is True

    # With backfill=False (default)
    scraper_live = ScraperAgent(run_dir=tmp_path)
    assert scraper_live.backfill is False


def test_articles_schema_validation_negative_cases():
    """Negative test proving articles.schema.json rejects missing fields and invalid types."""
    # 1. Missing required field 'url'
    bad_articles_missing_url = [
        {
            "source": "Sample Outlet",
            "title": "Valid Headline",
            "full_text": "Sample story full text content.",
        }
    ]
    is_valid, err = validate_contract(bad_articles_missing_url, "contracts/articles.schema.json")
    assert is_valid is False
    assert "'url' is a required property" in err

    # 2. Not an array (top-level must be array)
    bad_type = {"source": "Not an array"}
    is_valid, err = validate_contract(bad_type, "contracts/articles.schema.json")
    assert is_valid is False
    assert "is not of type 'array'" in err


# -----------------------------------------------------------------------------
# 8. Remediation Plan Fixes (Truncation, Summary, SSL, Date Clamp, HTML & Parallel)
# -----------------------------------------------------------------------------
def test_truncate_at_sentence_boundary():
    sentence = "Nepal Electricity Authority expanded transmission capacity across Gandaki Province. "
    long_text = sentence * 90  # > 6000 chars
    truncated = truncate_at_sentence_boundary(long_text, max_chars=6000)
    assert len(truncated) <= 6000
    assert truncated.endswith(".")

    # Fallback without punctuation must still not split mid-word
    words_only = ("hydropower " * 700).strip()
    truncated_words = truncate_at_sentence_boundary(words_only, max_chars=6000)
    assert len(truncated_words) <= 6000
    assert truncated_words.endswith("hydropower")


def test_build_text_summary_ignores_abbreviations():
    lead = ("A" * 495) + " "
    body = lead + "Dr. Sharma said Rs. 500 million was allocated by U.S. partners for the highway. The project completes next year."
    summary = build_text_summary(body)
    assert "Rs. 500 million" in summary
    assert summary.endswith("for the highway.")


def test_fetch_with_retry_handles_ssl_error(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    with patch("src.agents.scraper.requests.get", side_effect=requests.exceptions.SSLError("cert expired")):
        resp = scraper.fetch_with_retry("https://expired.example.com/feed", "NepalNewsBot/0.1", 0)
    assert resp is None


def test_parse_article_date_url_fallback_morning_clamp():
    morning_ref = datetime(2026, 11, 12, 8, 0, 0, tzinfo=KATHMANDU_TZ)
    dt = parse_article_date({}, "https://kathmandupost.com/national/2026/11/12/budget-update", ref_time=morning_ref)
    assert dt is not None
    assert dt == morning_ref
    assert (morning_ref - dt).total_seconds() == 0


def test_canonical_url_key_scheme_and_www():
    k1 = canonical_url_key("http://www.example.com/news/story-1/?utm_source=rss#top")
    k2 = canonical_url_key("https://example.com/news/story-1")
    assert k1 == k2 == "example.com/news/story-1"


def test_html_source_url_pattern_and_metadata_lookback(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path, cache_dir=tmp_path / "cache")
    homepage_html = """
    <html><body>
      <a href="/political">Political</a>
      <a href="/political/167501">Fresh Story</a>
      <a href="/social/150000">Old Story</a>
    </body></html>
    """
    fresh_article_html = """
    <html><head>
      <meta property="article:published_time" content="2026-11-12T06:00:00+05:45" />
    </head><body><article><p>""" + ("Fresh news content in Kathmandu Valley. " * 12) + """</p></article></body></html>
    """
    old_article_html = """
    <html><head>
      <meta property="article:published_time" content="2026-10-01T06:00:00+05:45" />
    </head><body><article><p>""" + ("Old archived news content in Pokhara. " * 12) + """</p></article></body></html>
    """

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/political/167501"):
            resp.text = fresh_article_html
        elif url.endswith("/social/150000"):
            resp.text = old_article_html
        else:
            resp.text = homepage_html
        return resp

    source_cfg = {
        "name": "Setopati English",
        "homepage": "https://en.setopati.com",
        "method": "html_only",
        "max_articles": 10,
        "include_url_patterns": [r"^/[a-zA-Z0-9_-]+/\d+$"],
    }
    ref_time = datetime(2026, 11, 12, 12, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
        with patch.object(scraper, "_is_url_allowed_by_robots", return_value=True):
            articles, stats, err = scraper._fetch_html_source(
                source_cfg,
                "NepalNewsBot/0.1",
                0,
                ref_time,
                set(),
                [],
                lookback_seconds=30 * 3600,
            )

    assert err is None
    assert stats["found"] == 2  # /political category link excluded
    assert len(articles) == 1
    assert articles[0]["url"].endswith("/political/167501")
    assert stats["drop_reasons"]["lookback"] == 1


def test_parallel_scrape_all_preserves_cross_source_and_order(sample_article_html, tmp_path):
    scraper = ScraperAgent(
        run_dir=tmp_path,
        config={"execution": {"allow_parallel": True}},
        cache_dir=tmp_path / "cache",
    )
    feed_a = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>A</title>
      <item><title>10 killed in electric microbus crash in Sindhuli</title>
      <link>https://a.news.com/1</link>
      <pubDate>Thu, 12 Nov 2026 07:00:00 +0545</pubDate></item>
    </channel></rss>"""
    feed_b = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>B</title>
      <item><title>10 killed as EV plunges 70m in Sindhuli</title>
      <link>https://b.news.com/1</link>
      <pubDate>Thu, 12 Nov 2026 07:30:00 +0545</pubDate></item>
    </channel></rss>"""

    mock_sources_cfg = {
        "settings": {"user_agent": "NepalNewsBot/0.1", "request_delay_seconds": 0},
        "sources": [
            {"name": "Source A", "rss": "https://a.news.com/feed", "method": "rss", "enabled": True, "robots_ok": True},
            {"name": "Source B", "rss": "https://b.news.com/feed", "method": "rss", "enabled": True, "robots_ok": True},
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        if "a.news.com/feed" in url:
            resp.content = feed_a
            resp.text = feed_a.decode("utf-8")
        elif "b.news.com/feed" in url:
            resp.content = feed_b
            resp.text = feed_b.decode("utf-8")
        else:
            resp.content = sample_article_html.encode("utf-8")
            resp.text = sample_article_html
        return resp

    ref_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert not errors
    assert [a["source"] for a in articles] == ["Source A", "Source B"]
    assert len(articles) == 2  # Cross-source coverage preserved for EditorAgent clustering


# -----------------------------------------------------------------------------
# 9. v2 Enrichment Features (Category, Importance, Missing Dates, Trust, Nepali)
# -----------------------------------------------------------------------------
def test_infer_category_hint_tiers():
    # Tier 1: RSS tags (skipping editorial priority tags like 'Cover'/'Exclusive')
    assert infer_category_hint("https://ok.com/story", "News", "Body", raw_tags=["Cover", "Accident"]) == "disaster"
    assert infer_category_hint("https://rajdhanidaily.com/id/151315/", "समाचार", "विवरण", raw_tags=["Exclusive", "विदेश"]) == "world"

    # Tier 2: URL section slug
    assert infer_category_hint("https://nagariknews.nagariknetwork.com/sports/karnali-105.html", "कर्णाली १०५", "खेल") == "sports"
    assert infer_category_hint("https://kathmandupost.com/province-no-2/2026/11/12/vote", "Vote", "Summary") == "politics"

    # Tier 3: Keyword fallback (Devanagari & English)
    assert infer_category_hint("https://example.com/news/123", "अर्थतन्त्र सुधार", "बजेट र बैंक ब्याजदर") == "economy"


def test_compute_importance_hint_position_and_tags():
    imp_top_cover = compute_importance_hint(1, raw_tags=["Cover"])
    imp_top_plain = compute_importance_hint(1, raw_tags=[])
    imp_mid_plain = compute_importance_hint(5, raw_tags=[])
    imp_low_plain = compute_importance_hint(20, raw_tags=[])

    assert imp_top_cover > imp_top_plain > imp_mid_plain > imp_low_plain
    assert 0.0 <= imp_low_plain <= 1.0
    assert imp_top_cover >= 0.90


def test_undated_feed_position_gate_and_metadata_recovery(sample_article_html, tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path, cache_dir=tmp_path / "cache")
    # Build an RSS feed with 7 undated entries:
    # - item 1: no RSS date, no HTML date -> kept (pos 1 <= 2), date_confidence='missing'
    # - item 2: no RSS date, HTML metadata has stale date -> dropped (lookback)
    # - item 3: no RSS date, pos 3 > max_undated_feed_position=2 -> dropped (undated_low_pos)
    feed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Undated Test</title>
      <item><title>Top Undated Story</title><link>https://u.news.com/story/1</link></item>
      <item><title>Stale Metadata Story</title><link>https://u.news.com/story/2</link></item>
      <item><title>Deep Undated Story</title><link>https://u.news.com/story/3</link></item>
    </rss>"""

    undated_html = """<html><head><title>Top Undated Story</title></head>
    <body><article><p>""" + ("Long enough undated breaking article text in Nepal. " * 12) + """</p></article></body></html>"""

    stale_html = """<html><head><meta property="article:published_time" content="2026-10-01T10:00:00+05:45" /></head>
    <body><article><p>""" + ("Long enough archived article text in Nepal. " * 12) + """</p></article></body></html>"""

    mock_sources_cfg = {
        "settings": {
            "user_agent": "NepalNewsBot/0.1",
            "request_delay_seconds": 0,
            "max_undated_feed_position": 2,
            "default_trust_weight": 0.70,
        },
        "sources": [
            {
                "name": "Undated Outlet",
                "rss": "https://u.news.com/feed",
                "method": "rss",
                "enabled": True,
                "robots_ok": True,
                "trust_weight": 0.85,
            }
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {}
        if url.endswith("/feed"):
            resp.content = feed_xml
            resp.text = feed_xml.decode("utf-8")
        elif url.endswith("/story/2"):
            resp.text = stale_html
        else:
            resp.text = undated_html
        return resp

    ref_time = datetime(2026, 11, 12, 12, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert not errors
    assert len(articles) == 1
    art = articles[0]
    assert art["url"] == "https://u.news.com/story/1"
    assert art["date_confidence"] == "missing"
    assert art["feed_position"] == 1
    assert art["source_trust"] == 0.85
    src_stats = stats["Undated Outlet"]
    assert src_stats["undated_kept"] == 1
    assert src_stats["drop_reasons"]["lookback"] == 1
    assert src_stats["drop_reasons"]["undated_low_pos"] == 1

    is_valid, err = validate_contract(articles, scraper.schema_contract_path)
    assert is_valid is True, f"Schema validation error: {err}"


def test_nepali_clustering_and_devanagari_number_validation():
    articles = [
        {
            "source": "Rajdhani Daily",
            "title": "भीमफेदी इभी दुर्घटना अपडेट: १४ जनाको मृत्यु",
            "url": "https://rajdhanidaily.com/id/151317/",
            "published_at": "2026-11-12T12:34:59+05:45",
            "text_summary": "मकवानपुरको भीमफेदीमा इभी दुर्घटना हुँदा १४ जनाको मृत्यु भएको छ।",
            "full_text": ("मकवानपुरको भीमफेदीमा इभी दुर्घटना हुँदा १४ जनाको मृत्यु भएको छ र ६ जना घाइते छन्। " * 8),
            "language": "ne",
            "category_hint": "disaster",
            "importance_hint": 0.95,
            "source_trust": 0.75,
            "date_confidence": "exact",
        },
        {
            "source": "Nagarik News",
            "title": "मकवानपुरको भीमफेदीमा इभी दुर्घटना, १४ जनाको मृत्यु",
            "url": "https://nagariknews.nagariknetwork.com/social/ev-crash-14.html",
            "published_at": "2026-11-12T13:00:00+05:45",
            "text_summary": "भीमफेदी गाउँपालिकामा भएको इभी दुर्घटनामा १४ यात्रुको ज्यान गएको प्रहरीले जनाएको छ।",
            "full_text": ("भीमफेदी गाउँपालिकामा भएको इभी दुर्घटनामा १४ यात्रुको ज्यान गएको प्रहरीले जनाएको छ। " * 8),
            "language": "ne",
            "category_hint": "disaster",
            "importance_hint": 0.60,
            "source_trust": 0.85,
            "date_confidence": "exact",
        },
    ]

    clusters = cluster_articles(articles, {"clustering": {"min_full_text_words": 20}, "categories": {}})
    assert len(clusters) == 1
    assert clusters[0]["source_count"] == 2
    assert clusters[0]["category_guess"] == "disaster"
    # Member articles sorted by source_trust descending (Nagarik News 0.85 before Rajdhani 0.75)
    assert clusters[0]["articles"][0]["source"] == "Nagarik News"

    # Devanagari digits in Nepali source text normalize cleanly to numeric values
    nums = extract_numbers_from_text("भीमफेदी इभी दुर्घटनामा १४ जनाको मृत्यु, ६ घाइते")
    assert 14.0 in nums
    assert 6.0 in nums


# =============================================================================
# 10. Refinement Plan Verification Tests (T1.1 - T3.5)
# =============================================================================

def test_thin_story_rejected_at_ingest(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    feed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>News</title>
      <item>
        <title>Quick Update On Roads</title>
        <link>https://example.com/road-update</link>
        <pubDate>Thu, 12 Nov 2026 07:00:00 +0545</pubDate>
      </item>
    </channel></rss>"""
    short_html = (
        "<html><body><article><p>"
        "Traffic resumed after a major landslide was cleared early this morning on the Prithvi Highway near Mugling. "
        "Authorities confirmed that heavy excavators worked overnight to remove large boulders blocking both lanes, "
        "allowing passenger buses and cargo trucks to proceed safely toward Kathmandu."
        "</p></article></body></html>"
    )

    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0},
        "sources": [{"name": "ShortNews", "rss": "https://example.com/feed", "method": "rss", "enabled": True}],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif "feed" in url:
            resp.text = feed_xml.decode("utf-8")
            resp.content = feed_xml
        else:
            resp.text = short_html
            resp.content = short_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert len(articles) == 0
    assert stats["ShortNews"]["drop_reasons"]["thin_story"] >= 1


def test_advertorial_rejected_at_ingest(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    feed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>News</title>
      <item>
        <title>Sponsored: Special Promotional Offer On Vehicles</title>
        <link>https://example.com/sponsored/deal</link>
        <pubDate>Thu, 12 Nov 2026 07:00:00 +0545</pubDate>
      </item>
    </channel></rss>"""
    promo_html = "<html><body><article><p>" + ("Sponsored content: Click here to buy and claim your promotional discount today with special offers and limited deals. " * 8) + "</p></article></body></html>"

    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0},
        "sources": [{"name": "PromoNews", "rss": "https://example.com/feed", "method": "rss", "enabled": True}],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif "feed" in url:
            resp.text = feed_xml.decode("utf-8")
            resp.content = feed_xml
        else:
            resp.text = promo_html
            resp.content = promo_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert len(articles) == 0
    assert stats["PromoNews"]["drop_reasons"]["advertorial"] >= 1


def test_robots_checked_on_rss_and_article(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    robots_text = "User-agent: *\nDisallow: /disallowed-feed\nDisallow: /blocked-article\n"
    with patch.object(scraper, "_get_robot_parser") as mock_rp:
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(robots_text.splitlines())
        mock_rp.return_value = (True, rp, None)
        can_feed, _ = scraper._check_robots_for_url("https://example.com/disallowed-feed", "Bot", 0)
        can_allowed, _ = scraper._check_robots_for_url("https://example.com/allowed-article", "Bot", 0)
        can_blocked, _ = scraper._check_robots_for_url("https://example.com/blocked-article", "Bot", 0)
        assert can_feed is False
        assert can_allowed is True
        assert can_blocked is False


def test_robots_failure_fails_closed(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "robots_policy": "fail_closed", "request_delay_seconds": 0},
        "sources": [{"name": "BlockedSource", "rss": "https://blocked.example.com/rss", "method": "rss", "enabled": True}],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        if "robots.txt" in url:
            resp.status_code = 500
            resp.text = "Server Error"
            resp.content = b"Server Error"
            return resp
        resp.status_code = 200
        return resp

    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(target_date="2026-11-15")

    assert len(articles) == 0
    assert "BlockedSource" in errors
    assert "ROBOTS_UNKNOWN" in errors["BlockedSource"]
    assert stats["BlockedSource"]["robots_unknown"] is True
    assert stats["BlockedSource"]["drop_reasons"]["robots_unavailable"] >= 1


def test_robots_fallback_flag_overrides(tmp_path, sample_feed_xml, sample_article_html):
    scraper = ScraperAgent(run_dir=tmp_path, allow_robots_fallback=True)
    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "robots_policy": "fail_closed", "request_delay_seconds": 0},
        "sources": [{"name": "FallbackSource", "rss": "https://fallback.example.com/rss", "method": "rss", "enabled": True}],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        if "robots.txt" in url:
            resp.status_code = 500
            resp.text = "Server Error"
            resp.content = b"Server Error"
            return resp
        elif "rss" in url:
            resp.status_code = 200
            resp.content = sample_feed_xml
            resp.text = sample_feed_xml.decode("utf-8")
            return resp
        else:
            resp.status_code = 200
            resp.content = sample_article_html.encode("utf-8")
            resp.text = sample_article_html
            return resp

    ref_time = datetime(2026, 11, 6, 12, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time, backfill=True)

    assert len(articles) > 0
    assert "FallbackSource" in stats


def test_429_respects_retry_after(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    url = "https://rate-limited.com/article"
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "3"}
    resp_429.text = "Too Many Requests"

    with patch.object(scraper, "_get_session") as mock_session_getter:
        sess = MagicMock()
        sess.get.return_value = resp_429
        mock_session_getter.return_value = sess
        res = scraper.fetch_with_retry(url, "Bot/1.0", 0.0)

    assert res.status_code == 429
    assert "rate-limited.com" in scraper._domain_cooldowns
    assert scraper._domain_cooldowns["rate-limited.com"] > 0


def test_degraded_pass_widens_lookback(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    manifest = {
        "date": "2026-11-15",
        "stages": {"scraper": {"status": "pending", "metrics": {}, "source_errors": {}}},
    }
    sample_art = {
        "source": "Sample",
        "title": "Story",
        "url": "https://sample.com/1",
        "published_at": "2026-11-14T10:00:00+05:45",
        "word_count": 100,
        "text_summary": "summary",
        "full_text": "full text " * 20,
    }
    call_counts = []

    def mock_scrape_all(lookback_hours=None, **kwargs):
        call_counts.append(lookback_hours)
        arts = [dict(sample_art, url=f"https://sample.com/{i}", title=f"Story {i}") for i in range(4)]
        return arts, {"Sample": {"found": 4, "extracted_ok": 4, "drop_reasons": {}}}, {}

    with patch.object(scraper, "scrape_all", side_effect=mock_scrape_all):
        outputs = scraper.run(manifest)

    assert "raw/articles.json" in outputs
    assert len(call_counts) == 2
    assert call_counts == [30, 48]
    arts_on_disk = json.loads((tmp_path / "raw" / "articles.json").read_text(encoding="utf-8"))
    assert all(a.get("degraded_run") is True for a in arts_on_disk)
    m_metrics = manifest["stages"]["scraper"]["metrics"]
    assert m_metrics.get("degraded") is True
    news_md = (tmp_path / "raw" / "news.md").read_text(encoding="utf-8")
    assert "DEGRADED INGEST" in news_md


def test_access_outage_is_hard_failure_not_degraded(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    manifest = {
        "date": "2026-11-15",
        "stages": {"scraper": {"status": "pending", "metrics": {}, "source_errors": {}}},
    }

    def mock_scrape_all(**kwargs):
        stats = {
            "Source1": {"found": 0, "extracted_ok": 0, "robots_unknown": True, "drop_reasons": {"robots_unavailable": 1}},
            "Source2": {"found": 0, "extracted_ok": 0, "drop_reasons": {"rate_limited": 1}},
        }
        errors = {"Source1": "ROBOTS_UNKNOWN", "Source2": "HTTP 429"}
        return [], stats, errors

    with patch.object(scraper, "scrape_all", side_effect=mock_scrape_all):
        with pytest.raises(RuntimeError, match="ACCESS_OUTAGE"):
            scraper.run(manifest)


def test_summary_skips_boilerplate_lead():
    body_with_lead = (
        "KATHMANDU, Nov 15 (RSS) — The government has announced a new infrastructure budget.\n\n"
        "Follow us on Facebook and Twitter for more updates. Subscribe to our newsletter.\n\n"
        "The Ministry of Finance allocated sixty billion rupees for national pride road projects across all seven provinces. "
        "The project will span four years and connect remote mountainous districts to the national highway network."
    )
    summary = build_text_summary(body_with_lead)
    assert "KATHMANDU, Nov 15 (RSS) —" not in summary
    assert "Follow us on Facebook" not in summary
    assert "The government has announced a new infrastructure budget" in summary or "The Ministry of Finance allocated" in summary


def test_jsonld_extraction_fallback():
    jsonld_html = """
    <!DOCTYPE html>
    <html>
    <head>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "NewsArticle",
        "headline": "Special Cabinet Decision On Trade Policy",
        "articleBody": "The Cabinet of Ministers endorsed the updated bilateral trade agreement with neighboring nations during an emergency meeting at the Prime Minister residence in Baluwatar. The deal aims to streamline cargo clearance and reduce transit costs.",
        "datePublished": "2026-11-15T08:00:00+05:45"
      }
      </script>
    </head>
    <body>
      <div><!-- No standard article paragraph tags --></div>
    </body>
    </html>
    """
    body, meta = extract_jsonld_article(jsonld_html)
    assert "The Cabinet of Ministers endorsed" in body
    assert meta.get("headline") == "Special Cabinet Decision On Trade Policy"
    assert meta.get("published_at") == "2026-11-15T08:00:00+05:45"


def test_cross_source_same_event_deduped(tmp_path, sample_article_html):
    scraper = ScraperAgent(run_dir=tmp_path)
    feed_a = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>A</title>
      <item><title>10 killed in electric microbus crash in Sindhuli</title>
      <link>https://a.news.com/1</link>
      <pubDate>Thu, 12 Nov 2026 07:00:00 +0545</pubDate></item>
    </channel></rss>"""
    feed_b = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>B</title>
      <item><title>10 killed as EV plunges 70m in Sindhuli</title>
      <link>https://b.news.com/1</link>
      <pubDate>Thu, 12 Nov 2026 07:30:00 +0545</pubDate></item>
    </channel></rss>"""

    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0, "dedupe_mode": "url+title+token"},
        "sources": [
            {"name": "Source A", "rss": "https://a.news.com/feed", "method": "rss", "enabled": True, "trust": 0.85},
            {"name": "Source B", "rss": "https://b.news.com/feed", "method": "rss", "enabled": True, "trust": 0.70},
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif "a.news.com/feed" in url:
            resp.text = feed_a.decode("utf-8")
            resp.content = feed_a
        elif "b.news.com/feed" in url:
            resp.text = feed_b.decode("utf-8")
            resp.content = feed_b
        else:
            resp.text = sample_article_html
            resp.content = sample_article_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    # Cross-source coverage is preserved so Editor clustering can group multi-outlet reporting
    assert len(articles) == 2
    sources = {a["source"] for a in articles}
    assert sources == {"Source A", "Source B"}


def test_nepali_article_marked_needs_translation(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    art_ne = scraper._extract_article(
        url="https://rajdhanidaily.com/id/151315/",
        html_text="<html><body><article><p>" + ("काठमाडौंमा नयाँ योजना सार्वजनिक गरिएको छ। " * 10) + "</p></article></body></html>",
        source={"name": "Rajdhani Daily", "language": "ne", "trust": 0.75},
        user_agent="Bot/1.0",
        delay_seconds=0.0,
        published_at="2026-11-15T10:00:00+05:45",
        title="काठमाडौंमा नयाँ योजना",
    )
    assert art_ne is not None
    assert art_ne["language"] == "ne"
    assert art_ne["needs_translation"] is True
    assert art_ne["translation_status"] == "pending"

    art_en = scraper._extract_article(
        url="https://kathmandupost.com/story/1",
        html_text="<html><body><article><p>" + ("The Kathmandu Valley witnessed steady economic growth this quarter. " * 8) + "</p></article></body></html>",
        source={"name": "Kathmandu Post", "language": "en", "trust": 0.85},
        user_agent="Bot/1.0",
        delay_seconds=0.0,
        published_at="2026-11-15T10:00:00+05:45",
        title="Kathmandu Valley Economic Growth",
    )
    assert art_en is not None
    assert art_en["language"] == "en"
    assert art_en["needs_translation"] is False
    assert art_en["translation_status"] == "not_required"


def test_devanagari_number_grounding_after_translation():
    source_ne = "दुर्घटनामा ५ जनाको मृत्यु भएको छ भने १० जना घाइते छन्।"
    nums_source = extract_numbers_from_text(source_ne)
    assert 5.0 in nums_source
    assert 10.0 in nums_source

    narration_en = "Five people were killed and 10 injured in the accident."
    nums_narration = extract_numbers_from_text(narration_en)
    assert 5.0 in nums_narration
    assert 10.0 in nums_narration
    assert set(nums_narration).issubset(set(nums_source))


def test_priority_score_prefers_recency():
    score_fresh = compute_priority_score(age_hours=2.0, feed_position=1, raw_tags=["politics"], source_trust=0.8)
    score_old = compute_priority_score(age_hours=28.0, feed_position=1, raw_tags=["politics"], source_trust=0.8)
    assert score_fresh > score_old
    assert score_fresh - score_old >= 0.20


def test_budget_exhaustion_stops_gracefully(tmp_path, sample_article_html):
    scraper = ScraperAgent(run_dir=tmp_path)
    feed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>News</title>
      <item><title>Story 1</title><link>https://example.com/1</link><pubDate>Sun, 15 Nov 2026 07:00:00 +0545</pubDate></item>
      <item><title>Story 2</title><link>https://example.com/2</link><pubDate>Sun, 15 Nov 2026 07:00:00 +0545</pubDate></item>
    </channel></rss>"""

    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "budget_seconds": 0.001, "request_delay_seconds": 0},
        "sources": [{"name": "BudgetSource", "rss": "https://example.com/feed", "method": "rss", "enabled": True}],
    }

    def mock_fetch(url, *args, **kwargs):
        import time
        time.sleep(0.01)
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif "feed" in url:
            resp.text = feed_xml.decode("utf-8")
            resp.content = feed_xml
        else:
            resp.text = sample_article_html
            resp.content = sample_article_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 15, 10, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert stats["BudgetSource"]["budget_exhausted"] is True


def test_html_predate_filter_avoids_deep_fetch(tmp_path, sample_article_html):
    scraper = ScraperAgent(run_dir=tmp_path)
    listing_html = """
    <html><body>
      <div class="card">
        <a href="/2026/11/15/recent-story">Recent Story</a>
        <time datetime="2026-11-15T08:00:00+05:45">Nov 15, 2026</time>
      </div>
      <div class="card">
        <a href="/2024/05/01/old-story">Old Story From Two Years Ago</a>
        <time datetime="2024-05-01T08:00:00+05:45">May 1, 2024</time>
      </div>
    </body></html>
    """
    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0},
        "sources": [{"name": "HtmlSource", "homepage": "https://example.com", "method": "html", "enabled": True}],
    }
    deep_fetched_urls = []

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif url == "https://example.com" or url == "https://example.com/":
            resp.text = listing_html
            resp.content = listing_html.encode("utf-8")
        else:
            deep_fetched_urls.append(url)
            resp.text = sample_article_html
            resp.content = sample_article_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 15, 12, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert any("recent-story" in u for u in deep_fetched_urls)
    assert not any("old-story" in u for u in deep_fetched_urls)


def test_html_dup_title_checked_before_fetch(tmp_path, sample_article_html):
    scraper = ScraperAgent(run_dir=tmp_path)
    listing_html = """
    <html><body>
      <a href="/link1">Government Announces New Highway Plan</a>
      <a href="/link2">Government Announces New Highway Plan</a>
    </body></html>
    """
    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0},
        "sources": [{"name": "HtmlDupSource", "homepage": "https://example.com", "method": "html", "enabled": True}],
    }
    fetched_articles = []

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif url == "https://example.com" or url == "https://example.com/":
            resp.text = listing_html
            resp.content = listing_html.encode("utf-8")
        else:
            fetched_articles.append(url)
            resp.text = sample_article_html
            resp.content = sample_article_html.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 15, 12, 0, 0, tzinfo=KATHMANDU_TZ)
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(ref_time=ref_time)

    assert len(fetched_articles) == 1
    assert stats["HtmlDupSource"]["drop_reasons"]["dup_title"] >= 1


def test_circuit_breaker_skips_dead_source(tmp_path):
    scraper = ScraperAgent(run_dir=tmp_path)
    state_file = tmp_path / "source_health.json"
    scraper._circuit_breaker_file = state_file

    scraper._record_circuit_breaker_outcome("DeadSource", kept=0, error="ERROR")
    scraper._record_circuit_breaker_outcome("DeadSource", kept=0, error="EMPTY")
    scraper._record_circuit_breaker_outcome("DeadSource", kept=0, error="ROBOTS_UNKNOWN")

    tripped = scraper._is_circuit_breaker_tripped("DeadSource")
    assert tripped is True

    scraper.force_source = "DeadSource"
    tripped_force = scraper._is_circuit_breaker_tripped("DeadSource")
    assert tripped_force is False


def test_parse_category_input_and_aliases():
    from src.agents.scraper import parse_category_input

    assert parse_category_input("1,5") == ["headlines", "health"]
    assert parse_category_input("politics, international") == ["politics", "international"]
    assert parse_category_input("all") == ["all"]
    assert parse_category_input("A") == ["all"]
    assert parse_category_input("") == ["all"]
    assert parse_category_input("8") == ["sports"]
    assert parse_category_input("health, sports") == ["health", "sports"]


def test_article_matches_categories():
    from src.agents.scraper import article_matches_categories

    # 1. All categories
    assert article_matches_categories("sports", selected_categories={"all"}) is True

    # 2. Headlines matching
    assert article_matches_categories("politics", feed_position=1, selected_categories={"headlines"}) is True
    assert article_matches_categories("politics", feed_position=5, raw_tags=["Exclusive", "Breaking"], selected_categories={"headlines"}) is True
    assert article_matches_categories("politics", feed_position=6, importance_hint=0.85, selected_categories={"headlines"}) is True
    assert article_matches_categories("politics", feed_position=8, importance_hint=0.3, title="Routine meeting", selected_categories={"headlines"}) is False

    # 3. Health matching
    assert article_matches_categories("health", selected_categories={"health"}) is True
    assert article_matches_categories("society", raw_tags=["hospital"], selected_categories={"health"}) is False

    # 4. International matching
    assert article_matches_categories("world", selected_categories={"international"}) is True
    assert article_matches_categories("international", selected_categories={"international"}) is True
    assert article_matches_categories("sports", selected_categories={"international"}) is False


def test_automated_category_fallback():
    from src.agents.scraper import prompt_category_selection

    # In automated / non-interactive mode, always returns ["headlines"]
    selected = prompt_category_selection(interactive=False)
    assert selected == ["headlines"]


def test_scraper_filters_by_category(tmp_path):
    import zoneinfo
    from src.agents.scraper import ScraperAgent

    scraper = ScraperAgent(run_dir=tmp_path, categories=["health"])
    feed_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>HealthNews</title>
      <item>
        <title>New hospital wing opens in Lalitpur with modern ICU</title>
        <link>https://health.news.com/icu-wing</link>
        <pubDate>Thu, 12 Nov 2026 07:00:00 +0545</pubDate>
        <category>Health</category>
        <description>The state hospital inaugurated a forty bed ICU facility.</description>
      </item>
      <item>
        <title>Parliament passes regular administrative procedures amendment</title>
        <link>https://health.news.com/parliament-bill</link>
        <pubDate>Thu, 12 Nov 2026 07:15:00 +0545</pubDate>
        <category>Politics</category>
        <description>Lawmakers approved the administrative guidelines without debate.</description>
      </item>
    </channel></rss>"""

    mock_sources_cfg = {
        "settings": {"user_agent": "Bot/1.0", "request_delay_seconds": 0},
        "sources": [
            {"name": "HealthSource", "rss": "https://health.news.com/feed", "method": "rss", "enabled": True, "trust": 0.85},
        ],
    }

    def mock_fetch(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "robots.txt" in url:
            resp.text = "User-agent: *\nAllow: /\n"
            resp.content = resp.text.encode("utf-8")
        elif "health.news.com/feed" in url:
            resp.text = feed_xml.decode("utf-8")
            resp.content = feed_xml
        else:
            p_text = "Detailed article body content describing the health development in comprehensive fashion for readers across Nepal. " * 8
            resp.text = f"<html><body><article><p>{p_text}</p></article></body></html>"
            resp.content = resp.text.encode("utf-8")
        return resp

    ref_time = datetime(2026, 11, 12, 10, 0, 0, tzinfo=zoneinfo.ZoneInfo("Asia/Kathmandu"))
    with patch.object(scraper, "_load_sources_config", return_value=mock_sources_cfg):
        with patch.object(scraper, "fetch_with_retry", side_effect=mock_fetch):
            articles, stats, errors = scraper.scrape_all(categories=["health"], ref_time=ref_time)

    # Health article should be kept, parliament politics article dropped by category
    assert len(articles) == 1
    assert "hospital wing" in articles[0]["title"].lower()
    assert stats["HealthSource"]["drop_reasons"]["category_filtered"] >= 1


def test_infer_category_hint_no_substring_false_positives():
    """Keywords must match whole words (e.g. 'league' must not match 'colleagues', 'sport' not 'passport')."""
    editor_cats = {
        "crime": ["crime", "police", "investigation"],
        "sports": ["sports", "cricket", "football", "league", "match"],
        "technology": ["technology", "ai", "cyber"],
    }
    # Regression: 'colleagues' previously matched 'league' twice and 'failed' matched 'ai'
    cat = infer_category_hint(
        url="https://kathmandupost.com/national/2026/11/18/police-inspector-found-dead-at-official-residence-in-gulmi",
        title="Police inspector found dead at official residence in Gulmi",
        text_summary="The officer was found unresponsive in his bed after colleagues broke open a window when he failed to wake up.",
        raw_tags=[],
        editor_categories=editor_cats,
    )
    assert cat == "crime"

    # URL slug 'passport' must not trigger 'sport'
    cat_passport = infer_category_hint(
        url="https://example.com/passport/renewal-process",
        title="Department eases passport renewal rules",
        text_summary="Applicants can now submit documents online.",
        raw_tags=["Passport"],
        editor_categories=editor_cats,
    )
    assert cat_passport != "sports"



"""Unit and integration tests for the Scraper Agent using local fixtures."""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest
import zoneinfo

from src.agents.scraper import (
    ScraperAgent,
    clean_title,
    is_duplicate_title,
    normalize_language_code,
    normalize_url,
    parse_article_date,
    KATHMANDU_TZ,
)
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
        if "broken.news.com" in url:
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
            articles, stats, errors = scraper.scrape_all(target_date="2026-11-06")

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
            articles, _, _ = scraper.scrape_all(target_date="2026-11-06")

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

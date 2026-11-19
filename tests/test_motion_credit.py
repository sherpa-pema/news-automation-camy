"""Unit tests for source credit sanitization and QA attribution enforcement."""
from __future__ import annotations

import pytest

from src.utils.overlay_plan import sanitize_credit


def test_sanitize_credit_single_source_string() -> None:
    res = sanitize_credit({"source": "Setopati English"})
    assert res == "Source: Setopati English"


def test_sanitize_credit_comma_split() -> None:
    res = sanitize_credit({"source": "The Kathmandu Post, Setopati English"})
    assert res == "Source: The Kathmandu Post, Setopati English"


def test_sanitize_credit_and_split() -> None:
    res = sanitize_credit({"source": "Kathmandu Post and Setopati English"})
    assert res == "Source: Kathmandu Post, Setopati English"


def test_sanitize_credit_dedup_and_punctuation() -> None:
    res = sanitize_credit({"source": "Setopati English., Setopati English, Onlinekhabar..."})
    assert res == "Source: Setopati English, Onlinekhabar"


def test_sanitize_credit_cap_at_three_outlets() -> None:
    res = sanitize_credit({"source": "Outlet One, Outlet Two, Outlet Three, Outlet Four, Outlet Five"})
    assert res == "Source: Outlet One, Outlet Two, Outlet Three"


def test_sanitize_credit_fallback_to_sources_list() -> None:
    res = sanitize_credit({"sources": ["Nepal Rastra Bank", "Kantipur"]})
    assert res == "Source: Nepal Rastra Bank, Kantipur"


def test_sanitize_credit_fallback_to_channel_name() -> None:
    res = sanitize_credit({}, channel_name="Nepal Daily Briefing")
    assert res == "Nepal Daily Briefing"

    res_empty = sanitize_credit({"source": ""}, channel_name="Custom Channel")
    assert res_empty == "Custom Channel"


def test_sanitize_credit_truncation() -> None:
    long_source = "Extremely Long Outlet Name Number One, Very Long Outlet Name Number Two, Another Super Long Name"
    res = sanitize_credit({"source": long_source})
    assert len(res) <= 48
    assert res.endswith("…")
    assert res.startswith("Source: Extremely Long Outlet Name")


def test_sanitize_credit_removes_duplicate_source_prefix() -> None:
    res = sanitize_credit({"source": "Source: Setopati English"})
    assert res == "Source: Setopati English"
    assert not res.startswith("Source: Source:")

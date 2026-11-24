"""Scraper Agent implementation for nepal-news-bot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import difflib
import email.utils
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
import urllib.robotparser
import zoneinfo

from bs4 import BeautifulSoup
import dateparser
import feedparser
import requests
from requests.adapters import HTTPAdapter
import trafilatura
import yaml

from src.agents.base import BaseAgent
from src.utils.editor_cluster import (
    _compile_word_boundary_re,
    extract_canonical_entities,
    is_advertisement,
    is_thin_story,
    tokenize,
)
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest, update_stage_status
from src.utils.nepali_date import parse_nepali_date, today_npt
from src.utils.office_events import emit_office_event
from src.utils.schema_validate import validate_contract

KATHMANDU_TZ = zoneinfo.ZoneInfo("Asia/Kathmandu")

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "ref_src",
    "_ga",
    "action_object_map",
    "action_type_map",
    "action_ref_map",
}


def normalize_url(url: str) -> str:
    """Normalizes an article URL by removing tracking query parameters and fragments."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    # Filter tracking parameters
    query_tuples = parse_qsl(parsed.query, keep_blank_values=False)
    filtered_queries = [(k, v) for k, v in query_tuples if k.lower() not in TRACKING_QUERY_PARAMS]
    clean_query = urlencode(filtered_queries)

    # Clean path: strip trailing slash unless path is just '/'
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        parsed.params,
        clean_query,
        "",  # Drop fragment
    ))
    return normalized


def canonical_url_key(url: str) -> str:
    """Returns a scheme- and www-agnostic deduplication key for an article URL."""
    norm = normalize_url(url)
    if not norm:
        return ""
    parsed = urlparse(norm)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    key = f"{netloc}{parsed.path}"
    if parsed.query:
        key = f"{key}?{parsed.query}"
    return key


ABBREV_END_RE = re.compile(
    r"(?:\b(?:rs|dr|mr|mrs|ms|prof|hon|gen|col|capt|sgt|lt|sr|jr|st|no|vol|fig|dept|govt|approx|est|vs|etc|eg|ie)|(?:[a-z]\.){1,}[a-z])\.$",
    re.IGNORECASE,
)


def _is_abbreviation_boundary(text: str, dot_pos: int) -> bool:
    """Returns True if the period at dot_pos is part of a known abbreviation or initialism."""
    if text[dot_pos] != ".":
        return False
    prefix = text[max(0, dot_pos - 20) : dot_pos + 1]
    return bool(ABBREV_END_RE.search(prefix))


def truncate_at_sentence_boundary(text: str, max_chars: int = 6000) -> str:
    """Truncates text at or below max_chars on a clean sentence/paragraph/word boundary."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    window_start = int(max_chars * 0.70)
    candidate_slice = stripped[:max_chars]

    # Prefer the last clean sentence boundary inside [0.70 * max_chars, max_chars]
    last_clean_end: int | None = None
    for m in re.finditer(r"[.!?।॥](?=\s|$)", candidate_slice[window_start:]):
        abs_pos = window_start + m.start()
        if not _is_abbreviation_boundary(candidate_slice, abs_pos):
            last_clean_end = window_start + m.end()

    if last_clean_end is not None:
        return candidate_slice[:last_clean_end].strip()

    # Next prefer a paragraph break inside the window
    para_idx = candidate_slice.rfind("\n\n", window_start)
    if para_idx != -1:
        return candidate_slice[:para_idx].strip()

    # Fallback: never split mid-word
    parts = candidate_slice.rsplit(None, 1)
    if len(parts) > 1 and parts[0]:
        return parts[0].strip()
    return candidate_slice.strip()


BOILERPLATE_PATTERNS = [
    r"\bsubscribe\b",
    r"\bsign\s*up\b",
    r"\bcookies?\b",
    r"\bto continue reading\b",
    r"\badvertisement\b",
    r"\bdownload our app\b",
    r"\ball rights reserved\b",
    r"\bterms of (?:use|service)\b",
    r"\bprivacy policy\b",
    r"सदस्यता",
    r"विज्ञापन",
    r"साइन\s*अप",
    r"एप डाउनलोड",
    r"सर्वाधिकार सुरक्षित",
]


def is_boilerplate(text: str) -> bool:
    """Checks whether text matches common cookie, subscription, or advertorial boilerplate."""
    t_low = text.lower()
    return any(bool(re.search(pat, t_low)) for pat in BOILERPLATE_PATTERNS)


def strip_datelines_and_bylines(para: str) -> str:
    """Strips common news datelines (e.g. 'KATHMANDU, Nov 12 —') and bylines from the lead."""
    cleaned = para.strip()
    cleaned = re.sub(r"^[A-Z\s,()]{2,45}(?:,\s*[A-Za-z\s\d(),]{1,30})?\s*[-—–]\s*", "", cleaned)
    cleaned = re.sub(r"^[A-Z][A-Za-z\s,().\d]{0,40}\s+[-—–]\s+", "", cleaned)
    cleaned = re.sub(r"^[\u0900-\u097F\s,()]{2,45}\s*[-—–]\s*", "", cleaned)
    cleaned = re.sub(r"^(?:by\s+[A-Za-z\s]{1,40}[-—–]?\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?i)\s*(?:read more|continue reading|थप पढ्नुहोस्|सम्बन्धित समाचार).*$", "", cleaned)
    return cleaned.strip()


def _truncate_summary_slice(
    text: str,
    min_cut: int = 500,
    max_cut: int = 650,
    fallback_max: int = 600,
) -> str:
    if len(text) <= fallback_max:
        return text

    search_region = text[min_cut:max_cut]
    for m in re.finditer(r"[.!?।॥](?=\s|$)", search_region):
        abs_pos = min_cut + m.start()
        if not _is_abbreviation_boundary(text, abs_pos):
            return text[: min_cut + m.end()].strip()

    parts = text[:fallback_max].rsplit(None, 1)
    if len(parts) > 1 and parts[0]:
        return parts[0].rstrip() + "..."
    return text[:fallback_max].rstrip() + "..."


def build_text_summary(
    full_text: str,
    min_cut: int = 500,
    max_cut: int = 650,
    fallback_max: int = 600,
    rss_entry: dict[str, Any] | None = None,
) -> str:
    """Extracts a high-quality lead summary (~600 chars), skipping boilerplate and datelines."""
    # 1. Feed summary / content:encoded when >= 120 chars and not boilerplate
    if rss_entry:
        raw_sum = rss_entry.get("summary") or rss_entry.get("description") or ""
        if not raw_sum and "content" in rss_entry and isinstance(rss_entry["content"], list):
            for c in rss_entry["content"]:
                if isinstance(c, dict) and c.get("value"):
                    raw_sum = c["value"]
                    break
        if raw_sum:
            soup = BeautifulSoup(raw_sum, "html.parser")
            clean_rss = clean_text_content(soup.get_text(separator=" ", strip=True))
            if len(clean_rss) >= 120 and not is_boilerplate(clean_rss):
                clean_rss = strip_datelines_and_bylines(clean_rss)
                if len(clean_rss) >= 60:
                    return _truncate_summary_slice(clean_rss, min_cut, max_cut, fallback_max)

    # 2. First clean paragraph of full_text after stripping datelines and bylines
    paras = [p.strip() for p in full_text.split("\n") if p.strip()]
    for p in paras:
        if is_boilerplate(p):
            continue
        clean_p = strip_datelines_and_bylines(p)
        if len(clean_p) >= 40 and not is_boilerplate(clean_p):
            return _truncate_summary_slice(clean_p, min_cut, max_cut, fallback_max)

    # 3. Truncation fallback
    return _truncate_summary_slice(full_text, min_cut, max_cut, fallback_max)


def compute_priority_score(
    age_hours: float | None,
    feed_position: int | None,
    raw_tags: list[str] | None,
    source_trust: float,
    priority_tags: list[str] | set[str] | None = None,
) -> float:
    """Computes a 0.0-1.0 priority score balancing recency, feed position, tags, and source trust.

    Formula: 0.45 * recency + 0.30 * position + 0.15 * tags + 0.10 * trust
    where recency = exp(-age_hours / 12) (undated = 0.5).
    """
    if age_hours is not None:
        recency = math.exp(-max(0.0, float(age_hours)) / 12.0)
    else:
        recency = 0.5

    pos = feed_position if (feed_position is not None and feed_position >= 1) else 10
    pos_score = 1.0 / math.log2(max(1, pos) + 1.0)

    tags_score = 0.0
    if raw_tags and priority_tags:
        p_set = {t.lower().strip() for t in priority_tags if str(t).strip()}
        for rt in raw_tags:
            rt_low = rt.lower().strip()
            if rt_low in p_set or any(p in rt_low for p in p_set):
                tags_score = 1.0
                break

    trust_score = max(0.0, min(1.0, float(source_trust)))
    raw_score = 0.45 * recency + 0.30 * pos_score + 0.15 * tags_score + 0.10 * trust_score
    return round(max(0.0, min(1.0, raw_score)), 3)


def clean_title(title: str) -> str:
    """Normalizes a headline for duplicate detection."""
    t = re.sub(r"[^\w\s]", "", title.lower())
    return " ".join(t.split())


def normalize_language_code(lang: str | None) -> str:
    """Normalizes a language string to ISO 639-1 two-letter code ('en', 'ne')."""
    if not lang:
        return "en"
    clean = lang.strip().lower()
    if clean in ("english", "en", "en-us", "en-gb"):
        return "en"
    if clean in ("nepali", "ne", "nep"):
        return "ne"
    return clean[:2]


def is_duplicate_title(title: str, existing_titles: list[str], threshold: float = 0.9) -> bool:
    """Checks if a title is nearly identical to any previously ingested title."""
    target_clean = clean_title(title)
    if not target_clean:
        return False
    for existing in existing_titles:
        ex_clean = clean_title(existing)
        ratio = difflib.SequenceMatcher(None, target_clean, ex_clean).ratio()
        if ratio >= threshold:
            return True
    return False


DEFAULT_PRIORITY_TAGS = {
    "cover",
    "breaking news",
    "breaking",
    "top story",
    "top stories",
    "exclusive",
    "trending",
    "featured",
    "मुख्य समाचार",
    "ब्रेकिङ",
}

AVAILABLE_CATEGORIES: dict[str, tuple[str, str, list[str]]] = {
    "1": ("headlines", "Headlines & Breaking News", ["breaking", "top stories", "cover", "exclusive"]),
    "2": ("politics", "National Politics", ["politics", "government", "parliament", "elections", "rajniti"]),
    "3": ("society_local", "Society & Local News", ["society", "local", "education", "transport", "disaster", "crime", "weather", "environment"]),
    "4": ("business_economics", "Business & Economics", ["business", "economy", "market", "finance", "trade"]),
    "5": ("health", "Health & Medicine", ["health", "hospital", "medical", "disease", "vaccine"]),
    "6": ("culture_entertainment", "Culture & Entertainment", ["culture", "festival", "heritage", "cinema", "arts", "tourism"]),
    "7": ("science_technology", "Science & Technology", ["technology", "telecom", "cyber", "ai", "digital", "science"]),
    "8": ("sports", "Sports", ["sports", "cricket", "football", "athletics", "tournament"]),
    "9": ("international", "International News", ["international", "world", "foreign", "global", "diplomacy"]),
}

CATEGORY_ALIAS_MAP: dict[str, set[str]] = {
    "headlines": {"headlines", "breaking", "top_stories", "cover", "exclusive"},
    "politics": {"politics", "national_politics", "political", "government", "parliament"},
    "national_politics": {"politics", "national_politics", "political", "government", "parliament"},
    "society_local": {"society_local", "society", "local", "education", "transport", "disaster", "crime", "weather", "environment", "climate"},
    "society": {"society_local", "society"},
    "local": {"society_local", "local"},
    "transport": {"society_local", "transport"},
    "disaster": {"society_local", "disaster"},
    "crime": {"society_local", "crime"},
    "weather": {"society_local", "weather"},
    "environment": {"society_local", "environment"},
    "climate": {"society_local", "climate"},
    "business_economics": {"business_economics", "economy", "business", "market", "finance"},
    "economy": {"business_economics", "economy"},
    "business": {"business_economics", "economy", "business"},
    "health": {"health", "medical"},
    "culture_entertainment": {"culture_entertainment", "culture", "entertainment", "tourism"},
    "culture": {"culture_entertainment", "culture"},
    "entertainment": {"culture_entertainment", "entertainment"},
    "tourism": {"culture_entertainment", "tourism"},
    "science_technology": {"science_technology", "technology", "science", "tech"},
    "technology": {"science_technology", "technology", "tech"},
    "science": {"science_technology", "science"},
    "sports": {"sports", "sport", "cricket", "football", "athletics"},
    "sport": {"sports", "sport"},
    "international": {"international", "world", "foreign", "global"},
    "world": {"international", "world", "foreign", "global"},
}

TAG_CATEGORY_MAP = {
    "politics": "politics",
    "political": "politics",
    "rajniti": "politics",
    "राजनीति": "politics",
    "प्रदेश": "politics",
    "economy": "economy",
    "economic": "economy",
    "business": "economy",
    "market": "economy",
    "money": "economy",
    "finance": "economy",
    "अर्थ": "economy",
    "अर्थतन्त्र": "economy",
    "बजार": "economy",
    "disaster": "disaster",
    "accident": "disaster",
    "earthquake": "disaster",
    "flood": "disaster",
    "landslide": "disaster",
    "विपद्": "disaster",
    "दुर्घटना": "disaster",
    "weather": "weather",
    "मौसम": "weather",
    "climate": "climate",
    "environment": "environment",
    "climate-environment": "environment",
    "वातावरण": "environment",
    "health": "health",
    "स्वास्थ्य": "health",
    "hospital": "health",
    "medical": "health",
    "medicine": "health",
    "disease": "health",
    "transport": "transport",
    "यातायात": "transport",
    "crime": "crime",
    "अपराध": "crime",
    "tourism": "tourism",
    "पर्यटन": "tourism",
    "technology": "technology",
    "tech": "technology",
    "science": "technology",
    "प्रविधि": "technology",
    "society": "society",
    "social": "society",
    "education": "society",
    "culture": "culture",
    "cultural": "culture",
    "संस्कृति": "culture",
    "कला": "entertainment",
    "साहित्य": "culture",
    "cinema": "entertainment",
    "film": "entertainment",
    "movie": "entertainment",
    "चलचित्र": "entertainment",
    "समाज": "society",
    "शिक्षा": "society",
    "sports": "sports",
    "sport": "sports",
    "cricket": "sports",
    "football": "sports",
    "खेलकुद": "sports",
    "खेल": "sports",
    "world": "world",
    "international": "world",
    "foreign": "world",
    "global": "world",
    "diplomacy": "world",
    "विदेश": "world",
    "अन्तर्राष्ट्रिय": "world",
    "entertainment": "entertainment",
    "मनोरञ्जन": "entertainment",
}

GENERIC_URL_SEGMENTS = {
    "news",
    "story",
    "stories",
    "id",
    "nepal",
    "kathmandu",
    "national",
    "article",
    "articles",
    "post",
    "detail",
    "en",
    "ne",
}

EXTRA_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "politics": [
        "सरकार", "संसद", "मन्त्री", "प्रधानमन्त्री", "निर्वाचन", "दल", "अध्यक्ष", "सांसद", "प्रदेश", "मन्त्रिपरिषद्",
    ],
    "economy": [
        "अर्थतन्त्र", "बजेट", "बैंक", "सेयर", "नेप्से", "राजस्व", "कर", "व्यापार", "निर्यात", "आयात", "मुद्रास्फीति",
    ],
    "disaster": [
        "दुर्घटना", "मृत्यु", "घाइते", "पहिरो", "बाढी", "भूकम्प", "आगलागी", "इभी", "accident", "killed", "injured",
    ],
    "weather": ["वर्षा", "मौसम", "हिमपात", "तापक्रम", "चिसो", "गर्मी", "हावाहुरी"],
    "health": [
        "health", "hospital", "doctor", "patient", "disease", "vaccine", "vaccination", "medicine", "medical",
        "diabetes", "blood pressure", "hypertension", "maternity", "heart", "cardiac", "trauma", "respiratory",
        "ebola", "rabies", "encephalitis", "dengue", "cancer", "clinic", "nurse", "nutrition", "mental health",
        "tuberculosis", "malaria", "surgery", "blood bank", "micronutrients", "energy drinks", "food safety",
        "स्वास्थ्य", "अस्पताल", "चिकित्सक", "बिरामी", "खोप", "रोग", "संक्रमण", "औषधि",
    ],
    "crime": ["प्रहरी", "पक्राउ", "ठगी", "भ्रष्टाचार", "तस्करी", "अनुसन्धान", "मुद्दा", "अदालत"],
    "sports": [
        "sports", "cricket", "football", "soccer", "basketball", "volleyball", "badminton",
        "athletics", "match", "matches", "tournament", "league", "championship", "playoff",
        "qualifier", "semifinal", "semi-final", "quarter-final", "stadium", "npl", "ipl",
        "world cup", "gold cup", "hat-trick", "wicket", "wickets", "batting", "bowling",
        "goal", "goals", "medal",
        "क्रिकेट", "फुटबल", "खेल", "खेलकुद", "खेलाडी", "रन", "विकेट", "लिग", "प्रतियोगिता", "जित",
    ],
    "world": [
        "iran", "ukraine", "russia", "china", "india", "usa", "united states", "europe", "gaza", "israel",
        "अमेरिका", "भारत", "चीन", "रुस", "युक्रेन", "विदेश", "अन्तर्राष्ट्रिय",
    ],
    "international": [
        "iran", "ukraine", "russia", "china", "india", "usa", "united states", "europe", "gaza", "israel", "foreign", "global",
        "अमेरिका", "भारत", "चीन", "रुस", "युक्रेन", "विदेश", "अन्तर्राष्ट्रिय", "संयुक्त राष्ट्रसंघ",
    ],
    "tourism": ["पर्यटन", "पर्यटक", "पर्वतारोहण", "होटल", "विमानस्थल"],
    "technology": ["प्रविधि", "इन्टरनेट", "साइबर", "डिजिटल", "सफ्टवेयर"],
    "society": ["शिक्षा", "विद्यालय", "विश्वविद्यालय", "समुदाय", "स्थानीय"],
    "culture": ["संस्कृति", "पर्व", "चाड", "दशैं", "तिहार", "छठ", "सम्पदा", "साहित्य"],
    "entertainment": ["मनोरञ्जन", "चलचित्र", "फिल्म", "सिनेमा", "कलाकार", "गीत", "संगीत"],
}


def extract_entry_tags(entry: Any) -> list[str]:
    """Extracts deduplicated RSS category/tag strings from a feedparser entry."""
    if not entry:
        return []
    raw = entry.get("tags") if isinstance(entry, dict) else getattr(entry, "tags", None)
    if not raw:
        return []
    seen: set[str] = set()
    tags: list[str] = []
    for item in raw:
        term = item.get("term") if isinstance(item, dict) else getattr(item, "term", None)
        if not term and isinstance(item, dict):
            term = item.get("label")
        if term:
            clean = str(term).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                tags.append(clean)
    return tags


def compute_importance_hint(
    feed_position: int | None,
    raw_tags: list[str] | None = None,
    priority_tags: set[str] | list[str] | None = None,
) -> float:
    """Computes a 0.0-1.0 importance hint from feed position decay and priority tags."""
    if feed_position is not None and feed_position >= 1:
        pos_score = 0.60 * math.exp(-0.12 * (feed_position - 1))
    else:
        pos_score = 0.15

    prio_set = {t.strip().lower() for t in (priority_tags or DEFAULT_PRIORITY_TAGS) if str(t).strip()}
    matched = 0
    for tag in raw_tags or []:
        t_low = tag.strip().lower()
        if t_low in prio_set or any(p in t_low for p in prio_set):
            matched += 1

    tag_boost = 0.0
    if matched == 1:
        tag_boost = 0.35
    elif matched >= 2:
        tag_boost = 0.40

    return round(min(1.0, pos_score + tag_boost), 3)


def _keyword_matches(kw: str, text_lower: str) -> bool:
    """Matches ASCII keywords on word boundaries to prevent substring false positives (e.g. 'league' in 'colleagues')."""
    kw_l = kw.strip().lower()
    if not kw_l:
        return False
    if kw_l.isascii():
        return bool(_compile_word_boundary_re(kw_l).search(text_lower))
    return kw_l in text_lower


def infer_category_hint(
    url: str,
    title: str,
    text_summary: str,
    raw_tags: list[str] | None = None,
    editor_categories: dict[str, list[str]] | None = None,
) -> str:
    """Infers a rough canonical topic category from RSS tags, URL section, and keywords."""
    # Tier 1: RSS tags
    for tag in raw_tags or []:
        t_low = tag.strip().lower()
        if t_low in DEFAULT_PRIORITY_TAGS or t_low in ("exclusive", "breaking news", "cover"):
            continue
        if t_low in TAG_CATEGORY_MAP:
            return TAG_CATEGORY_MAP[t_low]
        for key, cat in TAG_CATEGORY_MAP.items():
            if _keyword_matches(key, t_low):
                return cat

    # Tier 2: URL section slug
    if url:
        path_parts = [p.lower() for p in urlparse(url).path.split("/") if p]
        for seg in path_parts[:2]:
            if seg.isdigit() or seg in GENERIC_URL_SEGMENTS:
                continue
            if seg.startswith("province-"):
                return "politics"
            if seg in TAG_CATEGORY_MAP:
                return TAG_CATEGORY_MAP[seg]
            for key, cat in TAG_CATEGORY_MAP.items():
                if _keyword_matches(key, seg):
                    return cat

    # Tier 3: Keyword scoring (deduplicated editor.yaml categories + Nepali/sports/world extensions)
    combined = f"{title} {text_summary}".lower()
    merged_kws: dict[str, set[str]] = {}
    for cat, kws in (editor_categories or {}).items():
        merged_kws.setdefault(cat, set()).update(str(k).strip().lower() for k in kws if str(k).strip())
    for cat, kws in EXTRA_CATEGORY_KEYWORDS.items():
        merged_kws.setdefault(cat, set()).update(str(k).strip().lower() for k in kws if str(k).strip())

    best_cat = "other"
    best_score = 0
    for cat, kws in merged_kws.items():
        score = sum(1 for kw in kws if _keyword_matches(kw, combined))
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


def article_matches_categories(
    category_hint: str | None,
    raw_tags: list[str] | None = None,
    feed_position: int | None = None,
    importance_hint: float | None = None,
    title: str = "",
    selected_categories: set[str] | None = None,
) -> bool:
    """Checks whether an article matches the user-selected categories."""
    if not selected_categories or "all" in selected_categories:
        return True

    # 1. Check headlines
    if "headlines" in selected_categories:
        if feed_position is not None and feed_position <= 3:
            return True
        if importance_hint is not None and importance_hint >= 0.8:
            return True
        for t in raw_tags or []:
            t_low = t.lower()
            if t_low in DEFAULT_PRIORITY_TAGS or "break" in t_low or "exclusive" in t_low:
                return True
        title_low = title.lower()
        if any(kw in title_low for kw in ("breaking", "urgent", "ताजा", "विशेष", "मुख्य समाचार")):
            return True

    # 2. Check category hint & aliases
    cat = (category_hint or "other").strip().lower()
    for sel in selected_categories:
        if sel == "headlines":
            continue
        allowed_aliases = CATEGORY_ALIAS_MAP.get(sel, {sel})
        if cat in allowed_aliases:
            return True
        # Also check raw tags against aliases
        for t in raw_tags or []:
            if t.lower() in allowed_aliases:
                return True

    return False


def parse_category_input(user_input: str) -> list[str]:
    """Parses user input string into canonical category keys."""
    raw = user_input.strip().lower()
    if not raw or raw in ("a", "all"):
        return ["all"]

    tokens = [t.strip() for t in re.split(r"[,;\s]+", raw) if t.strip()]
    selected: list[str] = []
    num_to_cat = {k: v[0] for k, v in AVAILABLE_CATEGORIES.items()}

    for tok in tokens:
        if tok in num_to_cat:
            cat = num_to_cat[tok]
            if cat not in selected:
                selected.append(cat)
        elif tok in ("a", "all"):
            return ["all"]
        else:
            matched = False
            for slug, label, _ in AVAILABLE_CATEGORIES.values():
                if tok == slug or tok in label.lower():
                    if slug not in selected:
                        selected.append(slug)
                    matched = True
                    break
            if not matched and tok:
                if tok not in selected:
                    selected.append(tok)

    return selected or ["headlines"]


def prompt_category_selection(
    category_counts: dict[str, int] | None = None,
    interactive: bool | None = None,
    target_date: str | None = None,
) -> list[str]:
    """Presents category menu to user, falling back to headlines in automated mode."""
    is_auto = False
    if interactive is False:
        is_auto = True
    elif interactive is None:
        is_auto = not sys.stdin.isatty() or bool(os.getenv("PYTEST_CURRENT_TEST"))

    if is_auto:
        print("[AUTOMATED] Automated run detected: falling back to Headlines & Breaking News.")
        return ["headlines"]

    print("\n" + "=" * 65)
    print(f" Nepal News Bot — Category Selection ({target_date or 'Today'})")
    print("=" * 65)
    print("Select categories to scrape:")
    for num, (slug, label, _) in AVAILABLE_CATEGORIES.items():
        count_str = f" ({category_counts.get(slug, 0)} items)" if category_counts else ""
        print(f"  [{num}] {label:<26}{count_str}")
    print("  [A] All Categories")
    print("-" * 65)
    print("Enter choice (e.g. 1,2,5 or 'politics,health', [Enter] = Headlines): ", end="", flush=True)

    try:
        choice = sys.stdin.readline()
        if not choice:
            print("\n[AUTOMATED] No input received: falling back to Headlines & Breaking News.")
            return ["headlines"]
        res = parse_category_input(choice)
        print(f"-> Selected: {', '.join(res)}\n")
        return res
    except Exception:
        print("\n[AUTOMATED] Input error: falling back to Headlines & Breaking News.")
        return ["headlines"]


def parse_article_date_with_confidence(
    entry: dict[str, Any],
    url: str,
    ref_time: datetime | None = None,
) -> tuple[datetime | None, str]:
    """Extracts publication datetime (Asia/Kathmandu) and returns (dt, confidence)."""
    raw_date = None
    # 1. Check feed entry fields
    for key in ("published", "pubDate", "updated", "dc:date"):
        if entry.get(key):
            raw_date = str(entry[key])
            break

    dt = None
    confidence = "missing"
    if raw_date:
        # Feedparser sometimes provides parsed struct_time
        if entry.get("published_parsed"):
            try:
                st = entry["published_parsed"]
                dt_utc = datetime(*st[:6], tzinfo=timezone.utc)
                return dt_utc.astimezone(KATHMANDU_TZ), "exact"
            except Exception:
                pass

        # Check Bikram Sambat date first
        bs_d = parse_nepali_date(raw_date)
        if bs_d is not None:
            dt = datetime(bs_d.year, bs_d.month, bs_d.day, 12, 0, 0, tzinfo=KATHMANDU_TZ)
            confidence = "exact"
        else:
            try:
                dt = dateparser.parse(raw_date)
                if dt is not None:
                    # If parsed as a future AD date that falls into BS range (2070-2095), convert from BS
                    if 2070 <= dt.year <= 2095:
                        converted = parse_nepali_date(f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}")
                        if converted:
                            dt = datetime(converted.year, converted.month, converted.day, dt.hour, dt.minute, dt.second, tzinfo=KATHMANDU_TZ)
                    confidence = "exact"
            except Exception:
                dt = None

    # 2. Check URL for /YYYY/MM/DD/ pattern (e.g. Kathmandu Post)
    if dt is None and url:
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if match:
            try:
                year, month, day = map(int, match.groups())
                if 2070 <= year <= 2095:
                    converted = parse_nepali_date(f"{year:04d}-{month:02d}-{day:02d}")
                    if converted:
                        dt = datetime(converted.year, converted.month, converted.day, 12, 0, 0, tzinfo=KATHMANDU_TZ)
                        confidence = "url_date"
                if dt is None:
                    dt = datetime(year, month, day, 12, 0, 0)
                    confidence = "url_date"
            except Exception:
                dt = None

    if dt is None:
        return None, "missing"

    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KATHMANDU_TZ)
    else:
        dt = dt.astimezone(KATHMANDU_TZ)

    # Prevent fabricated noon timestamp from marking same-day articles as future on morning runs
    if confidence == "url_date" and ref_time is not None:
        ref_ktm = ref_time.astimezone(KATHMANDU_TZ) if ref_time.tzinfo else ref_time.replace(tzinfo=KATHMANDU_TZ)
        if dt.date() == ref_ktm.date() and dt > ref_ktm:
            dt = ref_ktm.replace(microsecond=0)

    return dt, confidence


def parse_article_date(
    entry: dict[str, Any],
    url: str,
    ref_time: datetime | None = None,
) -> datetime | None:
    """Extracts and parses publication date from feed entry or URL structure.

    Returns timezone-aware datetime in Asia/Kathmandu, or None if unparseable.
    """
    dt, _ = parse_article_date_with_confidence(entry, url, ref_time=ref_time)
    return dt


def extract_html_published_at(
    html_text: str,
    ref_time: datetime | None = None,
) -> datetime | None:
    """Attempts to extract an article publication timestamp from HTML metadata."""
    if not html_text:
        return None

    candidates: list[str] = []
    try:
        meta = trafilatura.extract_metadata(html_text)
        if meta and getattr(meta, "date", None):
            candidates.append(str(meta.date))
    except Exception:
        pass

    try:
        soup = BeautifulSoup(html_text, "html.parser")
        meta_specs = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("name", "pubdate"),
            ("name", "publishdate"),
            ("name", "date"),
            ("itemprop", "datePublished"),
        ]
        for attr, val in meta_specs:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                candidates.append(str(tag["content"]))
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag and time_tag.get("datetime"):
            candidates.append(str(time_tag["datetime"]))
    except Exception:
        pass

    for raw in candidates:
        try:
            dt = dateparser.parse(raw)
        except Exception:
            dt = None
        if dt is None:
            continue
        if dt.tzinfo is None:
            # Date-only strings (e.g. '2026-11-12') parse as 00:00:00
            dt = dt.replace(tzinfo=KATHMANDU_TZ)
        else:
            dt = dt.astimezone(KATHMANDU_TZ)
        if ref_time is not None:
            ref_ktm = ref_time.astimezone(KATHMANDU_TZ) if ref_time.tzinfo else ref_time.replace(tzinfo=KATHMANDU_TZ)
            if dt.date() == ref_ktm.date() and dt > ref_ktm:
                dt = ref_ktm.replace(microsecond=0)
        return dt

    return None


def extract_jsonld_article(html: str) -> tuple[str, dict[str, Any]]:
    """Extracts articleBody, headline, author, published date from JSON-LD schema if present."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for script in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
            try:
                raw_content = script.string or script.get_text()
                if not raw_content:
                    continue
                data = json.loads(raw_content)
                candidates = []
                if isinstance(data, list):
                    candidates.extend(data)
                elif isinstance(data, dict):
                    if "@graph" in data and isinstance(data["@graph"], list):
                        candidates.extend(data["@graph"])
                    else:
                        candidates.append(data)
                for item in candidates:
                    if not isinstance(item, dict):
                        continue
                    body = item.get("articleBody") or item.get("text")
                    if body and isinstance(body, str) and len(clean_text_content(body)) >= 50:
                        author_val = None
                        if "author" in item:
                            auth = item["author"]
                            if isinstance(auth, dict):
                                author_val = auth.get("name")
                            elif isinstance(auth, list) and auth and isinstance(auth[0], dict):
                                author_val = auth[0].get("name")
                            elif isinstance(auth, str):
                                author_val = auth
                        meta = {
                            "headline": item.get("headline") or item.get("name"),
                            "author": author_val,
                            "datePublished": item.get("datePublished") or item.get("dateCreated"),
                            "published_at": item.get("datePublished") or item.get("dateCreated"),
                        }
                        return clean_text_content(body), meta
            except Exception:
                continue
    except Exception:
        pass
    return "", {}


def extract_meta_description(html: str) -> str:
    """Extracts meta description or og:description from HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag = (
            soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"})
            or soup.find("meta", attrs={"name": "twitter:description"})
        )
        if tag and tag.get("content"):
            return clean_text_content(str(tag["content"]))
    except Exception:
        pass
    return ""


def clean_text_content(text: str) -> str:
    """Cleans extracted text by normalizing whitespace and line endings."""
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    # Collapse multiple consecutive blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


class ScraperAgent(BaseAgent):
    """Scraper Agent fetches RSS feeds from Nepali sources and extracts articles.

    Outputs:
        - raw/articles.json (validates against contracts/articles.schema.json)
        - raw/news.md
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any] | None = None,
        sources_path: Path | str = "config/sources.yaml",
        pipeline_path: Path | str = "config/pipeline.yaml",
        backfill: bool = False,
        lookback_hours: float | None = None,
        cache_dir: Path | str | None = None,
        allow_robots_fallback: bool = False,
        force_source: str | None = None,
        categories: list[str] | str | None = None,
        interactive: bool | None = None,
    ) -> None:
        super().__init__(run_dir, config)
        self.sources_path = Path(sources_path)
        self.pipeline_path = Path(pipeline_path)
        self.logger = setup_logger("scraper")
        self.last_domain_request: dict[str, float] = {}
        self._domain_locks: dict[str, threading.Lock] = {}
        self._domain_locks_guard = threading.Lock()
        self._robots_lock = threading.Lock()
        self._robots_failed: set[str] = set()
        self.robot_parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.backfill = backfill or (self.config.get("backfill", False) if self.config else False)
        self.lookback_hours = lookback_hours if lookback_hours is not None else (self.config.get("lookback_hours") if self.config else None)
        if categories is not None:
            if isinstance(categories, str):
                self.categories = parse_category_input(categories)
            else:
                self.categories = parse_category_input(",".join(categories))
        elif self.config and self.config.get("categories"):
            cfg_cats = self.config.get("categories")
            if isinstance(cfg_cats, str):
                self.categories = parse_category_input(cfg_cats)
            else:
                self.categories = parse_category_input(",".join(cfg_cats))
        else:
            self.categories = None
        self.interactive = interactive if interactive is not None else (self.config.get("interactive") if self.config else None)
        self.selected_categories: list[str] = self.categories or []
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.repo_root = repo_root
        is_external_run_dir = False
        try:
            is_external_run_dir = self.run_dir and not self.run_dir.resolve().is_relative_to(repo_root.resolve())
        except Exception:
            pass

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        elif is_external_run_dir:
            self.cache_dir = self.run_dir / "cache"
        else:
            self.cache_dir = repo_root / "data" / "cache"

        self._editor_categories_cache: dict[str, list[str]] | None = None
        self._editor_config_cache: dict[str, Any] | None = None
        self._runtime_settings: dict[str, Any] = {}
        self.allow_robots_fallback = allow_robots_fallback or (self.config.get("allow_robots_fallback", False) if self.config else False)
        self.force_source = force_source or (self.config.get("force_source") if self.config else None)
        self._sessions: dict[str, requests.Session] = {}
        self._domain_cooldowns: dict[str, float] = {}
        if is_external_run_dir:
            self._circuit_breaker_file = self.run_dir / "source_health.json"
        else:
            self._circuit_breaker_file = self.repo_root / "data" / "state" / "source_health.json"
        self._circuit_breaker_lock = threading.Lock()
        self._last_http_status: int | None = None
        self._start_time: float | None = None

    def _get_session(self, domain: str) -> requests.Session:
        with self._domain_locks_guard:
            if domain not in self._sessions:
                s = requests.Session()
                adapter = HTTPAdapter(max_retries=0)
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                self._sessions[domain] = s
            return self._sessions[domain]

    def _get_min_full_text_words(self) -> int:
        if self._editor_config_cache is None:
            try:
                ed_path = self.repo_root / "config" / "editor.yaml"
                if ed_path.exists():
                    self._editor_config_cache = yaml.safe_load(ed_path.read_text(encoding="utf-8")) or {}
            except Exception:
                self._editor_config_cache = {}
        return int(
            (self._editor_config_cache or {}).get("clustering", {}).get("min_full_text_words", 60)
        )

    def _load_editor_categories(self) -> dict[str, list[str]]:
        if self._editor_categories_cache is not None:
            return self._editor_categories_cache
        cats: dict[str, list[str]] = {}
        try:
            repo_root = Path(__file__).resolve().parent.parent.parent
            ed_path = repo_root / "config" / "editor.yaml"
            if ed_path.exists():
                data = yaml.safe_load(ed_path.read_text(encoding="utf-8")) or {}
                if isinstance(data.get("categories"), dict):
                    cats = {str(k): list(v or []) for k, v in data["categories"].items()}
        except Exception:
            pass
        self._editor_categories_cache = cats
        return cats

    def _get_domain_lock(self, domain: str) -> threading.Lock:
        with self._domain_locks_guard:
            lock = self._domain_locks.get(domain)
            if lock is None:
                lock = threading.Lock()
                self._domain_locks[domain] = lock
            return lock

    @property
    def name(self) -> str:
        return "scraper"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/articles.schema.json"

    def _load_sources_config(self) -> dict[str, Any]:
        """Loads and returns sources.yaml configuration."""
        path = self.sources_path
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            path = repo_root / path
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_pipeline_config(self) -> dict[str, Any]:
        """Loads pipeline.yaml configuration."""
        path = self.pipeline_path
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent.parent
            path = repo_root / path
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _respect_request_delay(self, url: str, delay_seconds: float) -> None:
        """Enforces polite crawl rate limiting per domain with jitter and rate limit cooldown."""
        domain = urlparse(url).netloc
        if not domain:
            return

        # Check domain-wide 429 rate limit cooldown
        cooldown_until = self._domain_cooldowns.get(domain, 0.0)
        now = time.monotonic()
        if cooldown_until > now:
            wait_time = cooldown_until - now
            self.logger.info(f"Rate limit cooldown active for {domain}: sleeping {wait_time:.1f}s")
            time.sleep(wait_time)

        lock = self._get_domain_lock(domain)
        with lock:
            last_time = self.last_domain_request.get(domain, 0.0)
            elapsed = time.time() - last_time
            jitter_max = float(self._runtime_settings.get("jitter_seconds", 0.5))
            jitter = random.uniform(0.0, jitter_max) if jitter_max > 0 else 0.0
            effective_delay = delay_seconds + jitter
            if elapsed < effective_delay:
                sleep_duration = effective_delay - elapsed
                time.sleep(sleep_duration)
            self.last_domain_request[domain] = time.time()

    def fetch_with_retry(
        self,
        url: str,
        user_agent: str,
        delay_seconds: float,
        timeout: int = 15,
        max_retries: int = 2,
        extra_headers: dict[str, str] | None = None,
        latency_sink: list[float] | None = None,
    ) -> requests.Response | None:
        """Fetches a URL with polite delay, custom headers, and retries for 5xx/429/transient errors.

        Does NOT retry 4xx (except 429 rate limit) or SSL certificate errors.
        """
        headers = {"User-Agent": user_agent}
        if extra_headers:
            headers.update(extra_headers)
        attempt = 0
        backoff = 1.0
        domain = urlparse(url).netloc

        while attempt <= max_retries:
            self._respect_request_delay(url, delay_seconds)
            t0 = time.monotonic()
            try:
                # Respect mocking of requests.get in unit tests
                if getattr(requests.get, "__module__", "").startswith("unittest.mock"):
                    response = requests.get(url, headers=headers, timeout=timeout)
                else:
                    session = self._get_session(domain) if domain else None
                    response = session.get(url, headers=headers, timeout=timeout) if session else requests.get(url, headers=headers, timeout=timeout)

                if latency_sink is not None:
                    latency_sink.append(round((time.monotonic() - t0) * 1000.0, 1))

                self._last_http_status = getattr(response, "status_code", None)

                # 429 Too Many Requests: dedicated rate limit handling
                if response.status_code == 429:
                    self.logger.warning(f"HTTP 429 Too Many Requests for {url}")
                    retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
                    cooldown = 5.0
                    if retry_after:
                        try:
                            cooldown = float(retry_after)
                        except ValueError:
                            try:
                                parsed_d = email.utils.parsedate_to_datetime(retry_after)
                                cooldown = max(0.0, (parsed_d - datetime.now(timezone.utc)).total_seconds())
                            except Exception:
                                cooldown = 5.0
                    cooldown = min(max(cooldown, 1.0), 30.0)
                    if domain:
                        self._domain_cooldowns[domain] = time.monotonic() + cooldown
                    attempt += 1
                    if attempt <= max_retries:
                        time.sleep(cooldown)
                        continue
                    return response

                # 4xx client errors (excluding 429): do not retry
                if 400 <= response.status_code < 500:
                    self.logger.warning(f"Client error HTTP {response.status_code} for {url}. No retry.")
                    return response

                # 5xx server errors: transient, retry with capped backoff
                if response.status_code >= 500:
                    self.logger.warning(f"Server error HTTP {response.status_code} on {url} (attempt {attempt+1}/{max_retries+1})")
                    attempt += 1
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff = min(backoff * 2.0, 10.0)
                        continue
                    return response

                return response

            except requests.exceptions.SSLError as e:
                self.logger.warning(f"SSL/TLS error for {url}: {e}. Skipping without retry.")
                return None

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self.logger.warning(f"Network error ({type(e).__name__}) for {url} (attempt {attempt+1}/{max_retries+1}): {e}")
                attempt += 1
                if attempt <= max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 10.0)
                    continue
                return None
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request error ({type(e).__name__}) fetching {url}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        return None

    def _load_cached_robots(self, domain: str, ttl_seconds: float = 86400.0) -> tuple[bool, str | None, bool]:
        """Returns (hit, body, is_failure) from disk robots cache if within TTL."""
        try:
            safe_domain = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
            cache_file = self.cache_dir / "robots" / f"{safe_domain}.json"
            if not cache_file.exists():
                return False, None, False
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            fetched_at = float(data.get("fetched_at", 0.0))
            is_failure = bool(data.get("failure", False))
            effective_ttl = 1800.0 if is_failure else ttl_seconds
            if (time.time() - fetched_at) <= effective_ttl:
                return True, data.get("body"), is_failure
        except Exception:
            pass
        return False, None, False

    def _save_cached_robots(self, domain: str, body: str | None, failure: bool = False) -> None:
        """Persists robots.txt body to disk cache (fail-safe)."""
        try:
            safe_domain = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
            robots_dir = self.cache_dir / "robots"
            robots_dir.mkdir(parents=True, exist_ok=True)
            cache_file = robots_dir / f"{safe_domain}.json"
            cache_file.write_text(
                json.dumps({"fetched_at": time.time(), "body": body, "failure": failure}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _get_robot_parser(
        self, domain: str, user_agent: str, delay_seconds: float
    ) -> tuple[bool, urllib.robotparser.RobotFileParser | None, str | None]:
        """Retrieves or loads RobotFileParser for domain.

        Returns (can_proceed, parser, error_code).
        Under fail_closed policy, network fetch failures result in (False, None, 'ROBOTS_UNKNOWN').
        """
        robots_policy = self._runtime_settings.get("robots_policy", "fail_closed")

        with self._robots_lock:
            if domain in self.robot_parsers:
                return True, self.robot_parsers[domain], None
            if domain in self._robots_failed and not self.allow_robots_fallback and robots_policy == "fail_closed":
                return False, None, "ROBOTS_UNKNOWN"

        hit, cached_body, is_failure = self._load_cached_robots(domain)
        if hit:
            if not is_failure and cached_body is not None:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(cached_body.splitlines())
                with self._robots_lock:
                    self.robot_parsers[domain] = rp
                return True, rp, None
            elif is_failure:
                if self.allow_robots_fallback or robots_policy != "fail_closed":
                    rp = urllib.robotparser.RobotFileParser()
                    rp.allow_all = True
                    with self._robots_lock:
                        self.robot_parsers[domain] = rp
                    return True, rp, None
                else:
                    return False, None, "ROBOTS_UNKNOWN"

        domain_lock = self._get_domain_lock(f"robots_{domain}")
        with domain_lock:
            with self._robots_lock:
                if domain in self.robot_parsers:
                    return True, self.robot_parsers[domain], None
                if domain in self._robots_failed and not self.allow_robots_fallback and robots_policy == "fail_closed":
                    return False, None, "ROBOTS_UNKNOWN"

            robots_url = f"https://{domain}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self.fetch_with_retry(robots_url, user_agent, delay_seconds, timeout=10, max_retries=1)
                if resp and resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    self._save_cached_robots(domain, resp.text, failure=False)
                    with self._robots_lock:
                        self.robot_parsers[domain] = rp
                    return True, rp, None
                elif resp and resp.status_code == 404:
                    rp.allow_all = True
                    self._save_cached_robots(domain, "", failure=False)
                    with self._robots_lock:
                        self.robot_parsers[domain] = rp
                    return True, rp, None
                else:
                    status_desc = resp.status_code if resp else "fetch_failed"
                    self.logger.warning(f"Could not read robots.txt for {domain} (status: {status_desc})")
                    self._save_cached_robots(domain, None, failure=True)
                    with self._robots_lock:
                        self._robots_failed.add(domain)
                    if self.allow_robots_fallback or robots_policy != "fail_closed":
                        self.logger.warning(f"robots.txt fallback allowed by flag for {domain}.")
                        rp.allow_all = True
                        with self._robots_lock:
                            self.robot_parsers[domain] = rp
                        return True, rp, None
                    else:
                        return False, None, "ROBOTS_UNKNOWN"
            except Exception as e:
                self.logger.warning(f"Error fetching robots.txt for {domain}: {e}")
                self._save_cached_robots(domain, None, failure=True)
                with self._robots_lock:
                    self._robots_failed.add(domain)
                if self.allow_robots_fallback or robots_policy != "fail_closed":
                    rp.allow_all = True
                    with self._robots_lock:
                        self.robot_parsers[domain] = rp
                    return True, rp, None
                else:
                    return False, None, "ROBOTS_UNKNOWN"

    def _check_robots_for_url(self, url: str, user_agent: str, delay_seconds: float) -> tuple[bool, str | None]:
        """Checks if URL is allowed according to robots.txt policy."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            return True, None
        ok, rp, err = self._get_robot_parser(domain, user_agent, delay_seconds)
        if not ok:
            return False, err or "ROBOTS_UNKNOWN"
        if rp is None:
            return True, None
        if not rp.can_fetch(user_agent, url):
            return False, "ROBOTS_DISALLOWED"
        return True, None

    def _is_url_allowed_by_robots(self, url: str, user_agent: str, delay_seconds: float) -> bool:
        """Backwards-compatible boolean check for robots.txt allowance."""
        allowed, err = self._check_robots_for_url(url, user_agent, delay_seconds)
        return allowed

    def _load_circuit_breaker_state(self) -> dict[str, Any]:
        try:
            if self._circuit_breaker_file.exists():
                return json.loads(self._circuit_breaker_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {"sources": {}}

    def _save_circuit_breaker_state(self, state: dict[str, Any]) -> None:
        try:
            self._circuit_breaker_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._circuit_breaker_file.with_name(
                f"{self._circuit_breaker_file.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(tmp_path, self._circuit_breaker_file)
        except Exception as e:
            self.logger.warning(f"Could not persist circuit breaker state: {e}")

    def _is_circuit_breaker_tripped(self, source_name: str, cb_cfg: dict[str, Any] | None = None) -> bool:
        if self.force_source and self.force_source.lower() == source_name.lower():
            return False
        with self._circuit_breaker_lock:
            state = self._load_circuit_breaker_state()
            src_data = state.get("sources", {}).get(source_name, {})
            cooldown = int(src_data.get("cooldown_runs_left", 0))
            if cooldown > 0:
                src_data["cooldown_runs_left"] = cooldown - 1
                state.setdefault("sources", {})[source_name] = src_data
                self._save_circuit_breaker_state(state)
                return True
            return False

    def _record_circuit_breaker_outcome(
        self,
        source_name: str,
        kept: int,
        error: str | None = None,
        cb_cfg: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        with self._circuit_breaker_lock:
            state = self._load_circuit_breaker_state()
            src_data = state.get("sources", {}).get(source_name, {})
            cfg = cb_cfg or self._runtime_settings.get("circuit_breaker") or {"failures": 3, "cooldown_runs": 3}
            failures_thresh = int(cfg.get("failures", 3))
            cooldown_runs = int(cfg.get("cooldown_runs", 3))

            if error:
                is_failure = True
            elif kept > 0:
                is_failure = False
            elif stats:
                found = int(stats.get("found", 0))
                drop_reasons = stats.get("drop_reasons") or {}
                cat_filtered = int(drop_reasons.get("category_filtered", 0))
                lookback_dropped = int(drop_reasons.get("lookback", 0))
                # If the feed responded and returned entries, but items were dropped due to topic/lookback, that is not an outage
                if found > 0 and (cat_filtered > 0 or lookback_dropped > 0):
                    is_failure = False
                else:
                    is_failure = True
            else:
                is_failure = True

            if is_failure:
                consecutive = int(src_data.get("consecutive_failures", 0)) + 1
                src_data["consecutive_failures"] = consecutive
                src_data["last_status"] = error or "EMPTY"
                if consecutive >= failures_thresh:
                    src_data["cooldown_runs_left"] = cooldown_runs
                    self.logger.warning(
                        f"Circuit breaker TRIPPED for {source_name}: {consecutive} consecutive failures. "
                        f"Entering cooldown for {cooldown_runs} runs."
                    )
            else:
                src_data["consecutive_failures"] = 0
                src_data["cooldown_runs_left"] = 0
                src_data["last_status"] = "OK"

            src_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            state.setdefault("sources", {})[source_name] = src_data
            self._save_circuit_breaker_state(state)

    @staticmethod
    def _init_source_stats() -> dict[str, Any]:
        return {
            "found": 0,
            "kept_filter": 0,
            "extracted_ok": 0,
            "dropped": 0,
            "undated_kept": 0,
            "newest_age_hours": None,
            "drop_reasons": {
                "missing_fields": 0,
                "dup_url": 0,
                "dup_title": 0,
                "dup_story": 0,
                "lookback": 0,
                "future": 0,
                "undated_low_pos": 0,
                "robots": 0,
                "robots_unavailable": 0,
                "rate_limited": 0,
                "thin_story": 0,
                "advertorial": 0,
                "url_filter": 0,
                "category_filtered": 0,
                "extract_fail": 0,
            },
            "avg_latency_ms": None,
            "max_latency_ms": None,
            "robots_unknown": False,
            "circuit_breaker_skipped": False,
            "budget_exhausted": False,
        }

    @staticmethod
    def _finalize_latency_stats(stats: dict[str, Any], latencies: list[float]) -> None:
        if latencies:
            stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
            stats["max_latency_ms"] = round(max(latencies), 1)

    def _load_feed_cache(self, source_name: str) -> dict[str, Any] | None:
        try:
            slug = re.sub(r"[^a-zA-Z0-9._-]", "_", source_name.lower())
            cache_file = self.cache_dir / "feeds" / f"{slug}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def _save_feed_cache(self, source_name: str, etag: str | None, last_modified: str | None, raw_xml: str) -> None:
        try:
            slug = re.sub(r"[^a-zA-Z0-9._-]", "_", source_name.lower())
            feeds_dir = self.cache_dir / "feeds"
            feeds_dir.mkdir(parents=True, exist_ok=True)
            cache_file = feeds_dir / f"{slug}.json"
            cache_file.write_text(
                json.dumps(
                    {
                        "etag": etag,
                        "last_modified": last_modified,
                        "raw_xml": raw_xml,
                        "cached_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _fetch_rss_source(
        self,
        source: dict[str, Any],
        user_agent: str,
        delay_seconds: float,
        ref_time: datetime,
        seen_urls: set[str],
        seen_titles: list[str],
        default_language: str = "en",
        lookback_seconds: float = 30 * 3600,
        selected_categories: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Fetches and processes an RSS source."""
        if selected_categories is None and getattr(self, "selected_categories", None):
            selected_categories = set(self.selected_categories)
        stats = self._init_source_stats()
        latencies: list[float] = []
        error_msg = None
        articles: list[dict[str, Any]] = []

        source_name = source["name"]
        publisher_name = source.get("publisher_name") or source_name
        effective_lookback_sec = float(source.get("lookback_seconds_override") or lookback_seconds)
        source_lang = normalize_language_code(source.get("language") or default_language)
        lang_policy = source.get("language_policy", self._runtime_settings.get("language_policy_default", "translate"))
        if source_lang != "en" and lang_policy == "skip":
            self.logger.info(f"{source_name}: Skipping source because language is {source_lang} and language_policy is skip.")
            return articles, stats, None

        cb_cfg = self._runtime_settings.get("circuit_breaker") or {"failures": 3, "cooldown_runs": 3}
        if self._is_circuit_breaker_tripped(source_name, cb_cfg):
            stats["circuit_breaker_skipped"] = True
            error_msg = "CIRCUIT_BREAKER_TRIPPED: Consecutive failures threshold reached; source in cooldown"
            self.logger.warning(f"{source_name}: {error_msg}")
            return articles, stats, error_msg

        rss_url = source.get("rss")
        if not rss_url:
            return articles, stats, "Missing RSS endpoint"

        # Check robots.txt for feed URL
        allowed, robots_err = self._check_robots_for_url(rss_url, user_agent, delay_seconds)
        if not allowed:
            if robots_err == "ROBOTS_UNKNOWN":
                stats["robots_unknown"] = True
                stats["drop_reasons"]["robots_unavailable"] += 1
                error_msg = f"ROBOTS_UNKNOWN: robots.txt could not be retrieved for {rss_url}"
                self.logger.warning(f"{source_name}: {error_msg}")
                self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
                self._finalize_latency_stats(stats, latencies)
                return articles, stats, error_msg
            else:
                stats["dropped"] += 1
                stats["drop_reasons"]["robots"] += 1
                error_msg = f"Feed URL disallowed by robots.txt: {rss_url}"
                self.logger.warning(f"{source_name}: {error_msg}")
                self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
                self._finalize_latency_stats(stats, latencies)
                return articles, stats, error_msg

        max_articles = source.get("max_articles", 20)

        extra_headers: dict[str, str] = {}
        cached_feed = None if self.backfill else self._load_feed_cache(source_name)
        if cached_feed:
            if cached_feed.get("etag"):
                extra_headers["If-None-Match"] = str(cached_feed["etag"])
            if cached_feed.get("last_modified"):
                extra_headers["If-Modified-Since"] = str(cached_feed["last_modified"])

        self.logger.info(f"Fetching RSS for {source_name}: {rss_url}")
        fetch_kwargs: dict[str, Any] = {"timeout": 15, "latency_sink": latencies}
        if extra_headers:
            fetch_kwargs["extra_headers"] = extra_headers
        resp = self.fetch_with_retry(rss_url, user_agent, delay_seconds, **fetch_kwargs)

        raw_content: bytes | str | None = None
        if resp and resp.status_code == 304 and cached_feed and cached_feed.get("raw_xml"):
            self.logger.info(f"{source_name}: RSS 304 Not Modified; using cached feed XML.")
            raw_content = cached_feed["raw_xml"]
        elif not resp or resp.status_code != 200:
            status = resp.status_code if resp else "timeout/connection failure"
            error_msg = f"Failed to fetch RSS feed ({status})"
            if resp and resp.status_code == 429:
                stats["drop_reasons"]["rate_limited"] += 1
            self.logger.warning(f"{source_name}: {error_msg}")
            self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
            self._finalize_latency_stats(stats, latencies)
            return articles, stats, error_msg
        else:
            raw_content = resp.content
            resp_headers = getattr(resp, "headers", {}) or {}
            etag = resp_headers.get("ETag") if hasattr(resp_headers, "get") else None
            last_mod = resp_headers.get("Last-Modified") if hasattr(resp_headers, "get") else None
            if (etag or last_mod) and isinstance(getattr(resp, "text", None), str):
                self._save_feed_cache(source_name, etag, last_mod, resp.text)

        feed = feedparser.parse(raw_content)
        if getattr(feed, "bozo", False) and not getattr(feed, "entries", None) and resp is not None:
            fallback_text = getattr(resp, "text", None)
            if isinstance(fallback_text, str) and fallback_text:
                self.logger.warning(f"{source_name}: feedparser bozo error ({getattr(feed, 'bozo_exception', '')}); retrying with decoded text.")
                feed = feedparser.parse(fallback_text)

        entries = feed.entries or []
        stats["found"] = len(entries)
        self.logger.info(f"{source_name}: {len(entries)} entries found in feed.")

        feed_ages: list[float] = []
        for entry in entries:
            e_url = normalize_url(getattr(entry, "link", ""))
            e_dt = parse_article_date(entry, e_url, ref_time=ref_time)
            if e_dt is not None:
                e_diff = (ref_time - e_dt).total_seconds()
                if e_diff >= -3600:
                    feed_ages.append(max(0.0, round(e_diff / 3600.0, 2)))
        if feed_ages:
            stats["newest_age_hours"] = min(feed_ages)

        max_undated_pos = int(
            source.get(
                "max_undated_feed_position",
                self._runtime_settings.get("max_undated_feed_position", 5),
            )
        )
        default_trust = float(self._runtime_settings.get("default_trust_weight", 0.70))
        source_trust = round(float(source.get("trust_weight") or source.get("trust") or default_trust), 2)

        for feed_pos, entry in enumerate(entries, 1):
            # Check time budget between articles
            if self._start_time is not None:
                budget_sec = float(self._runtime_settings.get("budget_seconds", 150.0))
                if (time.monotonic() - self._start_time) > budget_sec:
                    stats["budget_exhausted"] = True
                    self.logger.warning(f"{source_name}: Scraper time budget ({budget_sec}s) exhausted. Stopping gracefully.")
                    break

            raw_url = getattr(entry, "link", "")
            title = getattr(entry, "title", "").strip()
            if not raw_url or not title:
                stats["dropped"] += 1
                stats["drop_reasons"]["missing_fields"] += 1
                continue

            norm_url = normalize_url(raw_url)
            url_key = canonical_url_key(norm_url)
            # Deduplicate by canonical URL (scheme/www-agnostic)
            if norm_url in seen_urls or url_key in seen_urls:
                self.logger.debug(f"Dropping duplicate URL: {norm_url}")
                stats["dropped"] += 1
                stats["drop_reasons"]["dup_url"] += 1
                continue

            # Deduplicate by Title similarity (> 0.9) within source
            if is_duplicate_title(title, seen_titles, threshold=0.9):
                self.logger.debug(f"Dropping duplicate title: {title}")
                stats["dropped"] += 1
                stats["drop_reasons"]["dup_title"] += 1
                continue

            # Check robots.txt for article URL
            art_allowed, art_err = self._check_robots_for_url(norm_url, user_agent, delay_seconds)
            if not art_allowed:
                stats["dropped"] += 1
                if art_err == "ROBOTS_UNKNOWN":
                    stats["drop_reasons"]["robots_unavailable"] += 1
                else:
                    stats["drop_reasons"]["robots"] += 1
                continue

            # Parse date & apply lookback filter (or position gate if missing)
            pub_dt, date_conf = parse_article_date_with_confidence(entry, norm_url, ref_time=ref_time)
            age_hours = None
            if pub_dt is not None:
                diff_seconds = (ref_time - pub_dt).total_seconds()
                age_hours = round(diff_seconds / 3600.0, 2)
                if diff_seconds > effective_lookback_sec:
                    self.logger.debug(f"Dropping article outside lookback window ({pub_dt.isoformat()}): {title}")
                    stats["dropped"] += 1
                    stats["drop_reasons"]["lookback"] += 1
                    continue
                if diff_seconds < -3600:
                    self.logger.debug(f"Dropping future-dated article ({pub_dt.isoformat()}): {title}")
                    stats["dropped"] += 1
                    stats["drop_reasons"]["future"] += 1
                    continue
                pub_iso = pub_dt.isoformat()
            else:
                pub_iso = None
                if feed_pos > max_undated_pos:
                    self.logger.debug(
                        f"Dropping undated RSS entry at position #{feed_pos} (> {max_undated_pos}): {title}"
                    )
                    stats["dropped"] += 1
                    stats["drop_reasons"]["undated_low_pos"] += 1
                    continue

            stats["kept_filter"] += 1
            raw_tags = extract_entry_tags(entry)

            # Fast category filter gate before deep body extraction
            if selected_categories and "all" not in selected_categories:
                summary_raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                pre_cat = infer_category_hint(
                    norm_url, title, summary_raw, raw_tags=raw_tags,
                    editor_categories=self._load_editor_categories(),
                )
                pre_imp = compute_importance_hint(
                    feed_pos, raw_tags=raw_tags,
                    priority_tags=self._runtime_settings.get("priority_tags"),
                )
                if not article_matches_categories(
                    category_hint=pre_cat,
                    raw_tags=raw_tags,
                    feed_position=feed_pos,
                    importance_hint=pre_imp,
                    title=title,
                    selected_categories=selected_categories,
                ):
                    stats["dropped"] += 1
                    stats["drop_reasons"]["category_filtered"] += 1
                    continue

            # Extract article content
            article = self._extract_article(
                source_name=publisher_name,
                title=title,
                url=norm_url,
                published_at=pub_iso,
                rss_entry=entry,
                user_agent=user_agent,
                delay_seconds=delay_seconds,
                language=source_lang,
                scraped_at=ref_time.isoformat(),
                age_hours=age_hours,
                ref_time=ref_time,
                latency_sink=latencies,
                raw_tags=raw_tags,
                feed_position=feed_pos,
                source_trust=source_trust,
                date_confidence=date_conf,
            )

            if not article:
                stats["dropped"] += 1
                if getattr(self, "_last_http_status", None) == 429:
                    stats["drop_reasons"]["rate_limited"] += 1
                else:
                    stats["drop_reasons"]["extract_fail"] += 1
                continue

            # Quality gate: drop advertorials and thin stories
            if is_advertisement(article):
                self.logger.debug(f"Dropping advertorial article: {article['title']}")
                stats["dropped"] += 1
                stats["drop_reasons"]["advertorial"] += 1
                continue

            min_words = self._get_min_full_text_words()
            if is_thin_story(article, min_words=min_words):
                self.logger.debug(f"Dropping thin story article (<{min_words} words): {article['title']}")
                stats["dropped"] += 1
                stats["drop_reasons"]["thin_story"] += 1
                continue

            # If date was recovered from HTML metadata, enforce lookback window now
            if pub_dt is None and article.get("age_hours") is not None:
                diff_seconds = float(article["age_hours"]) * 3600.0
                if diff_seconds > effective_lookback_sec:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["lookback"] += 1
                    continue
                if diff_seconds < -3600:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["future"] += 1
                    continue
            elif article.get("date_confidence") == "missing":
                stats["undated_kept"] += 1

            if source.get("forced_category") and article.get("category_hint") in (None, "", "other"):
                article["category_hint"] = source["forced_category"]

            # Post-extraction category verification on full extracted text/summary
            if (
                selected_categories
                and "all" not in selected_categories
                and not source.get("forced_category")
                and not article_matches_categories(
                    category_hint=article.get("category_hint"),
                    raw_tags=article.get("raw_tags"),
                    feed_position=article.get("feed_position"),
                    importance_hint=article.get("importance_hint"),
                    title=article.get("title", ""),
                    selected_categories=selected_categories,
                )
            ):
                stats["dropped"] += 1
                stats["drop_reasons"]["category_filtered"] += 1
                continue

            articles.append(article)
            seen_urls.add(norm_url)
            seen_urls.add(url_key)
            seen_titles.append(title)
            stats["extracted_ok"] += 1
            if len(articles) >= max_articles:
                break

        self._record_circuit_breaker_outcome(source_name, len(articles), error_msg, cb_cfg, stats=stats)
        self._finalize_latency_stats(stats, latencies)
        return articles, stats, error_msg

    def _fetch_html_source(
        self,
        source: dict[str, Any],
        user_agent: str,
        delay_seconds: float,
        ref_time: datetime,
        seen_urls: set[str],
        seen_titles: list[str],
        default_language: str = "en",
        lookback_seconds: float = 30 * 3600,
        selected_categories: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Fetches and processes an HTML-only source by discovering links on the homepage."""
        if selected_categories is None and getattr(self, "selected_categories", None):
            selected_categories = set(self.selected_categories)
        stats = self._init_source_stats()
        latencies: list[float] = []
        error_msg = None
        articles: list[dict[str, Any]] = []

        source_name = source["name"]
        publisher_name = source.get("publisher_name") or source_name
        effective_lookback_sec = float(source.get("lookback_seconds_override") or lookback_seconds)
        source_lang = normalize_language_code(source.get("language") or default_language)
        lang_policy = source.get("language_policy", self._runtime_settings.get("language_policy_default", "translate"))
        if source_lang != "en" and lang_policy == "skip":
            self.logger.info(f"{source_name}: Skipping source because language is {source_lang} and language_policy is skip.")
            return articles, stats, None

        cb_cfg = self._runtime_settings.get("circuit_breaker") or {"failures": 3, "cooldown_runs": 3}
        if self._is_circuit_breaker_tripped(source_name, cb_cfg):
            stats["circuit_breaker_skipped"] = True
            error_msg = "CIRCUIT_BREAKER_TRIPPED: Consecutive failures threshold reached; source in cooldown"
            self.logger.warning(f"{source_name}: {error_msg}")
            return articles, stats, error_msg

        homepage = source.get("homepage", "").rstrip("/")
        if not homepage:
            return articles, stats, "Missing homepage URL"

        # Check robots.txt for homepage
        allowed, robots_err = self._check_robots_for_url(homepage, user_agent, delay_seconds)
        if not allowed:
            if robots_err == "ROBOTS_UNKNOWN":
                stats["robots_unknown"] = True
                stats["drop_reasons"]["robots_unavailable"] += 1
                error_msg = f"ROBOTS_UNKNOWN: robots.txt could not be retrieved for {homepage}"
                self.logger.warning(f"{source_name}: {error_msg}")
                self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
                self._finalize_latency_stats(stats, latencies)
                return articles, stats, error_msg
            else:
                stats["dropped"] += 1
                stats["drop_reasons"]["robots"] += 1
                error_msg = f"Homepage disallowed by robots.txt: {homepage}"
                self.logger.warning(f"{source_name}: {error_msg}")
                self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
                self._finalize_latency_stats(stats, latencies)
                return articles, stats, error_msg

        max_articles = source.get("max_articles", 5)
        include_patterns = [re.compile(p) for p in (source.get("include_url_patterns") or [])]
        exclude_patterns = [re.compile(p) for p in (source.get("exclude_url_patterns") or [])]
        max_undated_pos = int(
            source.get(
                "max_undated_feed_position",
                self._runtime_settings.get("max_undated_feed_position", 5),
            )
        )
        default_trust = float(self._runtime_settings.get("default_trust_weight", 0.70))
        source_trust = round(float(source.get("trust_weight") or source.get("trust") or default_trust), 2)

        self.logger.info(f"Fetching HTML homepage for {source_name}: {homepage}")
        resp = self.fetch_with_retry(homepage, user_agent, delay_seconds, timeout=15, latency_sink=latencies)
        if not resp or resp.status_code != 200:
            status = resp.status_code if resp else "connection failure"
            error_msg = f"Failed to fetch homepage ({status})"
            if resp and resp.status_code == 429:
                stats["drop_reasons"]["rate_limited"] += 1
            self.logger.warning(f"{source_name}: {error_msg}")
            self._record_circuit_breaker_outcome(source_name, 0, error_msg, cb_cfg)
            self._finalize_latency_stats(stats, latencies)
            return articles, stats, error_msg

        raw_markup: Any = None
        if hasattr(resp, "content") and isinstance(resp.content, (bytes, bytearray)):
            raw_markup = resp.content
        elif hasattr(resp, "text") and isinstance(resp.text, str):
            raw_markup = resp.text
        else:
            raw_markup = getattr(resp, "content", None) or getattr(resp, "text", "") or ""
        soup = BeautifulSoup(raw_markup, "html.parser")
        base_domain = urlparse(homepage).netloc

        candidate_entries: list[dict[str, Any]] = []
        seen_candidates: dict[str, dict[str, Any]] = {}
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            anchor_text = a.get_text(" ", strip=True)
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = href if href.startswith("http") else urljoin(homepage, href)
            parsed = urlparse(full_url)
            if parsed.netloc != base_domain or len(parsed.path) <= 5:
                continue
            if any(parsed.path.endswith(ext) for ext in (".css", ".js", ".png", ".jpg", ".svg", ".pdf")):
                continue
            if exclude_patterns and any(pat.search(parsed.path) for pat in exclude_patterns):
                continue
            if include_patterns and not any(pat.search(parsed.path) for pat in include_patterns):
                continue
            cand_key = canonical_url_key(full_url)
            if cand_key in seen_candidates:
                # If earlier link was an empty thumbnail <a>, adopt the non-empty headline link text
                if not seen_candidates[cand_key]["title"] and len(anchor_text) > 5:
                    seen_candidates[cand_key]["title"] = anchor_text
                continue

            # Pre-date extraction from listing (T3.2)
            cand_dt: datetime | None = None
            date_m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", parsed.path)
            if date_m:
                try:
                    cand_dt = datetime(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)), 12, 0, tzinfo=KATHMANDU_TZ)
                except Exception:
                    pass
            if cand_dt is None:
                time_node = a.find("time") or (a.parent and a.parent.find("time"))
                if time_node and time_node.get("datetime"):
                    try:
                        cand_dt = dateparser.parse(time_node["datetime"])
                        if cand_dt and cand_dt.tzinfo is None:
                            cand_dt = cand_dt.replace(tzinfo=KATHMANDU_TZ)
                    except Exception:
                        pass

            if cand_dt is not None:
                cand_age_sec = (ref_time - cand_dt).total_seconds()
                if cand_age_sec > effective_lookback_sec:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["lookback"] += 1
                    continue

            entry_obj = {
                "url": full_url,
                "title": anchor_text,
                "cand_dt": cand_dt,
            }
            seen_candidates[cand_key] = entry_obj
            candidate_entries.append(entry_obj)

        # Sort candidate links: newest-dated first, then undated preserving layout order
        candidate_entries.sort(
            key=lambda e: (e["cand_dt"] is not None, e["cand_dt"] or datetime.min.replace(tzinfo=KATHMANDU_TZ)),
            reverse=True,
        )

        stats["found"] = len(candidate_entries)
        self.logger.info(f"{source_name}: Found {len(candidate_entries)} candidate article links.")

        html_ages: list[float] = []
        for feed_pos, cand in enumerate(candidate_entries, 1):
            # Check time budget between articles
            if self._start_time is not None:
                budget_sec = float(self._runtime_settings.get("budget_seconds", 150.0))
                if (time.monotonic() - self._start_time) > budget_sec:
                    stats["budget_exhausted"] = True
                    self.logger.warning(f"{source_name}: Scraper time budget ({budget_sec}s) exhausted. Stopping gracefully.")
                    break

            raw_url = cand["url"]
            anchor_title = cand["title"]
            norm_url = normalize_url(raw_url)
            url_key = canonical_url_key(norm_url)
            if norm_url in seen_urls or url_key in seen_urls:
                stats["dropped"] += 1
                stats["drop_reasons"]["dup_url"] += 1
                continue

            # Check robots.txt for path
            art_allowed, art_err = self._check_robots_for_url(norm_url, user_agent, delay_seconds)
            if not art_allowed:
                self.logger.info(f"Skipping robots-disallowed path: {norm_url}")
                stats["dropped"] += 1
                if art_err == "ROBOTS_UNKNOWN":
                    stats["drop_reasons"]["robots_unavailable"] += 1
                else:
                    stats["drop_reasons"]["robots"] += 1
                continue

            # T3.3 Duplicate title check BEFORE deep-fetching
            if anchor_title and len(anchor_title.strip()) > 5:
                if is_duplicate_title(anchor_title.strip(), seen_titles, threshold=0.9):
                    self.logger.debug(f"Dropping duplicate title before deep-fetch: {anchor_title}")
                    stats["dropped"] += 1
                    stats["drop_reasons"]["dup_title"] += 1
                    continue
                seen_titles.append(anchor_title.strip())

            stats["kept_filter"] += 1

            # Fast category filter gate before deep HTML body extraction
            if selected_categories and "all" not in selected_categories:
                pre_cat = infer_category_hint(
                    norm_url, anchor_title, "", raw_tags=[],
                    editor_categories=self._load_editor_categories(),
                )
                pre_imp = compute_importance_hint(feed_pos, raw_tags=[])
                if not article_matches_categories(
                    category_hint=pre_cat,
                    raw_tags=[],
                    feed_position=feed_pos,
                    importance_hint=pre_imp,
                    title=anchor_title,
                    selected_categories=selected_categories,
                ):
                    stats["dropped"] += 1
                    stats["drop_reasons"]["category_filtered"] += 1
                    continue

            cand_pub_at = cand["cand_dt"].isoformat() if cand.get("cand_dt") else None
            cand_age = round((ref_time - cand["cand_dt"]).total_seconds() / 3600.0, 2) if cand.get("cand_dt") else None
            cand_conf = "url_date" if cand.get("cand_dt") else "missing"

            article = self._extract_article(
                source_name=publisher_name,
                title=anchor_title or "Untitled Article",
                url=norm_url,
                published_at=cand_pub_at,
                rss_entry={},
                user_agent=user_agent,
                delay_seconds=delay_seconds,
                language=source_lang,
                scraped_at=ref_time.isoformat(),
                age_hours=cand_age,
                ref_time=ref_time,
                latency_sink=latencies,
                raw_tags=[],
                feed_position=feed_pos,
                source_trust=source_trust,
                date_confidence=cand_conf,
            )

            if not article:
                stats["dropped"] += 1
                if getattr(self, "_last_http_status", None) == 429:
                    stats["drop_reasons"]["rate_limited"] += 1
                else:
                    stats["drop_reasons"]["extract_fail"] += 1
                continue

            # Quality gate: drop advertorials and thin stories
            if is_advertisement(article):
                self.logger.debug(f"Dropping advertorial HTML article: {article['title']}")
                stats["dropped"] += 1
                stats["drop_reasons"]["advertorial"] += 1
                continue

            min_words = self._get_min_full_text_words()
            if is_thin_story(article, min_words=min_words):
                self.logger.debug(f"Dropping thin story HTML article (<{min_words} words): {article['title']}")
                stats["dropped"] += 1
                stats["drop_reasons"]["thin_story"] += 1
                continue

            # Enforce lookback window if metadata yielded a publication timestamp
            if article.get("age_hours") is not None:
                diff_seconds = float(article["age_hours"]) * 3600.0
                if diff_seconds > effective_lookback_sec:
                    self.logger.debug(f"Dropping HTML article outside lookback ({article['age_hours']}h): {article['title']}")
                    stats["dropped"] += 1
                    stats["drop_reasons"]["lookback"] += 1
                    continue
                if diff_seconds < -3600:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["future"] += 1
                    continue
                html_ages.append(max(0.0, float(article["age_hours"])))
            elif article.get("date_confidence") == "missing":
                if feed_pos > max_undated_pos:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["undated_low_pos"] += 1
                    continue
                stats["undated_kept"] += 1

            if source.get("forced_category") and article.get("category_hint") in (None, "", "other"):
                article["category_hint"] = source["forced_category"]

            # Post-extraction category verification on full extracted text/summary
            if (
                selected_categories
                and "all" not in selected_categories
                and not source.get("forced_category")
                and not article_matches_categories(
                    category_hint=article.get("category_hint"),
                    raw_tags=article.get("raw_tags"),
                    feed_position=article.get("feed_position"),
                    importance_hint=article.get("importance_hint"),
                    title=article.get("title", ""),
                    selected_categories=selected_categories,
                )
            ):
                stats["dropped"] += 1
                stats["drop_reasons"]["category_filtered"] += 1
                continue

            articles.append(article)
            seen_urls.add(norm_url)
            seen_urls.add(url_key)
            seen_titles.append(article["title"])
            stats["extracted_ok"] += 1
            if len(articles) >= max_articles:
                break

        if html_ages:
            stats["newest_age_hours"] = min(html_ages)
        self._record_circuit_breaker_outcome(source_name, len(articles), error_msg, cb_cfg, stats=stats)
        self._finalize_latency_stats(stats, latencies)
        return articles, stats, error_msg

    def _extract_article(
        self,
        source_name: str = "",
        title: str = "",
        url: str = "",
        published_at: str | None = None,
        rss_entry: dict[str, Any] | None = None,
        user_agent: str = "NepalNewsBot/0.1",
        delay_seconds: float = 0.0,
        language: str = "en",
        scraped_at: str | None = None,
        age_hours: float | None = None,
        ref_time: datetime | None = None,
        latency_sink: list[float] | None = None,
        raw_tags: list[str] | None = None,
        feed_position: int | None = None,
        source_trust: float | None = None,
        date_confidence: str | None = None,
        html_text: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extracts article text with trafilatura / JSON-LD / RSS summary fallback chain."""
        if source and isinstance(source, dict):
            source_name = source.get("name", source_name)
            language = source.get("language", language)
            source_trust = source.get("trust_weight") or source.get("trust") or source_trust
        if rss_entry is None:
            rss_entry = {}

        # Enforce robots check on article URL
        allowed, art_err = self._check_robots_for_url(url, user_agent, delay_seconds)
        if not allowed:
            self.logger.info(f"Skipping article URL disallowed or blocked by robots.txt: {url} ({art_err})")
            return None

        extracted_text = ""
        winning_method: str | None = None
        extracted_title = title
        author = rss_entry.get("author")
        effective_date_conf = date_confidence or ("exact" if published_at else "missing")

        # 1. Fetch HTML with polite delay
        raw_html: str | None = html_text
        if raw_html is None:
            fetch_kwargs: dict[str, Any] = {"timeout": 15}
            if latency_sink is not None:
                fetch_kwargs["latency_sink"] = latency_sink
            html_resp = self.fetch_with_retry(url, user_agent, delay_seconds, **fetch_kwargs)
            if html_resp and html_resp.status_code == 200:
                if html_resp.encoding is None or html_resp.encoding.lower() == "iso-8859-1":
                    html_resp.encoding = html_resp.apparent_encoding or "utf-8"
                raw_html = html_resp.text

        if raw_html:
            try:
                raw_text = trafilatura.extract(
                    raw_html,
                    url=url,
                    include_links=False,
                    include_images=False,
                    output_format="txt",
                )
                if raw_text:
                    clean_cand = clean_text_content(raw_text)
                    if len(clean_cand) >= 200:
                        extracted_text = clean_cand
                        winning_method = "trafilatura"

                # JSON-LD fallback (T2.2)
                if len(extracted_text) < 200:
                    j_body, j_meta = extract_jsonld_article(raw_html)
                    if len(j_body) >= 60:
                        extracted_text = j_body
                        winning_method = "json_ld"
                        if (not extracted_title or len(extracted_title) < 5) and j_meta.get("headline"):
                            extracted_title = str(j_meta["headline"]).strip()
                        if not author and j_meta.get("author"):
                            author = str(j_meta["author"]).strip()
                        if published_at is None and j_meta.get("datePublished"):
                            try:
                                p_dt = dateparser.parse(str(j_meta["datePublished"]))
                                if p_dt:
                                    published_at = (p_dt.astimezone(KATHMANDU_TZ) if p_dt.tzinfo else p_dt.replace(tzinfo=KATHMANDU_TZ)).isoformat()
                                    effective_date_conf = "metadata"
                            except Exception:
                                pass

                # Meta description fallback
                if len(extracted_text) < 200:
                    meta_desc = extract_meta_description(raw_html)
                    if len(meta_desc) >= 60:
                        extracted_text = meta_desc
                        winning_method = "meta_description"

                # If original title was empty, short, or placeholder, extract from HTML metadata
                if len(extracted_title) < 5 or extracted_title.lower() == "untitled article":
                    meta = trafilatura.extract_metadata(raw_html)
                    if meta and meta.title:
                        extracted_title = meta.title.strip()
                    if meta and meta.author and not author:
                        author = meta.author.strip()
                    if len(extracted_title) < 5 or extracted_title.lower() == "untitled article":
                        try:
                            t_soup = BeautifulSoup(raw_html, "html.parser")
                            og_t = t_soup.find("meta", attrs={"property": "og:title"})
                            h1_t = t_soup.find("h1")
                            if og_t and og_t.get("content"):
                                extracted_title = str(og_t["content"]).strip()
                            elif h1_t and h1_t.get_text(strip=True):
                                extracted_title = h1_t.get_text(" ", strip=True)
                        except Exception:
                            pass

                # Recover publication timestamp from HTML metadata when missing
                if published_at is None:
                    meta_dt = extract_html_published_at(raw_html, ref_time=ref_time)
                    if meta_dt is not None:
                        published_at = meta_dt.isoformat()
                        effective_date_conf = "metadata"
                        if ref_time is not None:
                            ref_ktm = ref_time.astimezone(KATHMANDU_TZ) if ref_time.tzinfo else ref_time.replace(tzinfo=KATHMANDU_TZ)
                            age_hours = round((ref_ktm - meta_dt).total_seconds() / 3600.0, 2)
            except Exception as e:
                self.logger.debug(f"HTML extraction exception on {url}: {e}")

        # 2. Check if extraction succeeded or has >= 200 chars, else RSS fallback
        if len(extracted_text) < 200:
            self.logger.debug(f"Main text extraction under 200 chars ({len(extracted_text)}) for {url}. Attempting RSS fallback.")
            summary_raw = rss_entry.get("summary") or rss_entry.get("description") or ""
            if summary_raw:
                soup = BeautifulSoup(summary_raw, "html.parser")
                fallback_text = clean_text_content(soup.get_text(separator=" ", strip=True))
                if len(fallback_text) >= 50:
                    extracted_text = fallback_text
                    winning_method = "rss_summary"
                else:
                    self.logger.info(f"Article dropped: RSS summary too short (<50 chars) for {url}")
                    return None
            elif not extracted_text:
                self.logger.info(f"Article dropped: Extraction failed (<200 chars) and no RSS fallback for {url}")
                return None

        if not extracted_text or len(extracted_text) < 50:
            return None

        winning_method = winning_method or "trafilatura"

        # Clean title
        if not extracted_title or extracted_title == "Untitled Article":
            extracted_title = title if len(title) > 5 else "Nepal News Report"

        # Compute full word count before truncation (script-aware for Devanagari)
        norm_lang = normalize_language_code(language)
        if norm_lang == "ne":
            tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b|[\u0900-\u097F]+", extracted_text)
            word_count = len(tokens)
        else:
            word_count = len(extracted_text.split())

        full_text = truncate_at_sentence_boundary(extracted_text, max_chars=6000)

        # Generate clean summary (~600 chars) skipping boilerplate and datelines
        text_summary = build_text_summary(full_text, rss_entry=rss_entry)

        tags_list = list(raw_tags) if raw_tags is not None else extract_entry_tags(rss_entry)
        category_hint = infer_category_hint(
            url=url,
            title=extracted_title,
            text_summary=text_summary,
            raw_tags=tags_list,
            editor_categories=self._load_editor_categories(),
        )
        importance_hint = compute_importance_hint(
            feed_position=feed_position,
            raw_tags=tags_list,
            priority_tags=self._runtime_settings.get("priority_tags"),
        )
        default_trust = float(self._runtime_settings.get("default_trust_weight", 0.70))
        effective_trust = round(float(source_trust if source_trust is not None else default_trust), 2)

        priority_score = compute_priority_score(
            age_hours=age_hours,
            feed_position=feed_position,
            raw_tags=tags_list,
            source_trust=effective_trust,
            priority_tags=self._runtime_settings.get("priority_tags"),
        )

        needs_trans = (norm_lang != "en")
        trans_status = "pending" if needs_trans else "not_required"

        article: dict[str, Any] = {
            "source": source_name,
            "title": extracted_title,
            "url": url,
            "published_at": published_at,
            "text_summary": text_summary,
            "full_text": full_text,
            "language": norm_lang,
            "word_count": word_count,
            "author": author if author else None,
            "scraped_at": scraped_at or datetime.now(timezone.utc).isoformat(),
            "age_hours": age_hours,
            "category_hint": category_hint,
            "raw_tags": tags_list,
            "importance_hint": importance_hint,
            "feed_position": feed_position,
            "source_trust": effective_trust,
            "date_confidence": effective_date_conf,
            "needs_translation": needs_trans,
            "translation_status": trans_status,
            "priority_score": priority_score,
            "extraction": {"method": winning_method, "chars": len(extracted_text)},
        }
        return article

    def _should_run_parallel(self) -> bool:
        if self.config and "execution" in self.config:
            return bool(self.config["execution"].get("allow_parallel", True))
        try:
            pipeline_cfg = self._load_pipeline_config()
            return bool(pipeline_cfg.get("execution", {}).get("allow_parallel", True))
        except Exception:
            return True

    def _scrape_single_source(
        self,
        src: dict[str, Any],
        user_agent: str,
        delay_seconds: float,
        ref_time: datetime,
        default_language: str,
        lookback_seconds: float,
        selected_categories: set[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], str | None]:
        name = src.get("name", "Unknown Source")
        method = src.get("method", "rss")
        local_urls: set[str] = set()
        local_titles: list[str] = []

        if method == "rss":
            src_articles, stats, err = self._fetch_rss_source(
                src,
                user_agent,
                delay_seconds,
                ref_time,
                local_urls,
                local_titles,
                default_language=default_language,
                lookback_seconds=lookback_seconds,
                selected_categories=selected_categories,
            )
        elif method in ("html_only", "html"):
            src_articles, stats, err = self._fetch_html_source(
                src,
                user_agent,
                delay_seconds,
                ref_time,
                local_urls,
                local_titles,
                default_language=default_language,
                lookback_seconds=lookback_seconds,
                selected_categories=selected_categories,
            )
        else:
            self.logger.warning(f"Unknown method '{method}' for {name}")
            return name, [], self._init_source_stats(), f"Unknown method '{method}'"

        min_expected = int(src.get("min_expected_articles", 2))
        if not err and stats.get("extracted_ok", 0) < min_expected:
            self.logger.warning(
                f"Low yield for {name}: {stats.get('extracted_ok', 0)} articles "
                f"(expected >= {min_expected}; found={stats.get('found', 0)}, "
                f"drops={stats.get('drop_reasons', {})})"
            )

        return name, src_articles, stats, err

    def _extract_entities_from_article(self, article: dict[str, Any], entity_aliases: dict[str, list[str]]) -> set[str]:
        text = f"{article.get('title', '')} {article.get('text_summary', '')}"
        return extract_canonical_entities(text, entity_aliases=entity_aliases)

    def preflight_scan_feed_categories(
        self,
        active_sources: list[dict[str, Any]],
        user_agent: str,
        delay_seconds: float,
        ref_time: datetime,
    ) -> dict[str, int]:
        """Performs a lightweight header-only scan across active RSS feeds to estimate category availability."""
        counts: dict[str, int] = {slug: 0 for slug, _, _ in AVAILABLE_CATEGORIES.values()}
        editor_cats = self._load_editor_categories()
        prio_tags = self._runtime_settings.get("priority_tags")

        for src in active_sources:
            if src.get("method", "rss") != "rss" or not src.get("rss"):
                continue
            rss_url = src["rss"]
            allowed, _ = self._check_robots_for_url(rss_url, user_agent, delay_seconds)
            if not allowed:
                continue
            try:
                resp = self.fetch_with_retry(rss_url, user_agent, delay_seconds, timeout=5)
                if not resp or resp.status_code != 200:
                    continue
                feed = feedparser.parse(resp.content)
                for pos, entry in enumerate(feed.entries or [], 1):
                    raw_url = getattr(entry, "link", "")
                    title = getattr(entry, "title", "").strip()
                    if not raw_url or not title:
                        continue
                    norm_url = normalize_url(raw_url)
                    raw_tags = extract_entry_tags(entry)
                    summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                    cat_hint = infer_category_hint(norm_url, title, summary, raw_tags=raw_tags, editor_categories=editor_cats)
                    imp = compute_importance_hint(pos, raw_tags=raw_tags, priority_tags=prio_tags)

                    for slug, _, _ in AVAILABLE_CATEGORIES.values():
                        if article_matches_categories(cat_hint, raw_tags, pos, imp, title, {slug}):
                            counts[slug] = counts.get(slug, 0) + 1
            except Exception:
                continue
        return counts

    def scrape_all(
        self,
        target_date: str | None = None,
        source_filter: str | None = None,
        backfill: bool | None = None,
        lookback_hours: float | None = None,
        ref_time: datetime | None = None,
        categories: list[str] | str | None = None,
        interactive: bool | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
        """Executes scraping for all active sources (or a single filtered source).

        Returns:
            (articles, per_source_stats, source_errors)
        """
        self._start_time = time.monotonic()
        sources_cfg = self._load_sources_config()
        settings = sources_cfg.get("settings", {})
        self._runtime_settings = settings
        user_agent = settings.get("user_agent", "NepalNewsBot/0.1 (personal project)")
        delay_seconds = float(settings.get("request_delay_seconds", 2))
        default_language = normalize_language_code(settings.get("language", "en"))

        is_backfill = self.backfill if backfill is None else backfill
        effective_lookback_hours = (
            lookback_hours
            if lookback_hours is not None
            else (self.lookback_hours if self.lookback_hours is not None else settings.get("lookback_hours", 30))
        )
        lookback_seconds = float(effective_lookback_hours) * 3600

        # Establish reference time
        if ref_time is not None:
            pass
        elif is_backfill and target_date:
            try:
                y, m, d = map(int, target_date.split("-"))
                ref_time = datetime(y, m, d, 23, 59, 59, tzinfo=KATHMANDU_TZ)
            except Exception:
                ref_time = datetime.now(KATHMANDU_TZ)
        else:
            ref_time = datetime.now(KATHMANDU_TZ)

        self.logger.info(
            f"Scraper reference time: {ref_time.isoformat()} (Asia/Kathmandu) | "
            f"lookback: {lookback_seconds/3600:.1f}h | backfill: {is_backfill}"
        )

        active_sources: list[dict[str, Any]] = []
        for src in sources_cfg.get("sources", []):
            name = src.get("name", "Unknown Source")
            if source_filter and source_filter.lower() not in name.lower():
                continue
            if not src.get("robots_ok", True):
                self.logger.info(f"Skipping {name}: robots_ok is false.")
                continue
            if not src.get("enabled", True):
                self.logger.info(f"Skipping {name}: enabled is false.")
                continue
            active_sources.append(src)

        # Resolve selected categories
        selected_cats: list[str]
        effective_interactive = self.interactive if interactive is None else interactive

        if categories is not None:
            if isinstance(categories, str):
                selected_cats = parse_category_input(categories)
            else:
                selected_cats = parse_category_input(",".join(categories))
        elif self.categories is not None:
            selected_cats = list(self.categories)
        else:
            is_auto = (effective_interactive is False) or (
                effective_interactive is None
                and (not sys.stdin.isatty() or bool(os.getenv("PYTEST_CURRENT_TEST")))
            )
            if is_auto:
                print("[AUTOMATED] Automated run detected: falling back to Headlines & Breaking News.")
                selected_cats = ["headlines"]
            else:
                cat_counts = self.preflight_scan_feed_categories(active_sources, user_agent, delay_seconds, ref_time)
                selected_cats = prompt_category_selection(cat_counts, interactive=effective_interactive, target_date=target_date)

        self.selected_categories = selected_cats
        selected_categories_set = set(selected_cats)
        self.logger.info(f"Scraper targeted categories: {', '.join(selected_cats)}")

        # Expand active sources with dedicated category section feeds when specific topics are selected
        topic_cats = [c for c in selected_cats if c not in ("all", "headlines")]
        if topic_cats:
            extra_section_sources: list[dict[str, Any]] = []
            for src in active_sources:
                cat_feeds = src.get("category_feeds") or {}
                if not isinstance(cat_feeds, dict):
                    continue
                for cat_key in topic_cats:
                    feed_cfg = cat_feeds.get(cat_key)
                    if not isinstance(feed_cfg, dict) or not feed_cfg.get("url"):
                        continue
                    sec_method = feed_cfg.get("method", "rss")
                    sec_src = dict(src)
                    sec_src["name"] = f"{src['name']} ({cat_key})"
                    sec_src["publisher_name"] = src["name"]
                    sec_src["method"] = sec_method
                    sec_src["max_articles"] = int(feed_cfg.get("max_articles", 10))
                    sec_src["forced_category"] = cat_key
                    if sec_method == "rss":
                        sec_src["rss"] = feed_cfg["url"]
                    else:
                        sec_src["homepage"] = feed_cfg["url"]
                        sec_src["rss"] = None
                        if feed_cfg.get("include_url_patterns"):
                            sec_src["include_url_patterns"] = feed_cfg["include_url_patterns"]
                        if feed_cfg.get("exclude_url_patterns"):
                            sec_src["exclude_url_patterns"] = feed_cfg["exclude_url_patterns"]
                    # Category section feeds publish at lower frequency; give them 14-day lookback
                    # unless an explicit CLI lookback override was set on the agent instance.
                    if self.lookback_hours is None:
                        sec_src["lookback_seconds_override"] = 14 * 24 * 3600
                    extra_section_sources.append(sec_src)
            if extra_section_sources:
                self.logger.info(
                    f"Added {len(extra_section_sources)} dedicated category section feed(s) for {', '.join(topic_cats)}."
                )
                active_sources = extra_section_sources + active_sources

        ordered_results: list[tuple[str, list[dict[str, Any]], dict[str, Any], str | None]] = []
        run_parallel = self._should_run_parallel() and len(active_sources) > 1

        if run_parallel:
            max_workers = min(len(active_sources), 8)
            futures_by_idx: dict[int, Any] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for idx, src in enumerate(active_sources):
                    fut = pool.submit(
                        self._scrape_single_source,
                        src,
                        user_agent,
                        delay_seconds,
                        ref_time,
                        default_language,
                        lookback_seconds,
                        selected_categories=selected_categories_set,
                    )
                    futures_by_idx[idx] = fut
                for idx in range(len(active_sources)):
                    ordered_results.append(futures_by_idx[idx].result())
        else:
            for src in active_sources:
                ordered_results.append(
                    self._scrape_single_source(
                        src,
                        user_agent,
                        delay_seconds,
                        ref_time,
                        default_language,
                        lookback_seconds,
                        selected_categories=selected_categories_set,
                    )
                )

        # Load entity aliases from editor.yaml for cross-source entity matching
        entity_aliases: dict[str, list[str]] = {}
        try:
            ed_cfg = self.repo_root / "config" / "editor.yaml"
            if ed_cfg.exists():
                ed_data = yaml.safe_load(ed_cfg.read_text(encoding="utf-8")) or {}
                entity_aliases = ed_data.get("entity_aliases", {})
        except Exception:
            entity_aliases = {}

        dedupe_mode = self._runtime_settings.get("dedupe_mode", "url")
        all_articles: list[dict[str, Any]] = []
        per_source_stats: dict[str, dict[str, Any]] = {}
        source_errors: dict[str, str] = {}
        seen_urls: set[str] = set()

        for name, src_articles, stats, err in ordered_results:
            per_source_stats[name] = stats
            if err:
                source_errors[name] = err

            for cand in src_articles:
                cand["_feed_name"] = name
                cand_key = canonical_url_key(cand.get("url", ""))
                if cand_key and cand_key in seen_urls:
                    stats["dropped"] += 1
                    stats["extracted_ok"] = max(0, stats["extracted_ok"] - 1)
                    stats["drop_reasons"]["dup_url"] += 1
                    continue

                # Same-source deduplication (Tier 2 & Tier 3 within same outlet)
                # Cross-outlet stories are preserved so Editor clustering can group multi-source stories.
                cand_source = cand.get("source", "")
                cand_tokens = tokenize(f"{cand.get('title', '')} {cand.get('text_summary', '')}")
                cand_entities = self._extract_entities_from_article(cand, entity_aliases)

                dup_match_idx = -1
                for idx, existing in enumerate(all_articles):
                    if existing.get("source", "") != cand_source:
                        continue

                    # Tier 2: Title similarity >= 0.9 within same source
                    if is_duplicate_title(cand.get("title", ""), [existing.get("title", "")], threshold=0.9):
                        dup_match_idx = idx
                        break

                    # Tier 3: Token overlap >= 0.35 + entity match within same source
                    if "token" in dedupe_mode:
                        exist_tokens = tokenize(f"{existing.get('title', '')} {existing.get('text_summary', '')}")
                        exist_entities = self._extract_entities_from_article(existing, entity_aliases)
                        overlap = (
                            len(cand_tokens & exist_tokens) / len(cand_tokens | exist_tokens)
                            if (cand_tokens | exist_tokens)
                            else 0.0
                        )
                        if overlap >= 0.35 and bool(cand_entities & exist_entities):
                            dup_match_idx = idx
                            break

                if dup_match_idx != -1:
                    existing = all_articles[dup_match_idx]
                    cand_trust = float(cand.get("source_trust", 0.70))
                    exist_trust = float(existing.get("source_trust", 0.70))
                    cand_pub = cand.get("published_at") or ""
                    exist_pub = existing.get("published_at") or ""

                    if (cand_trust > exist_trust) or (cand_trust == exist_trust and cand_pub > exist_pub):
                        loser = existing
                        loser_src = loser.get("source", "")
                        loser_feed = loser.get("_feed_name") or loser_src
                        target_stat = per_source_stats.get(loser_feed)
                        if not target_stat:
                            for pk, pv in per_source_stats.items():
                                if pk == loser_src or pk.startswith(f"{loser_src} ("):
                                    target_stat = pv
                                    break
                        if target_stat:
                            target_stat["dropped"] += 1
                            target_stat["extracted_ok"] = max(0, target_stat["extracted_ok"] - 1)
                            target_stat["drop_reasons"]["dup_story"] += 1
                        self.logger.info(
                            f"Same-source duplicate: '{cand['title']}' ({cand['source']}) supersedes '{loser['title']}'"
                        )
                        if cand_key:
                            seen_urls.add(cand_key)
                        all_articles[dup_match_idx] = cand
                    else:
                        cand_src = cand.get("source", "")
                        cand_stat = per_source_stats.get(name) or per_source_stats.get(cand_src)
                        if not cand_stat:
                            for pk, pv in per_source_stats.items():
                                if pk == cand_src or pk.startswith(f"{cand_src} ("):
                                    cand_stat = pv
                                    break
                        if cand_stat:
                            cand_stat["dropped"] += 1
                            cand_stat["extracted_ok"] = max(0, cand_stat["extracted_ok"] - 1)
                            cand_stat["drop_reasons"]["dup_story"] += 1
                        self.logger.info(
                            f"Same-source duplicate: '{existing['title']}' retained over '{cand['title']}'"
                        )
                        continue
                else:
                    if cand_key:
                        seen_urls.add(cand_key)
                    all_articles.append(cand)

        for a in all_articles:
            a.pop("_feed_name", None)

        return all_articles, per_source_stats, source_errors

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Executes the Scraper Agent within the multi-agent pipeline."""
        try:
            return self._run_internal(manifest)
        finally:
            self._close_sessions()

    def _close_sessions(self) -> None:
        """Closes all cached requests.Session instances."""
        for s in list(self._sessions.values()):
            try:
                s.close()
            except Exception:
                pass
        self._sessions.clear()

    def _run_internal(self, manifest: dict[str, Any]) -> list[str]:
        target_date = manifest.get("date")
        pipeline_cfg = self._load_pipeline_config()
        scraper_st = next((st for st in pipeline_cfg.get("stages", []) if st.get("id") == "scraper"), {})

        if not self._runtime_settings:
            sources_cfg = self._load_sources_config()
            self._runtime_settings = sources_cfg.get("settings", {})

        min_eligible = int(scraper_st.get("min_eligible_articles", scraper_st.get("min_articles", 6)))
        degraded_floor = int(scraper_st.get("degraded_floor", 3))
        degradation_lookback = float(scraper_st.get("degradation_lookback_hours", 48.0))
        lookback_normal = float(self.lookback_hours if self.lookback_hours is not None else self._runtime_settings.get("lookback_hours", 30.0))

        effective_cats = (
            self.categories
            or manifest.get("selected_categories")
            or manifest.get("stages", {}).get("scraper", {}).get("metrics", {}).get("selected_categories")
        )
        if effective_cats and "all" not in effective_cats and len(effective_cats) == 1:
            min_eligible = min(min_eligible, 3)

        self.logger.info(f"Running Scraper Agent for run date: {target_date} (min_eligible={min_eligible})")

        articles, per_source_stats, errors = self.scrape_all(
            target_date=target_date,
            backfill=self.backfill,
            lookback_hours=lookback_normal,
            categories=effective_cats,
            interactive=self.interactive,
        )

        degraded_pass = False
        degradation_reasons: list[str] = []

        if len(articles) < min_eligible:
            # Check if shortfall is an access outage
            active_stats = list(per_source_stats.values())
            all_access_outage = bool(active_stats and all(
                st.get("robots_unknown")
                or (st.get("drop_reasons", {}).get("robots_unavailable", 0) > 0 and st.get("extracted_ok", 0) == 0)
                or (st.get("drop_reasons", {}).get("rate_limited", 0) > 0 and st.get("extracted_ok", 0) == 0)
                or (st.get("found", 0) == 0 and st.get("extracted_ok", 0) == 0 and sname in errors)
                for sname, st in per_source_stats.items()
            ))

            if all_access_outage:
                err_msg = "ACCESS_OUTAGE: All sources were unreachable, rate-limited, or blocked by robots.txt."
                self.logger.error(err_msg)
                manifest_file = self.run_dir / "manifest.json"
                if manifest_file.exists():
                    m = load_manifest(manifest_file)
                    update_stage_status(
                        m,
                        "scraper",
                        "failed",
                        error_message=err_msg,
                        source_errors=errors,
                        metrics={"total_articles": len(articles), "per_source_stats": per_source_stats},
                    )
                    save_manifest(m, manifest_file)
                raise RuntimeError(err_msg)

            # Attempt 2: Widened lookback ladder
            self.logger.info(f"Ingest shortfall ({len(articles)} < {min_eligible}). Widening lookback to {degradation_lookback}h...")
            articles_w, per_source_stats_w, errors_w = self.scrape_all(
                target_date=target_date,
                backfill=self.backfill,
                lookback_hours=degradation_lookback,
                categories=getattr(self, "selected_categories", effective_cats),
                interactive=False,
            )
            if len(articles_w) >= min_eligible:
                articles = articles_w
                per_source_stats = per_source_stats_w
                errors = errors_w
            elif len(articles_w) >= degraded_floor:
                articles = articles_w
                per_source_stats = per_source_stats_w
                errors = errors_w
                degraded_pass = True
                degradation_reasons.append(
                    f"Eligible stories ({len(articles)}) below target ({min_eligible}); accepted under degraded floor ({degraded_floor}) with {degradation_lookback}h lookback."
                )
            else:
                err_msg = f"INSUFFICIENT_SOURCES: Scraped {len(articles_w)} articles after lookback ladder ({degradation_lookback}h), below degraded floor ({degraded_floor})."
                self.logger.error(err_msg)
                manifest_file = self.run_dir / "manifest.json"
                if manifest_file.exists():
                    m = load_manifest(manifest_file)
                    update_stage_status(
                        m,
                        "scraper",
                        "failed",
                        error_message=err_msg,
                        source_errors=errors_w,
                        metrics={"total_articles": len(articles_w), "per_source_stats": per_source_stats_w},
                    )
                    save_manifest(m, manifest_file)
                raise RuntimeError(err_msg)

        if degraded_pass:
            for art in articles:
                art["degraded_run"] = True

        for src_name, src_stats in per_source_stats.items():
            emit_office_event(
                self.run_dir,
                "scraper_source_health",
                stage="scraper",
                status="error" if src_name in errors else "ok",
                message=errors.get(src_name) or f"{src_name}: {src_stats.get('extracted_ok', 0)} articles",
                extra={"source": src_name, **src_stats},
            )

        # Validate schema contract
        is_valid, validation_err = validate_contract(articles, self.schema_contract_path)
        if not is_valid:
            raise ValueError(f"articles.json schema validation failed: {validation_err}")

        # Write output files
        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        articles_file = raw_dir / "articles.json"
        with open(articles_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        news_md_file = raw_dir / "news.md"
        self._write_news_md(
            news_md_file,
            articles,
            target_date or str(datetime.now().date()),
            per_source_stats,
            degraded=degraded_pass,
            min_eligible=min_eligible,
        )

        output_files = ["raw/articles.json", "raw/news.md"]

        scraper_metrics = {
            "total_articles": len(articles),
            "total_words": sum(a.get("word_count", 0) for a in articles),
            "selected_categories": getattr(self, "selected_categories", ["headlines"]),
            "per_source_stats": per_source_stats,
            "degraded": degraded_pass,
            "degradation_reasons": degradation_reasons,
            "robots_fallback_allowed": self.allow_robots_fallback,
        }
        manifest["selected_categories"] = getattr(self, "selected_categories", ["headlines"])

        if "stages" in manifest and "scraper" in manifest["stages"]:
            update_stage_status(
                manifest,
                "scraper",
                manifest["stages"]["scraper"].get("status", "running"),
                output_files=output_files,
                source_errors=errors if errors else None,
                metrics=scraper_metrics,
            )

        # Update manifest
        manifest_file = self.run_dir / "manifest.json"
        if manifest_file.exists():
            m = load_manifest(manifest_file)
            update_stage_status(
                m,
                "scraper",
                "done",
                output_files=output_files,
                source_errors=errors if errors else None,
                metrics=scraper_metrics,
            )
            save_manifest(m, manifest_file)

        self.logger.info(f"Scraper Agent completed successfully with {len(articles)} articles (degraded={degraded_pass}).")
        return output_files

    def _write_news_md(
        self,
        file_path: Path,
        articles: list[dict[str, Any]],
        run_date: str,
        stats: dict[str, dict[str, Any]],
        degraded: bool = False,
        min_eligible: int = 6,
    ) -> None:
        """Writes the human-readable news.md digest grouped by source."""
        lines = [
            f"# Daily News Digest — {run_date}",
            f"> Automated ingestion by `nepal-news-bot` Scraper Agent.",
            "",
        ]
        if degraded:
            lines.extend([
                f"> **DEGRADED INGEST**: Bulletin ingested under degraded conditions ({len(articles)} eligible stories, target {min_eligible}).",
                "",
            ])
        lines.extend([
            "## Ingestion Summary",
            f"- **Selected Categories**: {', '.join(getattr(self, 'selected_categories', ['headlines']))}",
            f"- **Total Articles Available**: {len(articles)}",
            f"- **Total Words**: {sum(a.get('word_count', 0) for a in articles):,}",
            "",
            "### Per-Source Breakdown",
        ])

        for src, s in stats.items():
            extra_parts: list[str] = []
            if s.get("newest_age_hours") is not None:
                extra_parts.append(f"newest: {s['newest_age_hours']}h")
            if s.get("avg_latency_ms") is not None:
                extra_parts.append(f"avg_lat: {s['avg_latency_ms']}ms")
            drop_reasons = s.get("drop_reasons") or {}
            non_zero_drops = ", ".join(f"{k}={v}" for k, v in drop_reasons.items() if v)
            if non_zero_drops:
                extra_parts.append(f"drops: {non_zero_drops}")
            suffix = f" [{'; '.join(extra_parts)}]" if extra_parts else ""
            lines.append(
                f"- **{src}**: {s.get('extracted_ok', 0)} kept "
                f"(found: {s.get('found', 0)}, dropped: {s.get('dropped', 0)}){suffix}"
            )

        lines.extend(["", "---", ""])

        # Group articles by source
        grouped: dict[str, list[dict[str, Any]]] = {}
        for a in articles:
            grouped.setdefault(a["source"], []).append(a)

        for src, items in grouped.items():
            lines.append(f"## {src} ({len(items)} stories)")
            lines.append("")
            for idx, item in enumerate(items, 1):
                pub = item.get("published_at") or "Date not specified"
                date_conf = item.get("date_confidence", "exact" if item.get("published_at") else "missing")
                date_warn = " [DATE UNVERIFIED]" if date_conf == "missing" else ""
                lines.append(f"### {idx}. {item['title']}")
                lines.append(f"- **URL**: [{item['url']}]({item['url']})")
                lines.append(
                    f"- **Published**: `{pub}` ({date_conf}{date_warn}) | "
                    f"**Lang**: `{item.get('language', 'en')}` | **Words**: {item.get('word_count', 0)}"
                )
                lines.append(
                    f"- **Signals**: cat=`{item.get('category_hint', 'other')}` | "
                    f"imp=`{item.get('importance_hint', 0.0)}` | prio=`{item.get('priority_score', 0.0)}` | pos=`#{item.get('feed_position')}` | "
                    f"trust=`{item.get('source_trust', 0.70)}` | tags=`{item.get('raw_tags', [])}`"
                )
                lines.append("")
                lines.append(f"{item['text_summary']}")
                lines.append("")
            lines.append("---")
            lines.append("")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main() -> int:
    """Standalone CLI entry point for testing and debugging the Scraper Agent."""
    parser = argparse.ArgumentParser(description="Run Scraper Agent independently.")
    parser.add_argument("--date", default=today_npt(), help="Target run date (YYYY-MM-DD)")
    parser.add_argument("--source", default=None, help="Limit scraping to one specific source name")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse without saving outputs to disk")
    parser.add_argument("--backfill", action="store_true", help="Anchor lookback window to 23:59:59 of target date instead of current scrape time")
    parser.add_argument("--lookback-hours", type=float, default=None, help="Lookback window in hours (default: config value or 30)")
    parser.add_argument("--allow-robots-fallback", action="store_true", help="Allow scraping even if robots.txt is unavailable")
    parser.add_argument("--force-source", default=None, help="Bypass circuit breaker and force scraping specific source")
    parser.add_argument("--categories", default=None, help="Comma-separated categories to scrape (e.g. 'headlines,politics,health')")
    parser.add_argument("--automated", action="store_true", help="Run in automated mode without interactive category prompt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = repo_root / "data" / "runs" / args.date

    scraper = ScraperAgent(
        run_dir=run_dir,
        backfill=args.backfill,
        lookback_hours=args.lookback_hours,
        allow_robots_fallback=args.allow_robots_fallback,
        force_source=args.force_source,
        categories=args.categories,
        interactive=False if args.automated else None,
    )
    print(f"\n========================================================")
    print(f" Nepal News Bot — Scraper Agent Standalone")
    print(f" Target Date: {args.date} | Dry Run: {args.dry_run}")
    print(f" Backfill:    {args.backfill} | Lookback: {args.lookback_hours or 'default'}")
    if scraper.selected_categories:
        print(f" Categories:  {', '.join(scraper.selected_categories)}")
    if args.source:
        print(f" Filter Source: {args.source}")
    if args.allow_robots_fallback:
        print(f" Robots Fallback: ALLOWED")
    if args.force_source:
        print(f" Force Source: {args.force_source}")
    print(f"========================================================\n")

    articles, stats, errors = scraper.scrape_all(
        target_date=args.date,
        source_filter=args.source,
        backfill=args.backfill,
        lookback_hours=args.lookback_hours,
    )

    print("\n---------------- INGESTION STATS ----------------")
    print(f"{'Source':<26} | {'Found':<6} | {'Filtered':<8} | {'Extracted':<9} | {'Dropped':<7}")
    print("-" * 65)
    for src, s in stats.items():
        print(f"{src:<26} | {s['found']:<6} | {s['kept_filter']:<8} | {s['extracted_ok']:<9} | {s['dropped']:<7}")
    print("-" * 65)
    total_words = sum(a.get("word_count", 0) for a in articles)
    print(f"TOTAL ARTICLES: {len(articles)} | TOTAL WORDS: {total_words:,}")

    if errors:
        print("\n---------------- SOURCE ERRORS ------------------")
        for src, err in errors.items():
            print(f"- {src}: {err}")

    if articles:
        sample = articles[0]
        print("\n---------------- SAMPLE ARTICLE -----------------")
        print(f"Source:       {sample['source']}")
        print(f"Title:        {sample['title']}")
        print(f"URL:          {sample['url']}")
        print(f"Published:    {sample['published_at']}")
        print(f"Word Count:   {sample['word_count']}")
        print(f"Text Summary: {sample['text_summary']}")
        print(f"Full Text:    {sample['full_text'][:250]}... [truncated]")

    if not args.dry_run:
        # Validate schema
        is_valid, err = validate_contract(articles, scraper.schema_contract_path)
        if not is_valid:
            print(f"\n[ERROR] Schema validation failed: {err}")
            return 1
        print("\n[OK] articles.json validates against schema.")

        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        articles_path = raw_dir / "articles.json"
        with open(articles_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        news_md_path = raw_dir / "news.md"
        scraper._write_news_md(news_md_path, articles, args.date, stats)
        print(f"[OK] Saved outputs to:\n  - {articles_path}\n  - {news_md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

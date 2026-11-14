"""Scraper Agent implementation for nepal-news-bot."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import difflib
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import urllib.robotparser
import zoneinfo

from bs4 import BeautifulSoup
import dateparser
import feedparser
import requests
import trafilatura
import yaml

from src.agents.base import BaseAgent
from src.utils.logging import setup_logger
from src.utils.manifest import load_manifest, save_manifest, update_stage_status
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
    for m in re.finditer(r"[.!?](?=\s|$)", candidate_slice[window_start:]):
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


def build_text_summary(
    full_text: str,
    min_cut: int = 500,
    max_cut: int = 650,
    fallback_max: int = 600,
) -> str:
    """Extracts a lead summary (~600 chars) ending cleanly at a non-abbreviation sentence boundary."""
    if len(full_text) <= fallback_max:
        return full_text

    search_region = full_text[min_cut:max_cut]
    for m in re.finditer(r"[.!?](?=\s|$)", search_region):
        abs_pos = min_cut + m.start()
        if not _is_abbreviation_boundary(full_text, abs_pos):
            return full_text[: min_cut + m.end()].strip()

    parts = full_text[:fallback_max].rsplit(None, 1)
    if len(parts) > 1 and parts[0]:
        return parts[0].rstrip() + "..."
    return full_text[:fallback_max].rstrip() + "..."


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
    "culture": "society",
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
    "health": ["स्वास्थ्य", "अस्पताल", "चिकित्सक", "बिरामी", "खोप", "रोग", "संक्रमण"],
    "crime": ["प्रहरी", "पक्राउ", "ठगी", "भ्रष्टाचार", "तस्करी", "अनुसन्धान", "मुद्दा", "अदालत"],
    "sports": [
        "sports", "cricket", "football", "soccer", "match", "tournament", "league", "npl", "run", "runs", "wicket",
        "क्रिकेट", "फुटबल", "खेल", "रन", "विकेट", "लिग", "प्रतियोगिता", "जित",
    ],
    "world": [
        "iran", "ukraine", "russia", "china", "india", "usa", "united states", "europe", "gaza", "israel",
        "अमेरिका", "भारत", "चीन", "रुस", "युक्रेन", "विदेश", "अन्तर्राष्ट्रिय",
    ],
    "tourism": ["पर्यटन", "पर्यटक", "पर्वतारोहण", "होटल", "विमानस्थल"],
    "technology": ["प्रविधि", "इन्टरनेट", "साइबर", "डिजिटल", "सफ्टवेयर"],
    "society": ["संस्कृति", "पर्व", "शिक्षा", "विद्यालय", "विश्वविद्यालय", "समुदाय"],
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
            if key in t_low:
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
                if key in seg:
                    return cat

    # Tier 3: Keyword scoring (editor.yaml categories + Nepali/sports/world extensions)
    combined = f"{title} {text_summary}".lower()
    merged_kws: dict[str, list[str]] = {}
    for cat, kws in (editor_categories or {}).items():
        merged_kws.setdefault(cat, []).extend(kws)
    for cat, kws in EXTRA_CATEGORY_KEYWORDS.items():
        merged_kws.setdefault(cat, []).extend(kws)

    best_cat = "other"
    best_score = 0
    for cat, kws in merged_kws.items():
        score = sum(1 for kw in kws if kw.lower() in combined)
        if score > best_score:
            best_score = score
            best_cat = cat

    return best_cat


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

        try:
            dt = dateparser.parse(raw_date)
            if dt is not None:
                confidence = "exact"
        except Exception:
            dt = None

    # 2. Check URL for /YYYY/MM/DD/ pattern (e.g. Kathmandu Post)
    if dt is None and url:
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if match:
            try:
                year, month, day = map(int, match.groups())
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
    ) -> None:
        super().__init__(run_dir, config)
        self.sources_path = Path(sources_path)
        self.pipeline_path = Path(pipeline_path)
        self.logger = setup_logger("scraper")
        self.last_domain_request: dict[str, float] = {}
        self._domain_locks: dict[str, threading.Lock] = {}
        self._domain_locks_guard = threading.Lock()
        self._robots_lock = threading.Lock()
        self.robot_parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.backfill = backfill or (self.config.get("backfill", False) if self.config else False)
        self.lookback_hours = lookback_hours if lookback_hours is not None else (self.config.get("lookback_hours") if self.config else None)
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.cache_dir = Path(cache_dir) if cache_dir is not None else (repo_root / "data" / "cache")
        self._editor_categories_cache: dict[str, list[str]] | None = None
        self._runtime_settings: dict[str, Any] = {}

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
        """Enforces polite crawl rate limiting per domain (thread-safe per domain)."""
        domain = urlparse(url).netloc
        if not domain:
            return
        lock = self._get_domain_lock(domain)
        with lock:
            last_time = self.last_domain_request.get(domain, 0.0)
            elapsed = time.time() - last_time
            if elapsed < delay_seconds:
                sleep_duration = delay_seconds - elapsed
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
        """Fetches a URL with polite delay, custom headers, and retries for 5xx/transient errors.

        Does NOT retry 4xx or SSL certificate errors.
        """
        headers = {"User-Agent": user_agent}
        if extra_headers:
            headers.update(extra_headers)
        attempt = 0
        backoff = 1.0

        while attempt <= max_retries:
            self._respect_request_delay(url, delay_seconds)
            t0 = time.monotonic()
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                if latency_sink is not None:
                    latency_sink.append(round((time.monotonic() - t0) * 1000.0, 1))
                # 4xx client errors: do not retry
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

    def _load_cached_robots(self, domain: str, ttl_seconds: float = 86400.0) -> tuple[bool, str | None]:
        """Returns (hit, body) from disk robots cache if within TTL."""
        try:
            safe_domain = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
            cache_file = self.cache_dir / "robots" / f"{safe_domain}.json"
            if not cache_file.exists():
                return False, None
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            fetched_at = float(data.get("fetched_at", 0.0))
            if (time.time() - fetched_at) <= ttl_seconds:
                return True, data.get("body")
        except Exception:
            pass
        return False, None

    def _save_cached_robots(self, domain: str, body: str | None) -> None:
        """Persists robots.txt body to disk cache (fail-safe)."""
        try:
            safe_domain = re.sub(r"[^a-zA-Z0-9._-]", "_", domain)
            robots_dir = self.cache_dir / "robots"
            robots_dir.mkdir(parents=True, exist_ok=True)
            cache_file = robots_dir / f"{safe_domain}.json"
            cache_file.write_text(
                json.dumps({"fetched_at": time.time(), "body": body}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _is_url_allowed_by_robots(self, url: str, user_agent: str, delay_seconds: float) -> bool:
        """Checks if a URL is permitted according to robots.txt using urllib.robotparser."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            return True

        with self._robots_lock:
            if domain not in self.robot_parsers:
                rp = urllib.robotparser.RobotFileParser()
                hit, cached_body = self._load_cached_robots(domain)
                if hit:
                    if cached_body is not None:
                        rp.parse(cached_body.splitlines())
                    else:
                        rp.allow_all = True
                else:
                    robots_url = f"{parsed.scheme}://{domain}/robots.txt"
                    try:
                        resp = self.fetch_with_retry(robots_url, user_agent, delay_seconds, timeout=10, max_retries=1)
                        if resp and resp.status_code == 200:
                            rp.parse(resp.text.splitlines())
                            self._save_cached_robots(domain, resp.text)
                        else:
                            rp.allow_all = True
                            self._save_cached_robots(domain, None)
                    except Exception as e:
                        self.logger.warning(f"Could not read robots.txt for {domain}: {e}. Defaulting to allow.")
                        rp.allow_all = True
                self.robot_parsers[domain] = rp

            rp = self.robot_parsers[domain]

        return rp.can_fetch(user_agent, url)

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
                "lookback": 0,
                "future": 0,
                "undated_low_pos": 0,
                "robots": 0,
                "url_filter": 0,
                "extract_fail": 0,
            },
            "avg_latency_ms": None,
            "max_latency_ms": None,
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
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Fetches and processes an RSS source."""
        stats = self._init_source_stats()
        latencies: list[float] = []
        error_msg = None
        articles: list[dict[str, Any]] = []

        rss_url = source.get("rss")
        if not rss_url:
            return articles, stats, "Missing RSS endpoint"

        source_name = source["name"]
        max_articles = source.get("max_articles", 20)
        source_lang = normalize_language_code(source.get("language") or default_language)

        extra_headers: dict[str, str] = {}
        cached_feed = None if self.backfill else self._load_feed_cache(source_name)
        if cached_feed:
            if cached_feed.get("etag"):
                extra_headers["If-None-Match"] = str(cached_feed["etag"])
            if cached_feed.get("last_modified"):
                extra_headers["If-Modified-Since"] = str(cached_feed["last_modified"])

        self.logger.info(f"Fetching RSS for {source_name}: {rss_url}")
        fetch_kwargs: dict[str, Any] = {"timeout": 15}
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
            self.logger.warning(f"{source_name}: {error_msg}")
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
        source_trust = round(float(source.get("trust_weight", default_trust)), 2)

        for feed_pos, entry in enumerate(entries, 1):
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

            # Parse date & apply lookback filter (or position gate if missing)
            pub_dt, date_conf = parse_article_date_with_confidence(entry, norm_url, ref_time=ref_time)
            age_hours = None
            if pub_dt is not None:
                diff_seconds = (ref_time - pub_dt).total_seconds()
                age_hours = round(diff_seconds / 3600.0, 2)
                if diff_seconds > lookback_seconds:
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

            # Extract article content
            article = self._extract_article(
                source_name=source_name,
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
                stats["drop_reasons"]["extract_fail"] += 1
                continue

            # If date was recovered from HTML metadata, enforce lookback window now
            if pub_dt is None and article.get("age_hours") is not None:
                diff_seconds = float(article["age_hours"]) * 3600.0
                if diff_seconds > lookback_seconds:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["lookback"] += 1
                    continue
                if diff_seconds < -3600:
                    stats["dropped"] += 1
                    stats["drop_reasons"]["future"] += 1
                    continue
            elif article.get("date_confidence") == "missing":
                stats["undated_kept"] += 1

            articles.append(article)
            seen_urls.add(norm_url)
            seen_urls.add(url_key)
            seen_titles.append(title)
            stats["extracted_ok"] += 1
            if len(articles) >= max_articles:
                break

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
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Fetches and processes an HTML-only source by discovering links on the homepage."""
        stats = self._init_source_stats()
        latencies: list[float] = []
        error_msg = None
        articles: list[dict[str, Any]] = []

        homepage = source.get("homepage", "").rstrip("/")
        if not homepage:
            return articles, stats, "Missing homepage URL"

        source_name = source["name"]
        max_articles = source.get("max_articles", 5)
        source_lang = normalize_language_code(source.get("language") or default_language)
        include_patterns = [re.compile(p) for p in (source.get("include_url_patterns") or [])]
        exclude_patterns = [re.compile(p) for p in (source.get("exclude_url_patterns") or [])]
        max_undated_pos = int(
            source.get(
                "max_undated_feed_position",
                self._runtime_settings.get("max_undated_feed_position", 5),
            )
        )
        default_trust = float(self._runtime_settings.get("default_trust_weight", 0.70))
        source_trust = round(float(source.get("trust_weight", default_trust)), 2)

        self.logger.info(f"Fetching HTML homepage for {source_name}: {homepage}")
        resp = self.fetch_with_retry(homepage, user_agent, delay_seconds, timeout=15)
        if not resp or resp.status_code != 200:
            status = resp.status_code if resp else "connection failure"
            error_msg = f"Failed to fetch homepage ({status})"
            self.logger.warning(f"{source_name}: {error_msg}")
            self._finalize_latency_stats(stats, latencies)
            return articles, stats, error_msg

        soup = BeautifulSoup(resp.text, "html.parser")
        base_domain = urlparse(homepage).netloc

        candidate_links: list[tuple[str, str]] = []
        seen_candidates: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            anchor_text = a.get_text(strip=True)
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = href if href.startswith("http") else f"{homepage}/{href.lstrip('/')}"
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
                continue
            seen_candidates.add(cand_key)
            candidate_links.append((full_url, anchor_text))

        stats["found"] = len(candidate_links)
        self.logger.info(f"{source_name}: Found {len(candidate_links)} candidate article links.")

        html_ages: list[float] = []
        for feed_pos, (raw_url, anchor_title) in enumerate(candidate_links, 1):
            norm_url = normalize_url(raw_url)
            url_key = canonical_url_key(norm_url)
            if norm_url in seen_urls or url_key in seen_urls:
                stats["dropped"] += 1
                stats["drop_reasons"]["dup_url"] += 1
                continue

            # Check robots.txt for path
            if not self._is_url_allowed_by_robots(norm_url, user_agent, delay_seconds):
                self.logger.info(f"Skipping robots-disallowed path: {norm_url}")
                stats["dropped"] += 1
                stats["drop_reasons"]["robots"] += 1
                continue

            stats["kept_filter"] += 1

            article = self._extract_article(
                source_name=source_name,
                title=anchor_title or "Untitled Article",
                url=norm_url,
                published_at=None,
                rss_entry={},
                user_agent=user_agent,
                delay_seconds=delay_seconds,
                language=source_lang,
                scraped_at=ref_time.isoformat(),
                age_hours=None,
                ref_time=ref_time,
                latency_sink=latencies,
                raw_tags=[],
                feed_position=feed_pos,
                source_trust=source_trust,
                date_confidence="missing",
            )

            if not article:
                stats["dropped"] += 1
                stats["drop_reasons"]["extract_fail"] += 1
                continue

            # Enforce lookback window if metadata yielded a publication timestamp
            if article.get("age_hours") is not None:
                diff_seconds = float(article["age_hours"]) * 3600.0
                if diff_seconds > lookback_seconds:
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

            if is_duplicate_title(article["title"], seen_titles, threshold=0.9):
                stats["dropped"] += 1
                stats["drop_reasons"]["dup_title"] += 1
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
        self._finalize_latency_stats(stats, latencies)
        return articles, stats, error_msg

    def _extract_article(
        self,
        source_name: str,
        title: str,
        url: str,
        published_at: str | None,
        rss_entry: dict[str, Any],
        user_agent: str,
        delay_seconds: float,
        language: str = "en",
        scraped_at: str | None = None,
        age_hours: float | None = None,
        ref_time: datetime | None = None,
        latency_sink: list[float] | None = None,
        raw_tags: list[str] | None = None,
        feed_position: int | None = None,
        source_trust: float | None = None,
        date_confidence: str | None = None,
    ) -> dict[str, Any] | None:
        """Extracts article text with trafilatura and falls back to RSS description if needed."""
        extracted_text = ""
        extracted_title = title
        author = rss_entry.get("author")
        effective_date_conf = date_confidence or ("exact" if published_at else "missing")

        # 1. Fetch HTML with polite delay
        fetch_kwargs: dict[str, Any] = {"timeout": 15}
        if latency_sink is not None:
            fetch_kwargs["latency_sink"] = latency_sink
        html_resp = self.fetch_with_retry(url, user_agent, delay_seconds, **fetch_kwargs)
        if html_resp and html_resp.status_code == 200:
            try:
                raw_text = trafilatura.extract(
                    html_resp.text,
                    url=url,
                    include_links=False,
                    include_images=False,
                    output_format="txt",
                )
                if raw_text:
                    extracted_text = clean_text_content(raw_text)

                # If original title was empty or short, extract from HTML
                if len(extracted_title) < 5:
                    meta = trafilatura.extract_metadata(html_resp.text)
                    if meta and meta.title:
                        extracted_title = meta.title.strip()
                    if meta and meta.author:
                        author = meta.author.strip()

                # Recover publication timestamp from HTML metadata when missing
                if published_at is None:
                    meta_dt = extract_html_published_at(html_resp.text, ref_time=ref_time)
                    if meta_dt is not None:
                        published_at = meta_dt.isoformat()
                        effective_date_conf = "metadata"
                        if ref_time is not None:
                            ref_ktm = ref_time.astimezone(KATHMANDU_TZ) if ref_time.tzinfo else ref_time.replace(tzinfo=KATHMANDU_TZ)
                            age_hours = round((ref_ktm - meta_dt).total_seconds() / 3600.0, 2)
            except Exception as e:
                self.logger.debug(f"Trafilatura extraction exception on {url}: {e}")

        # 2. Check if extraction succeeded or has >= 200 chars
        if len(extracted_text) < 200:
            self.logger.debug(f"Main text extraction under 200 chars ({len(extracted_text)}) for {url}. Attempting RSS fallback.")
            summary_raw = rss_entry.get("summary") or rss_entry.get("description") or ""
            if summary_raw:
                soup = BeautifulSoup(summary_raw, "html.parser")
                fallback_text = clean_text_content(soup.get_text(separator=" ", strip=True))
                if len(fallback_text) >= 50:
                    extracted_text = fallback_text
                else:
                    self.logger.info(f"Article dropped: RSS summary too short (<50 chars) for {url}")
                    return None
            else:
                self.logger.info(f"Article dropped: Extraction failed (<200 chars) and no RSS fallback for {url}")
                return None

        # Clean title
        if not extracted_title or extracted_title == "Untitled Article":
            extracted_title = title if len(title) > 5 else "Nepal News Report"

        # Compute full word count before truncation, then truncate at clean sentence boundary
        word_count = len(extracted_text.split())
        full_text = truncate_at_sentence_boundary(extracted_text, max_chars=6000)

        # Generate abbreviation-aware lead summary (~600 chars)
        text_summary = build_text_summary(full_text)

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

        article: dict[str, Any] = {
            "source": source_name,
            "title": extracted_title,
            "url": url,
            "published_at": published_at,
            "text_summary": text_summary,
            "full_text": full_text,
            "language": normalize_language_code(language),
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
            )
        elif method == "html_only":
            src_articles, stats, err = self._fetch_html_source(
                src,
                user_agent,
                delay_seconds,
                ref_time,
                local_urls,
                local_titles,
                default_language=default_language,
                lookback_seconds=lookback_seconds,
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

    def scrape_all(
        self,
        target_date: str | None = None,
        source_filter: str | None = None,
        backfill: bool | None = None,
        lookback_hours: float | None = None,
        ref_time: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
        """Executes scraping for all active sources (or a single filtered source).

        Returns:
            (articles, per_source_stats, source_errors)
        """
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
                    )
                )

        all_articles: list[dict[str, Any]] = []
        per_source_stats: dict[str, dict[str, Any]] = {}
        source_errors: dict[str, str] = {}
        seen_urls: set[str] = set()

        for name, src_articles, stats, err in ordered_results:
            kept_articles: list[dict[str, Any]] = []
            for art in src_articles:
                url_key = canonical_url_key(art.get("url", ""))
                if url_key and url_key in seen_urls:
                    stats["dropped"] += 1
                    stats["extracted_ok"] = max(0, stats["extracted_ok"] - 1)
                    if "drop_reasons" in stats:
                        stats["drop_reasons"]["dup_url"] += 1
                    continue
                if url_key:
                    seen_urls.add(url_key)
                kept_articles.append(art)

            all_articles.extend(kept_articles)
            per_source_stats[name] = stats
            if err:
                source_errors[name] = err

        return all_articles, per_source_stats, source_errors

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Executes the Scraper Agent within the multi-agent pipeline."""
        target_date = manifest.get("date")
        pipeline_cfg = self._load_pipeline_config()

        # Extract min_articles from scraper stage in pipeline.yaml (default 5)
        min_articles = 5
        for st in pipeline_cfg.get("stages", []):
            if st.get("id") == "scraper":
                min_articles = st.get("min_articles", 5)
                break

        self.logger.info(f"Running Scraper Agent for run date: {target_date} (min_articles={min_articles})")

        articles, per_source_stats, errors = self.scrape_all(
            target_date=target_date,
            backfill=self.backfill,
            lookback_hours=self.lookback_hours,
        )

        for src_name, src_stats in per_source_stats.items():
            emit_office_event(
                self.run_dir,
                "scraper_source_health",
                stage="scraper",
                status="error" if src_name in errors else "ok",
                message=errors.get(src_name) or f"{src_name}: {src_stats.get('extracted_ok', 0)} articles",
                extra={"source": src_name, **src_stats},
            )

        # Check minimum articles threshold
        if len(articles) < min_articles:
            err_msg = f"INSUFFICIENT_SOURCES: Scraped {len(articles)} articles, below minimum threshold ({min_articles})."
            self.logger.error(err_msg)
            # Update manifest with failure info
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
        self._write_news_md(news_md_file, articles, target_date or str(datetime.now().date()), per_source_stats)

        output_files = ["raw/articles.json", "raw/news.md"]

        scraper_metrics = {
            "total_articles": len(articles),
            "total_words": sum(a.get("word_count", 0) for a in articles),
            "per_source_stats": per_source_stats,
        }

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

        self.logger.info(f"Scraper Agent completed successfully with {len(articles)} articles.")
        return output_files

    def _write_news_md(
        self,
        file_path: Path,
        articles: list[dict[str, Any]],
        run_date: str,
        stats: dict[str, dict[str, Any]],
    ) -> None:
        """Writes the human-readable news.md digest grouped by source."""
        lines = [
            f"# Daily News Digest — {run_date}",
            f"> Automated ingestion by `nepal-news-bot` Scraper Agent.",
            "",
            "## Ingestion Summary",
            f"- **Total Articles Available**: {len(articles)}",
            f"- **Total Words**: {sum(a.get('word_count', 0) for a in articles):,}",
            "",
            "### Per-Source Breakdown",
        ]

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
                    f"imp=`{item.get('importance_hint', 0.0)}` | pos=`#{item.get('feed_position')}` | "
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
    parser.add_argument("--date", default=str(datetime.now().date()), help="Target run date (YYYY-MM-DD)")
    parser.add_argument("--source", default=None, help="Limit scraping to one specific source name")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse without saving outputs to disk")
    parser.add_argument("--backfill", action="store_true", help="Anchor lookback window to 23:59:59 of target date instead of current scrape time")
    parser.add_argument("--lookback-hours", type=float, default=None, help="Lookback window in hours (default: config value or 30)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = repo_root / "data" / "runs" / args.date

    scraper = ScraperAgent(
        run_dir=run_dir,
        backfill=args.backfill,
        lookback_hours=args.lookback_hours,
    )
    print(f"\n========================================================")
    print(f" Nepal News Bot — Scraper Agent Standalone")
    print(f" Target Date: {args.date} | Dry Run: {args.dry_run}")
    print(f" Backfill:    {args.backfill} | Lookback: {args.lookback_hours or 'default'}")
    if args.source:
        print(f" Filter Source: {args.source}")
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

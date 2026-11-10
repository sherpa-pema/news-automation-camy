"""Scraper Agent implementation for nepal-news-bot."""
import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import sys
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


def parse_article_date(entry: dict[str, Any], url: str) -> datetime | None:
    """Extracts and parses publication date from feed entry or URL structure.

    Returns timezone-aware datetime in Asia/Kathmandu, or None if unparseable.
    """
    raw_date = None
    # 1. Check feed entry fields
    for key in ("published", "pubDate", "updated", "dc:date"):
        if entry.get(key):
            raw_date = str(entry[key])
            break

    dt = None
    if raw_date:
        # Feedparser sometimes provides parsed struct_time
        if entry.get("published_parsed"):
            try:
                st = entry["published_parsed"]
                # Convert struct_time (UTC) to datetime
                dt_utc = datetime(*st[:6], tzinfo=timezone.utc)
                return dt_utc.astimezone(KATHMANDU_TZ)
            except Exception:
                pass

        try:
            dt = dateparser.parse(raw_date)
        except Exception:
            dt = None

    # 2. Check URL for /YYYY/MM/DD/ pattern (e.g. Kathmandu Post)
    if dt is None and url:
        match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
        if match:
            try:
                year, month, day = map(int, match.groups())
                dt = datetime(year, month, day, 12, 0, 0)
            except Exception:
                dt = None

    if dt is None:
        return None

    # Ensure timezone awareness
    if dt.tzinfo is None:
        # Default assume Kathmandu local time if no offset was provided
        dt = dt.replace(tzinfo=KATHMANDU_TZ)
    else:
        dt = dt.astimezone(KATHMANDU_TZ)

    return dt


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
    ) -> None:
        super().__init__(run_dir, config)
        self.sources_path = Path(sources_path)
        self.pipeline_path = Path(pipeline_path)
        self.logger = setup_logger("scraper")
        self.last_domain_request: dict[str, float] = {}
        self.robot_parsers: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.backfill = backfill or (self.config.get("backfill", False) if self.config else False)
        self.lookback_hours = lookback_hours if lookback_hours is not None else (self.config.get("lookback_hours") if self.config else None)

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
        """Enforces polite crawl rate limiting per domain."""
        domain = urlparse(url).netloc
        if not domain:
            return
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
    ) -> requests.Response | None:
        """Fetches a URL with polite delay, custom headers, and retries for 5xx/transient errors.

        Does NOT retry 4xx errors.
        """
        headers = {"User-Agent": user_agent}
        attempt = 0
        backoff = 1.0

        while attempt <= max_retries:
            self._respect_request_delay(url, delay_seconds)
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                # 4xx client errors: do not retry
                if 400 <= response.status_code < 500:
                    self.logger.warning(f"Client error HTTP {response.status_code} for {url}. No retry.")
                    return response

                # 5xx server errors: transient, retry with backoff
                if response.status_code >= 500:
                    self.logger.warning(f"Server error HTTP {response.status_code} on {url} (attempt {attempt+1}/{max_retries+1})")
                    attempt += 1
                    if attempt <= max_retries:
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    return response

                return response

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self.logger.warning(f"Network error ({type(e).__name__}) for {url} (attempt {attempt+1}/{max_retries+1}): {e}")
                attempt += 1
                if attempt <= max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                return None
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        return None

    def _is_url_allowed_by_robots(self, url: str, user_agent: str, delay_seconds: float) -> bool:
        """Checks if a URL is permitted according to robots.txt using urllib.robotparser."""
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            return True

        if domain not in self.robot_parsers:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = self.fetch_with_retry(robots_url, user_agent, delay_seconds, timeout=10, max_retries=1)
                if resp and resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    rp.allow_all = True
            except Exception as e:
                self.logger.warning(f"Could not read robots.txt for {domain}: {e}. Defaulting to allow.")
                rp.allow_all = True
            self.robot_parsers[domain] = rp

        rp = self.robot_parsers[domain]
        allowed = rp.can_fetch(user_agent, url)
        return allowed

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
        stats: dict[str, Any] = {
            "found": 0,
            "kept_filter": 0,
            "extracted_ok": 0,
            "dropped": 0,
            "newest_age_hours": None,
        }
        error_msg = None
        articles: list[dict[str, Any]] = []

        rss_url = source.get("rss")
        if not rss_url:
            return articles, stats, "Missing RSS endpoint"

        source_name = source["name"]
        max_articles = source.get("max_articles", 20)
        source_lang = normalize_language_code(source.get("language") or default_language)

        self.logger.info(f"Fetching RSS for {source_name}: {rss_url}")
        resp = self.fetch_with_retry(rss_url, user_agent, delay_seconds, timeout=15)
        if not resp or resp.status_code != 200:
            status = resp.status_code if resp else "timeout/connection failure"
            error_msg = f"Failed to fetch RSS feed ({status})"
            self.logger.warning(f"{source_name}: {error_msg}")
            return articles, stats, error_msg

        feed = feedparser.parse(resp.content)
        entries = feed.entries or []
        stats["found"] = len(entries)
        self.logger.info(f"{source_name}: {len(entries)} entries found in feed.")

        feed_ages: list[float] = []
        for entry in entries:
            e_url = normalize_url(getattr(entry, "link", ""))
            e_dt = parse_article_date(entry, e_url)
            if e_dt is not None:
                e_diff = (ref_time - e_dt).total_seconds()
                if e_diff >= -3600:
                    feed_ages.append(max(0.0, round(e_diff / 3600.0, 2)))
        if feed_ages:
            stats["newest_age_hours"] = min(feed_ages)

        for entry in entries:
            raw_url = getattr(entry, "link", "")
            title = getattr(entry, "title", "").strip()
            if not raw_url or not title:
                stats["dropped"] += 1
                continue

            norm_url = normalize_url(raw_url)
            # Deduplicate by URL
            if norm_url in seen_urls:
                self.logger.debug(f"Dropping duplicate URL: {norm_url}")
                stats["dropped"] += 1
                continue

            # Deduplicate by Title similarity (> 0.9)
            if is_duplicate_title(title, seen_titles, threshold=0.9):
                self.logger.debug(f"Dropping duplicate title: {title}")
                stats["dropped"] += 1
                continue

            # Parse date & apply lookback filter
            pub_dt = parse_article_date(entry, norm_url)
            age_hours = None
            if pub_dt is not None:
                diff_seconds = (ref_time - pub_dt).total_seconds()
                age_hours = round(diff_seconds / 3600.0, 2)
                # If older than lookback window or more than 1h in the future (-3600s)
                if diff_seconds > lookback_seconds or diff_seconds < -3600:
                    self.logger.debug(f"Dropping article outside lookback window ({pub_dt.isoformat()}): {title}")
                    stats["dropped"] += 1
                    continue
                pub_iso = pub_dt.isoformat()
            else:
                # Prompt rule: If feed lacks dates, keep entry and set published_at to null
                pub_iso = None

            stats["kept_filter"] += 1

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
            )

            if article:
                articles.append(article)
                seen_urls.add(norm_url)
                seen_titles.append(title)
                stats["extracted_ok"] += 1
                if len(articles) >= max_articles:
                    break
            else:
                stats["dropped"] += 1

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
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Fetches and processes an HTML-only source by discovering links on the homepage."""
        stats: dict[str, Any] = {
            "found": 0,
            "kept_filter": 0,
            "extracted_ok": 0,
            "dropped": 0,
            "newest_age_hours": None,
        }
        error_msg = None
        articles: list[dict[str, Any]] = []

        homepage = source.get("homepage", "").rstrip("/")
        if not homepage:
            return articles, stats, "Missing homepage URL"

        source_name = source["name"]
        max_articles = source.get("max_articles", 5)
        source_lang = normalize_language_code(source.get("language") or default_language)

        self.logger.info(f"Fetching HTML homepage for {source_name}: {homepage}")
        resp = self.fetch_with_retry(homepage, user_agent, delay_seconds, timeout=15)
        if not resp or resp.status_code != 200:
            status = resp.status_code if resp else "connection failure"
            error_msg = f"Failed to fetch homepage ({status})"
            self.logger.warning(f"{source_name}: {error_msg}")
            return articles, stats, error_msg

        soup = BeautifulSoup(resp.text, "html.parser")
        base_domain = urlparse(homepage).netloc

        candidate_links: list[tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            anchor_text = a.get_text(strip=True)
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = href if href.startswith("http") else f"{homepage}/{href.lstrip('/')}"
            parsed = urlparse(full_url)
            # Only follow links on same domain
            if parsed.netloc == base_domain and len(parsed.path) > 5:
                # Basic check to avoid index pages
                if not any(parsed.path.endswith(ext) for ext in (".css", ".js", ".png", ".jpg", ".svg", ".pdf")):
                    candidate_links.append((full_url, anchor_text))

        stats["found"] = len(candidate_links)
        self.logger.info(f"{source_name}: Found {len(candidate_links)} internal links.")

        for raw_url, anchor_title in candidate_links:
            norm_url = normalize_url(raw_url)
            if norm_url in seen_urls:
                stats["dropped"] += 1
                continue

            # Check robots.txt for path
            if not self._is_url_allowed_by_robots(norm_url, user_agent, delay_seconds):
                self.logger.info(f"Skipping robots-disallowed path: {norm_url}")
                stats["dropped"] += 1
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
            )

            if article:
                # Update title from extracted page if anchor text was weak
                if is_duplicate_title(article["title"], seen_titles, threshold=0.9):
                    stats["dropped"] += 1
                    continue

                articles.append(article)
                seen_urls.add(norm_url)
                seen_titles.append(article["title"])
                stats["extracted_ok"] += 1
                if len(articles) >= max_articles:
                    break
            else:
                stats["dropped"] += 1

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
    ) -> dict[str, Any] | None:
        """Extracts article text with trafilatura and falls back to RSS description if needed."""
        extracted_text = ""
        extracted_title = title
        author = rss_entry.get("author")

        # 1. Fetch HTML with polite delay
        html_resp = self.fetch_with_retry(url, user_agent, delay_seconds, timeout=15)
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

        # Cap full text at 6000 characters
        full_text = extracted_text[:6000].strip()

        # Generate summary (~600 chars)
        if len(full_text) <= 600:
            text_summary = full_text
        else:
            # Try to cut at sentence end within 500-650 range
            match = re.search(r"[.!?]\s", full_text[500:650])
            if match:
                cut_idx = 500 + match.end()
                text_summary = full_text[:cut_idx].strip()
            else:
                text_summary = full_text[:600].rstrip() + "..."

        word_count = len(full_text.split())

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
        }
        return article

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

        all_articles: list[dict[str, Any]] = []
        per_source_stats: dict[str, dict[str, Any]] = {}
        source_errors: dict[str, str] = {}
        seen_urls: set[str] = set()
        seen_titles: list[str] = []

        sources_list = sources_cfg.get("sources", [])
        for src in sources_list:
            name = src.get("name", "Unknown Source")
            if source_filter and source_filter.lower() not in name.lower():
                continue

            if not src.get("robots_ok", True):
                self.logger.info(f"Skipping {name}: robots_ok is false.")
                continue

            if not src.get("enabled", True):
                self.logger.info(f"Skipping {name}: enabled is false.")
                continue

            method = src.get("method", "rss")
            if method == "rss":
                src_articles, stats, err = self._fetch_rss_source(
                    src,
                    user_agent,
                    delay_seconds,
                    ref_time,
                    seen_urls,
                    seen_titles,
                    default_language=default_language,
                    lookback_seconds=lookback_seconds,
                )
            elif method == "html_only":
                src_articles, stats, err = self._fetch_html_source(
                    src, user_agent, delay_seconds, ref_time, seen_urls, seen_titles, default_language=default_language
                )
            else:
                self.logger.warning(f"Unknown method '{method}' for {name}")
                continue

            all_articles.extend(src_articles)
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

        articles, stats, errors = self.scrape_all(
            target_date=target_date,
            backfill=self.backfill,
            lookback_hours=self.lookback_hours,
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
                    metrics={"total_articles": len(articles), "per_source_stats": stats},
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
        self._write_news_md(news_md_file, articles, target_date or str(datetime.now().date()), stats)

        output_files = ["raw/articles.json", "raw/news.md"]

        scraper_metrics = {
            "total_articles": len(articles),
            "total_words": sum(a.get("word_count", 0) for a in articles),
            "per_source_stats": stats,
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
            lines.append(f"- **{src}**: {s.get('extracted_ok', 0)} kept (found: {s.get('found', 0)}, dropped: {s.get('dropped', 0)})")

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
                lines.append(f"### {idx}. {item['title']}")
                lines.append(f"- **URL**: [{item['url']}]({item['url']})")
                lines.append(f"- **Published**: `{pub}` | **Words**: {item.get('word_count', 0)}")
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

# Scraper Agent Specification

## 1. Purpose
The Scraper Agent ingests today's news from verified Nepali news publishers. It prioritizes official RSS/Atom feeds, parses full article content with polite request throttling, respects publisher `robots.txt` directives, and outputs normalized structured article records alongside a combined Markdown digest.

## 2. Ground Rules
- **RSS First**: Always consume structured RSS/Atom feeds before attempting HTML scraping.
- **Polite Crawling**: Obey robots.txt disallow rules, use the custom User-Agent `NepalNewsBot/0.1 (personal project)`, and maintain at least 2 seconds delay between requests.
- **Data Integrity**: Retain publisher attribution, canonical URL, and publication timestamp for every article.
- **Stateless Operation**: Reads `config/sources.yaml` and writes solely to the specified run directory.

## 3. Inputs & Outputs
- **Inputs**:
  - `config/sources.yaml`: Verified sources list, RSS endpoints, request delays, and limits.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/raw/articles.json`: Structured list of parsed articles.
  - `data/runs/<YYYY-MM-DD>/raw/news.md`: Human-readable aggregated Markdown digest of the day's articles.

## 4. JSON Schema Contract
- Must satisfy: [articles.schema.json](../contracts/articles.schema.json)
- Expected fields per item: `source`, `title`, `url`, `published_at`, `text_summary`.

## 5. Tools & Modules
- `feedparser`: RSS and Atom XML parsing.
- `requests`: HTTP requests with custom headers and timeouts.
- `beautifulsoup4`: HTML cleanup and metadata extraction.
- `trafilatura`: Main body text extraction and boilerplate removal.

## 6. Failure Modes
- **Feed Offline or 500 Error**: Log warning, skip failing source, continue with other enabled sources.
- **Anti-Scraping / Captcha**: Log error, flag source in run logs, do not crash pipeline if minimum articles threshold is met.
- **Insufficient Articles (< 5 articles total)**: Fail stage with error code `INSUFFICIENT_SOURCES` to trigger retry.

## 7. Definition of Done Checklist
- [ ] Ingested articles from all active sources in `sources.yaml`.
- [ ] Verified `robots.txt` compliance and request delay of >= 2 seconds.
- [ ] At least 10 articles parsed across enabled outlets.
- [ ] Output `raw/articles.json` validates against `articles.schema.json`.
- [ ] Output `raw/news.md` written with source citations.

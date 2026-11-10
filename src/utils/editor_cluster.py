"""Clustering and shortlist generation for Editor Agent."""
from datetime import datetime
import difflib
import re
from typing import Any

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "nepal", "nepali", "says", "said"
}

OPINION_URL_PATTERNS = [
    "/opinion/", "/columns/", "/column/", "/commentary/",
    "/blogs/", "/blog/", "/op-ed/", "/editorial/", "/views/",
]


def tokenize(text: str) -> set[str]:
    """Extracts normalized alphanumeric word tokens excluding stopwords."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def is_advertisement(article: dict[str, Any]) -> bool:
    """Checks if an article is explicitly an advertisement or press release."""
    title = article.get("title", "").lower()
    url = article.get("url", "").lower()
    for marker in ["advertorial", "sponsored", "advertisement", "press release", "press-release"]:
        if marker in url or marker in title:
            return True
    return False


def detect_opinion(article: dict[str, Any]) -> tuple[bool, str]:
    """Detects whether an article is an opinion, editorial, column, or first-person commentary."""
    url = article.get("url", "").lower()
    title = article.get("title", "").lower()
    author = (article.get("author") or "").lower()
    full_text = article.get("full_text") or article.get("text_summary") or ""

    # 1. URL pattern
    for pattern in OPINION_URL_PATTERNS:
        if pattern in url:
            return True, f"URL pattern '{pattern}'"

    # 2. Title pattern
    title_match = re.search(r"\b(opinion|editorial|commentary|op-ed|column|perspective)\b", title, re.IGNORECASE)
    if title_match:
        return True, f"Title pattern '{title_match.group(1)}'"

    # 3. Author pattern
    author_match = re.search(r"\b(columnist|contributor|guest author)\b", author, re.IGNORECASE)
    if author_match:
        return True, f"Author designation '{author_match.group(1)}'"

    # 4. First-person pronoun heuristic
    words = re.findall(r"\b[A-Za-z]+\b", full_text)
    fp_count = 0
    fp_examples = []
    for w in words:
        if w in ("I", "my", "My", "mine", "Mine", "myself", "Myself", "we", "We", "our", "Our", "ours", "Ours", "ourselves", "Ourselves", "us"):
            fp_count += 1
            if len(fp_examples) < 5:
                fp_examples.append(w)

    if fp_count >= 5:
        return True, f"First-person pronoun count ({fp_count}) >= 5 (e.g. {', '.join(fp_examples)})"

    return False, ""


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_time_str(time_str: str) -> tuple[int, int]:
    """Parses a time string like '5pm', '5:00 pm', '17:00' into (hour, minute)."""
    clean_str = time_str.strip().lower().replace(".", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", clean_str)
    if not m:
        return (0, 0)
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return (hour, minute)


def detect_stale_preview(article: dict[str, Any], ref_time: datetime | None = None) -> tuple[bool, str]:
    """Detects whether an article is a preview of a scheduled future event whose event time has already passed."""
    text = (article.get("title", "") + " " + (article.get("full_text") or article.get("text_summary") or ""))
    m_phrase = re.search(r"\b(will|scheduled|set to|to face|to host)\b", text, re.IGNORECASE)
    if not m_phrase:
        return False, ""

    time_matches = list(re.finditer(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.))\b|\b([01]?\d|2[0-3]):([0-5]\d)\b", text, re.IGNORECASE))
    if not time_matches:
        return False, ""

    pub_str = article.get("published_at") or ""
    try:
        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
    except Exception:
        pub_dt = datetime.now().astimezone()

    if ref_time is None:
        ref_time = datetime.now(pub_dt.tzinfo)

    for tm in time_matches:
        time_str = tm.group(0)
        hour, minute = parse_time_str(time_str)

        start = max(0, tm.start() - 50)
        end = min(len(text), tm.end() + 50)
        window = text[start:end].lower()

        event_year = pub_dt.year
        event_month = pub_dt.month
        event_day = pub_dt.day

        m_date = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})\b", window)
        if m_date:
            event_month = MONTHS[m_date.group(1)]
            event_day = int(m_date.group(2))
        else:
            m_date2 = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", window)
            if m_date2:
                event_day = int(m_date2.group(1))
                event_month = MONTHS[m_date2.group(2)]

        try:
            event_dt = datetime(event_year, event_month, event_day, hour, minute, tzinfo=pub_dt.tzinfo)
        except Exception:
            continue

        if event_dt < ref_time:
            return True, f"Future event preview ('{m_phrase.group()}', time '{time_str}') on {event_dt.strftime('%Y-%m-%d %H:%M')} has passed relative to {ref_time.strftime('%Y-%m-%d %H:%M')}"

    return False, ""


def is_opinion_or_ad(article: dict[str, Any]) -> bool:
    """Legacy helper for backward compatibility."""
    if is_advertisement(article):
        return True
    is_op, _ = detect_opinion(article)
    return is_op


def is_thin_story(article: dict[str, Any], min_words: int = 60) -> bool:
    """Checks if article body is thin (under min_words)."""
    full_text = article.get("full_text") or article.get("text_summary") or ""
    words = full_text.strip().split()
    return len(words) < min_words


def guess_category(text: str, categories_cfg: dict[str, list[str]]) -> str:
    """Infers the most likely category based on keyword matches."""
    text_lower = text.lower()
    best_cat = "other"
    max_score = 0

    for cat, kws in categories_cfg.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > max_score:
            max_score = score
            best_cat = cat

    return best_cat


def articles_are_similar(a1: dict[str, Any], a2: dict[str, Any], title_ratio_thresh: float = 0.55, token_overlap_thresh: float = 0.35) -> bool:
    """Determines whether two articles cover the same event."""
    t1 = a1.get("title", "")
    t2 = a2.get("title", "")

    # 1. SequenceMatcher ratio on titles
    ratio = difflib.SequenceMatcher(None, t1.lower(), t2.lower()).ratio()
    if ratio >= title_ratio_thresh:
        return True

    # 2. Token overlap on titles
    toks1 = tokenize(t1)
    toks2 = tokenize(t2)
    if toks1 and toks2:
        overlap = len(toks1 & toks2) / len(toks1 | toks2)
        if overlap >= token_overlap_thresh:
            return True

    # 3. Combined title + summary overlap
    s1 = tokenize(a1.get("text_summary", ""))
    s2 = tokenize(a2.get("text_summary", ""))
    comb1 = toks1 | s1
    comb2 = toks2 | s2
    if comb1 and comb2:
        comb_overlap = len(comb1 & comb2) / len(comb1 | comb2)
        if comb_overlap >= 0.40 and len(toks1 & toks2) >= 2:
            return True

    return False


def parse_timestamp(ts_str: str | None) -> float:
    """Parses published_at string into epoch timestamp."""
    if not ts_str:
        return 0.0
    try:
        # ISO format
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        pass
    try:
        # RFC 2822
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(ts_str)
        return dt.timestamp()
    except Exception:
        return 0.0


def cluster_articles(
    articles: list[dict[str, Any]],
    editor_cfg: dict[str, Any],
    return_opinion: bool = False,
    return_stale: bool = False,
    ref_time: datetime | None = None,
) -> list[dict[str, Any]] | tuple[Any, ...]:
    """Filters and clusters articles about the same event across outlets."""
    min_words = int(editor_cfg.get("clustering", {}).get("min_full_text_words", 60))
    title_ratio = float(editor_cfg.get("clustering", {}).get("title_similarity_threshold", 0.55))
    token_overlap = float(editor_cfg.get("clustering", {}).get("token_overlap_threshold", 0.35))
    categories_cfg = editor_cfg.get("categories", {})
    allow_opinion = bool(editor_cfg.get("allow_opinion", False))
    allow_stale_preview = bool(editor_cfg.get("allow_stale_preview", False))

    # 1. Filter out ads and thin stories, detect opinion and stale preview on articles
    filtered_articles = []
    for art in articles:
        if is_advertisement(art):
            continue
        if is_thin_story(art, min_words=min_words):
            continue
        is_op, op_reason = detect_opinion(art)
        art["is_opinion"] = is_op
        art["opinion_reason"] = op_reason if is_op else None

        is_stale, stale_reason = detect_stale_preview(art, ref_time=ref_time)
        art["is_stale_preview"] = is_stale
        art["stale_preview_reason"] = stale_reason if is_stale else None

        filtered_articles.append(art)

    # 2. Cluster articles
    clusters: list[list[dict[str, Any]]] = []

    for art in filtered_articles:
        placed = False
        for c in clusters:
            # Check similarity against any existing article in cluster
            if any(articles_are_similar(art, existing, title_ratio, token_overlap) for existing in c):
                c.append(art)
                placed = True
                break
        if not placed:
            clusters.append([art])

    # 3. Format clusters
    formatted_clusters = []
    for idx, c in enumerate(clusters, 1):
        sources = sorted(list({a["source"] for a in c}))
        urls = [a["url"] for a in c]
        all_text = " ".join([f"{a.get('title', '')} {a.get('text_summary', '')}" for a in c])
        category = guess_category(all_text, categories_cfg)
        newest_ts = max((parse_timestamp(a.get("published_at")) for a in c), default=0.0)

        # Cluster opinion status: True if any member article is opinion
        is_opinion_cluster = any(a.get("is_opinion", False) for a in c)
        opinion_reasons = [a.get("opinion_reason") for a in c if a.get("opinion_reason")]
        op_reason_str = "; ".join(opinion_reasons) if opinion_reasons else None

        # Cluster stale preview status: True if any member article is stale preview
        is_stale_cluster = any(a.get("is_stale_preview", False) for a in c)
        stale_reasons = [a.get("stale_preview_reason") for a in c if a.get("stale_preview_reason")]
        stale_reason_str = "; ".join(stale_reasons) if stale_reasons else None

        # Primary title: pick longest or from multi-outlet
        primary_title = max((a.get("title", "") for a in c), key=len, default="News Item")

        formatted_clusters.append({
            "cluster_id": f"cluster_{idx:02d}",
            "primary_title": primary_title,
            "is_opinion": is_opinion_cluster,
            "opinion_reason": op_reason_str,
            "is_stale_preview": is_stale_cluster,
            "stale_preview_reason": stale_reason_str,
            "sources": sources,
            "source_count": len(sources),
            "urls": urls,
            "article_count": len(c),
            "newest_timestamp": newest_ts,
            "category_guess": category,
            "articles": [
                {
                    "source": a["source"],
                    "title": a["title"],
                    "url": a["url"],
                    "published_at": a.get("published_at"),
                    "text_summary": a.get("text_summary", ""),
                    "full_text": a.get("full_text", a.get("text_summary", "")),
                    "is_opinion": a.get("is_opinion", False),
                    "opinion_reason": a.get("opinion_reason"),
                    "is_stale_preview": a.get("is_stale_preview", False),
                    "stale_preview_reason": a.get("stale_preview_reason"),
                }
                for a in c
            ],
        })

    # 4. Rank clusters: more outlets first, then recency, then topic-variety penalty
    def cluster_sort_key(c: dict[str, Any]) -> tuple[int, float]:
        return (c["source_count"], c["newest_timestamp"])

    initial_sorted = sorted(formatted_clusters, key=cluster_sort_key, reverse=True)

    ranked_clusters = []
    category_counts: dict[str, int] = {}

    while initial_sorted:
        best_cand = None
        best_score = -1e9
        best_cand_idx = -1

        for i, cand in enumerate(initial_sorted):
            cat = cand["category_guess"]
            cat_count = category_counts.get(cat, 0)
            recency_normalized = cand["newest_timestamp"] / 1e9 if cand["newest_timestamp"] else 0.0
            score = (cand["source_count"] * 1000.0) + recency_normalized - (cat_count * 150.0)
            if score > best_score:
                best_score = score
                best_cand = cand
                best_cand_idx = i

        if best_cand:
            initial_sorted.pop(best_cand_idx)
            cat = best_cand["category_guess"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
            ranked_clusters.append(best_cand)

    # Separate news clusters from opinion and stale preview clusters
    news_clusters = [
        c for c in ranked_clusters
        if not c.get("is_opinion", False) and not c.get("is_stale_preview", False)
    ]
    opinion_clusters = [c for c in ranked_clusters if c.get("is_opinion", False)]
    stale_clusters = [
        c for c in ranked_clusters
        if not c.get("is_opinion", False) and c.get("is_stale_preview", False)
    ]

    # Re-index cluster_id cleanly
    for idx, c in enumerate(news_clusters, 1):
        c["cluster_id"] = f"cluster_{idx:02d}"

    for idx, c in enumerate(opinion_clusters, 1):
        c["cluster_id"] = f"opinion_{idx:02d}"

    for idx, c in enumerate(stale_clusters, 1):
        c["cluster_id"] = f"stale_{idx:02d}"

    top_limit = int(editor_cfg.get("clustering", {}).get("top_clusters_limit", 12))

    if allow_opinion and allow_stale_preview:
        shortlist = ranked_clusters[:top_limit]
        for idx, c in enumerate(shortlist, 1):
            c["cluster_id"] = f"cluster_{idx:02d}"
    elif allow_opinion and not allow_stale_preview:
        shortlist = [c for c in ranked_clusters if not c.get("is_stale_preview", False)][:top_limit]
        for idx, c in enumerate(shortlist, 1):
            c["cluster_id"] = f"cluster_{idx:02d}"
    elif not allow_opinion and allow_stale_preview:
        shortlist = [c for c in ranked_clusters if not c.get("is_opinion", False)][:top_limit]
        for idx, c in enumerate(shortlist, 1):
            c["cluster_id"] = f"cluster_{idx:02d}"
    else:
        shortlist = news_clusters[:top_limit]

    if return_opinion and return_stale:
        return shortlist, opinion_clusters, stale_clusters
    if return_opinion:
        return shortlist, opinion_clusters
    return shortlist

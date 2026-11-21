"""LLM-powered semantic news curation, clustering, and multi-source synthesis for EditorAgent.

Curates raw news into clean, national-priority event clusters, resolves differing angles
and languages across outlets, flags complementary information for multi-source synthesis,
and assigns accurate journalistic categories.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from typing import Any

from src.utils.editor_cluster import (
    DEFAULT_ENTITY_ALIASES,
    apply_cross_day_deduplication,
    detect_opinion,
    detect_stale_preview,
    extract_spine_numbers,
    is_advertisement,
    is_thin_story,
    parse_timestamp,
    rank_clusters_with_diversity,
)
from src.utils.llm_writer import _call_llm, _extract_json_object, detect_configured_provider

logger = logging.getLogger("editor")

CANONICAL_BROADCAST_CATEGORIES = [
    "national_politics",
    "society_local",
    "culture_entertainment",
    "business_economics",
    "science_technology",
    "sports",
]

BROADCAST_SLOTS: list[tuple[str, str, str]] = [
    ("slot_01", "national_politics", "National Politics"),
    ("slot_02", "society_local", "Society and Local News"),
    ("slot_03", "culture_entertainment", "Culture and Entertainment"),
    ("slot_04", "business_economics", "Business and Economics"),
    ("slot_05", "science_technology", "Science and Technology"),
    ("slot_06", "sports", "Sports"),
]

CATEGORY_TO_BROADCAST_SLOT: dict[str, str] = {
    "national_politics": "national_politics",
    "politics": "national_politics",
    "political": "national_politics",
    "government": "national_politics",
    "parliament": "national_politics",
    "society_local": "society_local",
    "society": "society_local",
    "social": "society_local",
    "local": "society_local",
    "transport": "society_local",
    "disaster": "society_local",
    "crime": "society_local",
    "health": "society_local",
    "weather": "society_local",
    "climate": "society_local",
    "environment": "society_local",
    "culture_entertainment": "culture_entertainment",
    "culture": "culture_entertainment",
    "cultural": "culture_entertainment",
    "entertainment": "culture_entertainment",
    "heritage": "culture_entertainment",
    "tourism": "culture_entertainment",
    "business_economics": "business_economics",
    "economy": "business_economics",
    "economic": "business_economics",
    "business": "business_economics",
    "market": "business_economics",
    "finance": "business_economics",
    "science_technology": "science_technology",
    "technology": "science_technology",
    "tech": "science_technology",
    "science": "science_technology",
    "sports": "sports",
    "sport": "sports",
    "cricket": "sports",
    "football": "sports",
    "international": "society_local",
    "world": "society_local",
}


def map_to_broadcast_slot(cat: str | None) -> str:
    """Maps any canonical or legacy category string to one of the 6 broadcast slots."""
    if not cat:
        return "society_local"
    return CATEGORY_TO_BROADCAST_SLOT.get(str(cat).strip().lower(), "society_local")


ALLOWED_CATEGORIES = [
    "national_politics",
    "society_local",
    "culture_entertainment",
    "business_economics",
    "science_technology",
    "sports",
    "politics",
    "economy",
    "disaster",
    "society",
    "transport",
    "crime",
    "weather",
    "climate",
    "health",
    "tourism",
    "culture",
    "entertainment",
    "science",
    "technology",
    "international",
    "world",
    "other",
]


def build_candidate_digest(
    articles: list[dict[str, Any]],
    max_candidates: int = 60,
    min_words: int = 60,
    ref_time: datetime | None = None,
    allow_opinion: bool = False,
    allow_stale_preview: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pre-filters raw articles and prepares a compact digest for the LLM curator.

    Returns:
        (eligible_articles, digest_items, opinion_articles, stale_articles)
    """
    eligible: list[dict[str, Any]] = []
    opinions: list[dict[str, Any]] = []
    stale_previews: list[dict[str, Any]] = []

    for art in articles:
        if is_advertisement(art):
            continue

        is_op, op_reason = detect_opinion(art)
        art["is_opinion"] = is_op
        art["opinion_reason"] = op_reason if is_op else None
        if is_op and not allow_opinion:
            opinions.append(art)
            continue

        if is_thin_story(art, min_words=min_words):
            continue

        is_stale, stale_reason = detect_stale_preview(art, ref_time=ref_time)
        art["is_stale_preview"] = is_stale
        art["stale_preview_reason"] = stale_reason if is_stale else None
        if is_stale and not allow_stale_preview:
            stale_previews.append(art)
            continue

        eligible.append(art)

    # Cap candidates prioritizing importance and trust if count exceeds limit
    if len(eligible) > max_candidates:
        capped_eligible = sorted(
            eligible,
            key=lambda a: (
                float(a.get("importance_hint", 0.0) or 0.0),
                float(a.get("source_trust", 0.70) or 0.70),
                parse_timestamp(a.get("published_at")),
            ),
            reverse=True,
        )[:max_candidates]
    else:
        capped_eligible = eligible

    digest: list[dict[str, Any]] = []
    for idx, art in enumerate(capped_eligible):
        summary = (art.get("text_summary") or art.get("full_text") or "")[:280].strip()
        digest.append({
            "id": idx,
            "source": art.get("source", "Unknown"),
            "title": art.get("title", "Untitled"),
            "published_at": art.get("published_at") or "Unknown",
            "summary": summary,
        })

    return capped_eligible, digest, opinions, stale_previews


def build_curator_prompt(digest: list[dict[str, Any]]) -> str:
    """Builds the editorial curation prompt."""
    digest_json = json.dumps(digest, separators=(",", ":"), ensure_ascii=False)
    cat_list = ", ".join(f'"{c}"' for c in ALLOWED_CATEGORIES)

    return (
        "You are the Senior Executive News Editor for Nepal News Bot, a daily 2-to-3 minute national broadcast "
        "television news program in Nepal.\n\n"
        "Below is a digest of candidate news stories scraped from Nepal's major outlets today in English and Nepali.\n"
        "Your task is to analyze these reports and produce an authoritative, broadcast-ready editorial curation JSON.\n\n"
        "EDITORIAL RESPONSIBILITIES:\n"
        "1. SEMANTIC EVENT CLUSTERING:\n"
        "   - Different outlets often report the same real-world event with different titles, angles, languages (English & Nepali), or update timestamps.\n"
        "   - Group all article IDs covering the exact same news event into ONE cluster. NEVER split updates or different outlets covering the same event into separate clusters.\n"
        "   - Standalone stories with only one outlet get their single article ID in their own cluster.\n\n"
        "2. MULTI-SOURCE INFORMATION SYNTHESIS DIRECTIVES:\n"
        "   - For clusters with 2 or more articles, write explicit 'synthesis_notes': identify what unique, complementary facts each outlet provides (e.g. 'Republica reports 14 fatalities; Kathmandu Post details the vehicle plunge and route; Nagarik reports APF rescue efforts').\n"
        "   - If different articles contain conflicting numbers (e.g. developing casualty or vote counts), specify 'conflicting_figures_resolution' directing the writer to strictly use the verified number from the latest update.\n\n"
        "3. NATIONAL NEWS RELEVANCE & EDITORIAL PRIORITY:\n"
        "   - Assign an 'editorial_priority' score from 1.0 to 10.0 reflecting its news value for a national Nepali evening broadcast:\n"
        "     * 8.0 - 10.0: High national impact (Supreme Court rulings, cabinet decisions, major fatal accidents/disasters, nationwide economic or transport disruptions).\n"
        "     * 5.0 - 7.9: Significant provincial politics, parliamentary committee updates, national sports, weather warnings, major cultural events.\n"
        "     * 1.0 - 4.9: Hyper-local municipal routine notices, minor local disputes, foreign wire stories (e.g. protests in Europe, foreign elections) UNLESS directly impacting Nepal or Nepali migrant workers.\n\n"
        "4. BROADCAST CATEGORIES & MULTI-HEADLINE COVERAGE:\n"
        "   The national broadcast covers 2 to 3 summarized headlines across each of the 6 sequential news domains:\n"
        "     1. Slot 1: National Politics ('national_politics') - 2-3 top national political stories\n"
        "     2. Slot 2: Society and Local News ('society_local') - 2-3 top civic, health, transport, disaster stories\n"
        "     3. Slot 3: Culture and Entertainment ('culture_entertainment') - 2-3 top cultural, festival, heritage, cinema stories\n"
        "     4. Slot 4: Business and Economics ('business_economics') - 2-3 top market, inflation, trade, finance stories\n"
        "     5. Slot 5: Science and Technology ('science_technology') - 2-3 top telecom, digital, AI, tech, science stories\n"
        "     6. Slot 6: Sports ('sports') - 2-3 top sports and tournament stories\n"
        f"   - Assign 'category' to one of the 6 canonical broadcast domains, or from: [{cat_list}].\n\n"
        "5. PRIMARY HEADLINE:\n"
        "   - Provide a clean, objective 'primary_title' in English (sentence case, max 10 words, no ending period).\n\n"
        "OUTPUT SCHEMA (Return a SINGLE valid JSON object, no markdown fences):\n"
        "{\n"
        '  "clusters": [\n'
        "    {\n"
        '      "article_ids": [0, 4],\n'
        '      "primary_title": "Fourteen killed in Makwanpur electric passenger van accident",\n'
        '      "category": "disaster",\n'
        '      "editorial_priority": 9.5,\n'
        '      "synthesis_notes": "Combine 14 fatalities from Article 0 with route details and plunge depth from Article 4.",\n'
        '      "conflicting_figures_resolution": "Earlier reports stated 6 dead; use the verified figure 14 from the latest update."\n'
        "    }\n"
        "  ],\n"
        '  "filtered_out_ids": [\n'
        '    {"id": 12, "reason": "Foreign wire story without domestic impact on Nepal"}\n'
        "  ]\n"
        "}\n\n"
        f"CANDIDATE DIGEST ({len(digest)} articles):\n"
        f"{digest_json}"
    )


def curate_and_cluster_with_llm(
    articles: list[dict[str, Any]],
    editor_cfg: dict[str, Any],
    prior_stories: list[dict[str, Any]] | None = None,
    ref_time: datetime | None = None,
    provider: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Executes LLM semantic curation and clustering.

    Returns (shortlist, opinion_clusters, stale_clusters) or None if LLM is unavailable or disabled.
    """
    # Guard: in automated test runs, skip live LLM calls unless explicitly enabled in editor_cfg
    curator_cfg = editor_cfg.get("llm_curator", {})
    if os.getenv("PYTEST_CURRENT_TEST") and not curator_cfg.get("enabled_in_tests", False):
        return None

    if not curator_cfg.get("enabled", True):
        return None

    chosen_provider, chosen_model = detect_configured_provider(provider)
    if not chosen_provider or not chosen_model:
        return None

    ref_year = ref_time.year if ref_time else datetime.now().year
    max_cand = int(curator_cfg.get("max_candidates", 40))
    min_words = int(editor_cfg.get("clustering", {}).get("min_full_text_words", 60))
    timeout = int(curator_cfg.get("timeout_sec", 60))
    top_limit = int(editor_cfg.get("clustering", {}).get("top_clusters_limit", 12))
    title_ratio = float(editor_cfg.get("clustering", {}).get("title_similarity_threshold", 0.55))
    token_overlap = float(editor_cfg.get("clustering", {}).get("token_overlap_threshold", 0.35))
    entity_aliases = editor_cfg.get("entity_aliases", DEFAULT_ENTITY_ALIASES)

    allow_op = bool(curator_cfg.get("allow_opinion", False) or editor_cfg.get("allow_opinion", False))
    allow_stale = bool(curator_cfg.get("allow_stale_preview", False) or editor_cfg.get("allow_stale_preview", False))

    eligible, digest, opinions, stale_previews = build_candidate_digest(
        articles,
        max_candidates=max_cand,
        min_words=min_words,
        ref_time=ref_time,
        allow_opinion=allow_op,
        allow_stale_preview=allow_stale,
    )
    if not eligible:
        return None

    logger.info(
        f"Running LLM Editorial Curator ({chosen_provider}:{chosen_model}) on {len(digest)} articles..."
    )

    prompt = build_curator_prompt(digest)
    messages = [{"role": "user", "content": prompt}]

    try:
        raw_reply = _call_llm(chosen_provider, chosen_model, messages, timeout=timeout)
        parsed = _extract_json_object(raw_reply)
    except Exception as e:
        logger.warning(f"LLM curator request failed ({e}); falling back to heuristic clustering.")
        return None

    clusters_data = parsed.get("clusters")
    if not isinstance(clusters_data, list) or not clusters_data:
        logger.warning("LLM curator returned empty or invalid clusters; falling back.")
        return None

    # Track which article indices have been assigned to avoid orphan high-value stories
    assigned_indices: set[int] = set()
    formatted_clusters: list[dict[str, Any]] = []

    for c_data in clusters_data:
        raw_ids = c_data.get("article_ids", [])
        valid_ids = [i for i in raw_ids if isinstance(i, int) and 0 <= i < len(eligible) and i not in assigned_indices]
        if not valid_ids:
            continue

        assigned_indices.update(valid_ids)
        c_members = [eligible[i] for i in valid_ids]

        sources = sorted(list({a["source"] for a in c_members}))
        urls = [a["url"] for a in c_members]
        newest_ts = max((parse_timestamp(a.get("published_at")) for a in c_members), default=0.0)
        max_source_trust = max((float(a.get("source_trust", 0.70) or 0.70) for a in c_members), default=0.70)
        max_importance = max((float(a.get("importance_hint", 0.0) or 0.0) for a in c_members), default=0.0)
        all_dates_missing = all(
            (a.get("date_confidence") == "missing" or not a.get("published_at"))
            for a in c_members
        )

        # Sort member articles reverse-chronologically (newest first)
        sorted_members = sorted(
            c_members,
            key=lambda a: (
                parse_timestamp(a.get("published_at")),
                float(a.get("source_trust", 0.70) or 0.70),
                float(a.get("importance_hint", 0.0) or 0.0),
            ),
            reverse=True,
        )

        # Detect genuine conflicting numeric figures for supersession
        article_num_sets = [
            frozenset(
                extract_spine_numbers(
                    f"{a.get('title', '')} {a.get('full_text') or a.get('text_summary') or ''}",
                    exclude_years={ref_year},
                )
            )
            for a in sorted_members
        ]
        has_differing_numbers = (
            len(sorted_members) >= 2
            and sum(1 for s in article_num_sets if len(s) > 0) >= 2
            and len(set(s for s in article_num_sets if len(s) > 0)) > 1
        )

        superseded_flags: list[bool] = []
        for a in sorted_members:
            a_ts = parse_timestamp(a.get("published_at"))
            if has_differing_numbers and newest_ts > 0.0 and a_ts < newest_ts:
                superseded_flags.append(True)
            else:
                superseded_flags.append(False)
        has_superseded = any(superseded_flags)

        category = str(c_data.get("category", "")).lower().strip()
        if category not in ALLOWED_CATEGORIES:
            category = "other"

        primary_title = str(c_data.get("primary_title") or sorted_members[0].get("title", "News Item")).strip()
        editorial_priority = float(c_data.get("editorial_priority", 5.0) or 5.0)
        synthesis_notes = c_data.get("synthesis_notes")
        conflict_res = c_data.get("conflicting_figures_resolution")

        formatted_clusters.append({
            "cluster_id": "temp",
            "primary_title": primary_title,
            "category_guess": category,
            "editorial_priority": editorial_priority,
            "synthesis_notes": synthesis_notes,
            "conflicting_figures_resolution": conflict_res,
            "llm_curated": True,
            "is_opinion": False,
            "opinion_reason": None,
            "is_stale_preview": False,
            "stale_preview_reason": None,
            "has_superseded_updates": has_superseded,
            "covered_recently": False,
            "repeat_reason": None,
            "sources": sources,
            "source_count": len(sources),
            "max_source_trust": round(max_source_trust, 2),
            "max_importance_hint": round(max_importance, 3),
            "all_dates_missing": all_dates_missing,
            "urls": urls,
            "article_count": len(c_members),
            "newest_timestamp": newest_ts,
            "articles": [
                {
                    "source": a["source"],
                    "title": a["title"],
                    "url": a["url"],
                    "published_at": a.get("published_at"),
                    "text_summary": a.get("text_summary", ""),
                    "full_text": a.get("full_text", a.get("text_summary", "")),
                    "language": a.get("language", "en"),
                    "category_hint": a.get("category_hint"),
                    "importance_hint": a.get("importance_hint"),
                    "feed_position": a.get("feed_position"),
                    "source_trust": a.get("source_trust"),
                    "date_confidence": a.get("date_confidence"),
                    "raw_tags": a.get("raw_tags", []),
                    "is_opinion": False,
                    "opinion_reason": None,
                    "is_stale_preview": False,
                    "stale_preview_reason": None,
                    "superseded": sup_flag,
                }
                for a, sup_flag in zip(sorted_members, superseded_flags)
            ],
        })

    # Apply 48-hour cross-day deduplication memory
    apply_cross_day_deduplication(
        formatted_clusters,
        prior_stories,
        title_ratio=title_ratio,
        token_overlap=token_overlap,
        entity_aliases=entity_aliases,
        ref_year=ref_year,
    )

    # Rank clusters: balance editorial priority, source count, topic diversity, and cross-day penalty
    def _curator_base_score(cand: dict[str, Any]) -> float:
        repeat_adj = -80.0 if cand.get("covered_recently") else 0.0
        missing_adj = -0.5 if cand.get("all_dates_missing") else 0.0
        return (
            (cand.get("editorial_priority", 5.0) * 20.0)
            + (min(cand.get("source_count", 1), 4) * 5.0)
            + (cand.get("max_source_trust", 0.70) * 3.0)
            + repeat_adj
            + missing_adj
        )

    ranked_clusters = rank_clusters_with_diversity(
        formatted_clusters,
        base_score_fn=_curator_base_score,
        category_penalty=12.0,
    )

    # Stratified Multi-Cluster Allocation: 2 to 3 clusters per broadcast category in exact sequence
    for c in ranked_clusters:
        c["broadcast_slot"] = map_to_broadcast_slot(c.get("category_guess"))

    clusters_per_slot = int(editor_cfg.get("clustering", {}).get("clusters_per_category", 4))
    slot_clusters: list[dict[str, Any]] = []
    assigned_ids: set[int] = set()

    for slot_id, slot_cat, slot_title in BROADCAST_SLOTS:
        matches = [
            c for c in ranked_clusters
            if id(c) not in assigned_ids and c.get("broadcast_slot") == slot_cat
        ][:clusters_per_slot]

        if matches:
            for item_idx, cand in enumerate(matches, 1):
                cand["broadcast_slot_id"] = slot_id
                cand["broadcast_slot_title"] = slot_title
                cand["slot_item_index"] = item_idx
                slot_clusters.append(cand)
                assigned_ids.add(id(cand))
        else:
            fallback_cand = next(
                (c for c in ranked_clusters if id(c) not in assigned_ids),
                None,
            )
            if fallback_cand:
                fallback_cand["broadcast_slot_id"] = slot_id
                fallback_cand["broadcast_slot_title"] = f"{slot_title} (Fallback)"
                fallback_cand["fallback_slot"] = True
                fallback_cand["slot_item_index"] = 1
                slot_clusters.append(fallback_cand)
                assigned_ids.add(id(fallback_cand))

    for cand in ranked_clusters:
        if id(cand) not in assigned_ids and len(slot_clusters) < top_limit:
            slot_clusters.append(cand)
            assigned_ids.add(id(cand))

    shortlist = slot_clusters if slot_clusters else ranked_clusters[:top_limit]
    for idx, c in enumerate(shortlist, 1):
        c["cluster_id"] = f"cluster_{idx:02d}"

    # Format opinion clusters
    opinion_clusters: list[dict[str, Any]] = []
    for idx, op in enumerate(opinions, 1):
        opinion_clusters.append({
            "cluster_id": f"opinion_{idx:02d}",
            "primary_title": op.get("title", "Opinion Piece"),
            "is_opinion": True,
            "opinion_reason": op.get("opinion_reason"),
            "is_stale_preview": False,
            "stale_preview_reason": None,
            "sources": [op["source"]],
            "source_count": 1,
            "urls": [op["url"]],
            "article_count": 1,
            "newest_timestamp": parse_timestamp(op.get("published_at")),
            "category_guess": "opinion",
            "articles": [op],
        })

    # Format stale preview clusters
    stale_clusters: list[dict[str, Any]] = []
    for idx, st in enumerate(stale_previews, 1):
        stale_clusters.append({
            "cluster_id": f"stale_{idx:02d}",
            "primary_title": st.get("title", "Stale Preview"),
            "is_opinion": False,
            "opinion_reason": None,
            "is_stale_preview": True,
            "stale_preview_reason": st.get("stale_preview_reason"),
            "sources": [st["source"]],
            "source_count": 1,
            "urls": [st["url"]],
            "article_count": 1,
            "newest_timestamp": parse_timestamp(st.get("published_at")),
            "category_guess": "stale",
            "articles": [st],
        })

    logger.info(
        f"LLM curation successfully formed {len(shortlist)} top news clusters "
        f"({len(opinion_clusters)} opinions, {len(stale_clusters)} stale)."
    )
    return shortlist, opinion_clusters, stale_clusters

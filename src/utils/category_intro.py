"""Category-aware broadcast intro resolution and template management.

Provides evergreen, date-agnostic spoken narrations for category-specific
broadcast intros (e.g. sports news, headlines & breaking news, politics, etc.)
allowing one-time TTS voice generation and persistent cross-run caching.
"""
from __future__ import annotations

import re
from typing import Any

# Canonical categories and their default metadata & date-agnostic evergreen narrations.
# All narrations strictly conform to broadcast TTS rules:
# - ZERO digits (0-9)
# - ZERO banned symbols (%, $, &, /, (), [])
# - 10 to 15 words (punchy, high-retention single-sentence hooks)
DEFAULT_CATEGORY_INTROS: dict[str, dict[str, Any]] = {
    "headlines": {
        "label": "Headlines & Breaking News",
        "headline_template": "Nepal Headlines and Breaking News for {spoken_date}",
        "headline_fallback": "Nepal Headlines and Breaking News",
        "narration": "Top headlines, breaking stories, and major developments from across the nation.",
        "visual_keywords": ["Kathmandu valley", "news studio", "breaking news desk"],
    },
    "sports": {
        "label": "Sports News",
        "headline_template": "Nepal Sports Roundup for {spoken_date}",
        "headline_fallback": "Nepal Sports Roundup",
        "narration": "Top sports news, major highlights, and match updates from Nepal and around the world.",
        "visual_keywords": ["sports stadium", "Nepal cricket", "athletes competing"],
    },
    "politics": {
        "label": "National Politics",
        "headline_template": "Nepal Political Briefing for {spoken_date}",
        "headline_fallback": "Nepal Political Briefing",
        "narration": "Major political shifts, parliamentary decisions, and government updates from across the nation.",
        "visual_keywords": ["parliament building", "news studio", "Kathmandu city"],
    },
    "business_economics": {
        "label": "Business & Economics",
        "headline_template": "Nepal Business Report for {spoken_date}",
        "headline_fallback": "Nepal Business Report",
        "narration": "Market trends, economic shifts, and major financial headlines from Nepal and global markets.",
        "visual_keywords": ["stock exchange", "news studio", "banking district"],
    },
    "health": {
        "label": "Health & Medicine",
        "headline_template": "Nepal Health Briefing for {spoken_date}",
        "headline_fallback": "Nepal Health Briefing",
        "narration": "Critical healthcare updates, medical breakthroughs, and wellness news from across the nation.",
        "visual_keywords": ["hospital building", "news studio", "medical clinic"],
    },
    "culture_entertainment": {
        "label": "Culture & Entertainment",
        "headline_template": "Nepal Culture and Entertainment for {spoken_date}",
        "headline_fallback": "Nepal Culture and Entertainment",
        "narration": "Vibrant arts, heritage festivals, and top entertainment headlines from across Nepal and beyond.",
        "visual_keywords": ["temple heritage", "cultural festival", "Kathmandu arts"],
    },
    "science_technology": {
        "label": "Science & Technology",
        "headline_template": "Nepal Technology Briefing for {spoken_date}",
        "headline_fallback": "Nepal Technology Briefing",
        "narration": "Cutting edge innovations, digital breakthroughs, and major technology headlines shaping tomorrow.",
        "visual_keywords": ["digital technology", "telecom tower", "computer laboratory"],
    },
    "society_local": {
        "label": "Society & Local News",
        "headline_template": "Nepal Local News Briefing for {spoken_date}",
        "headline_fallback": "Nepal Local News Briefing",
        "narration": "Grassroots voices, regional developments, and vital community stories from across Nepal.",
        "visual_keywords": ["community gathering", "Kathmandu streets", "local village"],
    },
    "international": {
        "label": "International News",
        "headline_template": "Nepal World Report for {spoken_date}",
        "headline_fallback": "Nepal World Report",
        "narration": "Global diplomacy, breaking world headlines, and major international developments from around the world.",
        "visual_keywords": ["world map", "diplomatic summit", "international news"],
    },
    "default": {
        "label": "Daily Briefing",
        "headline_template": "Nepal Daily Briefing for {spoken_date}",
        "headline_fallback": "Nepal Daily Briefing",
        "narration": "The top stories and major developments shaping the nation right now.",
        "visual_keywords": ["Kathmandu valley", "news studio", "map of Nepal"],
    },
}

# Alias mapping to canonical category keys
CATEGORY_SYNONYM_MAP: dict[str, str] = {
    "sports": "sports",
    "sport": "sports",
    "cricket": "sports",
    "football": "sports",
    "athletics": "sports",
    "headlines": "headlines",
    "breaking": "headlines",
    "breaking_news": "headlines",
    "top_stories": "headlines",
    "cover": "headlines",
    "politics": "politics",
    "national_politics": "politics",
    "political": "politics",
    "government": "politics",
    "parliament": "politics",
    "economy": "business_economics",
    "business": "business_economics",
    "business_economics": "business_economics",
    "market": "business_economics",
    "finance": "business_economics",
    "health": "health",
    "medical": "health",
    "medicine": "health",
    "culture": "culture_entertainment",
    "entertainment": "culture_entertainment",
    "culture_entertainment": "culture_entertainment",
    "tourism": "culture_entertainment",
    "technology": "science_technology",
    "science": "science_technology",
    "science_technology": "science_technology",
    "tech": "science_technology",
    "society": "society_local",
    "local": "society_local",
    "society_local": "society_local",
    "transport": "society_local",
    "disaster": "society_local",
    "crime": "society_local",
    "weather": "society_local",
    "climate": "society_local",
    "environment": "society_local",
    "international": "international",
    "world": "international",
    "foreign": "international",
    "global": "international",
}


def normalize_selected_categories(categories: list[str] | str | None) -> list[str]:
    """Normalizes raw category inputs into deduplicated canonical category keys."""
    if not categories:
        return []
    if isinstance(categories, str):
        tokens = [t.strip().lower() for t in re.split(r"[,;\s]+", categories) if t.strip()]
    else:
        tokens = [str(t).strip().lower() for t in categories if str(t).strip()]

    canonical: list[str] = []
    for tok in tokens:
        if tok in ("all", "a"):
            return ["all"]
        canon = CATEGORY_SYNONYM_MAP.get(tok, tok)
        if canon not in canonical:
            canonical.append(canon)
    return canonical


def get_canonical_category_keys() -> list[str]:
    """Returns the list of all supported canonical category keys (excluding default)."""
    return [k for k in DEFAULT_CATEGORY_INTROS if k != "default"]


def resolve_category_intro(
    categories: list[str] | str | None = None,
    spoken_date: str = "",
    custom_templates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolves category metadata, headline, evergreen narration, and keywords for intro scene.

    Args:
        categories: User-selected category list or comma-separated string.
        spoken_date: Human-readable spoken date string for the headline (e.g. 'Saturday, November 21, 2026').
        custom_templates: Optional overrides from config/editor.yaml intro_templates.

    Returns:
        Dictionary containing:
        - category_key: canonical category slug or 'all'/'default'
        - category_label: human readable category name
        - headline: category-adapted headline text
        - narration: evergreen, date-agnostic spoken broadcast narration
        - visual_keywords: list of 3-5 visual stock keywords
        - prompt_instruction: explicit directive for LLM writer and editor brief
    """
    norm_cats = normalize_selected_categories(categories)
    merged_templates: dict[str, dict[str, Any]] = {**DEFAULT_CATEGORY_INTROS}
    if custom_templates and isinstance(custom_templates, dict):
        for k, v in custom_templates.items():
            if isinstance(v, dict):
                merged_templates.setdefault(k, {}).update(v)

    # Determine canonical category
    cat_key = "default"
    if norm_cats and "all" not in norm_cats:
        if len(norm_cats) == 1:
            c = norm_cats[0]
            cat_key = c if c in merged_templates else "default"
        else:
            # Multi-category specific selection
            cat_key = "default"

    entry = merged_templates.get(cat_key, merged_templates["default"])
    label = entry.get("label", "Daily Briefing")

    # Format headline
    if spoken_date:
        hl_template = entry.get("headline_template", "Nepal Daily Briefing for {spoken_date}")
        headline = hl_template.format(spoken_date=spoken_date)
    else:
        headline = entry.get("headline_fallback", "Nepal Daily Briefing")

    narration = entry.get(
        "narration",
        "The top stories and major developments shaping the nation right now.",
    )
    visual_keywords = list(entry.get("visual_keywords", ["Kathmandu valley", "news studio", "map of Nepal"]))

    prompt_instruction = (
        f"The selected broadcast category is '{label}'. The intro scene (id='intro') MUST explicitly announce "
        f"this broadcast category using the date-agnostic wording: \"{narration}\"."
    )

    return {
        "category_key": cat_key,
        "category_label": label,
        "headline": headline,
        "narration": narration,
        "visual_keywords": visual_keywords,
        "prompt_instruction": prompt_instruction,
    }


def is_canonical_intro_narration(text: str, category_key: str | None = None) -> bool:
    """Returns True if the text matches the canonical evergreen narration for the given category (or any canonical category if category_key is None)."""
    norm_actual = re.sub(r"\s+", " ", text).strip().lower()
    if category_key:
        entry = DEFAULT_CATEGORY_INTROS.get(category_key)
        if not entry:
            return False
        norm_expected = re.sub(r"\s+", " ", entry["narration"]).strip().lower()
        return norm_actual == norm_expected

    for cat, entry in DEFAULT_CATEGORY_INTROS.items():
        norm_expected = re.sub(r"\s+", " ", entry["narration"]).strip().lower()
        if norm_actual == norm_expected:
            return True
    return False


def identify_category_from_intro(
    intro: dict[str, Any] | str | None = None,
    manifest: dict[str, Any] | None = None,
) -> str | None:
    """Identifies the canonical category key for an intro scene.

    Returns the canonical category key (e.g. 'headlines', 'sports') ONLY if the intro
    strictly matches the canonical evergreen narration or explicit canonical category.
    Returns None if the intro is custom, date-specific, or arbitrary test text.
    """
    text = ""
    explicit_cat = None
    if isinstance(intro, dict):
        text = str(intro.get("narration") or intro.get("headline") or "")
        explicit_cat = intro.get("category")
    elif isinstance(intro, str):
        text = intro

    # If manifest specifies a canonical category and the narration is canonical
    if manifest and isinstance(manifest, dict):
        sel_cats = manifest.get("selected_categories")
        if sel_cats:
            norm = normalize_selected_categories(sel_cats)
            if len(norm) == 1 and norm[0] in DEFAULT_CATEGORY_INTROS:
                if not text or is_canonical_intro_narration(text, norm[0]):
                    return norm[0]

    if explicit_cat and explicit_cat in DEFAULT_CATEGORY_INTROS:
        if not text or is_canonical_intro_narration(text, explicit_cat):
            return explicit_cat

    # Check exact match against all canonical evergreen narrations
    norm_actual = re.sub(r"\s+", " ", text).strip().lower()
    for cat_k, cat_v in DEFAULT_CATEGORY_INTROS.items():
        c_narr = re.sub(r"\s+", " ", cat_v.get("narration", "")).strip().lower()
        if norm_actual == c_narr:
            return cat_k

    return None


def get_intro_cache_filenames(category_key: str) -> tuple[str, str]:
    """Returns semantic filenames (wav_filename, meta_filename) for persistent category intro cache.

    Example:
        get_intro_cache_filenames('sports') -> ('intro_sports.wav', 'intro_sports.meta.json')
        get_intro_cache_filenames('headlines') -> ('intro_headlines.wav', 'intro_headlines.meta.json')
    """
    clean_cat = normalize_selected_categories([category_key])
    cat = clean_cat[0] if clean_cat and clean_cat[0] in DEFAULT_CATEGORY_INTROS else category_key
    if cat not in DEFAULT_CATEGORY_INTROS:
        cat = "default"
    return f"intro_{cat}.wav", f"intro_{cat}.meta.json"

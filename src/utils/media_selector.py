"""Media selection logic: license filtering, query ladders, duplicate prevention, and source prioritization."""
import logging
from typing import Any, Generator
from urllib.parse import urlparse
from src.utils.media_models import Candidate, PlannedShot
from src.utils.media_sources import build_media_sources
from src.utils.media_cache import MediaCacheManager
import imagehash

logger = logging.getLogger("media.selector")

NEPAL_MARKERS = {
    "nepal",
    "nepali",
    "nepalese",
    "kathmandu",
    "pokhara",
    "bagmati",
    "annapurna",
    "phewa",
    "himalaya",
    "himalayas",
    "himalayan",
    "everest",
    "sagarmatha",
    "bhaktapur",
    "lalitpur",
    "patan",
    "birgunj",
    "sindhupalchok",
    "mustang",
    "chitwan",
    "dharahara",
    "terai",
    "lumbini",
}

MAP_DIAGRAM_MARKERS = {"map", "diagram", "chart", "flag", "logo"}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".ogv", ".mov"}
DISALLOWED_EXTS = {".svg", ".pdf", ".tif", ".tiff", ".gif"}


def extract_clean_extension(url: str) -> str:
    """Extracts lowercase file extension from URL ignoring query parameters."""
    if not url:
        return ""
    clean = url.split("?")[0].split("#")[0].strip()
    idx = clean.rfind(".")
    if idx != -1:
        return clean[idx:].lower()
    return ""


def extract_noun_keyword(keyword: str) -> str:
    """Extracts base noun from multi-word keyword phrases (e.g. 'Phewa lake' -> 'lake')."""
    words = keyword.strip().split()
    if len(words) <= 1:
        return keyword.strip()
    return words[-1].strip()


STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "is", "are", "was", "were", "this", "that", "it", "as", "be",
    "its", "into", "over", "after", "about", "above", "across", "under", "again",
}

DEFAULT_FOREIGN_PLACES = [
    "india", "indian", "delhi", "mumbai", "himachal", "kinnaur", "sikkim",
    "tibet", "china", "pakistan", "bangladesh", "sri lanka", "uttarakhand",
    "ladakh", "kashmir", "kolkata"
]

KNOWN_PLACES = [
    "nepal", "nepali", "nepalese", "kathmandu", "pokhara", "lalitpur", "bhaktapur",
    "patan", "birgunj", "antalya", "turkey", "turkiye", "lumbini", "kirtipur",
    "kailali", "geta", "dhangadhi", "dharahara", "bagmati", "everest", "sagarmatha",
    "himalaya", "himalayas", "himalayan", "basantapur", "lukla", "chitwan", "mustang",
    "annapurna", "phewa"
]


def light_stem(w: str) -> str:
    """Performs light stemming on query tokens."""
    w = w.lower().strip()
    if len(w) > 4:
        if w.endswith("ies") and len(w) > 5:
            return w[:-3] + "y"
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("ed") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
    return w


def score_candidate_relevance(
    cand: Candidate,
    query: str,
    visual_keywords: list[str],
    scene: dict[str, Any],
    foreign_places: list[str] | None = None,
    is_sensitive: bool = False,
) -> tuple[float, list[str]]:
    """Calculates relevance score between 0.0 and 1.0 and records matched tokens.

    1. Share of the shot's query tokens (stopwords removed, light stemming) found in
       the candidate's title, tags and description.
    2. Bonus for an exact phrase (+0.2).
    3. Bonus when the candidate names a place that the scene names (+0.2).
    4. Foreign-place penalty: if candidate mentions a foreign place that the story text
       does not mention, subtract 0.5.
    5. Sensitive-scene person-tag rule: reject/0.0 if people visible or person tags.
    """
    import re

    if foreign_places is None:
        foreign_places = DEFAULT_FOREIGN_PLACES

    # Sensitive scene person-tag check
    if is_sensitive:
        person_markers = ["person", "people", "man", "woman", "child", "crowd", "girl", "boy"]
        cand_meta = f"{cand.title} {' '.join(cand.tags)}".lower()
        if cand.people_visible is True or any(pm in cand.tags for pm in person_markers) or any(pm in cand_meta for pm in person_markers):
            return 0.0, []

    # 1. Query tokens
    query_text = f"{query} {' '.join(visual_keywords)}"
    raw_query_words = [re.sub(r"[^\w]", "", w.lower()) for w in query_text.split()]
    query_words = [w for w in raw_query_words if w and w not in STOPWORDS and len(w) > 2]

    # Map stem -> raw word (for matched_tokens recording)
    q_stem_map: dict[str, str] = {}
    for w in query_words:
        st = light_stem(w)
        if st not in q_stem_map:
            q_stem_map[st] = w

    # Candidate text (title, tags, description)
    cand_desc = cand.extra.get("description", "") if isinstance(cand.extra, dict) else ""
    cand_text = f"{cand.title} {' '.join(cand.tags)} {cand_desc}".lower()
    raw_cand_words = [re.sub(r"[^\w]", "", w.lower()) for w in cand_text.split()]
    cand_stems = {light_stem(w) for w in raw_cand_words if w and w not in STOPWORDS}

    # Token match share
    matched_stems = set(q_stem_map.keys()) & cand_stems
    matched_tokens = sorted(list({q_stem_map[st] for st in matched_stems}))

    if q_stem_map:
        token_share = len(matched_stems) / len(q_stem_map)
    else:
        token_share = 0.5

    score = token_share

    # 2. Exact phrase bonus (+0.2)
    has_exact_phrase = False
    clean_q = query.strip().lower()
    if len(clean_q.split()) >= 2 and clean_q in cand_text:
        has_exact_phrase = True
    else:
        for kw in visual_keywords:
            clean_kw = kw.strip().lower()
            if len(clean_kw.split()) >= 2 and clean_kw in cand_text:
                has_exact_phrase = True
                break

    if has_exact_phrase:
        score += 0.2

    # 3. Place bonus (+0.2)
    story_text = f"{scene.get('headline', '')} {scene.get('narration', '')}".lower()
    scene_places = [p for p in KNOWN_PLACES if p in story_text]
    for fp in foreign_places:
        if fp in story_text and fp not in scene_places:
            scene_places.append(fp)

    has_place_match = False
    for p in scene_places:
        if p in cand_text:
            has_place_match = True
            if p not in matched_tokens:
                matched_tokens.append(p)

    if has_place_match:
        score += 0.2

    # 4. Foreign-place penalty (-0.5)
    cand_has_foreign_place = False
    for fp in foreign_places:
        if fp in cand_text:
            if fp not in story_text:
                cand_has_foreign_place = True
                break

    if cand_has_foreign_place:
        score -= 0.5

    # 5. Local library bonus: user-provided or local assets prioritized
    if cand.source in ("local_library", "user-provided"):
        score += 0.3

    final_score = max(0.0, min(1.0, round(score, 3)))
    return final_score, matched_tokens


class MediaSelector:
    """Coordinates search across sources according to strict editorial and licensing rules."""

    def __init__(
        self,
        config: dict[str, Any],
        cache_mgr: MediaCacheManager,
        offline: bool = False,
    ) -> None:
        self.config = config
        self.cache_mgr = cache_mgr
        self.offline = offline
        self.sources = build_media_sources(config, cache_mgr)

        # Source usage metrics
        self.metrics: dict[str, dict[str, int]] = {
            name: {"queried": 0, "candidates": 0, "used": 0, "skipped": 0}
            for name in self.sources
        }

        # Deduplication and variety tracking
        self.used_candidate_ids: set[str] = set()
        self.bad_candidate_ids: set[str] = set()
        self.used_sha256s: set[str] = set()
        self.used_phashes: list[imagehash.ImageHash] = []
        self.rejected_near_duplicates_count: int = 0

        # Query repetition tracking
        self.query_usage_counts: dict[str, int] = {}
        self.query_max_repeat: int = int(config.get("settings", {}).get("query_max_repeat", 2))

        # Map / diagram cap
        self.map_diagram_count: int = 0
        self.max_maps_per_run: int = int(config.get("settings", {}).get("max_maps_per_run", 1))

        # License policy
        self.allowed_licenses = set(lic.lower() for lic in config.get("allowed_licenses", []))
        self.allow_cc_by_sa = config.get("settings", {}).get("allow_cc_by_sa", False)

        # Blocklisted domains
        self.blocklisted_domains = set(d.lower() for d in config.get("blocklisted_domains", []))

        # Sensitive keywords
        self.sensitive_keywords = set(k.lower() for k in config.get("sensitive_keywords", []))
        self.person_tags = set(p.lower() for p in config.get("sensitive_scene_person_tags", []))
        self.watermark_markers = set(w.lower() for w in config.get("watermark_markers", []))

        # Relevance scoring configuration
        self.foreign_places = [fp.lower() for fp in config.get("foreign_places", DEFAULT_FOREIGN_PLACES)]
        self.min_relevance = float(config.get("settings", {}).get("min_relevance", 0.5))

    def is_candidate_nepal_specific(self, cand: Candidate) -> bool:
        """Checks if a candidate is explicitly related to Nepal in title or tags."""
        text = f"{cand.title} {' '.join(cand.tags)}".lower()
        return any(marker in text for marker in NEPAL_MARKERS)

    def is_scene_sensitive(self, scene: dict[str, Any]) -> bool:
        """Determines if a scene covers sensitive news requiring abstract/non-human imagery."""
        if scene.get("sensitive") is True:
            return True
        text = f"{scene.get('headline', '')} {scene.get('narration', '')}".lower()
        return any(kw in text for kw in self.sensitive_keywords)

    def is_domain_blocklisted(self, url: str | None) -> bool:
        """Checks if URL belongs to a blocklisted news-outlet domain."""
        if not url:
            return False
        try:
            domain = urlparse(url).netloc.lower()
            return any(blocked in domain for blocked in self.blocklisted_domains)
        except Exception:
            return False

    def is_license_allowed(self, license_name: str, source: str = "") -> bool:
        """Enforces allowed licenses, rejecting CC BY-SA (unless permitted), NC, ND, and unknown."""
        lic_clean = license_name.strip().lower()
        if not lic_clean:
            return False

        if "sa" in lic_clean or "sharealike" in lic_clean:
            if not self.allow_cc_by_sa:
                return False

        if "nc" in lic_clean or "noncommercial" in lic_clean or "nd" in lic_clean or "noderivs" in lic_clean:
            return False

        if "all rights reserved" in lic_clean or "unknown" in lic_clean:
            return False

        # Strings like "own work" taken from a Wikimedia Credit field must never populate the license
        if lic_clean in ("own work", "own work by original uploader", "self-published"):
            if source and source not in ("user-provided", "local_library"):
                return False

        # Match against configured allowed licenses
        return any(allowed in lic_clean or lic_clean in allowed for allowed in self.allowed_licenses)

    def is_candidate_acceptable(
        self,
        cand: Candidate,
        is_sensitive: bool,
        target_duration: float,
        prefer_kind: str = "any",
    ) -> tuple[bool, str | None]:
        """Validates all ground rules on a candidate."""
        # 1. Deduplication by ID & bad list
        if cand.id in self.used_candidate_ids:
            return False, "duplicate candidate ID"
        if cand.id in self.bad_candidate_ids:
            return False, "candidate marked bad from previous attempt"

        # 2. File type check (reject svg, pdf, tif, gif)
        ext = extract_clean_extension(cand.download_url)
        if not ext:
            ext = extract_clean_extension(cand.page_url)
        if ext in DISALLOWED_EXTS:
            return False, f"unsupported disallowed format ({ext})"
        if ext and ext not in ALLOWED_EXTS:
            return False, f"unsupported format ({ext})"

        # 3. Blocklisted domains
        if self.is_domain_blocklisted(cand.page_url) or self.is_domain_blocklisted(cand.download_url):
            return False, "blocklisted news domain"

        # 4. License verification
        if not self.is_license_allowed(cand.license_id, source=cand.source):
            return False, f"license '{cand.license_id}' not allowed"

        # 5. Map/diagram still cap
        if cand.kind == "image":
            meta_str = f"{cand.title} {' '.join(cand.tags)}".lower()
            if any(m in meta_str for m in MAP_DIAGRAM_MARKERS):
                if self.map_diagram_count >= self.max_maps_per_run:
                    return False, f"max {self.max_maps_per_run} map/diagram still reached for run"

        # 6. Watermark / logo check
        meta_str = f"{cand.title} {' '.join(cand.tags)}".lower()
        if any(wm in meta_str for wm in self.watermark_markers):
            return False, "watermark or logo detected in tags/title"

        # 7. Sensitive scene person check
        if is_sensitive:
            if cand.people_visible is True:
                return False, "people visible in sensitive scene"
            if any(pt in cand.tags for pt in self.person_tags):
                return False, "person tag detected in sensitive scene"
            if any(pt in cand.title.lower() for pt in self.person_tags):
                return False, "person mentioned in title for sensitive scene"

        # 8. Kind matching
        if prefer_kind != "any" and cand.kind != prefer_kind:
            return False, f"kind mismatch (expected {prefer_kind}, got {cand.kind})"

        # 9. Video duration check
        if cand.kind == "video" and cand.duration is not None:
            if cand.duration < (target_duration - 0.5):
                return False, f"video duration {cand.duration}s < required {target_duration}s"

        return True, None

    def build_query_ladder(self, scene: dict[str, Any], is_sensitive: bool) -> list[str]:
        """Constructs ordered query ladder: visual keywords -> simplified nouns -> category default."""
        if is_sensitive:
            return ["empty road landscape", "monsoon rain window", "city night skyline", "clouds mountain timelapse"]

        queries: list[str] = []
        visual_keywords = scene.get("visual_keywords", [])

        # 1. Visual keywords in order
        for kw in visual_keywords:
            clean = kw.strip()
            if clean and clean not in queries:
                queries.append(clean)

        # 2. Simplified noun terms
        for kw in visual_keywords:
            noun = extract_noun_keyword(kw)
            if noun and noun not in queries and len(noun) > 2:
                queries.append(noun)

        # 3. Category default from config
        category = self.detect_category(scene)
        cat_default = self.config.get("category_defaults", {}).get(category)
        if cat_default and cat_default not in queries:
            queries.append(cat_default)

        # 4. General fallback query
        gen_default = self.config.get("category_defaults", {}).get("general", "Kathmandu city skyline")
        if gen_default not in queries:
            queries.append(gen_default)

        return queries

    def detect_category(self, scene: dict[str, Any]) -> str:
        """Infers the scene category from scene metadata or headline/narration."""
        cat = scene.get("category")
        if cat and isinstance(cat, str) and cat.strip():
            return cat.strip().lower()
        text = f"{scene.get('headline', '')} {scene.get('narration', '')}".lower()
        if any(w in text for w in ["politics", "parliament", "minister", "government", "election"]):
            return "politics"
        if any(w in text for w in ["economy", "inflation", "market", "bank", "rupee", "tax"]):
            return "economy"
        if any(w in text for w in ["rain", "monsoon", "flood", "weather", "temperature", "cloud"]):
            return "weather"
        if any(w in text for w in ["climate", "glacier", "warming", "environment", "river cleanup", "waste"]):
            return "climate"
        if any(w in text for w in ["hospital", "health", "doctor", "medicine", "virus", "disease"]):
            return "health"
        if any(w in text for w in ["highway", "road", "transit", "traffic", "vehicle", "bus", "transport"]):
            return "transport"
        if any(w in text for w in ["hydroelectric", "hydropower", "electricity", "energy", "grid", "megawatt"]):
            return "energy"
        if any(w in text for w in ["tourism", "tourist", "trekking", "hotel", "annapurna", "lake", "mountain"]):
            return "tourism"
        if any(w in text for w in ["port", "customs", "cargo", "trade", "commercial"]):
            return "commerce"
        return "general"

    def iter_candidates_for_shot(
        self,
        scene: dict[str, Any],
        shot: PlannedShot,
        stills_in_scene: int,
        max_stills: int = 2,
    ) -> Generator[tuple[Candidate, str, str], None, None]:
        """Yields valid candidates across prioritized sources and query ladders.

        Allows consumer to try candidate 1, and on failure continue immediately to candidate 2.
        """
        is_sensitive = self.is_scene_sensitive(scene)
        category = self.detect_category(scene)
        query_ladder = self.build_query_ladder(scene, is_sensitive)
        allow_stills = stills_in_scene < max_stills

        source_stages = [
            ("local_library", "any"),
            ("wikimedia_commons", "video"),
            ("wikimedia_commons", "image" if allow_stills else "video"),
            ("pixabay", "video"),
            ("coverr", "video"),
            ("openverse", "image" if allow_stills else "video"),
            ("pixabay", "image" if allow_stills else "video"),
            ("unsplash", "image" if allow_stills else "video"),
            ("nasa", "any"),
            ("internet_archive", "video"),
        ]

        for source_name, prefer_kind in source_stages:
            src = self.sources.get(source_name)
            if not src:
                continue

            available, reason = src.is_available
            if not available:
                self.metrics[source_name]["skipped"] += 1
                continue

            for query in query_ladder:
                # Query cap: limit shots from same query to query_max_repeat per run
                if self.query_usage_counts.get(query, 0) >= self.query_max_repeat:
                    continue

                self.metrics[source_name]["queried"] += 1
                try:
                    candidates = src.search(
                        query=query,
                        kind=prefer_kind,
                        orientation="landscape",
                        offline=self.offline,
                        scene_category=category,
                        is_sensitive=is_sensitive,
                    )
                except Exception as e:
                    logger.warning(f"Error querying {source_name} with '{query}': {e}")
                    continue

                self.metrics[source_name]["candidates"] += len(candidates)

                for cand in candidates:
                    # Enforce that Wikimedia matches must be Nepal-specific
                    if source_name == "wikimedia_commons" and not self.is_candidate_nepal_specific(cand):
                        continue

                    ok, rej_reason = self.is_candidate_acceptable(
                        cand,
                        is_sensitive=is_sensitive,
                        target_duration=shot.duration,
                        prefer_kind=prefer_kind,
                    )
                    if ok:
                        cand.score, cand.matched_tokens = score_candidate_relevance(
                            cand=cand,
                            query=query,
                            visual_keywords=scene.get("visual_keywords", []),
                            scene=scene,
                            foreign_places=self.foreign_places,
                            is_sensitive=is_sensitive,
                        )
                        yield cand, query, source_name

    def mark_candidate_used(self, cand: Candidate, query: str, source_name: str) -> None:
        """Records candidate as used, updating query counts and metrics."""
        self.used_candidate_ids.add(cand.id)
        self.metrics[source_name]["used"] += 1
        self.query_usage_counts[query] = self.query_usage_counts.get(query, 0) + 1

        if cand.kind == "image":
            meta_str = f"{cand.title} {' '.join(cand.tags)}".lower()
            if any(m in meta_str for m in MAP_DIAGRAM_MARKERS):
                self.map_diagram_count += 1

    def mark_candidate_bad(self, candidate_id: str) -> None:
        """Marks candidate bad to skip in future queries."""
        self.bad_candidate_ids.add(candidate_id)

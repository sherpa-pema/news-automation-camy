"""Coverr media source (disabled by default, strict AI-generation exclusion)."""
import logging
import os
import re
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.coverr")

AI_MARKERS = {
    "artificial intelligence",
    "midjourney",
    "sora",
    "stable diffusion",
    "runway",
    "generated",
    "cgen",
    "ai-generated",
    "synthetic",
}


def is_marked_as_ai(title: str, tags: list[str], description: str) -> bool:
    """Checks for AI generation markers using word boundaries for short terms like 'ai'."""
    # Check exact tag match
    for t in tags:
        t_lower = t.strip().lower()
        if t_lower in ("ai", "ai-generated", "synthetic", "cgen", "midjourney", "sora", "stable diffusion", "runway"):
            return True

    text = f"{title} {description} {' '.join(tags)}".lower()
    # Word boundary match for 'ai'
    if re.search(r"\bai\b", text):
        return True

    for marker in AI_MARKERS:
        if marker in text:
            return True

    return False


class CoverrSource(BaseMediaSource):
    """Fetches video footage from Coverr if enabled."""

    @property
    def name(self) -> str:
        return "coverr"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("coverr", {})
        if not cfg.get("enabled", False):
            return False, "coverr source is disabled by default in config"
        key_var = cfg.get("api_key_env", "COVERR_API_KEY")
        api_key = os.getenv(key_var)
        if not api_key:
            return False, f"Missing required environment variable {key_var}"
        return True, None

    def search(
        self,
        query: str,
        kind: str = "any",
        orientation: str = "landscape",
        min_width: int = 1280,
        offline: bool = False,
        scene_category: str | None = None,
        is_sensitive: bool = False,
    ) -> list[Candidate]:
        available, reason = self.is_available
        if not available:
            logger.debug(f"Coverr skipped: {reason}")
            return []

        if kind == "image":
            return []

        cfg = self.config.get("sources", {}).get("coverr", {})
        api_key = os.getenv(cfg.get("api_key_env", "COVERR_API_KEY"))
        endpoint = cfg.get("endpoint", "https://api.coverr.co/videos")

        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"query": query, "page_size": 15}

        # Rate limit: 50 calls / hour = ~72 sec interval
        data = self.cache_mgr.get_json(endpoint, params=params, headers=headers, min_interval=2.0, domain_key="coverr.co", offline=offline)
        if not data or "hits" not in data:
            return []

        candidates: list[Candidate] = []
        for hit in data["hits"]:
            title = hit.get("title", "")
            tags = [t.lower() for t in hit.get("tags", [])]
            desc = hit.get("description", "")
            meta_str = f"{title} {' '.join(tags)} {desc}".lower()

            # Strict AI-generated exclusion rule: reject when unsure
            if is_marked_as_ai(title, tags, desc) or hit.get("is_ai"):
                logger.debug(f"Coverr video {hit.get('id')} rejected as AI-generated: {title}")
                continue

            if is_sensitive:
                person_markers = ["person", "people", "man", "woman", "child", "crowd"]
                if any(pm in meta_str for pm in person_markers):
                    continue

            cand = Candidate(
                source="coverr",
                id=f"coverr_{hit.get('id')}",
                kind="video",
                page_url=hit.get("page_url", f"https://coverr.co/videos/{hit.get('id')}"),
                download_url=hit.get("urls", {}).get("mp4") or hit.get("download_url"),
                width=int(hit.get("width", 1920)),
                height=int(hit.get("height", 1080)),
                duration=float(hit.get("duration", 0)),
                license_id="Coverr License",
                license_url="https://coverr.co/license",
                author="Coverr contributor",
                title=title,
                tags=tags,
                attribution_text="Video by Coverr (https://coverr.co)",
            )
            candidates.append(cand)

        return candidates

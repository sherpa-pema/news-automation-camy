"""Unsplash media source (disabled by default, stills only, download tracking)."""
import logging
import os
from pathlib import Path
import requests
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.unsplash")


class UnsplashSource(BaseMediaSource):
    """Fetches photos from Unsplash if enabled."""

    @property
    def name(self) -> str:
        return "unsplash"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("unsplash", {})
        if not cfg.get("enabled", False):
            return False, "unsplash source is disabled by default in config"
        key_var = cfg.get("api_key_env", "UNSPLASH_ACCESS_KEY")
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
            return []

        if kind == "video":
            return []

        cfg = self.config.get("sources", {}).get("unsplash", {})
        endpoint = cfg.get("endpoint", "https://api.unsplash.com/search/photos")
        api_key = os.getenv(cfg.get("api_key_env", "UNSPLASH_ACCESS_KEY"))

        headers = {"Authorization": f"Client-ID {api_key}"}
        orient = "landscape" if orientation == "landscape" else "portrait"
        params = {"query": query, "orientation": orient, "per_page": 15}

        data = self.cache_mgr.get_json(endpoint, params=params, headers=headers, min_interval=1.5, domain_key="unsplash.com", offline=offline)
        if not data or "results" not in data:
            return []

        candidates: list[Candidate] = []
        for res in data["results"]:
            user_info = res.get("user", {})
            user_name = user_info.get("name", "Unsplash photographer")
            tags = [t.get("title", "").lower() for t in res.get("tags", []) if isinstance(t, dict)]

            if is_sensitive:
                person_markers = ["person", "people", "man", "woman", "child", "crowd"]
                if any(pm in tags for pm in person_markers):
                    continue

            download_location = res.get("links", {}).get("download_location")
            image_url = res.get("urls", {}).get("regular")

            cand = Candidate(
                source="unsplash",
                id=f"unsplash_{res.get('id')}",
                kind="image",
                page_url=res.get("links", {}).get("html", "https://unsplash.com"),
                download_url=image_url,
                width=int(res.get("width", 1920)),
                height=int(res.get("height", 1080)),
                duration=None,
                license_id="Unsplash License",
                license_url="https://unsplash.com/license",
                author=user_name,
                title=res.get("alt_description") or f"Unsplash photo {res.get('id')}",
                tags=tags,
                attribution_text=f"Photo by {user_name} on Unsplash",
                extra={"download_location": download_location, "motion": "ken_burns"},
            )
            candidates.append(cand)

        return candidates

    def track_download(self, download_location: str) -> None:
        """Calls the Unsplash download endpoint per API guidelines."""
        cfg = self.config.get("sources", {}).get("unsplash", {})
        api_key = os.getenv(cfg.get("api_key_env", "UNSPLASH_ACCESS_KEY"))
        if not api_key or not download_location:
            return

        try:
            requests.get(
                download_location,
                headers={"Authorization": f"Client-ID {api_key}", "User-Agent": self.cache_mgr.user_agent},
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Failed to track Unsplash download: {e}")

    def download(self, candidate: Candidate, offline: bool = False) -> Path | None:
        """Downloads an Unsplash photo and triggers the required download tracking."""
        raw_path = self.cache_mgr.download_file(candidate.download_url, offline=offline)
        download_loc = (candidate.extra or {}).get("download_location")
        if download_loc:
            self.track_download(download_loc)
        return raw_path

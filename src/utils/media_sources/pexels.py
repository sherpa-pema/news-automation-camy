"""Pexels media source (disabled by default; key issuance paused; strictly no web scraping)."""
import logging
import os
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.pexels")


class PexelsSource(BaseMediaSource):
    """Pexels API provider. Only operates if enabled and PEXELS_API_KEY is present."""

    @property
    def name(self) -> str:
        return "pexels"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("pexels", {})
        if not cfg.get("enabled", False):
            return False, "pexels source is disabled in config (API key issuance paused)"
        key_var = cfg.get("api_key_env", "PEXELS_API_KEY")
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
            logger.debug(f"Pexels skipped: {reason}")
            return []

        # If enabled and key exists, query API:
        cfg = self.config.get("sources", {}).get("pexels", {})
        api_key = os.getenv(cfg.get("api_key_env", "PEXELS_API_KEY"))
        endpoint = cfg.get("endpoint", "https://api.pexels.com/videos/search")

        headers = {"Authorization": api_key}
        orientation_param = "portrait" if orientation == "vertical" else orientation
        params = {"query": query, "orientation": orientation_param, "per_page": 15}

        data = self.cache_mgr.get_json(endpoint, params=params, headers=headers, min_interval=0.5, domain_key="pexels.com", offline=offline)
        if not data or "videos" not in data:
            return []

        candidates: list[Candidate] = []
        for vid in data["videos"]:
            files = vid.get("video_files", [])
            if not files:
                continue
            best_file = None
            for f in files:
                ftype = f.get("file_type", "")
                if ftype and ftype != "video/mp4":
                    continue
                w = int(f.get("width") or 0)
                h = int(f.get("height") or 0)
                q = f.get("quality", "")
                if q == "hd" or (1280 <= w <= 1920) or (720 <= h <= 1080):
                    best_file = f
                    break
            if not best_file:
                best_file = files[0]
            cand = Candidate(
                source="pexels",
                id=f"pexels_{vid.get('id')}",
                kind="video",
                page_url=vid.get("url", "https://pexels.com"),
                download_url=best_file.get("link"),
                width=int(best_file.get("width") or 1920),
                height=int(best_file.get("height") or 1080),
                duration=float(vid.get("duration", 0)),
                license_id="Pexels License",
                license_url="https://www.pexels.com/license/",
                author=vid.get("user", {}).get("name", "Pexels creator"),
                title=f"Pexels video {vid.get('id')}",
                attribution_text=f"Video by {vid.get('user', {}).get('name', 'Pexels creator')} via Pexels",
            )
            candidates.append(cand)

        return candidates

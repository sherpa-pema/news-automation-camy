"""Openverse media source (stills only, strict CC0/PDM/BY licensing)."""
import logging
import os
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.openverse")


class OpenverseSource(BaseMediaSource):
    """Fetches CC0, PDM, and CC-BY images from Openverse."""

    @property
    def name(self) -> str:
        return "openverse"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("openverse", {})
        if not cfg.get("enabled", True):
            return False, "openverse source is disabled in config"
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
        if kind == "video":
            return []

        cfg = self.config.get("sources", {}).get("openverse", {})
        endpoint = cfg.get("endpoint", "https://api.openverse.org/v1/images/")
        token_var = cfg.get("token_env", "OPENVERSE_TOKEN")
        token = os.getenv(token_var)

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # Only request allowed open licenses: cc0, pdm, by (explicitly excluding by-sa, nc, nd)
        params = {
            "q": query,
            "license": "cc0,pdm,by",
            "page_size": 20,
        }

        data = self.cache_mgr.get_json(endpoint, params=params, headers=headers, min_interval=0.5, domain_key="openverse.org", offline=offline)
        if not data or "results" not in data:
            return []

        candidates: list[Candidate] = []
        for res in data["results"]:
            lic = res.get("license", "").lower()
            # Double check: exclude SA, NC, ND
            if "sa" in lic or "nc" in lic or "nd" in lic:
                continue

            width = int(res.get("width") or 1920)
            height = int(res.get("height") or 1080)

            if orientation == "landscape" and width < height:
                continue
            if orientation == "vertical" and height < width:
                continue

            tags = [t.get("name", "").lower() for t in res.get("tags", []) if isinstance(t, dict)]
            title = res.get("title", "")

            if is_sensitive:
                person_markers = ["person", "people", "man", "woman", "child", "crowd"]
                if any(pm in tags for pm in person_markers) or any(pm in title.lower() for pm in person_markers):
                    continue

            creator = res.get("creator") or "Unknown creator"
            foreign_landing_url = res.get("foreign_landing_url") or res.get("url")
            attr_req = "by" in lic.lower()
            attr_text = f"Photo by {creator} via Openverse ({lic.upper()})"

            cand = Candidate(
                source="openverse",
                id=f"openverse_{res.get('id')}",
                kind="image",
                page_url=foreign_landing_url,
                download_url=res.get("url"),
                width=width,
                height=height,
                duration=None,
                license_id=lic.upper(),
                license_url=res.get("license_url", "https://creativecommons.org/licenses/"),
                author=creator,
                title=title,
                tags=tags,
                attribution_text=attr_text,
                attribution_required=attr_req,
                extra={"motion": "ken_burns", "provider": res.get("provider", "openverse")},
                thumb_url=res.get("thumbnail") or res.get("url", ""),
            )
            candidates.append(cand)

        return candidates

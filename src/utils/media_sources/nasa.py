"""NASA Image and Video Library source with category gating and third-party checks."""
import logging
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.nasa")

NASA_ALLOWED_CATEGORIES = {"weather", "climate", "space", "earth"}
THIRD_PARTY_MARKERS = ["courtesy of", "copyright", "third-party", "getty", "reuters", "associated press"]


class NasaSource(BaseMediaSource):
    """Fetches NASA images and videos for weather, climate, space, and earth scenes."""

    @property
    def name(self) -> str:
        return "nasa"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("nasa", {})
        if not cfg.get("enabled", True):
            return False, "nasa source is disabled in config"
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
        # Strict category gating: only allowed for weather, climate, space, or Earth categories!
        cat_lower = (scene_category or "").lower()
        if cat_lower not in NASA_ALLOWED_CATEGORIES:
            return []

        cfg = self.config.get("sources", {}).get("nasa", {})
        endpoint = cfg.get("endpoint", "https://images-api.nasa.gov/search")

        media_type = "video,image"
        if kind == "video":
            media_type = "video"
        elif kind == "image":
            media_type = "image"

        params = {
            "q": query,
            "media_type": media_type,
            "page_size": 15,
        }

        data = self.cache_mgr.get_json(endpoint, params=params, min_interval=0.5, domain_key="nasa.gov", offline=offline)
        if not data or "collection" not in data or "items" not in data["collection"]:
            return []

        candidates: list[Candidate] = []
        for item in data["collection"]["items"]:
            data_list = item.get("data", [])
            if not data_list:
                continue

            entry = data_list[0]
            nasa_id = entry.get("nasa_id", "")
            title = entry.get("title", "")
            desc = entry.get("description", "")
            meta_str = f"{title} {desc}".lower()

            # Skip items with NASA logo, insignia, patch
            if any(term in meta_str for term in ["logo", "insignia", "patch"]):
                continue

            # Skip any item whose metadata indicates third-party content
            sec_creator = entry.get("secondary_creator", "").lower()
            photographer = entry.get("photographer", "").lower()
            if any(m in meta_str for m in THIRD_PARTY_MARKERS) or any(m in sec_creator for m in THIRD_PARTY_MARKERS):
                logger.debug(f"NASA item {nasa_id} skipped: third-party content detected")
                continue

            links = item.get("links", [])
            preview_href = links[0].get("href") if links else None
            if not preview_href:
                continue

            item_type = entry.get("media_type", "image")
            cand_kind = "video" if item_type == "video" else "image"

            download_url = preview_href
            if cand_kind == "video":
                collection_href = item.get("href")
                if collection_href:
                    manifest_data = self.cache_mgr.get_json(collection_href, min_interval=0.2, domain_key="nasa.gov", offline=offline)
                    if manifest_data and isinstance(manifest_data, list):
                        mp4s = [u for u in manifest_data if isinstance(u, str) and u.lower().endswith(".mp4")]
                        best_mp4 = None
                        for u in mp4s:
                            if "~medium.mp4" in u or "~orig.mp4" in u:
                                best_mp4 = u
                                break
                        if not best_mp4 and mp4s:
                            best_mp4 = mp4s[0]
                        if best_mp4:
                            download_url = best_mp4

            tags = [k.lower() for k in entry.get("keywords", [])]

            if is_sensitive:
                person_markers = ["person", "people", "man", "woman", "child", "crowd"]
                if any(pm in tags for pm in person_markers):
                    continue

            # NASA media policy: credit "NASA"
            cand = Candidate(
                source="nasa",
                id=f"nasa_{nasa_id}",
                kind=cand_kind,
                page_url=f"https://images.nasa.gov/details/{nasa_id}",
                download_url=download_url,
                width=1920,
                height=1080,
                duration=None,
                license_id="NASA media",
                license_url="https://www.nasa.gov/multimedia/guidelines/index.html",
                author="NASA",
                title=title,
                tags=tags,
                attribution_text="Imagery courtesy of NASA",
                attribution_required=False,
                extra={"motion": "ken_burns" if cand_kind == "image" else None},
                thumb_url=preview_href,
            )
            candidates.append(cand)

        return candidates

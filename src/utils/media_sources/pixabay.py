"""Pixabay media source for videos and photos."""
import logging
import os
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.pixabay")


class PixabaySource(BaseMediaSource):
    """Fetches stock videos and photos from Pixabay."""

    @property
    def name(self) -> str:
        return "pixabay"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("pixabay", {})
        if not cfg.get("enabled", True):
            return False, "pixabay source is disabled in config"
        key_var = cfg.get("api_key_env", "PIXABAY_API_KEY")
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
            logger.warning(f"Pixabay skipped: {reason}")
            return []

        api_key = os.getenv(self.config.get("sources", {}).get("pixabay", {}).get("api_key_env", "PIXABAY_API_KEY"))
        cfg = self.config.get("sources", {}).get("pixabay", {})
        candidates: list[Candidate] = []

        # 1. Video search
        if kind in ("any", "video"):
            endpoint = cfg.get("videos_endpoint", "https://pixabay.com/api/videos/")
            params = {
                "key": api_key,
                "q": query,
                "safesearch": "true",
                "per_page": 20,
            }
            # Pixabay videos endpoint has NO orientation parameter; we filter by dimensions of renditions!
            data = self.cache_mgr.get_json(endpoint, params=params, min_interval=0.6, domain_key="pixabay.com", offline=offline)
            if data and "hits" in data:
                for hit in data["hits"]:
                    hit_id = str(hit.get("id"))
                    page_url = hit.get("pageURL", f"https://pixabay.com/videos/id-{hit_id}/")
                    duration = float(hit.get("duration", 0))
                    tags = [t.strip().lower() for t in hit.get("tags", "").split(",") if t.strip()]
                    user = hit.get("user", "Pixabay contributor")

                    # Sensitive scene check: reject people tags
                    if is_sensitive:
                        person_markers = ["person", "people", "man", "woman", "child", "crowd", "girl", "boy"]
                        if any(pm in tags for pm in person_markers):
                            continue

                    videos_dict = hit.get("videos", {})
                    # Select rendition meeting target orientation and resolution (prefer large, then medium)
                    selected_rendition = None
                    selected_size = "medium"

                    for rend_name in ["large", "medium"]:
                        rend = videos_dict.get(rend_name)
                        if not rend or not rend.get("url"):
                            continue
                        w = int(rend.get("width", 0))
                        h = int(rend.get("height", 0))

                        # Check orientation filter
                        if orientation == "landscape" and w < h:
                            continue
                        if orientation == "vertical" and h < w:
                            continue

                        # Check minimum resolution
                        short_side = min(w, h)
                        if short_side >= 720:
                            selected_rendition = rend
                            selected_size = rend_name
                            break

                    if not selected_rendition:
                        continue

                    pic_id = hit.get("picture_id")
                    video_thumb = f"https://i.vimeocdn.com/video/{pic_id}_640x360.jpg" if pic_id else hit.get("videos", {}).get("tiny", {}).get("url", "")

                    cand = Candidate(
                        source="pixabay",
                        id=f"pixabay_v_{hit_id}",
                        kind="video",
                        page_url=page_url,
                        download_url=selected_rendition["url"],
                        width=int(selected_rendition["width"]),
                        height=int(selected_rendition["height"]),
                        duration=duration,
                        license_id="Pixabay Content License",
                        license_url="https://pixabay.com/service/license-summary/",
                        author=user,
                        title=f"Pixabay Video {hit_id}",
                        tags=tags,
                        attribution_text=f"Video by {user} via Pixabay",
                        attribution_required=False,
                        extra={"rendition": selected_size, "thumb_url": video_thumb},
                        thumb_url=video_thumb,
                    )
                    candidates.append(cand)

        # 2. Image search
        if kind in ("any", "image"):
            endpoint = cfg.get("photos_endpoint", "https://pixabay.com/api/")
            pixabay_orient = "horizontal" if orientation == "landscape" else "vertical"
            params = {
                "key": api_key,
                "q": query,
                "image_type": "photo",
                "orientation": pixabay_orient,
                "safesearch": "true",
                "per_page": 20,
            }
            data = self.cache_mgr.get_json(endpoint, params=params, min_interval=0.6, domain_key="pixabay.com", offline=offline)
            if data and "hits" in data:
                for hit in data["hits"]:
                    hit_id = str(hit.get("id"))
                    page_url = hit.get("pageURL", f"https://pixabay.com/photos/id-{hit_id}/")
                    tags = [t.strip().lower() for t in hit.get("tags", "").split(",") if t.strip()]
                    user = hit.get("user", "Pixabay contributor")

                    if is_sensitive:
                        person_markers = ["person", "people", "man", "woman", "child", "crowd", "girl", "boy"]
                        if any(pm in tags for pm in person_markers):
                            continue

                    download_url = hit.get("largeImageURL") or hit.get("webformatURL")
                    w = int(hit.get("imageWidth") or 1920)
                    h = int(hit.get("imageHeight") or 1080)

                    img_thumb = hit.get("webformatURL") or hit.get("previewURL") or download_url

                    cand = Candidate(
                        source="pixabay",
                        id=f"pixabay_img_{hit_id}",
                        kind="image",
                        page_url=page_url,
                        download_url=download_url,
                        width=w,
                        height=h,
                        duration=None,
                        license_id="Pixabay Content License",
                        license_url="https://pixabay.com/service/license-summary/",
                        author=user,
                        title=f"Pixabay Photo {hit_id}",
                        tags=tags,
                        attribution_text=f"Photo by {user} via Pixabay",
                        attribution_required=False,
                        extra={"thumb_url": img_thumb},
                        thumb_url=img_thumb,
                    )
                    candidates.append(cand)

        return candidates

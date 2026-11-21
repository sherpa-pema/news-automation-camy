"""Wikimedia Commons media source using MediaWiki API and extmetadata."""
import logging
import re
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.wikimedia")


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".ogv", ".mov"}
DISALLOWED_EXTS = {".svg", ".pdf", ".tif", ".tiff", ".gif"}


def clean_html(raw_html: str) -> str:
    """Strips HTML tags and unescapes entities."""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return " ".join(clean.split()).strip()


SA_PATTERN = re.compile(r"(?:^|[\s\-_/])(sa|sharealike)(?:$|[\s\-_/0-9])")
NC_PATTERN = re.compile(r"(?:^|[\s\-_/])(nc|noncommercial)(?:$|[\s\-_/0-9])")
ND_PATTERN = re.compile(r"(?:^|[\s\-_/])(nd|noderivs)(?:$|[\s\-_/0-9])")


class WikimediaCommonsSource(BaseMediaSource):
    """Fetches Nepal-specific images and videos from Wikimedia Commons."""

    @property
    def name(self) -> str:
        return "wikimedia_commons"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("wikimedia_commons", {})
        if not cfg.get("enabled", True):
            return False, "wikimedia_commons source is disabled in config"
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
        endpoint = self.config.get("sources", {}).get("wikimedia_commons", {}).get("endpoint", "https://commons.wikimedia.org/w/api.php")
        allow_cc_by_sa = self.config.get("settings", {}).get("allow_cc_by_sa", False)

        # Append "filetype:video" if kind == "video"
        search_term = query
        if kind == "video":
            search_term = f"{query} filetype:video"

        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrnamespace": "6",  # File namespace
            "gsrlimit": "20",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata|mime|mediatype",
            "iiurlwidth": "1920",
        }

        data = self.cache_mgr.get_json(endpoint, params=params, min_interval=0.5, domain_key="wikimedia.org", offline=offline)
        if not data or "query" not in data or "pages" not in data["query"]:
            return []

        candidates: list[Candidate] = []
        for page_id, page_data in data["query"]["pages"].items():
            imageinfo_list = page_data.get("imageinfo", [])
            if not imageinfo_list:
                continue

            info = imageinfo_list[0]
            orig_url = info.get("url", "")
            clean_url = orig_url.split("?")[0].lower()

            # Reject unsupported file types before ranking (svg, pdf, tif, gif, etc.)
            if any(clean_url.endswith(ext) for ext in DISALLOWED_EXTS):
                continue
            if not any(clean_url.endswith(ext) for ext in ALLOWED_EXTS):
                continue

            desc_url = info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/File:{page_data.get('title', '')}"
            width = info.get("width") or 0
            height = info.get("height") or 0
            mime = info.get("mime", "").lower()
            mediatype = info.get("mediatype", "").upper()

            is_video = "video" in mime or mediatype == "VIDEO" or any(clean_url.endswith(ext) for ext in [".webm", ".ogv", ".mp4", ".mov"])
            item_kind = "video" if is_video else "image"

            if kind != "any" and item_kind != kind:
                continue

            download_url = orig_url
            if not is_video and info.get("thumburl"):
                # Use pre-scaled 1080p rendition to avoid downloading huge originals
                download_url = info["thumburl"]
                width = info.get("thumbwidth") or width
                height = info.get("thumbheight") or height

            # Orientation check
            if orientation == "landscape" and width < height:
                continue
            if orientation == "vertical" and height < width:
                continue

            # Extract license from extmetadata
            ext = info.get("extmetadata", {})
            lic_short = ext.get("LicenseShortName", {}).get("value", "").strip()
            lic_url = ext.get("LicenseUrl", {}).get("value", "").strip()
            artist_raw = ext.get("Artist", {}).get("value", "Unknown author")
            artist = clean_html(artist_raw)
            credit_raw = ext.get("Credit", {}).get("value", "")
            credit = clean_html(credit_raw)
            categories = ext.get("Categories", {}).get("value", "")
            tags = [c.strip().lower() for c in categories.split("|") if c.strip()]

            # Strings like "Own work" from Credit/License must never populate the license
            lic_clean = lic_short.lower()
            if lic_clean in ("own work", "own work by original uploader", "self-published", "own", ""):
                # Do not accept "Own work" as a license name
                continue

            # Enforce license policies
            if SA_PATTERN.search(lic_clean):
                if not allow_cc_by_sa:
                    continue
            if NC_PATTERN.search(lic_clean) or ND_PATTERN.search(lic_clean):
                continue

            # Sensitive check
            if is_sensitive:
                person_markers = ["person", "people", "man", "woman", "child", "crowd", "portrait"]
                combined_text = f"{page_data.get('title', '')} {' '.join(tags)}".lower()
                if any(pm in combined_text for pm in person_markers):
                    continue

            attr_req = "BY" in lic_upper
            attr_text = f"{page_data.get('title', 'Media')} by {artist} via Wikimedia Commons ({lic_short})"
            thumb_url = info.get("thumburl") or orig_url

            candidates.append(
                Candidate(
                    source="wikimedia_commons",
                    id=f"wiki_{page_id}",
                    kind=item_kind,
                    page_url=desc_url,
                    download_url=download_url,
                    width=width,
                    height=height,
                    duration=None,  # probed later on download
                    license_id=lic_short,
                    license_url=lic_url,
                    author=artist,
                    title=clean_html(page_data.get("title", "")),
                    tags=tags,
                    attribution_text=attr_text,
                    attribution_required=attr_req,
                    people_visible=None,
                    extra={"thumb_url": thumb_url},
                    thumb_url=thumb_url,
                )
            )

        return candidates

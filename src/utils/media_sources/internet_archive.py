"""Internet Archive media source (disabled by default, strict PD/CC verification)."""
import logging
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.internet_archive")


class InternetArchiveSource(BaseMediaSource):
    """Searches Internet Archive moving image archive with verified licenses."""

    @property
    def name(self) -> str:
        return "internet_archive"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("internet_archive", {})
        if not cfg.get("enabled", False):
            return False, "internet_archive source is disabled by default in config"
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

        if kind == "image":
            return []

        cfg = self.config.get("sources", {}).get("internet_archive", {})
        search_endpoint = cfg.get("endpoint", "https://archive.org/advancedsearch.php")

        params = {
            "q": f"{query} AND mediatype:(movies)",
            "fl[]": ["identifier", "title", "licenseurl", "creator", "description"],
            "rows": 10,
            "output": "json",
        }

        data = self.cache_mgr.get_json(search_endpoint, params=params, min_interval=0.5, domain_key="archive.org", offline=offline)
        if not data or "response" not in data or "docs" not in data["response"]:
            return []

        candidates: list[Candidate] = []
        for doc in data["response"]["docs"]:
            lic_url = doc.get("licenseurl", "").lower()
            # Must have a verified public domain, cc0, or cc-by license URL
            is_valid_license = any(
                allowed in lic_url
                for allowed in ["publicdomain", "cc0", "licenses/by/"]
            )
            if not is_valid_license or "by-sa" in lic_url or "by-nc" in lic_url or "by-nd" in lic_url:
                continue

            ident = doc.get("identifier")
            mp4_filename = f"{ident}.mp4"
            meta_endpoint = cfg.get("metadata_endpoint", f"https://archive.org/metadata/{ident}")
            if not meta_endpoint.endswith(ident):
                meta_endpoint = f"{meta_endpoint.rstrip('/')}/{ident}"
            meta_data = self.cache_mgr.get_json(meta_endpoint, min_interval=0.5, domain_key="archive.org", offline=offline)
            if meta_data and "files" in meta_data:
                files = meta_data["files"]
                mp4_files = [f for f in files if isinstance(f, dict) and f.get("name", "").lower().endswith(".mp4")]
                if mp4_files:
                    best_f = sorted(mp4_files, key=lambda x: int(x.get("size") or 0), reverse=True)[0]
                    mp4_filename = best_f.get("name", mp4_filename)

            cand = Candidate(
                source="internet_archive",
                id=f"ia_{ident}",
                kind="video",
                page_url=f"https://archive.org/details/{ident}",
                download_url=f"https://archive.org/download/{ident}/{mp4_filename}",
                width=1920,
                height=1080,
                duration=None,
                license_id="Public Domain / CC BY",
                license_url=doc.get("licenseurl", ""),
                author=doc.get("creator", "Internet Archive"),
                title=doc.get("title", ident),
                attribution_text=f"Archive footage via Internet Archive ({ident})",
                attribution_required=("by" in lic_url),
            )
            candidates.append(cand)

        return candidates

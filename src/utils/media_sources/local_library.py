"""Local media library source: indexes user-provided media and sidecars in data/media_library/."""
import json
import logging
import shutil
from pathlib import Path
from typing import Any
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource
from src.utils.media_cache import MediaCacheManager
from src.utils.media_normalizer import probe_media

logger = logging.getLogger("media.local_library")

MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


class LocalLibrarySource(BaseMediaSource):
    """Indexes and matches local user footage and photos."""

    def __init__(self, config: dict[str, Any], cache_mgr: MediaCacheManager) -> None:
        super().__init__(config, cache_mgr)
        self.library_dir = Path(config.get("sources", {}).get("local_library", {}).get("path", "data/media_library"))
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.indexed_items: list[Candidate] = []
        self.warnings: list[str] = []
        self._index_library()

    @property
    def name(self) -> str:
        return "local_library"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("sources", {}).get("local_library", {})
        if not cfg.get("enabled", True):
            return False, "local_library source is disabled in config"
        return True, None

    def _index_library(self) -> None:
        """Indexes all media files with sidecars in library_dir."""
        self.indexed_items.clear()
        if not self.library_dir.exists():
            return

        for entry in self.library_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in MEDIA_EXTS:
                sidecar_path = entry.with_suffix(".json")
                if not sidecar_path.exists():
                    msg = f"Local library file {entry.name} ignored: missing sidecar JSON ({sidecar_path.name})"
                    logger.warning(msg)
                    self.warnings.append(msg)
                    continue

                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    msg = f"Local library sidecar {sidecar_path.name} invalid JSON: {e}"
                    logger.warning(msg)
                    self.warnings.append(msg)
                    continue

                # Validate sidecar fields
                required = ["title", "keywords", "license", "people_visible"]
                missing = [r for r in required if r not in meta]
                if missing:
                    msg = f"Local library sidecar {sidecar_path.name} missing fields: {missing}"
                    logger.warning(msg)
                    self.warnings.append(msg)
                    continue

                is_video = entry.suffix.lower() in VIDEO_EXTS
                width = 1920
                height = 1080
                duration = None

                try:
                    if is_video:
                        probe = probe_media(entry)
                        width = probe.get("width") or 1920
                        height = probe.get("height") or 1080
                        duration = probe.get("duration")
                    else:
                        from PIL import Image
                        with Image.open(entry) as im:
                            width, height = im.size
                except Exception as e:
                    logger.warning(f"Could not probe local file {entry.name}: {e}")

                keywords = [k.strip().lower() for k in meta.get("keywords", []) if k.strip()]

                candidate = Candidate(
                    source="user-provided",
                    id=f"local_{entry.stem}",
                    kind="video" if is_video else "image",
                    page_url=meta.get("origin_url") or "local",
                    download_url=str(entry.resolve()),
                    width=width,
                    height=height,
                    duration=duration,
                    license_id=meta.get("license", "own work"),
                    license_url=meta.get("license_url", ""),
                    author=meta.get("author", "User"),
                    title=meta.get("title", entry.stem),
                    tags=keywords,
                    attribution_text=f"Footage by {meta.get('author', 'User')}" if meta.get("author") else "User-provided footage",
                    attribution_required=("by" in str(meta.get("license", "own work")).lower()),
                    people_visible=bool(meta.get("people_visible", False)),
                    extra={"location": meta.get("location", ""), "notes": meta.get("notes", "")},
                    thumb_url=str(entry.resolve()),
                )
                self.indexed_items.append(candidate)

        logger.info(f"Local library indexed {len(self.indexed_items)} valid media items.")

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
        """Searches indexed items by keyword overlap."""
        query_words = set(w.lower() for w in query.replace(",", " ").split() if len(w) > 2)
        matches: list[tuple[int, Candidate]] = []

        for item in self.indexed_items:
            if kind != "any" and item.kind != kind:
                continue
            if is_sensitive and item.people_visible:
                continue

            item_tags = set(item.tags)
            # Check overlap between query words and item tags / title
            overlap = 0
            for qw in query_words:
                if any(qw in tag for tag in item_tags) or qw in item.title.lower():
                    overlap += 1

            if overlap > 0:
                matches.append((overlap, item))

        # Sort descending by overlap count
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches]


def add_to_library_cli(
    file_path: str | Path,
    keywords: list[str],
    license_name: str = "own work",
    license_url: str = "",
    author: str = "User",
    people_visible: bool = False,
    title: str | None = None,
    library_dir: str | Path = "data/media_library",
) -> Path:
    """CLI helper to import a file and write its sidecar JSON."""
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dest_dir = Path(library_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_file = dest_dir / src.name
    shutil.copy2(src, dest_file)

    sidecar_data = {
        "title": title or src.stem.replace("_", " ").title(),
        "keywords": keywords,
        "location": "Nepal",
        "license": license_name,
        "license_url": license_url,
        "author": author,
        "origin_url": "local",
        "people_visible": people_visible,
        "notes": "Imported via media agent CLI",
    }

    sidecar_path = dest_file.with_suffix(".json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    return dest_file

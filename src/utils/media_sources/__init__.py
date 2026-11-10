"""Media source providers registry and factory."""
from typing import Any
from src.utils.media_sources.base import BaseMediaSource
from src.utils.media_sources.local_library import LocalLibrarySource
from src.utils.media_sources.wikimedia_commons import WikimediaCommonsSource
from src.utils.media_sources.pixabay import PixabaySource
from src.utils.media_sources.coverr import CoverrSource
from src.utils.media_sources.openverse import OpenverseSource
from src.utils.media_sources.nasa import NasaSource
from src.utils.media_sources.internet_archive import InternetArchiveSource
from src.utils.media_sources.unsplash import UnsplashSource
from src.utils.media_sources.pexels import PexelsSource
from src.utils.media_sources.ytdlp import YtDlpSource
from src.utils.media_cache import MediaCacheManager

SOURCE_CLASSES: dict[str, type[BaseMediaSource]] = {
    "local_library": LocalLibrarySource,
    "wikimedia_commons": WikimediaCommonsSource,
    "pixabay": PixabaySource,
    "coverr": CoverrSource,
    "openverse": OpenverseSource,
    "nasa": NasaSource,
    "internet_archive": InternetArchiveSource,
    "unsplash": UnsplashSource,
    "pexels": PexelsSource,
    "ytdlp": YtDlpSource,
}


def build_media_sources(config: dict[str, Any], cache_mgr: MediaCacheManager) -> dict[str, BaseMediaSource]:
    """Instantiates all configured media sources."""
    sources: dict[str, BaseMediaSource] = {}
    sources_cfg = config.get("sources", {})

    for name, cls in SOURCE_CLASSES.items():
        sources[name] = cls(config, cache_mgr)

    return sources

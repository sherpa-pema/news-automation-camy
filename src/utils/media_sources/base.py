"""Base interface for all media sources."""
from abc import ABC, abstractmethod
from typing import Any
from src.utils.media_models import Candidate
from src.utils.media_cache import MediaCacheManager


class BaseMediaSource(ABC):
    """Abstract interface implemented by all media providers."""

    def __init__(self, config: dict[str, Any], cache_mgr: MediaCacheManager) -> None:
        self.config = config
        self.cache_mgr = cache_mgr

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier (e.g. 'local_library', 'pixabay')."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> tuple[bool, str | None]:
        """Checks if source is enabled and has necessary credentials.

        Returns:
            (True, None) if ready, or (False, reason) to skip with warning.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        kind: str = "any",  # "video", "image", or "any"
        orientation: str = "landscape",
        min_width: int = 1280,
        offline: bool = False,
        scene_category: str | None = None,
        is_sensitive: bool = False,
    ) -> list[Candidate]:
        """Searches the media source for candidates matching the query."""
        pass

"""Data models for Media Agent candidates, plans, and normalized shots."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """Represents a discovered media item from any source."""
    source: str  # local_library, wikimedia_commons, pixabay, coverr, openverse, nasa, etc.
    id: str
    kind: str  # "video" or "image"
    page_url: str
    download_url: str
    width: int
    height: int
    duration: float | None  # in seconds for videos
    license_id: str  # e.g. "CC0", "Pixabay Content License", "own work"
    license_url: str
    author: str
    title: str
    tags: list[str] = field(default_factory=list)
    attribution_text: str = ""
    attribution_required: bool = False
    people_visible: bool | None = None
    ai_generated: bool = False
    third_party: bool = False
    renditions: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    matched_tokens: list[str] = field(default_factory=list)
    thumb_url: str = ""


@dataclass
class PlannedShot:
    """A scheduled shot slice within a scene."""
    scene_id: str
    shot_index: int
    start: float  # Absolute time in narration.wav
    end: float    # Absolute time in narration.wav
    duration: float
    ideal_cut: float | None = None
    snapped_to_sentence: bool = False
    card_text: str | None = None


@dataclass
class PlannedScene:
    """A scene with its planned shots."""
    scene_id: str
    start: float
    end: float
    duration: float
    shots: list[PlannedShot] = field(default_factory=list)

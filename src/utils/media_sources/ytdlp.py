"""Optional yt-dlp source for candidate discovery (OFF by default, requires manual CLI approval).

NOTE:
YouTube's license label is often unreliable (it can be set by re-uploaders without rights),
and automated downloading from YouTube may conflict with its Terms of Service. Direct sources
(Wikimedia Commons, Pixabay, NASA, Local Library) are preferred. Content from news channels
is strictly forbidden under any license.
"""
import logging
from pathlib import Path
from typing import Any
from src.utils.media_models import Candidate
from src.utils.media_sources.base import BaseMediaSource

logger = logging.getLogger("media.ytdlp")


class YtDlpSource(BaseMediaSource):
    """Generates strictly allowlisted candidates requiring manual CLI approval before use."""

    @property
    def name(self) -> str:
        return "ytdlp"

    @property
    def is_available(self) -> tuple[bool, str | None]:
        cfg = self.config.get("ytdlp", {})
        if not cfg.get("enabled", False):
            return False, "yt-dlp is disabled by default in config"
        return True, None

    def search(
        self,
        query: str,
        kind: str = "video",
        orientation: str = "landscape",
        min_width: int = 1280,
        offline: bool = False,
        scene_category: str | None = None,
        is_sensitive: bool = False,
    ) -> list[Candidate]:
        available, reason = self.is_available
        if not available:
            logger.debug(f"yt-dlp skipped: {reason}")
            return []

        # If enabled, only search allowlisted domains and produce unapproved candidates
        return []

    def record_candidates(
        self,
        candidates: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Writes candidate records to ytdlp_candidates.md for human approval."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# yt-dlp Candidate Ledger (Approval Required)",
            "",
            "The following items were found via yt-dlp. To use an approved item, run:",
            "`python -m src.agents.media --date <DATE> --approve <candidate_id>`",
            "",
            "| ID | Title | URL | Channel | License | Duration |",
            "|---|---|---|---|---|---|",
        ]
        for c in candidates:
            lines.append(
                f"| {c.get('id')} | {c.get('title')} | {c.get('url')} | {c.get('channel')} | {c.get('license')} | {c.get('duration')}s |"
            )
        output_path.write_text("\n".join(lines), encoding="utf-8")

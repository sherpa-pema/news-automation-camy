"""Media Agent stub."""
from typing import Any
from src.agents.base import BaseAgent


class MediaAgent(BaseAgent):
    """Media Agent finds and downloads free/licensed visual assets per scene.

    Outputs:
        - media_manifest.json (validates against contracts/media_manifest.schema.json)
        - media/* (downloaded image/video assets)
    """

    @property
    def name(self) -> str:
        return "media"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/media_manifest.schema.json"

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Find free/licensed visuals based on scene visual keywords.

        TODO:
        1. Read script.json for scene IDs and visual_keywords.
        2. Query free media APIs (Pexels, Pixabay, Wikimedia Commons).
        3. STRICT: Never download photos or video from news outlets.
        4. If using yt-dlp, strictly filter for Creative Commons / Public Domain licenses.
        5. Download media files to media/ directory.
        6. Record origin URL, license, and attribution for every asset in media_manifest.json.
        """
        raise NotImplementedError("Media sourcing logic will be implemented in Step 2.")

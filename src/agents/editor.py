"""Editor Agent stub."""
from typing import Any
from src.agents.base import BaseAgent


class EditorAgent(BaseAgent):
    """Editor Agent selects top news stories and writes a broadcast narration script.

    Outputs:
        - script.json (validates against contracts/script.schema.json)
    """

    @property
    def name(self) -> str:
        return "editor"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/script.schema.json"

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Select top 5 stories, deduplicate across sources, and draft broadcast narration.

        TODO:
        1. Read raw/articles.json from the run directory.
        2. Filter, deduplicate, and rank stories by importance and recency.
        3. Call LLM to draft broadcast narration in our own words (never verbatim).
        4. Include mandatory source credit for each scene.
        5. Generate visual keywords for media sourcing.
        6. Validate target length (2-3 minutes, ~250-400 words) and save script.json.
        """
        raise NotImplementedError("Editor logic will be implemented in Step 2.")

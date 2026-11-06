"""Timeline Agent stub."""
from typing import Any
from src.agents.base import BaseAgent


class TimelineAgent(BaseAgent):
    """Timeline Agent produces word-level and scene-level timestamps from audio and script.

    Outputs:
        - timeline.json (validates against contracts/timeline.schema.json)
        - captions.srt (SRT subtitle file)
    """

    @property
    def name(self) -> str:
        return "timeline"

    @property
    def schema_contract_path(self) -> str | None:
        return "contracts/timeline.schema.json"

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Align narration audio with script text to produce word-level timestamps and subtitles.

        TODO:
        1. Read audio/narration.wav and script.json.
        2. Perform forced alignment / timestamped ASR (whisper-timestamped or aeneas).
        3. Compute word-level start/end offsets and map them into scene intervals.
        4. Write structured alignment to timeline.json.
        5. Generate captions.srt for video subtitling.
        """
        raise NotImplementedError("Timeline alignment logic will be implemented in Step 2.")

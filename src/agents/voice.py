"""Voice Agent stub."""
from typing import Any
from src.agents.base import BaseAgent


class VoiceAgent(BaseAgent):
    """Voice Agent converts script.json narration into speech using an open-source TTS model.

    Outputs:
        - audio/narration.wav (master narration audio)
        - audio/scenes/*.wav (per-scene audio)
    """

    @property
    def name(self) -> str:
        return "voice"

    @property
    def schema_contract_path(self) -> str | None:
        return None  # Validated via audio file checks (WAV, duration, loudness)

    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Synthesize broadcast narration audio from script.json.

        TODO:
        1. Read script.json from the run directory.
        2. Initialize open-source TTS engine (Kokoro-82M / Piper / Bark / Coqui).
        3. Synthesize speech per scene and save individual scene WAVs to audio/scenes/.
        4. Concatenate and apply audio loudness normalization (-16 LUFS) to audio/narration.wav.
        5. Verify total audio duration is within target window (100-200 seconds).
        """
        raise NotImplementedError("Voice synthesis logic will be implemented in Step 2.")

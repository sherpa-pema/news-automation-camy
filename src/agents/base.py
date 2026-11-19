"""BaseAgent abstract class defining the contract for all specialist agents."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from src.utils.schema_validate import validate_file


class BaseAgent(ABC):
    """Abstract base class for all specialist agents in the nepal-news-bot pipeline.

    Every specialist agent:
    1. Receives run paths and configuration strictly from the manifest.
    2. Executes independently without hidden runtime state.
    3. Validates its own generated artifacts against contract schemas before returning.
    """

    def __init__(self, run_dir: Path, config: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.config = config or {}
        self.last_metrics: dict[str, Any] | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique stage identifier for this agent."""
        pass

    @property
    @abstractmethod
    def schema_contract_path(self) -> str | None:
        """Relative path to the JSON Schema contract for this agent's primary output, if any."""
        pass

    @abstractmethod
    def run(self, manifest: dict[str, Any]) -> list[str]:
        """Executes the agent's task.

        Args:
            manifest: The current run manifest containing input files and state.

        Returns:
            list[str]: Relative paths of the output files created.
        """
        pass

    def validate(self, output_files: list[str]) -> tuple[bool, str | None]:
        """Validates that all expected output files exist and satisfy schema contracts.

        Returns:
            tuple[bool, str | None]: (True, None) if valid, or (False, reason) on failure.
        """
        for rel_path in output_files:
            abs_path = self.run_dir / rel_path
            if not abs_path.exists():
                return False, f"Expected output file not found on disk: {rel_path}"

        # If this agent produces a schema-governed JSON file, validate it
        if self.schema_contract_path and output_files:
            primary_json = next((f for f in output_files if f.endswith(".json")), None)
            if primary_json:
                is_valid, err = validate_file(self.run_dir / primary_json, self.schema_contract_path)
                if not is_valid:
                    return False, f"Schema validation failed for {primary_json}: {err}"

        return True, None

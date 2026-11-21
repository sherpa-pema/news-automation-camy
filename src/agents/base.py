"""BaseAgent abstract class defining the contract for all specialist agents."""
import json
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
            if rel_path.endswith(".json"):
                try:
                    with open(abs_path, "r", encoding="utf-8") as jf:
                        json.load(jf)
                except Exception as e:
                    return False, f"Invalid JSON syntax in {rel_path}: {e}"

        # If this agent produces a schema-governed JSON file, validate it
        if self.schema_contract_path and output_files:
            # Find the primary JSON matching the schema, or validate candidate JSONs
            candidate_jsons = [f for f in output_files if f.endswith(".json")]
            matched = False
            last_err = None
            for cj in candidate_jsons:
                is_valid, err = validate_file(self.run_dir / cj, self.schema_contract_path)
                if is_valid:
                    matched = True
                    break
                else:
                    last_err = (cj, err)
            if not matched and candidate_jsons:
                cj, err = last_err or (candidate_jsons[0], "Unknown schema validation failure")
                return False, f"Schema validation failed for {cj}: {err}"

        return True, None

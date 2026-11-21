"""JSON Schema validation utility stub."""
import functools
import json
from pathlib import Path
from typing import Any
import jsonschema


@functools.lru_cache(maxsize=32)
def _load_schema_cached(resolved_path: str) -> dict[str, Any]:
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    """Loads a JSON schema from disk with in-memory caching."""
    path = Path(schema_path)
    if not path.is_absolute():
        # Resolve relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        path = repo_root / path

    return _load_schema_cached(str(path.resolve()))


def validate_contract(data: Any, schema_path: str | Path) -> tuple[bool, str | None]:
    """Validates data against a target JSON Schema contract.

    Returns (True, None) on success, or (False, error_message) on failure.
    """
    schema = load_schema(schema_path)
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, f"Validation error at '{e.json_path}': {e.message}"
    except Exception as e:
        return False, f"Unexpected validation failure: {str(e)}"


def validate_file(file_path: str | Path, schema_path: str | Path) -> tuple[bool, str | None]:
    """Validates a JSON file on disk against a target schema."""
    target_file = Path(file_path)
    if not target_file.exists():
        return False, f"Target file does not exist: {target_file}"

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_contract(data, schema_path)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON syntax in {target_file}: {e}"

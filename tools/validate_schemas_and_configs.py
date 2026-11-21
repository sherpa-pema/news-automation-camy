"""Validates all YAML configuration files and JSON Schema contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import jsonschema
import yaml


def validate_all(repo_root: Path | str = ".") -> int:
    root = Path(repo_root).resolve()
    errors: list[str] = []

    # 1. Validate YAML configs
    config_dir = root / "config"
    yaml_files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("**/*.yaml"))
    print(f"Validating {len(yaml_files)} YAML config file(s)...")
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
        except Exception as exc:
            errors.append(f"YAML syntax error in {yf.relative_to(root)}: {exc}")

    # 2. Validate JSON Schemas
    contracts_dir = root / "contracts"
    schema_files = sorted(contracts_dir.glob("*.schema.json"))
    print(f"Validating {len(schema_files)} JSON Schema contract(s)...")
    for sf in schema_files:
        try:
            with open(sf, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            jsonschema.Draft202012Validator.check_schema(schema_data)
        except Exception as exc:
            errors.append(f"Schema validation error in {sf.relative_to(root)}: {exc}")

    if errors:
        print("\n[FAIL] Validation errors found:")
        for err in errors:
            print(f"  • {err}")
        return 1

    print("[PASS] All YAML configs and JSON Schema contracts are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(validate_all())

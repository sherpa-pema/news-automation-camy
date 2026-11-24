"""Deterministic validation for broadcast news scripts."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import yaml

from src.utils.editor_budget import count_syllables_in_text, estimate_scene_duration_sec
from src.utils.schema_validate import validate_contract

DEFAULT_ACRONYM_BLOCKLIST = [
    "NEA", "NRB", "NEPSE", "CIAA", "CIB", "CAAN", "NOC",
    "KMC", "TU", "UML", "RSP", "NCELL", "NTC", "IRD", "NIA",
]

DEFAULT_BLOCKED_WESTERN_TERMS = [
    "dollar bills", "us dollar", "capitol dome", "white house",
    "police cruiser", "nyse", "wall street", "euro notes",
]

UNIT_MULTIPLIERS: dict[str, float] = {
    "hundred": 100.0,
    "hundreds": 100.0,
    "thousand": 1e3,
    "thousands": 1e3,
    "हजार": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "लाख": 1e5,
    "million": 1e6,
    "millions": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "करोड": 1e7,
    "arab": 1e9,
    "arabs": 1e9,
    "अर्ब": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "kharba": 1e11,
    "kharbas": 1e11,
    "kharab": 1e11,
    "kharabs": 1e11,
    "खर्ब": 1e11,
}

WORD_TO_NUMBER: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "hundreds": 100,
    "thousand": 1000, "thousands": 1000,
    "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000,
    "million": 1000000, "millions": 1000000,
    "crore": 10000000, "crores": 10000000,
    "arab": 1000000000, "arabs": 1000000000,
    "billion": 1000000000, "billions": 1000000000,
    "kharba": 100000000000, "kharbas": 100000000000,
    "kharab": 100000000000, "kharabs": 100000000000,
}

SCALE_MULTIPLIER_VALUES = {100.0, 1000.0, 100000.0, 1000000.0, 10000000.0, 1000000000.0, 100000000000.0}


def parse_spoken_number_phrase(tokens: list[str]) -> list[float]:
    """Parses spoken number words into numeric values, supporting spoken decimals ('point') and scale multipliers."""
    numbers: list[float] = []
    current_val = 0.0
    current_acc = 0.0
    in_number = False
    in_decimal = False
    decimal_digits = ""
    last_multiplier = 0.0

    def flush_current() -> None:
        nonlocal current_val, current_acc, in_number, in_decimal, decimal_digits, last_multiplier
        if not in_number:
            return
        dec_val = 0.0
        if in_decimal and decimal_digits:
            try:
                dec_val = float("0." + decimal_digits)
            except ValueError:
                dec_val = 0.0
        total = current_val + current_acc + dec_val
        if total > 0 or in_decimal:
            numbers.append(round(total, 6))
        current_val = 0.0
        current_acc = 0.0
        in_number = False
        in_decimal = False
        decimal_digits = ""
        last_multiplier = 0.0

    n_toks = len(tokens)
    for idx, tok in enumerate(tokens):
        clean_tok = tok.lower().strip(",.")
        if clean_tok == "point":
            if in_number and idx + 1 < n_toks and tokens[idx + 1].lower().strip(",.") in WORD_TO_NUMBER:
                in_decimal = True
                last_multiplier = 0.0
                continue
            else:
                flush_current()
                continue

        # Allow 'and' inside compound phrases like "one hundred and five" or "one crore and fifty lakh"
        if clean_tok == "and" and in_number and last_multiplier >= 100.0 and idx + 1 < n_toks:
            next_tok = tokens[idx + 1].lower().strip(",.")
            if next_tok in WORD_TO_NUMBER and WORD_TO_NUMBER[next_tok] < last_multiplier:
                continue

        if in_decimal:
            if clean_tok in WORD_TO_NUMBER:
                val = WORD_TO_NUMBER[clean_tok]
                if val in SCALE_MULTIPLIER_VALUES:
                    dec_val = float("0." + decimal_digits) if decimal_digits else 0.0
                    base = (current_acc or 0.0) + dec_val
                    if val == 100.0:
                        current_acc = base * 100.0
                    else:
                        current_val += base * val
                        current_acc = 0.0
                        last_multiplier = val
                    in_decimal = False
                    decimal_digits = ""
                elif val in (20, 30, 40, 50, 60, 70, 80, 90) and idx + 1 < n_toks and 1 <= WORD_TO_NUMBER.get(tokens[idx + 1].lower().strip(",."), -1) <= 9:
                    # Multi-digit decimal like 'twenty five' -> 25
                    next_val = WORD_TO_NUMBER[tokens[idx + 1].lower().strip(",.")]
                    decimal_digits += str(int(val + next_val))
                elif 1 <= val <= 9 and idx > 0 and tokens[idx - 1].lower().strip(",.") in ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"):
                    # Already absorbed in previous tens step
                    pass
                elif 0 <= val <= 9:
                    decimal_digits += str(int(val))
                else:
                    decimal_digits += str(int(val))
            else:
                flush_current()
            continue

        if clean_tok in WORD_TO_NUMBER:
            in_number = True
            val = WORD_TO_NUMBER[clean_tok]
            if val == 100.0:
                current_acc = (current_acc or 1.0) * 100.0
                last_multiplier = 100.0
            elif val in SCALE_MULTIPLIER_VALUES:
                current_acc = (current_acc or 1.0) * val
                current_val += current_acc
                current_acc = 0.0
                last_multiplier = val
            else:
                # Support spoken year form like "twenty eighty three" (20 * 100 + 83 = 2083)
                if current_val == 0.0 and current_acc in (19.0, 20.0) and 10.0 <= val < 100.0:
                    current_acc = current_acc * 100.0 + val
                    # If this is a tens word (e.g. 20, 30, ..., 90) and next word is a unit (1..9), do not flush yet
                    next_clean = tokens[idx + 1].lower().strip(",.") if idx + 1 < n_toks else ""
                    next_val = WORD_TO_NUMBER.get(next_clean, -1.0)
                    if val in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0) and 1.0 <= next_val <= 9.0:
                        pass
                    else:
                        flush_current()
                elif 1900.0 <= current_acc <= 2099.0 and (current_acc % 10 == 0) and 1.0 <= val <= 9.0:
                    current_acc += val
                    flush_current()
                else:
                    if 1900.0 <= current_acc <= 2099.0:
                        flush_current()
                    current_acc += val
                last_multiplier = 0.0
        else:
            flush_current()

    flush_current()
    return numbers


DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def extract_numbers_from_text(text: str) -> set[float]:
    """Extracts all numeric quantities (digits, unit scales, and spelled words) from text."""
    found: set[float] = set()
    text = text.translate(DEVANAGARI_DIGITS)

    # Normalize currency prefixes by inserting a space: e.g. "Rs105.92" -> "Rs 105.92"
    norm_text = re.sub(r"\b(Rs|NRs|NPR|\$|€|£)\s*", r"\1 ", text, flags=re.IGNORECASE)
    norm_text = re.sub(r"([$€£])(\d)", r"\1 \2", norm_text)

    # 1. Regex for digit quantities with optional commas, decimals (supporting South Asian commas)
    digit_matches = re.findall(r"(?<![\d.])\d+(?:,\d{2,3})*(?:\.\d+)?(?![\d.])", norm_text)
    for m in digit_matches:
        try:
            val = float(m.replace(",", ""))
            found.add(val)
        except ValueError:
            pass

    # 2. Extract spelled-out numbers (including unit scales and spoken decimals)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    parsed_spoken = parse_spoken_number_phrase(words)
    for val in parsed_spoken:
        found.add(val)

    # 3. Currency / South Asian & Western multipliers (English + Devanagari) + compound descending pairs
    unit_regex = (
        r"(?<![\d.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*"
        r"(hundreds?|thousands?|lakhs?|lacs?|millions?|crores?|arabs?|billions?|kharbas?|kharabs?|हजार|लाख|करोड|अर्ब|खर्ब)"
        r"(?![a-zA-Z\u0900-\u097F])"
    )
    unit_matches = list(re.finditer(unit_regex, norm_text, re.IGNORECASE))
    compound_acc = 0.0
    prev_mult = 0.0
    prev_end = -1
    for m in unit_matches:
        try:
            base = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit_key = m.group(2).lower()
        mult_val = UNIT_MULTIPLIERS.get(unit_key)
        if not mult_val:
            continue
        scaled = base * mult_val
        found.add(scaled)
        found.add(base)

        gap = norm_text[prev_end:m.start()].strip().lower() if prev_end >= 0 else None
        if prev_end >= 0 and gap in ("", "and", "र") and prev_mult > mult_val:
            compound_acc += scaled
            found.add(compound_acc)
        else:
            compound_acc = scaled
        prev_mult = mult_val
        prev_end = m.end()

    return found


def find_number_spans_in_text(text: str) -> list[tuple[float, int, int]]:
    """Finds all numbers (digits and spelled words) along with their character span (start, end) in text."""
    spans: list[tuple[float, int, int]] = []
    text = text.translate(DEVANAGARI_DIGITS)

    # 1. Digits (with optional adjacent unit multiplier)
    unit_regex = (
        r"(?<![\d.])(\d+(?:,\d{2,3})*(?:\.\d+)?)"
        r"(?:\s*(hundreds?|thousands?|lakhs?|lacs?|millions?|crores?|arabs?|billions?|kharbas?|kharabs?|हजार|लाख|करोड|अर्ब|खर्ब)(?![a-zA-Z\u0900-\u097F]))?"
    )
    for m in re.finditer(unit_regex, text, re.IGNORECASE):
        try:
            base = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit_str = m.group(2)
        if unit_str and unit_str.lower() in UNIT_MULTIPLIERS:
            spans.append((base * UNIT_MULTIPLIERS[unit_str.lower()], m.start(), m.end()))
        else:
            spans.append((base, m.start(), m.end()))

    # 2. Spoken word phrases
    token_matches = list(re.finditer(r"\b[a-zA-Z]+\b", text))
    i = 0
    while i < len(token_matches):
        tm = token_matches[i]
        tok = tm.group(0).lower()
        if tok in WORD_TO_NUMBER or tok == "point":
            start_pos = tm.start()
            phrase_tokens: list[str] = []
            j = i
            last_mult = 0.0
            while j < len(token_matches):
                w = token_matches[j].group(0).lower()
                if w in WORD_TO_NUMBER:
                    phrase_tokens.append(w)
                    val = WORD_TO_NUMBER[w]
                    last_mult = val if val in SCALE_MULTIPLIER_VALUES else 0.0
                    j += 1
                elif w == "point":
                    phrase_tokens.append(w)
                    last_mult = 0.0
                    j += 1
                elif (
                    w == "and"
                    and last_mult >= 100.0
                    and j + 1 < len(token_matches)
                    and token_matches[j + 1].group(0).lower() in WORD_TO_NUMBER
                    and WORD_TO_NUMBER[token_matches[j + 1].group(0).lower()] < last_mult
                ):
                    phrase_tokens.append(w)
                    j += 1
                else:
                    break
            end_pos = token_matches[j - 1].end()
            parsed = parse_spoken_number_phrase(phrase_tokens)
            for val in parsed:
                spans.append((val, start_pos, end_pos))
            i = j
        else:
            i += 1

    return spans


def split_sentences(text: str) -> list[str]:
    """Splits text into clean sentences using text_normalizer.split_into_sentences."""
    try:
        from src.utils.text_normalizer import split_into_sentences
        return split_into_sentences(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?।॥])\s+", text.strip())
        return [s for s in sentences if s.strip()]


def load_pronunciation_keys(pronunciations_path: Path | str = "config/pronunciations.yaml") -> set[str]:
    """Loads lowercased keys from config/pronunciations.yaml."""
    p_path = Path(pronunciations_path)
    if not p_path.exists():
        return set()
    try:
        with open(p_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and isinstance(data.get("pronunciations"), dict):
            data = data["pronunciations"]
        if isinstance(data, dict):
            return {str(k).strip().lower() for k in data.keys() if str(k).strip()}
    except Exception:
        pass
    return set()


def is_bs_fiscal_or_date_context(sent: str, start_idx: int, end_idx: int) -> bool:
    """Checks whether a number span sits inside a Bikram Sambat fiscal year or date context."""
    window = sent[max(0, start_idx - 55) : min(len(sent), end_idx + 40)].lower()
    if re.search(
        r"\b(fiscal|bikram|sambat|bs|b\.s\.|year|budget|calendar|"
        r"baisakh|baishakh|jestha|jeth|ashad|asar|ashadh|shrawan|saun|"
        r"bhadra|bhadau|ashwin|asoj|kartik|mangsir|poush|push|magh|falgun|phagun|chaitra|chait|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
        window,
    ):
        return True
    prefix = sent[max(0, start_idx - 20) : start_idx].lower()
    if re.search(r"\b(in|for|of|since|by|from|through|during)\s*$", prefix):
        return True
    return False


def load_editor_config() -> tuple[dict[str, Any], set[str]]:
    """Loads editor.yaml and editor_whitelist.yaml."""
    cfg_path = Path("config/editor.yaml")
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    whitelist_path = Path("config/editor_whitelist.yaml")
    whitelist: set[str] = set()
    if whitelist_path.exists():
        with open(whitelist_path, "r", encoding="utf-8") as f:
            w_data = yaml.safe_load(f) or {}
            for noun in w_data.get("proper_nouns", []):
                whitelist.add(noun.lower())

    return cfg, whitelist


def get_imported_src_files(file_path: Path, repo_root: Path, visited: set[Path] | None = None) -> set[Path]:
    """Recursively finds all Python modules imported from src/ by the given file."""
    if visited is None:
        visited = set()
    file_path = file_path.resolve()
    if file_path in visited or not file_path.exists():
        return visited
    visited.add(file_path)

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception:
        return visited

    for node in ast.walk(tree):
        module_name = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    module_name = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("src."):
                module_name = node.module

        if module_name:
            rel_parts = module_name.split(".")
            target = repo_root.joinpath(*rel_parts).with_suffix(".py")
            if target.exists():
                get_imported_src_files(target, repo_root, visited)
            else:
                target_pkg = repo_root.joinpath(*rel_parts) / "__init__.py"
                if target_pkg.exists():
                    get_imported_src_files(target_pkg, repo_root, visited)
    return visited


def compute_script_validation_fingerprint(
    validator_path: Path | str | None = None,
    config_path: Path | str | None = None,
    whitelist_path: Path | str | None = None,
    pronunciations_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Computes SHA-256 digests covering everything that affects script validation:
    - editor_validator.py
    - every module it imports from src/
    - editor.yaml
    - editor_whitelist.yaml
    - the sensitive-trigger lists
    - contracts/script.schema.json
    - pronunciations.yaml
    """
    if repo_root:
        root = Path(repo_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent.parent
        if not (root / "contracts").exists() and Path("contracts").exists():
            root = Path(".").resolve()

    if validator_path:
        v_path = Path(validator_path).resolve()
    else:
        v_path = Path(__file__).resolve()

    if config_path:
        c_path = Path(config_path).resolve()
    else:
        c_path = root / "config" / "editor.yaml"
        if not c_path.exists():
            c_path = Path("config/editor.yaml").resolve()

    if whitelist_path:
        w_path = Path(whitelist_path).resolve()
    else:
        w_path = root / "config" / "editor_whitelist.yaml"
        if not w_path.exists():
            w_path = Path("config/editor_whitelist.yaml").resolve()

    if pronunciations_path:
        p_path = Path(pronunciations_path).resolve()
    else:
        p_path = root / "config" / "pronunciations.yaml"
        if not p_path.exists():
            p_path = Path("config/pronunciations.yaml").resolve()

    v_hash = hashlib.sha256(v_path.read_bytes()).hexdigest() if v_path.exists() else ""
    c_hash = hashlib.sha256(c_path.read_bytes()).hexdigest() if c_path.exists() else ""
    w_hash = hashlib.sha256(w_path.read_bytes()).hexdigest() if w_path.exists() else ""
    p_hash = hashlib.sha256(p_path.read_bytes()).hexdigest() if p_path.exists() else ""

    st_hash = ""
    if c_path.exists():
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                c_data = yaml.safe_load(f) or {}
            triggers = c_data.get("sensitive_triggers", [])
            st_bytes = json.dumps(sorted(triggers), ensure_ascii=False).encode("utf-8")
            st_hash = hashlib.sha256(st_bytes).hexdigest()
        except Exception:
            pass

    all_imported = get_imported_src_files(v_path, root)
    imported_modules_sha256: dict[str, str] = {}
    for mod_path in sorted(all_imported):
        if mod_path.resolve() == v_path.resolve():
            continue
        try:
            rel_name = mod_path.relative_to(root).as_posix()
        except ValueError:
            rel_name = mod_path.name
        mod_hash = hashlib.sha256(mod_path.read_bytes()).hexdigest()
        imported_modules_sha256[rel_name] = mod_hash

    # ONLY script.schema.json affects script validation
    script_schema_path = root / "contracts" / "script.schema.json"
    if not script_schema_path.exists() and Path("contracts/script.schema.json").exists():
        script_schema_path = Path("contracts/script.schema.json").resolve()
    script_schema_hash = (
        hashlib.sha256(script_schema_path.read_bytes()).hexdigest()
        if script_schema_path.exists()
        else ""
    )

    return {
        "validator_sha256": v_hash,
        "editor_config_sha256": c_hash,
        "editor_whitelist_sha256": w_hash,
        "sensitive_triggers_sha256": st_hash,
        "pronunciations_sha256": p_hash,
        "imported_modules_sha256": imported_modules_sha256,
        "script_schema_sha256": script_schema_hash,
    }


def compute_downstream_contracts_hash(
    repo_root: Path | str | None = None,
) -> dict[str, str]:
    """Computes SHA-256 digests for downstream schemas (recorded for information only, never blocks)."""
    if repo_root:
        root = Path(repo_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent.parent
        if not (root / "contracts").exists() and Path("contracts").exists():
            root = Path(".").resolve()

    contracts_dir = root / "contracts"
    if not contracts_dir.exists() and Path("contracts").exists():
        contracts_dir = Path("contracts").resolve()

    contracts_hash: dict[str, str] = {}
    if contracts_dir.exists():
        for s_file in sorted(contracts_dir.glob("*.schema.json")):
            try:
                rel_name = s_file.relative_to(root).as_posix()
            except ValueError:
                rel_name = s_file.name
            s_hash = hashlib.sha256(s_file.read_bytes()).hexdigest()
            contracts_hash[rel_name] = s_hash
    return contracts_hash


def compute_validator_fingerprint(
    validator_path: Path | str | None = None,
    config_path: Path | str | None = None,
    whitelist_path: Path | str | None = None,
    pronunciations_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Computes SHA-256 digests split into:
    - script_validation_fingerprint: validator, imported src/ modules, editor configs,
      whitelist, sensitive triggers, script.schema.json, pronunciations.yaml (GATES APPROVAL)
    - contracts_hash: downstream schemas (INFORMATIONAL ONLY, NEVER BLOCKS)
    """
    script_validation_fp = compute_script_validation_fingerprint(
        validator_path=validator_path,
        config_path=config_path,
        whitelist_path=whitelist_path,
        pronunciations_path=pronunciations_path,
        repo_root=repo_root,
    )
    contracts_hash = compute_downstream_contracts_hash(repo_root=repo_root)

    return {
        "script_validation_fingerprint": script_validation_fp,
        "contracts_hash": contracts_hash,
    }


def extract_script_validation_fingerprint(fp: dict[str, Any] | None) -> dict[str, Any]:
    """Extracts the 7 script validation components that gate approval, regardless of whether
    fp is in the new split structure or legacy flat structure."""
    if not fp or not isinstance(fp, dict):
        return {}
    if "script_validation_fingerprint" in fp and isinstance(fp["script_validation_fingerprint"], dict):
        return fp["script_validation_fingerprint"]
    return {
        "validator_sha256": fp.get("validator_sha256", ""),
        "editor_config_sha256": fp.get("editor_config_sha256", ""),
        "editor_whitelist_sha256": fp.get("editor_whitelist_sha256", ""),
        "sensitive_triggers_sha256": fp.get("sensitive_triggers_sha256", ""),
        "pronunciations_sha256": fp.get("pronunciations_sha256", ""),
        "imported_modules_sha256": fp.get("imported_modules_sha256", {}),
        "script_schema_sha256": fp.get("script_schema_sha256") or fp.get("schemas_sha256", {}).get("contracts/script.schema.json", ""),
    }


def verify_script_approval_gate(
    app_data: dict[str, Any],
    current_fp: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Verifies that the script approval gate passes.
    Only compares script_validation_fingerprint.
    Contracts hash for downstream schemas is recorded for information only and never blocks.
    """
    if current_fp is None:
        current_fp = compute_validator_fingerprint()
    app_fp = app_data.get("validator_fingerprint")
    if not app_fp:
        return False, "Missing validator_fingerprint in approval.json"

    req_fp = extract_script_validation_fingerprint(app_fp)
    cur_fp = extract_script_validation_fingerprint(current_fp)

    for key, cur_val in cur_fp.items():
        req_val = req_fp.get(key)
        if req_val != cur_val:
            return False, f"Fingerprint mismatch on {key}: approved {req_val} != current {cur_val}"
    return True, None


def validate_script_draft(
    draft_path: Path,
    shortlist_path: Path,
    budget: dict[str, Any],
) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    """Validates draft_script.json deterministically.
    
    Returns:
        (is_valid, errors, warnings, report_dict)
    """
    errors: list[str] = []
    warnings: list[str] = []
    structured_errors: list[dict[str, Any]] = []
    structured_warnings: list[dict[str, Any]] = []

    def add_global_error(msg: str) -> None:
        errors.append(msg)
        structured_errors.append({"scene_index": None, "scene_id": None, "message": msg})

    def add_global_warning(msg: str) -> None:
        warnings.append(msg)
        structured_warnings.append({"scene_index": None, "scene_id": None, "message": msg})

    report: dict[str, Any] = {
        "validator_fingerprint": compute_validator_fingerprint(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "structured_errors": [],
        "structured_warnings": [],
        "scene_results": {},
        "metrics": {},
    }

    if not draft_path.exists():
        msg = f"Draft script file not found at {draft_path}"
        add_global_error(msg)
        report["errors"] = errors
        report["structured_errors"] = structured_errors
        return False, errors, warnings, report

    try:
        with open(draft_path, "r", encoding="utf-8") as f:
            draft_data = json.load(f)
    except Exception as e:
        msg = f"Failed to parse draft JSON: {e}"
        add_global_error(msg)
        report["errors"] = errors
        report["structured_errors"] = structured_errors
        return False, errors, warnings, report

    # 1. Schema Validation against contracts/script.schema.json
    schema_path = Path("contracts/script.schema.json")
    if schema_path.exists():
        is_schema_valid, schema_err = validate_contract(draft_data, schema_path)
        if not is_schema_valid:
            add_global_error(f"Schema validation failed: {schema_err}")

    # Load shortlist for grounding and cluster checks
    shortlist = []
    if shortlist_path.exists():
        with open(shortlist_path, "r", encoding="utf-8") as f:
            shortlist = json.load(f)
    cluster_by_url = {}
    cluster_by_id = {}
    for c in shortlist:
        cluster_by_id[c["cluster_id"]] = c
        for a in c.get("articles", []):
            cluster_by_url[a["url"]] = c

    cfg, proper_noun_whitelist = load_editor_config()
    pron_keys = load_pronunciation_keys()
    sensitive_triggers = set(cfg.get("sensitive_triggers", []))
    acronym_blocklist = cfg.get("acronym_blocklist", DEFAULT_ACRONYM_BLOCKLIST)
    blocked_acronyms = {
        str(a).strip().lower(): str(a).strip()
        for a in acronym_blocklist
        if str(a).strip() and str(a).strip().lower() not in pron_keys
    }
    blocked_western_terms = [
        str(t).strip().lower()
        for t in cfg.get("visual_keyword_rules", {}).get("blocked_western_terms", DEFAULT_BLOCKED_WESTERN_TERMS)
        if str(t).strip()
    ]

    # Expand multi-word entries in whitelist & pronunciations for single-token lookups
    whitelist_token_words = set(proper_noun_whitelist)
    for entry in proper_noun_whitelist:
        for part in entry.split():
            whitelist_token_words.add(part)
    pron_token_words = set(pron_keys)
    for entry in pron_keys:
        for part in entry.split():
            pron_token_words.add(part)

    # Determine run Gregorian year for Bikram Sambat (BS) fiscal/date tolerance
    run_year = datetime.now(timezone.utc).year
    date_str = str(draft_data.get("date") or "")
    m_yr = re.match(r"^(\d{4})-\d{2}-\d{2}$", date_str)
    if m_yr:
        run_year = int(m_yr.group(1))
    bs_whitelist_years = {float(run_year + 56), float(run_year + 57), float(run_year + 58)}
    bs_short_years = {
        float((run_year + 56) % 100),
        float((run_year + 57) % 100),
        float((run_year + 58) % 100),
    }

    scenes = draft_data.get("scenes", [])

    # Format determination: Decouple format rules from draft scene count to avoid moving-target trap
    story_target_max = budget.get("scene_targets", {}).get("story", {}).get("max_words", 60)
    story_target_count = budget.get("scene_targets", {}).get("story", {}).get("count", 0)

    if draft_data.get("format") == "headlines" or budget.get("format") == "headlines":
        is_headlines = True
    elif draft_data.get("format") == "focused" or budget.get("format") == "focused":
        is_headlines = False
    elif len(scenes) in (7, 8):
        story_scenes = [s for s in scenes if s.get("type") == "story"]
        if story_scenes and all(len(split_sentences(s.get("narration", ""))) <= 1 for s in story_scenes) and story_target_max <= 28 and len(shortlist) >= 12:
            is_headlines = True
        else:
            is_headlines = False
    elif len(scenes) >= 12:
        is_headlines = True
    elif story_target_count >= 14 or story_target_max <= 28:
        is_headlines = True
    else:
        is_headlines = False

    if is_headlines:
        min_allowed_scenes = min(12, len(shortlist) + 2) if (shortlist and is_headlines) else 12
        if not (len(scenes) in (7, 8) or (min_allowed_scenes <= len(scenes) <= 26)):
            add_global_error(f"Expected {min_allowed_scenes} to 26 scenes (intro + story headlines + outro; legacy: Expected exactly 7 scenes), got {len(scenes)}")
    else:
        min_allowed_scenes = min(4, len(shortlist) + 2) if shortlist else 4
        if not (min_allowed_scenes <= len(scenes) <= 10):
            add_global_error(f"Expected {min_allowed_scenes} to 10 scenes in focused bulletin, got {len(scenes)}")

    broadcast_seq_cfg = None
    if cfg and "broadcast_sequence" in cfg:
        broadcast_seq_cfg = cfg["broadcast_sequence"]
    elif Path("config/editor.yaml").exists():
        try:
            with open("config/editor.yaml", "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
                broadcast_seq_cfg = y.get("broadcast_sequence")
        except Exception:
            broadcast_seq_cfg = None

    if broadcast_seq_cfg and isinstance(broadcast_seq_cfg, list):
        BROADCAST_SEQUENCE_RULES = []
        for idx, entry in enumerate(broadcast_seq_cfg, 1):
            s_id = entry.get("id", f"scene_{idx:02d}")
            s_cat = entry.get("category", "")
            s_aliases = set(entry.get("aliases", [s_cat])) | {s_cat}
            BROADCAST_SEQUENCE_RULES.append((idx, s_id, s_cat, s_aliases))
    else:
        BROADCAST_SEQUENCE_RULES = [
            (1, "scene_01", "national_politics", {"national_politics", "politics", "political", "government", "parliament"}),
            (2, "scene_02", "society_local", {"society_local", "society", "social", "local", "education", "health", "transport", "disaster", "crime", "weather", "climate", "environment"}),
            (3, "scene_03", "culture_entertainment", {"culture_entertainment", "culture", "cultural", "entertainment", "heritage", "festival", "arts", "tourism"}),
            (4, "scene_04", "business_economics", {"business_economics", "economy", "economic", "business", "market", "finance", "trade"}),
            (5, "scene_05", "science_technology", {"science_technology", "technology", "tech", "science", "digital", "telecom", "cyber"}),
            (6, "scene_06", "sports", {"sports", "sport", "cricket", "football", "athletics"}),
        ]

    seen_cluster_ids: set[str] = set()
    story_count = sum(1 for sc in scenes if sc.get("type", "story") == "story")

    total_words = 0
    total_syllables = 0
    total_est_duration_sec = 0.0
    wps = budget.get("words_per_second", 2.15)
    sps = budget.get("syllables_per_second", 3.3)
    current_max_slot_idx = 0

    for sc_idx, scene in enumerate(scenes):
        sc_id = scene.get("id", f"scene_{sc_idx}")
        sc_type = scene.get("type", "story")
        headline = scene.get("headline", "")
        narration = scene.get("narration", "")
        visual_keywords = scene.get("visual_keywords", [])
        source = scene.get("source", "")
        source_urls = scene.get("source_urls", [])
        evidence = scene.get("evidence", [])
        sensitive = scene.get("sensitive", False)

        sc_errors: list[str] = []
        sc_warnings: list[str] = []

        # Broadcast sequence check for 8-scene scripts
        if len(scenes) == 8 and 1 <= sc_idx <= 6:
            _, exp_id, exp_slot, valid_cats = BROADCAST_SEQUENCE_RULES[sc_idx - 1]
            cat_val = (scene.get("category") or "").strip().lower()
            if cat_val not in valid_cats:
                sc_errors.append(
                    f"Scene {sc_id} category '{cat_val}' violates broadcast sequence; expected '{exp_slot}' (Slot {sc_idx})"
                )

        # Monotonic sequential category block check for multi-scene scripts
        if (is_headlines or len(scenes) >= 14) and 1 <= sc_idx <= len(scenes) - 2:
            cat_val = (scene.get("category") or "").strip().lower()
            matched_slot_idx = None
            for slot_idx, (_, _, slot_name, valid_cats) in enumerate(BROADCAST_SEQUENCE_RULES):
                if cat_val in valid_cats:
                    matched_slot_idx = slot_idx
                    break
            if matched_slot_idx is None:
                sc_errors.append(
                    f"Scene {sc_id} category '{cat_val}' is not a recognized broadcast category"
                )
            elif matched_slot_idx < current_max_slot_idx:
                prev_slot_name = BROADCAST_SEQUENCE_RULES[current_max_slot_idx][2]
                sc_errors.append(
                    f"Scene {sc_id} category '{cat_val}' appears out of sequence after '{prev_slot_name}' (categories must follow sequential broadcast blocks)"
                )
            else:
                current_max_slot_idx = max(current_max_slot_idx, matched_slot_idx)

        # Headline checks
        hl_words = headline.strip().split()
        if len(hl_words) > 10:
            sc_errors.append(f"Headline exceeds 10 words ({len(hl_words)} words): '{headline}'")
        if headline.strip().endswith("."):
            sc_errors.append(f"Headline must not end with a period: '{headline}'")

        # Visual keywords count & cultural safety
        if not (3 <= len(visual_keywords) <= 5):
            sc_errors.append(f"Visual keywords count must be 3 to 5 (got {len(visual_keywords)})")

        for kw in visual_keywords:
            kw_lower = kw.strip().lower()
            for term in blocked_western_terms:
                if kw_lower == term or re.search(rf"\b{re.escape(term)}\b", kw_lower):
                    sc_errors.append(
                        f"Visual keyword '{kw}' matches blocked Western term '{term}'; use Nepal-appropriate stock keywords"
                    )
                    break

            kw_words = kw.strip().split()
            alpha_words = [w for w in kw_words if w.isalpha()]
            if len(kw_words) >= 2 and alpha_words and all(w[0].isupper() for w in alpha_words):
                if kw_lower not in proper_noun_whitelist:
                    sc_warnings.append(f"Visual keyword '{kw}' looks like a proper name or brand")

        # Narration / Spoken Text Word & Syllable Counts
        spoken_text = scene.get("spoken_text")
        target_spoken = spoken_text if spoken_text is not None else narration
        spoken_words = target_spoken.strip().split()
        word_count = len(spoken_words)
        syllable_count = count_syllables_in_text(target_spoken)
        sc_est_dur = estimate_scene_duration_sec(target_spoken, budget)
        total_words += word_count
        total_syllables += syllable_count
        total_est_duration_sec += sc_est_dur

        if sc_type == "intro":
            min_w = 10
            max_w = 25 if is_headlines else 30
            if not (min_w <= word_count <= max_w):
                sc_errors.append(f"Intro word count {word_count} outside {min_w}-{max_w} words")

            greeting_pattern = r"\b(good\s+(evening|morning|afternoon|day)|welcome\s+to\s+(our|the)\s+broadcast|hello\s+and\s+welcome)\b"
            if re.search(greeting_pattern, narration, re.IGNORECASE):
                sc_errors.append(
                    "Intro narration must not contain passive greetings ('Good evening', 'welcome to our broadcast'); use punchy evergreen category hook"
                )
        elif sc_type == "outro":
            min_w = 12 if is_headlines else 15
            max_w = 30 if is_headlines else 40
            if not (min_w <= word_count <= max_w):
                sc_errors.append(f"Outro word count {word_count} outside {min_w}-{max_w} words")
        else:
            min_w = 10 if is_headlines else 20
            max_w = 28 if is_headlines else 60
            if not (min_w <= word_count <= max_w):
                sc_errors.append(f"Story scene word count {word_count} outside {min_w}-{max_w} words")

        # Sentence count for story scenes
        sentences = split_sentences(narration)
        if sc_type == "story":
            if is_headlines:
                if len(sentences) != 1:
                    sc_errors.append(f"Headline bulletin story narration must be strictly 1 sentence (got {len(sentences)})")
            else:
                min_s = 2
                max_s = 4
                if not (min_s <= len(sentences) <= max_s):
                    sc_errors.append(f"Story narration must have {min_s} to {max_s} sentences (got {len(sentences)})")

        # TTS Rules (checked against spoken_text when provided, else narration)
        tts_text = spoken_text if spoken_text is not None else narration
        field_name = "spoken_text" if spoken_text is not None else "narration"

        if re.search(r"\d", tts_text):
            digits_found = re.findall(r"\d+", tts_text)
            sc_errors.append(f"Digits found in {field_name} (spell numbers out): {digits_found}")

        banned_symbols = re.findall(r"[%$&/\\@#^*+=<>~]", tts_text)
        if banned_symbols:
            sc_errors.append(f"Banned symbols in {field_name}: {set(banned_symbols)}")

        if re.search(r"[()\[\]{}]", tts_text):
            sc_errors.append(f"Parentheses or brackets found in {field_name}")

        # Non-ASCII / emoji check (allow Devanagari in Nepali mode)
        is_nepali = draft_data.get("language") == "Nepali"
        non_ascii = [ch for ch in tts_text if (ord(ch) > 127 and not (is_nepali and '\u0900' <= ch <= '\u097F'))]
        if non_ascii:
            sc_errors.append(f"Non-ASCII characters or emojis found: {set(non_ascii)}")

        # Case-insensitive Acronym Blocklist check
        seen_blocked_acronyms: set[str] = set()
        for tok in re.findall(r"\b[a-zA-Z]+\b", tts_text):
            tok_l = tok.lower()
            if tok_l in blocked_acronyms and tok_l not in seen_blocked_acronyms:
                seen_blocked_acronyms.add(tok_l)
                sc_errors.append(
                    f"Blocked acronym '{tok}' ({blocked_acronyms[tok_l]}) in {field_name} without pronunciation mapping; expand the full institution name"
                )

        # All-caps words (>1 letter) warning
        all_caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", tts_text)]
        if all_caps:
            sc_warnings.append(f"All-caps words found: {all_caps}")

        # Unmapped proper noun warning for TTS pronunciation coverage (all scene types)
        seen_unmapped_tts: set[str] = set()
        for sent in sentences:
            sent_words = sent.strip().split()
            for w_idx, w in enumerate(sent_words):
                if w_idx == 0:
                    continue
                clean_w = re.sub(r"['’]s$", "", w, flags=re.IGNORECASE)
                clean_w = re.sub(r"[^a-zA-Z]", "", clean_w)
                if (
                    len(clean_w) >= 2
                    and clean_w[0].isupper()
                    and clean_w.lower() not in whitelist_token_words
                    and clean_w.lower() not in pron_token_words
                    and clean_w.lower() not in seen_unmapped_tts
                ):
                    seen_unmapped_tts.add(clean_w.lower())
                    sc_warnings.append(
                        f"Potential unmapped proper noun for TTS: '{clean_w}' is absent from editor_whitelist.yaml and pronunciations.yaml"
                    )

        # Grounding & cluster checks for story scenes
        if sc_type == "story":
            if not source:
                sc_errors.append("Missing source outlet")
            if not source_urls:
                sc_errors.append("Missing source_urls")
            if not evidence:
                sc_errors.append("Missing evidence mappings")

            # Validate each evidence item explicitly against the shortlist
            covered_sentences = set()
            for ev in evidence:
                ev_idx = ev.get("sentence_index", -1)
                ev_url = ev.get("source_url", "")
                if ev_idx < 0 or ev_idx >= len(sentences):
                    sc_errors.append(f"Evidence sentence_index {ev_idx} out of range [0, {len(sentences)-1}]")
                else:
                    covered_sentences.add(ev_idx)
                if ev_url not in cluster_by_url:
                    sc_errors.append(f"Evidence source_url '{ev_url}' does not belong to any shortlisted cluster")

            for s_idx in range(len(sentences)):
                if s_idx not in covered_sentences:
                    sc_errors.append(f"Sentence {s_idx} has no corresponding evidence entry")

            # Resolve cluster for scene
            matched_clusters = []
            for u in source_urls:
                if u in cluster_by_url:
                    c = cluster_by_url[u]
                    if c not in matched_clusters:
                        matched_clusters.append(c)

            if not matched_clusters:
                sc_errors.append(f"source_urls {source_urls} do not match any shortlisted cluster")
                cluster_combined_text = ""
            else:
                cluster = matched_clusters[0]
                c_id = cluster.get("cluster_id") or cluster.get("id")
                if c_id:
                    if c_id in seen_cluster_ids and len(shortlist) >= story_count:
                        sc_errors.append(f"Duplicate story scene for cluster '{c_id}'")
                    seen_cluster_ids.add(c_id)

                articles = cluster.get("articles", [])
                non_sup_articles = [a for a in articles if not a.get("superseded", False)]
                target_articles = non_sup_articles if non_sup_articles else articles
                cluster_texts = [f"{a.get('title', '')} {a.get('full_text', '')}" for a in target_articles]
                cluster_combined_text = " ".join(cluster_texts)

                # Check evidence URLs belong to the matched cluster
                for ev in evidence:
                    ev_url = ev.get("source_url", "")
                    if ev_url in cluster_by_url and ev_url not in cluster.get("urls", []):
                        sc_errors.append(f"Evidence source_url '{ev_url}' not part of cluster URLs")

                # Sensitive trigger check
                if not sensitive:
                    cluster_lower = cluster_combined_text.lower()
                    for trig in sensitive_triggers:
                        pattern = rf"\b{re.escape(trig)}\b"
                        if re.search(pattern, cluster_lower):
                            sc_errors.append(f"Cluster text contains sensitive trigger word '{trig}', but sensitive is false")
                            break

                # Verbatim check: 8 consecutive words shared
                narration_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", narration.lower())
                cluster_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", cluster_combined_text.lower())
                cluster_token_set = set()
                for i in range(len(cluster_tokens) - 7):
                    cluster_token_set.add(tuple(cluster_tokens[i:i+8]))

                for i in range(len(narration_tokens) - 7):
                    ngram = tuple(narration_tokens[i:i+8])
                    if ngram in cluster_token_set:
                        bad_phrase = " ".join(ngram)
                        sc_errors.append(f"Verbatim 8-word overlap with source text: '{bad_phrase}'")
                        break

                # Number verification: numbers in spoken text (or narration) must appear in cluster text (with unit-scale & BS date tolerance)
                cluster_numbers = extract_numbers_from_text(cluster_combined_text)
                num_sentences = split_sentences(spoken_text) if spoken_text is not None else sentences

                for sent in num_sentences:
                    sent_spans = find_number_spans_in_text(sent)
                    sent_nums = {sp[0] for sp in sent_spans}
                    has_bs_year_in_sent = bool(sent_nums & bs_whitelist_years)
                    is_fiscal_sent = bool(re.search(r"\b(fiscal|bikram|sambat)\b", sent, re.IGNORECASE))

                    for num, start_idx, end_idx in sent_spans:
                        # Allow 0 or 1 for general speech
                        if num in (0.0, 1.0):
                            continue

                        # Bikram Sambat (BS) fiscal year / date context whitelist
                        if num in bs_whitelist_years and is_bs_fiscal_or_date_context(sent, start_idx, end_idx):
                            continue
                        if is_fiscal_sent and has_bs_year_in_sent and num in bs_short_years:
                            continue

                        # Check exact match in cluster (with float tolerance 1e-5 or relative 1e-6 for large numbers)
                        is_exact = any(
                            abs(num - c) < 1e-5 or (c != 0 and abs(num - c) / abs(c) < 1e-6)
                            for c in cluster_numbers
                        )
                        if is_exact:
                            continue

                        # Check if within 2% of any source number
                        close_match = None
                        min_rel_diff = 1.0
                        for c in cluster_numbers:
                            if c != 0:
                                diff = abs(num - c) / abs(c)
                                if diff <= 0.02 + 1e-7 and diff < min_rel_diff:
                                    min_rel_diff = diff
                                    close_match = c

                        if close_match is not None:
                            # Check if preceded by approximation modifier: about, around, over, more than, nearly
                            preceding_window = sent[max(0, start_idx - 45):start_idx]
                            mod_match = re.search(r"\b(about|around|over|more than|nearly)\b", preceding_window, re.IGNORECASE)
                            if mod_match:
                                sc_warnings.append(
                                    f"Approximated number {num} is within 2% of source number {close_match} following '{mod_match.group(1)}'"
                                )
                            else:
                                sc_errors.append(
                                    f"Number {num} is within 2% of source number {close_match} but lacks approximation modifier (about, around, over, more than, nearly)"
                                )
                        else:
                            sc_errors.append(f"Invented number {num} in narration does not appear in source cluster")

                # Proper nouns warning check (absent from cluster text)
                cluster_text_lower = cluster_combined_text.lower()
                for sent in sentences:
                    sent_words = sent.strip().split()
                    for w_idx, w in enumerate(sent_words):
                        if w_idx == 0:
                            continue  # First word of sentence is capitalized by grammar
                        clean_w = re.sub(r"['’]s$", "", w, flags=re.IGNORECASE)
                        clean_w = re.sub(r"[^a-zA-Z]", "", clean_w)
                        if clean_w and clean_w[0].isupper() and clean_w.lower() not in whitelist_token_words:
                            if clean_w.lower() not in cluster_text_lower:
                                sc_warnings.append(f"Proper noun '{clean_w}' in narration not found in cluster text")

                # High 4-word overlap warning (> 20%)
                if len(narration_tokens) >= 4:
                    narr_4grams = [tuple(narration_tokens[i:i+4]) for i in range(len(narration_tokens) - 3)]
                    cluster_4grams = set(tuple(cluster_tokens[i:i+4]) for i in range(len(cluster_tokens) - 3))
                    overlap_count = sum(1 for ng in narr_4grams if ng in cluster_4grams)
                    overlap_ratio = overlap_count / len(narr_4grams)
                    if overlap_ratio > 0.20:
                        sc_warnings.append(f"High 4-word sequence overlap with source ({overlap_ratio*100:.1f}% > 20%)")

        sc_struct_errors = [{"scene_index": sc_idx, "scene_id": sc_id, "message": e} for e in sc_errors]
        sc_struct_warnings = [{"scene_index": sc_idx, "scene_id": sc_id, "message": w} for w in sc_warnings]

        errors.extend(sc_errors)
        warnings.extend(sc_warnings)
        structured_errors.extend(sc_struct_errors)
        structured_warnings.extend(sc_struct_warnings)
        report["scene_results"][sc_id] = {
            "scene_index": sc_idx,
            "scene_id": sc_id,
            "errors": sc_errors,
            "warnings": sc_warnings,
            "structured_errors": sc_struct_errors,
            "structured_warnings": sc_struct_warnings,
            "word_count": word_count,
            "syllable_count": syllable_count,
            "est_duration_sec": sc_est_dur,
        }

    # Total word count budget check (10% tolerance)
    min_budget = budget.get("min_total_words", 250)
    max_budget = budget.get("max_total_words", 380)
    if is_headlines and len(scenes) < 20:
        min_budget = min(min_budget, max(60, int(min_budget * (len(scenes) / 20.0))))
    scene_max_sum = sum(
        (25 if s.get("type") == "intro" else (35 if s.get("type") == "outro" else (28 if is_headlines else 60)))
        for s in scenes
    )
    min_budget = min(min_budget, scene_max_sum)
    if not (min_budget <= total_words <= max_budget):
        add_global_error(f"Total word count {total_words} outside budget range [{min_budget}, {max_budget}]")

    dur_range = budget.get("duration_range_sec")
    if dur_range and isinstance(dur_range, (list, tuple)) and len(dur_range) == 2:
        min_dur = float(dur_range[0])
        max_dur = float(dur_range[1])
        if is_headlines and len(scenes) < 20:
            min_dur = min_dur * (len(scenes) / 20.0)
        if not (min_dur - 5.0 <= total_est_duration_sec <= max_dur + 5.0):
            add_global_error(
                f"Total estimated duration {total_est_duration_sec:.1f}s outside allowed range [{min_dur - 5.0:.1f}s, {max_dur + 5.0:.1f}s]"
            )

    report["metrics"] = {
        "total_words": total_words,
        "total_syllables": total_syllables,
        "words_per_second": wps,
        "syllables_per_second": sps,
        "total_est_duration_sec": round(total_est_duration_sec, 1),
        "scene_count": len(scenes),
    }

    is_valid = (len(errors) == 0)
    report["is_valid"] = is_valid
    report["errors"] = errors
    report["warnings"] = warnings
    report["structured_errors"] = structured_errors
    report["structured_warnings"] = structured_warnings

    return is_valid, errors, warnings, report


def write_script_preview_markdown(
    script_data: dict[str, Any],
    report: dict[str, Any],
    preview_path: Path,
) -> None:
    """Writes the script preview markdown file."""
    lines = []
    lines.append(f"# Broadcast Script Preview ({script_data.get('date', 'Today')})")
    lines.append("")
    lines.append(f"- **Language**: {script_data.get('language', 'English')}")
    lines.append(f"- **Origin**: `{script_data.get('origin', 'editor')}` (Mode: `{script_data.get('editor_mode', 'ide_agent')}`)")
    if "run_kind" in script_data:
        lines.append(f"- **Run Kind**: `{script_data['run_kind']}`")
    fp = report.get("validator_fingerprint") or script_data.get("validator_fingerprint")
    if fp:
        sv_fp = extract_script_validation_fingerprint(fp)
        lines.append(f"- **Validator Fingerprint**: `{sv_fp.get('validator_sha256', '')[:12]}...` (config: `{sv_fp.get('editor_config_sha256', '')[:12]}...`)")
    lines.append(f"- **Total Spoken Words**: {report['metrics'].get('total_words', 0)}")
    lines.append(f"- **Estimated Video Duration**: {report['metrics'].get('total_est_duration_sec', 0.0)}s (~{report['metrics'].get('total_est_duration_sec', 0.0)/60:.1f} mins)")
    lines.append(f"- **Status**: Awaiting Approval")
    lines.append("")

    if report.get("warnings"):
        lines.append("## Editorial Warnings for Reviewer")
        for w in report["warnings"]:
            lines.append(f"- [WARN] {w}")
        lines.append("")

    lines.append("## Broadcast Narration Scenes")
    for sc in script_data.get("scenes", []):
        sc_id = sc["id"]
        sc_type = sc.get("type", "story")
        hl = sc.get("headline", "")
        narr = sc.get("narration", "")
        src = sc.get("source", "")
        cat = sc.get("category")
        sens = sc.get("sensitive", False)
        keywords = ", ".join(sc.get("visual_keywords", []))
        evidence = sc.get("evidence", [])

        lines.append(f"### Scene: `{sc_id}` [{sc_type.upper()}]")
        lines.append(f"**Headline**: {hl}")
        lines.append("")
        lines.append(f"> \"{narr}\"")
        lines.append("")
        lines.append(f"- **Category**: `{cat}` | **Sensitive**: `{sens}`")
        lines.append(f"- **Attribution**: {src}")
        lines.append(f"- **Visual Keywords**: {keywords}")
        if evidence:
            lines.append("- **Supporting Evidence**:")
            for ev in evidence:
                lines.append(f"  - Sentence {ev['sentence_index']}: {ev['source_url']}")
        lines.append("")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

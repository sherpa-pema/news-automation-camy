"""Deterministic validation for broadcast news scripts."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import yaml

from src.utils.schema_validate import validate_contract

WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100000, "crore": 10000000, "million": 1000000, "billion": 1000000000,
}


def parse_spoken_number_phrase(tokens: list[str]) -> list[float]:
    """Parses spoken number words into numeric values, supporting spoken decimals ('point')."""
    numbers = []
    current_val = 0.0
    current_acc = 0.0
    in_number = False
    in_decimal = False
    decimal_digits = ""

    def flush_current():
        nonlocal current_val, current_acc, in_number, in_decimal, decimal_digits
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

    for tok in tokens:
        clean_tok = tok.lower().strip(",.")
        if clean_tok == "point":
            in_number = True
            in_decimal = True
            continue

        if in_decimal:
            if clean_tok in WORD_TO_NUMBER:
                val = WORD_TO_NUMBER[clean_tok]
                if 0 <= val <= 9:
                    decimal_digits += str(val)
                else:
                    decimal_digits += str(val)
            else:
                flush_current()
            continue

        if clean_tok in WORD_TO_NUMBER:
            in_number = True
            val = WORD_TO_NUMBER[clean_tok]
            if val == 100:
                current_acc = (current_acc or 1.0) * 100
            elif val in (1000, 100000, 1000000, 10000000, 1000000000):
                current_acc = (current_acc or 1.0) * val
                current_val += current_acc
                current_acc = 0.0
            else:
                current_acc += val
        else:
            flush_current()

    flush_current()
    return numbers


def extract_numbers_from_text(text: str) -> set[float]:
    """Extracts all numeric quantities (digits and spelled words) from text."""
    found = set()

    # Normalize currency prefixes by inserting a space: e.g. "Rs105.92" -> "Rs 105.92"
    norm_text = re.sub(r"\b(Rs|NRs|NPR|\$|€|£)\s*", r"\1 ", text, flags=re.IGNORECASE)
    norm_text = re.sub(r"([$€£])(\d)", r"\1 \2", norm_text)

    # 1. Regex for digit quantities with optional commas, decimals, suffixes
    digit_matches = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", norm_text)
    for m in digit_matches:
        try:
            val = float(m.replace(",", ""))
            found.add(val)
        except ValueError:
            pass

    # 2. Extract spelled-out numbers
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    parsed_spoken = parse_spoken_number_phrase(words)
    for val in parsed_spoken:
        found.add(val)

    # 3. Currency / multipliers: e.g. "Rs 50,000", "50 thousand", "10 million"
    multipliers = {
        "hundred": 100, "thousand": 1000, "lakh": 100000, "million": 1000000,
        "crore": 10000000, "billion": 1000000000,
    }
    for mult_name, mult_val in multipliers.items():
        pattern = rf"\b(\d+(?:\.\d+)?)\s+{mult_name}\b"
        for m in re.finditer(pattern, norm_text, re.IGNORECASE):
            try:
                base = float(m.group(1))
                found.add(base * mult_val)
                found.add(base)
            except ValueError:
                pass

    return found


def find_number_spans_in_text(text: str) -> list[tuple[float, int, int]]:
    """Finds all numbers (digits and spelled words) along with their character span (start, end) in text."""
    spans = []

    # 1. Digits
    for m in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text):
        try:
            val = float(m.group(0).replace(",", ""))
            spans.append((val, m.start(), m.end()))
        except ValueError:
            pass

    # 2. Spoken word phrases
    token_matches = list(re.finditer(r"\b[a-zA-Z]+\b", text))
    i = 0
    while i < len(token_matches):
        tm = token_matches[i]
        tok = tm.group(0).lower()
        if tok in WORD_TO_NUMBER or tok == "point":
            start_pos = tm.start()
            phrase_tokens = []
            j = i
            while j < len(token_matches):
                w = token_matches[j].group(0).lower()
                if w in WORD_TO_NUMBER or w == "point":
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
    """Splits text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


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
    report: dict[str, Any] = {
        "validator_fingerprint": compute_validator_fingerprint(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "scene_results": {},
        "metrics": {},
    }

    if not draft_path.exists():
        msg = f"Draft script file not found at {draft_path}"
        errors.append(msg)
        report["errors"] = errors
        return False, errors, warnings, report

    try:
        with open(draft_path, "r", encoding="utf-8") as f:
            draft_data = json.load(f)
    except Exception as e:
        msg = f"Failed to parse draft JSON: {e}"
        errors.append(msg)
        report["errors"] = errors
        return False, errors, warnings, report

    # 1. Schema Validation against contracts/script.schema.json
    schema_path = Path("contracts/script.schema.json")
    if schema_path.exists():
        is_schema_valid, schema_err = validate_contract(draft_data, schema_path)
        if not is_schema_valid:
            errors.append(f"Schema validation failed: {schema_err}")

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
    sensitive_triggers = set(cfg.get("sensitive_triggers", []))

    scenes = draft_data.get("scenes", [])
    if len(scenes) != 7:
        errors.append(f"Expected exactly 7 scenes (intro + 5 stories + outro), got {len(scenes)}")

    total_words = 0
    wps = budget.get("words_per_second", 2.15)

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

        sc_errors = []
        sc_warnings = []

        # Headline checks
        hl_words = headline.strip().split()
        if len(hl_words) > 10:
            sc_errors.append(f"Headline exceeds 10 words ({len(hl_words)} words): '{headline}'")
        if headline.strip().endswith("."):
            sc_errors.append(f"Headline must not end with a period: '{headline}'")

        # Visual keywords count
        if not (3 <= len(visual_keywords) <= 5):
            sc_errors.append(f"Visual keywords count must be 3 to 5 (got {len(visual_keywords)})")

        # Visual keywords person name check
        for kw in visual_keywords:
            kw_words = kw.strip().split()
            if len(kw_words) >= 2 and all(w[0].isupper() for w in kw_words if w.isalpha()):
                if kw.lower() not in proper_noun_whitelist:
                    sc_warnings.append(f"Visual keyword '{kw}' looks like a proper name or brand")

        # Narration Word Counts
        narration_words = narration.strip().split()
        word_count = len(narration_words)
        total_words += word_count

        if sc_type == "intro":
            if not (15 <= word_count <= 40):
                sc_errors.append(f"Intro word count {word_count} outside 15-40 words")
        elif sc_type == "outro":
            if not (15 <= word_count <= 40):
                sc_errors.append(f"Outro word count {word_count} outside 15-40 words")
        else:
            if not (20 <= word_count <= 60):
                sc_errors.append(f"Story scene word count {word_count} outside 20-60 words")

        # Sentence count for story scenes: 2 to 4 sentences
        sentences = split_sentences(narration)
        if sc_type == "story" and not (2 <= len(sentences) <= 4):
            sc_errors.append(f"Story narration must have 2 to 4 sentences (got {len(sentences)})")

        # TTS Rules
        if re.search(r"\d", narration):
            digits_found = re.findall(r"\d+", narration)
            sc_errors.append(f"Digits found in narration (spell numbers out): {digits_found}")

        banned_symbols = re.findall(r"[%$&/\\@#^*+=<>~]", narration)
        if banned_symbols:
            sc_errors.append(f"Banned symbols in narration: {set(banned_symbols)}")

        if re.search(r"[()\[\]{}]", narration):
            sc_errors.append("Parentheses or brackets found in narration")

        # Non-ASCII / emoji check
        non_ascii = [ch for ch in narration if ord(ch) > 127]
        if non_ascii:
            sc_errors.append(f"Non-ASCII characters or emojis found: {set(non_ascii)}")

        # All-caps words (>1 letter) warning
        all_caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", narration)]
        if all_caps:
            sc_warnings.append(f"All-caps words found: {all_caps}")

        # Grounding & cluster checks for story scenes
        if sc_type == "story":
            if not source:
                sc_errors.append("Missing source outlet")
            if not source_urls:
                sc_errors.append("Missing source_urls")
            if not evidence:
                sc_errors.append("Missing evidence mappings")

            # Validate each evidence item explicitly against the shortlist
            for ev in evidence:
                ev_idx = ev.get("sentence_index", -1)
                ev_url = ev.get("source_url", "")
                if ev_idx < 0 or ev_idx >= len(sentences):
                    sc_errors.append(f"Evidence sentence_index {ev_idx} out of range [0, {len(sentences)-1}]")
                if ev_url not in cluster_by_url:
                    sc_errors.append(f"Evidence source_url '{ev_url}' does not belong to any shortlisted cluster")

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
                cluster_texts = [f"{a.get('title', '')} {a.get('full_text', '')}" for a in cluster.get("articles", [])]
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

                # Number verification: numbers in narration must appear in cluster text
                cluster_numbers = extract_numbers_from_text(cluster_combined_text)

                for sent in sentences:
                    sent_spans = find_number_spans_in_text(sent)
                    for num, start_idx, end_idx in sent_spans:
                        # Allow 0 or 1 for general speech
                        if num in (0.0, 1.0):
                            continue

                        # Check exact match in cluster (with float tolerance 1e-5)
                        is_exact = any(abs(num - c) < 1e-5 for c in cluster_numbers)
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

                # Proper nouns warning check
                cluster_text_lower = cluster_combined_text.lower()
                for sent in sentences:
                    sent_words = sent.strip().split()
                    for w_idx, w in enumerate(sent_words):
                        if w_idx == 0:
                            continue  # First word of sentence is capitalized by grammar
                        clean_w = re.sub(r"[^a-zA-Z]", "", w)
                        if clean_w and clean_w[0].isupper() and clean_w.lower() not in proper_noun_whitelist:
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

        errors.extend(sc_errors)
        warnings.extend(sc_warnings)
        report["scene_results"][sc_id] = {
            "errors": sc_errors,
            "warnings": sc_warnings,
            "word_count": word_count,
            "est_duration_sec": round(word_count / wps, 1),
        }

    # Total word count budget check (10% tolerance)
    min_budget = budget.get("min_total_words", 250)
    max_budget = budget.get("max_total_words", 380)
    if not (min_budget <= total_words <= max_budget):
        errors.append(f"Total word count {total_words} outside budget range [{min_budget}, {max_budget}]")

    report["metrics"] = {
        "total_words": total_words,
        "words_per_second": wps,
        "total_est_duration_sec": round(total_words / wps, 1),
        "scene_count": len(scenes),
    }

    is_valid = (len(errors) == 0)
    report["is_valid"] = is_valid
    report["errors"] = errors
    report["warnings"] = warnings

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

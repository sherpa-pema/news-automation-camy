"""Standalone LLM broadcast script writer for zero-Antigravity operation.

Reads `<run_dir>/script/editor_brief.md` and writes `draft_script.json` and
`selection.json`, then runs `EditorAgent.validate_draft()` with up to 5 rounds
of surgical scene-level (or full-script fallback) self-correction on validation errors.

Supported providers (configured via environment variables or `.env`):
  - Zhipu/BigModel: ZHIPU_API_KEY (or BIGMODEL_API_KEY; model ZHIPU_MODEL, default glm-5.3-flash)
  - Gemini:         GEMINI_API_KEY (optional: GEMINI_MODEL, default gemini-2.5-flash)
  - Anthropic:      ANTHROPIC_API_KEY (optional: ANTHROPIC_MODEL, default claude-sonnet-4-5)
  - OpenAI:         OPENAI_API_KEY (optional: OPENAI_MODEL, default gpt-4o-mini)
  - Ollama:         OLLAMA_MODEL (optional: OLLAMA_URL, default http://localhost:11434)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
import requests


def detect_configured_provider(preferred: str | None = None) -> tuple[str | None, str | None]:
    """Detects an available LLM provider and model from env/.env."""
    load_dotenv(override=False)
    pref = preferred.strip().lower() if preferred else None
    order = [pref] if pref else ["zhipu", "gemini", "anthropic", "openai", "ollama"]
    for p in order:
        if p in ("zhipu", "bigmodel") and (os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")):
            return "zhipu", os.getenv("ZHIPU_MODEL", "glm-5.3-flash")
        if p == "gemini" and os.getenv("GEMINI_API_KEY"):
            return "gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if p == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        if p == "openai" and os.getenv("OPENAI_API_KEY"):
            return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if p == "ollama" and os.getenv("OLLAMA_MODEL"):
            return "ollama", os.getenv("OLLAMA_MODEL")
    return None, None


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extracts a JSON object from raw model output (stripping markdown fences)."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        end = cleaned.rfind("}")
        if end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed

    return json.loads(cleaned)


def _call_llm(provider: str, model: str, messages: list[dict[str, str]], timeout: int = 180) -> str:
    """Calls the configured LLM provider via standard HTTPS REST API with exponential backoff retries."""
    max_retries = 3
    base_backoff = 2.0
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            if provider == "zhipu":
                api_key = os.getenv("ZHIPU_API_KEY") or os.environ["BIGMODEL_API_KEY"]
                base_url = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
                thinking_type = os.getenv("ZHIPU_THINKING", "enabled")
                reasoning_effort = os.getenv("ZHIPU_REASONING_EFFORT", "low")
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 16384,
                    "thinking": {"type": thinking_type},
                    "reasoning_effort": reasoning_effort,
                }
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"Zhipu API error {resp.status_code}: {resp.text}")
                data = resp.json()
                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                text = str(content).strip()
                if not text:
                    raise RuntimeError(
                        f"Zhipu returned empty content (finish_reason={choice.get('finish_reason')}, "
                        f"usage={data.get('usage')})"
                    )
                return text

            if provider == "gemini":
                api_key = os.environ["GEMINI_API_KEY"]
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                contents = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                resp = requests.post(
                    url,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "contents": contents,
                        "generationConfig": {
                            "temperature": 0.2,
                            "responseMimeType": "application/json",
                        },
                    },
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            if provider == "anthropic":
                api_key = os.environ["ANTHROPIC_API_KEY"]
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 16384,
                        "temperature": 0.2,
                        "messages": messages,
                    },
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return "".join(block.get("text", "") for block in data.get("content", []))

            if provider == "openai":
                api_key = os.environ["OPENAI_API_KEY"]
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                        "messages": messages,
                    },
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            if provider == "ollama":
                base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
                resp = requests.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "format": "json",
                        "stream": False,
                        "messages": messages,
                    },
                    timeout=timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(base_backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"]

            raise ValueError(f"Unsupported LLM provider: {provider}")
        except (requests.RequestException, RuntimeError) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(base_backoff * (2 ** attempt))
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("Failed to call LLM after retries")


def _load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


_EXCLUDED_SECTIONS_RE = re.compile(
    r"\n---\s*\n##\s*[56]\.\s*Excluded\s+(?:Opinion|Stale)[\s\S]*$",
    re.IGNORECASE,
)
_DETAILS_BLOCK_RE = re.compile(
    r"(\n[ \t]*\*Summary\*:[\s\S]*?)\n[ \t]*<details><summary>Click to view extracted full text</summary>[\s\S]*?</details>",
    re.IGNORECASE,
)


def _compact_brief_for_llm(brief_text: str) -> str:
    """Strips excluded sections and redundant full-text <details> blocks to cut LLM prompt tokens."""
    if not brief_text:
        return brief_text

    # 1. Remove Sections 5 & 6 (Excluded Opinion / Stale Previews) which the writer must never use
    compacted = _EXCLUDED_SECTIONS_RE.sub("", brief_text)

    # 2. Collapse redundant <details> full_text blocks whenever *Summary* already carries >= 40 chars
    def _collapse_details(match: re.Match[str]) -> str:
        summary_line = match.group(1)
        summary_body = re.sub(r"\*Summary\*:|\*\*\[NEPALI[^\]]*\]\*\*", "", summary_line).strip()
        if len(summary_body) >= 40:
            return summary_line
        # Keep a short excerpt if summary was missing/very short
        block = match.group(0)
        code_match = re.search(r"```text\s*([\s\S]*?)```", block)
        if code_match:
            excerpt_words = code_match.group(1).strip().split()[:100]
            if excerpt_words:
                return f"{summary_line}\n    *Excerpt*: {' '.join(excerpt_words)}"
        return summary_line

    compacted = _DETAILS_BLOCK_RE.sub(_collapse_details, compacted)
    return compacted.strip()


def _is_word_budget_global_error(message: str) -> bool:
    low = (message or "").lower()
    return (
        "total narration word count" in low
        or "total word count" in low
        or "total words" in low
        or "word budget" in low
    )


def _build_surgical_repair_prompt(
    scenes: list[dict[str, Any]],
    failing_indices: list[int],
    structured_errors: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    budget: dict[str, Any],
    intro_info: dict[str, Any] | None = None,
) -> str:
    """Builds a targeted prompt containing only failing scene(s), their errors, matched cluster, and remaining budget."""
    locked_words = sum(
        len(scenes[i].get("narration", "").split())
        for i in range(len(scenes))
        if i not in failing_indices
    )
    global_min = int(budget.get("min_total_words", 250))
    global_max = int(budget.get("max_total_words", 380))
    rem_min = max(0, global_min - locked_words)
    rem_max = max(rem_min, global_max - locked_words)

    url_to_cluster: dict[str, dict[str, Any]] = {}
    for c in shortlist:
        for a in c.get("articles", []):
            url_to_cluster[a.get("url", "")] = c

    is_headlines_mode = len(scenes) >= 18
    narration_rule = (
        "- SINGLE-LINE HEADLINES ONLY: Story scene narration MUST be strictly ONE single sentence (10 to 25 words)."
        if is_headlines_mode
        else "- FOCUSED STORY FORMAT: Each story scene narration MUST have 2 to 3 clear spoken sentences (25 to 58 words), with evidence mapping every sentence_index (0, 1, ...) to a member source_url."
    )

    lines: list[str] = [
        "SURGICAL SCENE REPAIR: All other scenes in draft_script.json passed validation and are LOCKED.",
        "Fix ONLY the failing scene(s) below while keeping every locked scene untouched.",
        'Return a SINGLE valid JSON object (no markdown fences) with key "scenes": [...] containing ONLY the repaired scene object(s), preserving each scene\'s exact "id" and "type".',
        "",
        f"REMAINING GLOBAL WORD BUDGET across the {len(failing_indices)} failing scene(s): {rem_min} to {rem_max} total words (locked scenes already use {locked_words} words; total must stay within [{global_min}, {global_max}]).",
        "",
        "RULES REMINDER:",
        narration_rule,
        "- Narration: ZERO digits (0-9), ZERO %, $, &, /, (), [], or non-ASCII characters. Spell all numbers out.",
        "- Do NOT use blocked acronyms (e.g. NEA, NRB, NEPSE, CIAA, CIB, CAAN, NOC, KMC, TU, UML, RSP) unless expanded to full institution name.",
        "- Every number in narration must be grounded in the latest non-superseded source article of its matched cluster.",
        "- Never copy 8+ consecutive words from source text.",
        "- Visual keywords: 3 to 5 generic Nepal-appropriate stock terms (no blocked Western terms).",
        "- SENSITIVITY: If an error mentions 'sensitive is false' or the cluster text contains sensitive triggers (fatal, death, casualties, accident, suicide, rape, murder, arrest, etc.), you MUST set \"sensitive\": true on that scene object.",
        *(
            ["- CATEGORY SEQUENCE: Story scenes MUST follow the 6 sequential broadcast blocks: national_politics -> society_local -> culture_entertainment -> business_economics -> science_technology -> sports."]
            if is_headlines_mode
            else []
        ),
        "",
    ]

    SLOT_CATEGORIES = {
        1: "national_politics",
        2: "society_local",
        3: "culture_entertainment",
        4: "business_economics",
        5: "science_technology",
        6: "sports",
    }

    for idx in failing_indices:
        sc = scenes[idx]
        sc_id = sc.get("id", f"scene_{idx}")
        sc_type = sc.get("type", "story")
        if is_headlines_mode:
            sc_min, sc_max = (12, 25) if sc_type in ("intro", "outro") else (10, 25)
        else:
            sc_min, sc_max = (15, 38) if sc_type in ("intro", "outro") else (25, 58)
        if len(failing_indices) == 1:
            adj_min = max(sc_min, rem_min)
            adj_max = min(sc_max, rem_max)
            if adj_min <= adj_max:
                sc_min, sc_max = adj_min, adj_max

        sc_errs = [
            e.get("message", "")
            for e in structured_errors
            if e.get("scene_index") == idx
        ]

        # Resolve matched cluster
        matched_cluster = None
        for u in sc.get("source_urls", []):
            if u in url_to_cluster:
                matched_cluster = url_to_cluster[u]
                break
        if matched_cluster is None:
            for ev in sc.get("evidence", []):
                u = ev.get("source_url", "")
                if u in url_to_cluster:
                    matched_cluster = url_to_cluster[u]
                    break

        lines.append(f"=== FAILING SCENE index={idx}, id='{sc_id}', type='{sc_type}' ===")
        if (sc_type == "intro" or sc_id == "intro") and intro_info:
            lines.append(f"CATEGORY INTRO DIRECTIVE: {intro_info.get('prompt_instruction', '')}")
            lines.append(f"TARGET EVERGREEN INTRO: \"{intro_info.get('narration', '')}\"")
        elif sc.get("category"):
            lines.append(f"Category: '{sc.get('category')}'")
        elif is_headlines_mode and idx in SLOT_CATEGORIES:
            lines.append(f"Designated Broadcast Slot: '{SLOT_CATEGORIES[idx]}' (category MUST be '{SLOT_CATEGORIES[idx]}')")
        lines.append(f"Allowed word count for this scene: {sc_min}-{sc_max} words (currently {len(sc.get('narration', '').split())} words).")
        lines.append("Validation errors to fix:")
        for err in sc_errs:
            lines.append(f"  - {err}")
            if "sensitive is false" in err.lower():
                lines.append('    -> ACTION REQUIRED: You MUST set "sensitive": true on this scene object!')
        lines.append("Current scene JSON:")
        lines.append(json.dumps(sc, indent=2, ensure_ascii=False))

        if matched_cluster:
            lines.append(f"Matched cluster ({matched_cluster.get('cluster_id')}): {matched_cluster.get('primary_title')}")
            lines.append(f"Cluster URLs: {json.dumps(matched_cluster.get('urls', []))}")
            if matched_cluster.get("synthesis_notes"):
                lines.append(f"  * SYNTHESIS DIRECTIVE: {matched_cluster['synthesis_notes']}")
            if matched_cluster.get("conflicting_figures_resolution"):
                lines.append(f"  * CONFLICT RESOLUTION: {matched_cluster['conflicting_figures_resolution']}")
            for art in matched_cluster.get("articles", []):
                sup_note = " [OLDER UPDATE — DO NOT USE FOR NUMBERS]" if art.get("superseded") else ""
                lines.append(f"  * [{art.get('source')}] {art.get('title')} ({art.get('url')}){sup_note}")
                art_body = (art.get("full_text") or art.get("text_summary") or "").strip()
                art_words = art_body.split()
                if len(art_words) > 160:
                    art_body = " ".join(art_words[:160]) + " ... [TRUNCATED]"
                lines.append(f"    {art_body}")
        lines.append("")

    return "\n".join(lines)


def _extract_repaired_scenes(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(parsed.get("scenes"), list):
        return [s for s in parsed["scenes"] if isinstance(s, dict)]
    ds = parsed.get("draft_script")
    if isinstance(ds, dict) and isinstance(ds.get("scenes"), list):
        return [s for s in ds["scenes"] if isinstance(s, dict)]
    if isinstance(parsed.get("repaired_scenes"), list):
        return [s for s in parsed["repaired_scenes"] if isinstance(s, dict)]
    if "id" in parsed and "narration" in parsed:
        return [parsed]
    return []


def generate_and_validate_draft(
    run_dir: Path | str,
    *,
    provider: str | None = None,
    max_rounds: int = 5,
    categories: list[str] | str | None = None,
) -> dict[str, Any]:
    """Generates draft_script.json + selection.json via LLM and validates up to max_rounds."""
    from src.agents.editor import EditorAgent
    from src.utils.category_intro import resolve_category_intro
    from src.utils.editor_brief import format_spoken_date
    from src.utils.manifest import load_manifest
    from src.utils.office_events import emit_office_event

    rdir = Path(run_dir)
    script_dir = rdir / "script"
    brief_path = script_dir / "editor_brief.md"
    if not brief_path.exists():
        return {"ok": False, "error": f"Missing editor brief at {brief_path}. Run --morning first."}

    chosen_provider, chosen_model = detect_configured_provider(provider)
    if not chosen_provider or not chosen_model:
        return {
            "ok": False,
            "error": (
                "No standalone LLM provider configured in .env. Set one of: "
                "ZHIPU_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or OLLAMA_MODEL."
            ),
        }

    raw_brief_text = brief_path.read_text(encoding="utf-8")
    manifest_path = rdir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
    editor_agent = EditorAgent(rdir, editor_mode="llm_writer")
    shortlist = _load_json_if_exists(script_dir / "shortlist.json", [])
    budget = _load_json_if_exists(script_dir / "budget.json", {})
    focused_bulletin = 0 < len(shortlist) < 12
    brief_text = (
        _EXCLUDED_SECTIONS_RE.sub("", raw_brief_text).strip()
        if focused_bulletin
        else _compact_brief_for_llm(raw_brief_text)
    )

    eff_categories = (
        categories
        or manifest.get("selected_categories")
        or manifest.get("stages", {}).get("scraper", {}).get("metrics", {}).get("selected_categories")
    )
    spoken_date = format_spoken_date(manifest.get("date") or rdir.name)
    intro_info = resolve_category_intro(
        categories=eff_categories,
        spoken_date=spoken_date,
        custom_templates=editor_agent.editor_cfg.get("intro_templates"),
    )

    if focused_bulletin:
        system_prompt = (
            "You are a broadcast news writer for Nepal News Bot. Read the Editor Brief carefully and "
            "return a SINGLE valid JSON object (no markdown fences, no commentary outside JSON) with "
            "exactly two top-level keys:\n"
            '  "draft_script": { "language": "English", "origin": "editor", "run_kind": "...", "date": "YYYY-MM-DD", "scenes": [...] },\n'
            '  "selection": { "selected_clusters": [...], "skipped_clusters": [...] }\n'
            "CRITICAL RULES (FOCUSED 7-SCENE TOPIC BULLETIN):\n"
            f"1. Exactly 7 scenes: intro (id='intro', 15-35 words, announcing the broadcast category: \"{intro_info['narration']}\"), followed by 5 distinct story scenes (scene_01..scene_05, each from a different shortlisted cluster), and outro (id='outro', 15-35 words).\n"
            "2. STORY NARRATION: Each story scene narration MUST contain 2 to 3 crisp broadcast sentences (44 to 56 words per story scene) summarizing the verified facts from its cluster.\n"
            "3. Total spoken words across all 7 scenes MUST fall inside the allowed range in Section 1.\n"
            "4. Narration MUST contain ZERO digits (0-9), ZERO %, $, &, /, (), [], or non-ASCII characters. Spell all numbers out (e.g. 'forty percent', 'ten million').\n"
            "5. Every number/quantity mentioned in narration must exist in that story's source cluster (and use latest non-superseded update).\n"
            "6. Headlines: max 10 words, sentence case, NO ending period.\n"
            "7. visual_keywords: 3 to 5 generic Nepal-appropriate stock phrases per scene (no blocked Western terms).\n"
            "8. Story scenes: source_urls must come from the chosen cluster; evidence must map EVERY sentence_index (0, 1, ...) to a member article source_url; never copy 8+ consecutive words from source text; set sensitive: true if crime/death/fatal/accident/minor/children trigger applies.\n"
            "9. Categories: set each story scene's category to its topic (e.g. 'health' or 'society_local'); intro and outro categories MUST be null.\n"
            "10. TRANSLATION: Member articles marked NEPALI must be translated to English first. Narration stays ASCII-only (rule 4). Numbers must be transliterated from Devanagari (०-९) and must still trace to the Nepali source."
        )
    else:
        system_prompt = (
            "You are a broadcast news writer for Nepal News Bot. Read the Editor Brief carefully and "
            "return a SINGLE valid JSON object (no markdown fences, no commentary outside JSON) with "
            "exactly two top-level keys:\n"
            '  "draft_script": { "language": "English", "origin": "editor", "run_kind": "...", "date": "YYYY-MM-DD", "scenes": [...] },\n'
            '  "selection": { "selected_clusters": [...], "skipped_clusters": [...] }\n'
            "CRITICAL RULES (HEADLINES-ONLY BULLETIN):\n"
            f"1. Exactly 20 to 26 scenes: intro (12-25 words, announcing the broadcast category: \"{intro_info['narration']}\"), followed by 18 to 24 story headline scenes, and outro (12-25 words).\n"
            "2. SINGLE-LINE HEADLINES ONLY: Every story scene narration MUST contain strictly ONE single spoken sentence (10 to 22 words, target ~15 words) summarizing the core event. Absolutely NO descriptions, background paragraphs, historical context, or secondary sentences.\n"
            "3. Total spoken words across all scenes MUST fall inside the allowed range in Section 1.\n"
            "4. Narration MUST contain ZERO digits (0-9), ZERO %, $, &, /, (), [], or non-ASCII characters. Spell all numbers out.\n"
            "5. Every number/quantity mentioned in narration must exist in that story's source cluster (and use latest non-superseded update).\n"
            "6. Headlines: max 10 words, sentence case, NO ending period.\n"
            "7. visual_keywords: 3 to 5 generic Nepal-appropriate stock phrases per scene.\n"
            "8. Story scenes: source_urls must come from the chosen cluster; evidence must map sentence_index 0 to a member article source_url; never copy 8+ consecutive words from source text; set sensitive: true if crime/death/fatal/accident/minor trigger applies.\n"
            "9. STRICT SEQUENTIAL CATEGORY ORDER (3 to 4 single-line headlines per category block):\n"
            "   - Block 1: \"national_politics\" (3 to 4 scenes) - National Politics\n"
            "   - Block 2: \"society_local\" (3 to 4 scenes) - Society and Local News\n"
            "   - Block 3: \"culture_entertainment\" (3 to 4 scenes) - Culture and Entertainment\n"
            "   - Block 4: \"business_economics\" (3 to 4 scenes) - Business and Economics\n"
            "   - Block 5: \"science_technology\" (3 to 4 scenes) - Science and Technology\n"
            "   - Block 6: \"sports\" (3 to 4 scenes) - Sports\n"
            "   - All scenes in a category block must be grouped together before moving to the next category block.\n"
            "   - intro and outro categories MUST be null.\n"
            "10. MULTI-SOURCE SYNTHESIS: When a chosen cluster includes multiple member outlets, synthesize key facts into the single summary line. Cite the primary supporting source_url in evidence.\n"
            "11. TRANSLATION: Member articles marked NEPALI must be translated to English first. Narration stays ASCII-only (rule 4). Numbers must be transliterated from Devanagari (०-९) and must still trace to the Nepali source. Never paraphrase a translation into a different claim."
        )

    initial_user_message = {"role": "user", "content": f"{system_prompt}\n\n---\n\n{brief_text}"}
    full_messages: list[dict[str, str]] = [initial_user_message]

    last_errors: list[str] = []
    mode = "full"  # "full" or "surgical"
    repair_messages: list[dict[str, str]] = []
    failing_indices: list[int] = []
    current_draft_obj: dict[str, Any] = {}
    surgical_rounds = 0

    for round_idx in range(1, max_rounds + 1):
        emit_office_event(
            rdir,
            "substage_progress",
            stage="editor",
            current=round_idx,
            total=max_rounds,
            message=f"AI Writer ({chosen_provider}:{chosen_model}) round {round_idx}/{max_rounds} ({mode})...",
        )

        try:
            if mode == "surgical" and repair_messages and current_draft_obj.get("scenes"):
                surgical_rounds += 1
                raw_reply = _call_llm(chosen_provider, chosen_model, repair_messages)
                parsed = _extract_json_object(raw_reply)
                repaired_scenes = _extract_repaired_scenes(parsed)
                scenes_list = list(current_draft_obj["scenes"])
                id_to_f_idx = {
                    scenes_list[idx].get("id"): idx
                    for idx in failing_indices
                    if 0 <= idx < len(scenes_list) and scenes_list[idx].get("id")
                }
                for pos, rep_sc in enumerate(repaired_scenes):
                    rep_id = rep_sc.get("id")
                    target_idx: int | None = None
                    if rep_id in id_to_f_idx:
                        target_idx = id_to_f_idx[rep_id]
                    elif len(repaired_scenes) == len(failing_indices):
                        target_idx = failing_indices[pos]
                    if target_idx is not None and 0 <= target_idx < len(scenes_list):
                        orig_sc = scenes_list[target_idx]
                        merged_sc = dict(rep_sc)
                        merged_sc.setdefault("id", orig_sc.get("id"))
                        merged_sc.setdefault("type", orig_sc.get("type"))
                        target_errs = [
                            e.get("message", "")
                            for e in structured_errors
                            if e.get("scene_index") == target_idx
                        ]
                        if any("sensitive is false" in e.lower() for e in target_errs):
                            merged_sc["sensitive"] = True
                        scenes_list[target_idx] = merged_sc

                current_draft_obj["scenes"] = scenes_list
                (script_dir / "draft_script.json").write_text(
                    json.dumps(current_draft_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                raw_reply = _call_llm(chosen_provider, chosen_model, full_messages)
                parsed = _extract_json_object(raw_reply)
                current_draft_obj = parsed.get("draft_script") or parsed
                selection_obj = parsed.get("selection") or {"selected_clusters": [], "skipped_clusters": []}

                (script_dir / "draft_script.json").write_text(
                    json.dumps(current_draft_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (script_dir / "selection.json").write_text(
                    json.dumps(selection_obj, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning(f"Round {round_idx} LLM call or JSON extraction failed: {e}")
            if round_idx < max_rounds:
                retry_msg = f"Your previous response caused an error: {e}. Please respond strictly with a SINGLE valid JSON object."
                if mode == "surgical":
                    repair_messages.append({"role": "user", "content": retry_msg})
                else:
                    full_messages.append({"role": "user", "content": retry_msg})
                continue
            raise

        # Reload manifest before validate
        manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
        editor_agent.validate_draft(manifest)

        rep_path = script_dir / "validation_report.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
        if rep.get("is_valid"):
            return {
                "ok": True,
                "provider": chosen_provider,
                "model": chosen_model,
                "rounds": round_idx,
                "surgical_rounds": surgical_rounds,
                "warnings": rep.get("warnings", []),
                "metrics": rep.get("metrics", {}),
            }

        # Deterministic zero-latency pre-patch for missing sensitive=true flags
        structured_errors = rep.get("structured_errors") or []
        scenes_list = current_draft_obj.get("scenes") or []
        sensitive_patched = False
        for err in structured_errors:
            s_idx = err.get("scene_index")
            if s_idx is not None and 0 <= int(s_idx) < len(scenes_list):
                if "sensitive is false" in str(err.get("message", "")).lower():
                    if not scenes_list[int(s_idx)].get("sensitive", False):
                        scenes_list[int(s_idx)]["sensitive"] = True
                        sensitive_patched = True

        if sensitive_patched:
            current_draft_obj["scenes"] = scenes_list
            (script_dir / "draft_script.json").write_text(
                json.dumps(current_draft_obj, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
            editor_agent.validate_draft(manifest)
            rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
            if rep.get("is_valid"):
                return {
                    "ok": True,
                    "provider": chosen_provider,
                    "model": chosen_model,
                    "rounds": round_idx,
                    "surgical_rounds": surgical_rounds,
                    "warnings": rep.get("warnings", []),
                    "metrics": rep.get("metrics", {}),
                }
            structured_errors = rep.get("structured_errors") or []

        last_errors = rep.get("errors", ["Unknown validation failure"])
        round_diag = {
            "round": round_idx,
            "mode": mode,
            "provider": chosen_provider,
            "model": chosen_model,
            "is_valid": rep.get("is_valid", False),
            "errors": rep.get("errors", []),
            "structured_errors": rep.get("structured_errors", []),
            "warnings": rep.get("warnings", []),
            "metrics": rep.get("metrics", {}),
        }
        (script_dir / f"retry_round_{round_idx}.json").write_text(
            json.dumps(round_diag, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if round_idx == max_rounds:
            break

        # Determine whether errors are isolated to specific scene(s) or global
        scenes_list = current_draft_obj.get("scenes") or []
        global_errors = [e for e in structured_errors if e.get("scene_index") is None]
        non_budget_global_errors = [
            e for e in global_errors if not _is_word_budget_global_error(str(e.get("message", "")))
        ]
        budget_only_global = bool(global_errors) and not non_budget_global_errors
        failing_indices = sorted(
            {int(e["scene_index"]) for e in structured_errors if e.get("scene_index") is not None}
        )

        # If ONLY global total word count failed (all individual scenes passed), select 2-3 story scenes to adjust surgically
        if budget_only_global and not failing_indices and scenes_list:
            total_w = int((rep.get("metrics") or {}).get("total_words") or sum(len(s.get("narration", "").split()) for s in scenes_list))
            g_min = int(budget.get("min_total_words", 250))
            g_max = int(budget.get("max_total_words", 380))
            story_wcs = [
                (len(sc.get("narration", "").split()), i)
                for i, sc in enumerate(scenes_list)
                if sc.get("type") == "story"
            ]
            if story_wcs:
                if total_w > g_max:
                    story_wcs.sort(reverse=True)
                    picked = [i for _, i in story_wcs[:3]]
                    adj_msg = f"Global word count ({total_w}) exceeds max ({g_max}); tighten this scene slightly while preserving facts."
                else:
                    story_wcs.sort()
                    picked = [i for _, i in story_wcs[:3]]
                    adj_msg = f"Global word count ({total_w}) is below min ({g_min}); expand this scene slightly within its single-sentence limit."
                failing_indices = sorted(picked)
                structured_errors = list(structured_errors) + [
                    {"scene_index": idx, "scene_id": scenes_list[idx].get("id"), "message": adj_msg}
                    for idx in failing_indices
                ]

        # Check story selection diversity when enough clusters exist in shortlist
        diversity_ok = True
        if len(shortlist) >= 5 and (len(scenes_list) in (7, 8) or 14 <= len(scenes_list) <= 26):
            url_to_cid = {
                a.get("url", ""): c.get("cluster_id")
                for c in shortlist
                for a in c.get("articles", [])
            }
            story_cids: list[str] = []
            for sc in scenes_list:
                if sc.get("type") == "story":
                    cid = next((url_to_cid[u] for u in sc.get("source_urls", []) if u in url_to_cid), None)
                    if cid:
                        story_cids.append(cid)
            story_count = sum(1 for sc in scenes_list if sc.get("type") == "story")
            if len(story_cids) == story_count and len(set(story_cids)) < story_count:
                diversity_ok = False

        can_surgical = (
            (len(scenes_list) in (7, 8) or 14 <= len(scenes_list) <= 26)
            and not non_budget_global_errors
            and diversity_ok
            and 0 < len(failing_indices) < len(scenes_list)
        )

        if can_surgical:
            mode = "surgical"
            repair_prompt = _build_surgical_repair_prompt(
                scenes=scenes_list,
                failing_indices=failing_indices,
                structured_errors=structured_errors,
                shortlist=shortlist,
                budget=budget,
                intro_info=intro_info,
            )
            repair_messages = [{"role": "user", "content": repair_prompt}]
        else:
            mode = "full"
            scene_summary = []
            for sid, sres in (rep.get("scene_results") or {}).items():
                wc = sres.get("word_count")
                serrs = sres.get("errors") or []
                if serrs or (wc is not None):
                    scene_summary.append(f"- {sid}: {wc} words; errors={serrs}")
            metrics_info = rep.get("metrics") or {}
            g_min = int(budget.get("min_total_words", 250))
            g_max = int(budget.get("max_total_words", 380))
            per_scene_hint = (
                "keep each story scene strictly ONE sentence, 10-22 words"
                if len(scenes_list) >= 14 or len(shortlist) >= 12
                else "keep each scene strictly within its allowed word range"
            )
            feedback_parts = [
                "Validation failed with these exact errors. Fix ALL of them and return ONLY the full corrected JSON object with 'draft_script' and 'selection':",
                *[f"- {e}" for e in last_errors],
                "",
                f"Total words: {metrics_info.get('total_words')} (allowed range: {g_min}-{g_max}; {per_scene_hint}).",
                "Per-scene breakdown:",
                *scene_summary,
                "",
                "For any 'Verbatim 8-word overlap' error, rewrite that entire sentence using completely different phrasing and word order.",
            ]
            if not diversity_ok:
                feedback_parts.append("IMPORTANT: Ensure all story scenes come from DISTINCT shortlisted clusters.")
            if any("sensitive is false" in e.lower() for e in last_errors):
                feedback_parts.append(
                    "CRITICAL: If an error states 'Cluster text contains sensitive trigger word ..., but sensitive is false', you MUST set \"sensitive\": true on that scene object."
                )
            # Keep only initial brief + latest draft + latest feedback to prevent token snowballing across retries
            full_messages = [
                initial_user_message,
                {"role": "assistant", "content": raw_reply},
                {"role": "user", "content": "\n".join(feedback_parts)},
            ]

    return {
        "ok": False,
        "provider": chosen_provider,
        "model": chosen_model,
        "rounds": max_rounds,
        "surgical_rounds": surgical_rounds,
        "error": f"Draft failed validation after {max_rounds} attempts.",
        "validation_errors": last_errors,
    }

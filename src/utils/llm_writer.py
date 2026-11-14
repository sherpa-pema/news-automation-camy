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
    order = [preferred] if preferred else ["zhipu", "gemini", "anthropic", "openai", "ollama"]
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
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _call_llm(provider: str, model: str, messages: list[dict[str, str]], timeout: int = 180) -> str:
    """Calls the configured LLM provider via standard HTTPS REST API."""
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
        resp.raise_for_status()
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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        resp = requests.post(
            url,
            json={
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout,
        )
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
                "max_tokens": 4096,
                "temperature": 0.2,
                "messages": messages,
            },
            timeout=timeout,
        )
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
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _build_surgical_repair_prompt(
    scenes: list[dict[str, Any]],
    failing_indices: list[int],
    structured_errors: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    budget: dict[str, Any],
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

    lines: list[str] = [
        "SURGICAL SCENE REPAIR: All other scenes in draft_script.json passed validation and are LOCKED.",
        "Fix ONLY the failing scene(s) below while keeping every locked scene untouched.",
        'Return a SINGLE valid JSON object (no markdown fences) with key "scenes": [...] containing ONLY the repaired scene object(s), preserving each scene\'s exact "id" and "type".',
        "",
        f"REMAINING GLOBAL WORD BUDGET across the {len(failing_indices)} failing scene(s): {rem_min} to {rem_max} total words (locked scenes already use {locked_words} words; total must stay within [{global_min}, {global_max}]).",
        "",
        "RULES REMINDER:",
        "- Narration: ZERO digits (0-9), ZERO %, $, &, /, (), [], or non-ASCII characters. Spell all numbers out.",
        "- Do NOT use blocked acronyms (e.g. NEA, NRB, NEPSE, CIAA, CIB, CAAN, NOC, KMC, TU, UML, RSP) unless expanded to full institution name.",
        "- Every number in narration must be grounded in the latest non-superseded source article of its matched cluster.",
        "- Never copy 8+ consecutive words from source text.",
        "- Visual keywords: 3 to 5 generic Nepal-appropriate stock terms (no blocked Western terms).",
        "- SENSITIVITY: If an error mentions 'sensitive is false' or the cluster text contains sensitive triggers (fatal, death, casualties, accident, suicide, rape, murder, arrest, etc.), you MUST set \"sensitive\": true on that scene object.",
        "",
    ]

    for idx in failing_indices:
        sc = scenes[idx]
        sc_id = sc.get("id", f"scene_{idx}")
        sc_type = sc.get("type", "story")
        sc_min, sc_max = (15, 40) if sc_type in ("intro", "outro") else (20, 60)
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
                lines.append(f"    {art.get('full_text') or art.get('text_summary') or ''}")
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
) -> dict[str, Any]:
    """Generates draft_script.json + selection.json via LLM and validates up to max_rounds."""
    from src.agents.editor import EditorAgent
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

    brief_text = brief_path.read_text(encoding="utf-8")
    manifest_path = rdir / "manifest.json"
    manifest = load_manifest(manifest_path) if manifest_path.exists() else {"stages": {}}
    editor_agent = EditorAgent(rdir)
    shortlist = _load_json_if_exists(script_dir / "shortlist.json", [])
    budget = _load_json_if_exists(script_dir / "budget.json", {})

    system_prompt = (
        "You are a broadcast news writer for Nepal News Bot. Read the Editor Brief carefully and "
        "return a SINGLE valid JSON object (no markdown fences, no commentary outside JSON) with "
        "exactly two top-level keys:\n"
        '  "draft_script": { "language": "English", "origin": "editor", "run_kind": "...", "date": "YYYY-MM-DD", "scenes": [...] },\n'
        '  "selection": { "selected_clusters": [...], "skipped_clusters": [...] }\n'
        "CRITICAL RULES:\n"
        "1. Exactly 7 scenes: intro (15-40 words), scene_01..scene_05 (each 20-60 words, 2-4 sentences), outro (15-40 words).\n"
        "2. Total spoken words across all 7 scenes MUST fall inside the allowed range in Section 1.\n"
        "3. Narration MUST contain ZERO digits (0-9), ZERO %, $, &, /, (), [], or non-ASCII characters. Spell all numbers out.\n"
        "4. Every number/quantity mentioned in narration must exist in that story's source cluster (and use latest non-superseded update).\n"
        "5. Headlines: max 10 words, sentence case, NO ending period.\n"
        "6. visual_keywords: 3 to 5 generic Nepal-appropriate stock phrases per scene.\n"
        "7. Story scenes: source_urls must come from the chosen cluster; evidence must map EVERY sentence_index (0..N-1) to a member article source_url; never copy 8+ consecutive words from source text; set sensitive: true if crime/death/fatal/accident/minor trigger applies.\n"
        "8. Story category MUST be strictly one of: \"politics\", \"economy\", \"weather\", \"health\", \"transport\", \"crime\", \"disaster\", \"environment\", \"society\", \"technology\", \"tourism\", \"other\" (e.g. use \"other\" for sports, \"environment\" for climate, \"society\" for diaspora/community; intro and outro must use null).\n"
        "9. MULTI-SOURCE SYNTHESIS: When a chosen cluster includes multiple member outlets, do NOT summarize just one article. Actively combine complementary details across sentences (e.g. casualty figures from one outlet, vehicle/route details from another, and hospital/rescue status from a third). Every sentence must cite its specific source_url in evidence."
    )

    full_messages: list[dict[str, str]] = [
        {"role": "user", "content": f"{system_prompt}\n\n---\n\n{brief_text}"}
    ]

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

        last_errors = rep.get("errors", ["Unknown validation failure"])
        if round_idx == max_rounds:
            break

        # Determine whether errors are isolated to specific scene(s) or global
        structured_errors = rep.get("structured_errors") or []
        scenes_list = current_draft_obj.get("scenes") or []
        global_errors = [e for e in structured_errors if e.get("scene_index") is None]
        failing_indices = sorted(
            {int(e["scene_index"]) for e in structured_errors if e.get("scene_index") is not None}
        )

        # Check story selection diversity when enough clusters exist in shortlist
        diversity_ok = True
        if len(shortlist) >= 5 and len(scenes_list) == 7:
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
            len(scenes_list) == 7
            and not global_errors
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
            feedback_parts = [
                "Validation failed with these exact errors. Fix ALL of them and return ONLY the full corrected JSON object with 'draft_script' and 'selection':",
                *[f"- {e}" for e in last_errors],
                "",
                f"Total words: {metrics_info.get('total_words')} (aim for middle of budget range; keep story scenes 40-54 words so they never hit 60!)",
                "Per-scene breakdown:",
                *scene_summary,
                "",
                "For any 'Verbatim 8-word overlap' error, rewrite that entire sentence using completely different phrasing and word order.",
            ]
            if not diversity_ok:
                feedback_parts.append("IMPORTANT: Ensure all 5 story scenes come from 5 DISTINCT shortlisted clusters.")
            if any("sensitive is false" in e.lower() for e in last_errors):
                feedback_parts.append(
                    "CRITICAL: If an error states 'Cluster text contains sensitive trigger word ..., but sensitive is false', you MUST set \"sensitive\": true on that scene object."
                )
            full_messages.append({"role": "assistant", "content": raw_reply})
            full_messages.append({"role": "user", "content": "\n".join(feedback_parts)})

    return {
        "ok": False,
        "provider": chosen_provider,
        "model": chosen_model,
        "rounds": max_rounds,
        "surgical_rounds": surgical_rounds,
        "error": f"Draft failed validation after {max_rounds} attempts.",
        "validation_errors": last_errors,
    }

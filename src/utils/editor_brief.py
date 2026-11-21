"""Generates the editor brief markdown document for the Antigravity agent."""
from datetime import datetime
import json
from typing import Any

from src.utils.category_intro import normalize_selected_categories, resolve_category_intro



def format_spoken_date(date_str: str) -> str:
    """Formats YYYY-MM-DD into natural spoken words (e.g. Saturday, November 7, 2026 / Kartik 22, 2083 BS)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        ad_str = d.strftime("%A, %B %d, %Y").replace(" 0", " ")
        try:
            from src.utils.nepali_date import ad_to_bs, format_bs_date
            bs_d = ad_to_bs(d.year, d.month, d.day)
            bs_str = format_bs_date(bs_d.year, bs_d.month, bs_d.day, lang="en")
            return f"{ad_str} ({bs_str} BS)"
        except Exception:
            return ad_str
    except Exception:
        return date_str


def generate_editor_brief(
    run_date: str,
    budget: dict[str, Any],
    calib_msg: str,
    shortlist: list[dict[str, Any]],
    opinion_clusters: list[dict[str, Any]] | None = None,
    stale_preview_clusters: list[dict[str, Any]] | None = None,
    run_kind: str = "test",
    mode: str = "headlines",
    selected_categories: list[str] | str | None = None,
    custom_templates: dict[str, Any] | None = None,
) -> str:
    """Assembles the comprehensive editor brief markdown document."""
    spoken_date = format_spoken_date(run_date)
    wps = budget["words_per_second"]
    sps = budget.get("syllables_per_second", 3.3)
    target_words = budget["target_total_words"]
    min_words = budget["min_total_words"]
    max_words = budget["max_total_words"]
    is_headlines = (mode == "headlines" or len(shortlist) >= 12)

    intro_info = resolve_category_intro(
        categories=selected_categories,
        spoken_date=spoken_date,
        custom_templates=custom_templates,
    )

    lines = []
    lines.append(f"# Broadcast Editor Brief - {spoken_date}")
    lines.append("")
    lines.append(f"- **Run Date**: `{run_date}` | **Run Kind**: `{run_kind}`")
    norm_cats = normalize_selected_categories(selected_categories)
    if norm_cats and "all" not in norm_cats:
        lines.append(f"- **Selected Category Focus**: `{intro_info['category_label']}` (`{', '.join(norm_cats)}`)")
        lines.append(f"- **Intro Directive**: {intro_info['prompt_instruction']}")
    lines.append("")

    is_degraded = any(
        a.get("degraded_run")
        for c in shortlist
        for a in c.get("articles", [])
    ) or bool(budget.get("degraded"))
    if is_degraded:
        lines.append("> **DEGRADED INGEST**: Bulletin ingested under degraded conditions. Proceeding with available stories.")
        lines.append("")

    lines.append("> **Role**: You (the Antigravity agent) are acting as the Broadcast News Writer.")
    lines.append("> There are no external LLM calls or API keys. You will review the shortlist below,")
    lines.append("> select the top news clusters, write `draft_script.json` and `selection.json`,")
    lines.append("> and run deterministic validation.")
    lines.append("")
    lines.append("---")
    lines.append("## 1. Word Budget & Speech Rate Calibration")
    lines.append(f"- **Calibration**: {calib_msg}")
    lines.append(f"- **Speech Rate**: `{wps:.2f}` words per second (`{sps:.2f}` syllables per second)")
    lines.append(f"- **Target Video Duration**: 2:00 to 3:00 (120 to 180 seconds, target {budget['target_duration_sec']:.0f}s)")
    lines.append(f"- **Total Word Budget**: **{target_words} words** (allowed range: **{min_words} to {max_words} words**)")
    if is_headlines:
        lines.append("- **Scene Breakdown** (Headlines Bulletin: 20 to 26 scenes total, 3 to 4 single-line summarized headlines per category in exact sequential order):")
        lines.append("  - **1 Intro scene**: 12 to 25 words (target ~18 words)")
        lines.append("  - **18 to 24 Story scenes** (strictly 1 single spoken sentence per scene, 10 to 22 words each, target ~15 words each):")
        lines.append("    * Block 1: **National Politics** (`national_politics`) — 3 to 4 single-line headlines")
        lines.append("    * Block 2: **Society and Local News** (`society_local`) — 3 to 4 single-line headlines")
        lines.append("    * Block 3: **Culture and Entertainment** (`culture_entertainment`) — 3 to 4 single-line headlines")
        lines.append("    * Block 4: **Business and Economics** (`business_economics`) — 3 to 4 single-line headlines")
        lines.append("    * Block 5: **Science and Technology** (`science_technology`) — 3 to 4 single-line headlines")
        lines.append("    * Block 6: **Sports** (`sports`) — 3 to 4 single-line headlines")
        lines.append("  - **1 Outro scene**: 12 to 25 words (target ~18 words)")
    else:
        lines.append("- **Scene Breakdown** (Focused Topic Bulletin: 7 scenes total):")
        lines.append("  - **1 Intro scene**: 15 to 35 words (target ~25 words)")
        lines.append("  - **5 Story scenes** (2 to 3 broadcast sentences per scene, 44 to 56 words each):")
        lines.append("  - **1 Outro scene**: 15 to 35 words (target ~25 words)")
    lines.append("")
    lines.append("---")
    lines.append("## 2. Mandatory Writing Rules")
    lines.append("### Ground Truth & Attribution")
    lines.append("1. **Strictly Grounded**: Use ONLY facts present in the cluster's articles. No invented details, figures, context, or quotes. If a fact is unclear or missing, leave it out.")
    lines.append("2. **Complete Rewrite**: Never copy article sentences. Any run of 8 or more consecutive words shared with raw article text is strictly banned.")
    lines.append("3. **Attribution**: Explicitly attribute claims ('officials said', 'the Ministry of Finance reported').")
    lines.append("4. **Sensitivity & Protection**: Crime, fatal accidents, deaths, abuse, or stories involving minors MUST set `sensitive: true`. Never name minors, victims, or unconvicted suspects. No graphic details.")
    lines.append("5. **Temporal & Event Timing**: Never write preview wording (e.g. 'is scheduled to face', 'will begin at five') for an event whose start time has already passed at writing time. A source that says an event is scheduled does not show it happened, so never rewrite a preview as past tense; skip it.")
    lines.append("6. **Source Trust & Undated Items**: When outlets report conflicting figures or details, favor higher-trust outlets (`Trust` weight). Treat items marked `[DATE UNVERIFIED]` with caution.")
    lines.append("7. **Developing Stories (Superseded Updates)**: If a cluster contains articles marked `[OLDER UPDATE — DO NOT USE FOR NUMBERS]`, ground all casualty, vote, and financial counts strictly from the latest non-superseded article.")
    lines.append("8. **Cross-Day Repeats**: Clusters marked `[COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]` were already aired in the last 48 hours; skip them unless a fresh developing angle justifies inclusion.")
    lines.append("9. **Multi-Source Synthesis**: When a cluster contains multiple member outlets, synthesize complementary details across sentences (e.g. combine casualties from one outlet with vehicle/route context or quotes from another) while citing the exact source URL for each sentence in the `evidence` list.")
    lines.append("10. **Headlines-Only Bulletin (Single-Sentence Summaries)**: This broadcast is strictly a headlines bulletin. Every story scene MUST contain exactly ONE single spoken sentence (10 to 22 words) summarizing the core event. DO NOT write descriptions, background paragraphs, historical context, or secondary sentences. Story scenes MUST transition monotonically through the 6 category blocks (3 to 4 single-line headlines per block).")
    lines.append("")
    lines.append("### Broadcast Style & TTS Optimization")
    lines.append("1. **TTS-Friendly**: Short, punchy broadcast sentences. Standard English spelling for Nepali names and places.")
    lines.append("2. **BANNED Characters in Narration**: Narration text must contain no digits (0-9), no %, $, &, /, parentheses (), brackets [], or emojis. (Note: Headlines may use digits, e.g. 'COP31' or 'NPL Season 3', because headlines are visual titles only, not read aloud).")
    lines.append("3. **Spelled-Out Numbers**: Spell numbers the way they are spoken (e.g. 'fifty thousand rupees' instead of 'Rs 50,000', 'eighty percent' instead of '80%'). Every number mentioned must exist in the source cluster.")
    lines.append("4. **Headline Constraints**: Max 10 words, sentence case, no ending period. Headlines may contain digits where appropriate.")
    lines.append("5. **Visual Keywords**: Exactly 3 to 5 generic stock terms (e.g. 'Kathmandu street traffic', 'parliament building'). Keywords must strictly match the subject matter of the story (e.g. wild cliff honey hunting is not beekeeping; use 'mountain cliffs', 'wild honeycombs'). Never use person names or corporate brands.")
    lines.append("6. **Evidence**: Every narration sentence in story scenes must be mapped to a supporting member article URL via `evidence: [{sentence_index, source_url}]`.")
    lines.append("")
    lines.append("---")
    lines.append("## 3. Required Deliverable Files")
    lines.append(f"Write these two files to `data/runs/{run_date}/script/`:")
    lines.append(f"1. `draft_script.json` (The structured script)")
    lines.append(f"2. `selection.json` (Editorial rationales)")
    lines.append("")
    lines.append("### `draft_script.json` Schema Example:")
    example_draft = {
        "language": "English",
        "origin": "editor",
        "run_kind": run_kind,
        "date": run_date,
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": intro_info["headline"],
                "narration": intro_info["narration"],
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": intro_info["visual_keywords"],
                "evidence": []
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Example Headline in Sentence Case Without Period",
                "narration": "In Kathmandu, officials announced twenty new environmental initiatives today while community volunteers gathered to plant trees along the riverbank.",
                "category": "national_politics",
                "sensitive": False,
                "source": "The Kathmandu Post, Onlinekhabar English",
                "source_urls": [
                    "https://example.com/article1"
                ],
                "visual_keywords": [
                    "Kathmandu city",
                    "river cleanup",
                    "community volunteers",
                    "tree planting"
                ],
                "evidence": [
                    {
                        "sentence_index": 0,
                        "source_url": "https://example.com/article1"
                    }
                ]
            },
            {
                "id": "outro",
                "type": "outro",
                "headline": "Thank You for Tuning In",
                "narration": "That wraps up our daily briefing. Thank you for tuning in, and stay safe until tomorrow.",
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": ["Kathmandu night skyline", "sunset mountains", "Nepal landscape"],
                "evidence": []
            }
        ]
    }
    lines.append("```json")
    lines.append(json.dumps(example_draft, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### `selection.json` Schema Example:")
    example_selection = {
        "selected_clusters": [
            {
                "cluster_id": "cluster_01",
                "reason": "Highest national impact confirmed across two major outlets."
            }
        ],
        "skipped_clusters": [
            {
                "cluster_id": "cluster_06",
                "reason": "Lower priority regional announcement."
            }
        ]
    }
    lines.append("```json")
    lines.append(json.dumps(example_selection, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("## 4. Today's Shortlist (Top Clusters)")
    lines.append(f"Total shortlisted clusters: {len(shortlist)}")
    lines.append("")

    for c in shortlist:
        repeat_tag = " [COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]" if c.get("covered_recently") else ""
        slot_label = f" [SLOT: {c['broadcast_slot_title']}]" if c.get("broadcast_slot_title") else ""
        lines.append(f"### [{c['cluster_id']}] {c['primary_title']}{slot_label}{repeat_tag}")
        cluster_meta = f"- **Category Guess**: `{c.get('category_guess', 'other')}`"
        if c.get("broadcast_slot_id"):
            cluster_meta += f" | **Designated Slot**: `{c['broadcast_slot_id']}` ({c.get('broadcast_slot')})"
        if c.get("max_importance_hint") is not None:
            cluster_meta += f" | **Max Importance**: `{c['max_importance_hint']}`"
        if c.get("max_source_trust") is not None:
            cluster_meta += f" | **Top Trust**: `{c['max_source_trust']}`"
        if c.get("all_dates_missing"):
            cluster_meta += " | **[DATE UNVERIFIED — verify freshness]**"
        if c.get("covered_recently") and c.get("repeat_reason"):
            cluster_meta += f" | **[COVERED RECENTLY — REQUIRE NEW DEVELOPING ANGLE]** ({c['repeat_reason']})"
        lines.append(cluster_meta)
        sources = c.get("sources", [])
        lines.append(f"- **Confirmed Outlets ({c.get('source_count', len(sources))})**: {', '.join(sources)}")
        if c.get("synthesis_notes"):
            lines.append(f"- **Multi-Source Synthesis**: {c['synthesis_notes']}")
        if c.get("conflicting_figures_resolution"):
            lines.append(f"- **Numbers Resolution**: {c['conflicting_figures_resolution']}")
        lines.append("- **Member Articles**:")
        for a in c["articles"]:
            sup_tag = " **[OLDER UPDATE — DO NOT USE FOR NUMBERS]**" if a.get("superseded") else ""
            lines.append(f"  - **{a['source']}**: [{a['title']}]({a['url']}){sup_tag}")
            pub_str = a.get("published_at") or "Unknown"
            date_conf = a.get("date_confidence")
            meta_bits: list[str] = []
            if a.get("source_trust") is not None:
                meta_bits.append(f"Trust: {a['source_trust']}")
            if a.get("importance_hint") is not None:
                meta_bits.append(f"Imp: {a['importance_hint']}")
            if a.get("feed_position") is not None:
                meta_bits.append(f"Pos: #{a['feed_position']}")
            if a.get("category_hint"):
                meta_bits.append(f"Cat: {a['category_hint']}")
            if date_conf:
                meta_bits.append(f"Date: {date_conf}")
            is_nepali = bool(
                (a.get("language") and a.get("language") != "en")
                or a.get("needs_translation")
            )
            if is_nepali:
                meta_bits.append(f"Lang: {a.get('language', 'ne')}")
            warn_tag = " **[DATE UNVERIFIED]**" if date_conf == "missing" else ""
            meta_suffix = f" ({' | '.join(meta_bits)})" if meta_bits else ""
            lines.append(f"    *Published*: {pub_str}{meta_suffix}{warn_tag}{sup_tag}")
            nepali_marker = " **[NEPALI — TRANSLATE TO ENGLISH]**" if is_nepali else ""
            lines.append(f"    *Summary*:{nepali_marker} {a.get('text_summary', '').strip()}")
            lines.append("    <details><summary>Click to view extracted full text</summary>")
            lines.append("")
            if is_nepali:
                lines.append("    > **[NEPALI — TRANSLATE TO ENGLISH]**")
                lines.append("")
            lines.append("    ```text")
            full_txt = a.get("full_text", "").strip()
            # Truncate full text in brief to max 200 words to keep prompt concise
            words = full_txt.split()
            if len(words) > 200:
                full_txt = " ".join(words[:200]) + " ... [TRUNCATED FOR BRIEF]"
            # Indent text slightly for markdown formatting
            for t_line in full_txt.splitlines():
                lines.append(f"    {t_line}")
            lines.append("    ```")
            lines.append("    </details>")
        lines.append("")

    if opinion_clusters:
        lines.append("---")
        lines.append("## 5. Excluded Opinion & Commentary Pieces")
        lines.append(f"Total excluded opinion pieces: {len(opinion_clusters)}")
        lines.append("> These items were excluded from the news shortlist per `config/editor.yaml` (`allow_opinion: false`).")
        lines.append("")
        for oc in opinion_clusters:
            lines.append(f"### [{oc['cluster_id']}] {oc['primary_title']}")
            if oc.get("opinion_reason"):
                lines.append(f"- **Opinion Reason**: `{oc['opinion_reason']}`")
            lines.append(f"- **Outlets ({oc['source_count']})**: {', '.join(oc['sources'])}")
            lines.append("- **Member Articles**:")
            for a in oc["articles"]:
                lines.append(f"  - **{a['source']}**: [{a['title']}]({a['url']})")
                if a.get("opinion_reason"):
                    lines.append(f"    *Reason*: {a['opinion_reason']}")
                lines.append(f"    *Published*: {a.get('published_at') or 'Unknown'}")
                lines.append(f"    *Summary*: {a.get('text_summary', '').strip()}")
            lines.append("")

    if stale_preview_clusters:
        lines.append("---")
        lines.append("## 6. Excluded Stale Previews")
        lines.append(f"Total excluded stale previews: {len(stale_preview_clusters)}")
        lines.append("> These items announce a future event whose time has already passed, excluded per `config/editor.yaml` (`allow_stale_preview: false`).")
        lines.append("")
        for sc in stale_preview_clusters:
            lines.append(f"### [{sc['cluster_id']}] {sc['primary_title']}")
            if sc.get("stale_preview_reason"):
                lines.append(f"- **Stale Preview Reason**: `{sc['stale_preview_reason']}`")
            lines.append(f"- **Outlets ({sc['source_count']})**: {', '.join(sc['sources'])}")
            lines.append("- **Member Articles**:")
            for a in sc["articles"]:
                lines.append(f"  - **{a['source']}**: [{a['title']}]({a['url']})")
                if a.get("stale_preview_reason"):
                    lines.append(f"    *Reason*: {a['stale_preview_reason']}")
                lines.append(f"    *Published*: {a.get('published_at') or 'Unknown'}")
                lines.append(f"    *Summary*: {a.get('text_summary', '').strip()}")
            lines.append("")

    return "\n".join(lines)

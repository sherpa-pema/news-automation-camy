"""Generates the editor brief markdown document for the Antigravity agent."""
from datetime import datetime
import json
from typing import Any


def format_spoken_date(date_str: str) -> str:
    """Formats YYYY-MM-DD into natural spoken words (e.g. Saturday, November 7, 2026)."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.strftime("%A, %B %d, %Y").replace(" 0", " ")
    except Exception:
        return date_str


def generate_editor_brief(
    run_date: str,
    budget: dict[str, Any],
    calib_msg: str,
    shortlist: list[dict[str, Any]],
    opinion_clusters: list[dict[str, Any]] | None = None,
    stale_preview_clusters: list[dict[str, Any]] | None = None,
) -> str:
    """Assembles the comprehensive editor brief markdown document."""
    spoken_date = format_spoken_date(run_date)
    wps = budget["words_per_second"]
    target_words = budget["target_total_words"]
    min_words = budget["min_total_words"]
    max_words = budget["max_total_words"]

    lines = []
    lines.append(f"# Broadcast Editor Brief - {spoken_date}")
    lines.append("")
    lines.append("> **Role**: You (the Antigravity agent) are acting as the Broadcast News Writer.")
    lines.append("> There are no external LLM calls or API keys. You will review the shortlist below,")
    lines.append("> select the top 5 news clusters, write `draft_script.json` and `selection.json`,")
    lines.append("> and run deterministic validation.")
    lines.append("")
    lines.append("---")
    lines.append("## 1. Word Budget & Speech Rate Calibration")
    lines.append(f"- **Calibration**: {calib_msg}")
    lines.append(f"- **Speech Rate**: `{wps:.2f}` words per second")
    lines.append(f"- **Target Video Duration**: 2:00 to 3:00 (120 to 180 seconds, target {budget['target_duration_sec']:.0f}s)")
    lines.append(f"- **Total Word Budget**: **{target_words} words** (allowed range: **{min_words} to {max_words} words**)")
    lines.append("- **Scene Breakdown** (7 scenes total):")
    lines.append("  - **1 Intro scene**: 15 to 40 words (target ~25 words)")
    lines.append("  - **5 Story scenes**: 20 to 60 words each (target ~50 words each = ~250 words)")
    lines.append("  - **1 Outro scene**: 15 to 40 words (target ~25 words)")
    lines.append("")
    lines.append("---")
    lines.append("## 2. Mandatory Writing Rules")
    lines.append("### Ground Truth & Attribution")
    lines.append("1. **Strictly Grounded**: Use ONLY facts present in the cluster's articles. No invented details, figures, context, or quotes. If a fact is unclear or missing, leave it out.")
    lines.append("2. **Complete Rewrite**: Never copy article sentences. Any run of 8 or more consecutive words shared with raw article text is strictly banned.")
    lines.append("3. **Attribution**: Explicitly attribute claims ('officials said', 'the Ministry of Finance reported').")
    lines.append("4. **Sensitivity & Protection**: Crime, fatal accidents, deaths, abuse, or stories involving minors MUST set `sensitive: true`. Never name minors, victims, or unconvicted suspects. No graphic details.")
    lines.append("5. **Temporal & Event Timing**: Never write preview wording (e.g. 'is scheduled to face', 'will begin at five') for an event whose start time has already passed at writing time. A source that says an event is scheduled does not show it happened, so never rewrite a preview as past tense; skip it.")
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
        "date": run_date,
        "scenes": [
            {
                "id": "intro",
                "type": "intro",
                "headline": f"Nepal Daily Briefing for {spoken_date}",
                "narration": f"Good evening. Welcome to the Nepal Daily Briefing for {spoken_date}. Here are today's top stories from across the nation.",
                "category": None,
                "sensitive": False,
                "source": "Nepal News Desk",
                "source_urls": [],
                "visual_keywords": ["Kathmandu valley", "news studio", "map of Nepal"],
                "evidence": []
            },
            {
                "id": "scene_01",
                "type": "story",
                "headline": "Example Headline in Sentence Case Without Period",
                "narration": "In Kathmandu, officials announced twenty new initiatives today. Environmental departments confirmed that community volunteers gathered to plant trees along the riverbank.",
                "category": "environment",
                "sensitive": False,
                "source": "The Kathmandu Post, Onlinekhabar English",
                "source_urls": [
                    "https://example.com/article1",
                    "https://example.com/article2"
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
                    },
                    {
                        "sentence_index": 1,
                        "source_url": "https://example.com/article2"
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
                "visual_keywords": ["Kathmandu night skyline", "sunset mountains"],
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
        lines.append(f"### [{c['cluster_id']}] {c['primary_title']}")
        lines.append(f"- **Category Guess**: `{c['category_guess']}`")
        lines.append(f"- **Confirmed Outlets ({c['source_count']})**: {', '.join(c['sources'])}")
        lines.append("- **Member Articles**:")
        for a in c["articles"]:
            lines.append(f"  - **{a['source']}**: [{a['title']}]({a['url']})")
            lines.append(f"    *Published*: {a.get('published_at') or 'Unknown'}")
            lines.append(f"    *Summary*: {a.get('text_summary', '').strip()}")
            lines.append("    <details><summary>Click to view extracted full text</summary>")
            lines.append("")
            lines.append("    ```text")
            full_txt = a.get("full_text", "").strip()
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

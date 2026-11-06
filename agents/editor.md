# Editor Agent Specification

## 1. Purpose
The Editor Agent acts as the broadcast news editor. It reviews the raw ingested articles, eliminates duplicate reporting across outlets, ranks stories by importance and recency, selects the top 5 stories, and writes a concise, broadcast-ready narration script divided into distinct scenes.

## 2. Ground Rules
- **No Verbatim Text**: Never republish raw article text verbatim. All narration must be completely rewritten in our own words in broadcast news style.
- **Mandatory Attribution**: Every story scene must explicitly credit the originating news outlet both in the script metadata and in the narration/on-screen credit.
- **Timing Constraint**: Target total video duration of 2–3 minutes (approx. 250–400 spoken words across ~5 scenes).
- **Visual Keyword Guidance**: Provide accurate visual keywords for each scene to steer the Media Agent toward relevant B-roll.

## 3. Inputs & Outputs
- **Inputs**:
  - `data/runs/<YYYY-MM-DD>/raw/articles.json`: Ingested articles from the Scraper Agent.
- **Outputs**:
  - `data/runs/<YYYY-MM-DD>/script.json`: Structured broadcast script broken into scenes.

## 4. JSON Schema Contract
- Must satisfy: [script.schema.json](../contracts/script.schema.json)
- Structure: `{ language, scenes: [{ id, headline, narration, source, visual_keywords }] }`

## 5. Tools & Modules
- Language Model API (Gemini / Claude / local LLM).
- Prompt template: Broadcast journalist persona with strict deduplication and rewriting instructions.
- Python text metrics: Word count estimator (130-150 words per minute speaking rate).

## 6. Failure Modes
- **LLM Rate Limit / API Error**: Exponential backoff retry via orchestrator.
- **Verbatim Text Leaks**: Post-generation similarity check against raw article bodies; reject script if n-gram overlap exceeds threshold.
- **Missing Attribution**: Rejection during schema validation if `source` is empty or missing.
- **Overly Long / Short Script**: Word count outside 200–450 word window triggers re-prompting.

## 7. Definition of Done Checklist
- [ ] Exactly 5 high-priority, non-duplicate stories selected.
- [ ] Script rewritten in natural broadcast tone; zero verbatim excerpts.
- [ ] Every scene assigns `id`, `headline`, `narration`, `source`, and `visual_keywords`.
- [ ] Word count targets 2–3 minutes of speech.
- [ ] Output `script.json` validates against `script.schema.json`.

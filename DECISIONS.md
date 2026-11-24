# Architectural Decisions

Record major technical choices here (Date, Decision, Why, Trade-offs) so future work stays consistent.

## 2026-11-25 — Unified Bulletin Mode & VoiceAgent-Safe Dynamic Budgeting
- **Date**: 2026-11-25
- **Decision**: Authoritatively propagate bulletin format (`"headlines"` vs `"focused"`) from `EditorAgent.prepare()` into `budget.json`, `editor_brief.md`, and `editor_validator.py`. In `focused` mode (used when targeting specific categories or when shortlist yield is < 10 clusters), enforce `min_total_words >= 150` and allow 2-4 broadcast sentences per story (30-55 words) with 4-10 total scenes.
- **Why**: Low-yield runs previously suffered from a mathematical deadlock: `editor_validator.py` saw `story_max_words <= 28` from headlines configuration, forced 1-sentence constraints (producing at most 97 words for 4 scenes), but required minimum 120 words. Furthermore, any draft under ~130 words fails VoiceAgent's hard 60.0-second audio duration gate.
- **Trade-offs**: Bulletins with few stories now develop multi-sentence depth (~75-100s audio) rather than rapid 1-sentence blurbs (~40s audio), satisfying all downstream audio, timeline, and video validation gates without any regressions.

## 2026-11-25 — 14-Day Lookback Override for Category Section Feeds
- **Date**: 2026-11-25
- **Decision**: Apply `lookback_seconds_override = 14 * 24 * 3600` to dedicated topic section feeds unless an explicit CLI `--lookback-hours` override was provided by the user.
- **Why**: Category section feeds (e.g. Science & Tech, Sports) publish at weekly or bi-weekly frequencies. Previously, `_run_internal` always supplied `lookback_normal=30.0`, which bypassed the lookback override check and caused all category articles to be dropped as stale.
- **Trade-offs**: Dedicated topic feeds now retain high-value articles up to 14 days old, providing strong cluster yield for focused runs.

## 2026-11-25 — Source Health Circuit Breaker Fairness
- **Date**: 2026-11-25
- **Decision**: Update `_record_circuit_breaker_outcome()` to inspect `stats["found"]` and drop reasons. When a feed returns HTTP 200 and items are found, drops solely due to category filtering or age window do not increment `consecutive_failures`.
- **Why**: Healthy sources were falsely accumulating 3 consecutive failures during category-specific runs and tripping circuit breaker cooldowns for subsequent general runs.
- **Trade-offs**: Preserves circuit breaker protection against real network outages and dead feeds while preventing false-positive lockouts.

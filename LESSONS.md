# Engineering Lessons Learned

## 2026-11-25 — Downstream VoiceAgent Contract Preservation
- **Context**: Resolving mathematical deadlocks in EditorAgent on low-yield runs (e.g. 2 stories).
- **Rule**: Never lower Editor word budgets below the duration floor of downstream audio stages. VoiceAgent strictly enforces `min_duration_sec: 60.0` in `VoiceAgent.validate()`. At 2.15 words/second with natural punctuation pauses, a broadcast requires at least ~150 spoken words to guarantee audio duration `>= 60.0s`.
- **Solution**: For runs with low cluster yield or targeted category focus, switch to `"focused"` format where stories have 2-4 broadcast sentences (30-55 words each) rather than 1-sentence headlines. This guarantees both editorial depth and VoiceAgent duration compliance.

## 2026-11-25 — Circuit Breaker vs Topic Filtering Separation
- **Context**: Feed cooldowns in `data/state/source_health.json`.
- **Rule**: Zero returned articles is NOT a source failure if the HTTP response succeeded and articles were found in the feed. When runs filter by specific categories, off-topic drops (`category_filtered`) are normal business logic. The circuit breaker must inspect `stats["found"]` and `stats["drop_reasons"]` before incrementing `consecutive_failures`.

## 2026-11-25 — Unified Configuration Propagation
- **Context**: Divergence between `EditorAgent.prepare()`, `editor_brief.py`, and `editor_validator.py`.
- **Rule**: Whenever format or mode rules dictate validation constraints, the mode must be authoritatively written to `budget.json` (`format: headlines | focused`) during `prepare()`. Brief generation, writer system prompts, and validator must all consume `budget["format"]` so prompt instructions and validation checks never contradict each other.

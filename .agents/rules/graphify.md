---
trigger: always_on
description: Consult the graphify knowledge graph at graphify-out/ before searching files, and keep the graph updated after code changes.
---

# Codebase Navigation & Graph Maintenance Rules

## 1. Always Search and Navigate via Knowledge Graph First
- **Never grep or scan raw files blindly** when trying to locate where symbols, components, contracts, or features are implemented.
- For codebase or architecture questions, check `graphify-out/graph.json` first:
  - Run `graphify query "<question>"` to locate relevant files, concepts, and relationships.
  - Run `graphify explain "<concept>"` to inspect a specific node and its neighbors.
  - Run `graphify path "<node_A>" "<node_B>"` to trace paths/data-flow between subsystems.
  - Run `graphify section "<heading>"` to read targeted markdown sections directly without reading entire files.
- Consult `graphify-out/GRAPH_REPORT.md` for high-level module architecture and community clusters.

## 2. Mandatory Graph Update After Code Changes
- After creating, updating, refactoring, or deleting code files in `news-auto`, **always update the graph**:
  - Run `graphify update .` to re-extract AST symbols and sync `graphify-out/graph.json`.
  - For structural changes affecting documentation or specifications, run `/graphify . --update`.
- Ensure git commits trigger the installed post-commit hook, keeping the graph synchronized.

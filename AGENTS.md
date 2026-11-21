# Repository Guidelines: Graph-First Navigation & Maintenance

## Graph-First Codebase Exploration
- **Do not blindly grep or scan across every file**: The codebase has an indexed knowledge graph located in `graphify-out/`.
- When searching for files, symbols, modules, or dependencies, query graphify first:
  - `graphify query "<query>"`: finds relevant modules, nodes, and connections.
  - `graphify path "<source>" "<target>"`: reveals shortest structural path between concepts.
  - `graphify explain "<symbol>"`: details a node, its purpose, and its immediate neighbors.
  - `graphify section "<heading>"`: extracts Markdown sections without opening large doc files.
- Consult `graphify-out/GRAPH_REPORT.md` for community structures and god nodes.

## Mandatory Graph Updates on Code Modification
- Whenever source code files are added, modified, or refactored:
  - Run `graphify update .` to update the AST extraction and maintain fresh node/edge indices.
  - The repository has active git hooks (`post-commit`, `post-checkout`, `post-merge`) installed via `graphify hook install` to guarantee ongoing graph accuracy.

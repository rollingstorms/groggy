# Algorithms (algorithms.rs)

This module implements core graph algorithms for Groggy, available as methods on `FastGraph`:

- `bfs(start)` — Breadth-First Search
- `dfs(start)` — Depth-First Search
- `shortest_path(start, goal)` — Unweighted shortest path
- `connected_components()` — List of connected components
- `clustering_coefficient()` — Per-node clustering coefficient

**Python API:**
All are available as methods on `FastGraph` and return Python-friendly types.

**Rust:**
See `algorithms.rs` for implementation details.

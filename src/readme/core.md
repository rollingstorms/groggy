# FastGraph (core.rs)

`FastGraph` is the main graph structure for Groggy. It provides Python and Rust APIs for:
- Node and edge collections
- Attribute management
- Subgraph extraction
- Graph metadata

**Python API:**
- `FastGraph()` — create a new graph
- `.nodes()` / `.edges()` — get node/edge collections
- `.info()` — get graph info
- `.subgraph(node_ids, edge_ids)` — create a subgraph

Implements basic algorithms (BFS, DFS, shortest path, connected components, clustering coefficient) as methods.

**Rust:**
- See `core.rs` for struct and method details.

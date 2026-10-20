# Engine contract conformance baseline

This is the baseline for flat-list item 1. It records established behavior rather than redefining the engine. The focused checks are `tests/engine_contract.rs` and `tests/test_engine_contract.py`; broader CRUD coverage remains in `tests/modules/test_graph_core.py` and existing Rust tests.

| Established rule | Rust | Python |
| --- | --- | --- |
| An empty graph is a valid construction state. | `Graph::new()` | `Graph()` |
| Graphs default to undirected; direction is selected at creation. | `Graph::new_with_type(GraphType::Directed)` | `Graph(directed=True)` |
| One active node with no edges is the smallest meaningful graph. | `add_node`, membership, count, empty neighbors and degree | `add_node`, membership and counts |
| A newly created ordinary node has `entity_type = "base"`. | Typed attribute read | Python attribute read |

These rules are checked in both languages. They do not assert that every current view or historical read is already immutable.

Known contract gaps assigned to later items:

- Python `add_node(node_id=...)` currently accepts the argument but substitutes an auto-generated ID; stable external identity belongs to item 6.
- `Subgraph` pins IDs but reads live parent values; immutable state-pinned views belong to item 26.
- Timestamp snapshots select commit timestamps, not effective-time graph states; temporal reconstruction belongs to items 17–18 of the flat list.
- Direct mutations still require the legacy manual `commit`; routing them through automatically published versions belongs to item 9.

Item 2 live-state audit: `Graph` is the checked read boundary for node and edge attributes and topology. Node/edge entity handles now use it rather than querying retained pool slots directly. An inactive ID returns an error for entity-specific reads and is absent from active ID collections; bulk attribute-column reads retain their established aligned `None` result for inactive IDs. The focused checks are `tests/active_state_contract.rs` and `tests/test_active_state_contract.py`. Raw `GraphPool` access is a low-level storage API, not a live graph view. Algorithm-specific state and direction behavior remains under items 3 and 28.

Item 3 direction audit: Directed edge lookup and filtering, BFS/DFS, shortest path, all paths, and graph adjacency matrices now follow stored edge orientation. The pathfinding algorithm CSR cache uses the backing graph's direction; undirected paths retain reverse traversal. Focused checks are `tests/directed_contract.rs` and `tests/test_directed_contract.py`. Distinct weak/strong component definitions and algorithm-specific directed variants remain in item 28; stable path edge selection among parallel edges remains in item 4.

Item 4 edge identity audit: Parallel edges and self-loops have separate stable edge IDs. Shortest path now retains the chosen edge ID and its actual path weight; all simple paths enumerate parallel edge instances separately. Recorded snapshots preserve both parallel edge IDs and their typed values. Focused checks are `tests/edge_identity_contract.rs` and `tests/test_edge_identity_contract.py`. The proposed explicit endpoint-rewiring mutation is a separate history/change operation in item 13.

Item 5 mutation validation: Node/edge creation parses properties before allocating IDs. Rust `add_nodes_with_attrs` rejects system-managed node properties before allocation and retains columnar writes. Bulk node/edge removal validates active membership and duplicates for the whole input before changing topology. Python bulk setters collect conversion errors across entries and columns before calling Rust. Focused checks are `tests/mutation_validation.rs` and `tests/test_mutation_validation.py`, alongside the existing bulk-attribute checks. Duplicate removals now raise `InvalidInput`/`ValueError` before any write; missing or non-string `uid_key` values also reject node creation before allocation. Read-only Rust preflight checks let Python combine conversion and semantic errors in the same rejected call. Original input positions are retained even when an earlier ID cannot be decoded. Structurally malformed rows or columns report their structural error; fields that cannot be decoded cannot receive semantic validation. This establishes rejection atomicity for these input classes, not crash recovery or automatic version publication.

For each subsequent item, add focused Rust and Python conformance checks for the behavior being changed, then update this baseline when the contract or a documented gap changes.

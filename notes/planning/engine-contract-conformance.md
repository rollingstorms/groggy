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

For each subsequent item, add focused Rust and Python conformance checks for the behavior being changed, then update this baseline when the contract or a documented gap changes.

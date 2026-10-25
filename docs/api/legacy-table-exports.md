# Source-bound tables from graph and legacy accessors

The existing `Graph.table()`, `graph.nodes.table()`, `graph.edges.table()` and
legacy `Subgraph` table exports now use the same immutable capture/projection
path as `GraphView`. Each call captures current state without creating a delta.
Tables own their cell buffers and retain the source pin and exact row IDs.
Changing the graph afterward leaves old values and captures readable.

```python
from groggy import Graph, GraphTable
owner = Graph()
owner.add_nodes([{"x": 1}, {"x": 2}])
owner.add_edge(0, 1, w=3)
table = owner.table()
assert table.nodes.base_table().source_entity_ids == [0, 1]
assert table.edges.base_table().source_entity_ids == [0]
nodes = table.nodes.with_attributes("x", {0: 10, 1: 20})
edges = table.edges.with_attributes("w", {0: 4})
result = GraphTable(nodes, edges).update_source_graph(
    owner, node_columns=["x"], edge_columns=["w"], effective_at=123
)
assert owner.get_node_attr(1, "x") == 20
assert table.nodes.base_table()["x"].to_list() == [1, 2]
assert result["state_id"] == table.nodes.base_table().source_state_id + 1
```

A mixed `table()` export uses one shared capture for both components. Separate
calls to `nodes.table()` and `edges.table()` are separate captures, even at the
same numeric state, and cannot be presented as one capture for implicit mixed
writeback. Existing explicit mapped updates remain available.

Constrained accessor exports use checked selected membership, ascending unique
IDs, stable property ordering and row-aligned null padding. Explicitly present
null properties stay present; they are not dropped as “all null.” Stale selected
IDs now reject instead of producing fabricated empty rows or null endpoints.

```python
selected = owner.nodes[[1]].nodes
nodes = selected.table().base_table()
assert nodes.source_entity_ids == [1]
assert nodes.source_views()[0].node_attr(1, "x") == 20
owner.set_node_attr(1, "x", 99)
assert nodes["x"] == 20
```

Legacy `Subgraph` and accessors themselves still read their connected parent.
This change pins their **table outputs**, not all legacy object behavior. Each
fresh export sees current values and validates the current selection. A subgraph
with any stale selected member rejects the export; it does not partly export a
selection whose graph scope is no longer valid.

Manual dirty working state can be captured for inspection. Such a capture has
no accepted source state and remains read-only for checked source publication.
A later checkpoint does not upgrade an older capture's write authority.

```python
manual = Graph(manual=True)
manual.add_node(x=3)
working = manual.nodes.table().base_table()
assert working.source_state_id is None
assert working["x"] == 3
try:
    working.update_source_graph(manual, ["x"])
except ValueError:
    pass
else:
    raise AssertionError("dirty source unexpectedly authorized publication")
```

## Costs

A fresh legacy export captures the parent's read state before projecting columns.
This can copy state proportional to the full parent graph, including when only
a small selection is exported. It is not O(1), a borrowed table, or a COW capture.
For repeated reads, capture `view = graph.read_view()` once and reuse its table
exports. Cell buffers are still owned projections. Retained pins also retain
captured payloads until released.

FFI acquires the capture while holding the GIL because connected graph ownership
uses Rc/RefCell. Once captured, native membership selection and table projection
run with the GIL released. The separate FFI row-building/filtering implementations
have been removed. The current capture-versus-reuse benchmark is
`notes/benchmarks/table-export-capture-2026-10-25.json`; no fixed FFI latency or
memory budget is certified.

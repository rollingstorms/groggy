# Frozen graph-table source pins

Tables exported from `GraphView.nodes_table()`, `edges_table()` and `table()` now
retain their source capture and immutable row-to-entity IDs. Cell buffers are owned;
editing them does not update the graph. Explicit `update_source_graph` publishes
chosen property columns using the retained identity and returns a new GraphView.

```python
from groggy import Graph

graph = Graph()
graph.add_nodes([{"score": 0}, {"score": 1}, {"score": 2}])
original = graph.read_view()
table = original.nodes_table().base_table()
table = table.sort_by("score", ascending=False).head(2).select(["score"])
assert table.source_entity_ids == [2, 1]
assert table.source_entity_kind == "node"
assert table.source_state_id == graph.current_state_id
table["score"] = [2.0000000000000004, None]
result = table.update_source_graph(graph, ["score"], effective_at=10)
assert result["view"].node_ids == [1, 2]
assert graph.get_node_attr(2, "score") == 2.0000000000000004
assert original.node_attr(2, "score") == 2
```

Row identity survives column projection even when the physical `node_id`/`edge_id`
column is omitted. Clone, head/tail/slice, explicit row selection, masks, supported
filters/sorts, grouped row selections and duplicate removal propagate the exact
row order. Repeated row selection retains repeated IDs; source writeback rejects
ambiguous duplicate destinations until the caller reduces them explicitly.
Column edits/renames preserve row identity. Empty slices retain columns and an
empty source row map. Out-of-range selections reject even with no value columns.

`source_views()` returns frozen source selections, including read-only derivations.
`source_entity_ids` and `source_entity_kind` describe direct row binding;
`source_state_id` is the accepted-head precondition, not a guarantee that a write
will pass schema, shape or identity validation.

## Joins and derived identities

Inner/left/right/outer/cross joins retain both inputs' source captures and remain
read-only for automatic source writeback. Exact repeated captures/selections are
deduplicated; equal numeric state IDs from different captures/graphs are distinct.
Join rows may describe multiple entities or duplicate a fact, so the API does not
guess one entity per row. Their direct row identity/state is `None`.

```python
left = graph.read_view().nodes_table().base_table()
right_graph = Graph()
right_graph.add_nodes([{}, {}, {}])
right = right_graph.read_view().nodes_table().base_table()
joined = left.join(right, on="node_id", how="inner")
assert len(joined.source_views()) == 2
assert joined.source_entity_ids is None
assert joined.source_state_id is None
try:
    joined.update_source_graph(graph, ["score"])
except ValueError as error:
    assert "explicit identity" in str(error)
else:
    raise AssertionError("joined rows require explicit resolution")
```

Aggregated/reshaped rows and appended synthetic rows retain available source
context without an inferred entity binding. Existing explicit mapped writeback
remains available after the caller resolves identities/duplicates. This slice does
not certify all existing join key algorithms, specialized GraphTable domain
merges or TableArray aggregation adapters; their original broader work stays open.
No join persists graph entities or records a graph delta by itself.

## One mixed node/edge update

`NodesTable.from_base_table` and `EdgesTable.from_base_table` preserve the source
while validating structural columns. GraphTable source updates require both
components to retain the correct entity kinds and the **exact same capture**.
Use one original `view.table()`; separate calls to `graph.read_view()` intentionally
create different captures, even at the same head.

```python
from groggy import GraphTable, NodesTable, EdgesTable

graph.add_edge(0, 1, weight=1)
exported = graph.read_view().table()
nodes = exported.nodes.base_table()
edges = exported.edges.base_table()
nodes["score"] = [10, 11, 12]
edges["weight"] = [2.5]
edited = GraphTable(
    NodesTable.from_base_table(nodes),
    EdgesTable.from_base_table(edges),
)
before = graph.current_state_id
result = edited.update_source_graph(
    graph, node_columns=["score"], edge_columns=["weight"], effective_at=-10
)
assert result["state_id"] == before + 1
assert graph.get_edge_attr(0, "weight") == 2.5
```

## Validation, durability and costs

- Existing ID columns must still match retained row IDs. Existing edge endpoint
  columns must match captured endpoints. Editing a structural cell cannot retarget
  a property update; dropping it cannot erase immutable row identity.
- Writes choose complete property columns explicitly. Null cells assign explicit
  Null, including projection holes; omitted columns remain unchanged. Structural
  and managed identity columns cannot be selected. Graph-local reference values
  still require an explicit reference policy through ordinary GraphBatch APIs.
- Source updates use `apply_view_batch`: stale/foreign/dirty/read-only pins reject,
  schemas validate before mutation, and the result is prevalidated before one
  accepted delta. Empty writes retain that API's no-changes rejection policy.
- The same Python method accepts `DurableGraph`; installed and stricter optional
  schemas apply, rejected updates append nothing, and accepted values recover
  through the existing journal. Reopening requires a new source capture.
- Plain imported tables and current legacy Graph/Subgraph exports remain unbound.
  They use explicit mapping with `update_graph` or explicit independent
  materialization with `to_graph`; visible integer IDs alone grant no source pin.
- Pins are in-process metadata and are not persisted in CSV/JSON as capabilities.
  Frozen values remain readable after the owner disappears. Holding a sourced
  table retains the full source payload; derived tables share its Arc allocation.
  Row selections allocate their own row-ID map, and exported cells remain owned.
- Explicit source writeback includes full refreshed-view capture costs. Ordinary
  graph/table methods with no source metadata do not capture a graph implicitly.
  The original `update_graph` mapping API and its behavior remain available.

See [implementation inventory, verification and bounded costs](../../notes/planning/table-source-pins-2026-10-25.md).

See [table joins](table-joins.md) for typed/composite keys, stable output ordering,
column naming and unmatched-row behavior.

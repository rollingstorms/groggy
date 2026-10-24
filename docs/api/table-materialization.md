# Checked table-to-graph materialization

`GraphTable.to_graph(*, directed=False)` creates an independent publishing graph
from the complete table. All supported rows/properties are staged in Rust and
accepted together. A nonempty result starts at state 1, with no implicit
checkpoint; an empty graph stays clean at state 0. Ordinary later writes publish
normally. This is a new active graph, not recovery of the source graph's history.

```python
from groggy import GraphTable, NodesTable, EdgesTable

table = GraphTable(
    NodesTable.from_dict({"node_id": [200, 100], "name": ["Ada", "Lin"],
                          "external_id": ["ada", "lin"]}),
    EdgesTable.from_dict({"edge_id": [900], "source": [100], "target": [200],
                          "weight": [1.0000000001], "external_id": ["knows"]}),
)
graph = table.to_graph(directed=True)
assert graph.is_directed
assert graph.current_state_id == 1
assert graph.automatic_publication
assert not graph.has_uncommitted_changes()
assert graph.node_by_external_id("ada") == 0
assert graph.node_by_external_id("lin") == 1
assert graph.edge_by_external_id("knows") == 0
assert graph.edge_endpoints(0) == (1, 0)
assert graph.get_edge_attr(0, "weight") == 1.0000000001
assert graph.checkpoints() == []
assert graph.commit("materialized table", "example") == 1
```

## IDs, direction and properties

- New node and edge IDs follow their respective table row order, starting at zero.
  Source structural IDs identify rows and connect endpoints; sparse/high IDs do
  not cause sparse destination allocation. Duplicate source IDs reject.
- Tables currently do not store source graph direction. The default remains
  undirected; pass `directed=True` to request directed semantics. When copying a
  graph projection, use `source.table().to_graph(directed=source.is_directed)`.
- Managed external IDs are registered through the batch identity API. They remain
  usable for lookup/upsert. Missing/null cells mean unassigned; non-text, empty or
  conflicting IDs reject, including conflicts between nodes and edges.
- Inline property values retain native types, including f32/f64 vectors, binary,
  structured and temporal properties. NodeArray/EdgeArray values remap table IDs
  to new graph IDs, retaining order and repetitions. References to entities absent
  from the table reject.
- Table null cells become explicit Null properties. Projection tables do not
  distinguish absent properties from null cells, so materialization cannot recover
  that missing distinction. System metadata is handled separately: undeclared/null
  node entity type becomes canonical base, null external IDs remain unassigned,
  null labels remain absent, and null contains_subgraph cells have no reference.
- Graph-kind properties still obey native validation. Invalid labels or
  relationship types reject instead of logging a warning and dropping the value.

## Complete validation

Structural IDs and endpoints must be nonnegative full/compact integers, with no
invalid row silently filtered out. Every endpoint must identify a table node.
Default table policy allows self-loops and parallel edges. Explicit native policy
restrictions are checked; undirected reversed endpoint pairs count as parallel.
The table's diagnostic policy cannot permit dangling destination endpoints.

Null edge IDs retain the legacy assignment rule: fill them in row order after the
maximum declared edge ID (or after zero when none is declared). Assignment checks
integer overflow before materialization and never mutates the source table.
The explicit `auto_assign_edge_ids()` helper uses the same checked routine.

Meta-node identities and SubgraphRef/contains_subgraph values require portable
source pool contents. A table alone does not carry those contents, so conversion
rejects instead of silently creating a base node or copying a dangling pool ID.
The error leaves the source table and any source graph unchanged. There is no
partially returned destination graph and no warning-only property failure path.

## Native API and remaining scope

Rust `GraphTable::to_graph()` now returns the same clean publishing result;
`to_graph_with_config(GraphConfig)` selects destination configuration explicitly.
The old manually dirty native result is retired for this factory. FFI translates
the direction option and errors; it contains no conversion business logic and no
longer clones the entire table before conversion. This borrowed graph construction
currently retains the GIL, as other graph mutation/construction paths do.

[Writeback](table-writeback.md) explicitly updates an existing graph and is a
separate operation. Retained history/pool recovery, connected source identity and
read pins, and source-pool materialization remain separate checklist work.

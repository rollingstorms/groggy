# Explicit graph-table property writeback

`GraphTable.update_graph(graph, expected_state, *, node_mapping, edge_mapping,
node_columns, edge_columns, effective_at=None)` applies selected node and edge
property columns as **one checked graph mutation** and returns its state ID.
It updates existing entities. It does not create entities or change endpoints.

```python
from groggy import Graph, GraphTable, NodesTable, EdgesTable

graph = Graph()
a, b = graph.add_nodes([{"score": 1}, {"score": 2}])
edge = graph.add_edge(a, b, weight=1)
prior = graph.read_view()
before = graph.current_state_id
edited = GraphTable(
    NodesTable.from_dict({"node_id": [100, 200], "score": [10, 20]}),
    EdgesTable.from_dict({"edge_id": [900], "source": [100], "target": [200],
                          "weight": [3.5]}),
)
state = edited.update_graph(
    graph, before,
    node_mapping={100: a, 200: b}, edge_mapping={900: edge},
    node_columns=["score"], edge_columns=["weight"],
    effective_at=123,
)
assert state == before + 1
assert graph.get_node_attr(a, "score") == 10
assert graph.get_edge_attr(edge, "weight") == 3.5
assert prior.node_attr(a, "score") == 1
assert not graph.has_uncommitted_changes()
assert graph.commit("table checkpoint", "example") == state
```

## Identity and ambiguity

Both mappings are explicit caller assertions of destination identity. They cover
all rows of their components, with no extra keys, and map to active destination
entities. Node rows also resolve edge endpoints, even when no node property is
selected. Mapped endpoints must agree with each destination edge; reversal is
accepted for undirected graphs. Parallel edges remain distinct edge IDs.

Repeated table IDs or multiple rows targeting one destination reject, including
identical repeated assignments. A join does not create graph entities. Reduce or
select joined rows explicitly before writing them back. A stale expected state,
uncommitted legacy writes, missing/invalid structural cells, inactive targets,
endpoint disagreement, missing/duplicate selected columns and reserved columns
all reject before writes, deltas, versions or entity allocation. Invalid rows are
not silently filtered out.

These owned tables do not carry a verified source-graph identity or read pin
through joins. This API does not infer provenance from row numbers or prove that
the caller chose the intended graph; the explicit maps define that choice. The
expected state guards that destination's published version. Automatic source
lineage and connected pinned-table updates remain separate checklist work.

## Values and publication

Only selected columns are written. Unselected properties remain unchanged. A
selected null cell assigns explicit `Null`; it does not mean delete or leave
unchanged. Projection tables currently use null cells for missing properties,
so select columns deliberately. Typed inline values retain their native types,
including f64 scalars/vectors and temporal properties.

Structural columns (`node_id`, `edge_id`, `source`, `target`), node identity
(`entity_type`, `contains_subgraph`) and managed `external_id` cannot be updated
this way. Use explicit identity APIs for managed-ID changes. Graph-local
`SubgraphRef`, `NodeArray` and `EdgeArray` property values reject pending explicit
reference-remapping support; numeric vectors are ordinary values and are allowed.

Omit `effective_at` for one captured acceptance clock; supply signed Unix
microseconds to give all changes an explicit effective time. For distinct times
per assignment, use `GraphBatch`. `commit()` can label the resulting state as an
explicit checkpoint. An empty column selection validates the destination and
mappings but publishes nothing. A nonempty assignment uses ordinary batch
semantics even when its value matches the current value.

Like `Graph.apply_batch`, this explicit writeback requires a clean published
head and publishes once even on `Graph(manual=True)`. It does not change the
ordinary publication policy. The Rust entry point accepts a `TableWriteback`
request and returns `BatchResult`; the FFI only constructs that request and
translates the result/errors. Bulk values remain columnar. This borrowed graph
mutation currently retains the GIL, like the existing borrowed graph setters.

The separate `to_graph()` conversion materializes a new graph; it is not
writeback. Its legacy conversion/portability gaps and full pipeline publication
boundaries are not closed by this API.

# Terminal removals in atomic builders

A native step pipeline can calculate properties, then stage multiple node/edge
removals, and accept the complete operation as **one delta**. No working-graph
clone or intermediate publication is used.

```python
from groggy import Graph
from groggy.builder import step_pipeline

graph = Graph()
graph.add_nodes(3)
graph.add_edges([(0, 1), (1, 2)])
before = graph.read_view()
head = graph.current_state_id
remove = step_pipeline("remove middle node", [
    {"id": "graph.remove_nodes", "params": {"ids": [1]}},
], atomic=True)
remaining = graph.apply(remove, persist=True)
assert graph.current_state_id == head + 1
assert sorted(remaining.node_ids) == [0, 2]
assert sorted(remaining.edge_ids) == []
assert sorted(before.node_ids) == [0, 1, 2]
assert sorted(before.edge_ids) == [0, 1]
```

## Stage controls

| Stage | Target |
|---|---|
| `graph.remove_nodes` | Selected active node IDs; incident edges cascade in the owner graph, including edges outside the selection. |
| `graph.remove_edges` | Selected active edge IDs; parallel instances and self-loops retain their individual IDs. |

Exactly one target input is required:

- `ids`: nonnegative integer IDs. Duplicate removals reject the whole operation,
  including duplicates across stages.
- `source`: a matching node/edge map variable, or a node column. **Every key is a
  removal target; values are not interpreted as booleans.** A filtering step
  returns only matching keys, making it a suitable source.

An optional `effective_at` is a signed integer Unix timestamp in microseconds,
used for each removal in that stage. Omitted times use the single acceptance
clock. Node cascades use the existing canonical timing rules: the earliest
removed endpoint time, with an explicit edge removal taking precedence. Property
writes retain their existing effective-time rules.

```python
from groggy import Graph
from groggy.builder import step_pipeline

graph = Graph()
graph.add_nodes([{"discard": True}, {"discard": False}])
filtered = step_pipeline("remove matching keys", [
    {"id": "core.filter_nodes_by_attr", "params": {
        "attr": "discard", "target": "chosen",
        "predicate": {"type": "eq", "value": {"type": "Bool", "value": True}},
    }},
    {"id": "graph.remove_nodes", "params": {"source": "chosen"}},
], atomic=True)
graph.apply(filtered)
assert sorted(graph.node_ids) == [1]
```

## Execution boundary

Removal stages begin the **topology phase** of a local `atomic=True` builder.
Certified property loads, initialization, arithmetic and attachments can also
follow that phase; see [pending properties](pending-topology-properties.md).
Unadapted adjacency-dependent calculations must precede topology.
Further removals, explicit `graph.capture_staged`, immutable graph/table stages
and variable aliases may follow; see [staged graph captures](staged-graph-captures.md). `persist=True` is required. A certified prefix and all
removals share one checked publication; any target, duplicate, property/time
conflict, cancellation or final publication failure discards staging and restores
prior Context outputs. No-op removals create no version.

Property writes to removed entities, including edges removed by cascade, reject
under the existing GraphBatch resulting-state validation. They are not silently
dropped. A stale immutable write source rejects. The returned selection contains
only surviving originally selected IDs; it does not expand induced edges. The
Python result is frozen at the accepted state. A subsequent ordinary pipeline
algorithm receives that surviving selection.

Read-after-removal through explicit immutable staged captures is supported.
Ordinary calculations over an implicit topology overlay, topology loops and
arbitrary custom effect rollback remain unsupported. Additions and endpoint
revisions now share the same staged execution boundary. Outer
`Pipeline(..., atomic=True)` can compose certified builders and publish once;
see [outer atomic topology](outer-atomic-topology.md). A regular outer pipeline
retains its per-algorithm publication and failure boundary.

The implementation stages touched IDs/properties. Reconstructing the returned
selection scans selected nodes and edges; this is not constant time in removals.
The Python adapter's existing final frozen capture cost also remains. Cancellation
is checked during staging and before acceptance; acceptance itself is not an
interruptible transaction. Custom native step capability declarations remain a
trusted extension contract.

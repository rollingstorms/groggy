# Immutable captures of pending graph state

`graph.capture_staged` produces an immutable **selected pending working-state**
value inside a persistent local atomic builder. It includes properties staged so
far, explicit edge removals and node-removal cascades. Later stages and later
owner edits cannot alter the captured value.

```python
from groggy import Graph
from groggy.builder import step_pipeline

graph = Graph()
graph.add_nodes(3)
graph.add_edges([(0, 1), (1, 2), (2, 2)])
head = graph.current_state_id
program = [
    {"id": "graph.capture_staged", "params": {"output": "before"}},
    {"id": "graph.remove_nodes", "params": {"ids": [1]}},
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.nodes_table", "params": {"source": "pending", "output": "nodes"}},
    {"id": "table.emit", "params": {"source": "nodes", "name": "nodes"}},
    {"id": "graph.emit", "params": {"source": "before", "name": "before"}},
    {"id": "graph.emit", "params": {"source": "pending", "name": "pending"}},
]
_, profile = graph.apply(step_pipeline("inspect pending removal", program, atomic=True),
                         return_profile=True)
outputs = profile["outputs"]
assert graph.current_state_id == head + 1
assert sorted(outputs["before"].node_ids) == [0, 1, 2]
assert sorted(outputs["pending"].node_ids) == [0, 2]
assert sorted(outputs["pending"].edge_ids) == [2]
assert outputs["nodes"].source_entity_ids == [0, 2]
assert outputs["pending"].write_state_id is None
```

The only parameter is a required nonempty `output` variable name. Capture-only
programs publish no delta. A capture before any mutation gives the current
selected state; a capture after property work contains those pending properties.
A capture after removals retains only surviving selected entities. Retained
meta-group source catalogs are shared, so meta-node source expansion continues
to read its original immutable group.

## Effect ordering

The native interpreter classifies steps before running them. Existing property
calculations can precede topology effects. After a topology effect, only these
certified operations are accepted:

- Further staged removals.
- `graph.capture_staged`.
- The nine explicit immutable graph/table read stages and variable `alias`.

Thus a pending graph can be selected, projected to tables, sliced, and emitted
before acceptance. Ordinary property/structural calculations, topology loops,
opaque callbacks and unadapted algorithms still reject in this phase. Existing
property-only nested and outer atomic certification stays conservative.

`atomic=True` on the local builder and `persist=True` are required. Outer
`Pipeline(..., atomic=True)` still rejects staged captures/removals before any
algorithm executes. Explicit names do not decide generated naming, caching or
high-level DSL policy. Native extensions' capability declarations remain trusted.

## Pending versus accepted state

A staged capture is a read-only provisional value, **not an accepted-state pin**
and not an effective-time reconstruction. `write_state_id` is always `None`,
including when there are no pending changes. Each capture has its own identity;
independent captures cannot be merged as if they shared one source. Its internal
snapshot retains the base state number for projection bookkeeping; that number
must not be interpreted as an accepted version containing the staged changes.
Use a fresh owner `read_view()` after successful execution for accepted-state
writeback provenance, and the temporal APIs for effective-time queries.

Final canonical batch validation still decides whether all proposed changes can
be accepted. Invalid duplicate operations and temporal/property conflicts reject
the complete run. Late graph/table read errors also discard staged writes and
produced outputs, restoring previous Context outputs. Unaccepted intermediate
values are not returned in a successful result on failure.

## Ownership and cost

Captured values own selected immutable data and can outlive the writable owner
and be read on worker threads. The capture copies selected property values and
rebuilds selected adjacency; retained group sources share immutable payloads.
It does not clone a writable Graph, history or pool. This is explicit
materialization, with time/memory proportional to selected data (including value
sizes), plus the canonical base-capture cost when that source is not already
retained. It is not a lazy overlay or an O(1) snapshot. No new default capture,
permanent cache, checkpoint, index or version is introduced.

Cancellation is checked before and after materialization; its inner copy/kernel
is not interruptible. The builder still runs on the owner thread; this change
does not release the GIL around the whole pipeline. The PyGraph cyclic-GC
ownership limitation remains separate from these detached values.

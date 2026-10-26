# Owned snapshots and state restoration

Ordinary `snapshot_at_commit(state)` and `snapshot_at_timestamp(seconds)` retain
immutable selected values, direction, frozen group sources and configuration.
They do not copy the complete backing pool or all retained history. Their
`as_subgraph()` method creates the existing independent manual selection container,
now preserving all configuration fields and frozen source references. The selected
base is staged as a pending manual import; accept it before checked mutation.

Use `owned=True` when an independent graph must retain the original history,
branches, inactive ID/name reservations and arbitrary backing columns.

```python
from groggy import Graph, GraphBatch

g = Graph(directed=True, config={"query_timeout_ms": 123})
b = GraphBatch(effective_at=10)
n = b.add_node(value=None)
g.apply_batch(b, 0, "initial", "example")
first = g.current_state_id
g.writer().set_node_attr(0, "value", 42, effective_at=20)
latest = g.current_state_id
capture = g.snapshot_at_commit(first, owned=True)
assert capture.is_owned
assert capture.configuration()["query_timeout_ms"] == 123
del g

copy = capture.to_graph()
assert copy.current_state_id == first
assert copy.has_node_attribute(0, "value")
assert copy.get_node_attr(0, "value") is None
assert copy.snapshot_at_commit(latest).node_attr(0, "value").value == 42
assert copy.configuration()["query_timeout_ms"] == 123
assert copy.is_directed
copy.writer().set_node_attr(0, "value", "independent", effective_at=30)
assert copy.current_state_id > latest
assert capture.to_graph().get_node_attr(0, "value") is None
```

## Capture and restore contract

An owned capture copies the backing pool at capture time, including inactive
allocated IDs, unreferenced retained column values and raw/frozen group entries.
It also retains all captured versions, branches, tags, checkpoints, exact clocks,
external identity reservations, next state/entity IDs, configuration and the
ordinary mutation publication policy. Capturing an older recorded state selects
that state while retaining the other captured storage/history.

`to_graph()` restores an independent owner and selects the captured state on the
copy's selected branch. Other branch heads stay as captured. Later versions from
that selected branch remain in the copied history. New writes use the captured
allocation high-water marks; they do not reuse inactive IDs or numeric version IDs.
The copy has a distinct write provenance, so source views do not acquire authority
to update it. A capture may be restored repeatedly, independently.

The raw backing pool and selected historical values have distinct roles: values
from later accepted versions may remain retained in storage while selected reads
show only the captured recorded state. Membership-only legacy groups remain raw
membership; copying storage does not invent a historical source binding for them.
Frozen group revisions retain their source values and nested dependencies.

Full owned capture requires a clean published/manual checkpoint state. Ordinary
read snapshots can still inspect a recorded version while manual writes are pending.
A lightweight snapshot's `to_graph()` rejects explicitly; request an owned capture
when original writable history/backing storage is needed. Native APIs are
`Graph::owned_snapshot_at_commit` / `owned_snapshot_at_timestamp` and
`TemporalSnapshot::to_graph`. Low-level snapshots constructed directly from
HistoryForest have no owner configuration/backing context.

```python
light = copy.snapshot_at_commit(copy.current_state_id)
assert not light.is_owned
assert light.configuration() == copy.configuration()
try:
    light.to_graph()
except ValueError:
    pass
else:
    raise AssertionError("Full backing capture must be explicit")
```

`DurableGraph.snapshot_at_commit(state, owned=True)` captures the same core Graph
contents and can be used after the store closes. `to_graph()` returns an embedded
Graph. Journal ownership, locks and the store's separately installed schema policy
stay with the durable store; reopen/full-journal backup retain those policies.
These owned captures are transferable between in-memory owners. Persistent raw-pool
archive/import and storage compaction remain roadmap 10.3/10.4, rather than being
approximated by an active graph export.

## Configuration

Python `Graph(config={...})` now decodes and validates the native GraphConfig,
using the same translation as DurableGraph creation. The legacy positional or
`_config` argument is also honored; supplying both aliases rejects. Unknown keys
and wrong types reject. Explicit `directed=` overrides `config.graph_type`.
`Graph.configuration()` and `TemporalSnapshot.configuration()` expose copies.
Preserving a configuration field does not implement an otherwise unavailable
runtime feature configured by that field.

## Repeated switches and resets

State restoration reuses equal retained column values. A lazy per-column hash
lookup is built only when restoration first needs it; hash collisions compare
actual typed values before reuse. Ordinary mutations still append values. Appends
and bulk appends update an existing lookup; low-level positional overwrites or
resizes invalidate it. Repeating the same branch switches or resets does not keep
appending the same properties.

The lookup stores indices, not copies of property payloads. It adds memory
proportional to retained values in restored columns. A cold restoration builds the
lookup; warm state reconstruction still visits selected topology/properties and
history. Neither full captures nor full restores are O(1): explicit owned capture
copies all retained pool contents, and restoring creates an independently writable
pool. Owner-thread Graph/FFI assembly still runs with the GIL; this API does not
promise concurrent mutation during capture.

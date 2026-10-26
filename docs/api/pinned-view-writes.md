# Checked writes through immutable views

`Graph.apply_view_batch(view, batch, message, author, schema=None)` applies a
`GraphBatch` within a current pinned selection. It publishes one accepted state
and returns `state_id`, `added_nodes`, `added_edges` and a refreshed immutable `view`.
The input view remains unchanged. Explicit checked view writes publish even when
ordinary graph mutations use `manual=True`.

```python
from groggy import Graph, GraphBatch

graph = Graph()
a, b, outside = graph.writer().add_nodes_with_attrs([{"score": 1}, {}, {}], effective_at=0)
first, excluded = graph.writer().add_edges([(a, b), (a, b)], effective_at=0)
old = graph.read_view().select_edges([first])
batch = GraphBatch(effective_at=10)
new = batch.add_node(score=None)
batch.add_edge(b, new, weight=2.5)
batch.set_node_attr(a, "score", 1.0000000000000002, effective_at=5)
result = graph.apply_view_batch(old, batch, "selection update", "example")
updated = result["view"]
assert result["state_id"] == updated.write_state_id
assert old.node_attr(a, "score") == 1
assert updated.node_attr(a, "score") == 1.0000000000000002
assert updated.node_ids == [a, b, result["added_nodes"][0]]
assert updated.edge_ids == [first, result["added_edges"][0]]
assert excluded not in updated.edge_ids
assert graph.contains_edge(excluded)
```

## Source and scope

Current read pins carry in-process source-history, branch and accepted-head
provenance. Clones, filters and nested selections share that provenance.
`write_state_id` is the expected source state, or `None` for dirty/standalone
captures. It is a concurrency precondition, not an authorization or durability
promise. A foreign graph, different branch, stale head or dirty current graph
rejects. Dirty captures stay read-only after publication; capture a new view.
Historical/standalone snapshots and unrecorded `Graph.from_snapshot` materializations
are read-only sources for this API. Captures stay readable after source destruction.

Every existing mutation target must belong to the selection: node/edge property
writes, external identity assignments, explicit removals and existing endpoints of
new edges. New entities may be referenced using the usual batch handles. Native
property reference values retain the ordinary batch semantics; they are not extra
mutation targets or implicitly remapped.

Deleting a selected node requires **every edge its deletion would remove** to be
selected, including parallel, loop and external boundary instances. A narrower
selection rejects the whole batch; use a sufficiently broad view or an explicit
owner batch when outside effects are intended. No automatic selection widening.

```python
bad = GraphBatch()
bad.remove_node(a)
# Refresh the same narrow selection at the current head first.
narrow = graph.read_view().select_edges([first])
head = graph.current_state_id
try:
    graph.apply_view_batch(narrow, bad, "unsafe cascade", "example")
except ValueError as error:
    assert "outside the view" in str(error)
else:
    raise AssertionError("excluded parallel edge must prevent deletion")
assert graph.current_state_id == head
assert graph.contains_node(a)
```

The refreshed selection keeps surviving original members plus surviving entities
created by the batch. It does not induce extra owner edges or absorb excluded
parallel edges. Removed and transient add-then-remove entities stay absent. A
subsequent batch uses the returned view; reusing an older head fails explicitly.
An empty batch follows the existing checked-batch rejection policy.

## Atomic publication and returned data

Core batch validation, proposed-state schema checks, identity reservations and
explicit/default effective times use the existing mutation coordinator. All scope
checks and the refreshed immutable payload/membership are constructed **before**
publication. There is no fallible output capture after a successful mutation.
Read values/topology are also compared with the original capture to reject
untracked raw storage changes made after that capture.

This is not a global repair of native `pool_mut`/`space_mut` escapes. Raw untracked
edits made before capturing a pin remain outside the supported checked mutation
contract; the broader migration remains open. Source pins are in-process and are
not serialized write capabilities. A fresh `Graph.read_view()` retrieves current
state explicitly. `add_graph`/table materialization create independent data; a
view clone shares immutable data. Cached adjacency belongs to the frozen payload
and cannot become stale when its owner changes.

## Durable graph

`DurableGraph.read_view()` and `apply_view_batch(...)` use the same provenance,
scope and output preparation. The installed durable schema and optional stricter
per-call schema both apply. The accepted batch/time metadata is appended and
synced before in-memory publication through the existing journal boundary.
Rejected requests append nothing. Recovery replays the ordinary accepted batch;
view pins are not persisted as capabilities. Reopened stores require a new pin.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "selection.ggj"
    with DurableGraph.create(path) as store:
        seed = GraphBatch()
        seed.add_node(score=1)
        store.apply_batch(seed, 0, "seed", "example")
        pinned = store.read_view()
        change = GraphBatch()
        change.set_node_attr(0, "score", 2, effective_at=100)
        accepted = store.apply_view_batch(pinned, change, "change", "example")
        assert accepted["view"].node_attr(0, "score") == 2
        assert pinned.node_attr(0, "score") == 1
    with DurableGraph.open(path) as recovered:
        assert recovered.get_node_attr(0, "score") == 2
```

## Cost and remaining migration

This explicit immutable write path captures and validates the full current read
payload, stages a logical overlay and builds the next frozen adjacency. Its cost
includes full current values/topology, selected membership and batch changes; it
is not an O(1) scalar mutation API. Existing pins retain their own buffers until
released. Ordinary bulk owner writes remain the path for workloads that do not
need a new snapshot after each operation. No extra global cache is introduced.

FFI only translates batches, schemas and results. Current Graph/DurableGraph write
owners require the GIL; frozen read kernels retain their existing GIL release.
Legacy connected Subgraph/table reads, table source-pin propagation and arbitrary
effective-time reconstruction remain separate roadmap work. This API operates on
accepted-state pins. See [verification and bounded release costs](../../notes/planning/pinned-view-writes-2026-10-25.md).

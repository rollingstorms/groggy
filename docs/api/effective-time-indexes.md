# Immutable effective-time indexes

`Graph.build_effective_index()` and `DurableGraph.build_effective_index()` capture
the accepted first-parent lineage of the current branch. The resulting
`EffectiveTimeIndex` supports repeated historical reads, event windows, and
entity activation intervals. Build explicitly after publishing changes, and
build again explicitly when a new capture is needed.

```python
from groggy import Graph, GraphBatch, EffectiveTimeIndex

graph = Graph()
batch = GraphBatch(effective_at=10)
batch.add_node(score=1)
first = graph.writer().apply_batch(batch)["state_id"]
batch = GraphBatch(effective_at=30)
batch.set_node_attr(0, "score", 3)
graph.writer().apply_batch(batch)
batch = GraphBatch(effective_at=20)
batch.set_node_attr(0, "score", 2)
graph.writer().apply_batch(batch)

index = graph.build_effective_index()
assert isinstance(index, EffectiveTimeIndex)
assert index.branch == "main"
assert index.captured_head == graph.current_state_id
assert index.read_view_at(25).node_attr(0, "score") == 2
assert index.read_view_at(35).node_attr(0, "score") == 3
```

## Time coordinates and event rows

All timestamps are signed Unix microseconds. Booleans, floats, strings and values
outside i64 reject. Effective time orders changes; recorded time is the engine's
acceptance clock. A `recorded_by` cutoff applies only when explicitly supplied to
a query. Equal effective times use accepted lineage order, so the later accepted
assignment to the same key wins. Events themselves retain all accepted changes.

`events_between(start, end)` uses inclusive bounds `[start, end]`.
`node_events(id, start, end)` and `edge_events(...)` use entity-local postings.
Results follow effective time and canonical replay order. Reversed bounds reject,
including for unknown IDs. Unknown IDs otherwise return an empty list; inactive
IDs remain inspectable.

Each owned dictionary has `kind`, `entity_id`, `name`, `state_id`, `effective_at`,
and `recorded_at`. `kind` identifies node/edge addition/removal, edge endpoint
revision, or node/edge property change. `name` identifies a property and is None
for other changes. These
rows identify logical changes; their retained deltas hold the value payloads.
Assignments and unsets both have the corresponding property-change kind.

```python
events = index.node_events(0, 20, 30)
assert [row["effective_at"] for row in events] == [20, 30]
assert all(row["kind"] == "node_attr" and row["name"] == "score"
           for row in events)
assert len(index.events_between(20, 20)) == 1
cutoff = graph.publication_metadata(first)["recorded_at"]
assert index.read_view_at(25, recorded_by=cutoff).node_attr(0, "score") == 1
assert all(row["recorded_at"] <= cutoff
           for row in index.events_between(0, 100, recorded_by=cutoff))
```

The existing `build_temporal_index()` is a separate recorded-state selector with
its established state-ID/recording-time coordinates. Use `build_effective_index()`
for the per-change effective-time API described here. A single `GraphView` is a
captured state/selection; building its property/topology index does not recover a
whole historical lineage. See [view indexes](query-indexes.md).

## Structural activation

`node_activation_intervals(id)` and `edge_activation_intervals(id)` return
half-open `(start, end)` intervals `[start, end)`, with None for an open end.
An explicit recorded cutoff is also supported. Add/remove events at one effective
instant resolve in accepted replay order; zero-duration gaps/intervals coalesce.

```python
graph.writer().remove_nodes([0], effective_at=40)
new_index = graph.build_effective_index()
assert new_index.node_activation_intervals(0) == [(10, 40)]
assert new_index.node_events(0, 40, 40)[0]["kind"] == "node_removed"
assert index.node_activation_intervals(0) == [(10, None)]
assert index.read_view_at(50).node_attr(0, "score") == 3
assert new_index.read_view_at(50).node_ids == []
```

Activation describes structural add/remove history. For value/absence intervals
and explicitly stored boolean validity, see [property intervals](property-intervals.md).
See [group-membership indexes](group-membership-indexes.md) for retained source
membership and nested reverse references.
An edge activation interval alone does not certify endpoint
validity. `read_view_at()` applies the canonical complete affected endpoint
timeline checks and frozen group-source checks before returning a graph.

## Ownership, cost and recovery

Captures retain branch/head, precise clocks, frozen group sources and immutable
deltas. Later writes, branch switches, owner destruction, durable close/reopen,
and owner-side pruning do not retarget an existing index. `copy()` shares its
retained data and postings. Historical views carry their original read clocks
and have no implicit current-head write authority. Retained data stays in memory
while its readers need it.

Indexes are transient read objects: no automatic mutation maintenance, persisted
definitions or recovery rebuild. Reopening storage requires an explicit new
build for new readers. Construction and queries do not publish graph deltas,
append journal records, or label checkpoints. Pending manual changes must first
be explicitly published. Missing legacy clocks/source bindings use the same
explicit `legacy_times` and `group_sources` inputs as
[effective-time reads](effective-time-reads.md); known precise data cannot be
overridden. Construction validates event clocks; point reconstruction additionally
validates the selected timeline and source references.

The native implementation shares the existing effective replay engine. The
global sorted index and entity postings bound event-window searches by binary
search plus candidates visited (a recorded cutoff may discard candidates).
Activation uses dedicated lifecycle postings. Point reconstruction still replays
the selected event prefix and builds a graph snapshot; indexing does not make
whole-state reconstruction logarithmic. Index construction retains lineage and
builds postings explicitly; it adds no automatic work to ordinary graph mutations.
Native computation releases the Python GIL; result marshalling runs with it held.

Advanced native callers can use `EffectiveReadPlan::build_index()` after explicit
plan preparation. The prepared effective timestamp does not truncate the index.
A prepared recorded cutoff rejects: supply that optional cutoff on each query.

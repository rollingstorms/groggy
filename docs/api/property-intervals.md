# Effective-time property intervals

An explicit `EffectiveTimeIndex` now provides
`node_property_intervals(id, name, recorded_by=None)` and
`edge_property_intervals(...)`. Each result describes when the named property
held a value during that entity's active lifetimes. All bounds use signed Unix
microseconds, as in [effective-time reads](effective-time-reads.md).

```python
from groggy import Graph, GraphBatch

graph = Graph()
batch = GraphBatch(effective_at=10)
a, b = batch.add_node(), batch.add_node()
batch.add_edge(a, b, valid=True)
graph.writer().apply_batch(batch)
batch = GraphBatch(effective_at=30)
batch.set_edge_attr(0, "valid", False)
graph.writer().apply_batch(batch)
index = graph.build_effective_index()
assert index.edge_property_intervals(0, "valid") == [
    {"start": 10, "end": 30, "present": True, "value": True},
    {"start": 30, "end": None, "present": True, "value": False},
]
```

## Bounds, absence and null

Rows are owned dictionaries with `start`, `end`, `present`, and `value`. Intervals
are half-open `[start, end)`; None for `end` means open-ended. An absent property
has `present=False, value=None`. Explicit null has `present=True, value=None`.
Inactive periods produce no rows. A property never assigned to a known active
entity has absence rows covering its active lifetimes; an unknown ID returns [].

```python
batch = GraphBatch(effective_at=15)
batch.set_node_attr(0, "note", None)
graph.writer().apply_batch(batch)
batch = GraphBatch(effective_at=20)
batch.unset_node_attr(0, "note")
graph.writer().apply_batch(batch)
notes = graph.build_effective_index().node_property_intervals(0, "note")
assert notes == [
    {"start": 10, "end": 15, "present": False, "value": None},
    {"start": 15, "end": 20, "present": True, "value": None},
    {"start": 20, "end": None, "present": False, "value": None},
]
assert index.node_property_intervals(999, "note") == []
```

Equal effective times use accepted replay order. One instant resolves completely
before producing a row, so intermediate zero-duration values do not appear.
Adjacent intervals with the same exact typed payload coalesce. Floating widths,
signed-zero bits and NaN payloads remain distinct, using the engine's retained
value comparison rather than numeric coercion. Supported compact storage aliases
retain their existing logical equivalence. Property types may change over time.
For the uncoalesced accepted operations and their state/recorded coordinates, use
`node_events`/`edge_events` on the same index.

Creation/restoration starts a new property lifetime. Assignments outside an
active lifetime are ignored, as in canonical reconstruction. Values before
deactivation do not implicitly carry into a later activation; an explicit
restoration's property assignments determine its values. A removal/restoration
at the same timestamp can leave a continuous interval if the final value agrees.

## Recorded cutoffs and immutable captures

`recorded_by` restricts engine acceptance time only when supplied. The captured
branch/head already bounds every result. Backdated effective writes can correct
an interval in a new index without changing the old capture.

```python
first_state = index.captured_head
cutoff = graph.publication_metadata(first_state)["recorded_at"]
batch = GraphBatch(effective_at=25)
batch.set_edge_attr(0, "valid", False)
graph.writer().apply_batch(batch)
new_index = graph.build_effective_index()
assert new_index.edge_property_intervals(0, "valid")[0]["end"] == 25
assert index.edge_property_intervals(0, "valid")[0]["end"] == 30
assert new_index.edge_property_intervals(0, "valid", recorded_by=cutoff) == \
       index.edge_property_intervals(0, "valid")
```

Graph and DurableGraph use the same native index. Durable reopen requires an
explicit new build; existing indexes remain readable after owner close. Legacy
clock/source inputs and source pins follow the effective-index contract. Returned
containers can be edited without changing future results. Native work releases
the Python GIL; owned Python value conversion holds it. Invalid IDs/clocks/types
reject without publishing changes.

## Implementation and scope

The index stores property-specific postings pointing into retained deltas, plus
separate lifecycle postings. A query merges just the named property's events and
the entity's lifecycle events in canonical order. It does not scan unrelated
properties, replay whole graph snapshots, or duplicate property payloads at build
time. Per-index build cost includes one posting per property event and a map per
entity/property key; result values are cloned only for emitted rows. Large values
also incur exact comparison and output-copy costs. Clones share the postings.

These intervals describe individual stored values. A boolean property such as
`valid` can express fact validity explicitly, as above. Cross-property/domain
inference and automatic text/vector indexing are separate concerns. Edge
intervals do not certify endpoint validity; use canonical `read_view_at()` when
a valid whole graph is required. Group-membership indexes remain open in 18.2.
No index definitions are persisted or maintained automatically during mutation.

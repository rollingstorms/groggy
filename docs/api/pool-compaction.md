# Explicit pool compaction

`Graph.compact_pool()` removes duplicate retained property slots and returns
before/after node and edge slot counts. `Graph.optimize()` performs the same
maintenance and returns `None`. Both require an accepted, clean working state.

```python
import groggy as gr

g = gr.Graph()
g.add_node(value=1)
for _ in range(5):
    g.set_node_attr(0, "value", 1)
state = g.current_state_id
frozen = g.read_view()
report = g.compact_pool()
assert report["current_node_values"] < report["previous_node_values"]
assert g.current_state_id == state
assert frozen.node_attr(0, "value") == 1
assert g.optimize() is None
```

This is exact portable-value deduplication within each column: signed zero, NaN
payloads, scalar/vector widths, null, text and JSON spelling retain their value
identity. Internal small-value/compressed representations may share a slot when
they represent the same portable value. The first matching representation wins;
hashes only select candidates and collisions are checked for exact equality.

Every distinct raw value remains, including values without a current active
mapping. Branches, versions, inactive entity IDs, group references, external ID
reservations and all historical deltas remain. This does not delete a branch,
purge an entity, release an external name or discard unreachable distinct data.
Immutable graph/table read pins own their frozen values and remain readable.

Raw column offsets can change. Callers using low-level pool offsets must acquire
them again after maintenance; node/edge IDs and logical graph versions do not
change. All working mappings are remapped, including inactive mappings.

The explicit operation scans raw values and temporarily copies retained column
payloads. The native prepare/execute/finish API permits detached computation;
Python releases the GIL during execution. Capture and installation run on the
owner. A stale or foreign detached result rejects before changing storage.
Before/after counts measure slots, not bytes, unique history values or peak RSS.
Future restoration/writes can grow the columns again.

## Durable physical compaction

`DurableGraph.compact_storage(expected_state=...)` deduplicates raw columns and
writes a full retained baseline through the existing locked, synced Unix atomic
replacement path. It returns the ordinary storage checkpoint fields and a nested
`pool_compaction` dictionary with the slot report. `checkpoint_storage()` retains
its original meaning: replace the operation stream while preserving raw slots.

```python
import tempfile
from pathlib import Path
from groggy import DurableGraph, Graph

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    source = Graph()
    source.add_node(value=1)
    for _ in range(5):
        source.set_node_attr(0, "value", 1)
    with DurableGraph.create_from_graph(path, source) as store:
        state = store.current_state_id
        report = store.compact_storage(expected_state=state)
        slots = report["pool_compaction"]
        assert slots["current_node_values"] < slots["previous_node_values"]
        assert store.current_state_id == state
    with DurableGraph.open(path) as store:
        assert store.current_state_id == state
        assert store.check_integrity()["active_nodes"] == 1
```

The source owner rejects operations while detached maintenance runs. A failure
before replacement preserves both the original graph layout and named file.
After replacement, an error requires reopen; the replacement inode remains locked
until the owner/outcome is released. A successful operation installs the same
compacted layout in memory and in the verified file. Versions, schema revision,
publication clocks, group sources, inactive IDs and identity reservations remain.
A backup and subsequent appends remain recoverable.

Capture owns both a full portable baseline and a raw-column compaction input, so
this operation may temporarily use substantial additional memory. Execution and
file verification release Python's GIL. File size may shrink or grow with the
fixture and prior operation stream; retaining history still costs storage. No
automatic schedule, non-Unix atomic replacement or actual power-loss guarantee is
introduced by this API.

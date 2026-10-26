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

## Explicit orphan-value reclamation

Pass `reclaim_unreferenced=True` to `compact_pool` or `compact_storage` to discard
raw slots whose values are no longer referenced by **any retained version** or
working mapping in that column. This is an explicit destructive option; the default
remains deduplication with every distinct raw value retained.

```python
from groggy import Graph

g = Graph()
g.add_node(value=0)
g.create_branch("discard")
g.checkout_branch("discard")
g.set_node_attr(0, "orphan", "payload")
frozen = g.read_view()
g.checkout_branch("main")
# First explicitly prune the branch/history that owns the unwanted value.
g.prune_branch("discard")
report = g.compact_pool(reclaim_unreferenced=True)
assert report["reclaimed_unreferenced_node_values"] == 1
assert frozen.node_attr(0, "orphan") == "payload"
assert g.get_node_attr(0, "value") == 0
```

Both before and after values of every retained delta protect matching raw values.
Unreferenced versions still retained in history also protect values until explicit
history collection removes those versions. Every working mapping, including an
inactive mapping, remains a root. Exact portable identity and column/kind scope
apply; a numerically equal value under another column does not protect an orphan.

Frozen graph/table/group sources own independent payloads. Reclaiming an orphan
slot from the owner's pool does not delete those payloads or prune their sources.
Group registries, entity ID extents and external-name reservations remain retained.
Empty property columns remain declared; this operation does not migrate a schema.
A detached reclaiming result also rejects if the retained-history set changes.

The report's `reclaimed_unreferenced_node_values` and
`reclaimed_unreferenced_edge_values` count discarded orphan slots, including any
repeated slots for an orphan value. Remaining slot reduction is deduplication.
Native callers use `compact_pool_with_reclamation(true)` or
`compact_storage_with_reclamation(expected_state, true)`; the no-argument/default
methods preserve every distinct value. For durable graphs, raw-value removal uses
the same verified atomic replacement and failure protocol described above.

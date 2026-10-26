# Explicit history collection and branch pruning

`Graph.gc_history()` now removes unreachable recorded versions and unreferenced
recorded deltas. `Graph.prune_branch(name)` explicitly removes a non-current,
non-main branch reference and then collects all unreachable history. Both return
the number of recorded versions removed and require a clean state. They do not
publish another graph version or rewind lifetime allocation counters.

```python
from groggy import Graph

g = Graph()
g.add_node(value=0)
root = g.current_state_id
g.create_branch("discard")
g.checkout_branch("discard")
g.set_node_attr(0, "value", 1)
side = g.current_state_id
pin = g.read_view()
g.checkpoint("a label is not a permanent pin", "example")
g.checkout_branch("main")
assert g.prune_branch("discard") == 1
assert g.current_state_id == root
assert pin.node_attr(0, "value") == 1
assert g.checkpoints() == []
assert g.gc_history() == 0
try:
    g.snapshot_at_commit(side)
except ValueError:
    pass
else:
    raise AssertionError("Explicitly pruned state lookup must fail")
```

## Reachability and immutable ownership

Surviving branch/tag references retain their complete ancestry, including both
parents of a resolved merge. Native HistoryForest.garbage_collect also retains the
ancestry of explicitly supplied additional state IDs. State zero is the empty
root, not an actual commit. Shared ancestry and another branch's unique versions
are not collected. Native children/root/delta indexes are cleaned in one pass;
read/topology caches are invalidated once when versions are removed.

Existing immutable GraphViews, TemporalSnapshots, TemporalIndexes and prepared
EffectiveReadPlans keep their owned values/Arc payloads. A cold plan prepared
before collection can still reconstruct afterward. They do not impose permanent
pins on the owner's history lookup. Explicit owned graph captures retain their
own copied history and can still be restored independently.

```python
source = Graph()
source.add_node(value=0)
first = source.current_state_id
source.set_node_attr(0, "value", 9)
future = source.current_state_id
capture = source.snapshot_at_commit(first, owned=True)
copy = capture.to_graph()
assert copy.gc_history() == 1
assert capture.to_graph().snapshot_at_commit(future).node_attr(0, "value").value == 9
assert source.snapshot_at_commit(future).node_attr(0, "value").value == 9
```

Checkpoint labels referencing collected states are removed; they are not extra
audit/retention pins (the accepted A09 rule). There is no mandatory grace period
or automatic backup. Current/main/unknown branch pruning, dirty input and stale
durable state coordinates reject before mutation or journal I/O. Branch pruning
collects all unreferenced versions after deleting that reference, including any
unreferenced versions already present in an explicitly owned copy.

## Durable replay

DurableGraph exposes `prune_branch(name, expected_state=...)` and
`gc_history(expected_state=...)`. Omitting expected_state selects the current head.
Preparation calculates the exact sorted removal set; a checked collection record
is appended and synced before exposing changes. Recovery recomputes and verifies
that set. Four injected append/sync failure phases roll back bytes before any
branch/history change; retry and reopen preserve the accepted result. A no-op
collector writes no frame. The existing reopen-required rule covers indeterminate
rollback failures.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create(path) as stored:
        stored.writer().add_node(value=0)
        head = stored.current_state_id
        stored.create_branch("discard", head)
        stored.checkout_branch("discard", head)
        stored.writer().set_node_attr(0, "value", 1)
        stored.checkout_branch("main", stored.current_state_id)
        assert stored.prune_branch("discard", expected_state=head) == 1
    with DurableGraph.open(path) as recovered:
        assert recovered.current_state_id == head
        assert len(recovered.commit_history()) == 1
```

## Separate pool and physical maintenance

This is **history-only collection**. Raw entity slots, property column values,
stored groups and external-ID reservations remain allocated. A removed state's
lookup/reconstruction is unavailable through the owner; previously captured views
remain readable. History pruning does not release an old external name for reuse.
The journal still contains earlier frames and grows; replay applies the explicit
collection record. This does not physically erase or compact the file.

Use [explicit retained-storage purge](retained-purge.md) to reclaim orphan entities,
group registrations and names after pruning, and [pool compaction](pool-compaction.md)
for exact column compaction or optional orphan-value reclamation. Durable variants
use verified physical replacement; `checkpoint_storage` alone preserves raw slots.
Format/platform migration breadth and automatic scheduling remain separate work.
No automatic collection is enabled. History-only owner-thread collection/FFI
assembly retains its current GIL limitation.

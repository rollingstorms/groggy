# Explicit retained-storage purge

`Graph.preview_purge()` calculates a report without changing the graph.
`Graph.purge_unreferenced()` installs that same candidate. Both require a clean,
accepted state. Purge does not prune history or create a version/checkpoint.

```python
from groggy import Graph

g = Graph()
g.add_node(value=0)
g.create_branch("discard")
g.checkout_branch("discard")
orphan = g.add_node(value=9)
g.add_edge(0, orphan)
g.writer().set_node_external_id(orphan, "reusable")
pin = g.read_view()
g.checkout_branch("main")
assert g.preview_purge()["purged_nodes"] == 0  # retained branch owns its history
assert g.prune_branch("discard") > 0
state = g.current_state_id
preview = g.preview_purge()
assert g.is_node_allocated(orphan)
assert g.purge_unreferenced() == preview
assert g.current_state_id == state
assert not g.is_node_allocated(orphan)
assert pin.node_attr(orphan, "value") == 9
new = g.add_node()
assert new > orphan  # internal node/edge IDs are never recycled
```

## Roots and names

Every retained delta protects its entity targets, edge endpoints and before/after
property values. Nonhead and otherwise unreferenced versions remain roots until
explicit history collection removes them. Working mappings and retained property
references protect group registrations. Legacy membership groups also protect
owner-domain members/endpoints. Frozen group sources own separate payloads and ID
domains: removing an owner registration does not erase a nested source or an open
immutable graph/table/owned snapshot.

External names remain reserved while current properties or any retained history
reference the binding. Explicit purge releases names with no such root, including
orphaned previous names of otherwise retained entities. Name reuse does not reuse
the old internal ID. Typed node/edge ID arrays preserve their numeric values;
they do not secretly pin referents (the existing property-value contract).

```python
g.writer().set_node_external_id(new, "reusable")
assert g.node_by_external_id("reusable") == new
report = g.check_integrity()
assert report["purged_nodes"] == 1
assert report["purged_edges"] == 1
# Allocation counts are lifetime extents; has_node/has_edge describe active state.
assert g.is_node_allocated(new)
assert not g.has_node(orphan)
```

Reports contain newly purged half-open `node_ranges`/`edge_ranges`, counts,
removed group registration IDs, released names and reclaimed orphan raw property
slot counts. Deduplication reductions are separate from orphan slot counts.
Existing purge intervals are not counted twice. Owner raw offsets can change;
reacquire them after maintenance. Group references use content identities and may
be explicitly registered again from the same immutable content.

## Durable replacement and recovery

`DurableGraph.purge_unreferenced(expected_state=...)` returns storage-checkpoint
fields plus a nested `purge` report. It writes, syncs, verifies and atomically
replaces a complete retained baseline, then installs the identical memory layout.
The source owner is unavailable during detached maintenance. Before replacement,
a failure preserves original memory/file bytes; after replacement, an error
requires reopen. Existing writer locks cover both inodes through replacement.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create_from_graph(path, g) as store:
        state = store.current_state_id
        receipt = store.purge_unreferenced(expected_state=state)
        assert receipt["purge"]["purged_nodes"] == 0
    with DurableGraph.open(path) as store:
        assert not store.is_node_allocated(orphan)
        assert store.node_by_external_id("reusable") == new
        assert store.verify_storage()["graph"]["purged_nodes"] == 1
```

Portable full archives, owned graph captures, journal reopen and backups preserve
purge intervals and monotonic allocation counters. Legacy document v1 remains
readable; ledger-bearing dense archives use v2 and sparse endpoint archives use
v3. Ordinary nonpurged archives retain the v1 shape. Active-only exports remain
separate selected-data transfers.

## Cost and scope

This is explicit destructive maintenance, with no automatic history pruning,
schedule, permanent audit pins, grace period or mandatory backup. Capture and
exact stale-source validation encode/copy retained storage on the owner thread.
Candidate computation and durable I/O release Python's GIL. Peak temporary memory
can be substantial; this is not a mutation hot path.

After edge purging, raw endpoints use a sparse map so interior slots are reclaimed
and high lifetime IDs do not require an empty vector prefix on later insertion.
Ordinary dense graphs retain their existing layout. Sparse lookup trades dense
index locality for reclaimed memory; active traversal still uses GraphSpace's
CSR/indexing. Purged allocation ledgers use intervals rather than per-ID flags.
The local receipt in `notes/benchmarks/retained-purge-2026-10-26.json` measures
native/durable maintenance, not isolated RSS or a universal latency budget.
Unix atomic replacement and process-kill recovery are tested; automatic scheduling,
non-Unix replacement and actual power-loss qualification remain separate work.

Archives containing [typed endpoint revisions](endpoint-revisions.md) use document
v4. Readers accept v1–v4, and purge/checkpoint/compaction preserve that requirement
while retained revision records exist. Both old and new endpoint dependencies are
retention roots; ordinary no-revision formats/hashes remain compatible.

# Explicit journal storage checkpoints

`DurableGraph.checkpoint_storage(expected_state=None)` rewrites the journal into
one complete retained baseline. Subsequent checked changes append normally. It
does **not** create a graph version, change clocks, prune history, release IDs,
remove raw property values, or add a graph checkpoint label. The existing
`checkpoint()` remains an explicit graph-state annotation.

```python
from groggy import DurableGraph, GraphBatch
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "analysis.ggj"
    with DurableGraph.create(path) as store:
        batch = GraphBatch(effective_at=10)
        batch.add_node(score=1)
        store.apply_batch(batch, 0, "root", "example")
        state = store.current_state_id
        report = store.checkpoint_storage(expected_state=state)
        assert report["selected_state"] == state
        assert report["retained_versions"] == 1
        assert store.current_state_id == state
    with DurableGraph.open(path) as store:
        assert store.current_state_id == state
        assert store.read_view_at(10).node_attr(0, "score") == 1
```

The baseline contains all retained history/branches and original clocks, current
selected column indices, complete raw pool/ID/identity reservations, nested frozen
sources, graph configuration, graph checkpoint labels, current journal schema and
its revision. Open immutable payloads remain valid. A pruned version's original
operation frames no longer occur in the rewritten file, but retained raw columns
can still contain its values. This is not secure erasure or pool reclamation.

The report contains `previous_bytes`, `current_bytes`, `previous_records`,
`retained_versions`, and `selected_state`. Rewriting metadata/history can make a
file **larger**; compare measured sizes rather than assume compression.

```python
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "empty.ggj"
    with DurableGraph.create(path) as store:
        store.set_schema(None, 0, 0)
        first = store.checkpoint_storage()
        second = store.checkpoint_storage()
        assert first["selected_state"] == 0
        assert second["previous_records"] == 1
        assert store.schema_revision == 1
    with DurableGraph.open(path) as store:
        assert store.schema_revision == 1
```

## Replacement and failure behavior

The operation currently supports Unix and requires the actual journal path,
not a symlink. A stale expected state rejects before file changes. It prepares a
new sibling file, writes/syncs it with source permissions, verifies recovery, and
locks its inode before atomic replacement. The original inode stays locked until
the replacement is attached; the directory is then synced. A changed journal path
rejects instead of replacing an unrelated file.

Before replacement, an I/O/verification failure leaves the original journal and
writer usable. After replacement, any reported failure marks the writer unusable:
close and reopen the journal. Either retained baseline is complete; a directory
sync error means the replacement's crash persistence is uncertain. The tests
inject returned failures at five boundaries; they do not emulate a power failure.

Python performs detached file I/O and candidate verification outside the GIL.
Source capture/encoding stays on the owner thread. Native runtime integrations
can use the hidden prepare/execute/finish path; an abandoned plan/outcome leaves
the original owner requiring reopen rather than permitting stale appends.

## Remaining storage scope

Scheduling remains **explicit**, as agreed for MVP/first use. This replaces operation
replay with archive loading; parsing and validating retained data still scales
with retained data/history. It is not a constant-time opening or memory guarantee.
The baseline uses the existing **256 MiB frame limit**; oversize preparation rejects
without changing the source. Multi-frame/chunked recovery, format migrations,
non-Unix replacement, raw pool reclamation and optimized/automatic checkpoint
policy remain open under the existing storage/retention roadmap items.

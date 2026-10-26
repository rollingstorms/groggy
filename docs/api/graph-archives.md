# Complete retained graph archives

`Graph.save_archive(path)` writes a versioned, checksummed binary CBOR archive.
`Graph.load_archive(path)` restores an independent embedded graph. The destination
must be new; save validates and encodes first, syncs a sibling temporary, publishes
without replacing an existing file, and syncs the directory. A directory-sync
failure can leave a complete destination: inspect it before retrying.

```python
from groggy import Graph
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as directory:
    graph = Graph(directed=True)
    node = graph.add_node(score=1.5)
    first = graph.current_state_id
    graph.create_branch("first")
    graph.set_node_attr(node, "score", 2.5)
    path = Path(directory) / "analysis.gga"
    graph.save_archive(path)
    restored = Graph.load_archive(path)
    assert restored.get_node_attr(node, "score") == 2.5
    restored.checkout_branch("first")
    assert restored.get_node_attr(node, "score") == 1.5
    assert graph.get_node_attr(node, "score") == 2.5
```

The archive retains:

- configuration and automatic/manual publication policy;
- selected state/branch, all retained versions and deltas, accepted/effective clocks,
  merge parents, branches/tags and explicit checkpoint labels;
- raw attribute columns **with their positions**, including unused/older values,
  inactive allocation slots, sparse ID high-water marks and identity reservations;
- legacy membership-only groups and frozen nested source revisions/aliases;
- logical property types, exact floating bits, explicit null versus absence,
  structured/vector values and typed entity/group references.

Memory reuse caches, CSR/query caches, owner provenance and optimized value storage
representations are rebuilt. Compact/compressed values decode to their equivalent
logical types. Columns are not deduplicated, compacted or silently dropped. Open
views in a separate process are not serialized as external pins.

Manual pending edits reject: explicitly accept them through the existing manual
workflow first. Saving never publishes them or invents effective timestamps.

```python
with tempfile.TemporaryDirectory() as directory:
    graph = Graph(manual=True)
    graph.add_node(name="pending")
    path = Path(directory) / "manual.gga"
    try:
        graph.save_archive(path)
    except ValueError:
        assert not path.exists()
    else:
        raise AssertionError("dirty manual graph must reject")
    graph.commit("accept edits", "example")
    graph.save_archive(path)
    restored = Graph.load_archive(path)
    restored.add_node()
    assert restored.has_uncommitted_changes()
```

The existing `save_to_path/load_from_path` remains an **active JSON interchange**
format; it does not recover retained backing storage or original history. Journal
backup remains a journal operation, including its schema policy. An embedded
archive does not contain a DurableGraph lock, journal schema or schema revision.

Native `Graph.prepare_archive()` returns detached `PreparedArchive` bytes that can
be written on another thread. Python file I/O/checksumming releases the GIL;
graph capture/restore uses its thread-local owner. Explicit archives copy retained
storage and validate it; they are not a constant-time ordinary mutation path or
a bounded-recovery checkpoint. Typed node/edge ID arrays keep their stored IDs;
frozen sources keep their independent source ID domains. Version 1 rejects unknown versions and malformed,
truncated, checksum-invalid, dangling or inconsistent data. Future migrations and
physical journal replacement remain separate work.

## Continue an existing graph in a durable journal

```python
from groggy import DurableGraph, GraphBatch

with tempfile.TemporaryDirectory() as directory:
    source = Graph()
    source.add_node(name="retained")
    state = source.current_state_id
    path = Path(directory) / "import.ggj"
    with DurableGraph.create_from_graph(path, source) as store:
        assert store.current_state_id == state
        batch = GraphBatch()
        batch.add_node(name="continued")
        store.apply_batch(batch, state, "continue", "example")
        latest = store.current_state_id
    with DurableGraph.open(path) as store:
        assert store.current_state_id == latest
        assert store.snapshot_at_commit(state).node_attr(0, "name").value == "retained"
```

`DurableGraph.create_from_graph(path, graph, schema=None)` validates and stores the
complete source as one synced logical initialization record, with no fresh acceptance clock
or ID remapping. It leaves the source independent. Optional journal schema applies
to the imported selected state and subsequent writes; historical snapshots retain
their original values. Large journal baselines use bounded 8 MiB physical fragments rather than a
256 MiB aggregate limit. Dirty/schema-invalid sources reject before destination
creation. Initialization writes and syncs a locked sibling temporary, then publishes
without replacing an existing destination. Encoding failures do not leave a partial
named journal. Standalone archives keep their separate framing. Later records,
reopen, locking, partial-tail recovery and backup use the existing journal protocol.
Old journals remain readable; older binaries cannot read the new initialization
record. Storage migrations and recovery-checkpoint scheduling remain separate work.

## Explicit purge compatibility

Full archives also retain permanent purge intervals and compact sparse raw
endpoints after [explicit purge](retained-purge.md). Document readers accept legacy
v1, v2 purge ledgers and v3 sparse endpoint records; ordinary nonpurged writes
retain the v1 shape. Downgrades, malformed/overlapping purge intervals, duplicate
sparse records and retained references to purged IDs reject. Active-only exports
remain selected-data transfers with their own identity scope.

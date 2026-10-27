# Restore an explicitly selected retained state

`Graph.restore_state(state, expected_state=..., schema=...)` restores the working
graph and retargets its current branch to a retained published state ID. The state
argument is required. It restores active membership, ordered endpoints, typed
properties, external names and group references. State zero is the empty root,
not a retained published version, and rejects.

This is a reference/state switch. It does not publish an inverse mutation, create
a checkpoint, alter original effective/acceptance clocks, or delete history.
Other branches, retained future versions, ID allocation counters, external-name
reservations and immutable views survive. New writes from the restored state use
new monotonic IDs/versions and that selected ancestry. Graph configuration and
manual/automatic publication policy remain the owner's current configuration.

```python
from groggy import Graph

g = Graph()
g.add_nodes(2)
g.add_edge(0, 1, weight=7)
first = g.current_state_id
g.set_edge_endpoints(0, 1, 0)
g.set_node_attr(0, "value", "future")
future = g.current_state_id
pin = g.read_view()
versions = len(g.commit_history())
assert g.restore_state(first, expected_state=future) == first
assert g.edge_endpoints(0) == (0, 1)
assert pin.edge_endpoints(0) == (1, 0)
assert len(g.commit_history()) == versions
assert g.snapshot_at_commit(future).node_attr(0, "value").value == "future"
assert g.writer().restore_state(future) == future
assert g.get_node_attr(0, "value") == "future"
```

## Guards and dirty manual work

Omitting `expected_state` captures the current selected head at entry. A writer
session always supplies its captured head and optional schema; successful
restoration retargets that session. Stale heads, changed selected branches, foreign
owners, unknown/collected targets and malformed input reject before installation.
Boolean state IDs reject as type errors. An optional schema validates the complete
target, not only changed fields.

Pending manual changes reject, even when the target is already selected. Explicitly
call `reset_hard()` to discard those unaccepted changes first. An unchanged clean
target validates successfully without installing another state.

```python
manual = Graph(manual=True)
manual.add_node()
target = manual.commit("initial", "example")
manual.set_node_attr(0, "pending", True)
try:
    manual.restore_state(target)
except ValueError:
    pass
else:
    raise AssertionError("Pending manual edits require an explicit decision")
assert manual.has_uncommitted_changes()
manual.reset_hard()
assert manual.restore_state(target) == target
assert not manual.has_uncommitted_changes()
```

## Retention remains explicit

Restoring an earlier state does not make future versions immediately unreadable.
A later explicit `gc_history()` can collect versions no surviving branch/tag
references. Existing immutable payloads remain readable independently. Checkpoint
labels are not permanent pins. `purge_unreferenced()` and physical storage
compaction are separate operations; they never happen implicitly during restore.

```python
h = Graph()
h.add_node(value=0)
old = h.current_state_id
h.set_node_attr(0, "value", 1)
new = h.current_state_id
immutable = h.read_view()
h.restore_state(old)
assert h.snapshot_at_commit(new).node_attr(0, "value").value == 1
assert h.gc_history() == 1
assert immutable.node_attr(0, "value") == 1
assert h.get_node_attr(0, "value") == 0
```

## Durable replay and detached execution

`DurableGraph.restore_state` and its writer expose the same selection semantics.
The currently installed schema always applies; a per-call schema can add checks
but cannot weaken it. A checked reference record is appended and synced before
selection changes. Replay checks its original head, branch, target and schemas.
Append failures roll back bytes and leave memory untouched; indeterminate rollback
or unexpected installation failures require reopen. Restoring an already selected
clean state writes no journal frame. Reopen, backup and baseline replacement retain
the selected state and all still-retained history. Older readers do not understand
the new restore record and must reject it rather than silently skip it.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create_from_graph(path, g) as store:
        assert store.writer().restore_state(first) == first
        before = path.read_bytes()
        assert store.restore_state(first) == first
        assert path.read_bytes() == before
        store.verify_storage()
    with DurableGraph.open(path) as store:
        assert store.current_state_id == first
        assert store.snapshot_at_commit(future).node_attr(0, "value").value == "future"
```

Native callers may split work through `prepare_state_restore`,
`StateRestorePlan.execute` and `finish_state_restore`. The plan/outcome are owned
`Send + Sync` values with no mutable graph access. Reconstruction runs without the
Python GIL; capture, current-policy/reference validation and installation remain
owner operations. Logical values survive intervening valid pool compaction.
Installation rejects if the captured target was collected or the owner/head/branch
changed. This is a full retained-state replay and assembly, not a constant-time
pointer switch. It can require memory proportional to the selected graph.

A published state ID is distinct from an effective timestamp. Use
`read_view_at(effective_microseconds, recorded_by=...)` for an immutable reconstructed
clock selection. Restoration here changes the working graph to an explicit
published version; general tracked historical reverts remain a separate 13.3 item.

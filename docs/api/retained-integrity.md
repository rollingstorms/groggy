# Explicit retained-graph integrity checks

`Graph.check_integrity()` verifies a clean accepted graph and every retained
recorded state, including versions that are not branch/tag heads. It returns a
receipt with counts; invalid data raises an error and is not repaired.

```python
from groggy import Graph

graph = Graph()
graph.add_node(value=-0.0)
graph.create_branch("old")
graph.set_node_attr(0, "value", 2.0)
before = graph.current_state_id
report = graph.check_integrity()
assert report["selected_state"] == before
assert report["retained_versions"] == 2
assert report["verified_states"] == 3  # includes empty state zero
assert report["branches"] == 2
assert report["active_nodes"] == 1
assert report["schema_revision"] is None
assert graph.current_state_id == before
```

## Checked scope

- The recovery validator checks portable types, configuration, raw allocation and
  property indices, retained group closure, identities, parent/reference closure,
  content hashes, publication clock keys, branches and the selected current state.
- A separate walk verifies every retained first-parent transition: before-values,
  entity lifecycle, allocated endpoint identity, endpoint liveness and visible
  property ownership. A merge's delta remains its accepted resolution relative to
  the first parent; all retained parents/versions are checked independently.
- Errors from the deep walk identify the recorded state and entity/property. Invalid
  raw property encoding identifies the node/edge column and value offset, even
  when the value is not currently selected.
- Untimed manual versions remain supported and are counted under `legacy_versions`;
  this does not invent effective timestamps. Pending manual edits reject until
  explicitly accepted or discarded.

The receipt includes selected state/branch, active and allocated node/edge counts,
raw column/slot counts, retained/verified/legacy versions, branch/group/identity
references and historical maximum active node/edge counts. Allocated counts are ID
extents, including inactive/reserved IDs. Value counts are **raw slots**, not unique
values, facts or memory estimates; frozen-source payload values are excluded from
those slot counts. Group references include aliases.

## Durable graphs

`DurableGraph.check_integrity()` checks its accepted **in-memory retained graph**
and captured current schema policy/revision. It does **not reread an already open
journal file**. Closing and reopening verifies physical framing/checksums/recovery;
backup/replacement and OS writer-lock rules remain unchanged. A schema is checked
against the selected state, not retroactively imposed on every historical version.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, GraphBatch

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create(path) as durable:
        batch = GraphBatch()
        batch.add_node(value=1)
        durable.apply_batch(batch, 0, "initial", "example")
        durable.set_schema({}, durable.current_state_id, 0)
        before = path.read_bytes()
        receipt = durable.check_integrity()
        assert receipt["current_schema_checked"]
        assert receipt["schema_revision"] == 1
        assert path.read_bytes() == before
```

## Ownership and cost

This is an explicit full audit. It does not publish a version, consume IDs, prune,
rebuild secondary indexes, validate derived caches, or schedule maintenance. Native
`prepare_integrity_check()` freezes a portable input; `IntegrityCheckPlan.execute()`
can run on another thread and reports the captured state even if the source later
changes. Python releases the GIL during detached verification; capture stays on the
owner thread and copies retained portable data.

A linear verification walk reuses an owned snapshot; only first-parent snapshots
still needed by descendants are retained. Forks may require full snapshot copies,
and interleaved branch frontiers can increase memory. State validation scans
current topology/property ownership at each version. Cost therefore grows with
retained history and state sizes, not merely the size of the current graph. This
is not a cheap per-write hook or a universal time/memory guarantee.

Secondary-index/cache consistency, in-place physical-file verification, future
migration/platform coverage and service observability remain in the existing
index/storage/33.2 roadmap scope.

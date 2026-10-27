# Stable edge endpoint revisions

An endpoint revision changes an existing edge's ordered `(source, target)` pair.
Its ID, properties, relationship type and external identity remain. Node/edge
allocation counters do not advance. Parallel edges remain distinct; loops are
valid. `reverse_edges` swaps the ordered pair for each selected edge, preserving
graph direction. An undirected reversal changes stored orientation without
changing its undirected reachability.

```python
from groggy import Graph, GraphBatch

g = Graph()
batch = GraphBatch(effective_at=10)
a, b, c = [batch.add_node() for _ in range(3)]
batch.add_edge(a, b, weight=1)
g.apply_batch(batch, 0, "initial", "example")
pin = g.read_view()
first = g.current_state_id
g.writer().set_edge_endpoints(0, 2, 1, effective_at=20)
assert g.edge_endpoints(0) == (2, 1)
assert pin.edge_endpoints(0) == (0, 1)
assert g.read_view_at(19).edge_endpoints(0) == (0, 1)
assert g.read_view_at(20).edge_endpoints(0) == (2, 1)
assert g.get_edge_attr(0, "weight") == 1
assert g.diff_commits(first, g.current_state_id)["edge_endpoint_changes"] == [
    {"edge_id": 0, "old": (0, 1), "new": (2, 1)}
]
g.reverse_edges([0], effective_at=30)
assert g.edge_endpoints(0) == (1, 2)
```

## One batch and one clock default

Python/Rust writers expose `set_edge_endpoints` and
`set_edge_endpoints_bulk([(edge, source, target), ...])`. Ordinary Graph methods
follow its automatic/manual publication policy; writers always publish checked
changes. Explicit `effective_at` on Python direct conveniences selects checked
publication and requires a clean state. Native callers select explicit times
through a writer or GraphBatch. Empty ordinary bulk/reversal inputs are no-ops;
empty explicit writer batches remain invalid.

`GraphBatch.set_edge_endpoints` accepts an existing edge ID and existing node IDs
or node handles created by that batch. New edges already specify their initial
pair through `add_edge`; they are not endpoint-revision targets in a checked
batch. Repeated IDs, inactive/unknown targets, explicit edge-removal conflicts and
endpoints removed in the resulting graph reject the whole batch. Rewiring away
from a removed node is supported. Schema validation covers the complete result.

```python
batch = GraphBatch(effective_at=40)
new = batch.add_node()
batch.set_edge_endpoints(0, 1, new, effective_at=45)
batch.remove_node(2, effective_at=45)
batch.set_edge_attr(0, "weight", 2)
g.apply_batch(batch, g.current_state_id, "mixed change", "example")
assert g.edge_endpoints(0) == (1, 3)
assert g.read_view_at(44).edge_endpoints(0) == (1, 2)
assert g.read_view_at(45).edge_endpoints(0) == (1, 3)
assert not g.has_node(2)
```

Each accepted revision records typed before/after endpoints and an
`edge_endpoints` effective-time key. Without explicit times, one engine clock
supplies the batch default. Equal effective timestamps use later accepted order.
Assigning the same pair still records an event, as same-value property writes do.
Revisions before edge creation/after effective deletion and endpoint lifetime
contradictions reject before mutation or journal I/O. Later backdated removals
cannot turn an earlier accepted revision into an implicit edge resurrection.
Reads do not create mutation deltas.

## Manual compatibility and immutable ownership

Manual ordinary setters retain pending changes. Repeated revisions coalesce their
first old and final new pair. Revising a newly created pending edge changes its
initial pair; deleting a pending revised edge discards the unobservable revision.
`reset_hard` restores accepted endpoints and properties. A clean commit remains
an explicit checkpoint; accepting pending manual edits retains legacy untimed
history semantics and needs explicit legacy inputs for effective-time reads.

```python
manual = Graph(manual=True)
manual.add_nodes(3)
manual.add_edge(0, 1)
manual.set_edge_endpoints(0, 2, 1)
accepted = manual.commit("initial", "example")
manual.set_edge_endpoints(0, 0, 1)
assert manual.has_uncommitted_changes()
manual.reset_hard()
assert manual.edge_endpoints(0) == (2, 1)
assert manual.current_state_id == accepted
```

Recorded/effective views, owned captures and frozen group sources retain their own
endpoint pairs. Working GraphSpace invalidates topology caches even when edge IDs
and membership do not change. TemporalIndex stores endpoint timelines for recorded
neighbor/window queries; TemporalDelta and builder edge-diff/changed-entity paths
include endpoint revisions. Endpoint changes count as edge changes, not property
writes or node mutations. Ordinary connected Subgraph migration remains separate.

Legacy membership-only groups have no frozen source revision. Rewiring their
member edges explicitly rejects; it cannot silently reinterpret those groups.
Unreferenced registrations may be explicitly purged. Frozen-source groups are
supported. Raw `GraphPool.add_edge_with_id` remains a reconstruction helper and
rejects untracked overwrite. Trusted recorded-state installation restores the
revision selected by history; the raw current pair is not a timeless edge identity.

## Durable storage and retention

Use a DurableGraph writer or mixed GraphBatch for the same checked mutation.
The journal records the endpoint operation and original clocks; sync precedes
memory publication. Reopen, backup, explicit checkpoint/compaction and purge
preserve revisions, properties and identity. Retention traces both old and new
endpoint nodes. No implicit history pruning, ID reuse or new graph version is
introduced by maintenance.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create_from_graph(path, g) as store:
        state = store.writer().set_edge_endpoints(0, 0, 1, effective_at=50)
        store.checkpoint_storage(expected_state=state)
        store.compact_storage(expected_state=state, reclaim_unreferenced=True)
        store.purge_unreferenced(expected_state=state)
    with DurableGraph.open(path) as store:
        assert store.read_view_at(50).edge_endpoints(0) == (0, 1)
        store.verify_storage()
```

Full archives containing endpoint revisions use document v4. Readers accept v1–v4;
old no-revision delta hashes/document shapes remain compatible, and old binaries
cannot read the new fields/version. Load verifies shape/hashes/references/selected
state; `check_integrity` additionally audits every retained before-state. Durable
batch replay recomputes before endpoints from its accepted parent. Endpoint-specific
append fault tests cover four write/sync rollback phases. The existing generic
18-case process-kill maintenance matrix remains separate from actual power-loss
qualification.

Local release receipt: `notes/benchmarks/endpoint-revisions-2026-10-27.json`.
A 100k-edge batch measured ~166 ms mutation and ~102 ms effective read, excluding
setup, input construction and old-view capture. This is not FFI/RSS/whole-engine
latency. Bulk endpoint writes still execute on the owner thread, as existing
checked writes do; broader GIL migration remains under 19.2. Destructive rewind
and general tracked revert remain separate 13.3 work; three-way merging is 14.2.

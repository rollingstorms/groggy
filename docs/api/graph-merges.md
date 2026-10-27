# Checked two-parent graph merges

`prepare_merge(right_state, base=None, expected_state=None)` captures the selected
working state as the first parent and an explicit retained right-hand version.
It returns an immutable-input, one-shot `MergePlan`; reconstruction runs without
the GIL. `plan.resolve()` handles conflict-free inputs. Explicit conflicts require
a caller callback. `finish_merge(outcome, ...)` validates and publishes exactly one
version with both parents, then retargets the selected branch.

Independent node/edge/property changes combine. Separate IDs stay separate parallel
facts, including equal endpoint pairs and loops. Existing allocated IDs, external
names, labels/kinds and frozen source references survive; no new entity IDs are
allocated. Other branches and open immutable views retain their state. Automatic
or manual mutation policy is preserved; pending manual edits must be handled
explicitly before merging.

```python
from groggy import Graph

g = Graph()
g.add_node(value=0)
base = g.current_state_id
g.create_branch("right")
g.set_node_attr(0, "left", 1)
left = g.current_state_id
g.checkout_branch("right")
g.set_node_attr(0, "right", 2)
right = g.current_state_id
g.checkout_branch("main")
pin = g.read_view()
plan = g.prepare_merge(right, expected_state=left)
assert plan.base_state == base and plan.conflicts == []
outcome = plan.resolve()
merged = g.finish_merge(outcome)
assert g.get_node_attr(0, "left") == 1
assert g.get_node_attr(0, "right") == 2
assert pin.node_attr(0, "right") is None
assert g.snapshot_at_commit(left).node_attr(0, "right") is None
assert plan.is_consumed and outcome.is_consumed
g.check_integrity()
```

## Bases, conflicts and whole typed values

A missing base selects only a unique best common ancestor. Multiple best bases
(criss-cross ancestry) require an explicit common ancestor. Unrelated retained
roots require the explicit empty base `base=0`. Parents must be distinct published
states; unknown/zero parents and non-common explicit bases reject.

`plan.conflicts` is a copied list of dictionaries. Kinds are `node`, `edge`,
`node_attr`, `edge_attr`, `edge_endpoints` and provisional `topology`. Entity
conflicts carry complete base/left/right records or None for absence. Property
slots carry `present`, and when present, `value` and `dtype`; absence is distinct
from an explicit null. Endpoint conflicts carry ordered pairs.

A resolver receives each conflict dictionary. Returning `"base"`, `"left"` or
`"right"` selects the original native typed value/record exactly, preserving float
width, signed zero and NaN payload bits. Custom values use normal Python property
conversion (ordinary floats default to f64). There is no default winner.

```python
h = Graph()
h.add_node(value=0)
h.create_branch("other")
h.set_node_attr(0, "value", 1)
h.checkout_branch("other")
h.set_node_attr(0, "value", 2)
other = h.current_state_id
h.checkout_branch("main")
plan = h.prepare_merge(other)
conflict = plan.conflicts[0]
assert conflict["kind"] == "node_attr"
assert conflict["base"]["present"] and conflict["base"]["value"] == 0
outcome = plan.resolve(lambda conflict: "right")
h.finish_merge(outcome)
assert h.get_node_attr(0, "value") == 2
```

Custom answers are explicit dictionaries:

- `{"kind": "property", "present": True, "value": None}` assigns null;
  `{"kind": "property", "present": False}` unsets the property.
- `{"kind": "entity", "value": record_or_None}` selects a complete entity or
  explicit deletion. A record has `properties` and `endpoints` (None for a node,
  an ordered pair for an edge).
- `{"kind": "endpoints", "value": (source, target)}` revises an edge pair.
- `{"kind": "drop_edge"}` explicitly removes a dangling edge.
- `{"kind": "endpoint_nodes", "nodes": [{"id": id, "properties": {...}}]}`
  explicitly retains exactly the missing endpoint nodes. These must already be
  allocated IDs in this graph. Extra/duplicate IDs and missing references reject.

After direct answers, topology is recomputed: initial dependencies can disappear
and new ones can arise. Topology callbacks require concrete drop/retain answers;
a side string never implicitly cascades deletion or resurrects endpoint nodes.

```python
t = Graph()
t.add_nodes(2)
t.create_branch("with_edge")
t.remove_node(1)
t.checkout_branch("with_edge")
t.add_edge(0, 1)
with_edge = t.current_state_id
t.checkout_branch("main")
plan = t.prepare_merge(with_edge)
assert plan.conflicts[0]["kind"] == "topology"
outcome = plan.resolve(lambda conflict: {
    "kind": "endpoint_nodes",
    "nodes": [{"id": 1, "properties": {"retained": True}}],
})
t.finish_merge(outcome)
assert t.edge_endpoints(0) == (0, 1)
assert t.get_node_attr(1, "retained") is True
```

## Resolution clocks and first-parent reconstruction

The merge records an exact typed delta against its first parent, rather than
replaying every right-parent mutation into the selected effective timeline. The
right parent's original history and clocks remain readable by explicit state ID.
The accepted merge has one engine `recorded_at`; `effective_at` defaults to that
clock. Explicit per-change overrides use the same list shape as batch writes and
must target actual resolution changes. Duplicates/unrelated keys reject.

Initial attributes inherit a newly activated node/edge's effective time unless
overridden. Edge removals caused by removed endpoints inherit the earliest removed
endpoint time unless overridden. Complete affected topology timelines are validated
before publication, including backdated endpoint/lifetime conflicts. Accepted-order
ties retain the existing later-write-wins rule. `recorded_by` remains an explicit
read option; it is not automatically added to effective reads.

```python
from groggy import GraphBatch

clocked = Graph()
batch = GraphBatch(effective_at=10)
batch.add_node(value=0)
root = clocked.apply_batch(batch, 0, "root", "example")["state_id"]
clocked.create_branch("other")
clocked.writer().set_node_attr(0, "left", 1, effective_at=20)
clocked.checkout_branch("other")
clocked.writer().set_node_attr(0, "right", 2, effective_at=30)
other = clocked.current_state_id
clocked.checkout_branch("main")
outcome = clocked.prepare_merge(other).resolve()
clocked.finish_merge(outcome, effective_at=40, change_times=[{
    "kind": "node_attr", "entity": 0, "name": "right", "effective_at": 35,
}])
assert clocked.read_view_at(34).node_attr(0, "right") is None
assert clocked.read_view_at(35).node_attr(0, "right") == 2
```

Even a data-identical merge publishes one version to join two distinct parents;
its resolution delta and change-time map are empty. A subsequent clean commit or
checkpoint labels that state without publishing another version. No entity ID
is consumed by either operation. Retention follows both merge parents, including
when the right-hand branch is later pruned.

## Guarded acceptance and one-shot handles

Capture, resolution and read-only preflight do not mutate the owner. Finish
rechecks owner identity, selected branch/head, retained parents/base, allocated IDs,
frozen references, external identities, complete schema and effective timelines.
Legacy membership-only groups reject unsupported raw endpoint revisions; use
frozen source revisions. Stale/foreign/dirty/invalid outcomes reject before graph
mutation or journal I/O. Another branch advancing does not change an explicitly
selected right-parent version while that version remains retained.

Plans/outcomes are consumed once, including a failed resolution or native finish.
Python argument-conversion failures occur before outcome consumption. Native
callback failures propagate the original Python exception. A callback can itself
perform user-authorized graph work; if it advances the owner, later finish rejects
the now-stale outcome. Re-plan explicitly to retry. Input inspection dictionaries
are copies, and side selection reads the native captured values.

## Durable writer, recovery and schema

Graph and DurableGraph expose the same prepare/resolve/finish surface. Their Python
writers also provide `writer.merge(right, base=..., resolver=..., effective_at=...,
change_times=..., message=...)` and retarget the session after success. Native
GraphWriter offers `prepare_merge`/`finish_merge` with its author/schema/default
clock. Durable finish always enforces the currently installed schema plus an
optional stricter per-call schema; an old plan cannot weaken a changed policy.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.ggj"
    with DurableGraph.create_from_graph(path, g) as store:
        # Data is already combined, but this explicitly joins retained ancestry.
        version = store.writer().merge(right)
        store.verify_storage()
    with DurableGraph.open(path) as store:
        assert store.current_state_id == version
        assert store.snapshot_at_commit(left).node_attr(0, "left").value == 1
        store.check_integrity()
```

The journal stores the exact first-parent resolution delta, parent/base/branch
coordinates, original options/clocks and any additional schema. It does not write
the complete merged graph for a small change. Append/sync precedes memory
publication. Replay reconstructs the first parent, applies the declared delta,
recomputes its exact changes and verifies before/after values, clocks, policies and
coordinates before publishing. False before-values, redundant/duplicate changes,
unknown parents, wrong branches or malformed metadata reject without repairing
complete corrupted frames. Torn tails retain the existing recovery rule.

Four injected append failures and five actual SIGKILL boundaries cover pre-append,
partial header/payload, synced-before-memory and installed states. Retry/reopen,
writer locks, backup, baseline replacement, compaction, retained history and
continued monotonic allocation are verified. Indeterminate rollback or unexpected
post-sync installation errors require reopen. Older readers reject unknown merge
records; complete archives use version 5 when exact delta hash collisions require it.

Planning/delta reconstruction and native resolution assembly run outside the GIL,
with brief reentry for Python callbacks. Owner validation/schema checks and final
installation remain owner-bound. Full replay, candidate assembly and installation
still cost time/memory proportional to selected graph data. The 256 MiB per-record
journal limit applies to a large resolution delta; chunking/platform/service/scale
work remains in its existing roadmap scopes.

# Retained snapshot diffs

`temporal.diff_nodes` and `temporal.diff_edges` now support atomic property
pipelines. The shared native `TemporalDelta` also reports actual logical property
differences, replacing its empty property-change placeholders.

## Inputs and source identity

Supply `before` and `after` snapshot variable names. Without explicit names, the
existing native temporal Context supplies a reference snapshot (or window start)
and current retained state. Python specs can create both variables with
`temporal.snapshot_at`:

```python
import json
from groggy import Graph, AttrValue
from groggy.algorithms.base import AlgorithmHandle

class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}

graph = Graph()
a = graph.add_node(score=1)
before = graph.current_state_id
b = graph.add_node(score=1)
graph.set_node_attr(a, "score", 2)
after = graph.current_state_id
steps = [
    {"id": "temporal.snapshot_at", "params": {"state_id": before, "output": "before"}},
    {"id": "temporal.snapshot_at", "params": {"state_id": after, "output": "after"}},
    {"id": "temporal.diff_nodes", "params": {"before": "before", "after": "after", "output_prefix": "diff"}},
    {"id": "core.attach_node_attr", "params": {"source": "diff_nodes_added", "attr": "added"}},
    {"id": "core.attach_node_attr", "params": {"source": "diff_nodes_modified", "attr": "modified"}},
]
graph.view().apply(Steps(steps), persist=True)
assert graph.get_node_attr(b, "added") == 1
assert graph.get_node_attr(a, "modified") == 1
assert graph.current_state_id == after + 1
```

Snapshots must belong to the invocation owner's **same history instance**. Native
`TemporalDelta::compute` requires both inputs to share that instance. Numeric
state IDs alone cannot prove identity: unrelated graphs commonly use identical
IDs. A private shared token enforces provenance without cloning history, changing
storage formats or exposing a mutable owner. Cloned snapshots preserve it and
remain usable after their original Graph is dropped. A separately loaded or
copied Graph has a new instance and cannot be silently compared using its IDs.

Comparison is an explicit before→after state comparison. Reverse comparisons and
sibling branches in the same history are valid; numeric ordering does not imply
ancestry or elapsed time. It does not reconstruct arbitrary effective time.

## Output masks and logical values

With `output_prefix` (default `diff`), node diffs produce:

- `diff_nodes_added`: IDs present only in after, value 1.
- `diff_nodes_removed`: IDs present only in before, value 1.
- `diff_nodes_modified`: surviving IDs with at least one changed property, value 1.

Edge diffs use corresponding `diff_edges_*` edge maps. Existing added/removed
names remain stable; modified masks are additional outputs. A membership change
is not also a property-modified event. The native TemporalDelta carries sorted
property records with actual old/new logical values, stable entity-ID/name order,
and exact typed comparison using AttrValue semantics. Absence is None; explicit
null is Some(Null). Type, vector/structured and width changes follow logical
property equality. Summary/is_empty/affected entity APIs now include these records.

Results describe **net differences**. A value changed and then restored between
the selected states produces no property record. Entity creation and deletion
between two identical memberships is similarly absent. This is distinct from an
event log. Endpoint rewiring of a surviving edge currently returns an explicit
error because TemporalDelta has no endpoint-change record; roadmap 04.2/13.3
already tracks that capability.

```python
same = [
    {"id": "temporal.snapshot_at", "params": {"state_id": after, "output": "same"}},
    {"id": "temporal.diff_nodes", "params": {"before": "same", "after": "same"}},
    {"id": "core.attach_node_attr", "params": {"source": "diff_nodes_modified", "attr": "unexpected"}},
]
head = graph.current_state_id
graph.view().apply(Steps(same), persist=True)
assert graph.current_state_id == head
assert graph.get_node_attr(a, "unexpected") is None
```

## Scope, writes and failure

Like the retained snapshot inputs, diffs cover the **full owner state**. They do
not clip historical entities to the live invocation selection. Removed IDs remain
readable in result maps even when inactive now. Explicit attachments use the
existing owner-ID validation boundary: writing removed inactive IDs rejects;
active owner IDs can be addressed outside the selected view. This is intentional
full-owner history comparison, rather than a selected-topology traversal.

Earlier staged writes do not alter retained snapshots. Diff reads record no new
history. Persistent attachments share one enclosing acceptance, and transient
attachments return owned outputs:

```python
_, profile = graph.view().apply(Steps(steps), persist=False, return_profile=True)
assert profile["outputs"]["node_attributes.added"] == [(b, 1)]
assert profile["outputs"]["node_attributes.modified"] == [(a, 1)]
```

Missing/wrong-typed/foreign snapshots reject before installing results. Private
added/removed/modified maps are built before any destination changes. Cooperative
cancellation checks cover entity/property traversal, mask construction and final
installation; sorting and large value comparisons/clones have no bounded latency
promise. Atomic pipeline failure discards earlier staged property assignments.

The shared kernel borrows immutable membership/property maps instead of cloning
all source sets. It owns changed old/new values and sorts output IDs/property
records. Costs scale with compared entities, properties, changed payloads and
sorting. Each node/edge mask helper currently computes the full logical delta;
calling both repeats that work. No cache, source pin migration, whole-engine
performance improvement or certification of window/index helpers is implied.

# Checked temporal builder primitives

`temporal.decay`, `temporal.snapshot_at`, `temporal.diff_nodes` and
`temporal.diff_edges` support property-atomic algorithms,
loops and certified for-each bodies. They calculate private results and install
the destination only after validation and a final cancellation check. Explicit
attachments publish property changes through the enclosing algorithm's single
acceptance boundary; transient runs return owned property outputs.

## Decay maps and columns

Decay takes `attr` (a variable name), `half_life`, `output`, and optional
`time_delta` (default 0). Both time controls use the same caller-chosen unit.
The function is `value * exp(-ln(2) * (time_delta / half_life))`.
It does not read a clock, infer entity age, or generate historical deltas until
results are explicitly attached. This preserves the explicit-time change model.

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
graph.add_nodes([{}, {}])
steps = [
    {"id": "core.init_nodes", "params": {"target": "values", "value": 8}},
    {"id": "temporal.decay", "params": {"attr": "values", "half_life": 2.0, "time_delta": 4.0, "output": "result"}},
    {"id": "core.attach_node_attr", "params": {"source": "result", "attr": "score"}},
]
before = graph.current_state_id
graph.view().apply(Steps(steps), persist=True)
assert [graph.get_node_attr(n, "score") for n in range(2)] == [2.0, 2.0]
assert graph.current_state_id == before + 1
```

Sources may be sparse node maps, ordered node columns, or sparse edge maps.
Outputs retain the entity kind and keys; columns retain row order. Native callers
can construct `algorithms::steps::NodeColumn`, now re-exported beside
`StepVariables`. Duplicate column IDs reject. Source and destination names may
match. Every source ID must be active in the invocation selection, and edge
endpoints must be active selected nodes. Empty sources still validate controls.

Half-life must be finite and positive; elapsed time must be finite and
nonnegative. Values must be finite integer/floating scalars; booleans, null, text
and collections reject. Results are f64, including integer sources (large
integers therefore have f64 rounding). Invalid values no longer pass through
unchanged. Dividing elapsed time by half-life first handles subnormal half-lives
at elapsed=0 without NaN. Underflow to zero is allowed for long elapsed times.

Loading a graph property follows the existing staged overlay, so earlier
attachments are visible when explicitly loaded:

```python
staged = [
    {"id": "core.init_nodes", "params": {"target": "values", "value": 12}},
    {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "score"}},
    {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values", "default": 0}},
    {"id": "temporal.decay", "params": {"attr": "values", "half_life": 2.0, "time_delta": 4.0, "output": "values"}},
    {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "decayed"}},
]
before = graph.current_state_id
graph.view().apply(Steps(staged), persist=True)
assert graph.get_node_attr(0, "decayed") == 3.0
assert graph.current_state_id == before + 1
```

## Retained immutable snapshots

`temporal.snapshot_at` takes `output` and exactly one of `state_id` or `timestamp`.
State IDs are retained numeric identities. Timestamp selects the latest retained
acceptance at or before that **epoch second**, using greatest state ID for equal
seconds across the owner's retained history. It currently searches all retained
branches. Use an explicit state ID when branch identity matters. No qualifying
state is an error. Supplied coordinates must be nonnegative integers; floats,
booleans and negative values reject rather than truncate or wrap.

The result is a native immutable `TemporalSnapshot` variable covering the full
owner graph, including historical entities outside the current selection. It
excludes staged property writes and unaccepted manual working changes. It does
not switch the working graph, publish a checkpoint, or imply certification of
other temporal consumers. There is no new Python snapshot output conversion in
this change; Python can inspect retained states using the existing graph API.

```python
retained = graph.current_state_id
steps = [
    {"id": "core.init_nodes", "params": {"target": "values", "value": 20}},
    {"id": "core.attach_node_attr", "params": {"source": "values", "attr": "score"}},
    {"id": "temporal.snapshot_at", "params": {"state_id": retained, "output": "saved"}},
]
graph.view().apply(Steps(steps), persist=True)
assert graph.get_node_attr(0, "score") == 20
assert graph.snapshot_at_commit(retained).node_attr(0, "score").value == 12
```

This timestamp selector retains the existing acceptance-history behavior. It does
not reconstruct per-property effective time; use `Graph.read_view_at` for that
separate reconstruction API. Retained snapshot reconstruction cost depends on history, graph size and
checkpoint layout. Cancellation checks surround reconstruction; reconstruction
itself has no bounded cancellation latency promise.

## Existing temporal factories and remaining work

Supplied `before`, `after`, `output_prefix`, `index_var`, and `change_type` must
have their declared text type. Omitted optional values retain existing defaults.
`temporal.filter` state coordinates are nonnegative integers and range start must
not exceed end. Mark-changed types are `created`, `deleted`, `modified`, or `any`.
These checks run during preparation before any step can publish.

Node/edge diff semantics are documented in [Retained snapshot diffs](builder-temporal-diffs.md).
The four remaining helpers are now completed as described in
[Retained temporal windows](builder-temporal-windows.md). All eight native temporal
primitives support atomic property execution. Broader effective-time reconstruction
and pipeline/history/index lifecycle requirements retain their original scope.

Decay is O(source rows) time and output storage; ordered columns also use an
O(rows) duplicate-ID set. It does not clone the owner/history or build indexes.
Cancellation checks cover each decay row and pre-installation; allocation and
snapshot reconstruction do not promise bounded interrupt latency.

## Accepted history reads after pending edits

`temporal.snapshot_at`, `temporal.diff_nodes` and `temporal.diff_edges` may follow
pending topology in local or outer atomic programs and nested compute frames.
They read accepted retained history, including explicit sibling-branch state IDs;
they never give provisional entities an accepted creation state. Earlier staged
values, endpoint revisions, additions and removals do not alter these inputs.
Missing, purged or foreign-history sources still reject through the existing
validators, with whole-run rollback at the owning atomic boundary.

`graph.capture_staged` remains the explicit read of pending working state. Its
values and retained snapshots are separate immutable inputs. Historical result
IDs remain exact IDs: attaching a removed/out-of-frame ID fails rather than
filtering or mapping it onto a provisional entity. Exact child write restrictions
also apply to history maps. Selected lifecycle/window helpers retain their pending
topology guards until the staged-entity history policy is settled.

```python
from groggy.builder import step_pipeline


def history_step(kind, **params):
    return {"id": kind, "params": params}


history_graph = Graph()
history_graph.add_nodes(2)
before = history_graph.current_state_id
history_graph.set_node_attr(0, "score", 3)
accepted = history_graph.current_state_id
_, profile = history_graph.apply(step_pipeline("accepted versus pending", [
    history_step("graph.add_nodes", count=1, output="new"),
    history_step("core.init_nodes", value=99, target="values"),
    history_step("core.attach_node_attr", source="values", attr="score"),
    history_step("temporal.snapshot_at", state_id=before, output="before"),
    history_step("temporal.snapshot_at", state_id=accepted, output="after"),
    history_step("temporal.diff_nodes", before="before", after="after", output_prefix="diff"),
    history_step("core.attach_node_attr", source="diff_nodes_modified", attr="accepted_change"),
    history_step("graph.from_snapshot", source="after", output="old"),
    history_step("graph.emit", source="old", name="history"),
    history_step("graph.capture_staged", output="pending"),
    history_step("graph.emit", source="pending", name="working"),
], atomic=True), return_profile=True)
assert history_graph.current_state_id == accepted + 1
assert sorted(profile["outputs"]["history"].node_ids) == [0, 1]
assert profile["outputs"]["history"].node_attr(0, "score") == 3
assert sorted(profile["outputs"]["working"].node_ids) == [0, 1, 2]
assert profile["outputs"]["working"].node_attr(0, "score") == 99
assert history_graph.get_node_attr(0, "accepted_change") == 1
assert history_graph.get_node_attr(2, "accepted_change") is None
```

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
not reconstruct arbitrary per-property effective time; that roadmap work remains
open. Retained snapshot reconstruction cost depends on history, graph size and
checkpoint layout. Cancellation checks surround reconstruction; reconstruction
itself has no bounded cancellation latency promise.

## Existing temporal factories and remaining work

Supplied `before`, `after`, `output_prefix`, `index_var`, and `change_type` must
have their declared text type. Omitted optional values retain existing defaults.
`temporal.filter` state coordinates are nonnegative integers and range start must
not exceed end. Mark-changed types are `created`, `deleted`, `modified`, or `any`.
These checks run during preparation before any step can publish.

Node/edge diff semantics are documented in [Retained snapshot diffs](builder-temporal-diffs.md).
Window aggregation, lifecycle filtering/marking and temporal-window steps remain
uncertified for atomic property runs. They need an audit of branch/ancestry
semantics, index provenance, value histories and state-ID versus effective/
acceptance-time coordinates. Strict factory validation alone does not close those
concerns.

Decay is O(source rows) time and output storage; ordered columns also use an
O(rows) duplicate-ID set. It does not clone the owner/history or build indexes.
Cancellation checks cover each decay row and pre-installation; allocation and
snapshot reconstruction do not promise bounded interrupt latency.

# Retained temporal windows and lifecycle queries

The remaining four native temporal primitives support atomic property pipelines:
`temporal.window_aggregate`, `temporal.filter`, `temporal.mark_changed_nodes`, and
`temporal.window`. Together with snapshots, decay and diffs, this completes the
fixed inventory of eight existing temporal primitives.

## One retained branch and one write boundary

All four read a retained **first-parent lineage**, matching existing snapshot
reconstruction. By default its head is the owner's accepted current state. A
native Context temporal scope can select an explicit retained head. Sibling
branch changes are excluded. State coordinates must belong to that lineage;
zero denotes its empty root. Windows use inclusive **state-ID bounds**, not epoch
seconds or arbitrary property effective times. Gaps between numeric IDs are not
expanded or treated as missing snapshots. Reversed/unknown/foreign-branch bounds
reject before installing results.

Results cover the invocation's selected node IDs. Inactive historical IDs remain
inspectable if known to the retained lineage; live unpublished nodes have no
retained history. Unknown foreign IDs reject. Explicit attachments still require
active owner IDs. Reads ignore earlier staged properties, and persistent writes
from all helpers share the enclosing atomic pipeline's single acceptance.

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

def attach(source, attr):
    return {"id": "core.attach_node_attr", "params": {"source": source, "attr": attr}}

graph = Graph(manual=True)
graph.add_nodes([{"score": 1, "label": "index_0", "nullable": None}, {}])
graph.commit("first", "example")
first = graph.current_state_id
graph.set_node_attr(0, "score", 3)
graph.add_node()
graph.commit("second", "example")
second = graph.current_state_id
steps = [
    {"id": "temporal.window_aggregate", "params": {"attr": "score", "function": "sum", "start": first, "end": second, "output": "sum"}},
    attach("sum", "sum"),
    {"id": "temporal.filter", "params": {"predicate": "existed_at", "commit": first, "output": "existed"}},
    attach("existed", "existed"),
    {"id": "temporal.mark_changed_nodes", "params": {"start": second, "end": second, "change_type": "any", "output": "changed"}},
    attach("changed", "changed"),
    {"id": "temporal.window", "params": {"start": first, "end": first, "output": "window"}},
    attach("window", "window"),
]
graph.view().apply(Steps(steps), persist=True)
assert graph.current_state_id == second + 1
assert graph.get_node_attr(0, "sum") == 4.0
assert [graph.get_node_attr(n, "existed") for n in range(3)] == [1, 1, 0]
assert [graph.get_node_attr(n, "window") for n in range(3)] == [1, 1, 0]
assert graph.get_node_attr(2, "changed") == 1
```

`window_aggregate` and `mark_changed_nodes` accept paired optional `start`/`end`
parameters (new explicit access to their existing native window behavior).
Without these, a native Context must supply its window. Native constructors retain
existing signatures and add `with_window(start, end)`. `temporal.window` is now
registered using required start/end/output. Optional bounds require actual
nonnegative integers; partial pairs and floating truncation reject.

## Aggregation counts actual property-change records

Aggregation accepts attr, function, output and optional index_var (default
`temporal_index`). Functions are count, first, last, sum, avg/average, min and max.
Only recorded assignments to that property inside the window contribute. An
unchanged state contributes nothing, and there is no snapshot-per-ID fallback or
invented carry-in value. Reads create no deltas. Repeated recorded assignments are
not silently changed into time-weighted samples or reconstructed effective-time
values.

Count includes explicit null and returns zero for no changes. First/last preserve
the chosen logical step value, including literal `index_` text and explicit null;
no changes returns None. Property types unsupported by AlgorithmParamValue reject
for first/last rather than disappearing. Count can count those values without
converting them.

Numeric reducers use the same checked reduction kernel as other builders. They
accept finite integer/floating scalars and return f64; null/Boolean/text/collections
and nonfinite values reject instead of being silently skipped. Checked accumulation
preserves large integer cancellation, and a finite mean can succeed when a sum
would overflow. Empty numeric windows return None. Sum overflow errors.

```python
count = [
    {"id": "temporal.window_aggregate", "params": {"attr": "score", "function": "count", "start": first, "end": second, "output": "count"}},
    attach("count", "count"),
    {"id": "temporal.window_aggregate", "params": {"attr": "label", "function": "first", "start": first, "end": second, "output": "label"}},
    attach("label", "first_label"),
]
_, profile = graph.view().apply(Steps(count), persist=False, return_profile=True)
assert profile["outputs"]["node_attributes.count"] == [(0, 2), (1, 0), (2, 0)]
assert dict(profile["outputs"]["node_attributes.first_label"])[0] == "index_0"
```

## Lifecycle and exact event types

Filter predicates remain created_after(commit), created_before(commit),
existed_at(commit), and modified_in_range(start, end). Creation means the first
activation on the captured lineage. The legacy modified_in_range predicate means
**any recorded node event**, including creation/deactivation; use mark_changed_nodes
with change_type=modified for property events only. Filter and mark/window outputs
are dense selected node maps with integer 0/1 values.

Mark change_type is created, deleted, modified, any, or omitted=any. These tests
use actual delta membership. A node with created+modified events is not marked
as deleted merely because its summary is Multiple. Generated identity/property
assignments at creation count as property events too. Native ChangedEntities now
combines event types across additions and merges rather than overwriting them;
its Multiple summary does not replace the exact event tests.

Window existence uses every activation interval, with creation included and
deletion excluded. A node deactivated at state D does not exist at D; a later
restoration opens a new interval, leaving the intervening gap inactive. Point
property lookups cannot leak an old activation's attributes into a restoration
that supplies no property values. Existing lifetime-range accessors describe an
envelope; existence queries use the actual intervals.

```python
before = graph.current_state_id
bad = [
    {"id": "core.init_nodes", "params": {"target": "prefix", "value": 99}},
    attach("prefix", "prefix"),
    {"id": "temporal.window", "params": {"start": first, "end": 2**60, "output": "invalid"}},
]
try:
    graph.view().apply(Steps(bad), persist=True)
except RuntimeError:
    pass
else:
    raise AssertionError("unknown retained bound must reject")
assert graph.current_state_id == before
assert graph.get_node_attr(0, "prefix") is None
```

## Index ownership and costs

Graph::build_temporal_index() now captures its current retained lineage;
Graph::build_temporal_index_at(state) selects an explicit one. Native callers can
put its Arc into StepVariables under the expected index name to reuse it. Supplied
indexes must have the same owner token and captured head. A stale, foreign,
unscoped or wrong-typed supplied index errors, with no replacement or implicit
refresh. A captured older index remains valid with a matching explicit Context
head. Legacy TemporalIndex::from_history still constructs a whole-forest index;
checked temporal helpers reject it because it has no scoped provenance.

With no supplied variable, the temporal read explicitly derives a local ephemeral
index. It is not installed into StepVariables or a graph cache; failure cannot
leave a partially built cached index. This preserves the prior convenience path
without silently rebuilding a requested stale index or maintaining global indexes.
Index construction scans retained lineage deltas and allocates timelines and
activation intervals. Reuse avoids that work. Borrowed binary-searched property
history slices avoid cloning an entire history before conversion. Changed-event
queries iterate retained records, not every integer in a numeric range.

All outputs are prepared privately with cancellation checks before installation.
Cancellation covers lineage traversal, selected rows, history conversion and
change-mask loops; one commit's index processing, sorting, checked reductions and
large value clones do not promise bounded interrupt latency. Explicit index
construction/reuse and representative benchmarks are documented in the bundle
receipt. These APIs complete the eight primitives; arbitrary effective-time graph
reconstruction, broader index lifecycle and pipeline/pin requirements remain their
existing separate roadmap work.

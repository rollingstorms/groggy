# Temporal decay over pending selected values

`temporal.decay` now accepts node maps, ordered node columns and edge maps whose
IDs belong to the current pending selection. New IDs and revised edge endpoints
are read from that selected source. Explicit input values keep their supplied
identity and order; removed/foreign rows reject.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_nodes(2)
old = graph.read_view()
head = graph.current_state_id
graph.apply(step_pipeline("decay pending values", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("core.load_node_attr", attr="node_id", target="values"),
    stage("iter.loop", iterations=2, body=[
        stage("temporal.decay", attr="values", output="values", half_life=2.0, time_delta=2.0),
    ]),
    stage("core.attach_node_attr", source="values", attr="decayed"),
], atomic=True))
assert [graph.get_node_attr(n, "decayed") for n in [0, 1, 2]] == [0.0, 0.25, 0.5]
assert graph.current_state_id == head + 1
assert old.node_attr(1, "decayed") is None
```

The existing formula remains `value * exp(-ln(2) * time_delta / half_life)`.
Half-life must be finite and positive; elapsed time must be finite and
nonnegative. Inputs must be finite numbers. Underflow to zero is valid, including
very large elapsed-to-half-life ratios. Zero elapsed time with a subnormal
half-life remains valid. Result values are f64.

Elapsed time is supplied by the caller; this stage does not infer it from wall
clock, entity creation, retained-state IDs or property effective timestamps.
Computing a map does not rewrite historical validity. Explicit attachment stages
publish the computed values through the existing graph mutation/clock policy.
A loop can replace its input variable each iteration without publishing a delta
for each iteration.

The shared read boundary validates live selections and respects frozen selected
membership/endpoints. A pure native computation over a retained immutable source
can inspect a subsequently inactive entity without borrowing the current owner.
Public graph application still enforces its accepted-source write contract.

Decay reads selected working values. Accepted snapshot/diff readers remain
independent of those values. Selected lifecycle/window helpers follow surviving
pending membership while querying accepted lineage; atomic provisional IDs still
reject until their retained-history treatment is settled. No helper fabricates a
creation state or silently interprets a state-ID window as effective time.
Errors/cancellation retain the shared atomic rollback behavior. See
[temporal primitives](builder-temporal.md) and [selected windows](builder-temporal-windows.md).

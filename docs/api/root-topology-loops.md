# Sequential root loops with staged topology

An `iter.loop` at the root of an atomic builder can repeat supported topology,
pending-safe property and immutable read stages through the existing native step
executor. Nested root loops share the same pending plan. Each property/selection
read refreshes after mutations; publication happens once when the complete run
succeeds. A zero-iteration loop executes no body and publishes no change.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def root_loop_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_nodes(2)
head = graph.current_state_id
body = [
    root_loop_step("graph.add_nodes", count=1, output="new"),
    root_loop_step("graph.add_edges", edges=[[0, {"added": 0}]], output="edges"),
    root_loop_step("core.load_node_attr", attr="counter", default=0, target="old"),
    root_loop_step("core.init_scalar", value=1, target="one"),
    root_loop_step("core.add", left="old", right="one", target="next"),
    root_loop_step("core.attach_node_attr", source="next", attr="counter"),
]
operation = step_pipeline("root growth", [
    root_loop_step("iter.loop", iterations=3, body=body),
], atomic=True)
result = graph.apply(operation)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1, 2, 3, 4]
assert [graph.get_node_attr(n, "counter") for n in range(5)] == [3, 3, 3, 2, 1]
assert [graph.edge_endpoints(e) for e in graph.edge_ids] == [(0, 2)] * 3
```

## Reference and frame rules

`{"added": 0}` always denotes allocation position zero in the **whole atomic
batch**, including creation in earlier builders. It does not mean the most recent
node or the current iteration's allocation. In the example all three edges target
node 2; nodes 3 and 4 are isolated. Creation output maps are ordinary loop variables
and are overwritten when the same output name is assigned again.

Computed removal maps can select a different surviving entity each iteration.
Literal removals/revisions retain canonical duplicate validation: repeating the
same mutation is not silently treated as a no-op. Removals/revisions still target
accepted owner entities in the invocation selection; this adds no provisional
entity deletion or revision syntax.

The loop remains one mutation batch: final assignments, per-property effective
times and one publication are retained. Intermediate iterations are not separate
history states or checkpoints. Existing explicit topology clocks and table-attach
property clocks retain their normal behavior. Captures are immutable values; a
new capture is needed to read later iterations.

Property-only child `for_each` frames may execute within a growing root loop and
see the current children at each sampling step. Topology inside fixed child frames
remains rejected, including a topology loop nested inside one. Uncertified legacy
property readers cannot share a repeated topology body, even before its first
mutation, since later iterations would read after topology. Alternate batch plans,
streaming and unsupported callbacks retain their guards. Atomic and persistent
execution are required even when the iteration count is zero.

Errors and cooperative cancellation discard topology, attributes, provisional IDs
and emitted outputs from the failed run; prior outputs and the Context remain
usable. Earlier immutable graph views are unchanged.

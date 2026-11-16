# Classic path algorithms over pending graph state

The existing registry handles for BFS, DFS, Dijkstra and A* can follow supported
topology builders in an explicit outer atomic Pipeline. They consume the same
pending selected IDs, endpoints and property columns as builder primitives.
Provisional start/goal markers are valid working inputs. The existing path kernels,
result types, weight/heuristic rules and output attribute parameters are retained.

```python
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from groggy.builder import step_pipeline


def path_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes([{"source": False}, {}, {}])
graph.add_edges([(0, 1), (1, 2)])
head = graph.current_state_id
prepare = step_pipeline("pending path source", [
    path_step("graph.add_nodes", count=1, attributes={"goal": True}, output="new"),
    path_step("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 2]], output="edges"),
    path_step("graph.set_edge_endpoints", revisions=[[0, 2, 0]]),
    path_step("graph.remove_nodes", ids=[1]),
    path_step("core.init_edges", value=2, target="weights"),
    path_step("core.attach_edge_attr", source="weights", attr="weight"),
], atomic=True)
handles = [
    RustAlgorithmHandle("pathfinding.bfs", {"start_attr": "source", "output_attr": "hops"}),
    RustAlgorithmHandle("pathfinding.dfs", {"start_attr": "source", "output_attr": "order"}),
    RustAlgorithmHandle("pathfinding.dijkstra", {"start_attr": "source", "weight_attr": "weight", "output_attr": "cost"}),
    RustAlgorithmHandle("pathfinding.astar", {"start_attr": "source", "goal_attr": "goal", "weight_attr": "weight", "output_attr": "path"}),
]
copy = step_pipeline("consume classic output", [
    path_step("core.load_node_attr", attr="cost", target="costs"),
    path_step("core.attach_node_attr", source="costs", attr="copied_cost"),
], atomic=True)
result = Pipeline([prepare, *handles, copy])(graph.view(), atomic=True)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 2, 3]
assert [graph.get_node_attr(n, "hops") for n in [0, 3, 2]] == [0, 1, 2]
assert [graph.get_node_attr(n, "copied_cost") for n in [0, 3, 2]] == [0.0, 2.0, 4.0]
assert [graph.get_node_attr(n, "path") for n in [0, 3]] == [0, 1]
assert graph.get_node_attr(2, "path") is None
```

## One execution and publication contract

A native AlgorithmFrame retains the original continuation and privately binds a
pending read source when topology has changed. Later algorithms/builders receive
the continuation rather than an unpublished snapshot pretending to be an accepted
write source. Each invocation refreshes pending properties; final projection and
publication belong to the outer coordinator. Errors, cancellation and invalid
weights/heuristics discard the complete run, including provisional allocations.

The shared bulk endpoint reader uses the captured source when present and holds
one owner read guard for ordinary live iteration. It avoids per-edge owner locking
and does not clone writable graph/history state. Weight and heuristic validators
borrow captured or staged scalar values and retain canonical minimum parallel-edge
weights. Source direction, selected edge instances, loops and closure are preserved.

Result columns update the returned reachable/path rows; existing values on other
rows are not implicitly cleared. Marker values are ignored: presence selects the
source/goal, including explicit false or null, with the existing uniqueness rules.

Ordinary standalone/property-only paths retain their live execution path. The
[community wrappers](pending-classic-communities.md) and
[centrality wrappers](pending-classic-centrality.md) also share the pending input
boundary; their captured-table APIs remain available. General topology
loops, uncertified callbacks and alternate lowering remain separate checklist work.

The current native path workload receipt is documented in
`notes/planning/pending-classic-paths-2026-11-15.md`; different graph shapes are not
before/after performance comparisons or general scale guarantees.

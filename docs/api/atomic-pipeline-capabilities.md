# Atomic pipeline capability map

Numeric plan follow-up: [checked numeric loop plans](checked-numeric-loop-plans.md)
lists the 35 lowered instruction forms and shared arithmetic, topology, update
and fused kernels now supported by durable execution. Ordinary step-body loops,
edge-attribute registers and normalize(min/mean) remain outside this guarantee.

An atomic pipeline shares one pending graph mutation across certified builders and
classic algorithms. A successful run publishes one delta; failures and cancellation
restore prior outputs and leave graph state and allocations unchanged. Immutable
read inputs retain their captured state. A write invocation must still match its
owner, branch and accepted head before execution.

| Surface | Pending reads and writes | Remaining boundary |
| --- | --- | --- |
| Node/edge property maps and ordered node columns | Selected surviving/provisional rows; canonical bulk attachments; optional explicit effective time | Explicit times require atomic persistent execution. Unknown attachment parameters reject. |
| Structural, neighbor, path and checked compute primitives | Selected pending geometry and staged property columns | Uncertified extensions/legacy owner-only effects remain guarded. |
| Thirteen classic wrappers | BFS, DFS, Dijkstra, A*; PageRank, closeness, betweenness; components, LPA, Infomap, Louvain, Leiden, Girvan–Newman | Existing direction, weight, option and result conventions still apply. |
| Immutable graph/table intermediates | Explicit captures, named variables/outputs, selected reads and issued-table attachment | Earlier captures stay immutable; matching numeric IDs do not grant write authority. |
| Retained snapshots/diffs | Accepted retained history, independent of pending working edits | Unpublished changes do not become retained history. |
| Retained lifecycle/window helpers | Surviving accepted IDs after pending topology | Provisional-node history remains explicitly guarded pending its policy decision. |
| Root `iter.loop` | Sequential creation, accepted-entity removal/revision, refreshed properties, nested root loops | Added references remain whole-batch positions; canonical duplicate rules remain. |
| Fixed child `sample.for_each` | Exact immutable child selection with shared property staging and refreshed reads | Child topology changes remain rejected, including nested topology loops. |
| Other execution routes | Certified native step interpreter and ordinary supported algorithm calls | Bounded durable step-builder invocation is implemented; see [durable pipelines](durable-pipelines.md). Unported durable kernels, alternate batch plans, streaming and unsupported callbacks remain guarded. |

## All classic families in one publication

```python
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from groggy.builder import step_pipeline


def capability_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_nodes([{"source": False}, {}, {"goal": True}])
graph.add_edges([(0, 1), (1, 2)])
old = graph.read_view()
head = graph.current_state_id
prepare = step_pipeline("pending classic input", [
    capability_step("graph.add_nodes", count=1, output="new"),
    capability_step("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 2]], output="edges"),
    capability_step("graph.set_edge_endpoints", revisions=[[0, 2, 0]]),
    capability_step("graph.remove_nodes", ids=[1]),
    capability_step("core.init_edges", value=2, target="weights"),
    capability_step("core.attach_edge_attr", source="weights", attr="length"),
], atomic=True)
handles = [
    RustAlgorithmHandle("pathfinding.bfs", {"start_attr": "source", "output_attr": "hops"}),
    RustAlgorithmHandle("pathfinding.dfs", {"start_attr": "source", "output_attr": "order"}),
    RustAlgorithmHandle("pathfinding.dijkstra", {"start_attr": "source", "weight_attr": "length", "output_attr": "cost"}),
    RustAlgorithmHandle("pathfinding.astar", {"start_attr": "source", "goal_attr": "goal", "weight_attr": "length", "output_attr": "path"}),
    RustAlgorithmHandle("centrality.pagerank", {"output_attr": "rank"}),
    RustAlgorithmHandle("centrality.closeness", {"weight_attr": "length", "output_attr": "close"}),
    RustAlgorithmHandle("centrality.betweenness", {"weight_attr": "length", "output_attr": "between"}),
    RustAlgorithmHandle("community.connected_components", {"output_attr": "component"}),
    RustAlgorithmHandle("community.lpa", {"seed": 7, "output_attr": "label"}),
    RustAlgorithmHandle("community.infomap", {"seed": 7, "num_trials": 2, "output_attr": "map"}),
    RustAlgorithmHandle("community.louvain", {"output_attr": "louvain"}),
    RustAlgorithmHandle("community.leiden", {"output_attr": "leiden"}),
    RustAlgorithmHandle("community.girvan_newman", {"weight_attr": "length", "num_levels": 1, "output_attr": "girvan"}),
]
result = Pipeline([prepare, *handles])(graph.view(), atomic=True)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 2, 3]
assert sorted(result.edge_ids) == [0, 2, 3]
assert graph.edge_endpoints(0) == (2, 0)
assert graph.get_node_attr(2, "cost") == 2.0
assert graph.get_node_attr(2, "path") == 1
for node in result.node_ids:
    assert abs(graph.get_node_attr(node, "rank") - 1 / 3) < 1e-6
    assert graph.get_node_attr(node, "close") == 1.0
    assert graph.get_node_attr(node, "between") == 0.0
    for name in ["component", "label", "map", "louvain", "leiden", "girvan"]:
        assert graph.get_node_attr(node, name) is not None
assert old.edge_endpoints(0) == (0, 1)
assert old.node_attr(0, "rank") is None
```

Classic wrappers keep their existing kernels. Louvain, Leiden and Girvan–Newman
require undirected input. Weighted shortest-path and centrality lengths retain
canonical domain checks; directed closeness has explicit incoming/outgoing
semantics. Community kernels retain their own loop/parallel-edge definitions;
shortest-path minimum parallel-edge reduction is not a universal graph policy.

See [paths](pending-classic-paths.md), [centrality](pending-classic-centrality.md),
[community](pending-classic-communities.md), [root loops](root-topology-loops.md),
[attachment clocks](builder-attachment-times.md), [captured table writes](captured-table-attachment.md),
and [fixed child frames](pending-subgraph-frames.md) for detailed rules. This map
states bounded implemented capabilities; it is not a closure claim for the full
0.6.0 roadmap, service concurrency, or production performance/scale requirements.

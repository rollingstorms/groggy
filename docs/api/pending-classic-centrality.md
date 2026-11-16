# Classic centrality over pending selected graph state

PageRank, closeness and betweenness registry handles can follow supported topology
builders in an outer atomic Pipeline. They read the pending selected nodes, edge
instances, revised endpoints and staged properties. Result columns are visible to
later builders and algorithms, with one publication at the outer boundary.

```python
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from groggy.builder import step_pipeline


def centrality_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_nodes(3)
graph.add_edges([(0, 1), (1, 2)])
head = graph.current_state_id
prepare = step_pipeline("complete triangle", [
    centrality_step("graph.add_edges", edges=[[0, 2]], output="new_edges"),
    centrality_step("core.init_edges", target="weights", value=2),
    centrality_step("core.attach_edge_attr", source="weights", attr="length"),
], atomic=True)
handles = [
    RustAlgorithmHandle("centrality.pagerank", {"output_attr": "rank"}),
    RustAlgorithmHandle("centrality.closeness", {"output_attr": "close", "weight_attr": "length"}),
    RustAlgorithmHandle("centrality.betweenness", {"output_attr": "between", "weight_attr": "length"}),
]
result = Pipeline([prepare, *handles])(graph.view(), atomic=True)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1, 2]
for node in result.node_ids:
    assert abs(graph.get_node_attr(node, "rank") - 1 / 3) < 1e-6
    assert graph.get_node_attr(node, "close") == 1.0
    assert graph.get_node_attr(node, "between") == 0.0
```

The existing computation kernels, parameters and normalization conventions are
unchanged. Directed closeness preserves explicit incoming/outgoing behavior.
Weighted closeness and betweenness require strictly positive edge lengths.
Betweenness keeps equal minimum-length parallel edge instances as distinct paths;
more expensive parallel edges cannot contribute shortest paths. PageRank retains
its unweighted adjacency and optional staged personalization column.

Each wrapper uses the shared AlgorithmFrame and bulk endpoint authority. The
pending read source is private, while the original continuation is returned for
later pipeline operations. Frozen input direction and endpoints are authoritative;
ordinary live/property-only calls retain their live path and CSR cache behavior.
Errors discard the whole atomic run. Earlier immutable views remain unchanged.

The [six community wrappers](pending-classic-communities.md) also share the
pending input boundary. General topology-mutating loops and
alternate lowering remain open. This slice does not close broader checklist rows
or establish a new performance/scale guarantee.

# Classic community algorithms over pending selected graph state

The six registry wrappers for connected components, label propagation, Infomap,
Louvain, Leiden and Girvan–Newman accept supported pending topology in an outer
atomic Pipeline. They read selected edge instances and revised endpoints, include
provisional rows and stage result columns for later consumers. The outer run
publishes once. Errors discard changes, result columns and provisional allocations.

```python
from groggy import Graph, Pipeline
from groggy.algorithms.base import RustAlgorithmHandle
from groggy.builder import step_pipeline


def community_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes(3)
graph.add_edges([(0, 1), (1, 2)])
head = graph.current_state_id
prepare = step_pipeline("pending return route", [
    community_step("graph.add_edges", edges=[[2, 0]], output="return_edge"),
    community_step("core.init_nodes", target="seeds", value=42),
    community_step("core.attach_node_attr", source="seeds", attr="seed_label"),
], atomic=True)
components = RustAlgorithmHandle("community.connected_components", {
    "mode": "strong", "output_attr": "component",
})
labels = RustAlgorithmHandle("community.lpa", {
    "seed": 7, "seed_attr": "seed_label", "output_attr": "label",
})
result = Pipeline([prepare, components, labels])(graph.view(), atomic=True)
assert graph.current_state_id == head + 1
assert len({graph.get_node_attr(n, "component") for n in result.node_ids}) == 1
assert [graph.get_node_attr(n, "label") for n in result.node_ids] == [42, 42, 42]
```

## Existing algorithm semantics

- Component modes `undirected` and `weak` ignore orientation; `strong` preserves
  directed reachability. Undirected strong components reduce to ordinary components.
- Label propagation retains typed seed labels, deterministic tie handling and its
  optional random seed. Equal labels represent the same label identity.
- Infomap retains its existing directed/undirected model, trial count and seed.
- Louvain and Leiden retain their undirected, unweighted multilevel model. Parallel
  instances and loops reach their existing modularity kernels without collapsing
  them through the shortest-path minimum-weight policy.
- Girvan–Newman retains its undirected model and optional strictly positive edge
  lengths. Its local candidate removals do not mutate the owner's graph topology.

These wrappers reuse AlgorithmFrame, selected direction and bulk endpoint reads.
Connected-component CSR, result and edge-group caches belong to the private
pending scope, so an older warmed live cache cannot supply stale pending results.
The original continuation returns to the pipeline; a private pending pin does not
become an accepted write source. Ordinary live/property-only calls retain their
existing execution and cache path.

Together with the path and centrality families, all thirteen classic wrappers now
share pending selected input semantics. [Sequential root topology loops](root-topology-loops.md) are supported.
Unsupported custom effects, fixed child topology and alternate batch-plan lowering
remain guarded. Retained
lifecycle queries for provisional nodes remain a separate pending policy question.
No new storage schema, automatic index lifecycle or general scale guarantee is
introduced by this adaptation.

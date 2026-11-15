# Structural, neighbor and path reads after staged topology

The existing structural, neighbor, transformation and path primitives can read
pending topology in local atomic builders and shared outer atomic pipelines.
They use the selected immutable source and stage attachments into the same batch.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes(2)
graph.add_edge(0, 1)
head = graph.current_state_id
result = graph.apply(step_pipeline("replace direct path", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("graph.add_edges", edges=[[0, {"added": 0}], [{"added": 0}, 1]],
          output="edges", attributes={"weight": 2.0}),
    stage("graph.remove_edges", ids=[0]),
    stage("core.node_degree", target="degrees"),
    stage("core.attach_node_attr", source="degrees", attr="out_degree"),
    stage("core.init_scalar", target="start", value=0),
    stage("core.shortest_path_map", source="start", output="distances", weight_attr="weight"),
    stage("core.attach_node_attr", source="distances", attr="distance"),
], atomic=True))
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1, 2]
assert graph.get_node_attr(2, "out_degree") == 1
assert graph.get_node_attr(2, "distance") == 2.0
assert graph.get_node_attr(1, "distance") == 4.0
```

## Supported inventory and existing semantics

| Family | Primitives | Preserved semantics |
| --- | --- | --- |
| Structural | `node_degree`, `weighted_degree`, `edge_weight_sum`, `triangle_count`, `k_core_mark` | Degree is outgoing on directed graphs, incident on undirected graphs, once per loop. Weighted sums count both endpoints and twice per loop. Triangle and k-core require undirected input. |
| Neighbors | `collect_neighbor_values`, `neighbor_agg` | Collection follows outgoing directed / symmetric undirected adjacency. Aggregation honors its explicit `in`, `out` or `undirected` direction. Edge occurrences, including parallel edges, are retained. |
| Transformations | expression `map_nodes`, `neighbor_mode_update`, `update_in_place` | Explicit selected IDs, deterministic order and existing asynchronous visibility. Expression `neighbor_values` keeps its distinct-neighbor convention. Arbitrary callback mapping is not certified. |
| Paths | `shortest_path_map`, `k_shortest_paths`, `random_walk` | Existing direction, minimum parallel path costs, loopless k paths, seeded walk behavior and finite nonnegative weight rules. Staged weight properties are visible. |
| Variable reductions | `reduce_nodes`, `std`, `median`, `mode`, `quantile`, `entropy`, `histogram` | Read explicit variable maps, with existing checked numeric/category rules and owned histogram outputs. No graph capture is needed for reduction. |

All IDs are entity IDs, including provisional additions; no owner-liveness filter
is applied to a detached pending source. Removed/cascaded IDs do not become rows.
A stale explicit source map or path start still rejects where the primitive's
contract requires selected membership. Exact selected edges remain exact.

A shared Subgraph read boundary validates live selections and delegates frozen
reads to their captured properties/endpoints. Neighbor collection reuses the
existing checked CSR reader. No kernel builds adjacency from retained owner-pool
endpoints when its input is a pending or historical capture.

## Atomicity and remaining scope

The pending scope cache refreshes after staged mutations. All supported later
reads use that revision. Cancellation, invalid weights, wrong graph direction,
missing variables, invalid source IDs and final canonical validation failures
roll back topology, properties and outputs together. Retained outputs stay
immutable. [Property attachment rules](pending-topology-properties.md) still apply.

Explicit [captured table attachment](captured-table-attachment.md) can stage
selected result columns through retained row identity, including optional effective times.

Certified compute loops and message-passing blocks now share these reads;
see [nested execution](nested-pending-execution.md). Classic property-writing
algorithm wrappers, unadapted sampling/community steps, general topology loops
and opaque callback effects retain explicit guards after
topology. This slice does not certify those families by association. Journal-owner
dispatch and remaining DSL execution/lowering are separate open work.

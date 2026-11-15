# Core computation after pending topology changes

The existing filters, ordering, normalization, fused arithmetic, community
primitives and map samplers can follow supported topology changes in an explicit
atomic builder. They also compose across an outer atomic pipeline and inside
certified compute loops. One checked graph batch is accepted at the outer
boundary.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph(directed=True)
graph.add_nodes(2)
graph.add_edge(0, 1)
old = graph.read_view()
head = graph.current_state_id
graph.apply(step_pipeline("compute pending rows", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("graph.add_edges", edges=[[1, {"added": 0}]], output="edge"),
    stage("iter.loop", iterations=1, body=[
        stage("core.load_node_attr", attr="node_id", target="ids"),
        stage("core.normalize_values", source="ids", method="sum", target="shares"),
        stage("core.top_k", source="ids", k=1, order="desc", target="top"),
        stage("core.attach_node_attr", source="shares", attr="share"),
        stage("core.attach_node_attr", source="top", attr="rank"),
    ]),
], atomic=True))
assert graph.current_state_id == head + 1
assert graph.get_node_attr(2, "share") == 2 / 3
assert graph.get_node_attr(2, "rank") == 0
assert old.node_count() == 2
assert old.node_attr(0, "share") is None
```

## Supported stages

| Family | Existing step IDs | Read contract |
| --- | --- | --- |
| Filters and ordering | `core.filter_nodes_by_attr`, `core.filter_edges_by_attr`, `core.sort_nodes_by_attr`, `core.top_k` | Pending selected membership; attribute reads include staged assignments. |
| Numerical maps | `core.normalize_values`, `core.normalize_node_values`, `core.standardize`, `core.clip` | Explicit input maps; no implicit graph reload. |
| Fused kernels | `graph.fused_neighbor_mul_agg`, `core.fused_axpy`, `core.fused_madd` | Pending selected nodes; directional aggregation uses selected endpoints. |
| Community primitives | `core.community_seed`, `core.modularity_gain`, `core.label_propagate_step` | Pending selected topology and source direction, preserving existing label semantics. |
| Sampling masks | `core.sample_nodes`, `core.sample_edges`, `core.reservoir_sample` | Stable selected IDs; seeded sampling returns typed node/edge masks. |

These stages preserve their existing numerical definitions, sparse mask behavior,
null handling and parameter validation. Ordering breaks equal-value ties by ID,
with missing/null values last. Top-k and reservoir sources must contain selected
IDs, even when requesting zero results. A previously loaded map is an explicit
value: it does not gain new rows or drop removed rows automatically. Numerical
map operations retain that supplied map; attaching its output validates target
membership.

The fused neighbor kernel shares the checked directional CSR reader with ordinary
neighbor aggregation and message passing. Explicit direction overrides traversal
orientation; parallel occurrences and the existing self-loop convention are
preserved. The selected CSR provides stable neighbor ordering. Arithmetic remains
floating point and does not promise bitwise equivalence to earlier unordered
accumulations.

## Boundaries and failure behavior

Sampling masks do not change execution membership. Subgraph-producing
`sample.iterate_nodes`, `sample.iterate_edges`, neighborhood expansion and
for-each-subgraph stages remain guarded after pending topology. Their emitted
subgraphs need separate source-binding qualification. General topology-mutating
loops and classic property-writing algorithm wrappers also remain outside this
new certification; explicit captured algorithm tables remain available.

New/revised/removed topology, property assignments and output variables share the
existing atomic rollback boundary. Invalid values, foreign/removed source rows,
cancellation and acceptance failures cannot publish a partial prefix. Existing
immutable captures keep their original state. This certification adds no new
mutation coordinator, persistence format or Python-side graph computation.

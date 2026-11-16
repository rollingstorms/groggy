# Staged creation and captured algorithm tables

Persistent local atomic builders can create nodes and edges through the same
canonical mixed-batch validator used by ordinary checked writes. Creation stages
use provisional IDs; the owner allocates them only when the complete operation
is accepted. A failed later stage or final policy/timeline validation publishes
nothing and consumes no owner IDs.

```python
from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline

graph = Graph()
initial = GraphBatch(effective_at=0)
initial.add_node()
graph.writer().apply_batch(initial)
head = graph.current_state_id
program = [
    {"id": "graph.add_nodes", "params": {
        "count": 2, "output": "created", "effective_at": 10,
        "attributes": {"score": 1.0000000000000002},
    }},
    {"id": "graph.add_edges", "params": {
        "edges": [[0, {"added": 0}], [{"added": 0}, {"added": 1}]],
        "output": "edges", "effective_at": 20,
    }},
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.algorithm_table", "params": {
        "source": "pending", "output": "distances", "algorithm": "bfs", "start": 0,
    }},
    {"id": "table.emit", "params": {"source": "distances", "name": "distances"}},
]
result, profile = graph.apply(step_pipeline("create and compute", program, atomic=True),
                              return_profile=True)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1, 2]
table = profile["outputs"]["distances"]
assert table.source_entity_ids == [0, 1, 2]
assert table.column("value").to_list() == [0, 1, 2]
assert graph.get_node_attr(0, "value") is None
assert graph.read_view_at(15).edge_ids == []
assert graph.read_view_at(25).edge_ids == [0, 1]
```

## Creation inputs

- `graph.add_nodes`: nonnegative integer `count`, required nonempty `output`,
  optional shared `attributes` object and integer `effective_at`.
- `graph.add_edges`: `edges` containing `[source, target]` pairs, required
  nonempty `output`, optional shared `attributes` and integer `effective_at`.
- An integer endpoint is an existing active node in the execution selection.
  `{"added": position}` references the zero-based position across all node
  creation stages earlier in the shared atomic run (across builders in an outer
  atomic pipeline). Forward references reject.
- Existing-edge revisions accept the same explicit new-node references for
  their endpoints; see [endpoint revisions](pipeline-endpoint-revisions.md).
- Outputs are typed node/edge maps keyed by provisional IDs. Parallel edges and
  loops remain distinct instances. Zero-count/empty creation is a no-op.
- Attribute values use the existing scalar/vector/structured parameter conversion
  and canonical property policy. This input form does not invent typed entity or
  group references from arbitrary JSON objects.

Per-stage effective times become explicit creation-change overrides. Associated
property times inherit creation times unless independently overridden by the
existing batch model. Omitted times use the single final acceptance clock.
Effective-time validity, schema and managed identity checks still run at final
acceptance. Captures do not constitute accepted versions or timestamp guarantees.

## Algorithms over explicit captured state

`graph.algorithm_table` requires `source`, `output` and `algorithm`. It invokes
existing `AlgorithmRead` kernels on exactly that captured selection, including
pending additions, removals and endpoint revisions. It also accepts ordinary
historical captures. It never substitutes current owner liveness or properties.

| Algorithm | Additional options |
| --- | --- |
| `bfs`, `dfs` | Required `start` |
| `dijkstra` | Required `start`; optional `weight_attr` |
| `astar` | Required `start`, `goal`; optional `weight_attr`, `heuristic_attr` |
| `pagerank` | `damping`, `max_iter`, `tolerance`, `personalization_attr` |
| `betweenness` | `normalized`, `weight_attr` |
| `closeness` | `harmonic`, `normalized`, `direction`, `weight_attr` |
| `components` | `mode`: `undirected`, `weak`, `strong` |
| `louvain` | `max_iter`, `max_phases`, `resolution` |
| `leiden` | `max_iter`, `max_phases`, `resolution`, `seed`, `theta` |
| `infomap` | `teleportation`, `num_trials`, `max_iter`, `flow_max_iter`, `seed` |
| `girvan_newman` | `num_levels`, `modularity_threshold`, `weight_attr` |
| `lpa` | `max_iter`, `tolerance`, `seed_attr`, `seed` |

Defaults follow the native registry: PageRank uses 50 iterations and tolerance
1e-6; LPA tolerance is 0.01; closeness defaults to outgoing, harmonic and
unnormalized. Kernel domain restrictions still apply, including algorithms that
require undirected input and validation of negative/nonfinite edge weights.
Unknown algorithms/options reject explicitly. Existing cancellation checks remain
active, and any later error rolls back local staged changes and emitted outputs.

The result is an owned table retaining its captured source and original node IDs.
`value` holds distances, discovery order, scores or labels. BFS/Dijkstra omit
unreachable nodes; component labels include isolates. DFS and A* retain discovery
or path order. A* adds a repeated `cost` column; PageRank adds `converged`,
`iterations` and `max_update` columns so a finite iteration budget cannot silently
claim convergence. Empty results have zero rows.

## Remaining execution boundaries

The new stages require local `atomic=True` and `persist=True` for creation.
Algorithm tables are pure read stages and publish no score properties. Tables
from provisional captures retain `write_state_id=None`; they cannot be used as
accepted-state source writeback even after the enclosing program succeeds.
Use explicit [captured table attachment](captured-table-attachment.md) inside the
atomic run, or refresh an owner capture for later checked writeback.

Outer atomic pipelines now share these stages across builders; see
[outer atomic topology](outer-atomic-topology.md). General topology loops,
arbitrary callback effects, journal-owner pipeline dispatch, opaque extension
certification and remaining high-level DSL lowering stay open.
Certified property loads, initialization, arithmetic and attachments can now
follow topology; see [pending properties](pending-topology-properties.md).
Other calculations after topology still require explicit supported
adaptation; these table computations do not enable unadapted old property steps.

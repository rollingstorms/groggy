# Atomic topology across pipeline algorithms

`Pipeline(..., atomic=True)` shares one native pending mutation across supported
builder algorithms. Each topology builder still declares `atomic=True`, and the
outer run requires `persist=True`. Children stage changes; the outer coordinator
accepts once after all algorithms succeed.

```python
from groggy import Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

graph = Graph()
initial = GraphBatch(effective_at=0)
initial.add_node()
graph.writer().apply_batch(initial)
head = graph.current_state_id

create = step_pipeline("create", [
    {"id": "graph.add_nodes", "params": {
        "count": 1, "output": "created", "effective_at": 10,
    }},
], atomic=True)
connect = step_pipeline("connect", [
    {"id": "graph.add_edges", "params": {
        "edges": [[0, {"added": 0}]], "output": "connected", "effective_at": 20,
    }},
], atomic=True)
compute = step_pipeline("compute", [
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.algorithm_table", "params": {
        "source": "pending", "output": "paths", "algorithm": "bfs", "start": 0,
    }},
    {"id": "table.emit", "params": {"source": "paths", "name": "paths"}},
], atomic=True)

result, profile = Pipeline([create, connect, compute])(
    graph.view(), atomic=True, return_profile=True,
)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1]
assert sorted(result.edge_ids) == [0]
paths = profile["outputs"]["paths"]
assert paths.source_entity_ids == [0, 1]
assert paths.column("value").to_list() == [0, 1]
assert paths.source_views()[0].write_state_id is None
assert graph.read_view_at(15).edge_ids == []
```

## Selection and identity

The original selected IDs remain the execution basis until final acceptance.
`graph.capture_staged` explicitly reconstructs selected pending topology, including
earlier builders' changes. An `{"added": position}` endpoint uses the zero-based
node-creation position across the entire shared run. Step variables remain local
to each builder; emitted outputs use the existing shared Context output scope.

Final projection includes accepted new entities and surviving selected originals.
It excludes removed/cascaded entities and does not induce omitted parallel edges
or copy unrelated owner topology into the result. No provisional ID is filtered
through current owner liveness before acceptance. Retained captures stay immutable
through subsequent stages and after the owner changes or closes.

## Preflight and failure

All algorithms declare their native mutation effects before any prefix executes.
Certified property loads, initialization, arithmetic and attachments can follow
staged topology using the pending selection. See [supported property stages](pending-topology-properties.md).
Subsequent builders may also contain supported topology, explicit pending captures
and immutable-input computations. Supported structural, neighbor, expression and path readers also use that
selection; see [selected readers](pending-topology-readers.md). Unadapted
sampling/community steps and classic property-writing wrappers reject before execution. Use the existing
`graph.algorithm_table` kernels for computations on captured pending topology.
This ordering check is conservative, including statically empty topology stages.

Missing persistence, unsupported effects and invalid ordering reject in preflight.
Runtime errors, cancellation, owner/branch/head mismatches, invalid timelines and
canonical mutation-validation errors discard the entire pending operation.
Rejection consumes no owner IDs or accepted history and restores prior Context
outputs. Diagnostic timers may remain. The Context can be reused after failure.

The outer coordinator owns acceptance and output rollback. Native
`Pipeline::run_atomic_properties` keeps its existing name for compatibility;
`Algorithm::atomic_mutation_effects` adds conservative effect certification.
Existing property certification works before topology. Custom declarations are
trusted contracts; arbitrary callbacks do not acquire rollback safety by default.

## Boundaries

Ordinary outer execution retains per-algorithm publication. Empty shared plans
publish no version. Captured outputs retain provisional provenance even after a
successful acceptance; refresh an owner capture for accepted-state writeback.

General topology loops, remaining adjacency-dependent property adaptation,
journal-owner pipeline dispatch, arbitrary callback certification, remaining DSL
lowering and pipeline-history policy remain separate open work. The existing
owner-thread/GIL boundary is unchanged.

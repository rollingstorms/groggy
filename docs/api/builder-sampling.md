# Checked builder mask sampling

`core.sample_nodes`, `core.sample_edges` and `core.reservoir_sample` support
property-atomic builders and nested loops. Each returns a sparse entity map with
sampled IDs marked by integer 1. They select IDs without creating/deleting graph
entities. Node IDs and edge IDs retain their separate identity domains.

## Count, fraction and seed

Provide exactly one of `count` or `fraction` for node/edge sampling. Count is a
nonnegative integer no larger than the selected population. Fraction must be
finite and in [0, 1]; sample size is nearest integer, with positive half ties
rounded upward. Both controls together, malformed supplied values, and negative
seeds reject. Serialized Python/spec seeds are nonnegative i64 integers; native
constructors also accept u64 seeds.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_nodes([{} for _ in range(10)])
builder = AlgorithmBuilder("sample_four")
sampled = builder.sample_nodes(count=4, seed=19)
builder.attach_as("selected", sampled)
before = graph.current_state_id
graph.view().apply(builder.build(atomic=True), persist=True)
chosen = {n for n in range(10) if graph.get_node_attr(n, "selected") == 1}
assert len(chosen) == 4
assert graph.current_state_id == before + 1
```

Seeded samples traverse stable ascending IDs and use a local RNG; they neither
reseed nor consume the shared thread RNG. Unseeded calls initialize a fresh local
RNG and do not promise repeatability. Exact sampled sets can change when the
population or RNG implementation changes. A seed is not a portable stored sample.

```python
_, profile = graph.view().apply(builder.build(atomic=True), persist=False, return_profile=True)
owned = dict(profile["outputs"]["node_attributes.selected"])
assert set(owned) == chosen
assert graph.current_state_id == before + 1
```

Property attachment writes the sampled rows only. It does not clear a prior mask's
unsampled rows. Use a new output property, an explicit clearing operation, or an
owned transient output when replacing a previous selection.

## Reservoir sampling

The native step takes `source`, `target`, nonnegative `k`, optional `seed` and
`entity_type="nodes"|"edges"` (default nodes). It samples up to k source-map keys;
source values, including zero/null, do not act as predicate masks or weights.
Unlike exact count sampling, k above the available size returns every source key.
Source and target may alias. This API consumes a materialized map; it is not an
external streaming iterator API. Capacity is capped to the available population.

The low-level step spec can be used through an AlgorithmHandle:

```python
import json
from groggy.algorithms.base import AlgorithmHandle

class Steps(AlgorithmHandle):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        from groggy import AttrValue
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": self.steps}))}}

steps = [
    {"id": "core.init_nodes", "params": {"target": "source", "value": 0}},
    {"id": "core.reservoir_sample", "params": {"source": "source", "target": "sample", "k": 3, "seed": 7}},
    {"id": "core.attach_node_attr", "params": {"source": "sample", "attr": "reservoir"}},
]
graph.view().apply(Steps(steps), persist=True)
assert sum(graph.get_node_attr(n, "reservoir") == 1 for n in range(10)) == 3
```

Edge sampling treats parallel edges and self-loops as separate stored edge IDs:

```python
graph.add_edge(0, 1)
graph.add_edge(0, 1)
graph.add_edge(1, 1)
steps = [
    {"id": "core.sample_edges", "params": {"target": "sample", "fraction": 1.0, "seed": 7}},
    {"id": "core.attach_edge_attr", "params": {"source": "sample", "attr": "selected"}},
]
before = graph.current_state_id
graph.view().apply(Steps(steps), persist=True)
assert [graph.get_edge_attr(e, "selected") for e in range(3)] == [1, 1, 1]
assert graph.current_state_id == before + 1
```

## Validation, cancellation and costs

Selected nodes must be active. Edge sampling additionally validates selected edge
liveness and endpoint closure. Reservoir source IDs must belong to the active
selection, even for k=0. Errors/cancellation leave the result variable unchanged;
a failed atomic run discards prior staged attachments. Empty outputs alone create
no graph version. Nested for-each execution and whole sampler-pipeline transactions remain outside
atomic certification. Iteration/expansion/emission contracts are now covered by
[bounded subgraph samplers](builder-subgraph-sampling.md).

All three steps share Algorithm R with O(min(k,N)) reservoir capacity and a sparse
output map. Selected IDs are materialized in stable order; reservoir source keys
are sorted and checked by a linear merge against selected IDs. Thus this is not
an O(k) total-memory streaming interface. Validation scans nodes and, for edge
sampling, selected edges. Cancellation checks cover validation, candidate scanning
and result installation; allocation/sorting does not promise bounded interruption
latency. Bounded release receipts measure sampling plus property publication,
without a before/after speedup, FFI, RSS or production-scale guarantee.

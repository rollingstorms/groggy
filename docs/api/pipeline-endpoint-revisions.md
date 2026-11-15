# Edge endpoint revisions in atomic step pipelines

`graph.set_edge_endpoints` stages rows of `[edge_id, source_id, target_id]` in a
persistent local atomic builder. It changes existing edge records while retaining
edge IDs, properties, external identity and historical endpoint values.

```python
from groggy import Graph
from groggy.builder import step_pipeline

graph = Graph(directed=True)
graph.add_nodes(3)
edge = graph.add_edge(0, 1, weight=7)
before = graph.read_view()
head = graph.current_state_id
_, profile = graph.apply(step_pipeline("move edge and remove old endpoint", [
    {"id": "graph.set_edge_endpoints", "params": {"revisions": [[edge, 0, 2]]}},
    {"id": "graph.remove_nodes", "params": {"ids": [1]}},
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.emit", "params": {"source": "pending", "name": "pending"}},
], atomic=True), return_profile=True)
assert graph.current_state_id == head + 1
assert graph.edge_endpoints(edge) == (0, 2)
assert graph.get_edge_attr(edge, "weight") == 7
assert profile["outputs"]["pending"].edge_endpoints(edge) == (0, 2)
assert before.edge_endpoints(edge) == (0, 1)
```

## Contract

- `revisions` is required; each row has exactly three nonnegative integer IDs.
  Bool, fractional, negative and malformed IDs reject before execution.
- Every edge and both resulting endpoints must be active and in the original
  execution selection. New endpoints may be selected isolates.
- Optional `effective_at` is a signed integer Unix timestamp in microseconds;
  it applies to each endpoint revision in that stage. Omission uses the one
  acceptance clock. Existing temporal endpoint validation remains authoritative.
- An edge can be revised at most once in a batch, including across stages.
  Revising an explicitly removed edge or pointing at a removed node rejects.
- Empty rows add no delta. Loops and parallel edges retain separate IDs.
- Certified prefix property work, revisions and removals publish once. Late
  identity, property, clock or canonical validation errors discard all staging
  and produced outputs, restoring earlier Context outputs.

These are declarative batch stages. The canonical publisher applies endpoint
revisions **before** node-removal cascades, regardless of their staging order.
Moving an edge off a removed node can therefore preserve it. A capture evaluates
the current plan prefix with those same rules; a later revision can rescue an
edge absent from an earlier provisional capture. Explicit edge removal still
conflicts with revision. Captured adjacency and table endpoints use the revised
pairs; older captures keep their original pairs.

A local `atomic=True` builder and `persist=True` are required. Endpoint revisions
begin the same topology phase as removals: further certified topology stages,
explicit staged captures and immutable graph/table reads may follow. Ordinary
unadapted property/structural calculations and topology loops remain unsupported.
[Outer atomic pipelines](outer-atomic-topology.md) now share the same pending batch. There are no new naming/cache policies or
implicit checkpoints. Additions are supported through the same canonical pending batch.

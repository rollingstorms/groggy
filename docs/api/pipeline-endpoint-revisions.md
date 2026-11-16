# Edge endpoint revisions in atomic step pipelines

`graph.set_edge_endpoints` stages rows of `[edge_id, source, target]` in a
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

- `revisions` is required; each row has an accepted edge ID and two endpoint
  references. Numeric endpoints must be active nodes in the original execution
  selection; `{"added": position}` refers to a node created earlier in this atomic
  run. The position spans all creation stages/builders and loop iterations.
- The edge itself must be accepted and selected. This does not revise an edge
  created in the same batch. Bool, fractional, negative, malformed and forward
  references reject; numeric guesses at provisional node IDs are not references.
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
explicit staged captures, immutable graph/table reads and certified
[pending property stages](pending-topology-properties.md) may follow. Unadapted
calculations remain guarded. [Sequential root topology loops](root-topology-loops.md)
share this execution boundary; fixed child topology remains unsupported.
[Outer atomic pipelines](outer-atomic-topology.md) now share the same pending batch. There are no new naming/cache policies or
implicit checkpoints. Additions are supported through the same canonical pending batch.

## Rewire to a node created in the same run

Creation and revision use the same reference parser and selected-node validation.
The final publisher checks endpoint lifetimes and all conflicts before acceptance.
The old edge keeps its identity and properties; no intermediate graph version is
published.

```python
from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline

graph = Graph(directed=True)
initial = GraphBatch(effective_at=0)
initial.add_node()
initial.add_node()
initial.add_edge(0, 1, weight=2.5)
graph.writer().apply_batch(initial)
before = graph.read_view()
head = graph.current_state_id
_, profile = graph.apply(step_pipeline("new endpoint", [
    {"id": "graph.add_nodes", "params": {
        "count": 1, "output": "created", "effective_at": 10}},
    {"id": "graph.set_edge_endpoints", "params": {
        "revisions": [[0, 0, {"added": 0}]], "effective_at": 20}},
    {"id": "graph.remove_nodes", "params": {"ids": [1], "effective_at": 30}},
    {"id": "graph.capture_staged", "params": {"output": "pending"}},
    {"id": "graph.emit", "params": {"source": "pending", "name": "pending"}},
], atomic=True), return_profile=True)
assert graph.current_state_id == head + 1
assert graph.edge_endpoints(0) == (0, 2)
assert graph.get_edge_attr(0, "weight") == 2.5
assert graph.read_view_at(19).edge_endpoints(0) == (0, 1)
assert graph.read_view_at(20).edge_endpoints(0) == (0, 2)
assert profile["outputs"]["pending"].edge_endpoints(0) == (0, 2)
assert before.edge_endpoints(0) == (0, 1)
```

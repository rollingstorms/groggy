# Durable computed writeback and topology edits

The durable step-builder boundary supports `table.attach`,
`graph.set_edge_endpoints`, `graph.remove_nodes`, and `graph.remove_edges`.
They use the same sealed kernels as memory execution and the same single stored
batch as creation/property stages. This extends the
[durable pipeline contract](durable-pipelines.md).

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as folder:
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(4)]
    batch.add_edge(nodes[0], nodes[1])
    batch.add_edge(nodes[1], nodes[3])  # Incident edge outside the selection.
    graph.writer().apply_batch(batch)
    path = Path(folder) / "computed.journal"
    store = DurableGraph.create_from_graph(path, graph)
    source = store.read_view().select_nodes([0, 1, 2])
    program = Pipeline([step_pipeline("rewire and compute", [
        {"id": "graph.add_nodes", "params": {
            "count": 1, "output": "new", "effective_at": 10}},
        {"id": "graph.set_edge_endpoints", "params": {
            "revisions": [[0, 0, {"added": 0}]], "effective_at": 20}},
        {"id": "graph.remove_nodes", "params": {
            "ids": [1], "effective_at": 20}},
        {"id": "graph.capture_staged", "params": {"output": "pending"}},
        {"id": "graph.algorithm_table", "params": {
            "source": "pending", "algorithm": "bfs", "start": 0,
            "output": "distance"}},
        {"id": "table.attach", "params": {
            "source": "distance", "columns": {"value": "hops"},
            "effective_at": 30}},
        {"id": "graph.emit", "params": {
            "source": "pending", "name": "before_writeback"}},
    ], atomic=True)])
    result, profile = program.run_durable(store, source, return_profile=True)
    assert result.node_ids == [0, 2, 4]
    assert result.edge_ids == [0]
    assert result.edge_endpoints(0) == (0, 4)
    assert result.node_attr(4, "hops") == 1
    assert store.read_view().node_ids == [0, 2, 3, 4]
    assert store.read_view().edge_ids == [0]
    assert source.edge_endpoints(0) == (0, 1)
    assert profile["outputs"]["before_writeback"].node_attr(4, "hops") is None
    store.close()
    reopened = DurableGraph.open(path)
    assert reopened.read_view_at(19).edge_endpoints(0) == (0, 1)
    assert reopened.read_view_at(20).edge_endpoints(0) == (0, 4)
    assert reopened.read_view_at(29).node_attr(4, "hops") is None
    assert reopened.read_view_at(30).node_attr(4, "hops") == 1
    reopened.close()
```

## Exact boundaries

- Table attachment requires retained row identity from a capture issued by this
  invocation. Numeric IDs alone grant no attachment authority. Reordered rows and
  explicit source-column-to-property mappings retain their identities. Node and
  edge properties use shared canonical validation, including reserved keys,
  duplicate targets, null/absent handling, and semantic kinds.
- Captured algorithm results can attach explicit result/diagnostic columns. All
  thirteen captured kernels are covered by durable writeback tests. A table's
  captured source stays immutable after its values are attached.
- Explicit attachment time applies to every mapped property change; omitted time
  uses the one accepted operation's default. Writes create history; pure
  computation still does not. There is no implicit table writeback.
- Removal targets and revised edge IDs must be accepted entities in the execution
  selection. Removal/revision of newly created entities remains unsupported.
  Endpoints may explicitly reference earlier staged nodes using whole-batch
  `{"added": position}` references; guessed provisional numeric IDs reject.
- A selected node removal retains memory pipeline semantics: canonical incident
  edge cascades can extend outside the output selection. Rewiring an accepted
  edge away from that node can preserve it. Unselected external nodes remain.
  The separate `apply_view_batch` API retains its stricter selected-scope policy.
- Duplicated revisions, invalid temporal lifetimes, deleted-row writeback and
  conflicting canonical operations reject the entire proposal. Installed and optional stricter per-call schema,
  source/allocation guards, cancellation, journal failure and recovery retain the
  existing durable contract. No intermediate history versions are fabricated.

Unported arithmetic/neighbor/history producers, loops/composition, alternate
plans, streaming and custom effects retain preflight guards. This migration adds
no new policy for provisional history, cache names or pipeline-run retention.

An optional `schema=` dictionary on `Pipeline.run_durable` uses the same proposed-
state policy as checked batch publication. Native callers can use
`run_durable_with_schema` or `PreparedDurablePipeline::accept_with_schema`.
The additional policy is checked together with the latest installed policy and
is retained with the journal operation for recovery; it does not install a new
policy or schema revision. An explicitly supplied schema is also checked for an
empty run without publishing a version.

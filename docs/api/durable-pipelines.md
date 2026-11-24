# Durable pipeline execution

`Pipeline.run_durable(store, selected=None, return_profile=False, schema=None)` runs supported
atomic step builders through one journaled publication. Omitting `selected` uses
the whole current graph; an explicit input must be a current accepted view from
that durable owner. The returned `GraphView` is immutable and has accepted write
provenance. Intermediate captures remain immutable, unaccepted read values.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as folder:
    path = Path(folder) / "example.journal"
    store = DurableGraph.create(path)
    handle = step_pipeline("create and compute", [
        {"id": "graph.add_nodes", "params": {
            "count": 2, "output": "new", "effective_at": 10}},
        {"id": "graph.add_edges", "params": {
            "edges": [[{"added": 0}, {"added": 1}]],
            "output": "edges", "effective_at": 20}},
        {"id": "core.init_nodes_with_index", "params": {"target": "index"}},
        {"id": "core.attach_node_attr", "params": {
            "source": "index", "attr": "ordinal", "effective_at": 30}},
        {"id": "graph.capture_staged", "params": {"output": "pending"}},
        {"id": "graph.algorithm_table", "params": {
            "source": "pending", "algorithm": "bfs", "start": 0,
            "output": "distance"}},
        {"id": "table.emit", "params": {
            "source": "distance", "name": "distance"}},
    ], atomic=True)
    result, profile = Pipeline([handle]).run_durable(store, return_profile=True)
    assert result.node_ids == [0, 1]
    assert result.write_state_id == store.current_state_id == 1
    assert profile["outputs"]["distance"].column("value").to_list() == [0, 1]
    store.close()
    recovered = DurableGraph.open(path)
    assert recovered.read_view_at(15).edge_ids == []
    assert recovered.read_view_at(25).edge_endpoints(0) == (0, 1)
    assert recovered.read_view_at(30).node_attr(1, "ordinal") == 1
    recovered.close()
```

## Supported invocation inventory

| Registered steps | Durable behavior |
| --- | --- |
| `graph.add_nodes`, `graph.add_edges` | Stage creations; explicit `{"added": position}` references address the entire outer batch. Numeric endpoints denote accepted selected nodes. |
| `core.init_nodes`, `core.init_nodes_with_index`, `core.init_edges`, `core.init_scalar`, `core.graph_node_count`, `core.graph_edge_count` | Use the pending selection and deterministic node ordering; scalar values remain owned intermediates. |
| `core.load_node_attr`, `core.load_edge_attr`, `core.attach_node_attr`, `core.attach_edge_attr` | Read pending properties; attach through shared canonical validation. Explicit attachment clocks require persistent atomic execution. |
| `core.add/sub/mul/div/pow/min/max`, `core.abs/sqrt/exp/log`, `core.recip/compare/where/reduce_scalar/broadcast_scalar/mode_list` | Share the existing checked scalar/node/edge kernels; existing operand kinds and domain errors apply. Comparisons preserve exact integer ordering. |
| `core.normalize_values`, `core.normalize_node_values`, `core.standardize`, `core.clip` | Share native normalization over owned node/edge intermediates. |
| `iter.loop` with a fully lowered numeric `batch_plan` | Private register frame; shared checked execution, staged reads and simultaneous carried slots. See [numeric loop plans](checked-numeric-loop-plans.md). |
| `table.attach`, `graph.set_edge_endpoints`, `graph.remove_nodes`, `graph.remove_edges` | Explicit row-bound computed writeback and accepted-entity topology edits; see [writeback and cascades](durable-pipeline-writeback.md). |
| `graph.capture_staged` | Capture an immutable pending selection without allocating accepted IDs or inventing retained history. |
| `graph.select_nodes`, `graph.select_edges`, `graph.nodes_table`, `graph.edges_table`, `graph.degrees_table`, `table.select_columns`, `table.select_rows`, `graph.emit`, `table.emit` | Transform or emit explicit immutable variables. Earlier captures retain their values and membership. |
| `graph.algorithm_table` | Reuse the existing captured BFS, DFS, Dijkstra, A*, PageRank, betweenness, closeness, components, Louvain, Leiden, Infomap, Girvan–Newman and LPA kernels. Existing algorithm domains/options still apply. |

`graph.from_snapshot` has the same sealed implementation as the typed reads, but
this entry point currently has no supported retained-snapshot producer. It does
not make owner-bound temporal steps portable. Atomic builders are required;
ordinary classic algorithm handles, custom callbacks, ordinary step-body loops,
provisional-entity removal/revision, unported read steps, unlowered alternate plans
and streaming reject during preflight. These remain explicit migration
work in existing checklist rows. A supported effect flag alone grants no durable
invocation authority. This entry point does not add pipeline-run history, cache
policy, or a service concurrency model.

## Preparation, acceptance and failures

Rust `Pipeline::prepare_durable` captures the accepted source, complete
owner/branch/head coordinate, allocation high-water counters, and sealed native
kernel references. No custom Step/Algorithm method is invoked by its worker
execution. `DurablePipelinePlan::execute` uses the same `StepExecutor` and kernels
as memory execution and prepares owned changes and immutable output membership.
It holds no `Graph` or writable `Subgraph`. Dropping the proposal publishes nothing.

`PreparedDurablePipeline::accept` returns to the durable owner. It rechecks the
coordinate and any staged allocation expectation, then checks the latest
installed schema and any explicit stricter per-call schema and prepares the complete accepted output before journal
append/sync/publication. A per-call schema is retained with its operation for
replay and does not replace the installed schema; it is checked on empty runs
without creating a history version. Returning to the same branch/head after allocations on a
sibling branch does not authorize reinterpreting provisional IDs. Empty runs
validate the source and publish no version. Each nonempty run uses one engine
clock default unless explicit change timestamps were supplied. Reading or
computing alone records no property change.

`run_durable` restores earlier Context outputs on execution or acceptance failure.
Validation, cancellation and recoverable append errors consume no accepted IDs or
versions. An indeterminate append preserves the existing poison/reopen contract;
it returns no accepted result. Native callers using the explicit two-phase API
must treat Context outputs as pending until `accept` succeeds and discard them
on failed acceptance. Cancellation is checked again before acceptance; it cannot
undo an operation after journal acceptance has begun.

Python releases the GIL for detached preparation. Source capture, final schema and
publication work, and Python result conversion remain on the creator thread.
This does not authorize concurrent calls or writes against one durable owner.
Accepted output preparation currently reuses the full immutable proposed-state
projection used by checked view publication. It copies a snapshot, not a writable
graph/history/pool; selected-only memory scaling is not promised.

Evidence: [native workflow tests](../../tests/durable_pipeline_execution.rs),
[Python workflow and GIL tests](../../tests/test_durable_pipeline_execution.py),
[journal failure injection](../../src/storage/journal.rs), and
[completion record](../../notes/planning/durable-pipeline-execution-2026-11-24.md).

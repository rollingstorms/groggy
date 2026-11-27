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
| `core.filter_nodes_by_attr`, `core.filter_edges_by_attr`, `core.sort_nodes_by_attr`, `core.top_k` | Share selected pending property reads and existing predicate/rank rules; masks/ranks are owned maps, with graph changes only through explicit attachment. |
| `core.sample_nodes`, `core.sample_edges`, `core.reservoir_sample` | Seeded selected node/edge masks; reservoir inputs are validated even at zero k, and only explicit attachment writes graph properties. |
| `sample.iterate_nodes`, `sample.iterate_edges`, `sample.neighbors`, `sample.emit_subgraphs`, certified `sample.for_each` | Immutable selected collections and recursively prepared fixed child bodies, including pending IDs, nested loops/captures and sibling property visibility. Child topology and unported bodies reject. |
| Declarative `core.map_nodes` expressions | Share the existing evaluator with a private selected-read input adapter. Pending attributes, graph-default neighbors, checked operators, short-circuit and async source visibility retain their native rules. Rust callback mapping remains owner-only. |
| `alias`, `core.flow_update`, `core.residual_capacity` | Share owned immutable values and checked edge-map arithmetic; aliases remain independent under mutation/rebinding. Positive residuals retain their supplied edge IDs. |
| `core.reduce_nodes`, `core.std`, `core.median`, `core.quantile`, `core.mode`, `core.entropy`, `core.histogram` | Reduce explicitly loaded owned node maps; preserve native precision, empty-input and tie rules. Histogram outputs are owned count vectors. |
| `core.normalize_values`, `core.normalize_node_values`, `core.standardize`, `core.clip` | Share native normalization over owned node/edge intermediates. |
| `core.edge_weight_scale` | Selected pending edge properties; finite numeric factors/results, legacy unit fallback and nonnumeric pass-through; only attachment publishes. |
| `core.community_seed`, `core.modularity_gain`, `core.label_propagate_step` | Selected pending neighborhoods and ordered owned columns; retain seeding, modularity and typed label-vote definitions. |
| `core.shortest_path_map`, `core.k_shortest_paths`, `core.random_walk` | Shared selected BFS/Dijkstra, loopless Yen node sequences and locally seeded walks; preserve direction, parallel arcs, pending weights, restart rules and checked scalar/map outputs. |
| `temporal.decay` | Checked numerical decay over selected owned node maps, ordered columns and edge maps; no retained-history lookup or implicit clock. |
| `core.execution_block` in supported `message_pass` mode | Prepare the existing body on the owner; use selected pending adjacency and preserve ordered/synchronous updates, column order, directions and checked body operations. Streaming and unsupported body operations reject. |
| `iter.loop` with an ordinary `body` of supported steps | Prepare sealed child kernels recursively on the owner; nested loops share pending reads, captures, cancellation and one outer publication. |
| `iter.loop` with a fully lowered numeric/topology `batch_plan` | Private register frame; shared checked execution, staged reads and simultaneous carried slots. See [numeric loop plans](checked-numeric-loop-plans.md). |
| `core.weighted_degree`, `core.edge_weight_sum`, `core.triangle_count`, `core.k_core_mark` | Share the existing selected structural kernels. Weight sums use total incidence, signed finite weights and twice-per-loop contribution; triangles/k-core require undirected input. |
| `core.node_degree`, `core.neighbor_agg`, `core.collect_neighbor_values`, `core.neighbor_mode_update`, `core.update_in_place` | Share checked selected topology and immutable input maps. Asynchronous label updates keep stable ID order; pending revisions/removals are visible. |
| `core.fused_axpy`, `core.fused_madd`, `graph.fused_neighbor_mul_agg` | Retain native f64 arithmetic; invalid nonfinite inputs/results reject before publication. |
| `table.attach`, `graph.set_edge_endpoints`, `graph.remove_nodes`, `graph.remove_edges` | Explicit row-bound computed writeback and accepted-entity topology edits; see [writeback and cascades](durable-pipeline-writeback.md). |
| `graph.capture_staged` | Capture an immutable pending selection without allocating accepted IDs or inventing retained history. |
| `graph.select_nodes`, `graph.select_edges`, `graph.nodes_table`, `graph.edges_table`, `graph.degrees_table`, `table.select_columns`, `table.select_rows`, `graph.emit`, `table.emit` | Transform or emit explicit immutable variables. Earlier captures retain their values and membership. |
| `graph.algorithm_table` | Reuse the existing captured BFS, DFS, Dijkstra, A*, PageRank, betweenness, closeness, components, Louvain, Leiden, Infomap, Girvan–Newman and LPA kernels. Existing algorithm domains/options still apply. |

`graph.from_snapshot` has the same sealed implementation as the typed reads, but
this entry point currently has no supported retained-snapshot producer. It does
not make owner-bound temporal steps portable. Atomic builders are required;
ordinary classic algorithm handles, custom callbacks, loop bodies containing
unported steps, provisional-entity removal/revision, unported read steps,
unlowered alternate plans and streaming reject during preflight. These remain explicit migration
work in existing checklist rows. A supported effect flag alone grants no durable
invocation authority. This entry point does not add pipeline-run history, cache
policy, or a service concurrency model.

## Preparation, acceptance and failures

Rust `Pipeline::prepare_durable` captures the accepted source, complete
owner/branch/head coordinate, allocation high-water counters, and sealed native
kernel references, including recursively prepared ordinary loop bodies. No custom
Step/Algorithm method is invoked by its worker execution. `DurablePipelinePlan::execute` uses the same `StepExecutor` and kernels
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

## Ordinary nested bodies

A `body` uses the same supported invocation inventory as the outer builder.
Preparation checks every child, even in a zero-iteration loop. Unsupported or
malformed supplied `batch_plan` programs never fall back to a different body.
The worker uses captured kernel references and metadata; it does not instantiate
steps or call Step/Algorithm hooks. The existing owner-only memory fallback for
unported ordinary bodies remains available outside durable execution.

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from groggy import DurableGraph, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as folder:
    store = DurableGraph.create(Path(folder) / "loops.journal")
    def loop_step(kind, **params):
        return {"id": kind, "params": params}
    inner = loop_step("iter.loop", iterations=3, body=[
        loop_step("graph.add_nodes", count=1, output="new", effective_at=10),
    ])
    outer = loop_step("iter.loop", iterations=2, body=[inner])
    handle = step_pipeline("nested creation", [outer], atomic=True)
    result = Pipeline([handle]).run_durable(store)
    assert result.node_ids == [0, 1, 2, 3, 4, 5]
    assert store.current_state_id == 1
    store.close()
```

Loop variables use sequential `[initial, logical]` bindings, including when the
iteration count is zero. Immutable values share their payload until a mutable
consumer needs its own copy. Earlier captures remain unchanged by later writes.
As with other atomic builders, repeated writes to one key coalesce into its
final staged value and effective time; internal iterations do not create
separate retained versions. An inner failure or cancellation discards the
whole publication and restores earlier Context outputs.


## Structural step definitions

`core.weighted_degree` and `core.edge_weight_sum` retain their existing total
incidence definition on directed and undirected graphs: both endpoints receive
the weight, and a self-loop contributes twice. Missing weights default to one;
present weights must be finite numeric scalars. Signed weights are allowed and
nonfinite accumulated sums reject before publication. This differs from the
outgoing edge count produced by `core.node_degree` on directed graphs.

`core.triangle_count` and `core.k_core_mark` retain their existing undirected
contracts. Triangles use the simple loop-free projection. K-core peeling uses
selected edge instances and counts each self-loop once. Directed inputs reject;
no directed definition is inferred. Every step sees pending weights/endpoints
and the selected membership, including prior creations/removals in its batch.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

initial = Graph()
batch = GraphBatch(effective_at=0)
for _ in range(3):
    batch.add_node()
for source, target in [(0, 1), (1, 2), (2, 0)]:
    batch.add_edge(source, target)
initial.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "triangles.journal", initial)
    old_head = store.current_state_id
    handle = step_pipeline("triangle counts", [
        {"id": "core.triangle_count", "params": {"target": "counts"}},
        {"id": "core.attach_node_attr", "params": {
            "source": "counts", "attr": "triangles", "effective_at": 10}},
    ], atomic=True)
    result = Pipeline([handle]).run_durable(store)
    assert [result.node_attr(i, "triangles") for i in result.node_ids] == [1, 1, 1]
    assert store.current_state_id == old_head + 1
    store.close()
```


## Owned aggregation inputs

Aggregation steps consume the named node map, rather than reading graph
properties again. A later property write cannot change a previously loaded map;
an explicit new load sees pending values. Use `core.broadcast_scalar` to turn a
scalar statistic into a node map before attachment. A histogram is a count vector,
not a graph selection; its named `histograms.<target>` output can also be read
without attaching anything or publishing history.

`core.std` uses sample standard deviation and returns null for fewer than two
values. Empty reductions, median, quantile and mode return null; empty entropy
is zero and histogram bins are zero-filled. Numeric statistics require finite
numeric inputs. Mode and entropy accept typed categories; tied modes choose the
category whose contributor has the lowest node ID. The existing numerical kernels
and quantile interpolation rules are shared across both execution paths.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

input_graph = Graph()
batch = GraphBatch(effective_at=0)
for value in [1, 2, 3, 4]:
    batch.add_node(score=value)
input_graph.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "histogram.journal", input_graph)
    head = store.current_state_id
    handle = step_pipeline("owned histogram", [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
        {"id": "core.histogram", "params": {"source": "values", "target": "counts", "bins": 2}},
    ], atomic=True)
    result, profile = Pipeline([handle]).run_durable(store, return_profile=True)
    assert profile["outputs"]["histograms.counts"] == [2, 2]
    assert result.write_state_id == head == store.current_state_id
    store.close()
```


## Message-passing blocks

Supported `core.execution_block` message-pass bodies run through the same sealed
kernel as memory execution, including inside ordinary loops. The Python
`AlgorithmBuilder.message_pass` facade uses this route with an atomic builder.
Body parsing and certification happen during owner preparation; detached workers
do not instantiate steps or call the registry.

Ordered mode reads each earlier update in the target's row order; node maps use
ascending IDs and explicit node columns preserve their order. Synchronous mode
reads the original column throughout one block invocation. Direction defaults
to incoming and can be explicitly `in`, `out` or `undirected`. Parallel neighbor
incidences remain distinct; self-loops count once and `include_self` adds a node
only when it is absent from that neighbor row. These are the existing block
rules, rather than a new general adjacency definition.

Supported bodies retain neighbor collection/mode, unweighted neighbor sum,
constants, checked arithmetic, comparisons, conditional selection and apply.
Weighted or other reducer variants inside a block remain rejected; use the
standalone aggregation steps. Streaming, unsupported body operations and invalid
directions reject before execution, including within a zero-iteration loop.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import AlgorithmBuilder

graph = Graph(directed=True)
batch = GraphBatch(effective_at=0)
for value in [1, 2, 3]:
    batch.add_node(score=value)
batch.add_edge(0, 1)
batch.add_edge(1, 2)
graph.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "messages.journal", graph)
    builder = AlgorithmBuilder("synchronous messages")
    values = builder.load_attr("score")
    with builder.message_pass(target=values, ordered=False,
                              include_self=False, direction="in") as messages:
        messages.apply(builder.graph_ops.neighbor_agg(values, "sum"))
    builder.attach_as("answer", values)
    result = Pipeline([builder.build(atomic=True, optimize=False)]).run_durable(store)
    assert [result.node_attr(i, "answer") for i in result.node_ids] == [0, 1, 2]
    store.close()
```


## Flow maps and aliases

`alias` creates an independent logical binding to an immutable value. Its payload
is shared until a mutable consumer needs a private copy. Rebinding either name
leaves the other unchanged. Graph/table aliases retain their original immutable
pin, including after later graph changes or closing the owner.

`core.flow_update` accepts an edge-map delta or scalar delta and uses checked
integer/finite-float arithmetic. Map deltas use the union of supplied edge IDs;
missing deltas preserve the original value. `core.residual_capacity` retains
positive capacity-minus-flow values and positive flow-only values under their
supplied IDs. These primitives neither create reverse edges nor enforce complete
flow conservation. Explicit edge attachment still checks graph authority.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

graph = Graph(directed=True)
batch = GraphBatch(effective_at=0)
batch.add_node()
batch.add_node()
batch.add_edge(0, 1)
graph.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "flow.journal", graph)
    def flow_step(kind, **params):
        return {"id": kind, "params": params}
    handle = step_pipeline("flow update", [
        flow_step("core.init_edges", target="flow", value=2),
        flow_step("alias", source="flow", target="saved"),
        flow_step("core.init_scalar", target="delta", value=3),
        flow_step("core.flow_update", flow="flow", delta="delta", target="flow"),
        flow_step("core.init_edges", target="capacity", value=10),
        flow_step("core.residual_capacity", capacity="capacity", flow="flow", target="residual"),
        flow_step("core.attach_edge_attr", source="flow", attr="flow", effective_at=20),
        flow_step("core.attach_edge_attr", source="saved", attr="saved", effective_at=20),
        flow_step("core.attach_edge_attr", source="residual", attr="residual", effective_at=30),
    ], atomic=True)
    result = Pipeline([handle]).run_durable(store)
    assert result.edge_attr(0, "flow") == result.edge_attr(0, "residual") == 5
    assert result.edge_attr(0, "saved") == 2
    store.close()
```


## Declarative expressions

Declarative `core.map_nodes` and supported Python `AlgorithmBuilder.map_nodes`
programs use the existing expression evaluator with a private read-source
adapter. The public Rust `ExprContext`, `StepInput` and `Expr.eval` shapes remain
unchanged. Programmatic Rust mapping callbacks retain their separate owner-only
boundary and do not gain detached execution authority.

Attributes read staged properties. Named maps remain owned inputs; asynchronous
mapping uses the existing private working-source override, with deterministic
ascending node processing and complete-result installation. Invalid arithmetic,
missing inputs and unknown functions return explicit errors. Existing short
circuit and conditional rules remain.

Neighbor functions retain graph-default outgoing direction for directed graphs
and symmetric direction for undirected graphs. `neighbors` and `neighbor_count`
preserve incidences, while `neighbor_values` preserves one value per distinct
neighbor. A self-loop contributes its node once. This distinction is preserved
rather than inferred from another collection primitive.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import AlgorithmBuilder

graph = Graph()
batch = GraphBatch(effective_at=0)
for _ in range(3):
    batch.add_node()
for source, target in [(0, 1), (1, 2), (2, 0)]:
    batch.add_edge(source, target)
graph.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "expressions.journal", graph)
    builder = AlgorithmBuilder("structured expressions")
    ranks = builder.init_nodes(1.0)
    with builder.iterate(2):
        sums = builder.map_nodes("sum(ranks[neighbors(node)])", inputs={"ranks": ranks})
        ranks = builder.var("ranks", builder.core.add(builder.core.mul(sums, 0.85), 0.15))
    builder.attach_as("rank", ranks)
    result = Pipeline([builder.build(atomic=True, optimize=False)]).run_durable(store)
    assert all(abs(result.node_attr(i, "rank") - 3.295) < 1e-12 for i in result.node_ids)
    store.close()
```


## Filter masks and rank maps

Filtering steps produce owned maps containing integer `1` for matching node/edge
IDs. They do not remove or deactivate entities. Missing attributes are skipped;
explicit null is still a typed value for equality/inequality predicates.
Ordering comparisons preserve mixed integer/floating precision. Equality retains
its typed comparison rule; numeric range and ordering predicates use numeric
comparison. String containment and inclusive ranges retain their existing rules.

Sort and top-k produce ID-to-rank maps. They require a single comparable numeric,
text or Boolean domain; missing/null values sort last in either direction and
value ties break by ID. NaNs and incompatible domains reject. Even top-k zero
validates its entire source before returning an empty map. Computation alone
creates no version; explicit attachment uses the canonical checked batch.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

graph = Graph()
batch = GraphBatch(effective_at=0)
for score in [3, 1, 3, None]:
    batch.add_node(score=score)
graph.writer().apply_batch(batch)
with TemporaryDirectory() as folder:
    store = DurableGraph.create_from_graph(Path(folder) / "ranks.journal", graph)
    handle = step_pipeline("rank selected values", [
        {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
        {"id": "core.top_k", "params": {"source": "values", "order": "desc", "k": 2, "target": "top"}},
        {"id": "core.attach_node_attr", "params": {"source": "top", "attr": "rank", "effective_at": 10}},
    ], atomic=True)
    result = Pipeline([handle]).run_durable(store)
    assert result.node_ids == [0, 1, 2, 3]
    assert [result.node_attr(i, "rank") for i in result.node_ids] == [0, None, 1, None]
    store.close()
```

## Seeded sampling masks

Sampling returns owned ID-to-one masks, preserving graph membership and distinct
parallel/self-loop edge instances. Counts cannot exceed the selected population.
Fractions are finite in `[0, 1]`; the selected count rounds halfway upward.
Reservoir sampling takes at most the available source rows, validates all source
IDs even at zero k, and keeps no streaming state between invocations. A fixed
seed and stable population reproduce the mask. Without a seed, results may vary.
Only explicit attachment publishes these results; repeated masks are not a new
selection. Subgraph-producing samplers and foreach containers retain their
separate owner-bound execution limitation.

```python
from tempfile import TemporaryDirectory
from pathlib import Path
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as directory:
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    for _ in range(5):
        batch.add_node()
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "sampling.journal", graph)
    program = Pipeline([step_pipeline("sample masks", [
        {"id": "core.sample_nodes", "params": {"fraction": 0.5, "seed": 42, "target": "sample"}},
        {"id": "core.reservoir_sample", "params": {"source": "sample", "k": 2, "seed": 42, "target": "two"}},
        {"id": "core.attach_node_attr", "params": {"source": "two", "attr": "chosen", "effective_at": 20}},
    ], atomic=True)])
    result = program.run_durable(store)
    assert len(result.node_ids) == 5
    assert sum(result.node_attr(node, "chosen") == 1 for node in result.node_ids) == 2
    store.close()
```

## Normalization safety

Ordinary normalization factories apply defaults only to omitted controls. Explicit
malformed epsilon/method/bounds reject, including null, booleans and invalid
alias values. Numeric controls are validated during factory preparation, so bad
epsilon or clip ranges reject even inside a zero-iteration body. Existing clip
`min_value`/`max_value` precedence over `min`/`max` remains when both are valid.
Construction can therefore reject earlier than execution.

Ordinary normalization and standardization require finite, nonnegative epsilon
and finite numeric inputs. Clip bounds must be ordered and not NaN; infinite
bounds preserve the existing unbounded-side defaults. Invalid
controls reject even for empty maps; NaN bounds return an error rather than
reaching the underlying clamp operation. A nonfinite arithmetic intermediate or
result rejects the complete operation before output installation or publication.
This is a bounded f64 normalization contract, not a claim that all extreme-value
ratios are reconstructed through otherwise unrepresentable intermediates.

Normalization supports signed sum, arithmetic mean, minimum, maximum and minmax
range; standardization uses population variance. Empty sum normalization
rejects its zero denominator; empty mean/min/max/minmax, standardization and
clipping return empty maps when their controls are valid. Checked register plans
also support sum/mean/min/max, with the same finite and epsilon checks.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as directory:
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    batch.add_node(score=0.0)
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "normalize.journal", graph)
    head = store.current_state_id
    try:
        program = Pipeline([step_pipeline("invalid epsilon", [
            {"id": "core.load_node_attr", "params": {"attr": "score", "target": "values"}},
            {"id": "core.normalize_values", "params": {"source": "values", "target": "out", "epsilon": -1.0}},
        ], atomic=True)])
        program.run_durable(store)
    except Exception as error:
        assert "epsilon" in str(error)
    else:
        raise AssertionError("negative epsilon must reject")
    assert store.current_state_id == head
    store.close()
```

## Edge-weight transformation

`core.edge_weight_scale` uses the selected pending edge properties. Numeric scalar
weights produce f64 results; missing weights use one. Existing representable
nonnumeric values pass through unchanged. Attributes outside the step-value
representation retain the legacy unit fallback; this is not an arbitrary-property
conversion API. The default factor is one. An explicit factor must be numeric
and finite; scalar nonfinite input or overflow rejects without replacing prior
outputs or publishing staged changes. Loops and parallel edges retain their IDs.
Only explicit attachment writes graph properties.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as directory:
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    first, second = batch.add_node(), batch.add_node()
    batch.add_edge(first, second, weight=3.5)
    batch.add_edge(first, second)
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "weights.journal", graph)
    result = Pipeline([step_pipeline("scale weights", [
        {"id": "core.edge_weight_scale", "params": {"attr": "weight", "factor": 2, "target": "out"}},
        {"id": "core.attach_edge_attr", "params": {"source": "out", "attr": "scaled", "effective_at": 20}},
    ], atomic=True)]).run_durable(store)
    assert [result.edge_attr(edge, "scaled") for edge in result.edge_ids] == [7.0, 2.0]
    store.close()
```

## Community helper columns and numerical decay

Community seeding retains ascending node IDs, stable degree-hub ties and local
seeded RNG. Modularity gains use unweighted selected edge instances, their
existing directed/undirected objective, twice-per-undirected-loop degree, and
best nonnegative moves to neighboring communities. Label propagation uses one
explicit self vote in addition to stored self-loop incidences; typed canonical
serialization breaks frequency ties. Output columns retain supplied column order;
input maps use ascending ID order. Earlier aliases remain unchanged.

`temporal.decay` multiplies an owned numeric map/column by
`exp(-ln(2) * time_delta / half_life)`. Time delta is supplied by the caller and
is not a storage timestamp. Half-life is finite and positive, elapsed time finite
and nonnegative. Subnormal half-lives handle zero elapsed time and underflow
without NaN; node/edge membership is validated even at zero elapsed time.
Only attachment writes properties. This enables numerical decay without claiming
that retained snapshot/index/window producers run on workers.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as directory:
    graph = Graph()
    batch = GraphBatch(effective_at=0)
    nodes = [batch.add_node() for _ in range(3)]
    batch.add_edge(nodes[0], nodes[1])
    batch.add_edge(nodes[0], nodes[1])
    batch.add_edge(nodes[1], nodes[1])
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "community.journal", graph)
    result = Pipeline([step_pipeline("community and decay", [
        {"id": "core.community_seed", "params": {"strategy": "singleton", "target": "labels"}},
        {"id": "core.modularity_gain", "params": {"partition": "labels", "target": "gain"}},
        {"id": "core.label_propagate_step", "params": {"labels": "labels", "target": "next"}},
        {"id": "temporal.decay", "params": {"attr": "labels", "output": "half", "half_life": 2, "time_delta": 2}},
        {"id": "core.attach_node_attr", "params": {"source": "half", "attr": "half", "effective_at": 20}},
        {"id": "core.attach_node_attr", "params": {"source": "next", "attr": "next", "effective_at": 20}},
    ], atomic=True)]).run_durable(store)
    assert [result.node_attr(node, "half") for node in result.node_ids] == [0.0, 0.5, 1.0]
    assert [result.node_attr(node, "next") for node in result.node_ids] == [1, 0, 2]
    store.close()
```


## Immutable collections and fixed foreach frames

Portable collection steps carry `GraphViewArray` values without writable owners.
The Rust memory adapter retains its existing `SubgraphArray` output and explicit
owner association. This adds one public `StepValue` enum variant: exhaustive Rust
matches must include it; existing variants and accessors retain their contracts.
The standalone sampler facade still uses its owner-connected adapter. These
step-builder guarantees do not certify every standalone algorithm entry point.

All selector sources are checked before child execution: matching numeric IDs
alone do not establish ownership. Accepted views must share history identity;
pending views must have been issued by this atomic invocation. Child execution
refreshes pending values through the accepted root and projects exact membership,
including provisional IDs. It never grows to include a sibling's rows. Nested
loops and captures retain that boundary. Topology inside fixed children remains
rejected, including zero-iteration nested topology programs.

A missing child output preserves its input pin; an explicit empty array emits
zero children. Rebinding an alias does not mutate its source. Invalid selectors,
wrong reserved-output types, errors or cancellation preserve prior outputs and
roll back staged writes and allocations. Neighborhood direction, selected edge
scope, parallel instances, self-loops, ordered overlap and induced/per-item/unified
rules remain the existing native definitions.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline


def collection_step(kind, **params):
    return {"id": kind, "params": params}


with TemporaryDirectory() as directory:
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for _ in range(3):
        batch.add_node()
    batch.add_edge(0, 1)
    batch.add_edge(1, 2)
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "frames.journal", graph)
    program = Pipeline([step_pipeline("fixed edge frames", [
        collection_step("sample.iterate_edges", target="seeds"),
        collection_step("sample.for_each", source="seeds", target="mapped", body=[
            collection_step("core.load_node_attr", attr="counter", default=0, target="old"),
            collection_step("core.init_scalar", value=1, target="one"),
            collection_step("core.add", left="old", right="one", target="next"),
            collection_step("core.attach_node_attr", source="next", attr="counter", effective_at=20),
            collection_step("graph.capture_staged", output="pin"),
            collection_step("graph.emit", source="pin", name="child"),
        ]),
    ], atomic=True)])
    result, profile = program.run_durable(store, return_profile=True)
    first = profile["outputs"]["for_each.mapped.0.child"]
    assert first.node_ids == [0, 1]
    assert first.node_attr(1, "counter") == 1
    assert [result.node_attr(n, "counter") for n in range(3)] == [1, 2, 1]
    store.close()
    # Captured child values remain readable after the owner closes.
    assert first.node_attr(1, "counter") == 1
```

Membership sets and projected child values still require allocation. Reusing
source pins does not make wide captures free, and this portability change makes
no new latency, allocator/RSS or scale guarantee. Retained-history producers,
edge-aware registers, remaining frontend/alternate plans, streaming/custom
callbacks and the existing naming/cache/run-history questions remain open.


## Selected path primitives

`core.shortest_path_map` produces reachable-node maps: exact integer BFS distances
without `weight_attr`, or f64 Dijkstra distances with that option. Missing weights
remain unit cost; explicit weights must be finite, numeric and nonnegative.
`core.k_shortest_paths` produces JSON text containing loopless node sequences and
costs, using the minimum parallel-arc weight for each node-to-node transition.
Parallel edges do not turn an identical node sequence into multiple paths.
`core.random_walk` produces JSON text containing ordered walks, retaining per-edge
transition multiplicity, local seeded RNG, restart and zero-weight fallback rules.
These scalar text outputs can be broadcast explicitly before property attachment.

Source/target IDs must belong to the current selected frame. Validating a zero
path count or zero walk length does not skip ID/weight checks. Fixed foreach
children cannot inspect outside rows. Pending creations, removals, endpoint
revisions and property weights are read through the same immutable boundary as
other native kernels. Cost overflow, cancellation and domain errors preserve old
outputs and roll back the complete invocation. Search/tie/random-walk definitions
are unchanged; no new precision or path-enumeration scale claim is made.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import DurableGraph, Graph, GraphBatch, Pipeline
from groggy.builder import step_pipeline

with TemporaryDirectory() as directory:
    graph = Graph(directed=True)
    batch = GraphBatch(effective_at=0)
    for _ in range(3):
        batch.add_node()
    batch.add_edge(0, 1, cost=2.0)
    batch.add_edge(1, 2, cost=3.0)
    graph.writer().apply_batch(batch)
    store = DurableGraph.create_from_graph(Path(directory) / "paths.journal", graph)
    program = Pipeline([step_pipeline("selected distance", [
        {"id": "core.init_scalar", "params": {"target": "start", "value": 0}},
        {"id": "core.shortest_path_map", "params": {"source": "start", "weight_attr": "cost", "output": "distances"}},
        {"id": "core.attach_node_attr", "params": {"source": "distances", "attr": "distance", "effective_at": 20}},
    ], atomic=True)])
    result = program.run_durable(store)
    assert [result.node_attr(n, "distance") for n in range(3)] == [0.0, 2.0, 5.0]
    store.close()
```

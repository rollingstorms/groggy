# Atomic property builder runs

`builder.build(atomic=True)` opts a supported property-only builder pipeline into
staged execution. With `persist=True`, attachments are visible to later property
loads in the run, and success publishes one mixed node/edge batch. A failed or
cancelled run discards its staged writes.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
graph.add_node(score=7)
before = graph.current_state_id
prior = graph.read_view()
builder = AlgorithmBuilder("atomic property update")
builder.attach_as("score", builder.init_nodes(default=3.0000000001))
builder.attach_as("copied", builder.load_attr("score"))
builder.attach_as("score", builder.init_nodes(default=4.0))
graph.view().apply(builder.build(atomic=True), persist=True)
assert graph.current_state_id == before + 1
assert graph.get_node_attr(0, "score") == 4.0
assert graph.get_node_attr(0, "copied") == 3.0000000001
assert prior.node_attr(0, "score") == 7
assert not graph.has_uncommitted_changes()
```

## Staging and publication

- The native Context owns only touched property columns; it does not clone the
  graph, write tentative pool values or publish intermediate versions.
- Node/edge attachment stages validated values. Node/edge loads, edge-weight scale
  and smoothing read staged values first, then the unchanged owner graph.
  Repeated assignments across steps use the last value for each entity/property;
  duplicate targets within one supplied column still reject.
- Kind normalization happens before staging, so later loads see the same logical
  representation as an ordinary graph write. Final publication uses GraphBatch's
  full validation and one acceptance clock. An explicit checkpoint remains separate.
- Capture a clean published head before execution and recheck it before accepting.
  A stale/dirty head rejects. This explicit mode publishes once even when ordinary
  graph mutations use manual compatibility; it does not change that policy.
- A run with no property assignments, including empty attachment columns, publishes
  nothing. Cancellation is checked before accepting the staged batch. Native
  Context staging is cleared after ordinary execution errors/cancellation and can
  be reused.

## Supported steps and order

Native steps declare `supports_atomic_properties()`. The default is false, so
custom or unmigrated steps cannot accidentally bypass the staging contract.
Current supported families are initialization/counts, property loads/attachments,
scalar/vector arithmetic, normalization, variable flow/alias, stroke smoothing,
fused arithmetic/neighbor kernels, node degree and incident-weight sums,
undirected triangle/k-core calculations, property filtering and ordering, checked
statistical/category/neighbor aggregations, declarative expression maps and
neighbor/in-place label transformations, node/edge/reservoir mask sampling, and
ordered or synchronous message-pass blocks.
The declaration means writes use StepScope attachments and property reads honor
its overlay; topology remains unchanged. Unsupported steps reject while building
the native pipeline, before any step executes. Nested `iter.loop` bodies are also supported when every descendant is certified.
The complete body is checked even with zero iterations. Each iteration reads prior
staged assignments; a later failure discards the entire outer run. Loops check
cancellation between iterations and child steps. An alternate `batch_plan` rejects
in atomic mode because its executor bypasses property staging. Subgraph-producing sampling, temporal and other graph-dependent families still require migration/certification.
Sampler pipelines reject an atomic property request rather than ignoring it.

Atomic mode uses the builder's original ordered steps. The current IR optimizer
has no graph-property effect edges and can move attachments past later reads, so
its optimization pass is bypassed for atomic builds, including when optimize=True.
The normal build path keeps its existing optimizer behavior. Effect-aware
optimization remains AR7.8 work.

## Compatibility and limits

`build()` still defaults to atomic=False. Existing persistent pipelines keep their
behavior and are not certified as all-or-nothing runs. This explicit boundary is
being migrated before changing the default for every supported step family.
It covers one builder algorithm invocation, not an outer list of independent
algorithms. Native mutable storage escapes remain outside checked guarantees.

`persist=False` retains [transient output semantics](transient-builder-outputs.md):
no staging overlay is installed, graph reads stay unchanged and owned output
columns are returned. This mode does not implement a private simulated graph.
No run/provenance entities or intermediate mutation history are created.

The current path buffers property values in ordered maps and may execute more
steps than an optimized legacy build. Temporary memory scales with touched cells;
lookups/insertion are ordered-map operations. No constant-time, memory-budget,
FFI/GIL or production throughput guarantee is implied. Durable journal integration,
full arbitrary-step transactions and multi-client service isolation are separate.

## Nested iterations

The builder loop API produces the same native `iter.loop` steps. Atomic mode
keeps their original ordered body; transient execution still reads the owner graph.
No intermediate iteration publishes a version. Empty loops publish nothing.
Recursive validation instantiates loop descendants before any outer step runs;
large nested definitions therefore have a preparation cost. Loop working variables
are temporary computation values and are not separately recorded in graph history.

```python
loop_builder = AlgorithmBuilder("iterated properties")
with loop_builder.iterate(3):
    loop_builder.attach_as("score", loop_builder.load_attr("score") + loop_builder.init_nodes(default=1))
before_loop = graph.current_state_id
graph.view().apply(loop_builder.build(atomic=True), persist=True)
assert graph.get_node_attr(0, "score") == 7.0
assert graph.current_state_id == before_loop + 1
```

## Fused kernels and message passing

Explicit `core.fused_axpy`, `core.fused_madd` and
`graph.fused_neighbor_mul_agg` steps operate on run variables. Attach their results
through the usual property steps. Neighbor multiplication happens **before**
aggregation: sum each neighbor's value times that neighbor's scalar.

Both ordered and synchronous `message_pass` blocks update temporary variables. Property loads before
a block see prior staged attachments. A subsequent attachment enters the same outer
batch. Nodes run in ID order for node maps and existing column order for columns;
with `ordered=True`, updates become visible to later nodes (Gauss-Seidel).
With `ordered=False`, every node reads the same block input (Jacobi).

```python
message_graph = Graph(directed=True)
message_graph.add_nodes([{"label": 9}, {"label": 2}, {"label": 2}])
message_graph.add_edges([(0, 1), (0, 2)])
message_builder = AlgorithmBuilder("atomic messages")
labels = message_builder.load_attr("label")
with message_builder.message_pass(target=labels, include_self=True, direction="out") as messages:
    update = message_builder.core.mode(messages.pull(labels), tie_break="lowest")
    messages.apply(update)
message_builder.attach_as("label", labels)
message_before = message_graph.current_state_id
message_graph.view().apply(message_builder.build(atomic=True), persist=True)
assert message_graph.current_state_id == message_before + 1
assert [message_graph.get_node_attr(n, "label") for n in range(3)] == [2, 2, 2]
```

- Directions are explicit traversal choices: `in` (default), `out`, or
  `undirected`/`both`. Invalid names reject. Selected parallel edges contribute
  repeatedly; each stored self-loop contributes once. `include_self` adds one
  self contribution only when no selected self-loop already supplies it.
- Block options are now read from their JSON payload. Earlier wrapped options
  could silently use defaults. Direction now controls the selected CSR rather
  than sharing an outgoing-neighbor call for every direction. These fixes also
  apply to ordinary non-atomic blocks.
- Integer block arithmetic retains integer results and truncating integer
  division, with explicit overflow and zero-division errors. Floating division
  by zero also errors. These errors discard the outer atomic run's staged writes.
- Atomic blocks reject streaming and unsupported body operations
  before any pipeline step executes. Supported conditional operations are described
  below; malformed conditional inputs also reject during preparation.
- A block builds a selected directional CSR once per invocation, then evaluates
  its body in order. Repeated blocks rebuild that CSR; directional cache reuse
  remains a possible optimization. These paths do not enable the IR optimizer
  for atomic builds or change the default `atomic=False` policy.

## Conditional message passing

Block `compare` now accepts the same six operators as `core.compare`: `eq`, `ne`,
`lt`, `le`, `gt`, `ge`. It returns a numeric 0.0/1.0 mask. Block `where` takes
exactly three inputs in order: condition, true value, false value. It now evaluates
the condition and selects the corresponding value, preserving that value's type.
This also repairs ordinary non-atomic blocks that previously returned the first
input unconditionally.

```python
conditional_graph = Graph()
conditional_graph.add_nodes([{"score": -1}, {"score": 0}, {"score": 2}])
conditional_builder = AlgorithmBuilder("conditional messages")
values = conditional_builder.load_attr("score")
with conditional_builder.message_pass(target=values) as messages:
    mask = conditional_builder.core.compare(values, "gt", 0)
    messages.apply(conditional_builder.core.where(mask, 10, -10))
conditional_builder.attach_as("selected", values)
conditional_graph.view().apply(conditional_builder.build(atomic=True), persist=True)
assert [conditional_graph.get_node_attr(n, "selected") for n in range(3)] == [-10, -10, 10]
assert conditional_graph.current_state_id == 2
```

Block/map mask selection shares one native helper. Existing numeric-mask behavior
is retained: true means `abs(value) > f64::EPSILON`; zero, values at/below that
threshold and NaN are false. Non-numeric masks reject. This is the existing
builder convention, not a new general truthiness rule. Comparison preserves the
existing f64 conversion/tolerance behavior, including its precision limitations
for large integers; this change does not introduce exact typed comparisons.

Only the selected `where` value is resolved at that point. Earlier body operations
still execute eagerly in recorded order; `where` does not skip a division or other
computation already present in the body. A missing selected value or invalid mask
raises an error and discards the outer atomic run's staged writes. Incorrect
input counts and unknown/non-text comparison operators reject before any outer
step executes, including for an empty graph. Old two-input placeholder `where`
specifications now reject explicitly.

## Synchronous message passing

`ordered=False` now means Jacobi updates for both ordinary and atomic builders.
All target reads, including self and neighbor reads, use the target values captured
at block entry. Results are buffered separately and installed after all nodes
succeed. Later blocks/iterations see the completed result. Synchronous evaluation
currently runs serially; this option does not promise parallel execution.

```python
synchronous_graph = Graph(directed=True)
synchronous_graph.add_nodes([{"score": 1}, {"score": 2}, {"score": 3}])
synchronous_graph.add_edges([(0, 1), (1, 2)])
synchronous_builder = AlgorithmBuilder("synchronous messages")
synchronous_values = synchronous_builder.load_attr("score")
with synchronous_builder.message_pass(
    target=synchronous_values, ordered=False, include_self=False, direction="in"
) as messages:
    messages.apply(synchronous_builder.graph_ops.neighbor_agg(synchronous_values, "sum"))
synchronous_builder.attach_as("answer", synchronous_values)
synchronous_before = synchronous_graph.current_state_id
synchronous_graph.view().apply(synchronous_builder.build(atomic=True), persist=True)
assert [synchronous_graph.get_node_attr(n, "answer") for n in range(3)] == [0, 1, 2]
assert synchronous_graph.current_state_id == synchronous_before + 1
```

- Target maps retain their map type; columns retain their original node order and
  column type. Direction, parallel-edge, self-loop, conditional and include-self
  behavior is shared with ordered mode.
- An empty body preserves the target values; an empty graph produces no property
  assignments or accepted state. A later node failure discards pending block
  outputs and leaves the source variable intact. Atomic runs additionally discard
  any earlier staged property attachments. Cancellation checks remain between
  nodes/body operations and immediately before installing the completed results.
- Synchronous mode adds one output buffer proportional to the target node count.
  It does not clone the graph or publish per-node states. Streaming and outer
  pipeline transactions remain separate work.

## Degree and incident-weight steps

`core.node_degree`, `core.weighted_degree` and `core.edge_weight_sum` now support
atomic execution, including nested loops. They return temporary node variables;
attachments enter the shared outer batch. Weighted steps read staged edge
properties before falling back to owner values. In transient mode they continue
to read owner properties and return owned output columns.

```python
degree_graph = Graph(directed=True)
degree_graph.add_nodes(3)
degree_graph.add_edges([(0, 1), (0, 1), (0, 0), (2, 0)])
degree_builder = AlgorithmBuilder("atomic degree")
degree_values = degree_builder.node_degrees(degree_builder.init_nodes(default=0))
degree_builder.attach_as("degree", degree_values)
degree_before = degree_graph.current_state_id
degree_graph.view().apply(degree_builder.build(atomic=True), persist=True)
assert [degree_graph.get_node_attr(n, "degree") for n in range(3)] == [3, 0, 1]
assert degree_graph.current_state_id == degree_before + 1
```

Degree retains its existing selected-topology convention: outgoing degree on a
directed graph; both endpoints on an undirected graph, with each stored self-loop
counted once. Parallel edges count separately. An explicit source map/column selects
which nodes receive results; a missing or incorrectly typed source variable now
errors rather than falling back to all nodes. Source nodes outside the active
selection reject. Native results are checked i64 counts.

Both weighted primitives retain **total incidence weight** on directed and
undirected graphs: add each selected edge's weight to both endpoints, counting
self-loops twice. This is their existing definition; it differs from unweighted
self-loop counting above. The duplicate implementations now use one shared
native kernel. Missing weight properties default to 1; explicit null, booleans,
text, vectors and other nonnumeric values now error rather than silently becoming
1. Signed finite scalar weights are accepted. Nonfinite weights and overflowing
nonfinite sums reject, preserving atomic rollback. Float64 inputs retain f64
precision; integer weights convert to f64 for summation.

Selected edges are processed in cached ID order. The first ordering preparation
may sort edges; subsequent calls reuse it. Degree performs one selected-edge pass;
weighted primitives share one incidence pass. Undirected triangle and k-core
execution is described below.

## Undirected triangles and k-core membership

`core.triangle_count` and `core.k_core_mark` now support atomic pipelines and
nested loops on undirected graphs. Their outputs are temporary node maps; attach
them to publish with the surrounding run. No topology changes are made.

- Triangles contain three distinct nodes. Parallel edges do not multiply counts,
  and self-loops do not form triangles. Each triangle contributes one to each of
  its three nodes. Counts are checked i64 values.
- K-core repeatedly removes nodes whose remaining selected degree is less than
  `k`. It retains the existing undirected degree convention: parallel edges count
  separately and each stored self-loop counts once. `k=0` includes all selected
  nodes, including isolates. Negative `k` rejects during preparation.
- Both steps use only selected active nodes/edges and preserve original IDs.
  Triangle counting uses a simple projection; k-core preserves multiplicities.
  Cancellation/errors leave existing result variables intact and discard the
  outer atomic run's earlier staged property writes.
- Directed calls now explicitly reject. The previous implementation mixed
  outgoing neighbor counts with incorrect peeling/closure behavior; directed
  triangle definitions and directed core constraints require separate support.
  This correction applies to ordinary builders too.

Native step specifications can be composed using the same algorithm handle
interface as other builder pipelines:

```python
import json
from groggy import AttrValue
from groggy.algorithms.base import AlgorithmHandle

class StructuralSteps(AlgorithmHandle):
    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        steps = [
            {"id": "core.triangle_count", "params": {"target": "triangles"}},
            {"id": "core.k_core_mark", "params": {"k": 2, "target": "core"}},
            {"id": "core.attach_node_attr", "params": {"source": "triangles", "attr": "triangles"}},
            {"id": "core.attach_node_attr", "params": {"source": "core", "attr": "core"}},
        ]
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": steps}))}}

structural_graph = Graph()
structural_graph.add_nodes(4)
structural_graph.add_edges([(0, 1), (1, 2), (2, 0), (0, 1), (0, 0), (2, 3)])
structural_before = structural_graph.current_state_id
structural_graph.view().apply(StructuralSteps(), persist=True)
assert [structural_graph.get_node_attr(n, "triangles") for n in range(4)] == [1, 1, 1, 0]
assert [structural_graph.get_node_attr(n, "core") for n in range(4)] == [1, 1, 1, 0]
assert structural_graph.current_state_id == structural_before + 1
```

Preparation builds a local adjacency list with selected multiedges, then sorts its
rows. K-core's peeling pass is linear in that adjacency size. Triangle counting
orients the simple projection by node ID and intersects forward lists, visiting
each triangle once; dense workloads can still require cubic work. Temporary
memory scales with selected topology plus node results. This is not a constant
memory or production throughput guarantee.

## Property filters, sorting and top-k

`core.filter_nodes_by_attr`, `core.filter_edges_by_attr`,
`core.sort_nodes_by_attr` and `core.top_k` now support atomic execution and nested
loops. Property readers use the shared overlay; transient execution keeps reading
the owner graph. Filters return sparse maps containing 1 for matching IDs. Sorting
returns a rank for every selected node, and top-k returns ranks for at most k
source IDs. These operations do not change the graph's selected topology or
preserve original scores in their rank output. Attach explicitly to publish.

```python
class FilteringSteps(AlgorithmHandle):
    @property
    def id(self):
        return "builder.step_pipeline"

    def to_spec(self):
        steps = [
            {"id": "core.init_nodes", "params": {"target": "scores", "value": 10}},
            {"id": "core.attach_node_attr", "params": {"source": "scores", "attr": "score"}},
            {"id": "core.filter_nodes_by_attr", "params": {"attr": "score", "target": "mask", "predicate": {"type": "gt", "value": {"type": "Int", "value": 5}}}},
            {"id": "core.top_k", "params": {"source": "scores", "target": "top", "k": 2}},
            {"id": "core.attach_node_attr", "params": {"source": "mask", "attr": "selected"}},
            {"id": "core.attach_node_attr", "params": {"source": "top", "attr": "rank"}},
        ]
        return {"id": self.id, "params": {"steps": AttrValue(json.dumps({"atomic": True, "steps": steps}))}}

filter_graph = Graph()
filter_graph.add_nodes([{"score": 2}, {"score": 0}, {"score": 1}])
filter_before = filter_graph.current_state_id
filter_graph.view().apply(FilteringSteps(), persist=True)
assert [filter_graph.get_node_attr(n, "selected") for n in range(3)] == [1, 1, 1]
assert [filter_graph.get_node_attr(n, "rank") for n in range(3)] == [0, 1, None]
assert filter_graph.current_state_id == filter_before + 1
```

Predicates are decoded from their JSON payload, repairing the earlier tagged-wrapper
failure. Eq/Ne retain typed equality; e.g. integer 1 differs from floating 1.0.
Relational and range comparisons share the exact numeric comparator used by the
numerical core, avoiding mixed integer/float rounding near 2**53. Missing properties
do not match; present null can match an explicit None equality predicate. Other
property types retain their existing predicate support; values that cannot be
converted to step values do not match. Read errors propagate instead of silently
becoming nonmatches. Reversed or NaN range bounds reject during preparation.

Sort/top-k accept one comparable domain at a time: numeric scalars (including
ordered infinities), text, or booleans. Missing/null sort last in either direction.
NaNs, collections, unsupported property conversions and mixed domains reject
rather than being treated as equal. Equal values use ascending entity ID as a
stable tie-break in either direction. Source IDs must belong to the active selection.

Ordering validates values before sorting. Cancellation is checked during reads,
around sorting and before installing results; comparison sorting itself is not
interruptible. Failure leaves existing result variables intact and atomic runs
discard earlier staged writes. Sort/top-k still perform a full O(n log n) sort;
no heap-selection optimization is claimed. Statistical and neighbor aggregations are now supported as described in
[Checked builder aggregations](builder-aggregations.md), including the histogram
ownership correction and the standalone/message-pass distinction.

Expression and transformation semantics, examples and costs: [checked builder transformations](builder-transformations.md).

Mask sampling controls and examples: [checked builder sampling](builder-sampling.md).

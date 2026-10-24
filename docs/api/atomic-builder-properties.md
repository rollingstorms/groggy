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
fused arithmetic/neighbor kernels and ordered message-pass execution blocks.
The declaration means writes use StepScope attachments and property reads honor
its overlay; topology remains unchanged. Unsupported steps reject while building
the native pipeline, before any step executes. Nested `iter.loop` bodies are also supported when every descendant is certified.
The complete body is checked even with zero iterations. Each iteration reads prior
staged assignments; a later failure discards the entire outer run. Loops check
cancellation between iterations and child steps. An alternate `batch_plan` rejects
in atomic mode because its executor bypasses property staging. Sampling/temporal and other graph-dependent families still require migration/certification.
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

Ordered `message_pass` blocks also update temporary variables. Property loads before
a block see prior staged attachments. A subsequent attachment enters the same outer
batch. Nodes run in ID order for node maps and existing column order for columns;
updates become visible to later nodes in the block (Gauss-Seidel).

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
- Atomic blocks reject streaming, `ordered=False` and unsupported body operations
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

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
scalar/vector arithmetic, normalization, variable flow/alias and stroke smoothing.
The declaration means writes use StepScope attachments and property reads honor
its overlay; topology remains unchanged. Unsupported steps reject while building
the native pipeline, before any step executes. Nested `iter.loop` bodies are also supported when every descendant is certified.
The complete body is checked even with zero iterations. Each iteration reads prior
staged assignments; a later failure discards the entire outer run. Loops check
cancellation between iterations and child steps. An alternate `batch_plan` rejects
in atomic mode because its executor bypasses property staging. Fused/sampling/
temporal and other graph-dependent families still require migration/certification.
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

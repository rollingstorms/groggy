# Checked builder aggregations

The native reduction, statistics, category and neighbor aggregation steps now work
with `build(atomic=True)`. Property loads see earlier staged assignments; results
remain temporary until attached. One successful run publishes one mixed property
batch, using the normal shared acceptance clock. Histogram-only runs publish no
state. `persist=False` retains owner reads and returns owned outputs.

## Statistics and numerical consistency

`builder.core` exposes `std`, `median`, `quantile(values, q)`,
`mode_values`, `entropy`, and `reduce_nodes(values, reducer="sum")`, alongside
`reduce_scalar`. These methods lower through both Python builder paths. Native
algorithms and the shared numerical foundation own all calculations.

- Sum/mean use the exact product accumulator, including integer residuals when
  mixed with floats. Mean divides the accumulator before rounding, avoiding an
  overflowing intermediate sum. Both reduction entry points share this kernel.
- Standard deviation uses the checked centered sample kernel, retaining integer
  differences beyond f64's exact integer range. Median/quantile use the shared
  exact-order, checked linear quantile implementation.
- Numeric inputs must be finite scalar integers/floats. Present null, Boolean,
  text and collection values reject. Nonfinite final results also reject. Scalar
  statistical outputs are f64; mode retains the selected category's type.
- Empty reduce_nodes/std/median/quantile/mode_values yield None. Empty entropy is
  0, and std of a singleton is 0. The existing reduce_scalar empty result remains
  0, including its scalar pass-through behavior. Quantile q must be finite in
  [0,1]. None is an explicit value if deliberately broadcast as a property.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

stats_graph = Graph()
stats_graph.add_nodes([{"value": 2**60 + i} for i in range(3)])
stats_builder = AlgorithmBuilder("centered sample spread")
values = stats_builder.load_attr("value")
spread = stats_builder.core.std(values)
stats_builder.attach_as("spread", stats_builder.core.broadcast_scalar(spread, values))
stats_before = stats_graph.current_state_id
stats_graph.view().apply(stats_builder.build(atomic=True), persist=True)
assert stats_graph.get_node_attr(0, "spread") == 1.0
assert stats_graph.current_state_id == stats_before + 1
```

## Categories and owned histogram results

Mode/entropy count typed serialized categories rather than debug strings. Mode
ties select the category whose contributor has the smallest node ID. Entropy
accumulates categories in deterministic key order and returns Shannon entropy in
bits. Nonfinite floating category values reject. `mode_values` is a global
statistic; existing `core.mode` continues to reduce per-node lists with its own
explicit tie policy.

Histogram bins are counts, so `core.histogram` now returns a scalar count vector,
with an owned copy at `profile["outputs"][f"histograms.{handle.name}"]`. Its length
is the requested positive bin count, including zero bins. Empty input returns all
zeros; constant input enters bin zero. Internal bins are left-inclusive and the
maximum enters the final bin. Relative differences preserve narrow integer ranges
and close floating values; wide ranges use scaled distances to avoid overflow.

```python
hist_graph = Graph()
hist_graph.add_nodes([{"value": value} for value in [0, 1, 2, 3]])
hist_builder = AlgorithmBuilder("owned bins")
histogram = hist_builder.core.histogram(hist_builder.load_attr("value"), bins=2)
hist_before = hist_graph.current_state_id
_, hist_profile = hist_graph.view().apply(
    hist_builder.build(atomic=True), persist=True, return_profile=True
)
assert hist_profile["outputs"][f"histograms.{histogram.name}"] == [2, 2]
assert hist_graph.current_state_id == hist_before
```

Attaching the histogram handle directly as a node map now rejects instead of
writing bin counts onto unrelated nodes. This repairs the old bin-index/node-ID
alias. Explicitly broadcasting the scalar vector is a separate caller choice.
Histogram allocation is checked, but callers still choose and fund the bin count.

Histogram outputs are observable without an attachment. Builds containing these
steps keep their original ordered program until optimizer effects model owned
outputs; this also applies to ordinary non-atomic builds and nested loop bodies.

## Standalone neighbor aggregation

`builder.graph_ops.neighbor_agg(values, agg, weights=..., direction=...)` supports
sum, mean, min, max and mode. Direction defaults to `in`; `out` and `undirected`
(or `both`) are explicit choices over stored endpoints. Parallel selected edges
contribute repeatedly, and each stored self-loop contributes once.

```python
neighbor_graph = Graph(directed=True)
neighbor_graph.add_nodes([{"value": value} for value in [1, 2, 3]])
neighbor_graph.add_edges([(0, 1), (1, 2)])
neighbor_builder = AlgorithmBuilder("weighted incoming values")
values = neighbor_builder.load_attr("value")
weights = neighbor_builder.init_nodes(default=2)
result = neighbor_builder.graph_ops.neighbor_agg(values, weights=weights, direction="in")
neighbor_builder.attach_as("answer", result)
neighbor_before = neighbor_graph.current_state_id
neighbor_graph.view().apply(neighbor_builder.build(atomic=True), persist=True)
assert [neighbor_graph.get_node_attr(n, "answer") for n in range(3)] == [0, 2, 4]
assert neighbor_graph.current_state_id == neighbor_before + 1
```

Weights are per-contributor node multipliers for sum, not edge attributes. Other
weighted reducers reject explicitly. Missing source-map rows are omitted, retaining
sparse-map semantics; present invalid numeric values reject. A contributing node
must have a finite numeric weight when weights are supplied. Checked weighted
sums accumulate products before final rounding, avoiding intermediate product
overflow when the final result is representable. Empty sum/mean is 0; empty
min/max/mode is None. Mode uses the same typed categories and contributor-ID ties.

Selected nodes/edges must remain active and closed under endpoints. The native
step builds sorted directional CSR, checks cancellation and installs its result
only on success. A later error discards the outer atomic run's staged writes.
Message-pass bodies continue to support their documented unweighted sum operation;
weighted or other neighbor reducers there explicitly reject. Use standalone steps
for the complete aggregation set.

## Optimizer parity and limits

Direction and weight operands now survive both Python lowering paths and optimizer
rebuilds. The old rewrite of neighbor aggregation followed by output scaling was
removed: the fused neighbor primitive multiplies each contributor before reducing,
which changes the calculation when scales vary by node.

```python
parity_graph = Graph(directed=True)
parity_graph.add_nodes([{"value": value, "scale": scale}
                       for value, scale in [(1, 10), (2, 100), (3, 1000)]])
parity_graph.add_edges([(0, 1), (1, 2), (2, 0)])
parity_builder = AlgorithmBuilder("scale completed neighbor results")
values = parity_builder.load_attr("value")
scales = parity_builder.load_attr("scale")
result = parity_builder.graph_ops.neighbor_agg(values, direction="in")
parity_builder.attach_as("answer", result * scales)
parity_graph.view().apply(parity_builder.build(optimize=True), persist=True)
assert [parity_graph.get_node_attr(n, "answer") for n in range(3)] == [30, 100, 2000]
```

The corrected path may retain more temporary variables than the removed rewrite.
Statistics sort/allocate where needed, histogram storage scales with the bin count,
and standalone neighbor steps rebuild directional CSR per invocation. Cancellation
is checked around numerical kernels and before publication; sorting and individual
numerical kernels do not promise bounded interruption latency. These are library
semantics, not production memory/latency guarantees or multi-client isolation.
Streaming, arbitrary outer transactions, remaining step families and atomic-default
migration remain separate roadmap work.

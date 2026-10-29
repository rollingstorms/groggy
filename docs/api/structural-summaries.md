# Structural summaries on exact selected topology

`Graph`, connected `Subgraph`, and immutable `GraphView` expose
`clustering_coefficient(node_id=None)`, `transitivity()` and `density()`.
GraphView also exposes owned, ascending-ID `clustering_coefficients()` rows.
Python's former clustering/transitivity placeholders now delegate to Rust.

## Simple undirected clustering

Clustering and transitivity use the **unweighted simple undirected projection**
of the exact selected edges. Parallel edges coalesce and self-loops are ignored.
For a node with `d` distinct nonself neighbors, local clustering is the number
of edges between those neighbors divided by `d*(d-1)/2`. It is zero for `d < 2`.
The mean includes isolates and degree-one nodes as zeros. Transitivity is closed
connected triples divided by all connected triples; a triangle closes three
triples. Empty input and zero-denominator summaries return zero.

```python
import groggy as gr

graph = gr.Graph()
graph.add_nodes(4)
for source, target in [(0, 1), (1, 2), (0, 2), (0, 1), (0, 0)]:
    graph.add_edge(source, target)
view = graph.read_view()
assert graph.clustering_coefficient() == 0.75
assert graph.clustering_coefficient(node_id=0) == 1.0
assert graph.clustering_coefficient(node_id=3) == 0.0
assert graph.transitivity() == 1.0
assert view.clustering_coefficients() == [(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)]
```

Only selected edge instances can supply neighbors or close a triangle. An
excluded owner edge cannot contribute merely because its endpoints remain in the
selection. Values/weights on edges do not affect these unweighted summaries.
Directed clustering/transitivity definitions are not implemented: directed input
rejects explicitly, consistent with the existing triangle and k-core builder
steps. There is no silent undirected reinterpretation.

```python
wedge = graph.subgraph_from_edges([0, 1])
assert wedge.clustering_coefficient() == 0.0
assert wedge.transitivity() == 0.0
assert wedge.pin_view().clustering_coefficient() == 0.0
graph.remove_node(1)
assert view.transitivity() == 1.0  # old capture remains unchanged
assert graph.transitivity() == 0.0
```

A local node ID must be a nonnegative integer and belong to the selection. Boolean,
fractional, negative, overflowing and out-of-scope IDs reject. Summary reads do
not create properties, states or group revisions.

## Edge-instance density

Density retains edge-instance semantics: `m/[n*(n-1)]` for directed input and
`2*m/[n*(n-1)]` for undirected input, with zero for fewer than two nodes. All
selected instances, including parallel edges and loops, contribute to `m`, so
density can exceed one. This differs deliberately from the simple projection
used for clustering. Counts convert before multiplication to avoid usize
arithmetic overflow. Connected native/Python density now honors graph direction;
trait and captured paths use the same definition.

```python
directed = gr.Graph(directed=True)
directed.add_nodes(2)
directed.add_edge(0, 1)
assert directed.density() == 0.5
assert directed.induced_subgraph([0, 1]).density() == 0.5
assert directed.read_view().density() == 0.5
```

## Shared native kernels and preparation

Native `Subgraph.structural_read()`, `GraphView.structural_read()` and
`AlgorithmRead.structural_read()` produce an immutable `UndirectedStructure` for
repeated Rust computations. It shares sorted IDs and selected CSR, with no
property payloads, retained history, writable graph owner or fake GraphView.
The input survives later rewiring and owner release and can run on workers.
GraphView preparation uses its captured topology; connected preparation validates
current selected membership and reuses its instance/version-aware CSR cache.

Python Graph/Subgraph adapters prepare this topology-only input under the GIL,
then release the GIL for clustering/transitivity. Python GraphView convenience
methods release the GIL for both preparation and computation. The native reusable
input is not currently a separate Python object; `clustering_coefficients()` is
the bulk Python path for all local coefficients.

The existing builder `core.triangle_count` and structural summaries share one
checked, cancellation-aware triangle kernel. Its oriented-neighbor intersections
count each triangle once. Global summaries allocate a simple adjacency projection;
local summaries examine the selected ego neighborhood with temporary row storage,
without retaining all neighbor-pair combinations. Dense triangle counting can
still be expensive. Setup, sparse chains, a high-degree star and a dense clique
are measured separately in `benches/structural_summaries.rs`; this is no universal
large-graph guarantee.

Connected Subgraph retains live ownership; explicit GraphView input remains
immutable. These completed summaries do not complete the default connected-view
migration or all nonregistry analytics certification.

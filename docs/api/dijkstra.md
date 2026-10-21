# Dijkstra algorithm wrapper

`DijkstraShortestPath::new(start_attr, weight_attr, output_attr)` and
`algorithms.pathfinding.dijkstra(start_attr, weight_attr=None, output_attr="distance")`
retain their signatures. The registered single-source algorithm now validates its
source explicitly, supports traversal cancellation, and returns nonpersistent f64
results.

```python
from groggy import algorithms
from groggy.pipeline import apply

result, profile = apply(
    selection,
    algorithms.pathfinding.dijkstra("source", "length", "distance"),
    persist=False, return_profile=True,
)
rows = profile["outputs"]["pathfinding.dijkstra.distances"]
```

Exactly one selected node must carry `start_attr`. The value is ignored, including
false, zero and explicit null; it is not another node ID. Missing or ambiguous
markers reject before any result write. Empty input rejects for lack of a source;
a marked isolated singleton returns `(node, 0.0)`. This replaces an arbitrary
first-marker choice and corrects the old truthy/multiple-source documentation.

Every selected node/edge must be active and every edge endpoint selected, checked
before cache access. Every selected weight, including on unreachable edges, must
be finite nonnegative numeric Float/Int/SmallInt; missing is one. Zero lengths and
zero cycles are valid. Directed graphs follow outgoing edges; undirected graphs
traverse both ways. Parallel endpoint pairs use their minimum length without
changing underlying edge identities. Results contain distances, not path edge IDs.

The wrapper uses native heap/CSR traversal and shared heap ordering. Strict f64
improvements are retained, and nonfinite accumulated distances reject. Cancellation
is checked before validation, during heap traversal, and before final output.
No fixed latency is promised during setup or every high-degree neighbor scan.

Nonpersistent output is an owned list of `(original_node_id, f64_distance)` rows,
ordered by node ID, for reached nodes only. It uses the existing NodeScores
translation and writes no properties. Persistent runs validate all f32 distance
representations before one existing bulk property setter; overflow rejects with
no result values written. Rounding/underflow remains possible in f32 properties.
Previous output properties on unreached nodes remain unchanged. No new durable
transaction boundary or state-pinned historical input contract is introduced.

Traversal costs O((V+E) log(V+E)) with O(V+E) working entries, in addition to ordered
selection/preflight sorting. No speedup or production-scale capacity is claimed.
Low-level callback-based distance utilities and pipeline-step callers retain their
separate contracts and are not certified by this wrapper repair.

`tests/dijkstra_wrapper.rs` compares every three-node edge set in both graph
directions/every source, with zero/compact/missing weights, parallel instances,
loops and actual repeated cache hits, against independent Floyd-Warshall.
Further fixtures cover missing/ambiguous sources, cancellation, stale/open
selections, invalid unreachable weights, sparse/singleton/empty cases, tiny
improvements and f64 distances outside persistent f32 range.
`tests/test_dijkstra_wrapper.py` checks the rebuilt Python surface, marker values,
nonpersistent rows and rejection without partial writes. Broader external-backend,
scale, remaining caller and historical-input certification remains item 28 work.

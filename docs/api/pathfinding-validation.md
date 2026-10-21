# Checked traversal and Dijkstra inputs

Core Graph BFS/DFS require an active starting node. Shortest path and all-path enumeration require active start and end nodes, including the start-equals-end case. Unknown or deactivated IDs raise instead of appearing as phantom traversal results.

For `Graph.shortest_path(..., weight_attribute="weight")`, every active edge in the selected graph is checked before target early exit. Present weights must be integer (including compact integer storage) or floating scalars, finite and nonnegative. Zero is valid. A missing property retains the existing unit-weight default; explicit null, text, bool, NaN, infinity and negatives reject. Deactivated edges are excluded from this check. With no weight attribute, the unit-weight path remains unchanged.

```python
from groggy import Graph

graph = Graph()
a, b = graph.add_node(), graph.add_node()
edge = graph.add_edge(a, b, weight=3)
path = graph.shortest_path(b, a, weight_attribute="weight")
assert list(path.edge_ids) == [edge]
```

The selected path still retains exact edge-instance IDs and f64 total cost in Rust. Parallel edges remain separate facts. Graph-wide weighted validation costs O(E) in addition to the traversal; it is necessary before Dijkstra's early exit when weights have not been certified by a maintained index/schema.

The checked Dijkstra algorithm, weighted ShortestPathMap pipeline step and weighted closeness path share scalar validation and selection checks. Their distance-only endpoint map takes the minimum weight across selected parallel edges and supplies both orientations for an undirected graph. It does not deduplicate the underlying graph or change the direct path API's edge identity. Stale selected nodes/edges reject before computing/writing these checked results. Tiny improvements are compared directly rather than discarded using an absolute machine-epsilon offset; heap ordering/equality agree.

Dijkstra algorithm persistence still stores f32 property results. A distance outside finite f32 range now rejects before any result column write, instead of silently storing infinity. The pipeline variable map and direct Path cost use f64. This does not change the graph's wider property dtype contract.

## Remaining audit

These fixes do not complete all graph algorithms. Other weighted callers such as A*, betweenness and community methods still use the older weight collector and need their own domain/identity audits. Low-level callback-based Dijkstra utilities retain their existing signatures and require callers to supply valid nonnegative finite weights. Broader max-path/heuristic options, filters, cancellation, pinned view ownership, and algorithm-specific directed/weight definitions remain items 21/26/28/30. Algorithm persistence remains the legacy manual-grouping API, not a newly durable mutation boundary.

Tests cover live/dead endpoints, invalid/missing/zero/compact weights, reverse traversal, parallel minima, tiny-cost improvements, output range rejection, and direct distances compared with an independent Floyd-Warshall fixture. Python tests exercise the same public graph entry points. NetworkX was not available in the current test environment; no external graph backend comparison is claimed by this slice.

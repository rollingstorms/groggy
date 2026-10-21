# Checked traversal and weighted algorithms

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

## Checked A*

`pathfinding.astar` now uses the same selected-state and weight preflight, including unit defaults, compact integers, zero weights, reverse undirected traversal and minimum parallel-edge costs. It stores zero-based **node path order**, not distances or chosen edge IDs. Each marker attribute must be present on exactly one selected node (the marker's value does not select it). Missing/ambiguous markers reject. An unreachable goal returns an error. Only the selected path's output rows are written; unrelated pre-existing output values are not cleared.

An omitted heuristic is zero. For an explicit heuristic attribute, missing values are zero; present values must be finite nonnegative scalar numbers. The goal estimate must equal zero, and every selected oriented pair must satisfy `h(u) <= w(u,v) + h(v)`. Undirected graphs check both orientations; parallel edges use the minimum weight. This sufficient consistency requirement certifies early goal termination; admissible but inconsistent heuristics are currently rejected. Comparisons use the stored numeric values without an absolute epsilon. Invalid input rejects before result-column writes, including start-equals-goal. CSR cache construction may still occur before validation.

A* heap equality/order now agree and relaxation preserves tiny cost improvements. Path ties are not promised a canonical selection, and the output remains a node sequence, so callers needing selected edge identity should use the direct shortest-path API. Tests: `tests/astar_validation.rs` and `tests/test_astar_validation.py`.

## Checked node betweenness

`centrality.betweenness` validates selected active entities in both unit and weighted modes. A supplied length property must be finite, numeric and **strictly positive**; missing lengths are one. Zero is rejected because this Brandes implementation requires distances to advance along its predecessor DAG. Positive self-loops cannot be shortest paths and have no contribution. Unit mode ignores unused weight properties.

Paths count distinct edge instances. Equal minimum-length parallel edges contribute separate shortest paths; more expensive parallel edges are excluded. Weighted CSR is built locally and never replaces the shared all-edge topology cache. Unweighted CSR retains parallel instances. Directed raw scores count ordered source/target pairs; undirected raw scores divide by two. Normalized scores divide the ordered-pair accumulation by `(n-1)*(n-2)`, including isolates in `n`.

Weighted distance/tie comparisons use exact computed f64 values, without an absolute tolerance. A positive step that rounds away in f64 rejects, as do nonfinite path counts or results outside the persisted f32 score range. Failures preserve existing output properties. This is not arbitrary precision shortest-path counting; rational/decimal mathematical ties can differ after floating arithmetic. Validation adds O(E) work, and weighted execution needs local O(E) CSR storage. No speedup is claimed.

Checks include an independent exhaustive simple edge-path enumerator for directed/undirected, normalized/raw, weighted/unit multigraphs with loops and an isolate; cache reuse, tiny lengths, invalid weights, stale selections, rounded-away distances and path-count overflow are covered. See `tests/betweenness_validation.rs` and `tests/test_betweenness_validation.py`.

## Remaining audit

These fixes do not complete all graph algorithms. Other weighted callers such as Girvan–Newman still use the older weight collector and need their own domain/identity audits. Low-level callback-based Dijkstra utilities retain their existing signatures and require callers to supply valid nonnegative finite weights. Broader max-path/heuristic options, filters, cancellation, pinned view ownership, and algorithm-specific directed/weight definitions remain items 21/26/28/30. Algorithm persistence remains the legacy manual-grouping API, not a newly durable mutation boundary.

Tests cover live/dead endpoints, invalid/missing/zero/compact weights, reverse traversal, parallel minima, tiny-cost improvements, output range rejection, and direct distances compared with an independent Floyd-Warshall fixture. Python tests exercise the same public graph entry points. NetworkX was not available in the current test environment; no external graph backend comparison is claimed by this slice.

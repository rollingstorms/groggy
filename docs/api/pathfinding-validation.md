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

The checked Dijkstra algorithm and weighted ShortestPathMap pipeline step share scalar validation and selection checks. Weighted [closeness](closeness.md) additionally requires strictly positive lengths. Their distance-only endpoint map takes the minimum weight across selected parallel edges and supplies both orientations for an undirected graph. It does not deduplicate the underlying graph or change the direct path API's edge identity. Stale selected nodes/edges reject before computing/writing these checked results. Tiny improvements are compared directly rather than discarded using an absolute machine-epsilon offset; heap ordering/equality agree.

[Dijkstra wrapper](dijkstra.md) nonpersistent output now returns original-ID f64 distance rows and validates one selected source marker; it checks cancellation during traversal and before final output. Dijkstra algorithm persistence still stores f32 property results. A distance outside finite f32 range now rejects before any result column write, instead of silently storing infinity. The pipeline variable map and direct Path cost use f64. This does not change the graph's wider property dtype contract.

## Checked A*

`pathfinding.astar` now uses the same selected-state and weight preflight, including unit defaults, compact integers, zero weights, reverse undirected traversal and minimum parallel-edge costs. It stores zero-based **node path order**, not distances or chosen edge IDs. Each marker attribute must be present on exactly one selected node (the marker's value does not select it). Missing/ambiguous markers reject. An unreachable goal returns an error. Only the selected path's output rows are written; unrelated pre-existing output values are not cleared.

An omitted heuristic is zero. For an explicit heuristic attribute, missing values are zero; present values must be finite nonnegative scalar numbers. The goal estimate must equal zero, and every selected oriented pair must satisfy `h(u) <= w(u,v) + h(v)`. Undirected graphs check both orientations; parallel edges use the minimum weight. This sufficient consistency requirement certifies early goal termination; admissible but inconsistent heuristics are currently rejected. Comparisons use the stored numeric values without an absolute epsilon. Invalid input rejects before result-column writes, including start-equals-goal. Selected-state and weight preflight now run before cache access; marker/heuristic checks may still follow CSR construction.

A* heap equality/order now agree and relaxation preserves tiny cost improvements. Path ties are not promised a canonical selection, and the output remains a node sequence, so callers needing selected edge identity should use the direct shortest-path API. Tests: `tests/astar_validation.rs` and `tests/test_astar_validation.py`.

### A* nonpersistent output and cancellation

A nonpersistent run returns owned `(original_node_id, i64_path_index)` rows in
path order under `profile["outputs"]["pathfinding.astar.path"]`, using the existing
NodeIntegers bridge. `profile["stats"]["astar.path_cost"]` reports the f64 total
cost; `astar.count.path_nodes` reports path size. Persistent runs retain integer
path-index properties, so even a cost above f32 range is valid when its indices
fit i64. Start-equals-goal returns one row and zero cost.

```python
from groggy.pipeline import apply
from groggy import algorithms

result, profile = apply(
    selection,
    algorithms.pathfinding.astar("start", "goal", weight_attr="length"),
    persist=False, return_profile=True,
)
path_rows = profile["outputs"]["pathfinding.astar.path"]
```

Cancellation is checked before work, during heap traversal and before final
property/output publication, including trivial paths. Finite accumulated costs
and priorities are checked; path-index conversion rejects overflow. Unreachable
paths still raise without returning output rows. No fixed cancellation latency
is promised for every setup/high-degree scan. Existing node-to-CSR lookup may
scan selected IDs for each expansion; no throughput or scale improvement is
claimed. This operation retains the existing bulk mutation boundary and does
not complete state-pinned historical inputs.

`tests/astar_output.rs` compares all 512 three-node edge sets in both graph
directions and every source/goal pair, including zero-weight parallel instances,
against independent Floyd-Warshall costs. It checks actual cache reuse, path
adjacency/order, nonpersistent ownership, unreachable goals, cancelled trivial
paths, stale/open selections, sparse IDs and large f64 costs. Rebuilt Python
checks are in `tests/test_astar_output.py`. These supplement the earlier
heuristic/input tests without claiming external-backend or production-scale
certification.

## Checked node betweenness

`centrality.betweenness` validates selected active entities in both unit and weighted modes. A supplied length property must be finite, numeric and **strictly positive**; missing lengths are one. Zero is rejected because this Brandes implementation requires distances to advance along its predecessor DAG. Positive self-loops cannot be shortest paths and have no contribution. Unit mode ignores unused weight properties.

Paths count distinct edge instances. Equal minimum-length parallel edges contribute separate shortest paths; more expensive parallel edges are excluded. Weighted CSR is built locally and never replaces the shared all-edge topology cache. Unweighted CSR retains parallel instances. Directed raw scores count ordered source/target pairs; undirected raw scores divide by two. Normalized scores divide the ordered-pair accumulation by `(n-1)*(n-2)`, including isolates in `n`.

Weighted distance/tie comparisons use exact computed f64 values, without an absolute tolerance. A positive step that rounds away in f64 rejects, as do nonfinite path counts or results outside the persisted f32 score range. Failures preserve existing output properties. This is not arbitrary precision shortest-path counting; rational/decimal mathematical ties can differ after floating arithmetic. Validation adds O(E) work, and weighted execution needs local O(E) CSR storage. No speedup is claimed.

Checks include an independent exhaustive simple edge-path enumerator for directed/undirected, normalized/raw, weighted/unit multigraphs with loops and an isolate; cache reuse, tiny lengths, invalid weights, stale selections, rounded-away distances and path-count overflow are covered. See `tests/betweenness_validation.rs` and `tests/test_betweenness_validation.py`.

## Checked Girvan–Newman

`algorithms.community.girvan_newman(...)` exposes the existing Rust algorithm through a thin Python handle factory. It accepts undirected selected graphs and rejects directed input explicitly. All selected IDs and lengths are validated even with zero removal budget or a trivial graph. Weighted lengths are strictly positive finite numeric scalars, including compact integers, with missing values one. Unit mode ignores unused weight properties. Numerical limits match checked Brandes: exact computed f64 ties, advancing distances, finite path counts/dependencies; invalid input leaves previous output properties intact.

The local CSR stores edge-instance indices in both orientations. Equal-cost parallel edges contribute distinct shortest paths. Each iteration removes only one edge instance, recalculates scores, and permits more expensive surviving parallels to become shortest paths. Equal removal scores choose the lowest edge ID. Self-loops do not connect components or participate in removals/shortest paths; they remain in original-graph modularity, contributing twice to degree. The source graph's edges and shared topology cache are unchanged.

`num_levels` is a maximum number of individual non-loop edge removals. Zero returns initial components. The initial partition and every visited partition **after** removal are scored, including after the final permitted removal. Scoring always uses all original selected edge instances and degrees, rather than the progressively cut graph. Modularity is unweighted; supplied lengths do not mean connection strength. Equal computed partition scores retain the earlier partition. Community labels are assigned by component traversal in sorted node-ID order.

The default modularity stopping threshold is 0.0001. An explicit threshold must be finite and nonnegative; execution stops when the visited score falls below the best by more than that amount. Negative level budgets reject before unsigned conversion. The result is the best visited partition under this stopping rule, not a global modularity optimum or an exported dendrogram. Empty graphs, one-node graphs, isolates and disconnected selections are valid. Cancellation is checked before work and within shortest-path processing.

```python
from groggy import Graph, algorithms

graph = Graph()
nodes = [graph.add_node() for _ in range(6)]
for u, v in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]:
    graph.add_edge(nodes[u], nodes[v])
graph.induced_subgraph(nodes).apply(
    algorithms.community.girvan_newman(num_levels=1)
)
assert graph.get_node_attr(nodes[0], "community") != graph.get_node_attr(nodes[3], "community")
```

The edge-instance CSR and work buffers use O(V+E) additional memory. Recomputing all-source betweenness after each removal is expensive; this remains a small-graph algorithm and no throughput improvement is claimed. Tests include an independent exhaustive edge-path score reference before/after removals, parallel bridges, original-graph modularity, reversed stored orientations, loops, isolates, empty/singleton graphs, invalid input, numerical failures and cancellation. See `src/algorithms/community/girvan_newman_paths.rs`, `tests/girvan_newman_validation.rs` and `tests/test_girvan_newman_validation.py`.

## Remaining audit

These fixes do not complete all graph algorithms. Girvan–Newman has an explicit supported domain. Bounded Louvain/Leiden multilevel work is documented in [community modularity](community-modularity.md), with remaining limits made explicit. Other repaired algorithms have dedicated API documents; this does not certify every caller or input domain. Low-level callback-based Dijkstra utilities retain their existing signatures and require callers to supply valid nonnegative finite weights. Broader max-path/heuristic options, filters, cancellation, pinned view ownership, and algorithm-specific directed/weight definitions remain items 21/26/28/30. Algorithm persistence remains the legacy manual-grouping API, not a newly durable mutation boundary.

Tests cover live/dead endpoints, invalid/missing/zero/compact weights, reverse traversal, parallel minima, tiny-cost improvements, output range rejection, and direct distances compared with an independent Floyd-Warshall fixture. Python tests exercise the same public graph entry points. NetworkX was not available in the current test environment; no external graph backend comparison is claimed by this slice.

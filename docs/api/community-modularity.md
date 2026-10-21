# Louvain and Leiden local modularity

The checked Louvain and Leiden paths currently accept **undirected, unweighted selected graphs**. Edge attributes do not provide weights to these algorithms. Parallel edge instances remain separate; self-loops remain internal and contribute twice to degree. Selected nodes and edges must be active, and every selected edge's endpoints must be selected. Invalid or stale selections reject before output-column writes. Directed inputs reject explicitly, including trivial selections.

Both now optimize the resolution-dependent objective

```
Q_gamma = sum_C [L_C / m - gamma * (K_C / (2*m))^2]
```

Here `m` counts selected edge instances, `L_C` counts internal edge instances, and `K_C` sums degrees in a community. Gamma must be finite and strictly positive. It scales the null-model term, rather than multiplying the whole gain. This convention follows the [NetworkX modularity definition](https://networkx.org/documentation/networkx-3.2/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html). NetworkX is a reference only; Groggy executes its own Rust kernels.

The shared Rust `modularity_with_resolution` and `modularity_delta_with_resolution` helpers evaluate this objective and its full node-move gain. Existing `modularity` and `modularity_delta` signatures remain gamma-one wrappers. These low-level helpers require callers to supply consistent complete partition/degree/adjacency data and a valid gamma; algorithm constructors enforce gamma validity. Self-neighbors do not enter the changing internal-edge count because loops stay internal before and after a move. Parallel neighbors retain multiplicity. Community score terms are accumulated in sorted community-ID order.

## Selection and output behavior

Empty graphs are valid. Nonempty edgeless selections now receive distinct singleton labels in Louvain as well as Leiden. Isolates remain separate. Result writes retain the existing legacy bulk-column publication behavior and do not imply a durable checked-writer expansion. Parameter validation, cancellation or traversal failures leave prior result values intact.

Zero/negative iteration and phase budgets now reject instead of being silently coerced to one. Negative Leiden seeds reject instead of being cast to unsigned values. Cancellation is checked before trivial returns as well as during iteration.

Louvain visits sorted node IDs and candidate community IDs, accepting gains above its existing `1e-6` threshold. Leiden visits sorted IDs by default; an explicit seed shuffles that order with a local RNG. It does not alter global RNG state. Refinement/component processing and final labels have stable ordering. Same inputs, parameters and Leiden seed yield repeatable labels under the current implementation; this is not a promise to freeze partitions across future algorithm revisions.

## Remaining implementation gaps

These repairs complete the **local objective and input domain** slice. They do not make these full multilevel implementations:

- Louvain still performs one local node-move phase. Its positive `max_phases` argument is retained for compatibility but does not yet produce graph aggregation or additional coarse levels.
- Leiden currently alternates local moves and splitting disconnected communities. Its full refinement and aggregation procedure remains incomplete.
- Neither routine promises a globally optimal partition, arbitrary precision decisions, weighted modularity, or directed modularity.

The existing CSR/adjacency bulk structures remain in Rust. Validation adds O(V+E) work and a temporary endpoint map; sorted candidate sets and retained loop/parallel entries change costs, but no speedup is claimed. Large-graph performance and multilevel benchmarks remain future gates.

## Verification

`tests/community_resolution.rs` compares objective values and every possible single-node move against an independent adjacency-matrix definition across all 81 three-label assignments of a four-node multigraph, three resolutions, loops, parallel edges, and moves into an empty community. Additional tests cover gamma-dependent partitions, local objective checks, isolates, ID gaps, empty/edgeless inputs, invalid budgets, stale/directed selections, cancellation, seeded repeatability and global RNG isolation. `tests/test_community_resolution.py` verifies the Python factories and rebuilt Rust extension. This slice does not claim full backend partition parity.

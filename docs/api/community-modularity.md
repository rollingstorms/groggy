# Louvain and Leiden modularity levels

The checked Louvain and Leiden paths currently accept **undirected, unweighted selected graphs**. Edge attributes do not provide weights to these algorithms. Parallel edge instances remain separate; self-loops remain internal and contribute twice to degree. Selected nodes and edges must be active, and every selected edge's endpoints must be selected. Invalid or stale selections reject before output-column writes. Directed inputs reject explicitly, including trivial selections.

Both now optimize the resolution-dependent objective

```
Q_gamma = sum_C [L_C / m - gamma * (K_C / (2*m))^2]
```

Here `m` counts selected edge instances, `L_C` counts internal edge instances, and `K_C` sums degrees in a community. Gamma must be finite and strictly positive. It scales the null-model term, rather than multiplying the whole gain. This convention follows the [NetworkX modularity definition](https://networkx.org/documentation/networkx-3.2/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html). NetworkX is a reference only; Groggy executes its own Rust kernels.

The shared Rust `modularity_with_resolution` and `modularity_delta_with_resolution` helpers evaluate this objective and its full node-move gain for unit edge instances. Both multilevel algorithms use private count-weighted quotient kernels, verified against the same original-graph objective. Existing `modularity` and `modularity_delta` signatures remain gamma-one wrappers. These low-level helpers require callers to supply consistent complete partition/degree/adjacency data and a valid gamma; algorithm constructors enforce gamma validity. Self-neighbors do not enter the changing internal-edge count because loops stay internal before and after a move. Parallel neighbors retain multiplicity. Community score terms are accumulated in sorted community-ID order.

## Selection and output behavior

Empty graphs are valid. Nonempty edgeless selections now receive distinct singleton labels in Louvain as well as Leiden. Isolates remain separate. Result writes retain the existing legacy bulk-column publication behavior and do not imply a durable checked-writer expansion. Parameter validation, cancellation or traversal failures leave prior result values intact.

Zero/negative iteration and phase budgets now reject instead of being silently coerced to one. Negative Leiden seeds reject instead of being cast to unsigned values. Cancellation is checked before trivial returns as well as during iteration.

Louvain visits sorted node IDs and candidate community IDs, accepting gains above its existing `1e-6` threshold. Leiden shuffles its visit/refinement order with a local RNG; an omitted seed uses reproducible seed zero. It does not alter global RNG state. Same inputs, parameters and seed yield repeatable labels under the current implementation; this is not a promise to freeze partitions across future algorithm revisions.

## Louvain multilevel execution

Louvain now alternates local node moves and community aggregation. Each coarse connection carries the checked u64 count of original edge instances it represents. Edges internal to a community become self-loops; parallel connections combine by adding mass, never by dropping duplicates. A local weighted CSR carries that mass, with loops contributing twice to degree. The source graph and shared topology cache remain unchanged. Original nodes are mapped through every level and receive the final partition's labels; labels are opaque community numbers, not persistent graph entity IDs.

`max_iter` bounds sweeps at each level, while `max_phases` bounds levels. Execution stops earlier when no moves occur or no further aggregation is possible. Every level checks that modularity did not decrease beyond floating tolerance; aggregation checks that singleton coarse modularity matches its preceding partition. Count overflow, nonfinite objective/gain and violated invariants reject before result writes. Scoring still uses f64, so arbitrary-size exact count arithmetic or arbitrary precision decisions are not promised.

**Default change:** the registry and Python factory now use `max_phases=10` instead of the previous effective single phase. Python exposes the parameter after the existing `output_attr` argument, preserving the meanings of its first three positional arguments. Explicit `max_phases=1` requests the local-only variant. `Louvain::new` continues to use its caller-provided positive phase limit. Normal default partitions can change because later levels now run.

## Leiden refinement and aggregation

Leiden now runs a fast move queue, constrained singleton refinement, and aggregation based on the refined groups. The broader partition initializes the next quotient; it is not replaced by singleton coarse labels. This follows the stage distinction in [Traag, Waltman and van Eck, Algorithm A.2](https://traag.net/wp/wp-content/papercite-data/pdf/traag_leiden_algo_2018.pdf). The implementation optimizes modularity for the undirected unit-edge input domain, retaining count-weighted loops and parallels internally.

Refinement only merges a singleton into a connected neighboring group within its parent community. Degree-weighted attachment checks restrict eligible nodes and targets, and accepted merges have nonnegative gain. Random choices use stabilized exponential weights. `theta` is positive finite, defaults to 0.01, and is measured in **normalized modularity gain** units; its scaling and seeded trajectories need not match another library. Rust keeps the existing constructor and adds `with_theta`; Python appends `theta` after the existing positional arguments.

Fast moves can create a new singleton from an inherited group. `max_iter` now bounds queue visits as `max_iter * current_level_node_count`; overflow rejects. At a budget stop, disconnected remainders are split before result publication or aggregation. Every coarse node represents a connected refined group, so this repair preserves connected final communities when projected back to original IDs. Level scores cannot decrease beyond floating tolerance, and quotient scoring checks the inherited partition against the preceding original-level score. `max_phases` bounds levels (default ten). These are bounded heuristic results, not unbounded-convergence or subset-optimality certificates.

## Python scalar configuration

Raw Python scalar floats now reach algorithm parameters as f64 rather than being narrowed through f32 graph properties. Construction, sample pipelines and metadata type validation share one translation path. Existing explicit AttrValue parameters remain supported at their property precision; wrapping a value explicitly cannot recover precision already lost. Float-list/property precision is unchanged. Metadata validation checks parameter types; algorithm constructors enforce domains such as positive gamma/theta.

## Remaining verification and limits

- Neither routine promises global optimality, arbitrary precision decisions, user-supplied strength weights, directed modularity or CPM.
- Finite budgets do not establish the paper's asymptotic guarantees. External backend partition/quality comparisons and broader stochastic/large-scale validation remain open.

Graph structures and bulk result writes remain in Rust. The shared private quotient module is now `modularity_levels.rs`. Validation adds O(V+E) work and an endpoint map; quotient CSR, queue/refinement buffers and candidate maps require O(V+E) temporary storage. Bounded debug profiles are in `notes/benchmarks/louvain-levels-2026-10-21.md` and `notes/benchmarks/leiden-levels-2026-10-21.md`. Large-graph memory and release throughput remain future gates.

## Verification

`tests/community_resolution.rs` compares objective values and every possible single-node move against an independent adjacency-matrix definition across all 81 three-label assignments of a four-node multigraph, three resolutions, loops, parallel edges, and moves into an empty community. Additional tests cover gamma-dependent partitions, local objective checks, isolates, ID gaps, empty/edgeless inputs, invalid budgets, stale/directed selections, cancellation, seeded repeatability and global RNG isolation. `tests/test_community_resolution.py` verifies the Python factories and rebuilt Rust extension. This slice does not claim full backend partition parity.

`src/algorithms/community/modularity_levels.rs` checks quotient modularity against an independent original adjacency-matrix definition across all 243 three-label assignments of a five-node quotient and three resolutions, including loops, parallel mass, an isolate and a second aggregation. `tests/louvain_multilevel.rs` and `tests/test_louvain_multilevel.py` check actual improvement beyond local moves on a ring of cliques, monotonic phase scores, ID projection, repeatability, default/explicit phase limits and unchanged source topology.

`leiden_levels.rs` tests constrained connected-parent refinement, inherited coarse labels, moves into empty communities, degree eligibility, objective/connectedness invariants, extreme temperatures and invalid budgets. `tests/leiden_multilevel.rs` covers multilevel original-objective improvement, phase/visit bounds, connected output, ID gaps, loops/parallels, no-persistence mode and rejection before overwriting labels. `tests/test_leiden_multilevel.py` and `tests/test_algorithm_parameter_precision.py` exercise the rebuilt extension, temperature range and scalar f64 configuration.

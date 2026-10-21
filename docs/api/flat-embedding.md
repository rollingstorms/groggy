# Flat graph layout

`compute_flat_embedding` retains the native position API. The owned matrix helper
`compute_flat_embedding_matrix` and thin Python method share the same implementation:

```python
positions = graph.flat_embedding(
    embedding, iterations=100, learning_rate=0.03,
    edge_cohesion_weight=1.0, repulsion_weight=0.1,
    repulsion_power=1.5, spread_weight=0.05, radius_clip=0.98, seed=0,
)
```

## Topology and identity

Use every actual active edge instance, without a 100-edge cutoff or fabricated
chain/cross edges. Validate checked endpoints against the embedding's full active
node set. Unlabeled rows mean ascending active node IDs. Labeled rows must contain
one canonical `node_<id>` label for every active node, in their chosen row order;
reject duplicates, inactive IDs, malformed labels and row-count mismatches.
Output preserves that order and records its IDs; sparse IDs are not row offsets.

The objective uses graph topology. Input feature coordinates do not enter the
loss, so this graph layout does not promise to preserve high-dimensional metric
distances. Directed edges contribute the same geometric distance as reversed
edges, with each stored instance counted. Parallel instances contribute separately;
loops contribute zero cohesion but count in the edge mean's denominator.

A matrix alone cannot supply graph topology. The existing graph-free native
`compute_flat_embedding_from_matrix` now requires explicit zero cohesion for
multiple rows instead of constructing dummy edges. Empty/singleton graph-free
inputs remain valid. Supply the actual graph to retain cohesion. Graph-aware empty
input requires an empty graph; a nonempty matrix cannot silently stand for it.

## Working objective and gradients

For positions constrained to a disk, minimize:

- Cohesion weight times mean squared distance over edge instances.
- Repulsion weight times the mean of `(squared_distance + 1e-6)^(-power/2)` over
  **all unordered node pairs**.
- Minus spread weight times the determinant of the population 2D covariance.

Explicit native analytic gradients replace detached autodiff slices, constant
repulsion, omitted spread and no-op clipping. The general autodiff library is not
repaired or certified by this local solver change. Initialize using a local RNG;
repeatability requires the same row order, configuration and graph. Project into
the configured disk before/after steps. Up to 32 backtracking trials accept a
finite, nonincreasing-energy projected step. A zero gradient or unchanged accepted
step stops; an unaccepted step or nonrepresentable energy/gradient returns an
error, never a warning followed by fake success. No global optimum is promised.

Learning rate, repulsion power and radius must be finite positive. All weights
must be finite nonnegative; zero iterations explicitly requests clipped seeded
initialization. Radius projection uses scaled norms. Zero-edge graphs still run
repulsion/spread when requested; one node has no pair/cohesion force.

## Verification and limits

Native finite differences independently check cohesion, repulsion, covariance
spread and their combined gradients. Fixtures check actual topology beyond 100
edges, edge removal, sparse/reordered IDs, malformed identity, empty/singletons,
configuration errors, local RNG repeatability, owned results and disk bounds.
See `tests/flat_embedding.rs` and rebuilt Python `tests/test_flat_embedding.py`.

One objective/gradient evaluation costs O(E + N²) time and O(E + N) working memory;
backtracking can repeat it 32 times per iteration. Removing fake/capped forces
changes both results and cost. No speedup, large-graph capacity, convergence,
threading/GIL-release or cancellation guarantee is added. Results are owned
matrices; no graph properties/history change. The existing realtime visualization
caller's fallback policy remains a separate caller audit.

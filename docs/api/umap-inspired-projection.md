# Native UMAP-inspired neighbor layout

`viz::projection::algorithms::UMAPProjection` preserves the historical constructor,
trait/import path, and `name() == "umap"`. It implements a native weighted-neighbor
layout with sampled repulsion. **It is not full UMAP**: it does not implement fuzzy
simplicial-set union, smooth-kNN membership calibration, the fitted UMAP attraction
curve, or UMAP cross-entropy optimization. No equivalence or neighborhood-quality
threshold is claimed.

`with_seed(seed)` seeds local initialization and negative sampling; the projection
factory now propagates `ProjectionConfig.seed`. `project_matrix(&input)` returns
owned 2D rows with copied input labels and `umap_x`/`umap_y` columns. The legacy
position-vector trait uses the same computation in input order. Its graph argument
does not provide topology; like PCA/t-SNE this projects feature rows. Labels are
opaque metadata and must have a matching row count.

## Neighborhood and objective

Require finite, nonempty input with at least two features. Scale all input
coordinates by a common maximum absolute value before computing squared Euclidean
distances; finite precision still limits mixed-scale/nearly coincident values.
Find each row's exact `min(n_neighbors, N-1)` nearest other rows. Distance ties use
original row index. Sort checked finite distances without a partial-comparison
panic. For selected neighbor distances `d`, use:

- `rho`: nearest selected distance.
- `range`: farthest selected distance minus `rho`.
- Similarity `w = exp(-(d-rho)/range)`, or one when the range is zero.

This locally scaled weight is used by the objective; the former code discarded
the input distances. Directed neighbor selections contribute separately, so a
mutually selected pair contributes two springs. Equal-distance cutoffs retain
row-index tie behavior rather than implying permutation invariance among ties.

For each epoch, sample exactly `floor(N * negative_sample_rate)` ordered nonself
pairs uniformly, with replacement. Duplicate/reversed samples retain multiplicity;
samples may also be positive neighbors. With that sample fixed, minimize:

- `sum(0.5 * w * max(projected_distance - min_dist, 0)^2)` over neighbor selections.
- `sum(0.001 / sqrt(projected_squared_distance + 1e-6))` over negative samples.

Analytic gradients update **both endpoints** simultaneously. Each spring/sample
has equal and opposite forces, preserving the centroid up to rounding. Start each
epoch at learning rate `0.01 * (1 - epoch/n_epochs)`; up to 64 halvings accept a
finite step whose current sampled objective does not increase. Failure errors.
The objective changes with each sample, so no monotonic loss across epochs or
convergence guarantee is claimed. Coincident positions have zero smooth repulsive
gradient; no jitter is added during optimization. `min_dist` is an attraction
threshold, not a guaranteed minimum separation of output rows.

## Controls, behavior, and limits

Require positive `n_neighbors`, finite nonnegative `min_dist`, and finite
nonnegative `negative_sample_rate`. A requested count too large to represent or
allocate errors explicitly; float-to-integer saturation is not used. Zero epochs
returns finite seeded random coordinates after input/control validation. A singleton
returns the origin; empty input rejects. Oversized positive neighbor counts clamp
to `N-1`, preserving the previous behavior. Constant feature rows are valid and
use the documented tie rule. Seeded calls repeat without changing global RNG state.
Inputs/graph/history are unchanged.

Feature distances require `O(N²*D)` work; the implementation retains scaled
features and computes one distance row at a time, without an `N²` distance matrix. Exact neighbor sorts
require `O(N² log N)` work and `O(N*k)` stored springs. An epoch samples `M` pairs
and evaluates `O(N*k + M)` energy/gradient work per candidate, with up to 64
candidates. Working storage is `O(N*D + N*k + N + M)`. These are complexity descriptions,
not benchmark results. No approximate neighbor index, Python entry point, GIL
release, cancellation, streaming, global optimum, or production-scale guarantee is
added. External coordinate interpolation remains supported; incremental fitting
does not.

## Evidence

Private tests compare exact neighbor order/weights with known distances and verify
an independent hinge-plus-repulsion objective by central finite differences,
including inactive hinges, repeated/reversed terms, coincident points, overflow,
and balanced forces. `tests/umap_projection.rs` independently checks a full first
update from known nearest neighbors, centroid preservation, min-distance and
fractional sampling controls, metadata/ownership, extreme uniform feature scaling,
seeded factory/trait agreement, global RNG isolation, invalid domains, singleton,
constant rows and neighbor clamping.

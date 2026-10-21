# Dense t-SNE projection

`viz::projection::algorithms::TSNEProjection` keeps its existing constructor and
projection trait. `with_seed(seed)` uses local initialization; the projection
factory now honors `ProjectionConfig.seed`. Native `project_matrix(&input)`
returns owned coordinates with input row labels and `tsne_x`/`tsne_y` columns.
The graph argument of the position-vector trait does not supply topology;
projection uses input feature rows in their existing order, like PCA.

## Affinities and objective

Compute exact squared Euclidean feature distances, scaling all feature values by
one common maximum absolute value before squaring to avoid uniform extreme-scale
overflow/underflow. This common scaling does not change Euclidean neighbor order
or the intended entropy-calibrated affinities. Finite-precision limits still
apply to features with very different scales or nearly indistinguishable values.

For every row, fit the Gaussian precision by bounded binary search so conditional
entropy is `log(perplexity)` within `1e-8`. Shift distances by the nearest distance
and scale each row's distance range before exponentiation. Symmetrize conditionals
as `P_ij = (p(j|i) + p(i|j)) / (2*N)`. Diagonals are zero and total joint mass is
one. Uniform and nearest-tie entropy limits have explicit solutions.

For projected positions, use `w_ij = 1/(1 + squared_distance)` and normalize over
all ordered nonself pairs: `Q_ij = w_ij / sum(w)`. The native gradient minimizes
`-alpha * sum(P * log(w)) + log(sum(w))`; with `alpha=1`, this is KL divergence
up to the constant `sum(P*log(P))`. During the first 250 requested iterations,
`alpha=early_exaggeration`; subsequent iterations use one. Exaggeration multiplies
P in the objective/gradient, rather than the learning rate. Each phase can stop
early on zero gradients or updates too small to represent; the next requested
phase still runs. Up to 64 learning-rate halvings accept a finite nonincreasing
step within the current phase. A failed line search errors explicitly.

## Domains and limits

Require at least two rows and two features, all finite input values, matching
optional row-label count, finite perplexity in `[1,N-1]`, finite positive learning
rate, and finite exaggeration at least one. A row with `m` exactly tied nearest
neighbors cannot achieve perplexity below `m`; reject that request rather than
claiming the target was reached. All-identical input requires perplexity `N-1`.
Zero iterations returns seeded initialization after full affinity/control
validation. Labels are opaque copied metadata, not validated graph IDs.

This is an exact dense affinity/gradient implementation with bounded local
optimization, not a promise of identical results to another package. It omits
momentum, adaptive gains, Barnes-Hut/FFT approximation, and automatic rate choice.
There is no global optimum, quality threshold, or convergence guarantee. Dense
pair storage is `O(N²)`; distance work is `O(N²*D)`, calibration up to `O(200*N²)`,
and an optimizer iteration up to 64 candidate evaluations of `O(N²)` each.
No Python entry point, GIL-release, cancellation, streaming, or scale guarantee is
added. Interpolation support means the existing external position interpolator
can consume these coordinates, not incremental t-SNE fitting.

## Evidence

Native unit tests cover a known analytic conditional distribution, exact symmetric
normalization, entropy tie limits, scale invariance, an independent normalized
Student-t objective and finite differences with and without exaggeration, actual
KL reduction, and phase transition. `tests/tsne_projection.rs` checks ownership,
row metadata, seeded factory/trait agreement, global RNG isolation, uniform
extreme scaling, two-row inputs, invalid controls/shapes/nonfinite values and
unattainable tie cases. An older five-row smoke fixture has four equidistant
neighbors in one row: it now explicitly rejects perplexity three and exercises
perplexity four, instead of silently ignoring the requested entropy.

# Real matrix decompositions

The Rust `GraphMatrix` and Python `GraphMatrix` APIs now use in-house bounded Jacobi kernels for real SVD and symmetric eigendecomposition. Both compute with f64 intermediates and return matrix factors in the input's floating dtype. Integer inputs reject explicitly in Rust; Python GraphMatrix currently uses f64.

## Singular value decomposition

`svd()` returns `(U, singular_values, V_transpose)` with full factor shapes:

- An m by n input produces m by m U, min(m,n) nonnegative descending singular values, and n by n V-transpose.
- Reconstruct A with the first min(m,n) columns of U and corresponding rows of V-transpose. Singular values form the diagonal of the rectangular Sigma matrix.
- Rank-deficient and zero matrices receive completed orthonormal bases, rather than zero-filled vector columns. Empty rectangular inputs preserve full factor dimensions and have an empty singular-value list.

The one-sided Jacobi kernel rotates input columns directly; it does not form A-transpose times A. Wide matrices use the transposed tall problem with the factors swapped back. Inputs are scaled by their largest absolute entry to protect uniformly large/small magnitudes. Values below `epsilon_f64 * max(m,n) * ||scaled A||_F` are treated as numerical zero. A bounded 100-sweep budget reports nonconvergence explicitly.

Full factors require O(m squared + n squared) output memory. This is not a reduced-factor or optimized sparse SVD API. A very wide/tall matrix can have inexpensive input but expensive full outputs. Checked element/byte arithmetic rejects overflowing output dimensions; ordinary allocation failure is not a new recoverable-memory guarantee.

## Symmetric eigendecomposition

`eigenvalue_decomposition()` accepts a real square symmetric matrix. It returns descending algebraic eigenvalues and an orthonormal matrix whose columns are the matching eigenvectors, satisfying `A V = V Lambda`. Negative, zero and repeated eigenvalues are supported. Repeated eigenspaces and null spaces have no unique mathematical basis, so compare residuals/subspaces rather than exact vector signs against another library.

Symmetry and convergence use scaled tolerances of `8 * epsilon_f64 * max(n,1)`. Input pairs within the symmetry tolerance are averaged. Eigenvectors use a deterministic sign convention; a 100-sweep budget bounds iteration. General nonsymmetric and complex eigendecomposition remain unfinished and now raise explicitly instead of returning guessed eigenpairs.

Both methods reject NaN/infinite inputs and nonrepresentable finite-range results. Scaling and numerical-rank tolerances do not promise relative accuracy for arbitrary mixed extreme magnitudes or values below numerical resolution. f32 factors retain f32 output precision; wider precision/output guarantees remain item 29 and appendix A13 work. Decomposition gradients are not established by this repair.

## Example

```python
from groggy import GraphMatrix

# The current nested-list factory takes columns.
a = GraphMatrix.from_data([[2., 1.], [1., 2.]])
values, vectors = a.eigenvalue_decomposition()
assert abs(values[0] - 3.) < 1e-12
assert abs(values[1] - 1.) < 1e-12
u, singular, vt = a.svd()
assert len(singular) == 2
```

PCA's covariance path now receives complete orthogonal eigenvectors. Spectral embeddings use ascending node-ID row order and `node_<id>` row labels; tests verify a path-graph Laplacian and a rotated rank-two PCA plane. Other spectral choices, directed variants, sparse kernels and broader PCA edge cases remain under their own audits.

Verification: Rust tests check reconstruction, orthogonality, scaling, rank deficiency, empty shapes, f32/f64 factors, invalid domains, and an exhausted iteration budget. Python tests compare random, repeated, rank-deficient and ill-conditioned fixtures with NumPy SVD/eigvalsh through the built extension. `benches/matrix_decomposition_profile.py` records timings and residuals; debug measurements do not establish release throughput.

Method reference: [LAPACK DGESVJ documentation](https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj_ga7aec05d2a1523bbeee77ece21b12187c.html) describes the one-sided Jacobi SVD family. Groggy's implementation and stopping/rank rules are defined above; no LAPACK runtime backend was added.

Initial debug timings and residuals are available in the [decomposition baseline](../../notes/benchmarks/matrix-decompositions-2026-10-21.md).

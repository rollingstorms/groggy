# Sparse numerical materialization

CSR arithmetic, matrix products, selection/transpose, graph operators,
LU/Cholesky, determinant and coefficient-matrix solves have sparse-native routes.
They do not pass the coefficient matrix through a dense spectral kernel.
Numerical fill during sparse factorization can still grow substantially.

SVD (full/economy), symmetric/general eigen, QR, numerical rank/condition,
least-squares and explicit dense factor derivatives currently use dense work.
They may implicitly materialize a CSR input with at most **1,048,576 logical
cells** (`GraphMatrix::<T>::IMPLICIT_DENSE_ELEMENT_LIMIT`). Larger CSR inputs
return an error requesting explicit `to_dense()` before those consumers run.
This retains existing small-input behavior and prevents a sparse graph from
silently triggering a large dense input allocation.

The guard applies to every sparse input passing through the common dense
numerical conversion, including large sparse right-hand sides. The explicit
conversion is the opt-in; no hidden size-dependent approximate algorithm is
substituted. A dense input's shape and allocation arithmetic remain checked.
Input extraction uses validated logical storage, not per-cell optional reads.

This is a limit on implicit input materialization, **not** a total workspace or
RAM guarantee. Full SVD/eigen can need square outputs; thin SVD uses economy
factors, and factor derivatives may need additional dense scratch. Use a sparse
native routine when it matches the task. Large sparse iterative eigensolvers,
ordering/fill bounds and universal sparse spectral throughput remain future work.

Native/Python tests check early rejection on large CSR input, retained small
spectral results, and unaffected sparse-native solve/Cholesky routes.

## Expansion beyond spectral kernels

The same public `matrix::IMPLICIT_DENSE_ELEMENT_LIMIT` (also available through
GraphMatrix's existing associated constant) covers nonzero implicit-zero images
in CSR maps: exp, log, nonzero scalar shifts and dense comparison masks. Small
inputs retain IEEE values and canonical CSR results; above the limit callers must
explicitly convert to dense first. Sparse zero-preserving maps stay sparse and do
not hit this limit merely because their logical shape is large.

Nonfinite CSR scale and product fallback paths also check the logical input/grid
work before scanning it. Mixed nonfinite products cannot silently flatten a large
CSR operand. Explicit dense conversion opts into that work. Sparse nonzero fill
uses the same map policy. This does not bound arbitrary finite sparse-product or
factorization fill, inherently dense outputs, or total workspace memory.

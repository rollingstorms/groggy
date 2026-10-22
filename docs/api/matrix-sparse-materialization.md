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

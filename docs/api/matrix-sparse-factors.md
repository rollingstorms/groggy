# Sparse LU, Cholesky and direct solve

Canonical CSR square floating inputs now use in-house sparse numerical kernels
for lu_decomposition(), cholesky_decomposition(), solve(), slogdet(), and
larger-order determinant(). Existing APIs and positional coordinate conventions
remain. Sparse P/L/U or Cholesky factors stay canonical CSR; solve returns an
owned dense n×rhs solution. Inverse still explicitly solves against identity
and returns its existing dense result.

LU uses unscaled partial pivoting, sparse row maps and column adjacency to
select/update affected rows. Row swaps include already computed L columns.
Zero pivots may remain in valid singular factors; direct solve rejects them.
This retains dense LU's unscaled input convention, including independently tiny
and huge rows. It is not an exact-rank or arbitrary-intermediate-range promise.

Direct solve equilibrates each coefficient and RHS row, rejects a zero row or
lost normalization terms, then uses sparse LU and accurate triangular residual
sums. Factorization may create fill-in, but does not allocate a dense coefficient
matrix. RHS/solution buffers cost O(n*rhs). Sparse sign-log determinant uses
permutation parity and pivot log sums without materializing dense P/U.

Cholesky checks/averages symmetry with the same normalized tolerance as the
dense forward contract. Right-looking sparse Schur updates create numerical
fill using accurate two-product residuals. Require positive finite diagonal
pivots at working precision, rescale by the square root of coefficient scale,
and reject invalid/nonfinite/range/dtype domains. No fixed absolute pivot floor
is introduced. Independent positive rows and tiny diagonal values stay explicit.

Natural ordering and partial row pivoting are implemented; fill-reducing symbolic
ordering, rank-revealing variants, blocked/parallel kernels and throughput
certification remain performance work. Storage follows original nonzeros plus
actual fill and row/column structures. Fill can become dense for a sparse input.
Finite f32 narrowing and ordinary floating rounding/underflow remain possible.
Near-singular inputs can fail at working precision.

Tracked Cholesky checks bound sparse inputs by canonical structure/values without
making dense input copies. Generic cotangents and input gradients can themselves
be dense; current backward/explicit derivative buffers retain their documented
storage costs. General sparse QR/SVD/eigen, condition/rank and refinement paths
still use their existing dense reference domain and allocation policies; further
materialization controls remain in the numerical completion pass.

Native tests exercise 10,000×10,000 diagonal factors/solves with sparse factors,
scaling, extreme independent rows, singular/invalid and dtype cases. Python
compares seeded sparse LU reconstruction, NumPy solves/Cholesky, empty and
tracked-gradient behavior. External libraries are test references only.

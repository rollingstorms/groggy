# Pivoted real LU and determinant range

Item 29.29 replaces fixed-threshold LU and factorial recursive cofactor determinants
with a shared native partial-pivot factorization PA=LU for finite square f32/f64 input.
Choose the largest remaining pivot magnitude, swap working/permutation and previously
computed L rows, and eliminate with fused updates. There is no absolute epsilon cutoff.
Zero pivot columns remain zero, so singular matrices can still return valid P/L/U
factors; determinant and slogdet identify zero diagonal pivots as singular.

L is unit lower triangular, U upper triangular, and P a permutation. All factors own
their data. P and L retain permuted source row labels; P columns retain original row
identity, L columns and U rows share lu_i basis names, and U columns retain source
column names. Labels do not reorder the problem automatically. Malformed row metadata,
nonfinite input, non-square shape, nonfloating dtype and unrepresentable factors error.
f32 factors use f64 work followed by checked finite narrowing.

Determinant order zero is one. Orders one/two use the original finite values, with
accurately accumulated two-product subtraction at order two. Larger orders use LU
and exponentiate sign/log magnitude; final underflow/overflow produces signed zero/
infinity. This is a numerical f64 determinant estimate, not an exact integer determinant.
The new native/Python slogdet returns (sign, log(abs(det))) and avoids final product
range loss. Singular output is (0,-infinity); empty output is (1,0). Log magnitude is
a compensated sum of pivot logs. LU itself can still reject finite inputs if its
intermediate factors exceed range; slogdet does not eliminate that limitation.

Bounded work is O(n^3), scratch O(n^2), replacing factorial cofactor recursion. There
are no guessed values or ignored writes. Singular/ill-conditioned problems can have
large determinant sensitivity; no exact-rank, forward-error, gradient, complex/sparse
factorization or global conditioning guarantee is claimed.

Evidence: tests/matrix_lu_determinant.rs and tests/test_matrix_lu_determinant.py cover
PA=LU, permutations/triangularity, singular/zero/empty matrices, tiny/huge common scale,
range-separated diagonal determinants, slogdet versus NumPy, exact two-product
cancellation and rejected domains. benches/matrix_determinant_profile.py and paired
before/after JSON record small matching dev timings and whole-process RSS, not release
throughput or per-operation allocations.

Matching dev determinant medians changed from 0.0458 to 0.00442 ms at order four and 1.396 to 0.00808 ms at order six. These small fixtures demonstrate the recursion removal cost difference only; they are not release-scale throughput guarantees.

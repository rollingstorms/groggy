# Economy Householder QR

Item 29.27 replaces modified Gram-Schmidt, unnormalised near-zero vectors and fallback
values with a bounded in-house Householder factorization for finite f32/f64 input.
For an m by n matrix, k=min(m,n), Q is m by k and R is k by n, with A=QR and Q^T Q=I.
Tall/square output shapes remain; **wide matrices now use valid economy shapes**
instead of the old m-by-n Q with more columns than its dimension can make orthonormal.

Normalize input by its finite maximum magnitude, compute norm-stable Householder
reflectors, apply them with accurate dot products and fused updates, then build compact
Q by applying stored reflectors in reverse order. Explicitly zero R below its diagonal
and rescale its upper triangle. Zero/rank-deficient matrices retain orthonormal Q through
identity-completion reflectors; QR does not infer rank from an absolute epsilon.

All-zero and empty rectangular matrices are supported, including 0xN and Mx0 with
k=0. Inputs stay unchanged. Q retains source row labels and named qr_i basis columns;
R retains matching basis row labels and source column names. No graph binding or gradient
tape is inherited. Native integer input, nonfinite values, lost normalization range,
malformed row-label count and nonfinite/unrepresentable factors error explicitly.
f32 computation uses f64 work and checked finite narrowing; ordinary rounding/underflow
remain. This is real QR, not complex or pivoted/rank-revealing QR.

Work is bounded O(m*n*k + m*k*k); scratch/factors are O(m*n + m*k + k*n). There is no
full m-by-m Q allocation for tall matrices. No throughput or global conditioning
guarantee is claimed. Least squares, QR pivoting, gradients and generic lower-level QR
remain separate work.

Evidence: tests/matrix_qr.rs and tests/test_matrix_qr.py cover reconstruction,
orthogonality, triangularity and shapes for square/tall/wide/empty/zero/rank-deficient
inputs, scaling through 500 decimal orders, float32, identity and rejected domains.
Python NumPy is used only in tests to evaluate invariants, not as a backend.

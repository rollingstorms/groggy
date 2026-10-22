# Checked real Cholesky

Item 29.28 replaces fallback cells/ignored writes, fixed absolute symmetry thresholds
and unchecked factor conversion with a shared bounded native real Cholesky kernel.
Finite square f32/f64 input is required; native integer/nonfinite/rectangular input
rejects. Zero order returns an owned 0x0 factor.

Normalize by the finite maximum input magnitude. Reject input values completely lost
by normalization, and check symmetry within 8*epsilon(f64)*max(n,1) of the normalized
matrix scale. Accepted symmetric pairs are averaged before factoring. This is a
norm-relative working-precision symmetry policy, not exact bitwise symmetry or a
promise to repair materially asymmetric data.

Each lower-triangular entry uses an accurately accumulated residual A_ij minus the
existing factor products. A diagonal residual must be strictly positive; otherwise
return a positive-definiteness error at working precision. There is no absolute
pivot epsilon. Rescale L by sqrt(input scale), reject nonfinite intermediates and
unrepresentable output dtype, and retain ordinary f32 rounding/underflow.

L has source row labels and chol_i basis column names; labels do not reorder the
numeric problem. Source values stay unchanged, and no graph binding/gradient tape is
inherited. The owned factor satisfies A approximately L*L^T within floating errors
and the documented symmetry averaging. Near-singular positive-definite inputs can
fail at working precision; condition estimation, pivoted/modified/complex Cholesky,
sparse factors, gradients and throughput remain separate work.

The bounded dense work is O(n^3) with O(n^2) scratch. Evidence:
 tests/matrix_cholesky.rs and tests/test_matrix_cholesky.py cover reference factors,
reconstruction/triangularity across 500–600 decimal orders, f32, empty order, and
asymmetric/indefinite/singular/nonfinite/integer/shape rejection. NumPy is a test
reference only; the implementation uses Groggy's native kernels.

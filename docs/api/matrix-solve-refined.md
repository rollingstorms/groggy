# Solve residuals and refinement

`A.solve_refined(B, max_refinements=2)` returns `(solution, backward_error,
accepted_refinements)` in Python. Rust returns an owned `SolveReport<T>` with
those fields. `solve` and `inverse` retain their existing behavior.

The square, finite floating coefficient/RHS restrictions and direct solve's
explicit singularity/range errors apply. RHS may have multiple columns.
`max_refinements=0` reports on the direct solution; values above 16 are errors.
Zero-order systems return the original empty output shape and zero error.

The estimate is `||B-A*X||inf / (||A||inf*||X||inf + ||B||inf)`, where the
matrix infinity norm is the maximum absolute row sum. Residual products are
accumulated exactly before each residual cell is rounded to f64. Logarithmic
norm arithmetic avoids overflowing the denominator. A zero residual reports
zero; rounded subnormal residuals/ratios can underflow, and normalized norm
summands can lose tiny terms. This is a floating estimate, not a certified
bound or a forward-error guarantee for ill-conditioned systems.

Each attempt solves for a correction using the existing direct kernel, rounds
the candidate to the original element dtype, and recomputes its residual.
Only a strict decrease in the reported error is accepted; otherwise iteration
stops with the best previous result. A working-range failure is an explicit
error. There is no silent partial-success recovery from an invalid attempt.
No higher precision factorization, automatic condition threshold, or convergence
guarantee is implied. f32 corrections often stall at f32 rounding precision.

The solution owns its values and retains coefficient-column/RHS-column identity.
Inputs remain unchanged. Dense residual storage is O(n*rhs); each correction
currently refactors coefficients in O(n^3+n^2*rhs) rather than caching factors.
Release throughput, sparse refinement, and decomposition gradients remain open.

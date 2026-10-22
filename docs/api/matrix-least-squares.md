# Dense real least squares

Python: `A.least_squares(B, rcond=None)`; Rust:
`A.least_squares(&B, relative: Option<f64>)`.

Return an owned n×rhs minimum-norm solution for m×n coefficients and m×rhs
observations using the existing bounded, in-house real SVD. Both inputs must
have finite floating elements. Multiple RHS columns, overdetermined,
underdetermined, rank-deficient, zero and empty systems are supported.
Rows identify coefficient columns and columns identify RHS columns;
matching is positional. The sources remain unchanged.

Retain singular components when sigma/largest_sigma is strictly greater than
rcond; zero singular values are always discarded. The default is input-dtype
epsilon times max(m,n,1), shared with numerical rank. Explicit rcond must be
finite and nonnegative. A cutoff of zero retains computed nonzero components,
including floating residual directions of numerically rank-deficient input;
it does not recover exact rank. Values of one or more discard every component.
The result is minimum norm for the retained numerical model. Truncation can
intentionally increase residuals relative to the full mathematical system.

Normalize A and B by the largest absolute coefficient before decomposition.
Projection and reconstruction use exact finite dot accumulation followed by
f64 rounding; f32 output narrows with finite range checks. Reject nonfinite
inputs, incompatible rows, invalid cutoffs, lost normalization terms,
unrepresentable intermediate/output values, or failed decomposition/basis
convergence explicitly. Ordinary floating rounding and underflow in subsequent
arithmetic remain possible. There is no fallback external backend or guessed
solution after a failure.

Dense full-factor SVD requires O(m²+n²+m*n) storage in addition to RHS,
projection and output buffers; this is not an optimized thin/sparse solver.
Zero/empty/no-RHS cases avoid full-factor work. Allocation checks cover output
and projection dimensions; existing SVD allocation/convergence limitations
remain. This slice adds no gradients or production throughput guarantee.

Tests compare against NumPy least squares, check normal equations and null-space
orthogonality, and cover rank deficiency, multiple RHS, extreme common scaling,
cutoffs, dtype, coordinate identity, empty inputs and explicit errors.

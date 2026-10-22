# Real spectral derivatives

`eigen_jvp(H)` returns `(dvalues, dvectors)` for the existing real symmetric
eigen API. `eigen_vjp(values, vectors)` returns a symmetric Frobenius input
gradient for supplied eigenvalue and eigenvector cotangents. Input/tangent
shapes must match; tangents must be symmetric. Returned columns follow the
forward descending eigenvalue order and locally fixed vector signs.

`svd_jvp(H)` returns thin `(dU, dS, dVt)`, with k=min(m,n), shapes m×k, k,
and k×n. `svd_vjp(U, S, Vt)` takes matching thin cotangents and returns an m×n
input gradient. Full forward SVD remains unchanged; derivatives of arbitrary
full null-space completions are not represented by the thin derivative API.

These native/Python methods require finite real floating inputs. Eigenvalues
must be distinct, separated by more than 64*epsilon(f64)*max_dimension times
the largest absolute eigenvalue. Thin SVD additionally requires every singular
value above that relative resolution threshold. Repeated/unresolved values or
rank loss return explicit errors. These domain rules avoid assigning a unique
derivative to a nonunique basis; they are numerical policies, not exact-rank
proofs. Zero-axis shapes are supported. No gradient tape is inherited by these
explicit calls. Automatic factor-output integration is a separate work item.

Implement normalized in-house eigendecomposition/SVD, gap-based eigenvector
rotations, rectangular orthogonal-complement terms, and analytic adjoints.
Use accurate dot accumulations and checked exponent-based rescaling. Reject
lost normalization terms, invalid shapes/dtypes/symmetry, convergence and finite
working/output range failures. Ordinary rounding, underflow and f32 narrowing
remain possible, and small spectral gaps can amplify forward errors. Results
own their data and input gradients retain positional source labels.

Kernels use dense O(m²+n²+mn) scratch with full-factor forward decomposition.
No sparse, blocked, complex-input or throughput guarantee is introduced.
Tests use independent NumPy decompositions with locally aligned signs, central
differences and JVP/VJP adjoint identities across square/tall/wide and scaled
fixtures. NumPy is a test reference only.

Mathematical context: [Auto-Differentiating Linear Algebra](https://arxiv.org/abs/1710.08717).

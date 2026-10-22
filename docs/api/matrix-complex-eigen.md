# General real-input eigendecomposition

`A.complex_eigenvalue_decomposition()` returns a dedicated `ComplexEigenResult`:

- `eigenvalues_real`, `eigenvalues_imag`: f64 lists of length n.
- `eigenvectors_real`, `eigenvectors_imag`: owned n×n matrices in the input dtype.

Rust exposes the same fields in `ComplexEigenResult<T>`. Reconstruct each
complex column v and eigenvalue lambda from their real/imaginary components:
`A*v = lambda*v`. Columns have unit Euclidean norm before output-dtype
rounding; phase makes a largest component real and positive. Sort by descending
real then imaginary value, with original column position breaking ties.
Floating perturbations can change the order of nearly equal components.
Eigenvectors of nonsymmetric input need not be orthogonal.

The existing real symmetric `eigenvalue_decomposition()` API retains its
signature and supported domain. This new method accepts finite square real
floating matrices, including zero order. It adds no general complex matrix
storage or complex-valued input. Output rows identify source columns; output
columns identify the computed eigenbasis. Sources are unchanged.

## Native solver and limits

Use in-house complex Householder QR and full unitary similarities to obtain
an upper triangular form. Shifts use the trailing two-by-two roots, with
independent scaling for tiny blocks and an exceptional shift every 32 steps.
Deflation compares couplings against local diagonal scales. Bound iterations
to 256 per deflated eigenvalue, then return a convergence error.

Back substitution and transformation produce right eigenvectors. Accurate
complex dot accumulations verify residuals against the original normalized
input. A two-pass orthogonalization check rejects a numerically dependent
basis; it does not replace the returned eigenvectors with orthogonal ones.
Defective or unresolved repeated/ill-conditioned bases return explicit errors.
No Jordan chains or partial-result fallback are provided.

The internal tolerance is 64*n*epsilon(f64). Normalized pair residuals must be
no larger than eight times that tolerance times (Frobenius norm of A plus
absolute eigenvalue), before output narrowing. These checks provide numerical
backward diagnostics, not certified eigenvalue/forward-error bounds. Small
perturbations can change spectra of highly nonnormal input substantially.
Normalization rejects lost input terms, and finite output conversions reject
range overflow; ordinary rounding/underflow and f32 narrowing remain possible.
Conjugate pairs can have small independent numerical differences.

This is a dense reference implementation: full similarities cost O(n^3) per
QR step and multiple O(n²) buffers. It does not yet use a Hessenberg reduction,
implicit QR bulge chasing, balancing, sparse factors, gradients, or released-GIL
execution. Production throughput is still item 29.26.

## Verification

Native and Python tests cover rotations, real triangular spectra, zero/repeated
spectra, cyclic input, defective errors, mixed-scale blocks, dtype/range/shape
and bounded-budget errors. Seeded nonsymmetric inputs through order eight are
compared to NumPy eigenvalues and independently checked against A*v=lambda*v.

Algorithm context: [LAPACK ZGEEV](https://www.netlib.org/lapack/explore-html/d4/d68/group__geev_ga8656466cd9da86a0b3b6966e49a7ce40.html)
and [triangular eigenvector reconstruction](https://www.netlib.org/lapack/explore-html/da/d6a/group__trevc_ga2d7a7abf901cb7ff1a8ade8d6f11ac80.html).
Groggy implements its own bounded reference; it does not call LAPACK.

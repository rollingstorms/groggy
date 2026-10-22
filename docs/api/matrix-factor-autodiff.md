# Factor output automatic differentiation

Native/Python `backward_with_gradient(seed)` pulls back a caller-supplied,
finite, shape-matching cotangent. Each call replaces prior tape gradients;
`backward()` retains its all-ones seed. Gradient matrices retain input labels.

For an input bound with `requires_grad(true)` / Python `requires_grad_(True)`:

- `qr_decomposition()` tracks Q and R independently.
- `lu_decomposition()` tracks L and U; P is a discrete, untracked permutation.
- `eigenvalue_decomposition()` tracks the vector matrix; its value list is owned.
- `svd()` tracks square U/Vt factors; its singular-value list is owned. Tracked
  rectangular full SVD errors because arbitrary null-basis derivatives are not
  supported. Explicitly detach when full owned null bases are wanted.
- `eigen_tracked()` returns n×1 eigenvalue and n×n eigenvector matrices.
- `svd_tracked()` returns economy m×k U, k×1 singular values and k×n Vt matrices,
  with k=min(m,n). All three support independent tape seeds.
- Cholesky retains its existing tracked output and now accepts explicit seeds.

All outputs save the forward input. Mutation before producing a tracked output
requires rebinding; mutation after forward does not change its saved derivative
input. Repeated spectra, rank loss, unresolved QR sign charts and LU pivot ties
retain the explicit kernels' domain errors during backward. General complex
eigen gradients are unsupported and require explicit detachment for forward
computation. There is no derivative through a permutation or arbitrary null
basis. Value-list APIs remain unchanged; matrix-valued APIs expose the tape.

For a loss involving several outputs, use the existing explicit joint
`qr_vjp`, `lu_vjp`, `eigen_vjp` or `svd_vjp` with all cotangents. Independent
`backward_with_gradient` calls replace gradients rather than summing across
calls. Existing GraphMatrix arithmetic/tape composition limitations are not
removed by these factor routes; this is not a claim of universal autodiff.

Native `tests/matrix_factor_autodiff.rs` checks saved inputs, every output
route, malformed seeds, empty shapes and unsupported domains. Python
`tests/test_matrix_factor_autodiff.py` includes independent weighted eigenvalue
and singular-value finite differences, rectangular factors, explicit-kernel
agreement, repeat calls and mutation behavior.

The SVD routes now use [actual economy storage](matrix-economy-svd.md); `svd_thin()` also exposes owned value-list economy results.

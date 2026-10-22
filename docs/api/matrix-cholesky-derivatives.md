# Cholesky derivatives

`A.cholesky_jvp(H)` computes the directional derivative of the lower Cholesky
factor for a symmetric input tangent H. `A.cholesky_vjp(G)` pulls a factor
cotangent G back to a symmetric gradient on A. Rust and Python expose the same
methods. Inputs must be finite square floating matrices of matching shape,
and A must satisfy the forward Cholesky positive-definiteness/symmetry contract.
Zero order and native f32 are supported. Inputs remain unchanged.

Use `dL = L Phi(L^-1 H L^-T)`, where Phi keeps the lower triangle and halves
the diagonal. The pullback symmetrizes `Phi(L^T G)` and applies two transposed
triangular solves. These are in-house kernels with accurate dot accumulations;
no explicit inverse or external numerical backend is used. See
[Murray's derivation](https://arxiv.org/abs/1602.07527) for mathematical context.

The VJP uses the full-matrix Frobenius convention:
`sum(G * JVP(H)) = sum(VJP(G) * H)` for symmetric H. Thus varying a diagonal
coefficient contributes one gradient entry, while varying both entries of an
off-diagonal symmetric pair contributes their sum (twice the returned entry).
This differs from storing all sensitivity in one independent triangular entry.
Upper-triangle G values have no effect, but all supplied values must be finite.
The JVP output is lower triangular and retains forward-factor identity; the VJP
retains input coordinate labels. Labels do not reorder values.

Normalize A and directions separately, then rescale with decomposed binary
exponents to avoid forming overflowing/underflowing scale ratios first.
Reject lost normalization terms, nonsymmetric tangents, failed forward factors,
unrepresentable working/output ranges and dtype conversion errors. Symmetric
tangent validation uses the forward factor's norm-relative tolerance and averages
accepted pairs. Floating rounding/underflow and f32 narrowing remain possible;
no near-singular accuracy bound is promised.

These are explicit derivative calls, with owned outputs and no inherited tape.
Tracked forward factors also support the existing backward tape as described
below. Other decomposition derivatives remain separate checklist work. Each call recomputes the forward factor, costs O(n^3), and uses
O(n²) scratch. Cached/blocked/sparse derivatives and production throughput remain
open. Native analytic/scale/domain/identity tests and Python independent central
differences, symmetric-coordinate checks and adjoint identities verify this slice.

## Automatic backward tape

`A.requires_grad_(True).cholesky_decomposition()` in Python (native
`A.requires_grad(true)`) produces a tracked factor. Calling factor.backward()
uses the same pullback to populate the source leaf gradient. The existing tape
seeds an output-shaped matrix of ones, equivalent to differentiating the sum of
the factor entries; explicit cholesky_vjp supports other output cotangents.
Chained supported operations propagate to their saved input snapshots.
Repeated backward clears previous gradients, following the existing tape policy.

Binding gradients saves values. If input values change between binding and the
tracked forward call, reject explicitly; disable and re-enable tracking to bind
the new values. Mutations after a successful forward do not change its saved
input state. This prevents a new Cholesky forward from differentiating a stale
input. The guard is scoped to tracked Cholesky; broader mutation/tape ownership,
branch merging and unsupported operators are separate work. The existing general
grad accessor's metadata/locking policy is unchanged. Low-level Operation::Cholesky
forward/backward delegates to the same checked native factor/pullback.

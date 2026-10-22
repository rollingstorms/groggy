# Numerical completion pass — 2026-10-21

User request: finish the remaining linear algebra and numerical items together,
with separate reviewable commits and synchronized checklist evidence.

## Work queue

- [x] Real spectral derivatives: symmetric eigen and thin SVD JVP/VJP, distinct-spectrum/rank policy, native/Python finite-difference and adjoint references.
- [x] QR/LU JVP/VJP with locally stable rank/pivot policies and independent reconstruction/gradient references.
- [ ] Supported automatic differentiation routes, saved inputs, multi-output factors and explicit unsupported derivative domains.
- [ ] Numeric array arithmetic, conversion/shape/domain contracts, exact checked integer operations and Python dispatch.
- [ ] Lower-level matrix arithmetic, logical layouts, reductions, backend consistency and remaining matrix APIs/consumers.
- [ ] Sparse-aware canonical arithmetic, products, transpose/selection, conversions and numerical solve/factor paths with explicit storage/materialization policies.
- [ ] Release benchmark corpus covering fractional/integer/dense/sparse/vector work, allocation/peak-memory receipts and measured improvements.
- [ ] Python GIL behavior and explicit owned-kernel boundaries where useful.
- [ ] Complete regression/lint/API/doc review and accurate closure of 29.3–29.6/29.24–29.26.

Completion requires implemented behavior and recorded evidence. Historical slice
notes retain their scope; parent boxes remain open until the work above is done.
Unrelated development benchmark edits present at the start of this pass are
preserved separately.

## Completed slices in this pass

- 29.37: symmetric eigen/thin-SVD explicit derivatives, independent gradient checks.
- 29.38: QR/fixed-pivot LU explicit derivatives, independent gradient checks.
- 29.39: shared checked lower arithmetic, logical layouts, raw backend ABI guards.
- 29.40: canonical sparse arithmetic/products/transforms/conversions. Sparse solve/factors and remaining statistics still open.

- 29.41: sparse LU/Cholesky/row-equilibrated solve/sign-log determinants; independent references and 10,000-square storage fixtures.

- 29.42: native typed vector arithmetic, casts, comparisons, unique, scalar reads and Python dispatch; remaining helper/boundary review continues.

- 29.43: checked entrywise norms, integer-preserving and sparse zero-group matrix statistics, logical-layout and conversion boundaries.

- 29.44: checked math/helper domains, sparse transforms/concatenation/split and owned materialization.

- 29.45: seeded factor-output automatic routes and matrix-valued eigen/singular values; general arithmetic tape composition remains open.

- 29.46: owned numerical transfer and GIL release for untracked heavy calls; copy overhead, reductions/backward and release performance remain explicit.

- 29.47: 20-case release corpus, heap allocation/peak receipts and measured bulk-product/norm/reduction improvements; broad size/SIMD gates remain open.

- 29.48: checked attribute/ID/fancy-index bridges and BaseArray numeric conversion; 357 Python regressions.

- 29.49: checked logarithmic matrix powers and corrected noncommuting explicit/automatic derivatives.

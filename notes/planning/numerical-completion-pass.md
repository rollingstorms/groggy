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

- 29.50: actual economy SVD, thin tracked/gradient/spectrum/least-squares consumers and 25-case release receipts.

- 29.51: shared typed maps and checked public math/scalar/comparison/tile/repeat routes.

- 29.52: checked graph-matrix degree/normalization/transition operators and identity-preserving graph bridge; 424 Python tests.

## Concrete remaining audit work

- Expanded bounded size/fill/reference-dispatch and strict core lint gates pass in 29.60. Remaining functional closeout is 29.61–29.63; production targets remain A15/29.26/32 work.

- 29.53: shared raw-bound slice normalization and checked extreme steps; 437 Python regressions.

- 29.54: checked graph/view adjacency construction, real CSR, parallel accumulation and scoped identity/weights; 439 Python regressions.

- 29.55: scoped feature export, checked explicit attribute conversion and logical/CSR fill; 440 Python regressions.

- 29.56: matrix arithmetic/shared tape composition and joint factor losses; 446 Python regressions.

- 29.57: explicit sparse/dense numerical materialization policy; 447 Python regressions.

- 29.58: checked matrix constructors and shared matrix/array conversion; 454 Python regressions.

- 29.59: faithful scalar tape replay, deterministic execution and direct-division gradients; 455 Python regressions.

## Current checkpoint and proposed consolidation

After 29.59, all native suites and 455 rebuilt Python numerical/array tests pass.
The existing 25-case release corpus was rebuilt and rerun idle; see
`notes/benchmarks/numerical-current-2026-10-22.md`. Larger size/fill/SIMD and final
strict gates remain open. The user's matrix/advanced_matrix consolidation request
and related candidate review are recorded separately in
`module-consolidation-review-2026-10-22.md`; module moves are not yet implemented.

## Scheduled next work

Module moves are tracked in checklist AR7.2–AR7.3 before final numerical parent
closure: matrix/advanced_matrix/adjacency first, then canonical arrays/shared
numeric primitives, using the current tests and release receipt as migration
baselines. Preserve old imports. Finish a finite numerical finding inventory,
size/fill/dispatch measurement and API/lint gates; do not introduce unrelated
optional features as automatic blockers. See
`roadmap-finish-priorities-2026-10-22.md` for subsequent core work and workbook gates.

### AR7.2 — Canonical matrix implementation completed

Move the advanced storage/backend/arithmetic/neural implementations and adjacency
helpers into storage::matrix; generic autodiff has a direct matrix module. Legacy
advanced_matrix, storage adjacency and neural autodiff imports are re-exports of
the same types. Preserve GraphMatrix/UnifiedMatrix/owned-transfer responsibilities
and Python APIs; migrate maintained callers. Full locked offline native suites,
legacy type/trait/nested-import compatibility fixtures, focused Clippy (excluding
existing sort/match categories) and 455 rebuilt Python regressions passed. Rebuilt
25-case release corpus and idle rerun agree with every prior workload checksum at
1e-13 tolerance. Receipt: benches/numerical_matrix_consolidation_2026-10-22.json.
This is an ownership migration, not a new speedup or broad numerical completion.

### AR7.3 — Canonical array storage and shared numeric foundation completed

Make legacy BaseArray a dtype/name compatibility adapter over the generic shared
column, with explicit copy-on-write bridges and common selection/ID/reduction
paths. Shared storage::numeric owns scalar dtypes, conversion/comparison/norm
policies, reductions and accurate/checked accumulation, independent of array or
matrix ownership. Old paths remain aliases; matrix error adapters preserve errors.
Repair dropped invalid selection positions, truncated masks, slice overflow and
ID narrowing while preserving valid input and legacy slice conventions. Full
locked offline native suites, compatibility fixtures, focused Clippy (excluding
existing sort/match categories) and 455 rebuilt Python regressions passed. All 25
optimized idle-run workload checksums match the pre-migration receipt at 1e-13;
raw receipt is benches/numerical_array_foundation_2026-10-22.json. Broader numerical
performance gates and remaining module/view tasks stay open.

### Item 29.60 — Expanded release baseline and strict core gates

Extend the corpus from 25 to 44 cases, retaining original checksums, finite result
receipts and allocation/time samples. Cover dense through 512-square, million-value
vectors/dispatch, banded sparse fill/mixed products and 100k-square sparse-native
operations. Full locked offline native suites, 455 rebuilt Python regressions and
strict all-target core Clippy pass after behavior-preserving sort/match/test lint
cleanup. Record dispatch as reference (not SIMD acceleration), non-universal heap
measurement and unresolved production budgets. The finite closeout finding list
names remaining attribute/table, activation/tracking and sparse expansion repairs
under 29.61–29.63; broad parents remain open. See the numerical expanded benchmark
note and numerical-closeout-findings-2026-10-22.md.

### Item 29.63 — Shared sparse expansion and nonfinite fallback policy

Use one public implicit-dense cell limit for spectral input conversion, nonzero
CSR map images and nonfinite CSR scale/product scans. Guard sparse operands before
mixed nonfinite products can flatten them; retain explicit dense opt-in, small
IEEE results and large sparse zero-preserving maps. Full locked offline native
suites, strict all-target core Clippy and 462 rebuilt Python regressions pass,
including 10k-square early-error and just-over-limit explicit opt-in fixtures.
Sparse kernel parent 29.25 closes within the documented native domain and measured
coverage; arbitrary fill or production-memory guarantees are not implied.

### Item 29.61a — Checked attribute pair statistics and quantiles

Canonical BaseArray delegates to shared numeric attribute kernels. Preserve exact
integer ordering/centering, mixed integer/f32 residuals, pairwise missing cells,
average Spearman ties and legacy Kendall tau-a. Reject mismatched lengths, negative
ddof, invalid empty quantile arguments, nonfinite input and f32 output overflow.
Native/Python rational and boundary references are recorded in
`attribute-statistics-contract-2026-10-22.md`. Remaining window/cumulative/table
paths keep 29.61 and its numerical parent open.

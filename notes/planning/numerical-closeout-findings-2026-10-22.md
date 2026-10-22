# Finite numerical closeout findings — 2026-10-22

Scope: finish the existing numerical surfaces after AR7.2/AR7.3, without treating
optional algorithms/accelerators as new prerequisites. Current core reference:
full native suites, 455 rebuilt Python regressions, 44 release workloads and clean
strict all-target core Clippy. This inventory does not mark every numerical
parent complete. It records specific remaining public paths instead of the vague
"other routines remain unaudited" wording.

## Covered core families

- Checked typed arrays/scalars/conversion/comparison, common reductions and
  NumArray percentiles/Pearson; canonical column storage and compatibility paths.
- Dense/logical-layout/CSR arithmetic, products, selection, transforms, graph
  operators, constructors, numerical transfer and checked backend boundaries.
- Real LU/QR/Cholesky, determinants, solve/refinement, economy/full SVD,
  minimum-norm least squares, symmetric/general real-input eigen results and
  rank/condition diagnostics within their documented domains.
- Real supported decomposition JVP/VJP and seeded arithmetic/factor composition;
  explicit spectral/rank/chart restrictions, saved values and owned outputs.
- Bounded implicit sparse spectral conversion and sparse-native finite solve/
  factor/arithmetic routes. Non-zero-image and nonfinite expansion is NC3 below.
- Untracked heavy numerical GIL release; tracked calls retain ownership/GIL.
- Release correctness/allocation/time receipts and reference dispatch audit.

## Remaining concrete fixes

| ID | Checklist | Finding | Exit evidence |
| --- | --- | --- | --- |
| NC1 | 29.61 / 29.3 | Generic attribute-array quantile/corr/cov/window/cumulative paths and table aggregate/pivot/describe/sort paths still have separate f64/f32 logic. These can round large integers before comparison/centering, skip compact integers, mis-handle missing/domain inputs or silently overflow narrowed outputs. | Route compatible arithmetic through checked shared kernels; preserve each documented missing/value/output policy or report its deliberate correction; independently test large integers, mixed types, windows, empty/nonfinite inputs, sorting/pivot identity and Rust/Python behavior. Avoid inferring table writeback policy from numerical fixes. |
| NC2 | 29.62 / 29.4 | Matrix activation/dropout helpers contain ignored cell/conversion errors, metadata loss and tracking detachment; the lower GELU derivative does not differentiate its tanh forward approximation. Evaluation dropout scales values despite inverted training scaling. | Checked native math and controls, faithful derivative/domain handling, preserved metadata/storage where supported, reproducible masking evidence and explicit unsupported tracked boundaries. No silent zeros or discarded errors. |
| NC3 | 29.63 / 29.25 | Sparse maps with nonzero implicit-zero images and nonfinite sparse product/scale fallbacks can materialize or scan large logical grids outside the spectral input guard. | A common explicit expansion policy prevents unintended large-grid work, with exact small-input IEEE results, sparse-zero-preserving behavior and large early-error/opt-in fixtures. Preserve explicit dense conversion as an opt-in. |

These are completion repairs to existing APIs. They do not settle A10 writeback,
A13 broader numerical guarantees or A15 deployment budgets.

## Explicit capability limits, not silently completed features

Complex input matrices/general-complex eigen gradients, arbitrary full SVD null
basis gradients, universal autodiff, sparse iterative spectral methods and graph
matrix Conv2D integration are not provided by the bounded core above. Current
unsupported errors remain visible in the capability record. Any additional
implementation required by the owner's full vision remains tracked; this document
is not permission to retire a promise. Historical SIMD names use reference
kernels. External BLAS/NumPy availability is not claimed by exporting backend types.

## After the fixes

Reconcile 29.3–29.6/29.25–29.26 with actual supported scope and evidence. Keep
production-specific gates open until A15 supplies targets; do not invent a latency
SLA. Continue entity/view/state work on already accepted contracts while those
owner choices remain pending. General engine, service and Graphiti completion
remain separate roadmap milestones.

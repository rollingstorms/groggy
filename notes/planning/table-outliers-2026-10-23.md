# 07.3a.2g — Checked IQR outlier classification

The old table method skipped f64 properties, narrowed integer observations before
sorting and unwrapped unordered NaN comparisons. Delegate through the table
numeric adapter and the shared typed scalar ordering. Preserve existing
floor(n*q) quartiles, strict fences, source row alignment and nonnumeric=false.
Reject nonfinite numeric observations, missing/no-numeric columns and nonfinite
or negative factors explicitly.

Compare x-Q1+factor*(Q3-Q1) and x-Q3-factor*(Q3-Q1) to zero using split scalar
parts and exact finite product accumulation. Shared Dot gains an exact sign query
that normalizes existing bins without rounding to f64. Factor bounds, products or
sums therefore need not fit in f64, and i64 residual bits survive. Integer fast
accumulation remains available. The normal rounded total path shares its existing
normalization loop; no numerical total semantics change is intended.

Native cases cover high-i64 and mixed-width boundaries, sub-f32 values, strict
fence equality, extreme finite factors/values, subnormal products, missing cells
and explicit errors. Independent exact-sign cases cover underflow, large-product
cancellation, zero and unordered NaN. Python cases exercise public dispatch and
GIL-released computation. Release 1k/10k f64 and high-i64 fixtures record the
cost of the exact fences; no speedup or latency budget is promised.

Remaining property/vector/query/view/step adapters and ordinary Python float
inference stay open. This does not complete parent 07.3a or change mutation clocks.
See docs/api/table-outliers.md.

## Adaptive comparison after measurement

Initial exact accumulation for every observation measured roughly 8.6–9.8 ms per
10k rows. Compute correctly rounded fence bounds once. Round-to-nearest is
monotone: distinct rounded values certify the original order, even for integer
observations whose conversion loses low bits. When the rounded observation and
bound coincide, compare the exact residual instead. Zero factor or zero IQR uses
the existing typed ordering directly. No epsilon or heuristic threshold is used.

An independent integer/rational reference checks 160 combinations of sample size
and quarter-unit factor near i64::MAX, covering rank residues and both ordinary
and boundary dispatch. The existing extreme/subnormal/strict-boundary tests
remain. Before receipts refer to the staged all-exact implementation, not the
previous released f32-only method. Both implementations keep the same result
contract; final release receipts measure the adaptive path on the same fixtures.

Executed final gates: full locked offline native tests, strict all-target Clippy,
in-tree FFI rebuild, 33 focused Python regressions and 19 documentation examples.
Release before/after receipts are retained in
`benches/checked_table_outliers_2026-10-23.json`; scope/limits are in
`notes/benchmarks/checked-table-outliers-2026-10-23.md`. The measured 10k cases fall
from roughly 8.6–9.8 ms to 0.16 ms; boundary-collision workloads need more exact
arithmetic. Three existing FFI warnings remain; no external packages installed.

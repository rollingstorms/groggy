# 07.3a.2b — Retain f64 numerical property results

Remove the final f32 narrowing from checked attribute sums/statistics/windows,
pivots and grouped-array/table materialization. Preserve exact integer output
contracts and explicit NumArray f32 dtype. NumArray f64-to-property output retains
f64 without a narrowing error or silent rounding. Legacy f32 stored values stay
unchanged; higher precision applies to new computation results.

Cross-table sum/mean/std now stream borrowed value iterators through the existing
checked column reducers. Avoid intermediate i64 overflow and cancellation loss
across tables without cloning/flattening all rows. Sample std delegates to the
shared scaled typed kernel; it can succeed when sample variance exceeds f64.

The exact binned product accumulator can divide by a positive integer count before
rounding: normalize/subtract signed bins, perform base-2^32 long division, retain
the remainder in the sticky bit and round once. Preserve the existing unit-divisor
path. Attribute sums/means keep their normal compensated path and use the exact
accumulator only on intermediate overflow. Window means/covariance use the same
primitive. Streaming moments widen lazily, rescale before roots, and reject
nonfinite centered updates instead of silently clamping NaN to zero. Ordinary
expanding work remains linear.

Verification includes source-width/shape/key compatibility, sub-f32 precision,
representable means of maximum f64 values, cancellation after overflowing partial
sums, signed subnormal ties/remainders, covariance division, distinct variance/std
ranges, large window moments, true overflow and cross-chunk integer residuals.
Approximate correlations/std references use f64 tolerances rather than demanding
f32-rounded exact equality. Old f32-only overflow fixtures move to actual f64
limits, preserving meaningful rejection checks.

See docs/api/float64-numerical-results.md. Remaining profiling/display, legacy
algorithm/accessor, vector/text/CSV and default-inference work stays open; this is
not completion of parent 07.3a.

Executed: full locked offline native tests, strict all-target Clippy, in-tree FFI
rebuild, 97 focused Python output/statistics/property/predicate/builder/pipeline
tests (2 existing skips), 15 documentation examples, before/after ordinary-path
Criterion receipts and all 55 release numerical workloads. Original 25 checksums
remain identical; five statistics checksums change from retained f64 precision.
See notes/benchmarks/float64-numerical-outputs-2026-10-23.md and both JSON receipts.
Three pre-existing FFI warnings remain; no external packages installed.

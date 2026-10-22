# Shared logical matrix reductions

Item 29.22 unifies GraphMatrix global, axis, trace and legacy column reductions.
Read logical coordinates with checked access, including column-major dense and canonical
CSR inputs. Dense axis scans borrow storage for the operation and use checked strides;
other storage goes through checked element reads. Missing/unreadable cells now error.

Global Rust sum/mean/min/max now return GraphResult, intentionally changing their
public Rust return types. Update callers with ? or match. Python result types and
syntax stay unchanged; its bridge translates errors. Empty sum/mean remain zero;
empty global min/max remain None. Axis outputs always have the non-reduced dimension's
length, including zero for every empty inner reduction. This repairs the former empty
mean-axis vector inconsistency.

Floating sums and trace share the exact floating dot accumulator with unit factors.
Built-in integer sums/trace use checked i128 subtotals and final typed conversion;
output or accumulator overflow errors. Mean works independently of typed sum so a
finite convex mean can survive typed sum overflow. Integer hints retain tested high
offset/cancellation information until f64 output. Floating IEEE NaNs/infinities
propagate; any NaN propagates to min/max even after an earlier finite value. Extrema
retain original T and first ties, including signed zero.

Axis variance/std are sample statistics (N-1), while column_variances retains its
legacy population (N) definition. Both share scaled centered typed kernels. Std takes
the square root before rescaling, so it can remain finite/nonzero when variance
overflows/underflows. Finite singleton dispersion is zero, nonfinite singleton is
NaN; empty dispersion is zero for these matrix APIs. column_means returns f64 with
the same mean kernel. Axis statistics retain T and its established from_f64 conversion
(including integer truncation); no NumPy-style promotion is introduced.

Native global reduction materializes checked logical values in O(cells) scratch.
Axis reductions use O(reduced-axis length) scratch plus output, replacing the former
separate Welford implementation. This correctness consolidation changes allocation
behavior. Profile fixtures in benches/matrix_reductions_before/after_2026-10-21.json
use the existing matrix_statistics_profile.py with 128x96 dev f64 input. They measure
std axes and whole-process RSS only, not all reductions or per-operation allocations.
No production throughput or allocation improvement is claimed.

Remaining: lower-level MatrixStats/UnifiedMatrix/neural reductions, arbitrary-precision
mixed arithmetic, wider result dtypes/storage, parallel/SIMD/GIL work and release-scale
performance. Custom NumericType conversion/ordering behavior remains its own contract.

Evidence: tests/matrix_reductions.rs and tests/test_matrix_reductions.py cover global/
axis/column agreement, checked integer overflow/cancellation/high-offset differences,
population-versus-sample variance, large finite means, independent std range, later
NaNs, empty shape lengths and trace, alongside existing statistics/layout/CSR tests.

Dev 128x96 std-axis medians changed from 0.545 to 2.437 ms (rows) and 0.548 to 3.103 ms (columns). This reference-kernel consolidation costs about 4.47x/5.67x on these fixtures; the precision/consistency repair is not a performance improvement. The measured regression remains part of the production performance work.

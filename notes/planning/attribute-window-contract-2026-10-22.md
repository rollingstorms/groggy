# Attribute windows and cumulative closeout — 2026-10-22

Checklist 29.61b repairs existing BaseArray window, cumulative, shift and
fractional-change paths. Table aggregation/ordering/statistics remain in 29.61.

## Behavior

- Numeric cells include Int, SmallInt and Float. Missing/nonnumeric cells are
  skipped within a statistic; output remains Float (f32), with checked overflow.
- Rolling window size counts input rows; prefixes shorter than that size return
  Null. An all-missing window is Null. Zero size and unsupported operations fail
  even for empty input. Nonfinite numeric cells are rejected, including when a
  requested window exceeds the input length.
- Expanding statistics carry prior values across missing cells. Initial missing
  results are Null; cumsum preserves its initial zero convention. cummin/cummax
  start with Null. Standard deviation keeps the population denominator.
- Expanding/cumulative work uses one pass and fixed working state plus the output.
  An exact common product accumulator retains integer residuals and mixed-value
  cancellation for sums/means. Online moments are centered on an original typed
  value, retaining small differences near i64 extremes. Ordinary final f32
  rounding remains permitted; arbitrary exact real variance is not promised.
- Rolling recomputes each complete window, preserving its O(N*window) work bound.
  This slice does not claim an optimized sliding-moment algorithm.
- Shifts at or beyond the input length return the requested fill, including
  i32::MIN and empty inputs. No negation/length arithmetic can panic.
- Fractional change uses the typed difference before conversion. Missing or zero
  previous values produce Null; zero periods keep the legacy zero-for-nonzero
  convention. Nonfinite participating cells and overflowing outputs fail.

## Evidence

Native and Python tests cover high-offset/mixed variance, exact cancellation,
compact integers, missing rows, extreme shifts, fractional changes, empty
validation, nonfinite values and f32 overflow. The release corpus adds expanding
mean/std at three tenfold input sizes; timings and heap receipts are recorded
separately. Shared Rust logic remains behind the existing FFI methods.

Executed: full locked offline native suites, strict all-target core Clippy (clean),
rebuilt FFI and expanded Python numerical/array gate: **474 passed**. The idle
release runner completed **50 cases**; prior 44 checksums match within 1e-13.
Expanding std/mean scale approximately linearly across 1,700/17,000/170,000 rows,
with two output allocations per call; see
`notes/benchmarks/numerical-attribute-windows-2026-10-22.md`.

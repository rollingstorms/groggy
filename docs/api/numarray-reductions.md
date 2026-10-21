# Shared NumArray reduction kernels

The native `storage::array::numeric_reductions` module implements sum, mean,
minimum/maximum, sample variance, standard deviation and median over borrowed
int32/int64/float32/float64/bool slices. Existing generic NumArray statistical
methods delegate to those kernels through their established f64 conversion bounds;
those generic trait bounds have not been redesigned. Typed native callers can use
the slice kernels directly without converting integers to floats first.

Python NumArray delegates by dtype to the typed Rust kernels. The bridge no longer
copies each column into f64 vectors and implements its own statistics. Python now
also exposes the native median through `NumArray.median()`.

```python
from groggy import NumArray

NumArray([2**63-1, 1, -(2**63-1)], dtype="int64").sum()  # 1.0
NumArray([2**63-3, 2**63-2, 2**63-1], dtype="int64").var()  # 1.0
NumArray([0.0, 1e-300], dtype="float64").std()  # finite, despite variance underflow
```

## Output and edge cases

All nonempty results are f64, including integer/bool sums and extrema. Integer
selection/subtotals preserve typed information until final conversion; the f64
output still cannot represent every i64 value. No result-type widening or integer
return type is implied. Scalar input dtypes and source values remain unchanged.

- Empty sum is zero; empty mean/min/max/median/variance/std are `None`.
  Python empty var/std previously returned zero inconsistently; they now agree
  with the established native Option contract.
- Finite singleton variance/std remain zero for compatibility. Sample variance
  otherwise uses denominator `N-1`.
- Any NaN propagates to every nonempty result. Min/max do not ignore a later NaN;
  median screens NaNs before ordering/selection.
- Infinities participate in sum/mean/extrema/median with explicit IEEE behavior:
  opposite infinities produce NaN for sum/mean and for an even median pair;
  same-sign infinite pairs retain that infinity. Any nonfinite observation
  produces NaN variance/std, including nonfinite singletons.
- Unlike checked logical BaseArray reductions, NumArray does not reject IEEE
  values through GraphResult. Its established f64/Option result API is retained.
- Nulls are not a numeric dtype. Existing NumArray construction/conversion rules
  remain; this slice does not introduce missing-value filtering.

## Numerical kernels

Integer/bool kernels accumulate a checked i128 subtotal before final f64 conversion.
Floating totals use compensated accumulation and rescale when intermediate overflow
would otherwise obscure a finite result. Finite-input means use a scaled fallback
when the total overflows, so a finite convex mean can remain representable.

Variance centers integer differences before f64 conversion. Floating differences
are formed directly when finite, retaining nearby representable differences; if
they overflow, use scaled input differences. Accumulate normalized centered squared
deviations and rescale after dividing by `N-1`. Standard deviation takes the square
root before rescaling, so it can stay finite and nonzero even when variance is
infinite or underflows to zero. Correct rounding for every mixed-scale or
ill-conditioned input is not promised; extreme normalized underflow and ordinary
f64 rounding remain limits. Integer arithmetic hints preserve the tested i64
boundary cases without turning f64 statistics into arbitrary-precision results.

Median keeps typed ordering and uses an overflow-safe midpoint, including exact
integer addition before final conversion. The in-place selection helper uses
standard-library partitioning instead of the former recursively implemented
quickselect, which could exhaust the stack on repeated values. Original NumArray
values remain unchanged; the low-level mutable selector can partition its slice.

## Historical SIMD/auto entry points

`sum_simd`, `mean_simd`, `variance_simd`, `std_dev_simd`, min/max SIMD and auto
methods preserve their public names and now share reference numerical semantics.
The old unchecked vectorized accumulation is replaced by these reference kernels.
**These names currently provide no SIMD speed guarantee.** Reintroducing measured
acceleration while retaining the tested behavior remains under item 29.6.

The lower-level `simd_variance(data, mean)` retains its separate caller-supplied
center contract: it computes the sample moment around that value, which may differ
from variance around the actual mean. It retains its legacy zero result for zero
or one observation. NumArray variance/std wrappers use the actual-mean kernel,
without letting a rounded supplied center change their result.

## Evidence and limits

`tests/numarray_reductions.rs` covers typed integer cancellation/variance/median,
float overflow/underflow, scalar/SIMD/auto boundaries, finite/NaN/infinite domains,
explicit-center semantics and repeated-value selection. Python dtype/rational and
IEEE references are in `tests/test_numarray_reductions.py` alongside existing array
and matrix-statistics regressions. Raw source ownership and output contracts stay
explicit.

Scalar reductions scan in linear time with constant scratch; full-sort median uses
linear scratch and sorting, and NumArray selection median partitions a copy.
These are structural descriptions, not benchmark results. No acceleration,
throughput, GIL-release, ArrayArray, matrix reduction,
dtype-conversion or general arithmetic certification is added.

## Linear percentiles (29.10)

`NumArray.percentile(p)` retains rank `p*(N-1)` and linear interpolation.
The shared native typed slice kernel preserves ordering until interpolation;
`p=0.5` uses the safe typed median midpoint. Empty input or a probability outside
[0,1] (including NaN/infinity) returns None. Any input NaN propagates, including
at endpoint probabilities. Exact ranks return the selected scalar; interpolated
finite/infinite pairs return that infinity, opposed infinities return NaN.
Finite interpolation avoids overflowing endpoint differences. Input stays unchanged;
sorting uses O(N) scratch and O(N log N) work. Output is approximate f64, without
an arbitrary-precision interpolation guarantee. Existing generic NumArray bounds
remain; this slice adds no Python percentile method. `describe()` uses the repair.
Evidence: `tests/numarray_percentiles.rs`, analytic arithmetic-progression ranks,
integer/extreme floating midpoints, IEEE domains and source preservation.

## Pearson correlation (29.11)

`NumArray.correlate(other)` delegates to a shared native typed slice kernel.
The slice API accepts different scalar types for its two inputs; existing generic
NumArray bounds remain unchanged. This slice adds no Python correlation method.

Mismatched lengths, fewer than two pairs, or a finite constant input return None.
For matching lengths of at least two, any NaN/infinity returns Some(NaN), even if
the other input is constant. No pairs are silently dropped. Results are Pearson
correlations, without sample-size weighting or significance estimates.

Center integer differences before f64 conversion. Normalize each column's offsets
independently, using scaled inputs if subtraction would overflow, then center and
rescale residuals. Compensated dot products and separately square-rooted norms
avoid the old unscaled covariance/variance-product overflow and underflow. Clamp
final rounding excursions into [-1,1]. This does not promise arbitrary precision:
normalized values can still lose extreme mixed-scale contributions. Sources stay
unchanged. Work is linear; two owned f64 scratch columns use O(N) memory. No
throughput or allocation improvement is claimed.

Evidence: `tests/numarray_correlation.rs` uses a hand-calculated correlation,
affine sign/scale invariance, huge/tiny finite scales, opposite f64 extremes,
high-offset i64/f64 differences, orthogonal data, bool/mixed scalar inputs,
undefined/nonfinite precedence and source preservation.

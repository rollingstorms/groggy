# f64 numerical property results

Floating results from property-column sums, quantiles, covariance/correlation,
windows/cumulative operations, pivots and grouped table materialization now retain
f64 precision. Exact integer-only sums and existing integer count contracts stay
integer. NumArray-to-property conversion preserves its explicit floating dtype:
f32 arrays produce legacy f32 tags; f64 arrays produce f64 tags. Stored legacy
f32 inputs remain f32 and cannot regain precision already lost.

```python
import sys
from groggy import AttrValue, BaseArray

precise = 1 + 2**-40
values = BaseArray([AttrValue.float64(precise), AttrValue.float64(precise)])
assert values.sum() == 2 * precise
assert values.quantile(0.5) == precise
large = BaseArray([AttrValue.float64(sys.float_info.max)] * 2)
assert large.mean() == sys.float_info.max
assert list(large.expanding("mean")) == [sys.float_info.max] * 2
```

## Range and rounding

Column sums/means keep their compensated normal path. On intermediate overflow,
they rebuild the sum in the existing exact product accumulator. Positive integer
division occurs before its final f64 rounding, retaining representable means even
when their sums cannot fit. Covariance uses the same division primitive. Its
subnormal rounding includes the integer-division remainder as a sticky bit.

Standard deviation uses a shared scaled kernel so it can remain representable
when variance overflows. Streaming windows preserve linear expanding work and
allocate a wide moment accumulator only when the ordinary moment overflows;
large finite moment roots rescale before the square root. Nonfinite centered
updates fail explicitly; they never become a false zero through a NaN clamp.
For example, expanding std at `[-1e300, 1e300]` is supported, while a nonfinite
centered update at `[-f64::MAX, f64::MAX]` is rejected. This is not a universal
correct-rounding guarantee for every statistical formula.

Floating approximations may now expose low bits that f32 previously rounded
away (such as a correlation one ulp below one). Compare approximate statistics
with a justified tolerance. Final nonfinite results and integer range overflow
still fail on the checked property-statistics APIs. Typed numeric/ragged APIs
retain their established IEEE propagation and empty-input policies.

## Cross-table reductions

TableArray sum, mean and standard deviation use the checked column reducers over
borrowed chunk iterators. No concatenated value buffer or per-column intermediate
sum is required. Integer cancellation is retained across chunk boundaries.
Missing/empty chunks contribute no cells; no-cell sum is zero, while mean/std
require their documented sample counts. Nulls are skipped and nonnumeric cells
still fail. Existing group-key identity and row structure stay intact.

Ordinary Python property float inference has not switched yet. Use the explicit
constructors above for f64 input until consumer migration 07.3a.2 is complete.
Profile/display, legacy algorithm persistence/weight access and vector/text/CSV
consumers remain part of the default-migration gate.

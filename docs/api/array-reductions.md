# Checked logical-column reductions

`BaseArray<AttrValue>` now shares native reduction kernels with the older
`storage::array::base::BaseArray` for the sum/mean operations that older API
provides. Python `BaseArray.sum/mean/median/var/std` continue translating results
from Rust; no statistical business logic is added to the bridge.

```python
from groggy import BaseArray

BaseArray([2**63 - 1, 1, -(2**63 - 1)]).sum()  # 1
BaseArray([2**63 - 2, 2**63 - 1]).var()         # 0.5 (sample variance)
BaseArray([-(2**63), 2**63 - 1]).median()       # -0.5
```

## Input and output contract

- Accept logical `Int(i64)`, `SmallInt(i32)`, and finite `Float(f32)` values.
  Skip nulls. Reject booleans, text, collections, other nonnumeric values, NaN
  and infinities explicitly, with the offending input index.
- Empty/all-null input errors for all five reductions. A singleton supports
  sum/mean/median; variance and standard deviation require two non-null values.
- Integer-only sums return `Int(i64)`. Accumulate in checked i128 so an intermediate
  i64 overflow may cancel later; reject only if the final sum exceeds i64 range.
- Sums containing any float retain the existing `Float(f32)` output type, with
  compensated f64 accumulation and checked finite narrowing. A result that rounds
  to infinity errors. This repair does not widen the property type implicitly.
- Mean, median, variance and standard deviation return f64. Variance uses the
  existing **sample** denominator `N-1`; standard deviation is its square root.
  These statistics can succeed even when the sum cannot fit its i64/f32 output.
- Source values/order are unchanged. No graph mutation or historical delta is
  produced by a reduction.

Python floating inputs are stored as f32 by the established property conversion;
reference calculations must use those stored values, rather than assume f64 input
precision survives conversion. Integer precision remains i64 through conversion.

## Numerical behavior

Keep integer subtotal accumulation exact in i128, and float subtotal accumulation
compensated in f64. Split integer-to-float conversion into high and residual parts
when combining with floats, retaining small cancellation results near i64 limits.
Mean uses that compensated combined total and the non-null count.

For variance, subtract an actual input origin **before** converting integer
differences to f64. Use compensated centered means and a second squared-deviation
pass. Mixed integer/float offsets include integer conversion residuals. This
preserves small variations that would disappear if large i64 inputs were first
rounded individually to f64. The scalar input domain is i64/f32; this is not a
claim about arbitrary f64 matrices or NumArray kernels.

Median sorts a copied numeric sequence using exact integer/mixed numeric order,
with signed floating zeros equivalent to integer zero. Even integer medians add
in i128 before division; mixed pairs use compensated high/residual parts. Sorting
rejects nonfinite input first. Negative-zero medians retain their sign where the
selected scalar/pair has that sign.

Compensated arithmetic improves the tested cases; it is not arbitrary-precision
arithmetic or a correctly-rounded guarantee for every mixed-scale cancellation.
Final f32 sums retain normal f32 rounding. Other f64 statistics retain f64 rounding.

## Evidence and remaining work

`tests/array_reductions.rs` checks cancellation under input permutations, output
range errors, exact-reference sample variance over exhaustive small residual
sequences near both i64 limits, mixed int/float variance, median cancellation,
null/type/nonfinite domains and source ownership. A private comparator test checks
antisymmetry/transitivity across integer limits, fractional floats and signed zero.
`tests/test_array_reductions.py` compares the rebuilt Python API with exact rational
references over stored values. Bundled Python 3.9's direct integer variance loses
some high-offset cases, so those reference calculations use Fractions explicitly.

Sum/mean/variance/std scan in `O(N)` time with constant scratch space; median uses
`O(N)` scratch and `O(N log N)` sorting. These are complexity descriptions, not
benchmark results. This slice does not certify min/max, NumArray/ArrayArray/SIMD
reductions, matrix reductions, arithmetic/broadcasting, GIL release, or throughput.
Those remain under numerical roadmap items 29.3–29.6.

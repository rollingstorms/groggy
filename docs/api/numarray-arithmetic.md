# Typed numerical vectors and conversions

Native NumArray now offers checked_add/subtract/multiply, scalar_binary, dot,
true_divide, norm, typed unique and explicit conversion policies. Native
statistics are available across NumericType rather than requiring floating-only
standard-library operator bounds. Integer hints/order retain differences before
final f64 statistical outputs. The old native add Option API delegates to the
checked addition and returns None for a shape or arithmetic error.

Python NumArray supports +, -, *, / and reflected scalar operators, dot(),
norm(), comparisons and typed unique(). Array add/subtract/multiply/dot require
matching dtypes and lengths. Scalars broadcast; integer/bool scalar conversion
must be exact, while floating scalar conversion can round to the output dtype.
Integer overflow raises an error before returning any output; sources stay
unchanged. Bool add/multiply use the existing OR/AND numeric semiring.

True division returns float64 and rejects any zero divisor. Integer operands
are converted to f64 for that explicitly floating output, with ordinary rounding.
Dots preserve the array dtype: finite f64 products accumulate exactly before
rounding, integer products/subtotals use checked i128 and final range checks,
and f32 narrows after f64 accumulation. Norm is f64, uses scaled hypot and
propagates NaNs. Empty dots/norms are zero.

Scalar/array comparisons use native numeric order without first rounding i64
values to f64, including mixed integer/float scalar limits. IEEE NaN is unordered:
equality and ordered comparisons are false; not-equal is true. Membership uses
actual numeric equality. Unique retains dtype and first-occurrence values,
coalescing signed zeros and NaNs instead of parsing Python repr strings.

Existing dtype constructors and to_type(dtype) retain nearest-integer rounding
with ties away from zero; out-of-range/nonfinite integer conversion and finite
float narrowing overflow now raise errors instead of wrapping/saturating.
Floating rounding/underflow remain allowed by that explicit conversion.
`to_type(dtype, exact=True)` rejects representability loss. Bool construction
remains strict; to_type(bool) retains numeric truthiness, while exact=True
requires zero or one. Legacy basearray conversion retains f32 attribute storage,
with the same finite overflow/exactness checks.

Native NumArray.cast/GraphMatrix.cast default to exact numeric representability;
cast_with_policy opts into rounding/truncation. Matrix casts preserve labels and
CSR storage. Matrix Python storage remains float64; NumArray.reshape checks its
conversion to that boundary, requiring an explicit rounded to_type('float64')
first when integer precision would be lost. Numeric scalar reads used by reshape
now avoid copying the entire array for each element.

Outputs own immutable backing values; no graph binding or gradient tape is
inherited. Tests cover integer limits/cancellation/order, dtypes/shapes, scalar
broadcast, empty/nonfinite/signed-zero cases, conversion rules and independent
float norm references. Release performance remains part of the completion pass.

## Attribute and indexing boundaries (29.48)

The legacy attribute-to-NumArray bridge now performs inference in Rust without
`abs(i64::MIN)`. It keeps compact integer inference for existing table paths;
`BaseArray.to_num_array()` retains bool-only arrays, int64 integer arrays,
float64 mixed/floating arrays and an empty float64 result for empty input.
Null/non-numeric values reject conversion: filter or replace them explicitly
rather than interpreting a missing value as zero or losing array length.

`BaseArray.to_type('int64'/'float64', exact=False)` now shares native checked
conversions. Its existing default nearest-integer rounding remains; finite
range errors and NaN/infinity-to-integer conversions reject. `exact=True`
rejects fractional/precision loss. The i64/f64 aliases remain available.
Mixed float/integer promotion to float64 still rounds integers under the default
policy; request exact conversion when that loss is unacceptable.

Fancy indices supplied as numeric NumArrays must be exactly representable i64
positions. Integral floats remain accepted, fractional/nonfinite/out-of-range
positions reject, and bool dtype retains mask semantics. Native unsigned graph
IDs also reject if a signed int64 NumArray cannot represent them, instead of
wrapping. Tests: `numarray_bridge_boundaries.rs` and
`test_numarray_bridge_boundaries.py`, plus existing array compatibility suites.

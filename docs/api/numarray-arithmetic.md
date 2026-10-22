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

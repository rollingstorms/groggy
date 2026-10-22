# Floating matrix dot products

Item 29.15 replaces sequential floating product/accumulation in GraphMatrix.multiply
(and its Rust/Python matrix operators). Native f64 inputs now accumulate exact finite
products in fixed base-2^32 integer bins and round the final sum once to nearest,
ties to even. This retains residuals after cancellation even when individual products
would overflow, or when individually rounded products would underflow. No external
linear algebra backend is used.

The accumulator covers the full f64 product exponent range with 64 carry bits for
a usize-sized inner dimension. Positive and negative magnitudes normalize separately,
then subtract before final rounding. Storage is fixed scratch per dot, independent of
the inner dimension; the matrix shape and owned positional identity rules are unchanged.
Exact cancellation and empty dots return positive zero. A negative result rounded to
zero retains its sign. True output overflow yields signed infinity.

NaN operands, zero times infinity, or infinite products of both signs produce NaN.
Other infinite products produce their signed infinity. Finite product overflow is not
mistaken for an infinite input before cancellation.

f32 operands convert exactly to f64, share the accumulator, then narrow the final
f64 result to f32. This removes intermediate f32 product overflow but retains possible
double rounding at f32 boundaries; no correctly-rounded f32 promise is made. Integer,
bool and custom nonfloating NumericType paths are unchanged in this slice and still
need their separate overflow/domain audit. This kernel is not a general replacement
for all lower-level UnifiedMatrix/neural/sparse dot-product paths.

Evidence: tests/matrix_dot_product.rs covers cancellation across the exponent range,
underflow recovery, ties/sticky rounding, nonfinite domains and f32 product overflow.
tests/test_matrix_dot_product.py compares seeded finite dots to independent exact
Python Fraction products, including output overflow. Performance evidence is recorded
by benches/matrix_dot_profile.py and matching before/after JSON files. These are dev
extension timings and whole-process RSS, not release throughput or per-operation
allocation measurements. Fixed-bin reference accuracy has a cost; measured acceleration
and wider production-scale benchmarking remain under item 29.6.

Measured dev medians (same staged extension setup): 1x1024 by 1024x1, 0.200 to 0.231 ms; 16x32 by 32x16, 1.201 to 10.644 ms; 32x64 by 64x32, 9.368 to 48.532 ms. This reference repair regresses these small-product timings by about 1.15x/8.86x/5.18x. These fixtures establish the cost to address in measured acceleration; no speed improvement is claimed.

## Checked integer fast path (29.21)

Integral-valued finite factors with magnitude at most 2^63 first accumulate checked
i128 products/totals. If any product or subtotal cannot continue, or a non-integral
factor appears, flush the exact integer subtotal into the exponent bins and continue
there. No integer bits are discarded at this transition. Nonfinite checks precede
zero shortcuts so zero times infinity still produces NaN. All-integral totals round
to f64 only at the end; the numerical result contract is unchanged.

This optimization targets integral-valued floating workloads; general fractional
inputs still use the reference bins and retain their measured cost. Matching dev
fixtures are benches/matrix_dot_fast_profile.py and its before/after JSON files,
including a fractional control workload. Release-scale throughput, allocation counts,
SIMD/GIL work and efficient fractional kernels remain open under item 29.6.

Final rounding also checks discarded bits by whole 32-bit words, preserving halfway/sticky semantics while avoiding a bit-by-bit scan. Matching dev before/after medians were 0.230→0.179 ms (1x1024 dot), 16.331→1.470 ms (16x32 by 32x16 integral), 52.646→11.385 ms (32x64 by 64x32 integral), and 10.437→6.098 ms (16x32 fractional control). The raw contended after-run is retained separately; its samples vary substantially. These small noisy dev fixtures compare against the exact-bin reference, not a promise of speedup against the original inaccurate kernel or release workloads.

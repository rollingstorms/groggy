# Checked matrix arithmetic

Item 29.12 repairs the GraphMatrix `add` and `subtract` methods and Python `+`,
`-`, scalar `*`, scalar `/`, and unary negation. Binary addition/subtraction require
identical shapes and return errors on mismatch. Scalars broadcast over every cell.
Python `*` with a matrix retains matrix multiplication semantics.

Scalar addition/subtraction/division and negation execute in Rust without building
an extra constant matrix. Division uses each value divided by the scalar directly;
either signed zero is rejected, while small nonzero divisors remain valid. NaN,
infinity, overflow and underflow otherwise follow f64 arithmetic. This is not a
checked-finite arithmetic API. Division by zero rejects even for an empty matrix.

These methods allocate owned results, preserve all declared empty shapes, and
retain the left operand's positional row labels and column names. Binary arithmetic
is positional: labels do not cause joins or reorder the right operand. Sources stay
unchanged. Results do not inherit graph bindings, gradient tapes or cached properties.
Allocation/cell-access/set errors propagate through GraphResult to Python errors.

**Remaining work:** Rust `+`, `-` and `*` trait operators still have legacy plain-matrix
outputs and can substitute zeros on failure. Call checked methods instead. Migrating
those trait outputs and their consumers remains open. General integer overflow,
mixed dtype promotion, NumArray operator expansion, general sparse arithmetic,
matmul numerical stability, autodiff and measured throughput remain unaudited.
No Python reverse operators or general broadcasting rules are introduced here.

Evidence: `tests/matrix_arithmetic.rs` and `tests/test_matrix_arithmetic.py`, including
rectangular arithmetic, shape mismatch, source/label preservation, degenerate shapes,
smallest positive f64 division, signed zero and IEEE values.

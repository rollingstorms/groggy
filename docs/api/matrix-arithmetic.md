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

## Rust operator migration (29.14)

Borrowed and owned Rust `+`, `-` and `*` now return `GraphResult<GraphMatrix<T>>`
and delegate to the checked named methods. This intentionally changes their public
associated Output type. Update callers to handle the Result explicitly:

```rust
let sum = (&a + &b)?;
let difference = (a.clone() - b.clone())?;
let product = (&a * &b)?;
```

A dimension mismatch no longer produces plausible zero data. All-target compilation
checks repository callers; downstream crates using plain-matrix operator outputs
must migrate similarly. Named `add`, `subtract` and `multiply` keep their existing
GraphResult API. Python's valid-expression syntax and error translation stay intact.

Multiplication validates output element/byte counts and propagates allocation and
cell errors. It preserves left row labels and right column names; inner dimensions
are positional, without label alignment. Zero inner dimensions correctly produce a
zero product with the declared output shape. Scalar and elementwise checked arithmetic
share the output-size validation. This is not a guarantee that process-wide OOM or
arbitrary custom NumericType operations are recoverable.

General integer overflow, mixed dtype promotion, NumArray operator expansion,
general sparse arithmetic, matmul numerical stability, autodiff and measured
throughput remain unaudited. The product kernel still performs ordinary sequential
multiply/add accumulation in T, without a compensated dot-product guarantee.
No Python reverse operators or general broadcasting rules are introduced here.

Evidence: `tests/matrix_arithmetic.rs` and `tests/test_matrix_arithmetic.py`, including
rectangular arithmetic, shape mismatch, source/label preservation, degenerate shapes,
smallest positive f64 division, signed zero and IEEE values.

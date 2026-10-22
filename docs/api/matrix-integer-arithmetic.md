# Checked built-in integer matrix arithmetic

Item 29.17 makes checked GraphMatrix addition, subtraction and Hadamard multiplication
(and Rust + / - delegates) reject built-in i32/i64/i128 scalar overflow. Results are
owned and source values remain unchanged on success or error. The Hadamard path now
shares checked shape/allocation/cell handling and positional labels with add/subtract.

Integer matrix products accumulate exact integer products in a checked i128 subtotal,
then convert once to the output integer dtype. This retains i32/i64 cancellation when
an individual product would exceed the output dtype but the final result fits.
Reject product/subtotal i128 overflow and final output narrowing overflow. In particular,
i128 products can reject even when later cancellation would make the mathematical
answer fit. This is bounded checked arithmetic, not arbitrary precision.

Empty inner dimensions yield zero integer products. Built-in bool retains its prior
OR-of-AND semiring behavior. Floating paths retain their IEEE and dot-product contracts.
No automatic promotion, broadcasting or widening of the output dtype is introduced.

NumericType has additive checked-operation and exact-integer conversion hooks with
defaults for compatibility. Built-in integers override them; custom types own their
scalar-operation behavior and must provide integer hints to support the integer dot
path. Low-level NumericType.add/sub/mul, UnifiedMatrix backends, other integer matrix
routines and algorithm-builder policies are outside this slice.

Evidence: tests/matrix_integer_arithmetic.rs covers i32/i64/i128 boundaries, source
preservation, checked trait operators, widened subtotal cancellation, final/accumulator
overflow, bool and empty products. Python GraphMatrix is f64; its rebuilt numerical
regressions verify that this native integer repair does not alter that interface.

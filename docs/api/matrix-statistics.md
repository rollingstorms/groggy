# Matrix statistics and entrywise norms

`GraphMatrix.norm()` is the Frobenius norm. `norm_l1()` sums absolute
entries; `norm_inf()` returns the largest absolute entry. These last two are
entrywise norms, not induced matrix norms. Empty matrices return zero. All
three propagate NaN, validate storage/layout metadata and return native
`GraphResult<f64>` (Python floats or exceptions). Stable hypot accumulation
avoids squaring overflow/underflow; L1 uses the shared accurate sum kernel.

Global means and axis statistics use checked logical layouts. CSR operations
reduce stored entries and account for implicit zeros analytically. Column
reductions build a CSR transpose once; they do not materialize the matrix.
Floating statistics remain approximate; a sparse group combination may round
differently from the equivalent dense computation.

Axis identifiers retain Groggy's existing convention: `Rows` / Python `0`
reduces each row, `Columns` / Python `1` reduces each column. Axis outputs
retain native T. Integer means divide exact integer quotients/remainders before
final truncation toward zero, including i128 extremes. Other integer statistics
truncate the computed f64 output and reject unrepresentable values. f64 values
at positive 2^63/2^127 no longer saturate to i64/i128 MAX during conversion.

Axis std/variance use the sample denominator N-1; column_variances and lower
`MatrixStats.std` use the population denominator N. Empty core means/axis
statistics return zero. The lower global MatrixStats mean/std preserve NaN for
empty input. Finite singleton dispersion is zero; nonfinite observations yield
NaN dispersion. Lower statistics retain integer differences until scaling,
and never store squared residuals in the input dtype.

Evidence: `tests/matrix_statistics_checked.rs` and
`tests/test_matrix_statistics_checked.py` cover independent dense references,
large sparse shapes, extreme ranges, integer offsets, invalid layouts,
nonfinite data and empty reductions. This establishes numerical/storage
behavior; release performance remains a separate gate.

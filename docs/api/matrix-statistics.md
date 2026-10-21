# Matrix identity and axis statistics

Groggy's axis names retain their existing meaning: `Axis::Rows` / Python `0` reduces each row across its columns; `Axis::Columns` / Python `1` reduces each column across its rows. These labels are Groggy's API, not NumPy's axis numbering. The existing nested-list factory takes columns; this change preserves that behavior rather than silently transposing existing callers' inputs.

`GraphMatrix::eye(n)` now creates a diagonal identity instead of an all-ones matrix. The checked `identity(n)` uses the underlying identity constructor directly. Zero-sized identity and rectangular empty shapes are preserved rather than silently replaced by a 1x1 allocation fallback.

`var_axis` computes sample variance with divisor `n - 1`; `std_axis` computes its square root instead of returning sums. Empty and finite singleton reductions return zero, retaining the existing variance convention. NaN/infinite samples produce NaN where the result type supports it. Welford updates traverse the matrix in row-major order, with f64 intermediate accumulators and O(number of output values) auxiliary storage. Dense reductions hold one read guard and respect the storage strides; other storage uses the existing element-access path. Results retain the matrix's element type; unrepresentable finite conversions raise an error instead of silently substituting zero.

```python
from groggy import GraphMatrix

# The existing nested-list factory accepts columns.
columns = [[1., 4.], [2., 6.], [3., 8.]]
matrix = GraphMatrix.from_data(columns)
assert matrix.shape == (2, 3)
assert matrix.std_axis(0).to_list() == [1., 2.]
assert matrix.var_axis(0).to_list() == [1., 4.]
```

Aligned empty buffers retain a valid aligned allocation but expose zero elements. Checked shape/byte multiplication rejects overflow, and a shared buffer created from data requires exactly the requested element count. f32 conversion now retains representable NaN/infinity, while finite f64 values outside the f32 range still reject.

## Remaining numerical work

This repairs specific demonstrated placeholders, not the full numerical library. Current generic axis statistics retain integer-result conversion/truncation and use f64 intermediate precision; they do not promise exact statistics for arbitrary i64/i128 magnitudes. Wider statistical output types and precision guarantees remain under item 29 and appendix A13. The [real SVD and symmetric eigendecomposition repair](matrix-decompositions.md) now provides bounded iterations and explicit domain errors. General mean/reduction behavior, broadcasting, sparse operations, nonsymmetric/complex decomposition and gradient correctness still require their own audits. Legacy infallible allocation-failure fallbacks in constructors such as zeros/ones/eye are not removed by this change; checked constructors should be used where errors must be reported.

Verification: `tests/matrix_statistics.rs` checks identity multiplication, axis mapping, sample statistics, large offsets, f32 nonfinite values, singleton/empty shapes and invalid buffer sizes. `tests/test_matrix_statistics.py` compares each axis against Python's independent statistics library through the built extension. `benches/matrix_statistics_profile.py` provides a repeatable timing/RSS fixture; it reports whole-process peak RSS, not per-operation allocation or release-build guarantees.

The initial dense read-guard timing comparison is recorded in [benchmark notes](../../notes/benchmarks/matrix-statistics-2026-10-20.md), including the debug-build limitation and raw samples.

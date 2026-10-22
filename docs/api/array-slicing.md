# Shared native slice resolution

BaseArray, typed NumArray, BoolArray and native MatrixIndex ranges use the same
SliceIndex resolver. Raw optional start/stop/step values are normalized **once**
against the actual dimension length. Negative offsets, clamping and omitted
reverse bounds follow Python list slicing: `[::-1]` includes index zero, while
`[:-1:-1]` is empty. Selected values remain ordered owned copies.

The Python bridge now marshals raw slice fields instead of normalizing them
against an artificial maximum array length. Slice fields use the existing i64
native representation; out-of-range Python integers reject during marshalling.
Zero steps raise ValueError through the shared native validation method.

Native positions/counts use i128 intermediates and reserve the checked result
length before iteration. Extreme i64 steps cannot overflow a signed addition
or loop indefinitely. Allocation/platform limits return errors. Single negative
indices likewise avoid narrowing the dimension length to i64.

Matrix row extraction now reads selected cells rather than building every full
column just to take one row. Sparse cell-read validation still has its documented
row-access costs; this is not a claim of universal constant-time row extraction.

`tests/array_slice_boundaries.rs` checks defaults, bounds, extreme steps,
allocation limits, owned arrays and native matrix ranges.
`tests/test_array_slice_boundaries.py` compares 8,400 slice combinations against
Python lists across BaseArray, numeric/bool arrays and empty/small lengths, plus
matrix order/shape cases. Existing numerical and graph-matrix suites also pass.

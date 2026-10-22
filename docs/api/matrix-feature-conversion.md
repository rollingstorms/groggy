# Feature matrices and attribute conversion

`graph.to_matrix()` exports owned f64 numerical data. Native typed variants
`to_matrix_i64`, `to_matrix_f32` and `to_matrix<T>` retain their output dtype.
Rows follow ascending active node IDs, with `node_ID` row labels. Columns follow
sorted attribute names. Each included column has at least one numeric scalar
in this graph state; missing, null and nonnumeric cells retain legacy zero
imputation. This is an explicit feature-export policy, different from strict
BaseArray numeric conversion, which rejects nonnumeric/null inputs.

Reads resolve GraphSpace property indices into borrowed pool columns, including
after branch switches. Values are not taken from the latest global pool lookup.
Invalid references and allocation/shape overflow produce errors. Integer output
retains stored i64 values exactly; default f64 output can round large integers.

Explicit `FromAttrValue` conversion retains text parsing, null-to-zero and
finite float-to-integer truncation toward zero. Nonfinite/out-of-range integer
conversion and finite f32 overflow now return errors instead of saturating,
wrapping or silently introducing infinity. Explicit floating infinity/NaN stay
representable. Attribute Float storage is f32; conversion cannot restore
precision already absent from storage.

Lower-level `UnifiedMatrix.fill` validates dense logical addresses before any
writes. It fills selected logical cells while preserving padding; invalid strides
return errors without partial mutation. CSR fill retains CSR: nonzero fill stores
all logical cells, while zero fill clears stored values. Callers should expect
quadratic storage when filling a large sparse matrix with a nonzero value.

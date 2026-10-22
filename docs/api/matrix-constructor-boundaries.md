# Matrix construction and array boundaries

Native `try_zeros`, `try_ones`, `try_eye`, `identity`, `try_from_storage`,
`from_arrays`, `from_csr` and `from_row_major_data` return construction errors,
including shape/byte and default-column-label allocation overflow. Requested
shapes are never substituted with a fallback 1x1 matrix. Legacy infallible
`zeros`, `ones`, `eye` and `from_storage` retain their signatures and panic on
construction failure; use checked variants for external dimensions.

Python constructors use checked variants and raise exceptions. Python
GraphMatrix currently stores **float64**. Its zeros/ones dtype argument accepts
`float`, `float64` and `f64`; other types reject explicitly, replacing the old
behavior that accepted a dtype and silently ignored it. Typed native matrices
and Python NumArray provide other numerical dtypes.

`groggy.matrix`, `GraphMatrix.from_data`, column-array construction,
`from_base_array` and `from_flattened` use native checked numeric conversion.
Missing/nonnumeric cells and integers not exactly representable as f64 reject.
Compact integers and bools are numeric; they no longer become zeros through an
incomplete attribute match. Explicitly convert a NumArray to float64 with the
chosen rounding policy first when approximation is intended. Feature export
`graph.to_matrix()` retains its separately documented zero-imputation policy.

Factories keep Groggy's column-oriented nested-list convention. Explicit
`from_flattened`/`from_base_array` dimensions describe row-major values and use
the same native shape validation, including empty dimensions. NumArray reshape
uses that shared path.

`to_base_array(exact=False)` exports logical row-major values with checked
rounding into the attribute storage types; finite f64-to-f32 overflow errors.
`exact=True` also rejects representational precision loss. The native
`to_attribute_array(policy)` and `NumArray.from_attributes(values, policy)`
expose the same conversion rules. Attribute floating storage remains f32.

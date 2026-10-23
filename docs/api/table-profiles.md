# Checked table profiles

`BaseTable.profile()` summarizes each source column in a row. It includes f32 and
f64 numeric cells, skips null/text/Boolean/vector cells for numeric statistics,
and rejects nonfinite numeric observations with an error. It does not mutate the
source or record a graph change.

Mean, median, population standard deviation, null percentage and cardinality
retain f64 results. Minimum, maximum and rank quartiles retain the original scalar
value and type, including exact i64 extrema. Quartiles keep the existing
`floor(n * q)` rank convention for q=0.25 and q=0.75; they differ from describe's
interpolated quartiles. A single numeric observation has population std zero.

Unique counts use typed property identity, including one distinct null if present.
Repeated text no longer counts both its content and debug representation. Dtype
uses the most frequent non-null type; ties choose the lexically first type name.
F64 scalars/vectors report `float64`/`float64_vec`; existing f32 names remain
`float`/`float_vec`. Vectors do not become numeric scalar observations.

Empty columns have zero counts and null ratio/statistic fields; undefined ratios
are not NaN. An all-null nonempty column has 100% nulls and one unique typed value.
Source values already rounded to f32 are not recovered by wider computation.

```python
from groggy import BaseTable, AttrValue

x = 1 + 2**-40
profile = BaseTable.from_dict({"x": [AttrValue.float64(x)] * 2}).profile()
assert profile.column("mean").to_list() == [x]
assert profile.column("std").to_list() == [0.0]
```

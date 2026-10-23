# Checked table outliers

`BaseTable.check_outliers(column, factor=None)` returns one Boolean per source
row using Tukey IQR fences. The default factor is 1.5; explicit factors must be
finite and nonnegative. Numeric observations include i32/i64 and f32/f64 scalars.
Null, text, Boolean and vector cells are excluded from the quartile sample and
receive false flags. An empty/no-numeric sample or a missing column is an error.
Nonfinite scalar observations reject before sorting rather than panicking or
silently disappearing.

Keep the existing rank convention: sort numeric observations, choose Q1 at
`floor(n/4)` and Q3 at `floor(3*n/4)`. A value is an outlier only when it is strictly
below `Q1 - factor*(Q3-Q1)` or strictly above `Q3 + factor*(Q3-Q1)`. Equality is an
inlier. These rank quartiles are distinct from interpolated describe quantiles.

Ordering preserves integer distinctions before conversion. Fence comparisons use
the exact binary input values, split integer parts and the shared exact product
accumulator. Correctly rounded bounds certify order when rounded values differ; ties use an
exact residual comparison. This preserves correctness even when intermediate
products exceed f64 or underflow. The result reflects supplied binary
floating values, not an inferred decimal/rational interpretation. Stored f32
precision cannot be recovered by widening it.

The method reads an owned table without publishing graph changes. Python releases
the GIL during Rust work. Sorting takes O(n log n); each numeric row uses at most
two fixed-bin fence comparisons, only when rounded bounds cannot certify order.
Zero-factor or zero-IQR cases use direct typed comparison. This extra arithmetic preserves boundary
precision; no constant-time or interactive-latency guarantee is implied.

```python
from groggy import BaseTable

maximum = 2**63 - 1
source = BaseTable.from_dict({"x": [maximum-100, maximum-4, maximum-3,
                                    maximum-2, maximum-1, maximum, maximum, maximum]})
assert source.check_outliers("x").to_list() == [True] + [False] * 7
```

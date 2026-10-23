# Whole-table numeric dispatch

`describe`, `corr`, `cov`, `rolling_all`, `expanding_all`, `nlargest` and `nsmallest`
include explicit f64 columns. Compact integers also participate in all-column
windows. Boolean, text and vector cells are not numeric scalar observations.
These entry points delegate arithmetic to the same native checked column kernels
used by direct array operations.

Correlation and covariance errors propagate, including invalid methods, negative
ddof and nonfinite paired observations. Undefined results retain the column
kernel's documented result rather than becoming a success through a fallback.
Diagonal correlations now call that kernel too; they are not unconditionally 1
for constant/insufficient/invalid observations. Rank and percentile conventions
remain specific to each existing operation.

Matrix display materialization uses checked reads and the same typed scalar
property conversion as `NumArray.to_attributes()`: integers stay integers, f32
stays f32, f64 stays f64. A read failure returns an error instead of displaying a
fabricated zero. Display formatting itself can still round text according to the
selected theme; it does not change source properties. Array materialization no
longer allocates an intermediate vector of scalar-conversion inputs.

```python
from groggy import AttrValue, BaseTable

x = 1 + 2**-40
source = BaseTable.from_dict({"x": [AttrValue.float64(x), AttrValue.float64(1)]})
assert source.nlargest(1, "x").column("x").to_list() == [x]
assert "x" in source.describe().columns
assert source.expanding_all("mean").column("x_expanding_mean").to_list()[0] == x
```

CSV/JSON interchange and outlier classification now have their own checked
adapters; see table-interchange.md and table-outliers.md. Other table/view/step/
vector consumers and ordinary Python float defaults remain separate migration work.

# Floating consumer migration

Stored f64 scalars/vectors now flow through the supported array, aggregation,
property, table and graph visualization adapters without an implicit f32 cast.
Python scalar masks and sorting delegate to native BaseArray operations. Numeric
masks use the shared exact comparator and IEEE NaN rules, including `!=`.
Boolean/text domains remain separate; null orders below other values.

Ascending mixed-value sorting places null first, then numeric values (NaNs after
other numbers), then other typed properties in their logical order. This avoids
contradictory comparisons across f32/f64/text categories. Equal keys retain source
order. Scalar vector/structured equality uses logical property identity. This
array contract is separate from graph typed attribute equality.

Legacy edge/meta and table-array numeric aggregations now delegate to checked
column kernels. Floating computed sums/means retain f64; extrema retain the source
scalar. Integer cancellation is checked before output range errors. Counts and
caller-specific empty/nonnumeric conventions stay at the existing adapters.
Aggregation failures propagate instead of turning into silent null success.
Callback map results use the canonical property translator, preserving Boolean
and integer results rather than extracting them as floats first.

Graph position vectors and scalar coordinates accept f64; computed position and
smoothing properties retain f64. Table dtype/formatting, external numeric import
and legacy numeric lookups recognize f64. Explicitly typed f32 numerical storage,
getters and compression remain f32. Interchange formats without dtype metadata
cannot recreate the original floating tag.

Use `AttrValue.float32(value)` or `AttrValue.float_vec(values)` for explicit f32
properties. These permit rounding and nonfinite values but reject finite overflow
to infinity through the native conversion policy. `float64`/`float64_vec` preserve
f64. Empty explicit constructors keep the requested width.

```python
from groggy import AttrValue, BaseArray

x = 1 + 2**-40
values = BaseArray([AttrValue.float64(1), AttrValue.float64(x)])
assert (values > AttrValue.float64(1)).to_list() == [False, True]
assert values.sort(False).to_list() == [x, 1]
assert AttrValue.float32(0.1).value != 0.1
```

Ordinary Python property floats now default to f64; see
[default migration](float64-defaults.md). This migration does not complete the separate temporal-type, immutable legacy-view,
compiled-query or module compatibility retirement roadmap items.

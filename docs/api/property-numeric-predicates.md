# Property numeric predicates

Array `QueryEvaluator` comparisons and table numeric predicates share the native
numeric scalar comparator with graph numeric ordering. It accepts i32/i64 and
f32/f64 properties, retains integer values until comparison, and widens f32
without narrowing f64. Boolean and text properties retain their separate domains.

Supported comparison operators are `==`, `!=`, `<`, `<=`, `>` and `>=`.
Numeric query equality is exact, with IEEE unordered NaN behavior: equality and
ordering against NaN are false, inequality true. Infinities compare normally.
No epsilon-based equality or float-to-integer conversion is used. This is distinct
from typed property identity used by stored values and graph attribute-equality
filters; those existing identity rules are unchanged here.

Array floating literals parse at f64 precision. Tables parse integer thresholds
as i128 before considering floating syntax, preserving large-integer comparisons.
A previously rounded f32 value remains its actual stored value: querying a decimal
at f64 precision does not recover or hide earlier rounding. Array `>=`/`<=` tokens
are matched before their single-character prefixes. f64 vectors participate in
the existing length predicate. The supported `% 2 == 0` shortcut accepts only
finite floating multiples of two, avoiding truncation/saturation of fractions,
NaNs and infinities.

```python
import groggy as gr

value = 1 + 2**-40
table = gr.BaseTable.from_dict({"x": [gr.AttrValue.float64(1.0), gr.AttrValue.float64(value)]})
assert table.query("x > 1.0").column("x").to_list() == [value]
assert table.query("x <= 1.0").column("x").to_list() == [1.0]
```

Cross-table numeric extrema use the same exact comparator and reject unordered
NaNs. The remaining expression grammar/planning unification under AR7.7 is open:
this does not replace the parsers or promise compiled/broadcast predicate plans.
Ordinary Python property float defaults have not switched yet (07.3a.3).

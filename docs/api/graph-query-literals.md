# Graph query numeric literals

The native graph `QueryParser` shared by live subgraph and immutable GraphView
filters parses ordinary decimal/scientific literals as f64. Examples include
`1.0000000000009095`, `1e300`, `+1E-300`, `-.5` and `.5`. Integer syntax still
requires an i64 value and retains compact-integer inference where applicable.

Suffixes choose floating property identity explicitly: `1f32`, `0.1f32`,
`1e-3f32`, `1f64` and `0.1f64`. f32 literals parse directly at f32 precision;
f64 literals parse directly at f64 precision. Decimal-to-binary rounding,
including underflow, follows the selected width. Overflow/nonfinite results,
invalid exponents, unknown suffixes and malformed numbers reject explicitly.

Graph equality/inequality filters retain **typed property identity**, including
width distinctions and the existing stored zero/NaN rules. Thus `x == 1.0`
selects f64 1.0, while `x == 1.0f32` selects stored f32 1.0. Update queries against
existing f32 properties with an explicit suffix when equality is intended.
Ordering continues to compare scalar numbers across integer/floating widths
without narrowing; an already rounded f32 value is not reconstructed as its
original decimal input. Programmatic filters preserve their existing semantics.
Array/table numeric equality remains its separately documented numeric comparator.

The parser now requires the complete input to be consumed. Trailing identifiers,
extra values/parentheses, incomplete numbers and unterminated strings fail instead
of silently applying a valid prefix. The same syntax and rejection rules apply
to node and edge queries. Filtering a captured view continues to read its captured
state; parsing/filtering does not publish a graph change.

`null` is a case-insensitive contextual value literal. `x == null` selects only
an explicit Null property; an absent property matches neither `x == null` nor
`x != null`. `NOT (x == null)` also includes absence under the existing complement
rule. `null == null` can query a property named `null`; quoted `'null'` remains
text. Node and edge filters share these rules.

```python
from groggy import AttrValue, Graph

graph = Graph()
a = graph.add_node(x=AttrValue.float64(1 + 2**-40))
b = graph.add_node(x=AttrValue.float64(1e300))
view = graph.read_view()
assert view.filter_nodes("x == 1.0000000000009095").node_ids == [a]
assert view.filter_nodes("x >= 1E+300").node_ids == [b]
```

This is a graph literal/validation migration, not completion of AR7.7 parser or
compiled query planning unification. Ordinary Python property floats now default
to f64; old typed f32 stored values remain unchanged.

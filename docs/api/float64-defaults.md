# Python floating defaults

Ordinary Python floats now become `float64` properties. Numeric sequences
containing floats become `float64_vec`; empty ordinary sequences use that same
default. All-integer, Boolean and text sequences retain their respective types.
Mixed structured containers keep their existing JSON representation.

```python
from groggy import AttrValue, Graph

x = 1 + 2**-40
graph = Graph()
node = graph.add_node(score=x, embedding=[x, 1e300], labels=[])
assert graph.get_node_attr(node, "score") == x
assert graph.get_node_attr(node, "embedding") == [x, 1e300]
assert AttrValue(x).type_name == "float64"
assert AttrValue([]).type_name == "float64_vec"
assert AttrValue.float32(0.1).type_name == "float"
assert AttrValue.float_vec([]).type_name == "float_vec"
```

The shared translator covers direct and bulk graph writes, tables, callback
results and checked journal writers. Reserved empty node labels normalize to
an empty text vector. Wrapped algorithm parameters use the native parameter
converter, retaining f64 scalar/list precision.

## Explicit widths and compatibility

- `AttrValue.float32` and `AttrValue.float_vec` request f32. Finite overflow
  rejects; rounding and IEEE nonfinite values remain supported.
- NumPy float32 scalars, arrays and homogeneous sequences of float32 scalars
  retain f32 without importing NumPy during translation. NumPy float64 uses f64.
- Existing tagged f32 data remains f32 when loaded, exported or recovered.
  Widening cannot recover precision lost in earlier writes.
- Schema families `float` and `float_vec` accept either width; `float32`,
  `float64`, `float32_vec` and `float64_vec` require the specified width.
- Graph property equality is typed: use explicit f32 values or `f32` query
  literal suffixes to match stored f32. Array/table numeric predicates compare
  across widths using the shared numeric comparator.

Raw f64 vectors use eight bytes per element versus four for f32. Explicit f32
remains available for workloads that prefer the smaller representation. Plain
table CSV/JSON lacks complete dtype metadata; use the tagged graph codec for
width-preserving storage. The default switch does not rewrite existing files.

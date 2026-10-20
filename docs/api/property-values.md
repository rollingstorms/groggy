# Property values

Graph writers and `groggy.AttrValue` share one conversion policy. Graph getters, attribute columns, table cells and recorded snapshots decode supported values consistently.

| Python input | Logical property type | Python result |
| --- | --- | --- |
| `None` | Null | `None` |
| `bool` | Bool | `bool` |
| integer fitting i64 | Int | `int` |
| floating number | Float (f32) | `float` |
| `str` | Text | `str` |
| `bytes`, `bytearray` | Bytes | `bytes` |
| homogeneous boolean/integer/text sequence | BoolVec / IntVec / TextVec | `list` |
| numeric sequence containing floats | FloatVec (f32) | `list` |
| dictionary or heterogeneous JSON-compatible sequence | Json | decoded dictionary/list |

Booleans remain distinct from integers. Integers outside i64 raise rather than silently becoming floats. Floating properties retain the existing f32 precision; Python f64 values may round. Nested structured JSON numbers use JSON's representation rather than the native vector representation.

An empty sequence defaults to FloatVec. Use explicit constructors when type matters:

```python
from groggy import AttrValue, Graph

g = Graph()
node = g.add_node(
    flags=AttrValue.bool_vec([]),
    counts=AttrValue.int_vec([]),
    names=AttrValue.text_vec([]),
    embedding=AttrValue.float_vec([1, 2]),
    payload={"nested": [True, None, "source"]},
    binary=b"data",
    explicit_null=None,
)
state = g.commit("typed properties", "example")
snapshot = g.snapshot_at_commit(state)
assert snapshot.node_attr(node, "flags").type_name == "bool_vec"
assert snapshot.node_attr(node, "explicit_null").value is None
assert snapshot.node_attr(node, "missing") is None
assert g.nodes["payload"][0] == {"nested": [True, None, "source"]}
```

`AttrValue` wrappers can be passed directly to writers to preserve their exact stored type. Returned Python containers and binary values do not provide mutable access to graph storage.

## Structure and history

Structured dictionaries require string keys recursively. Cycles, more than 128 nesting levels, unsupported Python objects and non-finite JSON numbers raise conversion errors. Tuples normalize to lists. Dictionary keys are serialized in canonical order. Native float scalars/vectors may contain NaNs or infinities; the JSON restriction applies to structured JSON values.

Recorded snapshots retain typed property changes, including transitions between types. A present Null differs from an absent property in snapshot reads. Ordinary getters and attribute columns can project both to Python `None`; they do not alone establish presence. Current ordinary subgraphs retain live parent property reads; immutable state-pinned views remain roadmap item 26.

## Value identity and storage

Core equality, hashing and total ordering agree across compact integers, compact/compressed text and compressed float vectors. Collections compare by contents. Property identity treats all NaNs as equal and signed zero as equal; numerical arithmetic still uses IEEE behavior. Numeric-looking text retains numeric sort order, with literal spelling as a tie breaker; numeric and other text form separate ordered groups.

Compressed data decodes to its actual value. Invalid compression metadata raises on Python conversion rather than returning a placeholder. Float compression stores explicit little-endian bytes and preserves bit patterns.

This repair does not complete the property model. Native temporal values and a lossless typed disk codec remain roadmap work. Existing JSON graph export is narrower than the live property model and must not be used as a full-fidelity backup.

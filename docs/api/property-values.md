# Property values

Graph writers and `groggy.AttrValue` share one conversion policy. Graph getters, attribute columns, table cells and recorded snapshots decode supported values consistently.

| Python input | Logical property type | Python result |
| --- | --- | --- |
| `None` | Null | `None` |
| `bool` | Bool | `bool` |
| integer fitting i64 | Int | `int` |
| ordinary floating number | Float64 (f64) | `float` |
| explicit f32 / NumPy float32 | Float (f32) | `float` |
| `str` | Text | `str` |
| `bytes`, `bytearray` | Bytes | `bytes` |
| homogeneous boolean/integer/text sequence | BoolVec / IntVec / TextVec | `list` |
| numeric sequence containing ordinary floats | Float64Vec (f64) | `list` |
| aware `datetime` / `date` / `timedelta` | UtcInstant / Date / Duration | UTC `datetime` / `date` / `timedelta` |
| dictionary or heterogeneous JSON-compatible sequence | Json | decoded dictionary/list |

Booleans remain distinct from integers. Integers outside i64 raise rather than silently becoming floats. Ordinary floating properties retain f64 precision; explicitly typed f32 values remain f32. Nested structured JSON numbers use JSON's representation rather than the native vector representation.

Retained state restoration and storage keep signed-zero and NaN payload bits.
This is stricter than public property equality, which groups those representations
for identity. It does not change numerical IEEE comparison behavior.

An empty ordinary sequence defaults to Float64Vec. Use explicit constructors when type matters:

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

## Typed active graph export

`g.save_to_path(path)` writes a versioned active graph export; `Graph.load_from_path(path)` reads it. Both Python strings and path objects are accepted. Version 2 (`format: "groggy.graph"`) uses explicit logical type tags for all inline values above, including null, empty typed vectors, binary data, structured JSON and entity-ID arrays. Float scalars/vectors use unsigned bit patterns, preserving NaNs, infinities and signed zero. Optimized in-memory forms decode to their logical type; allocation/compression layout is not part of the file contract. Attribute names and entity IDs are emitted in stable order.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from groggy import Graph

with TemporaryDirectory() as directory:
    path = Path(directory) / "graph.json"
    graph = Graph()
    node = graph.add_node(binary=b"data", details={"source": [1, None]})
    graph.save_to_path(path)
    restored = Graph.load_from_path(path)
    assert restored.get_node_attr(node, "binary") == b"data"
    assert restored.get_node_attr(node, "details") == {"source": [1, None]}
```

Unversioned legacy scalar exports remain readable. Untagged legacy arrays/objects become structured JSON because the old file cannot distinguish vectors, bytes or entity-ID arrays. Invalid records, property tags, payloads, versions, duplicate entity IDs and missing edge endpoints raise errors. Structured JSON payloads must be valid JSON. Old exports cannot recover values that were omitted when originally saved.

Meta-nodes and `SubgraphRef` properties currently reject export because this format does not contain their backing pool contents. Export validation and encoding complete before the destination is opened, so a validation error preserves an existing file. File-write failures can still leave a partial file; atomic replacement and durable synchronization belong to roadmap item 10.

This format contains active topology, typed properties, active external identities and graph direction. It does **not** back up branches, history, inactive entities, old external-ID reservations, discarded allocation counters, subgraph pool contents, indexes or other graph configuration. Entity-ID arrays retain numeric values, not inactive referents. Default Python loading accepts an independent graph as one newly published import. `manual=True` retains deferred import compatibility. Checked journals retain supported history and frozen group sources. Explicit owned snapshots transfer the raw backing pool/history in memory; persistent arbitrary pool import and bounded storage lifecycle remain item 10. JSON here is active interchange, not a full engine backup.

Native temporal types are implemented; see [temporal properties](temporal-properties.md) for exact domains, timezone normalization and versioned payloads.

## Current read integrity

The native Graph scalar, row, requested-row and aligned-column readers share one
implementation in `api::graph::read_attributes`. A current index pointing at a
missing storage column or an out-of-range retained value returns `InvalidInput`,
including entity kind/ID, attribute and index. It cannot silently become absence.
Requested projections validate only their requested rows/keys. All-property row
reads validate every current binding in that row.

Scalar/row readers require active IDs. The existing aligned-column methods
`get_nodes_attrs`/`get_edges_attrs` retain None for inactive/unknown IDs, request
order and repeated IDs. Bulk row maps retain their existing ID/key map semantics.
An absent key stays omitted; a present native Null stays a property. No read
creates a state or changes history. Thin Python adapters keep their existing
projection/exception behavior; raw native storage escape hatches remain outside
the checked mutation API.

```python
from groggy import Graph
checked = Graph()
id = checked.add_node(present=None)
state = checked.current_state_id
frozen = checked.read_view()
assert checked.get_node_attr(id, "present") is None
assert checked.get_node_attr(id, "absent") is None
assert checked.snapshot_at_commit(state).node_attr(id, "present").type_name == "null"
assert checked.snapshot_at_commit(state).node_attr(id, "absent") is None
checked.set_node_attr(id, "present", 7)
assert frozen.node_attr(id, "present") is None
assert checked.current_state_id == state + 1
```

The twelve native entry points and their corruption/liveness/null/absence tests
are recorded in [the reader consolidation note](../../notes/planning/current-property-reads-2026-10-25.md).

# Native connected column alignment

Default selections now retain their source; see [immutable defaults](default-immutable-views.md).
The explicit connected examples below preserve the current-owner compatibility contract.

The native Rust Subgraph compatibility readers now retain every requested row:

- `node_ids()` and `edge_ids()` return ascending original IDs, reusing cached order.
- `get_node_attribute_column(name)` returns a typed value for each ascending node ID.
- `get_edge_attribute_column(name)` returns a typed value for each ascending edge ID.
- `get_node_attributes_for_nodes(ids, name)` follows the caller's exact sequence,
  including reversed order and repeated IDs.

Absent properties project to AttrValue::Null; explicit stored null remains null.
This fills an output cell only—it does not create a graph property or delta.
Use scalar presence/read APIs or the captured accessor presence column to
separate absence from explicit null. Empty inputs return empty vectors.

The previous helpers iterated HashSets and skipped absent cells, so vector length
and row identity could diverge. They now delegate to the existing attribute-first
`Graph.get_nodes_attrs`/`get_edges_attrs` bulk readers, resolving the storage column
once rather than calling a scalar getter for each entity. All requested IDs are
checked for membership/liveness before values are copied. Out-of-scope, unknown,
inactive and corrupt-index requests produce errors, not missing output rows.

These methods remain live reads of the connected owner. An explicit GraphView
keeps captured values through owner updates; no graph/property/history capture or
new mutation path is introduced here. The initial native helper repairs left Python adapters unchanged; the Python
whole-selection migration below now shares their ascending ID order and checked
column reads. Other accessor-specific absent-column policies remain unchanged.
The captured accessor path has its own documented ID-based
contract in [captured entities](captured-view-entities.md).

Compilable native examples/fixtures are maintained in
`tests/connected_column_alignment.rs`: sparse IDs, typed f64/vector/large integer
values, null/absent cells, explicit repeated order, rejected requests, empty
columns and the live/captured update boundary. Release costs are measured in
`benches/connected_column_alignment.rs` against equivalent aligned scalar reads.
ID ordering/setup is excluded from the timed bulk comparison; first ordering
still costs a sort. No scalar FFI or whole-engine throughput guarantee is implied.

## Python whole-selection columns

`Subgraph["property"]` now delegates directly to that checked Rust bulk reader.
It returns one BaseArray cell per ascending selected node ID. Read failures are
reported instead of being disguised as nulls. The `node_ids`/`edge_ids` properties,
`.nodes`/`.edges` and `to_nodes()`/`to_edges()` adapters now use the same native
ascending order. They continue to read the connected owner; no state is silently
captured or refreshed. Existing accessor-specific column policies remain separate.

Immutable `GraphView["property"]` uses its frozen node accessor's typed column.
`to_nodes()` and `to_edges()` return the same readonly captured entity adapters as
`.nodes` and `.edges`. Frozen column reads release the GIL and retain their original
source selection; no writable graph owner is stored in the view.

```python
from groggy import Graph

graph = Graph()
graph.add_nodes([{"score": 0.123456789012345}, {}, {"score": None}, {"score": 7}])
connected = graph.connected_view()
frozen = connected.pin_view()
assert connected.node_ids.to_list() == frozen.node_ids == [0, 1, 2, 3]
assert connected["score"].to_list() == frozen["score"].to_list() == [0.123456789012345, None, None, 7]
assert connected["unknown"].to_list() == [None] * 4
assert frozen.to_nodes().ids.to_list() == [0, 1, 2, 3]
graph.remove_node(0)
assert frozen["score"].to_list()[0] == 0.123456789012345
try:
    connected["score"]
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("Inactive selected IDs cannot become null-valued rows")
```

Scalar/column absence remains distinct from invalid identity. These columns still
project absence and explicit null to the same null cell; use captured `presence`
to distinguish them. Returned buffers are owned copies and do not write through.
This completes whole-selection column delegation, not the remaining live default
or effectful pipeline migration.

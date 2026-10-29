# Native connected column alignment

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
new mutation path is introduced here. Python's existing connected accessor order,
indexing and absent-column compatibility policies are unchanged by these native
helper repairs. The new captured accessor path has its own documented ID-based
contract in [captured entities](captured-view-entities.md).

Compilable native examples/fixtures are maintained in
`tests/connected_column_alignment.rs`: sparse IDs, typed f64/vector/large integer
values, null/absent cells, explicit repeated order, rejected requests, empty
columns and the live/captured update boundary. Release costs are measured in
`benches/connected_column_alignment.rs` against equivalent aligned scalar reads.
ID ordering/setup is excluded from the timed bulk comparison; first ordering
still costs a sort. No scalar FFI or whole-engine throughput guarantee is implied.

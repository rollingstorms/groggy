# Checked BaseTable entity conversions

`BaseTable.to_nodes(uid_key="node_id")` and `BaseTable.to_edges()` return owned
table components. They create no graph, allocate no entity IDs and publish no
history. Column value buffers retain their existing sharing behavior; table
schema/column handles are independent.

```python
from groggy import BaseTable
rows = BaseTable.from_dict({"uid": [70, 12], "score": [1.23456789012345, None]})
original_columns = rows.columns
nodes = rows.to_nodes("uid")
assert nodes.node_ids().to_list() == [70, 12]
assert nodes.base_table().columns == [
    "node_id" if name == "uid" else name for name in original_columns
]
assert nodes.base_table()["score"].to_list() == [1.23456789012345, None]
assert rows.columns == original_columns
```

The explicit node key is renamed to `node_id` in place in column order. Missing
keys and collisions with an existing `node_id` reject. Node IDs, edge IDs and
edge endpoints must be nonnegative integers; optimized SmallInt cells are
accepted. Node/edge identities must be unique. Text external IDs are properties,
not internal table IDs. Validation scans complete identity columns before
returning the result.

Edge conversion uses the existing EdgesTable aliases (`edge_ids`, `src`, `dst`,
and the other aliases accepted by `from_base_table`). Parallel endpoint pairs
and self loops are allowed. Without a node component there is no claim that an
outside graph contains those endpoints. Nullable edge IDs must be explicitly
assigned before checked conversion.

```python
edge_rows = BaseTable.from_dict({
    "edge_ids": [91, 92, 93], "src": [70, 70, 12], "dst": [12, 12, 12],
})
edges = edge_rows.to_edges()
assert edges.as_tuples() == [(91, 70, 12), (92, 70, 12), (93, 12, 12)]
bad = BaseTable.from_dict({"node_id": [7, 7]})
try:
    bad.to_nodes()
except ValueError as error:
    assert "Duplicate" in str(error)
else:
    raise AssertionError("duplicate node identity accepted")
```

## Retained source identity

A source-bound table retains automatic writeback only when the conversion still
describes the same captured entity kind and IDs. Edges must also retain the
captured source/target endpoints. Compatible renames preserve the binding.
Changing identity/kind/endpoints retains the source captures as read-only
context, requiring explicit identity resolution before source writeback. No
conversion invents graph ownership from arbitrary numeric rows.

```python
from groggy import Graph
graph = Graph()
graph.add_nodes([{"score": 1}, {"score": 2}])
graph.add_edge(0, 1)
original = graph.nodes.table().base_table()
same = original.rename({"node_id": "uid"}).to_nodes("uid").base_table()
assert same.source_entity_ids == [0, 1]
same["score"] = [8, 9]
same.update_source_graph(graph, ["score"], effective_at=123)
assert graph.get_node_attr(0, "score") == 8
assert original["score"].to_list() == [1, 2]
endpoint_rows = graph.edges.table().base_table().select(["source"])
different_kind = endpoint_rows.to_nodes("source").base_table()
assert different_kind.source_entity_ids is None
assert len(different_kind.source_views()) == 1
```

`NodesTable.from_base_table` / `EdgesTable.from_base_table` remain permissive
structural wrappers for inspecting/repairing imported rows. The explicit
BaseTable converters above are checked boundaries. These are separate from the
still-unsupported GraphTable converters that promise graph accessors or
SubgraphArray ownership; those contracts remain open in 27.1.

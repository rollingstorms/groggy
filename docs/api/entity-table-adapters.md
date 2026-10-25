# Entity table IDs and attribute columns

`NodesTable.with_attributes(name, attributes)` and
`EdgesTable.with_attributes(name, attributes)` produce a new table with a complete
ID-mapped property column. Inputs are either an integer-ID dictionary or a list
of `{"id": ..., "value": ...}` records. Physical row order is preserved;
unspecified rows become null, including replacement of an existing column.
Unknown IDs, repeated record IDs or ambiguous duplicate table IDs reject before
producing output. Python bool/float keys do not stand in for integer IDs.

```python
from groggy import BaseTable, NodesTable, EdgesTable
nodes = NodesTable.from_base_table(BaseTable.from_dict({"node_id": [7, 3]}))
changed = nodes.with_attributes("x", {3: 1.0000000000000002})
assert changed.base_table()["x"].to_list() == [None, 1.0000000000000002]
assert "x" not in nodes.base_table().columns
edges = EdgesTable.from_base_table(BaseTable.from_dict({
    "edge_id": [9, 5], "source": [7, 3], "target": [3, 7],
}))
changed_edges = edges.with_attributes("labels", [
    {"id": 9, "value": ["a", "b"]}, {"id": 5, "value": None}
])
assert changed_edges.base_table()["labels"].to_list() == [["a", "b"], None]
```

Native structural readers share checked nonnegative integer extraction. Small
and full integer representations are accepted; malformed, negative, fractional
or null IDs reject rather than independently shortening the ID/source/target
arrays. `auto_assign_edge_ids()` remains the explicit way to replace null edge
IDs. Option-returning native lookup/iteration methods retain their ability to
skip invalid rows, but now recognize small integers and never wrap negative IDs.

Python edge `edge_ids()`, `sources()` and `targets()` now return NumArrays with
**int64** values instead of float64 values, matching the existing node ID policy.
This is a dtype correction: callers relying on float arrays should explicitly
convert when needed. Native `as_tuples()` validates aligned columns once and
zips them in physical order; it no longer searches row-index lists repeatedly.
Malformed tuple rows now reject with their structural column/row.

```python
wide = 2**53 + 1
wide_edges = EdgesTable.from_base_table(BaseTable.from_dict({
    "edge_id": [wide], "source": [wide + 2], "target": [wide + 4],
}))
assert wide_edges.edge_ids().to_list() == [wide]
assert wide_edges.sources().to_list() == [wide + 2]
assert wide_edges.as_tuples() == [(wide, wide + 2, wide + 4)]
```

Attribute editing and endpoint filters retain source captures and direct row
identity. These operations edit owned table values; graph publication remains an
explicit checked update. Native preparation handles ID membership, null filling
and duplicates; FFI only marshals input and releases the GIL during the operation.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"x": 1}, {"x": 2}])
pinned = owner.read_view().table().nodes
edited = pinned.with_attributes("x", {1: 9}).base_table()
assert edited.source_entity_ids == [0, 1]
assert owner.get_node_attr(1, "x") == 2
edited.update_source_graph(owner, ["x"])
assert owner.get_node_attr(1, "x") == 9
assert pinned.base_table()["x"].to_list() == [1, 2]
```

Native ID-mapped columns validate and construct in O(rows + input entries).
Tuple extraction uses O(rows) work. This does not make owned column replacement
an O(1) mutation or certify the per-call FFI budget. Current measurements are in
`notes/benchmarks/entity-table-adapters-2026-10-25.json`.

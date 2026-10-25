# Table schema invariants

A BaseTable's physical columns and ordered schema must agree. The native
`with_column_order(columns, order)` constructor now requires every physical
column exactly once: duplicate, missing and unknown names reject. Declared
columns must have equal lengths, including zero-row tables. The separate
`from_columns` constructor keeps its existing explicit sparse null-padding rule.

Column selection through BaseTable, NodesTable, EdgesTable and GraphTable now
rejects repeated or unknown requested names. Specialized selections still retain
their required structural columns; an omitted structural column is different
from asking for the same structural column twice.

```python
from groggy import BaseTable, NodesTable
source = BaseTable.from_dict({"x": [1, 2], "y": [3, 4]})
assert source.select(["y", "x"]).columns == ["y", "x"]
try:
    source.select(["x", "x"])
except RuntimeError:
    pass
else:
    raise AssertionError("duplicate schema names were accepted")
assert source.columns == ["x", "y"]
```

Native `with_column` may initialize a truly empty 0×0 table from its first column.
Once columns are declared, even when there are no rows, adding/replacing one
column must match that row count. A zero-column projection with N rows retains N;
adding a column to it requires N values. This prevents changing only one field
and leaving the other arrays with different lengths.

```python
sparse = BaseTable.from_dict({"x": [1, 2], "y": [3]})
assert sparse["y"].to_list() == [3, None]
assert sparse.select([]).shape == (2, 0)
```

Validation happens before output construction. Rejected schema edits do not
change source tables, graph history or captured row identity.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"x": 1}, {"x": 2}])
table = owner.nodes.table().base_table()
assert table.select([]).source_entity_ids == [0, 1]
try:
    table.select(["node_id", "node_id"])
except RuntimeError:
    pass
else:
    raise AssertionError("duplicate IDs in schema were accepted")
assert table.source_entity_ids == [0, 1]
assert owner.get_node_attr(1, "x") == 2
```

These are schema validation corrections. They add no property coercion, implicit
row growth, graph selection policy or new serialization format. Checks are
linear in the requested schema width; no throughput or fixed FFI latency claim
is made.

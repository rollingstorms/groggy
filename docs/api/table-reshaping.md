# Typed entity filters and table reshaping

Node and edge `filter_by_attr(name, value)` methods compare logical property
values directly. Text is literal, including quotes, backslashes and query syntax;
column names need no query escaping. Vectors and nulls are compared as their
actual types. Numeric types follow property equality, including exact f64 and
small/full integer equivalence. This is separate from query numeric coercion.

```python
from groggy import BaseTable, NodesTable
nodes = NodesTable.from_base_table(BaseTable.from_dict({
    "node_id": [8, 3], "odd column": ['a"b', None],
}))
assert nodes.filter_by_attr("odd column", 'a"b').node_ids().to_list() == [8]
assert nodes.filter_by_attr("odd column", None).node_ids().to_list() == [3]
```

Typed filters retain the same source binding as any direct row selection. Both
FFI adapters marshal values and release the GIL for native filtering.

Generic pivots retain group order by first key occurrence, and reject repeated
index columns. Melt rejects repeated ID/value columns, overlap between those
sets, and output names that collide with each other or the ID columns. These
checks also apply to empty tables. Python melt now accepts the native identifier
selection through the keyword-only `id_vars` argument; existing positional
`melt(value_vars, var_name, value_name)` calls keep their meaning. Melt row-count multiplication is checked.

```python
table = BaseTable.from_dict({"g": [20, 10, 20], "key": ["x"] * 3, "v": [2, 4, 8]})
pivot = table.pivot_table(["g"], "key", "v", "sum")
assert pivot["g"].to_list() == [20, 10]
assert pivot["key_x"].to_list() == [10.0, 4.0]
```

Native `NodesTable`/`EdgesTable` implement the same reshape trait. Their results
must retain all structural columns as nonnegative integer IDs/endpoints, with a
unique entity ID per row. A reshape that drops IDs or multiplies one entity into
multiple rows rejects with guidance to use `base_table()` for a general table.
This preserves the public return types without inventing new entity IDs. Generic
BaseTable melt/pivot remain available for arbitrary analytical layouts.

```python
long_table = BaseTable.from_dict({"node_id": [8], "x": [2], "y": [3]}).melt(
    ["x", "y"], id_vars=["node_id"]
)
assert long_table["node_id"].to_list() == [8, 8]
assert long_table["value"].to_list() == [2, 3]
```

Analytical reshape outputs retain immutable source captures for inspection but
have no automatic source-row writeback identity. An aggregate or duplicated long
row does not assert a new graph entity. Input tables and graph history stay
unchanged.

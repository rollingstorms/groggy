# Table collection reductions

Table collection reductions execute in Rust. Both `table.group_by(keys)` and
`table.groupby(keys)` share the same reducer, numeric kernels and group-key
checks. Node and edge group adapters use it too. Python releases the GIL during
native reduction work.

```python
from groggy import BaseTable
source = BaseTable.from_dict({"kind": ["b", "a", "b"], "value": [1, 4, 2], "flag": [False, True, True]})
for groups in [source.group_by("kind"), source.groupby("kind")]:
    assert groups.sum()["value"].to_list() == [3, 4]
    assert groups.mean()["value"].to_list() == [1.5, 4.0]
    assert groups.count()["value"].to_list() == [2, 1]
    assert groups.all()["flag"].to_list() == [False, True]
    assert groups.agg({"value": "nunique"})["value"].to_list() == [2, 1]
```

## Result rows and columns

Grouped collections produce one row per table, in collection order. Group keys
appear once, sorted by name. Convenience methods preserve first-occurrence input
column order for the remaining eligible columns. Explicit `agg` specs sort their
output columns by name. Selecting a grouping key for reduction rejects a name
collision instead of doubling or overwriting key cells.

Native `TableArray::from_tables` and the core Python collection constructor reduce
an ungrouped collection into one row across all observations. Means use actual
observation counts, rather than averaging table means. The legacy Python
collection constructor retains its existing **one row per table** behavior; its
`columns=` argument is a projection, not a claim that every selected column is
a constant group key. The two collection wrapper classes remain distinct.

```python
CoreCollection = type(source.groupby("kind"))
collection = CoreCollection([
    BaseTable.from_dict({"x": [1]}),
    BaseTable.from_dict({"x": [3, 5, 7]}),
])
assert collection.mean()["x"] == 4.0  # one-row BaseTable indexing returns a scalar
assert collection.sum()["x"] == 16
assert collection.count()["x"] == 4
```

## Supported reductions

| Operation | Observations and result |
| --- | --- |
| `sum` | Non-null numeric scalars; exact integer accumulation, checked i64 output or finite f64 when floats occur. |
| `mean` / `avg` | Non-null numeric scalars; checked f64 output. |
| `min` / `max` | Non-null numeric scalars; exact mixed integer/float ordering, first original scalar on ties. |
| `count` | Number of non-null cells, checked i64 output. |
| `nunique` (explicit `agg`) | Distinct non-null properties using logical property identity, checked i64 output. |
| `all` | Non-null booleans; validates every cell even after false. |

Default numeric convenience methods exclude group keys and pure text/boolean
columns. A column containing any numeric scalar is eligible and then must contain
only numeric/null cells: mixed text/numeric data rejects rather than silently
skipping observations. Boolean columns use `all`; boolean properties are a
separate domain from numeric properties. `count` includes every non-key column.
Explicit numeric specs are strict too. Nonfinite numeric values and overflow
reject. Logical unique counts can count nonfinite properties by property identity.

All-null or empty numeric/boolean groups return Null; counts and unique counts
return zero. An empty collection has zero result rows, and explicit specs retain
empty output columns. It has no retained source tables or inferred group schema.
A zero-column nonempty collection still retains the correct result row count.

## Group metadata and missing columns

Group-key count must equal table count, and all groups must have the same key
names. If a key column is present in a table, every row must agree with its stored
key using property identity. Conflicting/stale metadata rejects before producing
output. Group keys may remain as metadata after projecting their physical columns
away. An explicitly constructed empty table group uses null keys when no key
observation exists; existing retained keys survive an empty row selection.

Explicit aggregation columns absent everywhere reject for a nonempty collection.
Columns present in only some tables contribute null observations elsewhere.
Unknown functions reject even for empty inputs. Group reductions retain all
source captures of the supplied tables, with no automatic output row writeback
identity. Original views stay immutable.

```python
from groggy import Graph
owner = Graph()
owner.add_nodes([{"kind": 1, "value": 2}, {"kind": 1, "value": 3}])
table = owner.read_view().nodes_table().base_table()
result = table.groupby("kind").agg({"value": "sum"})
assert result["value"] == 5
assert result.source_entity_ids is None
assert len(result.source_views()) == 1
owner.set_node_attr(0, "value", 99)
assert result.source_views()[0].node_attr(0, "value") == 2
```

## Flattening and costs

Native/core collection `concat()` flattens rows using the union of column names,
in first-input occurrence order, with null padding. Later-only columns are
preserved. It retains all source captures but infers no row writeback identity.
This vertical flatten is distinct from BaseTable's positional horizontal concat.

Numeric reductions stream existing columns through shared checked array kernels;
there is no flattening copy for native sum/mean/extrema. Counts/metadata validation
scan cells; unique counts allocate a set. Results and flatten outputs own columns,
and source captures retain their payloads. The legacy Python bridge currently
clones input BaseTables while preparing a native collection, so it has a copy
cost beyond native kernel measurements. No streaming-output, memory-budget,
100ns whole-reduction or large-scale latency guarantee is established.

Specialized graph-domain merges, collection ownership across zero-table
transformations and broader conversion adapters remain open under 27/AR7.10.
This slice certifies reduction/flatten outputs for their supplied input tables.

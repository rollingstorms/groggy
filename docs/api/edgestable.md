# EdgesTable API Reference

**Type**: `groggy.EdgesTable`

---

## Overview

Tabular view of edge data with columns for edge attributes.

**Primary Use Cases:**
- Analyzing edge attributes in tabular form
- Aggregating edge data
- Exporting edge information

**Related Objects:**
- `GraphTable`
- `BaseTable`
- `EdgesAccessor`

---

## Complete Method Reference

The following methods are available on `EdgesTable` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `as_tuples()` | `list` | ✓ |
| `auto_assign_edge_ids()` | `EdgesTable` | ✓ |
| `base_table()` | `BaseTable` | ✓ |
| `drop_columns()` | `EdgesTable` | ✓ |
| `edge_ids()` | `NumArray` | ✓ |
| `filter()` | `?` | ✗ |
| `filter_by_attr()` | `?` | ✗ |
| `filter_by_sources()` | `?` | ✗ |
| `filter_by_targets()` | `?` | ✗ |
| `from_csv()` | `?` | ✗ |
| `from_dict()` | `?` | ✗ |
| `from_json()` | `?` | ✗ |
| `from_parquet()` | `?` | ✗ |
| `group_by()` | `EdgesTableArray` | ✓ |
| `head()` | `EdgesTable` | ✓ |
| `interactive()` | `str` | ✓ |
| `interactive_embed()` | `str` | ✓ |
| `interactive_viz()` | `VizAccessor` | ✓ |
| `into_base_table()` | `BaseTable` | ✓ |
| `iter()` | `EdgesTableRowIterator` | ✓ |
| `ncols()` | `int` | ✓ |
| `nrows()` | `int` | ✓ |
| `rich_display()` | `str` | ✓ |
| `select()` | `EdgesTable` | ✓ |
| `shape()` | `tuple` | ✓ |
| `slice()` | `?` | ✗ |
| `sort_by()` | `EdgesTable` | ✓ |
| `sort_values()` | `EdgesTable` | ✓ |
| `sources()` | `NumArray` | ✓ |
| `tail()` | `EdgesTable` | ✓ |
| `targets()` | `NumArray` | ✓ |
| `to_csv()` | `?` | ✗ |
| `to_json()` | `?` | ✗ |
| `to_pandas()` | `DataFrame` | ✓ |
| `to_parquet()` | `?` | ✗ |
| `unique_attr_values()` | `?` | ✗ |
| `viz()` | `VizAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating EdgesTable

EdgesTable is accessed from GraphTable or directly from graph/subgraph:

```python
import groggy as gr

g = gr.generators.karate_club()

# From graph
edges_table = g.edges.table()

# From GraphTable
table = g.table()
edges_table = table.edges

# From subgraph
strong = g.edges[g.edges["weight"] > 5.0]
strong_edges = strong.table().edges
```

---

### Core Methods

#### `edge_ids()`

Get array of edge IDs.

**Returns:**
- `NumArray`: Edge IDs

**Example:**
```python
edges_table = g.edges.table()
ids = edges_table.edge_ids()
print(ids.to_list())
```

---

#### `sources()`

Get source node IDs for all edges.

**Returns:**
- `NumArray`: Source node IDs

**Example:**
```python
edges_table = g.edges.table()
srcs = edges_table.sources()
print(srcs.to_list())
```

**Notes:**
- For directed graphs: actual source nodes
- For undirected: arbitrary endpoint designation

---

#### `targets()`

Get target node IDs for all edges.

**Returns:**
- `NumArray`: Target node IDs

**Example:**
```python
edges_table = g.edges.table()
tgts = edges_table.targets()

# Build edge list
import pandas as pd
edge_list = pd.DataFrame({
    'source': edges_table.sources().to_list(),
    'target': edges_table.targets().to_list()
})
```

---

#### `shape()`

Get table dimensions.

**Returns:**
- `tuple[int, int]`: (rows, columns)

**Example:**
```python
edges_table = g.edges.table()
rows, cols = edges_table.shape()
print(f"{rows} edges, {cols} attributes")
```

---

#### `nrows()` / `ncols()`

Get number of rows or columns.

**Returns:**
- `int`: Row or column count

**Example:**
```python
print(f"{edges_table.nrows()} edges")
print(f"{edges_table.ncols()} columns")
```

---

#### `as_tuples()`

Get edges as list of tuples.

**Returns:**
- `list[tuple]`: List of (source, target) tuples

**Example:**
```python
edges_table = g.edges.table()
edge_list = edges_table.as_tuples()
print(edge_list[:5])  # [(0, 1), (0, 2), (1, 3), ...]
```

**Performance:** O(m) where m is edge count

---

### Display & Inspection

#### `head(n=5)` / `tail(n=5)`

Show first/last n rows.

**Returns:**
- `EdgesTable`: Subset table

**Example:**
```python
edges_table.head()     # First 5
edges_table.head(10)   # First 10
edges_table.tail(5)    # Last 5
```

---

#### `iter()`

Iterate over rows.

**Returns:**
- Iterator over rows

**Example:**
```python
for row in edges_table.iter():
    print(row)
```

---

### Selection & Filtering

#### `select(columns)`

Select specific columns.

**Parameters:**
- `columns` (list[str]): Column names to keep

**Returns:**
- `EdgesTable`: Table with selected columns

**Example:**
```python
# Select subset of attributes
weight_type = edges_table.select(["weight", "type"])
print(weight_type.head())

# Just topology
topology = edges_table.select(["source", "target"])
```

---

#### `drop_columns(columns)`

Remove specific columns.

**Parameters:**
- `columns` (list[str]): Columns to drop

**Returns:**
- `EdgesTable`: Table without specified columns

**Example:**
```python
# Remove metadata
clean = edges_table.drop_columns(["timestamp", "internal_id"])
```

---

### Sorting

#### `sort_by(column)` / `sort_values(column)`

Sort by column values.

**Parameters:**
- `column` (str): Column name to sort by

**Returns:**
- `EdgesTable`: Sorted table

**Example:**
```python
# Sort by weight
by_weight = edges_table.sort_by("weight")
print(by_weight.head())

# Heaviest edges first (if descending supported)
# by_weight_desc = edges_table.sort_by("weight", ascending=False)
```

**Notes:** Both methods are aliases

---

### Grouping

#### `group_by(column)`

Group rows by column value.

**Parameters:**
- `column` (str): Column to group by

**Returns:**
- `EdgesTableArray`: Array of grouped tables

**Example:**
```python
# Group by edge type
by_type = edges_table.group_by("type")

for type_table in by_type:
    print(f"Type: {type_table.nrows()} edges")

# Group by weight threshold
# (First create categorical column)
```

---

### Export

#### `to_pandas()`

Convert to pandas DataFrame.

**Returns:**
- `pandas.DataFrame`: DataFrame

**Example:**
```python
df = edges_table.to_pandas()
print(df.head())
print(df.describe())

# Edge weight distribution
import matplotlib.pyplot as plt
df['weight'].hist()
plt.show()
```

---

#### `to_csv(path)` / `to_parquet(path)` / `to_json(path)`

Export to file formats.

**Parameters:**
- `path` (str): File path

**Example:**
```python
edges_table.to_csv("edges.csv")
edges_table.to_parquet("edges.parquet")
edges_table.to_json("edges.json")
```

---

### Utility Methods

#### `auto_assign_edge_ids()`

Automatically assign IDs to edges if missing.

**Returns:**
- `EdgesTable`: Table with edge IDs assigned

**Example:**
```python
# If edges lack IDs
edges_with_ids = edges_table.auto_assign_edge_ids()
```

---

### Conversion

#### `base_table()` / `into_base_table()`

Convert to BaseTable.

**Returns:**
- `BaseTable`: Generic table

**Example:**
```python
base = edges_table.base_table()
```

**Notes:** Both methods are aliases

---

## Usage Patterns

### Pattern 1: Basic Edge Export

```python
edges_table = g.edges.table()

# To pandas
df = edges_table.to_pandas()
print(df.info())

# To CSV with edge list
edges_table.to_csv("edges.csv", index=False)
```

### Pattern 2: Edge List Construction

```python
edges_table = g.edges.table()

# Get edge list with attributes
edge_data = []
for src, tgt, eid in zip(
    edges_table.sources().to_list(),
    edges_table.targets().to_list(),
    edges_table.edge_ids().to_list()
):
    edge_data.append((src, tgt, eid))

# Or use as_tuples for basic edge list
edge_list = edges_table.as_tuples()
```

### Pattern 3: Weight Analysis

```python
# Sort by weight
by_weight = edges_table.sort_by("weight")

# Heaviest edges
print("Heaviest edges:")
heavy_df = by_weight.tail(10).to_pandas()
print(heavy_df[['source', 'target', 'weight']])

# Lightest edges
print("\nLightest edges:")
light_df = by_weight.head(10).to_pandas()
print(light_df[['source', 'target', 'weight']])
```

### Pattern 4: Edge Type Grouping

```python
# Group by type
by_type = edges_table.group_by("type")

# Analyze each type
for type_table in by_type:
    df = type_table.to_pandas()
    print(f"Edge type: {df['type'].iloc[0]}")
    print(f"  Count: {len(df)}")
    print(f"  Avg weight: {df['weight'].mean():.2f}")
    print(f"  Total weight: {df['weight'].sum():.2f}")
```

### Pattern 5: Network Flow Analysis

```python
import pandas as pd

# Get edge data
df = edges_table.to_pandas()

# Out-flow by source
out_flow = df.groupby('source')['weight'].agg(['sum', 'count', 'mean'])
print("Out-flow by node:")
print(out_flow.sort_values('sum', ascending=False).head())

# In-flow by target
in_flow = df.groupby('target')['weight'].agg(['sum', 'count', 'mean'])
print("\nIn-flow by node:")
print(in_flow.sort_values('sum', ascending=False).head())
```

### Pattern 6: Selective Export

```python
# Export only specific attributes
subset = edges_table.select(["source", "target", "weight"])
subset.to_csv("edge_list.csv")

# Remove metadata before export
public = edges_table.drop_columns(["internal_id", "timestamp"])
public.to_parquet("edges_public.parquet")
```

---

## Quick Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `edge_ids()` | `NumArray` | Get edge IDs |
| `sources()` | `NumArray` | Source node IDs |
| `targets()` | `NumArray` | Target node IDs |
| `as_tuples()` | `list` | (source, target) pairs |
| `shape()` | `tuple` | (rows, cols) |
| `nrows()` | `int` | Number of rows |
| `ncols()` | `int` | Number of columns |
| `head(n)` | `EdgesTable` | First n rows |
| `tail(n)` | `EdgesTable` | Last n rows |
| `select(cols)` | `EdgesTable` | Select columns |
| `drop_columns(cols)` | `EdgesTable` | Remove columns |
| `sort_by(col)` | `EdgesTable` | Sort by column |
| `group_by(col)` | `EdgesTableArray` | Group by column |
| `to_pandas()` | `DataFrame` | Export to pandas |
| `to_csv(path)` | None | Export to CSV |


---

## Object Transformations

`EdgesTable` can transform into:

- **EdgesTable → BaseArray**: `edges_table["column"]`
- **EdgesTable → DataFrame**: `edges_table.to_pandas()`
- **EdgesTable → AggregationResult**: `edges_table.agg({"weight": "sum"})`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/tables.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How EdgesTable works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains

## Additional Methods

#### `filter(predicate)`

Filter.

**Parameters:**
- `predicate`: predicate

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter(predicate=...)
```

---

#### `filter_by_attr(value)`

Filter By Attr.

**Parameters:**
- `value`: value

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter_by_attr(value=...)
```

---

#### `filter_by_sources(source_nodes)`

Filter By Sources.

**Parameters:**
- `source_nodes`: source nodes

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter_by_sources(source_nodes=...)
```

---

#### `filter_by_targets(target_nodes)`

Filter By Targets.

**Parameters:**
- `target_nodes`: target nodes

**Returns:**
- `None`: Return value

**Example:**
```python
obj.filter_by_targets(target_nodes=...)
```

---

#### `from_csv(path)`

From Csv.

**Parameters:**
- `path`: path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_csv(path=...)
```

---

#### `from_dict(data)`

From Dict.

**Parameters:**
- `data`: data

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_dict(data=...)
```

---

#### `from_json(path)`

From Json.

**Parameters:**
- `path`: path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_json(path=...)
```

---

#### `from_parquet(path)`

From Parquet.

**Parameters:**
- `path`: path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_parquet(path=...)
```

---

#### `interactive()`

Interactive.

**Returns:**
- `str`: Return value

**Example:**
```python
obj.interactive()
```

---

#### `interactive_embed()`

Interactive Embed.

**Returns:**
- `str`: Return value

**Example:**
```python
obj.interactive_embed()
```

---

#### `interactive_viz()`

Interactive Viz.

**Returns:**
- `VizAccessor`: Return value

**Example:**
```python
obj.interactive_viz()
```

---

#### `into_base_table()`

Into Base Table.

**Returns:**
- `BaseTable`: Return value

**Example:**
```python
obj.into_base_table()
```

---

#### `ncols()`

Ncols.

**Returns:**
- `int`: Return value

**Example:**
```python
obj.ncols()
```

---

#### `rich_display()`

Rich Display.

**Returns:**
- `str`: Return value

**Example:**
```python
obj.rich_display()
```

---

#### `slice(start, end)`

Slice.

**Parameters:**
- `start`: start
- `end`: end

**Returns:**
- `None`: Return value

**Example:**
```python
obj.slice(start=..., end=...)
```

---

#### `sort_values()`

Sort Values.

**Returns:**
- `EdgesTable`: Return value

**Example:**
```python
obj.sort_values()
```

---

#### `tail()`

Tail.

**Returns:**
- `EdgesTable`: Return value

**Example:**
```python
obj.tail()
```

---

#### `to_json(path)`

To Json.

**Parameters:**
- `path`: path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_json(path=...)
```

---

#### `to_parquet(path)`

To Parquet.

**Parameters:**
- `path`: path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_parquet(path=...)
```

---

#### `unique_attr_values()`

Unique Attr Values.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.unique_attr_values()
```

---

#### `viz()`

Viz.

**Returns:**
- `VizAccessor`: Return value

**Example:**
```python
obj.viz()
```

---


# NodesTable API Reference

**Type**: `groggy.NodesTable`

---

## Overview

Tabular view of node data with columns for node attributes.

**Primary Use Cases:**
- Analyzing node attributes in tabular form
- Aggregating node data
- Exporting node information

**Related Objects:**
- `GraphTable`
- `BaseTable`
- `NodesAccessor`

---

## Complete Method Reference

The following methods are available on `NodesTable` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `base_table()` | `BaseTable` | ✓ |
| `drop_columns()` | `NodesTable` | ✓ |
| `filter()` | `?` | ✗ |
| `filter_by_attr()` | `?` | ✗ |
| `from_csv()` | `?` | ✗ |
| `from_dict()` | `?` | ✗ |
| `from_json()` | `?` | ✗ |
| `from_parquet()` | `?` | ✗ |
| `group_by()` | `NodesTableArray` | ✓ |
| `head()` | `NodesTable` | ✓ |
| `interactive()` | `str` | ✓ |
| `interactive_embed()` | `str` | ✓ |
| `interactive_viz()` | `VizAccessor` | ✓ |
| `into_base_table()` | `BaseTable` | ✓ |
| `is_empty()` | `bool` | ✓ |
| `iter()` | `NodesTableRowIterator` | ✓ |
| `ncols()` | `int` | ✓ |
| `node_ids()` | `NumArray` | ✓ |
| `nrows()` | `int` | ✓ |
| `rich_display()` | `str` | ✓ |
| `select()` | `NodesTable` | ✓ |
| `shape()` | `tuple` | ✓ |
| `slice()` | `?` | ✗ |
| `sort_by()` | `NodesTable` | ✓ |
| `sort_values()` | `NodesTable` | ✓ |
| `tail()` | `NodesTable` | ✓ |
| `to_csv()` | `?` | ✗ |
| `to_json()` | `?` | ✗ |
| `to_pandas()` | `DataFrame` | ✓ |
| `to_parquet()` | `?` | ✗ |
| `unique_attr_values()` | `?` | ✗ |
| `viz()` | `VizAccessor` | ✓ |
| `with_attributes()` | `?` | ✗ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating NodesTable

NodesTable is accessed from GraphTable or directly from graph/subgraph:

```python
import groggy as gr

g = gr.generators.karate_club()

# From graph
nodes_table = g.nodes.table()

# From GraphTable
table = g.table()
nodes_table = table.nodes

# From subgraph
young = g.nodes[g.nodes["age"] < 30]
young_nodes = young.table().nodes
```

---

### Core Methods

#### `node_ids()`

Get array of node IDs.

**Returns:**
- `NumArray`: Node IDs

**Example:**
```python
nodes_table = g.nodes.table()
ids = nodes_table.node_ids()
print(ids.to_list())
```

---

#### `shape()`

Get table dimensions.

**Returns:**
- `tuple[int, int]`: (rows, columns)

**Example:**
```python
nodes_table = g.nodes.table()
rows, cols = nodes_table.shape()
print(f"{rows} nodes, {cols} attributes")
```

---

#### `nrows()` / `ncols()`

Get number of rows or columns.

**Returns:**
- `int`: Row or column count

**Example:**
```python
print(f"{nodes_table.nrows()} nodes")
print(f"{nodes_table.ncols()} columns")
```

---

#### `is_empty()`

Check if table is empty.

**Returns:**
- `bool`: True if no rows

**Example:**
```python
if nodes_table.is_empty():
    print("No nodes")
```

---

### Display & Inspection

#### `head(n=5)` / `tail(n=5)`

Show first/last n rows.

**Returns:**
- `NodesTable`: Subset table

**Example:**
```python
nodes_table.head()     # First 5
nodes_table.head(10)   # First 10
nodes_table.tail(5)    # Last 5
```

---

#### `iter()`

Iterate over rows.

**Returns:**
- Iterator over rows

**Example:**
```python
for row in nodes_table.iter():
    print(row)
```

---

### Selection & Filtering

#### `select(columns)`

Select specific columns.

**Parameters:**
- `columns` (list[str]): Column names to keep

**Returns:**
- `NodesTable`: Table with selected columns

**Example:**
```python
# Select subset of attributes
age_name = nodes_table.select(["age", "name"])
print(age_name.head())
```

---

#### `drop_columns(columns)`

Remove specific columns.

**Parameters:**
- `columns` (list[str]): Columns to drop

**Returns:**
- `NodesTable`: Table without specified columns

**Example:**
```python
# Remove sensitive attributes
public = nodes_table.drop_columns(["ssn", "password"])
```

---

### Sorting

#### `sort_by(column)` / `sort_values(column)`

Sort by column values.

**Parameters:**
- `column` (str): Column name to sort by

**Returns:**
- `NodesTable`: Sorted table

**Example:**
```python
# Sort by age
by_age = nodes_table.sort_by("age")
print(by_age.head())

# Descending (if supported)
# by_age_desc = nodes_table.sort_by("age", ascending=False)
```

**Notes:** Both methods are aliases

---

### Grouping

#### `group_by(column)`

Group rows by column value.

**Parameters:**
- `column` (str): Column to group by

**Returns:**
- `NodesTableArray`: Array of grouped tables

**Example:**
```python
# Group by city
by_city = nodes_table.group_by("city")

for city_table in by_city:
    print(f"City: {city_table.nrows()} nodes")
```

---

### Export

#### `to_pandas()`

Convert to pandas DataFrame.

**Returns:**
- `pandas.DataFrame`: DataFrame

**Example:**
```python
df = nodes_table.to_pandas()
print(df.head())
print(df.describe())
```

---

#### `to_csv(path)` / `to_parquet(path)` / `to_json(path)`

Export to file formats.

**Parameters:**
- `path` (str): File path

**Example:**
```python
nodes_table.to_csv("nodes.csv")
nodes_table.to_parquet("nodes.parquet")
nodes_table.to_json("nodes.json")
```

---

### Conversion

#### `base_table()` / `into_base_table()`

Convert to BaseTable.

**Returns:**
- `BaseTable`: Generic table

**Example:**
```python
base = nodes_table.base_table()
```

**Notes:** Both methods are aliases

---

## Usage Patterns

### Pattern 1: Basic Export

```python
nodes_table = g.nodes.table()

# To pandas
df = nodes_table.to_pandas()
print(df.info())

# To CSV
nodes_table.to_csv("nodes.csv", index=False)
```

### Pattern 2: Column Selection

```python
# Select specific attributes
subset = nodes_table.select(["id", "name", "age"])

# Export subset
subset.to_csv("nodes_basic.csv")
```

### Pattern 3: Sorting & Display

```python
# Sort by attribute
by_age = nodes_table.sort_by("age")

# Show oldest/youngest
print("Youngest:")
by_age.head(5)

print("\nOldest:")
by_age.tail(5)
```

### Pattern 4: Grouping Analysis

```python
# Group by category
by_dept = nodes_table.group_by("department")

# Analyze each group
for dept_table in by_dept:
    df = dept_table.to_pandas()
    print(f"Dept size: {len(df)}")
    print(f"Avg age: {df['age'].mean():.1f}")
```

---

## Quick Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `node_ids()` | `NumArray` | Get node IDs |
| `shape()` | `tuple` | (rows, cols) |
| `nrows()` | `int` | Number of rows |
| `ncols()` | `int` | Number of columns |
| `head(n)` | `NodesTable` | First n rows |
| `tail(n)` | `NodesTable` | Last n rows |
| `select(cols)` | `NodesTable` | Select columns |
| `drop_columns(cols)` | `NodesTable` | Remove columns |
| `sort_by(col)` | `NodesTable` | Sort by column |
| `group_by(col)` | `NodesTableArray` | Group by column |
| `to_pandas()` | `DataFrame` | Export to pandas |
| `to_csv(path)` | None | Export to CSV |

---

## See Also

- **[User Guide](../guide/tables.md)**: Comprehensive tutorial and patterns
- **[GraphTable API](graphtable.md)**: Parent container table
- **[EdgesTable API](edgestable.md)**: Edge data table
- **[NodesAccessor API](nodesaccessor.md)**: Dynamic node access

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
- `NodesTable`: Return value

**Example:**
```python
obj.sort_values()
```

---

#### `tail()`

Tail.

**Returns:**
- `NodesTable`: Return value

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

#### `with_attributes(attributes)`

With Attributes.

**Parameters:**
- `attributes`: attributes

**Returns:**
- `None`: Return value

**Example:**
```python
obj.with_attributes(attributes=...)
```

---


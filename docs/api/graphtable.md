# GraphTable API Reference

**Type**: `groggy.GraphTable`

---

## Overview

Tabular representation of graph data with separate nodes and edges tables.

**Primary Use Cases:**
- Exporting graph data to CSV/Parquet
- Converting to pandas DataFrames
- Tabular analysis of graph data

**Related Objects:**
- `Graph`
- `Subgraph`
- `NodesTable`
- `EdgesTable`

---

## Complete Method Reference

The following methods are available on `GraphTable` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `auto_assign_edge_ids()` | `GraphTable` | ✓ |
| `edges()` | `EdgesTable` | ✓ |
| `from_federated_bundles()` | `?` | ✗ |
| `get_bundle_info()` | `?` | ✗ |
| `head()` | `GraphTable` | ✓ |
| `is_empty()` | `bool` | ✓ |
| `load_bundle()` | `?` | ✗ |
| `merge()` | `?` | ✗ |
| `ncols()` | `int` | ✓ |
| `nodes()` | `NodesTable` | ✓ |
| `nrows()` | `int` | ✓ |
| `save_bundle()` | `?` | ✗ |
| `shape()` | `tuple` | ✓ |
| `stats()` | `dict` | ✓ |
| `tail()` | `GraphTable` | ✓ |
| `to_edges()` | `?` | ✗ |
| `to_graph()` | `Graph` | ✓ |
| `to_nodes()` | `?` | ✗ |
| `to_subgraphs()` | `?` | ✗ |
| `validate()` | `str` | ✓ |
| `verify_bundle()` | `?` | ✗ |
| `viz()` | `VizAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating GraphTable

GraphTables are created from graphs or subgraphs:

```python
import groggy as gr

g = gr.generators.karate_club()

# From graph
table = g.table()

# From subgraph
young = g.nodes[g.nodes["age"] < 30]
young_table = young.table()

# Load from bundle
table = gr.GraphTable.load_bundle("graph.bundle")
```

---

### Accessing Components

#### `nodes`

Access the nodes table.

**Returns:**
- `NodesTable`: Table of node data

**Example:**
```python
table = g.table()
nodes_table = table.nodes

# View nodes
print(nodes_table.head())
```

---

#### `edges`

Access the edges table.

**Returns:**
- `EdgesTable`: Table of edge data

**Example:**
```python
table = g.table()
edges_table = table.edges

# View edges
print(edges_table.head())
```

---

### Properties

#### `shape()`

Get dimensions (total rows, total columns).

**Returns:**
- `tuple[int, int]`: (rows, cols) across nodes and edges

**Example:**
```python
table = g.table()
rows, cols = table.shape()
print(f"Total: {rows} rows, {cols} columns")
```

---

#### `nrows()`

Get total number of rows.

**Returns:**
- `int`: Number of rows

**Example:**
```python
table = g.table()
print(f"{table.nrows()} total rows")
```

---

#### `ncols()`

Get total number of columns.

**Returns:**
- `int`: Number of columns

**Example:**
```python
table = g.table()
print(f"{table.ncols()} total columns")
```

---

#### `is_empty()`

Check if table has no data.

**Returns:**
- `bool`: True if empty

**Example:**
```python
table = g.table()
if table.is_empty():
    print("No data")
```

---

### Display Methods

#### `head(n=5)`

Show first n rows.

**Parameters:**
- `n` (int): Number of rows (default 5)

**Returns:**
- `GraphTable`: Table with first n rows

**Example:**
```python
table = g.table()
table.head()      # First 5 rows
table.head(10)    # First 10 rows
```

---

#### `tail(n=5)`

Show last n rows.

**Parameters:**
- `n` (int): Number of rows (default 5)

**Returns:**
- `GraphTable`: Table with last n rows

**Example:**
```python
table = g.table()
table.tail()      # Last 5 rows
table.tail(10)    # Last 10 rows
```

---

#### `stats()`

Get summary statistics.

**Returns:**
- `dict`: Statistics about the table

**Example:**
```python
table = g.table()
stats = table.stats()
print(stats)
# {'nodes': 34, 'edges': 78, 'node_attrs': ['age', 'club'], ...}
```

---

### Validation

#### `validate()`

Validate table structure.

**Returns:**
- `str`: Validation report

**Example:**
```python
table = g.table()
report = table.validate()
print(report)
# "Valid: 34 nodes, 78 edges, all edge endpoints exist"
```

**Notes:**
- Checks that edge endpoints exist in nodes table
- Validates data types
- Reports any inconsistencies

---

### Conversion

#### `to_graph()`

Convert table back to Graph.

**Returns:**
- `Graph`: New graph from table data

**Example:**
```python
table = g.table()

# Modify table data (via pandas, etc.)
# ...

# Convert back to graph
g2 = table.to_graph()
```

**Use cases:**
- Round-trip processing (Graph → Table → modify → Graph)
- Building graphs from table data
- Restoring from saved bundles

---

### I/O Operations

#### `save_bundle(path)`

Save complete graph to bundle file.

**Parameters:**
- `path` (str): File path to save to

**Returns:**
- `None`

**Example:**
```python
table = g.table()
table.save_bundle("my_graph.bundle")
```

**Notes:**
- Bundles store complete graph state
- Efficient binary format
- Preserves all attributes and structure

---

#### `load_bundle(path)` (class method)

Load graph from bundle file.

**Parameters:**
- `path` (str): File path to load from

**Returns:**
- `GraphTable`: Loaded table

**Example:**
```python
table = gr.GraphTable.load_bundle("my_graph.bundle")
g = table.to_graph()
```

---

### Utility Methods

#### `auto_assign_edge_ids()`

Automatically assign IDs to edges if missing.

**Returns:**
- `GraphTable`: Table with edge IDs assigned

**Example:**
```python
# If edges lack IDs
table_with_ids = table.auto_assign_edge_ids()
```

---

#### `viz()`

Access visualization methods.

**Returns:**
- `VizAccessor`: Visualization accessor

**Example:**
```python
viz = table.viz()
# Use viz methods
```

---

## Usage Patterns

### Pattern 1: Export to Pandas

```python
table = g.table()

# Get nodes as DataFrame
nodes_df = table.nodes.to_pandas()
print(nodes_df.head())

# Get edges as DataFrame
edges_df = table.edges.to_pandas()
print(edges_df.head())

# Combined analysis
import pandas as pd
print(f"Nodes: {len(nodes_df)}")
print(f"Edges: {len(edges_df)}")
print(f"Attributes: {list(nodes_df.columns)}")
```

### Pattern 2: Save/Load Graph

```python
# Save
g = gr.generators.karate_club()
table = g.table()
table.save_bundle("karate.bundle")

# Load later
table2 = gr.GraphTable.load_bundle("karate.bundle")
g2 = table2.to_graph()

# Verify
assert g2.node_count() == g.node_count()
assert g2.edge_count() == g.edge_count()
```

### Pattern 3: Round-Trip Processing

```python
# Graph to table
table = g.table()

# Export to pandas for processing
df = table.nodes.to_pandas()

# Modify
df['age_group'] = df['age'].apply(lambda x: 'young' if x < 30 else 'old')

# Save modified data
df.to_csv("nodes_modified.csv", index=False)

# Rebuild graph
import pandas as pd
nodes_df = pd.read_csv("nodes_modified.csv")
# Build new graph from modified data
```

### Pattern 4: Validation

```python
table = g.table()

# Check validity
report = table.validate()
print(report)

# Check if empty
if not table.is_empty():
    stats = table.stats()
    print(f"Valid graph: {stats}")
```

### Pattern 5: Inspection

```python
table = g.table()

# Quick overview
print(f"Shape: {table.shape()}")
print(f"Rows: {table.nrows()}, Cols: {table.ncols()}")

# Preview data
print("\nFirst 5 rows:")
table.head()

print("\nLast 5 rows:")
table.tail()

# Detailed stats
stats = table.stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

---

## Quick Reference

### Properties

| Method | Returns | Description |
|--------|---------|-------------|
| `shape()` | `tuple` | (rows, cols) |
| `nrows()` | `int` | Total rows |
| `ncols()` | `int` | Total columns |
| `is_empty()` | `bool` | Has data? |

### Components

| Method | Returns | Description |
|--------|---------|-------------|
| `nodes` | `NodesTable` | Nodes table |
| `edges` | `EdgesTable` | Edges table |

### Display

| Method | Description |
|--------|-------------|
| `head(n)` | First n rows |
| `tail(n)` | Last n rows |
| `stats()` | Summary statistics |

### I/O

| Method | Description |
|--------|-------------|
| `save_bundle(path)` | Save to file |
| `load_bundle(path)` | Load from file |
| `to_graph()` | Convert to Graph |

---

## Relationship with Other Tables

GraphTable is a container for NodesTable and EdgesTable:

```python
table = g.table()           # GraphTable

nodes_table = table.nodes   # NodesTable
edges_table = table.edges   # EdgesTable

# Each has export methods
nodes_df = nodes_table.to_pandas()
edges_df = edges_table.to_pandas()

nodes_table.to_csv("nodes.csv")
edges_table.to_parquet("edges.parquet")
```

**Key difference:**
- `GraphTable`: Complete graph representation (nodes + edges)
- `NodesTable`: Only node data
- `EdgesTable`: Only edge data


---

## Object Transformations

`GraphTable` can transform into:

- **GraphTable → NodesTable**: `table.nodes`
- **GraphTable → EdgesTable**: `table.edges`
- **GraphTable → DataFrame**: `table.to_pandas()`
- **GraphTable → Files**: `table.to_csv()`, `table.to_parquet()`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/tables.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How GraphTable works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains

## Additional Methods

#### `edges()`

Edges.

**Returns:**
- `EdgesTable`: Return value

**Example:**
```python
obj.edges()
```

---

#### `from_federated_bundles(bundle_paths)`

From Federated Bundles.

**Parameters:**
- `bundle_paths`: bundle paths

**Returns:**
- `None`: Return value

**Example:**
```python
obj.from_federated_bundles(bundle_paths=...)
```

---

#### `get_bundle_info(bundle_path)`

Get Bundle Info.

**Parameters:**
- `bundle_path`: bundle path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.get_bundle_info(bundle_path=...)
```

---

#### `merge(tables)`

Merge.

**Parameters:**
- `tables`: tables

**Returns:**
- `None`: Return value

**Example:**
```python
obj.merge(tables=...)
```

---

#### `nodes()`

Nodes.

**Returns:**
- `NodesTable`: Return value

**Example:**
```python
obj.nodes()
```

---

#### `to_edges()`

To Edges.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_edges()
```

---

#### `to_nodes()`

To Nodes.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_nodes()
```

---

#### `to_subgraphs()`

To Subgraphs.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.to_subgraphs()
```

---

#### `verify_bundle(bundle_path)`

Verify Bundle.

**Parameters:**
- `bundle_path`: bundle path

**Returns:**
- `None`: Return value

**Example:**
```python
obj.verify_bundle(bundle_path=...)
```

---


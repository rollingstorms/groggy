# Working with Tables

Tables provide a **tabular view** of graph data. Think pandas DataFrames for graphs - you get rectangular data that's easy to export, analyze, and transform.

---

## Table Types in Groggy

Groggy has four table types:

```python
import groggy as gr

g = gr.generators.karate_club()

# GraphTable - contains both nodes and edges
graph_table = g.table()

# NodesTable - just node data
nodes_table = g.nodes.table()

# EdgesTable - just edge data
edges_table = g.edges.table()

# BaseTable - low-level table operations
base_table = nodes_table.into_base_table()
```

**Hierarchy:**
```
GraphTable (nodes + edges)
├── NodesTable (inherits from BaseTable)
└── EdgesTable (inherits from BaseTable)
    └── BaseTable (core table operations)
```

---

## GraphTable: Complete Graph View

### Creating GraphTables

Get a table from a graph or subgraph:

```python
g = gr.Graph()
alice = g.add_node(name="Alice", age=29)
bob = g.add_node(name="Bob", age=55)
g.add_edge(alice, bob, weight=5)

# From graph
table = g.table()  # GraphTable

# From subgraph
sub = g.nodes[g.nodes["age"] > 30]
sub_table = sub.table()
```

### Accessing Nodes and Edges

GraphTable has separate nodes and edges tables:

```python
table = g.table()

# Get nodes table
nodes = table.nodes()  # NodesTable
print(f"Nodes shape: {nodes.shape()}")

# Get edges table
edges = table.edges()  # EdgesTable
print(f"Edges shape: {edges.shape()}")
```

### Inspecting GraphTables

```python
table = g.table()

# Shape
print(table.shape())  # (num_nodes, num_node_cols, num_edges, num_edge_cols)

# Row/column counts
print(f"Rows: {table.nrows()}, Cols: {table.ncols()}")

# Check if empty
if table.is_empty():
    print("Empty table")

# Statistics
stats = table.stats()
print(stats)  # Dict with node/edge counts, etc.

# Preview
preview = table.head(5)  # First 5 rows
tail = table.tail(5)     # Last 5 rows
```

### Converting Back to Graph

Materialize table as a graph:

```python
table = g.table()

# Convert to graph
new_graph = table.to_graph()

# Now you can modify it
new_graph.add_node(name="Charlie")
```

---

## NodesTable: Node Data

### Creating NodesTable

```python
g = gr.Graph()
g.add_node(name="Alice", age=29, role="Engineer")
g.add_node(name="Bob", age=55, role="Manager")
g.add_node(name="Carol", age=31, role="Engineer")

# Get nodes table
nodes = g.nodes.table()  # NodesTable
```

### Inspecting NodesTable

```python
# Shape (rows, columns)
print(nodes.shape())  # (3, 4) - 3 nodes, 4 columns (id + 3 attrs)

# Row/column counts
print(f"Nodes: {nodes.nrows()}")
print(f"Columns: {nodes.ncols()}")

# Preview
print(nodes.head(10))  # First 10 rows
print(nodes.tail(5))   # Last 5 rows

# Check if empty
if nodes.is_empty():
    print("No nodes")
```

### Selecting Columns

```python
# Select specific columns
selected = nodes.select(["name", "age"])
print(selected.shape())  # Fewer columns

# Drop columns
without_role = nodes.drop_columns(["role"])
```

### Sorting

```python
# Sort by single column
by_age = nodes.sort_by("age")
by_name = nodes.sort_by("name")

# Sort values (alternative)
sorted_nodes = nodes.sort_values("age")
```

### Grouping

```python
# Group by attribute
by_role = nodes.group_by("role")  # NodesTableArray

# Iterate groups
for group_table in by_role:
    print(f"Group size: {group_table.nrows()}")
```

### Getting Node IDs

```python
# Extract node IDs column
ids = nodes.node_ids()  # NumArray
print(ids.head())
```

### Iteration

```python
# Iterate rows
for row in nodes.iter():
    # Each row is a dict-like object
    print(row)
```

---

## EdgesTable: Edge Data

### Creating EdgesTable

```python
g = gr.Graph()
n0 = g.add_node()
n1 = g.add_node()
n2 = g.add_node()
g.add_edge(n0, n1, weight=5, type="friend")
g.add_edge(n0, n2, weight=2, type="colleague")

# Get edges table
edges = g.edges.table()  # EdgesTable
```

### Inspecting EdgesTable

```python
# Shape
print(edges.shape())  # (num_edges, num_columns)

# Counts
print(f"Edges: {edges.nrows()}")
print(f"Columns: {edges.ncols()}")

# Preview
print(edges.head(10))
print(edges.tail(5))
```

### Edge Endpoints

```python
# Get source nodes
sources = edges.sources()  # NumArray

# Get target nodes
targets = edges.targets()  # NumArray

# Zip together
for src, tgt in zip(sources, targets):
    print(f"{src} → {tgt}")
```

### Selecting and Sorting

```python
# Select columns
selected = edges.select(["weight", "type"])

# Drop columns
without_type = edges.drop_columns(["type"])

# Sort
by_weight = edges.sort_by("weight")
```

### Grouping

```python
# Group by type
by_type = edges.group_by("type")  # EdgesTableArray

for group in by_type:
    print(f"Type has {group.nrows()} edges")
```

### Getting Edge IDs

```python
# Extract edge IDs
ids = edges.edge_ids()  # NumArray
print(ids.head())
```

---

## Exporting Tables

### To pandas

Convert to pandas DataFrame:

```python
# Nodes to DataFrame
nodes = g.nodes.table()
nodes_df = nodes.to_pandas()
print(type(nodes_df))  # pandas.DataFrame

# Edges to DataFrame
edges = g.edges.table()
edges_df = edges.to_pandas()

# Analyze with pandas
mean_age = nodes_df['age'].mean()
print(f"Mean age: {mean_age:.1f}")
```

### To CSV

Export to CSV files:

```python
# Save nodes
g.nodes.table().to_csv("nodes.csv")

# Save edges
g.edges.table().to_csv("edges.csv")

# Note: Check if to_csv is available in your version
# May need to go through pandas:
# g.nodes.table().to_pandas().to_csv("nodes.csv")
```

### To Parquet

Export to Parquet (efficient columnar format):

```python
# Save as parquet
g.nodes.table().to_parquet("nodes.parquet")
g.edges.table().to_parquet("edges.parquet")

# Note: May need pandas route if not directly available:
# g.nodes.table().to_pandas().to_parquet("nodes.parquet")
```

---

## Graph Bundles

### Saving Complete Graphs

Bundle format saves entire graph (structure + attributes):

```python
# Save as bundle
g.save_bundle("my_graph.bundle")

# Later...
loaded_table = gr.GraphTable.load_bundle("my_graph.bundle")
restored_graph = loaded_table.to_graph()
```

**Bundle advantages:**
- Single file for complete graph
- Preserves all structure and attributes
- Fast to save/load
- Compressed storage

### Bundle Operations

```python
# Check bundle info
# info = gr.GraphTable.get_bundle_info("my_graph.bundle")

# Validate bundle
loaded = gr.GraphTable.load_bundle("my_graph.bundle")
validation = loaded.validate()
print(validation)  # "valid" or error message
```

---

## Common Patterns

### Pattern 1: Export for Analysis

```python
# Get nodes as DataFrame
nodes_df = g.nodes.table().to_pandas()

# Analyze with pandas
import pandas as pd

summary = nodes_df.describe()
print(summary)

# Group and aggregate
by_role = nodes_df.groupby('role')['age'].agg(['mean', 'count'])
print(by_role)
```

### Pattern 2: Filter and Export

```python
# Filter
active = g.nodes[g.nodes["active"] == True]

# Export filtered
active.table().to_csv("active_users.csv")
```

### Pattern 3: Combine Graph and Pandas

```python
# Start with graph operations
components = g.connected_components()
largest = components[0]

# Convert to table and pandas
df = largest.table().nodes().to_pandas()

# Analyze with pandas
stats = {
    'count': len(df),
    'mean_age': df['age'].mean(),
    'roles': df['role'].unique()
}
print(stats)
```

### Pattern 4: Table-Based Filtering

```python
# Get as table
nodes = g.nodes.table()

# Sort to find extremes
oldest = nodes.sort_by("age").tail(5)
youngest = nodes.sort_by("age").head(5)

# Convert to pandas for easier viewing
print(oldest.to_pandas())
print(youngest.to_pandas())
```

### Pattern 5: Edge Analysis

```python
# Get edges table
edges = g.edges.table()

# Sort by weight
heavy = edges.sort_by("weight").tail(10)

# Get as pandas
heavy_df = heavy.to_pandas()

# Add source/target names if available
# (would need to join with nodes table)
```

### Pattern 6: Group Statistics

```python
# Group nodes by attribute
by_role = g.nodes.table().group_by("role")

# Analyze each group
for group_table in by_role:
    df = group_table.to_pandas()
    role = df['role'].iloc[0]  # Get role name
    count = len(df)
    avg_age = df['age'].mean()

    print(f"{role}: {count} people, avg age {avg_age:.1f}")
```

### Pattern 7: Round-Trip Processing

```python
# Graph → Table → pandas
df = g.nodes.table().to_pandas()

# Process with pandas
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100])

# Save processed
df.to_csv("processed_nodes.csv", index=False)

# Note: To get back to graph, would need to rebuild:
# new_graph = gr.Graph()
# for _, row in df.iterrows():
#     new_graph.add_node(**row.to_dict())
```

---

## BaseTable: Low-Level Operations

### Converting to BaseTable

NodesTable and EdgesTable inherit from BaseTable:

```python
nodes = g.nodes.table()

# Get as BaseTable
base = nodes.into_base_table()  # BaseTable

# Or create reference
base_ref = nodes.base_table()  # BaseTable
```

**Use BaseTable when:**
- Generic table operations
- No need for node/edge-specific methods
- Interfacing with code expecting BaseTable

---

## Performance Considerations

### Memory

Tables are **snapshots**, not views:

```python
# Creates a copy
table = g.table()

# Table is independent of graph
g.add_node(name="New")
print(len(table.nodes()))  # Unchanged - table is snapshot
```

### When to Use Tables

**Use tables when:**
- ✅ Exporting to CSV/Parquet/pandas
- ✅ Tabular analysis (sorting, grouping)
- ✅ Sharing data with other tools
- ✅ Creating snapshots for reproducibility

**Avoid tables when:**
- ❌ Just want to filter (use subgraphs - they're views)
- ❌ Need to modify graph (tables are immutable)
- ❌ Memory constrained (tables copy data)

### Optimization Tips

**1. Export directly when possible:**

```python
# ✓ Direct export
g.nodes.table().to_csv("nodes.csv")

# vs. intermediate step
table = g.nodes.table()
table.to_csv("nodes.csv")
```

**2. Select columns before conversion:**

```python
# ✓ Filter columns first
nodes = g.nodes.table().select(["name", "age"])
df = nodes.to_pandas()

# vs. converting everything
all_nodes = g.nodes.table().to_pandas()
df = all_nodes[["name", "age"]]
```

**3. Use views for filtering:**

```python
# ✓ Filter with subgraph (view)
filtered = g.nodes[g.nodes["age"] > 30]
table = filtered.table()  # Table only filtered nodes

# vs. table then filter
table = g.nodes.table()
df = table.to_pandas()
filtered_df = df[df['age'] > 30]  # Converted all first
```

---

## Table Display

### Rich Display

Tables have enhanced display in notebooks:

```python
nodes = g.nodes.table()

# Rich display
display_str = nodes.rich_display()

# Interactive (if in notebook)
interactive = nodes.interactive()

# Interactive visualization
viz = nodes.interactive_viz()
```

### Head/Tail for Preview

```python
# Preview first/last rows
print(nodes.head(5))
print(nodes.tail(3))

# Head returns table, can chain
preview = nodes.head(100).to_pandas()
```

---

## Quick Reference

### GraphTable

| Operation | Method | Returns |
|-----------|--------|---------|
| Nodes | `table.nodes()` | `NodesTable` |
| Edges | `table.edges()` | `EdgesTable` |
| Shape | `table.shape()` | `tuple` |
| Head | `table.head(n)` | `GraphTable` |
| To Graph | `table.to_graph()` | `Graph` |
| Stats | `table.stats()` | `dict` |
| Validate | `table.validate()` | `str` |

### NodesTable

| Operation | Method | Returns |
|-----------|--------|---------|
| Shape | `nodes.shape()` | `tuple` |
| Head | `nodes.head(n)` | `NodesTable` |
| Sort | `nodes.sort_by(col)` | `NodesTable` |
| Select | `nodes.select([cols])` | `NodesTable` |
| Group | `nodes.group_by(col)` | `NodesTableArray` |
| To pandas | `nodes.to_pandas()` | `DataFrame` |
| Node IDs | `nodes.node_ids()` | `NumArray` |
| Iterate | `nodes.iter()` | Iterator |

### EdgesTable

| Operation | Method | Returns |
|-----------|--------|---------|
| Shape | `edges.shape()` | `tuple` |
| Head | `edges.head(n)` | `EdgesTable` |
| Sort | `edges.sort_by(col)` | `EdgesTable` |
| Select | `edges.select([cols])` | `EdgesTable` |
| Group | `edges.group_by(col)` | `EdgesTableArray` |
| To pandas | `edges.to_pandas()` | `DataFrame` |
| Sources | `edges.sources()` | `NumArray` |
| Targets | `edges.targets()` | `NumArray` |
| Edge IDs | `edges.edge_ids()` | `NumArray` |

---

## See Also

- **[GraphTable API Reference](../api/graphtable.md)**: Complete method reference
- **[NodesTable API Reference](../api/nodestable.md)**: NodesTable methods
- **[EdgesTable API Reference](../api/edgestable.md)**: EdgesTable methods
- **[BaseTable API Reference](../api/basetable.md)**: Low-level table operations
- **[Graph Core Guide](graph-core.md)**: Converting graphs to tables
- **[Subgraphs Guide](subgraphs.md)**: Using subgraphs as views vs. tables
- **[Arrays Guide](arrays.md)**: Column-based operations

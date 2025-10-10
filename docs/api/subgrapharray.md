# SubgraphArray API Reference

**Type**: `groggy.SubgraphArray`

---

## Overview

A collection of Subgraph objects, typically from algorithms like connected_components.

**Primary Use Cases:**
- Working with graph components
- Analyzing community structures
- Processing multiple subgraphs in parallel

**Related Objects:**
- `Subgraph`
- `Graph`
- `GraphTable`

---

## Complete Method Reference

The following methods are available on `SubgraphArray` objects. This reference is generated from comprehensive API testing and shows all empirically validated methods.

| Method | Returns | Status |
|--------|---------|--------|
| `collapse()` | `?` | ✗ |
| `collect()` | `list` | ✓ |
| `edges_table()` | `TableArray` | ✓ |
| `extract_node_attribute()` | `?` | ✗ |
| `group_by()` | `?` | ✗ |
| `is_empty()` | `bool` | ✓ |
| `map()` | `?` | ✗ |
| `merge()` | `Graph` | ✓ |
| `nodes_table()` | `TableArray` | ✓ |
| `sample()` | `SubgraphArray` | ✓ |
| `summary()` | `BaseTable` | ✓ |
| `table()` | `TableArray` | ✓ |
| `to_list()` | `list` | ✓ |
| `viz()` | `VizAccessor` | ✓ |

**Legend:**
- ✓ = Method tested and working
- ✗ = Method failed in testing or not yet validated
- `?` = Return type not yet determined

---

## Detailed Method Reference

### Creating SubgraphArray

SubgraphArray is typically returned from algorithms that produce multiple subgraphs:

```python
import groggy as gr

g = gr.generators.karate_club()

# From connected components
components = g.connected_components()  # → SubgraphArray
print(type(components))  # SubgraphArray

# From other algorithms
k_cores = g.k_core_decomposition()  # → SubgraphArray
communities = g.louvain()  # → SubgraphArray

# Each element is a Subgraph
for component in components:
    print(f"Component: {component.node_count()} nodes")
```

**Key Concept:** SubgraphArray is a collection of Subgraph objects, useful for analyzing disconnected components, communities, or other graph partitions.

---

### Core Methods

#### `is_empty()`

Check if array has no subgraphs.

**Returns:**
- `bool`: True if no subgraphs

**Example:**
```python
components = g.connected_components()
if components.is_empty():
    print("No components found")
else:
    print(f"{len(components)} components")
```

**Performance:** O(1)

---

#### `to_list()` / `collect()`

Convert to Python list of Subgraph objects.

**Returns:**
- `list[Subgraph]`: List of subgraphs

**Example:**
```python
components = g.connected_components()
comp_list = components.to_list()

for i, comp in enumerate(comp_list):
    print(f"Component {i}: {comp.node_count()} nodes, {comp.edge_count()} edges")
```

**Performance:** O(k) where k is number of subgraphs

**Notes:** `collect()` is an alias for `to_list()`

---

### Sampling & Filtering

#### `sample(n)`

Randomly sample n subgraphs.

**Parameters:**
- `n` (int): Number of subgraphs to sample

**Returns:**
- `SubgraphArray`: Sampled subgraphs

**Example:**
```python
components = g.connected_components()

# Sample 5 components
sample = components.sample(5)
print(f"Sampled {len(sample)} components")

# Use in chains
components.sample(3).neighborhood(depth=2).table()
```

**Performance:** O(n)

**Notes:**
- If n > array length, returns all subgraphs
- Sampling is random without replacement

---

### Aggregation Methods

#### `merge()`

Merge all subgraphs into single graph.

**Returns:**
- `Graph`: New graph containing all subgraphs

**Example:**
```python
components = g.connected_components()

# Merge back to single graph
merged = components.merge()
print(f"Merged: {merged.node_count()} nodes, {merged.edge_count()} edges")

# Should match original (if components cover full graph)
assert merged.node_count() == g.node_count()
```

**Performance:** O(V + E) where V, E are total nodes/edges

---

#### `summary()`

Get summary statistics about all subgraphs.

**Returns:**
- `BaseTable`: Table with statistics per subgraph

**Example:**
```python
components = g.connected_components()
summary = components.summary()

# Show summary
df = summary.to_pandas()
print(df[['nodes', 'edges', 'density', 'avg_degree']])
```

**Columns typically include:**
- `index`: Subgraph index
- `nodes`: Node count
- `edges`: Edge count
- `density`: Graph density
- `avg_degree`: Average degree

---

### Table Conversion

#### `table()`

Convert to array of GraphTables.

**Returns:**
- `TableArray`: Array of GraphTable objects

**Example:**
```python
components = g.connected_components()
tables = components.table()

# Export each component
for i, tbl in enumerate(tables):
    tbl.nodes.to_csv(f"component_{i}_nodes.csv")
    tbl.edges.to_csv(f"component_{i}_edges.csv")
```

---

#### `nodes_table()`

Get array of node tables.

**Returns:**
- `TableArray`: Array of NodesTable objects

**Example:**
```python
components = g.connected_components()
node_tables = components.nodes_table()

# Analyze nodes in each component
for i, nodes_tbl in enumerate(node_tables):
    df = nodes_tbl.to_pandas()
    print(f"Component {i}: {len(df)} nodes")
    print(f"  Avg age: {df['age'].mean():.1f}")
```

---

#### `edges_table()`

Get array of edge tables.

**Returns:**
- `TableArray`: Array of EdgesTable objects

**Example:**
```python
components = g.connected_components()
edge_tables = components.edges_table()

# Analyze edges in each component
for i, edges_tbl in enumerate(edge_tables):
    df = edges_tbl.to_pandas()
    print(f"Component {i}: {len(df)} edges")
    print(f"  Avg weight: {df['weight'].mean():.2f}")
```

---

### Indexing & Iteration

SubgraphArray supports indexing, slicing, and iteration:

**Example:**
```python
components = g.connected_components()

# Get specific component
largest = components[0]  # First component (Subgraph)
print(f"Largest: {largest.node_count()} nodes")

# Slice
top_three = components[:3]  # First 3 components (SubgraphArray)

# Negative indexing
smallest = components[-1]

# Iteration
for i, comp in enumerate(components):
    print(f"Component {i}: {comp.node_count()} nodes")

# Length
print(f"Total: {len(components)} components")
```

---

### Visualization

#### `viz()`

Access visualization methods.

**Returns:**
- `VizAccessor`: Visualization accessor

**Example:**
```python
components = g.connected_components()
viz = components.viz()
# Use viz methods for plotting components
```

---

## Usage Patterns

### Pattern 1: Component Analysis

```python
components = g.connected_components()

print(f"Total components: {len(components)}")

# Analyze each component
for i, comp in enumerate(components):
    print(f"\nComponent {i}:")
    print(f"  Nodes: {comp.node_count()}")
    print(f"  Edges: {comp.edge_count()}")
    print(f"  Density: {comp.density():.3f}")
    print(f"  Avg degree: {comp.degree().mean():.2f}")
```

### Pattern 2: Filter Components by Size

```python
components = g.connected_components()

# Get only large components
large_components = []
for comp in components:
    if comp.node_count() >= 10:
        large_components.append(comp)

print(f"{len(large_components)} components with ≥10 nodes")

# Analyze large components
for comp in large_components:
    print(f"Large component: {comp.node_count()} nodes")
```

### Pattern 3: Sample and Expand

```python
# Get connected components
components = g.connected_components()

# Sample a few components
sample = components.sample(3)

# Expand neighborhood around sampled components
expanded = sample.neighborhood(depth=2)

# Export
for i, subg in enumerate(expanded):
    tbl = subg.table()
    tbl.to_csv(f"expanded_component_{i}.csv")
```

### Pattern 4: Component Comparison

```python
components = g.connected_components()

if len(components) >= 2:
    comp_a = components[0]
    comp_b = components[1]

    print("Component comparison:")
    print(f"  Component A: {comp_a.node_count()} nodes")
    print(f"    Density: {comp_a.density():.3f}")
    print(f"    Diameter: {comp_a.diameter()}")

    print(f"  Component B: {comp_b.node_count()} nodes")
    print(f"    Density: {comp_b.density():.3f}")
    print(f"    Diameter: {comp_b.diameter()}")
```

### Pattern 5: Export All Components

```python
components = g.connected_components()

# Export each component separately
for i, comp in enumerate(components):
    # As graph
    comp_graph = comp.to_graph()
    comp_graph.save_bundle(f"component_{i}.bundle")

    # As tables
    comp.table().nodes.to_csv(f"component_{i}_nodes.csv")
    comp.table().edges.to_csv(f"component_{i}_edges.csv")

print(f"Exported {len(components)} components")
```

### Pattern 6: Merge Filtered Components

```python
components = g.connected_components()

# Filter components by criteria
selected = []
for comp in components:
    # Keep components with high average degree
    if comp.degree().mean() > 3.0:
        selected.append(comp)

print(f"Selected {len(selected)} high-connectivity components")

# Merge selected components
if selected:
    merged = gr.merge(selected)
    print(f"Merged graph: {merged.node_count()} nodes")
```

### Pattern 7: Summary Table Analysis

```python
components = g.connected_components()

# Get summary table
summary = components.summary()
df = summary.to_pandas()

# Analyze distribution
print("Component size distribution:")
print(df['nodes'].describe())

# Identify outliers
large_threshold = df['nodes'].quantile(0.9)
large_comps = df[df['nodes'] > large_threshold]
print(f"\nLarge components (>90th percentile):")
print(large_comps[['index', 'nodes', 'edges', 'density']])
```

### Pattern 8: Delegation Chain

```python
# Classic delegation chain:
# connected_components → sample → neighborhood → table → aggregate
result = (
    g.connected_components()      # SubgraphArray
     .sample(5)                   # SubgraphArray (5 components)
     .neighborhood(depth=2)       # SubgraphArray (expanded)
     .table()                     # TableArray
)

# Process results
for i, tbl in enumerate(result):
    print(f"Expanded component {i}:")
    print(f"  Nodes: {tbl.nodes.nrows()}")
    print(f"  Edges: {tbl.edges.nrows()}")
```

---

## Quick Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `is_empty()` | `bool` | Check if empty |
| `to_list()` | `list` | Convert to list |
| `collect()` | `list` | Alias for to_list() |
| `sample(n)` | `SubgraphArray` | Random sample |
| `merge()` | `Graph` | Merge all subgraphs |
| `summary()` | `BaseTable` | Statistics table |
| `table()` | `TableArray` | Convert to tables |
| `nodes_table()` | `TableArray` | Node tables |
| `edges_table()` | `TableArray` | Edge tables |
| `viz()` | `VizAccessor` | Visualization |
| `[i]` | `Subgraph` | Get by index |
| `[:n]` | `SubgraphArray` | Slice |
| `len()` | `int` | Number of subgraphs |


---

## Object Transformations

`SubgraphArray` can transform into:

- **SubgraphArray → Subgraph**: `arr[0]`, `arr.sample(n)`
- **SubgraphArray → GraphTable**: `arr.table()`
- **SubgraphArray → SubgraphArray**: `arr.neighborhood(depth=2)`

See [Object Transformation Graph](../concepts/connected-views.md) for complete delegation chains.

---

## See Also

- **[User Guide](../guide/subgraphs.md)**: Comprehensive tutorial and patterns
- **[Architecture](../concepts/architecture.md)**: How SubgraphArray works internally
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains

## Additional Methods

#### `collapse()`

Collapse.

**Returns:**
- `list`: Return value

**Example:**
```python
obj.collapse()
```

---

#### `collect()`

Collect.

**Returns:**
- `list`: Return value

**Example:**
```python
obj.collect()
```

---

#### `extract_node_attribute()`

Extract Node Attribute.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.extract_node_attribute()
```

---

#### `group_by(element_type)`

Group By.

**Parameters:**
- `element_type`: element type

**Returns:**
- `None`: Return value

**Example:**
```python
obj.group_by(element_type=...)
```

---

#### `map()`

Map.

**Returns:**
- `None`: Return value

**Example:**
```python
obj.map()
```

---


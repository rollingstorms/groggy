# Working with SubgraphArrays

A `SubgraphArray` is a collection of `Subgraph` objects, typically returned by algorithms that partition or analyze graphs. Think of it as an array where each element is a subgraph.

---

## What is a SubgraphArray?

SubgraphArrays are collections of related subgraphs:

```python
import groggy as gr

g = gr.generators.karate_club()

# Get connected components - returns SubgraphArray
components = g.connected_components()

print(type(components))  # SubgraphArray
print(len(components))   # Number of components
```

**Common sources:**
- `g.connected_components()` - One subgraph per component
- `g.nodes.group_by("attribute")` - One subgraph per group
- `g.edges.group_by("attribute")` - Groups of edges
- `sub.neighborhood(depth=k)` - K-hop neighborhoods

---

## Creating SubgraphArrays

### From Connected Components

The most common way to get a SubgraphArray:

```python
g = gr.generators.karate_club()

# Find all connected components
components = g.connected_components()

print(f"Found {len(components)} component(s)")

# Access individual components
largest = components[0]  # Subgraph
print(f"Largest: {largest.node_count()} nodes")
```

### From Grouping

Group nodes or edges by attribute:

```python
g = gr.Graph()
g.add_node(name="Alice", role="Engineer")
g.add_node(name="Bob", role="Manager")
g.add_node(name="Carol", role="Engineer")
g.add_node(name="Dave", role="Manager")

# Group by role - returns SubgraphArray
by_role = g.nodes.group_by("role")

print(f"Groups: {len(by_role)}")

# Each element is a subgraph of that group
for i, group in enumerate(by_role):
    print(f"Group {i}: {group.node_count()} nodes")
```

### From Neighborhoods

Expand neighborhoods around nodes:

```python
# Get 2-hop neighborhood around each node
neighborhoods = g.nodes[:5].neighborhood(depth=2)

# Returns SubgraphArray - one neighborhood per seed node
print(f"{len(neighborhoods)} neighborhoods")
```

---

## Accessing Elements

### Indexing

Access individual subgraphs by index:

```python
components = g.connected_components()

# First component
first = components[0]  # Subgraph
print(f"First: {first.node_count()} nodes")

# Last component
last = components[-1]  # Subgraph

# Slice
first_three = components[:3]  # Still a SubgraphArray
```

### Iteration

Loop over subgraphs:

```python
components = g.connected_components()

for i, comp in enumerate(components):
    print(f"Component {i}:")
    print(f"  Nodes: {comp.node_count()}")
    print(f"  Edges: {comp.edge_count()}")
    print(f"  Density: {comp.density():.3f}")
```

### To List

Convert to Python list:

```python
# Get as list
comp_list = components.to_list()
print(type(comp_list))  # list

# Or use collect()
comp_list = components.collect()
```

---

## Analyzing SubgraphArrays

### Summary Statistics

Get overview of all subgraphs:

```python
components = g.connected_components()

# Summary table with stats for each component
summary = components.summary()
print(summary)

# Shows: component index, node count, edge count, etc.
```

### Checking Contents

```python
# Check if empty
if components.is_empty():
    print("No components")

# Length
num_components = len(components)
print(f"{num_components} components")
```

### Size Analysis

Analyze component sizes:

```python
components = g.connected_components()

# Get sizes
sizes = [comp.node_count() for comp in components]

print(f"Sizes: {sizes}")
print(f"Largest: {max(sizes)} nodes")
print(f"Smallest: {min(sizes)} nodes")
print(f"Mean: {sum(sizes)/len(sizes):.1f} nodes")
```

---

## Transformations

### SubgraphArray → Tables

Convert to tabular format:

```python
components = g.connected_components()

# Combined table for all components
table = components.table()  # TableArray

# Node tables
nodes_table = components.nodes_table()  # TableArray

# Edge tables
edges_table = components.edges_table()  # TableArray
```

### SubgraphArray → Graph

Merge all subgraphs back into a single graph:

```python
components = g.connected_components()

# Merge components into one graph
merged = components.merge()  # Graph

print(f"Merged: {merged.node_count()} nodes")
```

**Use case:** Recombining filtered components

```python
# Get components
components = g.connected_components()

# Filter to large components only
large = [c for c in components if c.node_count() > 10]

# Create SubgraphArray from list
# (Note: May need to use internal constructor)
# For now, just work with list

# Merge selected components
# merged = SubgraphArray(large).merge()
```

### Individual Subgraph → Graph

Convert a single subgraph to graph:

```python
largest = components[0]
largest_graph = largest.to_graph()  # Graph

# Now you can modify it
largest_graph.add_node(name="NewNode")
```

---

## Filtering and Sampling

### Sampling Subgraphs

Get random sample:

```python
components = g.connected_components()

# Random sample of components
sample = components.sample(5)  # SubgraphArray

print(f"Sampled {len(sample)} components")
```

### Filtering by Condition

Use list comprehension to filter:

```python
components = g.connected_components()

# Large components only
large_components = [
    c for c in components
    if c.node_count() > 100
]

# Convert back to SubgraphArray
# (Note: May need internal constructor)
# For now, work with list

# Small components
small = [c for c in components if c.node_count() < 10]

# Dense components
dense = [c for c in components if c.density() > 0.5]
```

---

## Common Patterns

### Pattern 1: Find Largest Component

```python
components = g.connected_components()

# Find largest by node count
largest = max(components, key=lambda c: c.node_count())

print(f"Largest component:")
print(f"  Nodes: {largest.node_count()}")
print(f"  Edges: {largest.edge_count()}")

# Work with largest
largest_graph = largest.to_graph()
```

### Pattern 2: Component Size Distribution

```python
components = g.connected_components()

# Get size distribution
from collections import Counter

sizes = [c.node_count() for c in components]
size_dist = Counter(sizes)

print("Component size distribution:")
for size, count in sorted(size_dist.items()):
    print(f"  Size {size}: {count} component(s)")
```

### Pattern 3: Analyze Each Component

```python
components = g.connected_components()

results = []
for i, comp in enumerate(components):
    result = {
        'component_id': i,
        'nodes': comp.node_count(),
        'edges': comp.edge_count(),
        'density': comp.density(),
        'is_connected': comp.is_connected()
    }
    results.append(result)

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

### Pattern 4: Group Analysis

```python
# Group by attribute
by_role = g.nodes.group_by("role")

# Analyze each group
for i, group in enumerate(by_role):
    # Get representative role value
    if group.node_count() > 0:
        role_vals = group.nodes["role"]
        role = role_vals.head(1)[0] if len(role_vals) > 0 else "unknown"

        # Stats
        count = group.node_count()
        avg_age = group.nodes["age"].mean() if "age" in group.nodes.attribute_names() else 0

        print(f"{role}:")
        print(f"  Count: {count}")
        print(f"  Avg age: {avg_age:.1f}")
```

### Pattern 5: Export Components Separately

```python
components = g.connected_components()

# Export each component to separate file
for i, comp in enumerate(components):
    table = comp.table()
    table.to_csv(f"component_{i}.csv")

print(f"Exported {len(components)} components")
```

### Pattern 6: Filter and Merge

```python
components = g.connected_components()

# Keep only large components
large = [c for c in components if c.node_count() > 50]

print(f"Kept {len(large)} large components")

# Work with them individually
for comp in large:
    # Analyze large component
    density = comp.density()
    print(f"Component density: {density:.3f}")
```

### Pattern 7: Neighborhood Analysis

```python
# Get neighborhoods around seed nodes
seeds = g.nodes[:5]
neighborhoods = seeds.neighborhood(depth=2)

# Analyze each neighborhood
for i, nbh in enumerate(neighborhoods):
    center_id = seeds.node_ids()[i]
    print(f"Neighborhood around node {center_id}:")
    print(f"  Reached {nbh.node_count()} nodes")
    print(f"  Density: {nbh.density():.3f}")
```

---

## Advanced Usage

### Delegation Chains

SubgraphArrays enable powerful chains:

```python
result = (
    g.connected_components()    # → SubgraphArray
     .sample(5)                  # → SubgraphArray (5 components)
     .table()                    # → TableArray
)

# Each step transforms but maintains the collection structure
```

### Combining with Subgraph Methods

Each element supports subgraph methods:

```python
components = g.connected_components()

# Run algorithm on each component
for comp in components:
    # Each comp is a Subgraph
    sub_components = comp.connected_components()
    if len(sub_components) > 1:
        print(f"Component has {len(sub_components)} sub-components!")

    # Get adjacency
    adj = comp.adjacency_matrix()

    # Sample nodes
    sample = comp.sample(10)
```

---

## Performance Considerations

### Memory

```python
# SubgraphArray holds references to subgraphs
# Each subgraph is a view into the parent graph
# Low memory overhead

components = g.connected_components()
# ~O(num_components * (overhead per subgraph))
# Each subgraph stores node/edge IDs only
```

### Iteration

```python
# Iteration is efficient
for comp in components:
    # O(1) to access each component
    size = comp.node_count()  # O(1)
```

### Large Arrays

```python
# For large numbers of components
components = g.connected_components()

# Sample to reduce processing
sample = components.sample(100)

# Or filter early
large_only = [c for c in components if c.node_count() > 10]
```

---

## Limitations

### Not a Standard Array

SubgraphArray is not a numpy array or list:

```python
components = g.connected_components()

# ❌ No arithmetic
# components + 1  # Error

# ❌ No broadcasting
# components * 2  # Error

# ✓ But supports indexing and iteration
first = components[0]
for comp in components:
    pass
```

### Element Type

All elements are Subgraphs:

```python
components = g.connected_components()

for comp in components:
    print(type(comp))  # Always Subgraph
```

---

## Quick Reference

| Operation | Example | Returns |
|-----------|---------|---------|
| Index | `arr[0]` | `Subgraph` |
| Slice | `arr[:3]` | `SubgraphArray` |
| Length | `len(arr)` | `int` |
| Iterate | `for s in arr:` | Yields `Subgraph` |
| Sample | `arr.sample(n)` | `SubgraphArray` |
| To list | `arr.to_list()` | `list` |
| Collect | `arr.collect()` | `list` |
| Table | `arr.table()` | `TableArray` |
| Merge | `arr.merge()` | `Graph` |
| Summary | `arr.summary()` | `BaseTable` |
| Empty | `arr.is_empty()` | `bool` |

---

## See Also

- **[SubgraphArray API Reference](../api/subgrapharray.md)**: Complete method reference
- **[Subgraphs Guide](subgraphs.md)**: Working with individual subgraphs
- **[Graph Core Guide](graph-core.md)**: Graph algorithms that return SubgraphArrays
- **[Accessors Guide](accessors.md)**: Grouping operations
- **[Arrays Guide](arrays.md)**: Other array types in Groggy

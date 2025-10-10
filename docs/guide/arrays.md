# Working with Arrays

Arrays in Groggy provide **columnar access** to graph attributes. Think of them as single columns from a table, optimized for bulk operations and statistical analysis.

---

## Array Types in Groggy

Groggy has several array types:

```python
import groggy as gr

g = gr.generators.karate_club()

# NumArray - numeric data with statistics
ages = g.nodes["age"]  # NumArray (if age is numeric)
print(type(ages))

# BaseArray - generic attributes
names = g.nodes["name"]  # BaseArray

# NodesArray - node IDs
node_ids = g.nodes.ids()  # NodesArray

# EdgesArray - edge IDs
edge_ids = g.edges.ids()  # EdgesArray
```

**Type hierarchy:**
```
BaseArray (generic attributes)
├── NumArray (numeric with stats)
├── NodesArray (node IDs)
└── EdgesArray (edge IDs)
```

---

## BaseArray: Generic Attributes

### Getting BaseArrays

Access attribute columns:

```python
g = gr.Graph()
g.add_node(name="Alice", role="Engineer")
g.add_node(name="Bob", role="Manager")
g.add_node(name="Carol", role="Engineer")

# Get column as array
names = g.nodes["name"]  # BaseArray
roles = g.nodes["role"]  # BaseArray

print(type(names))  # BaseArray
```

### Basic Operations

```python
# Length
print(len(names))  # 3

# Indexing
print(names[0])  # "Alice"
print(names[-1])  # Last element

# Slicing
first_two = names[:2]  # BaseArray

# Check if empty
if names.is_empty():
    print("No values")
```

### Inspection

```python
# First/last elements
first = names.first()  # First element
last = names.last()   # Last element

# Preview
print(names.head(5))  # First 5
print(names.tail(3))  # Last 3

# Count
count = names.count()  # Number of elements

# Data type
dtype = names.dtype()  # e.g., "str", "int", "float"
```

### Unique Values

```python
roles = g.nodes["role"]

# Get unique values
unique_roles = roles.unique()  # BaseArray
print(unique_roles.to_list())  # ['Engineer', 'Manager']

# Count unique
num_unique = roles.nunique()
print(f"{num_unique} unique roles")
```

### Converting

```python
# To Python list
names_list = names.to_list()
print(type(names_list))  # list

# To numpy (if numeric)
# arr = names.to_numpy()  # May not work for non-numeric
```

---

## NumArray: Numeric Operations

### Getting NumArrays

Numeric attributes return NumArray:

```python
g = gr.Graph()
g.add_node(name="Alice", age=29)
g.add_node(name="Bob", age=55)
g.add_node(name="Carol", age=31)

# Numeric column → NumArray
ages = g.nodes["age"]  # NumArray
print(type(ages))
```

### Statistical Operations

```python
# Basic stats
mean_age = ages.mean()
max_age = ages.max()
min_age = ages.min()
sum_ages = ages.sum()

print(f"Ages: {min_age}-{max_age}, mean={mean_age:.1f}, sum={sum_ages}")

# Variance and standard deviation
variance = ages.var()
std_dev = ages.std()

print(f"Std dev: {std_dev:.2f}")
```

### Array Operations

```python
# All BaseArray operations work
first = ages.first()
last = ages.last()
unique_ages = ages.unique()

# Plus numeric-specific
total = ages.sum()
average = ages.mean()
```

### Edge Weight Example

```python
g = gr.Graph()
n0, n1, n2 = g.add_node(), g.add_node(), g.add_node()
g.add_edge(n0, n1, weight=5.0)
g.add_edge(n0, n2, weight=2.5)
g.add_edge(n1, n2, weight=1.0)

# Get weights as NumArray
weights = g.edges["weight"]

# Statistics
print(f"Total weight: {weights.sum()}")
print(f"Average weight: {weights.mean():.2f}")
print(f"Max weight: {weights.max()}")
```

---

## NodesArray: Node IDs

### Getting NodesArray

```python
g = gr.generators.karate_club()

# Get all node IDs
node_ids = g.nodes.ids()  # NodesArray
print(type(node_ids))

# From subgraph
sub = g.nodes[:10]
sub_ids = sub.node_ids()  # NumArray or NodesArray
```

### Operations

```python
# Length (number of nodes)
print(len(node_ids))

# Indexing
first_id = node_ids[0]
last_id = node_ids[-1]

# Slicing
subset_ids = node_ids[:5]

# To list
ids_list = node_ids.to_list()
```

### Using with Subgraphs

```python
# Get IDs from filtered nodes
young = g.nodes[g.nodes["age"] < 30]
young_ids = young.node_ids()

# Use IDs to select nodes
selected = g.nodes[young_ids.to_list()]
```

---

## EdgesArray: Edge IDs

### Getting EdgesArray

```python
g = gr.generators.karate_club()

# Get all edge IDs
edge_ids = g.edges.ids()  # EdgesArray

# From subgraph
heavy = g.edges[g.edges["weight"] > 3]
heavy_ids = heavy.edge_ids()
```

### Operations

```python
# Length
print(len(edge_ids))

# Access
first_edge = edge_ids[0]

# To list
ids_list = edge_ids.to_list()
```

---

## Common Patterns

### Pattern 1: Statistical Analysis

```python
# Get numeric attributes
ages = g.nodes["age"]
weights = g.edges["weight"]

# Compute statistics
stats = {
    'age': {
        'mean': ages.mean(),
        'std': ages.std(),
        'min': ages.min(),
        'max': ages.max()
    },
    'weight': {
        'mean': weights.mean(),
        'sum': weights.sum()
    }
}

print(stats)
```

### Pattern 2: Filtering by Value

```python
# Get array
ages = g.nodes["age"]

# Find indices where condition is true
# (Note: May need to do this via graph filtering)

# Better approach: filter graph first
young = g.nodes[g.nodes["age"] < 30]
young_ages = young.nodes["age"]

print(f"Young ages: {young_ages.to_list()}")
```

### Pattern 3: Unique Value Analysis

```python
roles = g.nodes["role"]

# Get unique values
unique_roles = roles.unique()
num_roles = roles.nunique()

print(f"{num_roles} unique roles:")
for role in unique_roles.to_list():
    # Count occurrences
    role_nodes = g.nodes[g.nodes["role"] == role]
    print(f"  {role}: {role_nodes.node_count()} nodes")
```

### Pattern 4: Aggregation

```python
# Get values
ages = g.nodes["age"]
weights = g.edges["weight"]

# Aggregate
total_age = ages.sum()
avg_age = ages.mean()
total_weight = weights.sum()

# Summary
print(f"Total age: {total_age}")
print(f"Average age: {avg_age:.1f}")
print(f"Total edge weight: {total_weight}")
```

### Pattern 5: Combining Arrays

```python
# Get multiple attributes
names = g.nodes["name"]
ages = g.nodes["age"]
roles = g.nodes["role"]

# Zip together
for name, age, role in zip(names.to_list(), ages.to_list(), roles.to_list()):
    print(f"{name} ({age}): {role}")
```

### Pattern 6: Exporting Arrays

```python
# Get as list
ages_list = g.nodes["age"].to_list()

# Use in other libraries
import numpy as np
ages_np = np.array(ages_list)

import pandas as pd
df = pd.DataFrame({
    'name': g.nodes["name"].to_list(),
    'age': ages_list
})
```

### Pattern 7: Edge Endpoint Analysis

```python
# Get sources and targets
sources = g.edges.sources()  # NumArray
targets = g.edges.targets()  # NumArray

# Analyze
print(f"Unique sources: {sources.nunique()}")
print(f"Unique targets: {targets.nunique()}")

# Most common source
from collections import Counter
source_counts = Counter(sources.to_list())
most_active = source_counts.most_common(5)

print("Most active sources:")
for node_id, count in most_active:
    print(f"  Node {node_id}: {count} outgoing edges")
```

---

## Array vs Table vs Subgraph

### When to Use Each

**Use Arrays when:**
- ✅ Need single column of data
- ✅ Statistical operations (mean, sum, etc.)
- ✅ Working with one attribute at a time
- ✅ Memory efficiency (single column)

**Use Tables when:**
- ✅ Need multiple columns
- ✅ Exporting to CSV/pandas
- ✅ Tabular analysis (sorting, grouping)

**Use Subgraphs when:**
- ✅ Filtering nodes/edges
- ✅ Need graph structure preserved
- ✅ Want to chain operations

### Comparison

```python
# Array - single column
ages = g.nodes["age"]  # NumArray
mean = ages.mean()

# Table - multiple columns
table = g.nodes.table()  # NodesTable
df = table.to_pandas()

# Subgraph - graph structure
young = g.nodes[g.nodes["age"] < 30]  # Subgraph
count = young.node_count()
```

---

## Performance Considerations

### Memory

Arrays are **views** into the attribute pool:

```python
# Get array - creates view, not copy
ages = g.nodes["age"]  # View into GraphPool

# Low memory overhead
# Arrays only store references to columns
```

### Operations

Statistical operations are optimized:

```python
# ✓ Fast - vectorized operation
ages = g.nodes["age"]
mean = ages.mean()

# ✗ Slow - Python loop
total = sum(g.nodes[nid]["age"] for nid in g.nodes.ids())
```

### When to Convert to List

```python
# Keep as array for stats
ages = g.nodes["age"]
mean = ages.mean()  # Fast

# Convert to list only when needed
ages_list = ages.to_list()
for age in ages_list:
    # Process individually
    pass
```

---

## Array Methods Summary

### BaseArray Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `count()` | `int` | Number of elements |
| `first()` | value | First element |
| `last()` | value | Last element |
| `head(n)` | `BaseArray` | First n elements |
| `tail(n)` | `BaseArray` | Last n elements |
| `is_empty()` | `bool` | Check if empty |
| `dtype()` | `str` | Data type |
| `unique()` | `BaseArray` | Unique values |
| `nunique()` | `int` | Count unique |
| `to_list()` | `list` | Convert to list |
| `[i]` | value | Index access |
| `[:n]` | `BaseArray` | Slicing |

### NumArray Additional Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `mean()` | `float` | Average |
| `sum()` | `float` | Sum |
| `min()` | `float` | Minimum |
| `max()` | `float` | Maximum |
| `std()` | `float` | Standard deviation |
| `var()` | `float` | Variance |

### NodesArray / EdgesArray

Same as BaseArray, specialized for node/edge IDs.

---

## Quick Reference

### Getting Arrays

```python
# From graph
ages = g.nodes["age"]          # NumArray (numeric attr)
names = g.nodes["name"]        # BaseArray (any attr)
node_ids = g.nodes.ids()       # NodesArray
edge_ids = g.edges.ids()       # EdgesArray

# From subgraph
sub = g.nodes[:10]
sub_ages = sub.nodes["age"]    # NumArray
sub_ids = sub.node_ids()       # NumArray

# From accessor
ids = g.nodes.array()          # NodesArray
```

### Common Operations

```python
# Statistics (NumArray)
mean = ages.mean()
std = ages.std()
total = ages.sum()

# Inspection (all arrays)
first = arr.first()
last = arr.last()
count = arr.count()
unique = arr.unique()

# Conversion
list_data = arr.to_list()

# Access
value = arr[0]
slice = arr[:5]
```

---

## See Also

- **[NumArray API Reference](../api/numarray.md)**: Numeric array operations
- **[NumArray API Reference](../api/numarray.md)**: Numeric array methods
- **[NodesArray API Reference](../api/nodesarray.md)**: Node ID arrays
- **[EdgesArray API Reference](../api/edgesarray.md)**: Edge ID arrays
- **[Tables Guide](tables.md)**: Multi-column tabular data
- **[Accessors Guide](accessors.md)**: Getting arrays from accessors
- **[Graph Core Guide](graph-core.md)**: Attribute access patterns

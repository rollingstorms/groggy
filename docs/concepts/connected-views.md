# Connected Views: The Transformation Graph

## Everything is a Graph

Even Groggy itself is a graph. The objects in the API are **nodes**, and the methods that transform one object into another are **edges**.

This conceptual model makes Groggy easier to learn: once you understand which objects can transform into which others, the entire API becomes intuitive.

---

## The Object Transformation Graph

```
                         ┌─────────┐
                         │  Graph  │
                         └────┬────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ↓                  ↓                  ↓
      ┌────────┐         ┌──────────┐      ┌─────────┐
      │Subgraph│         │GraphTable│      │BaseArray│
      └────┬───┘         └────┬─────┘      └────┬────┘
           │                  │                  │
           ↓                  ↓                  ↓
    ┌────────────┐       ┌────────┐        ┌─────────┐
    │SubgraphArr │       │NodesTab│        │ NumArray│
    └──────┬─────┘       │EdgesTab│        └─────────┘
           │             └────────┘
           ↓
      ┌────────┐
      │  Table │
      └────────┘
```

---

## Core Transformations

### From Graph

The Graph is the starting point for most operations.

#### Graph → Subgraph
Create a view into the graph:

```python
# Slice
sub = g.nodes[:10]              # First 10 nodes

# Filter
sub = g.nodes[g.nodes["age"] > 30]

# Specific IDs
sub = g.nodes[[0, 5, 10]]

# Explicit subgraph
sub = g.subgraph(nodes=[0, 1, 2])
```

#### Graph → GraphTable
Convert to tabular form:

```python
table = g.table()  # GraphTable with nodes + edges tables
```

#### Graph → BaseArray
Extract attribute columns:

```python
names = g["name"]              # All names
ages = g.nodes["age"]          # All ages
weights = g.edges["weight"]    # All edge weights
```

#### Graph → GraphMatrix
Matrix representations:

```python
A = g.to_matrix()              # Adjacency matrix
L = g.laplacian_matrix()       # Laplacian
D = g.degree_matrix()          # Degree matrix
```

---

### From Subgraph

Subgraphs are views that can transform similarly to graphs.

#### Subgraph → Graph
Materialize as a new graph:

```python
sub = g.nodes[:10]
new_graph = sub.to_graph()  # Independent copy
```

#### Subgraph → GraphTable
Convert to table:

```python
sub = g.nodes[g.nodes["active"] == True]
table = sub.table()
```

#### Subgraph → SubgraphArray
Many algorithms return arrays of subgraphs:

```python
components = g.connected_components()  # SubgraphArray
# Each component is a subgraph
```

#### Subgraph → BaseArray
Extract attributes from the subgraph:

```python
sub = g.nodes[:10]
names = sub["name"]  # Names of first 10 nodes
```

---

### From SubgraphArray

SubgraphArray is the key to delegation chains.

#### SubgraphArray → SubgraphArray (Filtering)
Chain operations while staying in the same type:

```python
components = g.connected_components()  # SubgraphArray

large = components.filter(lambda c: len(c.nodes) > 10)  # SubgraphArray
sampled = large.sample(5)                               # SubgraphArray
expanded = sampled.neighborhood(depth=2)                # SubgraphArray
```

#### SubgraphArray → GraphTable
Aggregate subgraphs into a table:

```python
components = g.connected_components()
table = components.table()  # GraphTable of all components
```

#### SubgraphArray → Subgraph
Extract individual subgraphs:

```python
components = g.connected_components()
first = components[0]           # First component (Subgraph)
last = components.last()        # Last component (Subgraph)
biggest = components.sorted_by_size().first()
```

---

### From GraphTable

GraphTable unifies node and edge tables.

#### GraphTable → Graph
Reconstruct a graph:

```python
table = g.table()
new_graph = table.to_graph()
```

#### GraphTable → NodesTable / EdgesTable
Access individual tables:

```python
table = g.table()
nodes = table.nodes  # NodesTable
edges = table.edges  # EdgesTable
```

#### GraphTable → BaseArray
Extract columns:

```python
table = g.table()
ages = table["age"]  # Get age column
```

#### GraphTable → AggregationResult
Aggregate data:

```python
table = g.table()
result = table.agg({
    "age": "mean",
    "weight": "sum"
})
```

---

### From BaseArray

Arrays provide columnar access to attributes.

#### BaseArray → NumArray
Numeric arrays unlock statistical operations:

```python
ages = g.nodes["age"]  # BaseArray
if ages.is_numeric():
    num_ages = ages.to_numeric()  # NumArray
    mean = num_ages.mean()
    std = num_ages.std()
```

#### BaseArray → Table
Convert array to single-column table:

```python
names = g["name"]
table = names.to_table()  # Table with "name" column
```

#### BaseArray → Python List
Materialize as Python list:

```python
names = g["name"]
name_list = names.to_list()  # ['Alice', 'Bob', ...]
```

---

### From GraphMatrix

Matrices represent graph structure or embeddings.

#### GraphMatrix → NumArray
Flatten or extract:

```python
A = g.to_matrix()
values = A.to_array()  # Flatten to 1D array
row = A.get_row(0)     # NumArray for row 0
```

#### GraphMatrix → Graph
Convert matrix back to graph:

```python
A = g.to_matrix()
new_graph = A.to_graph()  # Reconstruct from adjacency
```

#### GraphMatrix → NumPy
Export to numpy:

```python
A = g.to_matrix()
np_array = A.to_numpy()  # numpy ndarray
```

---

## Delegation Chain Examples

### Example 1: Component Analysis

```python
# Find large components, expand neighborhoods, summarize
result = (
    g.connected_components()       # Graph → SubgraphArray
     .filter(lambda c: len(c) > 5) # SubgraphArray → SubgraphArray
     .sample(3)                    # SubgraphArray → SubgraphArray
     .neighborhood(depth=2)        # SubgraphArray → SubgraphArray
     .table()                      # SubgraphArray → GraphTable
     .agg({"weight": "mean"})      # GraphTable → AggregationResult
)
```

**Transformation path:**
```
Graph → SubgraphArray → SubgraphArray → SubgraphArray →
SubgraphArray → GraphTable → AggregationResult
```

### Example 2: Attribute Processing

```python
# Get ages, filter, compute statistics
mean_adult_age = (
    g.nodes["age"]                 # Graph → BaseArray
     .filter(lambda x: x >= 18)    # BaseArray → BaseArray
     .to_numeric()                 # BaseArray → NumArray
     .mean()                       # NumArray → float
)
```

**Transformation path:**
```
Graph → BaseArray → BaseArray → NumArray → float
```

### Example 3: Subgraph to Analysis

```python
# Filter nodes, convert to subgraph, analyze
young_network = (
    g.nodes[g.nodes["age"] < 30]   # Graph → Subgraph
     .to_graph()                   # Subgraph → Graph
)

young_network.connected_components(inplace=True)
```

**Transformation path:**
```
Graph → Subgraph → Graph (with algorithms applied)
```

---

## Understanding Method Delegation

### How Delegation Works

When you call a method on a SubgraphArray, it might:

1. **Transform the array**: Returns another SubgraphArray
2. **Change type**: Returns a different object type
3. **Extract element**: Returns a single Subgraph

```python
components = g.connected_components()  # SubgraphArray

# 1. Transform array
filtered = components.sample(5)  # Returns SubgraphArray

# 2. Change type
table = components.table()  # Returns GraphTable

# 3. Extract element
first = components[0]  # Returns Subgraph
```

### Type Signatures Matter

Understanding return types helps you chain methods:

```python
g.connected_components()      # → SubgraphArray
  .sample(5)                  # SubgraphArray → SubgraphArray ✓
  .table()                    # SubgraphArray → GraphTable ✓
  .sample(3)                  # ❌ GraphTable has no sample() method
```

This error is prevented by understanding the transformation graph.

---

## The Power of Views

### Views Don't Copy Data

```python
# No data copying occurs here
sub = g.nodes[:1000]           # View
table = sub.table()            # View of view
array = table["name"]          # View of column

# Data is only copied on explicit materialization
graph_copy = sub.to_graph()    # Now data is copied
python_list = array.to_list()  # Now data is copied
```

### Immutable Views Prevent Bugs

```python
sub = g.nodes[:10]

# Modifying the subgraph view doesn't affect parent
# (because subgraphs are immutable views)

# To modify, you must be explicit:
sub_graph = sub.to_graph()  # Materialize
sub_graph.add_node()        # Modify the copy
```

---

## Common Transformation Patterns

### Pattern 1: Filter → Process → Aggregate

```python
result = (
    g.nodes[condition]          # Filter
     .to_subgraph()            # Process
     .table()                  # Convert
     .agg({"attr": "mean"})    # Aggregate
)
```

### Pattern 2: Algorithm → Sample → Analyze

```python
sampled_components = (
    g.connected_components()    # Algorithm
     .sorted_by_size()         # Sort
     .sample(5)                # Sample
     .table()                  # Convert
)
```

### Pattern 3: Extract → Transform → Export

```python
ages = (
    g.nodes["age"]             # Extract
     .to_numeric()             # Transform
     .to_numpy()               # Export
)
```

---

## Transformation Cheat Sheet

### Quick Reference

| From | Method | To | Use Case |
|------|--------|-----|----------|
| Graph | `nodes[...]` | Subgraph | Filter nodes |
| Graph | `table()` | GraphTable | View as table |
| Graph | `["attr"]` | BaseArray | Get column |
| Graph | `to_matrix()` | GraphMatrix | Linear algebra |
| Subgraph | `to_graph()` | Graph | Materialize |
| Subgraph | `table()` | GraphTable | View as table |
| SubgraphArray | `sample(n)` | SubgraphArray | Random sample |
| SubgraphArray | `table()` | GraphTable | Aggregate |
| SubgraphArray | `[i]` | Subgraph | Get element |
| GraphTable | `to_graph()` | Graph | Reconstruct |
| GraphTable | `agg({...})` | AggregationResult | Statistics |
| BaseArray | `to_numeric()` | NumArray | Math ops |
| BaseArray | `to_list()` | list | Export to Python |
| GraphMatrix | `to_numpy()` | ndarray | NumPy ops |

---

## Learning the Graph

The best way to learn Groggy is to internalize the transformation graph:

1. **Start with Graph**: Most operations begin here
2. **Know your current type**: What object do you have?
3. **Know where you want to go**: What object do you need?
4. **Find the path**: What methods connect them?

**Example thought process:**
```
I have: Graph
I want: Mean of ages > 30

Path:
1. Graph → BaseArray (via g.nodes["age"])
2. BaseArray → BaseArray (via .filter())
3. BaseArray → NumArray (via .to_numeric())
4. NumArray → float (via .mean())

Code:
g.nodes["age"].filter(lambda x: x > 30).to_numeric().mean()
```

---

## Next Steps

Now that you understand connected views and transformations:

- **[User Guide](../guide/graph-core.md)**: Practice these transformations with real examples
- **[API Reference](../api/graph.md)**: Detailed documentation for each object type
- **[User Guides](../guide/graph-core.md)**: See transformation chains in action

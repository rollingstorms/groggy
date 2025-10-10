# Accessors: NodesAccessor and EdgesAccessor

Accessors provide a unified interface for working with nodes and edges in Groggy. When you write `g.nodes` or `g.edges`, you're using accessors.

---

## What are Accessors?

Accessors are **entry points** for node and edge operations:

```python
import groggy as gr

g = gr.generators.karate_club()

# g.nodes is a NodesAccessor
nodes = g.nodes
print(type(nodes))  # <class 'groggy.NodesAccessor'>

# g.edges is an EdgesAccessor
edges = g.edges
print(type(edges))  # <class 'groggy.EdgesAccessor'>
```

**Think of accessors as:**
- **Namespaces** for node/edge operations
- **Gateways** to filtering and transformation
- **Collection-like objects** that support indexing and slicing

---

## NodesAccessor: Working with Nodes

### Basic Access

Get the nodes accessor:

```python
g = gr.Graph()
alice = g.add_node(name="Alice", age=29)
bob = g.add_node(name="Bob", age=55)

# Access via .nodes
nodes = g.nodes

# Count nodes
print(len(nodes))  # 2

# Check in graph
print(nodes.ids())  # NumArray([0, 1])
```

### Getting Node Attributes

Access attributes for all nodes:

```python
# Get entire column
names = g.nodes["name"]  # BaseArray
ages = g.nodes["age"]    # NumArray (numeric)

# Check what's available
attr_names = g.nodes.attribute_names()
print(attr_names)  # ['name', 'age']
```

### Filtering Nodes

Use boolean indexing to filter:

```python
g = gr.Graph()
alice = g.add_node(name="Alice", age=29, active=True)
bob = g.add_node(name="Bob", age=55, active=True)
carol = g.add_node(name="Carol", age=31, active=False)

# Single condition
young = g.nodes[g.nodes["age"] < 40]
print(f"Young nodes: {young.node_count()}")

# Multiple conditions
active_young = g.nodes[
    (g.nodes["age"] < 40) &
    (g.nodes["active"] == True)
]
print(f"Active young: {active_young.node_count()}")
```

**Returns a Subgraph:**

```python
sub = g.nodes[g.nodes["age"] > 30]
print(type(sub))  # <class 'groggy.Subgraph'>
```

### Slicing Nodes

Get subsets by position:

```python
# First 10 nodes
first_ten = g.nodes[:10]

# Nodes 5-15
middle = g.nodes[5:15]

# Every other node
evens = g.nodes[::2]

# Specific IDs
specific = g.nodes[[0, 5, 10]]
```

### Setting Node Attributes

Bulk update attributes:

```python
# Single node
g.nodes.set_attrs({alice: {"age": 30}})

# Multiple nodes
g.nodes.set_attrs({
    alice: {"age": 30, "verified": True},
    bob: {"verified": True}
})

# All nodes (via loop or comprehension)
updates = {nid: {"processed": True} for nid in g.nodes.ids()}
g.nodes.set_attrs(updates)
```

### Transformations

Convert to other types:

```python
# To table
nodes_table = g.nodes.table()  # NodesTable
df = nodes_table.to_pandas()

# To array
node_ids = g.nodes.ids()  # NumArray
node_array = g.nodes.array()  # NodesArray

# To matrix
node_matrix = g.nodes.matrix()  # GraphMatrix

# Get all nodes as subgraph
all_nodes = g.nodes.all()  # Subgraph
```

### Grouping Nodes

Group by attribute values:

```python
# Group by role
by_role = g.nodes.group_by("role")  # SubgraphArray

# Each element is a subgraph for that group
for i, group in enumerate(by_role):
    role = group.nodes["role"].head(1)[0]  # Get first value
    print(f"Group {i}: {group.node_count()} {role}s")
```

---

## EdgesAccessor: Working with Edges

### Basic Access

Get the edges accessor:

```python
g = gr.Graph()
alice = g.add_node(name="Alice")
bob = g.add_node(name="Bob")
e = g.add_edge(alice, bob, weight=5)

# Access via .edges
edges = g.edges

# Count edges
print(len(edges))  # 1

# Get IDs
print(edges.ids())  # NumArray([0])
```

### Getting Edge Attributes

Access attributes for all edges:

```python
# Get entire column
weights = g.edges["weight"]  # NumArray
types = g.edges["type"]      # BaseArray

# Check available attributes
attr_names = g.edges.attribute_names()
print(attr_names)  # ['weight', 'type']
```

### Filtering Edges

Use boolean indexing:

```python
g = gr.Graph()
n0 = g.add_node()
n1 = g.add_node()
n2 = g.add_node()

g.add_edge(n0, n1, weight=5, type="friend")
g.add_edge(n0, n2, weight=2, type="colleague")
g.add_edge(n1, n2, weight=1, type="friend")

# Heavy edges
heavy = g.edges[g.edges["weight"] > 3]
print(f"Heavy edges: {heavy.edge_count()}")

# By type
friendships = g.edges[g.edges["type"] == "friend"]
print(f"Friendships: {friendships.edge_count()}")

# Combined
heavy_friendships = g.edges[
    (g.edges["weight"] > 2) &
    (g.edges["type"] == "friend")
]
```

**Returns a Subgraph:**

```python
sub = g.edges[g.edges["weight"] > 3]
print(type(sub))  # <class 'groggy.Subgraph'>

# Subgraph contains both nodes and filtered edges
print(f"Nodes: {sub.node_count()}, Edges: {sub.edge_count()}")
```

### Slicing Edges

Get subsets by position:

```python
# First 10 edges
first_ten = g.edges[:10]

# Edges 5-15
middle = g.edges[5:15]

# Specific edge IDs
specific = g.edges[[0, 2, 4]]
```

### Edge Endpoints

Get source and target nodes:

```python
# All sources
sources = g.edges.sources()  # NumArray
print(sources.head())

# All targets
targets = g.edges.targets()  # NumArray
print(targets.head())

# Zip them together
for src, tgt in zip(sources, targets):
    print(f"Edge: {src} → {tgt}")
```

### Setting Edge Attributes

Bulk update:

```python
e1 = g.add_edge(n0, n1)
e2 = g.add_edge(n1, n2)

# Single edge
g.edges.set_attrs({e1: {"weight": 10}})

# Multiple edges
g.edges.set_attrs({
    e1: {"weight": 10, "validated": True},
    e2: {"weight": 5, "validated": False}
})
```

### Transformations

Convert to other types:

```python
# To table
edges_table = g.edges.table()  # EdgesTable
df = edges_table.to_pandas()

# To array
edge_ids = g.edges.ids()  # NumArray
edge_array = g.edges.array()  # EdgesArray

# To matrix
edge_matrix = g.edges.matrix()  # GraphMatrix
weight_matrix = g.edges.weight_matrix()  # GraphMatrix

# Get all edges as subgraph
all_edges = g.edges.all()  # Subgraph
```

### Grouping Edges

Group by attribute:

```python
# Group by type
by_type = g.edges.group_by("type")  # SubgraphArray

# Each element is a subgraph for that type
for i, group in enumerate(by_type):
    edge_type = group.edges["type"].head(1)[0]
    print(f"Type {edge_type}: {group.edge_count()} edges")
```

---

## Common Patterns

### Pattern 1: Filter Both Nodes and Edges

```python
# Active nodes
active_nodes = g.nodes[g.nodes["active"] == True]

# Strong edges
strong_edges = g.edges[g.edges["weight"] > 5]

# Combine: active nodes with strong edges
# Get nodes, then filter edges
active_strong = active_nodes.edges[
    active_nodes.edges["weight"] > 5
]
```

### Pattern 2: Bulk Attribute Update

```python
# Get IDs
node_ids = g.nodes.ids()

# Compute new values
ages = g.nodes["age"]
updated_ages = ages + 1  # Everyone ages a year

# Update in bulk
updates = {
    int(nid): {"age": int(new_age)}
    for nid, new_age in zip(node_ids, updated_ages)
}
g.nodes.set_attrs(updates)
```

### Pattern 3: Attribute Statistics

```python
# Node stats
mean_age = g.nodes["age"].mean()
max_age = g.nodes["age"].max()
age_std = g.nodes["age"].std()

print(f"Age: {mean_age:.1f} ± {age_std:.1f}, max: {max_age}")

# Edge stats
mean_weight = g.edges["weight"].mean()
total_weight = g.edges["weight"].sum()

print(f"Weight: mean={mean_weight:.2f}, total={total_weight}")
```

### Pattern 4: Conditional Updates

```python
# Find nodes to update
old_nodes = g.nodes[g.nodes["age"] > 50]
old_ids = old_nodes.node_ids()

# Update only those nodes
updates = {int(nid): {"category": "senior"} for nid in old_ids}
g.nodes.set_attrs(updates)
```

### Pattern 5: Accessor Chaining

```python
# Start with accessor
result = (
    g.nodes
     .table()                      # → NodesTable
     .head(100)                    # First 100 rows
     .to_pandas()                  # → DataFrame
)

# Edge version
edge_result = (
    g.edges
     .table()                      # → EdgesTable
     .to_pandas()                  # → DataFrame
)
```

### Pattern 6: Group and Analyze

```python
# Group nodes by role
by_role = g.nodes.group_by("role")

# Analyze each group
for group in by_role:
    role = group.nodes["role"].head(1)[0]
    count = group.node_count()
    avg_age = group.nodes["age"].mean()

    print(f"{role}: {count} people, avg age {avg_age:.1f}")
```

### Pattern 7: Edge Source/Target Analysis

```python
# Get all edges
sources = g.edges.sources()
targets = g.edges.targets()

# Find nodes with outgoing edges
unique_sources = set(sources)
print(f"Nodes with outgoing edges: {len(unique_sources)}")

# Find popular targets (in-degree)
from collections import Counter
target_counts = Counter(targets)
most_popular = target_counts.most_common(5)

print("Most popular targets:")
for node_id, count in most_popular:
    print(f"  Node {node_id}: {count} incoming edges")
```

---

## Accessor vs Direct Access

### What Accessors Are NOT

Accessors are not lists or dictionaries:

```python
# ❌ Not a list
# for node in g.nodes:  # Won't iterate node IDs
#     print(node)

# ✓ Get IDs first
for node_id in g.nodes.ids():
    print(node_id)

# ❌ Not a dict
# g.nodes[0]  # Doesn't get node 0's attributes

# ✓ Use attribute access
# g.nodes[0]["name"]  # Also doesn't work this way

# ✓ Get attributes via columns
names = g.nodes["name"]
first_name = names[0]
```

### What Accessors ARE

Accessors are **gateways**:

```python
# Gateway to filtering
filtered = g.nodes[g.nodes["age"] > 30]  # Subgraph

# Gateway to slicing
subset = g.nodes[:10]  # Subgraph

# Gateway to attributes
ages = g.nodes["age"]  # BaseArray/NumArray

# Gateway to transformations
table = g.nodes.table()  # NodesTable
ids = g.nodes.ids()      # NumArray
```

---

## Performance Tips

### Bulk Operations

Always prefer bulk operations:

```python
# ❌ Slow: Many small updates
for node_id in g.nodes.ids():
    g.nodes.set_attrs({node_id: {"processed": True}})

# ✓ Fast: Single bulk update
updates = {int(nid): {"processed": True} for nid in g.nodes.ids()}
g.nodes.set_attrs(updates)
```

### Column Access

Get entire columns, not individual values:

```python
# ❌ Slower
ages = []
for node_id in g.nodes.ids():
    # Can't actually do this, need different approach
    pass

# ✓ Faster
ages = g.nodes["age"]  # Get entire column at once
mean = ages.mean()
```

### Filter Once

Combine conditions instead of chaining:

```python
# Less efficient
result = (
    g.nodes[g.nodes["active"] == True]
     .nodes[g.nodes["age"] > 30]
)

# More efficient
result = g.nodes[
    (g.nodes["active"] == True) & (g.nodes["age"] > 30)
]
```

---

## Accessors on Subgraphs

Subgraphs also have accessors:

```python
# Create subgraph
sub = g.nodes[:100]

# Use accessors on subgraph
young = sub.nodes[sub.nodes["age"] < 30]
heavy_edges = sub.edges[sub.edges["weight"] > 5]

# Get attributes
ages = sub.nodes["age"]
weights = sub.edges["weight"]

# Transformations
node_table = sub.nodes.table()
edge_ids = sub.edges.ids()
```

**Note:** When filtering on a subgraph, filter using the subgraph's accessors:

```python
sub = g.nodes[:100]

# ✓ Correct: filter on sub
filtered = sub.nodes[sub.nodes["age"] < 30]

# ❌ Wrong: filter on g
# filtered = sub.nodes[g.nodes["age"] < 30]  # Mismatch!
```

---

## Common Gotchas

### 1. Accessors Are Not Iterables

```python
# ❌ Can't iterate directly
# for node in g.nodes:
#     print(node)

# ✓ Get IDs first
for node_id in g.nodes.ids():
    print(node_id)
```

### 2. Filtering Returns Subgraph

```python
# Filtering returns Subgraph, not accessor
result = g.nodes[g.nodes["age"] > 30]
print(type(result))  # Subgraph, not NodesAccessor

# To continue filtering, use result's accessor
further = result.nodes[result.nodes["active"] == True]
```

### 3. Slicing Returns Subgraph

```python
# Slicing also returns Subgraph
subset = g.nodes[:10]
print(type(subset))  # Subgraph

# Access subset's attributes via its accessor
ages = subset.nodes["age"]
```

### 4. Attribute Access Returns Array

```python
# Getting attribute returns array
ages = g.nodes["age"]
print(type(ages))  # NumArray, not NodesAccessor

# Arrays have different methods
mean_age = ages.mean()
max_age = ages.max()
```

---

## Quick Reference

### NodesAccessor Methods

| Operation | Example | Returns |
|-----------|---------|---------|
| Filter | `g.nodes[g.nodes["age"] > 30]` | `Subgraph` |
| Slice | `g.nodes[:10]` | `Subgraph` |
| Attribute | `g.nodes["name"]` | `BaseArray` |
| IDs | `g.nodes.ids()` | `NumArray` |
| Table | `g.nodes.table()` | `NodesTable` |
| Array | `g.nodes.array()` | `NodesArray` |
| Group | `g.nodes.group_by("role")` | `SubgraphArray` |
| All | `g.nodes.all()` | `Subgraph` |
| Set attrs | `g.nodes.set_attrs({id: {...}})` | None |

### EdgesAccessor Methods

| Operation | Example | Returns |
|-----------|---------|---------|
| Filter | `g.edges[g.edges["weight"] > 5]` | `Subgraph` |
| Slice | `g.edges[:10]` | `Subgraph` |
| Attribute | `g.edges["weight"]` | `NumArray` |
| IDs | `g.edges.ids()` | `NumArray` |
| Sources | `g.edges.sources()` | `NumArray` |
| Targets | `g.edges.targets()` | `NumArray` |
| Table | `g.edges.table()` | `EdgesTable` |
| Array | `g.edges.array()` | `EdgesArray` |
| Group | `g.edges.group_by("type")` | `SubgraphArray` |
| All | `g.edges.all()` | `Subgraph` |
| Set attrs | `g.edges.set_attrs({id: {...}})` | None |

---

## See Also

- **[NodesAccessor API Reference](../api/nodesaccessor.md)**: Complete method reference
- **[EdgesAccessor API Reference](../api/edgesaccessor.md)**: Complete method reference
- **[Graph Core Guide](graph-core.md)**: Graph operations
- **[Subgraphs Guide](subgraphs.md)**: Working with filtered views
- **[Arrays Guide](arrays.md)**: Array operations on attributes
- **[Tables Guide](tables.md)**: Tabular data operations

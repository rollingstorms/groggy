# Working with Subgraphs

A `Subgraph` is an **immutable view** into a portion of a Graph. It doesn't copy data—it just tracks which nodes and edges belong to the view.

---

## What is a Subgraph?

Think of a Subgraph as a **window** into your graph:

```python
import groggy as gr

# Full graph
g = gr.generators.karate_club()
print(f"Full graph: {g.node_count()} nodes")

# Subgraph - just a view
sub = g.nodes[:10]  # First 10 nodes
print(f"Subgraph: {sub.node_count()} nodes")
# No data was copied!
```

**Key characteristics:**
- **View, not copy**: References parent graph, no data duplication
- **Immutable**: Cannot modify the subgraph view directly
- **Cheap to create**: O(1) to create, stores only node/edge IDs
- **Transformable**: Can convert to Graph, Table, Array, or Matrix

---

## Creating Subgraphs

### Via Slicing

Use Python slice notation on nodes or edges:

```python
g = gr.generators.karate_club()

# First 10 nodes
first_ten = g.nodes[:10]

# Nodes 5 through 15
middle = g.nodes[5:15]

# Every other node
every_other = g.nodes[::2]

# First 5 edges
first_edges = g.edges[:5]
```

### Via Filtering

Filter by attribute conditions:

```python
g = gr.Graph()
alice = g.add_node(name="Alice", age=29, role="Engineer")
bob = g.add_node(name="Bob", age=55, role="Manager")
carol = g.add_node(name="Carol", age=31, role="Engineer")

# Engineers only
engineers = g.nodes[g.nodes["role"] == "Engineer"]
print(f"Engineers: {engineers.node_count()}")  # 2

# People over 30
older = g.nodes[g.nodes["age"] > 30]
print(f"Over 30: {older.node_count()}")  # 2
```

**Boolean operations:**

```python
# Combine conditions with & (and) or | (or)
young_engineers = g.nodes[
    (g.nodes["role"] == "Engineer") & (g.nodes["age"] < 30)
]

# Note: Use & and | for boolean arrays, not 'and'/'or'
```

### Via Specific IDs

Select nodes or edges by their IDs:

```python
# Specific node IDs
sub = g.nodes[[0, 5, 10, 15]]

# Specific edge IDs
edge_sub = g.edges[[0, 1, 2]]
```

### Via Explicit Subgraph Method

Use the `subgraph()` method for full control:

```python
# Specify both nodes and edges
sub = g.subgraph(
    nodes=[0, 1, 2, 3],
    edges=[0, 1]  # Only include specific edges
)
```

**Induced subgraphs (default):**

By default, selecting nodes automatically includes all edges between those nodes:

```python
# Selects nodes 0, 1, 2 and ALL edges between them
sub = g.nodes[[0, 1, 2]]

# This is an "induced" subgraph
```

---

## Working with Subgraphs

### Inspecting Subgraphs

Get basic information:

```python
sub = g.nodes[:100]

# Counts
print(f"Nodes: {sub.node_count()}")
print(f"Edges: {sub.edge_count()}")

# IDs
node_ids = sub.node_ids()  # NumArray
edge_ids = sub.edge_ids()  # NumArray

# Check if empty
if sub.is_empty():
    print("No nodes in subgraph")

# Check connectivity
if sub.is_connected():
    print("Subgraph is connected")
```

### Accessing Attributes

Access attributes just like on a Graph:

```python
# Via attribute name
names = sub["name"]  # BaseArray
ages = sub["age"]    # NumArray (if numeric)

# Via accessors
names = sub.nodes["name"]
weights = sub.edges["weight"]

# Statistical operations on numeric attributes
mean_age = sub.nodes["age"].mean()
max_weight = sub.edges["weight"].max()
```

### Graph Properties

Subgraphs support many graph analysis methods:

```python
# Degree
degrees = sub.degree()      # NumArray
in_deg = sub.in_degree()
out_deg = sub.out_degree()

# Density
density = sub.density()
print(f"Edge density: {density:.3f}")

# Adjacency
adj_matrix = sub.adjacency_matrix()  # GraphMatrix
adj_list = sub.adjacency_list()      # dict
```

---

## Transforming Subgraphs

### Subgraph → Graph

Materialize the view as an independent graph:

```python
sub = g.nodes[:100]

# Convert to full Graph (copies data)
new_graph = sub.to_graph()

# Now you can modify it
new_graph.add_node(name="NewPerson")
new_graph.commit("Added new person")
```

**When to materialize:**
- Need to modify the subset
- Want to persist it independently
- Need version control on the subset

### Subgraph → Table

Convert to tabular representation:

```python
sub = g.nodes[g.nodes["active"] == True]

# Get as table
table = sub.table()  # GraphTable

# Access node/edge tables
nodes_df = table.nodes.to_pandas()
edges_df = table.edges.to_pandas()

# Or directly
nodes_table = sub.nodes.table()  # NodesTable
```

**When to use tables:**
- Exporting to CSV/Parquet
- Analysis with pandas
- Tabular aggregations

### Subgraph → Matrix

Get matrix representations:

```python
# Adjacency matrix
A = sub.adjacency_matrix()
# or
A = sub.to_matrix()

# Adjacency list
adj = sub.adjacency_list()
print(adj[0])  # Neighbors of node 0
```

### Subgraph → Arrays

Extract specific data as arrays:

```python
# Node/edge IDs
node_ids = sub.node_ids()  # NumArray
edge_ids = sub.edge_ids()  # NumArray

# Attributes
ages = sub.nodes["age"]    # NumArray
names = sub.nodes["name"]  # BaseArray

# Accessors
nodes_accessor = sub.nodes  # NodesAccessor
edges_accessor = sub.edges  # EdgesAccessor
```

---

## Running Algorithms on Subgraphs

Subgraphs support graph algorithms:

### Connected Components

```python
# Find components within the subgraph
components = sub.connected_components()  # SubgraphArray

# Check if connected
if sub.is_connected():
    print("Subgraph is a single component")
else:
    print(f"Found {len(components)} components")
```

### Sampling

```python
# Random sample of nodes
sample = sub.sample(n=50)  # Returns Subgraph

# Sample gives you another subgraph view
print(f"Sampled {sample.node_count()} nodes")
```

### Neighborhood Expansion

```python
# Expand to k-hop neighborhood
expanded = sub.neighborhood(depth=2)  # SubgraphArray

# This returns an array of neighborhoods around each node
```

---

## Common Patterns

### Pattern 1: Filter → Analyze → Export

```python
# Filter to subset
active_users = g.nodes[g.nodes["active"] == True]

# Analyze
mean_age = active_users["age"].mean()
num_users = active_users.node_count()
print(f"Active users: {num_users}, mean age: {mean_age:.1f}")

# Export
active_users.table().to_csv("active_users.csv")
```

### Pattern 2: Slice → Check → Expand

```python
# Start with specific nodes
seed_nodes = g.nodes[[0, 1, 2]]

# Check properties
if seed_nodes.is_connected():
    print("Seeds are connected")

# Expand to neighborhood
expanded = seed_nodes.neighborhood(depth=2)
```

### Pattern 3: Chain Multiple Filters

```python
# Progressive filtering
result = (
    g.nodes[g.nodes["active"] == True]   # Filter 1
     .nodes[g.nodes["age"] > 25]         # Filter 2 (Note: re-filter on g.nodes)
     .nodes[g.nodes["role"] == "Engineer"]  # Filter 3
)

# Alternative: combine conditions
result = g.nodes[
    (g.nodes["active"] == True) &
    (g.nodes["age"] > 25) &
    (g.nodes["role"] == "Engineer")
]
```

### Pattern 4: Compare Subgraphs

```python
# Create two views
engineers = g.nodes[g.nodes["role"] == "Engineer"]
senior = g.nodes[g.nodes["age"] > 40]

# Compare sizes
print(f"Engineers: {engineers.node_count()}")
print(f"Senior: {senior.node_count()}")

# Get stats
print(f"Engineers mean age: {engineers['age'].mean():.1f}")
print(f"Senior mean age: {senior['age'].mean():.1f}")
```

### Pattern 5: Temporary Working Set

```python
# Create temporary view for analysis
temp = g.nodes[:1000]  # First 1000 nodes

# Do expensive computation on subset
components = temp.connected_components()
density = temp.density()

# Discard when done (temp is just a view)
# No cleanup needed!
```

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Create subgraph | O(1) | Just stores node/edge IDs |
| `node_count()` | O(1) | BitSet count |
| `contains_node()` | O(1) | BitSet lookup |
| `to_graph()` | O(V + E + A) | Full copy, expensive |
| `table()` | O(1) | Creates view |
| Access attributes | O(1) per access | Via parent graph |

### Memory

- **Subgraph storage**: ~40 bytes + 2 BitSets
- **BitSet overhead**: ~(num_entities / 8) bytes
- **No attribute duplication**: Attributes stay in parent GraphPool
- **Materialization**: `to_graph()` creates full copy

### Optimization Tips

**1. Delay materialization:**

```python
# ✓ Stay as view
sub = g.nodes[:1000]
result = sub.table().agg({"age": "mean"})

# ✗ Unnecessary copy
graph = sub.to_graph()
result = graph.table().agg({"age": "mean"})
```

**2. Chain filters efficiently:**

```python
# Efficient - single filtered view
result = g.nodes[cond1].nodes[cond2].nodes[cond3]

# Better - combine conditions
result = g.nodes[cond1 & cond2 & cond3]
```

**3. Bulk operations over loops:**

```python
# ✓ Fast
ages = sub.nodes["age"]
mean_age = ages.mean()

# ✗ Slow
total = sum(sub.nodes[n]["age"] for n in sub.node_ids())
```

---

## Subgraph Limitations

### Cannot Modify Directly

Subgraphs are immutable views:

```python
sub = g.nodes[:10]

# ✗ Cannot modify subgraph
# sub.add_node()  # No such method

# ✓ Materialize first, then modify
new_graph = sub.to_graph()
new_graph.add_node(name="NewPerson")
```

### Must Access Parent Graph for Modifications

To modify nodes in a subgraph, work through the parent:

```python
sub = g.nodes[g.nodes["active"] == True]

# Get IDs from subgraph
node_ids = sub.node_ids()

# Modify via parent graph
for nid in node_ids:
    g.nodes.set_attrs({nid: {"processed": True}})

# Or use bulk operations
g.nodes.set_attrs({nid: {"processed": True} for nid in node_ids})
```

### Parent Graph Must Stay Alive

The parent graph must not be deleted while subgraph exists:

```python
def create_subgraph():
    g = gr.Graph()
    g.add_node(name="Alice")
    return g.nodes[:1]  # Returns subgraph

# ✗ Dangerous - g goes out of scope
# sub = create_subgraph()  # Parent graph deleted!

# ✓ Keep parent alive
g = gr.Graph()
g.add_node(name="Alice")
sub = g.nodes[:1]  # Safe - g still in scope
```

---

## When to Use Subgraphs

**Use Subgraphs when:**
- ✅ Filtering by conditions
- ✅ Working with a portion of a large graph
- ✅ Temporary analysis without modifications
- ✅ Building delegation chains
- ✅ Memory efficiency matters

**Use Graph when:**
- ❌ Need to modify structure
- ❌ Need version control
- ❌ Persisting for later use
- ❌ Independent lifecycle required

---

## See Also

- **[Subgraph API Reference](../api/subgraph.md)**: Complete method reference
- **[SubgraphArray Guide](subgraph-arrays.md)**: Working with collections of subgraphs
- **[Graph Core Guide](graph-core.md)**: Parent graph operations
- **[Accessors Guide](accessors.md)**: NodesAccessor and EdgesAccessor details
- **[Object Transformations](../concepts/connected-views.md)**: Delegation chains

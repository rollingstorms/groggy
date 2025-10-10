# Graph Core

The `Graph` is the foundational object in Groggy. It represents a complete graph with nodes, edges, and their attributes.

---

## Creating a Graph

### Empty Graph

Start with an empty graph:

```python
import groggy as gr

g = gr.Graph()
print(len(g.nodes))  # 0
print(len(g.edges))  # 0
```

### From Built-in Generators

Use generators for quick experimentation:

```python
# Karate club network (classic dataset)
g = gr.generators.karate_club()

# Complete graph
g = gr.generators.complete_graph(5)

# Erdős-Rényi random graph
g = gr.generators.erdos_renyi(n=100, p=0.05)

# Path graph
g = gr.generators.path_graph(10)
```

See the Integration Guide for more generator examples.

---

## Adding Nodes

### Single Node

Add one node at a time:

```python
# Node without attributes
node_id = g.add_node()
print(node_id)  # 0 (integer ID)

# Node with attributes
alice = g.add_node(name="Alice", age=29, role="Engineer")
bob = g.add_node(name="Bob", age=55, active=True)
```

**Key points:**
- `add_node()` returns an integer ID
- Use this ID to reference the node later
- Any keyword arguments become node attributes
- Attribute names are arbitrary—make up whatever you need

### Multiple Nodes

Add many nodes at once:

```python
# List of attribute dicts
nodes = [
    {"name": "Alice", "age": 29},
    {"name": "Bob", "age": 55},
    {"name": "Carol", "age": 31}
]

node_ids = g.add_nodes(nodes)
print(node_ids)  # [0, 1, 2]
```

### Node Attribute Types

Nodes support various attribute types:

```python
node = g.add_node(
    # Primitives
    count=42,                    # int
    score=3.14,                  # float
    name="Alice",                # str
    active=True,                 # bool

    # Collections (if supported)
    tags=["python", "rust"],     # list
    metadata={"key": "value"}    # dict
)
```

---

## Adding Edges

### Single Edge

Connect two nodes:

```python
alice = g.add_node(name="Alice")
bob = g.add_node(name="Bob")

# Edge without attributes
edge_id = g.add_edge(alice, bob)

# Edge with attributes
edge_id = g.add_edge(alice, bob, weight=5, type="friendship")
```

### Multiple Edges

Add many edges at once:

```python
edges = [
    {"src": 0, "dst": 1, "weight": 5},
    {"src": 0, "dst": 2, "weight": 2},
    {"src": 1, "dst": 2, "weight": 1}
]

edge_ids = g.add_edges(edges)
```

### Self-loops and Multiple Edges

```python
# Self-loop
g.add_edge(alice, alice)

# Multiple edges between same nodes
g.add_edge(alice, bob, type="friend")
g.add_edge(alice, bob, type="colleague")
```

---

## Accessing Nodes and Edges

### Check Existence

```python
alice = g.add_node(name="Alice")

# Check if node exists
if g.contains_node(alice):
    print("Alice exists")

# Check if edge exists
edge = g.add_edge(alice, bob)
if g.contains_edge(edge):
    print("Edge exists")
```

### Count Nodes and Edges

```python
print(f"Nodes: {len(g.nodes)}")
print(f"Edges: {len(g.edges)}")
```

**Note:** `g.nodes` and `g.edges` are accessors, not lists. See [Accessors](accessors.md) for details.

---

## Working with Attributes

### Get Attributes

Access node/edge attributes:

```python
alice = g.add_node(name="Alice", age=29)

# Via nodes accessor
name = g.nodes[alice]["name"]

# Get column for all nodes
names = g.nodes["name"]  # Returns BaseArray
print(names.head())
```

### Set Attributes

Update attributes:

```python
# Single attribute
g.nodes.set_attrs({alice: {"age": 30}})

# Multiple attributes
g.nodes.set_attrs({
    alice: {"age": 30, "active": True},
    bob: {"age": 56}
})
```

### Bulk Attribute Operations

Operate on entire columns:

```python
# Get all ages
ages = g.nodes["age"]  # BaseArray

# Statistics (if numeric)
mean_age = ages.mean()
max_age = ages.max()

# Filter
young_ages = ages.filter(lambda x: x < 30)
```

---

## Querying the Graph

### Filtering Nodes

Use Pandas-style boolean indexing:

```python
# Single condition
young_nodes = g.nodes[g.nodes["age"] < 30]

# Multiple conditions
active_engineers = g.nodes[
    (g.nodes["active"] == True) &
    (g.nodes["role"] == "Engineer")
]

# Check result
print(len(young_nodes))  # Number of matches
```

### Filtering Edges

Same pattern for edges:

```python
# Heavy edges
heavy = g.edges[g.edges["weight"] > 3]

# By type
friendships = g.edges[g.edges["type"] == "friendship"]
```

### Slicing

Get subsets by index:

```python
# First 10 nodes
first_ten = g.nodes[:10]

# Specific nodes
subset = g.nodes[[0, 5, 10, 15]]

# Every other node
evens = g.nodes[::2]
```

---

## Graph Algorithms

### Connected Components

Find connected components:

```python
# Modify graph in place
g.connected_components(inplace=True, label="component")

# Check component assignments
components = g.nodes["component"]
print(components.unique())  # [0, 1, 2, ...]

# Count components
num_components = len(components.unique())
```

Or get components as subgraphs:

```python
# Returns SubgraphArray
components = g.connected_components()

# Work with individual components
largest = components.sorted_by_size().first()
print(f"Largest component: {len(largest.nodes)} nodes")
```

### Other Algorithms

```python
# Shortest paths (example - check API for actual implementation)
paths = g.shortest_paths(source=alice, weight="weight")

# PageRank
ranks = g.pagerank(damping=0.85)

# Centrality measures
betweenness = g.betweenness_centrality()
```

See [Algorithms Guide](algorithms.md) for comprehensive coverage.

---

## State Management & Version Control

### Committing Changes

Groggy has Git-like version control:

```python
# Make changes
g.add_node(name="Alice")
g.add_node(name="Bob")

# Commit the state
commit_id = g.commit("Added Alice and Bob")

# Make more changes
g.add_edge(0, 1)

# Commit again
g.commit("Connected Alice and Bob")
```

### Branching

Create branches to experiment:

```python
# Create a branch
g.create_branch("experiment")

# Switch to branch
g.checkout_branch("experiment")

# Make experimental changes
g.add_node(name="Charlie")

# Switch back to main
g.checkout_branch("main")

# Charlie doesn't exist on main branch
```

### Viewing History

```python
# List branches
branches = g.branches()
print(branches)

# Get commit history
commits = g.history()
for commit in commits:
    print(f"{commit.id}: {commit.message}")
```

---

## Converting and Exporting

### To Table

View graph as tables:

```python
# Nodes table
nodes_table = g.nodes.table()
print(nodes_table.head())

# Edges table
edges_table = g.edges.table()
print(edges_table.head())

# Both
graph_table = g.table()  # GraphTable
```

### To Matrix

Get matrix representations:

```python
# Adjacency matrix
A = g.adj()
#or 
A = g.adjacency_matrix()

# Laplacian matrix
L = g.laplacian_matrix()

# Degree array
D = g.degree()
```

### Save and Load

```python
# Save entire graph (structure + attributes)
g.save_bundle("my_graph.bundle")

# Load
loaded = gr.GraphTable.load_bundle("my_graph.bundle")
g2 = loaded.to_graph()
```

Export to files:

```python
# Parquet (efficient)
g.nodes.table().to_parquet("nodes.parquet")
g.edges.table().to_parquet("edges.parquet")

# CSV (human-readable)
g.nodes.table().to_csv("nodes.csv")

# Pandas
df = g.nodes.table().to_pandas()
```

---

## Complete Example

Putting it all together:

```python
import groggy as gr

# 1. Create graph
g = gr.Graph()

# 2. Add nodes with attributes
alice = g.add_node(name="Alice", age=29, role="Engineer")
bob = g.add_node(name="Bob", age=55, role="Manager", active=True)
carol = g.add_node(name="Carol", age=31, role="Analyst", active=True)
dave = g.add_node(name="Dave", age=42, role="Engineer", active=False)

# 3. Add edges with weights
g.add_edge(alice, bob, weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob, carol, weight=1)
g.add_edge(carol, dave, weight=3)

# 4. Query
print("=== Active Users ===")
active = g.nodes[g.nodes["active"] == True]
print(f"Active users: {len(active)}")

print("\n=== Young Engineers ===")
young_eng = g.nodes[
    (g.nodes["age"] < 35) &
    (g.nodes["role"] == "Engineer")
]
print(f"Young engineers: {len(young_eng)}")

# 5. Analyze
print("\n=== Graph Statistics ===")
g.connected_components(inplace=True, label="component")
print(f"Components: {len(g.nodes['component'].unique())}")
print(f"Mean age: {g.nodes['age'].mean():.1f}")

# 6. Export
print("\n=== Export ===")
g.nodes.table().to_csv("nodes.csv")
g.save_bundle("social_network.bundle")
print("✓ Saved to files")
```

---

## Performance Considerations

### Bulk Operations

Always prefer bulk operations:

```python
# ❌ Slow: N add_node calls
for data in node_data:
    g.add_node(**data)

# ✓ Fast: 1 add_nodes call
g.add_nodes(node_data)
```

### Attribute Access

Columnar access is faster:

```python
# ❌ Slower: iterate nodes
total = 0
for node in g.nodes:
    total += g.nodes[node]["age"]

# ✓ Faster: get column
ages = g.nodes["age"]
total = ages.sum()
```

### Memory

Views are cheap, copies are expensive:

```python
# Cheap: creates a view
sub = g.nodes[:1000]

# Expensive: copies data
new_graph = sub.to_graph()
```

---

## Common Patterns

### Pattern 1: Build → Query → Analyze

```python
# Build
g = gr.Graph()
# ... add nodes and edges

# Query
active = g.nodes[g.nodes["active"] == True]

# Analyze
g.connected_components(inplace=True)
```

### Pattern 2: Import → Transform → Export

```python
# Import
g = gr.from_pandas(nodes_df, edges_df)

# Transform
g.nodes.set_attrs({
    node: {"processed": True}
    for node in g.nodes.node_ids()
})

# Export
g.save_bundle("processed.bundle")
```

### Pattern 3: Experiment with Branches

```python
# Baseline
g.commit("baseline")

# Experiment
g.create_branch("experiment")
g.checkout_branch("experiment")
# ... make changes
g.commit("experimental changes")

# Compare
g.checkout_branch("main")
# ... compare results
```

---

## Next Steps

- **[Subgraphs](subgraphs.md)**: Work with graph subsets
- **[Accessors](accessors.md)**: Deep dive into `g.nodes` and `g.edges`
- **[Algorithms](algorithms.md)**: Comprehensive algorithm guide
- **[Graph API Reference](../api/graph.md)**: Complete API documentation

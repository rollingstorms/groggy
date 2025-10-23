# Quickstart Guide

This guide will get you up and running with Groggy in 5 minutes.

---

## Installation

If you haven't installed Groggy yet:

```bash
pip install groggy
```

See the [Installation Guide](install.md) for more options.

---

## Your First Graph

Let's build a small social network:

```python
import groggy as gr

# Create an empty graph
g = gr.Graph()

# Add nodes with attributes
alice = g.add_node(name="Alice", age=29, role="Engineer")
bob   = g.add_node(name="Bob",   age=55, role="Manager", club="Purple")
carol = g.add_node(name="Carol", age=31, role="Analyst", club="Blue")

# Add edges with weights
g.add_edge(alice, bob,   weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob,   carol, weight=1)

print(f"Created graph with {len(g.nodes)} nodes and {len(g.edges)} edges")
```

**Key points:**
- `add_node()` returns an integer ID
- Any keyword arguments become node attributes
- Edges use node IDs to connect nodes

---

## Inspect the Graph

### View as Tables

Groggy lets you view your graph as tables:

```python
# Nodes table
print(g.nodes.table().head())
#    id   name     age  role       club
# 0  0    Alice    29   Engineer   None
# 1  1    Bob      55   Manager    Purple
# 2  2    Carol    31   Analyst    Blue

# Edges table
print(g.edges.table().head())
#    src  dst  weight
# 0  0    1    5
# 1  0    2    2
# 2  1    2    1
```

### Access Attributes

Get attribute columns directly:

```python
# Get all names
names = g.nodes["name"]
print(names.head())
# ['Alice', 'Bob', 'Carol']

# Get all ages
ages = g.nodes["age"]
print(f"Mean age: {ages.mean():.1f}")
# Mean age: 38.3
```

---

## Query the Graph

Use Pandas-style filters:

```python
# Filter by attribute
blue_members = g.nodes[g.nodes["club"] == "Blue"]
print(f"Blue club members: {len(blue_members)}")
# Blue club members: 1

# Combine conditions
young_analysts = g.nodes[
    (g.nodes["age"] < 40) &
    (g.nodes["role"] == "Analyst")
]

# Filter edges by weight
heavy_edges = g.edges[g.edges["weight"] > 2]
```

---

## Run Algorithms

Groggy includes common graph algorithms:

```python
# Connected components (modifies graph)
g.connected_components(inplace=True, label="component")

# Check which component each node is in
print(g.nodes["component"].head())
# [0, 0, 0]  (all nodes in same component)

# Number of components
num_components = len(g.nodes["component"].unique())
print(f"Graph has {num_components} component(s)")

# Run multiple algorithms in one pass (PageRank + BFS)
g.nodes.set_attrs({alice: {"is_start": True}})

multi = gr.apply(
    g.view(),
    [
        gr.algorithms.centrality.pagerank(output_attr="pr"),
        gr.algorithms.pathfinding.bfs(start_attr="is_start", output_attr="dist"),
    ],
)

print(multi.nodes.table()[["pr", "dist"]].head())
```

---

## Delegation Chains

One of Groggy's signature features: chain operations together.

```python
# Find components, sample some, expand neighborhoods, summarize
result = (
    g.connected_components()     # Returns SubgraphArray
     .sample(1)                  # Sample 1 component
     .neighborhood(depth=1)      # Expand to neighbors
     .table()                    # Convert to table
     .agg({"weight": "mean"})    # Aggregate
)

print(result)
# Shows mean edge weight in the sampled neighborhood
```

**What happened:**
1. `connected_components()` → SubgraphArray (array of components)
2. `.sample(1)` → SubgraphArray (filtered to 1 component)
3. `.neighborhood(depth=1)` → SubgraphArray (expanded)
4. `.table()` → GraphTable (tabular view)
5. `.agg({...})` → AggregationResult (summary stats)

---

## Working with Subgraphs

Create subgraphs by slicing:

```python
# First 2 nodes
small_graph = g.nodes[:2]

# Specific nodes
subset = g.nodes[[0, 2]]  # Alice and Carol

# Convert back to full graph
new_graph = subset.to_graph()
```

---

## Bulk Attribute Updates

Set attributes for multiple nodes/edges at once:

```python
# Update node attributes
g.nodes.set_attrs({
    alice: {"status": "active", "team": "A"},
    bob:   {"status": "active", "team": "B"},
    carol: {"status": "inactive", "team": "A"}
})

# Update edge attributes
edge_0 = 0  # Edge ID
g.edges.set_attrs({
    edge_0: {"type": "friendship", "since": 2020}
})
```

---

## Save and Load

### Graph Bundles

Save the entire graph (structure + attributes):

```python
# Save
g.save_bundle("my_graph.bundle")

# Load
loaded = gr.GraphTable.load_bundle("my_graph.bundle")
g2 = loaded.to_graph()
```

### Export Tables

Export to various formats:

```python
# Parquet (efficient binary format)
g.nodes.table().to_parquet("nodes.parquet")
g.edges.table().to_parquet("edges.parquet")

# CSV (human-readable)
g.nodes.table().to_csv("nodes.csv")

# Pandas DataFrame
df = g.nodes.table().to_pandas()
print(df.head())
```

---

## Built-in Generators

Start with pre-built graphs:

```python
# Karate club network (classic dataset)
karate = gr.generators.karate_club()
print(karate.table())
# GraphTable with 34 nodes, 78 edges

# Complete graph
complete = gr.generators.complete_graph(5)

# Erdős-Rényi random graph
random = gr.generators.erdos_renyi(n=100, p=0.05)

# Path graph
path = gr.generators.path_graph(10)
```

---

## Visualization

Visualize your graph:

```python
# Basic visualization
g.viz.show()

# Color by attribute
g.viz.show(node_color="component")

# Size by attribute
g.viz.show(node_size="age")

# Combine styling
g.viz.show(
    node_color="club",
    node_size="age",
    edge_width="weight"
)
```

---

## Complete Example: Social Network Analysis

Putting it all together:

```python
import groggy as gr

# 1. Build the graph
g = gr.Graph()
alice = g.add_node(name="Alice", age=29)
bob = g.add_node(name="Bob", age=55, club="Purple", active=True)
carol = g.add_node(name="Carol", age=31, club="Blue", active=True)
dave = g.add_node(name="Dave", age=42, active=False)

g.add_edge(alice, bob, weight=5)
g.add_edge(alice, carol, weight=2)
g.add_edge(bob, carol, weight=1)
g.add_edge(carol, dave, weight=3)

# 2. Inspect
print("=== Node Table ===")
print(g.nodes.table().head())

print("\n=== Edge Table ===")
print(g.edges.table().head())

# 3. Query
print("\n=== Active Members ===")
active = g.nodes[g.nodes["active"] == True]
print(f"Found {len(active)} active members")

print("\n=== Older Users ===")
older = g.nodes[g.nodes["age"] > 30]
print(f"Found {len(older)} users over 30")

# 4. Analyze
print("\n=== Graph Analysis ===")
g.connected_components(inplace=True, label="component")
print(f"Components: {len(g.nodes['component'].unique())}")

mean_age = g.nodes["age"].mean()
print(f"Mean age: {mean_age:.1f}")

# 5. Chain operations
print("\n=== Delegation Chain ===")
result = (
    g.connected_components()
     .sample(1)
     .table()
     .head()
)
print(result)

# 6. Save
g.save_bundle("social_network.bundle")
print("\n✓ Graph saved to social_network.bundle")
```

---

## Next Steps

Now that you've seen the basics:

- **Learn the concepts**: Read [Concepts & Architecture](concepts/overview.md) to understand how Groggy works
- **Deep dive**: Explore the [User Guide](guide/graph-core.md) for comprehensive tutorials
- **API details**: Check the [API Reference](api/graph.md) for complete method documentation
- **User Guides**: Learn specific topics in the [User Guide](guide/graph-core.md)

---

## Quick Reference

### Creating Graphs
```python
g = gr.Graph()                          # Empty graph
g = gr.generators.karate_club()         # Built-in dataset
g = gr.from_pandas(nodes_df, edges_df)  # From DataFrames
```

### Adding Data
```python
node_id = g.add_node(attr1=val1, attr2=val2)
edge_id = g.add_edge(src, dst, weight=5)
```

### Querying
```python
filtered = g.nodes[g.nodes["attr"] > value]
subset = g.nodes[:10]  # First 10 nodes
```

### Algorithms
```python
g.connected_components(inplace=True, label="comp")
```

### Delegation Chains
```python
result = g.connected_components().table().agg({"weight": "mean"})
```

### I/O
```python
g.save_bundle("file.bundle")
g = gr.GraphTable.load_bundle("file.bundle").to_graph()
```

---

Happy graphing! 🎉

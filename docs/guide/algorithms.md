# Graph Algorithms

Groggy provides efficient graph algorithms implemented in Rust. These algorithms work on both full graphs and subgraphs, with results typically written back to the graph or returned as new objects.

---

## Algorithm Categories

Groggy's algorithms fall into several categories:

- **Traversal**: BFS, DFS
- **Components**: Connected components
- **Paths**: Shortest paths
- **Centrality**: Degree, betweenness, closeness
- **Community**: Community detection, clustering
- **Spectral**: Eigenvalue-based methods

---

## Graph Traversal

### Breadth-First Search (BFS)

Explore graph level by level:

```python
import groggy as gr

g = gr.generators.karate_club()

# BFS from a starting node
result = g.bfs(start=0)  # Returns Subgraph

print(f"BFS reached {result.node_count()} nodes")
```

**Use cases:**
- Finding shortest unweighted paths
- Level-order exploration
- Testing connectivity

### Depth-First Search (DFS)

Explore deeply before backtracking:

```python
# DFS from starting node
result = g.dfs(start=0)  # Returns Subgraph

print(f"DFS reached {result.node_count()} nodes")
```

**Use cases:**
- Detecting cycles
- Topological sorting
- Path finding

---

## Connected Components

Find groups of connected nodes:

### In-Place Labeling

Write component IDs to nodes:

```python
g = gr.generators.karate_club()

# Label nodes with component ID
g.connected_components(inplace=True, label="component")

# Check assignments
components = g.nodes["component"]
num_components = components.nunique()

print(f"Found {num_components} component(s)")

# Get nodes in component 0
comp_0 = g.nodes[g.nodes["component"] == 0]
print(f"Component 0: {comp_0.node_count()} nodes")
```

### As SubgraphArray

Get components as separate subgraphs:

```python
# Returns SubgraphArray
components = g.connected_components()

print(f"{len(components)} component(s)")

# Access individual components
largest = components[0]  # Subgraph
print(f"Largest: {largest.node_count()} nodes")

# Analyze each
for i, comp in enumerate(components):
    density = comp.density()
    print(f"Component {i}: {comp.node_count()} nodes, density={density:.3f}")
```

**Use cases:**
- Finding disconnected parts
- Analyzing component structure
- Filtering by component size

---

## Shortest Paths

### Single-Source Shortest Paths

Find shortest paths from one node to all others:

```python
g = gr.generators.karate_club()

# Shortest paths from node 0
paths = g.shortest_path(source=0)  # Returns Subgraph or distances

# With edge weights
g_weighted = gr.Graph()
n0, n1, n2 = g_weighted.add_node(), g_weighted.add_node(), g_weighted.add_node()
g_weighted.add_edge(n0, n1, weight=5.0)
g_weighted.add_edge(n0, n2, weight=2.0)
g_weighted.add_edge(n1, n2, weight=1.0)

# Weighted shortest paths
paths = g_weighted.shortest_path(source=0, weight="weight")
```

**Algorithms used:**
- Unweighted: BFS (O(V + E))
- Weighted: Dijkstra (O((V + E) log V))

**Use cases:**
- Finding optimal routes
- Distance calculations
- Network analysis

---

## Degree and Centrality

### Degree

Count connections per node:

```python
g = gr.generators.karate_club()

# All degrees
degrees = g.degree()  # NumArray
print(f"Mean degree: {degrees.mean():.2f}")
print(f"Max degree: {degrees.max()}")

# In-degree (directed graphs)
in_degrees = g.in_degree()

# Out-degree (directed graphs)
out_degrees = g.out_degree()
```

### Degree as Node Attribute

```python
# Store degrees as attribute
g.nodes.set_attrs({
    int(nid): {"degree": int(deg)}
    for nid, deg in zip(g.nodes.ids(), g.degree())
})

# Query by degree
high_degree = g.nodes[g.nodes["degree"] > 5]
print(f"High-degree nodes: {high_degree.node_count()}")
```

**Use cases:**
- Finding hubs
- Network analysis
- Filtering by connectivity

### Other Centrality Measures

```python
# Betweenness centrality (if available)
# betweenness = g.betweenness_centrality()

# Closeness centrality
# closeness = g.closeness_centrality()

# Eigenvector centrality
# eigenvector = g.eigenvector_centrality()

# PageRank
# pagerank = g.pagerank(damping=0.85)
```

---

## Neighborhood Analysis

### K-Hop Neighborhoods

Get nodes within k hops:

```python
g = gr.generators.karate_club()

# 2-hop neighborhood around node 0
neighborhood = g.neighborhood(node=0, depth=2)

# For multiple nodes
seeds = g.nodes[:5]
neighborhoods = seeds.neighborhood(depth=2)  # SubgraphArray

for i, nbh in enumerate(neighborhoods):
    print(f"Node {i} reaches {nbh.node_count()} nodes in 2 hops")
```

**Use cases:**
- Local structure analysis
- Feature extraction
- Subgraph sampling

### Neighborhood Statistics

```python
# Get neighborhood statistics (if available)
stats = g.neighborhood_statistics(depth=2)

# Might include:
# - Average neighborhood size
# - Clustering coefficients
# - Local densities
```

---

## Clustering and Communities

### Clustering Coefficient

Measure how clustered neighborhoods are:

```python
# Global clustering coefficient
# clustering = g.clustering_coefficient()
# print(f"Clustering: {clustering:.3f}")

# Per-node clustering
# node_clustering = g.node_clustering_coefficient()
```

**Interpretation:**
- 0.0 = No triangles (tree-like)
- 1.0 = Complete graph (everyone connected)

### Community Detection

Find natural groupings:

```python
# Louvain community detection (if available)
# communities = g.detect_communities(method="louvain")

# Label propagation
# communities = g.detect_communities(method="label_propagation")

# Results as node attribute
# g.nodes.set_attrs({
#     int(nid): {"community": int(comm)}
#     for nid, comm in zip(g.nodes.ids(), communities)
# })
```

**Use cases:**
- Social network analysis
- Modularity optimization
- Graph partitioning

---

## Spectral Methods

### Spectral Embeddings

Embed graphs in vector space:

```python
# Spectral embedding (if available)
# embeddings = g.spectral_embedding(dimensions=8)

# Returns matrix where each row is a node embedding
# print(embeddings.shape())  # (num_nodes, 8)
```

**Use cases:**
- Graph visualization
- Node classification
- Dimensionality reduction

### Laplacian Eigenmaps

```python
# Get Laplacian
L = g.laplacian_matrix()

# Compute eigenvectors (external library needed)
# import scipy.sparse.linalg as spla
# eigenvalues, eigenvectors = spla.eigsh(L, k=10, which='SM')
```

---

## Common Patterns

### Pattern 1: Filter by Algorithm Result

```python
# Run algorithm
g.connected_components(inplace=True, label="component")

# Filter by result
largest_comp_id = 0  # Assuming 0 is largest
largest = g.nodes[g.nodes["component"] == largest_comp_id]

# Work with filtered graph
print(f"Largest component: {largest.node_count()} nodes")
largest_graph = largest.to_graph()
```

### Pattern 2: Iterative Refinement

```python
# Start with full graph
current = g

# Iteratively filter
for i in range(3):
    # Run algorithm
    components = current.connected_components()

    # Keep largest
    current = components[0].to_graph()

    print(f"Iteration {i}: {current.node_count()} nodes")
```

### Pattern 3: Multi-Metric Analysis

```python
# Compute multiple metrics
degrees = g.degree()
# betweenness = g.betweenness_centrality()
# clustering = g.node_clustering_coefficient()

# Store all
g.nodes.set_attrs({
    int(nid): {
        "degree": int(deg),
        # "betweenness": float(bet),
        # "clustering": float(clust)
    }
    for nid, deg in zip(g.nodes.ids(), degrees)
})

# Query by combination
# important = g.nodes[
#     (g.nodes["degree"] > 5) &
#     (g.nodes["betweenness"] > 0.1)
# ]
```

### Pattern 4: Component Comparison

```python
components = g.connected_components()

# Analyze each
stats = []
for i, comp in enumerate(components):
    stats.append({
        'id': i,
        'nodes': comp.node_count(),
        'edges': comp.edge_count(),
        'density': comp.density(),
        'is_connected': comp.is_connected()
    })

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(stats)
print(df.sort_values('nodes', ascending=False))
```

### Pattern 5: Shortest Path Analysis

```python
# Find all paths from central node
central_node = 0
paths = g.shortest_path(source=central_node)

# Analyze reachability
reachable = paths.node_count()
total = g.node_count()
print(f"Reachability: {reachable}/{total} ({reachable/total*100:.1f}%)")

# Check if graph is connected
if reachable == total:
    print("Graph is connected")
else:
    print("Graph is disconnected")
```

### Pattern 6: Degree Distribution

```python
degrees = g.degree()

# Get distribution
from collections import Counter
degree_dist = Counter(degrees.to_list())

print("Degree distribution:")
for degree, count in sorted(degree_dist.items()):
    print(f"  Degree {degree}: {count} node(s)")

# Plot if needed
import matplotlib.pyplot as plt
plt.bar(degree_dist.keys(), degree_dist.values())
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution")
```

### Pattern 7: Subgraph Algorithms

```python
# Filter to subgraph
young = g.nodes[g.nodes["age"] < 30]

# Run algorithms on subgraph
young_components = young.connected_components()
print(f"Young network has {len(young_components)} component(s)")

# Compare to full graph
full_components = g.connected_components()
print(f"Full network has {len(full_components)} component(s)")
```

---

## Performance Considerations

### Algorithm Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| BFS/DFS | O(V + E) | Linear in graph size |
| Connected Components | O(V + E) | Union-find based |
| Shortest Path (unweighted) | O(V + E) | BFS |
| Shortest Path (weighted) | O((V + E) log V) | Dijkstra's |
| Degree | O(V + E) | Count edges per node |

### Large Graphs

For very large graphs:

```python
# Work with samples
sample = g.nodes.sample(1000)
sample_graph = sample.to_graph()
components = sample_graph.connected_components()

# Or work with subgraphs
core = g.nodes[g.nodes["degree"] > 5]  # High-degree subgraph
core_components = core.connected_components()
```

### Parallelization

Rust implementation uses parallelization when beneficial:

```python
# Algorithms automatically parallelize when appropriate
# No special configuration needed
components = g.connected_components()
```

---

## Algorithm Selection

### When to Use Each

**BFS:**
- ✅ Shortest unweighted paths
- ✅ Level-order traversal
- ✅ Checking connectivity

**DFS:**
- ✅ Cycle detection
- ✅ Topological sorting
- ✅ Deep exploration

**Connected Components:**
- ✅ Finding separate networks
- ✅ Analyzing fragmentation
- ✅ Filtering disconnected parts

**Shortest Paths:**
- ✅ Navigation/routing
- ✅ Distance calculations
- ✅ Network efficiency

**Degree:**
- ✅ Finding hubs
- ✅ Network topology
- ✅ Quick connectivity measure

**Centrality:**
- ✅ Identifying important nodes
- ✅ Ranking nodes
- ✅ Influence analysis

---

## Future Algorithms

Algorithms potentially coming to Groggy:

- **Betweenness centrality**: Bridge detection
- **PageRank**: Importance ranking
- **Community detection**: Modularity optimization
- **Minimum spanning tree**: Network design
- **Max flow**: Capacity analysis
- **Triangle counting**: Clustering analysis
- **K-core decomposition**: Core structure

Check the API reference for the latest available algorithms.

---

## Quick Reference

### Running Algorithms

```python
# Traversal
bfs_result = g.bfs(start=0)
dfs_result = g.dfs(start=0)

# Components
g.connected_components(inplace=True, label="component")
components = g.connected_components()  # SubgraphArray

# Paths
paths = g.shortest_path(source=0)
paths = g.shortest_path(source=0, weight="weight")

# Degree
degrees = g.degree()
in_deg = g.in_degree()
out_deg = g.out_degree()

# Neighborhoods
nbh = g.neighborhood(node=0, depth=2)
nbhs = g.nodes[:5].neighborhood(depth=2)
```

### Common Options

```python
# In-place modification
g.connected_components(inplace=True, label="component")

# Return value
components = g.connected_components()  # SubgraphArray

# With weights
paths = g.shortest_path(source=0, weight="weight")

# Depth parameter
nbh = g.neighborhood(node=0, depth=2)
```

---

## See Also

- **[Graph Core Guide](graph-core.md)**: Running algorithms on graphs
- **[Subgraphs Guide](subgraphs.md)**: Algorithms on subgraphs
- **[SubgraphArrays Guide](subgraph-arrays.md)**: Working with component results
- **[Matrices Guide](matrices.md)**: Matrix-based algorithms
- **[Performance Guide](performance.md)**: Optimizing algorithm performance

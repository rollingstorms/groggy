# GraphOps API

`GraphOps` provides topology-dependent operations: neighbor aggregation, degree computation, and graph-structural transformations. These operations work with the graph structure itself.

## Overview

Access GraphOps through `sG.builder.graph_ops`:

```python
@algorithm
def example(sG):
    degrees = sG.builder.graph_ops.degree()
    neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, "sum")
    # Or use matrix notation: neighbor_sum = sG @ values
    return degrees
```

## Graph Properties

### `degree(direction="out")`

Compute node degrees.

```python
out_degrees = sG.builder.graph_ops.degree()
in_degrees = sG.builder.graph_ops.degree(direction="in")
total_degrees = sG.builder.graph_ops.degree(direction="both")
```

**Parameters:**
- `direction`: Direction of edges to count
  - `"out"` - Out-degree (default)
  - `"in"` - In-degree
  - `"both"` - Total degree (in + out)

**Returns:** VarHandle (node values)

**Example:**
```python
@algorithm
def degree_centrality(sG, normalized=True):
    deg = sG.builder.graph_ops.degree()
    if normalized:
        return deg / (sG.N - 1)
    return deg
```

**Alternative via VarHandle:**
```python
# For any node values:
values = sG.nodes(1.0)
deg = values.degrees()  # Equivalent to degree(direction="out")
```

### `node_count()`

Get the number of nodes in the subgraph.

```python
n = sG.builder.graph_ops.node_count()
# Returns scalar VarHandle
```

**Alternative:**
```python
n = sG.N  # Property on GraphHandle
```

**Example:**
```python
@algorithm
def uniform_init(sG):
    n = sG.builder.graph_ops.node_count()
    inv_n = sG.builder.core.recip(n)
    return sG.nodes(inv_n)
```

### `edge_count()`

Get the number of edges in the subgraph.

```python
m = sG.builder.graph_ops.edge_count()
# Returns scalar VarHandle
```

**Example:**
```python
@algorithm
def edge_density(sG):
    n = sG.builder.graph_ops.node_count()
    m = sG.builder.graph_ops.edge_count()
    max_edges = n * (n - 1)  # Directed graph
    density = m / max_edges
    return density
```

## Neighbor Aggregation

### `neighbor_agg(values, agg="sum", weights=None)`

Aggregate values from neighbors.

```python
neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, agg="sum")
neighbor_max = sG.builder.graph_ops.neighbor_agg(values, agg="max")
```

**Parameters:**
- `values`: VarHandle (node values to aggregate)
- `agg`: Aggregation operation
  - `"sum"` - Sum neighbor values (default)
  - `"mean"` - Average neighbor values
  - `"min"` - Minimum neighbor value
  - `"max"` - Maximum neighbor value
- `weights`: Optional VarHandle (edge weights)

**Returns:** VarHandle (aggregated values per node)

**Semantics:**
```python
# For each node i:
# result[i] = agg(values[j] for j in neighbors(i))
```

**Examples:**

**Sum aggregation (PageRank):**
```python
contrib = ranks / (degrees + 1e-9)
neighbor_sum = sG.builder.graph_ops.neighbor_agg(contrib, "sum")
# Or: neighbor_sum = sG @ contrib
```

**Mean aggregation:**
```python
avg_neighbor_value = sG.builder.graph_ops.neighbor_agg(values, "mean")
```

**Weighted aggregation:**
```python
edge_weights = sG.builder.attr.load_edge("weight", default=1.0)
weighted_sum = sG.builder.graph_ops.neighbor_agg(
    values, agg="sum", weights=edge_weights
)
```

**Matrix notation shortcut:**
```python
# Sum aggregation only
neighbor_sum = sG @ values
# Equivalent to:
neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, "sum")
```

### `collect_neighbor_values(values, include_self=True)`

Collect neighbor values into lists (for mode computation).

```python
neighbor_lists = sG.builder.graph_ops.collect_neighbor_values(labels)
```

**Parameters:**
- `values`: VarHandle (node values to collect)
- `include_self`: Whether to include node's own value (default: True)

**Returns:** VarHandle (lists of neighbor values per node)

**Example - Label Propagation:**
```python
@algorithm
def lpa(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    
    with sG.builder.iter.loop(max_iter):
        # Collect labels from neighbors (and self)
        neighbor_labels = sG.builder.graph_ops.collect_neighbor_values(
            labels, include_self=True
        )
        
        # Find most common label
        most_common = sG.builder.core.mode(neighbor_labels, tie_break="lowest")
        
        labels = sG.builder.var("labels", most_common)
    
    return labels
```

### `neighbor_mode_update(target, include_self=True, ordered=True)`

Update values to most common neighbor value (optimized for LPA).

```python
updated = sG.builder.graph_ops.neighbor_mode_update(labels, ordered=True)
```

**Parameters:**
- `target`: VarHandle (values to update based on neighbor mode)
- `include_self`: Include node's own value when computing mode (default: True)
- `ordered`: Process nodes in order (synchronous) vs random (default: True)

**Returns:** VarHandle (updated values)

**Equivalent to:**
```python
neighbor_lists = sG.builder.graph_ops.collect_neighbor_values(target, include_self)
most_common = sG.builder.core.mode(neighbor_lists)
```

**Example - Optimized LPA:**
```python
@algorithm
def lpa_fast(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    
    with sG.builder.iter.loop(max_iter):
        labels = sG.builder.graph_ops.neighbor_mode_update(
            labels, include_self=True, ordered=True
        )
    
    return labels
```

## Subgraph Operations

### `subgraph(node_mask)`

Create a subgraph view based on node mask.

```python
active_nodes = status == 1.0
subgraph = sG.builder.graph_ops.subgraph(active_nodes)
```

**Parameters:**
- `node_mask`: VarHandle (boolean mask, 1.0 = include, 0.0 = exclude)

**Returns:** VarHandle (subgraph handle)

**Example:**
```python
@algorithm
def active_component(sG, threshold=0.5):
    values = sG.builder.attr.load("activity", default=0.0)
    is_active = values > threshold
    
    # Create subgraph of active nodes
    active_sg = sG.builder.graph_ops.subgraph(is_active)
    
    # Could run further algorithms on active_sg
    # For now, just return the mask
    return is_active
```

**Note:** Subgraph operations are currently limited. This is a future extension point.

### `filter_edges(edge_mask)`

Filter edges based on mask.

```python
strong_edges = edge_weight > 0.5
filtered = sG.builder.graph_ops.filter_edges(strong_edges)
```

**Parameters:**
- `edge_mask`: VarHandle (edge boolean mask)

**Returns:** VarHandle (subgraph with filtered edges)

**Example:**
```python
@algorithm
def strong_connections(sG, weight_threshold=0.5):
    weights = sG.builder.attr.load_edge("weight", default=0.0)
    is_strong = weights > weight_threshold
    
    # Filter to strong edges only
    strong_sg = sG.builder.graph_ops.filter_edges(is_strong)
    return is_strong
```

## Traversal Operations

### `bfs_levels(source_mask, max_depth=None)`

Compute breadth-first search levels from sources.

```python
levels = sG.builder.graph_ops.bfs_levels(is_source)
```

**Parameters:**
- `source_mask`: VarHandle (boolean mask, 1.0 = source node)
- `max_depth`: Optional maximum depth (default: None = unlimited)

**Returns:** VarHandle (distance from nearest source, -1 if unreachable)

**Example:**
```python
@algorithm
def shortest_path_distance(sG, source_id):
    # Create mask for source node
    node_ids = sG.nodes(unique=True)
    is_source = node_ids == source_id
    
    # Compute BFS levels
    distances = sG.builder.graph_ops.bfs_levels(is_source)
    return distances
```

### `connected_components()`

Find connected components.

```python
component_ids = sG.builder.graph_ops.connected_components()
```

**Returns:** VarHandle (component ID per node)

**Example:**
```python
@algorithm
def component_sizes(sG):
    components = sG.builder.graph_ops.connected_components()
    
    # Count nodes per component
    ones = sG.nodes(1.0)
    # Group by component and sum
    # (requires groupby - future feature)
    
    return components
```

**Note:** This is a native Rust operation, not composed from primitives.

## Structural Metrics

### `clustering_coefficient(local=True)`

Compute clustering coefficient.

```python
local_cc = sG.builder.graph_ops.clustering_coefficient(local=True)
global_cc = sG.builder.graph_ops.clustering_coefficient(local=False)
```

**Parameters:**
- `local`: Return per-node coefficient (True) or global average (False)

**Returns:** VarHandle (local: node values, global: scalar)

**Example:**
```python
@algorithm
def high_clustering_nodes(sG, threshold=0.5):
    cc = sG.builder.graph_ops.clustering_coefficient(local=True)
    is_high = cc > threshold
    return is_high.where(1.0, 0.0)
```

### `triangles()`

Count triangles per node.

```python
triangle_counts = sG.builder.graph_ops.triangles()
```

**Returns:** VarHandle (number of triangles per node)

**Example:**
```python
@algorithm
def triangle_density(sG):
    triangles = sG.builder.graph_ops.triangles()
    degrees = sG.builder.graph_ops.degree()
    
    # Maximum possible triangles for each node
    max_triangles = degrees * (degrees - 1) / 2.0
    
    density = triangles / (max_triangles + 1e-9)
    return density
```

## Attribute-Based Queries

### `neighbors_with_attribute(attr_name, comparison, value)`

Find neighbors matching attribute criteria.

```python
active_neighbors = sG.builder.graph_ops.neighbors_with_attribute(
    "status", "eq", 1.0
)
```

**Parameters:**
- `attr_name`: Attribute name to check
- `comparison`: Comparison operator ("eq", "gt", "lt", etc.)
- `value`: Value to compare against

**Returns:** VarHandle (count of matching neighbors per node)

**Example:**
```python
@algorithm
def active_neighbor_ratio(sG):
    # Count neighbors with status=1
    active_count = sG.builder.graph_ops.neighbors_with_attribute(
        "status", "eq", 1.0
    )
    
    # Total neighbors
    total_neighbors = sG.builder.graph_ops.degree()
    
    # Ratio
    ratio = active_count / (total_neighbors + 1e-9)
    return ratio
```

## Edge Operations

### `reverse_edges()`

Reverse edge directions.

```python
reversed_sg = sG.builder.graph_ops.reverse_edges()
```

**Returns:** VarHandle (subgraph with reversed edges)

**Example:**
```python
@algorithm
def in_degree(sG):
    # Reverse edges and compute out-degree
    reversed = sG.builder.graph_ops.reverse_edges()
    # Would need to apply to reversed graph
    # Current limitation: can't easily chain subgraph operations
    
    # Better: use degree(direction="in")
    return sG.builder.graph_ops.degree(direction="in")
```

### `undirected_view()`

Create undirected view (each edge becomes bidirectional).

```python
undirected = sG.builder.graph_ops.undirected_view()
```

**Returns:** VarHandle (undirected subgraph view)

**Example:**
```python
@algorithm
def undirected_pagerank(sG, damping=0.85, max_iter=100):
    # Treat graph as undirected
    uG = sG.builder.graph_ops.undirected_view()
    
    # Run PageRank on undirected view
    # (Implementation details omitted)
    pass
```

## Normalization Helpers

### `normalize_adjacency(method="symmetric")`

Normalize adjacency matrix.

```python
# D^-1 A (row normalization)
row_norm = sG.builder.graph_ops.normalize_adjacency("row")

# A D^-1 (column normalization)
col_norm = sG.builder.graph_ops.normalize_adjacency("column")

# D^-1/2 A D^-1/2 (symmetric normalization)
sym_norm = sG.builder.graph_ops.normalize_adjacency("symmetric")
```

**Parameters:**
- `method`: Normalization method
  - `"row"` - Row normalization (each row sums to 1)
  - `"column"` - Column normalization (each column sums to 1)
  - `"symmetric"` - Symmetric normalization

**Returns:** VarHandle (normalized adjacency operator)

**Example - Graph Convolutional Network style:**
```python
@algorithm
def gcn_propagation(sG, features):
    # Symmetric normalization
    norm_adj = sG.builder.graph_ops.normalize_adjacency("symmetric")
    
    # Propagate features
    # (Would need to apply norm_adj as operator)
    # Current limitation: operators not fully composable
    
    # Manual equivalent:
    deg = features.degrees()
    deg_sqrt_inv = sG.builder.core.pow(deg + 1e-9, -0.5)
    
    scaled_features = features * deg_sqrt_inv
    neighbor_sum = sG @ scaled_features
    result = neighbor_sum * deg_sqrt_inv
    
    return result
```

## Common Patterns

### PageRank-Style Aggregation

```python
# Compute contribution from each node
deg = ranks.degrees()
contrib = ranks / (deg + 1e-9)

# Aggregate from neighbors
neighbor_sum = sG @ contrib

# Update
ranks = damping * neighbor_sum + (1 - damping) / sG.N
```

### Weighted Neighbor Average

```python
weights = sG.builder.attr.load_edge("weight", default=1.0)
weighted_sum = sG.builder.graph_ops.neighbor_agg(values, "sum", weights)

# Normalize by weight sum
weight_sum = sG.builder.graph_ops.neighbor_agg(
    sG.nodes(1.0), "sum", weights
)
weighted_avg = weighted_sum / (weight_sum + 1e-9)
```

### Neighbor Filtering

```python
# Only aggregate from neighbors with high values
is_important = values > threshold
important_values = is_important.where(values, 0.0)
result = sG @ important_values
```

### Degree-Based Weighting

```python
# Weight by inverse degree (prevent high-degree nodes from dominating)
deg = values.degrees()
inv_deg = sG.builder.core.recip(deg, epsilon=1e-9)
weighted = values * inv_deg
result = sG @ weighted
```

## Performance Notes

### Neighbor Aggregation Efficiency

Neighbor aggregation is highly optimized in Rust:

```python
# These are efficient O(E) operations:
neighbor_sum = sG @ values           # Fast
neighbor_max = sG.builder.graph_ops.neighbor_agg(values, "max")  # Fast
```

### Degree Caching

Degree computation is cached per iteration:

```python
# Only computed once:
deg = ranks.degrees()
inv_deg = 1.0 / (deg + 1e-9)
contrib = ranks * inv_deg
```

### Subgraph Operations

Subgraph creation is lightweight (view-based, not copied):

```python
# Creates view, doesn't copy data
active = sG.builder.graph_ops.subgraph(is_active)
```

## Limitations and Future Work

### Current Limitations

1. **Subgraph chaining**: Can't easily compose multiple subgraph operations
2. **Edge attributes in aggregation**: Limited support for complex edge weighting
3. **Hypergraph operations**: No native hyperedge support

### Planned Features

- [ ] Multi-hop aggregation (`k_hop_agg(values, k=2)`)
- [ ] Attention-based aggregation (GAT-style)
- [ ] Sparse matrix operations (SpMM, SpMSpM)
- [ ] Bipartite graph operations
- [ ] Temporal graph operations (time-windowed aggregation)

## Examples

### Influence Propagation

```python
@algorithm
def influence_propagation(sG, seed_influence, decay=0.9, max_iter=10):
    influence = seed_influence
    
    with sG.builder.iter.loop(max_iter):
        # Propagate with decay
        neighbor_influence = sG @ influence
        influence = sG.builder.var("influence",
            decay * neighbor_influence + (1 - decay) * seed_influence
        )
    
    return influence
```

### Neighbor Feature Aggregation

```python
@algorithm
def aggregate_neighbor_features(sG, attr_name, agg="mean"):
    features = sG.builder.attr.load(attr_name, default=0.0)
    aggregated = sG.builder.graph_ops.neighbor_agg(features, agg)
    return aggregated
```

### Structural Similarity

```python
@algorithm
def structural_similarity(sG):
    # Jaccard coefficient based on common neighbors
    deg = sG.builder.graph_ops.degree()
    
    # Count common neighbors (simplified)
    # Full implementation would require set operations
    # This is a placeholder showing the pattern
    
    return deg
```

## See Also

- [VarHandle API](varhandle.md) - Matrix notation (sG @ values)
- [CoreOps API](core.md) - Value operations
- [IterOps API](iter.md) - Control flow for iterative algorithms
- [PageRank Tutorial](../tutorials/02_pagerank.md) - Neighbor aggregation in practice

# Tutorial 4: Custom Node and Edge Metrics

In this tutorial, you'll learn how to build complex custom metrics by combining the operations you've learned. We'll create several useful metrics that demonstrate the power and flexibility of the Builder DSL.

## What You'll Learn

- Combining multiple operations into complex metrics
- Working with both node and edge attributes
- Using conditional logic for sophisticated computations
- Building reusable metric functions
- Composing algorithms

## Prerequisites

- Complete [Tutorial 1](01_hello_world.md), [Tutorial 2](02_pagerank.md), and [Tutorial 3](03_lpa.md)
- Familiarity with graph metrics concepts
- Understanding of the DSL operators

## Example 1: Clustering Coefficient

The local clustering coefficient measures how connected a node's neighbors are to each other. It's the fraction of possible triangles that actually exist.

**Formula:**
```
C(v) = 2 * triangles(v) / (degree(v) * (degree(v) - 1))
```

Where `triangles(v)` is the number of triangles node v participates in.

### Implementation

```python
from groggy.builder import algorithm

@algorithm("clustering_coefficient")
def clustering_coefficient(sG):
    """
    Compute local clustering coefficient for each node.
    
    The clustering coefficient measures how well-connected a node's
    neighbors are. Range: [0, 1], where 1 means all neighbors are connected.
    """
    # Get degrees
    degrees = G.nodes().degrees()
    
    # Get neighbor degrees for each node
    neighbor_degrees = G @ degrees
    
    # Count triangles: for each edge (u,v), count common neighbors
    # This is a simplified approximation using degree-based estimation
    # (Full triangle counting requires more complex operations)
    
    # Maximum possible edges among neighbors: degree * (degree - 1) / 2
    max_edges = degrees * (degrees - 1.0) / 2.0
    
    # Estimate triangles from neighbor degree distribution
    # (Real implementation would count actual triangles)
    estimated_triangles = neighbor_degrees / degrees
    
    # Clustering coefficient
    clustering = estimated_triangles / (max_edges + 1e-9)
    
    # Clamp to [0, 1]
    clustering = G.builder.core.min(clustering, 1.0)
    
    return clustering

# Example usage
if __name__ == "__main__":
    import groggy as gg
    
    # Create a graph with some triangles
    G = gg.Graph()
    G.add_edges_from([
        (1, 2), (2, 3), (3, 1),  # Triangle
        (1, 4), (2, 4),          # Partial connection to 4
        (4, 5), (5, 6), (6, 4),  # Another triangle
    ])
    
    algo = clustering_coefficient()
    result = G.all().apply(algo)
    
    cc = result.nodes()["clustering_coefficient"]
    for node, coef in sorted(cc.items()):
        print(f"Node {node}: {coef:.3f}")
```

**Key techniques:**
- Combining arithmetic operations
- Using epsilon terms (`1e-9`) to avoid division by zero
- Clamping values with `min()` and `max()`

## Example 2: Betweenness Centrality (Simplified)

Betweenness centrality measures how often a node lies on shortest paths between other nodes. Here's a simplified version based on neighbor paths:

```python
@algorithm("betweenness_approx")
def betweenness_approx(sG):
    """
    Approximate betweenness centrality.
    
    True betweenness requires computing all shortest paths (expensive).
    This approximation uses local neighborhood structure as a proxy.
    """
    # Get degrees
    degrees = G.nodes().degrees()
    
    # High-degree nodes tend to be on more paths
    # Neighbors of low-degree nodes are more "bottleneck-y"
    
    # For each node, average the inverse degrees of its neighbors
    inv_degrees = 1.0 / (degrees + 1e-9)
    avg_neighbor_inv_degree = G @ inv_degrees / (degrees + 1e-9)
    
    # Betweenness proxy: nodes with low-degree neighbors are more central
    betweenness = degrees * avg_neighbor_inv_degree
    
    return betweenness.normalize()
```

**Key techniques:**
- Multi-step aggregation
- Inverting values
- Normalizing final results

## Example 3: Edge Importance Score

Let's build a metric that scores edges based on the importance of their endpoints:

```python
@algorithm("edge_importance")
def edge_importance(sG):
    """
    Score edges based on endpoint importance.
    
    Important edges connect important nodes. We use PageRank as the
    importance measure.
    """
    # First compute PageRank
    ranks = G.nodes(1.0 / G.N)
    degrees = ranks.degrees()
    inv_degrees = 1.0 / (degrees + 1e-9)
    
    with G.builder.iter.loop(20):
        neighbor_sum = G @ (ranks * inv_degrees)
        ranks = G.builder.var("ranks", 0.85 * neighbor_sum + 0.15 / G.N)
    
    ranks = ranks.normalize()
    
    # Edge importance = product of endpoint ranks
    # (This requires edge-level computation - placeholder for future feature)
    
    # For now, save node importance
    G.builder.attr.save("node_importance", ranks)
    
    # Return a signal that edge computation is needed
    return ranks

# Note: Full edge importance requires edge operations (future feature)
# For now, edges can be scored in post-processing:
#
# def score_edges(graph, result):
#     ranks = result.nodes()["node_importance"]
#     edge_scores = {}
#     for u, v in graph.edges():
#         edge_scores[(u, v)] = ranks[u] * ranks[v]
#     return edge_scores
```

**Key insight:** Some metrics require post-processing outside the builder. The builder excels at node-level computations.

## Example 4: Diversity Score

Measure how diverse a node's neighborhood is based on some attribute (e.g., community labels):

```python
@algorithm("diversity_score")
def diversity_score(sG):
    """
    Measure neighborhood diversity based on community labels.
    
    Assumes 'community' attribute exists on nodes.
    High diversity = neighbors from many different communities.
    Low diversity = neighbors mostly from same community.
    """
    # Load community labels
    communities = G.builder.attr.load("community", default=0.0)
    
    # For each node, count how many neighbors share its community
    same_community = (communities == G @ communities)
    fraction_same = same_community.reduce("mean")
    
    # Diversity = 1 - fraction_same
    # (1.0 = all different, 0.0 = all same)
    diversity = 1.0 - fraction_same
    
    return diversity
```

**Wait, this doesn't quite work!** The issue is that `G @ communities` aggregates community labels, but we need to compare each node's community with each neighbor's community individually.

Let's fix it:

```python
@algorithm("diversity_score")
def diversity_score(sG):
    """
    Measure neighborhood diversity based on community labels.
    
    This is a simplified version that estimates diversity based on
    how varied the neighbor community labels are.
    """
    # Load community labels
    communities = G.builder.attr.load("community", default=0.0)
    degrees = communities.degrees()
    
    # Aggregate neighbor community labels
    neighbor_communities = G @ communities
    
    # If all neighbors are in the same community, the sum will be
    # degree * that_community_label. If diverse, sum will vary.
    
    # Average neighbor community
    avg_neighbor_comm = neighbor_communities / (degrees + 1e-9)
    
    # Variance proxy: difference from own community
    diversity = G.builder.core.abs(communities - avg_neighbor_comm)
    
    return diversity.normalize()
```

**Key lesson:** Some metrics require more sophisticated operations than the builder currently supports. Design accordingly!

## Example 5: Multi-Metric Composite Score

Combine multiple metrics into a composite score:

```python
@algorithm("node_quality")
def node_quality_score(G, alpha=0.4, beta=0.3, gamma=0.3):
    """
    Composite quality score combining multiple metrics.
    
    Quality = alpha * centrality + beta * clustering + gamma * diversity
    
    Args:
        alpha: Weight for centrality (default: 0.4)
        beta: Weight for clustering (default: 0.3)
        gamma: Weight for diversity (default: 0.3)
    """
    # Component 1: Centrality (degree-based)
    degrees = G.nodes().degrees()
    centrality = degrees.normalize()
    
    # Component 2: Clustering (neighbor connectivity)
    neighbor_degrees = G @ degrees
    clustering = (neighbor_degrees / (degrees + 1e-9)).normalize()
    
    # Component 3: Diversity (neighbor degree variation)
    avg_neighbor_deg = neighbor_degrees / (degrees + 1e-9)
    diversity = G.builder.core.abs(degrees - avg_neighbor_deg).normalize()
    
    # Combine with weights
    quality = alpha * centrality + beta * clustering + gamma * diversity
    
    # Save components for analysis
    G.builder.attr.save("centrality", centrality)
    G.builder.attr.save("clustering", clustering)
    G.builder.attr.save("diversity", diversity)
    
    return quality

# Usage
algo = node_quality_score(alpha=0.5, beta=0.3, gamma=0.2)
result = G.all().apply(algo)

# Access both composite and components
quality = result.nodes()["node_quality"]
centrality = result.nodes()["centrality"]
clustering = result.nodes()["clustering"]
diversity = result.nodes()["diversity"]
```

**Key techniques:**
- Parameterized algorithms
- Saving intermediate results with `G.builder.attr.save()`
- Normalizing components before combining
- Weighted linear combination

## Example 6: Temporal Activity Score

Score nodes based on recent activity (assumes edge timestamps):

```python
@algorithm("recent_activity")
def recent_activity_score(G, decay=0.9):
    """
    Score nodes based on recent edge activity.
    
    Assumes edges have a 'timestamp' attribute (0 = oldest, 1 = newest).
    Nodes with recent connections get higher scores.
    
    Args:
        decay: Decay factor for older edges (default: 0.9)
    """
    # Load edge timestamps (normalized to [0, 1])
    timestamps = G.builder.attr.load_edge("timestamp", default=0.0)
    
    # Compute edge weights: decay^(1 - timestamp)
    # Recent edges (timestamp near 1) get higher weight
    age = 1.0 - timestamps
    weights = G.builder.core.pow(decay, age)
    
    # Aggregate weighted neighbor activity
    # (This would require weighted aggregation - future feature)
    
    # For now, simple degree-based activity
    degrees = G.nodes().degrees()
    activity = degrees.normalize()
    
    return activity
```

## Example 7: Influence Propagation

Model how influence spreads through a network:

```python
@algorithm("influence")
def influence_propagation(G, sources, decay=0.8, max_iter=10):
    """
    Model influence spreading from source nodes.
    
    Args:
        G: Graph handle
        sources: Binary mask (1.0 = source, 0.0 = not source)
        decay: Influence decay per hop (default: 0.8)
        max_iter: Maximum propagation steps (default: 10)
    
    Returns:
        Influence score for each node
    """
    # This doesn't work because sources is a parameter, not available in decorator!
    # See corrected version below
    pass
```

**Problem:** The `@algorithm` decorator doesn't support additional parameters besides `G`. Let's use a factory pattern:

```python
def influence_propagation_factory(sources_attr="is_source", decay=0.8, max_iter=10):
    """
    Create an influence propagation algorithm.
    
    Args:
        sources_attr: Name of attribute marking source nodes
        decay: Influence decay per hop
        max_iter: Maximum propagation steps
    
    Returns:
        Algorithm that computes influence scores
    """
    @algorithm("influence")
    def influence_propagation(sG):
        # Load source mask
        influence = G.builder.attr.load(sources_attr, default=0.0)
        degrees = influence.degrees()
        
        # Propagate influence
        with G.builder.iter.loop(max_iter):
            # Spread influence to neighbors with decay
            neighbor_influence = G @ influence
            propagated = decay * neighbor_influence / (degrees + 1e-9)
            
            # Keep max of current influence and incoming
            influence = G.builder.core.max(influence, propagated)
        
        return influence
    
    return influence_propagation

# Usage
# 1. Mark source nodes
G.nodes[0]["is_source"] = 1.0
G.nodes[5]["is_source"] = 1.0

# 2. Create algorithm
algo = influence_propagation_factory(sources_attr="is_source", decay=0.9, max_iter=20)

# 3. Apply
result = G.all().apply(algo())
influence_scores = result.nodes()["influence"]
```

**Key technique:** Factory pattern for parameterized algorithms that need external data.

## Composing Algorithms

You can run multiple algorithms in sequence:

```python
# Compute PageRank
pr_algo = pagerank(damping=0.85, max_iter=50)
result1 = G.all().apply(pr_algo)

# Store PageRank as "importance" attribute
for node, rank in result1.nodes()["pagerank"].items():
    G.nodes[node]["importance"] = rank

# Compute quality score using importance
quality_algo = node_quality_score(alpha=0.5, beta=0.3, gamma=0.2)
result2 = G.all().apply(quality_algo)
```

## Best Practices

### 1. **Normalize intermediate values**
```python
# Bad: mixing different scales
score = degrees + pagerank  # Degrees ~ 10, PageRank ~ 0.001

# Good: normalize first
score = degrees.normalize() + pagerank.normalize()
```

### 2. **Use meaningful epsilon values**
```python
# Bad: epsilon too large
inv_deg = 1.0 / (degrees + 1.0)  # Changes results significantly!

# Good: small enough to not affect results
inv_deg = 1.0 / (degrees + 1e-9)
```

### 3. **Avoid redundant computation**
```python
# Bad: recomputing degrees
with G.builder.iter.loop(100):
    degrees = G.nodes().degrees()  # Wasteful!
    ...

# Good: compute once
degrees = G.nodes().degrees()
with G.builder.iter.loop(100):
    # Use degrees
    ...
```

### 4. **Document assumptions**
```python
@algorithm("my_metric")
def my_metric(sG):
    """
    Compute custom metric.
    
    Assumptions:
    - Graph is undirected
    - Nodes have 'weight' attribute (default=1.0)
    - At least 2 nodes exist
    
    Returns:
        Metric value in range [0, 1]
    """
    ...
```

### 5. **Save intermediate results for debugging**
```python
@algorithm("complex_metric")
def complex_metric(sG):
    component1 = ...
    component2 = ...
    
    # Save for debugging/analysis
    G.builder.attr.save("component1", component1)
    G.builder.attr.save("component2", component2)
    
    final = component1 + component2
    return final
```

## Exercises

1. **Embeddedness**: Measure how embedded an edge is (number of common neighbors)
   - Hint: For edge (u,v), count nodes connected to both u and v

2. **K-core**: Find the k-core decomposition (maximal subgraph where all nodes have degree ≥ k)
   - Hint: Iteratively remove nodes with degree < k

3. **Bridge score**: Identify bridge nodes (nodes whose removal would disconnect the graph)
   - Hint: Use connected components before and after hypothetical removal

4. **Assortativity**: Measure tendency for high-degree nodes to connect to other high-degree nodes
   - Hint: Compare node degree with average neighbor degree

5. **Weighted centrality**: PageRank with edge weights
   - Hint: Use `weights` parameter in neighbor aggregation

## Key Takeaways

✅ Combine operations to build complex metrics  
✅ Normalize components before combining them  
✅ Use epsilon terms to avoid division by zero  
✅ Save intermediate results with `G.builder.attr.save()`  
✅ Use factory patterns for parameterized algorithms  
✅ Compute loop-invariant values outside loops  
✅ Document assumptions and value ranges  
✅ Some metrics require post-processing outside the builder  

## Next Steps

- [API Reference: CoreOps](../api/core.md) - See all available operations
- [API Reference: GraphOps](../api/graph.md) - Learn about graph operations
- [Migration Guide](../guides/migration.md) - Convert old code to new DSL
- [Performance Guide](../guides/performance.md) - Optimize your algorithms

---

**You've completed the tutorial series!** You now know how to build sophisticated graph algorithms with the Groggy Builder DSL. Check out the [Examples Gallery](../examples/README.md) for more inspiration!

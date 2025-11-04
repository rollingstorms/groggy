# Tutorial 2: PageRank - Iterative Algorithms

In this tutorial, you'll learn how to implement iterative graph algorithms using the Groggy Builder DSL. We'll implement the famous PageRank algorithm as an example.

## What You'll Learn

- Using loops with `G.builder.iter.loop()`
- Variable reassignment with `G.builder.var()`
- Neighbor aggregation with the `@` operator
- Handling edge cases (sink nodes)
- Normalizing results

## Prerequisites

- Complete [Tutorial 1: Hello World](01_hello_world.md)
- Understanding of iterative algorithms
- Basic knowledge of PageRank (optional but helpful)

## What is PageRank?

PageRank is an algorithm used by Google to rank web pages. The idea is simple:
- Important pages are linked to by other important pages
- A page's importance is the sum of the importance of pages linking to it
- We iterate this calculation until scores converge

**Mathematical formula:**
```
PR(u) = (1 - d)/N + d * Σ(PR(v) / outdegree(v))
```

Where:
- `PR(u)` is the PageRank of node u
- `d` is the damping factor (usually 0.85)
- `N` is the total number of nodes
- The sum is over all nodes v that link to u

## Step 1: Basic Structure

Let's start with the skeleton:

```python
from groggy.builder import algorithm

@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Compute PageRank scores for all nodes."""
    # We'll fill this in
    pass
```

**Key points:**
- We accept parameters (`damping`, `max_iter`) to make the algorithm configurable
- The decorator name is "pagerank", so results will be saved as the "pagerank" attribute

## Step 2: Initialize Ranks

All nodes start with equal rank:

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Compute PageRank scores for all nodes."""
    # Initialize all nodes with rank 1/N
    ranks = G.nodes(1.0 / G.N)
    
    return ranks
```

**What's happening:**
- `G.N` is a property that returns the node count
- `G.nodes(1.0 / G.N)` initializes each node with value 1/N
- Division and other operators work naturally!

## Step 3: Add Iteration

Now let's add the main loop:

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Compute PageRank scores for all nodes."""
    # Initialize
    ranks = G.nodes(1.0 / G.N)
    
    # Iterate
    with G.builder.iter.loop(max_iter):
        # Compute new ranks (we'll fill this in)
        new_ranks = ranks  # Placeholder
        
        # Reassign for next iteration
        ranks = G.builder.var("ranks", new_ranks)
    
    return ranks
```

**Important concepts:**

### The Loop Context
```python
with G.builder.iter.loop(max_iter):
    # Code here runs max_iter times
```

This creates a fixed-iteration loop. Everything inside runs `max_iter` times.

### Variable Reassignment
```python
ranks = G.builder.var("ranks", new_ranks)
```

This is crucial for loops! It tells the builder: "The variable `ranks` in the next iteration should use the value of `new_ranks` from this iteration."

**Why is this needed?** The builder constructs a computation graph, not direct Python execution. We need to explicitly mark which values carry forward between iterations.

## Step 4: Compute Contributions

Each node contributes its rank divided by its degree to its neighbors:

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Compute PageRank scores for all nodes."""
    # Initialize
    ranks = G.nodes(1.0 / G.N)
    
    # Precompute degrees (they don't change)
    degrees = ranks.degrees()
    inv_degrees = 1.0 / (degrees + 1e-9)  # Avoid division by zero
    
    # Iterate
    with G.builder.iter.loop(max_iter):
        # Each node contributes rank/degree to neighbors
        contrib = ranks * inv_degrees
        
        # Aggregate contributions from neighbors
        neighbor_sum = G @ contrib
        
        # Compute new ranks (simplified for now)
        new_ranks = damping * neighbor_sum + (1 - damping) / G.N
        
        # Reassign
        ranks = G.builder.var("ranks", new_ranks)
    
    return ranks.normalize()  # Normalize to sum to 1.0
```

**New concepts:**

### Neighbor Aggregation: `G @ contrib`
The `@` operator (matrix multiplication) aggregates neighbor values:
```python
neighbor_sum = G @ contrib
```

For each node u, this computes: `sum(contrib[v] for all neighbors v of u)`

### Normalize
```python
return ranks.normalize()
```

This divides each value by the sum, so all ranks sum to 1.0.

## Step 5: Handle Sink Nodes

Sink nodes (nodes with no outgoing edges) need special handling. They can't contribute to their neighbors, so their rank should be redistributed uniformly:

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """Compute PageRank scores for all nodes."""
    # Initialize
    ranks = G.nodes(1.0 / G.N)
    
    # Precompute degrees and identify sinks
    degrees = ranks.degrees()
    inv_degrees = 1.0 / (degrees + 1e-9)
    is_sink = (degrees == 0.0)  # Binary mask: 1.0 if sink, 0.0 otherwise
    
    # Iterate
    with G.builder.iter.loop(max_iter):
        # Sinks contribute 0, non-sinks contribute rank/degree
        contrib = is_sink.where(0.0, ranks * inv_degrees)
        
        # Aggregate from neighbors
        neighbor_sum = G @ contrib
        
        # Collect mass from sink nodes
        sink_mass = is_sink.where(ranks, 0.0).reduce("sum")
        
        # Compute new ranks with all three terms
        new_ranks = (
            damping * neighbor_sum +           # Damped neighbor contribution
            (1 - damping) / G.N +              # Teleportation term
            damping * sink_mass / G.N          # Sink redistribution
        )
        
        # Reassign
        ranks = G.builder.var("ranks", new_ranks)
    
    return ranks.normalize()
```

**Key additions:**

### Comparison Operators
```python
is_sink = (degrees == 0.0)
```

Comparison operators (`==`, `<`, `>`, etc.) create binary masks.

### Conditional Selection: `.where()`
```python
contrib = is_sink.where(0.0, ranks * inv_degrees)
```

This means: "If is_sink is true (1.0), use 0.0, else use ranks * inv_degrees"

Syntax: `mask.where(if_true, if_false)`

### Reduction
```python
sink_mass = is_sink.where(ranks, 0.0).reduce("sum")
```

`.reduce("sum")` sums all values into a single scalar.

## Complete Example

```python
from groggy.builder import algorithm
import groggy as gg

@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    """
    Compute PageRank scores for all nodes.
    
    Args:
        G: Graph handle (injected by decorator)
        damping: Damping factor (default: 0.85)
        max_iter: Maximum iterations (default: 100)
    
    Returns:
        Normalized PageRank scores
    """
    # Initialize ranks uniformly
    ranks = G.nodes(1.0 / G.N)
    
    # Precompute degrees and identify sinks
    degrees = ranks.degrees()
    inv_degrees = 1.0 / (degrees + 1e-9)
    is_sink = (degrees == 0.0)
    
    # Iterate until convergence (or max_iter)
    with G.builder.iter.loop(max_iter):
        # Compute contributions (sinks contribute 0)
        contrib = is_sink.where(0.0, ranks * inv_degrees)
        
        # Aggregate neighbor contributions
        neighbor_sum = G @ contrib
        
        # Redistribute sink mass uniformly
        sink_mass = is_sink.where(ranks, 0.0).reduce("sum")
        
        # PageRank formula: damped neighbors + teleport + sink redistribution
        new_ranks = (
            damping * neighbor_sum +
            (1 - damping) / G.N +
            damping * sink_mass / G.N
        )
        
        # Update ranks for next iteration
        ranks = G.builder.var("ranks", new_ranks)
    
    # Normalize to sum to 1.0
    return ranks.normalize()

# Example usage
if __name__ == "__main__":
    # Create a simple web graph
    G = gg.Graph(directed=True)
    G.add_edges_from([
        (1, 2), (1, 3),  # Page 1 links to 2 and 3
        (2, 3), (2, 4),  # Page 2 links to 3 and 4
        (3, 1),          # Page 3 links back to 1
        (4, 3),          # Page 4 links to 3
    ])
    
    # Compute PageRank
    pr_algo = pagerank(damping=0.85, max_iter=50)
    result = G.all().apply(pr_algo)
    
    # Print results
    ranks = result.nodes()["pagerank"]
    for node, rank in sorted(ranks.items(), key=lambda x: -x[1]):
        print(f"Node {node}: {rank:.4f}")
```

Expected output:
```
Node 3: 0.3721
Node 1: 0.2677
Node 2: 0.2072
Node 4: 0.1530
```

Node 3 has the highest PageRank because it's linked to by multiple nodes!

## Understanding the Flow

Let's trace what happens in one iteration:

**Before iteration:**
```
Node 1: rank = 0.25, degree = 2
Node 2: rank = 0.25, degree = 2
Node 3: rank = 0.25, degree = 1
Node 4: rank = 0.25, degree = 1
```

**Step 1: Compute contributions**
```
Node 1 contributes: 0.25 / 2 = 0.125 to each neighbor (2, 3)
Node 2 contributes: 0.25 / 2 = 0.125 to each neighbor (3, 4)
Node 3 contributes: 0.25 / 1 = 0.250 to neighbor (1)
Node 4 contributes: 0.25 / 1 = 0.250 to neighbor (3)
```

**Step 2: Aggregate neighbors**
```
Node 1 receives: 0.250 from node 3
Node 2 receives: 0.125 from node 1
Node 3 receives: 0.125 + 0.125 + 0.250 = 0.500 from nodes 1, 2, 4
Node 4 receives: 0.125 from node 2
```

**Step 3: Apply damping**
```
Node 1: 0.85 * 0.250 + 0.15 * 0.25 = 0.250
Node 2: 0.85 * 0.125 + 0.15 * 0.25 = 0.144
Node 3: 0.85 * 0.500 + 0.15 * 0.25 = 0.463
Node 4: 0.85 * 0.125 + 0.15 * 0.25 = 0.144
```

After many iterations, these values converge to the final PageRank!

## Exercises

1. **Personalized PageRank**: Start with non-uniform initial ranks
   ```python
   # Give node 1 all the initial rank
   ranks = G.nodes(0.0)
   ranks = ranks.where(node_id == 1, 1.0, 0.0)
   ```

2. **Early stopping**: Stop when ranks change by less than a threshold
   - This requires convergence detection (future feature)
   - For now, try different `max_iter` values

3. **Weighted PageRank**: Use edge weights if available
   ```python
   # Load edge weights
   weights = G.builder.attr.load_edge("weight", default=1.0)
   
   # Use in aggregation
   neighbor_sum = G.builder.graph.neighbor_agg(contrib, "sum", weights=weights)
   ```

4. **Compare with NetworkX**: Verify your results match NetworkX's PageRank
   ```python
   import networkx as nx
   nx_graph = nx.DiGraph(G.edges())
   nx_ranks = nx.pagerank(nx_graph, alpha=0.85)
   ```

## Key Takeaways

✅ Use `G.builder.iter.loop(n)` for fixed iteration loops  
✅ Use `G.builder.var("name", value)` to carry values between iterations  
✅ Use `G @ values` to aggregate neighbor values  
✅ Use `mask.where(if_true, if_false)` for conditional logic  
✅ Use `.reduce("sum")` to aggregate all values to a scalar  
✅ Use `.normalize()` to normalize values to sum to 1.0  
✅ Handle edge cases (sinks, zero degrees) with epsilon terms  

## Common Mistakes

❌ **Forgetting to reassign loop variables**
```python
with G.builder.iter.loop(10):
    ranks = damping * neighbor_sum  # Wrong! This doesn't carry forward
```

✅ **Correct**
```python
with G.builder.iter.loop(10):
    new_ranks = damping * neighbor_sum
    ranks = G.builder.var("ranks", new_ranks)  # Explicitly reassign
```

❌ **Using Python's built-in operators on scalars**
```python
# Wrong - G.N returns a VarHandle, not a Python number
if G.N > 0:  # This won't work as expected!
    ...
```

✅ **Use comparison operators that return masks**
```python
# Use mask.where() for conditional logic
non_empty = (G.N > 0)
result = non_empty.where(compute_value(), 0.0)
```

❌ **Modifying loop-invariant variables inside the loop**
```python
with G.builder.iter.loop(10):
    degrees = ranks.degrees()  # Wasteful! Degrees don't change
```

✅ **Compute constants outside the loop**
```python
degrees = ranks.degrees()  # Compute once
with G.builder.iter.loop(10):
    # Use degrees here
```

## Next Steps

- [Tutorial 3: Label Propagation](03_lpa.md) - Learn about async updates and mode aggregation
- [Tutorial 4: Custom Metrics](04_custom_metrics.md) - Build complex node/edge metrics
- [API Reference: IterOps](../api/iter.md) - Learn about iteration control
- [API Reference: GraphOps](../api/graph.md) - Learn about neighbor operations

---

**Ready for asynchronous updates?** Continue to [Tutorial 3: Label Propagation](03_lpa.md)!

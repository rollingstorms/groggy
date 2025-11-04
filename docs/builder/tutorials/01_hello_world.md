# Tutorial 1: Hello World - Your First Algorithm

Welcome to the Groggy Builder DSL! This tutorial will walk you through creating your first custom graph algorithm.

## What You'll Learn

- How to use the `@algorithm` decorator
- Initialize node values
- Perform basic arithmetic operations
- Attach results back to the graph

## Prerequisites

- Groggy installed (`pip install groggy`)
- Basic Python knowledge
- Understanding of graph concepts (nodes, edges)

## The Scenario

Let's create a simple algorithm that computes a "popularity score" for each node based on its degree (number of connections). The formula will be:

```
popularity = degree / max_degree
```

This normalizes degrees to a 0-1 range, where 1.0 means the node has the most connections.

## Step 1: Import and Decorator

```python
from groggy.builder import algorithm

@algorithm("node_popularity")
def compute_popularity(sG):
    """Compute normalized popularity scores based on degree."""
    pass  # We'll fill this in
```

The `@algorithm` decorator transforms your function into a reusable algorithm that can be applied to any graph or subgraph.

**Key points:**
- `sG` represents the **subgraph** your algorithm operates on (could be the full graph or a subset)
- The decorator name (`"node_popularity"`) is what the result attribute will be called
- Return a `VarHandle` to automatically save results

**Why `sG` not `G`?** All operations work on subgraphs, not full graphs. Using `sG` reminds us we're always working with a (potentially filtered) subset.

## Step 2: Get Degrees

```python
@algorithm("node_popularity")
def compute_popularity(sG):
    """Compute normalized popularity scores based on degree."""
    # Get the degree of each node
    degrees = sG.nodes().degrees()
    
    return degrees
```

**What's happening:**
- `sG.nodes()` initializes a variable representing all nodes in the subgraph (with default value 0.0)
- `.degrees()` is a fluent method that computes the degree for each node
- We return `degrees` to save it as the "node_popularity" attribute

## Step 3: Normalize

Now let's normalize by the maximum degree:

```python
@algorithm("node_popularity")
def compute_popularity(sG):
    """Compute normalized popularity scores based on degree."""
    # Get degrees
    degrees = sG.nodes().degrees()
    
    # Find max degree
    max_degree = degrees.reduce("max")
    
    # Normalize to [0, 1]
    popularity = degrees / (max_degree + 1e-9)  # Add small epsilon to avoid division by zero
    
    return popularity
```

**New concepts:**
- `.reduce("max")` aggregates all degree values to find the maximum
- Operator overloading: `degrees / (max_degree + 1e-9)` uses Python's division operator
- The `1e-9` prevents division by zero if all nodes have degree 0

## Step 4: Apply to a Graph

Now let's use our algorithm on a real graph:

```python
import groggy as gg

# Create a simple graph
G = gg.Graph()
G.add_edges_from([
    (1, 2), (1, 3), (1, 4),  # Node 1 has degree 3
    (2, 3), (2, 5),           # Node 2 has degree 3
    (3, 4),                   # Node 3 has degree 3
    (4, 5),                   # Node 4 has degree 3
])

# Create our algorithm
pop_algo = compute_popularity()

# Apply to all nodes
result = G.all().apply(pop_algo)

# Access results
popularity_scores = result.nodes()["node_popularity"]
print(popularity_scores)
# {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.667}
```

**Key points:**
- `G.all()` creates a subgraph containing all nodes and edges
- `.apply(algo)` executes the algorithm on the subgraph
- Results are accessed via `result.nodes()["attribute_name"]`

## Complete Example

Here's the complete code:

```python
from groggy.builder import algorithm
import groggy as gg

@algorithm("node_popularity")
def compute_popularity(sG):
    """Compute normalized popularity scores based on degree."""
    degrees = G.nodes().degrees()
    max_degree = degrees.reduce("max")
    popularity = degrees / (max_degree + 1e-9)
    return popularity

# Create a graph
G = gg.Graph()
G.add_edges_from([
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 5),
    (3, 4),
    (4, 5),
])

# Apply algorithm
algo = compute_popularity()
result = G.all().apply(algo)

# Get results
scores = result.nodes()["node_popularity"]
for node, score in sorted(scores.items()):
    print(f"Node {node}: {score:.3f}")
```

Output:
```
Node 1: 1.000
Node 2: 1.000
Node 3: 1.000
Node 4: 1.000
Node 5: 0.667
```

## Exercises

Try modifying the algorithm to:

1. **Bonus for outdegree**: Give extra weight to nodes with high outdegree
   ```python
   popularity = (0.6 * in_degrees + 0.4 * out_degrees) / max_degree
   ```

2. **Log scaling**: Use logarithmic scaling instead of linear
   ```python
   popularity = G.builder.core.log(degrees + 1) / G.builder.core.log(max_degree + 1)
   ```

3. **Percentile rank**: Compute what percentile each node is in
   - Hint: You'll need to sort and compute ranks

## Key Takeaways

✅ Use `@algorithm` to define reusable graph algorithms  
✅ `G.nodes()` initializes node values  
✅ `.degrees()` computes node degrees  
✅ `.reduce("max")` aggregates values  
✅ Operator overloading (`/`, `+`, etc.) makes math natural  
✅ Return a `VarHandle` to save results automatically  

## Next Steps

- [Tutorial 2: PageRank](02_pagerank.md) - Learn about iterative algorithms
- [API Reference: Core Operations](../api/core.md) - See all available operations
- [API Reference: VarHandle](../api/varhandle.md) - Learn about operator overloading

## Common Mistakes

❌ **Forgetting the decorator**
```python
# Wrong - no decorator
def compute_popularity(sG):
    return G.nodes().degrees()
```

✅ **Correct**
```python
@algorithm("node_popularity")
def compute_popularity(sG):
    return G.nodes().degrees()
```

❌ **Not returning the result**
```python
@algorithm("popularity")
def compute_popularity(sG):
    degrees = G.nodes().degrees()
    # Forgot to return!
```

✅ **Correct**
```python
@algorithm("popularity")
def compute_popularity(sG):
    degrees = G.nodes().degrees()
    return degrees  # or manually save with G.builder.attr.save()
```

❌ **Using Python sum() instead of .reduce()**
```python
# Wrong - trying to use Python's sum on VarHandle
max_degree = sum(degrees)  # This won't work!
```

✅ **Correct**
```python
# Use .reduce() to aggregate
max_degree = degrees.reduce("max")
```

---

**Ready for more?** Continue to [Tutorial 2: PageRank](02_pagerank.md) to learn about iterative algorithms with loops!

# Builder DSL Quick Start

Get started with the Groggy Builder DSL in 5 minutes!

## Installation

```bash
pip install groggy
```

## Your First Algorithm

```python
from groggy import Graph
from groggy.builder import algorithm

# Define an algorithm
@algorithm("popularity")
def popularity(sG):
    """Compute popularity score based on degree."""
    degrees = sG.nodes().degrees()
    max_deg = degrees.reduce("max")
    return degrees / (max_deg + 1e-9)

# Create a graph
g = Graph()
g.add_edges([(1, 2), (1, 3), (2, 3), (3, 4)])

# Run the algorithm
algo = popularity()
result = g.all().apply(algo)
scores = result.nodes()["popularity"]

print(scores)  # {1: 0.67, 2: 0.67, 3: 1.0, 4: 0.33}
```

## Core Concepts

### 1. The `@algorithm` Decorator

Wraps your function to create a reusable algorithm:

```python
@algorithm("my_algorithm")
def my_algorithm(sG, param1=default):
    # Your algorithm logic
    return result_variable
```

### 2. The Subgraph Parameter (`sG`)

**Important:** Always use `sG` (not `G`) as the first parameter.

Why? Because all operations work on **subgraphs** (potentially filtered portions of the graph), not necessarily the full graph.

```python
@algorithm("example")
def example(sG):  # ✅ Correct: reminds us it's a subgraph
    ...

def example(G):   # ❌ Wrong: implies full graph
    ...
```

### 3. Initialize Node Values

```python
# All nodes with value 1.0
ones = sG.nodes(1.0)

# All nodes with value 0.0 (default)
zeros = sG.nodes()

# Each node with unique ID
ids = sG.nodes(unique=True)

# Graph properties
n = sG.N  # Number of nodes
m = sG.M  # Number of edges
```

### 4. Operators Work Naturally

```python
# Arithmetic
result = a + b
result = a * 2.0
result = a / (b + 1e-9)
result = a ** 2

# Comparison (returns mask)
mask = values > 0.5
mask = degrees == 0

# Conditional selection
result = mask.where(if_true, if_false)

# Neighbor aggregation
neighbor_sum = sG @ values
```

### 5. Fluent Methods

```python
# Get degrees
degrees = nodes.degrees()

# Aggregate to scalar
total = values.reduce("sum")
avg = values.reduce("mean")
max_val = values.reduce("max")

# Normalize
normalized = values.normalize()  # Sum to 1.0
```

### 6. Iteration

```python
with sG.builder.iter.loop(max_iter):
    # Compute new values
    new_values = some_computation(old_values)
    
    # Reassign for next iteration
    old_values = sG.builder.var("old_values", new_values)
```

## Common Patterns

### Pattern 1: Node Metric

```python
@algorithm("degree_centrality")
def degree_centrality(sG):
    degrees = sG.nodes().degrees()
    n = sG.N
    return degrees / (n - 1.0 + 1e-9)
```

### Pattern 2: Iterative Algorithm

```python
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    ranks = sG.nodes(1.0 / sG.N)
    deg = ranks.degrees()
    
    with sG.builder.iter.loop(max_iter):
        contrib = ranks / (deg + 1e-9)
        neighbor_sum = sG @ contrib
        ranks = sG.builder.var("ranks",
            damping * neighbor_sum + (1 - damping) / sG.N)
    
    return ranks.normalize()
```

### Pattern 3: Conditional Logic

```python
@algorithm("positive_influence")
def positive_influence(sG):
    # Load existing attribute
    sentiment = sG.builder.attr.load("sentiment", default=0.0)
    
    # Filter to positive nodes
    is_positive = sentiment > 0
    
    # Propagate only from positive nodes
    pos_values = is_positive.where(sentiment, 0.0)
    neighbor_avg = sG.builder.graph.neighbor_agg(pos_values, "mean")
    
    return neighbor_avg
```

### Pattern 4: Multiple Outputs

```python
@algorithm("metrics")
def compute_metrics(sG):
    degrees = sG.nodes().degrees()
    
    # Save intermediate results
    sG.builder.attr.save("degree", degrees)
    
    # Compute and return final metric
    normalized = degrees / degrees.reduce("max")
    return normalized
```

## Available Operations

### Arithmetic
- `+`, `-`, `*`, `/`, `**` (power)
- `//` (floor division), `%` (modulo)

### Comparison
- `==`, `!=`, `<`, `<=`, `>`, `>=`

### Logical
- `~` (invert), `&` (and), `|` (or)

### Graph
- `sG @ values` - Aggregate neighbors (sum)
- `nodes.degrees()` - Get degrees
- `sG.builder.graph.neighbor_agg(values, op)` - Aggregate with op ("sum", "mean", "max", "min", "mode")

### Reduction
- `values.reduce("sum")` - Sum all values
- `values.reduce("mean")` - Average
- `values.reduce("max")` - Maximum
- `values.reduce("min")` - Minimum

### Utility
- `values.normalize()` - Normalize to sum to 1.0
- `mask.where(if_true, if_false)` - Conditional selection

## Next Steps

**Learn More:**
- [Tutorial 1: Hello World](tutorials/01_hello_world.md) - Basics
- [Tutorial 2: PageRank](tutorials/02_pagerank.md) - Iterative algorithms
- [Tutorial 3: Label Propagation](tutorials/03_lpa.md) - Community detection
- [Tutorial 4: Custom Metrics](tutorials/04_custom_metrics.md) - Advanced patterns

**API Reference:**
- [CoreOps](api/core.md) - Arithmetic and reductions
- [GraphOps](api/graph.md) - Topology operations
- [AttrOps](api/attr.md) - Attribute loading/saving
- [IterOps](api/iter.md) - Control flow

## Tips

1. **Always use `sG`** as the parameter name (not `G`)
2. **Add small epsilon** when dividing: `a / (b + 1e-9)`
3. **Use `sG.builder.var()`** to reassign variables in loops
4. **Return a VarHandle** to automatically save results
5. **Check tutorials** for more examples and patterns

## Common Mistakes

### ❌ Using `G` instead of `sG`
```python
def my_algo(G):  # Wrong - implies full graph
    ...
```

### ✅ Use `sG`
```python
def my_algo(sG):  # Correct - reminds us it's a subgraph
    ...
```

### ❌ Forgetting the decorator
```python
def pagerank(sG):  # Just a function, not an algorithm
    return ranks
```

### ✅ Add `@algorithm`
```python
@algorithm("pagerank")
def pagerank(sG):  # Now it's an executable algorithm
    return ranks
```

### ❌ Division by zero
```python
result = a / b  # Could fail if b == 0
```

### ✅ Add epsilon
```python
result = a / (b + 1e-9)  # Safe even if b == 0
```

## Get Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/groggy/issues)
- **Docs:** [Full Documentation](https://groggy.readthedocs.io)
- **Examples:** See `docs/builder/tutorials/` for more

Happy graph computing! 🎉

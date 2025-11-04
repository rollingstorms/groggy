# Builder DSL Tutorials

Welcome to the Groggy Builder DSL tutorial series! These hands-on tutorials will teach you how to build custom graph algorithms using the intuitive, Python-native builder interface.

## Tutorial Series

### [1. Hello World - Your First Algorithm](01_hello_world.md)
**Time:** 15 minutes  
**Difficulty:** Beginner

Learn the basics:
- Using the `@algorithm` decorator
- Initializing node values
- Basic arithmetic operations
- Applying algorithms to graphs

**You'll build:** A node popularity metric based on degree

---

### [2. PageRank - Iterative Algorithms](02_pagerank.md)
**Time:** 30 minutes  
**Difficulty:** Intermediate

Learn about iterations:
- Fixed-iteration loops
- Variable reassignment between iterations
- Neighbor aggregation with `@` operator
- Handling edge cases (sinks)
- Normalization

**You'll build:** The famous PageRank algorithm

---

### [3. Label Propagation - Asynchronous Updates](03_lpa.md)
**Time:** 25 minutes  
**Difficulty:** Intermediate

Learn about async updates:
- Mode aggregation (most common value)
- Synchronous vs asynchronous updates
- `neighbor_mode_update()` for efficiency
- Community detection

**You'll build:** A label propagation community detection algorithm

---

### [4. Custom Metrics - Advanced Compositions](04_custom_metrics.md)
**Time:** 45 minutes  
**Difficulty:** Advanced

Learn to combine operations:
- Multi-component metrics
- Conditional logic
- Saving intermediate results
- Factory patterns for parameterized algorithms
- Best practices and common patterns

**You'll build:** Multiple custom metrics including clustering coefficient, composite scores, and influence propagation

---

## Learning Path

```
Start Here
    ↓
Tutorial 1: Hello World
    ↓
Tutorial 2: PageRank ←──────┐
    ↓                        │
Tutorial 3: LPA              │ Can do in
    ↓                        │ either order
Tutorial 4: Custom Metrics ──┘
    ↓
Check out the API Reference
    ↓
Build your own algorithms!
```

## Quick Reference

### Core Concepts by Tutorial

| Concept | Tutorial | Description |
|---------|----------|-------------|
| `@algorithm` decorator | 1 | Define reusable algorithms |
| `sG.nodes()` | 1 | Initialize node values |
| `.degrees()` | 1 | Compute node degrees |
| `.reduce("sum")` | 1 | Aggregate values to scalar |
| Operator overloading | 1, 2 | Use `+`, `*`, `/`, etc. naturally |
| `sG.builder.iter.loop(n)` | 2 | Fixed iteration loops |
| `sG.builder.var()` | 2 | Carry values between iterations |
| `sG @ values` | 2 | Aggregate neighbor values |
| `.where()` | 2 | Conditional selection |
| `.normalize()` | 2 | Normalize values |
| Mode aggregation | 3 | Find most common value |
| `neighbor_mode_update()` | 3 | Async label updates |
| `sG.nodes(unique=True)` | 3 | Initialize with unique IDs |
| Multi-step composition | 4 | Combine operations |
| `sG.builder.attr.save()` | 4 | Save intermediate results |
| Factory patterns | 4 | Parameterized algorithms |

### Common Patterns

**Initialize values:**
```python
all_ones = sG.nodes(1.0)
all_zeros = sG.nodes(0.0)
unique_ids = sG.nodes(unique=True)
```

**Aggregate neighbors:**
```python
neighbor_sum = sG @ values              # Sum
neighbor_avg = sG.builder.graph.neighbor_agg(values, "mean")
neighbor_max = sG.builder.graph.neighbor_agg(values, "max")
neighbor_mode = sG.builder.graph.neighbor_agg(values, "mode")
```

**Conditional logic:**
```python
mask = (values > threshold)
result = mask.where(if_true, if_false)
```

**Loops:**
```python
with sG.builder.iter.loop(max_iter):
    new_value = compute(old_value)
    old_value = sG.builder.var("old_value", new_value)
```

**Normalize:**
```python
normalized = values.normalize()  # Sum to 1.0
normalized_max = values / values.reduce("max")  # Scale to [0, 1]
```

## Prerequisites

- Python 3.8+
- Groggy installed: `pip install groggy`
- Basic graph theory concepts (nodes, edges, degree)
- Python programming basics

## Getting Help

- **API Reference:** See `/docs/builder/api/` for detailed documentation
- **Examples:** Check `/docs/builder/examples/` for more algorithm implementations
- **Issues:** Report bugs or ask questions on [GitHub Issues](https://github.com/yourusername/groggy/issues)

## What's Next?

After completing these tutorials, explore:

1. **[API Reference](../api/README.md)** - Detailed documentation of all operations
2. **[Migration Guide](../guides/migration.md)** - Convert old builder code to new DSL
3. **[Performance Guide](../guides/performance.md)** - Optimize your algorithms
4. **[Examples Gallery](../examples/README.md)** - More algorithm implementations

## Contributing

Found a mistake or have suggestions? We welcome contributions!

- Submit a PR with corrections
- Suggest new tutorials
- Share your own algorithms as examples

---

**Ready to start?** Begin with [Tutorial 1: Hello World](01_hello_world.md)!

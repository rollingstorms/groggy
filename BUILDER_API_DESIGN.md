# Builder API Design Decisions

## Overview

This document explains key design choices in the Groggy builder DSL, particularly around naming conventions and API surface.

---

## Core Design Principle: Subgraph-First Architecture

### Why `sG` (subgraph) instead of `G` (graph)?

**TL;DR**: All operations in Groggy happen on **subgraphs**, not graphs. Using `sG` makes this explicit.

### The Subgraph Model

In Groggy, you never directly operate on a `Graph` object in algorithms. Instead:

```python
# Create a graph
graph = Graph()
graph.add_node()
graph.add_edge(n0, n1)

# But algorithms always receive a *view* (subgraph)
subgraph = graph.view()              # Entire graph
subgraph = graph.view(node_mask=...) # Filtered subset
subgraph = graph.component(0)        # Single component

# Apply algorithm to the subgraph
result = subgraph.apply(my_algorithm)
```

### Why This Matters

1. **Operations are scoped**: `sG @ values` aggregates over neighbors *in the subgraph*, not the full graph
2. **Filtering is first-class**: You can easily run algorithms on subsets without copying data
3. **Consistency**: `sG.N` is "number of nodes in this view", not "all nodes in the graph"
4. **Future-proof**: Enables distributed/partitioned graphs where "the graph" doesn't exist in one place

### Algorithm Signatures

```python
# ✅ Correct: Parameter is a subgraph
@algorithm("pagerank")
def pagerank(sG, damping=0.85, max_iter=100):
    ranks = sG.nodes(1.0 / sG.N)  # Nodes in this view
    with sG.iterate(max_iter):
        ...
    return ranks

# ❌ Misleading: Looks like full graph
@algorithm("pagerank")
def pagerank(G, damping=0.85):
    ranks = G.nodes(1.0 / G.N)  # Which N? Full or view?
    ...
```

### Usage Pattern

```python
# The subgraph is created at apply-time
graph = Graph()
# ... add nodes/edges ...

# Apply to full graph
pr_full = pagerank()
result1 = graph.view().apply(pr_full)

# Apply to component
pr_component = pagerank(max_iter=50)
result2 = graph.component(0).apply(pr_component)

# Apply to filtered view
large_nodes = lambda n: n.degree > 10
pr_filtered = pagerank(damping=0.9)
result3 = graph.view(node_filter=large_nodes).apply(pr_filtered)
```

---

## Iteration API: `sG.iterate()` vs `sG.builder.iter.loop()`

### The Problem

Original design had deeply nested syntax:

```python
with sG.builder.iter.loop(100):
    ranks = sG.builder.var("ranks", ...)
```

This violated the DSL principle of **natural, readable code**.

### The Solution

Move iteration methods directly to `GraphHandle`:

```python
with sG.iterate(100):
    ranks = sG.var("ranks", ...)
```

### Why This Works

1. **Natural semantics**: "iterate over the subgraph" is intuitive
2. **Consistent with other methods**: `sG.nodes()`, `sG @ values`, `sG.N`
3. **Shorter**: 2 fewer levels of nesting
4. **Applies to context**: Works for both node and edge operations

### Node vs Edge Context

The iteration applies to whatever entity type you're working with:

```python
# Node-level iteration (default)
with sG.iterate(100):
    ranks = ...  # operates on nodes

# Edge-level iteration (future)
with sG.edges.iterate(10):
    weights = ...  # operates on edges
```

---

## Trait Namespaces: When to Use `sG.builder.*`

### The Rule

- **Never use**: `sG.builder.iter.*` → use `sG.iterate()` instead
- **Never use**: `sG.builder.var()` → use `sG.var()` instead
- **Do use**: `sG.builder.core.*` for core operations not in `VarHandle`
- **Do use**: `sG.builder.graph_ops.*` for advanced graph operations
- **Do use**: `sG.builder.attr.*` for attribute I/O

### Why Keep Some Namespaces?

These traits provide **specialized operations** that don't fit on handles:

```python
# ✅ Core ops that aren't operators
inv_n = sG.builder.core.recip(n, epsilon=1e-9)
broadcast = sG.builder.core.broadcast_scalar(value, template)

# ✅ Graph ops beyond @ operator
mode_labels = sG.builder.graph_ops.neighbor_mode_update(labels, ordered=True)
collected = sG.builder.graph_ops.collect_neighbor_values(values)

# ✅ Attribute I/O
weights = sG.builder.attr.load("weight", default=1.0)
sG.builder.attr.save("pagerank", ranks)
```

These operations are:
- Too specialized for operator overloading
- Not frequent enough to warrant handle methods
- Domain-specific (belong in traits)

---

## Operator Overloading: What's on `VarHandle`?

### Common Operations → Operators

```python
# Arithmetic
c = a + b
d = a * 2.0
e = a / (b + 1e-9)

# Comparison
mask = a > 0.5
is_zero = a == 0.0

# Logical
combined = mask1 & mask2
inverted = ~mask
```

### Fluent Methods

```python
# Aggregation
total = values.reduce("sum")

# Normalization
normalized = ranks.normalize()

# Conditional
result = mask.where(a, b)

# Graph topology
deg = ranks.degrees()
```

### When NOT to Add to VarHandle

Don't add operations that:
1. Require multiple parameters beyond `self` and one other value
2. Have complex semantics (like `broadcast_scalar`)
3. Are rarely used
4. Are domain-specific (graph topology, attributes, control flow)

Keep `VarHandle` **lean and focused** on common value operations.

---

## Summary of API Layers

| Layer | Purpose | Example | When to Use |
|-------|---------|---------|-------------|
| **Operators** | Common math | `a + b`, `a * 2` | Always prefer for arithmetic |
| **VarHandle methods** | Fluent operations | `ranks.normalize()` | Common single-operand ops |
| **sG methods** | Graph context | `sG.nodes()`, `sG.iterate()` | Graph-level initialization/control |
| **sG.builder.core** | Core ops | `recip()`, `broadcast_scalar()` | Specialized value ops |
| **sG.builder.graph_ops** | Topology | `neighbor_mode_update()` | Advanced graph algorithms |
| **sG.builder.attr** | Attributes | `load()`, `save()` | Attribute I/O |
| **sG.builder.iter** | Control (legacy) | `until_converged()` | Only for unimplemented features |

---

## Design Philosophy

1. **Readability > Brevity**: Code should read like mathematical notation
2. **Subgraph-first**: All operations scope to the input view
3. **Trait separation**: Keep domains (core, graph, attr) separate
4. **Progressive disclosure**: Common ops are short; specialized ops are namespaced
5. **Zero overhead**: Syntactic sugar, not runtime cost

---

## Future Extensions

### Convergence-based iteration
```python
with sG.until_converged(ranks, tol=1e-6):
    ranks = sG.var("ranks", ...)
```

### Edge-level operations
```python
edge_weights = sG.edges(1.0)
with sG.edges.iterate(10):
    edge_weights = sG.edges.var("weights", ...)
```

### Gradient/autograd (matrix view only)
```python
M = sG.to_matrix()
M.requires_grad_(True)
loss = (M @ x - target).pow(2).sum()
loss.backward()
```

Gradients stay **opt-in** and localized to the matrix view - they never leak into the graph algorithm DSL.

---

## Conclusion

The API design prioritizes:
1. **Natural syntax** for common operations (`sG.iterate()` not `sG.builder.iter.loop()`)
2. **Semantic clarity** (subgraph vs graph)
3. **Domain separation** (traits for specialized ops)
4. **Future extensibility** (edges, convergence, gradients)

This creates a DSL that's **easy to read, write, and extend** while maintaining clear boundaries between concerns.

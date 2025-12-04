# Algorithm Builder DSL

The Groggy Algorithm Builder is a powerful Domain-Specific Language (DSL) for composing custom graph algorithms. It provides:

- **Natural mathematical syntax** with operator overloading
- **Trait-based organization** (Core, Graph, Attribute, Iteration)
- **Zero FFI overhead** through IR compilation
- **Decorator-based definitions** for clean, readable code

## Quick Start

```python
import groggy as gr

b = gr.builder("pagerank_builder")
ranks = b.init_nodes(default=1.0)
degrees = b.node_degrees(ranks)
inv_deg = b.core.recip(degrees, epsilon=1e-9)

with b.iterate(20):
    contrib = b.core.mul(ranks, inv_deg)
    neighbor_sum = b.map_nodes(
        "sum(contrib[neighbors(node)])",
        inputs={"contrib": contrib},
    )
    ranks = b.core.add(b.core.mul(neighbor_sum, 0.85), 0.15)
    ranks = b.normalize(ranks, method="sum")

b.attach_as("pagerank", ranks)
pr_algo = b.build()
result = graph.view().apply(pr_algo)
```

## Key Concepts

### Subgraph-First Philosophy

All algorithms operate on **subgraph views**, not full graphs. This means:

- Use `sG` (not `G`) as the parameter name to emphasize subgraph semantics
- `sG.N` is the node count in the subgraph view
- Operations are scoped to the provided view
- Create views with `graph.view()` or `graph.view().filter(...)`

### VarHandle: The Core Abstraction

`VarHandle` represents a variable in the algorithm IR. It supports:

- **Arithmetic operators**: `+`, `-`, `*`, `/`
- **Comparison operators**: `==`, `<`, `>`, `<=`, `>=`, `!=`
- **Fluent methods**: `.normalize()`, `.reduce()`, `.where()`, `.degrees()`
- **Matrix notation**: `sG @ values` for neighbor aggregation

### Trait-Based Operations

Operations are organized into semantic domains:

| Trait | Purpose | Example |
|-------|---------|---------|
| **CoreOps** | Value operations | `x + y`, `x.reduce("sum")` |
| **GraphOps** | Topology operations | `sG @ values`, `sG.builder.graph_ops.degree()` |
| **AttrOps** | Attribute I/O | `sG.builder.attr.load("weight")` |
| **IterOps** | Control flow | `sG.builder.iter.loop(100)` |

## Architecture

```
Algorithm Definition (Python)
         ↓
    @algorithm decorator
         ↓
    AlgorithmBuilder
         ↓
    IR (steps: list of dicts)
         ↓
    Rust Execution Engine
         ↓
    Result (Subgraph with attributes)
```

## Documentation Structure

- **API Reference** - Detailed trait and class documentation
  - [VarHandle](api/varhandle.md) - Variable handles and operators
  - [CoreOps](api/core.md) - Arithmetic and value operations
  - [GraphOps](api/graph.md) - Topology and neighbor operations
  - [AttrOps](api/attr.md) - Attribute loading and saving
  - [IterOps](api/iter.md) - Control flow and iteration

- **[Tutorials](tutorials/README.md)** - Step-by-step guides
  - [Hello World](tutorials/01_hello_world.md) - Your first algorithm
  - [PageRank](tutorials/02_pagerank.md) - Iterative algorithm
  - [Label Propagation](tutorials/03_lpa.md) - Asynchronous updates
  - [Custom Metrics](tutorials/04_custom_metrics.md) - Building custom centrality

## Design Principles

### 1. Natural Syntax

Algorithm definitions should read like mathematical notation:

```python
# Before (verbose)
result = builder.core.add(builder.core.mul(x, 2.0), 1.0)

# After (natural)
result = x * 2.0 + 1.0
```

### 2. Semantic Organization

Operations grouped by conceptual domain:

- **Core**: Pure value operations (math, logic, aggregation)
- **Graph**: Topology-dependent operations (neighbors, degrees)
- **Attr**: Data I/O operations (load, save, attach)
- **Iter**: Control flow operations (loops, convergence)

### 3. Zero Overhead

The DSL compiles to efficient Rust execution:

- All operators build IR, not immediate execution
- Single FFI call per algorithm application
- Future: JIT compilation and fusion optimization

### 4. Composability

Algorithms are functions that return `VarHandle`:

```python
@algorithm
def weighted_pagerank(sG, weight_attr="weight", damping=0.85):
    weights = sG.builder.attr.load_edge(weight_attr, default=1.0)
    ranks = sG.nodes(1.0 / sG.N)
    # ... use weights in computation
    return ranks
```

## Comparison to Other DSLs

| Library | Syntax Style | Domain | Compilation |
|---------|-------------|--------|-------------|
| **Groggy Builder** | Mathematical + fluent | Graphs | IR → Rust |
| JAX | NumPy-like | Tensors | JIT (XLA) |
| PyTorch | Imperative + autograd | Tensors | Eager + JIT |
| NetworkX | Imperative | Graphs | Pure Python |
| GraphBLAS | Matrix algebra | Graphs | Native C |

## Performance Characteristics

- **Setup overhead**: ~100-500µs (IR construction)
- **Execution**: Native Rust speed (~1-10µs per operation)
- **FFI overhead**: Single call per algorithm (~100ns)
- **Memory**: Columnar storage, minimal copies

## Migration from Old Syntax

The builder supports both old and new syntax for backward compatibility:

```python
# Old style (still works)
builder = AlgorithmBuilder("my_algo")
nodes = builder.init_nodes(1.0)
result = builder.core.add(nodes, 1.0)
builder.attach_as("output", result)
algo = builder.build()

# New style (recommended)
@algorithm
def my_algo(sG):
    nodes = sG.nodes(1.0)
    return nodes + 1.0
```

See the tutorials and guides for migration patterns.

## Advanced Features

### Custom Initialization

```python
# Uniform values
nodes = sG.nodes(1.0)

# Unique IDs
labels = sG.nodes(unique=True)

# Proportional to node count
ranks = sG.nodes(1.0 / sG.N)
```

### Conditional Operations

```python
# Mask-based selection
is_sink = (degrees == 0.0)
contrib = is_sink.where(0.0, ranks * inv_deg)

# Equivalent to:
# if is_sink: contrib = 0.0
# else:       contrib = ranks * inv_deg
```

### Reductions

```python
# Sum all values
total = values.reduce("sum")

# Mean, min, max
avg = values.reduce("mean")
min_val = values.reduce("min")
max_val = values.reduce("max")
```

### Neighbor Aggregation

```python
# Sum neighbor values
neighbor_sum = sG @ values

# Or explicitly:
neighbor_sum = sG.builder.graph_ops.neighbor_agg(values, agg="sum")

# With weights:
weighted_sum = sG.builder.graph_ops.neighbor_agg(
    values, agg="sum", weights=edge_weights
)
```

## Examples Gallery

### Degree Centrality

```python
@algorithm
def degree_centrality(sG, normalized=True):
    degrees = sG.builder.graph_ops.degree()
    if normalized:
        return degrees / (sG.N - 1)
    return degrees
```

### Label Propagation

```python
@algorithm
def lpa(sG, max_iter=10):
    labels = sG.nodes(unique=True)
    with sG.builder.iter.loop(max_iter):
        labels = sG.builder.graph_ops.neighbor_mode_update(
            labels, include_self=True, ordered=True
        )
    return labels
```

### Betweenness Approximation

```python
@algorithm
def betweenness_approx(sG, samples=100):
    # Simplified example - real implementation more complex
    centrality = sG.nodes(0.0)
    # ... sampling and path computation
    return centrality.normalize()
```

## Debugging

### Print IR

```python
builder = AlgorithmBuilder("debug")
# ... define algorithm ...
algo = builder.build()
print(algo.to_json())  # See generated IR
```

### Trace Execution

```python
@traced  # Future: print execution trace
@algorithm
def debug_pagerank(sG):
    # Will log each operation
    ...
```

## Roadmap

**Current (Week 3)**: Documentation and optimization foundation

**Near-term**:
- [ ] JIT compilation (`@compiled` decorator)
- [ ] Convergence detection (`sG.builder.iter.until_converged()`)
- [ ] Automatic fusion optimization
- [ ] GPU acceleration

**Long-term**:
- [ ] Matrix views with autograd
- [ ] Distributed execution
- [ ] Visual algorithm builder (Groggy Studio)
- [ ] Domain-specific optimizations

## Contributing

See the repository contributing guidelines for details.

When adding new operations:
1. Add to appropriate trait (core/graph/attr/iter)
2. Update FFI bindings in Rust
3. Add tests and documentation
4. Update examples

## Further Reading

- [VarHandle API](api/varhandle.md) - Operator overloading and fluent methods
- [CoreOps API](api/core.md) - Arithmetic and value operations
- [GraphOps API](api/graph.md) - Topology operations
- Additional historical docs available in the repository for deeper context

# About Groggy

## What is Groggy?

Groggy is a **high-performance graph analytics library** that bridges the gap between graph theory and practical data science. It combines:

- **Graph topology**: Nodes, edges, and the relationships between them
- **Tabular data**: Columnar attribute storage for efficient bulk operations
- **Rust performance**: Memory-safe, high-speed core implementation
- **Python ergonomics**: Intuitive, chainable API that feels natural

## Who is Groggy For?

### Data Scientists
- Work with graph data using familiar pandas-like operations
- Query, filter, and aggregate without leaving your comfort zone
- Seamless integration with NumPy, pandas, and the PyData ecosystem

### ML Engineers
- Build graph neural networks with automatic differentiation
- Efficient feature engineering on graph-structured data
- High-performance embeddings and spectral analysis

### Network Analysts
- Analyze social networks, knowledge graphs, and complex systems
- Run classic graph algorithms (connected components, centrality, etc.)
- Visualize and explore graph structure interactively

### Researchers
- Git-like version control for reproducible graph experiments
- Time-travel queries to analyze graph evolution
- Extensible architecture for custom algorithms

## What Makes Groggy Different?

### 1. Everything is a View

In Groggy, when you work with a graph, you're typically working with **immutable views**:
- Subgraphs are views into the main graph
- Tables are snapshots of graph state
- Arrays are columnar views of attributes
- Matrices represent graph structure or embeddings

This design enables powerful composition without unnecessary copying.

### 2. Delegation Chains

Groggy's signature feature: objects know how to transform into other objects.

```python
result = (
    g.connected_components()    # → SubgraphArray
     .sample(5)                 # → SubgraphArray (filtered)
     .neighborhood(depth=2)     # → SubgraphArray (expanded)
     .table()                   # → GraphTable
     .agg({"weight": "mean"})   # → AggregationResult
)
```

Once you learn the transformation patterns, the entire API becomes intuitive.

### 3. Columnar Storage

Attributes are stored **separately** from graph structure:
- **GraphSpace**: Which nodes/edges are alive (topology)
- **GraphPool**: Where attributes are stored (columnar data)

This separation enables:
- Efficient bulk attribute operations
- Time-series tracking of graph changes
- Memory-efficient storage and versioning

### 4. Three-Tier Architecture

```
┌──────────────────────────────────────┐
│        Python API Layer              │  Intuitive, chainable
│  (Graph, Table, Array, Matrix)       │
├──────────────────────────────────────┤
│          FFI Bridge                  │  Pure translation
│         (PyO3 bindings)              │
├──────────────────────────────────────┤
│         Rust Core                    │  High-performance
│  (Storage, State, Algorithms)        │  algorithms
└──────────────────────────────────────┘
```

- **Rust Core**: All algorithms, storage, and performance-critical code
- **FFI Bridge**: Pure translation layer, no business logic
- **Python API**: User-facing interface optimized for developer experience

## Design Philosophy

### "Everything is a Graph"

Even Groggy itself is a graph:
- **Nodes** = Object types (Graph, Subgraph, Table, Array, Matrix)
- **Edges** = Methods that transform one type into another

This conceptual model makes the library easier to learn and use.

### Performance First, Ergonomics Close Second

- All core operations meet O(1) amortized complexity targets
- Memory usage scales linearly with data size
- FFI overhead <100ns per call for simple operations
- But never at the expense of a confusing API

### Test-Driven Documentation

Every documented feature has a working test that validates it. If it's in the docs, it works. If it works, it's in the docs.

### Columnar Thinking

Optimize for **bulk operations** over single-item loops:
- Process entire attribute columns at once
- Cache-friendly data access patterns
- Leverage SIMD and parallelization where possible

## Project Goals

### Short Term
- Comprehensive API coverage for core graph operations
- Solid foundation of graph algorithms
- Excellent documentation with real-world examples
- Robust testing and benchmarking

### Medium Term
- Advanced graph neural network support
- Integration with PyTorch Geometric and DGL
- Distributed graph processing capabilities
- Rich visualization ecosystem

### Long Term
- Industry-standard graph analytics platform
- Reference implementation for graph data structures
- Foundation for graph machine learning research
- Community-driven algorithm library

## Project Status

Groggy is under active development. The core architecture is stable, but the API is still evolving based on user feedback and real-world usage patterns.

Current version: **0.5.1**

See the [changelog](https://github.com/rollingstorms/groggy/releases) for recent updates and the [roadmap](https://github.com/rollingstorms/groggy/milestones) for planned features.

## Community

Groggy is open source (MIT License) and welcomes contributions:

- **Code**: Bug fixes, new features, performance improvements
- **Documentation**: Tutorials, examples, typo fixes
- **Testing**: Edge cases, performance benchmarks, real-world use cases
- **Ideas**: Feature requests, API design discussions, architectural feedback

Join the community:

- **GitHub**: [rollingstorms/groggy](https://github.com/rollingstorms/groggy)
- **Issues**: [Bug reports and feature requests](https://github.com/rollingstorms/groggy/issues)
- **Discussions**: [Questions and ideas](https://github.com/rollingstorms/groggy/discussions)

## License

Groggy is released under the **MIT License**.

See [LICENSE](https://github.com/rollingstorms/groggy/blob/main/LICENSE) for full text.

---

Ready to get started? Check out the [Installation Guide](install.md) or jump straight to the [Quickstart](quickstart.md).

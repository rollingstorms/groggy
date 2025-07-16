# Groggy: Graph Language Engine (Python & Rust)

Groggy is a powerful, easy-to-use graph language engine written in Rust with a Python API. It is designed for high-performance processing and analysis of large, dynamic graphs, with a focus on simplicity, scalability, and agent/LLM-friendliness.

---

## Architecture Overview

- **Graph**: Entry point; exposes node/edge collections and graph-wide operations.
- **NodeCollection / EdgeCollection**: Manage nodes/edges with batch, filter, and attribute operations.
- **NodeProxy / EdgeProxy**: Interface for individual nodes/edges and their attributes.
- **AttributeManager**: Efficient attribute access and batch updates.
- **GraphState & Snapshots**: Versioned state, branching, and reproducible analysis.
- **ContentPool & ColumnarStore**: Content-addressable and columnar storage for performance.
- **GraphStore**: Versioned, branchable storage engine.

---

## Python API (Mirrors Rust)

### Main Graph Class
```python
import groggy as gr
g = gr.Graph(directed=True)
g.nodes.add({'n1': {'type': 'A'}, 'n2': {'type': 'B'}})
g.edges.add([('n1', 'n2', {'weight': 1.0})])
print('memory stats:', g.info())
```

### Node & Edge Collections
```python
g.nodes.add({'n3': {'attr': 'X'}})
g.nodes.remove(['n1'])
g.nodes.filter('type="A"')
g.edges.add([('n2', 'n3')])
g.edges.filter('weight > 0.5')
```

### Attribute Management
```python
g.nodes.attr.set({'n2': {'score': 0.8}})
attrs = g.nodes.attr.get(['n2'], ['score'])
```

---

## Rust Backend Highlights
- SIMD-optimized, content-addressable, and columnar storage
- Unified attribute/filter management
- Fast snapshotting and branching
- Native Python bindings via PyO3

---

## Design Principles
- **Simplicity**: Intuitive, LLM/agent-friendly API
- **Performance**: Rust core, optimized for large graphs
- **Extensibility**: Easy to add new algorithms/storage
- **Data Science Ready**: ML/AI pipeline integration
- **Consistent**: Python and Rust APIs mirror each other

---

## Example Use Cases
- Data science workflows
- ML/AI graph data pipelines
- Real-time analytics
- LLM/agent-based graph reasoning

---

## Further Reading
- See `agents.md` for agent/LLM integration patterns
- See `docs/` for full API reference

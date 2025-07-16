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
from groggy import Graph
G = Graph(directed=True)
G.nodes.add({'n1': {'type': 'A'}, 'n2': {'type': 'B'}})
G.edges.add([('n1', 'n2', {'weight': 1.0})])
paths = G.shortest_path('n1', 'n2')
```

### Node & Edge Collections
```python
G.nodes.add({'n3': {'attr': 'X'}})
G.nodes.remove(['n1'])
G.nodes.filter(type='A')
G.edges.add([('n2', 'n3')])
G.edges.filter(weight__gt=0.5)
```

### Attribute Management
```python
G.nodes.attr.set({'n2': {'score': 0.8}})
attrs = G.nodes.attr.get(['n2'], ['score'])
```

### Subgraphs & Snapshots
```python
subG = G.subgraph(node_filter='type==A')
snap = G.snapshot()
```

### Algorithms
```python
G.bfs('n1')
G.dfs('n2')
G.connected_components()
G.pagerank()
```

### Utilities
```python
from groggy.utils import create_random_graph, convert_networkx_graph
G2 = create_random_graph(100, 0.1, use_rust=True)
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

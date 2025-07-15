# Groggy for Agents & LLMs

Groggy is designed for seamless integration with intelligent agents and LLMs, enabling:
- Dynamic graph construction and querying
- Fast, cold-start-friendly context setup
- Intuitive, explainable API for code generation and reasoning

---

## Agent/LLM Usage Patterns

### 1. Fast Graph Instantiation
```python
from groggy import Graph
G = Graph(directed=True)
```

### 2. Bulk Operations for Efficiency
```python
G.nodes.add({'n1': {'type': 'A'}, 'n2': {'type': 'B'}})
G.edges.add([('n1', 'n2', {'weight': 1.0})])
```

### 3. Attribute Management
```python
G.nodes.attr.set({'n1': {'score': 0.9}})
attrs = G.nodes.attr.get(['n1'], ['score'])
```

### 4. Filtering & Subgraphs
```python
filtered = G.nodes.filter(type='A')
subG = G.subgraph(node_filter='type==A')
```

### 5. Algorithms for Reasoning
```python
paths = G.shortest_path('n1', 'n2')
components = G.connected_components()
```

### 6. Provenance & State Tracking
```python
snap = G.snapshot()
G.save_state('experiment-1')
```

---

## Agent Integration Guidelines
- Use batch operations for speed and atomicity
- Leverage attribute manager for schema enforcement
- Use subgraphs and snapshots for context isolation
- Prefer high-level API for LLM code generation
- Use provenance/state features for reproducibility

---

## LLM/Agent Best Practices
- Always check `.info()` for graph/collection metadata
- Use `.filter()` and `.attr.get()` for efficient queries
- Avoid direct storage manipulation—use provided API
- For large graphs, use Rust backend for optimal speed

---

## Example: LLM Workflow
```python
from groggy import Graph
G = Graph()
G.nodes.add({'n1': {'role': 'root'}, 'n2': {'role': 'leaf'}})
subG = G.subgraph(node_filter='role=="root"')
paths = G.bfs('n1')
```

---

Groggy is built to make agent-based graph reasoning, planning, and analytics as frictionless as possible.

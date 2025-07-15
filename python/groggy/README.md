# Groggy Python API: Module Guide

This directory contains the Python API for Groggy, a high-performance graph engine with a Rust backend. Each module is documented below with its responsibilities, main classes/functions, and usage examples.

---

## Modules

- **graph.py**: Main Graph class, entry point for all graph operations.
- **collections/**: NodeCollection and EdgeCollection management.
- **algorithms.py**: Graph algorithms (BFS, DFS, shortest path, etc).
- **analysis.py**: Change tracking, provenance, and diff utilities.
- **utils.py**: Utilities for graph generation, conversion, and benchmarking.
- **views.py**: Read-only views and snapshots.
- **graph/state.py**: State and branch management.
- **graph/subgraph.py**: Subgraph creation and metadata.

---

## Quick Start Example
```python
from groggy import Graph
G = Graph(directed=True)
G.nodes.add({'n1': {'type': 'A'}})
G.edges.add([('n1', 'n2')])
```

---

For full API reference, see each module's README below.

# Groggy Python: collections/ Module

Manages node and edge collections, including batch operations, filtering, and attribute management.

- **base.py**: Abstract BaseCollection API.
- **nodes.py**: NodeCollection, NodeProxy, NodeAttributeManager.
- **edges.py**: EdgeCollection, EdgeProxy, EdgeAttributeManager.

## Example: Node/Edge Operations
```python
from groggy import Graph
G = Graph()
G.nodes.add({'n1': {'score': 1.0}})
G.edges.add([('n1', 'n2', {'weight': 2.0})])
filtered = G.nodes.filter(score=1.0)
```

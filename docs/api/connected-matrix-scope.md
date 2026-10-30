# Connected matrix scope validation

Connected `Subgraph.adjacency_matrix(...)`, `adj(...)` and `to_matrix()` validate
the complete selected node/edge membership and endpoint closure against the current
owner. Inactive selected IDs raise errors; the export does not silently remove rows
or edge contributions. Directed/undirected, sparse/dense and weight rules remain
those of the existing [graph matrix constructor](graph-matrix-construction.md).

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes(3)
graph.add_edge(0, 1, weight=2.5)
connected = graph.connected_view()
frozen = connected.pin_view()
owned = connected.adjacency_matrix(sparse=True, weight_attr="weight")
assert owned.shape == (3, 3)
assert owned.get(0, 1) == 2.5
graph.remove_node(2)  # isolated, but still part of connected's selected row identity
try:
    connected.adjacency_matrix(sparse=True, weight_attr="weight")
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("A selected matrix must not silently lose its third row")
assert owned.shape == (3, 3)
assert frozen.adjacency_matrix(sparse=True, weight_attr="weight").get(0, 1) == 2.5
assert graph.view().adjacency_matrix(sparse=True).shape == (2, 2)
```

Endpoint-only changes within selected nodes remain visible to the live connected
export. A revision to an endpoint outside that selection rejects. A fresh current
selection explicitly reflects changed membership. An earlier immutable GraphView
continues to export its captured values/topology, and an earlier returned matrix
is independent owned numerical data.

The method uses the existing instance-sensitive native scope validator and shared
ascending ID caches. It no longer builds two temporary vectors by filtering away
inactive IDs. Warm scope validation is reused; first validation and matrix building
retain their existing selected-size costs. Attribute/weight reading still goes
through the canonical matrix constructor. No new graph owner, mutation, snapshot
format or GIL behavior is introduced by this guard.

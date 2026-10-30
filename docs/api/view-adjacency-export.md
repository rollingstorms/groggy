# Selected adjacency export

`Graph.adjacency_list()`, connected `Subgraph.adjacency_list()` and immutable
`GraphView.adjacency_list()` return dictionaries from original node IDs to lists
of neighbor IDs. Keys and neighbor rows are ascending; isolates retain empty rows.
Lists preserve parallel edge occurrences and one occurrence per self-loop.
Directed inputs export outgoing edges; undirected inputs include both endpoints.

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes(4)
for edge in [(0, 1), (0, 1), (1, 1), (2, 0)]:
    graph.add_edge(*edge)
expected = {0: [1, 1], 1: [1], 2: [0], 3: []}
assert graph.adjacency_list() == expected
connected = graph.connected_view()
frozen = connected.pin_view()
assert connected.adjacency_list() == frozen.adjacency_list() == expected
subset = frozen.select_edges([1, 2])
assert subset.adjacency_list() == {0: [1], 1: [1]}
rows = subset.adjacency_list()
rows[0].clear()
assert subset.adjacency_list()[0] == [1]
graph.remove_edge(1)
assert frozen.adjacency_list() == expected
try:
    connected.adjacency_list()
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("Invalid connected selections cannot skip missing edges")
```

Only selected edge instances contribute. Node selection does not implicitly
reintroduce excluded parallel edges or owner-wide routes. Connected inputs check
current active membership and endpoint closure before reusing CSR. Stale or
malformed inputs raise errors rather than returning partial topology. Their
subsequent calls still read the current owner; this is not an immutable default
migration. Captured inputs use their frozen topology after owner writes or deletion.

Returned dictionaries/lists are independent owned data. Sorting is an export
convention; `neighbors()` may retain its existing edge-occurrence order. Both
represent the same selected outgoing multiset. There is no property capture,
checkpoint, mutation publication, implicit index building or writable owner clone.

## Native preparation and GIL

`Subgraph.adjacency_read()` and `GraphView.adjacency_read()` return an immutable
`AdjacencyRead` containing shared row IDs and CSR only. It stores no property
payload, history pin or writable owner. It can outlive its source and run on a
worker. Its `adjacency_list()` returns an owned ordered map. Connected preparation
shares the validated CSR cache; captured preparation shares the existing native
AlgorithmRead topology builder and releases its property/source handle afterward.

Python connected/graph adapters prepare on the owner thread, then release the GIL
for detached map construction. Captured preparation and export both release the
GIL. Python dictionary/list conversion still occurs under the GIL.

The former connected adapter scanned every selected edge for each node and
ignored direction/read errors. The new path traverses CSR rows once, sorts each
output neighbor list, and builds an ordered map. Sorting/ordered-map construction
and output memory remain real costs. This replaces O(nodes × edges) scanning;
it is not a zero-copy adjacency representation. Initial capture, cold CSR building
and Python marshalling are measured separately where applicable.

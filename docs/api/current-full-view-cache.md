# Current full-view cache

`Graph.view()` retains the existing connected `Subgraph` API. Its cache now uses
an active-membership revision from the Rust core. An unchanged revision returns
the same Python object without allocating node/edge ID vectors.

Equal entity counts do not prove equal membership. Removing/replacing a node or
edge, or trading an edge for a node, invalidates the cache even if the total count
stays unchanged. Restoration also invalidates through space instance identity,
including states that rebuild the same numeric version.

```python
from groggy import Graph

graph = Graph()
graph.add_nodes(2)
graph.add_edge(0, 1)
connected = graph.view()
frozen = connected.pin_view()
assert graph.view() is connected
graph.remove_edge(0)
graph.add_node()  # total node+edge count is still three
current = graph.view()
assert current is not connected
assert current.node_count() == 3
assert current.edge_count() == 0
assert graph.view() is current
assert frozen.node_ids == [0, 1]
assert frozen.edge_ids == [0]
```

Property writes and endpoint-only revisions preserve membership, so they retain
the cached connected object. Its live property/adjacency readers still follow the
current owner and the native topology cache has its separate endpoint-sensitive
revision. This does not pin a connected object's values or change its defaults.
Use `read_view()` or `pin_view()` for immutable state.

Native `GraphSpace::membership_revision()` returns a cloneable/comparable opaque
`MembershipRevision` token. It covers scalar and bulk active-ID activation and
deactivation. No-op membership calls, property mappings and endpoint-only changes
do not change it. It retains only an instance marker, carries no graph payload,
and is not a retained state ID, timestamp or write capability.

The benchmark `benches/current_full_view_cache.py` times the actual Python method
against an unchanged full selection on release in-tree builds. Construction and
immutable capture are excluded. Timing receipts distinguish this narrow cache
operation from capture, algorithm execution, publication and overall FFI latency.

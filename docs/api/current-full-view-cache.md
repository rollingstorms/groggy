# Current full-view cache

`Graph.view()` returns an immutable `Subgraph` read paired with explicit write
methods. Its weak cache uses a complete readable-state revision from the Rust
core. An unchanged revision returns the same still-owned Python object without
allocating node/edge ID vectors or recapturing properties.

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

Property writes and endpoint-only revisions refresh the default capture even
when membership is unchanged. Earlier views keep their original values/topology.
The cache holds a Python weak reference and native weak payload/capture references;
it does not retain an otherwise unowned snapshot. Branch/head and space-instance
changes also invalidate it, as do explicit retained-catalog revisions.

`graph.connected_view()` explicitly exposes the previous live-owner semantics.
Owner operations share a separate lightweight membership cache, avoiding full
property captures for scalar degree and similar work. Its topology cache keeps
its separate endpoint-sensitive revision.

Native `GraphSpace::membership_revision()` returns a cloneable/comparable opaque
`MembershipRevision` token. It covers scalar and bulk active-ID activation and
deactivation. No-op membership calls, property mappings and endpoint-only changes
do not change it. It retains only an instance marker, carries no graph payload,
and is not a retained state ID, timestamp or write capability.

The benchmark `benches/default_immutable_views.py` separately measures cold capture,
warm default and explicit connected calls, and warmed owner degree.
`benches/current_full_view_cache.py` times the actual Python method
against an unchanged full selection on release in-tree builds. Construction and
immutable capture are excluded. Timing receipts distinguish this narrow cache
operation from capture, algorithm execution, publication and overall FFI latency.

# Specialized graph selections

Native `FilteredSubgraph`, `ComponentSubgraph`, and `NeighborhoodSubgraph` now
contain one canonical `Subgraph` selection, plus their filter/component/center
metadata. They share selected membership, ordering, scope validation and topology
caches instead of maintaining separate traversal implementations.

## Selected topology

BFS, DFS, shortest hop paths, neighbors, total degree, connected components and
node/edge selection use the canonical selected topology. Traversal cannot leave
through an excluded node or edge and later re-enter the result. Induction keeps
all selected parallel instances and self-loops; it never adds owner-only edges.
Connected induction scans selected edges once, and explicit edge selection
resolves only requested selected instances. Neither needs to capture owner
property values; native execution scopes remain live until `.freeze()` is called. Default Python
selections are frozen on return; `pin_view()` exposes their owner-free source.
A component wrapper is not assumed to be connected merely because of its name:
its actual selected components are computed. Derived results retain their
specialized wrapper type and metadata context.

Empty induced/edge selections now produce empty specialized results, matching the
base selection contract. Older filtered/component implementations rejected some
empty requests. IDs outside the parent selection are intersected during explicit
node/edge selection, matching existing base behavior. This is distinct from
addressing a nonexistent BFS start or degree ID, which rejects.

Live topology reads validate selected liveness and endpoint closure. An owner
removal or invalid endpoint revision invalidates a connected selection; retained
pool contents do not supply current edges. Filter complement now prunes edges
whose endpoints are not in the complementary selected node set.

## Ownership and capture

`as_subgraph()` borrows the canonical native selection. Cloning that selection or
its specialized wrapper shares node/edge membership instead of cloning both hash
sets. Neighborhood Python conversion uses this shared representation and retains
its existing neighborhood type label. Cache entries keep the same instance/version
validity rules as the base subgraph.

Each native wrapper exposes `freeze()` and `pin_view()`. The component and neighborhood Python
classes also expose it, and public neighborhood-array items retain their existing
base `Subgraph` API. Capturing returns a readonly `GraphView` with the selected
values/topology from that moment. Specialized metadata remains on the wrapper;
this does not add another history or membership representation.

```python
import groggy as gr

graph = gr.Graph()
graph.add_nodes(4)
for source, target in [(0, 1), (1, 2), (2, 3)]:
    graph.add_edge(source, target)
selected = graph.neighborhood([0], radius=1)[0]
frozen = selected.pin_view()
assert frozen.node_ids == [0, 1]
assert frozen.edge_ids == [0]
graph.remove_node(1)
assert frozen.bfs(0).node_ids == [0, 1]
```

Explicit native constructors create current-owner execution scopes; `.freeze()`
keeps specialized metadata while binding read operations to one immutable source.
Frozen filter combinations, complements, additional criteria, statistics and
matches preserve that source. `reapply_filter()` explicitly returns a fresh capture
of current owner data without changing the earlier filter. Frozen neighborhood
expansion, exact hop rows and merges retain the original source and central-node
metadata, with checked hop overflow. Default Python neighborhood/component outputs
are frozen on return. Native integration examples remain in
`tests/specialized_selection_scope.rs` and the binding regression module.

## Default component neighborhoods

ComponentsArray.neighborhood() returns one actual one-hop neighborhood per
component, centered at its lowest node ID, using both incoming and outgoing
connections within that captured component. It retains edge instances between
selected nodes and the captured graph direction. The input components remain
unchanged. Empty collections return an empty array; isolated nodes remain
single-node neighborhoods. Errors propagate rather than dropping a component.

```python
from groggy import Graph
neighborhood_graph = Graph(directed=True)
neighborhood_graph.add_nodes(4)
neighborhood_graph.add_edges([(0, 1), (1, 2), (2, 3)])
component_capture = neighborhood_graph.connected_components()
neighborhood_graph.remove_node(1)
near_first = component_capture.neighborhood()
assert near_first[0].node_ids.to_list() == [0, 1]
assert component_capture[0].node_count() == 4
```

For a different center, depth or traversal workflow, select a component and use
its explicit neighborhood/traversal API. Native GraphView.neighborhood_from_first
accepts hop count and direction and preserves an empty selection's source/type.
The old implementation computed a neighborhood and discarded it, returning the
whole component. The convenience method no longer returns that placeholder.

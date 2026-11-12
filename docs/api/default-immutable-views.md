# Default immutable graph selections

`graph.view()`, node/edge selection, filtering, traversal, grouping, neighborhoods,
component arrays, and returned algorithm selections retain their creation-time
properties and topology. Reading or deriving a selection does not publish a delta.
Removing or rewiring an entity in the working graph leaves earlier views readable.

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes([{"score": 1}, {"score": 2}])
graph.add_edge(0, 1)
old = graph.view()
graph.set_node_attr(0, "score", 9)
graph.remove_edge(0)
assert old["score"].to_list() == [1, 2]
assert old.neighbors(0).to_list() == [1]
assert graph.view().edge_count() == 0
```

## Reads and explicit effects

A default `Subgraph` is an immutable read paired with its explicit write owner.
`pin_view()` returns its same source as an owner-free `GraphView`, suitable for
worker reads. `query()`, tables, matrices, visualization, copies and derived
selections read the original capture. Lazy components retain that capture before
an item is first accessed. Expanded meta-nodes read their retained group revision.

`nodes`, `edges`, `to_nodes()` and `to_edges()` return immutable entity adapters.
Their `ids` is a property; `table()` returns an owned, source-bound component table.
`graph.nodes` and `graph.edges` remain the working graph's explicit accessors.

```python
selected = graph.view()
assert selected.nodes.ids.to_list() == [0, 1]
assert selected.nodes[0]["score"] == 9
assert selected.to_nodes().table().base_table().source_entity_ids == [0, 1]
selected.set_node_attrs({"score": {0: 10, 1: 20}})
assert selected["score"].to_list() == [9, 2]
assert graph.view()["score"].to_list() == [10, 20]
```

Bulk setters, composition and persistent algorithms validate the captured history,
branch, accepted head and exact source values before using the owner. A stale or
unpublished bound source rejects before its effect. An explicit setter leaves its
input unchanged. An algorithm returns a refreshed immutable result: continue from
that result, or obtain a fresh selection for the next write. Atomic pipeline
publication rules remain those of the pipeline API. Component and subgraph-array collapse resolve
all source plans and apply one checked mutation for the entire collection. See
[atomic collection collapse](meta-composition.md#atomic-collection-collapse) for
rejection rules and the remaining multistage pipeline boundary.

```python
try:
    selected.set_node_attrs({"score": {0: 30}})
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("The old source must reject a second write")
assert graph.get_node_attr(0, "score") == 10
```

## Capture identity and compatibility

Related default selections at the same state share one source while a caller holds
it, so their set operations can compose. Independent `read_view()` captures retain
their distinct identity; cross-capture algebra still rejects. A property, endpoint,
membership, branch, head, restoration or retained-catalog change refreshes the
default source. The cache is weak: it keeps no unowned frozen payload alive.

`graph.connected_view()` explicitly requests the previous live-owner behavior.
Its selected IDs remain fixed; current values/endpoints can change, and inactive
or malformed selections reject. Default-facing results derived from compatibility
execution scopes are captured again when returned. Native `Subgraph::new` remains
an explicit execution-scope constructor; use `.freeze()` to bind its reads.
Specialized native wrappers also expose `.freeze()` while keeping their metadata.

```python
live = graph.connected_view()
frozen = graph.view()
graph.set_node_attr(0, "score", 11)
assert live["score"].to_list() == [11, 20]
assert frozen["score"].to_list() == [10, 20]
```

Capture and export costs remain proportional to the state being captured or output
being materialized. A warm default-view cache check is independent of graph size.
Owner operations use a separate lightweight execution cache and do not capture all
properties just to compute a degree. Keeping a selection alive retains its source
snapshot, including the full captured state used by selected reads; it is not a
constant-memory detached copy of only the selected rows.

## Collection readers

SubgraphArray column extraction, table projection, merge and visualization read
captured sources. Multi-column tables align rows by ascending selected node ID
and represent absence as null. The legacy single-column selector returns present
observations (including explicit null), preferring node properties and falling
back to edge properties only when no selected node has that property.

Collection column requests pass captured native views to Rust in one call with
the GIL released. Many groups can run across the existing worker pool; one or
few large groups retain row-parallel projection. Results keep collection order
and ascending IDs within each group. Tables and observations are detached owned
outputs; projection does not retain an extra sorted-membership/column cache or
publish a delta. Native callers can use `GraphView::project_node_columns` and
`GraphView::project_attribute_observations` for the same collection semantics.

```python
from groggy import Graph
collection_graph = Graph(directed=True)
collection_graph.add_nodes([{"x": value} for value in range(5)])
collection_graph.add_edges([(0, 4), (1, 3), (1, 3), (3, 3)])
collection = collection_graph.subgraph_from_edges([1, 2, 3]).neighborhood([1, 3], hops=1)
collection_graph.remove_node(1)
collection_graph.set_node_attr(3, "x", 999)
assert collection[["x"]][0].column("x").to_list() == [1, 3]
owned = collection.merge()
assert owned.is_directed
assert sorted(owned.node_ids) == [1, 3]
assert sorted(owned.edge_ids) == [1, 2, 3]
assert owned.get_node_attr(3, "x") == 3
```

Collection merge unions exact membership from one capture, then uses the existing
checked owned graph materialization. Overlaps count once; no connecting edges or
filler nodes are invented. Native GraphView.merge_all performs one membership
pass and rejects mixed captures. Visualization reads that union directly without
constructing a temporary working graph. A zero-item array has no source coordinate;
its merge retains the existing fresh default-empty graph behavior. A nonempty
array of empty selections still retains its captured source/direction.

Array table/grouping failures and component predicate/truth-conversion errors
propagate instead of silently dropping entries. Caller callbacks keep their own
side effects; this is error propagation, not callback rollback. The component-neighborhood convenience method now returns actual captured
one-hop results; see [specialized selections](specialized-selections.md).
Generic eager marker iterators retain separate unfinished pipeline work.

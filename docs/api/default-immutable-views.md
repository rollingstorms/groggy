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
publication rules remain those of the pipeline API. A component collection resolves
all its sources before its sequential composition effects; this does not introduce
a new outer atomic transaction.

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

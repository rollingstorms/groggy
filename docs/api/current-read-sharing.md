# Shared current read payloads

`Graph.read_view()`, `Graph.query()` and native `pin_current_state()` reuse the
same frozen values, membership and adjacency when the graph has not changed and
an earlier reader still holds that payload. DurableGraph and explicit connected
Subgraph captures use the same core path. This avoids materializing the entire
working graph for each repeated capture.

```python
from groggy import Graph

graph = Graph()
graph.add_node(value=1)
first = graph.query()
second = graph.read_view()
assert first.node_attr(0, "value") == second.node_attr(0, "value") == 1

graph.set_node_attr(0, "value", 2)
third = graph.query()
assert third.node_attr(0, "value") == 2
assert first.node_attr(0, "value") == second.node_attr(0, "value") == 1
```

## Ownership and invalidation

The graph keeps a **weak reference**, so releasing the last reader releases the
cached payload. The next capture then materializes the current state again.
A reader may be a graph view, a derived selection, a table, an explicit index,
or a native pinned read state. The cache creates no extra history retention root.

Property assignment, unsetting, membership changes and endpoint revisions
invalidate reuse. This includes manual property edits whose topology generation
and accepted state ID have not changed. Physical pool maintenance conservatively
invalidates reuse; existing readers retain their owned logical values.
Accepted state identity and graph direction are also checked before reuse.

Each capture receives its own provenance identity, current branch/writeback
coordinates and retained group catalog. Clones and selections derived from one
capture keep that capture's identity. Independently exported node and edge tables
remain independent captures; sharing their payload does not authorize combining
them for graph writeback. Capture a single GraphTable or derive both tables from
one GraphView when shared provenance is needed.

```python
graph.create_branch("other")
graph.checkout_branch("other")
other = graph.query()
assert other.node_attr(0, "value") == 2
# The old captures still belong to their original source branch.
assert first.node_attr(0, "value") == 1
```

## Costs and boundaries

The first capture and the first capture after invalidation still copy the full
active logical state. Keeping a stable view is useful when running multiple
queries or algorithms. Shared captures avoid that copy, but derived selections,
query result lists and explicit indexes can still allocate separately. Retaining
readers across many edits retains each reader's frozen state until it is released.

This optimization changes no publication or checkpoint behavior and builds no
secondary index. It applies to current-state capture; historical/effective-time
reconstruction retains its existing cache policy. It also does not change the
legacy connected Subgraph's live-parent behavior. Use an explicit immutable
capture until the remaining connected-view migration is complete.

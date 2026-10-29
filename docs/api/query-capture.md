# Composable immutable query captures

`graph.query()` now returns the existing immutable `GraphView`. The native method
previously returned an unimplemented error; it now uses the same capture owner as
`read_view()`. DurableGraph and connected Subgraph expose the same entry point.
There is no separate query builder state or second filtering implementation.

```python
from groggy import Graph, GraphView

graph = Graph()
a = graph.add_node(score=3)
b = graph.add_node(score=1)
graph.add_edge(a, b, weight=2)
query = graph.query()
assert isinstance(query, GraphView)
assert query.query_nodes("score > 0", order_by="score") == [b, a]
assert query.query_edges("weight > 0", order_by="weight") == [0]
result = query.filter_nodes("score >= 3")
assert result.node_ids == [a]
assert len(result.edges_table()) == 0
graph.set_node_attr(a, "score", -1)
assert query.node_attr(a, "score") == 3
assert graph.query().query_nodes("score > 0") == [b]
```

## Results, ordering and scope

`query_nodes(filter, *, order_by=None, descending=False, offset=None, limit=None)`
and `query_edges(...)` scan the capture and return owned ordered ID lists. Filters
accept the existing typed predicates or filter syntax. Their ordering/page engine
is shared with `GraphQueryIndex`: exact mixed numeric comparisons, supported
scalar domains, missing/Null last, ascending ID ties, checked offsets/limits and
full matching-row validation even for an empty page. NaN, unsupported scalar
ordering and incompatible mixed domains reject explicitly. Without `order_by`,
IDs are ascending, or descending when requested.

```python
assert query.query_nodes("score > 0", order_by="score", offset=1, limit=1) == [a]
page = query.select_nodes(query.query_nodes("score > 0", order_by="score", limit=1))
assert page.node_ids == [b]
assert page.node_attr(b, "score") == 1
assert query.query_nodes("never_assigned == null") == []
```

ID list order belongs to the page. A GraphView is a graph selection with canonical
ascending membership enumeration; selecting page IDs does not add a row-order
contract to the graph. Node selection keeps only internal edges within its parent
edge scope; edge selection preserves edge instances and their endpoints.
Selections, filtering and traversal retain the original state/time pins.

Ordinary `Graph.filter_nodes/filter_edges` still return their existing connected
Subgraph type. Use `.query()` on that selection to explicitly freeze its current
state. A stale selection rejects capture rather than silently dropping inactive
IDs. The broader connected-view migration remains 26.1/AR4/AR7.5.

## Explicit execution choice

These GraphView query methods deliberately scan captured values. They do not
build, repair or recover an index. For repeated selective lookup, build requested
indexes explicitly and use their query methods. Missing indexes retain their
explicit error/`allow_scan=True` contract; adding the capture API does not change
it. Both methods use the same semantics and ordering kernel.

```python
index = query.build_index(node_properties=["score"], edge_properties=["weight"])
assert index.query_nodes("score > 0", order_by="score") == \
       query.query_nodes("score > 0", order_by="score")
assert index.query_edges("weight > 0", order_by="weight") == \
       query.query_edges("weight > 0", order_by="weight")
```

Captures, scans, explicit index builds and owned page lists allocate memory.
The first owner capture materializes its current read state. Repeated captures
share that frozen payload while a reader holds it and the working state is
unchanged; see [current read sharing](current-read-sharing.md). Each capture keeps
separate provenance. Reuse the resulting query when a stable state is intended. Queries do not publish deltas or checkpoints,
and durable captures do not retain writer locks. Detached scan/order work releases
the Python GIL. Native callers use `Graph::query()` and
`GraphView::query_nodes/query_edges` with the shared `QueryOptions`.

## Historical queries

`read_view_at(T, recorded_by=U)` returns the same GraphView query surface with
effective/recorded pins. A temporal query has no implicit current-head writeback.
An index built on it and graph/table selections derived from it keep those pins.

```python
from groggy import GraphBatch

temporal = Graph()
batch = GraphBatch(effective_at=10)
batch.add_node(score=1)
first = temporal.writer().apply_batch(batch)["state_id"]
batch = GraphBatch(effective_at=20)
batch.set_node_attr(0, "score", 2)
temporal.writer().apply_batch(batch)
cutoff = temporal.publication_metadata(first)["recorded_at"]
past = temporal.read_view_at(25, recorded_by=cutoff)
selected = past.select_nodes(past.query_nodes("score > 0", order_by="score"))
assert selected.node_attr(0, "score") == 1
assert selected.read_time == past.read_time
assert selected.write_state_id is None
```

This advances the existing pinned-query scope in 21.2. It does not finish the
legacy connected Subgraph migration or add another query language. Traversal,
finite patterns, explicit group indexes and table adapters compose through the
existing GraphView surface. See [ordered pinned queries](ordered-pinned-queries.md)
and [immutable views](../immutable-read-views.md).

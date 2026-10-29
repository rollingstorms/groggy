# Ordered pages and bounded captured queries

For the explicit `graph.query()` capture and ordered scan methods on GraphView,
see [query captures](query-capture.md). They share the ordering kernel below.

`GraphQueryIndex.query_nodes/query_edges` filter matching IDs, order them, then
apply pagination. `GraphView.within_hops` creates a bounded immutable graph
selection. These reads create no graph version or checkpoint and rebuild no index.

```python
from groggy import Graph, GraphBatch, NodeFilter, EdgeFilter

g = Graph(directed=True)
batch = GraphBatch(effective_at=10)
nodes = [batch.add_node(score=value) for value in [30, 10, 10, None]]
batch.add_node()
batch.add_edge(nodes[0], nodes[1], weight=3)
batch.add_edge(nodes[0], nodes[1], weight=1)
batch.add_edge(nodes[1], nodes[2], weight=2)
batch.add_edge(nodes[2], nodes[0], weight=None)
g.apply_batch(batch, 0, "fixture", "example")
head = g.current_state_id
view = g.read_view()
index = view.build_index(node_properties=["score"], edge_properties=["weight"])
all_nodes = NodeFilter.not_filter(NodeFilter.node_set([]))
assert index.query_nodes(all_nodes, order_by="score") == [1, 2, 0, 3, 4]
assert index.query_nodes(all_nodes, order_by="score", descending=True, offset=1, limit=2) == [1, 2]
assert index.query_edges(EdgeFilter.and_filters([]), order_by="weight") == [1, 2, 0, 3]
assert g.current_state_id == head
```

## Scalar ordering and pages

Without `order_by`, IDs are ascending (descending when requested). Property sorting
accepts one comparable scalar domain among matching rows: numeric, text, Boolean,
date, UTC instant or duration. Numeric comparisons preserve large integers across
i64/f32/f64; text aliases decode consistently, once per row for compressed text.
NaNs, vector/binary/structured values and incompatible mixed domains reject.
Infinities retain numeric ordering. This follows the scalar ordering conventions
used by builder steps, extended to native temporal values.

Missing properties and explicit Null sort last in both directions. Equal property
values break ties by ascending ID in both directions. Numeric ordering can compare
values equally while typed property equality still distinguishes their widths.

`offset=None` means zero; `limit=None` means all remaining matches. Booleans,
fractional numbers, negative or overflowing counts reject. Large valid offsets
return empty without offset-plus-limit arithmetic. All matching scalar rows are
validated even for `limit=0` or offsets past the end. An empty match contains no
values to order and remains valid.

Results are ordered ID lists. GraphView represents membership: passing a page to
`select_nodes/select_edges` produces a valid graph selection whose usual ID
accessors remain ascending. It does not retain an independent table row order.

## Explicit index availability

```python
try:
    index.query_nodes(NodeFilter.node_set([]), order_by="unindexed")
except ValueError as error:
    assert "index" in str(error)
else:
    raise AssertionError("A requested ordering index must exist")
assert index.query_nodes(all_nodes, order_by="unindexed", allow_scan=True) == [0, 1, 2, 3, 4]
```

The ordering column and filter requirements must be explicitly indexed unless
`allow_scan=True`. Availability checks precede Boolean/empty-result shortcuts.
Named absent columns may have explicit empty indexes; empty ordering names reject
even when scanning is allowed. Existing find/filter methods keep their behavior.

## Bounded traversal

```python
outgoing = view.within_hops([0], 1)
assert outgoing.node_ids == [0, 1]
assert outgoing.edge_ids == [0, 1]
assert view.within_hops([0], 1, direction="incoming").node_ids == [0, 2]
assert view.within_hops([0], 1, direction="both").node_ids == [0, 1, 2]
assert view.within_hops([0, 0, 4], 0).node_ids == [0, 4]
assert view.within_hops([], 10).node_ids == []
```

Traversal includes seeds and nodes reachable within at most `max_hops`. Duplicate
seeds coalesce as in `select_nodes`. Every seed must belong to the captured
selection, even for zero-hop requests. Empty seeds select nothing. Counts and
seed IDs use strict nonnegative integer conversion.

Directions are outgoing (default), incoming or both. Undirected graphs follow
either endpoint in every mode. Iterative breadth-first traversal visits each node
once, including cyclic/parallel/loop input and very large hop bounds. The result
retains **all parent-selected internal edges** among reached nodes, including
parallel IDs; it is an induced selection, not only traversal-tree edges.

```python
restricted = view.select_edges([0, 1])
assert restricted.within_hops([0], 50).node_ids == [0, 1]
try:
    restricted.within_hops([0, 2], 0)
except ValueError as error:
    assert "seed" in str(error)
else:
    raise AssertionError("An out-of-selection seed must reject")
local_index = outgoing.build_index(node_properties=["score"], topology=False)
assert local_index.query_nodes(all_nodes, order_by="score") == [1, 0]
```

## State/time ownership and costs

```python
past = g.read_view_at(15)
bounded = past.within_hops([0], 1)
assert bounded.read_time == past.read_time
assert bounded.write_state_id is None
past_index = bounded.build_index(node_properties=["score"])
g.set_node_attr(0, "score", -1)
assert past_index.query_nodes(all_nodes, order_by="score") == [1, 0]
assert bounded.node_attr(0, "score") == 30
```

Captures preserve properties, membership, branch/effective/recorded clocks and
existing writeback provenance rules through owner mutation/closure. Native calls
release Python's GIL over immutable data. Rust entry points use `QueryOptions`
and `TraversalDirection` under `groggy::query`.

Property pages validate all matches, partition the requested range and sort only
that page. They still read matching values and retain per-match keys; small limits
do not promise bounded-memory streaming, cursor pagination or constant query time. Bounded traversal visits selected
incident edges and constructs the induced selection using visited/queue storage,
without allocating a dense array indexed by the maximum ID. Retained memory
follows the existing shared GraphView capture.

Arbitrary multi-edge pattern matching, specialized group/temporal interval indexes,
ordinary connected Subgraph migration and wider query/expression consolidation
remain in the original checklist. This slice does not certify those features.

## ID pages from existing postings

With no `order_by`, simple property-presence and exact-equality queries page
directly from their existing immutable sorted ID postings. Both ascending and
descending requests copy only the returned page. Exact lookup shares the same
binary-search bounds as the ordinary indexed predicate path. No new index is
built and no automatic update/recovery is introduced.

Missing requested indexes still reject unless `allow_scan=True`. Boolean/range/
topology predicates retain the ordinary executor. Property ordering still validates
all matching scalar values, including empty or out-of-range pages, before selecting
a page. The fast path changes allocation work, not query or validation semantics.

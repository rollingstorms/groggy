# Captured selection algebra and grouping

GraphView composes node and edge membership over one immutable capture. The
results share its frozen properties, topology, time coordinates, retained groups
and checked-write provenance. These reads neither publish a state nor construct a
writable graph. Python captured operations release the GIL during computation.

## Exact membership algebra

`merge_with` unions node/edge sets. `intersect_with` intersects them independently.
`subtract_from` removes the other selection's nodes and edges, including every
edge whose endpoint was removed. Union never induces extra edges between members.
Intersections can contain isolated nodes even when they share no edge instances.

```python
import groggy as gr

graph = gr.Graph()
a, b, c, isolated = graph.add_nodes(4)
first = graph.add_edge(a, b)
second = graph.add_edge(a, b)
last = graph.add_edge(b, c)
root = graph.read_view()
left = root.select_edges([first])
right = root.select_edges([second, last])
assert left.merge_with(right).edge_ids == [first, second, last]
assert left.intersect_with(right).node_ids == [a, b]
assert left.intersect_with(right).edge_ids == []
remaining = root.subtract_from(root.select_nodes([b]))
assert remaining.node_ids == [a, c, isolated]
assert remaining.edge_ids == []
```

Both operands must descend from the **same capture**, such as selections of one
`root` or its `.copy()`. Independent `graph.read_view()` calls are different
captures even when they currently share a frozen payload. Views from different
states, effective times, groups or graphs must not silently identify unrelated
entities merely because their numeric IDs match. Capture the desired source once,
then select both operands from it. No cross-state merge or value reconciliation
is implied by membership union.

`calculate_similarity(other, metric="jaccard")` compares node membership only.
Supported metrics are `jaccard` (intersection/union), `dice` (twice the intersection
/ summed sizes), `cosine` (intersection / geometric mean of sizes), and `overlap`
(intersection / smaller size). Empty denominators return zero. It requires the
same capture too. Native `find_overlaps` returns nonempty intersections in input
order.

## Typed grouping

`group_by_nodes(attribute)` partitions nodes having that property, in the existing
canonical AttrValue order. Each result includes only selected edges with both
endpoints in its node group. `group_by_edges(attribute)` groups exact selected
edge instances and their endpoints. Missing properties are skipped; explicit null
is a real group. Numeric keys retain their existing typed equality rules: integer
values do not narrow through f64, and storage-compatible compact/text variants use
the library's logical equality. Collection, reference and temporal values retain
their existing AttrValue equality and order.

```python
graph.set_node_attr(a, "group", "red")
graph.set_node_attr(b, "group", "red")
graph.set_node_attr(c, "group", None)
graph.set_edge_attr(first, "kind", "fact")
group_source = graph.read_view()
node_groups = group_source.group_by_nodes("group")
assert [v.node_ids for v in node_groups] == [[c], [a, b]]
assert node_groups[1].edge_ids == [first, second]
edge_groups = group_source.group_by_edges("kind")
assert edge_groups[0].edge_ids == [first]
assert edge_groups[0].node_ids == [a, b]
graph.remove_node(a)
assert group_source.group_by_nodes("group")[1].node_ids == [a, b]
```

Node grouping builds key groups once and examines member incidences to retain
internal selected edges. Edge grouping visits each selected edge once before
constructing groups. The grouping map uses typed comparison; large collection
keys can make comparisons expensive. Every result shares the source capture,
and can be passed to captured algorithms, tables, queries or later algebra.

## Connected compatibility adapters

Legacy `Subgraph.merge_with`, `intersect_with`, `subtract_from` and node/edge
grouping reuse these native kernels. Python's three former set-operation
placeholders now delegate to Rust. Connected algebra requires the same graph
owner, validates both selections against one current capture, and preserves the
legacy connected return type and descriptive grouping metadata. Set subtraction
now prunes dangling incident edges. The original connected objects and their
returned connected adapters remain live; explicitly `.pin_view()` for immutable
results. The complete connected-default ownership migration remains open.

# Captured entities and aligned property columns

`GraphView.nodes` and `.edges` expose read-only `GraphViewEntities`. Their entity
handles, iterators, filters and child selections share the same captured state,
selected membership, clocks and retained groups. They have no working graph owner.
Property columns are detached typed `BaseArray` values, aligned with ascending
original IDs. Python releases the GIL for ordering, bulk projection and selection.

## IDs and positions are explicit

Integer brackets and `by_id(id)` always mean **original entity ID** on this new
captured API. `.at(position)` uses ascending ID order, including negative
positions. Node/edge `.ids` returns an exact integer array. Positional order is
built lazily: creating an accessor, checking length/membership or reading an entity
by ID does not sort or copy all IDs.

```python
import groggy as gr

graph = gr.Graph()
graph.add_nodes(5)
graph.remove_node(1)
first = graph.add_edge(0, 2)
parallel = graph.add_edge(0, 2)
last = graph.add_edge(2, 3)
graph.set_node_attr(0, "score", 2.5)
graph.set_node_attr(2, "score", None)
graph.set_node_attr(4, "score", 2**53 + 1)
view = graph.read_view()
nodes = view.nodes
assert nodes.ids.to_list() == [0, 2, 3, 4]
assert nodes[2].id == nodes.at(1).id == 2
assert nodes.at(-1).id == 4
assert 2 in nodes and 1 not in nodes
assert nodes[4].score == 2**53 + 1
```

ID lists/tuples select those IDs, coalescing duplicates; slices select positions
in the sorted ID order. All child graph selections report stable ascending IDs,
including reverse slices: a graph membership set does not promise row order.
`select(iterable)` accepts explicit IDs. Out-of-scope IDs, invalid counts and
boolean/fractional/negative IDs reject; bad positions raise IndexError.
Existing connected accessors retain their current indexing behavior; switching
their defaults requires the remaining compatibility migration.

## Aligned columns, null and absence

`nodes["score"]`, `nodes.score` and `nodes.column("score")` read present columns.
String brackets/`column` also allow an absent column and return aligned nulls.
Attribute notation raises AttributeError when no selected entity has the column.
Missing cells and present nulls both project to None; `presence(name)` distinguishes
them. Scalar entity lookup raises KeyError for an absent property, while explicit
null returns None. `keys`, membership and `to_dict` contain only present keys.
Large columns use the library's existing ordered Rayon projection; small columns
remain serial. Results preserve typed integers, f64, bytes, temporal/collection
values and the existing storage-compatible logical equality rules.

```python
assert nodes["score"].to_list() == [2.5, None, None, 2**53 + 1]
assert nodes.presence("score").to_list() == [True, True, False, True]
assert "score" in nodes[2] and "score" not in nodes[3]
assert nodes[2].to_dict()["score"] is None
assert nodes[nodes.presence("score")].node_ids == [0, 2, 4]
assert nodes[gr.BaseArray([False] * 4)].node_count() == 0
assert nodes[::-2].node_ids == [2, 4]
```

Masks must be exact-length boolean BaseArrays. Null/nonboolean mask values reject;
empty results are valid captured graphs. Lists/tuples are ID selections, not
implicit masks. Retain one accessor when projecting several columns to reuse
its ID ordering. Columns are owned arrays without automatic graph writeback.
`attribute_names()` lists the selected property names.

## Scope-correct entity handles

`GraphViewEntity.kind` is `node` or `edge`. Node `degree`/`neighbors` use selected
captured instances and the canonical direction/loop/parallel conventions. Edge
`source`, `target` and `endpoints` are frozen. Operations for the wrong entity kind
reject explicitly. `meta_subgraph()` retrieves a captured meta-node's retained
source revision. No mutable Node/Edge owner is consulted.

```python
scoped = view.select_edges([first, last])
assert scoped.nodes[0].degree == 1
assert scoped.nodes[0].neighbors == [2]
assert scoped.edges[first].endpoints == (0, 2)
assert scoped.nodes[[0, 2]].edge_ids == [first]  # excluded parallel edge stays excluded
assert [node.id for node in scoped.nodes] == [0, 2, 3]
assert view.nodes.all().node_ids == [0, 2, 3, 4]  # includes the isolate
assert view.edges.all().node_ids == [0, 2, 3]  # edge endpoints only
assert scoped.nodes.filter("score >= 2").node_ids == [0]
```

Node `all()` preserves the exact parent selection including isolates. Edge `all()`
keeps selected instances and their endpoints. Node/edge filters use the canonical
captured executor. Entity attribute syntax (`node.score`) and string brackets
(`node["score"]`) read the same captured property. There are no attribute setters.

## Explicit writeback returns another capture

```python
source = graph.read_view()
old = source.nodes[0]
batch = gr.GraphBatch()
batch.set_node_attr(0, "score", 9)
result = graph.apply_view_batch(source, batch, "explicit update", "example")
assert result["view"].nodes[0].score == 9
assert old.score == 2.5
assert view.nodes[0].score == 2.5
```

Handles/accessors/iterators survive owner mutation, removal, purge and release,
and can read on workers. Every live handle can retain the captured payload; drop
it when its snapshot is no longer needed. Owned arrays may copy large property
values. This completes the explicit captured accessor path, not the full connected
default/mutation-facade migration. See the existing checked pinned-view write API
for publication, stale-source and scope validation.

Native connected Subgraph column helpers now retain aligned missing rows and use
the existing bulk storage reader. See [native column alignment](connected-column-alignment.md)
for their live ownership and explicit-sequence rules.

Native GraphViewEntities.present_column(name) returns owned observations in
ascending entity-ID order, omitting absent cells while retaining explicit null.
column(name) remains the aligned projection; presence(name) distinguishes its
null padding from explicitly stored null. SubgraphArray's compatibility column
selectors delegate to these captured readers.

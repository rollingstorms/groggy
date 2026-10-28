# Captured graph patterns

`GraphPattern` describes a finite conjunction of named nodes and edge instances.
It supports branches, cycles, self-loops, disconnected components and isolated
nodes. Node and edge predicates use the existing `NodeFilter`/`EdgeFilter` objects
or query strings. This is a native pattern API; it does not parse Cypher.

## Build explicitly, then match

```python
from groggy import Graph, GraphPattern

graph = Graph(directed=True)
a = graph.add_node(kind="person")
b = graph.add_node(kind="person")
c = graph.add_node(kind="place")
e0 = graph.add_edge(a, b, relation="knows")
e1 = graph.add_edge(a, b, relation="knows")
e2 = graph.add_edge(b, c, relation="visits")
index = graph.build_index(node_properties=["kind"], edge_properties=["relation"])
pattern = GraphPattern(
    nodes=[{"name": "a", "filter": 'kind == "person"'},
           {"name": "b"}, {"name": "c"}],
    edges=[{"name": "ab", "source": "a", "target": "b",
            "filter": 'relation == "knows"'},
           {"name": "bc", "source": "b", "target": "c",
            "filter": 'relation == "visits"'}],
)
result = index.match_pattern(pattern)
assert result.bindings == [
    {"nodes": {"a": a, "b": b, "c": c}, "edges": {"ab": e0, "bc": e2}},
    {"nodes": {"a": a, "b": b, "c": c}, "edges": {"ab": e1, "bc": e2}},
]
assert not result.truncated
```

`Graph.build_index`, `Subgraph.build_index`, `DurableGraph.build_index` and
`GraphView.build_index` explicitly capture their selected state and build transient
indexes. A connected `Subgraph` captures its values when `build_index` is called;
this does not change the legacy subgraph itself into a frozen view. Invalid stale
selections reject rather than silently dropping identities. Rebuilding remains
explicit. No index definitions are persisted or automatically maintained.

## Binding rules and scope

Names must be nonempty and unique across nodes and edges. Edge endpoints must name
declared nodes. At least one node variable is required; a one-node pattern works.
Node and edge reuse are allowed by default (homomorphic matching). Set
`distinct_nodes=True` and/or `distinct_edges=True` to reject reuse within that
kind. Parallel edge IDs produce separate bindings. There is no implicit merging
of facts, endpoint pairs, or result rows.

Edge direction is `outgoing` by default, relative to the declared source/target;
`incoming` reverses that relation and `both` accepts either. In an undirected
source all three accept either endpoint order. A self-loop edge is considered
once for each applicable binding. Matching uses only the source selection's nodes
and edge instances, including its direction and effective/recorded-time pins.

```python
row = result.match_view(1)
assert row.node_ids == [a, b, c]
assert row.edge_ids == [e1, e2]
graph.set_node_attr(a, "kind", "changed")
assert row.node_attr(a, "kind") == "person"
assert result.source_view.node_attr(a, "kind") == "person"
# Returned Python dictionaries are owned copies, not edits to the result.
copy = result.bindings
copy.clear()
assert len(result) == 2
```

`match_view(row)` includes exactly the bound node and edge IDs, coalescing reused
IDs and retaining isolates. It does **not** induce all source edges among those
nodes. The source view and each match remain immutable and readable after owner
mutation or closure. Effective-time results retain their clocks and no write
capability. This differs from a node subset's induced internal-edge selection.

## Paging and explicit search limits

Rows are lexicographically ordered by node IDs in declared node-variable order,
then edge IDs in declared edge-variable order. Node/edge declaration order is part
of this API. `offset` and `limit` select a deterministic page, with no offset+limit
integer addition. A limit probes for one more complete match: `truncated` is true
only when another complete match was observed. Thus `limit=0` reports whether a
match exists. Huge offsets can still require extensive search.

```python
page = index.match_pattern(pattern, limit=1, max_expansions=100)
assert page.bindings == result.bindings[:1]
assert page.truncated
last = index.match_pattern(pattern, offset=1, limit=1)
assert last.bindings == result.bindings[1:]
assert not last.truncated
```

`max_expansions` caps attempted variable assignments. Exhaustion raises an error
and returns no partial result. `expansions` reports the assignments attempted by a
successful execution. Predicate, topology and candidate preparation are excluded:
this is **not a wall-time or memory budget**. Omitted limits permit exhaustive
search and potentially exponential output. Use bounded pages and assignment
budgets for exploratory queries. Counts must be nonnegative machine-sized Python
integers; booleans, floats and overflowing integers reject.

## Index requirements

Topology must be built for patterns with edges. Every predicate must have its
requested secondary index unless `allow_scan=True` is explicit. All predicates
are prepared before searching, including predicates behind an empty candidate
set. Explicit scan permission uses captured state and may build a local endpoint
lookup for that operation; it does not install an index or rebuild the original.

```python
unindexed = graph.build_index(topology=False)
scanned = unindexed.match_pattern(pattern, allow_scan=True)
# This index captures the graph after the property edit above.
assert len(scanned) == 0
assert not unindexed.has_topology
```

Finite patterns are composed with unary predicates and existing bounded hop
selection. Optional/negative clauses, aggregation, cross-variable property
comparisons and a variable-length path language are not supported by this API.

# Explicit indexes over immutable graph views

`GraphView.build_index(...)` explicitly builds named property indexes and optional
topology indexes. A full `graph.read_view()` gives global selection; a selected
view bounds every posting and result. Index construction never publishes a graph
change. Values and source state stay shared with the captured view; postings store
IDs rather than copies of large property values. Index clones share postings.

```python
from groggy import Graph, GraphQueryIndex, NodeFilter, EdgeFilter, AttributeFilter

g = Graph()
a = g.add_node(score=2**53 + 1, labels=["Person"])
b = g.add_node(score=2**53, labels=["Person", "Evidence"])
c = g.add_node(score=None)
e = g.add_edge(a, b, relationship_type="supports", weight=1.5)
parallel = g.add_edge(a, b, relationship_type="supports", weight=2.5)
head = g.current_state_id
view = g.read_view()
index = view.build_index(node_properties=["score", "labels"],
                         edge_properties=["weight", "relationship_type"])
assert isinstance(index, GraphQueryIndex)
assert index.find_nodes("score > 9007199254740992.0") == [a]
assert index.find_nodes("score == null") == [c]
assert index.nodes_with_label("Person").node_ids == [a, b]
assert index.find_edges('relationship_type == "supports"') == [e, parallel]
assert index.find_edges(EdgeFilter.connects_nodes(a, b)) == [e, parallel]
assert g.current_state_id == head
```

`find_nodes`/`find_edges` return ascending IDs. `filter_nodes`/`filter_edges` return
immutable GraphViews with the original captured clocks, values and membership
scope. Node results keep only selected internal edges; edge results retain their
endpoints and distinct parallel edge IDs. Materializing a node view still needs
to select its induced edges; use ID queries when that graph result is unnecessary.

## Lifecycle

The index is an immutable read object. A graph delta changes the working graph,
not an existing index. `index.rebuild(new_view)` explicitly constructs a new index
with the same column/topology configuration. The old index and its results remain
readable after mutation, deactivation, branch switching and owner closure. Nothing
implicitly rebuilds in response to reads or writes.

```python
old_result = index.filter_nodes("score > 9007199254740992.0")
g.set_node_attr(a, "score", -1)
assert index.find_nodes("score > 9007199254740992.0") == [a]
current = index.rebuild(g.read_view())
assert current.find_nodes("score > 9007199254740992.0") == []
assert old_result.node_attr(a, "score") == 2**53 + 1
assert index.copy().node_properties == ["labels", "score"]
```

Indexes are derived read structures and are not journal records or serialized
pool data. After reopening storage, explicitly capture and build/rebuild the
needed indexes. An existing captured index keeps its own retained values and
does not start reading the reopened owner. Rejected publication changes neither
the working graph nor existing indexes. This policy requires no automatic index
recovery, delta maintenance, background rebuild or persistence migration.

## Explicit availability

Only requested property columns are indexed. Absent named columns have valid
empty indexes. Missing indexes fail, including inside Boolean branches that
would short-circuit to no matches. `allow_scan=True` explicitly permits the
existing scan executor for a query requiring unavailable indexes. There is no
implicit fallback. Empty or duplicate property names reject index construction.

```python
properties = view.build_index(node_properties=["score", "missing"], topology=False)
assert not properties.has_topology
assert properties.find_nodes(NodeFilter.has_attribute("missing")) == []
try:
    properties.find_edges("weight > 1")
except ValueError as error:
    assert "index" in str(error)
else:
    raise AssertionError("Missing indexes must fail")
assert properties.find_edges("weight > 1", allow_scan=True) == [e, parallel]
```

## Predicate semantics

Equality keeps existing typed logical property identity: i64, f32 and f64 are
distinct as already documented; compact integer/text representations preserve
their logical aliases. Null is present, while absence is absent. Inequality
operates only on present values. Boolean NOT complements the captured selection.
Empty node AND matches nothing; empty edge AND matches all captured edges, as in
the established scan API.

Range indexes share exact scalar numeric comparisons without narrowing large
integers. NaNs never satisfy range comparisons; infinities retain their ordering.
Dates, UTC instants and durations have separate compatible temporal domains.
Unsupported ordering domains match nothing, following the existing predicates.
Generic equality also supports vector, structured, binary and reference values.

An indexed temporal property supports explicit time windows, for example:

```python
from datetime import date

dated = Graph()
dated.add_node(observed=date(2024, 1, 1))
dated.add_node(observed=date(2024, 2, 1))
dates = dated.read_view().build_index(node_properties=["observed"], topology=False)
window = NodeFilter.and_filters([
    NodeFilter.attribute_filter("observed", AttributeFilter.greater_than_or_equal(date(2024, 1, 1))),
    NodeFilter.attribute_filter("observed", AttributeFilter.less_than(date(2024, 2, 1))),
])
assert dates.find_nodes(window) == [0]
```

This indexes the named property's values. To query graph state at an effective
time, first use `read_view_at(...)` and build over that captured state. The index
and its children retain the same effective/recorded cutoff:

```python
from groggy import GraphBatch

temporal = Graph()
batch = GraphBatch(effective_at=10)
batch.add_node(value=1)
temporal.apply_batch(batch, 0, "initial", "example")
batch = GraphBatch(effective_at=20)
batch.set_node_attr(0, "value", 2)
temporal.apply_batch(batch, temporal.current_state_id, "change", "example")
past = temporal.read_view_at(15)
past_index = past.build_index(node_properties=["value"], topology=False)
result = past_index.filter_nodes("value == 1")
assert result.node_ids == [0]
assert result.read_time == past.read_time
assert result.write_state_id is None
```

## Topology and limits

`topology=True` indexes selected endpoint pairs, incidence, neighbor membership
and degrees. Endpoint queries honor graph direction; undirected endpoint order
is interchangeable. Neighbor predicates retain their existing incident-neighbor
definition, including incoming directed edges. Degrees share GraphView's selected
loop/multiedge semantics. These queries do not expand the captured edge scope.

These indexes use one explicit immutable lifecycle. Working-Graph index catalogs,
specialized retained group-membership indexes and temporal event/validity-interval
indexes remain separate work in 18.2. A group ID stored as an ordinary property
can already use the generic equality index. Ordinary live Subgraph migration and
broader query languages remain open; no automatic maintenance or production
latency guarantee is implied.

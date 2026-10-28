# Immutable retained group membership indexes

`GraphView.build_group_index()` explicitly indexes typed group references in its
selected node/edge properties and all nested frozen source revisions reachable
from them. Graph, DurableGraph and connected Subgraph also offer
`build_group_index()` as capture/build adapters. The result is an immutable
`GroupMembershipIndex`, with explicit rebuilds and no mutation maintenance.

```python
from groggy import Graph, GroupMembershipIndex

source = Graph()
source.add_nodes(2)
source.set_node_attr(0, "value", 7)
source.add_edge(0, 1)
parent = Graph()
child = parent.retain_group(source.read_view(), kind="child")
parent.create_meta_node(child)
root = Graph()
outer = root.retain_group(parent.read_view(), kind="parent")
meta = root.create_meta_node(outer)
index = root.build_group_index()
assert isinstance(index, GroupMembershipIndex)
assert index.root_groups == [outer]
assert index.group_ids == sorted([outer, child])
assert index.children(outer) == [child]
assert index.parents(child) == [outer]
assert index.node_references(child) == []
assert index.node_references(child, recursive=True) == [(meta, "contains_subgraph")]
```

## Identity and membership

Membership always takes a group revision: `contains_node(group, node)` and
`contains_edge(group, edge)` inspect that revision's frozen source selection.
An entity number from a different source graph does not establish shared identity.
`group_view(group)` returns the exact frozen source with its own state/time pins,
typed properties, direction, selected membership and nested sources. It has no
implicit working-graph write authority.

```python
assert index.contains_node(child, 1)
assert not index.contains_node(outer, 1)
assert index.contains_edge(child, 0)
assert not index.contains_edge(outer, 0)
assert index.group_view(child).node_attr(0, "value") == 7
assert index.group_view(child).write_state_id is None
source.set_node_attr(0, "value", 9)
assert index.group_view(child).node_attr(0, "value") == 7
```

`group_ids` lists canonical frozen revision IDs, including nested revisions;
`root_groups` lists those directly referenced by selected root properties.
Both are sorted. Unreferenced retained pool entries are excluded. A legacy alias
is resolved in its own source context before indexing; the public key is the
canonical revision ID, not an ambiguous legacy number. A membership-only legacy
reference without a frozen source rejects construction. Plain integer values
are not treated as typed group references.

Unknown/out-of-scope group revisions raise ValueError. Within a known revision,
an unknown node/edge simply is not a member. IDs must be nonnegative integers;
booleans, fractional/string IDs and values exceeding native ID width reject.

## Reverse references and nested groups

`node_references(group)` and `edge_references(group)` return sorted owned
`(entity_id, property_name)` pairs from the root selection. All typed reference
properties are included, not only `contains_subgraph`. Different properties on
one entity remain distinct. `recursive=True` includes root references to any
ancestor revision that reaches the requested group. Multiple routes to the same
root property produce one pair. `children(group)` and `parents(group)` report
direct links among reachable group revisions, not root entity IDs.

```python
from groggy import AttrValue

assert root.retain_group(index.group_view(child), kind="child") == child
root.writer().set_node_attr(meta, "evidence", AttrValue.subgraph_ref(child))
new_index = index.rebuild(root.read_view())
assert new_index.node_references(child) == [(meta, "evidence")]
assert new_index.node_references(child, recursive=True) == [
    (meta, "contains_subgraph"), (meta, "evidence")]
assert index.node_references(child) == []
assert index.copy().group_ids == index.group_ids
```

Reading a nested source does not install a standalone writable reference in the
root owner. Explicitly retain that source first when reusing it in a new root
property, as above; the index itself never changes the owner's storage.

Selections bound root reference postings and which revisions can be reached.
Following a selected reference deliberately exposes its explicitly frozen source;
it does not intersect source-local IDs with selected root entity IDs.

```python
empty = root.read_view().select_nodes([]).build_group_index()
assert empty.group_ids == []
assert empty.root_groups == []
assert new_index.view.node_ids == [meta]
```

## Lifetime, costs and limits

Old indexes and derived source views remain readable after source/root mutation,
deactivation, branch switches, owner drop, durable close/reopen and owner-side
pruning. An effective-time root capture preserves its branch/head/read clocks;
each source keeps its separate frozen pins. Reopening storage requires an
explicit new capture/build for new readers. Construction publishes no graph
change and appends no durable record or checkpoint. Connected Subgraph captures
use its existing checked pinning path; its broader live-view migration is separate.

Build visits selected root properties and the properties of each reachable
revision once, sharing frozen views and storing reference/parent/child postings.
Membership uses the existing frozen membership hash sets. Direct reference
lookups visit matching postings; recursive lookups traverse reachable ancestors
and deduplicate/sort their matching root properties. Output costs grow with
result size; recursive enumeration does not enumerate every possible path.
Index clones share postings and values. Native work releases the Python GIL;
argument/result translation holds it. There are no persisted index definitions,
automatic publication updates or recovery rebuilds.

Property equality/label/type/topology indexes remain in
[query indexes](query-indexes.md). Per-change events, activation and value
intervals remain in [effective-time indexes](effective-time-indexes.md) and
[property intervals](property-intervals.md). These families complete the listed
secondary-index scope of checklist 18.2; broader query migration, service scale,
and Graphiti ingest retain their separate roadmap items.

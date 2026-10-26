# Atomic meta-node composition

`selection.collapse(...)` prepares the complete result before modifying the
owner graph: node aggregations, retained membership, boundary edge attributes,
and optional removal of original nodes. On a publishing graph, success creates
one accepted version, with the same default effective/recorded acceptance clock
for its changes. `commit()` can then label that version as a checkpoint.
`Graph(manual=True)` stages the same operation without publishing; a later
explicit commit accepts its pending changes.

```python
from groggy import Graph

graph = Graph(directed=True)
a, b, outside = graph.add_nodes([{"amount": 2}, {"amount": 4}, {}])
graph.add_edges([(outside, a, {"weight": 1}), (b, outside, {"weight": 3})])
before = graph.current_state_id
prior = graph.read_view()
meta = graph.induced_subgraph([a, b]).collapse(
    node_aggs={"total": ("sum", "amount")},
    edge_aggs={"weight": "sum"},
    node_strategy="collapse",
    allow_missing_attributes=False,
)
assert graph.current_state_id == before + 1
assert graph.get_node_attr(meta.id, "total") == 6
assert set(graph.node_ids) == {outside, meta.id}
assert set(meta.expand().node_ids) == {a, b}
assert prior.node_attr(a, "amount") == 2
assert {graph.edge_endpoints(e) for e in graph.edge_ids} == {
    (outside, meta.id), (meta.id, outside)
}
```

## Aggregation and validation

- Node source/target mappings are honored. Count counts selected nodes.
- Strict mode rejects a source with no observations. Lenient mode supplies zero
  for sum/mean/min/max and empty text for first/last/concatenation. Explicit native
  defaults apply only when no observations exist. Null is an observation and
  remains governed by the chosen aggregation's existing numeric/value policy.
- Node IDs and original edge IDs are processed in increasing order. First/last
  follow this order. Text concatenation accepts normal and compact text;
  `concat_unique` sorts distinct logical strings.
- Unknown functions, reserved/duplicate node targets, inactive or non-internal
  selections, unsupported managed external-ID copying and aggregation failures
  reject before entity IDs, group contents, deltas or versions are created.
  Capacity reservations may occur during validation. Allocation aborts and native
  callers corrupting storage through mutable escape hatches are outside this
  rejection guarantee. Python collapse retains its existing `RuntimeError`
  translation for execution failures.

## Boundary edges

Aggregate creates one edge per external endpoint and direction. Directed incoming
and outgoing edges remain distinct. Undirected edges retain the conventional
meta-node-to-external storage orientation. Internal edges are retained as group
membership and are not emitted as self-loops. `keep_external` creates one new edge
per original boundary edge, preserving parallel facts and attributes, except
requested generated metadata (`edge_count=1`, entity marker, source/target).
`drop_all` emits none. New boundary edges always have new internal IDs; managed
external IDs must be assigned through explicit upserts.

The native configuration also supports count-only edges, per-group minimum edge
counts and suppressing automatic edges to existing meta-nodes. Explicit selectors,
preset resolution and real external-pair contraction are described in
[Composer options and contraction](composer-options.md).

## Retained scope

Canonical node identity remains meta; preset/custom domain kinds become semantic
labels. Group storage here retains immutable membership only; portable pool
persistence and full pinned source-state group revisions remain separate checklist
work. This completes the three composer requirements in 09.4b.2b, not every native
constructor, mutable storage accessor, algorithm or table writeback.

Native `Graph::create_meta_node(reference)` also uses prepared execution in both
publishing and manual modes. It validates retained-reference existence before
allocating an ID; membership may include inactive entities. Reference validation
checks storage membership without cloning the group's node/edge sets. This does
not make global `get_meta_nodes()` selection constant time: that API still scans
active nodes. See [creation evidence](../../notes/planning/prepared-meta-creation-2026-10-25.md).

## Frozen temporal source reads

New composition also pins the selected source properties/topology as an immutable
group revision. `graph.read_view_at(T).meta_subgraph(meta_id)` reads that explicit
source, including after original members are deactivated. Subnode updates do not
change older group revisions. To choose another source state, retain a new group
and explicitly publish its reference; see [effective-time reads](effective-time-reads.md).
Connected legacy `expand()` adapters retain their compatibility contract; canonical
frozen group reads use GraphView.meta_subgraph. Their broad migration remains 26.1.

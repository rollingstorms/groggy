# Automatic publication for ordinary Python graph mutations

`Graph()` now publishes checked mutation versions by default. Successful ordinary
creation, scalar/bulk setters, semantic kinds, managed identities/upserts and
removals are visible in recorded history immediately. Each supported bulk call
publishes once. Rejected inputs do not consume entity IDs or change the head.

```python
from groggy import Graph

graph = Graph()
assert graph.automatic_publication
nodes = graph.add_nodes([{"name": "Ada"}, {"name": "Lin"}])
assert graph.current_state_id == 1
edge = graph.add_edge(*nodes, weight=1.5)
assert graph.current_state_id == 2
before = graph.read_view()
graph.set_node_attrs({"score": {nodes[0]: 1, nodes[1]: 2}})
assert graph.current_state_id == 3
assert graph.snapshot_at_commit(3).node_attr(nodes[0], "score").value == 1
assert before.node_attr(nodes[0], "score") is None
assert not graph.has_uncommitted_changes()
assert graph.commit("analysis checkpoint", "owner") == 3
assert len(graph.commit_history()) == 3
```

The native coordinator validates and timestamps the complete operation before
execution. Prepared execution uses internal bulk storage paths, with ordinary
publication dispatch disabled until completion; it does not recursively publish
one version per row. Initial edge properties are included in the edge creation
version, rather than a second setter version. Node-removal cascades share the
same version. Empty count/row/column/removal operations are no-ops.

## Clocks, writers and checkpoints

Ordinary calls use engine acceptance microseconds for their effective-time
default. Use `graph.writer()` or `GraphBatch` for explicit batch/per-change
effective timestamps, schema checks and optimistic expected states. Ordinary
writes advance the head, so an older writer session becomes stale and must be
explicitly refreshed. Passing a schema on one call does not install that schema
on unrelated calls.

`commit()` labels the current published state as an explicit checkpoint; it does
not publish another mutation delta. `commit_history()` contains mutation states,
while `checkpoints()` contains explicit labels. Immutable retained views and
snapshots continue reading their selected versions after ordinary updates.
`reset_hard()` selects the current published head; it does not undo already
published mutations. Earlier retained states remain available through snapshot
reads and existing branch selection. Full revert/effective-time reconstruction
remains separate roadmap work.

## Explicit manual compatibility

Use `Graph(manual=True)` to retain deferred/manual writes. Such pending writes
must be committed or reset before using the checked writer. The prior dirty
`commit()` adapter publishes them once and records a checkpoint. The default
migration does not silently absorb pending manual writes into a new checked
operation.

Rust `Graph::new_publishing()` enables the coordinator policy. Use the fallible
`insert_nodes`, `add_nodes_with_attrs`, `try_add_node`, and ordinary fallible
mutators. `set_automatic_publication` changes policy only on a clean graph.
Existing Rust `Graph::new`/directional/config constructors and infallible
`add_node`/`add_nodes` remain manual compatibility APIs. Retiring or changing
those signatures is a remaining migration step; their errors are not discarded
or converted to panics.

## Imports and pooled operations

`Graph.load_from_path(path)` accepts active contents as one new published import
with current acceptance clocks; it does not recreate the source history. Use
`manual=True` to retain the previous deferred-import behavior. Graph-table and
subgraph-array factories also return publishing graphs after accepting their
privately assembled contents. Existing checked journals retain their recovery
semantics and original clocks.

Ordinary `add_graph` stages base topology and properties as one batch. Managed
external identities require explicit upserts. Cross-pool/meta references require
explicit portable copying and reject before allocation; copying pool-local IDs
as though they were portable could otherwise produce dangling references.

Meta-node creation publishes its required system attributes together. Meta
classification accepts all logical text storage representations, including
compact text. Higher-level collapse plans now stage their aggregations, retained
membership, boundary edges and removals as one checked operation; see
[meta composition](meta-composition.md) for semantics and limits. Remaining
compound algorithm/table writeback routines still need an audit. Explicit mapped
[table property writeback](table-writeback.md) now accepts mixed selected columns
as one checked batch. [Table materialization](table-materialization.md) now accepts
supported active rows/properties into a new publishing graph; connected provenance
and portable retained pool/history copying remain open. Mutable native
pool/space escape hatches and the infallible Rust compatibility surface remain
outside this step. Native [subgraph bulk writes](subgraph-writes.md) now preserve
complete inputs for owner validation, including algorithm output attachment;
the all-edge convenience setter publishes one complete column.

Automatic publication retains more mutation history than deferred writes.
Prefer bulk operations for bulk workloads. The native receipt in
`benches/automatic_publication_2026-10-23.json` records workload time and version
counts; bulk and scalar workloads deliberately retain different histories. It
provides neither an FFI latency guarantee nor a memory/throughput SLA.

# Publishing writes

`Graph.writer(author="writer", schema=None)` is the checked transition from manually grouped writes to automatic publication. Every successful scalar/bulk writer call stages one batch and publishes one state. Initial properties are published with entity creation. Read through the owning graph; an older recorded snapshot remains unchanged.

```python
from groggy import Graph, GraphBatch

g = Graph()
w = g.writer(author="example")
nodes = w.add_nodes_with_attrs([{"name": "Ada"}, {"name": "Lin"}])
first = g.snapshot_at_commit(w.state_id)
edge = w.add_edge(*nodes, relationship_type="KNOWS")
w.set_node_attrs({"score": {nodes[0]: 1, nodes[1]: 2}}, effective_at=100)
assert not g.has_uncommitted_changes()
assert len(g.commit_history()) == 3
assert first.node_attr(nodes[0], "score") is None
```

Writer column inputs map property names to `{id: value}` mappings or `(id, value)` pairs. The writer also supports scalar setters, kinds, removals, bulk node rows, edge rows, and `apply_batch` for mixed operations. Empty mutation batches reject instead of publishing an empty version. A supplied schema is private to the session and applied to each resulting graph. Python copies the definition so later caller edits do not alter the policy.

Python sessions capture an expected state. A second writer moving the head makes the first stale until `refresh()` explicitly targets the latest head. Dirty legacy writes reject publication until they are resolved. Rust `graph.writer(author).with_schema(&schema)` exclusively borrows the graph; its methods use the same Rust coordinator. Legacy graph methods remain available for compatibility and continue to require manual `commit`; they do not inherit a writer's policy. Raw pool/space access remains a low-level interface outside checked mutation guarantees.

## Effective and recorded time

Times are **signed Unix microseconds**; negative values represent dates before the Unix epoch. The engine captures `recorded_at` once after validation and before applying the accepted mutation. It is not caller supplied, a durability acknowledgement time, or a guarantee of monotonically advancing wall-clock time. State IDs distinguish publications even if clocks coincide.

`GraphBatch(effective_at=T)` provides a default effective time. Omitting it uses the captured recorded clock. Individual staged changes accept `effective_at=` overrides. Initial property changes inherit their entity's explicit creation time unless overridden individually. Bulk writer setters accept an effective-time default for all assignments; use a mixed batch for individual overrides.

```python
batch = GraphBatch(effective_at=100)
node = batch.add_node(name="source", effective_at=50)
batch.set_node_attr(node, "score", 0.5, effective_at=-10)
result = w.apply_batch(batch, "timed node")
metadata = g.publication_metadata(result["state_id"])
assert metadata["effective_at"] == 100
scores = [row for row in metadata["change_times"] if row["name"] == "score"]
assert scores[0]["effective_at"] == -10
```

`publication_metadata(state_id)` returns `recorded_at`, the effective default, and a stable list of actual change times. Entries identify change kind, entity ID, optional property name, and effective time. Before/after values remain in the same version's typed delta. A legacy manual commit returns `None`. Unknown/duplicate override targets, invalid reference handles, non-integer timestamps, booleans and integer overflow reject before mutation. Callers cannot override recorded time through a raw batch field.

Implicit edge removal takes the earliest effective removal time of its removed endpoints. An explicit edge removal uses its own effective override/default instead. Timestamp reconstruction in item 16 will consume these records; the old `snapshot_at_timestamp` still selects whole-second commit times and is not an effective-time query. These changes do not establish durable publication or crash recovery, which require the storage work in item 10.

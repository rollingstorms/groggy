# Publishing writes

`Graph()` publishes ordinary writes automatically. `Graph.writer(author="writer", schema=None)` adds an explicit optimistic session, author, schema and effective-time controls. Every successful scalar/bulk writer call stages one batch and publishes one state. Initial properties are published with entity creation. Read through the owning graph; an older recorded snapshot remains unchanged.

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

Python sessions capture an expected state. A second writer moving the head makes the first stale until `refresh()` explicitly targets the latest head. Dirty legacy writes reject publication until they are resolved. Rust `graph.writer(author).with_schema(&schema)` exclusively borrows the graph; its methods use the same Rust coordinator. Explicit `Graph(manual=True)` and native `Graph::new_manual` retain deferred compatibility writes; native node insertion is fallible in both policies. Ordinary Python writes publish independently and invalidate older sessions; they do not inherit a writer's policy. Raw pool/space access remains a low-level interface outside checked mutation guarantees.

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

Implicit edge removal takes the earliest effective removal time of its removed endpoints. An explicit edge removal uses its own effective override/default instead. Timestamp reconstruction in item 16 will consume these records; the old `snapshot_at_timestamp` still selects whole-second commit times and is not an effective-time query. For durable publication and crash recovery, use the [DurableGraph journal](durable-storage.md); an ordinary Graph writer remains in-memory.

## Managed identities and upserts

A mixed batch can explicitly assign or rename external IDs using `set_node_external_id` and `set_edge_external_id`, including handles for entities created in that batch. Name collisions, duplicate assignments and inactive resulting targets reject before allocation or reservation. Schema checks see the proposed identity properties. Reservation/index publication accompanies the typed property delta; old names remain retained after renames.

```python
batch = GraphBatch(effective_at=100)
node = batch.add_node(name="identity")
batch.set_node_external_id(node, "source-identity")
result = w.apply_batch(batch, "named source")
assert g.node_by_external_id("source-identity") == result["added_nodes"][0]
assert w.upsert_nodes([{"external_id": "person", "name": "Ada"}])
state = w.state_id
assert w.upsert_nodes([{"external_id": "person", "name": "Ada"}])
assert w.state_id == state
```

The writer supports managed node/edge upserts and explicit renames. Idempotent upserts publish no version when their normalized payloads are unchanged; they still check the expected head, dirty-state guard and session schema. Edge upserts preserve endpoints, direction and parallel instance IDs. Python uses native `publish_upsert_nodes` / `publish_upsert_edges` to return both IDs and the exact resulting state, avoiding a separate mutable-head read. Ordinary `Graph.upsert_nodes` / `upsert_edges` now publish once by default; explicit manual graphs retain deferred grouping. Custom input identity keys are consumed into the same managed `external_id` property.

## Durable publication

`DurableGraph.writer()` uses the same staged methods and expected-head checks, with a synced journal before publication. Its installed schema applies even when the session supplies no schema. See [durable storage](durable-storage.md) for recovery behavior and current pool/checkpoint/concurrency limits.

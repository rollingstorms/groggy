# External identity and upserts

Groggy allocates internal numeric node and edge IDs. Caller-controlled identity uses a nonempty, case-sensitive text `external_id`, with one uniqueness namespace across nodes, edges and groups in a graph. UUID syntax and normalization belong to the caller or adapter.

```python
from groggy import Graph

graph = Graph(directed=True)
alice, bob = graph.upsert_nodes([
    {"external_id": "person-alice", "name": "Alice", "group_id": "example"},
    {"external_id": "person-bob", "name": "Bob", "group_id": "example"},
])
[fact] = graph.upsert_edges([
    {"external_id": "fact-1", "source": "person-alice", "target": "person-bob",
     "relation": "KNOWS", "group_id": "example"},
])
assert graph.node_by_external_id("person-alice", group_id="example") == alice
assert graph.edge_by_external_id("fact-1") == fact
first = graph.commit("people and fact", "example")

# Patch the same entity. Unspecified properties remain unchanged.
assert graph.upsert_nodes([
    {"external_id": "person-alice", "name": "Alicia"},
]) == [alice]
graph.commit("rename display name", "example")
assert graph.snapshot_at_commit(first).node_by_external_id("person-alice") == alice
```

Each upsert call validates all rows before writing or allocating IDs. Duplicate external IDs within a call reject the entire call, including identical duplicates. A repeated identical successful payload is a no-op. Node and edge calls are separate write boundaries; a combined ingest transaction and automatic publication remain roadmap item 9. Direct writes currently require an explicit `commit` for historical snapshots.

Edge endpoints can be numeric internal IDs or active external node IDs. Existing directed edges must keep their ordered endpoints; undirected input can reverse equivalent endpoints. Distinct external edge IDs can identify parallel facts or self-loops. Endpoint changes require a future explicit rewire operation.

`upsert_nodes(rows, uid_key="uuid")` and `upsert_edges(rows, uid_key="uuid")` can read another input key. That key is consumed into the managed `external_id` column. It is not also stored as an ordinary property. Edge identity keys must differ from `source` and `target`.

Use `set_node_external_id(node, name)` or `set_edge_external_id(edge, name)` for explicit initial assignment or rename. The same entity keeps its numeric ID. Old names remain reserved, even after removal, branch switching or reset of uncommitted work. An upsert cannot silently resurrect an inactive identity or claim its old name. `allocated_entity_for_external_id(name)` returns `("node", id)` or `("edge", id)` for retained bindings, including inactive ones; reading the corresponding values still requires the appropriate historical snapshot. Active lookup returns `None` for removed entities, old names and a nonmatching group.

The `external_id` column is readable through ordinary property APIs but writable only through the dedicated identity API. Scalar/bulk setters, ordinary creation, mixed batches and subgraph writes reject direct assignment. `add_node(node_id=...)` rejects custom numeric IDs. `add_graph` rejects copying a graph with managed identities before allocating; use explicit upserts to choose identity semantics.

Legacy `add_nodes(..., uid_key=...)` still returns an ordinary-property mapping. It rejects duplicate input or existing values before allocation. Later ordinary property changes are not constrained by that convenience API; ambiguous string resolution or forward mapping raises an error. Managed identity provides the lasting uniqueness guarantee.

Active identity lookup uses a retained primary index and checks the selected graph state and optional group property. Snapshot identity lookup scans pinned snapshot properties. A snapshot materialized as an independent graph rebuilds only its selected-state bindings. Current JSON export/import preserves active IDs and current names; complete history and retained reservations need the versioned storage work in item 10. Purging reservations belongs to item 15.

Rust exposes `NodeUpsert`, `EdgeUpsert`, `ExternalEntityRef` and the same methods on `Graph`. Upsert records contain the external ID and a property map; edge records also contain numeric endpoints.

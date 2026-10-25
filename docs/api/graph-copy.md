# Copy active graphs

`destination.add_graph(source)` copies active base nodes, active edge instances and
current typed properties. Source IDs are sorted before assigning fresh destination
IDs; inactive allocation gaps are preserved only in the source. The destination
keeps its direction/configuration. This API copies endpoint records without changing
the destination's directed/undirected interpretation.

```python
from groggy import Graph
source = Graph(directed=True, manual=True)
source.add_nodes([{}, {"score": 1.0000000000000002}, {}])
source.remove_node(0)
source.add_edges([(1, 2), (1, 2), (2, 2)])
destination = Graph(directed=True)
destination.add_node()
before = destination.current_state_id
destination.add_graph(source)
assert destination.current_state_id == before + 1
assert destination.get_node_attr(1, "score") == 1.0000000000000002
assert [destination.edge_endpoints(i) for i in [0, 1, 2]] == [(1, 2), (1, 2), (2, 2)]
assert source.has_uncommitted_changes()
```

Publishing mode accepts one operation/version. `manual=True` uses the same prepared
validation and bulk installation but leaves a pending change for an explicit commit.
A dirty publishing destination rejects a nonempty copy. Empty sources are no-ops.
Copying never commits or mutates the source, including a dirty source.

```python
manual = Graph(manual=True)
manual.add_graph(source)
assert manual.current_state_id == 0
assert manual.has_uncommitted_changes()
manual.commit("copy", "owner")
assert not manual.has_uncommitted_changes()
```

Native typed `NodeArray` and `EdgeArray` property references are remapped using the
complete source-to-destination maps, preserving order and duplicates, across both
node and edge properties. References to inactive/missing source entities reject.
Ordinary integers, integer vectors and JSON are opaque data and retain their values.
Graph tables use the same reference-remapping function when materializing.

Managed external IDs require explicit upserts; meta nodes and pooled `SubgraphRef`
values require portable source pool contents and reject. `contains_subgraph` cannot
smuggle a pooled reference through an untyped non-null value. Corrupt source value
indices and dangling endpoints reject instead of silently disappearing. Validation
precedes destination ID allocation and writes, in both modes.

```python
managed = Graph()
managed.upsert_nodes([{"external_id": "existing"}])
empty = Graph(manual=True)
try:
    empty.add_graph(managed)
except ValueError as error:
    assert "explicit upserts" in str(error)
else:
    raise AssertionError("managed identities need explicit upserts")
assert empty.add_node() == 0
```

This is independent active materialization, not copying source history, branch IDs,
retained inactive entities or immutable source provenance. Source capture/staging is
linear in active values/topology plus scanning current attribute-index bindings,
with sorting for deterministic ID assignment. Values are owned by the destination;
large buffers must be copied once. See the [bounded release receipt and verification
notes](../../notes/planning/graph-copy-2026-10-25.md).

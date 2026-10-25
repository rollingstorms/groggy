# Specialized graph-table merges

`GraphTable.merge(tables, strategy="domain_prefix")` combines complete node/edge
tables in Rust. It preserves property columns, remaps endpoints and typed local
references, and retains supplied immutable source captures. It returns owned
combined tables; it does not mutate an owner graph or publish a graph state.
Python releases the GIL over native merge execution.

```python
from groggy import Graph, GraphTable
left = Graph()
left.add_nodes([{"x": 1}, {"x": 2}])
left.add_edges([(0, 1, {"weight": 1.125})])
right = Graph()
right.add_nodes([{"x": 3}, {"x": 4}])
right.add_edges([(0, 1, {"weight": 2.125})])
left_table, right_table = left.read_view().table(), right.read_view().table()
merged = GraphTable.merge([left_table, right_table])
assert merged.nodes.base_table()["node_id"].to_list() == [0, 1, 2, 3]
assert merged.nodes.base_table()["x"].to_list() == [1, 2, 3, 4]
assert merged.edges.base_table()["source"].to_list() == [0, 2]
assert merged.edges.base_table()["target"].to_list() == [1, 3]
assert merged.edges.base_table()["weight"].to_list() == [1.125, 2.125]
```

## Six existing strategies

| Strategy | Identity and conflict behavior |
| --- | --- |
| `fail` | Preserve IDs and all rows; reject any node-ID or edge-ID collision between inputs. |
| `keep_first` | Equal IDs identify one entity; keep the first complete record. |
| `keep_second` | Equal IDs identify one entity; keep the last complete record, including endpoints. |
| `merge_attributes` | Equal IDs identify one entity; later physically present property columns replace earlier cells, including Null. Missing columns preserve earlier cells. Colliding edge IDs must have identical ordered endpoints. |
| `domain_prefix` | Keep inputs independent; assign dense IDs to every entity in domain/row order. |
| `auto_remap` | Keep inputs independent; preserve first/noncolliding IDs and remap later collisions to the lowest available nonnegative ID, reserving every original input ID first. |

All outputs have stable first-occurrence entity order. Replacing a record does
not move its output row. Structural columns come first, then property columns in
first-input occurrence order. Missing cells are Null. Keep policies choose whole
records; attribute merge does not merge vector elements or JSON internals.

Both remapping policies add `_domain` and `_original_id` columns to nodes and
edges. Input columns with those names reject instead of being overwritten.
Default domains are `domain_0`, `domain_1`, etc. Domain labels are metadata; IDs
remain nonnegative integers. Autoremap is distinct from dense remapping.

```python
first = GraphTable.merge([left_table, right_table], strategy="keep_first")
last = GraphTable.merge([left_table, right_table], strategy="keep_second")
assert first.nodes.base_table()["x"].to_list() == [1, 2]
assert last.nodes.base_table()["x"].to_list() == [3, 4]
assert GraphTable.merge([left_table, right_table], strategy="auto_remap").nodes.base_table()["node_id"].to_list() == [0, 1, 2, 3]
```

## Validation and typed references

Every input is validated, including a single input. Node/edge IDs must be unique
within that table and nonnegative integer cells. Null edge IDs require explicit
assignment before merge. Endpoints must exist in their own input domain: there
is no fallback to an identically numbered node from another domain.

`NodeArray` and `EdgeArray` properties use the same complete domain-local maps as
endpoints, including references to edges appearing later in an input. Plain
integers, vectors, text and JSON are opaque user data and are not rewritten.
f32/f64 values, vectors, temporal values, bytes and other supported inline
properties are copied without string conversion. Dangling typed references
reject, including in records a keep policy would discard.

Raw `SubgraphRef` values reject because relocating pooled contents is the separate
portable pool requirement (07.3 / 10.3 / 12.2), not an ID-only merge. No pooled
contents are silently dropped or reinterpreted. Merge does not infer graph
direction from table rows; independent materialization chooses direction explicitly.

Merged external IDs must be text/null and globally unique across surviving nodes
and edges. They are preserved, never automatically prefixed or regenerated.
Callers must resolve conflicting external identity explicitly. `merge_attributes`
rejects endpoint conflicts; callers can select a whole-edge policy or independent
remapping. Self-loops and parallel edges survive when permitted by input policies.

The result uses the strongest input validation strictness, the intersection of
loop/parallel-edge permissions, and the union of reference-validation requirements.
Auto-repair remains enabled only if every input enabled it; merge performs no
silent repair. Output validation failures return errors before any result is
exposed. Endpoint pair checks retain the existing table policy's ordered-pair
semantics; undirected materialization additionally checks its configured topology.

## Source identity and explicit graph materialization

Merged components retain their supplied source captures with read-only provenance.
They never infer a unique row owner, even if a policy keeps every ID unchanged.
Existing tables/views remain unchanged. Explicit mapped writeback or independent
materialization remains a separate caller action.

```python
nodes = merged.nodes.base_table()
assert len(nodes.source_views()) == 2
assert nodes.source_entity_ids is None
left.set_node_attr(0, "x", 99)
assert nodes.source_views()[0].node_attr(0, "x") == 1
fresh = merged.to_graph()
fresh.set_node_attr(0, "x", 42)
assert left.get_node_attr(0, "x") == 99
```

`from_federated_bundles(paths, domain_names=...)` delegates loaded tables to dense
domain merge. Supplied names must be nonempty, unique and match the bundle count;
they are actually emitted on node/edge rows. Existing bundle loader/format
limitations remain: this does not certify CSV bundles as lossless portable storage.

## Costs and scope

Merge uses domain ID maps, column buffers and row positions; it avoids repeated
UID scans and borrows input source metadata without cloning input tables solely
for provenance. It eagerly constructs output cells and validates the result.
Holding output source captures retains their frozen payloads. Memory and time
scale with input/output topology and property content; no total memory budget,
streaming, matched speedup or FFI latency guarantee is established.

This closes the six-policy specialized merge and federated-name delegation
family under 27/AR7.10. Portable pooled contents remain in their existing checklist
items; broader graph/table conversion adapters remain under 27.

[Collection ownership](table-collection-ownership.md) retains source captures and
known schema across supported empty collection transformations.

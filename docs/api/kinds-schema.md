# Labels, relationship types and schema

Semantic labels describe what a node represents. They are independent of the protected `entity_type` property, which continues to identify Groggy's base/meta storage structures. A base or meta node can carry multiple semantic labels, such as `Entity` and `Person`. No label is implicitly assigned to new nodes.

`labels` is a node text-vector property containing distinct, nonempty, case-sensitive names. Accepted values are stored in sorted order. Empty labels clear the set; a bare empty Python sequence is normalized to an empty text vector. Null, duplicate labels, whitespace-only names and other element types reject the operation. Nonblank names are preserved literally; no trimming, case folding or Unicode normalization occurs. The singular `label` remains an ordinary property.

`relationship_type` is an optional edge text property. A null assignment explicitly clears it. Nonempty names follow the same literal matching rules. Parallel edges and self-loops retain their own IDs even when their types match; setting a type does not change edge endpoints. Node `relationship_type` and edge `labels` assignments reject with a wrong-kind error.

```python
from groggy import Graph

g = Graph()
ada = g.add_node(labels=["Person", "Entity"], name="Ada")
source = g.add_node(labels=["Episodic"])
fact = g.add_edge(source, ada, relationship_type="MENTIONS")
assert g.node_labels(ada) == ["Entity", "Person"]
assert g.get_node_attr(ada, "entity_type") == "base"
assert g.nodes_with_label("Person") == [ada]
assert g.edges_with_relationship_type("MENTIONS") == [fact]
g.set_node_labels(ada, ["Entity"])
g.set_edge_relationship_type(fact, "SUPPORTS")
assert g.edge_relationship_type(fact) == "SUPPORTS"
```

The helper queries return sorted active entity IDs and currently scan the relevant column. They are not indexed lookups; indexes belong to roadmap item 18. Inactive entity getters raise; scans omit inactive IDs. Labels/types use normal property history, bulk setters, upserts and mixed batches. Recorded snapshots retain earlier typed values, and typed active graph exports preserve them. Ordinary subgraphs retain their current live-read behavior until item 26.

## Explicit schema validation

`g.validate_schema(schema)` returns all applicable findings without changing the graph, recording a version, or installing a write policy. Rules select nodes by semantic label and edges by relationship type. Every matching label applies; extra properties, unknown labels and untyped entities remain flexible.

```python
schema = {
    "node_labels": {
        "Entity": {
            "name": {"type": "text"},
            "age": {"type": "int", "required": False, "nullable": True},
        }
    },
    "relationship_types": {
        "SUPPORTS": {"confidence": {"type": "float"}}
    },
}
findings = g.validate_schema(schema)
assert len(findings) == 1
assert findings[0]["entity_type"] == "edge"
assert findings[0]["entity_id"] == fact
assert findings[0]["selector"] == "SUPPORTS"
assert findings[0]["property"] == "confidence"
g.set_edge_attr(fact, "confidence", 0.9)
assert g.validate_schema(schema) == []
```

A property rule has `type`, `required` (default true) and `nullable` (default false). Required controls absence; nullable controls a present Null. Optional but nonnullable properties may be absent, but cannot contain Null. The logical types are `any`, `int`, `float`, `bool`, `text`, `int_vec`, `float_vec`, `bool_vec`, `text_vec`, `bytes`, `json`, `node_array`, `edge_array`, and `subgraph_ref`. Storage optimizations share the underlying logical family. Integers and floats are distinct; no coercion occurs. `any` still obeys nullable. JSON validation checks its outer property type, not a nested document schema.

Findings contain `entity_type`, `entity_id`, `selector`, `property` and `message`, ordered by nodes then edges, ID, label/type and property. Unknown schema fields or types and empty selectors raise errors. Multiple labels can impose conflicting constraints; validation reports each violated rule. Rust exposes `GraphSchema`, `PropertyConstraint`, `PropertyKind`, `SchemaViolation` and the same checked graph helpers.

## Schema-controlled batch publication

Pass `schema=` to `apply_batch` to enforce the rules on that call's complete proposed resulting graph:

```python
from groggy import Graph, GraphBatch

graph = Graph()
batch = GraphBatch()
batch.add_node(labels=["Entity"], name="Ada")
policy = {"node_labels": {"Entity": {"name": {"type": "text"}}}}
result = graph.apply_batch(batch, graph.current_state_id, "person", "example", schema=policy)
assert graph.current_state_id == result["state_id"]
assert not graph.has_uncommitted_changes()
```

Rust provides `apply_batch_with_schema(batch, expected_state, message, author, &schema)`; the existing `apply_batch` remains the flexible path. Omit the Python `schema` argument or pass `None` to use that same flexible path. Passing a schema does not install a policy on future writes.

Validation combines staged properties with unchanged live values, applies label/type changes together with their required properties, and excludes removed nodes/edges plus edges removed through node deletion. It also checks untouched surviving entities, so a previously invalid graph must be repaired or have those entities/selectors removed in the batch before this policy can succeed. Unknown labels and untyped entities remain allowed under the schema model.

Every schema finding is reported before allocation or property/history writes. Rejection preserves current state, history, dirty tracking and allocation counters; success publishes one state using the existing batch coordinator. Proposed entity IDs shown in errors are prospective IDs and have not been reserved. Existing stale-state and uncommitted-legacy-write guards still apply. Schema checking uses an overlay of staged assignments and reads existing columns, rather than cloning/reconstructing the graph. It scans the proposed graph; it is not an indexed incremental check. Existing constraints on required, nullable and logical property types are identical to explicit auditing.

## Remaining enforcement work

Schema validation via `validate_schema` is an explicit audit; schema-controlled batches enforce policy only for the call. Installing a graph-wide enforced schema still requires every direct writer to join the checked publication boundary in item 9. Existing infallible Rust creation methods cannot report required-field errors; they need a compatible checked path or an explicit policy restriction. This slice does not silently impose required fields on those methods.

Schema policy versioning/persistence, ontology migrations, relationship endpoint constraints, uniqueness constraints beyond managed external IDs, and indexed label/type selection remain open. Existing files using incompatible values in `labels` or `relationship_type` now fail checked loading explicitly; callers must migrate those properties. Graphiti-specific label mapping remains the adapter's responsibility.

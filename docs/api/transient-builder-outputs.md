# Nonpersistent builder attribute outputs

`persist=False` now applies to native builder node/edge attachment steps and
stroke smoothing. They validate prospective columns, then return owned results
instead of changing graph properties or creating mutation versions/deltas.
Retrieve them with `return_profile=True` in `profile["outputs"]`.

```python
from groggy import Graph
from groggy.builder import AlgorithmBuilder

graph = Graph()
nodes = graph.add_nodes(2)
before = graph.current_state_id
builder = AlgorithmBuilder("transient scores")
builder.attach_as("score", builder.init_nodes(default=1.0000000001))
_, profile = graph.view().apply(
    builder.build(), persist=False, return_profile=True,
)
assert profile["outputs"]["node_attributes.score"] == [
    (node, 1.0000000001) for node in nodes
]
assert graph.current_state_id == before
assert not graph.has_uncommitted_changes()
assert all(graph.get_node_attr(node, "score") is None for node in nodes)
```

## Result contract

- Node attachment outputs use `node_attributes.<property>` keys; edge attachments
  use `edge_attributes.<property>`. Each value is a list of `(entity_id, value)`
  tuples sorted by ID. Node and edge columns with the same property name remain
  distinct outputs.
- Output values use the existing typed Rust/Python bridge: integers, f64 values,
  vectors, text, booleans, structured values and explicit Null are retained as
  produced by the builder's existing parameter/value conversion rules. Results
  are owned; editing the returned Python list does not edit the graph.
- A repeated attachment to one output key replaces that emitted column. The final
  column for that key is returned; outputs do not represent an accumulated private
  graph or a replay of all intermediate emissions. Empty attachment columns have
  empty typed results.
- The owner still validates IDs, duplicate targets, kinds and system-managed
  property restrictions. A malformed multi-column attachment emits none of its
  columns. Cancellation rejects before that attachment publishes/emits results.
- The graph remains unchanged, including pending manual changes and its existing
  publication policy. Later graph-reading steps therefore read the original graph;
  use pipeline variables to pass computed intermediate values between steps.
- Smoothing exposes its two target columns through the same node output keys.
  It does not overwrite coordinate inputs when persistence is disabled.

## Persistence and remaining migration

Explicit `persist=True` retains existing property publication behavior, and public
apply wrappers still default to True for compatibility. Adopting the accepted
transient-default direction across all execution APIs remains part of the pipeline
migration; this slice repairs the previously ignored explicit False flag.

Persistent multi-step pipelines are not yet one atomic mutation: an attachment can
publish before a later step fails. Full staging/read-after-write and compound
failure semantics remain 09.4b.2 / AR7.8 / 30. Per-step checks do not certify the
entire run. This change adds no pipeline-run/provenance history entities.

Native `AlgorithmOutput` adds `NodeAttributes` and `EdgeAttributes` owned column
variants. The shared `StepScope` attachment methods select persistence versus
outputs and perform canonical native validation. FFI only translates values; no
storage format or Python-side graph algorithm was introduced.

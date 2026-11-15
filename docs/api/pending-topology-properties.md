# Properties after staged topology

Supported property primitives can run after node/edge additions, removals or
endpoint revisions in a local atomic builder or a shared outer atomic pipeline.
They read the selected pending state and attach into the same canonical batch.
The owner publishes one delta after the complete run succeeds.

```python
from groggy import Graph, Pipeline
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_node()
head = graph.current_state_id
create = step_pipeline("create", [
    stage("graph.add_nodes", count=1, output="new", attributes={"score": 3.5}),
    stage("graph.add_edges", edges=[[0, {"added": 0}]], output="edges"),
], atomic=True)
compute = step_pipeline("compute", [
    stage("core.load_node_attr", target="scores", attr="score", default=2.0),
    stage("core.init_scalar", target="two", value=2.0),
    stage("core.mul", left="scores", right="two", target="scaled"),
    stage("core.attach_node_attr", source="scaled", attr="scaled"),
    stage("core.load_node_attr", target="again", attr="scaled"),
    stage("core.attach_node_attr", source="again", attr="copied"),
], atomic=True)
result = Pipeline([create, compute])(graph.view(), atomic=True)
assert graph.current_state_id == head + 1
assert sorted(result.node_ids) == [0, 1]
assert graph.get_node_attr(0, "scaled") == 4.0
assert graph.get_node_attr(1, "scaled") == 7.0
assert graph.get_node_attr(1, "copied") == 7.0
```

## Certified stages

- Node/edge property loads and attachments; edge weight scaling.
- Node/edge initialization, ordered node indices and selected entity counts.
- Explicit variable arithmetic: binary operations, unary math, reciprocal,
  comparisons, conditional selection, scalar reduction/broadcast and list mode.

Initialization and property loads include provisional new IDs and exclude removed
nodes and cascaded edges. An exact edge selection stays exact. Attachments validate
all targets, duplicate rows, property kinds and reserved keys before staging.
A map collected before a removal is **not** silently filtered at attachment:
its removed targets reject the operation. Recreate the map over the pending
selection to target survivors. Reloads see earlier staged attachments.

The original execution selection remains available to later topology stages,
including endpoint revisions that rescue an edge from a pending node cascade.
The final canonical batch still validates the entire plan and its timelines.
Conflicting staged writes/removals reject rather than silently dropping writes.

## Clocks, ownership and cost

These primitives keep their existing attachment parameters and timestamp policy.
They do not add an explicit timestamp parameter. Canonical batch acceptance
assigns default times; creation timestamps and existing explicit topology times
retain their usual rules. This computes pending working state, not a separate
historical effective-time query.

Pending reads use the existing detached immutable capture. A builder reuses that
scope while the batch revision is unchanged and refreshes it after property or
topology effects. Materialization copies selected state, including selected
attributes; it is not an O(1) live view. Pure variable arithmetic does not capture.
Runs without staged topology retain the existing property overlay path. Cancellation
is checked at stage boundaries; the capture copy itself is not cooperatively
interrupted.

Step variables remain local to each builder. Explicit emitted graph/table values
remain immutable with provisional provenance. A failure or cancellation discards
staged topology/properties and restores the outer output scope without consuming
IDs or publishing history.

## Remaining boundary

Classic property-writing algorithm wrappers and unadapted adjacency-dependent
primitives (such as `core.node_degree` and neighbor collection) still reject after
topology. Use explicit `graph.capture_staged` with the supported captured graph
kernels for those reads. General topology loops, table-result attachment,
remaining execution/DSL lowering and journal-owner dispatch remain open.

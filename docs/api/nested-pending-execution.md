# Nested computation over pending graph state

Certified `iter.loop` bodies can compute and attach properties after staged
topology. Each child step uses the same pending-read refresh boundary as a
top-level builder step. Later loads see earlier staged assignments, including
assignments from a previous iteration or a nested loop.

```python
from groggy import Graph
from groggy.builder import step_pipeline


def stage(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
graph.add_node()
head = graph.current_state_id
body = [
    stage("core.load_node_attr", attr="counter", target="counter", default=0),
    stage("core.init_scalar", target="one", value=1),
    stage("core.add", left="counter", right="one", target="next"),
    stage("core.attach_node_attr", source="next", attr="counter"),
    stage("core.load_node_attr", attr="counter", target="reload"),
    stage("core.attach_node_attr", source="reload", attr="mirror"),
]
graph.apply(step_pipeline("nested counter", [
    stage("graph.add_nodes", count=1, output="new"),
    stage("iter.loop", iterations=2, body=[stage("iter.loop", iterations=3, body=body)]),
], atomic=True))
assert graph.current_state_id == head + 1
for node in [0, 1]:
    assert graph.get_node_attr(node, "counter") == 6
    assert graph.get_node_attr(node, "mirror") == 6
```

## Supported nested effects

Bodies may combine certified pending property/selected-reader steps, explicit
immutable-input operations, pending captures and explicit captured-table
attachments. Capture/attachment bodies require a local `atomic=True` builder and
`persist=True`, including at zero iterations. They share the outer pending batch
and output scope. They do not publish a delta per iteration.

Body capabilities are checked recursively before a prefix executes. General
topology mutations inside loops, streaming blocks, alternate batch plans and
opaque callbacks remain uncertified. Legacy owner-based property readers remain
restricted to execution before pending topology. A mixed body cannot silently
upgrade an unadapted descendant's read contract.

## Typed bindings and immutable inputs

`loop_vars=[["initial", "logical"]]` preserves the complete existing step value,
including immutable graph/table/snapshot/index values. Explicit variable names
retain their prior sequential aliasing behavior. Captured inputs are not
silently refreshed: a table alias keeps the source snapshot's values and row
identity. Only ordinary property loads against the execution scope refresh after
staged writes. Attachment still requires authority from the current atomic run.

## Message passing

Supported `core.execution_block(mode="message_pass")` bodies also use the selected
pending endpoints. Neighbor aggregation and message passing share the native
checked directional CSR builder. Provisional IDs, endpoint revisions, exact edge
selection, parallel edges and stored self-loops retain the existing conventions.

`ordered=True` keeps deterministic ordered Gauss–Seidel visibility;
`ordered=False` keeps synchronous Jacobi visibility. Both compute privately and
install their variable result after success. Weights and unsupported block
operations retain their existing explicit rejection rules. Streaming remains
unsupported.

## Ownership, cancellation and cost

A nested frame can borrow a parent's capture only when its capture token and
pending revision still match. After staging changes, the next selected read
refreshes. The capture authority registry is weak and retains no graph payload. Expired
authority entries are reclaimed with amortized sweeps; live earlier captures
remain authorized, and oversized dead registry capacity can be released.
This avoids copying a current source merely to enter another frame; it does not
make materialization O(1) or introduce a global result cache.

The original execution selection remains available for final acceptance.
Cancellation, body failures and canonical publication failures discard the shared
pending graph mutation and restore outputs. Cancellation is checked before and
after each executed child. Snapshot-copy cancellation remains bounded by the
existing materialization implementation. See [pending properties](pending-topology-properties.md)
and [table attachment](captured-table-attachment.md) for clocks and provenance.

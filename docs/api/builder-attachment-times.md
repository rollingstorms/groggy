# Effective timestamps for direct builder attachments

`core.attach_node_attr` and `core.attach_edge_attr` accept an optional
`effective_at` signed integer in Unix microseconds. Node maps, ordered node columns
and edge maps use the existing canonical property staging boundary. Explicitly
timed attachments require `atomic=True` and `persist=True`. Omitting the parameter
retains existing execution and default batch-clock behavior.

```python
from groggy import Graph, GraphBatch
from groggy.builder import step_pipeline


def timed_attachment_step(kind, **params):
    return {"id": kind, "params": params}


graph = Graph()
initial = GraphBatch(effective_at=0)
a = initial.add_node()
b = initial.add_node()
initial.add_edge(a, b)
graph.writer().apply_batch(initial)
head = graph.current_state_id
operation = step_pipeline("timed computed properties", [
    timed_attachment_step("core.init_nodes", target="scores", value=9),
    timed_attachment_step("core.attach_node_attr", source="scores", attr="score", effective_at=40),
    timed_attachment_step("core.init_edges", target="weights", value=2),
    timed_attachment_step("core.attach_edge_attr", source="weights", attr="weight", effective_at=80),
], atomic=True)
graph.apply(operation)
assert graph.current_state_id == head + 1
assert graph.read_view_at(39).node_attr(0, "score") is None
assert graph.read_view_at(40).node_attr(0, "score") == 9
assert graph.read_view_at(79).edge_attr(0, "weight") is None
assert graph.read_view_at(80).edge_attr(0, "weight") == 2
```

Each attachment sets one effective time for its supplied rows. Different attachment
stages may supply different times within the same publication. The canonical
coordinator retains its existing temporal rules: topology clocks must preserve
endpoint lifetimes; property changes outside an entity's effective active lifetime
remain retained in accepted history but do not carry into a later activation in
effective-time reconstruction. Accepted-state and effective-time reads remain
distinct. Provisional rows use the same rule. A later assignment to the same
key replaces its pending value **and** clock; omitting its time restores the batch
clock. This is one final assignment per key, not a history record for every
intermediate computation. Separate accepted writes at an identical effective time
use the established later-accepted-write tie rule.

All targets are validated before staging, including fixed child-frame confinement.
Invalid clocks, timeline conflicts, unsupported logical values, late errors and
cancellation retain whole-run rollback. Existing immutable views keep their prior
values; archive/history reconstruction retains accepted explicit clocks.

The only supported attachment parameters are `source`, `attr`, and `effective_at`.
Unknown keys and noninteger clock values reject during construction rather than
being ignored. Omit an unused optional clock; an explicit null is not a timestamp.
Ordinary untimed/transient attachments retain their existing behavior. Direct
non-atomic timed computation remains explicit through ordinary graph writer/batch
APIs, or by making the builder atomic.

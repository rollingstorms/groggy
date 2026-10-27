# Writer owner, branch and state expectations

A Python writer captures its owning graph, selected branch and published head.
Every batch, node/edge upsert, checkpoint, merge preparation and explicit restore validates that
coordinate inside the native owner operation. Switching branches invalidates the
session even if both branches currently point to the same state ID.

```python
from groggy import Graph

g = Graph()
g.add_node(value=0)
g.create_branch("analysis")
writer = g.writer()
g.checkout_branch("analysis")
assert writer.branch == "main"
try:
    writer.set_node_attr(0, "value", 1)
except ValueError as error:
    assert "coordinate" in str(error)
else:
    raise AssertionError("A branch switch requires explicit writer refresh")
assert g.get_node_attr(0, "value") == 0
writer.refresh()
assert writer.branch == "analysis"
writer.set_node_attr(0, "value", 1)
assert writer.state_id == g.current_state_id
g.checkout_branch("main")
assert g.get_node_attr(0, "value") == 0
```

`writer.refresh()` explicitly captures the currently selected owner/branch/head.
Successful writes and tracked/reference restores advance the captured state while
retaining the same branch. Failure leaves the session's expectation unchanged.
Merge callbacks cannot change the coordinate of an already-running writer call
by refreshing the same Python writer; the call uses its local captured coordinate.
Native Rust writers hold an exclusive mutable borrow throughout their lifetime,
which already prevents external branch switching during the session.

## Direct coordinates

Graph and DurableGraph expose immutable `write_coordinate()` markers with
`state_id` and `branch`. Advanced callers can pass `expected_coordinate` to
`apply_batch`, `publish_upsert_nodes`, `publish_upsert_edges`, `checkpoint`,
`prepare_merge` and `restore_state`. The existing
expected-state check still applies; foreign owners reject even when their branch
names and state IDs match. All checks occur before state publication or durable
journal append. Normal direct calls without a coordinate continue to target the
currently selected branch with their supplied head expectation.

These markers contain an owner identity, branch name and expected state. They
contain no graph payload, history record or mutable graph reference. They are
safe to transport across threads, pin no history, grant no mutation authority and
are not a lock or lease. Validation must happen inside the exclusive operation;
a previous successful validation does not authorize a later unchecked mutation.
They are not portable serialization or a service authorization mechanism.

Graph remains thread-confined. DurableGraph retains its exclusive file writer
lock and checked append/sync/replay contract. This correction does not introduce
a multi-client service, writer queue, HTTP interface or automatic branch switch.
Those remain separate 09.5/33.3 scope. There is no journal/archive format change.

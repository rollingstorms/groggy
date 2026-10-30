# Property access respects selected membership

Ownership update: [default selections are immutable](default-immutable-views.md).
The earlier live-scope details below apply to explicit native execution scopes
and `graph.connected_view()`; default bound writes require a matching source.

Shared connected-subgraph property getters now enforce the selection in addition
to the owner's active-state rules. An active node outside a node selection is
not readable through it. A parallel edge
whose endpoints are selected still needs its own selected edge identity.

```python
from groggy import Graph

graph = Graph()
a = graph.add_node(chosen=True, score=1)
b = graph.add_node(chosen=False, score=2)
selected = graph.filter_nodes("chosen == true")
assert selected.get_node_attribute(a, "score") == 1
try:
    selected.get_node_attribute(b, "score")
except RuntimeError as error:
    assert "outside" in str(error)
else:
    raise AssertionError("A selection must not read an excluded entity")
```

Python connected bulk accessors retain their existing selection-bound writes:
a valid prefix is never written when an excluded input fails.

```python
head = graph.current_state_id
try:
    selected.set_node_attrs({"score": {a: 7, b: 8}})
except (ValueError, RuntimeError, KeyError, PermissionError):
    pass
else:
    raise AssertionError("An excluded property target must reject the batch")
assert graph.get_node_attr(a, "score") == 1
assert graph.current_state_id == head
```

Algorithm Context and StepScope property reads also check the current selection
and owner liveness before returning an existing staged value. A value staged by
an earlier wider algorithm does not grant read access from a narrower selection.
Outer-pipeline failure still discards staging under the existing atomic contract.

Explicit native bulk writes and algorithm property-map attachments keep their
existing owner-wide active-ID contract. Retained temporal diffs intentionally
cover full owner states, even for an empty invocation selection; explicitly
attaching those maps can target active owner IDs outside that selection. Those
maps can still be returned as owned transient outputs. This repair does not
change that writeback contract or silently discard map entries. Checked pinned
`apply_view_batch` retains its separate exact selected-write scope. Unifying
legacy read/write ownership is still part of 26.1/AR3, not an incidental change
to the historical diff API.

This is a property-boundary repair within 26.1/09.4b.2. It does not turn connected
Subgraph into an immutable view or certify arbitrary custom topology effects.
Explicit `selected.query()`/`pin_view()` captures remain the immutable read path;
GraphView already enforces its own frozen selected membership. Existing connected
Python getter adapters retain their established RuntimeError translation.

The added valid-input work is a membership hash lookup per read, plus an active
membership check before returning a staged value. It creates no snapshot or
mutation work. Explicit owner access remains separate from selected reads. Wider
ownership/topology migration remains on the original roadmap.

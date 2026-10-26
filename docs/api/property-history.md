# Property history: assignment, null and absence

Assignments retain whole typed values; deltas describe property changes, not
byte-level differences within an embedding, text or other encoded value. An
explicit unset removes the property's current mapping and records absence. It
does not delete the entity, erase backing pool values or rewrite immutable views.

```python
from groggy import Graph, GraphBatch

g = Graph()
b = GraphBatch(effective_at=10)
n = b.add_node(value=None)
g.apply_batch(b, 0, "initial", "example")
old = g.read_view_at(10)
g.writer().unset_node_attr(0, "value", effective_at=20)
assert old.has_node_attribute(0, "value")  # Explicit null.
assert old.node_attr(0, "value") is None
assert not g.read_view_at(20).has_node_attribute(0, "value")
g.writer().set_node_attr(0, "value", 42, effective_at=30)
assert not g.read_view_at(29).has_node_attribute(0, "value")
assert g.read_view_at(30).node_attr(0, "value") == 42
```

## One checked operation

`GraphBatch.unset_node_attr` / `unset_edge_attr` accept existing IDs and handles
created by that batch, plus an optional per-property `effective_at`. Batches may
mix sets, unsets and topology operations. A key cannot be both set and unset for
the same entity in one batch; duplicate targets reject the complete operation.
Targets must be active in the resulting state. Managed entity identity and
external IDs cannot be unset; a meta-node must retain `contains_subgraph`.
Schema enforcement checks the proposed absence before allocation or journal I/O.

`Graph.unset_node_attr` / `unset_edge_attr` and bulk `unset_node_attrs` /
`unset_edge_attrs` honor the graph's publication policy. Bulk inputs map property
names to entity ID lists. An empty direct Graph bulk call is a no-op, matching
ordinary bulk setters; an explicit empty batch remains invalid.

`GraphWriter` offers the same scalar and bulk methods with effective timestamps.
Each writer call is one publication. `DurableGraph.writer()` uses these same
batches and persists absence plus the original acceptance/effective clocks.

```python
from groggy import GraphBatch

batch = GraphBatch(effective_at=40)
batch.unset_node_attr(0, "value", effective_at=45)
batch.set_node_attr(0, "other", None)
g.apply_batch(batch, g.current_state_id, "mixed", "example")
assert g.read_view_at(44).has_node_attribute(0, "value")
assert not g.read_view_at(45).has_node_attribute(0, "value")
assert g.read_view_at(45).has_node_attribute(0, "other")
```

An unset of an already absent key still records an absence event: backdating
that event can affect reconstructed history. Same-value assignments likewise
remain events. Separate publications are never collapsed together. Explicit
recorded-time cutoffs and equal-effective-time accepted-order ties follow
[effective-time reads](effective-time-reads.md).

## Coalescing and native history API

Within one manual pending change set, repeated scalar/bulk sets and unsets of a
key retain the first old value and final new value in operation order. No
intermediate manual states are published. This applies equally to nodes and edges.

Native `ChangeSet` and recorded `Delta` property tuples now have optional old
**and** new values. `None` means absent; `Some(AttrValue::Null)` means present with
null. Low-level strategy/index-delta new indices are optional for the same reason.
Native graph/index property histories return `(state_id, Option<AttrValue>)`.
Existing assignment-only recorded delta content hashes remain compatible.

Python history methods retain typed `AttrValue` wrappers: a null is an object
whose `.value is None`, while an unset entry is bare `None`.

```python
history = g.node_attr_history(0, "value", 1, g.current_state_id)
assert history[0][1].value is None
assert history[1][1] is None
assert history[2][1].value == 42
assert history[3][1] is None
```

Temporal window `count` counts all changes, including unsets. `first`/`last`
preserve the selected event (absence produces the existing empty result value).
Numeric reductions reject null/absence and nonnumeric values instead of silently
using an older value or discarding a change. History entries themselves preserve
the null/absence distinction.

Frozen retained groups keep their source values after later unsets. Unsetting a
user property containing a group reference does not purge that source. Durable
reopen and full backup reproduce typed history; pooled storage import/compaction
remain separate roadmap requirements. New journal unset fields are optional and
omitted when unused, so existing assignment-only records remain readable.

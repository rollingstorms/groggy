# Logical property change records

`groggy::state::change::{AttributeChange, EntityType}` is the authoritative Rust
representation of a logical property difference between selected states.
`state::state`, `state::history` and the existing `state` root exports preserve
the same names as compatibility re-exports of those exact types.

An `AttributeChange` retains its public fields: node/edge `entity_type`, u64
`entity_id`, `attr_name`, `old_value` and `new_value`. `None` means absence;
`Some(AttrValue::Null)` is an explicit null. Values retain their native tags and
storage semantics. Snapshot `StateDiff.attribute_changes` and history
`CommitDiff.attr_changes` use the same record; commit diff construction moves the
records directly instead of converting an identical second structure.

These records are logical comparisons, not a new serialized publication log.
Publication clocks/effective-time keys, retained delta lineage and storage-column
offsets keep their distinct responsibilities. This move changes no persisted
format, ID width, mutation behavior, clock policy or retention decision. Full
storage lifecycle and service guarantees remain separate work.

`errors::EntityType` is a diagnostic scope that also includes states, branches
and attributes. `algorithms::steps::sampling::EntityType` selects node/edge
collections using its established plural wire names. They intentionally remain
separate types; they are not duplicate property-change targets.

## Canonical mutation records and exact comparison

`state::change::ChangeSet` now owns logical mutation payloads. The older
`state::change_tracker::ChangeSet`, `state::ChangeSet` and other existing exports
are aliases of that same type. Fields and history/archive/journal encodings are
unchanged. `Delta::from_changes` sorts and moves owned buffers without cloning
property vectors or strings. Legacy hash bytes remain unchanged; exact collision
checks still protect signed zero and NaN payload variants.

One native snapshot comparison builds topology and typed property changes for
GraphSnapshot differences, recorded history differences, merges and tracked
restoration. Exact payload comparison is shared with archive validation and delta
interning. Different float widths, signed-zero bits and NaN payloads differ.
Supported compact/small/compressed storage representations normalize according to
the portable codec; logical AttrValue ordering/equality remains unchanged.
Inactive rows cannot supply before-values when an identity is activated. Changes
are sorted by entity kind, ID and property name. Removed entity properties remain
in their retained parent history, not redundant per-key unset records.

`GraphSnapshot.try_diff_with(other)` returns GraphResult and reports compared
invalid/unencodable payloads. Existing infallible `diff_with` remains compatible;
it conservatively reports unencodable compared values as changed. Recorded
history readers use the checked API. This is comparison, not a full integrity
scan; use Graph.check_integrity for retained graph validation.

```python
import math
from groggy import Graph

g = Graph()
g.add_node(value=0.0)
plus = g.current_state_id
g.set_node_attr(0, "value", -0.0)
minus = g.current_state_id
change = g.diff_commits(plus, minus)["attribute_changes"][0]
assert math.copysign(1, change["old"]) == 1
assert math.copysign(1, change["new"]) == -1
restored = g.restore_state(plus, mode="tracked")
assert len(g.diff_commits(minus, restored)["attribute_changes"]) == 1
assert g.diff_commits(plus, restored)["attribute_changes"] == []
```

## Physical references have a separate explicit representation

`state::change::IndexChangeView` borrows actual pending node/edge IDs, endpoint
changes and before/after column offsets. It owns no logical values or mutation
authority. ChangeTracker diagnostics inspect this view without allocating fake
properties or copying every pending change. Endpoint-only changes participate in
modified-edge inspection. Memory estimates describe occupied pending record bytes;
spare capacity, hash buckets and pool-owned value payloads are excluded.

`ChangeTracker.create_delta()` returns GraphResult<DeltaObject>. Its indexed
projection retains complete physical changes in `DeltaObject.index_changes`,
including topology and both attribute offsets. `new_with_indices` is fallible and
rejects unequal column lengths, repeated or unordered entity indices. Existing
value-column DeltaObject construction still has its original role; indexed
records are never silently converted into empty value deltas. Exact equality
checks payloads after a hash match.

`create_change_set()` / `create_changeset()` are now fallible. They return actual
topology-only logical records when there are no property offsets. If properties
are pending, they reject and direct the caller to
`create_changeset_resolved(pool)`. Resolution requires the corresponding retained
pool, preserves first-before/last-after, returns actual typed values, distinguishes
Null from absence, and rejects unavailable indices. The old fabricated
`Text("index_N")` payloads are removed. Physical offsets are not portable history.

Native custom TemporalStorageStrategy implementations now supply
`indexed_changes()`. Shared default projection/resolution methods replace the
parallel conversion implementations. Callers of the formerly infallible delta or
no-pool extraction methods must handle GraphResult. Python API shapes are unchanged.

GraphSpace owns selected state and membership, not a pending write queue. Its old
`create_change_delta` helper now returns an explicit error instead of claiming an
empty history delta. Use the Graph coordinator or an explicit tracker. Selected
GraphSpace, retained HistoryForest, effective-time keys/indexes and physical column
references keep separate responsibilities; they are not interchangeable records.

No persisted format or automatic retention/index rebuild policy changes in this
consolidation. Broader coordinator, legacy view, query/index, service and throughput
work remains in its existing checklist rows.

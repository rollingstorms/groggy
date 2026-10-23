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
AR7.9 migration and effective-time reconstruction remain separate work.

`errors::EntityType` is a diagnostic scope that also includes states, branches
and attributes. `algorithms::steps::sampling::EntityType` selects node/edge
collections using its established plural wire names. They intentionally remain
separate types; they are not duplicate property-change targets.

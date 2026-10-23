# AR7.9a — Shared logical property-change owner

Move the identical snapshot/history AttributeChange and Node/Edge EntityType
records under state::change. Preserve every old Rust import path via exact
re-exports, public fields and u64 identity representation. Remove the history
conversion map; CommitDiff consumes the StateDiff record vector directly. Migrate
FFI enum matching to the authoritative path without changing Python output.

Preserve absent versus null, typed values, forward/reverse diff behavior and
existing selected-state reconstruction. No serialization, delta clock, retention,
publication, reset/revert or pool-storage policy is changed. This is a completed
bounded ownership slice, not completion of AR7.9 as a whole.

Diagnostic EntityType and sampling EntityType are intentionally different domains
with additional variants/wire names. Keep them separate; unifying by name alone
would make the model less precise. See docs/api/change-records.md.

Verification covers cross-path Rust record assignment through snapshot diffs,
null/absence in both directions, full native suites, strict all-target lint,
in-tree FFI build and Python commit-diff/history/property/recovery surfaces.
No numeric/kernel change or performance claim; no new benchmark is needed for
this type-ownership and redundant-conversion removal.

Executed: locked offline full cargo test; strict all-target cargo clippy; in-tree
FFI rebuild; 97 focused Python commit-diff, property/history, publication, identity,
retained-view and temporal regressions. Three pre-existing FFI warnings remain.

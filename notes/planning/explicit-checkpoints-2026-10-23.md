# 12.3a / 09.4a — Explicit checkpoint boundary and checked constructors

The owner accepted automatic checked publication and explicit commit/checkpoint
semantics. This step supplies the checkpoint boundary before ordinary mutation
routing changes. It does not mark the broad default migration complete.

## Implemented

- Checkpoint records label a clean published state with branch/message/author and
  engine acceptance microseconds. No new state ID, empty delta, changed mutation
  clock or permanent retention pin is created. Repeated and empty-state labels
  work; public Python records are owned copies.
- `Graph.checkpoint`, optimistic `checkpoint_at`, writer checkpoint and schema
  policy checks share native validation. Stale/dirty states reject before record
  publication. Native GraphWriter and the Python convenience writer retain their
  selected state ID after checkpointing.
- Clean `commit()` now checkpoints the selected published state. Dirty legacy
  `commit()` remains a compatibility adapter: publish pending changes once, then
  record the resulting state. Ordinary direct mutations are not yet migrated.
- Durable annotations sync before memory visibility; journal replay checks the
  referenced selected state/branch and preserves the original clock. Backups
  retain these records. Old records remain supported; older readers reject the
  new record kind instead of silently ignoring it.
- Rust `try_add_node`/`try_add_nodes` expose checked fallible construction without
  forcing a panic/error discard into existing infallible compatibility methods.
  Bulk insertion publishes once, rejected dirty writes consume no IDs and a zero
  count remains a no-op. Further ordinary Rust/Python default migration remains
  09.4; no implementation approval is requested again.

## Verification

Targeted native/Python tests cover repeated/empty checkpoints, no duplicate delta
or state ID, original publication clocks, retained views, stale sessions, schema
failure, legacy compatibility, branches, journal recovery and backup. Native
journal fault tests cover pre/post-header/payload/sync rollback, crash after sync
before memory visibility, and semantically invalid checkpoint records.

Verification: 4 native checkpoint integration tests and 5 Python checkpoint tests
passed. Full native suites: 907 passed, 52 ignored. Root Python API baseline
(excluding the owner's independent algorithm benchmark): 996 passed, one existing
skip; this does not certify the old `tests/modules/` survey scripts. All 24
selected documentation examples passed. `cargo clippy --offline --locked
--all-targets -- -D warnings` passed. The in-tree extension and changed Python writer adapter were refreshed;
three existing unrelated FFI warnings remain. Black is unavailable in the existing
environment; changed Rust regions and new modules/tests were formatted and
whitespace checked. No checkpoint schedule or
automatic pruning policy is introduced. This is an explicit label operation,
not the future materialized-recovery checkpoint in 10.4.

# Explicit retained-state restoration — 2026-10-27 UTC

Existing **13.3 component**, following completed coordinated endpoint revisions
(04.2). Branch: `groggy-roadmap-2026-10-20`. This implements the accepted decision
record's explicit working-state switch, with destructive pruning separately chosen.
No new checklist rows or inferred policy for tracked inverse mutations.

## Delivered boundary

Native Graph, DurableGraph and both writer sessions expose `restore_state` with
an explicit retained published target and expected current head. Working active
membership, ordered endpoint pairs, whole typed properties, external bindings and
frozen group references are installed; the current branch head moves to that
state. Other branches and all retained versions survive. Allocation high-water
marks, reserved names, publication metadata, graph configuration, manual/automatic
policy and independent immutable payloads survive. No mutation version, delta,
checkpoint or new effective/acceptance clock is created. Clean same-state selection
is a validated no-op. State zero is not a published version and rejects.

Dirty pending manual state rejects: the caller must explicitly reset_hard before
switching. Native detached `StateRestorePlan`/`StateRestoreOutcome` carry logical
values and an owner/head/branch coordinate; execution is Send + Sync and needs no
mutable graph. Stale heads/branches, foreign reopened owners and targets collected
between capture/install reject before mutation. Intervening pool compaction is
safe while the logical target remains retained. Full topology/allocated-ID/source/
identity preflight precedes installation. Trusted recorded installation shares the
existing snapshot payload installer; raw untracked endpoint overwrites stay guarded.

FFI only translates arguments, errors and ownership boundaries. Reconstruction
runs without the Python GIL; capture, schema/reference validation and install are
owner-bound. Python state IDs reject bool. Writers update their expected head only
after success. Ordinary Graph accepts an optional full-target schema. Durable
restoration enforces the currently installed policy plus any stricter per-call
schema, including a policy changed after detached reconstruction was prepared.

Durable RestoreState records include original head, selected branch, explicit
target and additional schema. Append/sync completes before changing memory. Replay
validates the same coordinates and policies. Four injected append-failure phases
preserve bytes, state, versions and immutable values; retry/reopen succeeds. A
synced record before memory installation recovers the target. Valid-checksum frames
with stale head, wrong branch, absent target or failing extra schema reject on
reopen without repairing bytes. Same-state restoration appends no frame. Backup,
explicit GC, checkpoint replacement and physical verification are covered. Older
readers reject the new record; archive schema remains unchanged by this operation.

## Verification

- Native **1,342 passed / 52 ignored**, including five new integration cases and
  two journal failure/replay cases (four append phases and five replay variants).
- Root Python **2,025 passed / 1 skipped**; existing owner benchmark and nested
  legacy survey exclusions remain. **176 executable docs examples**, four new.
- Strict all-target Clippy; rebuilt/staged in-tree FFI with its three pre-existing
  unrelated warnings; serial process-kill parent **1 passed / 18 cases**.
- Scoped Rust formatting, Python/stub syntax and diff checks. No install/push.

Commands: locked offline `cargo test --quiet`, strict `cargo clippy --all-targets
-- -D warnings`, the separate ignored `killed_checkpoint_process` serial gate,
in-tree FFI `cargo rustc`, root Python/doc runner, and `cargo bench --bench
state_restore`. Logs are under `/tmp/groggy-state-restore-*` for this local run.

## Cost receipt

`../benchmarks/state-restoration-2026-10-27.json` contains three native release
samples each. Fixture: a chain with one integer property per node, then a complete
column update; restore the first version while retaining the second and its old
immutable pin. Setup, input construction and old-pin capture are excluded.

| Nodes / edges | Plan capture median | Detached reconstruction | Validation/install |
| --- | --- | --- | --- |
| 1k / 999 | 0.50 µs | 0.269 ms | 0.838 ms |
| 100k / 99,999 | 2.54 µs | 30.27 ms | 126.42 ms |

No FFI/RSS/production latency claim. Full state replay/assembly remains proportional
to selected data; only the small immutable capture is cheap. This is a capability
receipt, not an overall-engine before/after comparison. Existing broader 19.2/32
performance audits remain open.

## Fixed remaining work

**205 checked / 63 open**, same 268 rows. Existing 13.3 now has endpoint reversal,
pending rollback and retained-state selection. It stays partial because a general
tracked inverse mutation and its recording/effective-time policy are separate.
Restoration never chooses destructive pruning implicitly; the explicit GC/purge
APIs already supply that separate mechanism. Next core item is caller-resolved
three-way merge planning/acceptance (14.2). No production-service or Graphiti
scope is pulled forward. See `docs/api/state-restoration.md` for executable usage.

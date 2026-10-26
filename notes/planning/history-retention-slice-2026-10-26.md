# Explicit history collection — existing 15.1 / 15.2

Completed the history/reference component. The broad pool/physical-storage parents
remain open; no checklist rows added and counts stay **200 checked / 68 open**.

## Implemented

- Graph.gc_history replaces the no-op with explicit reachability collection.
- Graph.prune_branch explicitly removes a non-current/non-main reference and then
  collects all unreferenced history. Dirty/current/main/unknown operations reject.
- Surviving branches/tags and all merge parents retain complete ancestry. Native
  extra keep states also retain their ancestors. Empty state zero is not counted
  as a commit. Acceptance/entity high-water marks never rewind.
- Recorded deltas, child/root indexes and checkpoint labels are cleaned; read and
  topology caches clear once after actual deletion. Index cleanup walks stored
  entries/links once. Preparing a durable receipt additionally sorts removed IDs.
- Checkpoints are metadata, not permanent pins (A09). There are no extra audit
  pins, mandatory grace period or automatic backup. Immutable views/snapshots,
  temporal indexes, cold effective plans and explicit full owned captures retain
  independent payloads and remain readable.
- Durable collection frames store expected head, optional removed branch and exact
  sorted version set. Replay recomputes/verifies the set. No-op collection writes
  no frame. Reopen/full backup preserve shared/frozen source data and reservations.
- Four injected append/sync failure phases roll back bytes before pruning is
  exposed, then retry/reopen succeed. Indeterminate rollback keeps the existing
  poison/reopen rule. Older binaries cannot consume the new collection variant;
  new binaries continue reading existing journal records.

## Evidence

- Native: **1,216 passed / 51 ignored**, including five reachability/ownership/
  merge/reservation tests and four durable prune failure phases.
- Root Python contract suite: **1,959 passed / 1 skipped**, including six pruning
  cases. **151 executable docs examples** pass. Owner benchmark work and legacy
  nested survey suites are excluded from this Python contract gate.
- Strict all-target Clippy; in-tree FFI build (three unrelated existing warnings),
  scoped Rust formatting, Python/stub syntax and diff checks.
- Release `cargo bench --offline --locked --bench history_collection`:

| Property versions | Unreachable versions removed | Prune time |
| --- | --- | --- |
| 1,000 | 500 | 0.27 ms |
| 10,000 | 5,000 | 2.60 ms |
| 50,000 | 25,000 | 15.00 ms |

The fixture protects a shared branch and holds a cold effective plan through
collection. Raw column counts stay unchanged (50,001 values at 50k versions), and
the old plan reconstructs afterward. One local bounded run per size; no speedup,
RSS or production latency claim. Receipt: `notes/benchmarks/history-collection-2026-10-26.json`.

## Fixed remaining retention inventory

1. Raw pool/entity/value/group reclamation and complete reference closure — open
   in 15.1/15.2; identity reservations are deliberately not released by history GC.
2. Materialized storage checkpoints, persistent compaction/format upgrades and
   crash-safe replacement — open in 10.4/15.2. Journal frames still contain prior
   content; this is logical history pruning, not physical file erasure.
3. Actual optimize/compactor behavior — open; the existing optimize no-op is not
   presented as a working compactor.

This component adds no automatic collection policy. See `docs/api/history-retention.md`.

# Explicit storage checkpoint and physical replacement (10.4 component)

Follow-on to completed 10.3 (`9d78131d`) in the owner-authorized priority window
that began 10:33 UTC. No checklist rows were added. 10.4 remains partial; counts
remain **201 checked / 67 open**.

## Completed component

`DurableGraph.checkpoint_storage(expected_state)` replaces the physical operation
stream with one materialized full retained baseline. Existing graph versions,
clocks, selected state/branch, branches/merge ancestry, raw columns, ID/identity
reservations, frozen sources and graph checkpoint labels survive. Journal schema
and its original revision survive; the next append follows baseline sequence zero.
Old archive-initialization records without a schema revision still default to zero.
No graph version or new clock is created, and scheduling remains explicit.

- Write/sync a sibling temporary, preserve source permissions, verify complete
  recovery, lock the replacement inode, check the owned path again, then atomically
  replace and sync the directory. Keep the original inode locked until the new
  file lock is attached. The implementation currently requires Unix.
- Reject stale requests and changed/symlink paths. Never replace a file that has
  replaced the owned journal. Path mismatch invalidates the old writer.
- Errors before replacement keep the original writer/file usable; errors after
  replacement retain the new inode lock and require reopen. Injected tests cover
  before-write, after-sync, before-replace, after-replace and after-directory-sync
  return failures. These are not power-loss emulation.
- Native prepare/execute/finish separates thread-local graph capture from owned
  file work. Python executes file I/O and candidate recovery outside the GIL.
  During detached work the owner rejects operations; abandoned/mismatched outcomes
  require reopen. Outcomes carry replacement locks even on post-rename errors.
- Explicit rewrite does not guarantee a smaller file. Pruned operation records are
  removed from the new stream; raw values/identity reservations are not purged.
  Open immutable payloads remain valid. Backup and incomplete-tail recovery still
  work after replacement; subsequent writes and branch switches are verified.

API/examples: `../../docs/api/storage-checkpoints.md`.

## Measurement

`cargo bench --offline --locked --bench storage_checkpoint` compares release
reopen of an operation stream with reopen after explicit baseline replacement.
The fixture creates a shared branch halfway, explicitly prunes the other branch,
and preserves all raw values. At 10,000 original property updates:

- retained versions **5,001**, raw values **10,001**, prior frames **10,007**;
- reopen **73.05 ms → 24.46 ms**;
- file **3,924,828 → 2,359,889 bytes**;
- checkpoint including sync/candidate verification **49.45 ms**.

Receipt: `../benchmarks/storage-checkpoint-2026-10-26.json`. This single local
fixture is not a guaranteed latency, throughput or RSS bound. Archive/history
loading still scales with retained data; replay after a checkpoint scales with
subsequent frames.

## Fixed remaining 10.4 scope

1. Chunked/multi-frame baselines beyond the current 256 MiB frame limit, measured
   large retained-data opening/RSS targets and future format migration tooling.
2. Non-Unix replacement and additional subprocess/power-loss/failure corpora where
   required; this slice tests injected failures and OS locks on the local platform.
3. Scheduling/budget decisions remain deferred until MVP/first use per A02. No
   automatic checkpoint threshold or retention pin policy was invented.

Raw entity/property/group/identity reclamation and actual optimize/compactor remain
15/06.3. Native default/escape migration remains 09.4b.2. These are the existing
workstreams; this slice does not close them or invent new checklist entries.

## Final verification

Full native **1,237 passed / 51 ignored**, root Python contract **1,973 passed /
1 skipped**, **156 executable docs examples**, and strict all-target Clippy pass.
Native storage coverage includes four unit tests (one exercising all five fault
phases), two integration workflows and the existing journal/import/retention
suites. Four public Python checkpoint workflows plus existing archive/journal
contracts pass. The owner benchmark and legacy nested Python survey suites are
not part of the root contract gate. In-tree extension rebuild retains three
pre-existing FFI warnings. Scoped formatting/syntax/diff checks pass; no packages
were installed (Black/isort remain unavailable).

# Shared recorded-read plans and detached diffs — 2026-10-26

Existing scope: 04.2 comparison dependency, 19.2 ownership/GIL boundary and AR7.9
shared history ownership. No new checklist rows; **204 checked / 64 open**.
Coordinated rewiring/reversal remains the next endpoint requirement.

## Implemented

- `HistoryForest.prepare_recorded_read` captures logical first-parent history;
  `prepare_commit_diff` captures two selections. `Graph.prepare_commit_diff`
  delegates to the same native path. Plans are Send + Sync and contain shared
  immutable records, not mutable graph access or owner mutation authority.
- `HistoryForest.reconstruct_state_at` and `diff_commits` execute these same
  kernels. Recorded and effective reads now use one immutable lineage cache;
  no duplicate ancestry walker or new unbounded per-state cache was added.
- Cold capture validates ancestry and rejects unknown/missing/cyclic parents.
  A matching cached head shares an Arc lineage array. Empty-root capture does
  not evict the useful head. Cloned plans share the arrays instead of copying
  every retained handle. Existing history collection invalidates the cache.
- Plans survive later mutation, history pruning, explicit pool purge and owner
  destruction. They hold the logical typed values needed for independent replay.
  Frozen group registries/graph restoration ownership remain GraphView/owned
  snapshot responsibilities; logical group IDs alone do not grant those rights.
- Python `diff_commits` releases the owner borrow and GIL during reconstruction
  and typed comparison. Preparation and Python output marshalling remain under
  the GIL. The bridge does type/error translation only.
- Identical-state diffs validate/capture the lineage, then return a correct empty
  diff without reconstructing two copies of the graph. Unknown identical state
  IDs still reject. First-parent resolved-merge semantics remain unchanged.

## Cost receipt

`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline
--locked --bench recorded_diff`; five local release samples, fixture setup
excluded. Receipt: `../benchmarks/recorded-diff-2026-10-26.json`.

| Fixture | First cold capture | Warm capture median (samples 2–5) | Detached execution median (samples 2–5) |
| --- | --- | --- | --- |
| 100k nodes / one recorded version | 12 μs | 83 ns | 8.26 ms |
| 1 node / 50k property versions | 6.70 ms | 62.5 ns | 2.94 ms |

Identical-state execution measures roughly 21–63 ns on these warm local samples,
excluding capture. This is native plan execution, not Python/FFI latency. Capture
can still take O(ancestry) on a cold or alternating head, while reconstruction and
output cost grow with actual data. Only the most recent head is cached. No RSS,
1B-graph, production throughput or universal 100 ns call-budget claim.

An intermediate uncached implementation spent about 5 ms repeatedly capturing the
50k-version lineage. That observation prompted reuse of the existing read context;
the final receipt separates cold and warm costs rather than hiding the first walk.
This is not an overall-engine before/after speed comparison.

## Verification

- Full native: **1,315 passed / 52 ignored**, including captured-plan lifetime,
  detached Send/Sync execution after prune/purge/owner destruction, repeated reads,
  empty/unknown states, resolved merges, malformed ancestry, shared cache identity
  and identical-state fast-path regressions.
- Root Python: **2,018 passed / 1 skipped**, including a real 30k-node recorded
  diff while another Python thread progresses with ordinary bytecode switching
  disabled. Existing owner benchmark and nested legacy survey exclusions remain.
- **168 executable documentation examples**.
- Dedicated serial process-kill matrix: **1 passed / 18 cases**; full reconstruction
  remains compatible with checkpoint/compaction/purge recovery.
- Strict offline all-target Clippy; rebuilt/staged in-tree extension, three known
  unrelated FFI warnings; scoped formatting, syntax and diff checks.

## Remaining original scope

No new endpoint mutation or storage format is introduced. The fixed remaining
04.2/13.3 inventory is in `endpoint-difference-foundation-2026-10-26.md`, followed
by three-way merge planning/caller resolution (14.2). Other historical Python
entry points have not all migrated to detached execution, so 19.2/AR ownership
parents remain open. Deferred Graphiti/service/scheduling decisions are unchanged.

# 12.2 exact owned snapshots — exit review

Completed the existing fixed inventory without adding checklist rows:

1. **Configuration restoration:** Graph-produced temporal snapshots capture every
   GraphConfig field. Lightweight as_subgraph materialization retains configuration,
   direction, allocation high-water marks and frozen group sources. Regular Python
   Graph config/_config now uses shared strict decoding instead of ignoring input.
2. **Owned temporal materialization/raw-pool transfer:** explicit owned captures
   copy all retained backing storage, including unmapped/inactive values and raw
   membership entries, plus identity reservations, complete captured history and
   clocks, references, tags, checkpoints, branches and allocation counters.
   to_graph restores an independent mutable owner; source destruction/later writes
   and other restores do not change the capture. Capturing an older version selects
   it on the copy's current branch while retaining other branches/later versions.
   New write provenance prevents numeric-ID lookalikes from authorizing source writes.
3. **Repeated-switch allocation bounds:** restoration looks up equal retained
   values rather than appending duplicates. Lazy per-column lookup stores one
   representative index per distinct value, with collision buckets only when
   unequal values actually share a hash. Typed equality checks collisions; append/
   bulk append maintain the lookup, positional overwrite/resize invalidates it.
   Repeated branch/reset cycles add zero duplicate column values.

## Explicit costs and limits

Ordinary snapshots do not capture the full raw pool/history. Full captures are
explicit (`owned=True`, native owned_snapshot_at_commit/timestamp); to_graph
requires that capture. Full capture/restore costs scale with retained contents.
Lightweight as_subgraph retains the established manual selection-container contract;
contextless native Graph::from_snapshot compatibility is unchanged. A low-level
HistoryForest snapshot has no graph-owner configuration/backing context.

In-memory ownership transfer is complete here. Persistent arbitrary pool/history
archive/import remains 10.3, storage compaction/checkpoints/migrations 10.4.
Durable source capture restores the embedded core Graph, not a journal lock or
store-level schema policy. Reopen/full backup remain the durable policy path.
Membership-only groups do not acquire invented historical source bindings.
Several GraphConfig fields describe runtime features still tracked elsewhere;
preserving their values does not certify those features as implemented.

## Release receipt

`cargo bench --offline --locked --bench owned_snapshot_restoration`, five warm
switch-pair repetitions; chain, two recorded states, typed node/edge properties.

| Nodes | Cold switch pair | Warm pair median | Owned capture | Owned restore |
| --- | --- | --- | --- | --- |
| 1,000 | 3.39 ms | 3.04 ms | 0.88 ms | 1.02 ms |
| 10,000 | 51.72 ms | 44.25 ms | 13.07 ms | 15.87 ms |
| 100,000 | 1.58 s | 1.55 s | 566 ms | 241 ms |

All cases preserve column lengths across six switch pairs. At 100k nodes, lengths
are 100,000 entity kinds / 200,000 score values / 99,999 edge weights before and
after. Score-column estimated payload plus lookup is 3,435,304 bytes. This is a
local bounded fixture, not an RSS/production/latency guarantee or a measured
speedup against the prior branch. Reconstruction and adjacency/space assembly
remain substantial costs; 32/10.4 retain scale/read-checkpoint optimization.
Receipt: `notes/benchmarks/owned-snapshot-restoration-2026-10-26.json`.

## Verification

- Native: **1,197 passed / 51 ignored**, including owned captures, config equality,
  inactive/raw values, branches/clocks/counters, source destruction, write provenance,
  100 repeated switch/reset cycles and lookup invalidation/collision checks.
- Python: **1,933 passed / 1 skipped**; **144 executable docs examples**.
- Strict all-target Clippy; in-tree FFI build (three pre-existing unrelated warnings),
  scoped Rust formatting, Python/stub syntax and diff checks.

No checklist rows added. Native publication/compound migration remains 09.4b.2;
whole-pipeline publication is the next focused boundary.

## Measured reconstruction follow-through

The initial receipt above exposed quadratic node membership in recorded replay.
`recorded-membership-scaling-2026-10-26.md` records its repair and the matched
second receipt: 100k warm switch-pair median **1.554 s → 0.564 s**, owned capture
**566 ms → 100 ms**, with column lengths still unchanged. The original receipt is
kept as the baseline rather than overwritten.

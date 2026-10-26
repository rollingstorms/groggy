# Recorded membership reconstruction scaling (12.2 / 32)

The 12.2 release receipt exposed a concrete quadratic path: applying an initial
recorded delta checked the growing active-node vector for every added node.
Bulk removal similarly scanned the active vector once per removed node.

Reconstruction now builds membership once when nodes are added, appends only new
IDs, and filters removals once. Property-only deltas do not build that set.
Iteration order, duplicate handling, add-then-remove semantics, retained before
states and property cleanup are unchanged. The extra temporary membership memory
is O(active + changed IDs), without growing retained columns.

Matched release fixture (`owned_snapshot_restoration`), five warm repetitions:

| Nodes | Before warm switch pair median | After median |
| --- | --- | --- |
| 1,000 | 3.04 ms | 2.87 ms |
| 10,000 | 44.25 ms | 33.37 ms |
| 100,000 | 1.554 s | 0.564 s |

At 100k, owned capture changes from 566 ms to 100 ms because constructing the
selected historical snapshot uses this path. Owned restore remains about 236 ms.
Column lengths remain unchanged. Both receipts are retained under
`notes/benchmarks/owned-snapshot-restoration{,-linear}-2026-10-26.json`.
This is one local bounded workload, not a whole-engine speedup guarantee. Snapshot
payload cloning and graph-space assembly still scale with selected content;
read/recovery checkpoint and larger scale work stay in 10.4/32.

Verification includes recorded membership order/duplicates/removals, the owned
snapshot/branch/reset tests, strict all-target Clippy and the native suite
(**1,205 passed / 51 ignored**, including the concurrent native boundary slice).
No new checklist rows; this is a measured follow-through on 12.2 and existing 32.

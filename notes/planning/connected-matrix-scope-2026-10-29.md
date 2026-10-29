# Connected matrix scope consistency — 2026-10-29

Existing 26.1 / AR2 / AR4 read-facade scope; no new checklist row.

## Reproduced defect and repair

Four pre-fix Python cases showed dense/sparse connected exports silently dropping
an inactive selected node or edge. The method filtered its cached node/edge IDs
against current liveness, so row shape or edge contributions could disappear
without an error. The existing matrix contract already required stale explicit
IDs to reject. Rewiring outside selected nodes already rejected in the shared
matrix constructor and remains covered.

Subgraph.adjacency_matrix now calls the canonical instance-sensitive selected-scope
validator, then passes the shared ascending ID slices directly into the existing
matrix constructor. This removes two temporary filtered ID vectors. Warm scope
validation reuses its stamp; changes to selected topology invalidate it. No new
snapshot, owner, mutation or matrix arithmetic path is added. Aliases adj/to_matrix
inherit the fix. Valid endpoint revisions inside selected nodes remain live reads.
Previously returned matrices stay owned, and captured GraphView matrices continue
using their immutable state even after node/edge removal.

## Verification and costs

Two native tests cover stale node/edge rejection for dense/sparse, original owned
and frozen data retention, inside endpoint revisions and outside closure failures.
Five Python cases cover both matrix storage forms, aliases and rewiring. Focused
matrix scope/captured-export suite: ten cases. Full gate:
**1,503 native / 54 ignored**, strict all-target core Clippy,
**2,559 root Python / 1 skipped**, **249 executed documentation examples**.
Full-ABI release extension rebuilt and atomically staged; three existing unrelated
FFI warnings remain. Scoped formatting, Python syntax and diff checks pass. Owner
benchmark/review work remains untouched and excluded from the root Python gate.

A release-extension sparse directed-chain benchmark measures five warmed bound
method calls per size; graph setup, warmup and scalar verification are excluded.
Sparse operator construction/Python wrapper are included. Separate quiet build
runs, not alternating pairs:

| Nodes | Former filtered IDs | Checked shared IDs |
| --- | --- | --- |
| 1,000 | 0.358 ms | 0.347 ms |
| 100,000 | 46.012 ms | 40.699 ms |

No universal speedup, memory/RSS, GIL or cold-preparation guarantee. Both still
construct complete owned numerical operators. Raw data:
`../benchmarks/connected-matrix-scope-2026-10-29.json`.
API/example: `../../docs/api/connected-matrix-scope.md`.

**211 checked / 57 open** remains unchanged: source defaults and effectful view
migration retain their original scope. Reader consistency is improved here; a
current connected Subgraph is still not an immutable snapshot.

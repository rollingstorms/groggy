# Current full-view cache correctness and cost — 2026-10-29

Existing scope: 26.1 / AR4 / AR7.5. This repairs the connected compatibility
entry point; it does not close immutable default migration.

## Problem and implementation

Seven pre-fix Python regressions reproduced wrong cached membership after
same-cardinality node replacements, edge replacements, edge-to-node trades
(automatic and manual publication), and explicit restoration to a retained state.
The former key summed lengths of freshly allocated node/edge ID vectors: it
collided across distinct selections and performed O(N+E) work on every warm hit.

GraphSpace now owns an opaque MembershipRevision: existing instance identity plus
a separate membership counter. All eight scalar/bulk activation/deactivation
paths change it only when active IDs change. Duplicate activations and missing
removals leave membership stable. Property writes and endpoint-only revisions
also preserve it; their value/topology readers retain separate invalidation.
Restored spaces have distinct identity even when numeric versions coincide.
Python's Graph.view cache marshals this native key instead of copying ID vectors.
Reading/comparing the key is O(1), without allocating a new identity per mutation.
The token retains no graph payload and is not a durable ID or write capability.

## Evidence

Two native regressions cover all mutation paths, stable no-ops, property/endpoint
independence and equal-version restoration. Eight Python cases cover the seven
former failures and cached-object reuse with correct live property/neighbor reads.
Explicit pin_view remains immutable; Graph.view remains the existing connected
compatibility API with selected membership and live owner reads.

A release-extension Python benchmark measures seven samples of 128 calls to an
already bound Graph.view method. The initial full-view construction and immutable
capture are excluded. Quiet local before/after medians:

| Chain nodes | Former key | Membership revision |
| --- | --- | --- |
| 1,000 | 2,444.664 ns/call | 39.711 ns/call |
| 100,000 | 431,954.758 ns/call | 40.688 ns/call |

These are nonalternating local runs, not a universal FFI latency certification.
Only unchanged warm hits are measured; initial construction still copies IDs.
Raw samples: `../benchmarks/current-full-view-cache-2026-10-29.json`.
Harness: `../../benches/current_full_view_cache.py`.
API example: `../../docs/api/current-full-view-cache.md`.

## Verification and remaining work

- Full native: **1,494 passed / 54 ignored**, no failures.
- Strict all-target core Clippy passed.
- Full-ABI release extension rebuilt and atomically staged; three unrelated
  existing FFI warnings remain.
- Root Python: **2,519 passed / 1 skipped** (owner's untracked benchmark excluded).
- **245 executed documentation examples** passed.
- Scoped formatting, Python syntax and diff checks passed.

No checklist rows added or broad parents closed: **211 checked / 57 open**.
Remaining priorities are immutable default/effectful facade migration, general
compound topology acceptance, and full immutable payload construction costs.

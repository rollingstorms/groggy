# Writer coordinates — embedded session correction


## Writer coordinate guard — 2026-10-27 (09.5 / AR3 / 19.2)

Reproduced and fixed a Python session writing to a different branch when both
branches shared a state ID. Writers now capture owner, branch and expected head;
native acceptance validates the full coordinate before mutation or durable append.
All batch/upsert/checkpoint/merge/restore entries are covered. Explicit refresh
retargets the session. A callback refresh cannot replace an in-flight call's
coordinate. Immutable markers share a small owner/branch binding and pin no history.

Native writers remain protected by their exclusive graph borrow. Physical
checkpoint/compaction preserves the owning identity; reopen creates a new owner
and rejects old markers. No persisted format or result shape changes.

Verification: 1,386 native / 56 ignored; 2,049 root Python / 1 skipped; 184
executable documentation examples, strict all-target Clippy and rebuilt release
FFI (three existing warnings). Bounded matched Python release medians on the same
extension: scalar writer update 8.47 → 8.66 microseconds (about 0.19 microseconds
extra); 1k-row bulk batch 1.464 → 1.459 ms (within local timing variation).
Receipt: `../benchmarks/write-coordinate-overhead-2026-10-27.json`. This does not
certify an FFI latency budget or production/RSS throughput.

This advances existing 09.5/AR3/19.2 without closing their remaining service,
compound topology and wider API scopes. Inventory remains **208 checked / 60 open**.
No new rows.

## Fixed covered entry inventory

- Batch mutations: all add/remove/set/unset/identity/kind bulk conveniences.
- Managed node and edge upserts, including no-op upserts.
- Explicit checkpoints.
- Retained merge preparation; final outcomes independently retain owner/head/branch guards.
- Both required restoration modes.

The capture/advance objects contain only an identity marker, shared branch name
and expected head. They are Send + Sync, immutable and independent of graph data.
They grant no authorization, mutex ownership, lease or retained-state pin. Native
validation happens under the same owner operation as preparation/publication.
Existing head/dirty/schema/time validation remains in effect. The direct APIs
accept optional expected_coordinate; Python writer sessions always supply one.

Regression tests cover same-head branch switches for every entry, unchanged
journal bytes/checkpoints/versions/next IDs on rejection, foreign owners, explicit
refresh, callback refresh, immutable cross-thread markers, manual dirty state,
no history pin, maintenance and reopen. Full suite caught the independent upsert
wrapper path and it was migrated before acceptance.

Multi-client service queues/isolation, broader immutable Subgraph migration and
compound topology execution remain their original open scope. This correction
does not claim that a local thread-confined Graph is a multi-user service.

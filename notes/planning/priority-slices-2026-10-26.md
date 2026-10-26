# Priority slices — 2026-10-26

Branch: `groggy-roadmap-2026-10-20`. After completing 16.2, the owner authorized up
to 90 minutes of priority slices, aiming to complete existing tasks and document
remaining work. The follow-on window began 03:49 UTC. This record consolidates
its outcomes and the remaining core priorities. No new checklist rows were added.

Current inventory after the second window's storage slices: **201 checked / 67
open**. Completed full retained archive/import (10.3, `9d78131d`) and explicit
recovery baseline/locked replacement component (10.4). The first-window receipts
below retain their original counts; second-window results follow at the end.

## Completed work and commits

| Existing scope | Commit | Delivered |
| --- | --- | --- |
| 16.2, completed before the follow-on window | `f932d381` | Effective reconstruction, endpoint timeline validation, frozen nested sources, explicit legacy inputs, resolved first-parent merge reads and cached event prefixes. |
| 11.2 | `ebe087d8` | Whole typed property history, null/absence distinctions, node/edge scalar/bulk/mixed unsets, exact clocks, coalescing and durable replay. |
| 12.2 | `e2d28cf1` | Full configuration, explicit owned raw-pool/history capture and independent restoration, inactive ID/name reservations and bounded repeated-switch column allocation. |
| 12.2 / existing 32 | `a58a4f31` | Removed quadratic recorded node membership reconstruction; single membership/removal passes preserve semantics. |
| Components of 09.4b.2 | `0efefed9` | One acceptance for certified outer property pipelines, failure/cancel rollback, shared staging across algorithms, and complete selected manual import baselines. |
| 14.1 | `346dda4c` | Branch-selected recorded timestamps, immutable/identity lifecycle, durable policy/recovery/failure verification and Python manual reset bridge. |
| History component of 15.1 / 15.2 | `f03be97d` | Real history-only collection, explicit branch pruning, shared/merge ancestry, immutable payloads, durable removal receipts and replay/failure/backup checks. |

Follow-on checklist movement: **197 checked / 71 open → 200 checked / 68 open**.
The three existing rows closed were 11.2, 12.2 and 14.1. Native boundary and
retention parents remain partial, with smaller fixed remainders described below.
Counts include overlapping parents/children; they are not a completion percentage.

## Measured results

- Matched 100k-node branch switch-pair median: **1.554 s → 0.564 s** after the
  membership repair; owned capture **566 ms → 100 ms**. Repeated switches append
  zero duplicate column values. Full replay/space assembly remains nontrivial.
- A 50k-property-version fixture prunes **25,000 unreachable versions in 15 ms**,
  protecting shared ancestry and an open cold read plan. Raw column counts remain
  50,001; history collection is not pool or physical-file compaction.

Receipts: [owned snapshot baseline](../benchmarks/owned-snapshot-restoration-2026-10-26.json),
[repaired membership](../benchmarks/owned-snapshot-restoration-linear-2026-10-26.json),
[history collection](../benchmarks/history-collection-2026-10-26.json).
These are bounded local fixtures, not production/RSS/whole-engine speed guarantees.

## Final verification

- Native `cargo test --offline --locked --quiet`: **1,216 passed / 51 ignored**.
- Root Python contract suite: **1,959 passed / 1 skipped**; **151 executable docs
  examples** pass. The owner's benchmark test and legacy nested survey suites are
  excluded from this Python gate; this does not claim every historical test passes.
- Strict all-target Clippy passes. The extension is built/staged in tree; its three
  pre-existing unrelated warnings remain. No wheel/pip/package installation.
- Scoped Rust formatting, Python/stub syntax, diff checks and local doc-link checks.
  Black/isort are unavailable in the existing environment; no packages were added.
- Owner changes to development benchmark files and their untracked receipt/test
  remain untouched. All implementation slices above have separate commits.

## Fixed next-work inventory after the first window

1. **Native mutation boundaries (09.4b.2):** migrate native default/infallible
   construction, audit mutable raw pool/space escapes, and certify or stage remaining
   topology/uncertified multi-stage algorithms. Certified property pipelines and
   selected import baselines are complete. Effect-aware IR/default execution policy
   remains AR7.8/30.1; outer execution cannot undo earlier optimizer reordering.
2. **Complete persistent backing storage (10.3/10.4):** arbitrary raw-pool/existing
   graph archive/import, materialized recovery checkpoints, physical compaction,
   format migrations and crash-safe replacement. Current reopen cost still grows
   with journal operations; in-memory owned transfer is complete.
3. **Remaining retention (15.1/15.2/06.3):** reclaim raw entities/values/groups and
   identity reservations only with complete retained-reference closure. Implement
   the actual optimize/compactor path. History-only pruning is complete and does
   not release names, recycle IDs or physically erase journal content.
4. **Topology history and merging (04.2/13.3/14.2):** typed endpoint changes,
   reversals, three-way conflict detection and caller resolution, including explicit
   parallel fact preservation. Already-resolved merge reconstruction is complete.
5. **Views/query/index/execution closure:** remaining connected-view migration,
   one liveness/adjacency policy, global secondary index lifecycle, typed pipeline
   intermediates/caching and final compatibility/conformance (02/03/18/19/21/26/30/AR).
6. **Existing numerical/scale/release gates:** finish the stated remaining numerical
   parent scopes and representative throughput/RSS/large-graph evidence; extend
   failure injection, capability matrix and release docs (28/29/31/32/34).
7. **Deferred integration:** Graphiti ingest/driver, provenance/retrieval-specific
   work and retrieval benchmarks remain behind the core as previously agreed.

These are the existing workstreams, not new requirements discovered by this pass.
No additional owner answers were needed for the implemented slices. Service
concurrency/deployment and deferred adapter choices retain their original workbook
scope. The flat checklist remains the authority for individual entries.

## Second priority window — began 10:33 UTC

Completed 10.3: full retained graph archives and existing-graph durable journal
initialization. Exact backing values/column positions, allocation and identity
reservations, branches/history/clocks/configuration and frozen independent source
domains are preserved. Evidence: `retained-archive-import-2026-10-26.md`; local
100k-node save/load receipt: `../benchmarks/graph-archive-2026-10-26.json`.
**201 checked / 67 open**, no new rows. The first window's table/gates above remain
a historical receipt. The next slice targets explicit 10.4 storage checkpoints and
crash-safe journal replacement; automatic schedule remains deferred.

### Second completed storage slice (10.4 component)

Explicit materialized journal baselines and locked atomic Unix replacement are
implemented. Graph versions/raw storage/clocks/schema revision survive; file work
and candidate verification release Python's GIL. Five injected failure boundaries,
continued writes, backup, tail recovery and changed-path rejection are verified.
See `storage-checkpoint-2026-10-26.md`; local 10k-update opening improved from
73.05 ms to 24.46 ms. Chunking/scale/migration/non-Unix/crash-corpus scope remains,
and A02 scheduling is still deferred. Counts **201 checked / 67 open**.

### Second-window verification

- Native full gate: **1,237 passed / 51 ignored**.
- Root Python contract: **1,973 passed / 1 skipped**; **156 executable docs
  examples**. Owner benchmark and legacy nested survey suites remain excluded.
- Strict all-target Clippy; scoped Rust formatting, Python/stub syntax, diff checks.
- Both new release benchmarks are recorded under `notes/benchmarks/`.
- No dependency install, push, auto-checkpoint schedule or Graphiti scope change.
  Owner benchmark/development work remains untouched.

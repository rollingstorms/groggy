# Priority slices — 2026-10-26

Branch: `groggy-roadmap-2026-10-20`. After completing 16.2, the owner authorized up
to 90 minutes of priority slices, aiming to complete existing tasks and document
remaining work. The follow-on window began 03:49 UTC. This record consolidates
its outcomes and the remaining core priorities. No new checklist rows were added.

Current inventory after the continuation: **215 checked / 53
open**. Explicit state restoration advances existing 13.3 without adding rows. Completed full retained archive/import (10.3, `9d78131d`) and explicit
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

### Exact floating restoration follow-up

Found and repaired signed-zero/NaN representation reuse under existing 12.2/10.3.
Public equality and history hashes remain canonical; restoration keys preserve
exact scalar/vector/compressed bits and matching-slot reuse. Node/edge, owned,
archive and durable checkpoint/reopen/backup workflows have native/Python coverage.
Evidence and repeat benchmark: `restoration-float-bits-2026-10-26.md`. No new rows,
counts remain **201 checked / 67 open**.

### Final gates and current next-work inventory

After the exact-bit repair: **1,239 native passed / 51 ignored**, **1,975 root
Python passed / 1 skipped**, **156 docs examples**, strict all-target Clippy.
The preceding second-window gate counts are retained as intermediate receipts.
No new checklist rows were added; one existing parent (10.3) closed, 10.4's concrete
recovery/replacement component completed, and restoration repaired within 12.2.

1. Native default/infallible construction, mutable raw-storage escapes and
   uncertified/topology compound acceptance (remaining 09.4b.2 inventory).
2. Remaining 10.4 chunking/large-data opening and RSS/migration/platform scope.
   Checkpoint scheduling remains deferred per A02; explicit Unix baselines work.
3. Raw pool/entity/value/group/identity reclamation with reference closure and
   actual optimize, then endpoint rewiring/reversals and conflict-aware merges.
4. Existing connected-view/query/index/pipeline and numerical/algorithm/conformance
   remainder. The flat checklist preserves their individual IDs and limits.
5. Graphiti/provenance/retrieval integration remains deferred behind the core.

Owner development/benchmark changes remain untouched; implementation is committed
in coherent slices, with no push or automatic maintenance policy introduced.

## Third priority window — began 11:18:59 UTC

Completed native default/infallible migration under 09.4b.2, including validated
configuration and registered atomic accessor node creation. Evidence and receipt:
`native-default-migration-2026-10-26.md`. The fixed component inventory is now
**one completed, two remaining**: raw-storage escapes and uncertified/topology
compound acceptance. Counts remain **201 checked / 67 open**, no new rows.
The raw audit identified a pooled-attribute setter that silently returned success;
its rejection behavior belongs to the same existing boundary scope.

Third-window first-slice gates: **1,246 native passed / 51 ignored**, **1,986
root Python passed / 1 skipped**, **157 docs examples**, strict all-target Clippy.
The owner benchmark and legacy nested survey suites remain excluded.

### Raw owner mutation boundary completed

Sealed public pool/space mutation and repaired silent unsupported pooled attribute
operations. Preserved all 46 raw corruption/recovery cases as internal unit tests.
Evidence: `raw-storage-boundary-2026-10-26.md`. Native boundary fixed inventory is
now **two completed, one remaining** (uncertified/topology multi-stage acceptance),
with **201 checked / 67 open** and no new rows.

Raw-boundary gates: **1,250 native passed / 51 ignored**, **1,986 root Python
passed / 1 skipped**, **157 docs examples**, strict all-target Clippy.

### 2026-10-26 — classic algorithm outer property acceptance

All 13 built-in classic registry algorithms now share builder Context property
staging, including their weight/marker/heuristic/personalization/seed reads.
Success publishes once; late failure rolls back. Evidence and remaining limits:
`classic-atomic-algorithms-2026-10-26.md`. Existing 09.4b.2 remains open for topology
compound acceptance and unsupported custom effects; fixed component inventory
remains two complete plus this narrowed final component. No new checklist rows:
**201 checked / 67 open**. Broader IR/default policy remains AR7.8/30.1.

Third-window final gates: **1,254 native passed / 51 ignored**, **1,999 root
Python passed / 1 skipped**, **158 documentation examples**, strict all-target
Clippy. Rebuilt the in-tree extension; no wheel/package installation. Python
owner benchmark and legacy nested survey suites remain excluded.

The requested work window is complete. Three coherent implementation slices:
native default/fallible insertion and accessor creation (`914476f6`); sealed raw
mutation and explicit pooled attribute rejection (`34d735b0`); classic algorithm
shared property acceptance (this commit). Checklist remains **201 checked / 67
open** because the completed components sit inside existing 09.4b.2. Native
defaults and raw escapes are closed; classic property wrappers are certified;
topology compound acceptance remains open. No new checklist rows were added.

Next: prepare topology compound effects with explicit ID/remapping and read-overlay
semantics; then finish existing storage chunking/open-time/RSS and reclamation
requirements. Checkpoint scheduling and Graphiti remain deferred per owner.

### Fourth priority window — chunked storage baselines (2026-10-26)

Window began 12:52:12 UTC. Existing 10.4 chunked-baseline component is complete:
bounded 8 MiB streaming fragments, ordered/full-stream checksum verification,
scratch-backed recovery and locked atomic initial creation/replacement. A 566 MB
retained journal reopens in 1.30 s / 1.73 GB peak RSS locally; decoded memory
remains material work. Evidence: `chunked-storage-baselines-2026-10-26.md`.
Gates: 1,259 native / 51 ignored; 2,000 root Python / 1 skipped; 158 docs; strict
Clippy. No new checklist rows: **201 checked / 67 open**. 10.4 remains partial
for decoded-memory/migration/platform/process-failure scope; A02 schedule deferred.

### Borrowed archive-state validation completed

Following 242e80e2, selected-state validation now borrows current column values
instead of cloning current properties and encoding two complete maps. Paired
release medians (three alternating pairs, 270 MiB unique text / 566 MB journal):
open **1.464 → 1.306 s**, isolated-process peak RSS **1,723,629,568 →
1,203,109,888 bytes** (~11% less time, ~30% less RSS). This is a local fixture,
not a universal bound. Receipt: `borrowed-archive-validation-2026-10-26.md`.
Gates: **1,261 native / 51 ignored**, **2,000 root Python / 1 skipped**, **158
documentation examples**, strict all-target Clippy and in-tree extension rebuild.
No new checklist rows; **201 checked / 67 open**. Decoded graph/history still
scale with retained data. The owner GraphTable grouped-view answer was reconfirmed
and its stale appendix pending marker corrected in d5bdc24a.

### Checkpoint process-death recovery verified

Existing 10.4/31.4 scope now includes actual SIGKILL/reopen tests at six boundaries,
including partial large-fragment output and before/after atomic replacement.
Complete state/clocks/history/schema/checkpoints, released locks and continued
append are verified. See `checkpoint-process-recovery-2026-10-26.md`. This does
not certify sudden power loss/non-Unix replacement or add orphan maintenance.
Gates: **1,263 native / 51 ignored**, strict all-target Clippy; production API
unchanged. No new rows; **201 checked / 67 open**.

### Recorded-state reconstruction completed without per-version whole-state copies

HistoryForest borrows retained deltas; GraphSnapshot uses one owning replay buffer
and a generation/order membership index. Existing immutable APIs, selected lineage,
IDs/types/clocks and write semantics remain. Native prefix parity and 10k-version
scalar-add regressions pass. Three-pair release fixture (10k nodes / 1k property
versions): **1.025 s → 1.722 ms**, requested allocations **4.18 GB → 4.62 MB**;
ordinary mutation/FFI latency is not measured. Scalar-add scratch peak grows by
~40 KiB in its small fixture. Evidence: `recorded-reconstruction-2026-10-26.md`.
Gates: **1,265 native / 51 ignored**, **2,000 root Python / 1 skipped**, **158
documentation examples**, strict Clippy and rebuilt extension. No new rows;
**201 checked / 67 open**. Full retained integrity scanning is the next 33.2 slice.

### Process-test isolation repair

A full parallel run exposed transient descriptor inheritance during fork/exec,
which can retain an unrelated fixture's flock briefly after its owner closes.
The six-boundary SIGKILL matrix now has a dedicated ignored-by-default serial
gate, executed successfully locally and wired after normal Rust tests in the CI
workflow. Default tests keep the other lock tests parallel. Production locking is
unchanged; the serial gate is reported separately from default ignored counts.
No new checklist row. See `checkpoint-process-recovery-2026-10-26.md`.

### Retained-memory integrity diagnostic completed (33.2 component)

Native/Python Graph and DurableGraph now expose check_integrity, verifying
portable recovery invariants plus every retained first-parent transition, including
unreferenced/nonhead versions. Structured counts and contextual errors are returned
without mutation. Detached verification releases the Python GIL; capture remains
owner-bound. Durable scope is retained memory/current schema, not rereading an open
journal. Evidence and exact limits: `retained-integrity-2026-10-26.md`.
Gates: **1,270 native / 52 ignored**, separate serial SIGKILL gate **1 passed**;
**2,004 root Python / 1 skipped**, **160 docs**, strict Clippy and rebuilt extension.
No new checklist rows; **201 checked / 67 open**. Broad 33.2 retains original
index/cache/physical-file/service diagnostic scope.

### JSON mutation recoverability gap repaired (07.4 / 05.3)

A reproduced native path accepted malformed typed JSON before later save failure.
The shared graph property boundary now applies the same intrinsic syntax rule as
retained storage, rejecting whole scalar/bulk/create/batch/upsert inputs before
values/versions/IDs/reservations change. Valid text and large-number spelling remain
exact; ordinary Text/non-JSON values keep their distinct behavior. Four native
regressions and a release parse-cost receipt are recorded in
`json-property-validation-2026-10-26.md`.
Gates: **1,274 native / 52 ignored**, serial SIGKILL gate **1 passed**; **2,004
root Python / 1 skipped**, **160 docs**, strict Clippy, in-tree extension rebuild.
No new checklist rows; **201 checked / 67 open**.

## Fourth-window consolidated exit

Work began 12:52:12 UTC, was interrupted during reference-build setup, and resumed
with the retained worktree/source state preserved. All implementation work is
committed on `groggy-roadmap-2026-10-20`; no push was performed.

| Commit | Completed existing component |
| --- | --- |
| 242e80e2 | 10.4: streaming large baselines, complete fragment verification and atomic initial publication; 566 MB recovery evidence. |
| 416f6db2 | 10.4/32.2: borrowed current-column archive validation; paired reopen time/RSS improvement. |
| 146f3fc1 | 10.4/31.4: six abrupt checkpoint process-death boundaries. |
| eee51be9 | Isolate the process-death matrix in an explicitly executed serial CI gate; remove its parallel flock-inheritance race. |
| 5363241f | 12.2/16.2/32.2: borrowed deltas and single-buffer recorded reconstruction, stable node order and allocation evidence. |
| 2a0a6c99 | 33.2: explicit full retained-memory/schema integrity reports, all-version transition verification and contextual errors. |
| 09e3be15 | 07.4/05.3: reject malformed native JSON before scalar/bulk/create/batch/upsert mutation, exact valid-text retention. |
| d5bdc24a | Reconfirm the already implemented explicit-column grouped-view converter and resolve its stale appendix marker. |

Final verification: **1,274 native passed / 52 ignored**, dedicated serial SIGKILL
matrix **1 passed** covering six phases, **2,004 root Python passed / 1 skipped**,
**160 executable docs examples**, strict all-target Clippy and rebuilt in-tree FFI.
The 52 default ignores include the separately executed process matrix; the other
51 are pre-existing. Owner benchmark and legacy nested survey suites remain
excluded. Three pre-existing FFI warnings remain. Scoped Rust formatting/diff
checks pass; no external packages were installed.

All benchmark results are local fixture evidence, not whole-engine guarantees.
The paired 270 MiB text fixture improved median reopen **1.464 → 1.306 s** and
peak RSS **1.724 → 1.203 GB**. The prepared 10k-node/1k-version reconstruction
fixture improved **1.025 s → 1.722 ms** with far fewer requested allocations.
Decoded data and branch-frontier memory still scale with retained content; JSON
syntax checking adds its documented parse cost.

Checklist remains **201 checked / 67 open**, with no added checkbox rows. These
are completed components inside the existing broad parents, not completion of
arbitrary topology transactions, raw reclamation, all property/index/view/API
contracts, three-way merges, services or deferred Graphiti/retrieval work.

### Existing priority work after this window

1. Raw entity/value/group/identity reclamation and real compaction (15/06.3), using
   tested explicit baseline replacement and preserving shared ancestry/open owned
   payloads. Existing identity/reference closure rules remain part of that work.
2. Typed endpoint changes/reversals (04.2/13.3), then explicit conflict-aware
   three-way merges (14.2). The new integrity validator's fixed endpoint check must
   migrate with the established endpoint-change contract when that feature lands.
3. Remaining 09.4b.2/AR7.8 compound execution/default-policy scope. No built-in
   owner-topology mutation steps were found in the current algorithm registry;
   certified property effects work, unsupported effects reject. Do not invent an
   arbitrary topology pipeline API simply to force a parent checkbox closed.
4. Existing index/live-view/query/numerical/algorithm certification remainders and
   migration/platform/physical-file diagnostics stay under their original IDs.
   A02 scheduling/budgets and Graphiti/retrieval remain deferred as recorded.

The temporary managed reference checkout was archived after paired measurement.
Owner development/benchmark modifications and untracked benchmark fixtures remain
untouched.

## Fifth priority window — explicit retained-column compaction

Owner authorized up to another 90 minutes on the existing priorities. The selected
order follows the open checklist, core-first direction and storage dependencies;
it is an execution choice, not an owner ranking of all 67 unchecked entries.

15.2's no-op optimize component is now real: native/Python exact raw-column
compaction, complete working-index remapping, detached execution and stale-owner
checks. All distinct raw values remain; history, groups, inactive IDs/reservations
and immutable pins are preserved. See `pool-compaction-2026-10-26.md` for native,
Python, archive and local release evidence. No new rows; **201 checked / 67 open**.
Destructive distinct-value/entity/group collection and identity release still need
complete reference closure. Durable physical installation is the next component.

### Durable exact-column compaction completed

15.2/10.4 now includes explicit `compact_storage`, preserving all distinct raw
values and retained history while physically replacing the baseline and installing
the same remapped columns in memory. Five injected failures plus twelve actual
SIGKILL/reopen cases preserve old/new layout, clocks/schema/history and locks.
Warm-cache release fixture: **8.58 → 5.80 MB**, reopen **34.37 → 22.31 ms** with
all eleven versions retained. Details/limits: `pool-compaction-2026-10-26.md`.
Gates: **1,279 native / 52 ignored**, serial matrix **1 passed / twelve cases**,
**2,009 root Python / 1 skipped**, **162 docs**, strict Clippy, rebuilt extension.
No new rows; **201 checked / 67 open**. Raw duplicate reclamation and verified
physical compaction are done; deleting unreachable distinct values/entities/groups
and releasing external identities remain separate destructive reference-closure
work. Ordinary checkpoints continue to preserve raw column positions.

### Explicit orphan raw-property reclamation completed

The same compaction API now accepts `reclaim_unreferenced=True`. All retained
before/after property values and working mappings protect raw slots; nonhead and
unreferenced retained versions are included. History-root changes invalidate a
detached result. Frozen source payloads and owned snapshots remain independent;
entities, groups and identity reservations remain retained. Default compaction
continues to preserve all distinct raw values.

Existing 15.2 now has actual history collection, duplicate-slot compaction,
orphan raw-property reclamation and verified physical replacement. Remaining
retention scope is raw entity/group reclamation and identity/reference closure,
not the already implemented property collector. A 1k-node fixture reclaims 20k
orphan values and reduces physical baseline ~6.35 MB → ~0.73 MB while retaining
both remaining versions and an open discarded-state view. Exact scope/limits and
receipts: `pool-compaction-2026-10-26.md`. No new checkbox rows; **201 checked /
67 open** because the broad retention parents still include that remaining scope.

Reclamation gates: **1,285 native passed / 52 ignored**, serial process matrix
**1 passed / twelve cases**, **2,011 root Python passed / 1 skipped**, **163 docs
examples**, strict Clippy and rebuilt extension. Both-column failure atomicity and
malformed-unused-value rejection have internal corruption regressions. Existing
owner benchmark/nested legacy survey exclusions remain; no external packages/push.

### Physical-file integrity verification completed

Existing 33.2 now includes actual named-file verification: the same recovery parser
performs read-only framing/checksum/sequence/semantic replay, followed by every
retained-state transition and current schema check. Partial tails are reported,
not silently repaired; an error blocks the owner until close/reopen. Independent
file cursor and retained inode locks protect detached operation and append
position. Native/Python tests include complete byte corruption, partial tails,
changed paths, foreign/dropped owners, valid-checksum semantic corruption and real
chunked baseline input. Scope/receipt: `storage-verification-2026-10-26.md`.

Final window gates: **1,292 native passed / 52 ignored**, dedicated serial SIGKILL
matrix **1 passed / twelve cases**, **2,014 root Python passed / 1 skipped**,
**164 docs examples**, strict all-target Clippy, rebuilt in-tree extension, scoped
formatting/diff checks. The owner's benchmark/nested legacy survey exclusions and
three FFI warnings remain. No packages installed or push performed.

## Fifth-window consolidated exit

Four completed existing implementation slices, each committed separately:

1. `338e4da2`: real native/Python exact property-pool compaction and optimize.
2. `1e996a49`: verified locked physical compaction with atomic memory installation.
3. `a1d9ef48`: optional orphan raw-value reclamation with all retained history roots.
4. Physical journal verification: shared read-only recovery parser plus full
   retained audit, with failure/lock/chunked-input coverage and a local cost receipt.

Original scope still open: raw entity/group reclamation and external identity
release with full reference closure (15/06.3); endpoint changes/reversals and
three-way merging (04.2/13.3/14.2); remaining execution/default, view/index/query,
numerical/scale/release contracts under their existing checklist entries. Format
migrations/non-Unix support and the deferred schedule/service/Graphiti decisions
retain their prior scope. No additional questions or checkbox rows were created.

**201 checked / 67 open** remains the literal row count. This window completed
property collection/physical maintenance and diagnostic components inside broad
15.2/10.4/33.2; it does not claim those broad parents or all 67 entries are finished.
Owner development/benchmark files and receipts remain unstaged and untouched.

## Sixth priority window — complete explicit retention closure

Completed existing **06.3/15.1/15.2**: previewable root-traced purge of orphan
entities, values, groups and names, permanent interval ledgers, sparse endpoint
reclamation, complete archive/owned/journal persistence and verified locked
physical replacement with identical memory installation. Existing history-only
pruning remains explicit and separate. A09 policy and open immutable payloads are
preserved, including nested frozen group sources and current pin write authority.

Concrete issue fixed inside this scope: dense interior holes and giant high-ID
prefix allocation after purge. Ordinary graphs keep dense endpoints; purged graphs
use sparse raw records. Local 100k-orphan journal fixture shrinks 8.8 MB → 1.6 KB;
raw endpoint lookup trades ~3 ns dense for ~15 ns sparse. Details/receipts/gates:
`retained-purge-2026-10-26.md`. No claim about universal latency/RSS/production use.

**204 checked / 64 open**, same 268 rows. Final gates: **1,308 native / 52 ignored**,
serial **18-case** SIGKILL matrix, **2,017 root Python / 1 skipped**, **167 docs
examples**, strict all-target Clippy and rebuilt extension. Owner exclusions and
three unrelated FFI warnings remain; owner benchmark work is untouched.

Next core order: endpoint changes/reversals **04.2/13.3**, then caller-resolved
three-way merging **14.2**, remaining execution/default boundaries, view/index/query
and numerical/scale/release closure. Scheduling/Graphiti/service remain deferred.

### Endpoint comparison/safety foundation (04.2 / 13.3)

Endpoint-only snapshot differences previously disappeared; raw restoration could
silently overwrite an existing endpoint pair. Canonical typed endpoint diffs now
preserve sorted stable edge identities/before-after tuples and reach commit diffs
and Python output. Raw dense/sparse reconstruction rejects untracked rewiring,
and snapshot restoration preflights before replacing working membership.
Coordinated rewiring/reversal and recorded/effective/durable mutation remain open;
no wire format or owner policy changed. Fixed remaining inventory and evidence:
`endpoint-difference-foundation-2026-10-26.md`.

Gates: **1,310 native / 52 ignored**, **2,017 root Python / 1 skipped**, **168 docs
examples**, rebuilt extension and strict all-target Clippy. Same exclusions/known
FFI warnings. **204 checked / 64 open**, no new checkbox rows.

### Shared recorded-read plans and detached diffs

Recorded reconstruction and effective reads now share the existing immutable
lineage cache. Native captured diff/read plans survive prune/purge/owner destruction;
Python recorded diff reconstruction/comparison releases the GIL. Cloned plans share
lineage arrays, and identical-state diffs skip reconstruction after validation.
Cold ancestry walks and Python output marshalling remain owner-bound. A local
50k-property-version fixture measures 6.70 ms cold capture, 62.5 ns matching warm
capture and 2.94 ms detached diff execution; this is not whole-engine/FFI latency.
Evidence/limits: `recorded-diff-plans-2026-10-26.md`.

Final gates: **1,315 native / 52 ignored**, serial **18-case** process-kill matrix,
**2,018 root Python / 1 skipped**, **168 docs examples**, strict all-target Clippy
and rebuilt extension. Existing owner/legacy exclusions and known FFI warnings
remain. **204 checked / 64 open**, no new rows. Coordinated endpoint mutation,
reversal and merge planning remain the next core work.

## Endpoint revisions completed — 2026-10-27 UTC

Existing **04.2** is complete: stable-ID scalar/bulk/mixed endpoint changes and
ordered reversal retain properties, names, typed before/after history and explicit/
default clocks. Recorded/effective reads, indexes/caches, branch/older owned restore,
manual coalescing/reset, frozen sources and full durable replay/maintenance are
verified. Legacy membership-only groups explicitly reject unsupported rewiring;
no implicit source migration or new owner decision. Evidence/cost/limits:
`endpoint-revisions-2026-10-27.md`. General revert remains **13.3**; three-way
merge planning remains **14.2**.

**205 checked / 63 open**, same 268 rows. Gates: **1,335 native / 52 ignored**,
**2,022 root Python / 1 skipped**, **172 docs examples**, strict all-target Clippy,
rebuilt extension and serial **18-case** process-kill matrix. Existing owner/legacy
exclusions and known FFI warnings remain. No new rows, external installs or push.

### Explicit retained-state restoration (13.3 component) — 2026-10-27

Graph/DurableGraph and writers now restore an explicitly selected retained
published state, retargeting only the working/current-branch selection. Topology,
typed properties, external bindings and group revisions restore; future history,
other branches, allocation/name reservations, clocks and immutable pins remain.
Dirty/stale/foreign/collected/schema targets reject before mutation. Native detached
reconstruction runs off-owner/GIL, and durable reference records validate and replay
after sync with rollback/recovery/backup/maintenance coverage. No new mutation or
implicit history pruning. Same-state durable restores append no frame.

Existing **13.3 remains partial** for general tracked inverse mutations and their
recording/effective-time policy; this reference switch does not choose that policy.
**205 checked / 63 open**, same 268 rows. Gates: **1,342 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 docs examples**, strict Clippy, in-tree FFI
and serial 18-case recovery matrix. Local 100k-node restoration: **30.27 ms replay /
126.42 ms validation-install**; owner capture **2.54 µs**, no RSS/production claim.
Evidence: `state-restoration-2026-10-27.md`; API: `docs/api/state-restoration.md`.
Next core scope is caller-resolved three-way merges (14.2), with tracked revert
policy still explicit. No new checklist rows or owner-policy assumptions.

### Native three-way merge planner (14.2 component) — 2026-10-27

Native snapshot merge now combines independent node/edge/property/endpoint changes
and inventories typed conflicts with whole base/left/right values. Explicit caller
answers resolve entity/property/endpoints; recomputed dangling dependencies require
explicit edge deletion or exact missing-node retention. Parallel IDs remain distinct;
absence/Null, float types, signed zero and NaN payloads are preserved. Shared
structural validation replaces a public panic and duplicated integrity/restore checks.
Common-ancestor selection now returns all best bases and rejects ambiguous criss-cross
selection rather than silently choosing an older DFS match.

**14.2 is partial**, not complete: owning Graph capture/acceptance, full identity/
schema/source/time guards, one two-parent publication, durable replay and Python
surfaces remain. Evidence and fixed remainder: `snapshot-merge-planning-2026-10-27.md`.
**205 checked / 63 open**, same 268 rows. Gates: **1,351 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 Python docs examples** plus one executable
Rust merge example, strict Clippy, rebuilt FFI and serial 18-case recovery matrix.
Local 100k-node pure planning median **274.29 ms**, final validation **6.72 ms**;
no publication/FFI/RSS/production claim. No new checklist rows.

### Owner-bound merge capture/preflight (14.2 component) — 2026-10-27

Native Graph now captures owner/head/branch/base and two explicit retained parent
versions into a detached read plan. Typed caller resolution produces a privately
provenanced unpublished outcome. Read-only preflight rejects stale/foreign/changed
branches, collected parents/bases, unallocated IDs, missing group references,
conflicting external identity and optional full-target schema violations. Valid
compaction preserves logical results; explicit empty base handles unrelated roots.
Capture/resolution/validation do not allocate IDs/versions or change working state.

**14.2 remains partial**: exact first-parent delta and clock/timeline preparation,
current-policy revalidation, guarded one-version two-parent publication, durable
record/replay/failure evidence and Python APIs remain. There is no unchecked
finish_merge write API. Fixed evidence/remainder: `owner-merge-capture-2026-10-27.md`.
**205 checked / 63 open**, same 268 rows. Gates: **1,356 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 Python docs examples**, strict Clippy and
rebuilt/staged in-tree FFI. The preceding planner's 18-case recovery gate remains
the latest storage check; this component has no journal/storage change.

## Merge publication closure — 2026-10-27

Existing 14.2 is complete: exact first-parent deltas, explicit typed conflict
resolution, effective clocks/topology, guarded two-parent publication, Python
writer bridge and delta-only durable replay. Verification: 1,369 native tests,
2,034 root Python tests, 181 executable documentation examples, strict all-target
Clippy and 23 serial real process-kill scenarios across merge/maintenance.
See `merge-publication-exit-review-2026-10-27.md` for boundaries and benchmarks.
No new checklist rows. Inventory: **206 checked / 62 open**.

## Explicit restore modes completed — 2026-10-27

Owner approved one restore_state API with required mode="tracked" or mode="switch".
Tracked restoration records one exact current-parent delta with explicit/default
effective clocks; switch retargets the branch without publication. Graph,
DurableGraph and native/Python writers share the contract. Exact delta preparation
and checked snapshot publication are shared with merges.

Existing 13.3 is complete: native/Python compatibility, stale/dirty/schema/clock
guards, frozen sources, history/identity preservation, durable replay/rollback,
backup, explicit source-target pruning and compaction, plus five real SIGKILL
cases. Full verification: 1,376 native / 56 ignored; 2,040 root Python / 1 skipped;
182 executable docs; strict all-target Clippy; rebuilt in-tree FFI. The final
focused rebuilt restore/merge suite also passes (18 tests). Three existing FFI
warnings remain. No new checklist rows: **207 checked / 61 open**.

Local release medians for 100k-node chains with all properties changed: switch
reconstruction 30.9 ms + validation/install 132.5 ms; tracked replay/comparison
240.2 ms + validation/publication/install 206.8 ms. This measures full-state
restoration, not ordinary mutation overhead. Receipt:
`../benchmarks/state-restore-modes-2026-10-27.json`. Full replay/assembly costs,
RSS/production targets and broader lifetime/GIL work remain explicit limits.

Next priority is existing canonical state/change ownership (AR7.9), followed by
compound topology acceptance and remaining view/index/query boundaries.

## Canonical change-record closure — 2026-10-27

Existing AR7.9 is complete within its typed-record/representation/compatibility
scope. One native ChangeSet owner and snapshot difference builder serve recorded
diffs, merge and tracked restore. Exact value comparison also serves archive
validation and delta interning. Signed zero, NaN payloads, dtype and inactive-row
regressions are covered; owned payload buffers move into immutable deltas.

Physical offsets have an explicit borrowed IndexChangeView and complete indexed
DeltaObject projection. Fabricated index_N values and the silently empty adapter
are removed. Pool-independent logical extraction now rejects property offsets;
explicit retained-pool resolution preserves actual values and absence. Raw-space
delta extraction rejects because GraphSpace owns no pending changes. Native
fallibility/strategy compatibility changes are documented; persisted formats and
Python shapes are unchanged.

Verification: 1,383 native / 56 ignored; 2,042 root Python / 1 skipped; strict
all-target Clippy and rebuilt FFI. All **183 executable documentation examples** pass, including the new
change-record guide. Matched alternating-order local release medians:
100k property-free snapshot differences 7.05 → 5.31 ms; 10k nodes with 256-wide
f64 vectors and one change 12.14 → 8.20 ms. These are two fixture comparisons,
not whole-engine/RSS/production guarantees. Receipt:
`../benchmarks/canonical-state-changes-2026-10-27.json`.

Fixed inventory now **208 checked / 60 open**; no new rows. Remaining native
compound topology acceptance and coordinator compatibility guards stay in
09.4b.2/AR3; broader view/index/query/release work remains unchanged.

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

## Current topology-cache integrity — 2026-10-27 (33.2 component)

Explicit check_integrity now captures and verifies the current eligible GraphSpace
cache before archive reconstruction, which previously replaced it with fresh
derived data. Edge columns, active IDs/endpoints, isolate membership, neighbor
maps and CSR rows/offsets/edge IDs are checked with multiplicity. Cold/stale caches
are left untouched; invalid headers and poisoned locks return contextual errors.
The existing incidence factory is shared by cache construction and validation.
Sorting/comparison run in the detached phase; ordinary mutations do no extra work.

Four native regression groups cover valid empty/directed/undirected/parallel/loop/
isolate inputs, malformed columns/ranges/content/versions, detached captures,
stale/cold behavior and poisoning. Python checks cover warm caches, rejected
rewiring, successful rewiring and immutable old views. Final gates: 1,390 native /
56 ignored, 2,051 root Python / 1 skipped, 185 executable documentation examples,
strict all-target Clippy and rebuilt release FFI (three existing warnings).

Bounded release receipt: `../benchmarks/current-topology-integrity-2026-10-27.json`.
For the 100k-node one-version fixture, median cold capture/verify was 179/376 ms;
warm capture/verify was 177/422 ms. Warm verification additionally checks current
cached incidences. This measures explicit full retained integrity, not ordinary
write/query latency, RSS or a production SLA.

The current-cache component is complete. Existing 33.2 stays partial for secondary
index/other cache/service diagnostics, and AR2 still owns broader adjacency-policy
conformance. Receipt fields and storage formats are unchanged. Inventory remains
**208 checked / 60 open**, fixed 268 rows.

## Verified Python build profile and compatibility remainder — 2026-10-27

This continuation verified an in-tree CPython 3.9.6 release extension, built with
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=0 and an explicit PYO3_PYTHON executable.
On macOS, the direct cargo rustc build also supplied -undefined dynamic_lookup
linker arguments; staging used an atomic file replacement. No wheel/package
installation was performed. Three existing FFI compiler warnings remain.

Building from python-groggy/ with its forced ABI3-forward-compatibility setting
exposed a configuration conflict with the direct PyDate/PyDateTime/PyDelta imports
in ffi/utils/temporal.rs. The verified build disables that setting. Aligning the
marshaller/build configuration and testing the declared Python-version matrix
remains existing 19.2/34.2 work; forward-ABI3 and newer Python compatibility are
not certified by these test totals. No checklist row or owner-policy question is
added. All six implementation slices are committed; inventory is 208/60.

## Atomic pipeline write coordinates — 2026-10-27

The property overlay now captures the canonical owner/branch/head write coordinate
for standalone atomic builders and outer atomic pipelines. Acceptance validates
that complete coordinate even for an empty batch. A same-head branch switch or
foreign owner rejects before property publication; existing stale/dirty checks
remain. Failed outer runs discard staged properties and partial outputs, preserve
prior outputs and allow context reuse on a fresh invocation. No entity IDs are
allocated by rejection. The coordinate retains identity only.

Focused algorithm suite: **232 passed / 1 ignored**, including two new coordinate
regressions. Full native suite: **1,392 passed / 56 ignored**. This closes
a concrete coordinator gap within existing AR3/09.4b.2; the fixed checklist remains
**208 checked / 60 open**. Built-in pipeline algorithms write properties, not graph
topology; uncertified custom effects still reject explicit atomic execution.

## Python datetime marshalling in both ABI profiles — 2026-10-27

Fixed the configuration conflict recorded above. The configured limited-ABI build
previously failed on PyDate/PyDateTime/PyDelta imports (reproduced before this
change). An extension build script now forwards the actual PyO3 configuration and
platform linker arguments. The normal CPython ABI retains the existing direct C
marshaller; limited ABI uses cached Python standard-library types and base field
descriptors. Both share canonical native temporal validation and exact integer
components. Cached type identity rejects __class__ spoofing; subclass overrides
cannot substitute stored fields, and invalid astimezone return types reject.

No packages were installed. Offline locked release cargo builds succeeded with
forward compatibility set to both 1 and 0 on CPython 3.9.6, without manual linker
flags. Limited ABI: **2,053 Python tests passed / 1 skipped**, **185 documentation
examples**. Full ABI: the **17 temporal regressions** passed, including the two
new subclass/spoof cases; the normal direct path is preserved. Three pre-existing
FFI warnings remain. Wider declared Python versions/PyPy/platforms are unverified;
19.2/34.2 remain open. Inventory stays **208 checked / 60 open**.

## Contextual Null query literals — 2026-10-27 (21.1 / AR7.7 component)

The shared native graph parser now accepts case-insensitive null in value position,
without reserving property identifiers or changing quoted text. Explicit Null,
absence and Boolean complement retain the existing typed predicate semantics.
Complete-input validation still rejects malformed/trailing tokens. Two native
regressions and one rebuilt Python regression cover both entity kinds, case,
quoted text, a property named null, missing properties and rejection. This is a
small usable query completion within the existing inventory, not a new checklist
row or a claim that all query/parser migration is finished. Counts stay 208/60.

## Explicit transient secondary indexes — 2026-10-27 (18.2 / 18.3)

Owner A11.1 decision: keep indexes transient; rebuild explicitly. Implemented
GraphQueryIndex over exact immutable captures, named property equality and exact
numeric/temporal ranges, labels/relationship types and optional selected topology.
Missing requested indexes fail before Boolean execution unless scanning is explicit.
Existing scan APIs and durable external-ID reservations retain their contracts.

18.3 is complete with mutation/deactivation/branch/effective history/rejection/
owner closure/reopen/thread coverage. Native **1,399 passed / 56 ignored**, then
the extended journal restore test passed at four actual injected failure boundaries.
Rebuilt Python **2,069 passed / 1 skipped**, **190 documentation examples**, strict
core Clippy. Five alternating-order release samples and construction cost are
recorded in `captured-query-indexes-2026-10-27.md` and its linked benchmark JSON.

Fixed inventory: **209 checked / 59 open**, no rows added. 18.2 still owns
specialized group membership and temporal interval indexes plus broader query
integration. Ordinary immutable-Subgraph migration and compound topology execution
retain their existing scopes. Workbook answers, appendix and roadmap are updated.

## Ordered/paged captured queries and bounded traversal — 2026-10-27

Completed existing 21.1 components: native/Python scalar and ID ordering, exact
mixed numeric/tie/missing policies, overflow-safe pages and explicit ordering-index
availability; captured multi-source outgoing/incoming/both hop traversal with
complete seed validation and induced selected-edge closure. Finite property pages
validate all matches, partition the rank interval and sort the selected page.
Read/time pins and owner-independent threaded reads are preserved; no publication.

Evidence: `ordered-pinned-queries-2026-10-27.md`. Final **1,404 native / 56 ignored**,
**2,089 Python / 1 skipped**, **195 doc examples**, strict all-target core Clippy.
Independent exhaustive small-topology distance reference, exact ranking/page
fixtures and release checksums pass. The measured 100k shuffled-property page
improved from a local full-sort reference median 38.25 ms to 28.02 ms; all-match
validation/memory remains and no production bound is claimed.

Fixed inventory remains **209 checked / 59 open**, no rows added. 21.1 retains
arbitrary multi-edge patterns and wider ordinary Graph/Subgraph integration;
21.2/26.1, specialized 18.2 indexes and compound topology retain their stated
existing scopes. Updated the finish plan's stale next-work paragraph to reflect
already completed endpoint/restore/merge/native-boundary work.


## Finite captured graph patterns — 2026-10-27

Completed 21.1 with arbitrary finite named graph conjunctions, existing predicates,
all directions, explicit reuse/distinctness, parallel identities, assignment
budgets, deterministic pages and exact-bound pinned graph views. Shared endpoint
postings serve entity filters and pattern joins. Graph/Subgraph/DurableGraph offer
explicit capture/build entry points. Legacy connected query migration stays open.

Evidence: `graph-patterns-2026-10-27.md`; **1,410 native / 56 ignored**,
**2,103 Python / 1 skipped**, **199 docs**, strict all-target core Clippy. Analytic
release receipts cover 1k/10k/100k chain nodes. Existing 21.1 closes; no new rows.
Fixed inventory: **210 checked / 58 open**.


## Builder operand/effect-order repairs — 2026-10-27

Default compilation now preserves interleaved effects and subsequent reads for
algorithm/sampler builders. CSE keeps operand positions, limits reuse to known
value operations, and retains observable execution blocks in DCE. Reproduced wrong
results (7 instead of 2 after a write, 16 instead of 0 for reversed subtraction)
are fixed. Evidence: `builder-optimization-order-2026-10-27.md`.

Sixteen new cases; **2,119 root Python / 1 skipped**, **201 docs**; native baseline
unchanged at **1,410 / 56 ignored**, strict Clippy. Local 1k/10k alternating
optimization/reference receipts document the conservative fallback cost. No rows
added/closed: **210 checked / 58 open**. Broader typed effects/alias/lowering and
compound topology acceptance remain AR7.8/30.1/09.4b.2. The existing unfinished
builder power facade is documented under 30.1 without adding an audit row.


## Checked builder power — 2026-10-27

The preceding builder power gap is implemented for registered step pipelines:
explicit/reflected ** and CoreOps.pow share the existing checked native expression
kernel over node/edge maps and scalar operands, including exact integers, checked
real powers and atomic failure behavior. Receipt: `builder-power-2026-10-27.md`.
**1,413 native / 56 ignored**, **2,144 Python / 1 skipped**, **204 docs**, strict
all-target core Clippy. Bounded release receipts validate every result at 1k/100k
map rows. No rows added/closed; **210 checked / 58 open**. Other facades, alternate
plans, complete typed effects and compound topology remain in the existing scope.


## Checked elementary builder math — 2026-10-27

Completed existing abs/sqrt/exp/log and exact element-wise min/max facade methods
through shared native scalar kernels/comparison and existing map/atomic execution.
Legacy/IR lowering, selected scope, scalar fusion/broadcast, integer precision,
invalid domains/controls, cancellation and atomic rollback are covered.
`builder-elementary-math-2026-10-27.md`: **1,416 native / 56 ignored**, **2,212 Python
/ 1 skipped**, **207 docs**, strict all-target Clippy, bounded 1k/100k unary release
receipts. Initial matrix GIL worker-progress timing failure passed isolated/full
rechecks and is recorded under existing 29.26. No rows added/closed: **210 / 58**.
Other facades/effects/alternate plans and compound topology remain open.


## Tiny-divisor repair completed — 2026-10-28

Completed the interrupted native builder division fix: valid tiny/subnormal
nonzero divisors are accepted; exact signed zero still rejects. Node/edge and
Python lowering modes, atomic rollback and cross-language subnormal conversion
are verified. `builder-division-2026-10-28.md`: **1,417 native / 56 ignored**,
**2,236 Python / 1 skipped**, **208 docs**, strict all-target core Clippy. No rows
added/closed; **210 checked / 58 open**. Broader existing builder/effect and
compound-topology scopes remain unchanged.


## Effective-time event/activation indexes — 2026-10-28

Completed the existing 18.2 component with explicit immutable EffectiveTimeIndex
captures on Graph/DurableGraph: indexed global/entity event windows, structural
activation intervals and repeated canonical historical reconstruction. Explicit
recorded cutoffs, inactive IDs, retained source clocks, branch/head isolation and
transient rebuild policy are preserved. No new replay engine or automatic
mutation maintenance. `effective-time-indexes-2026-10-28.md`: **1,420 native /
56 ignored**, **2,250 root Python / 1 skipped**, **211 docs**, strict all-target
core Clippy, local 100/10k property-event release receipts. Whole-state reads
still replay the selected prefix; the larger narrow window measured 0.603 µs
indexed versus 7.346 µs scanned. Group-membership and attribute/fact validity
indexes remain within 18.2. No rows added/closed: **210 checked / 58 open**.


## Exact property-value intervals — 2026-10-28

Completed the next existing 18.2 component: EffectiveTimeIndex now returns exact
node/edge property-value and absence intervals inside active lifetimes, with
explicit recorded cutoffs, accepted-order ties and canonical retained-value
coalescing. Property/lifecycle postings skip unrelated history; no whole-graph
snapshot replay or automatic mutation maintenance. Stored boolean properties
express fact validity through the same API. `property-intervals-2026-10-28.md`:
**1,423 native / 56 ignored**, **2,274 root Python / 1 skipped**, **214 docs**,
strict all-target core Clippy and bounded 100/10k unrelated-event release receipts.
The larger fixture measured 0.173 µs indexed versus 4.128 µs scanned; build cost
was 5.549 ms. Group-membership indexes remain in 18.2. No new rows or premature
parent closure: **210 checked / 58 open**.


## Secondary-index family completed — 2026-10-28

GroupMembershipIndex completes the remaining 18.2 family: selected typed root
node/edge references, reachable nested revision links, source-qualified frozen
membership and deterministic direct/recursive reverse references. Canonical IDs
resolve local legacy aliases per source; plain integers do not imply references.
GraphView/Graph/DurableGraph/Subgraph expose explicit capture/build; rebuild/copy
preserve old captures and source/time pins. No implicit storage registration or
automatic maintenance. `secondary-indexes-exit-review-2026-10-28.md` reviews the
complete fixed index-family inventory and closes the existing 18.2 row.

Gates: **1,428 native / 56 ignored**, **2,285 root Python / 1 skipped**, **218 docs**,
strict all-target core Clippy. Native release 101/10,001-root fixtures compare
indexed reverse lookup against a borrowed selected-property scan, checking every
result. At 10,001 roots, medians were 0.203 µs for one direct hit, 644.510 µs for
the scan, and 0.779 ms to emit all 10,001 recursive references; build from an
already captured view cost 2.031 ms. No production/RSS/whole-owner-capture claim.

Fixed checklist moves **210 checked / 58 open → 211 checked / 57 open**, with no
new rows. Next core priorities retain their original scope: 09.4b.2 compound
execution/effects, 26.1/AR4/AR7.5 connected-view ownership, and 21.2 legacy pinned
query migration. Wider scale/storage/platform and service/Graphiti items remain.


## Canonical immutable query entry point — 2026-10-28

Completed an existing 21.2 component: native Graph.query no longer returns an
unimplemented placeholder. Graph/DurableGraph/Subgraph expose canonical GraphView
capture; GraphView.query_nodes/query_edges share the existing column predicates
and exact ordering/page kernel with explicit indexes. FFI options translation is
shared. No new query engine, automatic index work or changed legacy filter return
type. `query-capture-2026-10-28.md`: **1,431 native / 56 ignored**, **2,305 root
Python / 1 skipped**, **222 docs**, strict all-target core Clippy and checked
1k/100k native release receipts. At 100k nodes, current read capture costs 138.665
ms; ten-row pages measured 33.357 ms scanned versus 9.233 µs indexed, with a
separate 52.162 ms explicit index build. These expose existing materialization
costs, not a production latency claim.

21.2 remains partial with the legacy connected-view migration. Inspection also
identified owner-only selected-property checks in shared SubgraphOperations and
the Context overlay; that follow-up remains inside 26.1/09.4b.2, with no new row.
Fixed checklist remains **211 checked / 57 open**.


## Selected property read guards and writeback clarification — 2026-10-28

Shared connected property getters and Context/StepScope staged reads now enforce
selected membership and current liveness. Excluded parallel edge identities and
values cached by wider scopes cannot bypass the read boundary. No new state owner,
snapshot, mutation path or FFI business logic. `selected-property-scope-2026-10-28.md`
records native/manual/staged and Python coverage plus bounded 1k/10k scalar-read
release comparisons.

Full-suite review confirmed that native map attachment intentionally supports
active owner IDs outside an invocation selection for full-owner retained diffs.
That behavior is preserved, including the existing empty-invocation regression;
no attachment entries are dropped or regression tests weakened. Broader ownership
unification stays in 26.1/AR3. Python bulk accessor and pinned apply_view_batch
selection contracts remain unchanged.

Gates: **1,433 native / 56 ignored**, **2,309 root Python / 1 skipped**, **224 docs**,
strict all-target core Clippy. The query-capture and selected-read slices are
committed separately. No rows added/closed: **211 checked / 57 open**.


## Repeated immutable capture reuse — 2026-10-28

Completed the measured repeated-capture component of 21.2/AR7.5/32.2. A weak
GraphSpace cache shares frozen ReadData while readers hold it, with independent
capture identity and fresh branch/write coordinates/group catalogs. Selected
property/topology/mapping changes invalidate reuse, including dirty manual edits
without a topology-generation change. Existing independent-table-export rejection
is preserved. At 100k nodes the quiet release fixture measured **109.158 ns**
median shared capture, versus **93.910 ms** initial cold capture; mutation/release
then requires another materialization. See `current-read-sharing-2026-10-28.md`.

Gates: **1,441 native / 56 ignored**, strict all-target core Clippy, **2,314 root
Python / 1 skipped**, **226 docs**, full-ABI release extension rebuilt/staged.
No new checklist rows or broad parent closure: **211 checked / 57 open**.
First-capture allocation and legacy connected ownership remain priority work.


## Sparse canonical read adjacency — 2026-10-28

Canonical frozen reads now allocate only nonempty adjacency/incidence entries.
Active isolates/sinks return shared immutable empty rows and absent IDs remain
absent, preserving parallel/loop/direction/order semantics. Bounded incidence
reservations avoid dense construction growth. See
`pinned-read-adjacency-2026-10-28.md` for native endpoint-reference coverage and
two-process matched 100k-node isolate/sparse/chain/star release receipts. Cold
captures improved 12–25% on these fixtures; sparse peak RSS falls, chain peak RSS
is unchanged, and the outgoing-star three-row probe is 26.7% slower because sinks
need an extra membership check. This trade-off is explicit, not a universal
throughput claim.

Gates: **1,442 native / 56 ignored**, strict all-target core Clippy, **2,314 root
Python / 1 skipped**, **227 docs**, rebuilt/staged full-ABI release extension.
No parent closure/new rows: **211 checked / 57 open**. The next measured priority
is a whole ingest/query/neighborhood/effective-read/recovery workload at larger
sizes; connected ownership and compound topology acceptance remain open.


## Million-node core workflow baseline — 2026-10-28

Existing 32.1/32.2 now has a complete 1k/100k/1m directed-ring workflow receipt: one
bulk publication, cold canonical capture, explicit property index/page, bounded
neighborhood, later correction/effective reconstruction and retained archive
recovery after dropping the owner. See `core-workflow-scale-2026-10-28.md`. The
million case completes with 8.257 s bulk acceptance, 75.618 µs indexed ten-row
page, 1.951 µs three-hop neighborhood, 5.908 s effective reconstruction and 26.669 /
14.599 s archive save/recovery. Peak process RSS is 4,076.422 MiB and includes
every stage/allocator retention, not logical-heap bytes. Small payload/two-version/
low-degree fixtures do not establish production budgets.

32.2 changes from open to partial, with no new rows or checkbox closure:
**211 checked / 57 unchecked**. The measured follow-up is archive encoding/
validation allocation lifetime; preserve full recovery and corruption validation.
The new release benchmark passes all three sizes and strict all-target Clippy;
the unchanged core retains the preceding full native/Python/docs gates.


## Archive preparation allocation lifetime — 2026-10-28

Completed the measured 10.4/32.2 encoding component: one framed byte vector,
detached document released before complete round-trip validation, and prepared
buffer growth slack released. Format/header/checksum, full corruption validation
and synced no-clobber publication remain unchanged.
`single-buffer-archives-2026-10-28.md` records matched complete workflow receipts:
100k save 1.525 → 1.365 s with process peak 1,181.969 → 871.469 MiB; million save
24.731 → 20.674 s, but whole-process peak 4,062.031 → 4,282.734 MiB in one pair.
No million-scale peak-memory improvement is claimed; reconstruct/recover memory
needs stage-specific measurement and stays open.

Gates: **1,442 native / 56 ignored**, strict all-target Clippy, **2,314 root Python
/ 1 skipped**, **227 docs**, rebuilt/staged full-ABI extension. No new rows or
parent closure: **211 checked / 57 unchecked**. A measured next retrieval component
is paging directly from existing presence/equality postings instead of cloning
all matching IDs for a small ID-ordered page.


## Bounded pages from existing index postings — 2026-10-28

Completed the measured 21.2/32.2 allocation component: simple ID-ordered presence/
exact-equality queries borrow existing sorted postings and allocate only the
returned page. Equality bounds and ascending/descending overflow-safe ID paging
are shared with the ordinary index/scan paths. Missing indexes still reject unless
scanning is explicit; property ordering still validates every matching scalar even
for empty pages. No new query type/index/recovery/retention work.

`query-page-postings-2026-10-28.md`: two native scan-parity/validation cases and one
Python scope/lifetime/count case, plus two alternating release processes per
variant at 1k/100k/1m. Million-node forward exact pages measure **79.173 → 2.508 µs**;
presence pages **7,606.391 → 0.073 µs** by avoiding full-match materialization.
These are repeated native measurements excluding capture/index build/Python.

Gates: **1,444 native / 56 ignored**, strict all-target core Clippy, **2,315 root
Python / 1 skipped**, **227 docs**, rebuilt/staged full-ABI release extension.
Fixed inventory remains **211 checked / 57 unchecked**, no new rows. This hour
completes five committed slices (capture reuse, sparse frozen adjacency, million
workflow baseline, archive framing lifetime, direct posting pages). Functional
next priorities remain connected immutable ownership and compound topology;
large retained reconstruction/recovery memory remains explicitly open.

## Captured community kernels — 2026-10-29

Completed the functional 26.1/AR7.5 dependency: Louvain, Leiden, Infomap,
Girvan–Newman and label propagation now consume canonical captured input through
native AlgorithmRead/GraphView and GIL-releasing Python GraphView methods. Shared
kernels preserve legacy connected preparation, staged labels/weights and checked
writeback; read computations return owned rows and never publish. This completes
the immutable input surface for all thirteen classic registry algorithms.

Evidence/remaining boundaries: `captured-communities-2026-10-29.md`; executable
controls and examples: `docs/api/captured-community.md`. Five native parity/
lifecycle/domain cases and 79 Python cases pass, including worker transfer and
effective-time corrections across durable close/reopen. Ten release cases compare
connected, convenience and reusable captured preparation without claiming an
all-size speedup. No whole Graph, history or writable owner is cloned.

Gates: **1,449 native / 56 ignored**, strict all-target core Clippy,
**2,394 root Python / 1 skipped**, **229 executable doc examples**, rebuilt and
atomically staged full-ABI release extension. Python stubs and scoped formatting
are updated. Three existing unrelated FFI warnings remain.

No new checklist rows or broad parent closure: **211 checked / 57 unchecked**.
Connected default ownership/accessors (26.1/AR4), legacy filter result migration
(21.2) and compound topology/custom effects (09.4b.2) remain priority work.
Hierarchical Infomap and broad algorithm certification retain their existing scope.

## Captured membership algebra and grouping — 2026-10-29

Completed the next 26.1/AR7.5 read dependency: exact GraphView membership algebra,
node-set similarity, native overlap intersections and typed node/edge grouping.
Connected adapters reuse the same kernel and preserve their connected return type
and descriptive metadata. Fixed native dangling-edge subtraction and cross-owner
ID mixing, and exposed Python's three former set-operation placeholders.
`captured-view-algebra-2026-10-29.md` records semantics, tests, release preparation
costs and the fixed remaining ownership boundary.

Gates: **1,454 native / 54 ignored**, strict all-target core Clippy,
**2,405 root Python / 1 skipped**, **231 executable documentation examples**,
rebuilt and atomically staged full-ABI extension. The two fewer ignored examples
are old group-by documentation snippets replaced by the executable captured guide;
no running regression was removed. Six release fixtures report capture separately
from grouping/algebra; no universal speedup or memory claim. Three unrelated FFI
warnings remain. Stubs, scoped formatting, Python syntax and diff checks pass.

Inventory remains **211 checked / 57 unchecked**; no new rows or parent closure.
Next functional component: scope-correct captured/connected traversal trees and
hop paths. Connected BFS/DFS/shortest-path currently traverse the full owner before
filtering, which can leak routes through excluded intermediates. This belongs to
the existing selected ownership/algorithm contracts, not a new audit workstream.

## Scoped traversal trees, paths and restoration cache identity — 2026-10-29

Completed the next 26.1/AR7.5 selected-read component. GraphView provides BFS/DFS
discovery trees, reachability, shortest-hop subgraphs and empty/weak/strong
connectivity. One traversal kernel consumes captured adjacency or selected live
topology. Connected adapters no longer traverse excluded intermediates/instances
and post-filter afterward; induced children preserve parent edges, and component
fallback/cache uses exact selected CSR. Their connected ownership remains explicit.

Measured and fixed a compatibility regression before commit: connected BFS on a
100k chain changed from a provisional 37.500 ms repeated capture path to 3.375 µs
with topology-only reads and instance/version-cached selection validation. First
validation/preparation still costs 31.966 ms; whole-owner reference is 0.875 µs
and cannot implement selected scope. No universal performance claim. Restoration
can repeat numeric space versions, so scope/CSR/component cache keys now include
space identity; an equal-version topology-switch regression proves invalidation.
Evidence: `captured-view-traversal-2026-10-29.md` and its paired raw release receipt.

Gates: **1,460 native / 54 ignored**, strict all-target core Clippy,
**2,430 root Python / 1 skipped**, **234 executable documentation examples**,
rebuilt/atomically staged full-ABI extension, scoped formatting/syntax/diff checks.
Six new native cases include all 512 three-node directed graphs; 25 Python cases
exercise captured/connected APIs, cache restoration and scope. Existing owner
benchmark/review changes remain excluded. Three unrelated FFI warnings remain.

No new rows or broad parent closure: **211 checked / 57 unchecked**. Next read
parity component is native/captured structural summaries and the existing Python
clustering/transitivity placeholders; full connected accessors/default ownership
and compound topology/custom effects retain their original scope.

## Topology-only structural summaries and former Python placeholders — 2026-10-29

Completed the next 26.1/AR7.5/28.15 read parity component. Graph, connected
Subgraph and GraphView now expose actual native simple-undirected local/mean
clustering and transitivity; GraphView also returns bulk owned coefficient rows.
Selected topology excludes owner-only triangle edges, loops and parallel mass.
Directed clustering definitions reject explicitly, as existing triangle/k-core
steps do. Native inherent/trait/captured/Python density shares one direction-aware
edge-instance definition and avoids usize denominator multiplication overflow.

UndirectedStructure shares immutable CSR and sorted IDs for repeated native
computations, retaining no properties/history/writable owner/GraphView. Python
connected adapters prepare only topology then release the GIL. Builder triangles
and summaries use one checked cancellation-aware kernel. No fake partial graph or
full owner clone. Evidence, domains and raw release baseline:
`structural-summaries-2026-10-29.md`; all 64 four-node simple graphs match independent
closed-triple counts. Cold setup, chain/star/clique computations and cached
preparation are measured separately without a before/after speedup claim.

Gates: **1,466 native / 54 ignored**, strict all-target core Clippy,
**2,460 root Python / 1 skipped**, **237 executable documentation examples**,
rebuilt/atomically staged full-ABI extension, scoped Rust formatting and syntax/
diff checks. Four native integration/two unit cases and 30 Python cases added;
no existing regression weakened. Three unrelated FFI warnings remain.

Inventory remains **211 checked / 57 unchecked**; no new checkbox rows or broad
parent closure. Captured algebra/grouping, thirteen classic algorithms, scoped
traversal and structural summaries now cover the corresponding read prerequisites.
Connected column/accessor ownership and full read/write/default return migration
still require work in existing 26.1/AR4/AR7.5/21.2; topology/custom compound effects
remain in 09.4b.2.

## Explicit captured entity/column access — 2026-10-29

Completed the 26.1/AR7.5/21.2 explicit captured accessor component: view.nodes and
view.edges provide read-only scoped entities, stable ID lookup, explicit sorted
positions, exact selection/filter/slice/mask operations and aligned owned typed
columns with presence bits. Entities/iterators retain only the canonical view;
meta-node source revision lookup, scope-correct topology, source/worker lifetime
and explicit writeback are covered. ID order materializes lazily; scalar lookup
does not sort or allocate all IDs. Legacy connected defaults remain unchanged.

Matched four-process release measurements identified and improved large-column
projection using the existing ordered Rayon bulk path: 100k owned columns
25.006 → 5.430 ms, presence 19.638 → 6.461 ms. Native scalar receipts stay independent
of ID ordering; no 100ns Python FFI or general memory guarantee. Evidence and
fixed remainder: `captured-view-entities-2026-10-29.md`.

Gates: **1,472 native / 54 ignored**, strict all-target Clippy,
**2,478 root Python / 1 skipped**, **241 executable documentation examples**,
rebuilt/atomically staged full-ABI extension, scoped formatting, syntax and diff
checks. Six native cases include the parallel ordering boundary; 18 public Python
cases cover null/absence, meta/time/read-only/scope and owner lifetime. Three
unrelated FFI warnings remain. No new rows or parent closure: **211 / 57**.
Remaining window focus: native legacy column helpers still skip absent values;
repair alignment through the existing attribute-first bulk reader.

## Native compatibility column alignment — 2026-10-29

Completed the final window slice in existing 26.1/AR7.5. Native Subgraph complete
columns align to ascending node_ids/edge_ids; missing cells no longer disappear.
Explicit requested node sequences retain order and repetitions. Whole-request
scope/liveness checks preserve rejection, then the existing attribute-first bulk
reader resolves the storage column once. No new graph capture/publication path;
live ownership and Python connected accessor defaults retain their boundaries.
Evidence: `connected-column-alignment-2026-10-29.md`; four native cases and a 1k/
100k release comparison with equivalent aligned scalar reads. This fixes row
identity, without claiming a general throughput improvement.

Final gates: **1,476 native / 54 ignored**, strict all-target core Clippy,
**2,478 root Python / 1 skipped**, **241 executable documentation examples**,
rebuilt/atomically staged full-ABI extension, scoped formatting/syntax/diff checks.
The native-only guide links the compilable integration examples. Three unrelated
FFI warnings remain. Owner benchmark/review edits are preserved outside commits.

This window completes five committed functional slices: captured algebra/grouping,
scoped traversal/cache restoration, structural summaries, captured entity/column
access, and native compatibility column alignment. No new checkbox rows or broad
parent closures: **211 checked / 57 unchecked**. The execution checkpoint in
`roadmap-finish-priorities-2026-10-22.md` now distinguishes those completed read
prerequisites from actual connected/default/filter ownership, mutable/effectful
facades, specialized adapters and topology/custom compound acceptance. Cold/
invalidated full captured value materialization and retained recovery memory
remain the existing 10.4/32.2 performance boundary.

### Closing review: confirmed next legacy CSR liveness boundary

A throwaway Python reproduction after the final gates confirms an existing
26.1/AR7.5 inconsistency still needs the next slice. Create nodes 0/1 with edge
0→1, retain connected induced_subgraph([0,1]), warm selected.neighbors(0), then
remove node 1. The connected neighbors call still returns [1.0], and neighbors(1)
returns [0.0], while selected.pin_view() correctly rejects the stale selection.
The new captured entities and scoped traversal paths do not use this stale live
read contract and retain their documented behavior.

Cause to address: Subgraph::get_or_build_csr_internal checks versioned cache state
but its miss path reads retained pool endpoints without the canonical selected
liveness guard. Reuse validate_query_scope before CSR reuse/build, then check the
legacy degree adapter's separate path and verify removal/restore/cache cases.
The instance-sensitive validation/cache stamp needed for an inexpensive guarded
path is already implemented this window. This is a concrete reproduction under
the existing ownership item, not a new checkbox or completed repair. No code was
changed after the final verified alignment commit to conceal this remaining gap.

### Connected CSR liveness and degree blocker — 2026-10-29

Completed the closing reproduction above. Selected CSR readers now validate
canonical active membership and endpoint closure before both cache reuse and
rebuild. Warm/cold node/edge removal and rewiring reject, equal-version restored
spaces invalidate correctly, and frozen reads remain readable independently.
Native total degree now includes incoming directed instances, matching the
established incident-edge rule (loops once). Python's three separate degree edge
scans are replaced by thin translation to one immutable topology-only Rust input;
row computation releases the GIL. Explicit requests preserve ordering/repetition,
implicit rows sort by ID, invalid IDs reject whole requests, and undirected
in/out degrees use incident counts. Full-owner edge counting still requires a
valid original selection and does not admit owner-only result IDs.

Receipt: `connected-degree-reads-2026-10-29.md`; API guide and native/Python
regressions cover scope, direction, loops, parallels, empty/invalid requests,
prepared worker lifetime and absence of publication. The existing full-owner
native property-attachment contract is unchanged. No whole value/state capture
or new retention policy was introduced.

Release `connected_degree_reads` receipt in `notes/benchmarks/`: 100k-chain cold
preparation 21.532 ms; median warm preparation 0.500 microseconds; median counting
all 100k rows 2.699 ms; averaged warmed native neighbor/incident degree calls
0.074/0.041 microseconds. Five bulk/preparation samples and 10k scalar repetitions
are checksum-verified. Absolute fixture timings only, not before/after speedup,
Python FFI-budget certification or broader scale guarantees. Cold/full-owner
preparation and legacy default ownership retain their documented boundaries.

Final gates: **1,479 native / 54 ignored**, strict all-target core Clippy,
**2,497 root Python / 1 skipped**, **242 executed documentation examples**,
full-ABI release extension rebuilt/atomically staged. Restoration assertions were
extended and rerun after the full native suite; strict Clippy includes the new
benchmark. Scoped formatting/syntax/diff checks pass. Three existing unrelated
pipeline FFI warnings remain; owner benchmark/review edits remain excluded.

This completes the demonstrated compatibility blocker, not the broad ownership
migration. No new checklist rows or premature parent closures: **211 checked /
57 unchecked**. Actual default/filter immutable ownership and compound topology
acceptance remain the next existing priority scopes.

### Specialized selection ownership and topology — 2026-10-29

Completed the next existing 26.1/AR2/AR7.5 component. A pre-fix native regression
proved that filtered BFS traversed owner-only routes and returned nodes outside
its reachable selected topology. Filtered/component/neighborhood wrappers now own
one canonical Subgraph selection plus their specialized metadata. Their separate
membership/owner fields and duplicated traversal implementations are retired;
neighbor/degree, actual components, BFS/DFS, shortest hop paths and node/edge
selection use the same selected-scope validation and topology kernels.

Typed native result helpers move canonical membership directly into metadata
wrappers without copying output hash sets through trait objects. Canonical
connected induction/edge selection also stop capturing full owner property values:
induction scans selected edge instances once, and explicit edge selection resolves
only the selected requested instances. Parallel edges, loops and empty outputs
retain the base selection contract. Filter complement prunes dangling crossing
edges. Specialized clones share membership; Python neighborhood conversion shares
the canonical native selection and preserves its existing type label. All three
native wrappers expose explicit capture; component/neighborhood extension adapters
and stubs expose that same frozen path. No new graph/history representation.

Receipt: `specialized-selection-scope-2026-10-29.md`; native regressions cover all
three variants, both directedness modes, selected-route isolation, disconnected
components, complete parallel/loop induction, invalid/empty behavior, metadata,
shared clones, warm stale-scope rejection and frozen worker lifetime. The public
Python neighborhood bridge and extension registration are covered too. The prior
canonical traversal/degree/restoration suites remain passing.

Release raw receipt in `notes/benchmarks/specialized-selection-scope-2026-10-29.json`:
100k-chain median warm three-hop BFS 1.206–1.465 microseconds across base and all
three variants; four-node induction 6.494–7.333 ms (one selected-edge scan); selecting
three edges 2.958–3.541 microseconds. Small-criteria shared filter cloning averages
0.071 microseconds at 100k nodes. Current object struct sizes are 160/304/184/200
bytes for base/filtered/component/neighborhood, excluding heap allocations. Five
samples and output/membership checks; absolute warmed topology-only evidence,
not a universal speedup, heap-total or Python FFI-budget claim. First validation
and explicit immutable full-value capture keep their documented costs.

Final gates: **1,483 native / 54 ignored**, strict all-target core Clippy,
**2,500 root Python / 1 skipped**, **243 executable documentation examples**,
full-ABI release extension rebuilt/atomically staged. Scoped formatting, Python
syntax and diff checks pass. Three pre-existing unrelated pipeline FFI warnings
remain; owner benchmark/review changes are preserved outside the commit.

No new checklist rows or broad parent closure: **211 checked / 57 unchecked**.
This completes specialized membership/traversal consolidation and removes the
reproduced selected-route defect. Actual immutable default/filter returns,
effectful facades, the remaining viz adapter boundary and general compound
topology acceptance retain their existing priority scopes.

### Complete canonical operations through checked view writes — 2026-10-29

Completed the existing scoped-write component of 26.1/AR4. Six pre-fix Python
regressions reproduced escaped node/edge unsets, excluded-edge/new-endpoint
rewrites, stale unset values in refreshed views and lost surviving rewired edges.
Scope checks now include all prepared mutation targets, and projection follows
actual addition/revision/assignment/unset/removal order. Refreshed views retain
the immutable group catalog for meta-source access. Exact source comparisons use
canonical retained payload equality and normalize empty/absent property rows.
Exhaustive PreparedBatch matches now make future added fields a compile-time
review requirement at both scope and projection consumers. Identity targets remain
represented by their normalized property columns; clock fields add no targets.

Receipt: `view-batch-completeness-2026-10-29.md`. Native/Python evidence covers
whole-request rejection without allocation/publication, refreshed absence,
rewiring before node cascade, additions and identities, reverse/loop endpoints,
explicit effective states, next-write authorization, old immutable inputs,
retained meta-source workers, and durable byte-preserving rejection/recovery.
Unchanged NaN sources were already supported and remain regression-covered.

Release view_batch_publication baseline: a chain with one f64 scalar plus a
16-element f64 vector per node, two-node/one-edge write scope and original capture
retained through five sequential scalar writes. Quiet standalone medians: 2.303 ms
at 1k and 337.125 ms at 100k, including full refreshed payload construction.
Raw receipt: `notes/benchmarks/view-batch-publication-baseline-2026-10-29.json`.
This exposes the existing full-state output cost; there is no O(1) view-write claim.

Gates: **1,487 native / 54 ignored**, strict all-target core Clippy,
**2,511 root Python / 1 skipped**, **244 executed documentation examples**,
full-ABI release extension rebuilt/atomically staged, scoped formatting/syntax/
diff checks. Three unrelated existing FFI warnings remain; owner edits excluded.
Inventory remains **211 checked / 57 open**: immutable defaults and general
multi-stage topology execution are still their existing broader boundaries.

### Reuse accepted payloads for scoped publication — 2026-10-29

Completed a measured existing 26.1/AR4/AR7.5 and 10.4/32.2 component. Source checks
reuse current immutable data identity after the unchanged provenance/head guards;
distinct payloads still receive exact fallback comparison. Full output copying
moves after ordinary preparation and selected-scope validation. Graph/durable
success then shares one helper to seed the weak current-read cache with the
prebuilt accepted full payload. New captures retain fresh identity and current
branch/group metadata; no extra retention pin or new snapshot format.

Receipt: `view-write-payload-reuse-2026-10-29.md`. New native cases verify payload
sharing independent of selected membership, immutable original values, ordinary
mutation invalidation, forced same-head maintenance fallback, fresh catalogs,
durable adoption/reopen and weak lifetime/distinct capture identity. All ten
existing raw/corrupt pinned-write boundary cases remain passing.

Same release fixture as `9b364219`, three alternating standalone process pairs,
15 samples per size/variant: medians **1.464→0.517 ms at 1k** and **310.443→76.283 ms
at 100k**, 2.83×/4.07× on this workload. Whole-fixture peak RSS in one extra run per
variant is 594.281/589.359 MiB; construction is included, so no significant peak
memory improvement or isolated allocation claim. Raw paired data is in
`notes/benchmarks/view-write-payload-reuse-2026-10-29.json`.

Full output copying/index construction still scales with owner payload/topology.
No O(1), production memory/throughput, durable-fsync or Python latency guarantee.
Gates: **1,492 native / 54 ignored**, strict all-target core Clippy,
**2,511 root Python / 1 skipped**, **244 executed documentation examples**, rebuilt/
atomically staged extension and scoped formatting/syntax/diff checks. Three old
unrelated FFI warnings remain; owner edits stay excluded. No new checklist rows
or broad ownership/default/compound closure: **211 checked / 57 unchecked**.


### Current connected full-view cache repair — 2026-10-29

Completed another existing 26.1/AR4/AR7.5 component. Seven reproduced failures
showed that aggregate entity counts reused stale full-view membership after
same-size replacements/trades and state restoration. GraphSpace now supplies an
instance-sensitive membership revision across all eight scalar/bulk mutation
paths. Property and endpoint-only writes preserve cached membership; separate
read caches still refresh values/topology. Restoration distinguishes equal
numeric versions. The Python bridge no longer allocates whole ID vectors for a
warm cache check. Explicit immutable capture and legacy return types are unchanged.

Receipt: `current-full-view-cache-2026-10-29.md`. Quiet release-extension warm-hit
medians were **2,445→40 ns at 1k** and **431,955→41 ns at 100k**. These are bound
Python method calls in nonalternating local runs; initial view construction and
immutable capture are excluded, so no universal FFI budget claim.

Gates: **1,494 native / 54 ignored**, strict all-target core Clippy,
**2,519 root Python / 1 skipped**, **245 executed documentation examples**, rebuilt/
atomically staged release extension and scoped formatting/syntax/diff checks.
Three unrelated existing FFI warnings remain; owner edits excluded.
No rows added or broad parents closed: **211 checked / 57 open**. Next remains
immutable default/effectful facade migration and compound topology acceptance;
full refreshed-output copying also remains linear.


### Complete integer view sampling — 2026-10-29

Completed an existing 26.1/AR4/AR7.5 operation: graph and connected integer sampling
formerly discarded the sampled IDs and returned the entire input. Native connected
and captured methods now share the checked reservoir kernel with pipeline masks,
induce only selected instances, cap counts and preserve frozen source/time/group/
write pins. Collection failures propagate; sampler handles release on failed runs.
Measured zero-size shortcuts remove sorting/induction after live scope validation.
Read defaults and full immutable sampler/effectful pipeline migration remain open.

Receipt: `view-node-sampling-2026-10-29.md`. At 100k, paired local native zero-size
medians are **2.352 ms→0.208 µs connected**, **1.308 ms→0.083 µs captured**; the
baseline is the correct implementation before only the zero shortcut. Sixteen-node
samples measure **4.609 ms / 1.428 ms** with setup/capture excluded. No universal
latency/RSS claim. Gates: **1,498 native / 54 ignored**, strict all-target Clippy,
**2,537 root Python / 1 skipped**, **246 docs**, rebuilt/staged release extension,
scoped format/syntax/diff checks. Three existing FFI warnings remain. Pending next-
slice column-indexing tests and owner benchmark work excluded from this slice gate.
No rows added or broad parents closed: **211 checked / 57 open**.


### Whole-selection Python column facade — 2026-10-29

Completed another existing 26.1/AR4/AR7.5 component. Subgraph["property"] now uses
the checked native bulk reader rather than a per-row FFI scalar loop that hid
errors as nulls. Selected node/edge properties and adapters share ascending IDs;
GraphView indexing and to_nodes/to_edges reuse frozen readonly accessors. Three
pre-fix cases reproduced the old boundary; eight final cases cover row/absence/
error/type/lifetime behavior. Broader live defaults/effectful facades remain open.

Receipt: `view-column-indexing-2026-10-29.md`. Release before/after medians are
**0.103→0.113 ms at 1k**, **32.798→23.702 ms at 100k** for complete f64 columns;
separate nonalternating local builds, no universal performance/RSS claim.
Gates: **2,545 root Python / 1 skipped**, **247 docs**, rebuilt/staged extension,
syntax/diff checks. Native core unchanged from **1,498 / 54 ignored** full suite
and strict all-target Clippy. Three existing unrelated FFI warnings remain;
owner benchmark/review work untouched. **211 checked / 57 open**, no rows added.


### Selected adjacency export — 2026-10-29

Completed the existing 26.1/AR2/AR4/AR7.5 adjacency facade. Graph's placeholder and
captured missing method now export full selected rows. Connected export replaces
per-node full-edge scans, wrong directed reverse edges and swallowed read errors
with validated CSR input. Detached AdjacencyRead stores shared IDs/CSR only, permits
owner-free workers and supports GIL release. Stable sorted neighbor IDs preserve
parallel/loop occurrences and isolated rows; excluded edges stay excluded.

Receipt: `view-adjacency-export-2026-10-29.md`. Actual warmed Python before/after:
**14.906→0.120 ms at 1k**, **382.161→0.778 ms at 5k**; new 100k export **29.443 ms**
including dictionary/list conversion, with no extrapolated old 100k runtime. Native
100k cold/warm prep is **24.495 ms / 2.208 µs**, detached map export **2.152 ms**;
captured CSR preparation **17.522 ms**, initial owner capture separate. Nonpaired
local receipts, no production/RSS/universal latency claim.

Gates: **1,501 native / 54 ignored**, strict all-target core Clippy,
**2,554 root Python / 1 skipped**, **248 docs**, rebuilt/staged release extension,
scoped format/syntax/diff checks. Three unrelated old FFI warnings remain and owner
work is untouched. Live defaults/effectful migration stays open; **211 checked /
57 open**, no new rows.


### Connected matrix scope consistency — 2026-10-29

Completed another existing 26.1/AR2/AR4 reader component. Four pre-fix cases showed
dense/sparse connected exports silently losing inactive selected rows or edges.
Canonical scope validation now rejects stale selections and shared ascending ID
slices replace two filtered temporary vectors. Aliases inherit the guard; valid
inside rewiring remains live, while captured/previous owned matrices remain frozen.

Receipt: `connected-matrix-scope-2026-10-29.md`. Quiet warmed sparse export before/
after medians: **0.358→0.347 ms at 1k**, **46.012→40.699 ms at 100k**; separate local
builds, no universal memory/latency claim. Gates: **1,503 native / 54 ignored**,
strict all-target core Clippy, **2,559 root Python / 1 skipped**, **249 docs**,
rebuilt/staged release extension, scoped formatting/syntax/diff checks. Three
existing unrelated FFI warnings remain; owner changes remain untouched.
**211 checked / 57 open**, no new rows or broad parent closure.

### Current block and fixed next boundary

This block committed integer sampling, whole-selection column/index adapters,
adjacency export and matrix scope consistency. These finish actual operations
within 26.1/AR2/AR4/AR7.5; the numerical solver work is unchanged. Remaining work
includes binding effectful apply/sampler/collapse facades to captured reads and
explicit checked owner writes, then migrating default returns. Legacy Python
selection conversion and visualization adapters also retain their original scope:
Subgraph.to_graph still constructs an empty graph placeholder, and its viz getter
still constructs a full-owner GraphDataSource. They require selected/captured
conversion rather than silent scope widening. These are existing facade gaps,
not new checklist rows; no claim that all remaining facade methods are audited.


### Selected graph materialization — 2026-10-29

Completed existing 26.1/AR4/AR7.5 conversion: connected and captured to_graph now
preserve selected IDs/direction/typed properties and referenced frozen groups in
an independent publishing graph. Out-of-scope local references reject; full source
history/configuration recovery remains explicit owned capture. Six pre-fix Python
cases reproduced the old empty/missing converter. Native group-source archive
recovery, stale selection, empty copy and independent publication are verified.
Receipt: `view-materialization-2026-10-29.md`. Absolute release copy medians:
**2.987 ms / 595.559 ms at 1k/100k**, including initial import publication, source
capture separate; no speedup against the incorrect placeholder or RSS claim.
Gates: **1,506 native / 54 ignored**, strict all-target Clippy, **2,565 root Python /
1 skipped**, **250 docs**, rebuilt/staged extension and scoped checks. Three old
unrelated FFI warnings remain. **211 checked / 57 open**, no new rows. Visualization
and frozen defaults/checked effectful binding remain the same parent scope.


### Selected immutable visualization — 2026-10-29

Completed existing 26.1/AR4/AR7.5 visualization adapter scope. Two pre-fix real
server cases reproduced full-parent leakage/missing captured .viz. Both now use
GraphDataSource.from_view and the existing checked entity policy; selected IDs,
parallel/loop instances, direction, typed properties/positions/weights and snapshot
lifetime remain intact. Captured preparation releases the GIL. Real isolated HTTP/
websocket snapshot tests and native worker/source-identity cases pass. Receipt:
`view-visualization-2026-10-29.md`. Sixteen-node source medians **11.125/11.292 µs**
from 1k/100k parents after capture; full-source medians **0.668/150.908 ms**, no
server/RSS/universal speed claim. Gates: **1,508 native / 54 ignored**, strict
Clippy, **2,567 root Python / 1 skipped**, **251 docs**, rebuilt/staged release
extension and scoped checks. **211 checked / 57 open**; main remaining boundary is
frozen default read binding and explicit checked execution/writeback.


### Default immutable ownership completed — 2026-10-30

Closed existing **26.1 / AR4 / AR7.5** with default frozen selections, preserved
specialized derivations, lazy component/retained meta sources, checked explicit
effects and weak canonical capture caches. No checklist rows added: **214 checked /
54 open / 268 total**. Verification: **1,514 native / 54 ignored**, strict Clippy,
**2,574 Python / 1 skipped**, **255 executable docs**. Warm held defaults measure
66–68 ns at 1k/100k; cold 100k capture remains 147.879 ms. Full scope and limits:
`default-immutable-views-2026-10-30.md`. Next existing priorities are compound
topology/effect acceptance and legacy query migration.


## 2026-11-05 — pinned query closure and atomic collection collapse

Closed existing 21.2 after the immutable default/filter migration exit review.
Completed the collection-collapse component of existing 09.4b.2/AR3: all plans
prepare against one source and publish once; late failures preserve topology,
properties, retained groups and IDs. Both ComponentsArray and SubgraphArray use
the native declarative boundary. Destructive overlap/cross-plan rewiring rejects
explicitly. See `pinned-query-exit-review-2026-11-05.md` and
`compound-collection-collapse-2026-11-05.md` for full gates, compatibility and raw
release receipts. The 500-component fixture improves 249.147 → 19.219 ms median
wall time while reducing 500 acceptances to one. Generic topology pipelines and
typed effects remain open. No rows added: **215 checked / 53 open / 268 total**.


## 2026-11-05 — checked native lazy scalar arrays

Completed a scalar execution component within existing 29.3/30.1/AR7.7/AR7.8.
Random filtering and semantic operation reordering are replaced by a fully
validated, statement-ordered plan with compiled queries and private seeded
sampling. Unsupported graph/type-changing operations reject explicitly. Array
and graph languages share lexer/boolean parsing while preserving intentional
numeric equality and nonfinite-literal policies. Full native/Python/docs gates
pass. Million-row i64 chain: 21.453 ms / 45 requested allocations / 8,002,473 peak
additional live bytes; direct eager reference 0.776 ms. Interpreter throughput
remains an explicit existing numerical/execution gap. Receipt:
`lazy-scalar-arrays-2026-11-05.md`. No new rows: **215 checked / 53 open / 268 total**.


## 2026-11-05 — captured collection projection and materialization

Closed four demonstrated current-owner bypasses in existing 26.1/AR4/AR7.5:
SubgraphArray table columns, observations, merge and visualization. Canonical
source reads preserve sorted IDs, null/absence, direction, exact parallel edges
and retained groups. Native merge_all validates one capture and unions once;
materialization delegates to checked owned-copy code. Summaries use native
density and collection errors propagate. Full gates: 1,532 native / 54 ignored,
strict Clippy, 2,591 Python / 1 skipped and 257 docs. Large-group table projection
improves 35.594 → 11.132 ms; 100-small-group observations regress 14.071 → 39.848 ms
and retain an explicit existing performance gap. See
`subgraph-array-projection-2026-11-05.md`. No new rows: **215 checked / 53 open / 268 total**.


## 2026-11-05 — measured typed numeric predicate lowering

Completed a numerical/execution component within existing 29.6/29.26/AR7.8.
Typed numeric plans dispatch once per primitive column and preserve the generic
policy for heterogeneous nonnumeric cells. Threshold normalization uses the
existing Exact conversion policy; source values/literal identity never change.
Differential bit checks cover f32/f64 edge cases and wide integers. Full gates:
1,533 native / 54 ignored, strict Clippy, 2,591 Python / 1 skipped, 257 docs.
Million-row i64 chain improves 21.453 → 8.497 ms, AttrValue 24.265 → 14.672 ms and
f64 filter 26.159 → 14.787 ms; peak requested bytes remain stable. Direct eager
reference remains 0.742 ms, so production/compiled-throughput gaps stay open.
Receipt: `typed-scalar-lowering-2026-11-05.md`. No rows added: **215 / 53 / 268**.


## 2026-11-05 — actual captured component neighborhoods; window handoff

Completed the component neighborhood convenience path in existing
26.1/AR2/30.1/AR7.8. Both wrappers now return actual one-hop captured selections
from the lowest node ID, delegate traversal to Rust, release the GIL and propagate
errors. They no longer discard computed results and return whole components.
Original source values/edge instances survive owner changes; empty/isolated and
explicit native direction/depth controls are covered. Full final gates:
**1,535 native / 54 ignored**, strict Clippy, **2,593 maintained root Python /
1 skipped**, **258 executable docs**. Receipt:
`component-neighborhood-defaults-2026-11-05.md`.

This work window closes existing 21.2 and completes five implementation slices;
no rows were added. **215 checked / 53 open / 268 total**. Remaining priorities:
compound topology and typed graph/table pipeline execution (09.4b.2/AR3/30.1/AR7.8),
measured small-group captured projection costs (32.2/AR7.5), and wider numerical/
conformance/scale/release gates. Adapter/retrieval/service deferrals are unchanged.


## 2026-11-11 — bulk captured collection projection

Completed the measured many-small-group regression within existing 32.2/AR7.5.
Native comparison isolates scheduling: unchanged readers take 32.599 ms across
100 groups sequentially versus 4.727 ms across existing workers; sorting is
1.064 ms. One GIL-free native collection call now preserves frozen sources,
input order, sorted entity rows, explicit null/absence and missing-column errors.
No persistent column/membership cache was added. Quiet Python release medians
for 100k nodes/100 groups improve tables 49.426 → 9.837 ms and observations
36.039 → 6.282 ms; single-group changes are small and mixed. Full native gate:
1,537 passed / 54 ignored. Maintained Python: 2,594 passed / 1 skipped; 258 docs.
Receipt: `bulk-captured-projection-2026-11-11.md`. Strict all-target Clippy and
whitespace checks passed. Broad scale/pipeline parents remain open:
**215 checked / 53 open / 268 total**, no rows added.


## 2026-11-11 — captured incident-edge composition planning

Frozen collapse/contraction reuse the validated source's existing incident rows,
retaining stable edge ordering, distinct parallel IDs, exact internal traversal
and one mixed acceptance. Connected compatibility retains its scan. Exhaustive
small-selection differential coverage includes direction, endpoints, tombstones,
loops, all external strategies and meta selectors. Native 2,000-component median
improves 192.406 → 40.904 ms; Python 203.355 → 47.797 ms. These are separate
fixtures, not an FFI-overhead comparison or a linear-total-scaling claim.
Receipt: `captured-incident-composition-2026-11-11.md`.
Full gates: 1,538 native / 54 ignored; strict all-target Clippy; 2,594 maintained
Python / 1 skipped; 258 executable docs. Three existing FFI warnings remain.
Generic topology intermediates and typed pipeline work remain open:
**215 checked / 53 open / 268 total**, no rows added.


## 2026-11-11 — typed immutable graph/table pipeline inventory

Completed nine native read primitives within 30.1/AR7.8: explicit retained
snapshot → graph, node/edge selection, node/edge tables, selected rows/columns,
and explicit graph/table outputs. Typed variables preserve captured data, IDs,
null/absence alignment, exact values and table provenance. Generic aliasing and
shared output-scope completion preserve representation and roll back local/outer
atomic outputs without deep-copying prior buffers. A thin step_pipeline factory
uses the existing BuiltAlgorithm/interpreter; naming policy/IR behavior is not
redefined. Native 100k snapshot-to-table workflow measures 67.908 ms; retaining
one million previous output rows does not copy its buffer. Receipt and exact
remaining domains: `typed-pipeline-values-2026-11-11.md`.
Full gates: **1,543 native / 54 ignored**, strict all-target Clippy, **2,611
maintained Python / 1 skipped**, **260 executable docs**. Rebuilt in-tree release
extension retains three existing FFI warnings. Current staged/topology graph
intermediates, remaining DSL/IR lowering, naming/cache/streaming remain open.
**215 checked / 53 open / 268 total**, no checklist rows added.


## 2026-11-11 — native display ownership, compatibility and rendering safety

Completed the native ownership/caller migration component of AR7.6/AR7.11.
Typed windows/schema, configuration, renderers, themes and existing CSS now have
one owner under display::render; core/viz display paths remain type-identical
aliases. Native storage/graph callers use the owner directly. Fixed reproduced
UTF-8 JSON clipping and signed-minimum HTML failures plus zero-budget underflow.
Shared bounded clipping uses an ASCII prefix fast path; old positive-budget ASCII
layouts retain all 12 corpus fingerprints. Native prefix work for one million
characters improves 142.943 → 0.128 μs in a same-binary comparison; whole rendering
still includes full tooltip escaping. Full render timings are mixed and preserved
in the receipts, including quiet repeats. No universal throughput claim.
Receipt: `display-foundation-2026-11-11.md`.

Final gates: **1,547 native / 54 ignored**, strict all-target Clippy, rebuilt release
in-tree extension, **2,612 maintained Python / 1 skipped**, **261 executable docs**.
Three existing FFI compile warnings remain. The Python gate also emitted a
PyGraph unsendable-finalization warning (15 total warnings instead of the usual
14); this is recorded under existing 19.2 without attributing it to the test where
GC reported it. It remains unresolved; no unsafe Send workaround was introduced.

Historical Python formatter adapters/shared scalar presentation policy and wider
compatibility/performance gates remain in AR7.6. Current/staged topology pipeline
intermediates, naming/cache policy and broader scale/numerical/release gates
retain their prior scope. **215 checked / 53 open / 268 total**; no rows added.
The status overview now distinguishes its earlier 209/59 record from this current
inventory, and the workbook appendix records the implemented typed read stages
without filling the unanswered A14.1 naming decision.


## November 12 — ownership diagnosis and atomic terminal removals

- 19.2: a minimal isolated cyclic-GC diagnostic reproduces wrong-thread PyGraph
  finalization without algorithms. Owner-thread Graph collection and worker
  collection of detached views pass. The full-suite allocation path and safe
  writable lifetime design remain open (`graph-thread-lifetime-2026-11-12.md`).
- 09.4b.2/AR3: local atomic builders combine property calculations with multiple
  node/edge removals in one canonical acceptance; strict controls, map-key targets,
  cascade times, rollback and exact surviving output selections are complete.
  Generic staged topology reads/additions/other effects remain open.
  Receipt: `terminal-topology-pipeline-2026-11-12.md`.
- Verification: full native 1,551 passed/54 ignored, final focused native 5 passed,
  strict all-target Clippy, rebuilt in-tree release, maintained Python 2,633
  passed/1 skipped/15 warnings, 263 documentation examples. Ownership warning
  and three existing FFI build warnings remain. Measured native removal medians
  0.362 ms (1k nodes) and 33.164 ms (100k); selected membership scans remain linear.

**215 checked / 53 open / 268 total**, no rows added. Next core execution work:
explicit immutable staged graph/table values after topology, then broader
addition/revision effects and outer execution contracts. Naming/cache policy,
scale promises and deferred Graphiti/service work retain their prior boundaries.


### November 12 follow-on — selected pending graph/table reads completed

`graph.capture_staged` captures explicit immutable selected values before/after
pending properties and removals. Existing graph/table primitives and aliases can
read them after topology staging; native effect classification rejects unadapted
calculations. One canonical acceptance/output rollback boundary is retained.
Captures keep typed values, exact identities, meta-group sources and worker-safe
lifetimes, but have no accepted-state write pin. Capture-only runs add no history.

This is selected-data materialization, not a lazy/constant-time overlay. The same
release binary measures removal-only medians 0.426/38.466 ms (1k/100k nodes),
versus 1.177/193.629 ms when also capturing a pending graph, projecting its node
table and emitting both. No implicit capture is introduced on ordinary mutations.
Full scopes, raw samples and limitations: `staged-graph-capture-2026-11-12.md`.

Final checks: **1,555 Rust passed / 54 ignored**, final targeted native **13 passed**,
strict all-target Clippy, rebuilt/staged release, **2,645 maintained Python passed /
1 skipped / 15 warnings**, **264 executable docs**. The known writable graph GC
warning and three existing FFI compile warnings remain. Naming/cache workbook
question A14.1 stays unanswered; no policy inferred.

Next core execution scope: addition/endpoint-revision stages, ordinary algorithms
over staged topology and wider nested/outer effect execution; broader DSL,
streaming, scale and release contracts remain. **215 checked / 53 open / 268 total**;
no rows added and no broad parent checked off for this bounded component.

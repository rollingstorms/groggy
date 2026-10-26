# Priority slices — 2026-10-26

Branch: `groggy-roadmap-2026-10-20`. After completing 16.2, the owner authorized up
to 90 minutes of priority slices, aiming to complete existing tasks and document
remaining work. The follow-on window began 03:49 UTC. This record consolidates
its outcomes and the remaining core priorities. No new checklist rows were added.

Current inventory after core retention closure: **204 checked / 64
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

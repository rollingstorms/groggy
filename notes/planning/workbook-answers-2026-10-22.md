# Owner workbook answers — 2026-10-22

Source: the owner's dictated answers to the open-questions appendix and subsequent
explicit clarification replies in this chat. Accepted directions authorize future
implementation; they do not claim that the code already implements them. Only
answered questions change status. Unanswered or qualified choices remain visible.

## Answer mapping

| Question | Owner answer / clarification | Recorded direction and status |
| --- | --- | --- |
| A01.1 | Subnode changes and meta-node changes are distributed; confirmed new reference per revision. | **Confirmed:** target subnode and meta properties independently. An explicit membership/source-pin revision receives a new immutable reference; older references remain readable. Do not infer automatic propagation to every containing group. |
| A01.2 | Suggested. | **Confirmed:** copy/materialize the selected state by default; transfer history only explicitly. |
| A02.1 | Defer checkpoint schedule until MVP/first use; question a maximum budget; benchmark opening. | **Confirmed:** measure recovery/open time now; defer scheduling thresholds and mandatory log/disk limits. Do not invent an opening SLA. |
| A02.2 | Suggested; asks whether checkpoint is compressed state. | **Confirmed:** verified side-by-side format conversion, retaining original until verification. A checkpoint is saved state/recovery information; compression is optional. |
| A03.1 | Yes, migrate defaults now. | **Confirmed (2026-10-23):** checked automatic publication becomes the default for ordinary Graph mutations now; commit means an explicit checkpoint. Publication errors must remain visible and a failed batch must publish nothing. |
| A03.2 | Check examples, but change everything now. | **Confirmed:** audit real repository examples as compatibility fixtures and proceed with the requested terminology/API migration now. A03.1 now separately confirms automatic-publication defaults. |
| A04.1 | Only explicit changes/effective time; confirmed later accepted write wins equal-time conflicts. | **Confirmed:** mutation events carry effective time; reads/computation create no changes. At identical effective time for the same key, later accepted version wins. Explicit recorded cutoff can reveal the earlier accepted value. |
| A04.2 | Reject it. | **Confirmed, overrides suggestion:** reject an effective edge whose endpoint validity does not support it; do not silently retain/hide invalid topology as the default. Backdated combinations need whole-proposed-timeline validation. |
| A05.1 | Remain absent. | **Confirmed:** unset persists until a later set; no fallback to an older value. |
| A05.2 | Restore unless explicit. | **Confirmed correction (2026-10-23):** restoration requires an explicit retained state; do not select a pre-deactivation state automatically. |
| A06.1 | Suggested. | **Confirmed:** first-class UTC instant, calendar date and duration first; distinct versioned types. Local/timezone-aware calendar types need their own precise rules later. |
| A06.2 | Change default; asks whether f32 was for memory. | **Confirmed:** ordinary Python floats default to f64. This intentionally supersedes the proposed preserve-f32 default. Preserve existing typed data during storage migration; do not pretend widening recovers already-rounded bits. |
| A07.1 | “You could, I guess.” | **Tentative:** no definite choice between branch schema identity and global installed policy. Keep the current policy until clarified. |
| A07.2 | Okay. | **Confirmed:** explicit compatible migration stages, each atomic and separately recorded. |
| A09.1 | No. | **Proposal rejected:** do not infer extra permanent named-checkpoint/audit retention roots. **Confirmed (2026-10-23):** purge is explicit and must preserve shared branches and open immutable views. |
| A09.2 | No. | **Proposal rejected:** the combined grace-period/backup/audit proposal is not accepted. **Confirmed (2026-10-23):** no mandatory grace period or backup; explicit purge still preserves shared branches and open immutable views. This is currently a data science library policy, not a production database retention guarantee. |
| A10.1 | Conditional yes if joining duplicates things. | **Conditional interpretation:** repeated table rows can map to the same source entity; a join does not itself create graph entities. Reject ambiguous/conflicting writeback without explicit target mapping/reduction. No silent last-row-wins update. |
| A10.2 | One batch. | **Confirmed:** one mixed atomic writeback for unambiguously identified node and edge updates. |
| A11.2 | Fail. | **Confirmed, overrides suggestion:** fail when an index required by the selected query execution is unavailable; scanning must be explicit, not an automatic fallback. This does not make today's ordinary scan API require a nonexistent optional index. |
| A12.1 | Okay. | **Confirmed:** preserve documented compatibility default; expose explicit weak/strong component modes. This does not accept unrelated provisional self-loop degree changes. |
| A12.2 | Yes. | **Confirmed:** one deterministic optimum by default; equal-optimum enumeration is a separate bounded operation. |
| A13.2 | Change “S64/X64” default; confirmed f64 floating point. | **Confirmed:** f64 floating-point defaults/support; this is not an integer-width change. Keep real error/convergence domains and residual/tolerance evidence. Existing numerics already largely use f64; property storage/FFI still require migration. |
| A14.2 | Correct. | **Confirmed:** pipeline runs are transient by default; persistence/provenance is explicit. |
| A15.1 | Include compute time and wall time. | **Confirmed measurement addition:** report elapsed computation and end-to-end wall time; label process CPU separately when collected. Specific graph/vector/history/hardware fixtures and budgets remain unspecified. |
| A16.1 | Explicitly. | **Recorded interpretation:** reject duplicate assignments by default; identical coalescing is an explicit option and conflicting assignments still fail. No implicit retry-key policy is chosen. |
| A17.1 | Real use later; use a test example. | **Confirmed sequencing:** use a synthetic documented integration fixture initially; a real Graphiti compatibility release/use workload is not yet selected. Integration stays deferred behind core work; do not claim a hand-made fixture proves upstream compatibility. |

## Explicit clarification replies

- “S64/X64” means **f64 floating point**; ordinary Python floats should default to f64.
- For two explicit writes at the same effective timestamp, **later accepted write wins**.
- Membership/source-pin group revisions use **new immutable references**, retaining the old reference.

- Ordinary mutations adopt checked automatic publication now; commit is an explicit checkpoint.
- Entity restoration requires an explicit retained state.
- Explicit purge preserves shared branches and open immutable views, without extra permanent checkpoint/audit pins or mandatory grace/backup. Scope is currently a data science library.

## Questions not answered by the original message

A08.1–A08.2 (rewiring/merge),
A11.1 (index recovery versus rebuild), A13.1 (shape/broadcast policy), A14.1
(stage naming), A15.2 (service limits), A16.2 (durable retry key), and A17.2
(adapter correction/fact identity). A07.1 remains tentative. A15.1/A17.1 supply
measurement/fixture direction, not complete production targets. A03.1, A05.2 and A09 were clarified on 2026-10-23. A question
being unanswered in this message does not undo earlier accepted behavior; existing
exact-shape operations and other established contracts remain in force.

## Answers to the owner's explanatory questions

### Is a checkpoint a compressed state?

A logical checkpoint identifies a saved state. A physical recovery checkpoint can
materialize that state plus recovery metadata and a journal position, so opening
starts there and replays later changes. Compression is an independent encoding
choice. It need not discard retained history. Current DurableGraph opening still
replays the full journal; the existing full-log backup is not a compact checkpoint.
See `docs/api/durable-storage.md` and A02.

### Was f32 chosen to save memory?

Current `AttrValue::Float`/`FloatVec` store f32. Comments in `src/types.rs` emphasize
embeddings/coordinates and cache-friendly vector storage. Python conversion in
`python-groggy/src/ffi/utils/helpers.rs` narrows incoming floats and float lists.
This supports a compact-storage motivation; an original documented scalar-default
rationale has not been established. f32 numeric elements use 4 bytes versus 8 for
f64. Vector payloads therefore double when widened; total graph memory need not
double because containers, IDs, indexes and enum layouts have other costs.
Historical f32 values must remain faithfully tagged/readable; lost precision
cannot be reconstructed simply by widening them. Future defaults and explicitly
typed storage are distinct from rewriting retained data.

### Does a join duplicate nodes in the graph?

A join can repeat one source node/edge across several table rows. It does not
itself allocate graph entities. Writing multiple differing assignments back to
that one identity needs explicit mapping/reduction; accepted mixed updates are
one atomic batch. This is the condition used to interpret A10.1.

## Implementation impact

- Retained references can proceed with per-revision immutable identity and explicit
  selected-state copies; preserve distributed mutation targets and historical pins.
- Temporal replay now has an accepted equal-time ordering and endpoint rejection
  rule. Validate full proposed timelines atomically, including backdated writes.
- f64 property/default migration and native temporal tags must update the typed
  value bridge, schema, codecs, history/journal and numerical/query adapters.
  These are requested changes, not already completed by f64 matrix support.
- Mixed table writeback, index-failure policy and transient pipelines have clearer
  contracts; source identity/expected-state checks remain required.
- Checkpoint scheduling stays deferred. Measure actual opening cost before setting
  thresholds. No production maximum or SLA is inferred from a small baseline.
- Unanswered merge/rebuild/shape/service/retry/adapter choices remain open. No
  implementation or destructive retention policy is selected by blank fields.

## Initial opening-time measurement

A bounded warm-cache baseline now exists in
[`notes/benchmarks/journal-open-development-2026-10-22.md`](../benchmarks/journal-open-development-2026-10-22.md).
With one live node, median open wall time is approximately 0.130 / 5.684 / 56.260 ms
for 1 / 101 / 1,001 accepted batches. Seven samples per case; process CPU is
recorded separately. The current in-tree development extension replays the whole
journal. This is not release/cold-device/large-graph performance or a production
budget. Checkpoint scheduling and maximum-log limits remain deferred as requested.

## A06.1 implementation — 2026-10-23

07.3b now implements distinct native UTC instant, calendar date and fixed elapsed
duration properties. Rust/Python share exact microseconds, calendar years 1..9999
and the full Python timedelta domain. Aware datetimes normalize to UTC; naive
datetimes reject. Versioned payloads retain values through history, export and
checked journal recovery. See [implementation and verification](temporal-properties-2026-10-23.md).
Local/zoned calendar types and temporal arithmetic remain separate future scope.

## A03.1 checkpoint boundary — 2026-10-23

12.3a now supplies explicit checkpoints of clean published states, including
writer sessions and durable recovery. Clean `commit()` labels that state without
a duplicate mutation delta. Dirty legacy `commit()` retains a publication
compatibility path. 09.4a adds native fallible checked constructors; switching
ordinary Rust/Python defaults and auditing pooled/meta/compound mutations remains
09.4b under the existing authorization. See [implementation record](explicit-checkpoints-2026-10-23.md).

## A03.1 ordinary Python defaults — 2026-10-23

09.4b.1 switches ordinary Python creation/mutation defaults to checked publication.
Bulk calls publish once, ordinary time defaults use the acceptance clock, and
explicit writer/batch effective timestamps remain available. `commit()` records
an explicit checkpoint. `Graph(manual=True)` names the legacy deferred path.
Remaining native infallible/default compatibility and multi-stage operation audits
are 09.4b.2, already authorized. See [implementation record](automatic-publication-2026-10-23.md).

## 2026-10-23 — atomic compound meta/collapse publication (09.4b.2a)

Modern plans and legacy collapse routes now stage node aggregations, retained
membership, boundary edges and optional removals before accepting one version.
Rejected calls consume no entity IDs or stored group contents. Directed boundary
orientation, deterministic aggregation, parallel copying, strict/default handling
and standalone staged meta-edge helpers are covered. Manual mode remains deferred.
See `atomic-meta-composition-2026-10-23.md` and the API example for evidence/limits.
Native compatibility, algorithm/table boundaries and portable pool state remain
open; 09.4b.2b records real contraction/custom identity/explicit-selector gaps.

## 2026-10-23 — explicit mixed table writeback (09.4b.2c / 27.2a)

Implements A10's one-batch update direction with `GraphTable.update_graph`:
explicit one-to-one row-ID mappings, selected property columns, expected-state
and mapped-endpoint checks, and one checked native publication. Ambiguous join
rows, invalid/inactive targets and late property errors reject before mutation.
Null is explicit, omitted columns stay unchanged, and clock defaults/explicit
times are shared by the batch. Source graph/pin propagation is not inferred;
27.1, connected table views, legacy materialization, native compatibility and
whole-pipeline boundaries remain open. See `table-writeback-2026-10-23.md`.

## 2026-10-23 — whole-input subgraph write validation (09.4b.2d / 05.3a)

Native subgraph node bulk/column setters now preserve all supplied rows for the
owner validator. Inactive/unknown/duplicate targets reject the complete write,
including both algorithm attachment representations. Native all-edge convenience
writes stage one column, producing one accepted version with no per-edge partial
publication. Manual compatibility and valid empty operations keep their existing
behavior. See `subgraph-write-validation-2026-10-23.md`; broader native migration,
pipeline boundaries and legacy table materialization remain open.

## 2026-10-24 — checked table materialization (09.4b.2e / 27.2b)

Table-to-graph conversion now prepares all rows/properties in one native batch,
returns one clean publishing state, remaps row IDs/reference arrays and registers
managed external IDs. Invalid/dangling rows, kind errors, identity conflicts and
missing pooled/meta contents reject instead of logging losses or creating base
fallbacks. Null edge-ID assignment is checked and shared with the explicit helper.
Direction is selectable with the undirected default retained. Portable pool/history
recovery and connected source pins remain open. See `table-materialization-2026-10-24.md`.

## 2026-10-24 — nonpersistent builder attachment (09.4b.2f / 30.1a)

Native node/edge attachment and smoothing now honor persist=False and expose
owned typed output columns without changing graph properties, pending mutations
or versions. Canonical target/kind validation and per-attachment cancellation
remain active. Both map/column routes use one native boundary; the FFI only
translates results. Persistent multi-step atomicity and the accepted default
migration remain open. See `transient-builder-outputs-2026-10-24.md`.

## 2026-10-24 — atomic property builder boundary (09.4b.2g / 30.1b)

Explicit build(atomic=True) stages supported property writes, serves later loads
from staged values, and publishes one checked mixed batch after success. Ordinary
errors/cancellation discard the buffer; stale/dirty heads reject; empty writes
publish nothing. Unsupported steps reject before execution. Atomic builds preserve
original order by bypassing the current effect-unaware IR optimizer. This adds no
whole graph clone or FFI graph logic. Existing defaults and broader step/pipeline
migration remain open. See `atomic-builder-properties-2026-10-24.md`.

### Atomic nested property loops — 2026-10-24

Implemented 09.4b.2h: recursively certified loop descendants share the outer
property staging context, see earlier iteration writes and accept once. A later
error discards all iterations; empty/zero-iteration bodies remain checked and
cancellation is checked between child steps and iterations. Alternate batch plans
reject until their executor honors staging. This follows existing workbook
authority; remaining step families and automatic atomic defaults stay open.
Details: `atomic-builder-loops-2026-10-24.md`.

### Atomic fused kernels and message-pass blocks — 2026-10-24

Implemented 09.4b.2i under the existing staged property contract. Certified fused
kernels and ordered supported message-pass blocks compose with nested loops and
accept once. Block options now honor the JSON payload, direction uses selected
CSR, and arithmetic overflow/zero division return errors. Streaming/unordered/
placeholder-where blocks reject before atomic execution. Legacy where condition
semantics remain an explicit 30.1/AR7.8 repair; effect-aware optimization, alternate
batch plans, remaining families and atomic defaults stay open. Verification:
940 native tests, 1,112 root Python tests, 39 doc examples and strict Clippy;
four release ring workload receipts. See `atomic-builder-blocks-2026-10-24.md`.

### Conditional message-pass execution — 2026-10-24

Implemented 09.4b.2j: compare and three-input where now honor the existing builder
comparison/mask contract inside ordered message-pass blocks. Malformed arities
and operators reject before execution; selected-value errors discard the outer
atomic run. This supersedes the previous placeholder-where gap. Existing numeric
tolerance is preserved; synchronous/unordered execution, remaining step families,
effect-aware optimization and atomic defaults remain parent work. Details and
verification: `atomic-builder-conditionals-2026-10-24.md`.

### Synchronous message passing — 2026-10-24

Implemented 09.4b.2k: ordered=False now evaluates every node against fixed block
input and installs buffered outputs after success. This is serial Jacobi, distinct
from ordered Gauss-Seidel, and works in ordinary and atomic builders. Types, column
order, selected topology and outer staging remain consistent. Later-node failures
leave source variables intact and atomic runs discard prefix writes. Synchronous
block semantics are no longer open; streaming and remaining unsupported families,
optimizer effects and defaults remain parent work. Details and verification:
`synchronous-message-pass-2026-10-24.md`.

### Atomic degree and incident-weight steps — 2026-10-24

Implemented 09.4b.2l: node_degree, weighted_degree and edge_weight_sum now compose
with atomic staging and nested loops. Weighted paths share one overlay-aware
validated kernel; malformed/nonfinite values and sum overflow error. Degree
retains its directional/loop convention, uses one edge pass and rejects invalid
source variables. Weighted total incidence retains double self-loop contribution.
Triangle/k-core certification and remaining execution-boundary parent work stay
open. Details, compatibility changes and evidence:
`atomic-structural-builder-steps-2026-10-24.md`.

### Undirected triangle and k-core steps — 2026-10-24

Implemented 09.4b.2m: distinct-node triangle counting ignores loop/parallel-edge
artifacts; queue-based k-core peeling retains the established multiedge/once-per-
loop degree convention. Both support atomic staging, scoped IDs, cancellation and
empty graphs. Negative k rejects. Directed calls explicitly reject until separate
definitions are implemented; prior directed behavior was inconsistent. This
supersedes the undirected triangle/k-core migration gap, while directed support
and remaining execution-boundary parent work remain open. Details and evidence:
`atomic-triangle-core-steps-2026-10-24.md`.

### Atomic property filtering and ordering — 2026-10-24

Implemented 09.4b.2n: node/edge predicate filtering, property sorting and top-k now
support staged reads and atomic loops. Predicate payload decoding and range
validation are repaired; read errors propagate. Sorting uses exact numeric order,
ID ties, null/absent-last behavior and explicit incomparable-domain/NaN rejection.
Sparse masks and rank-map outputs are preserved. Statistical/neighbor aggregation
review remains next; other execution-boundary parent work stays open. Details,
compatibility notes and evidence: `atomic-filtering-builder-steps-2026-10-24.md`.

### Combined aggregation builder bundle — 2026-10-24

Owner requested approximately 3x larger slices. This batch completes 09.4b.2o–2q
together: shared checked statistical reductions and public scalar methods; stable
mode/entropy and owned histogram vectors; all five checked standalone neighbor
reducers with direction/weight lowering parity. Histogram bin/node identity aliasing
and an invalid optimizer output-scaling rewrite are repaired. Histogram-containing
programs retain ordered observable outputs until optimizer effects support them.
Message-pass weighted/non-sum requests reject explicitly; broader execution parents
remain partial. Verified with 953 native tests, 1,312 root Python tests, 48 doc
examples, strict Clippy and 28 release workloads. Details and compatibility changes:
`atomic-aggregation-builder-bundle-2026-10-24.md`.

### Combined expression/transformation bundle — 2026-10-24

Completed 09.4b.2r–2t together at the requested larger cadence: checked expression
operators/literals and Python parsing parity; staged property reads and private
async source overlays; deterministic selected neighbor label updates and checked
sparse copies. Native callback errors propagate, but opaque callbacks remain
uncertified. Preserve the public ExprContext shape and the distinct incidence
versus unique-neighbor APIs. Scalar maps/copies avoid edge scans; async maps copy
only their source column. Remaining community/sampling/temporal/path certification,
optimizer effects, native escapes/default compatibility and outer transaction
parents stay open. See `atomic-transformation-builder-bundle-2026-10-24.md` and
`../../docs/api/builder-transformations.md` for details, compatibility changes,
costs and verification evidence.

### Combined mask sampling bundle — 2026-10-24

Completed 09.4b.2u–2w together: node/edge count/fraction masks and map-key reservoir
sampling now support property-atomic builders and nested loops. Stable ID order,
local seeded RNGs, strict controls, selected identity/endpoint validation and
cancellation replace nondeterministic shared RNG state and unchecked paths.
Reservoir capacity is capped to available keys, including oversized k; zero/null
source values remain eligible. Existing sparse attachments do not clear previous
unsampled rows. Subgraph-producing sampling and whole sampler transactions remain
open, alongside community/temporal/path audits and broader execution parents.
See `atomic-sampling-builder-bundle-2026-10-24.md` and
`../../docs/api/builder-sampling.md` for contracts, costs and verification.

### Combined community helper bundle — 2026-10-24

Completed 09.4b.2x–2z: checked deterministic singleton/degree/random seeds;
synchronous selected label propagation with map/column parity; actual local
modularity move gains replacing the old edge-fraction proxy. Directed and
undirected objectives handle stored loops/parallel edges, strict integer partitions
and neighboring candidate communities; independent exhaustive fixtures recompute
the full objective after moves. Typed outputs, cancellation and atomic staging
compose through existing attachment/load paths. Broader execution, composer and
remaining sampling/temporal/path audits stay open. See
`atomic-community-builder-bundle-2026-10-24.md` and
`../../docs/api/builder-community.md` for contracts, costs and verification.

### Bounded subgraph sampler bundle — 2026-10-24

Completed 09.4b.2aa–2ac: ordered selected node/edge iteration; invocation-bounded
outgoing/undirected neighborhoods with direct map/array seed parity; consistent
induced emission across source types/modes. Selected edge omissions, parallel
edges and loops remain respected; foreign owners and malformed selectors reject.
Results are legacy owner-connected selectors, not pinned value snapshots. Nested
for-each stays uncertified; its already remaining child-effect/output audit is now
explicit 09.4b.2ad, including first-result truncation and explicit-empty handling.
This is an existing parent-work decomposition, not new scope. Verified with 969
native tests, 1,516 Python tests, 64 doc examples, strict Clippy and 12 release
workloads. See `bounded-subgraph-sampler-bundle-2026-10-24.md` and
`../../docs/api/builder-subgraph-sampling.md`.

### Checked nested sampler execution bundle — 2026-10-24

Completed the explicit open 09.4b.2ad and related 09.4b.2ae–2af: recursive child
capability checks, shared outer property staging, complete ordered child-array
flattening with explicit-empty semantics, validated item outputs and private owned
frames. Transient entity columns merge by ID across items/steps and prior rows;
non-column payloads retain per-item keys. Failure restores prior owned outputs
and target arrays while the outer atomic algorithm discards pending properties.
Standalone sampler transactions and pinned views remain separate open parents.
Verified with 975 native tests, 1,532 Python tests, 68 doc examples and strict
Clippy; release receipt covers 12 workloads. See
`atomic-foreach-sampler-bundle-2026-10-24.md` and
`../../docs/api/builder-foreach.md`.

### Checked path builder bundle — 2026-10-24

Completed 09.4b.2ag–2ai: shared checked selected/staged-weight topology for BFS/
Dijkstra maps, repaired loopless Yen node paths and isolated seeded per-edge walks.
Root exclusions, full f64 candidate ordering/deduplication, parallel minima versus
walk multiplicity, reverse traversal, missing/null distinctions and strict input/
overflow/cancellation handling are explicit. Native DFS references enumerate small
graph paths independently. Temporal steps remain next; broader execution/native
compatibility/source-pin and standalone sampler parents stay open. See
`atomic-path-builder-bundle-2026-10-24.md` and
`../../docs/api/builder-paths.md` for contracts, costs and gate/receipt evidence.


## Checked temporal builder continuation — 2026-10-24

Completed **09.4b.2aj–2al**: checked finite node/edge decay, ordered node-column
parity, and retained immutable snapshot selection with strict temporal factory
parameters. Existing helper implementations now have builder registry entries.
Decay uses explicit elapsed time and creates deltas only through attachments;
loading earlier staged properties follows the existing overlay. Retained snapshots
exclude those pending writes and do not create checkpoints or change the working
graph. Snapshot timestamps keep the existing whole-history acceptance epoch-second
selector (state-ID tie-break), not arbitrary effective-time reconstruction.

Other temporal steps remain uncertified: diff snapshot/source identity and
branch semantics, index provenance/lifecycle intervals, and window-value history
resolution need follow-up. Existing workbook decisions remain in force; no new
owner choice or approval gate is introduced. See
`checked-temporal-builder-bundle-2026-10-24.md` and
`../../docs/api/builder-temporal.md` for contracts, costs and verification.


## Retained diff completion and progress accounting — 2026-10-24

Completed the existing node/edge temporal diff family under **30.1 / AR7.8**:
real logical property records, stable added/removed/modified masks, immutable
snapshot provenance and checked atomic output construction. Full-owner retained
comparisons allow reverse/sibling states within one history; foreign history
instances and unrepresented endpoint rewiring reject explicitly. No new policy
question or checkbox row was added. The fixed eight-primitive temporal inventory
now has four complete/four remaining, reduced from two complete/six remaining.

The flat header's stale 158 count was corrected. The 189 checked/79 unchecked
inventory overlaps broad parents, children, deferred scope and architecture work;
it is not a burn-down count. Audit repairs now update existing requirements, with
new capabilities explicitly identified as scope additions. Full parent exit
criteria stay in force. See `roadmap-status-2026-10-24.md`,
`retained-temporal-diffs-2026-10-24.md` and
`../../docs/api/builder-temporal-diffs.md`.


## Temporal primitive family complete — 2026-10-24

Finished the four remaining existing helpers under **30.1 / AR7.8**:
window_aggregate, filter, mark_changed_nodes and window. They share owner/head-
checked first-parent retained indexes, actual sparse state records and activation
intervals. Aggregation counts property-change events, preserves literal text/null
and uses shared checked numeric reductions; lifecycle windows exclude deletion
and respect restoration gaps; change-type masks use exact delta membership.
Supplied stale/foreign/unscoped indexes reject rather than silently refresh.
Absent indexes remain ephemeral local read structures, with no cache/variable
side effect. Existing native signatures remain and explicit window bounds expose
these helpers through Python specs; no FFI business logic was added.

The fixed temporal inventory is now **8 complete / 0 remaining**, with no new
checklist rows. The broad pipeline, global index lifecycle and arbitrary effective-
time reconstruction parents are not redefined or falsely closed. Verified with
1,000 native tests, 1,695 Python tests, 81 documentation examples, strict Clippy
and 12 release workloads. See `temporal-window-completion-2026-10-24.md` and
`../../docs/api/builder-temporal-windows.md`.


## Composer semantics complete — 2026-10-24

Closed existing **09.4b.2b**, without adding checklist rows: preset defaults now
precede explicit overrides, domain kinds are labels on canonical meta nodes, and
explicit existing-meta selectors are available in Rust/Python. `contract_all`
implements actual external connectivity through the selected interior. The owner
chose **one edge per reachable external endpoint pair**. Attribute reduction uses
each distinct original corridor edge once, in edge-ID order; this is a documented
implementation contract, not a promise of flow-equivalent capacity. The summary
meta-node is isolated in contraction mode.

The checklist is **190 checked / 78 unchecked**, with overlapping parents still
open. Full gates: 1,006 native tests, 1,716 Python tests, 84 documentation examples
and strict Clippy. Ten bounded release workloads expose repeated edge scanning
during bulk source removal; that measured performance repair follows under
existing bulk/performance requirements. See `composer-completion-2026-10-24.md`
and `../../docs/api/composer-options.md`.


## Checked pinned-view write slice — 2026-10-25

Closed original **26.2**, reducing the inventory to **191 checked / 77 unchecked**
without adding a checklist row. Current GraphView captures carry source history,
branch and accepted-head provenance through clones/selections/filters. Graph and
DurableGraph apply_view_batch prepare mixed writes against that pin, require every
existing mutation target/endpoint and deletion cascade to stay selected, then
publish one state and return a new immutable view. Returned data is constructed
and validated before publication. Earlier views keep their original values and
edge scope; no excluded parallel edges are induced into the result.

Schemas, managed identities and explicit/default change times use the existing
coordinator. Durable writes enforce installed plus per-call schemas and use the
existing append/sync/publication boundary. Recovery replays ordinary batch records;
old process pins cannot write to a reopened store. Dirty/standalone/unrecorded
snapshot captures are read-only; foreign, different-branch, stale and out-of-scope
requests reject before mutation/append. Raw untracked changes after capture reject;
native escapes made before capture remain outside the supported contract and
09.4b.2 remains open.

Ten native regressions (including 80 chained overlay/oracle comparisons), sixteen
Python cases, updated generated stubs, three executable examples and six release
workloads pass. Full gates: 1,030 native, 1,746 Python, 91 documentation examples,
strict Clippy and in-tree FFI build. The explicit refreshed-snapshot path owns full
current values/topology and is not an O(1) scalar writer. See
`pinned-view-writes-2026-10-25.md` and `../../docs/api/pinned-view-writes.md`.
Legacy connected Subgraph/table reads, table source provenance, portable group
contents and arbitrary effective-time reconstruction keep their original scope.


## Frozen graph-table source slice — 2026-10-25

Advanced existing **27.1 / 27.2 / AR7.10**. GraphView owned exports now carry shared
frozen source pins and immutable row-to-entity IDs through the native table
foundation, including the thread-transferable projection path used by Python.
Selections, filters, sorting and column projections preserve exact row identity
even after visible ID columns are dropped. Five generic joins keep source pins
but require explicit identity resolution before automatic source writeback.

BaseTable chosen-column and GraphTable mixed-column updates use the existing
checked view coordinator for Graph/DurableGraph, retain schema/time behavior, and
publish one delta. Forged structural IDs/endpoints, duplicate rows, wrong kinds,
independently captured bundles and stale/foreign/read-only sources reject. New
from_base_table adapters preserve provenance. Imports remain unbound; explicit
mapped writeback and independent materialization remain available. Empty slices
keep columns; repeated drop indices remove rows once.

Ten native regressions, thirteen Python cases, three executable examples and
release observations at 1k/10k nodes pass. Full baseline: 1,040 native, 1,759 Python,
94 documentation examples and strict Clippy. Source pins retain the full frozen
payload; selections share that allocation and own row-ID maps. See
`table-source-pins-2026-10-25.md` and `../../docs/api/table-source-pins.md`.
This finishes the explicit frozen export/base-row/generic-join/source-write family.
Specialized domain merges, TableArray adapters, broad join-key conformance and
legacy connected reads keep the existing parents partial; no new checkbox rows
were added and the inventory remains **191 checked / 77 unchecked**.


## 2026-10-25 — generic table join conformance (27 / AR7.10)

Closed the five-family generic join engine inventory using one Rust typed hash
planner and columnar materializer. Single/composite keys follow property identity;
null matches null, storage variants match and f32/f64 remain distinct. Stable
outer ordering and coalesced right-only shared keys replace arbitrary hash order
and lost keys. Generated-name collisions now reject rather than overwrite data.
Existing family naming and right-join swapped-left behavior remain explicit.
Python's existing multi-key syntax now executes and releases the GIL over native
work. Native TableArray positional joins delegate all five families. Frozen
source contexts remain immutable and joined rows require explicit write identity.

Evidence: `table-join-conformance-2026-10-25.md`, seven native regressions including
300 independent nested-loop comparisons, eight Python cases, 1,047 native /
1,767 root Python tests, 97 executable docs and strict all-target Clippy. Release
native receipt: `notes/benchmarks/table-joins-2026-10-25.json`; no matched speedup
or memory guarantee. Checklist remains 191 checked / 77 unchecked: no new rows.
The generic join family is complete; broad 27/AR7.10 remain partial for specialized
merges, grouping/set/conversion semantics and TableArray aggregation adapters.


## 2026-10-25 — stable typed table row operations (27 / AR7.10)

Closed the generic grouping/dedup/set/stack/concat inventory. One private Rust
row-key mechanism now serves joins, both group APIs, grouped aggregates and set
operations. Groups and distinct rows keep first-occurrence order. Canonical
property identity replaces debug/delimited strings. Group/dedup selections retain
exact source IDs; combinations/aggregates keep read-only source contexts.
Concat now preserves column order, pads shorter inputs, and rejects colliding
names. Stack preserves all rows with identical ordered schemas. Both native
combination methods now have thin Python adapters; affected BaseTable row/group
calls release the GIL. Grouped aggregation rejects output collisions and invalid
empty-input specs, and aggregation columns have deterministic order.

Evidence: `table-row-conformance-2026-10-25.md`, seven native regressions including
432 independent set/stack oracle comparisons, ten Python cases, 1,054 native /
1,777 root Python tests, 100 executable docs and strict all-target Clippy. Receipt:
`notes/benchmarks/table-rows-2026-10-25.json`; no matched speedup or memory guarantee.
Checklist remains 191 checked / 77 unchecked with no added rows. Specialized
GraphTable domain merges, TableArray aggregate/group-key policies and broader
conversion adapters remain open; generic joins and this row family are complete.


## 2026-10-25 — TableArray reductions and supplied-table flattening (27 / AR7.10)

Closed the existing collection reduction/group-key validation/flatten inventory.
A Rust coordinator now emits each key once, validates constant/consistent group
metadata, uses sorted explicit specs and checks collisions/missing/unknown specs.
Grouped collections emit one row per group; native/core ungrouped collections
reduce all observations, with correctly weighted means. Legacy Python constructor
per-table behavior remains; its columns= projection no longer acts as grouping.
Both Python wrappers and specialized node/edge adapters delegate numeric/logical
reductions to Rust with the GIL released. Edge group metadata now survives.
Flattening retains later-only columns, pads missing values, preserves zero-column
row counts and retains supplied source captures without guessed writeback IDs.

Evidence: `table-collection-conformance-2026-10-25.md`, seven native regressions,
twelve Python cases, 1,061 native / 1,789 root Python tests, 103 executable docs
and strict all-target Clippy. Receipt: `notes/benchmarks/table-collections-2026-10-25.json`.
Native kernel timings exclude the legacy bridge's BaseTable cloning cost; no
matched speedup, memory or FFI latency guarantee is claimed. Checklist stays
191 checked / 77 unchecked, with no new rows. Specialized merges and broader
collection ownership/conversion adapters remain open; this family is complete.


## 2026-10-25 — specialized graph-table merges (27 / AR7.10)

Closed all six existing GraphTable merge strategies and federated-domain naming.
One checked Rust plan replaces attribute-dropping, empty-output and cross-domain
fallback paths. Sparse IDs/endpoints and typed reference arrays use complete local
maps. Fail/keep/attribute/dense/auto-remap policies now have explicit stable row,
property and conflict behavior. Output retains source captures read-only, preserves
validation policies conservatively and checks surviving external identity.
Federated names are validated and emitted. Python merge/loading release the GIL.

Evidence: `graph-table-merge-conformance-2026-10-25.md`, ten native tests including
256 independent topology-oracle combinations, thirteen Python cases, 1,071 native /
1,802 root Python tests, 106 executable docs and strict all-target Clippy. Receipt:
`notes/benchmarks/graph-table-merge-2026-10-25.json`; no matched speedup or memory/
FFI latency claim. Checklist remains 191 checked / 77 unchecked, no new rows.
Specialized merges are complete; broader collection ownership/conversions stay
under existing 27/AR7.10, while pooled portability stays in 07.3/10.3/12.2.


## 2026-10-25 — collection ownership through empty results (27 / AR7.10)

Closed the supported empty grouping/selection/schema/callback/join ownership
inventory. A shared native context retains schema, observed roles, declared keys
and frozen source captures even when no tables remain. Projection/rename and
actual join column plans keep metadata coherent; invalid metadata cannot be
laundered by selecting nothing. Predicates execute once and Python callback
errors propagate. Native/core and legacy node/edge group adapters retain captures.

Rust compatibility: TableArray now has private context; direct struct literals
must use existing constructors. Public table/key fields and Python call shapes
remain. Native with_empty_source validates schema-only adapter metadata.

Evidence: `table-collection-ownership-2026-10-25.md`, eight native regressions,
ten Python cases, 1,079 native / 1,812 root Python tests, 109 executable docs and
strict all-target Clippy. Receipt: `notes/benchmarks/table-collection-context-2026-10-25.json`;
context preparation may scan cells, with no O(1), memory or FFI latency promise.
No new rows: 191 checked / 77 unchecked, including nine explicitly deferred and
14 overlapping architecture entries. Broader graph/table conversions and existing
legacy-view/portable-storage parents remain open; this ownership family is done.


### Native graph-table feature conversions — 2026-10-25

Closed both existing domain conversions and their three node dtype conveniences
under 27.1/AR7.10. One GraphTable-owned implementation fixes column-major data
being interpreted as row-major, preserves empty shapes and feature names, and
validates unique nonnegative integer row IDs. Physical row/feature order and
existing scalar conversion policies remain; errors identify the feature and row.
Outputs are independent numerical copies. No Python method or new module added.

Evidence: `graph-table-feature-conformance-2026-10-25.md`, five native regressions
(including 32 shapes in both domains), API contract `docs/api/graph-table-features.md`
and native 1k/10k/100k-row benchmark receipt. This completes the feature conversion
family; broader reshaping/bundle boundaries remain in the same parent. No new
checklist rows: 191 checked / 77 unchecked.


### Typed filters and checked reshapes — 2026-10-25

Closed node/edge typed equality filtering, stable generic pivot grouping,
checked melt schema/row counts, specialized pivot/melt identity validation and
the Python melt identifier adapter under 27.1/AR7.10. Existing positional Python
melt calls remain valid; `id_vars` is an optional keyword. General long layouts
use BaseTable; specialized outputs cannot discard or duplicate graph identity.

Evidence: `table-reshape-conformance-2026-10-25.md`. Gates: **1,090 native passed /
51 ignored; 1,819 root Python passed / 1 skipped; 112 executable docs passed;
strict all-target Clippy passed**. Current-cost receipt:
`notes/benchmarks/table-reshaping-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite/bundle adapter scope remains open.


### Typed and verified standalone table bundles — 2026-10-25

Closed the existing save/load/validate/info/verify bundle family under
27.1/AR7.10. New v3 CBOR stores ordered logical inline values and policy using
the shared property codec, exact float bits, checked framing/checksum and atomic
synced replacement of one authoritative file. V2 CSV reads now verify historical
checksums, metadata, manifest and schema/counts; resave migrates to v3. FFI
persistence logic was replaced with native delegation and GIL release.

Pooled refs reject rather than becoming strings; source pins/history are detached,
not represented as portable. Legacy CSV cannot recover lost type information.
Topology validation and encoded integrity remain separate. Whole-table memory
cost and post-publication directory-sync reconciliation are documented.

Evidence: `graph-table-bundle-conformance-2026-10-25.md`; frozen v2 fixture;
8 native / 11 Python regression cases. Gate: **1,098 native passed / 51 ignored;
1,830 root Python passed / 1 skipped; 115 executable docs passed; strict
all-target Clippy passed**. Benchmark receipt:
`notes/benchmarks/graph-table-bundles-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite adapters and separate full retained-pool,
history and storage lifecycle requirements remain open.


### Entity table adapters and 27.2 completion — 2026-10-25

Completed node/edge ID-mapped property attachment, typed structural readers,
linear tuple extraction, exact Python integer ID arrays and source-aware
attribute publication. Unknown/duplicate attribute IDs reject; malformed
structural reads no longer compact rows independently. Existing source capture,
head, schema and durable publication gates are reused.

27.2 is now checked: the immutable source-bound explicit table update contract
is complete, including Python mixed property updates with one delta and explicit
effective time. Remaining legacy live-view migration and composite conversions
stay in 26.1/27.1/AR7.10; full retained-pool portability keeps its original items.

Evidence: `entity-table-adapters-2026-10-25.md`; six native and thirteen Python
regressions. Gate: **1,104 native passed / 51 ignored; 1,843 root Python passed /
1 skipped; 118 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/entity-table-adapters-2026-10-25.json`.
No new checklist rows: **192 checked / 76 unchecked** after closing 27.2.


### Checked bulk table lookups and neighbor reads — 2026-10-25

Closed the existing native get_nodes/get_edges/neighbors adapter family under
27.1/AR7.10. Bulk identity lookup scans IDs once and copies selected first whole
rows; duplicate requests collapse and unknown IDs remain omitted. Structural
errors reject consistently. Neighbors scan borrowed endpoints once, preserving
the existing external-neighbor set semantics without copied edge tables/arrays.
No public API or graph direction/subsetting policy was added.

Evidence: `graph-table-bulk-reads-2026-10-25.md`; four native regressions, including
all 32 subsets of an independent looped/multiedge oracle. Gate: **1,108 native
passed / 51 ignored; 1,843 root Python passed / 1 skipped; 118 executable docs
passed; strict all-target Clippy passed**. Receipt:
`notes/benchmarks/graph-table-bulk-reads-2026-10-25.json`.
At 100k input rows with 100 requested IDs, current medians are 2.222 ms node
lookup, 2.345 ms edge lookup and 3.486 ms external neighbors; no matched speedup,
RSS or FFI claim. No new rows: **192 checked / 76 unchecked** after 27.2 closure.
Composite subsetting/routing and legacy view migration remain open. The induced
node-subset edge policy question remains unanswered; it was not silently applied.


### Legacy table output pin migration — 2026-10-25

Closed existing native/Python Graph, connected Subgraph and full/constrained
accessor table export paths under 26.1/27.1/AR7.10. All share pinned projection,
source row identity, stable selected IDs and explicit-null preservation. Mixed
exports share one capture; stale selections reject; dirty working captures stay
read-only. Removed separate FFI row-building logic. Legacy source objects remain
connected; GraphTable edge-retention policy is unchanged.

Evidence: `legacy-table-export-migration-2026-10-25.md`; 6 native / 12 Python cases.
Gate: **1,114 native passed / 51 ignored; 1,855 root Python passed / 1 skipped;
121 executable docs passed; strict all-target Clippy passed**. Fresh capture costs
are proportional to parent state; current capture/reuse measurements are in
`notes/benchmarks/table-export-capture-2026-10-25.json`. No COW, O(1), RSS or fixed
FFI budget claim. No new rows: **192 checked / 76 unchecked**. Broad legacy-view
and composite adapter exits remain open.


### Shared table schema validation — 2026-10-25

Closed ordered constructor, projection and column-growth invariants under
27.1/AR7.10. Every physical column must appear exactly once in an ordered schema;
projections reject duplicate/unknown requests; declared zero-row schemas cannot
be expanded by changing only one column. Truly empty initialization, sparse
from_columns padding and N-row zero-column projections retain their contracts.
Python node/edge projection delegates to native validation with GIL release.

Evidence: `table-schema-invariants-2026-10-25.md`; 6 native / 6 Python cases.
Gate: **1,120 native passed / 51 ignored; 1,861 root Python passed / 1 skipped;
124 executable docs passed; strict all-target Clippy passed**. These are O(schema
width) checks, without a new performance claim/kernel. No new rows: **192 checked /
76 unchecked**. Composite routing/subsetting and broad legacy migration remain.


### Composite routing and shared component metadata — 2026-10-25

Closed native sort/group/pivot/melt operand routing and GraphTable component
metadata sharing under 27.1/AR7.10. Edge melt now respects its identifier component
when values have shared names. Existing sort/reshape node preference and grouping
ambiguity rejection remain. Clones/groups share table/schema objects above the
already shared BaseArray value buffers; mutable access detaches metadata and
retains the existing column COW/replacement rules. Initial progress commentary
overstated prior value-copy costs; evidence/docs explicitly correct that claim.

Evidence: `graph-table-component-conformance-2026-10-25.md`; 6 routing / 5 sharing
native cases, including component and buffer pointer checks, independent edits
and source publication. Gate: **1,131 native passed / 51 ignored; 1,861 root Python
passed / 1 skipped; 127 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/graph-table-component-sharing-2026-10-25.json`.
Clone/drop timing is amortized over 10000 iterations to exceed clock resolution;
no matched speedup, RSS, whole-graph COW or FFI latency guarantee.

27.1 now has a fixed explicit remainder: node-subset edge policy; composite
metadata/rectangular Table interpretation; existing unsupported to_nodes/to_edges/
to_subgraphs converter contracts. These pre-existing facade contracts remain
within the original parent, not new checklist rows. Legacy object migration stays
in 26.1; full retained-pool/history portability remains separately tracked.
Inventory: **192 checked / 76 unchecked**, unchanged by adding no rows.

### Checked table visualization captures — 2026-10-25

Replaced all three Python table `.viz` graph-rebuilding loops with native checked
entity adaptation (27.1/AR7.10 and 28.15). Sparse/optimized IDs, parallel edges,
loops and typed properties retain their original identity; malformed/duplicate
IDs and dangling endpoints reject before server creation. No temporary graph,
new IDs or history are allocated. Direction is explicit natively; the existing
Python undirected default remains. Empty/oversized GraphDataSource windows are
safe. Evidence/API: `table-visualization-2026-10-25.md` and
`docs/api/table-visualization.md`; 7 native / 8 Python cases and 3 API examples.
Full gate: **1,138 native passed / 51 ignored; 1,869 root Python passed / 1 skipped;
130 executable docs passed; strict all-target Clippy passed**.

At 100k nodes, the current native direct capture median is about 80ms versus
798ms for checked graph materialization plus capture on the same dense fixture.
These are current alternatives, not a speedup against the old incorrect FFI
loop or an RSS/FFI/server guarantee. Receipt:
`notes/benchmarks/table-visualization-2026-10-25.json`.
Existing table DataSource compatibility readers and interactive_viz options,
converter ownership, node-subset and rectangular metadata contracts remain open.
Inventory remains **192 checked / 76 unchecked**; no new rows added.

### Shared immutable visualization captures — 2026-10-25

GraphDataSource clones now share their entire immutable payload and stable
process-local identity; counts derive from stored vectors and duplicate payload
helpers are consolidated. Python viewer server reuse now uses that identity,
replacing a temporary stack-address key that could alias separate captures.
Same accessor reuses its server; a fresh `.viz` accessor is a separate capture.

Evidence: `viz-snapshot-sharing-2026-10-25.md`; one pointer/last-owner test, four
native public-contract cases and one process-isolated Python real HTTP/websocket
server test (including sparse IDs in actual server snapshots). Gate: **1,143
native passed / 51 ignored; 1,870 root Python passed / 1 skipped; 130 executable
docs passed; strict all-target Clippy passed**. Matched native clone/drop at 100k
nodes: median 26.06ms before / 10ns after, amortized over 100 iterations with the
same benchmark source. Initial capture and owned payload exports still copy;
this is not a graph-capture/RSS/FFI/server latency claim. Receipts:
`notes/benchmarks/viz-snapshot-clone-{before,after}-2026-10-25.json`.
Existing 27.1/AR7.10/28.15 scope; **192 checked / 76 unchecked**, no new rows.

### Checked BaseTable entity conversions — 2026-10-25

Closed the existing native BaseTable converter family and exposed thin Python
`to_nodes(uid_key="node_id")` / `to_edges()` adapters (27.1/AR7.10). Custom node
keys now rename correctly; edge aliases use existing native normalization.
Complete identity/endpoint validation rejects malformed or duplicate IDs, while
parallel edges/self loops remain valid. Conversions create table components,
share column payloads, and allocate no graph entities or history.

Writable source binding survives only for the same captured kind/IDs/endpoints;
changed identity/kind/topology retains read-only source captures. Compatible
renames can publish one explicit source update. Import-oriented from_base_table
wrappers remain permissive. Evidence/API: `checked-table-conversions-2026-10-25.md`
and `docs/api/checked-table-conversions.md`; 7 native / 11 Python cases, 3 examples.
Gate: **1,150 native passed / 51 ignored; 1,881 root Python passed / 1 skipped;
133 executable docs passed; strict all-target Clippy passed**. Validation scans
identity columns; no speedup, RSS or FFI latency claim.

The fixed 27.1 remainder still names GraphTable node-subset policy, rectangular
metadata, and its unsupported graph-accessor/subgraph converter contracts.
Those are distinct from the now-complete BaseTable data-only adapters. No new
checklist rows; inventory remains **192 checked / 76 unchecked**.

### 12.3 exit review — explicit checkpoint requirement complete

Closed the original 12.3 parent after tracing native commit/checkpoint, writer,
Python default publication and durable annotation/replay through the existing
passing conformance suites. The former Partial status predated ordinary Python
default migration. Clean labels add no delta or state ID; dirty/manual commit
publishes once for compatibility; stale/dirty/schema failures and durable
rollback/recovery are covered. Evidence: `checkpoint-exit-review-2026-10-25.md`.

This is a status correction, not a new capability or a claim that native default
migration (09.4b.2), materialized recovery checkpoints/compaction (10.4), portable
pooled snapshots (12.2) or effective-time reconstruction (16.2) are complete.
Current gate remains 1,150 native / 1,881 root Python / 133 executable docs,
with 51 native ignored and one Python skip; strict all-target Clippy passed.
Inventory now **193 checked / 75 unchecked**, with no rows added or removed.

### GraphTable owner decisions implemented — 2026-10-25

Owner confirmed induced node subsets, separate component shapes/rectangular
operations through .nodes/.edges, and table-component to_nodes/to_edges aliases.
Implemented checked head/tail/slice and one-pass induced node grouping, exact row
provenance, separate native/Python shapes/schemas and independent component
wrappers. GraphTable no longer implements rectangular Table; native subsets are
fallible and Python raises standard errors. Existing aggregate counts remain
explicitly separate from shape. Also fixed BaseTable's end-versus-length slice
bug found by the new middle/reversed range cases.

Evidence/API: `graph-table-decisions-2026-10-25.md` and
`docs/api/graph-table-components.md`. Gate: **1,157 native passed / 51 ignored;
1,890 root Python passed / 1 skipped; 134 executable docs; strict all-target
Clippy passed**. Current 100k-node/100-group native cost about 31ms; receipt
`notes/benchmarks/graph-table-induced-groups-2026-10-25.json`. The earlier bundle
policy is a different workload; no matched speedup/RSS/FFI claim.

27.1 remaining inventory is now its unresolved to_subgraphs contract and native
legacy display/streaming schema integration. These are existing converter and
metadata boundaries, not new requirements. Broader 26.1/portable history remain
separate. Inventory **193 checked / 75 unchecked**, no new rows.

### Native GraphTable streaming projection and cache identity — 2026-10-25

Closed the remaining native table-schema integration boundary in 27.1/AR7.10.
A private graph_streaming adapter produces one explicitly tagged display
projection with component-prefixed columns, typed values and null padding;
headers/schema/width agree and windows span both components. Core shapes remain
separate. Clone-stable process-local presentation IDs refresh on derived outputs,
mutable component access, policy edits and explicit ID assignment, preventing
same-shaped branches from aliasing cached windows. No graph history is produced.

Evidence: `graph-table-streaming-2026-10-25.md`; four native cases. Full gate:
**1,161 native passed / 51 ignored; 1,890 root Python passed / 1 skipped;
134 executable docs; strict all-target Clippy passed**. Current induced-group
receipt: `notes/benchmarks/graph-table-streaming-identity-2026-10-25.json`;
100k-node/100-group median about 28.2ms, not a speedup/RSS/FFI claim.

The fixed 27.1 remainder is now **to_subgraphs grouping/ownership**, currently
awaiting the owner choice (explicit column groups / connected components /
deferral). Native legacy graph-data/layout adapters are unchanged and remain in
existing AR7.6/28.15 scope. Broader 26.1 and portable-history work remain separate.
Inventory **193 checked / 75 unchecked**, with no new rows.

### Native graph-table layouts preserve isolated nodes — 2026-10-25

Fixed the existing 28.15/AR7.6 compatibility layout adapter: using the complete
checked table capture retains isolated nodes, including a one-node graph, with
original IDs/order. Metadata no longer infers direction by reciprocal-edge
heuristics; the native table compatibility path defaults undirected and the
checked capture offers explicit direction. Removed the quadratic guess without
claiming all legacy graph readers, layout options or solver quality are certified.

Evidence: `graph-table-native-layout-2026-10-25.md`; two native cases. Full gate:
**1,163 native passed / 51 ignored; 1,890 root Python passed / 1 skipped;
134 executable docs; strict all-target Clippy passed**. Native current grid case
on 100k nodes (chain plus isolate, no properties) costs about 43ms including
complete capture, not a speedup versus the old incomplete node set. Receipt:
`notes/benchmarks/graph-table-native-layout-2026-10-25.json`.

The immediate remaining table decision is to_subgraphs grouping/ownership;
the three earlier owner answers are implemented and recorded. Original wider
workbook questions and 16.2 effective-time reconstruction remain open.
Inventory **193 checked / 75 unchecked**, no added rows.

## 2026-10-25 — Explicit grouping and execution priority

Owner accepted GraphTable.to_subgraphs(by=...) as immutable graph views grouped
by an explicit node column. Groups retain only internal edges and exact IDs;
standalone-table direction is explicit. This closes the fixed 27.1/AR7.10 facade
inventory rather than extending it with adjacent audits.

In response to the owner's waterfall/prioritization concern, resume core vertical
workflows: timed bulk mutation → immutable timestamp reconstruction → query/table
→ durable reopen. Implemented read-side 16.2 slice and optional-cutoff/pin behavior
16.3; atomic endpoint timeline rejection at publication remains the next priority.
The initial full-replay benchmark is evidence for optimization choices, not a
production latency promise. See effective-time-read-slice-2026-10-25.md.

## 2026-10-25 — Endpoint lifetime rule implemented

The accepted A04 rule now rejects invalid effective edge/endpoint intervals during
checked preparation, before mutation/allocation or durable append. Retained edge
history participates, including edges already removed from current state. Journal
recovery uses the original acceptance clock. Explicit timed topology with unknown
legacy lifetimes rejects; untimed legacy compatibility is retained without a
historical temporal guarantee. Evidence: effective-topology-publication-2026-10-25.md.

## 2026-10-26 — Effective-time contract complete (16.2)

Frozen source revisions now implement the accepted A01 separation: subnode changes
and meta property changes remain distinct; changed source pins produce new group
references, with old sources retained. Missing legacy clocks/member-source pins
require explicit caller inputs for that read, never guesses or immutable-history
rewrites. Known precise clocks/frozen refs cannot be overridden. Resolved merge
reads follow existing first-parent resolution semantics; planning stays 14.2.
See effective-time-exit-review-2026-10-26.md. Property unset remains the next 11.2
priority. The owner authorized up to 90 minutes of priority slices after 16.2,
aiming at task completion, separate coherent commits and explicit open-work notes.

## 2026-10-26 — 11.2 property history completed

Implemented the accepted distinction between absence and explicit Null, including
node/edge scalar/bulk/mixed unsets and explicit/default effective times. Later
accepted equal-time events win; an absent key stays absent until an assignment.
Old typed values and frozen sources remain retained, and durable recovery preserves
clocks and history. Fixed mixed scalar/bulk coalescing. See
`property-history-exit-review-2026-10-26.md` and `docs/api/property-history.md`.
12.2/10.3/10.4 retain owned snapshot/pool import/storage lifecycle requirements.

## 2026-10-26 — owned snapshot contract implemented (12.2)

Full configuration and frozen references survive lightweight materialization;
explicit owned temporal captures retain all backing storage/history/reservations
and restore independent owners. Repeated branch/reset restoration no longer
appends duplicate values. No owner answers changed and no checklist rows added.
See `owned-snapshot-exit-review-2026-10-26.md` and `docs/api/owned-snapshots.md`.
Persistent archive/import and storage lifecycle remain 10.3/10.4.

## 2026-10-26 — outer pipeline and import publication boundaries

Completed certified whole-pipeline property acceptance and complete selected
snapshot import baselines under existing 09.4b.2. Errors/cancellation discard
earlier staged updates; children share one acceptance; untouched imported topology
and properties remain in history. **199 checked / 69 open**, no new rows. The
parent now has three explicit open components: native default/infallible migration,
raw pool/space escapes and uncertified/topology algorithm acceptance. Effect-aware
IR remains AR7.8. Evidence/gates: `outer-pipeline-boundary-2026-10-26.md`.

## 2026-10-26 — branch lifecycle completed (14.1)

Closed the existing creation/switching/identity/read/recovery inventory. Recorded
timestamp selection now uses selected first-parent ancestry; siblings and later
unselected retained states cannot leak into it. Frozen groups, dirty manual
workflows, empty branches and durable schema/failure boundaries are verified.
**200 checked / 68 open**, no new rows. See `branch-lifecycle-exit-review-2026-10-26.md`.
Explicit destructive pruning/history and pool reclamation remain 15/10.4.

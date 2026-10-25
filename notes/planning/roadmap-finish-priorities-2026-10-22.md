# Roadmap finish priorities — 2026-10-22

User direction: add module consolidation to the flat checklist and prioritize
completion of the existing roadmap while the owner finishes the workbook.
This is an execution order, not a replacement vision or a claim that unfinished
items are absent from the existing engine. Original 01–34 numbering is retained;
AR7 tracks consolidation. Pending workbook suggestions are not accepted decisions.

## Current execution checkpoint

Implemented: AR7.2–AR7.4 consolidation, numerical completion slices and receipts,
AR7.5a–l immutable reads through components and centralities, AR7.9a canonical
logical change records, and 07.3a f64 representation, supported consumer migration
and ordinary Python default switching, plus 07.3b native temporal properties.
See the flat checklist for each supported scope and evidence.

Next: finish native compatibility and compound mutation boundaries (09.4b.2).
Python ordinary defaults are publishing (09.4b.1), using the explicit checkpoint
boundary (12.3a). Reconstructed effective-time states follow. Source-preserving table adapters and broader legacy frozen
read guarantees remain open; owned projections do not complete those parents.

## Owner answers received — 2026-10-22

See [the answer record](workbook-answers-2026-10-22.md) and updated appendix.
Group revision references/selected-state copying, same-effective-time ordering,
endpoint validity rejection, persistent absence after unset, f64 floating defaults,
UTC instant/date/duration types, staged schema migration, mixed table writeback,
explicit index failure, component/path defaults and transient pipeline runs now
have recorded directions. These decisions are not implementation completion.

The accepted f64 default intentionally supersedes earlier preserve-f32-default
recommendations. Implement tagged f64 properties and bridge/codec/history/schema/
numerical/query migration before declaring the final property format complete;
keep existing tagged data readable. Native matrix f64 support alone does not do
this. Endpoint validity now rejects invalid timelines rather than silently hiding
an edge. No generic silent index-to-scan fallback is authorized.

On 2026-10-23 the owner confirmed migrating checked automatic-publication
defaults now, requiring an explicit retained restoration state, and explicit purge
preserving shared branches/open immutable views without extra permanent audit/
checkpoint pins or mandatory grace/backup. Current scope is a data science library.
A07.1 is tentative; A08 and other unanswered
fields remain open. Checkpoint scheduling is explicitly deferred until MVP/first
use; opening-time measurement can proceed now without inventing a log budget.
A10.1 concerns duplicated source rows, not automatic creation of graph entities.

## Immediate work, without waiting for the whole workbook

1. **AR7.2 — one matrix module.** Use the current 455 Python regressions, full
   native suites and 25-case release receipt as the migration baseline. Inventory
   exports/callers, move existing implementations into explicit internal modules,
   preserve legacy imports through re-exports, and compare behavior/performance.
   This changes ownership/organization while preserving current semantics.
2. **AR7.3 — array/numeric foundations.** Resolve duplicate BaseArray ownership
   and locate shared scalar/dtype/conversion/accurate arithmetic primitives below
   both arrays and matrices. Preserve typed arrays, precision and null policies.
3. **Close the numerical pass deliberately.** Convert the broad remaining 29
   parents into a finite inventory of supported surfaces and concrete findings.
   Complete the size/sparsity/fill/dispatch sweep, public API/doc and lint gates;
   separate working reference dispatch from actual SIMD acceleration claims.
   Fix discovered failures. Parent completion requires evidence across its stated
   scope; an optional enhancement does not automatically become a new blocker,
   and an unfinished promise requires implementation or explicit owner deferral.
4. **AR7.4/AR7.5 with AR1/AR2/AR4 — entity/view foundations.** Unify duplicate
   handles and shared membership/state ownership; make immutable read views pin
   the selected state and propagate it into numerical/query/table results.
   Current live-parent Subgraph behavior does not satisfy that accepted contract.
   Preserve retained inactive-ID inspection separately from current membership.

## Then follow the dependencies

### 1. State, values, changes and temporal reconstruction

Finish 04.2/05.3/06.3/07.3–07.4/09.4/10.3/11.2/12.2–12.3/13.3/16.2–16.3,
with AR3/AR6 and shared change records (AR7.9). Exact retained referents and typed
changes underpin state restoration and time queries. Checkpoint operations and
reconstruction are separate: accepting a property/bulk mutation records its delta
with explicit effective time or the accepted batch default; reading/computing does
not create a change. A requested snapshot is a checkpoint. Optional recorded_by
remains explicit. Test unambiguous cases and prepare the machinery now; await the
specific unanswered policies before choosing correction/lifecycle semantics.

### 2. Durable lifecycle, branches and retention

Finish 08.3/10.4/14/15. Portable retained storage and exact states precede trustworthy
merges/reclamation. Implement checkpoint/recovery/format-validation mechanisms
separately from choosing automatic scheduling, schema migration, merge resolution
or destructive retention policies. Verify crash recovery before publication and
reclamation claims. Do not silently purge history while the workbook is pending.

### 3. Indexed queries, tables, algorithms and pipelines

Finish 18/21/27/remaining 28/30, alongside AR7.7–AR7.8/AR7.10. Index and query results
must share active/state/time semantics. Tables need explicit identity/writeback.
Pipelines consume corrected primitives and pinned inputs. Remaining Infomap,
non-registry analytics and algorithm certification remain named work. Pipeline
run-history and ambiguous writeback are decision gates, not permission to invent
policy. Independent algorithm/reference work can proceed while other gates wait.

### 4. Core verification, presentation and operations

Carry 01.2/19/31/32/34 through every preceding step. Then finish the complete
comparison/failure/scale reports, display consolidation (AR7.6), smaller facade
work (AR5), compatibility retirement gate (AR7.11) and operational integrity.
09.5/33.3 need an explicit service boundary: embedded correctness does not complete
the multi-user objective. Benchmark target fixtures can be published provisionally;
owner-specific performance budgets remain unanswered until selected.

### 5. Deferred integration remains part of finishing the full vision

After the core milestone, resume 17/20/22–25/32.4: provenance/episodes, Graphiti
round-trip/idempotency, lexical/exact-vector/hybrid retrieval and operation driver.
Provenance/storage primitives can be prepared earlier when their contracts are
settled. Choose the Graphiti release/fixture before adapter compatibility claims.
Completing the core must not be reported as completing all 34 items.

## Workbook reading priority

The owner does not need to complete every page before implementation resumes.
Use the existing open-questions appendix; do not overwrite answers or treat blank
fields as approval.

| Priority | Appendix | What the answers unlock |
| --- | --- | --- |
| First | A01, A03–A06 | Retained group/meta identity/copy rules; migration of legacy writes/names; same-time corrections and endpoint validity; unset/restore semantics; temporal types and property float precision. Most now have owner answers, including publication defaults and explicit restoration state; see the answer record. |
| Next | A02, A07–A11 | Checkpoint/upgrade budgets, schema evolution, merge conflicts, retention/purge, table writeback and index loading/fallback. Some answered; schema-branch/merge/rebuild choices still need resolution before their lifecycle behavior is finalized. |
| Alongside algorithm/numerical closure | A12–A13, A16 | Compatibility choices, numerical guarantees and duplicate/retry extensions. Preserve existing established direction, exact-shape, checked precision and rejection rules until an explicit change is accepted. |
| Before pipeline/service commitments | A14–A15 | Intermediate naming/run provenance and actual scale/service budgets. These do not block module moves or independent correctness work. |
| After core, before Graphiti integration | A17 | Compatibility release and correction-versus-parallel-fact adapter identity. |

If a selected task reaches an unanswered policy gate, finish all independent
preparation and report that specific dependency; continue the next independent
checklist item. No elapsed-time assumption or provisional recommendation counts
as the owner's answer.

## Completion discipline

- Each task identifies its stable checklist ID, concrete behavior and exit checks
  before implementation; commit coherent completed items separately.
- Preserve existing functionality, import compatibility and accepted decisions.
- Add verification at the layer changed, including Rust/Python and applicable
  persistence/state/time behavior. Capture before/after performance for structural
  numerical/graph kernel changes.
- Keep broad unchecked parents accurate; historical completed slices retain their
  scope. Finish conformance, public documentation and examples as part of the item.
- Record discovered defects under the responsible item. Distinguish optional new
  capabilities from necessary repairs so finishing does not become endless scope
  expansion. Deferring original vision work requires the owner's direction.


### AR7.5i completed — frozen components

Connected component views now share captured state and existing kernels, with the
accepted default/weak/strong modes. See `frozen-components-contract-2026-10-22.md`.
Next independent read migration: centrality. Restore/purge/publication policy
questions remain pending; this slice does not choose them.


### AR7.5j completed — frozen PageRank

Captured PageRank now shares one validated power-iteration kernel with the live
wrapper. Owned results expose finite-budget convergence status. See
`frozen-pagerank-contract-2026-10-22.md`. Next independent read migrations:
closeness and betweenness; broader AR7.5 remains open.


## AR7.5k — Frozen closeness (2026-10-22)

Completed captured classic/harmonic closeness reads using the existing distance/
scoring kernel and a shared compact incoming CSR transpose. Preserve defaults,
normalization and strict selected weight rules; return owned f64 rows without
publication. See `frozen-closeness-contract-2026-10-22.md`. Betweenness and broad
AR7.5 remain open.


## AR7.5l — Frozen Brandes betweenness (2026-10-23)

Completed captured node betweenness over shared Brandes kernels, preserving
parallel shortest-instance counts, selected isolates and normalization. Repaired
dropped live transient outputs and indexer overflow; strengthened cancellation.
See `frozen-betweenness-contract-2026-10-23.md`. The centrality read slices are
complete; remaining algorithms, registry/pipeline and broad AR7.5 stay open.


## AR7.9a — Canonical logical change records (2026-10-23)

Snapshot and history attribute differences now use one record/target type owned
by state::change, with exact legacy re-exports. Commit diffs move the existing
record vector directly; FFI output is unchanged. Preserve typed/null/absence
semantics and leave diagnostic/sampling scopes separate. See
`change-record-owner-2026-10-23.md`; broad AR7.9 and temporal publication/storage
migration remain open.


## 07.3a.1 — f64 property foundation (2026-10-23)

Distinct f64 scalar/vector tags and bit-pattern codecs now preserve wide values
through the tested property/history/storage lifecycle; explicit Python constructors
and width-aware schemas are available. See `float64-property-foundation-2026-10-23.md`.
This implements a dependency of the accepted f64 decision. Consumer migration
07.3a.2 and the ordinary Python default switch 07.3a.3 remain open; the parent is
not complete, and stored legacy f32 values are not widened automatically.


## 07.3a.2a — Shared exact numeric predicates (2026-10-23)

Array/table numeric predicates and cross-table extrema now use one wide scalar
comparison owner, shared with graph numeric ordering. Fix epsilon/narrowing,
operator-prefix and parity faults; preserve typed graph equality and storage
identity. See `float64-numeric-predicates-2026-10-23.md`. Remaining arithmetic/
outputs, broader expression planning and the Python default switch remain open.


## 07.3a.2b — f64 numerical results and range (2026-10-23)

Checked numerical property outputs retain f64; explicit NumArray f32 stays typed.
Stream cross-table reducers, preserve cross-chunk integer cancellation, divide
exact accumulations before final rounding and rescale wide moment roots. Preserve
finite/error and empty policies; use f64 tolerances for approximate statistics.
See `float64-numerical-outputs-2026-10-23.md`. Remaining consumers and ordinary
Python default inference stay open.

## 2026-10-23 — checked table profiling (07.3a.2c)

Table profiles now include f64 and use shared checked numeric kernels, preserve
source extrema/rank quartiles and f64 statistics, reject nonfinite observations,
and fix text uniqueness/empty ratios. Existing population/rank conventions stay
explicit. See `table-profiles-2026-10-23.md` and `docs/api/table-profiles.md`.
Remaining property consumers and ordinary float-default migration remain open.

## 2026-10-23 — f64 algorithm properties (07.3a.2d)

Dijkstra, closeness, betweenness and PageRank persistence retains owned f64
scores, replacing prior f32-only output bounds. Shared edge weight/capacity
access recognizes f64, and compatibility handles inherit the canonical policy.
See `algorithm-float64-properties-2026-10-23.md`. This is a property-tag migration,
not completion of remaining consumer/default or publication-policy work.

## 2026-10-23 — whole-table dispatch and typed display (07.3a.2e)

F64 participates in whole-table ranking/statistics/windows; pairwise errors and
diagonal semantics come from the shared column kernel. Matrix display and array
materialization share one typed scalar converter, with checked reads and no
intermediate scalar-input buffer. See `table-numeric-dispatch-2026-10-23.md`.
Remaining text/vector/step/view adapters and ordinary defaults stay open.

## 2026-10-23 — checked table outliers (07.3a.2g)

F64-aware typed ordering and exact IQR fence comparisons retain high-integer
precision and handle extreme finite products without materializing rounded f64
fences. Invalid factors/nonfinite cells fail; row alignment/rank conventions
remain. Python releases the GIL. See `table-outliers-2026-10-23.md`. Remaining
consumer/default inference work stays open.

## 2026-10-23 — graph numeric literals (07.3a.2h)

Graph decimal/scientific literals retain f64 with explicit f32/f64 suffixes for
typed equality. Complete-input validation rejects malformed/overflowing tokens,
incomplete strings and trailing input. Live/frozen node and edge filters share
the parser without changing property identity or numeric ordering. See
`graph-query-literals-2026-10-23.md`; broader AR7.7 and defaults remain open.

## 2026-10-23 — f64 consumer/default completion (07.3a.2i–3)

Ordinary Python floats and numeric float collections now default to f64. Explicit
f32 constructors/NumPy float32 inputs and old tagged f32 data retain their width.
Remaining supported consumers delegate to checked native kernels; wrapped
algorithm parameters now share the native converter. Empty labels and table
schema families handle both widths. Direct/bulk writes, history/export/journal
recovery and old-format reads are covered. Root API baseline: 976 passed, one
existing skip; full native suites, strict all-target Clippy and 22 documentation
examples passed. See `float64-defaults-2026-10-23.md` for scope and evidence.

07.3a is complete. Next is 07.3b native UTC instant/date/duration; the wider 07.3
property-format parent, publication migration and legacy immutable-view work stay
open. Earlier dated default-migration gates above record intermediate states.

## 2026-10-23 — native temporal properties (07.3b)

UTC instants, calendar dates and fixed durations now have distinct native types,
checked constructors/deserializers and version-1 tagged payloads. Python aware
datetime normalizes to UTC; naive datetime rejects. Exact microseconds, shared
calendar limits and full timedelta range round trip through direct/bulk writes,
tables, retained snapshots, schemas, export and checked journal recovery.
Typed same-domain comparisons preserve identity and numeric exclusion. Shared
Rust/Python fixtures, limits, fractional offsets and DST folds have coverage.
See `temporal-properties-2026-10-23.md` and `docs/api/temporal-properties.md`.

This closes 07.3b, not the broader property-format/pooled-reference parent or
mutation effective-time reconstruction. Next: 09.4 ordinary checked publication
and 12.3 explicit checkpoint semantics under the confirmed owner decision.

## 2026-10-23 — explicit checkpoint boundary (12.3a / 09.4a)

Clean `commit()` and explicit checkpoint/writer operations now label the selected
published version without allocating another state or changing mutation clocks.
Legacy dirty `commit()` publishes once for compatibility. Durable checkpoint
records sync before visibility and recover original metadata through replay and
backup; stale/dirty/schema failures, crash boundaries and invalid record targets
have coverage. Checkpoints do not add permanent retention pins or materialized
recovery images. New native fallible insertion conveniences publish checked bulk
operations. See `explicit-checkpoints-2026-10-23.md`.

Ordinary mutation defaults, including pooled/meta and compound write paths,
remain 09.4b. The owner's authorization to migrate them stands; this foundation
does not close the broad publication/default or storage lifecycle parents.

## 2026-10-23 — ordinary Python publication defaults (09.4b.1)

Python `Graph()` and active graph imports now return publishing graphs; explicit
`manual=True` retains compatibility. Native coordinator policy routes supported
ordinary creation, scalar/bulk/column setters, removals/cascades, kinds and managed
identities through checked publication. Creation includes initial properties;
prepared execution suppresses recursive routing and restores the policy. Base
graph copying stages before allocation; unsafe cross-pool references reject.
Meta creation publishes required system fields together, and classification now
accepts logical compact text. Graph-producing factories accept their private
imports once before return. See `automatic-publication-2026-10-23.md`.

The remaining Rust default/infallible compatibility surface and higher-level
compound meta/collapse, algorithm and table writeback boundaries remain 09.4b.2.
The broad 09.4 parent and pool portability/effective-time reconstruction are not
closed by this core default migration. Existing owner authorization remains.

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

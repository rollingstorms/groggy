# Groggy roadmap status — 2026-10-24

Branch: `groggy-roadmap-2026-10-20`. The [flat checklist](graph-engine-flat-checklist.md)
currently contains **191 checked and 77 unchecked entries**. Entries overlap:
parent workstreams and smaller repair slices both appear, and unchecked entries
include partial, open, review and deferred work. These counts are an inventory,
not a completion percentage or an estimate of remaining implementation effort.

Count correction: the earlier summary omitted three already checked A* entries
because its heading parser treated the literal asterisk as formatting. The earlier atomic-builder pass
added two checked slices; three other entries were a counting correction. The
recent loop/block, structural, filtering, aggregation, transformation, mask-sampling, community and bounded-selector continuation slices are counted individually alongside their open parents. Counts now use checkbox markers independently of headings.

## Fixed remaining-work inventory

The 77 unchecked entries are not 77 independent tasks: 14 are overlapping AR
architecture entries, 63 are numbered entries (including nested publication
parents), and nine of those numbered entries explicitly say Deferred. The old
flat-checklist header also still said 158 checked; it is corrected to 189.
Recent implementations were recorded as new checked slices while broad parents
stayed open. That made the open total insensitive to substantial completed work.

From this pass, audit repairs update existing requirements/evidence. New
capabilities are identified as scope additions. Each current workstream uses a
fixed implementation inventory with stated exits, so completing components reduces
its remaining count without manufacturing broad parent completion.

### Temporal builder family — existing eight primitives

| Existing primitive | Current status | Verified boundary |
| --- | --- | --- |
| snapshot_at | Complete | Retained-state selector certified in 09.4b.2al; effective-time reconstruction remains a separate original requirement. |
| decay | Complete | Checked maps/columns certified in 09.4b.2aj–2ak. |
| diff_nodes | Complete | Retained owner provenance, added/removed/modified maps, property comparison and atomic failure behavior verified. |
| diff_edges | Complete | Same; unrepresented endpoint rewiring rejects under existing 04.2/13.3. |
| window_aggregate | Complete | Correct change-value history, branch/index provenance, sparse state IDs, null/absence and checked reductions. |
| filter | Complete | Correct branch-scoped lifecycle/changes, retained-index provenance and failure/cancellation behavior. |
| mark_changed_nodes | Complete | Exact change-type membership, aggregation over windows and index provenance. |
| window | Complete | State-ID/time units, reactivation/deletion boundaries and branch-scoped existence. |

**Remaining: six → four → zero. All eight primitives complete; no checklist rows
added.** These are the same primitive names present before the audit. This closes
the fixed temporal family under 30.1; other pipeline/intermediate/pin requirements
and arbitrary effective-time reconstruction (16.2) retain their original scope.
See `temporal-window-completion-2026-10-24.md` for implementation/gate evidence.

## Recent completed slices

| Slice | What is now supported |
| --- | --- |
| 26.2 | Checked current-view batch writes, exact source/scope validation, prevalidated refreshed pins and durable journal publication/recovery. |
| 12.3a | Explicit checkpoints label accepted states without creating duplicate mutation deltas. |
| 09.4b.1 | Ordinary Python mutations default to checked publication, including initial properties and bulk operations. Explicit manual compatibility remains. |
| 09.4b.2a | Compound collapse prepares aggregations, retained membership, boundary edges and removals before accepting one version. |
| 09.4b.2c / 27.2a | Explicitly mapped mixed node/edge table property updates validate identity/endpoints and publish one batch. |
| 09.4b.2d / 05.3a | Subgraph/algorithm bulk writes reject complete inputs with inactive targets; all-edge convenience writes publish once. |
| 09.4b.2e / 27.2b | Table graph materialization preserves supported typed values/identities, remaps references and rejects unsupported data without silent downgrade/loss. |
| 09.4b.2f / 30.1a | Builder node/edge attachment and smoothing honor persist=False and return owned typed columns without graph mutation. |
| 09.4b.2g / 30.1b | Explicit atomic property builders stage touched columns, read staged values and accept once; errors/cancellation discard the buffer. |
| 09.4b.2h | Nested property loops recursively validate descendants, share staged values across iterations and discard the whole run on failure. |
| 09.4b.2i | Fused variable kernels and supported ordered message-pass blocks compose with atomic staging; options/direction and arithmetic errors are checked. |
| 09.4b.2j | Conditional blocks honor compare/where semantics with shared masks, early input/operator validation and atomic rollback. |
| 09.4b.2k | Synchronous message passing reads fixed block input and installs buffered outputs together, preserving column order and outer atomic staging. |
| 09.4b.2l | Degree and incident-weight primitives honor selected topology, staged weights and checked input/output rules. |
| 09.4b.2m | Undirected triangle and k-core primitives use checked selected topology, corrected counting/peeling and atomic staging; directed variants reject explicitly. |
| 09.4b.2n | Property filters and ordering honor staged values, exact numeric order, deterministic ties and explicit validation. |
| 09.4b.2o–2q | Shared checked statistics, deterministic categories/owned histogram vectors, five neighbor reducers and public/optimizer lowering parity. |
| 09.4b.2r–2t | Checked expression operators/literals, scoped synchronous/async maps and deterministic selected label/copy transformations; opaque callbacks remain uncertified. |
| 09.4b.2u–2w | Stable isolated node/edge mask sampling and bounded map-key reservoirs with strict controls, selected identity validation and atomic publication. |
| 09.4b.2x–2z | Checked seeding, synchronous propagation and actual directed/undirected local modularity gains verified against independent full-objective moves. |
| 09.4b.2aa–2ac | Ordered bounded selector iteration, selected neighborhood expansion and consistent induced emission; legacy owner connections remain explicit. |
| 09.4b.2ad–2af | Recursive for-each property staging, complete child selector flattening and private merged/per-item owned output frames. |
| 09.4b.2ag–2ai | Checked staged-weight distance maps, loopless Yen node paths and isolated seeded edge-incidence walks. All eight temporal primitives are now complete under 30.1; broader pipeline work remains. |
| 09.4b.2aj–2al | Finite selected decay maps, ordered column parity and retained immutable snapshot builder selection with strict temporal controls. |
| 09.4b.2b | Preset/override precedence, canonical meta identity with semantic labels, explicit meta selectors and real external endpoint contraction. |
| 30.1 temporal family | All eight primitives complete: retained branch indexes, exact lifecycle/change windows, checked aggregation and failure-safe outputs. |



These are extensions and repairs to an already substantial graph engine. They do
not represent building its basic engine contract from scratch.

## Broader areas

| Area | Completed foundation | Still open |
| --- | --- | --- |
| Numerical core and architecture | Canonical matrix/array foundations, entity handles, many arithmetic/statistical/solver/factorization/sparse/operator and algorithm repairs, bounded release receipts. | Remaining promised derivative/composition domains, uncovered consumers and representative performance/memory/scale gates; broad numerical parents remain partial. |
| Mutation/publication | Checked mixed batches, ordinary Python defaults, property clocks, checkpoints, compound collapse and recent table/subgraph boundaries. | Remaining native default/infallible compatibility, mutable pool/space escapes, whole-pipeline boundaries and broader pipeline semantics. |
| Properties, state and history | f64 defaults, typed temporal properties, shared logical change records, retained snapshots and immutable read APIs. | Complete value lifecycle, effective-time reconstruction/validation, rewiring/revert, exact portable snapshots and portable pooled/meta contents. |
| Views and tables | Explicit pinned read views, frozen matrix/table exports, mapped writeback and checked independent materialization. | Migrate legacy connected reads and broader graph/table conversion adapters. Frozen BaseTable selections/projections and generic joins now retain source pins; direct/mixed source property writeback is checked. Generic joins, grouping, dedup and set operations share typed keys and stable ordering; row combinations retain source context, and concat rejects name collisions. TableArray reductions/group-key validation and lossless supplied-table flatten now share a Rust coordinator. All six specialized graph-table merges and federated domain naming now preserve attributes, domain-local reference maps, validation policy and source pins. Collection context now survives empty grouping/selections, schema transforms, callbacks and native joins. Broader graph/table conversions and broad table parents remain partial. |
| Storage, branches and schema | Checked journal/recovery foundations, checkpoint annotations, existing branch/identity behavior and explicit schema checks. | Full pooled retention, lifecycle/compaction/format migration, complete branch lifecycle, three-way merge, purge and schema evolution. |
| Queries, indexes and pipelines | Existing native filter/query/index/builder machinery and many repaired algorithm paths. | Consistent index lifecycle, pinned temporal query results, shared expression/execution boundaries, pipeline intermediates/history and complete algorithm certification. |
| Integration and service | Existing embedded engine and test/benchmark infrastructure. | Graphiti ingest/driver/provenance/retrieval remain deferred behind core; service concurrency, multi-user deployment and broad comparison/scale gates remain open. |

## Continue in this order

1. Finish the remaining native mutation/publication and compound pipeline audits
   under 09.4b.2; composer leaf 09.4b.2b is now complete.
2. Complete legacy view/source-pin migration and retained pool/meta recovery;
   these underpin dependable history and connected table behavior.
3. Finish timestamp reconstruction and timeline validation, then complete
   rewiring/revert, branch/merge/retention and storage lifecycle behavior.
4. Complete query/index/pipeline consistency and remaining architecture moves.
5. Close documented numerical/algorithm/performance/conformance/release gaps with
   representative fixtures and measured limits.
6. Resume Graphiti provenance, ingest, retrieval and driver integration after the
   core behavior is dependable.

This order does not create a new approval gate. Existing workbook answers remain
the implementation authority; unresolved policy decisions stay visible in the
appendix and their corresponding checklist entries.

## Current verification baseline

Latest collection-ownership slice (2026-10-25): **1,079 native tests passed, 51 ignored; 1,812 root
Python API tests passed, 1 skipped; 109 executable documentation examples passed;
strict all-target core Clippy passed.** The root Python gate excludes the unrelated
owner benchmark test and the separate legacy modules survey. Local release
receipts are bounded observations, not production throughput/memory guarantees.

A matched composer before/after comparison now measures the bulk-removal repair
(`bulk-removal-2026-10-25.md`); broader performance comparison remains follow-up
under the existing performance gates. As requested, implementation priorities continue
first; comparison must match workload, precision and history semantics and record
time, allocations, peak memory and retained-history growth.

The explicit nested-sampler child 09.4b.2ad is now complete alongside related
selector-output and owned-frame slices 09.4b.2ae/2af. This closes one unchecked
entry while adding two checked implementation slices. Standalone sampler
transactions and pinned view/source provenance remain open parent work.


### Native graph-table feature conversions — 2026-10-25

Completed native node/edge feature conversion and the existing f64/f32/i64 node
conveniences under 27.1/AR7.10. Fixed row-major layout, empty shape/name retention,
and strict ID validation through one GraphTable-owned implementation. Five new
native tests cover both domains, 32 shapes and conversion/ownership errors.

Latest gate: **1,084 native passed / 51 ignored; 1,812 root Python passed /
1 skipped; 109 executable docs passed; strict all-target Clippy passed**.
Evidence: `graph-table-feature-conformance-2026-10-25.md`; benchmark receipt:
`notes/benchmarks/graph-table-features-2026-10-25.json`.

No new checkbox rows: **191 checked / 77 unchecked**, including overlapping
parents and nine explicitly deferred entries. The feature-matrix conversion
family is closed; the broader graph/table reshaping and bundle adapter boundaries
remain under the existing parent.


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

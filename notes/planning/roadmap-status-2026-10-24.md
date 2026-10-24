# Groggy roadmap status — 2026-10-24

Branch: `groggy-roadmap-2026-10-20`. The [flat checklist](graph-engine-flat-checklist.md)
currently contains **177 checked and 79 unchecked entries**. Entries overlap:
parent workstreams and smaller repair slices both appear, and unchecked entries
include partial, open, review and deferred work. These counts are an inventory,
not a completion percentage or an estimate of remaining implementation effort.

Count correction: the earlier summary omitted three already checked A* entries
because its heading parser treated the literal asterisk as formatting. The earlier atomic-builder pass
added two checked slices; three other entries were a counting correction. The
nested-loop, fused/block, conditional, synchronous, two structural, filtering and three aggregation, three transformation, three sampling and three community extensions add nineteen further checked slices. Counts now use checkbox markers independently of headings.

## Recent completed slices

| Slice | What is now supported |
| --- | --- |
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



These are extensions and repairs to an already substantial graph engine. They do
not represent building its basic engine contract from scratch.

## Broader areas

| Area | Completed foundation | Still open |
| --- | --- | --- |
| Numerical core and architecture | Canonical matrix/array foundations, entity handles, many arithmetic/statistical/solver/factorization/sparse/operator and algorithm repairs, bounded release receipts. | Remaining promised derivative/composition domains, uncovered consumers and representative performance/memory/scale gates; broad numerical parents remain partial. |
| Mutation/publication | Checked mixed batches, ordinary Python defaults, property clocks, checkpoints, compound collapse and recent table/subgraph boundaries. | Remaining native default/infallible compatibility, mutable pool/space escapes, whole-pipeline boundaries and unfinished composer semantics. |
| Properties, state and history | f64 defaults, typed temporal properties, shared logical change records, retained snapshots and immutable read APIs. | Complete value lifecycle, effective-time reconstruction/validation, rewiring/revert, exact portable snapshots and portable pooled/meta contents. |
| Views and tables | Explicit pinned read views, frozen matrix/table exports, mapped writeback and checked independent materialization. | Migrate legacy connected reads, preserve source identity/pins through selections/joins, checked writes through pinned views. |
| Storage, branches and schema | Checked journal/recovery foundations, checkpoint annotations, existing branch/identity behavior and explicit schema checks. | Full pooled retention, lifecycle/compaction/format migration, complete branch lifecycle, three-way merge, purge and schema evolution. |
| Queries, indexes and pipelines | Existing native filter/query/index/builder machinery and many repaired algorithm paths. | Consistent index lifecycle, pinned temporal query results, shared expression/execution boundaries, pipeline intermediates/history and complete algorithm certification. |
| Integration and service | Existing embedded engine and test/benchmark infrastructure. | Graphiti ingest/driver/provenance/retrieval remain deferred behind core; service concurrency, multi-user deployment and broad comparison/scale gates remain open. |

## Continue in this order

1. Finish the remaining native mutation/publication and compound pipeline audits
   under 09.4b.2, including the explicitly recorded composer gaps.
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

Latest community-bundle change: **964 native tests passed, 52 ignored; 1,489 root
Python API tests passed, 1 skipped; 60 executable documentation examples passed;
strict all-target core Clippy passed.** The root Python gate excludes the unrelated
owner benchmark test and the separate legacy modules survey. Local release
receipts are bounded observations, not production throughput/memory guarantees.

The matched before/after performance comparison remains follow-up under the
existing performance gates. As requested, implementation priorities continue
first; comparison must match workload, precision and history semantics and record
time, allocations, peak memory and retained-history growth.

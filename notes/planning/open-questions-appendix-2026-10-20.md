# Groggy open questions appendix - 2026-10-20

Companion to the design workbook. Baseline: `c01f20d4`, branch `groggy-roadmap-2026-10-20`. The PDF has printable/fillable answer fields. Roadmap numbers below use the staged 1-34 list.

**Status (updated 2026-10-22):** owner answers and explicit clarifications are recorded below and in [the answer record](workbook-answers-2026-10-22.md). Unanswered choices and suggestions remain provisional. An accepted answer authorizes implementation; it does not mark that implementation complete. Existing owner decisions remain authoritative. Blank answers do not authorize a new direction. Core implementation can continue where accepted decisions already provide enough guidance.

## Reading order

A01-A06 cover storage/history dependencies; A07-A11 cover schema, merge, retention, views and indexing; A12-A15 cover algorithms, numerical behavior, pipelines and operating targets. A16 covers duplicate/retry policy; A17 is deferred Graphiti integration.

### Implementation priority while this workbook is being completed

See [the finish plan](roadmap-finish-priorities-2026-10-22.md) for per-task gates.
Prioritize **A01 and A03–A06** for upcoming state/history/temporal decisions,
then **A02 and A07–A11** for durable lifecycle, merge, retention, table and index
policies. A12–A13/A16 accompany relevant compatibility/verification work;
A14–A15 precede pipeline/service commitments; A17 stays behind core completion.
Module consolidation and accepted read/ownership/numerical correctness work can
continue now. This reading order does not supply answers or override established
choices; unanswered decision fields below remain unchanged.

## Established constraints

Immutable state-pinned views; stale writes conflict; one version per accepted checked batch; separate effective/recorded time; explicit optional recorded_by; inactive history retained until explicit purge; global identity; flexible properties with optional schema; one serialized writer first; undirected default; Neo4j/NetworkX comparison targets; Graphiti integration deferred behind core/numerical completion.

## Current boundary

Checked writers and DurableGraph implement documented automatic publication and journal recovery. Legacy Graph calls still use manual grouping. Effective-time reconstruction, pooled/meta persistence, pinned ordinary subgraphs/tables, compact checkpoints and storage migration remain unfinished.

## A01 - Pooled references and retained content

Roadmap: 10, 12, 26 / AR1, AR4, AR6. Timing: NEXT CORE DEPENDENCY.

**Current code / problem:** Stored subgraphs currently retain sets of IDs. Removing a node or edge edits those stored sets, and switching state replaces the backing pool. The durable journal rejects pooled references because the referenced content cannot yet be recovered reliably.

**Already agreed:** Old views must remain immutable. Inactive IDs remain historically inspectable until explicit purge.

**Suggested direction (provisional):** Represent a stored subgraph as an immutable selection pinned to its source state. Retain the values and topology needed by that state. Distinguish this from making a detached, independent copy.

**Example:** A meta-node groups Ada and Lin at version 4. Removing Lin at version 5 must not rewrite what the group contained at version 4.

### A01.1 - When a grouped subgraph is edited explicitly, should the meta-node receive a new reference, or should a separate group identity link its successive immutable selections?

Suggested: a new reference per revision; add stable group identity only where useful.

Decision (2026-10-22): Confirmed: subnode/meta property edits target their own entities. Group membership/source-pin revisions receive new immutable references; old references remain readable.

### A01.2 - When a subgraph is copied into another graph, should copying include its retained source history, or materialize only the selected state?

Suggested: materialize the selected state by default; history transfer is explicit.

Decision (2026-10-22): Confirmed: materialize selected state by default; retained history transfer is explicit.

Notes / conditions: ____________________

Code / record context:

- `src/storage/pool.rs :: store_subgraph / update_stored_subgraphs_remove_node`
- `src/api/graph.rs :: reset_to_snapshot`

## A02 - Checkpoints and storage evolution

Roadmap: 10, 12, 33 / AR6. Timing: BEFORE COMPACT RECOVERY.

**Current code / problem:** The journal recovers accepted inline writes and full-history backups. Opening and backup verification replay the full log, so their cost grows with retained operations. There is no compact checkpoint or format migration tool yet.

**Already agreed:** Use versioned storage, an append-only log and periodic snapshots. A checkpoint is an explicit immutable state, distinct from accepting a mutation.

**Suggested direction (provisional):** Use a manifest naming verified checkpoint and log segments. Publish a new manifest only after its files are synced. Start with explicit checkpoints and a configurable scheduling policy; choose default thresholds from measured recovery cost.

**Example:** A graph with ten million small updates should not require replaying every update just to open its latest state.

### A02.1 - What recovery-time and temporary-disk budgets should set the checkpoint schedule on your first real machine?

Suggested: measure first; set an opening-time target and a maximum log budget.

Decision (2026-10-22): Measure opening/recovery time. Defer checkpoint schedule and mandatory budget thresholds until MVP/first use.

### A02.2 - For a storage-format upgrade, is a verified side-by-side conversion acceptable before switching the manifest, even if it temporarily needs a second copy?

Suggested: side-by-side conversion, preserving the original until verification succeeds.

Decision (2026-10-22): Confirmed: verified side-by-side conversion; preserve original until verified. Checkpoint compression is optional, not its definition.

Notes / conditions: ____________________

Code / record context:

- `src/storage/journal.rs :: open / backup`
- `docs/api/durable-storage.md :: Current limits`

## A03 - Migrating legacy writes and names

Roadmap: 9, 12, 19 / AR3, AR5. Timing: BEFORE CHANGING DEFAULT APIS.

**Current code / problem:** Graph.writer and DurableGraph provide checked automatic publication. Legacy Graph methods still accumulate dirty changes for manual commit. The old commit name therefore has different semantics from the intended explicit checkpoint operation.

**Already agreed:** One accepted batch creates one version. Keep compatibility while introducing clearer public language.

**Suggested direction (provisional):** Add explicit version/checkpoint/state-switch names first. Keep manual grouping behind the existing legacy API during a documented transition. Never make an existing infallible constructor silently hide a publication error.

**Example:** Existing code adds two nodes and then commits once. A default change could otherwise create three versions or change what commit means.

### A03.1 - Should automatic publication become the default only in a major release, or remain selected through a durable/checked graph type indefinitely?

Suggested: explicit checked API now; change defaults only with a documented major release.

Decision (2026-10-23): Migrate defaults now: ordinary mutations publish checked changes automatically; commit denotes an explicit checkpoint. Preserve atomic failure and visible errors.

### A03.2 - Which existing scripts, notebooks or downstream APIs must be included as compatibility fixtures before renaming commit and checkout?

Suggested: collect your real examples and test them before deprecating any names.

Decision (2026-10-22): Audit real examples/fixtures; proceed with terminology/API changes now. A03.1 now confirms migrating automatic-publication defaults.

Notes / conditions: ____________________

Code / record context:

- `src/api/writer.rs :: GraphWriter`
- `src/api/graph.rs :: commit / checkout_branch`

## A04 - Temporal conflicts and topology

Roadmap: 11, 16 / AR3. Timing: BEFORE EFFECTIVE-TIME READS.

**Current code / problem:** Automatic deltas already retain engine-recorded time and per-change effective times in signed Unix microseconds. Effective-time reconstruction is still missing. Several writes may claim the same key and effective timestamp, and topology can be backdated too.

**Already agreed:** Resolve one value per key at a time. recorded_by is an explicit optional query cutoff. Each result pins its branch/version and times.

**Suggested direction (provisional):** Break equal-effective-time ties by accepted version order, not wall-clock recorded time. Use half-open validity intervals. Require active endpoints for an edge in a reconstructed view; retain all contributing records for inspection.

**Example:** Version 8 says score=3 effective at T; version 9 corrects it to 4 at the same T. An as-known-at-version-8 view still shows 3.

### A04.1 - Should later accepted changes win at the same effective timestamp, including corrections submitted with an earlier effective time?

Suggested: later accepted version wins; an explicit recorded cutoff can reveal the earlier value.

Decision (2026-10-22): Confirmed: explicit effective-time mutation events; later accepted version wins same-key/effective-time ties. Reads/computation do not create changes.

### A04.2 - If an edge is effective before one endpoint exists, should ingestion reject that combination or retain it but hide the edge outside endpoint validity?

Suggested: retain the record and expose only topology with active endpoints; provide a diagnostic.

Decision (2026-10-22): Reject unsupported endpoint validity, overriding the retain-and-hide suggestion. Validate proposed timelines atomically.

Notes / conditions: ____________________

Code / record context:

- `src/api/graph/publication.rs :: batch_publication_at`
- `src/temporal/snapshot.rs`

## A05 - Unset, null and deletion timelines

Roadmap: 7, 11, 16. Timing: BEFORE PROPERTY-HISTORY COMPLETION.

**Current code / problem:** Snapshots distinguish a present null from an absent property. Full unset operations and temporal deletion rules still need completion. Removing a graph entity and permanently purging its retained history are different operations.

**Already agreed:** Null is a value. Absence means no property assignment. Retained historical records are not implicitly erased by deletion.

**Suggested direction (provisional):** Record SetProperty and UnsetProperty separately. Let unset end an assignment from its effective time forward. Keep entity deactivation independent of property history, and require an explicit restore operation to reactivate an identity.

**Example:** A score is 5 at T1, explicitly null at T2, and unset at T3. These are three different observable states.

### A05.1 - After an unset at T3, should the key remain absent until another assignment, rather than falling back to its earlier value?

Suggested: absence persists until another set; never silently resurrect an old value.

Decision (2026-10-22): Confirmed: remain absent until another explicit assignment.

### A05.2 - When an entity is explicitly restored, should its previous properties return automatically, or must the caller choose a retained state/payload?

Suggested: restore from an explicit retained state, with an optional patch.

Decision (2026-10-23): Require an explicit retained state when restoring an entity. This corrects the earlier provisional default-restoration interpretation.

Notes / conditions: ____________________

Code / record context:

- `src/utils/strategies.rs :: create_change_set_resolved`
- `src/state/history.rs :: Delta`

## A06 - Native temporal types and precision

Roadmap: 7, 29. Timing: BEFORE FINAL PROPERTY FORMAT.

**Current code / problem:** Current integer properties are i64; floating properties and vectors are f32. The value bridge and journal preserve existing inline logical types. Native dates, times and durations, plus wider numeric types, remain unfinished.

**Already agreed:** Property values carry their type. Flexible mode permits type changes; enforced schema may constrain them.

**Suggested direction (provisional):** Keep existing values stable. Add explicit f64 and native temporal constructors if required. Separate UTC instants, local calendar dates and durations; encode them with distinct versioned tags rather than guessing from strings.

**Example:** A calendar birthday is not an instant in UTC. A numerical decomposition may need f64 even when stored embeddings use f32.

### A06.1 - Which first-class temporal types are essential for the next core release: UTC instant, date, local datetime, timezone-aware datetime, duration?

Suggested: UTC instant, calendar date and duration first; add local/timezone types with precise rules.

Decision (2026-10-22): Confirmed suggestion: UTC instant, calendar date and duration first; distinct typed/versioned representations.

Implementation (2026-10-23): 07.3b complete with exact microseconds, UTC normalization, distinct native types, typed schemas and lossless history/export/journal round trips. See [implementation record](temporal-properties-2026-10-23.md) for domains and verification.

### A06.2 - Should ordinary Python floats continue mapping to f32 for compatibility, with an explicit f64 constructor, or change default in a major release?

Suggested: preserve f32 compatibility and expose f64 explicitly before changing defaults.

Decision (2026-10-22): Confirmed: change ordinary Python float default to f64. Preserve existing typed data during migration; see answer record for memory discussion.

Notes / conditions: ____________________

Code / record context:

- `src/types.rs :: AttrValue`
- `src/storage/value_codec.rs :: StoredValue`

## A07 - Schema evolution across branches

Roadmap: 8, 10, 14. Timing: BEFORE ONTOLOGY MIGRATION.

**Current code / problem:** DurableGraph has a persisted global schema revision. A branch switch is checked against the current installed policy. Older branch data can therefore fail a policy introduced after that branch was created.

**Already agreed:** Schema is opt-in; flexible mode remains available. All matching labels apply. Semantic labels are independent of base/meta identity.

**Suggested direction (provisional):** Keep the current installed policy as the write gate, while recording migration provenance. Offer an explicit historical schema audit separately from enforcement. Never silently rewrite an old branch to fit a new policy.

**Example:** A new policy requires a name. A historical branch contains unnamed nodes and cannot currently become the working head under that policy.

### A07.1 - Should branches eventually carry their own schema revision, or should one installed graph-wide policy continue governing all working heads?

Suggested: retain a global write policy initially; add branch schema identity only with a concrete need.

Decision (2026-10-22): Tentative owner answer: “You could, I guess.” Global versus branch schema policy remains unresolved.

### A07.2 - When a migration cannot validate a whole graph, should it remain unpublished until complete, or use explicitly named intermediate compatible revisions?

Suggested: explicit compatible migration stages, each atomic and separately recorded.

Decision (2026-10-22): Confirmed: explicit compatible migration stages, each atomic and recorded.

Notes / conditions: ____________________

Code / record context:

- `src/storage/journal.rs :: set_schema / checkout_branch`
- `src/api/graph/schema.rs :: check_schema_at`

## A08 - Rewiring and merge conflict rules

Roadmap: 4, 13, 14. Timing: BEFORE REVERT AND THREE-WAY MERGE.

**Current code / problem:** Edge IDs preserve parallel instances. Ordinary edge upsert rejects changed endpoints. State diff is implemented, but explicit rewiring and three-way merge are not complete. Merge needs deterministic conflict boundaries.

**Already agreed:** Parallel facts may coexist. Caller-selected resolution governs conflicts; old recorded values remain inspectable.

**Suggested direction (provisional):** Use a dedicated RewireEdge event preserving edge ID. Merge independent property keys automatically, report competing writes to the same key, and treat delete/edit or incompatible rewires as conflicts. Record the chosen resolution.

**Example:** One branch rewires edge 7; another changes its weight. These can combine if both endpoint and schema rules remain valid.

### A08.1 - If two branches rewire the same undirected edge to reversed but equivalent endpoints, should merge coalesce them or report a stored-orientation conflict?

Suggested: coalesce equivalent undirected endpoints; directed endpoint order remains significant.

Decision: ____________________

### A08.2 - Should independently edited properties on the same entity merge automatically, with conflicts limited to overlapping fields and lifecycle changes?

Suggested: merge disjoint keys; report delete/edit, same-key and incompatible topology conflicts.

Decision: ____________________

Notes / conditions: ____________________

Code / record context:

- `src/api/graph/identity.rs :: edge upserts`
- `src/state/history.rs :: CommitDiff / Delta`

## A09 - Retention, purge and audit policy

Roadmap: 15, 33 / AR1, AR6. Timing: BEFORE DESTRUCTIVE MAINTENANCE.

**Current code / problem:** Identity aliases and allocated IDs remain reserved. Full-journal backups retain history. Compaction and garbage collection are not finished; removing a branch name alone does not define which storage may safely be reclaimed.

**Already agreed:** Deletion and purge are explicit separate operations. Retained branches, states and views must keep their referenced history.

**Suggested direction (provisional):** Trace reachability from branches, named checkpoints and active readers. Use a previewable purge plan, a grace period and verified replacement segments. Never reclaim referenced content as an incidental side effect of optimization.

**Example:** Deleting a branch must not remove a shared ancestor or data still needed by an open immutable view.

### A09.1 - Should named checkpoints or a separate audit pin keep history even after all branch names referencing it are deleted?

Suggested: named checkpoints pin history; optional explicit audit pins use the same retention mechanism.

Decision (2026-10-23): No extra permanent checkpoint/audit pins. Purge remains explicit and must preserve shared branches and open immutable views.

### A09.2 - What grace period and backup requirement should protect an explicit destructive purge, and should strict audit retention be an optional graph policy?

Suggested: configurable grace period and an optional strict audit policy; preview the affected states.

Decision (2026-10-23): No mandatory grace period or backup. Explicit purge must preserve shared branches and open immutable views. Current scope is a data science library, with production retention policies deferred.

Notes / conditions: ____________________

Code / record context:

- `src/state/ref_manager.rs`
- `src/api/graph.rs :: gc_history / optimize`

## A10 - Table identity and explicit writeback

Roadmap: 26, 27 / AR4. Timing: BEFORE CONNECTED TABLE MIGRATION.

**Current code / problem:** Ordinary subgraphs retain IDs but can read live parent values; graph tables currently materialize columns. Joins can duplicate one source entity across several rows. Writing such a result needs a clear mapping back to graph identity.

**Already agreed:** Views are immutable and state-pinned. A join is a result until an explicit write. Stale writes conflict and successful writes produce a new state.

**Suggested direction (provisional):** Carry hidden source identity and state pins through table operations. Write back only unambiguous editable property columns. Keep topology edits on explicit graph operations. Require an explicit rule when rows map repeatedly to one target.

**Example:** Joining a person to three facts produces three rows for the same node. Assigning three different ages cannot be an implicit last-row-wins update.

### A10.1 - Should ambiguous or aggregated join results reject writeback unless the caller supplies an explicit target mapping or reduction?

Suggested: reject ambiguous writeback; allow explicit mapping/reduction with validation.

Decision (2026-10-22): Conditional yes: reject repeated-target/aggregated ambiguity without explicit mapping/reduction. Joins duplicate rows, not graph entities; see answer record.

### A10.2 - Should a table writeback be able to update nodes and edges together in one mixed batch, or should each source kind require a separate command?

Suggested: allow one mixed atomic writeback when every row has an unambiguous source identity.

Decision (2026-10-22): Confirmed: one mixed atomic node/edge writeback batch.

Notes / conditions: ____________________

Code / record context:

- `src/subgraphs/subgraph.rs`
- `src/storage/table/base.rs`

## A11 - Indexes and explicit rebuild control

Roadmap: 18, 32 / AR2. Timing: BEFORE SECONDARY INDEXES.

**Current code / problem:** The managed external-ID index exists. Label/type selection still scans columns, and broader property/time indexes remain planned. The owner requested global indexes and no rebuild unless explicit; derived recovery/cache work needs careful terminology.

**Already agreed:** Indexes are global rather than per label by default. Updates consume the same typed changes as graph state.

**Suggested direction (provisional):** Maintain declared indexes incrementally at publication. Keep user-requested rebuild distinct from cold recovery. An absent or unusable optional index should have explicit status and a correct scan fallback, never silently return incomplete results.

**Example:** An optional score index is missing after restore. Results must stay correct even if the engine scans until you explicitly rebuild it.

### A11.1 - Does the explicit-rebuild rule allow automatic restoration of persisted index state during open, while leaving any full reindex operation manual?

Suggested: restore/replay is automatic; deliberate full reindexing remains explicit.

Decision: ____________________

### A11.2 - When an optional index is unavailable, should queries fall back to scans with an explain flag, or fail unless the caller opts into a scan?

Suggested: correct scan fallback with inspectable execution status and an optional require-index mode.

Decision (2026-10-22): Fail when the requested index is unavailable; scanning must be explicitly selected, not a silent fallback.

Notes / conditions: ____________________

Code / record context:

- `src/api/graph/identity.rs :: ExternalIdentityIndex`
- `src/query/query.rs; src/temporal/index.rs`

## A12 - Traversal and algorithm edge cases

Roadmap: 3, 21, 28 / AR2. Timing: DURING CORE ALGORITHM AUDIT.

**Current code / problem:** Core direction and parallel-edge paths have repairs. Individual algorithms still need complete rules for self-loops, ties, components and weight domains. A general undirected default does not settle each directed algorithm definition.

**Already agreed:** Keep the basic algorithm set as core. Preserve edge-instance identity and pin results to selected state. Dijkstra requires nonnegative weights.

**Suggested direction (provisional):** Use explicit weak/strong component methods for directed graphs. Count an undirected self-loop twice for degree but once in edge iteration. Resolve equally weighted path ties by a documented deterministic rule, without merging parallel edges.

**Example:** A directed A-to-B edge places the nodes in one weak component but two strongly connected components.

### A12.1 - Should an ambiguous directed components call require a weak/strong mode, or preserve a documented legacy default and add explicit methods?

Suggested: preserve compatibility with a clear default and add explicit weak/strong names.

Decision (2026-10-22): Confirmed suggestion: retain a documented compatibility default and add explicit weak/strong modes.

### A12.2 - Should shortest-path APIs return one deterministic optimum by default, with all equal optima available only through a separately bounded operation?

Suggested: one deterministic path by default; all-path enumeration requires caller bounds.

Decision (2026-10-22): Confirmed: one deterministic optimum by default; all-optima enumeration is separate and bounded.

Notes / conditions: ____________________

Code / record context:

- `src/query/traversal.rs`
- `src/algorithms/mod.rs; src/state/topology/csr.rs`

## A13 - Numerical shapes, sparse data and errors

Roadmap: 29. Timing: BEFORE NUMERICAL COMPLETION CLAIMS.

**Current code / problem:** Groggy has substantial array and matrix code. The roadmap identifies placeholder or incorrect identity, axis standard deviation, eigendecomposition and SVD behavior. CSR graph adjacency is not yet a full sparse numerical contract.

**Already agreed:** Complete Groggy's existing numerical design, using in-house implementations. NumPy API parity and Graphiti vector search are not the immediate goal.

**Suggested direction (provisional):** Specify shape/dtype rules per operation, accumulate sensitive reductions in f64, and expose convergence/unsupported-domain errors. Keep sparse-to-dense conversion explicit and test mathematical reconstruction/residuals against trusted references.

**Example:** Broadcasting two differently sized vectors or accidentally densifying a huge sparse matrix must not succeed with an unexplained shape or memory cost.

### A13.1 - Should the first repaired numerical release require exact-shape elementwise inputs plus explicit scalar operations, adding broader broadcasting only where already promised?

Suggested: exact shapes and explicit scalars first; preserve any verified existing broadcast promises.

Decision: ____________________

### A13.2 - What precision guarantees are essential: f64 computation/output support, per-operation tolerances, and explicit errors for nonconvergence or unsupported matrix domains?

Suggested: explicit f64 support, documented tolerances and residual-based verification; no placeholder answers.

Decision (2026-10-22): Confirmed f64 floating-point defaults/support (“S64/X64” clarified as f64); keep explicit errors and tolerance/residual verification.

Notes / conditions: ____________________

Code / record context:

- `src/storage/matrix/matrix_core.rs :: std_axis / svd / eigenvalue_decomposition`
- `src/storage/array/base_array.rs`

## A14 - Pipeline results, naming and caching

Roadmap: 30 / AR5. Timing: AFTER PRIMITIVE REPAIRS.

**Current code / problem:** The pipeline/builder already exists and should be extended from its working behavior. Intermediate result names, cache ownership, provenance and whether a run becomes a graph entity remain unresolved.

**Already agreed:** Pin each run to an input state. Keep pipeline execution on the corrected core and numerical operations.

**Suggested direction (provisional):** Give a run a typed result map and inspectable plan. Cache pure stages by graph/state, operation, parameters and implementation version. Keep recording run provenance opt-in so an ordinary read does not mutate its own input graph.

**Example:** Running a path analysis twice on state 12 can reuse a result. Running it on state 13 cannot reuse a state-12 answer as if it were current.

### A14.1 - Should pipeline intermediates be explicitly named typed values, or should names be optional labels over generated stable stage identifiers?

Suggested: generated stable stage IDs with optional readable names and typed outputs.

Decision: ____________________

### A14.2 - Should a run be transient by default, with explicit persist-result/provenance operations rather than automatically creating history entities?

Suggested: transient by default; persisting results or provenance is explicit.

Decision (2026-10-22): Confirmed: transient runs by default; result/provenance persistence is explicit.

Notes / conditions: ____________________

Code / record context:

- `src/algorithms/builder.rs`
- `src/algorithms/pipeline.rs`

## A15 - Scale targets and service boundaries

Roadmap: 32, 33. Timing: BEFORE SCALE AND MULTIUSER CLAIMS.

**Current code / problem:** The ambition ranges from small graphs to a billion entities. DurableGraph currently has one OS writer lock and replay-based recovery; Python holds the GIL during I/O. Multiuser service coordination is still future work.

**Already agreed:** Support embedded use and a service wrapper. Begin with one serialized writer per graph and immutable snapshot readers.

**Suggested direction (provisional):** Pick one real machine and workload first. Measure ingest rate, p50/p95 retrieval latency, resident memory, retained history size and restart time. Expose queue limits and conflicts at the service boundary; keep core identity separate from user authentication.

**Example:** One million nodes with 384-component f32 vectors already need about 1.54 GB for vector payloads alone, before edges, history or indexes.

### A15.1 - What first benchmark fixture should we optimize for: node/edge counts, properties, vector dimensions, history depth, RAM and storage device?

Suggested: start with a measured million-node tier, plus tiny and stress fixtures; set budgets from your hardware.

Decision (2026-10-22): Add compute elapsed time and end-to-end wall time; label CPU time separately. Exact workload/hardware/budgets remain open.

### A15.2 - Which limits should the first service enforce: maximum queued writes, timeout/cancellation rules, graph access boundaries and acceptable stale-read behavior?

Suggested: serialized bounded write queue, explicit timeouts, state-pinned reads and authorization in the wrapper.

Decision: ____________________

Notes / conditions: ____________________

Code / record context:

- `src/storage/journal.rs :: DurableGraph`
- `src/utils/config.rs`

## A16 - Duplicate inputs and retry identity

Roadmap: 5, 6, 9, 10. Timing: POLICY REVIEW / RETRY DESIGN.

**Current code / problem:** Current bulk setters and managed upserts reject duplicate targets within one call, even if repeated payloads match. That is a recorded implementation policy for review. A synced batch can also survive a crash before the caller receives its acknowledgement.

**Already agreed:** Invalid batches fail atomically. Separate identical managed upsert calls are idempotent. A missing acknowledgement does not prove that a durable write was rejected.

**Suggested direction (provisional):** Keep duplicate assignments within a batch as explicit errors; offer caller-side normalization separately. For mixed-write retry safety, consider an optional graph-scoped request key whose original result is returned on identical replay and whose reuse with different contents fails.

**Example:** Two rows assign different scores to node 7 in one batch. Separately, a network retry repeats an already accepted add-node batch after its reply was lost.

### A16.1 - Should bulk setters and upserts keep rejecting duplicate targets inside one call, or coalesce identical repeated assignments under an explicit option?

Suggested: reject by default; any coalescing is explicit and conflicting values still fail.

Decision (2026-10-22): Recorded interpretation: reject duplicates by default; coalescing identical repeats is explicit and conflicts still fail.

### A16.2 - Should generic mixed batches gain an explicit durable request key for safe retries, or should callers reconcile accepted state and external IDs themselves?

Suggested: optional graph-scoped request key with payload checking and replay of the original result.

Decision: ____________________

Notes / conditions: ____________________

Code / record context:

- `src/api/batch.rs :: GraphBatch preparation`
- `src/storage/journal.rs :: apply_batch_with_schema`

## A17 - Deferred Graphiti compatibility

Roadmap: 17, 20, 22-25. Timing: DEFERRED - NOT A CORE BLOCKER.

**Current code / problem:** Graphiti ingest, its driver and lexical/vector/hybrid retrieval are deferred until the Groggy core is ready. A compatibility target and lossless interchange fixture have not been selected. These questions are parked for later.

**Already agreed:** Keep raw evidence/provenance, support multiple embeddings under named attributes, start vector retrieval with exact search, and prefer operation interfaces over a Cypher translator.

**Suggested direction (provisional):** Freeze a specific Graphiti release only when integration work begins. Define a lossless fixture for episodes, entities, facts, IDs, groups, times and embeddings; then test idempotent ingest and explicit unsupported-operation errors.

**Example:** A corrected fact must preserve its earlier evidence and recorded/effective history rather than disappear during adapter translation.

### A17.1 - When integration resumes, which Graphiti release and representative ingest/search fixture should define the compatibility target?

Suggested: pin a released version at integration start and run its actual operation tests.

Decision (2026-10-22): Use a synthetic documented test example initially; select the real use case/release later. Graphiti remains deferred.

### A17.2 - What adapter-level identity rule distinguishes correcting an existing fact from adding a parallel fact, especially when its endpoints change?

Suggested: preserve explicit source fact IDs; endpoint changes require a declared correction or a new identity.

Decision: ____________________

Notes / conditions: ____________________

Code / record context:

- `notes/planning/graph-engine-implementation-roadmap.md :: deferred items`
- `notes/planning/graph-engine-design-responses-2026-10-20.md :: integration sequencing`

## Source records

- [Implementation roadmap](graph-engine-implementation-roadmap.md)

- [Owner answers and accepted recommendations](graph-engine-design-responses-2026-10-20.md)

- [Durable-storage guarantees and limits](../../docs/api/durable-storage.md)

The earlier decision record is chronological. Later implementation entries supersede earlier statements about what is still missing. The old focused-follow-up questions are not all still open. This appendix does not reopen the already accepted versioning, immutable-view or recorded-time model.

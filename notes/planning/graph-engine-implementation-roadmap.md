# Groggy graph engine implementation roadmap

Status: **staged implementation plan; compare each item against the [current-state review](../architecture/current-state-review-2026-10-19.md) and the [decision record](graph-engine-design-responses-2026-10-20.md).** This list originally overstated missing foundations. Many core types, APIs, and architectural choices already exist. Each item should begin by identifying what Groggy already implements; the work is to repair a demonstrated mismatch or add a clearly needed capability, not redesign settled foundations.

The [flat implementation checklist](graph-engine-flat-checklist.md) breaks these items into individually checkable slices, including completed foundations, remaining core work, deferred integrations, and architecture changes.

The owner's first-pass answers, accepted recommendations, and remaining open choices are recorded in the [2026-10-20 design responses](graph-engine-design-responses-2026-10-20.md). The companion [open-questions appendix](open-questions-appendix-2026-10-20.md) collects remaining choices as A01-A17, with provisional suggestions and the staged roadmap references.

Groggy's original vision is a Rust graph engine with a Python API, separate topology and columnar properties, tracked changes, connected graph and table views, and graph analytics. See [Origins & Design](../../docs/concepts/origins.md), [Design Decisions](../../docs/appendices/design-decisions.md), and the [temporal extensions plan](temporal-extensions-plan.md).

The concrete integration target is [Graphiti](https://github.com/getzep/graphiti/blob/main/README.md): Graphiti produces episodes, entities, facts, provenance, time fields, and embeddings; Groggy stores and queries them. This plan does not assume Groggy performs Graphiti's LLM extraction. Graphiti's [driver interface](https://github.com/getzep/graphiti/blob/main/graphiti_core/driver/driver.py) defines the eventual backend compatibility target.

The stages describe related work, with Graphiti-specific integration and retrieval deferred until the Groggy core is finished. Each item may call for verification, repair, completion, or a new capability. Item numbers remain continuous so individual work items are easy to reference. The owner's "number 25" in the later flat list means the Graphiti ingest adapter, numbered 20 here; its "number 27" means lexical search, numbered 22 here. The related Graphiti driver is numbered 25 here.

## Immediate next implementation slice

1. **Completed on this branch:** the existing `set_node_attrs`, `set_edge_attrs`, and column setter paths collect all inactive or unknown IDs and duplicate assignments before writing, with attribute and input positions in the error. Duplicate assignments are rejected as an implementation choice pending owner review. Vectorized pool writes and current Rust/Python entry points remain in use. Python conversion errors now aggregate with read-only Rust preflight diagnostics before rejected calls return.
2. **Explicit batch API implemented in Rust and Python:** `GraphBatch` combines node/edge insertion, property updates, and removals. It validates semantic errors against the proposed resulting state before calling existing bulk writers. This establishes all-or-nothing behavior for rejected input in a single-process graph; crash-safe rollback and durable publication belong to the storage milestone.
3. **Batch publication implemented; legacy migration pending:** `apply_batch` checks the expected state and publishes one typed delta and state on success. Existing `commit`/`checkout` and direct mutation methods remain callable with their current manual-commit behavior. Route those direct methods through one-operation batches only with a compatible transition and explicit tests.
4. Optional schema enforcement now checks the complete proposed batch result before allocation, using the same rules as explicit auditing. It validates new and unchanged surviving entities, combines label/property changes, and excludes explicit/incident removals. Ordinary Graph policy is per call/session; the checked writer publishes direct/bulk methods, and DurableGraph persists an installed policy.
5. Rust and Python tests now cover multiple semantic errors with zero writes, stale-state rejection, one successful state publication, and an unchanged older snapshot. Journal failure injection now covers write, sync, rollback and sync-before-memory boundaries; future secondary indexes still need equivalent checks.

## Architecture refactor thread (cross-stage)

These are changes to the *existing* engine's internal responsibilities, not a replacement engine or a new prerequisite contract. Make each change behind the current Rust and Python APIs. Record current behavior with focused conformance tests first, distinguish intentional behavior changes from structural refactors, and measure hot paths before and after. The companion design workbook prompts decisions on ownership, view lifetime, and history representation before implementation.

- **AR1 — One authority for active topology (items 1–3, 18, 28).** `GraphSpace` owns live node and edge membership; `GraphPool` owns allocated slots and columnar values. Route public reads, indexes, views, and algorithms through the same active-state check. Retain tombstones or stored slots where history needs them, without treating allocation as liveness.
- **AR2 — One adjacency and direction policy (items 3–4, 21, 28).** Build a narrow internal topology read interface over the existing topology machinery. It must return edge IDs and endpoints consistently for directed and undirected graphs, parallel edges, self-loops, and historical snapshots. Remove algorithm-local choices about reverse edges. Avoid virtual dispatch on hot paths unless benchmarks justify it.
- **AR3 — One coordinated mutation and typed change record (items 5, 9, 11–15).** A successful mutation updates live state and records the actual before and after values, identity, and topology change together. Commits, diffs, reset/revert, and snapshot reconstruction consume that record. Establish failure and rollback behavior before exposing transactions; do not infer history later from column offsets or placeholder strings.
- **AR4 — Explicit view and table ownership (items 12, 26–27).** Implement the accepted pinned, immutable view semantics: a later parent mutation does not change an older view. A write routed through a view checks its expected version and returns a new view or conflict. Decide whether graph tables are borrowed views or materialized results where current code and documentation disagree. Preserve existing public methods through a compatible migration.
- **AR5 — A smaller coordinator and thin adapters (items 19, 21, 25, 30).** Keep `Graph` as the stable public facade while moving mutation coordination, topology reads, history, and table conversion into cohesive internal components. Keep FFI limited to translation, and put Graphiti episode, provenance, UUID, and temporal mapping in an adapter above the core. Split crates or files only where this yields a real responsibility boundary.
- **AR6 — Honest persistence boundary (items 10, 20, 25, 33–34).** Decide whether Groggy is an in-memory engine with export or Graphiti's durable system of record. If durable, persist snapshots and the typed change record with versioned recovery and migration. Keep existing export functions distinct from a transactional storage guarantee, and make incomplete or no-op public operations explicit rather than silently succeeding.

## Current completion order — 2026-10-22

Module consolidation is now implementation work in flat-checklist **AR7.1–AR7.11**,
with the inventory complete and module moves still open. Preserve the original
34 roadmap items and existing functionality/imports. Start with matrix and shared
array/numeric foundations, close the numerical gates, then entity/view ownership,
exact state/history/time, durable lifecycle, query/table/algorithm/pipeline surfaces
and the core release/service gates. Verification runs throughout; Graphiti-specific
integration remains behind the core milestone and stays on the full roadmap.

The [finish priorities](roadmap-finish-priorities-2026-10-22.md) map checklist IDs to
dependencies and specific workbook questions. Module moves and already-decided
correctness work can proceed while the owner completes the workbook. Blank
answers do not settle new mutation, temporal, merge, purge or service policy.

## Stage 1 — Verify and repair core invariants

Preserve Groggy's established graph model and make its behavior consistent across public APIs. The one-node, zero-edge graph is the smallest meaningful graph; an empty `Graph` is also supported as a construction state. This stage tests and repairs implementation against the existing contract.

1. **Verified baseline on this branch — codify the existing engine contract.** The focused Rust and Python conformance checks cover empty construction, default and explicit direction, and the one-node, zero-edge minimum. The [contract baseline](engine-contract-conformance.md) records established rules and routes known gaps to their later items. Extend the suite as each contract area is repaired rather than treating this baseline as proof that all APIs already conform.
2. **Core live-state read audit completed on this branch.** `GraphSpace` membership gates graph attribute and topology reads, including `incident_edges`; node/edge entity handles and subgraph attribute helpers now route through checked `Graph` reads instead of retained pool slots. Focused Rust and Python checks cover removed entities. Low-level `GraphPool` inspection remains separate from a live graph view; algorithm-specific traversal and direction issues remain in items 3 and 28.
3. **Core direction paths repaired on this branch.** Edge lookup/filtering, BFS/DFS, shortest path, all paths, adjacency matrices, and pathfinding CSR cache now apply graph direction. Focused Rust and Python checks include a one-way edge and an undirected control. Algorithm-specific definitions, including weak/strong components, remain under item 28; parallel-edge path identity remains under item 4.
4. **Edge instance identity verified on this branch.** Parallel edges and self-loops retain distinct stable IDs across storage and recorded snapshots. Shortest path now returns the edge instance selected by its weight, and all simple paths retain parallel edge instances rather than reconstructing an arbitrary edge afterward. Rust and Python checks cover these rules. Explicit edge relationship types belong to item 8; endpoint rewiring belongs to item 13.
5. **Mutation validation repaired on this branch.** Creation parses properties and checks reserved node fields before allocating IDs. Bulk property writes and removals reject inactive, unknown, and duplicate targets before writing. Conversion and semantic errors aggregate across decodable rows and columns, retaining original input positions. Rust/Python tests verify rejected calls preserve state and IDs. Zero-count Rust node allocation now returns an empty list without changing IDs or activating nodes; the pool also handles zero explicitly. External-ID uniqueness, schema, and automatic publication follow in items 6, 8, and 9. See the [conformance record](engine-contract-conformance.md).
6. **Managed external identities implemented on this branch.** Rust/Python node and edge upserts preserve IDs, reject conflicts before allocation, and make identical payloads a no-op. One namespace spans entity kinds, groups and retained branches. Renames/removal retain reservations; snapshots read pinned identities. Legacy UID mappings reject ambiguity. See the [identity API](../../docs/api/external-identity.md). Durable reservations and purge remain items 10 and 15.
7. **Complete the property model — value bridge repaired; still in progress.** Rust/Python graph reads, columns, table cells and recorded snapshots now preserve existing scalar, binary, typed-vector and structured values. Equality, ordering and hashing agree across storage optimizations; corrupt compressed values raise on decoding. Explicit null differs from absence in snapshots, and typed before/after records retain type changes. See [property values](../../docs/api/property-values.md). Versioned active exports now preserve inline logical types and float bits, with strict decode errors and legacy reads. Meta-node/pool references reject explicitly until pool storage exists. Native temporal types remain; current numeric property widths are i64/f32. Preserve all intended types through Rust, Python, history and disk before closing this item.
8. **Semantic kinds and explicit schema audit implemented; enforcement open.** Multiple versioned node labels remain independent of base/meta storage identity; edges carry an optional named relationship type. Shape checks run before checked creation, bulk updates, upserts and mixed batches. Optional schema auditing reports required/nullable/type violations for every matching label/type without mutating state. See [kinds/schema API](../../docs/api/kinds-schema.md). Item 9 adds schema-controlled batch publication; graph-wide enforcement still needs a compatible checked direct-write path; schema policy versioning and ontology migration remain open.

## Stage 2 — Durable writes and graph history

Make changes atomic and persistent, then complete the Git-like history promised by the original design.

9. **Implement atomic write batches and transactions.** Rust/Python `GraphBatch` now publishes one in-memory state for a validated mixed write using existing bulk paths. Optional Rust/Python schema-controlled publication now checks the whole proposed resulting state before allocation, including unchanged survivors and implicit edge removals. On ordinary Graph, policy remains per call/session; DurableGraph can install a persistent global policy. A checked `writer` session now publishes direct/bulk operations through batches while retaining legacy APIs. Automatic publications carry engine-recorded time, batch effective-time defaults and validated per-change overrides. Mixed batches and checked upserts now coordinate managed identity columns/reservations with publication. The checked publication milestone now includes synced durable journal/replay, global installed-schema enforcement and expected-head checks. Legacy manual mutation remains explicitly separate; pooled/meta mutations and service concurrency remain outside this checked surface. Publication records deltas and one engine-assigned recorded time for the batch.
10. **Build durable storage and recovery.** Version 2 active graph export now preserves inline logical property types and rejects unsupported pooled references; it is not a complete engine backup. DurableGraph now provides a versioned checksummed CBOR journal preserving checked inline writes, allocation counters/reservations, schema revisions, configuration, branches, typed history and original clocks. Restart, partial records, corruption, subprocess exit and injected I/O failures are tested in Rust/Python. Full-journal backups now use a verified synced temporary file and publish without overwriting; they reopen with the ordinary recovery path. Stored membership and backing pool data now survive deactivation, reset and branch switches; pooled/meta serialization is still rejected. Remaining: durable subgraph/meta contents, importing existing graphs, periodic checkpoints, compaction, format upgrades and bounded recovery. See [storage guarantees/limits](../../docs/api/durable-storage.md).
11. **Complete property history.** The first slice made commits resolve typed node and edge values and added tests for historical reads. Finish coalescing, deletion, unset/null distinctions, effective and recorded times, and all supported value types before treating the history contract as complete.
12. **Make snapshots and state switching exact.** The first slice preserved direction and ID allocation across snapshots and branch switches. Retained subgraph membership, endpoints and old column indices now survive state switching because the selected space is rebuilt without replacing the pool. Missing meta referents reject before allocation; detached snapshot materialization rejects absent backing references. Extend reconstruction to all configuration and portable/pinned referents, bound repeated-switch column allocation, and verify state-pinned reads independently of the working head. See [retained subgraphs](../../docs/api/retained-subgraphs.md). Treat an explicit `commit` as a requested immutable checkpoint of the current published state after a compatible migration; automatic batch versions and deltas remain the source for reconstruction between checkpoints.
13. **Complete diff, revert, and reset semantics.** `reset_hard` restores the recorded state and preserves allocated IDs. Rust and Python `diff_commits` now compare reconstructed states in either direction, including typed property before/after values and topology, with deterministic ordering. Revert, endpoint rewiring, and destructive pruning policy remain.
14. **Finish branches and merges.** Implement a three-way merge for node, edge, and property changes, with conflict reporting and resolution. This follows reliable commits and snapshots.
15. **Define retention and garbage collection.** Preserve referenced commits and provenance while allowing compaction of unreachable history. Existing `gc_history` and `optimize` are no-ops.

## Stage 3 — Temporal knowledge graph ingest

Complete Groggy's temporal and provenance primitives. The Graphiti-specific ingest adapter at item 20 is deferred until the core graph, history, view, table, query, and analytics work is finished.

16. **Add effective time separately from recorded time.** Graphiti facts have valid and invalid times, and property changes also need an effective timestamp. Use an explicit per-change time when supplied, otherwise the batch's publication clock; the engine separately records when the batch was accepted. Reconstruct a branch-scoped graph view for “what was true at T”, using the latest published version on that branch at view creation. Apply a `recorded_by=U` cutoff only when the caller explicitly requests “what did we know by U?”; the resulting view remains pinned. This uses the same immutable view and graph-to-table interface as a present-state selection; derived subgraphs and tables inherit the state/time pins. The current `snapshot_at_timestamp` selects only a whole-second commit timestamp across all commits and is insufficient for these reads.
17. **Model episodes and provenance directly.** Store raw episode identity and content, the entities and facts derived from it, and links back to the source. Invalidate or supersede facts without erasing their evidence.
18. **Extend primary and secondary indexes.** Item 6 adds an allocation-lifetime external-ID index with active-state/group checks; historical identity lookup currently scans snapshot properties. Add group, label, edge type, endpoint, property equality/range, and time-window indexes. Preserve correctness under update, deletion, state switching and recovery; persist allocation reservations under item 10.
19. **Expose one coherent Python API.** Mirror the Rust contract for IDs, transactions, queries, temporal reads, properties, and errors. Verify the built Python extension end to end; passing Rust tests does not establish Python behavior.
20. **Deferred — build a Graphiti data adapter.** After the Groggy core is finished, prove a round trip for Graphiti episodes, entities, facts, embeddings, attributes, UUIDs, group IDs, provenance, and time fields. Re-ingesting the same episode must be idempotent; invalidating a fact must preserve its history.

## Stage 4 — Queries, retrieval, and Graphiti driver

Finish Groggy's native graph query layer. Lexical, embedding, and hybrid retrieval at items 22–24 are deferred until the vector/matrix core is correct and the broader Groggy core is finished. The Graphiti driver at item 25 follows the deferred ingest adapter.

21. **Create a usable graph query layer.** Provide indexed node and edge selection, directed pattern matching, bounded traversal, property filters, temporal filters, ordering, and pagination. Define results independently of internal storage layout.
22. **Deferred — add full-text search.** After the core, index retained text such as episode content, entity names and summaries, and fact text; support ranked keyword retrieval and group filtering.
23. **Deferred — add embedding search.** After the vector/matrix library is correct and measured, store embeddings with dimensions and model metadata, support nearest-neighbor search, and keep search indexes synchronized with updates and persistence. This is distinct from completing the core vector library.
24. **Deferred — combine retrieval modes.** After the core, merge keyword, vector, temporal, and graph-neighborhood results with inspectable scores. Graphiti's search relies on this combination.
25. **Deferred — implement a Graphiti storage driver.** After the core and ingest adapter, implement Graphiti's required query, session, transaction, indexing, graph-operation, and search behavior. Serve operations natively where possible and implement the required query behavior for remaining calls; run Graphiti ingest and search flows against it.

## Stage 5 — Connected computing and analytics

Complete Groggy's broader structural-computing vision on top of a correct graph core.

26. **Make connected views trustworthy.** Implement the agreed immutable, state-pinned view contract. Current subgraphs can retain IDs while reading live parent values; audit and migrate those paths. Writes routed through a view must check its expected version and return a new view or conflict.
27. **Complete graph-to-table and table-to-graph operations.** Preserve the structural-computing vision: columnar filtering, joins, aggregations, bulk updates, and conversions should remain connected to graph identity and history rather than becoming disconnected copies.
28. **Correct core traversal and graph algorithms.** Validate start and end nodes, respect direction and active state, handle weights correctly, and reject unsupported inputs. Core BFS/DFS/path methods now reject inactive/unknown starts and endpoints. Checked Dijkstra paths reject invalid weights, decode compact integers, preserve reverse/parallel minimum distances, and retain tiny improvements. Algorithm/pipeline Dijkstra and weighted closeness share validation; f32 result overflow rejects before persistence. A* now shares weight/selection validation, rejects ambiguous markers and invalid/inconsistent heuristics, and preserves tiny improvements. Node betweenness now validates strictly positive weights, preserves shortest edge-instance multiplicity, fixes directed raw scaling, and rejects numerical overflow. Girvan–Newman now validates its undirected/positive-length domain, removes individual edge instances, and scores every visited partition against original topology, including the final allowed removal. Louvain/Leiden local moves now use actual modularity resolution, retain loops/parallel instances, and validate undirected/live selections and parameters. Leiden seeded ordering is repeatable without global RNG changes. Louvain now executes bounded multilevel aggregation with count-weighted coarse edges/loops, original-ID projection and objective-preservation checks; registry/Python defaults allow ten levels, with explicit one-level mode preserved. Leiden now runs bounded fast moves, constrained stochastic refinement and refined aggregation while carrying the broader partition into each quotient; connected output, preserved objective and visit bounds are tested. Scalar Python algorithm configuration now preserves f64 precision. Infomap now evaluates the two-level map equation with recorded uniform-node teleportation, bounded stationary flow, real seeded trials and independent entropy checks; hierarchical Infomap/acceleration remain open. LPA now uses immutable interned label values, real neighbor votes, explicit tie/direction rules, local seeded order, checked controls/live selections and nonpersistent component output, with exhaustive small-graph references. Connected components now uses consistent weak/undirected and strong direction policies, validates before cache reads, supports cancellation and verifies iterative kernels against exhaustive reachability. PageRank now honors undirected reversal and personalized dangling mass, checks controls/personalization/live selections, preserves explicit bounded-approximation diagnostics and returns nonpersistent f64 scores, with exhaustive small-graph linear-solve references. Closeness now handles undirected traversal consistently, exposes outgoing/incoming and harmonic/classic normalized/raw options, validates strictly positive lengths, preserves cache direction and returns nonpersistent f64 rows; exhaustive Floyd-Warshall references cover all supported variants. BFS/DFS wrappers now validate a unique selected presence marker and live/closed inputs, support cancellation, return exact nonpersistent integer rows and define deterministic DFS order independently of shared CSR history. Dijkstra now also validates one source marker before cache access, checks cancellation in heap traversal/before output and returns ordered nonpersistent f64 distances; exhaustive all-source zero-weight multigraph references cover its wrapper. A* now also returns exact nonpersistent path rows and f64 cost diagnostics, validates state/weights before cache reads and honors cancellation for trivial paths; exhaustive endpoint-pair references cover the follow-up. PCA now shares one checked matrix/visualization caller with scaled preprocessing/scatter, explicit singleton/constant behavior, retained labels and a thin Python projection method; analytic/NumPy SVD geometry checks cover its supported domain. Spectral now accumulates loop/parallel adjacency, preserves isolate zero modes, rejects directed input, separates threshold from normalization and enforces component-null-space/dimension rules with sorted native/Python row identity. Native operator and NumPy eigenpair references cover the slice. Flat layout now uses every real active edge with checked row identity and a native finite-difference-tested objective/gradient solver, replacing fabricated/truncated edges, detached gradients and placeholder forces/clipping. A thin Python method returns owned coordinates. Composite interleaving now alternates all child columns in checked chunks, with row-ID alignment, shape/control validation and stable weighted-average accumulation; the factory and thin Python matrix combiner share one native component. N-dimensional energy/force-directed layout now uses actual all-edge spring/all-pair repulsion energy and analytic gradients, checked controls/identity, owned warm starts, seeded factory initialization and bounded monotonic backtracking; independent derivative/topology tests replace placeholders. Dense t-SNE now fits entropy-calibrated symmetric affinities and normalized Student-t probabilities, applies exaggeration to attraction rather than learning rate, and checks analytic gradients/seeded row-preserving output with bounded descent. The native UMAP-inspired solver now checks exact neighbors and sample budgets, uses local distance weights and balanced derivatives, and preserves seeded owned identity with bounded finite step acceptance; its documented neighbor/spring model does not claim full UMAP fuzzy-set/cross-entropy equivalence. Realtime flat preprocessing now propagates failures through the owned matrix helper with row labels and strict supplied controls, and unknown layout names reject instead of broadcasting success. WebSocket layout acknowledgements now reflect actual engine completion/failure rather than premature success. Layout-change requests now prepare all fallible computation and timestamp work before publishing configuration, controllers, caches, animation and coordinates; failures preserve prior state and queues, while successful switches retain queued property/structural changes and emit a coherent envelope. Embedding/projection controls now also stage candidate configuration and computation before publication; string embedding requests reject unknown/ignored controls and acknowledge actual execution. Quality-only updates explicitly reject the previous no-op behavior. Other visualization/step-caller audits, quality computation, remaining realtime controls/visual error presentation and composite child-seed policy, plus external/stochastic/scale verification remain open. See [realtime layout errors](../../docs/api/realtime-layout-errors.md). See [UMAP-inspired projection](../../docs/api/umap-inspired-projection.md). See [t-SNE projection](../../docs/api/tsne-projection.md). See [energy embedding](../../docs/api/energy-embedding.md). See [composite embedding](../../docs/api/composite-embedding.md). See [flat layout](../../docs/api/flat-embedding.md). See [spectral embedding](../../docs/api/spectral-embedding.md). See [PCA projection](../../docs/api/pca-projection.md). See [Dijkstra wrapper](../../docs/api/dijkstra.md). See [BFS/DFS](../../docs/api/bfs-dfs.md). See [closeness](../../docs/api/closeness.md). See [PageRank](../../docs/api/pagerank.md). See [connected components](../../docs/api/connected-components.md). See [label propagation](../../docs/api/label-propagation.md). See [Infomap domains and limits](../../docs/api/infomap.md). See [community modularity](../../docs/api/community-modularity.md). See [pathfinding validation](../../docs/api/pathfinding-validation.md).
29. **Complete and optimize Groggy's vector and matrix libraries.** Keep Groggy's existing API design rather than pursuing NumPy API parity. Audit vector/array arithmetic, shapes and broadcasting where promised, reductions, dense and sparse matrix operations, decompositions, numerical edge cases, and columnar interoperation. The identity and axis-statistics slice now repairs eye, sample variance/std, stable accumulation and empty/invalid shape handling, with Rust/Python reference checks. Real SVD and symmetric eigendecomposition now use bounded in-house Jacobi kernels, with reconstruction/orthogonality checks and NumPy comparisons. Logical BaseArray sum/mean/median/sample variance/std now share checked native kernels, preserving output types and null policy while repairing overflow, cancellation and high-offset integer variance with native/exact-rational Python references. Logical-column min/max now also use exact mixed numeric ordering and return original typed values with checked finite domains. NumArray sum/mean/extrema/sample variance/std/median now share typed native and thin Python kernels, with checked scaling, integer-difference precision, consistent IEEE/empty behavior and safe selection; historical SIMD/auto entry points currently use those reference kernels pending measured acceleration. ArrayArray and percentile/correlation reductions, general nonsymmetric/complex eigendecomposition, sparse behavior, decomposition gradients and full precision/output guarantees still need completion. See [array reductions](../../docs/api/array-reductions.md) and [NumArray reductions](../../docs/api/numarray-reductions.md). See [decomposition domains and limits](../../docs/api/matrix-decompositions.md). See [matrix statistics](../../docs/api/matrix-statistics.md). Verify results against trusted numerical references, then measure allocation, memory use, and throughput on representative data before relying on these primitives for PCA, spectral methods, or ML pipelines.
30. **Finish the algorithm builder around corrected primitives.** Support repeatable graph and table pipelines, typed intermediate results, cancellation, and historical inputs. This fulfills the analytics vision after underlying computations are dependable.

## Stage 6 — Verification and operations

Prove compatibility and reliability at realistic scale, then document actual capabilities.

31. **Run differential and failure testing.** Compare core graph, query, and temporal behavior with Neo4j and graph-algorithm results with NetworkX on shared, clearly defined fixtures. Test restart and failed mid-batch writes. Defer Graphiti corpus comparisons, including duplicate episode ingest and search, until the adapter and driver are implemented.
32. **Measure and optimize the intended workload.** Benchmark ingestion throughput, indexed lookup, temporal reads, neighborhood expansion, vector/array and dense/sparse matrix operations at realistic sizes. Add hybrid-search benchmarks after that feature is implemented. Use results to improve memory layout, indexing, compaction, and concurrency.
33. **Finish operational tools.** Provide migration/version checks, backups and restore, integrity checks, observability, and clear limits for graph size, concurrent writers, and recovery time.
34. **Publish an honest capability matrix.** Mark features implemented only when their Rust API, Python API, persistence, and correctness tests meet the contract. Update old roadmap completion claims accordingly.

The near-term milestone is a coherent Groggy core: reliable writes, temporal reconstruction, immutable connected views and tables, native graph queries, corrected graph algorithms, and capable vector/matrix libraries, with persistence, reference-result comparisons, and failure testing. After that, items 20 and 22–25 provide Graphiti ingest, retrieval, and backing-database milestones. The remaining operations and scale work establishes production readiness.

Numerical item 29.10 completes the existing NumArray linear percentile path using shared typed kernels and explicit IEEE/rank behavior. Correlation is repaired in 29.11; arithmetic, ArrayArray and measured acceleration remain open.

Numerical item 29.11 completes the existing NumArray Pearson correlation path with typed centering, independent scaling and compensated normalized moments. It uses two O(N) scratch columns; benchmarking and other correlation surfaces remain open. ArrayArray and arithmetic operators are next under 29.3.

Numerical item 29.12 repairs checked matrix add/subtract and Python arithmetic with native scalar kernels, exact-zero divisor validation and preserved positional labels/empty shapes. Legacy Rust operator result-type migration, broader numerical overflow rules, NumArray expansion and ArrayArray remain open.

Numerical item 29.13 repairs ArrayArray f64 kernels and centralizes legacy attribute projection/multi-aggregation in Rust, preserving named keys. Table scalar floats remain f32; finite narrowing overflow now errors. Exact grouped i64 statistics, lossless f64 table storage, other grouped-table implementations and throughput remain open.

Numerical item 29.14 removes silent fallbacks from borrowed/owned Rust matrix +, - and * by returning GraphResult. Named methods and Python syntax keep their checked APIs. Product output sizing and positional row/column identity are checked; downstream Rust callers must handle the intentionally changed associated Output type. Integer overflow, stable dot products and other matrix operations remain open.

Numerical item 29.15 repairs GraphMatrix floating dot products with exact finite f64 accumulation and one rounding; f32 narrows afterward. Native/rational Python tests and matching dev profiles delimit the slice. Linear solves/inverse, integer/custom arithmetic, sparse/lower-level routines, remaining decompositions and production throughput still require work.

Numerical item 29.16 repairs dense floating solve/inverse using direct row-equilibrated partial-pivot elimination. Multiple RHS, source/coordinate identity, scale-aware behavior and explicit domain/range errors have native and rational Python coverage. Condition estimates/refinement, integer arithmetic, remaining indexing/reductions, sparse/complex solvers and remaining decompositions still require work.

Numerical item 29.17 checks built-in integer GraphMatrix add/subtract/Hadamard overflow and matrix-product i128 accumulation/output conversion. Arbitrary precision, other integer backends, exact grouped integer reductions and remaining matrix/sparse/decomposition/conditioning work remain open.

Numerical item 29.18 repairs integer-only ArrayArray sum/mean/std precision through typed native reductions for both single and multi aggregation. Explicit float projection, mixed-type working limits and f32 materialization remain documented; other grouped-table paths and storage widening are separate.

Numerical item 29.19 repairs owned column/2D selection, raw Python slicing/masks, logical reshape and transpose with explicit validation, identity and empty-shape behavior. Remaining reductions, lower-level views/lazy/sparse paths and general decomposition/performance work remain open.

Numerical item 29.20 establishes canonical CSR construction/access/update/clone and native/Python checked-consumer correctness, with dense output semantics. Efficient sparse kernels, sparse factorizations, lower-level backend coverage and production-scale evidence still remain under 29.4/29.6.

Numerical item 29.21 introduces a checked i128 fast path for integral-valued floating dots, retaining exact transitions to reference bins. Matching dev profiles measure integral and fractional-control workloads; this does not close release-scale throughput, generic fractional acceleration, allocation accounting, SIMD or GIL work.

Numerical finish status after 29.15–29.21: floating dot accuracy, direct floating solves/inverse, checked built-in integer arithmetic, typed grouped integer statistics, owned indexing/transforms, canonical CSR basics and an initial measured dot fast path have concrete slices. The six broad areas are not complete. Flat items 29.22–29.26 explicitly retain remaining reductions, conditioning/real factorizations, general complex eigen/gradients, efficient sparse kernels and production performance gates; parent items 29.3–29.6 remain open.

Numerical item 29.22 repairs GraphMatrix global/axis/trace and legacy column reductions, with typed stable kernels, checked Rust results, explicit IEEE/empty behavior and preserved sample/population conventions. Remaining work is now 29.23–29.26 plus lower-level/neural reduction consistency and production performance; the parent numerical categories remain open.

Numerical item 29.27 repairs real QR with bounded economy Householder factors and checked scale/domain/identity behavior. Wide shapes migrate to m×k and k×n. Cholesky/LU/determinant/rank/conditioning/refinement and least-squares work still remain under 29.23.

Numerical item 29.28 repairs finite real Cholesky using scaled relative symmetry checks, accurate residuals and explicit positive-definite/range/dtype validation. LU/determinant/rank/condition/refinement/least-squares work remains under 29.23.

Numerical item 29.29 repairs finite real LU and determinant/slogdet with partial pivoting, singular/empty support, checked factor range/identity and bounded cubic work. Rank, condition estimation, residual/refinement and least squares remain under 29.23; complex/sparse variants and production performance remain separate.

Numerical item 29.30 repairs rectangular/scale-aware numerical rank and adds an independent normalized 2-norm condition estimate with native/Python APIs and singular-spectrum references. Residual reporting/refinement and least squares are the remaining 29.23 solver slices.

### Item 29.31 — Solve diagnostics and refinement, 2026-10-21
Add owned native SolveReport and thin Python tuple without changing solve/inverse. Report a normwise backward-error estimate against original coefficients using exact-dot rounded residuals and log norms; it is not a forward-error/certified guarantee. Bound correction attempts to 16 and accept only strict improvement after output-dtype rounding, preserving coordinate identity and explicit range failures. Repeated dense refactorization is deliberate initial scope; factor reuse, sparse refinement and performance remain open. See docs/api/matrix-solve-refined.md.

### Item 29.32 — Remove hidden SVD truncation, 2026-10-21
Follow-up inspection found a spectral floor in the underlying SVD that overrode the explicit numerical-rank cutoff and could report infinite condition for a diagonal matrix with a resolved tiny singular value. Rotate nonzero columns and retain computed nonzero norms instead. Preserve bounded convergence/basis errors rather than invent missing directions. Add diagonal tests through 1e-300 alongside existing decomposition references. Finite-precision rank uncertainty remains explicit.

### Item 29.33 — Real minimum-norm least squares, 2026-10-21
Use the bounded in-house real SVD to solve rectangular/rank-deficient multiple-RHS systems, retaining components above an explicit relative cutoff (default shared with numerical rank). Normalize finite floating coefficients/RHS, use accurate dot projections/reconstruction and preserve owned unknown/RHS identity. Reject shape/dtype/cutoff/convergence/working-range failures. Empty/zero systems return the minimum-norm zero result. Dense full factors and repeated allocation remain explicit costs; sparse/thin optimization, gradients and release throughput remain open. This finishes the bounded real factorization/diagnostic list in 29.23, while 29.24–29.26 remain open. See docs/api/matrix-least-squares.md.

### Item 29.34 — Dedicated complex eigen result, 2026-10-21
Implement the caller-selected separate result with f64 real/imaginary eigenvalues and owned real/imaginary eigenvector matrices, retaining the real symmetric API. General finite real-input eigenpairs use in-house bounded shifted complex Householder QR/full similarities, local deflation, independently scaled trailing shifts, triangular back substitution and residual/basis checks. Reject defective/unresolved bases, convergence and range failures explicitly. This dense reference does not claim Hessenberg/implicit-QR performance, balancing, complex input, certified forward bounds or gradients. Decomposition differentiation and sparse/performance work remain open under 29.24–29.26. See docs/api/matrix-complex-eigen.md.

### Item 29.35 — Explicit Cholesky derivatives, 2026-10-21
Add native/Python cholesky_jvp and cholesky_vjp using owned outputs and shared scaled triangular/dot kernels. Define symmetric tangents and symmetric Frobenius cotangents explicitly: an off-diagonal pair contributes twice the returned entry. Upper-factor cotangents are irrelevant but must be finite. Preserve forward SPD/domain validation, dtype/coordinate identity and explicit working-range errors. Independent central differences and adjoint checks cover the math; tape integration, other decomposition derivatives and cached/blocked/sparse/performance work remain open. See docs/api/matrix-cholesky-derivatives.md.

### Item 29.36 — Cholesky automatic tape integration, 2026-10-21
Track checked forward Cholesky factors and add native Operation::Cholesky forward/backward delegation to the same factor and symmetric Frobenius pullback. The current backward seeds ones and clears prior gradients; explicit VJP remains available for arbitrary cotangents. Validate the bound input snapshot against values used for the tracked forward and reject stale values until explicit disable/rebind. Later source mutation leaves the completed forward snapshot stable. Native/Python chained, repeated, empty/f32, finite-difference and stale/post-forward mutation tests cover this operation. Broader tape ownership, mutation, branch merging, general grad metadata/locking and other decomposition gradients remain separate. See docs/api/matrix-cholesky-derivatives.md.

### Item 29.37 — Symmetric eigen and thin SVD derivatives
Implement native/thin Python eigen_jvp/eigen_vjp and svd_jvp/svd_vjp with separated-spectrum/full-rank policies, accurate dot operations, checked normalized rescaling and symmetric Frobenius input gradients. Thin SVD output excludes arbitrary full null-space completions. These explicit calls own outputs; automatic routes and complex derivatives remain in the numerical completion queue. Full locked offline native suite and 269 rebuilt Python numerical regressions passed; focused Clippy passed excluding the existing sort/match categories. See docs/api/matrix-spectral-derivatives.md and notes/planning/numerical-completion-pass.md.

### Item 29.38 — QR/LU derivative kernels
Add explicit qr_jvp/qr_vjp and lu_jvp/lu_vjp with checked analytic kernels and triangular solves. QR requires a resolved leading block and locally stable Householder signs; LU holds permutations fixed and rejects unresolved pivot ties/rank loss. Ignore structurally fixed cotangents only after finite validation. Native full suites and 280 rebuilt Python numerical regressions passed; focused Clippy passed excluding existing sort/match categories. Automatic output routes and performance remain in the completion pass. See docs/api/matrix-triangular-derivatives.md.

### Item 29.39 — Checked lower matrix/backend kernels
Extract the shared typed Product reference, align GraphMatrix.matmul/multiply and lower UnifiedMatrix arithmetic/reductions/logical materialization, and validate raw backend slices/concrete types before pointer casts. i128's historical Int64 dtype tag is not a valid raw memory layout; typed matrix kernels continue supporting it while raw wrappers reject. Native GEMM/element outputs are prepared before publication, and malformed dense strides return errors. Full locked offline native suites and 280 rebuilt Python numerical regressions passed; focused Clippy passed excluding existing sort/match categories. This repairs correctness/safety and makes no throughput claim; optimized sparse/vector/remaining routine and performance work continues. See docs/api/matrix-checked-backend.md.

### Item 29.40 — Sparse-native operations
Add canonical CSR arithmetic/products, mixed finite products, transpose, repeated selections, reshape, scalar mapping and explicit owned conversions. Preserve logical IEEE nonfinite interactions using full traversal when required; canonical sparse zeros normalize to implicit positive zero. Remove dense copies from sum/extrema and correct the legacy dense() alias to actually materialize dense storage. Full locked offline native suites and 284 rebuilt Python numerical regressions passed; added scalar/reshape/signed-zero fixtures passed (four native sparse-operation tests and five Python sparse-operation tests). Focused Clippy passed excluding existing sort/match categories. Sparse solve/factors, remaining statistics and production performance remain open. See docs/api/matrix-sparse-operations.md.

### Item 29.41 — Sparse factorization and solve kernels
Implement sparse row-map/column-adjacency partial-pivot LU and right-looking positive-definite Cholesky with numerical fill-in, canonical sparse factors, accurate triangular solves and sparse sign-log determinants. LU retains unscaled working values; solves equilibrate rows like the dense reference, preserving independently tiny/huge representable rows. Save-state comparison for tracked CSR Cholesky avoids dense input copies. Dense RHS/output, generic gradient and remaining spectral/QR materialization costs remain explicit. Full locked offline native suites and 298 rebuilt Python numerical regressions passed; focused Clippy passed excluding existing sort/match categories. See docs/api/matrix-sparse-factors.md.

### Item 29.42 — Native typed vector operations and conversions
Add checked dtype-preserving vector arithmetic and accurate typed dots, f64 true division, stable norms, native mixed integer/float comparisons, first-occurrence typed unique and explicit exact/rounding policies. Move constructor/to_type conversion math out of FFI; retain nearest-integer rounding for explicit conversions but reject wrapping/saturation/finite floating narrowing overflow. Native NumArray statistics now support NumericType with retained integer hints; native matrix casts preserve labels/CSR. Matrix-bound reshape checks precision and scalar reads avoid repeated full-vector copies. Full locked offline native suites and 305 rebuilt Python numerical regressions passed; focused Clippy passed excluding existing sort/match categories. Further matrix/statistical, automatic-gradient and release/GIL gates remain in the pass. See docs/api/numarray-arithmetic.md.

### Item 29.43 — Checked matrix statistics and sparse reductions
Replace overflow-prone entrywise norms and lower typed residual squares with stable shared kernels; retain exact integer axis means, reject rounded upper-bound integer casts, validate logical layouts and propagate NaNs. Compute CSR means/axis statistics from stored entries plus analytic zero groups, including a single transpose for column reductions. Native full locked offline suites, three dedicated native fixtures and 310 rebuilt Python numerical regressions passed; focused Clippy passed with the existing sort/match categories excluded. Broader helper APIs, automatic gradients and release/GIL gates remain open. See docs/api/matrix-statistics.md.

### Item 29.44 — Checked numerical helper operations
Remove silent zero substitutions from lower math functions, preserve floating IEEE domains, reject integer abs overflow and retain NaNs through ReLU. Add fallible owned maps, CSR-aware maps/reshape/slice/concatenation, checked helper shape allocations, prepared in-place writes, owned materialization and metadata-preserving GraphMatrix split/concatenate. Full locked offline native suites and 312 rebuilt Python numerical regressions passed; focused Clippy passed excluding existing sort/match categories. Generic callback zip materialization, remaining neural/autodiff and release/GIL gates remain explicit. See docs/api/matrix-helper-operations.md.

### Item 29.45 — Seeded factor output automatic differentiation
Add finite shape-checked backward_with_gradient, factor-output tape routes for QR/LU/symmetric eigen/thin SVD, matrix-valued eigen/singular values, saved-input checks and gradient label preservation. Keep existing value lists owned; reject tracked rectangular full SVD and unsupported general complex eigen derivatives explicitly. Explicit joint VJPs remain the route for multi-output loss pullbacks; independent tape calls replace prior gradients. Full locked offline native suites and 319 rebuilt Python numerical tests passed, including independent weighted spectral finite differences; focused Clippy passed excluding existing sort/match categories. Universal arithmetic tape composition and release/GIL gates remain open. See docs/api/matrix-factor-autodiff.md.

### Item 29.46 — Owned numerical GIL boundary
Add checked Send numerical snapshots preserving CSR/labels while excluding graph references and tapes; reconstruct owned matrices on the receiving side. Thin FFI marshalling releases GIL for untracked heavy products, solves, factors, diagnostics and explicit decomposition derivatives. Tracked matrix inputs retain the owning tape/GIL path. Full locked offline native suites and 324 rebuilt Python numerical regressions passed, including independent-thread progress during dense products/QR/Cholesky and sparse Cholesky; focused Clippy passed excluding existing sort/match categories. Copy overhead and production throughput remain measured gates, and generic reductions/backward do not yet release GIL. See docs/api/matrix-numerical-threads.md.

### Item 29.47 — Release numerical corpus and measured bulk-kernel improvements
Add a repeatable 20-case optimized numerical corpus with seven samples, allocator requests/additional-live-heap peaks and checked result receipts. Route dense GraphMatrix products/reduction materialization through shared bulk kernels and replace sequential hypot norms with scaled accurate squares. Recorded roughly 1.85x fractional products, 9.16x integer products, 1.40x norms and 2.01x dense transfer-roundtrip+sum improvement for these cases, with explicit extra product scratch and instrumented/noisy timing limitations. Preserve the contended receipt separately. Full locked offline native suites and 324 rebuilt Python numerical regressions passed; focused library/benchmark Clippy passed excluding existing sort/match categories. Full strict Clippy still reports eight preexisting errors. Production size sweeps/SIMD/general tape gates remain open. See notes/benchmarks/numerical-completion-2026-10-22.md.

### Item 29.48 — Checked attribute, ID and fancy-index bridges
Move attribute inference/promotion and exact positional conversions into Rust, eliminate i64::MIN inference overflow, reject native IDs outside signed array range and propagate errors through graph/subgraph/table/accessor wrappers. BaseArray conversions now use native checked rounding/exact policies and reject nulls/non-numeric values instead of inserting zeros or discarding length; preserve existing bool/int64/float64 promotion and aliases. Integral floating NumArray positions remain accepted; fractional/nonfinite/range failures reject while bool masks remain masks. Full locked offline native suites and 357 rebuilt Python numerical/array compatibility tests passed; focused Clippy passed excluding existing sort/match categories. General arithmetic tape composition and broader production gates remain open. See docs/api/numarray-arithmetic.md.

### Item 29.49 — Matrix powers and correct noncommuting gradients
Replace linear repeated multiplication with checked exponentiation by squaring and replace the incorrect elementwise matrix-power backward rule with recursive matrix Frechet JVP/VJP. Retain sparse identity/results, positional labels, saved-input checks, zero/empty behavior and explicit nonfloating derivative errors; add native/Python power_jvp/power_vjp and untracked GIL routing. Full locked offline native suites, focused final boundary checks and 364 rebuilt Python numerical/array tests passed, including independent powers/finite differences/adjoints and u32::MAX sparse identity. Focused Clippy passed excluding existing sort/match categories. General arithmetic tape composition remains open. See docs/api/matrix-powers.md.

### Item 29.50 — True economy SVD and bounded rectangular consumers
Share bounded Jacobi rotations across full/economy modes, completing only k requested thin columns. Add svd_thin, retain full svd, and route tracked SVD, explicit thin derivatives, numerical spectrum/rank/condition and minimum-norm least squares through actual economy factors with thin indexing/labels. Full locked offline native suites, large rectangular native fixtures and 376 rebuilt Python numerical/array regressions passed; focused library/benchmark Clippy passed excluding existing sort/match categories. Idle release receipts compare 512x4 full/thin factors (306.05ms/0.514ms, 4.19MB/53.6KB additional peak heap) and 10,000x4 consumers around 1.0–1.12MB heap. These instrumented shape-specific results do not certify sparse spectral work or universal performance. See docs/api/matrix-economy-svd.md.

### Item 29.51 — Unified public mathematical/scalar/comparison routes
Add a shared typed fallible map for dense/logical/CSR storage and reuse it for casts, native boolean comparison masks and checked public math/scalar methods. Remove Python abs/comparison loops and the scalar_multiply placeholder, preserving floating mask output compatibility and large-integer scalar precision. Tile/repeat now use checked selections, retain labels/CSR/empty dimensions and reject size overflow. Full locked offline native suites and 382 rebuilt Python numerical/array regressions passed; focused Clippy passed excluding existing sort/match categories. Graph numerical transforms, general tape composition and broader release gates remain open. See docs/api/matrix-helper-operations.md.

### Item 29.52 — Correct graph-matrix operators and native bridge
Use checked typed row degrees and sparse diagonals, correct generalized normalization powers, expose explicit normalized-adjacency/true normalized-Laplacian/transition methods, and repair high-level normalized Laplacian and transition routes. Preserve the parameterized legacy adjacency formula/default under its old name, document isolate/directed/row-degree/range policies, require explicit gradient detachment for unsupported row-degree pullbacks, and remove redundant alias reconstruction that lost labels/storage. Transition normalization scales rows before summing to handle overflowing finite raw degrees. Full locked offline native suites and 424 rebuilt Python numerical/array/graph-matrix compatibility tests passed; focused Clippy passed excluding existing sort/match categories. Graph adjacency construction/storage selection, array slice normalization and general arithmetic tape composition remain open. See docs/api/graph-matrix-transforms.md.

### Item 29.53 — Shared checked slice bounds and extreme steps
Normalize raw optional array/matrix range bounds once against actual dimension lengths, clamp negative offsets correctly, preserve omitted reverse bounds and use i128 positions/precomputed checked result counts to avoid signed step overflow. Marshal Python slice fields directly, expose shared zero-step validation and remove an unused duplicate normalization helper. Matrix row reads no longer materialize unrelated full columns. Full locked offline native suites and 437 rebuilt Python numerical/array/graph-matrix regressions passed, including 8,400 independent Python-list slice combinations; focused Clippy passed excluding existing sort/match categories. Slice fields retain the explicit native i64 input bound. See docs/api/array-slicing.md.

### Item 29.54 — Checked graph-to-matrix construction
Share one native CSR builder across dense/sparse adjacency exports; sum parallel edge instances accurately with checked integer narrowing, count undirected loops once, validate IDs/endpoints, preserve canonical full-graph coordinates and explicit node order, and propagate scoped weight errors. Honor storage format and reject unsupported raw-ID coordinates. View exports use selected edges and produce owned numerical snapshots; live parent semantics remain explicit. Rectangular row-ID labels now match shape. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 439 rebuilt Python numerical/array regressions passed, including 10,000-node CSR and independent edge-selection/cancellation fixtures. See docs/api/graph-matrix-construction.md.

### Item 29.55 — Scoped feature export and numerical boundaries
Resolve feature values through GraphSpace indices rather than latest pool values; sort columns, retain node row IDs and legacy zero imputation, validate references/shape/allocation and preserve typed integers. Explicit attribute conversion now checks integer range/nonfinite values and finite f32 overflow while retaining text/null/truncation policy. Dense fill validates all logical addresses before writes and preserves padding; CSR fill is supported with explicit full-storage cost for nonzero values. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 440 rebuilt Python regressions passed. Branch-switch, large integer, overflow and padded/invalid-layout fixtures cover the repaired boundaries. See docs/api/matrix-feature-conversion.md.

### Item 29.56 — Compose matrix arithmetic and factor tapes
Use unique computation IDs and validated tape unions to preserve shared ancestors across cloned sibling expressions and independent inputs. Track matrix add/subtract/Hadamard/product/transpose, dense/CSR conversions, scalar arithmetic and abs/exp/log/sqrt; propagate repeated-input contributions and check finite floating values/saved bindings. Repair forward traversal past unrelated later IDs and add identity/transpose/scale/math pullbacks with explicit singular-domain errors. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 446 rebuilt Python regressions passed. Independent NumPy finite differences cover a joint U/S/Vt loss; QR reconstruction, cloned branches, sparse/repeated operands and stale bindings have native/Python fixtures. Value reductions, unsupported transforms/activations and concurrent shared backward are not certified. See docs/api/matrix-arithmetic-autodiff.md.

### Item 29.57 — Explicit sparse numerical materialization policy
Bound implicit CSR input materialization for dense numerical consumers to 1,048,576 logical cells, exposing the limit natively and requiring explicit to_dense() opt-in above it. Retain small sparse spectral behavior and sparse-native coefficient solve/factor routes; validate logical input storage through bulk extraction and checked reservation. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 447 rebuilt Python regressions passed, including early large-CSR spectral rejection and unaffected 4096-square sparse solve/Cholesky. This input guard is not a total-memory/fill or iterative sparse spectral guarantee. See docs/api/matrix-sparse-materialization.md.

### Item 29.58 — Checked constructors and shared matrix/array conversion
Add checked zero/one/identity/storage wrappers including label allocation, remove fallback 1x1 substitutions, and retain legacy infallible signatures with explicit failure panic. Python uses checked constructors and rejects unsupported dtype claims instead of ignoring them. Unify numeric factory/column/BaseArray/flattened conversion through native exact policies; preserve compact integers, reject missing/nonnumeric/inexact cells, support empty row-major shapes, and check attribute export narrowing with an exact option. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 454 rebuilt Python regressions passed. Native overflow/metadata/attribute and Python factory/shape/dtype/precision fixtures cover these boundaries. Feature export retains its separately documented imputation. See docs/api/matrix-constructor-boundaries.md.

### Item 29.59 — Replay scalar primals and deterministic tape execution
Store scalar shifts/divisors as actual forward operations rather than derivative-only identity/reciprocal nodes. Pull scalar division gradients back by direct division so representable results survive reciprocal overflow. Use deterministic node-ID topological ordering, reject missing dependencies before publishing operation nodes and route matrix-product contributions through shared repeated-input accumulation. Full locked offline native suites, focused Clippy (excluding existing sort/match categories) and 455 rebuilt Python regressions passed. Native replay/ordering/dependency and native/Python subnormal-cotangent fixtures cover the repairs. See docs/api/matrix-arithmetic-autodiff.md.

### Proposed module consolidation — 2026-10-22

User proposes one matrix module replacing the matrix/advanced_matrix split.
The review identifies arrays, entity handles, graph-view ownership, display,
query/execution, shared state/change records and table boundaries as related
candidates, with distinct concepts retained. See
`module-consolidation-review-2026-10-22.md` for evidence and migration order.
This is a proposal, not a completed refactor or approval of every candidate.

### AR7.2 — Canonical matrix implementation completed

Move the advanced storage/backend/arithmetic/neural implementations and adjacency
helpers into storage::matrix; generic autodiff has a direct matrix module. Legacy
advanced_matrix, storage adjacency and neural autodiff imports are re-exports of
the same types. Preserve GraphMatrix/UnifiedMatrix/owned-transfer responsibilities
and Python APIs; migrate maintained callers. Full locked offline native suites,
legacy type/trait/nested-import compatibility fixtures, focused Clippy (excluding
existing sort/match categories) and 455 rebuilt Python regressions passed. Rebuilt
25-case release corpus and idle rerun agree with every prior workload checksum at
1e-13 tolerance. Receipt: benches/numerical_matrix_consolidation_2026-10-22.json.
This is an ownership migration, not a new speedup or broad numerical completion.

### AR7.3 — Canonical array storage and shared numeric foundation completed

Make legacy BaseArray a dtype/name compatibility adapter over the generic shared
column, with explicit copy-on-write bridges and common selection/ID/reduction
paths. Shared storage::numeric owns scalar dtypes, conversion/comparison/norm
policies, reductions and accurate/checked accumulation, independent of array or
matrix ownership. Old paths remain aliases; matrix error adapters preserve errors.
Repair dropped invalid selection positions, truncated masks, slice overflow and
ID narrowing while preserving valid input and legacy slice conventions. Full
locked offline native suites, compatibility fixtures, focused Clippy (excluding
existing sort/match categories) and 455 rebuilt Python regressions passed. All 25
optimized idle-run workload checksums match the pre-migration receipt at 1e-13;
raw receipt is benches/numerical_array_foundation_2026-10-22.json. Broader numerical
performance gates and remaining module/view tasks stay open.

### Item 29.60 — Expanded release baseline and strict core gates

Extend the corpus from 25 to 44 cases, retaining original checksums, finite result
receipts and allocation/time samples. Cover dense through 512-square, million-value
vectors/dispatch, banded sparse fill/mixed products and 100k-square sparse-native
operations. Full locked offline native suites, 455 rebuilt Python regressions and
strict all-target core Clippy pass after behavior-preserving sort/match/test lint
cleanup. Record dispatch as reference (not SIMD acceleration), non-universal heap
measurement and unresolved production budgets. The finite closeout finding list
names remaining attribute/table, activation/tracking and sparse expansion repairs
under 29.61–29.63; broad parents remain open. See the numerical expanded benchmark
note and numerical-closeout-findings-2026-10-22.md.

### Item 29.63 — Shared sparse expansion and nonfinite fallback policy

Use one public implicit-dense cell limit for spectral input conversion, nonzero
CSR map images and nonfinite CSR scale/product scans. Guard sparse operands before
mixed nonfinite products can flatten them; retain explicit dense opt-in, small
IEEE results and large sparse zero-preserving maps. Full locked offline native
suites, strict all-target core Clippy and 462 rebuilt Python regressions pass,
including 10k-square early-error and just-over-limit explicit opt-in fixtures.
Sparse kernel parent 29.25 closes within the documented native domain and measured
coverage; arbitrary fill or production-memory guarantees are not implied.

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

# Groggy graph engine implementation roadmap

Status: **staged implementation plan; compare each item against the [current-state review](../architecture/current-state-review-2026-10-19.md) and the [decision record](graph-engine-design-responses-2026-10-20.md).** This list originally overstated missing foundations. Many core types, APIs, and architectural choices already exist. Each item should begin by identifying what Groggy already implements; the work is to repair a demonstrated mismatch or add a clearly needed capability, not redesign settled foundations.

The owner's first-pass answers, accepted recommendations, and remaining open choices are recorded in the [2026-10-20 design responses](graph-engine-design-responses-2026-10-20.md).

Groggy's original vision is a Rust graph engine with a Python API, separate topology and columnar properties, tracked changes, connected graph and table views, and graph analytics. See [Origins & Design](../../docs/concepts/origins.md), [Design Decisions](../../docs/appendices/design-decisions.md), and the [temporal extensions plan](temporal-extensions-plan.md).

The concrete integration target is [Graphiti](https://github.com/getzep/graphiti/blob/main/README.md): Graphiti produces episodes, entities, facts, provenance, time fields, and embeddings; Groggy stores and queries them. This plan does not assume Groggy performs Graphiti's LLM extraction. Graphiti's [driver interface](https://github.com/getzep/graphiti/blob/main/graphiti_core/driver/driver.py) defines the eventual backend compatibility target.

The stages describe related work, with Graphiti-specific integration and retrieval deferred until the Groggy core is finished. Each item may call for verification, repair, completion, or a new capability. Item numbers remain continuous so individual work items are easy to reference. The owner's "number 25" in the later flat list means the Graphiti ingest adapter, numbered 20 here; its "number 27" means lexical search, numbered 22 here. The related Graphiti driver is numbered 25 here.

## Immediate next implementation slice

1. **Completed on this branch:** the existing `set_node_attrs`, `set_edge_attrs`, and column setter paths collect all inactive or unknown IDs and duplicate assignments before writing, with attribute and input positions in the error. Duplicate assignments are rejected as an implementation choice pending owner review. Vectorized pool writes and current Rust/Python entry points remain in use. Python conversion errors for malformed values are a separate parser concern.
2. **Explicit batch API implemented in Rust and Python:** `GraphBatch` combines node/edge insertion, property updates, and removals. It validates semantic errors against the proposed resulting state before calling existing bulk writers. This establishes all-or-nothing behavior for rejected input in a single-process graph; crash-safe rollback and durable publication belong to the storage milestone.
3. **Batch publication implemented; legacy migration pending:** `apply_batch` checks the expected state and publishes one typed delta and state on success. Existing `commit`/`checkout` and direct mutation methods remain callable with their current manual-commit behavior. Route those direct methods through one-operation batches only with a compatible transition and explicit tests.
4. Rust and Python tests now cover multiple semantic errors with zero writes, stale-state rejection, one successful state publication, and an unchanged older snapshot. Extend failure injection to storage/index publication boundaries when those components exist.

## Architecture refactor thread (cross-stage)

These are changes to the *existing* engine's internal responsibilities, not a replacement engine or a new prerequisite contract. Make each change behind the current Rust and Python APIs. Record current behavior with focused conformance tests first, distinguish intentional behavior changes from structural refactors, and measure hot paths before and after. The companion design workbook prompts decisions on ownership, view lifetime, and history representation before implementation.

- **AR1 — One authority for active topology (items 1–3, 18, 28).** `GraphSpace` owns live node and edge membership; `GraphPool` owns allocated slots and columnar values. Route public reads, indexes, views, and algorithms through the same active-state check. Retain tombstones or stored slots where history needs them, without treating allocation as liveness.
- **AR2 — One adjacency and direction policy (items 3–4, 21, 28).** Build a narrow internal topology read interface over the existing topology machinery. It must return edge IDs and endpoints consistently for directed and undirected graphs, parallel edges, self-loops, and historical snapshots. Remove algorithm-local choices about reverse edges. Avoid virtual dispatch on hot paths unless benchmarks justify it.
- **AR3 — One coordinated mutation and typed change record (items 5, 9, 11–15).** A successful mutation updates live state and records the actual before and after values, identity, and topology change together. Commits, diffs, reset/revert, and snapshot reconstruction consume that record. Establish failure and rollback behavior before exposing transactions; do not infer history later from column offsets or placeholder strings.
- **AR4 — Explicit view and table ownership (items 12, 26–27).** Implement the accepted pinned, immutable view semantics: a later parent mutation does not change an older view. A write routed through a view checks its expected version and returns a new view or conflict. Decide whether graph tables are borrowed views or materialized results where current code and documentation disagree. Preserve existing public methods through a compatible migration.
- **AR5 — A smaller coordinator and thin adapters (items 19, 21, 25, 30).** Keep `Graph` as the stable public facade while moving mutation coordination, topology reads, history, and table conversion into cohesive internal components. Keep FFI limited to translation, and put Graphiti episode, provenance, UUID, and temporal mapping in an adapter above the core. Split crates or files only where this yields a real responsibility boundary.
- **AR6 — Honest persistence boundary (items 10, 20, 25, 33–34).** Decide whether Groggy is an in-memory engine with export or Graphiti's durable system of record. If durable, persist snapshots and the typed change record with versioned recovery and migration. Keep existing export functions distinct from a transactional storage guarantee, and make incomplete or no-op public operations explicit rather than silently succeeding.

## Stage 1 — Verify and repair core invariants

Preserve Groggy's established graph model and make its behavior consistent across public APIs. The one-node, zero-edge graph is the smallest meaningful graph; an empty `Graph` is also supported as a construction state. This stage tests and repairs implementation against the existing contract.

1. **Codify and verify the existing engine contract.** Collect the accepted behavior already expressed in Groggy's design docs, APIs, and tests into a focused conformance suite for Rust and Python. Use a single node with no edges as the minimal graph case. Document actual gaps or contradictions before changing behavior; extend the contract only for requirements Groggy has not yet decided.
2. **Unify live-state semantics.** Make every public read honor the same active node and edge state. The first slice moved `has_edge_between` and `edge_endpoints` to GraphSpace membership; audit the remaining read paths.
3. **Fix directed graph behavior throughout.** Edge lookup, paths, traversals, and algorithms must respect edge direction. Current path queries can traverse a directed edge backward.
4. **Define and implement edge multiplicity.** Give each relationship its own stable ID and explicit type; verify parallel edges and self-loops across storage, queries, and history. Graphiti can hold distinct facts between the same two entities.
5. **Make mutations consistent and validated.** Reuse the existing Rust and Python bulk APIs, including `set_node_attrs`, `set_edge_attrs`, `add_nodes`, and `add_edges`. The first slices now reject invalid edge endpoints and all inactive, unknown, or duplicate property targets before writing; Rust and Python regression tests cover them. Continue auditing other bulk paths and Python value conversion so all invalid input classes receive the agreed atomic, complete error report.
6. **Establish stable external IDs.** Map Graphiti UUIDs to Groggy's internal IDs without depending on insertion order. Support idempotent upserts, uniqueness constraints, and lookup by UUID and group.
7. **Complete the property model.** Preserve all supported values through Rust, Python, history, and disk: nulls, dates and times, arrays, vectors, and structured attributes as well as scalars. Define missing-property and type-change behavior.
8. **Add labels, relationship types, and schema rules.** Support `Entity`, `Episodic`, and other node kinds; named edge kinds; required fields; and optional ontology constraints while retaining flexible columnar attributes.

## Stage 2 — Durable writes and graph history

Make changes atomic and persistent, then complete the Git-like history promised by the original design.

9. **Implement atomic write batches and transactions.** Rust/Python `GraphBatch` now publishes one in-memory state for a validated mixed write using existing bulk paths. Next: migrate direct methods, add a batch-wide effective-time default and per-change overrides, coordinate indexes, and make episode ingest durably atomic across entities, facts, and provenance. Publication records deltas and one engine-assigned recorded time for the batch.
10. **Build durable storage and recovery.** The existing `Graph::save_to_path` writes a limited JSON representation and skips complex property types. Add a versioned on-disk format preserving IDs, schema, active state, properties, indexes, and history; test restart and interrupted writes.
11. **Complete property history.** The first slice made commits resolve typed node and edge values and added tests for historical reads. Finish coalescing, deletion, unset/null distinctions, effective and recorded times, and all supported value types before treating the history contract as complete.
12. **Make snapshots and state switching exact.** The first slice preserved direction and ID allocation across snapshots and branch switches. Extend reconstruction to all graph configuration and retained references, and verify state-pinned reads independently of the working head. Treat an explicit `commit` as a requested immutable checkpoint of the current published state after a compatible migration; automatic batch versions and deltas remain the source for reconstruction between checkpoints.
13. **Complete diff, revert, and reset semantics.** `reset_hard` restores the recorded state and preserves allocated IDs. Rust and Python `diff_commits` now compare reconstructed states in either direction, including typed property before/after values and topology, with deterministic ordering. Revert, endpoint rewiring, and destructive pruning policy remain.
14. **Finish branches and merges.** Implement a three-way merge for node, edge, and property changes, with conflict reporting and resolution. This follows reliable commits and snapshots.
15. **Define retention and garbage collection.** Preserve referenced commits and provenance while allowing compaction of unreachable history. Existing `gc_history` and `optimize` are no-ops.

## Stage 3 — Temporal knowledge graph ingest

Complete Groggy's temporal and provenance primitives. The Graphiti-specific ingest adapter at item 20 is deferred until the core graph, history, view, table, query, and analytics work is finished.

16. **Add effective time separately from recorded time.** Graphiti facts have valid and invalid times, and property changes also need an effective timestamp. Use an explicit per-change time when supplied, otherwise the batch's publication clock; the engine separately records when the batch was accepted. Reconstruct a branch-scoped graph view for “what was true at T”, using the latest published version on that branch at view creation. Apply a `recorded_by=U` cutoff only when the caller explicitly requests “what did we know by U?”; the resulting view remains pinned. This uses the same immutable view and graph-to-table interface as a present-state selection; derived subgraphs and tables inherit the state/time pins. The current `snapshot_at_timestamp` selects only a whole-second commit timestamp across all commits and is insufficient for these reads.
17. **Model episodes and provenance directly.** Store raw episode identity and content, the entities and facts derived from it, and links back to the source. Invalidate or supersede facts without erasing their evidence.
18. **Build primary and secondary indexes.** At minimum: UUID, group, node label, edge type, endpoints, property equality/range, and time-window indexes. Keep every index correct under update, deletion, checkout, and recovery.
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
28. **Correct core traversal and graph algorithms.** Validate start and end nodes, respect direction and active state, handle weights correctly, and reject unsupported inputs. Dijkstra currently accepts negative weights, and some traversals return a nonexistent start node as a result.
29. **Complete and optimize Groggy's vector and matrix libraries.** Keep Groggy's existing API design rather than pursuing NumPy API parity. Audit vector/array arithmetic, shapes and broadcasting where promised, reductions, dense and sparse matrix operations, decompositions, numerical edge cases, and columnar interoperation. The matrix layer currently has placeholder or incorrect behavior in `eye`, axis standard deviation, eigendecomposition, and SVD. Verify results against trusted numerical references, then measure allocation, memory use, and throughput on representative data before relying on these primitives for PCA, spectral methods, or ML pipelines.
30. **Finish the algorithm builder around corrected primitives.** Support repeatable graph and table pipelines, typed intermediate results, cancellation, and historical inputs. This fulfills the analytics vision after underlying computations are dependable.

## Stage 6 — Verification and operations

Prove compatibility and reliability at realistic scale, then document actual capabilities.

31. **Run differential and failure testing.** Compare core graph, query, and temporal behavior with Neo4j and graph-algorithm results with NetworkX on shared, clearly defined fixtures. Test restart and failed mid-batch writes. Defer Graphiti corpus comparisons, including duplicate episode ingest and search, until the adapter and driver are implemented.
32. **Measure and optimize the intended workload.** Benchmark ingestion throughput, indexed lookup, temporal reads, neighborhood expansion, vector/array and dense/sparse matrix operations at realistic sizes. Add hybrid-search benchmarks after that feature is implemented. Use results to improve memory layout, indexing, compaction, and concurrency.
33. **Finish operational tools.** Provide migration/version checks, backups and restore, integrity checks, observability, and clear limits for graph size, concurrent writers, and recovery time.
34. **Publish an honest capability matrix.** Mark features implemented only when their Rust API, Python API, persistence, and correctness tests meet the contract. Update old roadmap completion claims accordingly.

The near-term milestone is a coherent Groggy core: reliable writes, temporal reconstruction, immutable connected views and tables, native graph queries, corrected graph algorithms, and capable vector/matrix libraries, with persistence, reference-result comparisons, and failure testing. After that, items 20 and 22–25 provide Graphiti ingest, retrieval, and backing-database milestones. The remaining operations and scale work establishes production readiness.

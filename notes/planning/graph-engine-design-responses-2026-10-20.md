# Graph engine design responses — 2026-10-20

Status: **confirmed directions are authorized for implementation; unresolved choices remain provisional.** This captures the owner's answers to the design workbook and separates them from recommendations. The owner confirmed on 2026-10-20 that Neo4j is the reference backend and bulk writes must fail atomically with every invalid input identified in the error. The [roadmap](graph-engine-implementation-roadmap.md) remains a staged plan rather than a completion claim.

## Confirmed implementation direction — 2026-10-20

The owner accepted the following recommendations and authorized implementation. This section is the decision record for the staged work; the status of each rule in code must still be verified separately.

1. A successful atomic batch publishes one logical version. A single mutation is a one-operation batch. Versions use changes plus periodic snapshots, not full graph copies per write.
2. A view is pinned to one version and is immutable. A write routed through a view checks its expected version and returns a new version/view; a stale write conflicts.
3. Deletion changes membership in new states while the stable ID and older values remain in retained history. Purge is a distinct explicit operation, and IDs cannot be reused while referenced.
4. Timed changes retain effective time and recorded time. Queries at a state and time resolve one value per entity-property key; corrections do not overwrite previous records.
5. An acknowledged durable batch must recover completely after restart or not appear. Use an append-only log, periodic snapshots and a versioned manifest; specify the exact flush guarantee before claiming durability.
6. Begin with concurrent state-pinned readers and one serialized writer per graph. A multiuser service can coordinate access above the core.
7. Prefer public language such as apply batch, version, switch state, fork branch, compare and purge. Existing `commit`/`checkout` APIs remain as compatibility aliases during migration.
8. A reset is an explicit switch to an earlier state with a separately specified destructive pruning policy. Shared or referenced history is not removed implicitly.
9. Bulk operations fail atomically if any input is invalid and identify **every** invalid input in the error. Neo4j is the confirmed comparison backend.

First implementation slice: active-edge reads now use GraphSpace membership and graph direction; historical commits resolve typed property values from append-only pool columns; snapshots preserve graph direction; the Rust edge-batch API rejects the whole invalid batch before allocation. Python edge-batch parsing validates all rows before insertion. State switches and hard resets now preserve allocated ID high-water marks, and hard reset restores the last recorded state. These are foundations, not completion of automatic version publication, immutable live views, transactional persistence, or multiuser concurrency.

Existing bulk API clarification: Rust and Python already expose `set_node_attrs` and `set_edge_attrs`; Rust also has `set_node_attr_column`, `add_nodes`, and `add_edges`. The property setters validate IDs before applying changes and record attribute changes, so this work should extend their behavior rather than introduce replacement bulk setters. They do not publish a new version per call. The next implementation slice should compose these optimized operations behind one atomic, version-publishing boundary for mixed node, edge, and property changes. Preserve the existing public methods as compatibility paths into that boundary.

Bulk setter implementation update: `set_node_attrs`, `set_edge_attrs`, and `set_node_attr_column` now collect all inactive/unknown target IDs and duplicate assignments before writing, reporting attribute and input positions in one error. Bulk node setters also enforce the existing single-setter rule that `entity_type` is system managed. Rejecting duplicate assignments avoids order-dependent results and is an implementation choice for review, not a previously settled owner decision. Python value-conversion failures remain a separate parser limitation.

Batch implementation update: Rust `GraphBatch` and Python `GraphBatch` can stage new nodes, edges referring to those new nodes, property assignments, and removals. `apply_batch` checks an expected published state, rejects a dirty legacy working graph, validates semantic inputs before mutation, and records one new state on success. Existing single and bulk methods still use manual `commit`; the accepted one-operation-batch rule requires a compatibility migration. This batch path is in-memory only, with no durability claim yet.

Batch validation uses current membership lookups by ID plus the batch's new and removed IDs. It does not copy or scan the full active graph for each write. An edge cannot be added with an endpoint removed by the same batch.

Comparison implementation update: `diff_commits(from, to)` now compares reconstructed published states in Rust and Python and reports added/removed node and edge IDs plus typed property before/after values in deterministic order. Unknown state IDs fail explicitly. This is a state comparison; it does not yet report every intermediate action between those states, and revert or endpoint rewiring remains future work.

Temporal and checkpoint clarification: Record a typed delta when a successful write batch is published, never merely because a graph computation or historical read runs. Each change may carry a caller-supplied effective timestamp; when omitted, use the batch's publication clock as its default. The engine assigns one recorded timestamp and one logical version to the whole batch. Existing bulk setters should retain their vectorized write paths while accepting a batch-wide effective-time default and, where needed, per-item overrides. A later correction may have an old effective time and a new recorded time. Historical reads must select a branch or state lineage and reconstruct the graph at a requested effective time, optionally limited to information recorded by a second timestamp. This includes topology, active membership, properties, and stable IDs.

The owner's current description treats `commit` as an explicit immutable snapshot/checkpoint of the current published state. Automatic batch publication still creates the logical versions and deltas needed between checkpoints; a checkpoint need not copy the entire graph if a compact representation can reconstruct the same state. The current `commit` method instead publishes manually accumulated changes, so changing its meaning requires a compatibility migration. Current `snapshot_at_timestamp` only selects a recorded commit timestamp (in whole seconds) across all commits; it does not provide effective-time reconstruction or branch-scoped bitemporal reads.

View clarification: A historical timestamp is another selector for the same state-pinned graph view model, not a separate kind of graph. A view fixes graph identity, lineage/version, effective time and optional recorded-time cutoff; subgraph filtering and graph-to-table conversion inherit those pins. The view may reconstruct lazily and share immutable storage. A table derived from it shows the same state and time, whether implemented as a lazy table view or a materialized table value. Current `Subgraph` fixes its ID sets but reads live parent values, while `g.table()` materializes current columns; both paths need an explicit state-pinned contract before they can be described as immutable historical views.

Recorded-time query clarification: The `recorded_by` cutoff is supplied only when the caller explicitly asks for an as-known-then view. An effective-time query without that cutoff uses the latest published version on the selected branch at view creation and remains pinned to that version afterward. Groggy still records the batch's acceptance time automatically as history metadata; making the query cutoff optional does not make that metadata optional.

Integration sequencing clarification: The owner deferred item 25 in the recent flat implementation list, the Graphiti ingest adapter, until Groggy's core graph, history, view, table, query, and analytics ideas are finished. In the staged roadmap this is item 20; its related Graphiti storage driver is roadmap item 25 and follows the adapter. Keep Graphiti as a future compatibility target and motivation for lossless core primitives, without treating integration as the next milestone.

Numerical-core priority clarification: The owner also deferred item 27 in the recent flat list, lexical search (roadmap item 22), along with Graphiti-oriented vector search and hybrid retrieval (roadmap items 23–24). This does not defer Groggy's vector and matrix *libraries*. Complete and optimize those existing numerical APIs as Groggy designed them: dependable vector/array operations, dense and sparse matrix behavior, shapes and broadcasting where promised, reductions, decompositions, and performance on columnar data. Use reference-result comparisons with NetworkX for graph algorithms and Neo4j for graph/query behavior while core work proceeds; Graphiti corpus comparison waits for integration.

## Emerging model

- **Identity and lifetime:** GraphSpace should be the authority for membership in the selected state; GraphPool stores entities, properties and history references. An inactive ID remains inspectable in history until an explicit destructive purge. Never reuse an ID while any retained state references it.
- **States and changes:** Every successful mutation should have a typed change record with entity ID, operation, affected field or endpoints, old and new typed values, effective time, recorded time, and origin/actor as available. Logical values must be recoverable even if storage deduplicates them by immutable content reference. A delta is the record of a transition; a binary or numerical difference between values is an optional compression technique, not the only source of truth.
- **Version language:** The public words `commit` and `checkout` are under review. Internally distinguish a mutation, an atomic batch, a named state/version, a branch, and a state switch. Existing API names remain until a compatible migration is designed. Actions and states may also be represented as graph entities, but that should not make the kernel's own recovery depend on querying itself.
- **Views:** A view is an immutable selection and value snapshot at a state. A method called *through* a view can propose a mutation to the owning graph; on success it creates a new state and returns a new view/result. The old view retains its original contents. This differs from current `Subgraph`, which fixes IDs but reads live values from its shared parent.
- **Graphiti scope:** Strengthen Groggy's graph engine first. The Graphiti release, lossless field list, driver scope and durability requirements are unresolved. The target may be a multiuser graph database; storage and concurrency requirements must be set before treating a Graphiti driver as the minimum engine.

## Architecture answers and recommendations

| Topic | Owner direction | Recommendation / unresolved choice |
| --- | --- | --- |
| AR1 — liveness | Inactive IDs remain historically inspectable until explicitly deleted. GraphSpace filters the pool. | Make membership state-relative: `active_in(state, id)`. Distinguish deactivation from destructive purge, and forbid purge while a retained state references the ID. |
| AR2 — iteration | Recommendation requested. | A graph read is pinned to one state; reject unknown/deleted start IDs. Iterate active edge **instances** by stable edge ID, once per edge, in ascending ID order for deterministic public output. The shared `adjacencies(node, mode)` primitive yields `(neighbor_id, edge_id)` for each edge, retaining parallel-edge multiplicity. Outgoing traversal follows source→target in directed graphs; undirected traversal exposes both directions without duplicating the stored edge. Preserve today's `neighbors()` multiplicity initially, and add a separately named `unique_neighbors()` if useful. A self-loop appears once in edge iteration; define degree convention separately. Provide explicit in/out/both modes rather than caller-specific reverse flags. |
| AR3 — change and visibility | Recover graph, IDs, properties and deltas; revisit `commit`/`checkout` terms. | Record typed before/after mutations plus topology and time metadata. A batch is all-or-nothing and becomes visible when its state/version is atomically published. Readers pinned to an older state finish on that state. Durable recovery also needs schema, index definitions, branch heads and a stable storage manifest; indexes themselves may be rebuilt if that is a documented recovery choice. |
| AR4 — views | Views are immutable snapshots, while methods can change the graph. | Split snapshot data from mutation command routing. Mutating through a view must not rewrite the view; return a new state/view or result. Decide whether a stale view may mutate the current graph or must fail on an expected-version mismatch; recommend the latter by default. |
| AR5 — sequencing | Recommendation requested. | First establish a narrow state-pinned read interface and a coordinated mutation path while keeping `Graph`/Python signatures. Next move queries, algorithms and views onto it; then repair history and persistence. Extract facade/FFI files only after responsibility boundaries are real. |
| AR6 — durability | Unknown while Graphiti use is being worked out. | Defer the exact guarantee. For a multiuser database, choose an explicit promise later: e.g. a successful durable save/transaction survives process crash, while a weaker export operation is documented separately. Do not equate a JSON save with transactional recovery. |

## Roadmap answers

### Stage 1 — core invariants

1. **Existing contract.** The engine contract already exists. The unresolved addition is state/version semantics for the intended use. Graphiti may need entities, facts and provenance; simple vector search does not inherently require storing raw text, but text is needed if Groggy must return, audit, re-embed or full-text search it. Keep this a use-case decision.
2. **Live state.** GraphSpace owns membership and filters GraphPool reads. Inactive IDs remain addressable in retained history. An immutable snapshot cannot become stale; a *derived cache* can, so key it by immutable state ID (and query parameters) or invalidate it on each mutation. The current `GraphSpace` version counter only covers topology changes, not a complete immutable state.
3. **Direction.** Graphs default to undirected unless explicitly directed, matching today's Python constructor. Mixed-direction edges are undecided. Recommend keeping graph-wide direction first and adding edge-level direction only with a concrete workload. Directed variants are needed for path/reachability, components, centrality, and any algorithm whose transition probabilities depend on outdegree; validate each algorithm's definition.
4. **Edge identity.** Preserve distinct edge IDs for parallel facts and self-loops. The owner leans toward permitting an explicit endpoint change as a recorded mutation. Recommend a dedicated `rewire_edge(id, new_source, new_target)` operation with atomic index changes; leave ordinary endpoint fields immutable. Whether a Graphiti fact's changed endpoints represent the same fact is an adapter-level identity rule.
5. **Bulk writes.** Confirmed: if any input is invalid, fail the entire batch atomically and identify every invalid input in the error. Validate endpoints, IDs, schema, types and uniqueness before allocation. Duplicate edge payloads can create parallel edges; duplicate external IDs inside a batch should be rejected or coalesced by a declared upsert policy, never silently filtered.
6. **External IDs.** UUID uniqueness is global. External IDs may change only explicitly. Conflicting upserts fail atomically. Updating an external ID must update its unique index in the same batch.
7. **Properties.** Scalars, temporal values, collections and vectors should be first class, and property type belongs to the value. Null is an explicit value; absent means no property assignment. A type change can be one typed `SetProperty(old,new)` event, rather than a compulsory delete-plus-add. Use schema constraints if a key must keep one type. The durable format should be versioned and need not be JSON.
8. **Labels/schema.** Multiple labels and ontology migration remain open. A graph may opt into an explicit enforced schema for nodes and edges; retain a flexible mode. Do not equate Groggy's existing `base`/`meta` `entity_type` with Graphiti labels.

### Stage 2 — states and durability

9. **Batch boundary.** Treat one episode ingest as one atomic batch even if it updates multiple Meta nodes, entities and facts. Recommend readers see the previous immutable state until publication. Start with one serialized writer per graph and concurrent snapshot readers; a service can coordinate multiple users. Revisit concurrent writers only if workload demands it.
10. **Storage.** Favor a versioned snapshot plus append-only change log, with both needed for bounded recovery and full history. "Fully recoverable after save" needs a precise crash point and acknowledgement rule before implementation. JSON size is a format tradeoff, not a durability definition.
11. **Value history.** Retain actual typed old/new values or immutable value references whose targets cannot be collected while referenced. Explicitly record set, unset, null assignment, and entity deletion separately. Content-addressed large blobs/vectors can deduplicate equal values later; first ensure correctness.
12. **State switch.** The owner wants it to change the working graph. Retain all referenced states by default; explicitly prune unwanted branches. Use a new public term if `checkout` suggests the wrong workflow. Rebuild or version indexes against the selected state. A snapshot is the stable read representation, while a state switch selects the graph's new working head.
13. **Reset/dirty state.** The owner leans toward a destructive reset; automatic change tracking versus explicit state recording remains open. A dirty state exists under the current API when `add_node` or `set_node_attr` changes the live graph after the last `commit` but before a new commit. Switching branches today rejects that case. With automatic versioning, there may be no dirty interval; distinguish rollback of an uncommitted batch from reverting a published version.
14. **Merge.** Preserve parallel edges. Define conflicts for delete-versus-edit, competing property assignments, endpoint rewires, and schema changes. Prefer caller-selected resolution with optional policy and an interactive tool above the core. Record the chosen resolution.
15. **Pruning/compaction.** Explicit branch deletion should remove history exclusive to that branch; shared ancestors or states referenced by another branch/view must remain. Auditing policy is open. Crash-safe compaction should write and verify a new snapshot/segment set, atomically publish a new manifest, then reclaim old unreachable segments after active readers release them. Git similarly separates reachable objects from unreachable ones and delays pruning of the latter; see the [Git cruft-pack documentation](https://git-scm.com/docs/cruft-packs).

### Stage 3 — temporal ingest

16. **Time.** An explicit effective timestamp on each property/fact change is desired, defaulting to the current clock when omitted. Also retain *recorded time* (when Groggy accepted the change) so a late correction can answer what was believed at an earlier date. At any selected state and time, one entity-property key resolves to at most one value; distinct fact edges may still overlap or contradict. Define "current" as latest published state evaluated at the query's default time, normally now.
17. **Provenance.** Episodes can be Meta nodes, and one underlying node can appear in several Meta-node collections. Raw episode content is intended to remain until explicit deletion. Corrections/withdrawals are recorded as property or relationship changes with provenance, not silent overwrite. Clarify whether multiple episodes jointly support one fact.
18. **Indexes.** Treat validity as time-qualified property/fact state rather than just a boolean. Global indexes are preferred. Maintain indexes incrementally on every published change; explicit rebuild is a recovery/repair tool. A temporal query must still apply the selected state/time, or a current-only index can return stale hits.
19. **Python.** Current Python `GraphQuery.filter_nodes` returns a `PySubgraph` with fixed IDs and a shared parent, so it is not an immutable value snapshot; `g.table()` materializes fresh columns. Audit remaining wrappers before labeling all views/copies. GIL means CPython's Global Interpreter Lock: on ordinary builds it limits simultaneous Python bytecode execution ([Python glossary](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)). `py.allow_threads` is currently used in visualization FFI paths, with a pipeline use only noted as a future possibility. Long Rust computations should release it when they hold no Python objects and the graph's Rust synchronization is safe. Version Python API compatibility explicitly, with deprecation for renamed state methods.
20. **Graphiti adapter.** Target release, exact lossless fields and object-versus-interchange format are open. Research the data contract before choosing. Strengthen Groggy independently first.

### Stage 4 — query and retrieval

21. **Query engine.** `QueryEngine` remains in `src/query/query.rs`. `Graph::find_nodes`/`find_edges` and Python `GraphQuery.filter_nodes`/`filter_edges` use it; `Graph::query()` is only the unimplemented *complex builder*. Preserve and extend the working filter API before selecting a new language.
22. **Text retrieval.** Phrase/prefix search is not required. Vector similarity is useful but cannot replace lexical search when the requirement is exact words or document retrieval. If Groggy stores only embeddings, it cannot reproduce source text from them. Text reindexing should consume the same logical property-change event as other indexes; it need not compute a delta of a delta.
23. **Vector retrieval.** Start with exact search over stored vectors, using the numerical layer and a defined similarity metric. This is O(number of vectors × dimension) until an approximate index is added. Multiple embeddings per entity can live under distinct named attributes; record dimension/model/version per embedding attribute or index, not just one graph-global value if models may coexist.
24. **Hybrid search.** Method chaining is a suitable API, but score fusion needs a rule because cosine similarity, text ranking, recency and graph distance have different scales. Start with explicit filters plus independently inspectable scores and a simple weighted normalization only when needed. Evaluate with a small query set whose expected relevant entities/facts are hand labeled.
25. **Driver.** Prefer Graphiti operation interfaces over a Cypher translator. Unsupported calls should raise explicit errors. Upstream compatibility requires pinning a target version and running its own ingest/search tests in CI when the driver exists; target version is open.

### Stage 5 — connected computing

26. **Views.** The owner explicitly chose immutable snapshot views. A view may route an update to the graph but remains pinned to its original state. Cache entries are keyed by state ID, graph ID and query parameters; immutable snapshot caches do not become stale. A "latest" alias or mutable working-head cache must refresh after publication.
27. **Tables.** A join is a result/view until an explicit write. Property edits through a table are graph mutations producing typed change records. Decide whether topology columns can be edited through a table; recommend an explicit edge operation. Existing `g.table()` materializes a copy, so making it a live view would be an API/semantic migration, not a rename.
28. **Algorithms.** Treat the existing basic algorithm set as core. Default results should be tied to an immutable selected state, with an explicit "latest" shortcut if desired. Weight constraints depend on the algorithm: Dijkstra requires nonnegative weights; other algorithms may allow negative values or no weights.
29. **Numerics.** Prefer in-house implementations, including sparse numerical work if promised. CSR currently represents sparse graph adjacency; it does not by itself establish a complete sparse matrix API. Set precision/error and sparse semantics per operation before claiming parity with mature numerical libraries.
30. **Builder.** Preserve the substantial existing pipeline/builder work and review its current distance to the desired workflow before redesign. Naming, caching and versioning of intermediate/pipeline results remain open; pin every run to an input state first.

### Stage 6 — proof and operations

31. **Reference testing.** Neo4j is the confirmed reference backend. Compare observable results first and document any justified divergence. Inject failures at every persistence publication boundary and representative ingest stage, including restart during write.
32. **Scale and latency.** One million through one billion entities is an ambition, not one benchmark tier. Define small/medium/large measured tiers and a first target. Interactive latency cannot literally be zero; set percentile targets for representative queries. Embeddings increase storage and computation unless they replace retained text by product choice; they do not automatically speed arbitrary graph operations. Memory budget depends on whether Groggy is embedded, service-backed, or both.
33. **Deployment.** A common core can support embedded single-process use and a multiuser service wrapper. The service needs authentication, tenancy/isolation, concurrent snapshot readers, coordinated writers, backup and observability. Choose consistency/availability targets later. Useful diagnostics include write latency/failures, recovery time, state/index lag, cache hit rate, memory, active readers, and rejected/conflicted mutations.
34. **Completion.** "Implemented" means the intended Rust and Python behavior, persistence where promised, correctness and failure tests, and accurate docs. Do not retire original aspirations merely because they are deferred; mark their state honestly. Update examples/docs with each release rather than only after the full roadmap.

## Decisions needing a focused follow-up

1. Does "inactive ID explicitly deleted" mean a visibility change, a permanent purge, or both as separate operations? The recommendation above assumes both.
2. May a view targeting old state send a write to the latest graph, or must it first refresh? The recommendation is optimistic version checking and rejection on stale state.
3. Are versions created automatically for every mutation, only for successful batches, or only when explicitly named? This determines whether a dirty working state exists.
4. Is a property value allowed to change type without a schema migration? The recommendation is yes in flexible mode, no when a schema fixes the type.
5. Define the policy for duplicate external IDs within one batch: reject them all, or coalesce only an explicitly declared idempotent upsert.

## Item 5 implementation record — 2026-10-20

Node/edge creation must parse all supplied Python properties before entity allocation. The Rust `add_nodes_with_attrs` boundary owns system-managed node-property validation and writes accepted rows through existing columnar setters. A malformed value, row, missing external key, or reserved node property leaves IDs unallocated.

Bulk removals now validate all IDs first. Duplicate targets reject the call, consistently with duplicate property assignments; valid node removals may share incident edges and remove each edge once. Rejected calls leave topology and change tracking unchanged.

Conversion errors aggregate across properties, node rows and bulk setter columns. Read-only Rust preflight validates decodable targets and reserved properties even when another value fails conversion, producing one combined report with original input positions. Temporary placeholder values never reach a write when any error exists. Structurally malformed input is reported at its row/column boundary. Automatic versions, schema enforcement, external-ID uniqueness, and crash recovery remain assigned to their roadmap items.

Verification: the full locked Rust test suite passes, as do 71 focused Python/conformance/CRUD checks against the rebuilt extension. Rust formatting and diff whitespace checks pass. Strict Clippy remains blocked by nine existing sort/match lints outside this change; Black, isort and pre-commit are unavailable in the local runtime. The regenerated 37-page workbook was rendered and its affected pages visually checked.

## Item 6 implementation record — 2026-10-20

Managed identity uses the `external_id` property and a separate allocation-lifetime index owned by the graph coordinator. The property is versioned with the existing typed change record; the index retains every assigned name and its internal node/edge identity across branch switches and reset. Uniqueness is global within one graph instance, across entity kinds and groups. IDs are nonempty, opaque, case-sensitive strings; a later Graphiti adapter can normalize UUID objects. An explicit `group_id` filter limits reads without changing uniqueness.

`upsert_nodes` and `upsert_edges` validate complete typed inputs, reject duplicate input IDs even when payloads match, and patch only supplied properties. Identical payloads cause no new dirty state. Edge endpoint changes reject the upsert; undirected input may reverse equivalent endpoints. Python parses every row first and combines syntax errors with semantic failures from other decodable rows, retaining input positions. Custom input key names are consumed into `external_id`, not mirrored into a second identity property.

`set_node_external_id` and `set_edge_external_id` explicitly assign or rename an identity. Old names and removed entities remain reserved, consistently with retained numeric allocation IDs; reset also retains reservations from discarded uncommitted work. Normal lookup sees the selected state's active name. Allocation lookup can resolve retained names to an internal entity for historical inspection. Reusing an inactive/old name in an upsert fails; restoring state or explicitly renaming the same entity is required. A snapshot materialized as an independent graph has only its selected-state bindings.

The dedicated property cannot be set through ordinary creation/setters, bulk columns, mixed batches, or subgraph writers. Legacy `uid_key` remains an ordinary-property convenience with creation-time duplicate checks and ambiguity errors; it is not a lasting property uniqueness constraint. Unsupported `add_node(node_id=...)` now raises before allocation instead of substituting a different numeric ID. `add_graph` rejects copying managed identities before allocation; callers use explicit upserts until identity-aware graph merge exists.

The audit also repaired subgraph bulk writers that bypassed coordinator validation and omitted change tracking. Those writes now produce ordinary property deltas. Current JSON export/import now preserves compact/compressed text and restores active numeric IDs and endpoints directly, including sparse IDs. Complete history, inactive/old-name reservation metadata, discarded allocation counters and crash recovery remain item 10. Node and edge upsert calls are separate direct write boundaries using the current manual `commit`; mixed episode publication remains item 9.

Verification: the full locked Rust suite passes, including 13 external-identity conformance tests. The rebuilt Python extension passes the regression suites and all 14 focused identity tests, including node/edge batch bypass checks and combined syntax/semantic diagnostics. The new API documentation example runs successfully. Formatting and diff checks pass. Strict Clippy still reports the same nine existing sort/match lints outside this feature. The 37-page workbook was regenerated and the identity page visually checked.

## Item 7 implementation record — property bridge repair, 2026-10-20

The existing property enum remains the source of value types. Python conversion is centralized and fallible: binary values return bytes, homogeneous sequences retain their boolean/integer/text/float type, and structured JSON returns decoded containers. Explicit vector constructors preserve empty-vector types; typed snapshot values can be reused as writer inputs. Structured values require string keys, reject cycles and excessive nesting, and do not accept non-finite JSON numbers. Native float properties remain f32 and integer properties i64; oversized integers reject instead of falling back to a lossy float.

Core value identity now obeys equality/hash/ordering laws across all existing variants and storage optimizations. NaNs and signed zero are canonicalized for property identity, without changing numerical arithmetic. Collection ordering compares contents. Numeric-looking text preserves natural numeric ordering with a literal tie breaker and separates numeric from nonnumeric text to ensure transitivity. Compression validates decoded size and uses safe explicit little-endian float decoding.

Recorded snapshots retain typed values and distinguish explicit Null from absence. Python structures are returned as owned copies. Ordinary subgraphs remain live reads until item 26. Native temporal types, precision extensions and complete disk round trips remain unfinished; item 7 is not closed by this repair. Unexposed accessor mutation helpers were identified during testing; their public API design remains item 27 rather than being added incidentally here.

Verification: the full locked Rust suite passes, including four property identity/compression/history tests. The rebuilt Python extension passes 101 focused property, identity, validation, direction, batch, diff, array and table checks; 79 existing table checks are skipped by their gates. Tests exercise supported values through bulk setters and mixed-batch publication as well as graph/table/snapshot reads. The documented example runs. Rust formatting and diff whitespace checks pass. Strict Clippy reports the same nine pre-existing lints outside this change. Black, isort and pre-commit remain unavailable locally. The 37-page workbook was regenerated and item 7 visually checked.

## Item 7 implementation record — typed active graph export, 2026-10-20

Replaced the two scalar-only graph serialization loops with a shared strict logical value codec and a dedicated graph persistence module. Version 2 exports preserve inline types, null, empty vectors, binary data, structured JSON and float bit patterns, independent of compact/compressed memory layout. Active IDs, endpoints, direction and managed identity names round-trip. Python save/load methods directly delegate to this core path and accept filesystem paths. Legacy unversioned scalar documents remain readable; untagged legacy containers are JSON, not guessed vectors. Unknown versions, malformed records/properties and duplicate numeric IDs fail explicitly.

This is an active graph interchange format, not durable full-engine storage. Meta-nodes and pooled subgraph references reject export because backing pool contents are absent. Validation and encoding finish before opening the destination, preserving existing data on validation errors. I/O failure atomicity, history/branches, inactive identities and reservations, allocation high-water state, indexes and pool recovery remain item 10. Native temporal values are still open under item 7. JSON does not select the eventual durable storage backend.

Verification: the full locked Rust suite passes, including four dedicated typed export tests covering bitwise float preservation, inline value families, deterministic encoding, legacy input and malformed documents. The rebuilt extension passes 65 focused Python property/persistence, identity, mutation, batch and diff checks. Both property API examples run. Formatting and whitespace checks pass. The workbook was regenerated and item 7 visually checked.

Strict Clippy still reports the same nine pre-existing sort/match lints outside this change; no new lints were reported in the persistence modules. Black, isort and pre-commit remain unavailable locally.

## Item 8 implementation record — semantic kinds and explicit schema audit, 2026-10-20

Node `labels` is a versioned TextVec of distinct nonblank literal names, stored in sorted order. A bare empty vector clears labels and normalizes to TextVec. Edge `relationship_type` is optional nonblank text; null explicitly clears it. These semantic kinds do not replace protected base/meta `entity_type`, change endpoints, or deduplicate parallel edge identities. New unlabeled/untyped entities remain valid. Getters use checked active-state reads; label/type selection scans columns and returns sorted active IDs until item 18 indexing.

Shared core validation runs before allocation or writes in checked creation, scalar/bulk/column mutation, external-ID upserts and GraphBatch preparation. Internal bulk normalization preserves the existing columnar storage path. Upserts normalize labels before equality checks so repeating an unsorted label list remains a no-op. Snapshots retain typed prior values, and active export preserves semantic kind properties.

`GraphSchema` defines optional per-label and per-relationship rules, with required, nullable and logical type constraints. Every matching label applies; unknown labels and extra properties remain flexible. `validate_schema` is a deterministic, structured audit, with no state mutation or installed write policy. Unknown schema fields/types and blank selectors reject explicitly.

Graph-wide schema enforcement remains open and is coupled to item 9: current infallible Rust `add_node`/`add_nodes` cannot report missing required fields, and multi-property writes need proposed-state validation. Enforcing only selected setters would leave bypasses. A compatible checked creation/mutation path, schema policy persistence/versioning, ontology migration and endpoint constraints must follow; this audit does not establish those guarantees. Original enforced-schema aspirations remain on the roadmap.

Verification: the full locked Rust suite passes, including five kind/schema conformance tests. The rebuilt extension passes 94 focused Python kind/schema, property, persistence, identity, validation, direction, batch and diff checks. Tests cover all matching labels, stable historical values, typed export, invalid allocation-free creation, bulk rejection, idempotent upserts and one mixed-batch publication. API examples run and generated method stubs are updated. Formatting and whitespace checks pass. Strict Clippy still reports the same nine pre-existing sort/match lints outside this work; Black, isort and pre-commit remain unavailable locally. The 37-page workbook was regenerated and the affected kind/schema and mutation pages visually checked.

## Item 9 implementation record — schema-controlled publication, 2026-10-20

Rust `apply_batch_with_schema` and Python `apply_batch(..., schema=...)` validate the entire proposed resulting graph before allocating entities, appending property values, recording changes or publishing history. A read-only assignment overlay reuses existing columns; it does not clone/reconstruct the graph. The same validator supplies explicit schema auditing and enforcement, preserving identical required/nullability/logical-type semantics. The opt-in path scans the proposed graph, including untouched surviving entities; it is not an incremental index.

Proposed-state validation combines new labels/types and their property updates, supplies the system base entity_type for new nodes, and excludes explicitly removed nodes/edges and incident edges lost through node removal. Batch preparation normalizes semantic kinds before validation, so an empty label assignment has its accepted TextVec type. Schema findings aggregate in deterministic order and report prospective IDs, which remain unallocated on rejection. Existing expected-state and clean-working-state guards apply before execution. A successful schema-controlled write uses the existing bulk coordinator and publishes one state.

The schema is explicitly per call. It is not installed on the graph, so future flexible calls and legacy direct mutations do not inherit it. This closes the proposed-state enforcement slice shared by items 8 and 9, not graph-wide policy. A previously invalid graph can succeed if the batch repairs/removes the invalid entities or selectors; unrelated writes cannot silently preserve violations under the supplied schema.

Direct-write migration review: automatically publishing inside the current nested scalar/bulk writers would publish multiple intermediate states during creation and mixed batches. Current infallible Rust add_node/add_nodes also cannot report validation/publication failures, while legacy users expect dirty state followed by manual commit. The next migration must separate checked public mutation coordination from internal column writers and offer a compatible checked/publication path; changing only selected setters would leave bypasses. Graph-wide installed/schema policy persistence, effective-time defaults/overrides, index publication, checkpoint naming and crash recovery remain open. No new durability or concurrent-reader guarantee is claimed here.

Verification: the full locked Rust suite passes, including seven schema-batch tests for single publication, old snapshots, combined label/property updates, unchanged invalid survivors, removals, high-water allocation and stale/dirty guards. The rebuilt extension passes 105 focused Python schema, kinds, property, persistence, identity, mutation, direction, batch and diff checks. Schema API examples execute, and the generated apply_batch stub exposes the keyword-only schema argument. Formatting and whitespace checks pass. Strict Clippy still reports the same nine pre-existing sort/match lints outside this work; Black, isort and pre-commit remain unavailable locally. The 37-page workbook was regenerated and affected pages visually checked.

## Item 9 ongoing implementation — publishing writer and change times, 2026-10-20

Added an opt-in checked direct-write session (`Graph.writer`) over the existing batch publication coordinator. Creation with properties, scalar/bulk columns, kinds and removals each publish one state. Rust borrows the graph for the session; Python sessions retain an expected head and require explicit refresh after another writer advances it. A copied/borrowed schema policy applies to each session call. Legacy methods remain an explicit compatibility path and dirty legacy work rejects publication. This transition avoids publishing nested internal setters and does not silently change established manual-commit APIs.

Automatic publications capture one engine-assigned recorded clock in signed Unix microseconds. Batch effective time defaults to that clock, or an explicit batch time; per-change overrides target actual staged operations and reject unknown/duplicate targets before allocation. Initial properties inherit creation time unless explicitly overridden. Implicit incident-edge removals use the earliest removed endpoint time; explicit edge removal controls its own time. Metadata keys the existing typed delta without duplicating its property payloads, and immutable commit hashes include publication metadata. Legacy manual commits retain their existing behavior and report no automatic publication metadata. Effective-time reconstruction remains item 16; these records do not change the old commit-second timestamp query into an effective-time query.

The full Rust suite and focused writer/time/batch checks pass at this checkpoint. Managed identities/index publication, complete policy coverage, durable publication and recovery are still being worked through before claiming item 9 complete.

## Item 9 ongoing implementation — identity/index publication, 2026-10-20

Mixed batches now support explicit managed identity assignment and rename for existing IDs or new handles. Read-only preparation checks duplicate identities and names, retained resulting targets, graph-lifetime reservations and schema constraints before allocation. Managed values are inserted into prepared columns after ordinary system-property writes have been rejected. Execution uses the validated internal column writers, publishes reservations alongside property deltas, and records the external_id change's effective time. Current identity lookup and prior snapshots agree with their selected state; old aliases remain reserved.

Checked writers now expose managed upserts and renames. Upsert planning reuses existing core validation without allocating, and normalizes properties before detecting a no-op. An identical normalized upsert checks expected head and schema but publishes no new state. Native Python publishing upserts return the exact state with IDs, avoiding adoption of another writer's later head. Legacy upserts retain their manual-grouping behavior. The next dependency is crash-safe durable publication/replay (item 10); item 9 remains open while this foundation is implemented.

Verification at the identity checkpoint: the full locked Rust suite passes, including three mixed-identity tests, two checked-writer tests, four publication-time tests and seven schema-publication tests. The rebuilt Python extension passes 122 focused checks across these features and prior conformance suites. All publishing API examples run. Formatting and whitespace checks pass; strict Clippy reports only the same nine existing sort/match lints. Durability is the next dependency; this entry does not mark item 9 complete.


## Items 9–10 durable publication record — 2026-10-20

The checked transaction boundary now has a durable implementation: DurableGraph owns the graph and a locked versioned CBOR journal. It validates all proposed data/identities/policies, captures the acceptance clock, appends and syncs a record, then publishes the prepared mutation. Only the checked surface promises this boundary; legacy Graph manual writes retain compatibility. Installed schema revisions persist globally and per-call schemas can only add checks. Branch actions are journaled and preserve allocation-lifetime reservations. The immutable Rust read handle cannot call pool_mut, which now requires an exclusive graph borrow.

Recovery restores accepted versions, typed deltas, exact float bits, original publication clocks, deterministic history hashes, branches, configuration, policies and high-water counters. Partial final records are truncated; complete corruption errors without truncation. Append failure rolls back when possible. Failed rollback produces an indeterminate result and poisons the open store until reopen. Synced transactions can survive an unacknowledged crash; managed identities support reconciliation. A 256 MiB record bound rejects before writes. One exclusive OS writer lock is the current concurrency boundary; Python holds the GIL during I/O/replay.

Item 9's checked publication mechanism is implemented for its documented scalar, columnar, mixed-batch and managed-upsert operations. This does not claim full engine persistence or multi-client service support. Item 10 remains active for pooled/meta references, graph import, checkpoints, compaction/migration and bounded recovery; item 12 retains explicit commit-to-checkpoint migration. Journal recovery is operation replay, not effective-time graph reconstruction (item 16). Stored subgraph references are explicitly rejected until backing pool contents can be retained correctly.

Verification: targeted durable tests cover every inline logical value family, stricter per-call schema replay, transient entity lifecycle, retained branches/IDs and publication clocks. Failure tests include interrupted header/payload boundaries, corrupt lengths/checksums, write/sync/rollback failure, and sync-before-publication recovery. Python additionally exits a subprocess without closing and verifies recovery. See docs/api/durable-storage.md for the exact supported boundary.

Final journal verification: full locked Rust suite passed (416 library tests, 2 ignored, plus integration/doc suites), and 122 focused Python tests passed against the rebuilt extension. Strict Clippy retains eight pre-existing sort/match diagnostics; the new large-record enum warning was fixed with boxed payload storage. Workbook pages 9–10 were regenerated and visually checked.


## Item 10 backup and retained-reference review — 2026-10-20

DurableGraph.backup produces an independent complete journal through a synced sibling temporary file, replay verification, no-overwrite publication and directory sync. It preserves retained branches/history/reservations as well as active values; reopening uses the same recovery API. Failed copy/verification removes the temporary file. A directory-sync error after publication can leave a destination requiring reconciliation. This is a full log backup, not a compact checkpoint; verification replays the log.

The next pooled-reference repair must address two existing issues together: node/edge removal currently mutates stored subgraph membership, and branch reset replaces the backing pool. Serializing an opaque SubgraphRef alone would therefore not make historical content safe. Keep pooled/meta journal writes rejected until immutable referents and exact snapshot restoration are implemented; this intersects items 12 and 26. Preserve the original vision and explicit deletion policy rather than silently exporting incomplete data.

Backup verification: the full locked Rust suite passes, including eight durable integration tests and six journal failure-injection tests; 123 focused Python checks pass against the rebuilt extension. Formatting/diff checks pass. Strict Clippy reports the same eight pre-existing sort/match diagnostics and no new storage diagnostics. Updated workbook item 10 was rendered and checked.


## Items 10/12 retained membership repair - 2026-10-20

Continued accepted correctness work while the owner reviews the appendix. Working deletion no longer edits stored selection membership. Reset/branch switch now rebuilds GraphSpace on the existing pool, preserving historical column indices, allocated endpoints and subgraph backing records. A missing meta referent rejects before allocation. Reference-hash collisions reject without replacement; compatibility mutation helpers are deprecated no-ops. Detached temporal materialization explicitly rejects missing pool backing rather than creating dangling references.

Python regression checks also exposed redundant protected entity_type writes in collapse. Those writes were removed, centralized checked meta creation is reused, and strict collapse now honors the same node-removal strategy as allow-missing collapse. Nested original meta identities remain intact. This repairs retained membership and collapse behavior, not full immutable expanded property reads or durable meta storage. Repeated switches still append restored property values; bounding that allocation remains item 12. No appendix choice about group revision identity or detached history transfer was assumed.

Verification: full locked Rust suite passes; three new retained-subgraph checks and 45 focused Python retention/durability/schema/kind checks pass against the rebuilt extension. Source workbook/PDF questions remain unchanged while the owner answers them. See docs/api/retained-subgraphs.md for the supported boundary and remaining work.


## Item 29 identity and axis-statistics repair - 2026-10-20

Continued numerical repairs that follow the existing API while workbook decisions remain with the owner. eye now builds a diagonal identity. Checked identity construction no longer goes through an infallible zero-matrix fallback. Empty aligned buffers retain logical zero length, so 0x0 identities and rectangular empty shapes preserve dimensions. Checked shape/byte multiplication and shared-buffer data lengths reject invalid allocation inputs.

Axis standard deviation previously returned sums. It now derives from sample variance with the existing n-1 divisor and Groggy row/column axis naming. Variance/std use row-major Welford updates with f64 intermediate accumulators; their result type remains T. Existing zero behavior for empty/finite singleton reductions is retained. f32 conversion preserves NaN/infinity; finite output conversion failure is explicit. This does not settle widened integer statistical outputs or precision guarantees in appendix A13. It does not complete SVD, eigendecomposition, sparse algebra, other reductions or gradients. See docs/api/matrix-statistics.md for those limits.


Final numerical verification: the full locked Rust suite passes (416 library tests, two ignored, plus integration/doc tests), including five new matrix-statistics checks. The rebuilt extension passes 133 focused Python checks, including statistics comparisons and retained-subgraph/durable regressions. Nested-list construction was verified as column-oriented; that behavior is preserved and documented, rather than changed implicitly. Dense reductions hold one read guard and obey row/column storage strides. The recorded 1024x384 f64 debug fixture improves from roughly 36 ms to 17 ms per axis reduction, with raw samples and limits in notes/benchmarks/matrix-statistics-2026-10-20.md. Strict Clippy retains the same eight pre-existing sort/match diagnostics; no new numerical diagnostics were reported. The owner's workbook PDFs remain at their answerable baseline.


## Item 29 real decomposition repair - 2026-10-21

Replaced the guessed eigenpairs and zero-filled SVD factors with in-house bounded Jacobi kernels. Real SVD now handles tall/wide/square, rank-deficient, zero and empty inputs with full orthogonal factors and min(m,n) sorted singular values. It rotates columns directly, scales finite input, uses a documented numerical-rank floor and fails explicitly on exhausted iteration budget or non-finite/unrepresentable results. Real symmetric eigendecomposition returns complete orthogonal column eigenvectors and descending algebraic eigenvalues; general nonsymmetric/complex input is still unsupported and rejects instead of fabricating an answer. Both f32/f64 Rust matrices are supported using f64 intermediate computation. These are engineering rules for the current kernels, not a new accepted answer to the wider precision/output questions in appendix A13.

Integration checks cover PCA distances in a rotated rank-two plane and actual path-graph Laplacian eigenvectors. That exposed arbitrary spectral row ordering; spectral matrices now use sorted active node IDs with node_<id> row labels. Broader directed/sparse spectral definitions and decomposition gradients remain separate work. Full output memory is quadratic; no reduced-factor, sparse or general-eigen completion is claimed. Workbook PDFs remain at the owner's answering baseline.

Verification: full locked Rust suite passes (417 library tests, two ignored, plus integration/doc suites), including eight decomposition integration tests and a forced exhausted-budget unit test. The rebuilt extension passes 152 focused Python checks, with 19 NumPy comparison tests covering random, ill-conditioned, repeated, rank-deficient, rescaled and invalid inputs. Clippy remains blocked by eight existing sort/match diagnostics. Debug timing/residual baselines are recorded in notes/benchmarks/matrix-decompositions-2026-10-21.md. Algorithm family reference: https://www.netlib.org/lapack/explore-html/d9/deb/group__gesvj_ga7aec05d2a1523bbeee77ece21b12187c.html .


## Item 28 checked traversal and weighted Dijkstra repair - 2026-10-21

Continued accepted algorithm correctness work without choosing open workbook policies. Graph BFS/DFS reject unknown/inactive starts; shortest/all-path methods validate both endpoints before execution. Weighted Graph shortest path validates every active edge before target early exit. Missing weights remain unit weight; explicit nonnumeric/null/bool, negative and non-finite values reject. Compact integer storage now decodes as its real weight rather than silently becoming one.

Dijkstra algorithm, weighted ShortestPathMap step and weighted closeness reuse checked scalar/selected-entity validation. Distance-only maps choose minimum selected parallel weights and populate both orientations for undirected graphs, preserving the underlying edge instances. Dijkstra relaxations no longer discard small improvements using an absolute epsilon, and heap equality/order agree. f32 persisted algorithm distances outside finite range reject before writing any result column; direct path/pipeline-variable costs retain f64. This remains legacy algorithm persistence, not a durable write expansion. Other weighted algorithms, option handling, cancellation and immutable-view migration remain item 28/related work.

Verification: full locked Rust suite passes (417 library tests, two ignored, plus integration/doc suites). Seven focused pathfinding integration tests cover invalid/live endpoints, missing/compact/zero/invalid weights, reverse and parallel paths, pipeline parity, tiny costs, f32 output range rejection and comparison against an independent Floyd-Warshall fixture. The rebuilt extension passes 163 focused Python checks including 11 pathfinding input/identity cases. Clippy retains eight pre-existing sort/match diagnostics. NetworkX was not installed, so no NetworkX/Neo4j differential-comparison completion is claimed. Workbook PDFs remain at the owner's answerable baseline.


## Item 28 checked A* repair - 2026-10-21

A* reuses selected-state and nonnegative-weight validation for both weighted and unit-weight calls. Undirected reverse edges, compact integers, missing unit weights and minimum parallel pair costs are respected. Start/goal attributes must each occur on exactly one selected node. These are presence markers, preserving the existing value-independent convention.

The supported heuristic domain is explicitly finite nonnegative numeric estimates, missing zero, goal zero, and consistency on every selected oriented pair using minimum parallel weight. This is sufficient for early goal termination; admissible but inconsistent heuristics reject for now. No heuristic means zero. Heap equality/order agree and exact relaxation preserves tiny improvements. Preflight includes trivial paths, before any output-column mutation. Output remains zero-based node path order and only path rows are written; it does not expose edge-instance selection or clear unrelated previous output rows. Other algorithms and canonical ties remain open. Focused native and Python tests cover reverse/parallel compact weights, valid and invalid heuristics, ambiguous markers, early-exit validation and tiny improvements. The user's workbook PDF baseline remains unchanged while answers are pending.

Verification: locked offline full Rust suite passed (417 library tests, 2 ignored, plus integration/doc suites including five new A* cases). Rebuilt Python extension: 60 focused pathfinding/algorithm API/apply tests passed, including 14 new A* cases. Strict Clippy remains blocked by the same eight existing sort/match diagnostics outside this slice; no new A* diagnostic. Formatting and diff whitespace checks passed.


## Item 28 node betweenness repair - 2026-10-21

Node Brandes validates selected live entities before trivial returns/cache use. Weighted lengths must be finite positive numeric scalars (compact integers included), missing one; unit mode ignores unused weight properties. Zero lengths reject for this implementation's strictly advancing predecessor DAG. Equal minimum parallel edge instances contribute separate shortest paths; expensive parallel alternatives are excluded through a local weighted CSR that does not contaminate the shared all-edge cache. Positive self-loops contribute no shortest paths.

Directed raw scores are no longer incorrectly halved. Normalized scores use the appropriate directed/undirected factor including isolates. Exact f64 distance/tie comparisons replace absolute epsilon comparisons and heap equality/order agree. Rounded-away positive steps, path-count overflow and nonfinite/out-of-f32-range output reject before persistence. Distinct edge-instance paths are the explicit multigraph convention; no external backend parity is claimed. Weighted CSR adds O(E) local storage and validation work; no performance improvement is claimed. Exhaustive simple-path reference fixtures cover weighted/unit, directed/undirected, normalized/raw results and cache isolation. Girvan–Newman's direction, pair-based removal, modularity evaluation and weighted domain still require a separate repair; this does not close item 28.

Verification: locked offline full Rust suite passed (417 library tests, 2 ignored, plus integration/doc suites including five new betweenness cases). Rebuilt extension: 73 focused Python tests passed, including 13 new betweenness cases. Strict Clippy reports the same eight existing diagnostics outside this slice; targeted test Clippy passes with those two existing lint categories explicitly allowed. Targeted Rust formatting and diff whitespace checks pass.


## Item 28 Girvan–Newman repair - 2026-10-21

Replaced endpoint-pair masks with a local edge-instance CSR. Both orientations are traversed for undirected input; directed graphs now explicitly reject rather than produce an accidental projection. Equal shortest parallel instances contribute separately and each removal affects exactly one original edge ID. More expensive surviving parallels become eligible after shorter ones are removed. Removal ties choose lowest ID; equal computed partition scores keep the earlier partition. Original graph topology is unchanged; only final node labels are persisted through the existing legacy result-column path.

Validate live selection, positive finite numeric lengths (missing one), finite nonnegative modularity threshold and nonnegative level budget before publication, including zero-budget/trivial cases. Shared checked validation decodes compact integers. Self-loops stay in original modularity with degree two but are not removed or traversed for shortest paths. Unit BFS and weighted heap paths retain separate efficient frontiers with reusable buffers. Reject unrepresentable advancing distances and path-count/dependency overflow; cancellation is checked during traversal.

The budget counts individual non-loop removals. Score initial components and every post-removal partition, including the final allowed removal, using all original edges and degrees. Lengths affect pathfinding only; modularity remains unweighted, preserving the distinction from strength weights. Default stop threshold is 0.0001 below the best score; this returns the best visited partition, not a global optimum or full dendrogram. Empty/singleton/disconnected graphs and isolates are valid. Added a thin Python community.girvan_newman factory. Edge-instance CSR and buffers cost O(V+E) additional memory; no performance improvement is claimed. Other community/algorithm-domain audits remain item 28. Workbook PDF baseline remains unchanged pending user answers.

Incidental item 5 fix, committed separately as `a121fc0b`: the empty-graph test exposed `add_nodes(0)` underflow in pool range allocation. Graph and pool zero-count operations now preserve the next ID and return no nodes; the pool tuple explicitly uses `(next, next, [])` for the empty case. Native tests cover initial and nonzero counters.

Verification: locked offline full Rust suite passed (420 library tests, 2 ignored, plus integration/doc suites). Three new private-kernel tests compare exhaustive edge-instance path scores and numerical failures; six integration tests cover partition/scoring/input behavior and two allocation tests cover the incidental fix. Rebuilt extension: 91 focused Python tests passed, including 18 new Girvan–Newman cases. Strict Clippy reports the same eight existing sort/match diagnostics; targeted Girvan–Newman test Clippy passes with those two existing lint categories explicitly allowed. Targeted Rust formatting and whitespace checks pass.


## Item 28 Louvain/Leiden local objective and domain repair - 2026-10-21

Resolution now scales the modularity null-model term through shared objective/move-gain helpers. Louvain previously ignored gamma; Leiden multiplied the entire move gain and therefore did not change preferred moves. Existing gamma-one helper signatures remain compatible. Objective checks use a separate full adjacency-matrix definition across all small partitions and moves, including self-loops and parallel edge instances. This is objective-level conformance, not full NetworkX partition parity.

Both algorithm paths explicitly require an undirected live selection and preserve loop/parallel multiplicity in their unweighted modularity data. Louvain now writes singleton labels for nonempty edgeless selections. Gamma must be finite positive; budgets positive; Leiden seed nonnegative. Validation/cancellation precedes trivial returns and output publication. Sorted candidates/node IDs remove unordered local choices; Leiden uses a local seeded shuffle and deterministic refinement/relabeling without changing process RNG state. Preserve columnar result writes and core Rust computation; no new durable mutation boundary or performance improvement is claimed.

Full Louvain graph aggregation/multiple levels and full Leiden refinement/aggregation remain open. Louvain still executes one local phase despite a positive legacy max_phases parameter; document this explicitly. Leiden still uses connectivity splitting refinement. Infomap, label propagation, connected-components and wider algorithm domain audits remain item 28; immutable result-view migration remains item 26. Preserve the user's workbook PDF baseline while answers are pending. See docs/api/community-modularity.md for exact domains and evidence.

Verification: locked offline full Rust suite passed (420 library tests, 2 ignored, plus integration/doc suites including six new community-resolution checks). Rebuilt extension: 110 focused Python tests passed, including 19 new resolution/domain cases. Strict Clippy remains blocked by the same eight existing sort/match diagnostics outside this slice; targeted test Clippy passed with only those two existing lint categories explicitly allowed. Targeted Rust formatting and diff whitespace checks pass. The shared objective convention is linked to official NetworkX documentation in docs/api/community-modularity.md; no external backend was installed or executed.


## Item 28 Louvain multilevel aggregation - 2026-10-21

The formerly ignored phase budget now controls successive local-move/quotient levels. Original unit edge instances are aggregated into checked u64 masses; internal connections become loops and loop degree is doubled. Preserve the original objective and map final labels through every level back to original node IDs, including gaps and isolates. A private count-weighted CSR kernel replaces per-candidate rescanning of the original local adjacency. Source topology and shared subgraph cache are not modified. Check finite gains/objectives, nondecreasing phase scores and aggregation objective equality before final legacy bulk-column persistence. Mass overflow or invalid internal partitions reject.

Registry and Python default max_phases is now ten; explicit one requests local-only execution. The Python parameter is appended after existing output_attr to preserve positional meanings. Rust's explicit constructor is unchanged. Defaults can yield different, improved partitions; output labels remain opaque. max_iter bounds sweeps per level, and early termination remains possible. No global optimum, general external edge-weight support, arbitrary precision or large-scale performance guarantee is added.

Independent full adjacency-matrix tests check all small quotient partitions across three resolutions and repeated aggregation. A ring-of-cliques fixture demonstrates that later levels improve the original-graph objective beyond the local phase. Record bounded debug before/after profiles and full raw repetitions in benches/ and notes/benchmarks/louvain-levels-2026-10-21.md; old ten-phase requests were ignored, so those timings are not equivalent-work comparisons. Full Leiden refinement/aggregation and other community algorithm audits remain item 28. Workbook PDF baseline remains unchanged.

Verification: locked offline full Rust suite passed (422 library tests, 2 ignored, plus integration/doc suites). Two new private quotient tests, three multilevel integration tests and four Python multilevel cases pass. Rebuilt extension: 114 focused Python tests passed. Strict Clippy reports the same eight existing sort/match diagnostics outside this slice; targeted multilevel test Clippy passes with those two existing categories explicitly allowed. Targeted Rust formatting and diff whitespace checks pass. Debug before/after profile stores all repetitions and whole-process RSS; no large-scale or general release speedup is claimed.


## Item 28 Leiden refinement and aggregation - 2026-10-21

Replace connectivity-only refinement with bounded fast moves, constrained stochastic singleton refinement and refined aggregation carrying the broader parent partition into the next level (paper Algorithm A.2). Share count-weighted quotient storage/objective checks with Louvain; rename its private module from louvain_levels to modularity_levels. Fast moves support leaving an inherited group for an empty singleton and requeue affected neighbors. max_iter bounds visits as level size times the requested multiplier; overflow rejects. Split disconnected budget-stop remainders before publication/aggregation, preserve original ID mappings and source topology, and check nondecreasing scores and inherited quotient score equality. Refined groups merge only inside their parent, through actual connections, subject to degree-weighted attachment checks and nonnegative modularity gains.

Use one local RNG across levels. Omitted seed means reproducible seed zero; no global RNG state changes. Refinement uses stabilized exponential selection with theta=0.01 by default in normalized modularity units. Add Rust with_theta without changing the existing constructor and append Python theta after existing positional arguments. Require finite positive theta. Actual default partitions and budget interpretation can change; finite caps are not claims of global/subset optimality, full unbounded convergence, CPM/directed/strength-weight support or external backend parity. Remaining Infomap/LPA/components and option/domain audits stay item 28. Preserve the workbook PDF baseline.

Temperature range tests exposed a bridge bug: Python scalar configuration was narrowed through f32 AttrValue before reaching f64 algorithm parameters. Fix separately in a3aea1d6, preserving raw scalar floats through regular/sample pipeline construction and shared metadata type translation, while retaining explicit AttrValue compatibility and existing list/property precision. Business logic and domain validation stay in Rust. Tests cover tiny/large/near-one float values, explicit wrapped inputs and existing pipeline behavior.

Bounded debug profiles store raw repetitions and whole-process RSS in benches/ and notes/benchmarks/leiden-levels-2026-10-21.md. The before/after algorithms do different work, so timing differences are not equivalent-work speedups or large-scale guarantees. API details and primary algorithm reference are in docs/api/community-modularity.md.

Verification: locked offline full Rust suite passed (426 library tests, 2 ignored, plus integration/doc suites). Four new private Leiden checks and four integration tests pass; rebuilt extension passes 146 focused Python tests, including eight multilevel/temperature cases, four scalar precision cases and the existing pipeline regression suite. Strict Clippy remains blocked by eight existing sort/match diagnostics outside this slice; targeted Leiden test Clippy passes with only those existing categories allowed. Targeted Rust formatting and diff whitespace checks pass. No external backend was installed/executed. The bounded profiles store all repetitions; they do not establish asymptotic or release performance.


## Item 28 Infomap two-level map equation - 2026-10-21

The previous Infomap implementation was label voting, with teleportation and trial count unused. Replace that specific placeholder with in-house sparse stationary flows, the two-level codebook objective, incremental node moves and actual independently ordered trials. Retain a one-module baseline and choose the lowest complete trial score; rebuild scores each sweep and reject increased/nonfinite lengths before final legacy bulk-column writes. All-module candidates include a new singleton because jumps connect beyond stored edges. Preserve source topology, ID mapping, active-selection validation and local seeded RNG behavior.

Define the supported model explicitly: directed/undirected unit edge instances, parallel multiplicity, doubled undirected loop incidence, recorded uniform-node teleportation in [0,1], dangling rows jumping uniformly even at zero teleportation. Use a uniform initial distribution and lazy stationary iteration; zero-jump reducible graphs follow the resulting limiting mixture. Require original-operator L1 residual <=1e-12 within flow_max_iter (default10000); failure is explicit and leaves existing labels. This is a bounded numerical criterion, not a uniqueness or arbitrary-precision claim. Edgeless nonempty selections have uniform recorded-jump flow and a one-community codebook.

Preserve the Rust constructor, add with_flow_budget, and expose a thin Python infomap factory. Budgets positive, seed nonnegative, missing seed reproducible zero. Python metadata validation now permits integer-to-float parameter widening, matching AlgorithmParams::get_float while keeping bool distinct; this supports teleportation=1 without a false Python type failure. Hierarchical Infomap, alternative teleportation models, strength weights and scalable candidate search remain open. Worst-case candidate work is quadratic per sweep; record the real extra cost of replacing a label vote with map-equation work in notes/benchmarks/infomap-2026-10-21.md. Do not compare the old placeholder as equivalent objective performance.

Independent direct entropy tests enumerate all small partitions and node moves across directed/undirected modes and three teleportation settings; analytic rates cover dangling/periodic/reducible/loop/parallel cases. Native/Python integration checks actual modules, meaningful teleportation/trials, failure preservation, seed isolation and explicit input domains. External backend comparisons remain open. LPA and connected components remain the next item 28 audits. Workbook PDF baseline remains unchanged.

Verification: locked offline full Rust suite passed (429 library tests, 2 ignored, plus integration/doc suites). Three private direct-entropy/flow checks and five native Infomap integration tests pass. Rebuilt extension: 160 focused Python tests passed, including 14 new Infomap cases and existing pipeline/type bridge regressions. Strict Clippy remains at the same eight existing sort/match diagnostics; targeted Infomap test Clippy passed with those two existing lint categories explicitly allowed. Targeted Rust formatting and diff whitespace checks pass. Debug profiling records the added cost of real objective work and is not an equivalent-work speedup comparison.


## Item 28 label propagation repair — 2026-10-21

The existing CSR LPA engine remains in place. Fixes remove an implicit self-vote, mutable representative-value corruption, separate identities for equal initial labels, silent coercion of invalid budgets, and the ignored Python random seed. Preserve the Rust constructor and first three Python positional arguments; add Rust with_seed and trailing Python tolerance/seed_attr options. Directed graphs count outgoing neighbors; undirected graphs count both orientations; parallel instances vote separately and loops once. Retain a tied current label, otherwise prefer the first initial value in ascending node order. This deterministic tie policy and bounded asynchronous iteration are explicit; they do not promise global optimality or unconditional convergence.

Validation/cancellation reject before final bulk persistence. Native nonpersistent runs return grouped original IDs. The direct-value reference covers every three-node directed edge set with both unique and repeated labels; Rust and rebuilt Python tests cover failure/seed/API cases. No external implementation parity or speedup is claimed. Full strict Clippy remains blocked by eight pre-existing sort/collapsible-match errors; focused tests pass with exactly those existing lint categories exempted. Item 28 remains partial.


## Item 28 connected components repair — 2026-10-21

Keep the existing CSR/BFS and explicit-stack Tarjan kernels. Repair weak/undirected orientation on directed input and strong orientation on undirected input, including CSR cache policy. Validate live/closed selections before reading cached assignments; reject stale selections rather than silently inspect retained pool entities. Add traversal cancellation and final pre-write checks; replace the Tarjan stack panic with an error. Dense index span arithmetic is checked, and the existing u32 DFS/label representation now has an explicit selected-node bound. Export ComponentMode so the public Rust constructor is usable. Correct the Python documentation's obsolete Union-Find claim.

Independent transitive closure verifies all 512 three-node edge sets in directed/undirected graphs and all modes with repeated runs. Extra fixtures cover sparse IDs, loops/parallels, nonpersistent results, stale/cancelled/invalid selections and a 10,000-node path. Full Rust tests pass; 178 focused rebuilt Python tests pass across the current repairs. Focused Clippy passes with the two pre-existing lint categories exempted; the full strict gate remains blocked by the same eight pre-existing errors. No kernel speedup, external backend parity, unconditional cancellation latency or production-scale claim is made. Next registered-algorithm work: PageRank, complete closeness definitions/options, and BFS/DFS algorithm-wrapper checks; hierarchical Infomap and wider scale/reference validation remain open.


## Item 28 PageRank direction and personalization repair — 2026-10-21

Preserve the native CSR power iteration, Rust/Python signatures and maximum-coordinate tolerance convention. Repair ignored undirected reversal and personalized dangling mass, reject nonfinite tolerance and invalid budgets, and validate live/closed selections before empty returns/cache reads. Personalization accepts Float/Int/SmallInt with missing weight one, rejects other/negative/nonfinite values and all-zero totals. Loops count one transition incidence and parallel instances retain their mass. Final ranks are normalized in f64; existing persistent properties remain f32.

Keep compatible bounded approximation on exhausted max_iter; report pr.converged and pr.max_update rather than silently promise an exact stationary result. Add an owned NodeScores output for native nonpersistent execution and thin FFI translation to Python profile rows. This enum extension requires exhaustive Rust callers to handle the new variant; component-output tests now do so explicitly. Metadata version advances to 0.2.0 and its Python metadata expectation follows. No new durable boundary, edge-strength option or speedup is claimed.

All three-node edge sets in both graph directions with uniform/personalized resets and damping 0/0.85 match an independent direct linear solve. Rust and rebuilt Python failure/output/API checks cover the supported slice. Full strict Clippy remains blocked by the same eight pre-existing errors; targeted checks exempt only those existing categories. Near-one damping, broader backend/history/scale certification and other item 28 work remain open. Next: complete closeness definitions/options, then BFS/DFS algorithm wrappers.


## Item 28 closeness definitions and checked traversal — 2026-10-21

Fix unweighted undirected reversal and live-state/cache/empty validation. Preserve the native constructor and first two Python positional arguments; expose trailing harmonic, direction and normalized options plus typed Rust direction/normalization builders. Defaults remain outgoing raw harmonic scores. Classic raw is reachable_count / distance_sum; normalized classic adds reachable_count / (N-1), while normalized harmonic divides reciprocal sum by N-1. Disconnected isolates score zero. Strictly positive finite lengths replace silent skipping/undefined zero-length scores; this does not change Dijkstra's zero-length domain.

Use reusable BFS/heap buffers with traversal cancellation, checked sparse/dense index spans, a private incoming transpose, ordered original-ID f64 output and all-score f32 preflight before existing bulk persistence. Nonpersistent rows use NodeScores; FFI remains translation only. The existing heap entry type is visible within the crate to share ordering semantics. All supported variants match an independent exhaustive small-graph Floyd-Warshall reference; Rust/Python checks cover failures, compatibility and cache isolation. No performance/backend/production-scale claim is added. Item 28 remains partial; BFS/DFS wrappers follow.


## Item 28 BFS/DFS wrappers and real cache coverage — 2026-10-21

Replace arbitrary first-marker selection with an explicit exactly-one presence rule matching the existing single-source implementation; false/zero/null mark a source and the stored value is ignored. Preserve constructors/factory signatures, reject inactive/open selections before cache access, propagate attribute errors, and honor advertised cancellation during queue/frame traversal and before bulk results. Empty inputs reject for lack of a source; marked isolates return zero. BFS writes integer hop distances; DFS writes zero-based preorder with ascending node-ID neighbor order, sorting a private CSR copy so earlier caller cache order cannot change results. Explicit DFS frames avoid recursion and duplicate pending-node stacks.

Nonpersistent native NodeIntegers outputs expose exact ordered pairs to Python profiles through translation-only FFI. This additional enum variant requires exhaustive Rust matches to cover integer rows. Only reached nodes are written persistently; old properties on unreached nodes remain under existing bulk setter semantics. Full snapshot/durability boundaries are unchanged.

Independent small references cover every three-node edge set in both directions/every source with loops/parallels and reversed insertion order. Carry the returned selection into the next run to guarantee the cache-hit tests really reuse populated topology caches; also correct the preceding closeness fixture to do this. Additional fixtures cover invalid markers/selections, cancellation, sparse/singleton/20,000-node paths and rebuilt Python pipelines. No speedup/backend/scale certification is claimed. Dijkstra still needs the corresponding source/cancellation/nonpersistent wrapper follow-up; other step/domain/history/reference work remains under item 28.


## Item 28 Dijkstra wrapper completion — 2026-10-21

Complete the follow-up exposed by the BFS/DFS audit: require exactly one selected start-property presence, propagate attribute-read errors, validate the full selected graph and weights before cache reads, and check cancellation during native heap traversal and before final output. Preserve nonnegative lengths including zero, compact/missing values, reverse undirected traversal, parallel minimum distances and strict f64 improvements. Source marker values are ignored, including false/zero/null; the old arbitrary first marker and misleading truthy/multiple-source wording are removed. Constructors and factory signatures stay stable; metadata advances to 0.2.0.

Return owned ordered original-ID f64 distances through the existing NodeScores output in nonpersistent mode. Only persistent writes require f32 representation, all checked before one existing bulk setter. Previous properties on unreachable nodes remain under the existing setter behavior. Native exhaustive all-source multigraph fixtures use an independent Floyd-Warshall reference, including repeated populated CSR caches; failure/range/API checks run in Rust and rebuilt Python. No speedup, new durable boundary, low-level utility certification or backend/scale parity is claimed. Remaining algorithm/step/history audits stay open.


## Item 28 A* output and trivial-path cancellation — 2026-10-21

The Dijkstra completion exposed a neighboring wrapper gap: A* nonpersistent mode discarded its path, and a cancelled start-equals-goal path could bypass traversal cancellation. Return ordered original-ID/i64 path-index rows through existing NodeIntegers and expose f64 path cost in execution diagnostics; retain persistent integer-index semantics and original signatures. Check cancellation before work and before final output, and validate selected state/lengths before cache reads. Retain consistent-heuristic validation and noncanonical tie-path behavior. Guard finite accumulated costs/priorities and checked i64 indices. Metadata advances to 0.2.0.

Independent exhaustive small-graph Floyd-Warshall costs cover all endpoint pairs, both graph directions and zero-weight parallel edges using populated shared caches; fixtures also check cancellation before trivial paths, stale/open selections, sparse IDs, nonpersistent ownership and large f64 cost without requiring a float property. Rebuilt Python exercises returned paths/diagnostics and preserved old properties. Existing per-expansion node-to-CSR scans and wider algorithm/step/history/reference/scale work remain; no speedup or new transaction boundary is claimed.


## Item 28 PCA projection caller repair — 2026-10-21

The visualization PCA caller had a singleton N-1 denominator, naive column statistics, an arbitrary deviation floor and unscaled covariance products. Add a public native owned project_matrix result, used by the existing position trait and a thin Python pca_projection method. Keep center/standardize constructor semantics, use population deviation (matching the old column variance denominator), treat constant-feature scale as one, and define singleton behavior. Column normalization bounds preprocessing statistics; common scaling bounds the mean scatter before the existing native symmetric solve. Restore score scale and reject nonfinite preprocessing/output. Retain row identity and expose copied optional row_labels in Python.

Use feature-space scatter rather than allocating full observation-space SVD outputs; this retains the squared-conditioning limitation explicitly. Analytic rank-two/scaled/constant/singleton/metadata cases and NumPy SVD score-geometry references cover the slice. No fitted-model, whitening, sparse acceleration, safe GIL-release or production-scale claim is added. The audit also found fabricated topology in flat_embedding and simplified t-SNE/UMAP-inspired callers; record those under the still-partial broader analytics slice rather than close item 28.15 after PCA/spectral alone.


## Item 28 spectral Laplacian and null-space repair — 2026-10-21

The existing spectral path overwrote parallel adjacency, used threshold-dependent degree normalization, gave isolates normalized diagonal one, and silently symmetrized directed graphs. Keep the symmetric in-house eigensolver, but explicitly support undirected unit edge instances, accumulate parallels/loops and use adjacency row degree (loops once). Construct D-A first and normalize on both sides with zero inverse degree for isolates. Validate endpoints, finite positive thresholds, dimensions and checked dense size before unsafe accesses.

Skip all component null modes plus eigenvalues within the selected cutoff; establish minimum nullity from the dense Laplacian support instead of blindly forcing one skip or depending on a tiny numerical cutoff. Keep the existing Rust builder/constructor names, add a thin Python spectral_embedding with explicit skip_nullspace, sorted owned row identity and matrix/PCA interoperation. Explicit include mode supports a singleton zero mode; the default correctly has no nonconstant singleton/edgeless modes. Two existing sparse-34-node fixtures requested five modes although only five nodes are connected, so update them to require four and reject the fifth.

Native exact operator values, exhaustive three-node loop/parallel/disconnected support cases, and NumPy eigenpair/owned-identity checks cover the slice. Full graph/table snapshot migration, directed variants, sparse acceleration, safe GIL release, cancellation and production-scale behavior are not added. Broader analytics remains partial: fabricated flat topology, composite interleave TODO panic and simplified projection/energy callers require subsequent work, already recorded in the flat checklist.


## Item 28 flat-layout topology and effective solver — 2026-10-21

Fixing edge extraction alone would have left topology ineffective: sliced autodiff tensors were detached, failed gradients returned initialization as success, repulsion was constant, spread omitted and clipping a no-op. Keep the public configuration and position entry point while implementing a local explicit native objective/gradient kernel for mean edge cohesion, regularized all-pair repulsion and negative population covariance determinant. Use local seeded initialization, disk projection and at most 32 descent backtracking trials; errors remain errors. This does not certify the wider autodiff library.

Use all real active edge IDs, checked endpoints and validated row identities, retaining parallel/loop instances without truncation. A shared embedding row-ID helper supports canonical unlabeled rows and explicit labeled permutations. The graph-free wrapper no longer invents a graph: multirow cohesion requires explicit graph input or zero cohesion. Expose an owned graph-aware matrix helper and translation-only Python method. Feature values are not in this topology-based loss; no high-dimensional distance-preservation claim is made.

Independent derivative/topology/identity/bounds/failure tests replace previous placeholder shape/no-panic tests. Native/Python checks cover edge influence after the former cutoff and disappearance on removal. All-pair work and bounded line-search costs are documented; no speedup, global convergence, GIL-release, cancellation or production-scale promise is added. Realtime caller fallback policy and broader analytics remain open.


## Item 28 composite interleaving and checked aggregation — 2026-10-21

Replace the public interleave TODO panic with round-robin complete column chunks, retaining final/ragged tails and every child column. Preserve existing concatenation per-child dimension semantics; weighted averages require equal actual widths. Move composition into one cohesive native module shared by the existing factory and translation-only Python combine_embeddings. Resolve row identity using the flat-layout helper, align children by active IDs, and label owned output canonically instead of silently averaging misaligned rows.

Preflight empty children, zero requested dimensions/chunks, malformed identities/shapes, finite values and finite nonnegative weights; avoid the neighboring weighted-average empty-index/zero-total failures. Normalize weights by their maximum and scale/compensate per-cell accumulation to keep convex averages within finite input range even at f64 extremes. Preserve feature provenance for concatenation/interleaving and positional-column semantics. Handle missing source feature names without indexing panics.

Independent column-round ordering, factory, identity, invalid-input and extreme-value fixtures cover Rust and rebuilt Python. Child algorithm/seed propagation, general transforms, gradients, safe GIL release, cancellation and production-scale behavior remain outside this slice. The broader analytics audit remains partial after both requested defect fixes.


## Item 28 N-dimensional energy/force-directed repair — 2026-10-21

Replace the constant 1.0 energy and 0.01 gradients, topology-free magnitude terms and ten-edge cutoff with a native analytic spring-electric objective: sum 0.5*a*(sqrt(distance_squared+1e-6)-ideal)^2 over all active nonloop edge instances, plus b/sqrt(distance_squared+1e-6) over unordered node pairs. Preserve each parallel/reverse spring; directed orientation has symmetric geometric influence. General autodiff remains a separate audit rather than a dependency of this solver.

Validate finite positive learning rate, finite nonnegative strengths/ideal distance, positive representable dimension, finite coordinates and complete unique active row identities. Default initialization uses sorted active IDs; native optimize_positions supports owned warm starts with labeled row permutations. Outputs retain row identity without graph/history changes. Both energy and force-directed factory engines honor configured seeds. Force-directed no longer advertises an incremental interface; energy has an explicit warm-start method.

Use bounded backtracking (64 halvings) with optional linear rate annealing and actual initial/final energy history, accepting only finite nonincreasing energy. Return explicit errors on overflow or failed line search. Zero gradients/unrepresentable updates stop; a fixed iteration bound is an approximation, not certified convergence. Smoothed coincident points can remain stationary, and disconnected/repulsion-only layouts need not have bounded optima. Custom/stress modes reject even at zero iterations. No Python API, acceleration, GIL, cancellation or scale guarantee is added.

Independent finite differences in multiple dimensions, translation/balanced-force checks, real energy descent, cutoff/topology/parallel/identity/failure fixtures and factory RNG reproducibility cover the repair. Broader analytics 28.15 and certification 28.16 stay open. See docs/api/energy-embedding.md and tests/energy_embedding.rs.


## Item 28 dense t-SNE repair — 2026-10-21

Replace fixed-variance conditionals, unnormalized Student-t values and learning-rate exaggeration with entropy-calibrated per-row Gaussian affinities, symmetric joint P of unit mass, globally normalized Q and analytic exact dense objective gradients. A bounded binary search targets log(perplexity); explicit uniform/nearest-tie limits handle boundary cases. Reject unattainable perplexity rather than fabricate distances or silently ignore it. One older smoke fixture has four tied nearest neighbors in a row, so perplexity three now explicitly rejects and four succeeds.

Use phase-specific finite nonincreasing backtracking, with P exaggerated during the first 250 requested updates. Preserve the later phase even if the earlier phase stops. Validate controls/input/shape/finite values and row metadata, use common feature scaling before distances, preserve owned row labels, and propagate factory seed to a local RNG. Keep existing constructor/trait/import path; add native project_matrix/with_seed. No Python API or approximation/convergence/scale/cancellation/GIL claim is added. UMAP-inspired behavior and other analytics remain open.

Known analytic entropy probabilities, an independent normalized Student-t objective, finite differences, actual KL reduction/phase transition, and native public factory/trait/identity/RNG/range/failure tests provide scoped evidence. See docs/api/tsne-projection.md.


## Item 28 native UMAP-inspired scope and repair — 2026-10-21

Keep the existing public UMAPProjection constructor/name as a native inspired neighbor layout, and explicitly distinguish it from full UMAP (fuzzy simplicial union, smooth-kNN membership calibration, fitted attraction curves and cross-entropy). This repair does not certify full UMAP compatibility or replace the method with a dependency. Move the solver into its own native module, retaining the existing import path.

Replace unchecked NaN distance sorting, ignored distance weights, one-endpoint asynchronous forces, unchecked negative-count casts and unseeded factory calls. Shared finite/common-scaled features feed deterministic exact kNN with row-index tie breaks. Compute one distance row at a time to retain rowwise neighbor-search storage rather than introducing a quadratic distance matrix. Directed neighbor similarities use exp(-(distance-nearest)/(farthest-nearest)), or one for a zero range. All selected neighbor springs contribute a weighted squared hinge beyond min_dist; sampled unordered geometric effects use 0.001/sqrt(distance_squared+1e-6) with ordered nonself draws and retained multiplicity. Each gradient updates both endpoints. Hold sampled pairs fixed during bounded finite nonincreasing backtracking; losses need not decrease across epochs because samples change.

Validate positive neighbors, finite nonnegative min_dist/sample rate, input/shape/metadata, and representable/allocatable floor(N*rate) sample counts. Preserve positive-neighbor clamping, support singleton origin, add owned native project_matrix and local with_seed, and propagate factory seed. Fractional rates explicitly floor their per-epoch count. No graph/history mutations, Python API, GIL/cancellation/scale or quality guarantee is added. Independent derivative/force-balance/known-neighbor first-step tests plus public metadata/RNG/domain/range fixtures cover this scope. See docs/api/umap-inspired-projection.md.


## Item 28 realtime flat-layout failure propagation — 2026-10-21

Remove the silent original-embedding fallback on flat solver and matrix conversion failures. Reuse compute_flat_embedding_matrix so topology/row validation, solver/construction errors and owned row labels are preserved, rather than rebuilding unlabeled coordinates. Preprocessing is skipped only when flat_embed is absent or explicitly false/zero. Parse supplied flat flags/controls strictly with named errors, retain defaults only for missing values, and expose the native repulsion_power/radius_clip/seed controls alongside the existing fields. Disabled preprocessing ignores its unused numeric controls.

Unknown LayoutChanged algorithms now return a public engine error rather than logging a warning and broadcasting the unapplied command as successful. Tests cover every flat parameter, native default equivalence, sparse/permuted identity/ownership, malformed/domain/row errors and async failed-projection position/timestamp/broadcast preservation. The server also swallowed layout errors while the bridge acknowledged success before computation. Both server run paths now acknowledge actual engine completion/failure to the requester, and ChangeLayout no longer receives a premature bridge success; the browser logs rejection as an error. Native server/bridge tests cover queued versus completed controls, unavailable engines and valid/invalid/flat-failing requests. Layout-change controller/configuration bookkeeping still precedes recomputation; full command rollback and visual error presentation need separate review. No generalized transaction, rendering or coordinate-ownership claim is made. See docs/api/realtime-layout-errors.md.


## Item 28 prepare layout changes before publication — 2026-10-21

Address layout-request failure preservation by staging the result rather than mutating settings and trying to reconstruct them afterward. Share a read-only preparation path with ordinary layout computation. Resolve accepted parameter caches, compute any missing embedding, project coordinates, derive controls, check finite coordinates before/after centering, and validate publication time before state changes. A failing cached/uncached request leaves prior controller/view/policies, configuration, parameters/cache, embedding, animation, timing/positions, pending deltas and queues untouched, and emits no successful engine update.

After success, publish accepted state and effective honeycomb cell size, cache the embedding, reset interpolation and activate the layout controller. Use a prevalidated timestamp to emit one coherent envelope with matching settings/coordinates and optional view; preserve older pending parameter deltas. Empty parameter maps retain accepted-cache restoration semantics. A layout switch now clears only obsolete coalesced position deltas; the former flush-and-discard also lost coalesced properties, which now remain queued alongside structural updates.

Snapshot-style native failure tests cover fresh/cached embeddings, preprocessing errors, later projection failure after cell-size derivation, embedding failure and nonfinite coordinates. Successful retry covers coherent publication, controller activation, cache restoration and property/structural retention. Scope is recoverable layout-request errors, not atomicity of all controls, concurrent-reader isolation, durable transactions, or network acknowledgement rollback. Embedding/projection/quality updates still need their own staging review. See docs/api/realtime-layout-errors.md.


## Item 28 embedding/projection control publication — 2026-10-21

Extend candidate preparation to typed embedding and projection configuration updates and string embedding requests. Compute the required fresh/missing embedding and current-layout coordinates/derived settings against candidate configuration; validate publication time before publishing any state. Errors preserve prior configuration, cached embedding/layout parameters, animation, controller/view, coordinates/timing, pending settings and queues. Successful updates publish a coherent configuration/layout/position envelope and retain nonmotion updates. Fresh embeddings reactivate dimension-dependent controller state after publication; projection changes reuse the embedding when available.

String embedding requests now execute immediately as controls, reject unknown method names instead of falling back to spectral, and receive actual success/failure acknowledgements in both WebSocket server paths. Reject nonempty unsupported string parameter maps instead of silently ignoring them. Preserve legacy names and native energy_nd/force_directed_nd aliases. New embedding computation on an empty graph rejects rather than treating stale display coordinates as the requested embedding.

Quality-only recomputation is still a placeholder; UpdateQualitySettings now returns an explicit NotImplemented error without changing state, instead of storing settings and claiming a no-op succeeded. Quality evaluator/optimization certification remains open. Native state-signature failure tests, coherent successful typed/string changes and server/bridge acknowledgement tests cover the bounded repair. Other control families and visualization correctness remain open; proceed next to numerical arithmetic/reductions (29.3).


## Item 29 logical-column reductions — 2026-10-21

Begin the arithmetic/reductions audit with the five BaseArray logical-column statistics and share native sum/mean kernels with the older BaseArray path. Preserve scalar i64/compact-i32/f32 input, null skipping, integer-only i64 sums, float-containing f32 sums, f64 statistics and sample N-1 variance. Reject nonnumeric/nonfinite input, empty/all-null data, insufficient variance observations and unrepresentable sum outputs explicitly; no implicit property widening or NumPy API redesign is introduced.

Use checked i128 integer subtotal accumulation so intermediate i64 overflow may later cancel, plus compensated f64 float accumulation with integer conversion residuals. Center exact integer differences before f64 conversion for two-pass sample variance. Sort copied median values by coherent mixed numeric order, treating floating signed zeros as equivalent to integer zero; add even integer pairs in i128 before conversion and retain cancellation in mixed pairs. Input data and graph/history are unchanged.

Independent native permutations, exhaustive exact integer-residual variance fixtures, numeric comparator laws, source/domain/range cases and Python Fraction references cover the slice. Bundled Python 3.9's direct large-integer variance also collapses small differences; use explicit rational references rather than treating that result as authoritative. No arbitrary-precision/correct-rounding/throughput guarantee is added. NumArray, ArrayArray, SIMD, min/max, matrix reductions and arithmetic operators remain for subsequent slices of 29.3/29.4. See docs/api/array-reductions.md.


## Item 29 exact logical-column extrema — 2026-10-21

Extend the shared numeric order to BaseArray min/max: choose original values and preserve first ties without rounding i64 candidates to f64/f32. Reject nonfinite values even in singleton/late positions and retain null skipping and explicit empty errors. Preserve legacy text/compact-text/boolean extrema and reject incompatible mixtures. Native permutation/limit/type/domain fixtures and rebuilt Python comparisons cover the scope; NumArray, other reductions and arithmetic remain separate work. See docs/api/array-reductions.md.


## Item 29 shared typed NumArray reductions — 2026-10-21

Remove statistical business logic and per-call f64 column copies from the NumArray Python bridge. Dispatch int32/int64/float32/float64/bool slices to shared Rust sum/mean/extrema/sample variance/std/median kernels; expose Python median. Preserve f64/Option outputs and existing native generic bounds. Integer hints retain exact subtotal/difference/median information before final f64 conversion; output remains f64 rather than an exact i64 return.

Use compensated/rescaled floating totals/means and centered normalized dispersion; compute std before rescaling so a valid std survives variance overflow/underflow. Median uses typed order and safe midpoint; in-place selection uses standard-library partitioning instead of recursively partitioning repeated data. Define empty sum zero, empty other statistics None, finite singleton dispersion zero, NaN propagation, IEEE infinite sum/mean/median behavior, and nonfinite dispersion NaN. Python empty var/std now agree with native None, replacing the inconsistent Python zero. These NumArray IEEE semantics remain distinct from checked logical BaseArray errors.

Historical SIMD/auto entry points now share the reference kernels; no current vectorized speed guarantee is made. The explicit-mean low-level simd_variance retains its caller-centered moment contract. Reintroduce acceleration only with semantic verification and measurements under 29.6. Native typed/extreme/dispatch/nonfinite/repeated-selection tests and rebuilt Python dtype/rational/IEEE checks cover the slice. Percentiles/correlation, ArrayArray, conversions, arithmetic, matrix operations and throughput remain open. See docs/api/numarray-reductions.md.

### Item 29 linear percentiles — 2026-10-21
Retain linear rank p*(N-1), Option output and current generic bounds. Reject invalid probabilities with None; propagate observation NaNs, including endpoint requests. Safe midpoint/interpolation retains finite representable extremes; f64 output remains approximate. No new Python API is required for this native repair.

### Item 29 Pearson correlation — 2026-10-21
Retain Option and Pearson paired semantics. Mismatched/short/finite constant input returns None; nonfinite matching inputs of length at least two produce NaN. No implicit pair dropping. Preserve integer differences before final f64 moments and independently scale both columns. Accept explicit O(N) scratch pending measured optimization; no new FFI API or external backend is introduced.

### Item 29 checked matrix arithmetic — 2026-10-21
Use GraphResult methods and Python errors for mismatched shapes. Preserve positional identity on owned arithmetic results; no label join/reordering or inherited gradient tape. Scalar division rejects signed zero exactly, computes direct quotients and retains other IEEE behavior. Keep the source-compatible Rust trait operators visible as remaining debt pending result-type migration.

### Item 29 ArrayArray numeric reductions — 2026-10-21
Retain zero for empty inner arrays and the existing permissive numeric projection (numbers/bools included, null/text/collections skipped). Apply shared stable f64 kernels and NaN propagation. Move conversion/aggregation out of FFI, preserve named keys and reject output-name collisions. AttrValue scalar floats are f32; preserve that storage format, reject finite narrowing overflow and document remaining rounding/underflow. Exact grouped i64 moments and lossless f64 storage need separate work.

### Item 29 Rust matrix operator results — 2026-10-21
Choose an explicit source-level Rust operator migration: +, - and * return GraphResult for borrowed and owned matrices, sharing checked named methods rather than returning substitute zeros or panicking. Propagate errors and check output sizes. Product metadata comes from left rows and right columns; operands are positional. Downstream callers use ? or match; Python syntax stays unchanged. NumericType overflow and stable dot-product accumulation remain separate work.

### Item 29 floating dot products — 2026-10-21
Use an in-house fixed-bin reference accumulator to avoid intermediate floating overflow, underflow and cancellation loss in GraphMatrix products. Preserve IEEE input behavior and round finite f64 dots once. f32 still narrows through f64. Record the measured cost before optimizing; do not claim the lower-level numerical backends are covered.

### Item 29 direct solve and inverse — 2026-10-21
Use direct row-equilibrated partial-pivot elimination and the same kernel for inverse(identity). Reject exact zero pivots, nonfinite inputs, lost normalization terms and unrepresentable ranges; remove fixed absolute singularity thresholds and guessed values. Restrict to floating matrices explicitly. Preserve unknown/RHS coordinate identity; label matching is positional. Conditioning and iterative refinement are separate requirements, not implied by a finite solution.

### Item 29 integer matrix arithmetic — 2026-10-21
Reject integer overflow rather than debug panic/release wrap. Keep output dtype stable; use checked i128 products/subtotals and one final narrowing for built-in integer matmul. This allows output-range cancellation but rejects accumulator overflow explicitly. Retain bool OR/AND semantics. Custom NumericType hooks have compatible defaults and own their arithmetic contract.

### Item 29 typed grouped integer statistics — 2026-10-21
Preserve original i64 scalars for statistical reductions rather than materialize f64 first. Share single/multi aggregation and generic result-table validation. Keep numeric inclusion/empty/count rules and f32 stored outputs explicit; widening storage and exact mixed arithmetic are separate work.

### Item 29 matrix selections and transforms — 2026-10-21
Keep selections owned with positional identity, including duplicates and empty axes. Validate indices before reading or allocating. Resolve raw Python slices in the Rust core at actual dimensions. Reshape follows logical row order and resets labels only when the coordinate shape changes; transpose exchanges axis identity. Lower-level views and graph bindings are separate contracts.

### Item 29 canonical CSR — 2026-10-21
Use sorted unique in-range columns and no stored zeros as the explicit CSR contract. Reject malformed/duplicate construction; checked set inserts/removes canonically. Clones own independent sparse vectors. Existing checked GraphMatrix consumers may densify; expose this behavior and its cost rather than claim sparse optimization. Distinguish storage kind from numeric zero density.

### Item 29 measured floating dot acceleration — 2026-10-21
Optimize the exact reference kernel only with a behavior-preserving checked path. Integral-valued factors may accumulate in i128; flush exact subtotal bits to the bin accumulator when needed. Keep nonfinite checks ahead of zero shortcuts. Measure integral and fractional controls separately and retain the remaining fractional/reference cost explicitly.

### Item 29 shared matrix reductions — 2026-10-21
Use checked GraphResult global reductions and shared native typed kernels across axes/columns/trace. Keep sample axis and population column variance conventions explicit, along with typed-statistic conversion rather than introduce promotion. Preserve non-reduced axis length for empty inputs. Collect owned logical values/slices for consistent kernels and record this scratch/performance cost; lower-level/neural reductions are separate.

### Item 29 economy QR — 2026-10-21
Use real economy Householder QR with k=min(m,n), avoiding impossible wide orthonormal output and unnecessary full tall Q. Retain zero/rank-deficient orthonormal completion without epsilon rank decisions. Reject input/range/dtype failures explicitly; preserve source and basis identity in owned factors. Complex/pivoted QR, gradients and least squares are separate work.

### Item 29 scaled real Cholesky — 2026-10-21
Use a norm-relative symmetry policy after scaling, average accepted pairs, and require strictly positive residual pivots at working precision. Reject unsupported/range errors instead of guessing factors. Support native f32/f64 and empty order, retaining owned source/basis identity. Conditioning, pivoted/complex/sparse factors and gradients remain separate.

### Item 29 LU and determinant — 2026-10-21
Use bounded real partial-pivot PA=LU without absolute singularity epsilons. A zero pivot can remain in a valid singular factorization; determinant/sign-log expose singularity. Preserve owned coordinate/permutation identity and checked dtype range. Empty determinant is one, order-two subtraction is accurate, and larger determinants exponentiate pivot log sums. Expose slogdet to retain final range; do not claim exact integer or unbounded intermediate range.

### Item 29 numerical rank and condition estimates — 2026-10-21
Define rank relative to the largest singular value and input dtype precision, with an explicit cutoff override. Condition number reports the untruncated normalized spectral ratio, independently of rank cutoff. Use normalized finite floating SVD inputs to avoid absolute-scale/output-range artifacts. Keep these as numerical estimates and preserve solve policies; residual/refinement/least squares remain separate.

Verification: full locked offline native suite passed (442 library tests, two ignored, plus integration/doc suites); rebuilt Python numerical regressions: 189 passed. Focused Clippy passed with the existing eight lint findings excluded; strict all-target Clippy still reports those same eight existing findings.

### Item 29 complex eigen result — accepted design decision, 2026-10-21
The caller selected a dedicated complex eigendecomposition result exposing real/imaginary eigenvalues and eigenvector matrices. Preserve the existing real symmetric API and real numeric matrix storage. The general complex solver, convergence/defective-matrix policy, and its independent verification remain implementation work under 29.24.

### Item 29.31 — Solve diagnostics and refinement, 2026-10-21
Add owned native SolveReport and thin Python tuple without changing solve/inverse. Report a normwise backward-error estimate against original coefficients using exact-dot rounded residuals and log norms; it is not a forward-error/certified guarantee. Bound correction attempts to 16 and accept only strict improvement after output-dtype rounding, preserving coordinate identity and explicit range failures. Repeated dense refactorization is deliberate initial scope; factor reuse, sparse refinement and performance remain open. See docs/api/matrix-solve-refined.md.

29.31 verification: full locked offline native suites passed; rebuilt Python numerical suite: 194 passed, including independent Fraction residual checks and an overflowing denominator fixture. Focused Clippy passes with only the existing sort/match categories excluded.

### Item 29.32 — Remove hidden SVD truncation, 2026-10-21
Follow-up inspection found a spectral floor in the underlying SVD that overrode the explicit numerical-rank cutoff and could report infinite condition for a diagonal matrix with a resolved tiny singular value. Rotate nonzero columns and retain computed nonzero norms instead. Preserve bounded convergence/basis errors rather than invent missing directions. Add diagonal tests through 1e-300 alongside existing decomposition references. Finite-precision rank uncertainty remains explicit.

29.32 verification: full locked offline native suites passed; rebuilt Python numerical suite: 197 passed. Focused Clippy passed with the same existing sort/match categories excluded.

### Item 29.33 — Real minimum-norm least squares, 2026-10-21
Use the bounded in-house real SVD to solve rectangular/rank-deficient multiple-RHS systems, retaining components above an explicit relative cutoff (default shared with numerical rank). Normalize finite floating coefficients/RHS, use accurate dot projections/reconstruction and preserve owned unknown/RHS identity. Reject shape/dtype/cutoff/convergence/working-range failures. Empty/zero systems return the minimum-norm zero result. Dense full factors and repeated allocation remain explicit costs; sparse/thin optimization, gradients and release throughput remain open. This finishes the bounded real factorization/diagnostic list in 29.23, while 29.24–29.26 remain open. See docs/api/matrix-least-squares.md.

29.33 verification: full locked offline native suites passed; rebuilt Python numerical suite: 215 passed. Focused Clippy passed with only the existing sort/match categories excluded.

### Item 29.34 — Dedicated complex eigen result, 2026-10-21
Implement the caller-selected separate result with f64 real/imaginary eigenvalues and owned real/imaginary eigenvector matrices, retaining the real symmetric API. General finite real-input eigenpairs use in-house bounded shifted complex Householder QR/full similarities, local deflation, independently scaled trailing shifts, triangular back substitution and residual/basis checks. Reject defective/unresolved bases, convergence and range failures explicitly. This dense reference does not claim Hessenberg/implicit-QR performance, balancing, complex input, certified forward bounds or gradients. Decomposition differentiation and sparse/performance work remain open under 29.24–29.26. See docs/api/matrix-complex-eigen.md.

29.34 verification: full locked offline native suites passed (443 library tests passed, two ignored, plus integration/doc suites). Rebuilt Python numerical suite: 242 passed, including 27 complex-eigen cases. Focused Clippy passed with existing sort/match categories excluded. Strict all-target Clippy still reports exactly the same eight existing sort/match findings; it is not clean.

### Item 29.35 — Explicit Cholesky derivatives, 2026-10-21
Add native/Python cholesky_jvp and cholesky_vjp using owned outputs and shared scaled triangular/dot kernels. Define symmetric tangents and symmetric Frobenius cotangents explicitly: an off-diagonal pair contributes twice the returned entry. Upper-factor cotangents are irrelevant but must be finite. Preserve forward SPD/domain validation, dtype/coordinate identity and explicit working-range errors. Independent central differences and adjoint checks cover the math; tape integration, other decomposition derivatives and cached/blocked/sparse/performance work remain open. See docs/api/matrix-cholesky-derivatives.md.

29.35 verification: full locked offline native suites passed; rebuilt Python numerical regression suite: 253 passed, including independent Cholesky central differences and adjoint identities. Focused Clippy passed with the existing sort/match categories excluded.

### Item 29.36 — Cholesky automatic tape integration, 2026-10-21
Track checked forward Cholesky factors and add native Operation::Cholesky forward/backward delegation to the same factor and symmetric Frobenius pullback. The current backward seeds ones and clears prior gradients; explicit VJP remains available for arbitrary cotangents. Validate the bound input snapshot against values used for the tracked forward and reject stale values until explicit disable/rebind. Later source mutation leaves the completed forward snapshot stable. Native/Python chained, repeated, empty/f32, finite-difference and stale/post-forward mutation tests cover this operation. Broader tape ownership, mutation, branch merging, general grad metadata/locking and other decomposition gradients remain separate. See docs/api/matrix-cholesky-derivatives.md.

29.36 verification: full locked offline native suites passed; rebuilt Python numerical regression suite: 258 passed, including tracked Cholesky finite-difference, stale snapshot, chain and repeated-backward checks. Focused Clippy passed with existing sort/match categories excluded. Strict all-target Clippy still reports the same eight existing sort/match findings and is not clean.

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

### Module consolidation and finish priorities — 2026-10-22

The owner requested adding consolidation to the flat checklist and prioritizing
completion while the workbook is pending. AR7.1 records the completed inventory;
AR7.2–AR7.11 track module/caller consolidation and compatibility verification.
The proposed execution order starts with matrix/array foundations and numerical
closeout, then entity/view ownership, state/history/time, durable lifecycle and
remaining core surfaces, with Graphiti integration after core completion. The
finish plan maps remaining IDs and specific appendix dependencies. This planning
update supplies no answers to blank workbook questions and marks no module moves
or original roadmap parents complete. See `roadmap-finish-priorities-2026-10-22.md`.

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

### Item 29.61a — Checked attribute pair statistics and quantiles

Canonical BaseArray delegates to shared numeric attribute kernels. Preserve exact
integer ordering/centering, mixed integer/f32 residuals, pairwise missing cells,
average Spearman ties and legacy Kendall tau-a. Reject mismatched lengths, negative
ddof, invalid empty quantile arguments, nonfinite input and f32 output overflow.
Native/Python rational and boundary references are recorded in
`attribute-statistics-contract-2026-10-22.md`. Remaining window/cumulative/table
paths keep 29.61 and its numerical parent open.

### Item 29.61b — Attribute windows, cumulative statistics and shifts

Shared checked native windows include compact integers and preserve typed
centering/mixed cancellation, population standard deviation and existing missing
cell/output conventions. Expanding work is linear; rolling retains its explicit
window-scan bound. Shifts beyond length and i32::MIN return fill safely; fractional
changes retain integer differences. Invalid empty operations, nonfinite inputs
and narrowed overflow fail explicitly. See `attribute-window-contract-2026-10-22.md`.
Table numeric paths remain open in 29.61.

### Item 29.61c — Table numerical adapters and NC1 closure

Route table sum/mean/extrema through checked canonical columns, use exact mixed
numeric ordering for single/multi-column sort/top-N and match pivot keys by the
same typed identity used for unique keys. Reject ambiguous output column names.
Describe computes one checked population summary per column. Canonical sum dtypes
replace the old rounded-integer guess; overflow/nonfinite cases fail explicitly.
Full native suites, strict core Clippy and 560 rebuilt Python regressions pass
(one existing Python case skipped). See `table-numeric-contract-2026-10-22.md`.
NC1/29.61 closes; NC2 remains open. This does not settle table writeback policy.

### Item 29.62 — Checked activations, dropout and shared Python routes

Remove ignored conversions/cell errors and duplicate FFI activation loops. Preserve
metadata, use faithful GELU derivatives and actual logical-row softmax, compose
ReLU/GELU/sigmoid/tanh/softmax tapes, and reject unsupported tracked leaky/ELU/
training-dropout calls. Inverted dropout evaluates as identity, validates both
modes/endpoints and supports storage-equivalent seeded masks. Shared sparse
expansion guards and owned-data/GIL boundaries apply. Full native suites and
570 rebuilt Python regressions pass (one existing skip); strict core lint and
55-case release receipts are recorded in `matrix-activation-contract-2026-10-22.md`
and the numerical activation benchmark note. This closes the NC2 helper inventory,
not universal gradients or the deferred convolution API.

### AR7.4 — Canonical entity handle ownership with compatibility adapters

Node/Edge own handle state under entities; legacy EntityNode/EntityEdge wrap them
under explicit compatibility modules. Storage imports are facades and actual
callers use entities. Checked current constructors and unchecked historical
signatures/trait overrides remain intact; property writes/history/liveness are
preserved. Native ownership/mutation/restoration/inactive-reference fixtures and
rebuilt Python live/history/topology gates are recorded in
`entity-handle-consolidation-2026-10-22.md`. This does not freeze graph views.

### AR7.5a — Shared immutable read payload and current capture

Graph current captures and TemporalSnapshot share Arc-owned read state with exact
current values, membership and indexed adjacency. Capture includes dirty changes
without a checkpoint/delta and rejects dangling referents. Clone/read-worker
transfer retains frozen values. Do not use checkpoint ID/topology version as a
property-sensitive read cache key. Ordinary graph/table view read/write ownership
remains open; see `pinned-read-state-foundation-2026-10-22.md`.

### AR7.5b — Immutable scoped selections and Python read API

GraphView shares frozen state and membership across copies/children; parent edge
scope is preserved for non-induced and parallel facts. Edge-aligned adjacency/
incidence gives degree-proportional scoped reads with explicit directed/loop
counts. Graph.read_view/Subgraph.pin_view are native-backed Python entry points;
null presence, out-of-scope errors, owner lifetime and read-worker behavior are
covered by native tests and 176 rebuilt Python regressions (one existing skip).
Strict core Clippy passes; both documented examples execute. Record indexed
capture cost versus cheap pin/view sharing in the immutable graph-view benchmark
note. Ordinary Subgraph/table/algorithm read-owner migration remains open; see
`immutable-selection-contract-2026-10-22.md`.

### AR7.5c — Frozen matrix exports and shared graph projection

GraphView.to_matrix/adjacency_matrix use the selected captured state, including
weights, directed/parallel/loop semantics, sorted node identity and isolates.
Live Graph exports share the native kernels. Matrix results are owned; Python
releases the GIL for projection and transfers numerical data without a working
graph reference. Existing null/missing/nonnumeric policies remain explicit.
Numeric text conversion now supports compact/compressed encodings, rejecting
invalid text or corrupted compression. Full native suites and strict all-target
core Clippy pass; the rebuilt Python numerical/read/history gate has 671 passed
and one existing skip. All three documented examples execute. Bounded export
costs and remaining adapter migration are recorded in
`frozen-matrix-export-contract-2026-10-22.md`.

### AR7.5d — Row-aligned table exports and shared projection

Graph/Subgraph/GraphView table exports share native attribute-first projection
kernels and stable IDs/column order. Sparse properties no longer shift onto the
wrong Subgraph rows. Stale selections, unreadable references and structural-name
collisions fail explicitly. Frozen tables retain captured values/endpoints;
Python transfers owned columns with the GIL released during projection. Results
remain independent owned tables, without source-pin joins or graph writeback.
Full native suites and strict all-target core Clippy pass; rebuilt Python
numerical/read/history/table gate: 677 passed, one existing skip. All four
read-view examples execute. Policy boundaries, regressions and measured costs:
`frozen-table-export-contract-2026-10-22.md`. AR7.5/26.1/27.1 remain open.

### AR7.5e — Shared scoped live/frozen query predicates

QueryEngine, Subgraph and GraphView share column-oriented node/edge predicates.
Legacy nested queries stay inside the parent selection and preserve excluded
parallel-edge scope. Frozen results share the captured pin. Range comparisons
retain exact integer/mixed ordering; incident/Boolean conventions remain explicit.
Full native suites plus five focused native fixtures pass; strict all-target
core Clippy passes. Rebuilt Python core/subgraph/read/history/property/algorithm
gate: 255 passed, five existing skips. All five read-view examples execute.
`frozen-query-contract-2026-10-22.md` records boundaries and measured costs.
Broader query-language/index and algorithm migration remain open.

### AR7.5f — Immutable algorithm input and shared BFS/DFS kernels

AlgorithmRead shares captured selection, stable node-row identity and CSR without
a working owner. GraphView/native/Python integer-row traversal survives owner
mutation/drop and accepts recorded historical pins. Legacy pipeline wrappers
share the kernels while retaining current marker/telemetry/writeback ownership.
Full native suites, strict all-target core Clippy and 276 rebuilt Python tests
pass (five existing skips). Six documented examples execute. Exhaustive small
topology, cancellation, thread transfer and long-path evidence is recorded in
`frozen-traversal-contract-2026-10-22.md`; broader algorithm migration remains open.

### AR7.5g — Shared checked frozen Dijkstra distances

Live/frozen inputs share weight validation/minimum parallel-pair assembly and
one cancellable distance kernel. GraphView reads captured weights and selected
CSR, with owned f64 rows and no property writes. Python releases the GIL; legacy
pipeline marker/writeback behavior remains intact. Full native suites, strict
core Clippy and 316 rebuilt Python tests pass (five existing skips). Seven read
examples execute. Scope and costs: `frozen-dijkstra-contract-2026-10-22.md`.

### AR7.5h — Shared checked frozen A* paths

Live/frozen readers share heuristic validation and one CSR-indexed A* kernel.
Frozen paths reuse native Path and preserve actual selected edge IDs, deterministic
parallel ties and owned f64 cost. Python GraphPath is immutable data; captured
preparation/search/path assembly release the GIL. Full native suites and strict
core Clippy pass; rebuilt weighted-path/read/history Python gate: 325 passed,
five existing skips. Eight examples execute. Scope, edge assembly and costs:
`frozen-astar-contract-2026-10-22.md`. Broader algorithms remain open.

## Owner workbook answers — 2026-10-22

The owner supplied appendix answers and confirmed three clarifications: f64 means
64-bit floating point/default Python floats; later accepted writes win same-key/
effective-time ties; explicit group membership/source-pin revisions receive new
immutable references. Distributed subnode and meta-node property edits retain
their own mutation targets. Selected-state copy is the default; history transfer
is explicit. Invalid endpoint validity rejects (overriding retain-and-hide), unset
stays absent, and the first temporal types are UTC instant/date/duration.

Other answers address checkpoint measurement/deferred scheduling, verified format
conversion, immediate terminology migration, staged schema migration, table mixed
atomic writeback, explicit index failure, compatible component modes, one optimal
path, transient pipelines, timing metrics, explicit duplicate coalescing and a
synthetic future integration fixture. The complete question-by-question record,
qualifications, unfilled questions and pending clarification requests are in
`workbook-answers-2026-10-22.md`; the appendix source now contains the answers.

These are implementation directions, not completed code. In particular, f64
matrix operations do not mean property storage has stopped narrowing to f32.
Existing typed storage must remain readable; no historical precision can be
recovered after prior narrowing. Publication default/restore-source/purge
clarifications and all omitted fields remain unchosen. No destructive maintenance
is authorized by an ambiguous answer. The opening benchmark is a bounded warm
in-tree development baseline, not a production budget or checkpoint schedule.


## AR7.5i — Frozen component reads (2026-10-22)

Completed shared BFS/Tarjan component reads over AlgorithmRead/GraphView, with
immutable component views and exact selected edge identity. Preserve undirected
default and explicit weak/strong modes (accepted A12.1); no properties, deltas,
checkpoints or live caches are written. Native/Python and independent closure
checks cover scope, orientation and lifetime. See
`frozen-components-contract-2026-10-22.md`. Centrality and broad AR7.5 remain open.


## AR7.5j — Frozen PageRank (2026-10-22)

Completed captured PageRank using shared personalization validation and the
existing power-iteration kernel. Owned f64 results explicitly report convergence,
iterations and the final maximum update before mass normalization. Preserve live
persistence/telemetry, selected parallel occurrences and frozen properties;
Python releases the GIL and reads publish no mutation. See
`frozen-pagerank-contract-2026-10-22.md`. Closeness/betweenness and broad AR7.5
remain open; ordinary f64 property migration is separate.


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

# Module consolidation review — 2026-10-22

Status: reviewed and now tracked as AR7.1–AR7.11 in the flat checklist at the
user's request. The inventory, AR7.2 matrix move and AR7.3 array/numeric foundation are complete.
Remaining consolidation tasks are open; numerical closeout precedes entity/view work. Existing functionality and public imports should remain supported
through compatibility re-exports during migration.

## First: one matrix module

`storage/matrix` contains the public GraphMatrix, decompositions and derivatives.
`storage/advanced_matrix` contains storage, arithmetic, dtype, memory and autodiff.
Dependencies run both ways: advanced checked kernels import matrix dot products,
and advanced autodiff invokes GraphMatrix factor/derivative routines.
`storage/adjacency` is already a GraphMatrix alias plus graph builders.

Consolidate these under `storage/matrix`, with explicit internal responsibilities:

- public GraphMatrix API and owned numerical transfer;
- dense/CSR storage and logical layout;
- checked arithmetic kernels and linear algebra;
- graph construction/operators;
- automatic differentiation;
- execution/backend capabilities and memory buffers.

Keep the distinction between a public matrix and its internal storage. Removing
the advanced namespace does not require forcing graph references, tape ownership
and thread-transfer data into one struct. Move scalar dtype/conversion/accurate
accumulation primitives into a small shared numeric foundation because arrays
also use them. Keep convolution/layer-specific functionality distinct from
ordinary matrix arithmetic; autodiff itself is not exclusively a neural feature.

## Other candidates, in recommended order

| Area | Concrete evidence | Proposed boundary |
| --- | --- | --- |
| Arrays | `storage/array/base.rs` defines nongeneric BaseArray; `base_array.rs` defines the generic, publicly re-exported BaseArray<T>. | One canonical generic column representation; migrate still-needed metadata/behavior and retain aliases while callers move. Preserve NumArray/BoolArray specializations. |
| Entity handles | `storage/node.rs`/`edge.rs` define EntityNode/EntityEdge; `entities/node.rs`/`edge.rs` define graph-backed Node/Edge wrappers. Both families have callers. | One node/edge handle family under entities, with storage owning pooled data. Preserve trait/API behavior and compatibility names. |
| Graph views | Subgraph, FilteredSubgraph and other specialized wrappers carry related graph references, node/edge sets and behavior. | Share membership/state ownership and adjacency semantics; keep filter/component/neighborhood metadata as specialized behavior. Resolve AR4 state pins explicitly; moving files alone cannot freeze today's live-parent views. |
| Display | `display` and `viz/display` define separate DisplayConfig types and rendering systems; Python has another display package. `core` largely re-exports visualization. | One display configuration/data/formatting foundation; Rust/Python/interactive renderers are adapters. Keep core only as a migration facade, then retire misleading imports. |
| Query expressions | Array QueryEvaluator parses expressions itself; query/query_parser has a different parser/filter model, and the Python builder has expression parsing/IR. | Share defined scalar predicate/expression semantics and supported lowering. Keep graph traversal planning separate from scalar evaluation; these languages need comparison before unification. |
| Algorithm execution | Algorithm pipelines, step builders, batch plans/executors and Python builder IR are spread across several layers. | One explicit validation/lowering/execution contract, with algorithm composition and step/batch representations as levels. Consolidate overlapping implementations only after mapping supported features. |
| State/delta records | `state/history.rs` and `state/state.rs` each define AttributeChange and EntityType; delta/history/state representations overlap. | Canonical entity/change value types and explicit conversions. Keep history lineage, selected state and temporal indexing distinct responsibilities. |
| Tables | BaseTable, NodesTable, EdgesTable, GraphTable and TableArray have different responsibilities but broad combined surfaces. | One column/table foundation and clear ownership/update adapters for graph-backed tables. Preserve genuinely distinct node/edge bundles and collections. This needs boundary cleanup more than a type merger. |

## Migration sequence

1. Finish and benchmark the current numerical correctness slice.
2. Inventory matrix exports, callers and feature/error behavior; add import/API
   compatibility fixtures for the migration.
3. Move matrix internals and adjacency helpers behind one module, preserving
   GraphMatrix behavior and legacy advanced_matrix imports temporarily.
4. Consolidate array foundations and shared numeric primitives.
5. Consolidate entity handles and view ownership, tied to AR1/AR2/AR4.
6. Address display and execution/query boundaries in separate commits.
7. Consolidate shared change records and table ownership after contracts are clear.

The aim is one authoritative implementation per responsibility. A directory
rename is insufficient, and distinct concepts should retain separate types.
Do not combine GraphPool, GraphSpace and HistoryForest: retained values, selected
membership/property references and historical lineage are distinct responsibilities.
Likewise, temporal validity and recorded lineage should not be collapsed.

## AR7.4 completed ownership migration

Canonical Node/Edge handle state now backs historical adapters under entities;
storage/node and storage/edge are re-export facades. Source callers have moved.
Preserve constructor validation differences and historical specialized trait
behavior explicitly; see `entity-handle-consolidation-2026-10-22.md`. AR7.5 pinned
view ownership remains a separate behavioral migration.

## AR7.5a shared read-state foundation

Current captures and historical TemporalSnapshot now share immutable state/value/
membership/adjacency ownership. Ordinary graph/table view migration remains open:
`pinned-read-state-foundation-2026-10-22.md`. Measured fresh capture cost requires
related views to share a pin; property writes do not advance topology version, so
that version cannot identify cached property state.

## AR7.5b immutable selection representation

GraphView now owns canonical immutable membership over the shared read pin, with
explicit native/Python factories and edge-scoped nested selections. It has no
working mutation owner. Existing adapters still need migration; do not relabel
their live reads. Read-state incidence/index cost is measured and documented in
`immutable-selection-contract-2026-10-22.md` and its benchmark note.

## AR7.5c frozen numerical projections

Live Graph and immutable GraphView now share matrix feature/adjacency projection
kernels. The view adapter reads selected captured values, with no working owner;
Python transfers owned numerical data across the GIL boundary. Storage encoding
no longer changes numeric text conversion. See
`frozen-matrix-export-contract-2026-10-22.md` for policies, tests and receipts.
Existing table/query/algorithm consumers still require migration.

## AR7.5d shared table projection

Graph and Subgraph no longer assemble tables in separate per-entity loops.
GraphView uses the same attribute-first kernel over captured maps. Owned column
transfer is distinct from display/server-bearing tables and writable graph owners.
This repairs sparse-property row alignment in legacy Subgraph exports and makes
ordering/error policy consistent. Source-pin propagation through table selections/
joins and graph writeback remain separate; see
`frozen-table-export-contract-2026-10-22.md`.

## AR7.5e shared query read source

QueryEngine/Subgraph/GraphView use one predicate executor and live/frozen read
adapters. Attribute matching is column-oriented and borrows values; live incident
topology is constructed lazily once per query. Subgraph FFI now delegates scoped
selection construction to Rust, fixing whole-parent leakage. Scalar ranges use
the canonical exact comparison helper. This does not consolidate the separate
array expression language, algorithm pipeline IR or pending query-language scope.
See `frozen-query-contract-2026-10-22.md`.

## AR7.5f immutable algorithm input

AlgorithmRead holds captured selection plus shared ordered node IDs/index and CSR,
without a working graph owner. Frozen traversal and legacy pipeline BFS/DFS share
one integer-row kernel implementation; marker/result writeback adapters remain
separate. Native input clones share preparation; Python convenience methods
prepare per call and release the GIL. Remaining weighted/centrality/component and
pipeline readers still need migration. See
`frozen-traversal-contract-2026-10-22.md`.

### AR7.5g — Shared checked frozen Dijkstra distances

Live/frozen inputs share weight validation/minimum parallel-pair assembly and
one cancellable distance kernel. GraphView reads captured weights and selected
CSR, with owned f64 rows and no property writes. Python releases the GIL; legacy
pipeline marker/writeback behavior remains intact. Full native suites, strict
core Clippy and 316 rebuilt Python tests pass (five existing skips). Seven read
examples execute. Scope and costs: `frozen-dijkstra-contract-2026-10-22.md`.

## AR7.5h shared A* search and owned path bridge

Live and captured inputs share heuristic domains/consistency and one CSR-indexed
A* kernel. Search no longer performs a linear node-ID lookup on each heap pop.
Frozen results reuse query::Path and retain actual selected edge instances;
Python GraphPath only translates immutable owned data. The legacy pipeline keeps
its marker/path-index/telemetry adapter. See `frozen-astar-contract-2026-10-22.md`.


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

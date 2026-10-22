# Module consolidation review — 2026-10-22

Status: proposal following the user's request to unify matrix/advanced_matrix
and identify similarly ambiguous boundaries. No module moves are implemented by
this review. Existing functionality and public imports should remain supported
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

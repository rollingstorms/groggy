# Checked table visualization conversion — 2026-10-25

Existing 27.1 / AR7.10 conversion and 28.15 visualization boundaries; no new
checklist rows or broad parent completion.

## Repair

The three Python table `.viz` getters previously rebuilt a temporary manual graph
in FFI, allocated dense IDs but wrote attributes/edges using original table IDs,
accepted only Int structural cells and ignored mutation failures. Sparse rows
could lose their properties and edges; optimized SmallInt rows disappeared.

`src/storage/table/visualization.rs` now owns strict structural reads and typed
row adaptation. `GraphDataSource::from_entities` checks unique identities and
endpoint closure before accepting an owned capture, then uses existing native
label/position/weight formatting. Node/edge row order and original IDs are
preserved. Edge-only captures synthesize only endpoint nodes in ascending ID
order with empty properties; mixed captures reject dangling endpoints. Direction
is explicit natively; Python retains its former undirected default.

FFI delegates with the GIL released, translates errors and wraps the result.
No graph allocation/publication/history or typed-reference remapping occurs.
The shared GraphDataSource window reader also clamps out-of-range starts and
uses saturating end arithmetic, avoiding panic/overflow for empty or oversized
requests. Reference attributes stay opaque with original IDs; pooled contents
are not expanded or serialized by this adapter.

## Verification

Seven native cases cover sparse/optimized IDs, parallel edges/self loops,
typed/null/reference/JSON attributes, numeric weights and positions, explicit
direction, invalid/duplicate IDs, dangling endpoints, source write isolation,
empty/overflow windows and layout ID preservation. Eight Python cases exercise
all three getters, sparse/empty inputs and atomic rejection before server start.
Three executable API examples are in `docs/api/table-visualization.md`.

Full native: 1,138 passed / 51 ignored. Root Python: 1,869 passed / 1 skipped.
Documentation and strict Clippy receipts are recorded with the slice completion.

## Performance boundary

`benches/table_visualization.rs` compares current native direct checked capture
against checked `to_graph` materialization plus GraphDataSource capture on the
same dense chain (1k/10k/100k nodes, two node properties, one edge property).
Three observations per case; fixtures, assertions, metadata reads and destruction
excluded. At 100k nodes median direct capture was about 80ms versus 798ms for the
current checked materialization alternative. This is not a measured speedup
against the old incorrect FFI loop, nor an RSS/Python/GIL/server/FFI budget claim.
Both alternatives preserve the same IDs on this dense fixture; sparse-ID
behavior is verified independently. Receipt:
`notes/benchmarks/table-visualization-2026-10-25.json`.

## Remaining existing boundaries

The three unsupported graph-accessor/subgraph converters retain their unresolved
ownership/identity contracts. GraphTable subset/rectangular metadata choices
remain open. Native table DataSource compatibility readers, interactive_viz
option/fallback paths and other viewer controls remain outside this repair.
An owned visualization capture is not the connected-view migration in 26.1 or
portable pooled/history materialization in 07.3/10.3/12.2.

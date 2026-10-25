# Graph-table feature conversion conformance — 2026-10-25

Existing requirements: 27.1 / AR7.10. This closes the two native domain
conversions and their three existing typed conveniences without adding checklist
rows or changing the broader parent status.

## Completed inventory

| Existing adapter | Status |
| --- | --- |
| Node `to_matrix<T>` | Complete: one shared checked feature conversion |
| Node f64/f32/i64 conveniences | Complete: delegate to that same conversion |
| Edge `edges_to_matrix<T>` | Complete: same implementation, structural exclusions and feature names |

Both old implementations traversed columns but passed data to a row-major
constructor. Node conversion lost feature width on empty input; edge conversion
also lost feature names. The shared implementation resolves columns once,
validates lengths and IDs, checks allocation size, and fills rows in physical
order. It preserves the existing `FromAttrValue` dtype/null/text/boolean rules.
Malformed/duplicate IDs now reject even when no features exist. Errors include
feature and row. Edge IDs do not become misleading node labels.

These matrices are owned numerical copies. They do not inherit connected table
writeback identity or mutate source graph history. No Python method or new matrix
module is introduced. Implementation remains owned by GraphTable.

## Evidence

`tests/graph_table_features.rs` covers 32 small shapes in both domains (including
zero dimensions), nonsequential IDs, physical feature order, exact f64, nulls,
booleans/numeric text, integer parsing/range and f32 range errors, invalid and
duplicate IDs, contextual errors and owned-copy independence.

`benches/graph_table_features.rs` measures 1k/10k/100k rows with eight features
in each domain, five optimized native observations. Input construction and
output assertions are excluded; complete validation/conversion/matrix allocation
is included. Receipt: `notes/benchmarks/graph-table-features-2026-10-25.json`.
No matched speedup, FFI latency, RSS or allocation claim is made.

API contract: `docs/api/graph-table-features.md`.

Remaining under the existing parents: graph/table reshaping and legacy bundle
conversion boundaries. This slice does not certify CSV type/history portability
or close legacy connected-read and pooled-portability requirements.

## Performance observations

Median native conversion times in this receipt (rows × 8 features):

| Rows | Node matrix | Edge matrix |
| --- | --- | --- |
| 1,000 | 0.066 ms | 0.035 ms |
| 10,000 | 0.653 ms | 0.330 ms |
| 100,000 | 8.163 ms | 3.882 ms |

Node conversion also constructs row-label strings. These are current machine
observations, not a regression comparison or a guaranteed latency budget.


Verification: **1,084 native tests passed, 51 ignored; 1,812 root Python tests
passed, 1 skipped; 109 executable documentation examples passed; strict all-target
Clippy and scoped formatting checks passed**. The in-tree extension was rebuilt.
Native gates: `cargo test --offline --locked --quiet` and
`cargo clippy --offline --locked --all-targets -- -D warnings`, with
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`.

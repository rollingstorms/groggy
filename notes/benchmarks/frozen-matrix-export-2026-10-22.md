# Frozen matrix export timing receipts — 2026-10-22

Locked offline release Criterion runner, idle; 200 ms warmup, 1 second
measurement, 10 samples. Ring fixtures with one scalar node attribute. Graph
setup/read-state capture excluded, output construction/drop included. These are
native costs; no FFI latency, allocation/RSS, old-exporter speedup or SLA claim.

| Nodes / edges | Current live scalar features | Frozen scalar features | Frozen unweighted CSR adjacency |
| --- | ---: | ---: | ---: |
| 1,000 / 1,000 | 0.266 ms | 0.305 ms | 0.522 ms |
| 10,000 / 10,000 | 2.763 ms | 4.053 ms | 5.755 ms |

The frozen feature path is slower than the live path on these fixtures: frozen
column discovery traverses per-node captured maps, while live reads use pooled
attribute metadata. Both must materialize an owned matrix and metadata. Related
exports reuse the capture; this does not make projection constant time. Captured
map layout/column-oriented acceleration remains an optimization opportunity, not
a correctness blocker or a measured speedup. Broader budgets remain open.

Receipt: `benches/frozen_matrix_export_2026-10-22.json`; fixture and timing setup:
`benches/read_state.rs`. Run only the export cases with
`read_state --bench --noplot 'read_state/(live_features|frozen_features|frozen_sparse_adjacency)'`.

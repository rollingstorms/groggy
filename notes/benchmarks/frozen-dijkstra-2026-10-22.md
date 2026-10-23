# Frozen Dijkstra timing receipts — 2026-10-22

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Undirected rings with default unit weights. Capture/setup
excluded. Prepared calls include weight validation/minimum-pair map, heap traversal
and owned row creation/drop; convenience calls additionally prepare CSR.

| Nodes / edges | Prepared Dijkstra | Convenience Dijkstra |
| --- | ---: | ---: |
| 1,000 / 1,000 | 251.01 µs | 322.20 µs |
| 10,000 / 10,000 | 2848.78 µs | 3597.49 µs |

These are bounded native costs, not FFI latency, memory/RSS, a previous-kernel
speedup or a production SLA. The timing fixture uses unit defaults; varying and
invalid weights are covered by correctness fixtures. Receipt:
`benches/frozen_dijkstra_2026-10-22.json`; fixture: `benches/read_state.rs`.

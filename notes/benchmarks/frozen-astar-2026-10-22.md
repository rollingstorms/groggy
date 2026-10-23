# Frozen A* timing receipts — 2026-10-22

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Undirected rings, start row zero and goal halfway around
the ring, default unit weights and zero heuristic. Setup/capture excluded.
Prepared calls include weight/heuristic checks, search, owned path/selected-edge
reconstruction and drop; convenience also prepares CSR.

| Nodes / edges | Prepared A* | Convenience A* |
| --- | ---: | ---: |
| 1,000 / 1,000 | 366.13 µs | 424.40 µs |
| 10,000 / 10,000 | 4094.41 µs | 4734.90 µs |

These are bounded native costs, not FFI latency, allocation/RSS, a previous-engine
speedup or production SLA. Varying weights and nonzero/invalid heuristics are
covered by correctness fixtures rather than this timing fixture. Path instance
reconstruction scans selected edge IDs, not unrelated captured adjacency.
Receipt: `benches/frozen_astar_2026-10-22.json`; fixture: `benches/read_state.rs`.

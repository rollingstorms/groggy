# Immutable traversal timing receipts — 2026-10-22

Locked offline release Criterion, idle retained run; 200 ms warmup, 1 second
measurement, 10 samples. Undirected rings. Current graph capture/setup excluded;
preparation includes sorting IDs, index and CSR. Clone shares prepared state.
Prepared BFS/DFS include owned result creation/drop; convenience BFS also includes
input preparation. No FFI latency, allocation/RSS, previous-kernel speedup or
production-budget claim.

| Nodes / edges | Prepare | Input clone | Prepared BFS | Prepared DFS | Convenience BFS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 63.35 µs | 19.97 ns | 8.72 µs | 12.36 µs | 77.55 µs |
| 10,000 / 10,000 | 862.43 µs | 19.20 ns | 81.41 µs | 126.80 µs | 841.35 µs |

Preparation dominates repeated BFS work here; native callers can prepare once
and reuse the input. Python convenience methods explicitly prepare each call.
Clones share the view/row index/CSR rather than reconstructing topology. The DFS
kernel still sorts a private CSR copy to preserve the existing general-kernel
contract; its cost is included. These are bounded native fixtures, not a claim
that capture/preparation/traversal is constant time or an FFI budget certificate.

Receipt: `benches/frozen_traversal_2026-10-22.json`; fixture: `benches/read_state.rs`.
Filter: `read_state/(algorithm_prepare|algorithm_clone|prepared_bfs|prepared_dfs|frozen_bfs)`.

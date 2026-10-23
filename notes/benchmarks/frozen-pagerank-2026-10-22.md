# Frozen PageRank timing receipts — 2026-10-22

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Existing undirected ring fixture with one node value
column, 1k/10k nodes and edges. Capture/setup excluded. Prepared calls reuse CSR;
convenience also prepares CSR. Both include personalization validation,
normalization, iteration, owned result allocation and destruction.

Damping 0.85, budget 50, maximum-coordinate update tolerance 1e-6. Uniform
personalization converges in one iteration on these symmetric rings. Explicit
personalization uses the captured integer `value=node ID` column. Convergence
metadata is collected separately through the same native kernel.

| Nodes / edges | Prepared uniform | Convenience uniform | Prepared personalized | Convenience personalized | Personalized iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 13.92 µs | 71.55 µs | 207.42 µs | 261.26 µs | 32 |
| 10,000 / 10,000 | 142.58 µs | 832.23 µs | 1638.56 µs | 2462.18 µs | 18 |

All cases report convergence. Personalized final updates are approximately
9.405e-7 and 9.440e-7. Fixed coordinate tolerance is not a graph-size-independent
error bound; the different iteration counts must be considered when comparing
sizes. This fixture is not a worst-case convergence or iteration-budget test.

These bounded native measurements do not establish a previous-engine speedup,
FFI latency, allocation/RSS or production SLA. Invalid inputs, disconnected and
directed graphs, sinks, parallel edges, loops, and exhausted budgets are checked
by correctness fixtures rather than this ring timing fixture.

Receipt: `benches/frozen_pagerank_2026-10-22.json`; fixture:
`benches/read_state.rs`.

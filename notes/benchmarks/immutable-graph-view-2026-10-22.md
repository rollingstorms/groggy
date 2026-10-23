# Immutable selection timing receipts — 2026-10-22

Locked offline release Criterion runner, idle, 200 ms warmup/1 second measurement/
10 samples. Same ring fixtures as the read-state baseline. Setup excluded and
output drop included. Native regression estimates, not FFI latency or a service
budget; no heap/RSS measurement is made here.

| Nodes / edges | Indexed current capture | Read-pin clone | Full view from pin | Selected-view clone |
| --- | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 0.756 ms | 10.25 ns | 10.06 ns | 11.67 ns |
| 10,000 / 10,000 | 9.645 ms | 10.29 ns | 10.03 ns | 11.67 ns |

Adding edge-aligned adjacency and incoming/outgoing incidence support increases
fresh capture cost relative to the prior 0.574/7.628 ms fixtures (approximately
32%/26%). It also retains O(V+E) indexing storage. This is an explicit trade-off
for degree-proportional, edge-correct scoped reads; it is not reported as a capture
speedup. Related selections share one pin, so they do not repay that capture cost.
Production budget/index compaction choices remain open.

Receipt: `benches/immutable_graph_view_2026-10-22.json`. Clone/full-view estimates
remain size-independent across these fixtures. This does not measure selection
construction or certify constant time for capture, sorting, or arbitrary queries.

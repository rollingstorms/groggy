# Frozen Brandes timing receipts — 2026-10-23

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Induced undirected paths selected from captured 1k/10k
rings: 100/1000 selected nodes and 99/999 edges. Normalized all-source scores.
Capture/selection excluded; outputs and destruction included. Prepared unweighted
calls reuse CSR; explicit lengths validate and prepare minimum-instance CSR.
Convenience also prepares the base captured CSR.

| Selected nodes / edges | Prepared hops | Convenience hops | Prepared explicit lengths | Convenience explicit lengths |
| --- | ---: | ---: | ---: | ---: |
| 100 / 99 | 168.18 µs | 180.89 µs | 620.46 µs | 625.56 µs |
| 1,000 / 999 | 14098.46 µs | 13752.27 µs | 59893.74 µs | 59989.42 µs |

The explicit `length` attribute is absent, exercising the documented unit-length
default with weighted validation/filtering/Dijkstra. The 1000-node weighted cases
use Criterion's flat sampling and mean estimates; other cells use slope estimates.
Small prepared/convenience reversals are measurement variation, not a speedup
claim; full estimates/samples are retained. General weighted topology, ties,
parallel instances, loops and directed/disconnected graphs are correctness
fixtures, not this timing fixture.

These are bounded native costs, not FFI latency, allocation/RSS, production scale
or a before/after optimization comparison. Receipt:
`benches/frozen_betweenness_2026-10-23.json`; fixture: `benches/read_state.rs`.

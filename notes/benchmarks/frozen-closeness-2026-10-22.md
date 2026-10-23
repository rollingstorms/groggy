# Frozen closeness timing receipts — 2026-10-22

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Induced undirected paths selected from the existing
captured 1k/10k ring: 100/1000 selected nodes, 99/999 edges. Harmonic outgoing
unnormalized hop distances. Capture/selection excluded. Prepared calls reuse CSR;
convenience includes CSR preparation. All-source distance computation, owned rows
and destruction are included.

| Selected nodes / edges | Prepared | Convenience |
| --- | ---: | ---: |
| 100 / 99 | 81.14 µs | 85.33 µs |
| 1,000 / 999 | 6229.44 µs | 6268.55 µs |

These are bounded all-source native costs, not a previous-engine speedup, FFI
latency, allocation/RSS or production SLA. The quadratic traversal cost dominates
at the larger fixture. Directed incoming transposition, weighted paths and
fragmentation are correctness fixtures, not this timing fixture.
Receipt: `benches/frozen_closeness_2026-10-22.json`; fixture: `benches/read_state.rs`.

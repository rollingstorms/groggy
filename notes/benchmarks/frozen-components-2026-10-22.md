# Frozen component timing receipts — 2026-10-22

Locked offline release Criterion; idle retained run, 200 ms warmup, 1 second
measurement, 10 samples. Existing undirected ring fixture with one node value
column, 1k/10k nodes and edges. Capture/setup excluded. Prepared calls reuse CSR;
convenience calls also prepare CSR. Both include kernel computation, immutable
result membership allocation and result destruction. One component per ring.

| Nodes / edges | Prepared weak | Convenience weak | Prepared strong | Convenience strong |
| --- | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 129.45 µs | 190.31 µs | 135.76 µs | 199.23 µs |
| 10,000 / 10,000 | 1449.55 µs | 2158.25 µs | 1499.04 µs | 2207.45 µs |

These bounded native measurements do not establish a previous-engine speedup,
FFI latency, allocation/RSS or production SLA. Directed symmetric preparation,
isolate-heavy fragmentation and many SCCs are correctness fixtures rather than
this timing fixture. Returning component membership requires owned sets while
state/property values remain shared with the captured input.

Receipt: `benches/frozen_components_2026-10-22.json`; fixture:
`benches/read_state.rs`.

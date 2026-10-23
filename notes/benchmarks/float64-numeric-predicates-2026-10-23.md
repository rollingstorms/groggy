# f64 numeric predicate scan receipts — 2026-10-23

Locked offline release Criterion; no concurrent build/test during retained
measurements; 200 ms warmup, 1 second measurement, 10 samples. Input row i stores
f64 `1+i*2^-40`, threshold `1.0000000005`; both scans assert exactly n-550 matches.
Setup excluded; output is a match count, without row materialization or FFI.

| Rows | Array QueryEvaluator scan | Table scalar-predicate scan |
| --- | ---: | ---: |
| 1,000 | 129.226 µs | 14.895 µs |
| 10,000 | 1233.334 µs | 146.613 µs |

Both include existing per-value scalar parsing. The array path also repeats its
expression/operator handling; sharing the exact scalar comparator does not make
these parsers one compiled plan. That larger AR7.7 work remains open. These are
bounded native costs, not a speedup comparison, FFI latency or production SLA.
The first array case was rerun idle after documentation-example verification.
Receipt: `benches/float64_numeric_predicates_2026-10-23.json`; fixture:
`benches/read_state.rs`.

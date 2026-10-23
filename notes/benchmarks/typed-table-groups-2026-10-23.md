# Typed table grouping — 2026-10-23

Locked offline release Criterion; 200 ms warmup, 1 s measurement, 10 samples.
Seventeen repeated explicit f64 keys 1+(i%17)*2^-40 and integer unit values.
Input setup excluded; public aggregation-spec clone, grouping, result construction
and destruction included. No concurrent build/test.

| Rows | Median |
| ---: | ---: |
| 1,000 | 89.115 µs |
| 10,000 | 764.693 µs |

Current implementation measurements only: no before/after speedup, FFI latency,
RSS, JSON parser/I/O timing or large-graph capacity claim. Grouping borrows typed
keys, hashes their logical content and copies keys only into result groups. It
does not allocate a debug string for each grouping cell. Group order remains
unspecified. JSON round-trip parsing prioritizes precision; parser throughput is
not certified by this grouping benchmark.

Raw samples: `benches/typed_table_groups_2026-10-23.json`.

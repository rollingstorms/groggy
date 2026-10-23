# Table profile measurements — 2026-10-23

Native locked offline release Criterion, 200 ms warmup, 1 s measurement, 10
samples. One source column; setup excluded, profile result construction/destruction
included. No concurrent builds/tests. F64 values are 1+i*2^-40; i64 values lie
near i64::MAX. These are bounded current measurements, not a before/after
speedup, FFI latency, allocation/RSS or large-graph guarantee.

| Column | Rows | Median |
| --- | ---: | ---: |
| profile_f64 | 1,000 | 100.277 µs |
| profile_f64 | 10,000 | 870.800 µs |
| profile_high_integer | 1,000 | 101.726 µs |
| profile_high_integer | 10,000 | 886.710 µs |

Raw samples/estimates: `benches/table_profiles_2026-10-23.json`.
Profiling hashes all typed values and sorts numeric observations. It is not an
O(1) graph mutation path. Borrowed typed hashes avoid per-value debug strings;
the numeric adapter retains source references plus scalar hints.

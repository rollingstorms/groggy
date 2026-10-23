# Checked outlier classification — 2026-10-23

Native locked offline release Criterion; 200 ms warmup, 1 s measurement, 10
samples. Sorted inputs have one low extreme followed by increasing values:
f64 uses 1+i*2^-40 (extreme -1e300), i64 lies near i64::MAX. Setup excluded;
result allocation/destruction included. No concurrent build/test.

Before is the staged implementation accumulating exact comparisons for every
row, not the prior released f32-only method. After uses correctly rounded bounds
to certify distinct rounded order and exact residuals only for collisions. Both
implementations have the same checked result contract.

| Input | Rows | All-exact median | Adaptive median |
| --- | ---: | ---: | ---: |
| outliers_f64 | 1,000 | 856.512 µs | 18.573 µs |
| outliers_f64 | 10,000 | 8562.784 µs | 158.510 µs |
| outliers_high_integer | 1,000 | 972.368 µs | 499.438 µs |
| outliers_high_integer | 10,000 | 9801.335 µs | 163.799 µs |

These fixtures are favorable to sorted-run handling and have data-dependent
boundary collisions. Near i64::MAX the 1k integer fixture still needs many exact
residuals; the larger spread in the 10k fixture separates its rounded fences.
No universal speedup, random-input/worst-case latency, FFI budget, RSS or graph
capacity claim is made. Sorting is O(n log n) in general; collision comparisons
use a fixed-size exact accumulator. Raw samples/estimates:
`benches/checked_table_outliers_2026-10-23.json`.

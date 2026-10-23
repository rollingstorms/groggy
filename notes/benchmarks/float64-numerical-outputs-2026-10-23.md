# f64 numerical output receipts — 2026-10-23

Locked offline release Criterion, 200 ms warmup, 1 second measurement, 10 samples,
no concurrent build/test during retained runs. Inputs are 1k/10k f64 properties
`1+i*2^-40`, owned arrays and ten borrowed table chunks. Setup excluded; ordinary
finite-data computation/output destruction included, no overflow fallback or FFI.

The baseline is the staged intermediate f64-output/borrowed-table implementation
before the exact overflow fallback and scaled-std refinements. It is not a prior
released-engine checkout. This comparison checks ordinary-path cost of the range
repair, not the cost of the complete f32-to-f64 migration.

| Rows | Operation | Staged baseline | Final |
| --- | --- | ---: | ---: |
| 1,000 | Attribute mean | 1.881 µs | 1.907 µs |
| 1,000 | Attribute sum | 1.934 µs | 1.923 µs |
| 1,000 | Cross-table mean | 2.630 µs | 2.595 µs |
| 1,000 | Cross-table sum | 2.634 µs | 2.589 µs |
| 10,000 | Attribute mean | 18.958 µs | 18.946 µs |
| 10,000 | Attribute sum | 19.276 µs | 18.921 µs |
| 10,000 | Cross-table mean | 24.054 µs | 23.864 µs |
| 10,000 | Cross-table sum | 23.936 µs | 23.993 µs |

Differences are small measurements, not universal speedup/SLA claims. The rare
wide moment path allocates one fixed-size exact accumulator; these ordinary-data
fixtures do not measure that path or RSS. Raw samples/estimates are retained in
`benches/float64_numerical_outputs_2026-10-23.json`.

The release numerical corpus also completes all 55 cases with finite checksums.
All original 25 checksums match `numerical_completion_current_2026-10-22.json`.
Against the earlier 52-case table/statistics corpus, only five overlapping
statistics checksums change: expanding std at 1,700/17,000/170,000 and high-integer
table describe at 1,700/17,000. They now retain f64 output bits instead of rounding
to f32. Shape/domain references still pass. Full receipts including process-wide
System allocation requests/peaks (not RSS) are in
`benches/numerical_float64_outputs_2026-10-23.json`.

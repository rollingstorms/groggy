# Economy SVD release receipts — 2026-10-22

The release corpus now has 25 cases. After Cargo/rustc jobs completed, the built
benchmark binary was run directly with seven measured samples per case.
These are instrumented System-allocator timings and additional-live-heap peaks,
not RSS, stack or universal throughput guarantees. Fixture preparation is
excluded; factor outputs/checksums and their drop are included.

| Case | Median ms | Additional peak heap bytes |
| --- | ---: | ---: |
| 512×4 full SVD | 306.0467 | 4,194,664 |
| 512×4 economy SVD | 0.5141 | 53,600 |
| 10,000×4 economy SVD | 10.1778 | 1,040,352 |
| 10,000×4 numerical rank | 10.0587 | 1,040,352 |
| 10,000×4 least squares, one RHS | 10.7080 | 1,120,384 |

Full and economy 512×4 factors use the same Jacobi rotations and return identical
singular-value checksums. The roughly 595× timing and 78× heap differences in
this shape come from omitting full null-basis completion/allocation and its
extra labels. They do not imply similar ratios for square matrices or other
spectra. The original full API remains available and intentionally still pays
for its full output.

Raw samples: `benches/numerical_completion_economy_2026-10-22.json`.
Reproduce with the `numerical_completion` Cargo bench target and an output path
in `GROGGY_NUMERICAL_PROFILE`, as documented in the earlier completion benchmark
notes. The initial twenty-case before/after receipts remain unchanged.

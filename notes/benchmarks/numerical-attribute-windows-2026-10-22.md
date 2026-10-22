# Attribute expanding release receipts — 2026-10-22

`benches/numerical_completion.rs` now runs 50 cases. Build locked/offline in release,
then execute the runner idle. One warmup, seven measured samples; input setup
excluded, output allocation/drop included. Original 44 case checksums match the
expanded receipt within 1e-13. This is a scaling receipt, not an old/new timing
comparison or a production latency guarantee.

| Rows | Expanding population std | Expanding mean | Additional live heap |
| --- | ---: | ---: | ---: |
| 1,700 | 0.058 ms | 0.171 ms | 81,640 bytes |
| 17,000 | 0.584 ms | 1.752 ms | 816,040 bytes |
| 170,000 | 5.847 ms | 17.387 ms | 8,160,040 bytes |

Both operations allocate twice (the output vector and canonical column owner).
Tenfold input growth produces approximately tenfold work and output memory on
this fixture. Working state is fixed-size; stack/RSS are not measured. Each final
result is checked against the uniform integer distribution's mean 0/std sqrt(24).

Receipt: `benches/numerical_attribute_windows_2026-10-22.json`. Full native suites,
strict all-target core Clippy and **474 rebuilt Python regressions** pass. Rolling
retains its O(N*window) scan bound and is not certified by this expanding receipt.

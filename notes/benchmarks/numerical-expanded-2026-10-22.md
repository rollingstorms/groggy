# Expanded numerical release baseline — 2026-10-22

Extends the existing 25-case corpus to **44** bounded workloads. Build offline in
release mode, then run the already-built binary with no Cargo/rustc jobs running.
One warmup and seven measured samples; fixture setup is excluded. Operations
include output creation/drop and checksum. System allocator instrumentation counts
process allocation/reallocation requests and additional live-heap peaks, not RSS,
stack or total process memory. Raw receipt:
`benches/numerical_expanded_2026-10-22.json`.

| Workload | Median ms | Additional peak heap bytes |
| --- | ---: | ---: |
| Dense fractional product, 128-square | 27.087 | 393,568 |
| Dense fractional product, 256-square | 172.128 | 1,573,216 |
| Dense fractional product, 512-square | 1,333.939 | 6,291,808 |
| Million-element add + sum | 3.888 | 8,388,648 |
| Million-element dot | 7.103 | 0 |
| Million-element norm | 13.855 | 0 |
| Million-element reference sum | 1.710 | 0 |
| Million-element historical SIMD-name sum | 1.659 | 0 |
| Million-element auto sum | 1.724 | 0 |
| CSR 2000-square product, bandwidth 1 | 0.613 | 176,346 |
| CSR 2000-square product, bandwidth 4 | 11.596 | 405,722 |
| CSR 2000-square product, bandwidth 16 | 64.382 | 1,192,154 |
| CSR diagonal 100,000-square solve | 180.103 | 60,000,080 |
| CSR diagonal 100,000-square Cholesky | 31.930 | 48,000,192 |
| CSR 100,000-node cycle operator construction | 36.995 | 15,094,272 |

Bandwidth products retain CSR. Input/output nonzeros are respectively
2,000/2,000, 8,000/14,000 and 32,000/62,000. All rows have exactly representable
power-of-two-normalized weights; product row-sum checks validate setup before
measurement. This bounded banded fill pattern does not predict arbitrary graph
factorization fill. Large operator construction measures a matrix builder, not
full graph ingestion or retained history.

All samples have finite checksums. The original 25 workloads agree with the
pre-consolidation receipt at 1e-13 relative/absolute tolerance. New coverage also
includes column-major storage and mixed CSR/dense multiple-column products.

The reference/SIMD-name/auto reduction calls share the same checked reference
kernel, confirmed by code and identical results. Timing noise is not evidence of
SIMD acceleration. Compiler autovectorization is not certified. The 512-square
product remains costly: accurate in-house arithmetic is not a NumPy-speed claim.
No new precision mode or external backend is substituted.

Full locked offline native suites, 455 rebuilt Python regressions and strict
`cargo clippy --offline --locked --all-targets -- -D warnings` passed after the
lint cleanup. Existing Python extension warnings remain distinct from this core
Clippy command. Production latency/memory budgets, actual workload selection and
billion-entity goals remain A15/29.26/32 work. This is a measured baseline, not a
universal throughput or memory guarantee.

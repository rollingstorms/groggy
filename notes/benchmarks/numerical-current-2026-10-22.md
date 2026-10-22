# Current numerical checkpoint — 2026-10-22

Source commit: `b19aa9b7` on `groggy-roadmap-2026-10-20`.
Full locked offline native suites and 455 rebuilt Python numerical/array
regressions passed. Focused Clippy passed excluding the previously recorded
sort/match lint categories; this is not a new strict all-target clean claim.

Rebuilt the existing optimized 25-case numerical corpus offline, then reran its
already-built binary with no Cargo/rustc jobs running. One warmup and seven
measured samples per case, with process-wide System allocator instrumentation.
Fixture setup is excluded; output creation/drop and checksum are included.
Additional live heap is not RSS, stack or a whole-process memory guarantee.

| Workload | Median ms | Additional peak heap bytes |
| --- | ---: | ---: |
| Dense fractional 64x64 product | 4.531 | 98,656 |
| Dense integer 64x64 product | 0.868 | 98,656 |
| CSR diagonal 10,000-square solve | 15.277 | 6,000,080 |
| Full SVD, 512x4 | 314.754 | 4,194,664 |
| Economy SVD, 512x4 | 0.481 | 53,600 |
| Economy SVD, 10,000x4 | 10.055 | 1,040,352 |

The retained receipt is `benches/numerical_completion_current_2026-10-22.json`.
All 25 workload checksums agree with the earlier economy receipt at 1e-13
relative/absolute tolerance. These instrumented measurements are a current
checkpoint, not evidence of a further optimization or universal performance.
The broader size/fill sweep, SIMD/dispatch review and strict final gates remain
open in the numerical completion pass.

Reproduction uses the existing `benches/numerical_completion.rs` runner and the
command in `numerical-completion-2026-10-22.md`.

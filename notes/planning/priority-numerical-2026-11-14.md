# Numerical throughput pass — 2026-11-14

Existing 29.6/29.26/32.3 scope. Reuse the 55-case `numerical_completion` release
corpus, including fractional/integer dense products, column-major storage,
million-element vectors, sparse bandwidth/fill, sparse-native solves/factors,
large tall decompositions, activations and data transfer. Seven measured samples
per case follow one warm call; fixture setup is excluded. No new dependency,
external numerical backend or SIMD claim is introduced.

Raw baseline: `notes/benchmarks/numerical-priority-2026-11-14.json`.
Raw follow-up: `notes/benchmarks/numerical-priority-after-2026-11-14.json`.
Both run on macOS arm64, Rust 1.96.1, optimized builds, without concurrent local
builds/tests. These are ordered corpus runs, not randomized process-pair trials.
Allocation counters measure requested Rust bytes, not RSS or allocator retention.

## Measured repair

The dense fallback traversed the right-hand matrix by a strided column for every
output cell. For inner/output widths at least 32, materialize its columns
contiguously once, release the old right-hand copy, then allocate the output.
The same shared checked/compensated product kernel consumes every term in the
same k order. No sum reassociation, zero skipping or reduced integer precision.
Logical row/column-major inputs still pass through their existing checked export.
Sparse-native and guarded expansion paths retain their existing dispatch.

| Dense square size | Before median | After median | Additional requested peak, both |
| --- | --- | --- | --- |
| 128 | 26.096 ms | 23.460 ms | 393,568 bytes |
| 256 | 163.609 ms | 151.215 ms | 1,573,216 bytes |
| 512 | 1,154.310 ms | 1,035.924 ms | 6,291,808 bytes |

The observed large-case reductions are about 8–10%, with unchanged requested
peak storage. Smaller cases show small variation. This remains a scalar checked
implementation; a 512 product still costs roughly a second on this fixture.
Do not describe it as external-backend parity or a universal performance target.

## Verification

Two native regressions compare logical layouts and cancellation-sensitive
floating values against checked vector dots bit-for-bit, retain checked integer
output overflow and confirm zero-times-infinity still yields NaN. Full native,
Python and documentation gates are recorded in the combined priority receipt.

Remaining: broader workload targets, allocation/RSS budgets, tuned kernels,
concurrency/GIL evidence for every promised operation and genuine acceleration
where appropriate. This closes the bounded profiling/repair slice, not all
numerical production or sparse/decomposition-domain promises.

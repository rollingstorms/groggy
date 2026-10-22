# Numerical completion release measurements — 2026-10-22

Baseline native code: d99a09c9. Compared change: dense GraphMatrix products
now use the shared checked lower matmul; logical reduction buffers use checked
bulk materialization; vector/matrix norms use a scaled accurate sum of squares
instead of a sequential hypot fold. No precision mode/backend replacement.
A separate shape check preserves the existing incompatible-dimensions error.

Run:

```
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target \
  GROGGY_NUMERICAL_PROFILE=/tmp/numerical.json \
  cargo bench --offline --locked --bench numerical_completion
```

Twenty seeded release workloads on macOS/aarch64; one warmup and seven measured
samples per case. The custom System allocator counts allocations/reallocations,
requested bytes and maximum additional live heap bytes per operation. Inputs
are prepared outside measurements; output construction/drop and result checksum
are included. Allocation instrumentation affects timings. This is heap request
accounting, not RSS, stack usage or a certified whole-process memory budget.
The benchmark runner is sequential; allocator counters cover the process.

| Workload | Before ms | After ms | Ratio before/after |
| --- | ---: | ---: | ---: |
| 64×64 fractional product | 9.197 | 4.982 | 1.85 |
| 64×64 integer product | 7.936 | 0.866 | 9.16 |
| 256×256 Frobenius norm | 0.987 | 0.704 | 1.40 |
| 256×256 numerical-data roundtrip + sum | 1.473 | 0.731 | 2.01 |

The bulk product copies logical input buffers, increasing measured additional
heap peak from 67,950 to 98,656 bytes in the 64×64 fixtures. Norms retain their
524,288-byte logical copy; scaled normalized products are accurately accumulated
before square root/rescaling. CSR cases retain sparse storage. The 10,000-square
CSR solve fixture peaks at about 6MB additional heap, including factor scratch,
dense RHS/output, labels and checksum.

Unchanged workloads vary about 2–10% between the retained before/after runs;
there is no claim that all cases got faster. An initial after run overlapped
compilation and distorted some timings substantially (CSR solve 46ms vs roughly
16ms without compilation). That raw receipt is retained separately and excluded
from the comparison above. The after comparison was rerun from the already-built
binary with no Cargo/rustc jobs running. Future comparisons should run on an idle
machine and inspect raw samples, not assert CI wall-clock thresholds.

Raw receipts live in `benches/numerical_completion_{before,after}_2026-10-22.json`
and `benches/numerical_completion_after_contended_2026-10-22.json`. All 20 checksums
are finite and agree before/after to 1e-13 relative/absolute tolerance.
The corpus also covers dense axis statistics, typed vector arithmetic/dots,
CSR products/statistics/transforms/factors/solve, QR/SVD/eigen, integer population
statistics and owned thread-transfer costs.

GIL worker-progress tests cover the behavior independently. This corpus does not
certify SIMD coverage, every embedding dimension, billion-entity graph workloads,
universal autodiff, or a 100ns Python FFI budget. Broader size sweeps, bare timing
without allocation instrumentation, sparse fill-in and supported derivative
workload profiles remain gates for the broad production-performance parent.

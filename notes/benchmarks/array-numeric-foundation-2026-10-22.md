# Shared array/numeric foundation checkpoint — 2026-10-22

AR7.3 moves scalar types/conversion/reductions/accurate accumulation into a common
numeric module, preserving old imports. Matrix error adapters delegate to those
kernels. Legacy BaseArray uses the canonical shared column and copy-on-write.

Rebuild and idle-rerun the existing 25-case optimized corpus with its allocator
instrumentation. All seven samples in all workloads agree with the b19aa9b7
baseline checksums at 1e-13 relative/absolute tolerance. Receipt:
`benches/numerical_array_foundation_2026-10-22.json`.

Observed medians: 64x64 fractional product 4.536ms (baseline 4.531ms), integer
product 0.289ms (0.868ms), 100k fractional dot 0.627ms (0.644ms), and 10k-square
CSR solve 17.114ms (15.277ms). This is a compatibility/performance observation,
not a claim that every case improves. Source reorganization and compiler choices
can change optimization; ordinary machine noise remains. Heap figures retain
the process-allocation/additional-live-heap scope, not RSS or stack.

Full native suites, explicit storage/type/selection/ID compatibility fixtures,
focused Clippy and 455 rebuilt Python regressions passed. Broad size/fill/dispatch
and final strict gates remain work in the numerical completion pass.

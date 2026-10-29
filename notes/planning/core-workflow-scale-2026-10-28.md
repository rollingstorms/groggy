# Core workflow scale baseline — 2026-10-28

Existing 32.1/32.2 work, using implemented core interfaces after `f9975e6c`.
No new requirements/checklist rows; no external graph libraries or packages.

`benches/core_workflow_scale.rs` accepts a node count divisible by 100 and at
least 1,000. It creates a directed ring with the same number of edges in one
accepted batch at effective time 10, using one repeating i64 bucket property plus
normal entity_type. It captures a GraphView, explicitly builds a transient bucket
index, compares scan/indexed ten-row pages to arithmetic expected IDs, checks
three-hop induced selections, accepts one property correction at time 20, and
reconstructs the original value at time 15. It then releases read/index owners,
saves a complete retained archive, drops the original graph and restores it.
Recovery checks head/counts/wraparound edge, distributed property samples, and
historical reconstruction. Temporary archive files are removed after each run.

Each size runs in a separate process. There is one full workflow observation per
size, not statistical evidence for the ingest/recovery stages. Indexed page and
neighborhood timings are medians of five batches of 100 operations. OS peak RSS
includes setup, every stage, allocator retention and macOS memory-pressure effects;
it is not a direct logical-heap or incremental capture measurement. Wall/user/
system times are process totals from `/usr/bin/time -l`. Local filesystem/cache,
degree distribution, two accepted versions and small scalar values limit scope.

Raw evidence: `notes/benchmarks/core-workflow-scale-2026-10-28.json`.

| Nodes / edges | Bulk accept | Cold capture | Index build | Indexed 10-row page | 3-hop neighborhood | Effective read | Archive save / recovery | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 / 1,000 | 4.278 ms | 0.792 ms | 0.191 ms | 1.513 µs | 2.323 µs | 2.171 ms | 26.194 / 5.865 ms | 20.766 MiB |
| 100,000 / 100,000 | 0.600 s | 0.143 s | 0.051 s | 10.117 µs | 2.092 µs | 0.291 s | 1.336 / 0.831 s | 1,203.469 MiB |
| 1,000,000 / 1,000,000 | 8.257 s | 2.240 s | 0.751 s | 75.618 µs | 1.951 µs | 5.908 s | 26.669 / 14.599 s | 4,076.422 MiB |

The million fixture completed in 66.28 s wall / 50.00 s user / 10.71 s system,
with a 290,157,519-byte archive. Property correction measured 1.050 ms; the old
capture remained unchanged. Scan/page measured 340.210 ms versus 75.618 µs for
the explicitly indexed page. Shared current capture already has separate evidence;
this fixture measures its first cold capture. Neighborhood selection stays bounded
on this low-degree shape; it does not certify hub-heavy traversal budgets.

## Concrete follow-up

Archive save is a measured bottleneck. Inspection found simultaneous payload and
framed-byte buffers, plus an owned archive document kept alive during a complete
round-trip restore validation. Reduce those allocation lifetimes while retaining
the existing format, full prepublication validation, corruption rejection and
atomic no-clobber file publication. Compare matched release executables before/
after; record actual outcomes rather than attributing unrelated timing noise.

32.2 is partial: these three fixtures cover a complete bounded core workflow,
not billion-scale operation, deep/wide history, varied properties/degree shapes,
concurrent services or journal throughput. Existing journal/recovery receipts
retain their own scopes. No universal latency/memory budget has been chosen.

The unchanged core passed **1,442 native / 56 ignored**, **2,314 root Python /
1 skipped**, **227 docs** before this benchmark-only addition. The new release
benchmark assertions pass at all three sizes and strict all-target core Clippy
includes the new target. The owner’s separate benchmark/review files are excluded.
Reproduce with offline locked release `cargo bench --bench core_workflow_scale
--no-run`, then run its emitted executable with 1000, 100000 or 1000000.

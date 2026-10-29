# Sparse frozen adjacency — 2026-10-28

Existing AR7.5/03.2/32.2 capture component; no new checklist row.
This follows `current-read-sharing-2026-10-28.md`: a shared read avoids repeated
copies, but cold/current/historical captures still construct frozen adjacency.

## Implemented

PinnedReadState no longer inserts empty adjacency and incidence entries for every
active node. Only actual outgoing/undirected and incident rows are stored.
Immutable empty vectors serve active isolates and directed sinks; the existing
ExistenceIndex distinguishes active empty rows from absent IDs. Public return
types, ordered neighbor/edge alignment, parallel occurrences, direction and loop
rules stay the same. No FFI/Python business logic or alternate read owner.

Capacity reservations use bounded incidence counts to avoid repeated hash-table
growth on connected inputs. They are capacity bounds, not inserted empty rows.
No raw ID-sized array is allocated; sparse/arbitrarily large IDs retain the same
membership semantics. The same ReadData construction serves current and retained
captures and remains independent of later mutation, pool maintenance and owner
lifetime. Weak current capture caching and separate capture identity are retained.

One added native test compares all three read APIs against explicit endpoint
incidences across 1,000-node directed/undirected fixtures, parallel edges, a loop,
sinks, isolates, arbitrary absent IDs and node removal while an old pin survives.
The existing full suite covers the shared query/table/algorithm/temporal paths.
Guide example: `docs/api/current-read-sharing.md`.

## Matched release evidence

`benches/pinned_read_adjacency.rs` runs a single 100k-node fixture per process.
The baseline executable comes from commit `98004d6d`; the changed executable uses
the same benchmark source/toolchain. Two processes per fixture/variant reverse
execution order. Each process performs five cold captures, releasing the reader
between them, followed by five whole-node row-probe passes. Each probe reads
neighbors, aligned edge IDs and incident IDs and validates total counts. Values
and fixture construction are excluded from capture timings, but OS peak process
RSS includes construction, all capture passes and allocator-retained memory.

Raw evidence: `notes/benchmarks/pinned-read-adjacency-2026-10-28.json`.
Medians below combine ten timing samples and two peak-RSS observations per variant.
These local bounded fixtures do not establish a production budget or arbitrary
degree/payload/history performance.

| Directed fixture | Edges | Cold before → after | Process peak before → after | Three-row probe before → after |
| --- | --- | --- | --- | --- |
| Isolates | 0 | 50.996 → 38.198 ms | 165.016 → 154.078 MiB | 132.555 → 47.397 ns/node |
| Sparse chain plus isolates | 1,000 | 51.064 → 38.658 ms | 165.375 → 154.477 MiB | 132.220 → 93.548 ns/node |
| Chain | 99,999 | 118.912 → 104.590 ms | 196.742 → 196.875 MiB | 129.450 → 124.695 ns/node |
| Outgoing star | 99,999 | 86.249 → 70.955 ms | 188.789 → 181.820 MiB | 114.900 → 145.576 ns/node |

The star probe is **26.7% slower**: most nodes have incoming edges but no outgoing
row, so the sparse representation checks membership after an adjacency miss.
This is a stated trade-off for lower sparse capture allocation and faster cold
construction, not a claim that every read is faster. Chain peak RSS is effectively
unchanged in this process-level measurement. Wider algorithm/workflow and scale
budgets remain 28.16/32; future tuning should use representative whole workloads.

## Gates and remaining work

Final gates: **1,442 native passed / 56 ignored**, strict all-target core
Clippy, **2,314 root Python passed / 1 skipped**, **227 executable docs**.
The CPython 3.9 full-ABI release extension was rebuilt and atomically staged in
tree; three unrelated FFI warnings remain. Commands use offline locked Cargo with
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`, full cargo test,
strict all-target Clippy, release FFI build and the benchmark target. Scoped Rust
formatting and diff checks pass. Owner benchmark/review files remain excluded.
The first capture still copies all logical values and stores nonempty per-node
vectors; dense/sparse workload choice, full memory targets and million/billion
scale certification remain open. Default connected Subgraph ownership remains
26.1/AR4/AR7.5; this slice changes only the canonical frozen read data.
Fixed checklist remains **211 checked / 57 unchecked**.

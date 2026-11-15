# Projected pending frame materialization — 2026-11-15

## Completed optimization

The pending batch retains one immutable accepted source, prepares the root's
membership/endpoints once per structural revision and materializes each exact
frame directly. Only selected property values are copied. Staged properties retain
column-first storage; frame reads look up selected rows in each staged column.

New property assignments mark target IDs for membership validation. Topology
changes invalidate that validation and recheck every staged target, including
siblings outside the current frame. Geometry plans append only; their lengths
identify structural revisions. Every capture still validates the owner's write
coordinate. Cache state belongs to one atomic batch and drops with it. Older
issued captures remain immutable and their table-attachment authority stays weak.
No writable Graph/history clone, per-child publication or FFI computation is added.

## Matched native measurements

Three fresh processes per revision, five observations per case, same fixture and
System allocator counters. Baseline production source: `808893c4`, with the shared
fixture/allocation harness applied before production changes. After receipts use
this implementation. Rust 1.96.1, native release on this macOS arm64 workstation.
Builds and fixtures are excluded; execute/materialization/acceptance are included.
Reported values are medians of the three process medians. Requested bytes count
allocation/reallocation requests, not RSS, allocator overhead or retained memory.
Timing includes counter overhead. Earlier development observations are excluded.

| Initial nodes | Staged node | Before ms | After ms | Before requested bytes | After requested bytes | Before peak extra bytes | After peak extra bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | No | 0.406 | 0.420 | 582,980 | 587,445 | 158,204 | 161,920 |
| 100 | Yes | 3.526 | 0.541 | 5,969,824 | 772,779 | 221,688 | 181,520 |
| 1,000 | No | 4.242 | 4.320 | 5,965,616 | 6,035,393 | 1,565,784 | 1,619,420 |
| 1,000 | Yes | 364.614 | 5.715 | 638,093,178 | 7,691,619 | 2,089,920 | 1,612,284 |

For the 1,000-node staged-topology fixture: roughly 64× lower median time,
98.8% less requested allocation volume and 22.9% lower peak additional requested
bytes. Property-only fixtures incur a small topology-cache allocation cost (about
1.2% more requested bytes at 1,000 nodes); their timing differences are small and
variable. This is evidence for these node-only workloads, not a universal scale,
memory or throughput guarantee. Overlapping/wide frames still copy their values.

Receipts: `notes/benchmarks/child-frame-allocations-{before,after}-2026-11-15-{1,2,3}.json`.
Reproduce with `cargo bench --offline --locked --bench pending_frame_allocations`
and `GROGGY_ATOMIC_BUILDER_PROFILE=<path>`. Ordinary uninstrumented timing uses
`atomic_builder_properties --child-frames`; both binaries share the same fixture.

## Qualification

Locked/offline native suite: **1,599 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, formatting and whitespace
checks pass. The in-tree release extension was rebuilt without installing a wheel;
three existing binding warnings remain. Focused Python: **118 passed**.
Full maintained Python: **2,980 passed / one skipped / 14 established warnings**.
All **279 executable documentation examples** pass. Black/isort/pre-commit remain
unavailable; no external packages were installed.

The new native regressions cover sibling-target revalidation after removal,
endpoint rescue/cascade refresh, immutable earlier frames, dirty foreign targets
with unchanged topology and unchanged accepted state on rejection. The existing
pending frame, table authority, source/selection, nested execution and topology
suites remain maintained. An initial test used a nonexistent insert_edges helper;
it was corrected to the existing add_edges API before qualification.

## Existing roadmap

Advances 09.4b.2, 03.2/AR2, 30.1, AR7.8 and measured performance evidence for 32.3.
The fixed inventory remains **215 checked / 53 unchecked**, with **44 unchecked
release rows in scope**. Broad remaining execution/lowering/scale requirements are
not closed by a workload optimization. No rows added. Owner files are untouched.

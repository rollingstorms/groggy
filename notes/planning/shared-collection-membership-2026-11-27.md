# Shared collection membership and empty frames — 2026-11-27

Finite performance/ownership follow-up within 30.1, AR4, AR7.8, AR7.11 and 32.3.
No new stable row; checklist counts remain 217 checked / 51 open.

- [x] Share immutable selected membership in memory and portable collection adapters.
- [x] Validate all empty-body foreach inputs, then preserve original pins without constructing child frames.
- [x] Requalify existing semantics and measure matched qualified artifacts.

`SelectionPart` shares node/edge membership Arcs from its original subgraph/view.
Derived portable values retain those same immutable sets; owner-connected memory
pass-through still retains its complete original subgraph identity/caches. A
body that performs work still creates its checked fixed child frame. An empty
body cannot observe working values, so after source/membership validation it
preserves the original selection without projecting a child snapshot. Cancellation,
invalid owners, stale IDs, endpoint closure and old target bindings keep their
existing checks. No precision, timing, mutation, history or Python API policy changes.

Local source `57cf012911bc4e0f8040d6b30bba85a2db286413` passes **1,714 native tests**
(54 ignored), **3,706 packaged Python tests** (one skipped, 23 expected warnings)
and **306 documentation examples**, plus strict all-target Clippy and formatting.
The 175 native step tests and 32 independent collection worker geometry cases
remain passing. All 500 production sources match the runtime; all 62 packaged
Python/stub bytes match and Python 3.8 syntax checks pass. See the
[local receipt](../benchmarks/shared-collection-membership-local-qualification-2026-11-27.json)
and [wheel receipt](../benchmarks/shared-collection-membership-wheel-qualification-2026-11-27.json).

## Matched costs

Both qualified wheels use the same benchmark script and environment. Only three
production files differ from baseline `5f19e15e`: selected membership access,
collection adapters and the empty foreach body path. Fresh graph construction
stays outside operation timing. Small and larger corpora each use ABBA order,
three warmups per process, and respectively 22 or six samples per case/runtime.

| Nodes | Workload | Before ms | After ms | Wall change | CPU change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | per_node_passthrough | 0.368 | 0.263 | -28.5% | -28.8% |
| 100 | overlapping_edges_write | 0.701 | 0.701 | -0.1% | -0.2% |
| 1,000 | per_node_passthrough | 3.439 | 2.334 | -32.1% | -32.1% |
| 1,000 | overlapping_edges_write | 7.392 | 7.597 | +2.8% | +2.8% |
| 10,000 | per_node_passthrough | 38.183 | 33.405 | -12.5% | -12.5% |
| 10,000 | overlapping_edges_write | 79.072 | 80.533 | +1.8% | +2.0% |
| 100,000 | per_node_passthrough | 540.207 | 479.864 | -11.2% | -11.1% |
| 100,000 | overlapping_edges_write | 1089.734 | 1088.114 | -0.1% | +0.2% |

Read-only medians improve 11–32% in these fixtures. Nonempty write medians range
from about −0.2% to +2.8%; this is not evidence of a general write speedup. The
initial portable implementation's small readonly overhead motivated this change;
its [earlier measurements](../benchmarks/read-collections-performance-2026-11-27.json)
remain recorded at their original source rather than being overwritten.

Whole-process macOS peak RSS for two fresh processes per runtime is **788–792 MiB
before, 734–739 MiB after**. This includes wheel unpacking/imports, graph setup,
warmups and both 10k/100k workloads. It is not a per-operation allocation/RSS
measurement. OS peak memory-footprint readings vary more and are retained in the
raw reports. No production budget, allocator count, broad topology distribution,
million/billion-scale or universal latency guarantee is established. All raw
samples, process timing output, wheel hashes and source/script pins are in the
[performance receipt](../benchmarks/shared-collection-membership-performance-2026-11-27.json).

Manual cross-platform qualification was dispatched for this exact code source as
[run 46078071480](https://github.com/rollingstorms/groggy/actions/runs/46078071480).
Its final result and receipts are recorded separately when complete.

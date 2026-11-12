# Bulk captured collection projection — 2026-11-11

Closes the specific many-small-group throughput regression recorded in
`subgraph-array-projection-2026-11-05.md`, within existing **32.2 / AR7.5**.
This is evidence under the existing inventory, not a new checklist row or a
claim that the broad scale/release gates are complete.

## Evidence and implementation

The reproduced release baseline for 100k nodes in 100 groups was 49.426 ms for
two-column tables and 36.039 ms for present observations (median of five).
A native comparison changes only group scheduling, keeping the column reader
unchanged and excluding Python/GIL transitions: 100 groups take 32.599 ms
sequentially versus 4.727 ms across eight existing Rayon workers. Sorting those
groups alone takes 1.064 ms. This establishes group scheduling as a substantial
cost in this fixture; it does not isolate every property/hash/cache cost.

Native `GraphView::project_node_columns` and
`GraphView::project_attribute_observations` now own collection projection.
Enough groups/rows use the existing worker pool; one/few selections retain the
canonical per-column row-parallel reader. The adapter captures sources under
the GIL, makes one GIL-free native call, then wraps owned results. No graph owner
or Python object crosses into workers. Group order and ascending original IDs
are unchanged. Null, absence, exact types, node-first/edge fallback, unknown
attribute errors and independent captures retain their previous contracts.
No new persistent sorted-membership or column cache is introduced.

Quiet rebuilt-extension measurements, five samples, fixture creation and output
validation excluded, milliseconds of median wall time:

| Nodes | Groups | Tables before → after | Observations before → after |
|---:|---:|---:|---:|
| 1,000 | 1 | 0.126 → 0.155 | 0.075 → 0.070 |
| 100,000 | 1 | 11.641 → 11.997 | 7.879 → 7.665 |
| 100,000 | 4 | 11.065 → 10.903 | 8.703 → 7.973 |
| 100,000 | 100 | 49.426 → 9.837 | 36.039 → 6.282 |
| 100,000 | 1,000 | 56.646 → 14.664 | 36.312 → 7.040 |

The intended 100-group case improves about 5×/5.7× and is faster than the old
incorrect current-owner reader's historical 50.646/14.071 ms observation.
One-group differences are small and mixed; no uniform speedup is claimed.
Parallel reads use more process CPU than wall time. Raw receipts retain both;
there is no allocation/RSS or production-throughput claim. Source snapshots
still retain their full captured state and owned outputs scale with output size.

Reproduction:

- `cargo bench --offline --locked --bench captured_view_entities` includes the
  isolated native group scheduling comparison and sort-only measurement.
- `benches/subgraph_array_projection.py` measures the Python public workflow.
- `notes/benchmarks/subgraph-array-bulk-{before,after,profile}-2026-11-11.json`
  retain raw samples and fixture descriptions.

## Completion boundary

Native regressions compare grouped outputs against canonical scalar readers
after owner changes/removal, including empty groups, independent source states,
duplicate/missing names, explicit null and edge fallback. Python coverage adds
64 captured groups, row ordering, exact cells and a missing attribute in the
last group: no partial result or publication is returned.

Broader ingestion/history/storage/memory/scale work remains in 32.2. Typed
graph/table pipeline and topology effects remain in 30.1/AR7.8/09.4b.2.
Verification: **1,537 native passed / 54 ignored**, **2,594 maintained root
Python passed / 1 skipped**, **258 executable documentation examples**. The
release extension was rebuilt in tree; its three existing FFI warnings remain.
Strict root all-target Clippy and final whitespace checks are recorded in the
slice ledger. No external packages or wheel installation were used.

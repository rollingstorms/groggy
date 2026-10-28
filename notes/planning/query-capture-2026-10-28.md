# Canonical immutable query entry point — existing 21.2 component

Replaced native `Graph.query()`'s unimplemented placeholder with the existing
immutable GraphView capture. Graph, DurableGraph and connected Subgraph expose
this entry point in Python; connected selections explicitly freeze current
values and reject stale membership. No new query object, parser or execution
engine is introduced. Native callers previously received Result<()> containing
an unimplemented error; they now receive Result<GraphView>.

GraphView.query_nodes/query_edges add owned ordered/paged ID results through the
same column predicate executor and scalar ordering/page kernel used by explicit
GraphQueryIndex. Scan queries do not implicitly build/repair indexes. Native
QueryOptions and one shared FFI options marshaller preserve strict pagination,
exact large numeric comparisons, missing/Null ordering, deterministic ties and
validation of every matching value even for limit=0. Filtering/traversal and
selected graph/table results preserve immutable pins. ID-list ordering is not
silently converted into ordered graph membership.

Three native integration tests and twenty Python cases cover independent expected
pages, scan/index agreement, selected parallel/loop edge scope, empty/overflow
pages, exact i64/f64 comparison, invalid scalar domains, mutation/deactivation,
stale connected selections, owner closure, threads, explicit effective/recorded
cutoffs, durable bytes/head preservation/reopen and strict controls. Guide:
`docs/api/query-capture.md` includes four executable blocks.

Final gates: **1,431 native passed / 56 ignored**, **2,305 root Python passed /
1 skipped**, **222 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The CPython 3.9
full-ABI release extension was rebuilt and atomically staged in tree; three
pre-existing FFI warnings remain. Root Python verification excludes the owner's
separate benchmark test and nested historical surveys. No external packages or
graph libraries were installed/used. Owner notebook/benchmark/review work is
excluded from this commit.

Commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target` with
offline locked cargo test, focused `--test query_capture`, strict all-target
Clippy and `cargo bench --bench query_capture`; Python uses the rebuilt extension,
focused/root contract suites and executable docs runner.

## Bounded release costs

`notes/benchmarks/query-capture-2026-10-28.json`: one bulk-created score column,
largest 100 values matching, ten-row descending page. Every result is checked
against analytic IDs. Five alternating samples use ten queries each. Fixture
creation is excluded; root capture and index build are measured separately.

| Nodes | Capture | Index build after capture | Scan/page median | Indexed/page median |
| --- | --- | --- | --- | --- |
| 1,000 | 0.474 ms | 0.099 ms | 72.525 µs | 7.479 µs |
| 100,000 | 138.665 ms | 52.162 ms | 33.357 ms | 9.233 µs |

These are local native release results, without Python marshalling or RSS. Owner
capture materializes current read state and is not O(1); query reuse avoids
recapturing when a fixed state is intended. Selective explicit index lookup avoids
rescanning but has build and retention costs. This entry point reuses the existing
capture implementation; it adds no automatic mutation work and makes no production
latency claim. Larger capture/storage/scale gates retain their existing scope.

21.2 remains partial: legacy filter APIs still return connected Subgraph and their
full immutable migration remains tied to 26.1/AR4/AR7.5. During inspection, shared
SubgraphOperations property getters/bulk setters and the Context property overlay
also showed owner-only rather than selected-membership checks. That concrete scope
gap is follow-up work within the existing 26.1/09.4b.2 boundaries; it is not fixed
by this query entry point. Compound topology/custom effects remain
09.4b.2/AR3/AR7.8. No new checklist rows or parent closure:
**211 checked / 57 unchecked**.

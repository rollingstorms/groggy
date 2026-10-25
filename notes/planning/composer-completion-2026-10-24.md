# Composer completion — 2026-10-24

Closes existing checklist **09.4b.2b**. Owner decision: contraction creates one
edge per reachable external endpoint pair. No simple-path enumeration.

## Contract

Preset defaults apply first; explicit options and aggregation overrides follow.
Canonical `entity_type=meta` remains stable; semantic kinds are text-vector labels.
Explicit meta-target selectors validate active external meta identities before
mutation. Ordinary external endpoints remain eligible. Components/subgraph arrays
retain their existing per-selection publication boundaries.

Contraction honors precisely the selected internal edge set, directedness and
external reachability, including disconnected interiors, loops and cycles. Each
distinct original edge in the source-to-target reachability corridor contributes
once, including cyclic detours and parallel instances. Original edge-ID order
defines First/Last; edge_count counts contributing edges. Existing direct external
edges remain distinct. The summary meta-node is isolated. Collapse removes selected
originals; Extract retains them. Reducers describe columns, not flow-equivalent
networks. Worst-case output is quadratic in external endpoints; graph traversal
per source/target pair remains a documented cost.

## Evidence

Six native regressions include an independent Floyd reachability oracle over
2,048 directed and 64 undirected small graphs, exact corridor counts and weights,
selected-edge exclusion, preset precedence and rejection without mutation.
Twenty-one Python cases plus 12 existing atomic-collapse cases pass. Full gates:
1,006 native passed / 52 ignored; 1,716 Python passed / 1 skipped; 84 executable
documentation examples; strict all-target Clippy. In-tree FFI build succeeded.

## Release observations

`benches/composer_completion.rs`, five observations per case. Fixture/setup and
validation excluded; measured execution includes aggregations, group retention,
source removal and one accepted publication. Median milliseconds:

| Mode | 1,000 nodes | 10,000 nodes |
| --- | ---: | ---: |
| preset | 2.3763 | 103.6460 |
| selector_include | 1.7114 | 104.4105 |
| selector_empty | 1.5242 | 101.5752 |
| contract_directed | 3.7222 | 208.7543 |
| contract_undirected | 3.8566 | 210.1142 |

Receipt: `../benchmarks/composer-completion-2026-10-24.json`. These bounded chain
fixtures do not establish general scale or memory guarantees. The nonlinear
growth exposed repeated retained-edge scans in per-node bulk removal. A matched
after measurement will evaluate that existing bulk/performance requirement.

Pinned/portable meta payloads, complete native mutation migration and compound
pipeline boundaries remain open in their existing checklist entries.

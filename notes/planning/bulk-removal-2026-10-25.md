# Bulk cascade repair — 2026-10-25

Existing scope: 05.3, 09.4b.2 and performance gates 32. No new checkbox row.

## Change

Both node-removal execution and publication-clock preparation previously called
incident-edge discovery separately for each selected node, scanning the retained
edge pool repeatedly. They now share one active-edge scan per phase. Each original
edge instance is considered once, independent of retained tombstones. Bulk space
removal touches requested IDs directly, avoiding whole-active-set retain passes.
Single-node removal delegates to the same prepared cascade.

Implicit cascades keep the earliest removed endpoint effective time. Explicit edge
removal overrides still win. All lookups/target validation precede cascade writes;
shared edges, parallel instances and loops are recorded once. This preserves the
existing publication/manual compatibility boundary and immutable property history.

CSR snapshots include isolated node IDs, so node membership changes now invalidate
the topology cache even without an edge change. Existing Arc arrays remain frozen.
Empty removals do not invalidate caches or publish states. Internal cache versions
are not accepted graph state IDs and need not advance once per individual edge.

## Matched release observations

Same optimized `composer_completion` benchmark, five observations per case. Fixture
creation/configuration/verification excluded. Includes the complete compose plan,
retained group, aggregations, source removal and one publication. Before receipt:
`../benchmarks/composer-completion-2026-10-24.json` (f3219ffb); after receipt:
`../benchmarks/bulk-removal-after-2026-10-25.json`. Medians in milliseconds:

| Nodes | Mode | Before | After | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | preset | 2.3763 | 0.7968 | 2.98x |
| 1,000 | selector_include | 1.7114 | 0.6952 | 2.46x |
| 1,000 | selector_empty | 1.5242 | 0.5969 | 2.55x |
| 1,000 | contract_directed | 3.7222 | 1.8260 | 2.04x |
| 1,000 | contract_undirected | 3.8566 | 1.9465 | 1.98x |
| 10,000 | preset | 103.6460 | 6.7992 | 15.24x |
| 10,000 | selector_include | 104.4105 | 6.7509 | 15.47x |
| 10,000 | selector_empty | 101.5752 | 6.7333 | 15.09x |
| 10,000 | contract_directed | 208.7543 | 19.6692 | 10.61x |
| 10,000 | contract_undirected | 210.1142 | 20.9246 | 10.04x |

These local runs were sequential on the same machine/build profile, not alternating
statistical trials. Ratios describe these bounded chain/port fixtures only. No RSS,
allocation, FFI latency, dense-boundary or billion-node guarantee is established.
The benchmark's receipt scope string remains unchanged to preserve fixture parity.

## Verification

- `cargo test --offline --locked --quiet`: 1,010 passed / 52 ignored.
- Four new native regressions cover manual/publishing modes, both directions,
  parallel/loop/shared cascades, history, untouched edges, mixed explicit clocks,
  invalid requests, empty operations and isolated-node cache invalidation.
- In-tree FFI build; root Python suite: 1,720 passed / 1 skipped, including four
  new cross-language cascade/history cases. Temporary Python sources refreshed
  after missing staging files caused collection errors; final gate passed.
- 84 executable documentation examples; strict all-target Clippy; diff checks.

The original broad mutation migration, cache/view migration and scale gates remain
open. This is a measured repair within those requirements.

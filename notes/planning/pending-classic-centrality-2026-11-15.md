# Pending classic centrality — 2026-11-15

## Completed family

The three classic registry wrappers (PageRank, closeness, betweenness) now use
AlgorithmFrame and the shared bulk endpoint reader. They consume pending selected
IDs, revised endpoints, direction and staged columns after supported topology
builders, return the original live continuation, and stage results into the outer
batch. The coordinator remains the sole publication boundary. No new kernel,
FFI business logic, persistence format or public algorithm option was introduced.

Ordinary live/property-only calls retain their existing preparation and CSR cache
path. Weighted betweenness also reads per-edge endpoints from the selected source
when retaining equal minimum-length parallel instances. Incoming closeness and
strictly positive weighted lengths retain their prior semantics. PageRank keeps
its optional personalization column and existing unweighted edge multiplicity.

## Verification

- The native classic pipeline regression now covers all three centralities after
  warming the old live CSR, then staging node/edge creation, endpoint revision,
  cascade removal and weights. A triangle provides finite independent expected
  values; later builder reads must match and there is one publication.
- 42 new Python cases compare pending centrality with separately accepted selected
  execution across direction, parallel edges, loops, sparse IDs, provisional rows,
  weighted and incoming closeness, personalization and normalization options.
  They also test empty selections, invalid staged inputs and late managed-property
  failures that must release provisional allocations and discard all changes.
- Focused Python centrality/path/classic integration gate: **79 passed**.
- Full locked/offline native suite: **1,604 passed / 54 ignored / zero failed**,
  223 suite results. Strict all-target Clippy, Rust formatting and whitespace pass.
- Rebuilt in-tree release extension: **3,131 maintained Python passed / one
  skipped / 14 established warnings**, **284 executable documentation examples**.
  No package install, new artifact-platform qualification or publication.

The first focused test run exposed a test harness mistake: RustAlgorithmHandle
is passed to Pipeline, rather than called directly. Correcting the reference
execution fixed that harness error. The consecutive reference also explicitly
refreshes immutable selection after publication. The old preflight guard test now
uses a deliberately uncertified custom algorithm instead of newly supported
PageRank, retaining the guard coverage as built-in support grows. No production
fallback was added.

## Checklist and remaining work

Advances existing 09.4b.2, 03.2/AR2, 30.1 and AR7.8, with conformance under
01.2/19.2/28.16. The fixed inventory remains 215 checked / 53 unchecked (44 open
release rows, nine deferred). Six classic community wrappers remain to adapt;
general topology-mutating loops, unsupported custom effects and alternate plans
remain guarded. Wide pending captures still copy selected logical properties;
this functional adaptation makes no new throughput or memory guarantee.

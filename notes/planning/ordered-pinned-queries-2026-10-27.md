# Ordered/paged captured queries and bounded traversal — 21.1

This completes three components of the existing native query work: deterministic
ordering, pagination and bounded directional traversal. It adds no checklist rows
and does not claim arbitrary pattern matching or ordinary connected Subgraph
migration is complete. The full-vision inventory remains 209 checked / 59 open.

## Native contract and architecture

- `query::QueryOptions` describes a scalar property or ID ordering, direction,
  offset and optional limit. `GraphQueryIndex::query_nodes/query_edges` reuse the
  existing captured filter executor, validate requested index availability, then
  share one ordering/pagination implementation. Existing find/filter APIs retain
  their behavior. Returned lists preserve query order; GraphView remains a set.
- One comparable scalar domain is required among matching rows. Numeric ordering
  shares the canonical exact i64/f32/f64 comparator; text aliases, Boolean and
  native temporal values are supported. Null/absence sort last in both directions;
  property ties use ascending ID. NaN, mixed unsupported domains and non-scalars
  reject before results, including for limit zero or offsets past the end.
- Ordering/filter indexes must exist unless scanning is explicitly permitted.
  An ordering column cannot silently trigger a scan for an empty query. Counts
  use strict Python integer conversion and overflow-safe native page bounds.
- After complete scalar validation, property pages partition to the requested
  rank interval and sort only that page with the same total comparator. This
  preserves full-sort results while avoiding sorting discarded rows. All-match
  key/filter memory and validation remain; no streaming/cursor claim is made.
- `GraphView::within_hops` and `query::TraversalDirection` perform iterative
  multi-source captured BFS. Outgoing/incoming/both orientation is explicit;
  undirected input follows either endpoint. Seeds validate completely, duplicate
  seeds coalesce, zero/empty/very large bounds are safe, and cycles/loops/parallel
  edges do not repeatedly visit nodes. Induced results keep all parent-selected
  internal edge instances, not only traversal-tree edges.
- Original source/time pins survive nested selection, owner mutation/closure and
  threads. Reads publish no changes and rebuild no indexes. FFI performs only
  type/filter/error translation and releases the GIL over immutable native data.
  Python type stubs and the API guide document ordered IDs versus graph membership.

API examples: `docs/api/ordered-pinned-queries.md` (five executable blocks).

## Verification

Five native integration tests cover exact mixed numeric ranking, both directions,
property ties, explicit Null/absence, scalar domains and aliases, empty/error
pages, extreme offsets/limits, missing index errors and explicit scan permission.
Independent all-pairs distance reference checks every three-node topology in both
graph directions with full, node-selected and edge-selected scope, all three
traversal modes, each selected start and bounds 0–4. Additional fixtures cover
parallel loops, isolates, multi-source seeds, invalid seeds, maximum hop bounds,
effective captures, owner mutation/drop and unchanged publication state.

Twenty new Python cases cover the public signatures and supported scalar domains,
query ordering/ties/pages, control conversion, selected edge closure, effective
provenance and shared-thread lifetime. A 100k-row shuffled-property release fixture
checks the page against an independently sorted integer-key reference.

Final gates: **1,404 native tests passed / 56 ignored**, **2,089 root Python
passed / 1 skipped**, **195 executable documentation examples**, and strict
all-target core Clippy. Scoped Rust formatting, Python syntax and diff checks pass.
The final in-tree release extension was rebuilt and atomically staged after the
partition optimization; three existing unrelated FFI warnings remain. No external
packages or graph libraries were installed/used. The separate owner algorithm
benchmark and notebook/review work were excluded from staging.

Native commands used `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`
with `cargo test --offline --locked`, focused `--test ordered_pinned_queries`, and
`cargo clippy --offline --locked --all-targets -- -D warnings`. Python verification
uses root `tests/test_*.py` excluding the owner's `test_algorithm_benchmark.py`,
followed by the documentation example runner. The existing verified CPython 3.9
full-ABI release build profile was used; wider Python/platform certification is
unchanged.

## Bounded release evidence

Reproducer: `cargo bench --offline --locked --bench ordered_queries` with the
same target environment. Snapshot capture and property-index build are reported
separately from queries. Raw five-sample before/after receipts are in
`../benchmarks/ordered-pinned-queries-2026-10-27.json`. The reference version used
complete sorting by the identical scalar/ID comparator followed by page slicing;
the new version partitions and sorts only the selected rank interval. Both check
returned IDs against independently sorted shuffled integer keys.

| Fixture / query | Full-sort reference median | Partitioned page median |
| --- | ---: | ---: |
| 1,000 nodes, 100-row page after offset 10 | 166.6 microseconds | 61.2 microseconds |
| 100,000 nodes, 100-row page after offset 10 | 38.25 milliseconds | 28.02 milliseconds |

The 100k-node directed-chain fixture returned a two-hop induced view in a median
1.34 microseconds and a hundred-hop induced view in 60.64 microseconds in the final
run. These are absolute bounded observations; the traversal implementation did
not change between before/after runs, so differences between its samples are not
a claimed optimization. Runs were sequential rather than alternating control
binaries, and no RSS, universal throughput or production target is certified.

Pagination still filters and validates all matching rows and allocates per-match
keys; it is not a bounded-memory streaming or cursor implementation. Ordinary
mutations pay no new query/index work.

## Remaining fixed scope

- 21.1: arbitrary bounded multi-edge pattern matching and broader ordinary
  Graph/Subgraph query integration. Existing endpoint filters and captured bounded
  traversal are implemented; do not describe them as a general pattern language.
- 21.2 / 26.1 / AR4: wider pinned-query/ordinary live Subgraph migration remains.
- 18.2: specialized retained group membership and temporal event/validity interval
  indexes remain. Existing named temporal-property ranges are a separate feature.
- 09.4b.2 / AR3: typed compound topology execution remains; these query reads do not
  implement an operation transaction coordinator.

Graphiti and service work remain deferred under the original priorities. No new
owner policy or open-question row was introduced.

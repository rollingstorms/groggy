# Finite captured graph patterns — 21.1

This slice completes the remaining finite native selection inventory: arbitrary
finite graph-shaped conjunctions and owner-facing explicit capture/build entry
points. It uses the previously implemented filters, ordering/pages, bounded hop
selection and transient index lifecycle. No new checklist row is added.

## Contract and implementation

- `query::{GraphPattern, PatternNode, PatternEdge, PatternOptions, PatternMatches}`
  defines named node/edge bindings, per-variable existing predicates and explicit
  directions. Variables are unique across both entity kinds; endpoints must name
  declared nodes. One node is sufficient; zero nodes reject.
- Branches, cycles, self-loops, disconnected components, isolates and parallel edge
  instances are supported. Default homomorphic binding permits node/edge reuse;
  independent distinctness controls make injective matching explicit. Neither
  endpoint pairs nor result rows are implicitly deduplicated.
- The query index and matcher share one captured endpoint/incidence implementation.
  Existing label/property/degree semantics are unchanged. No query index work is
  added to ordinary mutations. A missing topology/property index fails unless
  scanning is explicitly authorized. Every predicate is prepared before search,
  even behind empty candidates. A scan's endpoint lookup is operation-local.
- Iterative depth-first frames avoid recursion. Bound endpoint joins intersect
  sorted candidate sets and preserve stable declaration-order node-ID then
  edge-ID ordering. Result offset/limit never add together; one observed extra
  complete match proves truncation. Budget exhaustion is an error with no partial
  return. `max_expansions` counts attempted assignments, excluding predicate,
  topology and candidate preparation, so it is not a time/memory bound.
- Results own immutable source pins. `match_view(row)` keeps exactly bound node
  and edge IDs (including isolates, coalescing reused IDs), not every induced edge
  among the nodes. Source effective/recorded clocks survive selection, owner
  mutation/closure and threads. Binding dictionaries returned to Python are owned
  copies. Result construction grants no new write authority.
- Graph, connected Subgraph and DurableGraph expose `build_index` in native and
  Python APIs. These explicitly capture and build; durable reads append nothing.
  A stale connected selection rejects instead of silently dropping identities.
  This does not migrate the legacy Subgraph object's live-parent read behavior.
- FFI marshals shapes/filters/strict integer controls, translates errors and
  releases the GIL for matching and captured index construction. Rust owns the
  matching algorithm. Curated exports/type stubs and the executable API guide
  cover the public surface.

Guide: `docs/api/graph-patterns.md`.

## Scope boundaries

Finite graph patterns plus unary predicates are supported. Optional/negative
clauses, aggregation, cross-variable property comparisons and a variable-length
path language are not promised by this API. Existing `within_hops` supplies
bounded reachability. An unrestricted disconnected/reusable pattern can have
exponential output; callers should bound pages/assignments. Large offsets and
candidate preparation can still require substantial work.

Remaining pinned legacy query/ordinary Subgraph migration stays in 21.2, 26.1 and
AR4/AR7.5. Specialized retained group-membership/event/validity indexes stay in
18.2. Compound mutation coordination stays in 09.4b.2/AR3. Those existing rows must
not be checked off by completion of the finite query API. Service and Graphiti
work retain their recorded deferrals.

## Verification

Six native integration tests compare every three-node topology (directed and
undirected, complete and edge-selected scopes, all traversal directions and
reuse/distinct modes) against independent complete Cartesian enumeration. Further
fixtures cover branching/cycles/disconnected clauses, filtered parallel edges,
isolates, exact match membership, invalid names/endpoints, missing indexes behind
empty candidates, explicit scans, overflow-safe pages, truthful truncation,
budget errors, effective clocks, owner mutation/drop, threads and a 2,000-node-
variable iterative pattern.

Fourteen rebuilt Python cases verify exported classes/signatures, filters, exact
bindings, copy isolation, ordinary write-coordinate preservation, strict counts,
direction/reuse, durable no-write construction, historical clocks and owner/
thread lifetimes. Four new documentation examples execute.

Final gates: **1,410 native passed / 56 ignored**, **2,103 root Python passed /
1 skipped**, **199 executable documentation examples**, strict all-target core
Clippy, scoped formatting, Python/stub syntax and diff checks. Native commands use
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target` with offline locked
cargo test/clippy. The existing CPython 3.9 full-ABI release extension was rebuilt
and atomically staged in tree; three existing unrelated FFI warnings remain. No
packages or external graph libraries were installed/used. The owner's benchmark
and notebook/review changes are excluded from staging.

## Bounded release evidence

Reproducer: `cargo bench --offline --locked --bench graph_patterns` with the same
target environment. Five samples at 1k, 10k and 100k directed-chain nodes check
every two-edge match against the analytic chain IDs. Raw receipt:
`../benchmarks/graph-patterns-2026-10-27.json`.

| Nodes | Capture + topology index | Full-result median | 100-row page median |
| --- | ---: | ---: | ---: |
| 1,000 | 1.16 ms | 0.798 ms | 0.091 ms |
| 10,000 | 13.03 ms | 10.89 ms | 0.251 ms |
| 100,000 | 250.19 ms | 156.82 ms | 1.514 ms |

The page attempts 505 assignments, versus 499,993 for the 100k full result. Page
preparation still captures/scans the selected node-ID domain, so this is not a
constant-cost or bounded-memory page promise. Measurements exclude Python result
marshalling and use no property predicates. These are absolute local release
observations, with no before/after, RSS, dense-graph or production throughput
claim. Ordinary graph mutation paths pay no additional query work.

## Checklist movement

Existing **21.1 is complete** for the finite native selection scope above.
Fixed inventory moves **209 checked / 59 unchecked → 210 checked / 58 unchecked**,
with all 268 existing rows retained. The broader unfinished scopes listed above
remain unchecked.

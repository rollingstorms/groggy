# Selected retained temporal queries after pending topology — 2026-11-15

## Completed fixed reader family

All four existing selected lifecycle/window helpers now consume the common
pending selected frame: window_aggregate, filter, mark_changed_nodes and window.
They can follow removals/endpoint revisions in local and outer atomic programs,
loops and exact for-each frames. The selected surviving IDs determine result rows;
accepted first-parent history determines values/existence/creation/change masks.
Pending property edits never become inferred historical records. Removed working
IDs are not imported back into the selected frame.

The shared retained-index provenance/range/cancellation validators remain intact.
Atomic provisional IDs still reject as unknown to retained lineage; no accepted
creation state is fabricated. The owner question about empty history for these
IDs remains open. Ordinary manual working-node behavior is unchanged. Late errors
preserve graph/history, output rollback and unconsumed allocation coordinates.
No new API, storage format, persistent index or FFI computation.

## Qualification and measured cost

Locked/offline native suite: **1,602 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, formatting and whitespace
checks pass. Rebuilt the in-tree release extension without wheel installation;
three established binding warnings remain. Focused Python: **119 passed**.
Full maintained Python: **3,065 passed / one skipped / 14 established warnings**;
all **282 executable documentation examples** pass.

One new native shared-executor regression proves all four helpers use surviving
membership while ignoring staged values. Fifty-two Python regressions cover all
seven aggregation functions, four predicates, four change masks, empty-root and
nonempty windows, directed/undirected endpoints/cascades, local/outer loops and
for-each, late provisional-ID failure and invalid bounds. Existing retained
lineage, foreign index, numeric/error, property-only and snapshot suites remain
maintained. Black/isort/pre-commit remain unavailable. Owner files untouched.

Current native release fixture, five observations per case; medians:

| Surviving nodes | Flat, three repetitions | Three-iteration loop |
| --- | --- | --- |
| 100 | 3.461 ms | 3.403 ms |
| 1,000 | 34.637 ms | 33.831 ms |

Five accepted score changes per node, one staged removal, staged score=99, four
helpers repeated three times, one acceptance. Historical sums remain 15. Includes
ephemeral index construction, pending captures, validation and execution; excludes
fixtures, initial capture, algorithm construction, assertions and compilation.
One-process observations, not before/after speedup, allocation/RSS, FFI or general
scale guarantees. Rebuilding ephemeral retained indexes can dominate repeated
queries; caller-supplied matching indexes remain supported. No implicit persistent
index lifecycle is introduced.

Receipt: `notes/benchmarks/pending-temporal-windows-2026-11-15.json`; reproduce with
atomic_builder_properties `--pending-windows` and GROGGY_ATOMIC_BUILDER_PROFILE.

## Existing roadmap

Advances 09.4b.2, 30.1, AR7.8 and temporal/API conformance under 01.2/19.2.
**215 checked / 53 unchecked**, including **44 unchecked release rows**; no rows
added or broad unfinished parents closed. Remaining nearby work includes classic
wrapper adjacency/input migration, general topology loops and alternate lowering.

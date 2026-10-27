# Explicit captured query indexes — 18.2 / 18.3

## Accepted lifecycle

Owner clarification for A11.1 on 2026-10-27: **keep indexes transient; rebuild
explicitly**. Secondary indexes are immutable read objects, not writable graph
state. They are not serialized definitions, journal records or automatically
maintained publication caches. New state or reopened storage needs an explicit
capture/build. Existing captured indexes remain readable. Durable external-ID
reservations retain their separate existing contract.

A11.2 remains unchanged: requested indexed execution fails if an index is
unavailable, unless the caller explicitly permits scanning. Existing ordinary
scan APIs do not require optional indexes.

## Delivered

- Native `query::GraphQueryIndex`, public Python `GraphQueryIndex`, and
  `GraphView.build_index(node_properties=..., edge_properties=..., topology=...)`.
- Named global/selected property equality indexes preserve typed logical identity,
  compact aliases, Null versus absence, vectors, structured and binary values.
  Exact numeric range postings keep i64 precision across f32/f64 comparisons;
  NaNs are excluded from ranges. Date, UTC instant and duration ranges have
  separate compatible domains.
- Explicit labels postings and ordinary relationship-type property indexing.
  Optional selected endpoint, incidence, neighbor and degree indexes preserve
  direction, parallel IDs, loops, isolates and the existing GraphView conventions.
- Existing native filters and query strings, recursive Boolean filters, sorted ID
  results and immutable graph results. Empty node/edge AND conventions remain
  compatible with the existing scan executor. Complete availability checks happen
  before execution, including Boolean branches that would otherwise short-circuit.
- `rebuild(new_view)` explicitly constructs the same requested configuration for
  another capture. Clones share postings; indexed values share frozen source state.
  Postings hold IDs rather than copying large attribute payloads. FFI only
  translates filters/types/errors and releases the GIL for native execution.
- API guide: `docs/api/query-indexes.md`; curated exports and type stubs.

## Verification

- Full native suite: **1,399 passed / 56 ignored**. Five index integration tests
  cover typed predicates and Boolean combinations against the existing Groggy
  scan executor, plus every three-node topology (512 masks) for both directions
  with full, node-selected and edge-selected scopes. Parallel loops, numeric
  boundaries, absent columns, owner closure and effective captures are included.
- The subsequently extended existing journal restoration failure test was rerun
  with the correct full filter and **one test executed and passed**. It covers
  four fault boundaries (before header, after header, after payload, after sync),
  unchanged bytes/head/properties and old/new index behavior through accepted
  restoration and reopening. A prior mistyped filter executed zero tests; that
  output is not verification evidence.
- Rebuilt in-tree release extension: **2,069 Python tests passed / 1 skipped**,
  excluding the owner's separate algorithm benchmark file; **190 documentation
  examples** execute. Index tests cover invalid input and allocation/journal
  preservation, branch switching, deactivation, durable reopen, effective clocks,
  transient manual captures, shared-thread reads and explicit scan fallback.
- Strict all-target core Clippy passes. Scoped Rust formatting and diff checks
  pass. Native/FFI builds used existing offline dependencies; no packages installed.
  The release extension was atomically staged in the source tree. Python ABI
  compatibility limits remain those already recorded under 19.2/34.2.

Commands: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo test
--offline --locked`, the focused `storage::journal::state_restore_failure_tests::
restore_append_failures_leave_selection_bytes_history_and_pins_unchanged` lib test,
and `cargo clippy --offline --locked --all-targets -- -D warnings` with the same
target environment. Python verification uses root `tests/test_*.py` excluding
`test_algorithm_benchmark.py`, followed by the API documentation example runner.

## Bounded release measurement

Reproducible benchmark: `cargo bench --offline --locked --bench query_indexes`
with the target environment above. Raw five alternating-order samples:
`notes/benchmarks/captured-query-indexes-2026-10-27.json`.

At 100,000 nodes / 99,999 chain edges, median native ID-query times:

| Query | Indexed | Explicit scan |
| --- | ---: | ---: |
| One matching property value | 2.51 microseconds | 21.25 milliseconds |
| Numeric range returning 100 rows | 3.32 microseconds | 21.22 milliseconds |
| One matching endpoint pair | 0.129 microseconds | 4.51 milliseconds |

One measured property-plus-topology index construction took 91.8 milliseconds.
Capture and index construction are excluded from query timings. Both paths use
the same frozen data and return sorted IDs; graph-view materialization is excluded.
These are local bounded fixtures, not production/RSS/FFI/billion-node guarantees.
Ordinary graph mutations do not automatically pay index construction or maintenance.

## Exit and remaining work

**18.3 is complete** for the accepted explicit transient lifecycle. The fixed
268-row checklist becomes **209 checked / 59 open**, without adding rows.

18.2 remains partial: specialized retained group-membership and temporal event/
validity-interval indexes and broader working-Graph catalog/query integration.
Indexing a named temporal property already supports explicit value windows;
indexing an effective-time GraphView preserves its clocks. Neither is a general
temporal interval index. A selected view retains its underlying captured state;
a small selection is not a promise of proportional retained memory.

Ordinary connected Subgraph migration, compound topology execution, broader query
planning, service deployment and Graphiti remain in their existing scopes. The
new index representation does not change those contracts or settle other workbook
questions.

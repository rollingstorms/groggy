# Pending topology property primitives — 2026-11-15

## Completed implementation slice

Local atomic builders and shared outer atomic runs now allow certified property
loads, initialization, arithmetic and attachments after staged topology. Reads
include newly allocated provisional IDs, omit removals/cascades and preserve the
exact selected scope. The shared batch publishes once after final validation.

This advances existing rows **09.4b.2, 30.1, AR3 and AR7.8**. It does not add a
checklist row or close their broader remaining requirements. The inventory stays
**215 checked / 53 unchecked / 268 total**, with **44 unchecked rows in 0.6.0 scope**.

## Architecture and contracts

- `AtomicStepEffect::PendingProperties` certifies primitives that can consume a
  detached pending Subgraph binding. Other property-certified steps retain their
  conservative base-topology restriction. Variable-only math is explicitly
  `ImmutableInputs`.
- The existing batch capture constructs the selected pending read. A builder
  caches that binding by an internal batch read revision, invalidated by staged
  properties/additions/removals/endpoint revisions. Pure arithmetic never captures.
  The original selection remains available for later topology and final acceptance.
- Context reads recognize the immutable pending binding; attachments reuse the
  ordinary bulk kind/reserved-key/duplicate validation with pending membership as
  its liveness source. No owner mutation or separate transaction engine is added.
- Removed or out-of-scope map targets reject; stale maps are not implicitly
  filtered. The entire input validates before any property column is staged.
- Reloads observe earlier property attachments, including full f64 values. New
  entities receive initialization and attachments without being owner-active yet.
- Existing canonical clocks, creation defaults, schema validation, coordinate
  checks, cancellation and output rollback remain authoritative. This slice does
  not add attachment timestamp parameters or reinterpret effective-time queries.
- Pending-scope construction failures finish the current profiling step before
  propagating to the owning rollback boundary.

## Verification

Executed with Rust 1.96.1 and macOS arm64 CPython 3.9.6:

- Full locked/offline Rust suite: **1,582 passed / 54 ignored / zero failed**,
  across 223 suite results, including doctests.
- Strict all-target Clippy: **passed**, including the updated benchmark.
- Rust formatting and `git diff --check`: **passed**.
- Maintained root Python surface: **2,791 passed / one skipped / 14 established warnings**.
- Curated executable documentation: **272 examples passed**.
- The release extension was rebuilt in-tree with `maturin develop --release
  --skip-install`, without installing a wheel. The three established binding
  compiler warnings are unchanged.

Commands used `CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=python-groggy/target` for native gates:

```sh
cargo test --offline --locked
cargo clippy --offline --locked --all-targets -- -D warnings
cargo fmt --all -- --check
```

Python/docs verification used `PYTHONPATH=python-groggy/python`, the configured
Python 3.9 interpreter, and `notes/scripts/verify_immutable_view_migration.py`.
Black, isort and pre-commit executables are unavailable locally; no external
packages were installed. Unrelated owner review/benchmark work is untouched.

Coverage includes directed/undirected graphs, sparse IDs, parallel edges and
loops, selected topology, added entities, endpoint revisions and cascades,
read-after-write, later topology invalidation, empty-scope creation, duplicate
provisional targets, reserved keys, invalid kinds, stale maps, rollback and
Context reuse. Native and Python guard tests now use still-unadapted adjacency
readers where property initialization is intentionally newly supported.

The user-facing contract and executable example are in
[`pending-topology-properties.md`](../../docs/api/pending-topology-properties.md).

## Cost and remaining work

Pending scope materialization copies selected state/attributes. The revision
cache avoids recapturing consecutive unchanged reads; it does not establish a
new throughput budget or remove the cost after each changing attachment. Runs
without pending topology retain the existing property overlay. No broad
performance or new cross-platform qualification claim is made.

The native release receipt uses five observations per case, with fixture,
initial capture and algorithm construction excluded. It times initialization,
count/index reads, attachment, reload, second attachment and one publication:

| Initial nodes | Property overlay, zero new nodes | Pending topology, one new node |
| --- | --- | --- |
| 1,000 | 3.595 ms | 3.525 ms |
| 10,000 | 27.308 ms | 37.132 ms |

These are medians from one local process. The 1k difference is too small to
support a performance conclusion; the 10k case records the additional pending materialization/creation cost; this is
not a before/after optimization comparison or a release performance budget.
Machine-readable observations:
[`pending-topology-properties-2026-11-15.json`](../benchmarks/pending-topology-properties-2026-11-15.json).
Reproduce with `cargo bench --offline --locked --bench atomic_builder_properties
-- --pending-properties`, setting `GROGGY_ATOMIC_BUILDER_PROFILE` to the desired
receipt path and the same native target/incremental environment as the gates.

Unadapted adjacency-dependent primitives and classic property-writing algorithm
wrappers still reject after topology. Explicit captured degree/algorithm-table
reads remain supported. Captured-result attachment, general topology loops,
journal-owner dispatch, remaining DSL execution/lowering, naming/cache policy
and pipeline-history policy stay in the existing open parents.

The next bounded execution slice is to connect supported captured computation
results to explicit validated attachments, or adapt the remaining adjacency
readers against the same pending source. Neither should bypass row provenance,
selection, final batch validation or the single acceptance boundary.

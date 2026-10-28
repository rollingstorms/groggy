# Exact property-value intervals — existing 18.2 component

Completed `EffectiveTimeIndex.node_property_intervals` and
`edge_property_intervals`, native and Python. Queries return maximal half-open
intervals of an entity's exact property value during its structural active
lifetimes. Inactivity produces no rows; absence differs from explicit Null.
Unknown IDs return no rows, while a never-assigned property has absence intervals
for its entity's active lifetimes. Explicit recorded cutoffs and accepted-order
effective-time ties follow the existing replay contract. No additional clock is
recorded or inferred.

Property-specific postings share the existing immutable event/delta lineage.
Each query merges only the named property's events and the entity's lifecycle
events, without scanning unrelated properties or reconstructing graph snapshots.
Birth/restoration resets property state, and out-of-lifetime assignments are
ignored. One timestamp resolves before emitting intervals; equal exact adjacent
values coalesce using the canonical retained-value comparator. Float widths,
signed-zero bits and NaN payloads remain distinct. Typed vectors, structured and
temporal values preserve their existing conversions.

The implementation is a private child of the existing effective-read owner,
not a new public subsystem or independent graph replay engine. Public native
`PropertyInterval` carries an optional typed value. FFI only validates/translates
controls, releases the GIL, and marshals owned dictionaries with explicit
`present`/`value` fields. Existing captures and copies retain clocks/values and
share index postings; rebuilds remain explicit and transient. Build memory adds
one ordinal per property event and entity/property lookup maps, without copying
property payloads. Emitted rows copy their values and exact comparison can cost
time proportional to large values.

## Verification

Three additional native integration tests and 24 Python cases cover independent
expected intervals and canonical point-read comparisons, Null/absence, backdated
writes, accepted-order ties, lifecycle removal and same/later restoration,
unknown/never-assigned properties, explicit recorded cutoffs, inactive writes,
extreme signed clocks, exact float width/zero/NaN payloads, typed containers and
temporals, owned output mutation, durable close/reopen, shared-thread reads,
independent explicit legacy captures, and strict invalid IDs/clocks/names.

Final gates: **1,423 native passed / 56 ignored**, **2,274 root Python passed /
1 skipped**, **214 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The existing
CPython 3.9 full-ABI release extension was rebuilt and atomically staged in tree;
three pre-existing FFI warnings remain. Root Python verification excludes the
owner's separate algorithm benchmark and nested historical surveys. No external
packages or graph libraries were installed/used. Owner benchmark/notebook/review
changes remain excluded from the commit.

Native commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`
with offline locked `cargo test`, focused `--test effective_time_index`, strict
`cargo clippy --all-targets -- -D warnings`, and `cargo bench --bench
property_intervals`. Python runs the rebuilt extension's focused/root contract
suites and executable documentation runner. Guide: `docs/api/property-intervals.md`.

## Bounded release measurement

`notes/benchmarks/property-intervals-2026-10-28.json` records five alternating
samples with 100 result-checked iterations each. One continuously active node has
two assignments to `x`, with 100 or 10,000 intervening assignments to `noise`.
An independent flat timeline reference scans that known monotonic fixture and
constructs the same two value intervals. Both paths allocate owned outputs and
check every result. Indexed timing additionally converts native interval rows
to reference tuples. Median local timings, excluding fixture creation:

| Unrelated property events / total logical events | Index build | Indexed intervals | Flat timeline scan |
| --- | --- | --- | --- |
| 100 / 104 | 38.38 µs | 0.190 µs | 0.103 µs |
| 10,000 / 10,004 | 5.549 ms | 0.173 µs | 4.128 µs |

The small fixture exposes lookup overhead; the larger fixture demonstrates that
unrelated events are skipped. These are native release measurements for one
bounded workload, not production/RSS/Python throughput claims. Explicit builds
still scan retained events and allocate postings. Queries remain proportional to
relevant property/lifecycle candidates, value comparison and emitted payloads;
no logarithmic full-graph reconstruction or free history retention is claimed.

## Existing remainder

A stored boolean property can express fact validity with these same intervals;
no new domain inference is implied. Edge property intervals alone do not certify
endpoint validity; canonical point reconstruction retains that responsibility.
Specialized retained group-membership indexes remain under 18.2. Wider connected
Subgraph ownership (26.1/AR4/AR7.5), legacy pinned query migration (21.2), and
compound topology/effect acceptance (09.4b.2) keep their existing scope. No new
checklist rows: **210 checked / 58 unchecked**, with 18.2 still partial.

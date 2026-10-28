# Effective-time event and activation indexes — existing 18.2 component

Completed explicit `Graph.build_effective_index()` and
`DurableGraph.build_effective_index()` with the native `EffectiveTimeIndex`,
curated Python export and type stubs. The index owns one immutable first-parent
branch/head capture. It offers reconstructed reads, inclusive effective-time
event windows, entity-local events, and half-open structural activation intervals.
Optional recorded cutoffs remain explicit per query. Inactive IDs remain
inspectable. Changes at equal effective times follow accepted replay order.

The implementation extends the existing effective replay owner in
`src/state/effective_read.rs`; it does not introduce another reconstruction
algorithm. Native owner entry points live with query capture entry points.
Global sorted events, entity postings and separate lifecycle postings share
retained lineage; lifecycle queries do not scan every property event. Python
only translates strict clocks/IDs/results and releases the GIL for native work.
Publication metadata and event rows share one change-key marshaller.

Captures keep precise/explicit legacy clocks and frozen group source revisions.
Later writes, branch switches and owner closure do not retarget them. Indexes
remain transient under A11.1: rebuild explicitly for new state or reopened
storage. There are no persisted definitions, automatic write maintenance or
recovery rebuilds. Durable capture leaves journal bytes and accepted head
unchanged and does not retain a storage lock. The pre-existing recorded-state
`build_temporal_index()` keeps its separate coordinate contract.

## Verification

Three native integration cases and fourteen Python cases cover independent
publication-metadata window references, canonical point reconstruction,
backdated/equal-time writes, property Null/unset, rewiring, inactive IDs,
restoration with same/later activation times, branch isolation, owner drop,
shared-thread reads, durable close/reopen, frozen sources, explicit incompatible
legacy clock captures, manual pending rejection, empty roots, strict controls,
and inclusive extreme timestamp bounds/reversed ranges.

Final gates: **1,420 native passed / 56 ignored**, **2,250 root Python passed /
1 skipped**, **211 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The CPython 3.9
full-ABI release extension was rebuilt and atomically staged in tree; three
pre-existing FFI warnings remain. Root Python verification excludes the owner's
separate algorithm benchmark and nested historical surveys. No external packages
or graph libraries were installed/used. Owner benchmark/notebook/review work is
excluded from this commit.

Commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target` with
offline locked `cargo test`, focused `--test effective_time_index`, strict
`cargo clippy --all-targets -- -D warnings`, and `cargo bench --bench
effective_time_index`. The FFI release build uses the existing Python interpreter
and manifest; Python runs the root contract suite and executable documentation
runner. Public guide: `docs/api/effective-time-indexes.md`.

## Bounded release receipt

`notes/benchmarks/effective-time-index-2026-10-28.json` records five alternating
samples per fixture; each window timing includes 100 result-checked iterations.
One node receives 100 or 10,000 monotonic property updates. Independent publication
metadata supplies the flat property-event scan reference. Both methods return
the final ten events; all returned records and reconstructed terminal values are
checked. Times below are local sample medians, excluding fixture creation.

| Property updates / total logical events | Index build | Indexed window | Flat property-event scan | Full-prefix reconstructed read |
| --- | --- | --- | --- | --- |
| 100 / 102 | 41.71 µs | 0.578 µs | 0.526 µs | 8.96 µs |
| 10,000 / 10,002 | 4.656 ms | 0.603 µs | 7.346 µs | 0.932 ms |

The small fixture shows index overhead; the larger narrow-window fixture shows
the avoided scan. These are native release measurements, including window
allocation/result checking, without Python marshalling, RSS measurements or a
production throughput claim. Point reconstruction still replays the selected
prefix and builds a snapshot. Event windows use binary search plus candidate
visitation; recorded cutoffs can discard candidates. Build/retained-memory costs
are explicit, and ordinary mutations do no automatic secondary-index work.

## Remaining existing scope

18.2 stays partial: specialized retained group-membership and attribute/fact
validity interval indexes remain. Activation intervals describe structural
add/remove events, not attribute validity or endpoint certification. Point reads
retain canonical timeline/source validation. No new checklist rows were added;
the fixed inventory remains **210 checked / 58 unchecked**. Ordinary connected
Subgraph ownership (26.1/AR4/AR7.5), broader legacy pinned query migration (21.2)
and compound topology/effect acceptance (09.4b.2) keep their existing scope.

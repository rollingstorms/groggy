# Atomic effective endpoint validation — 16.2

## Delivered

Checked topology preparation validates complete effective edge lifetimes against
both endpoint lifetimes before allocating IDs, mutating the graph or appending a
durable frame. This includes future invalid intervals, per-change overrides,
implicit cascade times and retained edges already removed from the current graph.
Equal effective timestamps use accepted state order and the reader's lifecycle
phases. Signed minimum/maximum timestamps require no sentinel arithmetic.

A history-owned index stores per-entity lifecycle events and retained incident
edge IDs. Accepted versions append events incrementally; property-only preparation
skips the validator, and property publications only advance the cache head.
Validation overlays proposed events without accepting them and sorts affected
entity timelines. Unrelated graph size and property-history depth do not cause a
full replay. Switching ancestry rebuilds the selected branch's index, and explicit
history GC clears it. The cache uses a Mutex to preserve HistoryForest's sharing
traits; mutation through exclusive ownership uses direct access. It holds no
property payloads or graph snapshots and is not persisted as canonical data.

Journal recovery prepares with the original recorded clock, then verifies exact
stored metadata. Reopening must not substitute the current wall clock for an
implicit birth/death time. Invalid proposals leave journal bytes/head unchanged.
Existing invalid timed journals now reject on replay; silently reopening them
would preserve the violation this slice is intended to prevent.

## Compatibility and remaining work

Explicit timed topology changes require precisely timed single-parent ancestry.
Untimed legacy manual writes retain compatibility but cannot claim historical
endpoint validity when their birth/death clocks were never recorded. Property-only
changes remain supported. Precise read reconstruction still rejects legacy/merge
ancestry. These limits are surfaced in the error/API docs, without guessing times.

The independent reader boundary check remains for raw HistoryForest construction
that bypasses checked Graph publication. 16.2 remains partial for retained pooled/
group history, merge/legacy migration and read-index/checkpoint work. Property
unset remains 11.2. No additional checklist rows were introduced or parents falsely
closed: **196 checked / 72 open**.

## Evidence

Native cases cover pre-birth edges, rejected-ID reuse, historical removed edges,
cascade override failure/retry, future intervals, loops/parallel IDs, branch index
rebuilds, extreme clocks, compatibility boundaries and durable frame/reopen clocks.
Python verifies atomic rejection, ID reuse and durable bytes through the rebuilt
extension. Existing mixed scoped-write fixtures now give endpoints explicit birth
times rather than testing a timed edge against unknown/incompatible births.

Release benchmark (`cargo bench --bench effective_topology_publication`):

| Nodes | Prior property versions | Median full edge publication |
| --- | --- | --- |
| 1,000 | 0 | 6.67 µs |
| 1,000 | 1,000 | 4.54 µs |
| 100,000 | 0 | 2.63 µs |
| 100,000 | 1,000 | 2.21 µs |

Twenty-one samples each; isolated nodes born at zero, publish parallel edges at
2000 and verify the reconstructed edge count. This is a bounded local current-cost
receipt, not a prior-version speedup, memory budget, FFI, high-degree, deep lifecycle
or production-capacity claim. Machine-readable receipt lives in notes/benchmarks/.
Full verification results are in the roadmap status log.

Final gate: 1,178 native passed / 51 ignored; 1,903 root Python passed / 1 skipped;
136 executable documentation examples; strict all-target Clippy passed. Three
focused rebuilt Python regressions also passed after adding identity-reservation
assertions. Scoped rustfmt and git diff --check passed. No external installs.

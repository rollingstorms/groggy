# Exact pending subgraph execution frames — 2026-11-15

## Completed behavior slice

The atomic coordinator now records the original selected execution root. Native
StepScope/StepExecutor distinguish a dynamic root selection from an exact child
frame; this distinction survives nested loops. Pending reads validate/materialize
the root and then project the child. Sibling assignments no longer appear as
foreign targets and provisional root nodes do not leak into every child.
Dynamic captures still respect the incoming node/edge selection plus surviving
added entities; they do not expand a narrowed selection to the whole root.

A PendingSelections capability refreshes property-only pending sources before
producing immutable children. The four existing iterate/neighbor/emit primitives
now use selected source endpoints/direction and preserve pins in their outputs.
For-each composes conservative descendant effects and uses the common native step
executor. Child execution reads refresh, while supplied snapshots/tables remain
immutable. Explicit capture/table bodies retain their persistent/local-atomic
requirements. General child topology mutations, alternate plans, streaming and
uncertified effects retain guards; legacy owner readers remain before topology.

Loops and for-each share nested-body parsing and top-level literal inference.
Empty programs no longer become numeric lists, typed vectors stay consistent,
dependency names are retained and malformed parameter/dependency fields reject.
The old independent loop parser and duplicate nested value translator are retired.

The graph, deltas and accepted IDs publish once at the outer boundary. Late child
errors restore prior outputs and discard all earlier sibling/topology effects.
Existing ordinary/non-atomic sampler behavior remains available.

## Qualification

Locked/offline native suite: **1,597 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, Rust formatting and whitespace
checks pass. Rebuilt the in-tree release extension without installing a wheel;
three established binding warnings remain. Focused Python: **149 passed**.

Full maintained Python gate: **2980 passed / one skipped / 14 established warnings**,
plus **279 executable documentation examples**.

Coverage includes sparse IDs, directed/undirected, stored loops, exact edge
instances, provisional IDs, revisions/cascades, sibling overlap, nested frames,
property-only producer loops, immutable earlier outputs, table aliases/timestamps,
late child failures, stale seed selections, empty-child preflight and nested literal
and grammar consistency. Native root/child tests also prove dynamic narrowing and
rejection of out-of-root fixed membership. Existing sampling/foreach/loop suites
remain maintained. Initial tests incorrectly omitted the induced self-loop on a
reached node and used an unqualified test-module import; fixtures were corrected.
Black/isort/pre-commit remain unavailable. Unrelated owner files are untouched.

## Measured cost and next optimization

Native release, five observations per case, singleton iteration with per-child
load/increment/attach and one acceptance:

| Initial nodes | Property-only | One staged node |
| --- | --- | --- |
| 100 | 0.929 ms | 5.296 ms |
| 1,000 | 4.825 ms | 325.293 ms |

Receipt: `notes/benchmarks/pending-child-frames-2026-11-15.json`; reproduce with
atomic_builder_properties `--child-frames` and GROGGY_ATOMIC_BUILDER_PROFILE.
Fixtures/initial capture, compilation, algorithm construction and assertions are
excluded; frame production, pending materialization, validation and acceptance
are included. One process and current-fixture observations, not before/after
speedup, FFI, memory or general scale guarantees. An earlier development run was
slower; the qualified receipt above is the reported observation.

The repeated full-root value copy makes write-heavy pending child iteration
expensive. Next: retain one accepted source for the batch, prepare pending topology
by structural revision, validate changed property targets incrementally, and build
exact frame snapshots directly. Preserve immutable sources, earlier capture
authority, endpoint/selection closure, late rollback and final canonical acceptance.
Measure requested allocations as well as matched native release throughput before
claiming improvement. No additional mutation or FFI computation path is needed.

## Existing roadmap

Advances 09.4b.2, 03.2/AR2, 30.1 and AR7.8; conformance/API evidence contributes to
01.2/19.2. Inventory remains **215 checked / 53 unchecked**, including **44 unchecked
release rows**. No checkbox rows added or broad unfinished parents closed.

# Portable flow values and aliases — 2026-11-25

Finite component of existing 09.4b.2, 30.1, 32.3 and AR7.8. All three steps in
`steps/flow.rs` now use the shared sealed read boundary. Stable flat counts remain
217 checked / 51 open; broad execution and scale parents remain partial.

## Completed component checks

- [x] Port scalar/map flow updates with exact checked integer and finite-float
  arithmetic, preserving map-union semantics and unchanged passthrough values.
- [x] Port positive residual-capacity maps without inventing reverse-edge IDs or
  imposing a new conservation/domain policy.
- [x] Port generic aliases for owned maps/scalars and immutable graph/table values.
- [x] Replace eager alias payload cloning with existing shared immutable bindings
  and copy-on-write mutable access; preserve independent mutation and rebinding.
- [x] Verify real worker execution, sparse selected edges, pending new edges,
  earlier aliases, exact effective clocks/reopen and full failure rollback.
- [x] Measure the public repeated-alias path against a matched unpacked-wheel
  baseline and no-alias control; record scope and residual copy costs.

## Contract and implementation

Owner adapters delegate to their sealed kernels. Only immutable input values and
already-prepared metadata cross workers. No map arithmetic moved into FFI.
Integer overflow/nonfinite arithmetic still rejects before installing results.
Map flow updates retain the union of input edge IDs; scalar updates cover flow
keys. Residual results retain positive capacity-minus-flow and positive flow-only
values under their existing IDs. These are map primitives, not a new max-flow
algorithm; attachment continues to validate graph authority.

Aliases use the same shared value mechanism as loop initial bindings. Mutable
map/column access obtains a private copy on first aliased mutation. Rebinding a
name never changes an earlier alias. Graph/table aliases keep their original pin
and survive closing the owner. This changes internal copy cost, not result kinds
or naming/cache/history policy.

## Evidence

The flow unit suites retain checked overflow/nonfinite, sparse union, positive
residual and empty-input rules. The new alias regression mutates both map and
column copies, checks source independence and self-aliasing, and preserves an
existing target when a source is missing.

`tests/durable_flow_steps.rs` compares scalar and edge-map deltas against exact
integer expectations above f64 precision, memory execution, worker execution and
reopen. It covers selected parallel/self-loop edges, pending new edges, old
aliases, graph/table pin lifetime and rollback after missing-source/overflow
failures. `tests/test_durable_flow_steps.py` adds public mixed numeric, owned value,
immutable output lifetime and failure cases. The executable contract is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Local gates pass **1,682 native tests** (54 ignored), **3,525 packaged Python
tests** (one skipped, 23 expected validation warnings), and **298 documentation
examples**. Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax
and all 62 packaged Python/stub byte comparisons pass. All 500 production sources
match runtime `46fe0212ebf7eaa5b78a407bd94af960fe569d44`. See the
[local receipt](../benchmarks/portable-flow-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-flow-wheel-qualification-2026-11-25.json).
Earlier platform receipts remain historical until the explicit new candidate
qualification finishes. No production latency, allocator or RSS guarantee is
inferred from the transient alias benchmark.


## Matched alias timings

[Raw samples and artifact hashes](../benchmarks/owned-alias-performance-2026-11-25.json)
compare qualified runtime `23399898` with this runtime using the unchanged
`benches/owned_alias.py`. Each runtime has three sequential processes and nine
observations per case; builds/tests are finished before timing. Setup, compile
and warm-up are excluded, final transient attachment and Python marshalling are
included. Every result is checked independently and graph history is unchanged.

| Route | Rows | Aliases | Before ms | After ms | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,000 | 10 | 0.154 | 0.151 | -2.0% |
| control | 10,000 | 10 | 2.011 | 1.700 | -15.5% |
| control | 1,000 | 100 | 0.146 | 0.144 | -1.0% |
| control | 10,000 | 100 | 1.866 | 1.715 | -8.1% |
| alias | 1,000 | 10 | 0.228 | 0.146 | -36.0% |
| alias | 10,000 | 10 | 2.483 | 1.698 | -31.6% |
| alias | 1,000 | 100 | 0.892 | 0.151 | -83.1% |
| alias | 10,000 | 100 | 9.499 | 1.807 | -81.0% |

At 100 aliases, medians are 81–83% lower. No-alias control medians also vary
(roughly 1–15% lower), so these are descriptive results for the combined sealed
loop/immutable-sharing migration, not an isolated per-instruction causal estimate.
First mutation, durability sync, allocator/RSS and production SLAs are excluded.

## Remaining scope

First mutation of a shared map/column can still copy its payload. The benchmark
isolates read-only repeated aliasing and final output marshalling; it does not
qualify arbitrary mutation-heavy workloads. Other read producers/containers,
retained history, edge-aware registers, normalization options, fixed-child and
provisional topology, alternate IR and streaming stay in the existing parents.
A14 naming/cache/history and A15 production budgets remain unanswered.

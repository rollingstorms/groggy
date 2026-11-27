# Portable community helpers and decay — 2026-11-27

Finite component of existing 03.2, 09.4b.2, 28.16, 29.3, 30.1 and AR7.8. All
three community helpers and numerical decay now share selected memory/durable
reads. No stable checklist row is added; counts remain 217 checked / 51 open.

## Completed component checks

- [x] Share singleton/degree/random community seeding through sealed read kernels.
- [x] Share modularity gains and typed synchronous label propagation unchanged.
- [x] Preserve pending directed/undirected incidences, loops, parallel edges,
  sparse selections, column order, local RNG and stable tie rules.
- [x] Share numerical decay over node maps, ordered columns and edge maps, with
  subnormal controls, in-place bindings and immutable aliases preserved.
- [x] Compare actual workers, memory and reopening against independent modularity,
  label-vote, seed and arithmetic expectations.
- [x] Preserve old outputs, selected-domain validation at zero elapsed time,
  effective attachment clocks and all staged write/journal/ID rollback.

Owner adapters delegate to native read kernels; FFI receives no graph algorithms.
The same pending selected-neighborhood helper is used, preserving existing
incidences and graph-default direction. Modularity still computes best
nonnegative moves to neighboring communities in the unweighted objective, with
undirected loop degree counted twice. Label propagation adds a self vote to the
stored incidence votes and resolves ties by canonical typed serialization.
Community helpers produce ordered node columns, enabling the existing decay
column path on workers without a new data type.

Decay remains a numerical operation with caller-supplied elapsed time. It does
not read retained owner history, infer timestamps for provisional IDs or change
property validity. Existing half-life/value checks, zero-time behavior and
underflow handling remain. Inputs are immutable; only explicit attachment writes
graph properties, with one accepted delta for the complete pipeline.

Evidence: `tests/durable_community_steps.rs` checks six directedness/strategy
cases with independent objective recomputation for each candidate modularity
move, explicit vote counts and seeding rules. It covers pending endpoint changes,
removal/creation, sparse selected IDs, outside isolation, real workers, memory,
recovery and rejection. Existing native community tests retain exhaustive small
objective comparisons and column/order/error cases. `tests/durable_decay_steps.rs`
checks three layouts, nested bodies, aliases, removed-row rejection at zero time,
ID/journal rollback and no-history computation. Existing temporal tests retain
control, cancellation and duplicate-column checks. Public Python tests add 11
community and 18 decay cases. The combined executable example is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Local qualification passes **1,706 native tests** (54 ignored), **3,641 packaged
Python tests** (one skipped, 23 expected warnings) and **304 documentation examples**.
Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax and all 62
packaged Python/stub comparisons pass. All 500 production source files match
runtime `fb969e53397621f7ca276b2ca4f939960dfc6705`. See [local receipt](../benchmarks/portable-community-decay-local-qualification-2026-11-27.json)
and [wheel receipt](../benchmarks/portable-community-decay-wheel-qualification-2026-11-27.json). No new platform or performance
claim is made. Retained-history producers, collections/foreach, edge registers,
remaining frontend and topology domains remain in existing parents.

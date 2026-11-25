# Portable aggregation steps — 2026-11-25

Finite component of existing 09.4b.2, 30.1, 30.3 and AR7.8. The aggregation
module's eight native-read steps now share memory/durable execution: seven owned
map reductions join the previously portable neighbor aggregator. Global flat
counts remain 217 checked / 51 open; broader execution parents remain partial.

## Completed component checks

- [x] Port reduce-nodes sum/mean/min/max and sample standard deviation.
- [x] Port median and linearly interpolated quantile using the existing typed
  numerical kernel, without narrowing source integers before that kernel.
- [x] Port typed-category mode and entropy, preserving contributor-ID tie rules.
- [x] Port histogram scalar count vectors and owned named outputs.
- [x] Run supported statistics inside ordinary nested bodies on a worker, using
  already-loaded immutable inputs; explicit later loads see pending writes.
- [x] Preserve no-op history for read-only/empty computations, failure rollback,
  accepted-ID allocation and effective-time reconstruction after reopen.

## Contract and architecture

Owner adapters delegate to the sealed read implementation. Computation consumes
named owned node maps; no writable graph owner crosses the worker boundary and
no second statistics implementation is added. FFI remains translation only.
Use explicit broadcast before node attachment; histogram outputs do not carry
graph identity. No graph mutation is recorded merely for computing a statistic.

Existing sample/empty behavior is preserved: fewer than two observations give
null standard deviation; empty reduction/median/quantile/mode return null;
empty entropy is zero and histogram bins are zero-filled. Numeric aggregation
requires finite numeric values. Mode and entropy retain typed categories and
mode ties choose the lowest contributor ID. Result and interpolation types remain
the existing ones. No storage, cache, naming or run-history policy changes.

## Evidence

`tests/durable_aggregation_steps.rs` exercises every aggregation and all four
reduce-nodes modes against arithmetic expectations, memory execution, real worker
execution and reopening. It checks selected noncontiguous IDs and outside rows,
empty/read-only output behavior, and restoration of previous histogram outputs
and allocation/journal state after a late numeric failure. A four-epsilon bound
is used only for the independent sample-standard-deviation reference; memory and
durable output bits must agree exactly.

`tests/test_durable_aggregation_steps.py` covers public Python nested invocation,
empty results, exact large-integer mode ties, integer residuals in standard
deviation, extreme finite medians, category ties, owned histogram mutation,
pending reload versus a frozen prior input and late-failure rollback.
Unsupported-step fixtures now use retained owner snapshot capture, keeping
preflight rejection coverage after median becomes portable.

Local qualification passes **1,675 native tests** (54 ignored), **3,489 packaged
Python tests** (one skipped, 14 established warnings), and **296 documentation
examples**. Strict all-target Clippy, root formatting, whitespace, changed Python
3.8 syntax and all 62 packaged Python/stub byte comparisons pass. All 500
production sources match runtime `a305dac5f200886b0fcf8d96792fa4b498669713`.
See [local receipt](../benchmarks/portable-aggregation-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-aggregation-wheel-qualification-2026-11-25.json). The executable contract and example
are in [durable pipelines](../../docs/api/durable-pipelines.md). No new platform
or performance improvement claim is made by this adapter migration.

## Remaining scope

Other composition containers, unported retained-history producers, edge-aware
registers, normalization options, fixed-child/provisional topology and remaining
expression/IR and streaming routes stay in existing execution rows. A14 naming,
cache/history and A15 production budgets remain unanswered.

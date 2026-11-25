# Portable declarative expressions — 2026-11-25

Finite component of existing 03.2, 09.4b.2, 30.1 and AR7.8. Declarative
`core.map_nodes` now shares memory/durable execution through one evaluator;
programmatic mapping callbacks retain their separate owner-only boundary.
Stable flat counts remain 217 checked / 51 open; broader frontend work stays
partial.

## Completed component checks

- [x] Add a private expression read-source adapter while preserving public
  `ExprContext`, `StepInput` and `Expr.eval` layouts/signatures.
- [x] Reuse the existing evaluator for native owner and detached execution;
  preserve checked operations, constants, attributes and short-circuit rules.
- [x] Preserve pending attribute reads, graph-default neighborhood definitions
  and the distinction between incidence counts and distinct neighbor values.
- [x] Preserve synchronous and ordered async working-source visibility and
  complete-result installation, including nested ordinary loops and aliases.
- [x] Verify staged creation/removal/endpoint changes, selected-outside isolation,
  worker execution, memory parity, effective-time recovery and failure rollback.
- [x] Exercise supported public expression/structured-iteration builders with
  optimization enabled and disabled, without widening callback authority.

## Architecture and contract

A private input enum supplies either the existing owner input/overlay or the
selected read scope. Both feed the same `eval_context` implementation; the public
expression context stays unchanged and no evaluator is duplicated in FFI.
Owner mapping adapters delegate to the sealed expression kernel. Run-local
inputs/working maps live only during evaluation and carry no registry hooks.

Named maps remain immutable owned values. Async mapping clones its working source
once, processes ascending selected IDs, and exposes that source override to later
nodes. Publication/output installation still waits for the whole operation to
succeed. Properties read the current staged overlay; a previously loaded map
remains unchanged unless it is the explicit async source being updated.

Default neighborhoods are outgoing on directed graphs and symmetric on undirected
graphs. `neighbors`/`neighbor_count` preserve edge incidences; `neighbor_values`
preserves one value per distinct neighbor. Self-loops retain one incidence.
This preserves the established expression contract rather than substituting the
standalone collector's parallel-value semantics.

## Evidence

`tests/durable_expression_steps.rs` checks 16 directedness/async/expression cases
against independent arithmetic, memory execution, actual worker execution and
reopen. It includes pending creations, endpoint revisions and node removal,
noncontiguous selected IDs, outside rows, loops, parallel edges, staged property
reads and a non-evaluated division-by-zero branch. Late numeric failure restores
prior outputs and discards pending IDs in both update modes.

`tests/test_durable_expression_steps.py` exercises the public builder with both
optimization settings, default graph direction, distinct neighbor values, async
visibility, structured iteration/arithmetic/aliases, exact integers above f64
precision, pending properties and failure rollback. The executable contract is
in [durable pipelines](../../docs/api/durable-pipelines.md).

Local qualification passes **1,684 native tests** (54 ignored), **3,539 packaged
Python tests** (one skipped, 23 expected validation warnings), and **299
documentation examples**. Strict all-target Clippy, formatting, whitespace,
Python 3.8 syntax and all 62 packaged Python/stub byte comparisons pass. All
500 production source files match runtime `05e71ca435c21e120e4e1a585d300ed193a471ea`.
See [local receipt](../benchmarks/portable-expression-local-qualification-2026-11-25.json)
and [wheel receipt](../benchmarks/portable-expression-wheel-qualification-2026-11-25.json). This adapter migration makes no
new performance or cross-platform claim. The concurrently dispatched artifact
run is anchored to the preceding flow/alias candidate; its scope is recorded
separately.

## Remaining scope

Rust callbacks remain owner-only. Other producers/containers, retained history,
edge-aware registers, normalization options, fixed-child/provisional topology,
remaining alternate IR and streaming stay in existing execution parents.
A14 naming/cache/history and A15 production budgets remain unanswered.

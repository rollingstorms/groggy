# Typed captured graph/table pipeline values — 2026-11-11

Completes a fixed nine-stage immutable read inventory within existing
**30.1 / AR7.8**. Original broad parents remain open; no checklist rows added.

## Implemented workflow

Retained TemporalSnapshot → immutable GraphView → exact selected graph → owned
node/edge table → selected rows/columns → named typed outputs. New native
StepValue variants retain GraphView and shared immutable BaseTable data. Typed
accessors fail on missing/wrong-kind variables. Canonical graph/table readers
perform every selection/projection; no writable graph reconstruction, entity-ID
remapping or FFI algorithm implementation is introduced.

Nine registered primitives: graph.from_snapshot, graph.select_nodes,
graph.select_edges, graph.nodes_table, graph.edges_table, table.select_columns,
table.select_rows, graph.emit and table.emit. All control/type failures propagate;
noninteger/negative IDs/rows and unsupported controls reject before execution.
Empty lists remain valid values. Exact IDs, selected edge instances, explicit
null/absence alignment, wide integers/f64 and source row provenance survive
owner changes/removal and owner lifetime. Table output edits do not mutate other
outputs or their graph source.

The existing native alias now clones the actual StepValue instead of probing a
fixed subset of map/scalar types. Graph/table values, snapshots and temporal
indexes retain their representation/source. Graph/table aliases share immutable
payloads rather than copying full captured graphs/tables.

Python `groggy.builder.step_pipeline` delegates native StepSpecs through the
existing BuiltAlgorithm/interpreter and result wrappers. It copies the submitted
program and retains statement order/native aliases, bypassing legacy encoding/IR
optimization. There is no second execution engine or silently inferred naming
policy. Existing DSL behavior and default atomic=False are unchanged.

## Atomic output ownership

These explicit-input read steps certify for the existing property-only atomic
scope: an explicitly retained state excludes in-flight attachments. Local atomic
builders now preserve/restore pre-existing Context outputs on execution and
publication failure, matching the outer atomic coordinator. Both use one native
output-scope completion helper. Prior outputs move out of the scope; successful
results merge back by explicit key, failures discard partial outputs. Retained
row buffers are not deep-cloned merely to support rollback.

Native tests check buffer pointer identity on both success/failure, previous-name
overwrite rollback, context reuse and staged property rollback. This is an output
ownership guarantee, not an arbitrary custom callback rollback/sandbox guarantee.
An inherited persistent atomic batch already uses the outer output scope;
children share that scope. Transient nested scopes also move their maps rather
than duplicating retained buffers.

## Bounded release evidence

`cargo bench --bench atomic_builder_properties -- --typed-values`, output selected
through GROGGY_ATOMIC_BUILDER_PROFILE. Raw receipt:
`notes/benchmarks/typed-pipeline-values-2026-11-11.json`.

Five samples, construction/compilation and output validation excluded:

- Retained snapshot reconstruction → alias → node table → selected score column
  → emitted table: median **0.954 ms at 1k nodes**, **67.908 ms at 100k nodes**.
  This includes existing reconstruction work; it does not measure zero-copy
  reconstruction or promise streaming/linear total cost.
- Empty local atomic run retaining zero versus one million existing result rows:
  median **0.708 μs for each**. The original row-buffer pointer remains identical
  after every run. This supports the scoped no-deep-copy claim; no allocation/RSS,
  Python GIL/FFI or general throughput certification is inferred.

## Remaining original inventory

Current staged graph captures, topology mutation intermediates/read-after-write,
remaining high-level DSL/IR lowering, automatic naming/cache policy, streaming
and interrupted inner-kernel cancellation remain under 30.1/AR7.8/09.4b.2.
A14.1 remains unanswered; this slice keeps explicit existing variable/result
names. Historical read inputs here are explicit retained-state snapshots, not a
claim that every algorithm/step accepts every temporal/graph/table domain.

API and executable examples: `docs/api/typed-pipeline-values.md`. Both local and
outer atomic modes, persistent/transient calls, late source/type/selection errors,
empty values, caller-program copying, cancellation and immutable result lifetimes
have focused native/Python regressions. Full gates: **1,543 native passed / 54 ignored**, strict root all-target Clippy,
rebuilt release in-tree extension, **2,611 maintained Python passed / 1 skipped**,
and **260 executable documentation examples**. Three existing FFI warnings remain.
Scoped whitespace checks pass. Fixed checklist: **215 checked / 53 open / 268 total**.

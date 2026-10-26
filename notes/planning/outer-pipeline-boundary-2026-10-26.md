# Outer property pipeline and selected-import boundaries — 09.4b.2

Completed two components within the existing native boundary inventory. No new
checklist rows and no claim that the entire native migration is finished.

## Whole-pipeline acceptance

- Algorithm capability defaults to false. Certified builder algorithms prove all
  descendant steps use the property overlay; all outer algorithms certify before
  execution. Uncertified algorithms/effects reject before earlier steps can write.
- Native run_atomic_properties and Python Pipeline.run/call, module apply,
  Graph.apply and Subgraph.apply accept explicit atomic=True. Profiles report it.
- Persistent children share the outer batch irrespective of their local atomic
  flag; later algorithms read staged node/edge values. The outer operation accepts
  once with ordinary GraphBatch clocks. Scope and graph ownership remain checked.
- Runtime/final validation failure, owner changes and cancellation (including
  after the last algorithm) discard all staging. Context pending state and active
  step state clear. Existing outputs survive errors; partial outputs are discarded.
  Successful outputs merge. A native Context/compiled pipeline can be reused.
- persist=False keeps existing transient original-graph reads and owned outputs;
  empty/read-only runs create no version. Normal execution keeps its prior
  per-algorithm publication policy.

The supplied compiled step order is the execution contract. Mutation-sensitive
Python children should use build(atomic=True) or disable optimization, since outer
execution cannot undo earlier optimizer reordering. AR7.8 still owns the complete
effect-aware IR migration. This boundary does not certify unsupported topology,
streaming, classic algorithm wrappers or durable pipeline execution. GIL ownership
constraints remain those of the existing Rc graph assembly.

## Complete materialized import baseline

Graph::from_snapshot and lightweight temporal as_subgraph now stage complete
selected topology/properties as pending manual input. Checked publication rejects
until that baseline is accepted. Manual commit records the complete base; explicit
into_publishing re-seeds the selected current state and imports it once with new
acceptance clocks. Retained indices are reused, untouched values survive later
history reconstruction, allocation marks survive and malformed endpoints reject.
Owned to_graph already restores accepted history and is unchanged by this rule.

## Verification

- **1,205 native tests passed / 51 ignored**, including four outer failure/cancel/
  unsupported/owner/output tests and three real selected-import regressions.
- **1,948 Python tests passed / 1 skipped**, including 15 outer pipeline cases;
  **146 documentation examples** execute.
- Strict all-target Clippy, in-tree FFI build (three existing unrelated warnings),
  scoped Rust formatting, Python syntax and diff checks.

## Fixed remaining native boundary inventory

| Existing component | Status |
| --- | --- |
| Checked ordinary/bulk setters and scalar/mixed batches | Complete for documented checked paths. |
| Compound composition/copy and table writeback/materialization | Complete within checked/typed supported scopes already recorded. |
| Certified builder property families and whole outer property pipeline | Complete with explicit atomic mode; broader IR/default policy remains AR7.8/30.1. |
| Selected snapshot import baseline | Complete; raw persistent archive/import remains 10.3. |
| Native default/infallible constructor migration | Open: legacy new/add_node/add_nodes remain manual compatibility; fallible publishing alternatives exist. |
| Mutable raw pool/space escape audit | Open: exclusive ownership alone does not certify publication or history completeness. |
| Uncertified/topology and other multi-stage algorithm acceptance | Open: certify/migrate actual effects; no blanket atomic claim. |

Checklist counts remain **199 checked / 69 open**, since the 09.4b.2 parent still
contains those three open components. This is a fixed remainder, not new audit rows.

# Next execution slice: durable pipeline preparation boundary

Context, 2026-11-16. The first extraction below is now implemented; see
[owned pipeline effects](owned-pipeline-effects-2026-11-16.md). Durable dispatch
and its invocation adapter remain open. Selected materialization is now also
extracted; see [immutable pending reads](immutable-pending-materialization-2026-11-16.md). Existing scope: 09.4b.2, 30.1, AR3/AR7.8 and the
persistence boundary. This does not add a checklist row or claim implementation.

## Current source boundaries

- `src/algorithms/pipeline.rs::Pipeline::run_atomic_properties` validates effects
  and source binding, creates one PropertyBatch, runs steps against one Graph
  owner, then calls accept_selection. Output restoration and Context cleanup cover
  execution/publication failure. The returned continuation belongs to that owner.
- `src/algorithms/property_batch.rs::PropertyBatch::accept_selection` immediately
  accepts, then projects original and added IDs against the accepted owner.
  `accept` validates the WriteCoordinate, moves staged values/clocks/topology into
  one GraphBatch, and calls Graph::apply_batch. Batch conversion and backend
  dispatch currently share this function.
- `src/storage/journal.rs::DurableGraph` owns a Graph directly. graph() exposes
  checked shared reads only. apply_batch_with_schema encodes the stored batch,
  prepares canonical publication with installed/additional policy, and executes
  through the journal's persist/sync/publication boundary. It deliberately exposes
  no mutable Graph handle.
- `python-groggy/src/ffi/api/durable.rs` marshals native batch/view operations.
  It is not the place for an algorithm executor or an alternate mutation policy.

## Completed first implementation step

Extract the existing owned PropertyBatch-to-GraphBatch conversion from backend
acceptance. An internal prepared-effects value should retain:

1. The expected WriteCoordinate, including owner/branch/head binding.
2. The owned canonical batch payload with explicit property/topology clock choices.
3. Selection projection information, including global added positions and whether
   topology changed. This is pending plan information, not an accepted source pin.

Ordinary in-memory acceptance should consume that value through the existing
Graph coordinator, then project the accepted selection. Preserve empty-run behavior,
one publication/default engine clock, value movement, source checks and output
rollback. No new public facade is needed for this first refactor. Existing native
classic/root-loop/clock fixtures and application workflows are its finite exit.

This seam is useful preparation for durable dispatch; it does **not** make durable
execution complete. Do not expose a prepare method as safe durable execution until
the input/read-owner issue below is handled.

## Read-owner issue for the following step

Algorithms currently receive Subgraph, whose graph() gives an Arc<OwnerCell<Graph>>.
PropertyBatch staging validates against that owner, and retained snapshot/window
reads use its accepted history. DurableGraph cannot safely lend the same mutable
owner outside its publication path. A deep cloned scratch Graph would copy history
and obscure provenance/cost; merely moving the Graph temporarily into an Arc does
not guarantee recovery if continuations, outputs or callbacks retain that Arc.
Returning a writable Subgraph from durable execution would also expose a route
around journaling. Preserve this boundary rather than adding an unsafe mutable
escape or silently accepting a detached numeric-ID match.

The next backend adapter must provide the selected immutable read/history and
coordinate/allocation authority needed by existing checked execution without
publishing through a scratch owner. Reuse existing GraphView/AlgorithmRead/selected
CSR and shared step kernels. Retain explicit accepted-versus-pending history rules,
fixed frame confinement, global allocation references and retained group identity.
Specify supported/certified effects before dispatch; unsupported callbacks stay
explicit. Return immutable result views/owned results after durable acceptance.

## Durable acceptance exits

- One stored operation and one visible publication for a supported compound run;
  preserve installed/additional schema checks and the existing sync-before-visible
  rule. Preparation must not publish or allocate accepted IDs.
- Same branch/head/owner guards and exact selection/typed values/clocks as memory
  execution. Preserve default clock semantics; no fabricated intermediate history.
- A late validation, execution or cancellation error leaves graph, journal, IDs and
  prior outputs unchanged according to the existing coordinator contract.
- Journal append/sync failure follows existing poison/reopen behavior; do not
  report success or hand out an accepted write source after an uncertain append.
- Reopen/archive/backup retains the accepted operation and ID high-water counters.
  Exercise existing native journal failure injection and the archive/journal
  application workflow, plus selected classic/root-loop/attachment-clock examples.

These exits restate existing contracts. Service concurrency, new durability levels,
checkpoint scheduling and retained pipeline-run history are not decided here.

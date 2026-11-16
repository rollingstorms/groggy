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
- `PropertyBatch::prepare` now moves values/clocks/topology into owned effects
  independently of acceptance (`property_batch/prepared.rs`). Acceptance validates
  the complete coordinate and dispatches to Graph::apply_batch, then projects
  original survivors and global added ranges. Selected read materialization now
  lives in `property_batch/capture.rs`, independent of Graph/Subgraph access.
  capture_scoped retains the checked live-owner adapter.
- `src/storage/journal.rs::DurableGraph` owns a Graph directly. graph() exposes
  checked shared reads only. apply_batch_with_schema encodes the stored batch,
  prepares canonical publication with installed/additional policy, and executes
  through the journal's persist/sync/publication boundary. It deliberately exposes
  no mutable Graph handle.
- `python-groggy/src/ffi/api/durable.rs` marshals native batch/view operations.
  It is not the place for an algorithm executor or an alternate mutation policy.

## Completed first implementation step

The completed PropertyBatch-to-GraphBatch extraction separates conversion from
backend acceptance. Its internal prepared-effects value retains:

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

## Next coherent implementation boundary (design, not implemented)

The remaining coupling is invocation, not snapshot copying. Public
Algorithm::execute takes/returns an owner-bound Subgraph; Step::apply receives
StepScope, whose subgraph() and property/adjacency helpers are also owner-bound.
StepValue::SubgraphArray retains that ownership even though GraphView/table outputs
are detached. Both registries currently instantiate those public trait objects.
Atomic effect certification proves staging/pending-read behavior, not portability
to an owner-free durable invocation. Do not treat its existing flags as proof that
a callback cannot reach a mutable owner.

| Responsibility | Existing authority | Required next adaptation |
| --- | --- | --- |
| Source identity/selection | execution_scope, write coordinates, bound source validation | Capture a checked immutable source, complete coordinate and allocation high-water authority at entry. Recheck at acceptance. |
| Selected pending values/geometry | PropertyBatch capture kernel | Feed explicit frame membership and the accepted pin directly; retain dynamic-root/fixed-child rules and issued-capture authority. |
| Core step invocation | StepExecutor + public StepScope adapter | Introduce an internal read-oriented frame used by certified kernels. Keep public owner-bound adapters for compatibility; no dummy owner or panicking “subgraph unavailable” fallback. |
| Registered/custom effects | Step/Algorithm registries and effect preflight | Bind portable kernel capability explicitly in internal registration. Legacy/custom callbacks remain memory-only unless migrated; reject unsupported durable programs before a prefix executes. |
| Computation | AlgorithmRead, selected CSR, existing scalar/table kernels | Reuse these kernels; migrate leaf invocation, not arithmetic/algorithm definitions into a second interpreter. |
| Retained-history reads | Existing recorded/effective read plans | Lend retained read authority independently from the pending snapshot; do not invent history for provisional entities. |
| Acceptance | Owned effects + canonical durable journal coordinator | Check installed/additional schema, coordinate/allocation authority and complete output preparation before persist/sync/publication. |
| Output/lifetime | Immutable GraphView, table and owned result variants | Durable results retain no writable Subgraph; older captures remain immutable through later writes, reopen and pruning. |

Keep one executor for sequencing, refresh, cancellation and outputs. An internal
registered-kernel adapter can preserve public Step/Algorithm interfaces while
migrating built-ins incrementally. Merely duplicating the JSON step dispatch in
DurableGraph would create another execution engine. Merely adding an optional
owner to StepScope while leaving subgraph() callable would preserve the escape or
introduce runtime panics. Neither solves this boundary.

### First complete durable program target

A bounded end-to-end target can combine node/edge creation, explicit batch-global
references, scalar property initialization/attachment with effective clocks, an
immutable pending capture, one existing captured AlgorithmRead computation and
owned table/graph outputs. Run the identical program through memory and durable
adapters, including late failure, cancellation, installed schema, append failure,
reopen and allocation checks. Extend supported kernels only after that whole
program passes. Fixed child topology, legacy callbacks, alternate batch plans and
unported retained-history helpers keep explicit preflight guards; do not infer
support from matching IDs or a successful pure computation.

### Preserve the existing cascade distinction

DurableGraph::apply_view_batch_with_schema already offers checked view writes and
prepares its returned immutable view before journaling. Its strict view scope can
reject removal cascades outside the view. Existing selected atomic **pipelines**
permit a selected node's canonical owner-wide incident-edge cascade, even when an
incident edge is outside their output selection. The new selected endpoint fixture
in tests/test_staged_revision_endpoints.py exercises this: removing node 1 also
removes its owner edge to unselected node 3, while a revised selected edge is
rescued to new nodes. Returned membership remains selected.

A durable pipeline cannot silently substitute the strict view-writer policy and
reject that valid program. Reuse canonical batch preparation/publication and its
schema checks, with a separately checked invocation selection and prepared output
projection; do not add a generic “skip validation” switch. This is preservation of
current behavior, not a new cascade rule. Output materialization must finish before
journal acceptance so a fallible capture cannot turn an accepted write into a
reported execution failure.

The first adapter need not promise GIL release. Detached execution may release it
only after its complete input/context/output types carry no live owner and satisfy
compiler-checked thread/lifetime constraints. Existing typed values include both
detached outputs and owner-bound subgraph arrays, so qualifying one read kernel
alone does not qualify the entire invocation. Service scheduling, pipeline-run
history and new checkpoint/durability policies remain separate existing scope.

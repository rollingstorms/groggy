# Compound collection collapse — 2026-11-05

Completes the collection-collapse component of existing **09.4b.2 / AR3**.
No checkbox rows are added. Generic multistage topology pipelines remain open.

## Change

ComponentsArray.collapse and SubgraphArray.collapse now prepare all declarative
MetaNodePlans against one captured owner state, validate all effects and retained
groups, then use one mixed GraphBatch acceptance. Previously a later failure could
leave earlier components collapsed; frozen subgraph arrays could also fail on the
second selection after the first publication made its source stale.

Rust owns aggregation, source/owner validation, planned identity resolution,
boundary generation, group preparation and publication. FFI parses shared options
and marshals results. Single and collection paths share the same preparation and
application code. Each new meta-node keeps its own selected immutable source.
Every generated boundary endpoint and its source/target metadata uses the right
planned meta ID. No whole working graph or history is cloned for staging.

Publishing mode creates one state. Manual mode stages one complete operation and
leaves acceptance to an explicit checkpoint. Empty collections create no state.
A collection is evaluated against its original source, never against an earlier
plan's unpublished effects. Stale/mixed-owner sources, overlapping destructive
selections and boundary targets removed by another plan reject before effects.
Cross-plan destructive rewiring requires an explicit future policy; this API
makes no implicit choice. Identical retained revisions deduplicate; conflicting
staged group references reject before installation, with capacity reserved for
the complete group batch.

## Completion boundary

This does not add arbitrary custom callbacks, streaming topology mutation or
read-after-topology-write pipeline stages. Existing property pipelines still use
their certified overlay. Topology changes still use checked declarative batches;
custom algorithms/steps remain uncertified by default. Native capability opt-in
is a trusted extension contract, not an isolation sandbox for arbitrary callbacks.

Existing **09.4b.2 / AR3 / 30.1 / AR7.8** therefore remain open. The separate query
exit review closed 21.2: fixed checklist **215 checked / 53 open / 268 total**.

## Verification and bounded release evidence

- Full `cargo test --offline --locked`: **1,521 passed / 54 ignored**.
- Strict root all-target Clippy: **passed**.
- Rebuilt release in-tree extension; focused collapse/composer/default-view tests:
  **46 passed**.
- Maintained root Python gate: **2,581 passed / 1 skipped**; **256 executable
  documentation examples passed**. The separately maintained owner benchmark
  remains excluded by the existing verifier. Three pre-existing FFI warnings remain.
- `benches/compound_composition.py`, with builds quiet, compares the previous
  `22a05d11` extension against this change. Median wall time for 10 / 100 / 500
  disconnected pairs: **0.344 / 11.018 / 249.147 ms before**, **0.232 / 2.022 /
  19.219 ms after**. At 500 components this is approximately **13x faster**;
  acceptance count changes from 500 to one. Raw receipts include separate process
  CPU and wall samples: `notes/benchmarks/compound-collection-{before,after}-2026-11-05.json`.

Construction is excluded. This is a bounded synthetic topology/acceptance workflow,
not a general production throughput guarantee. Full cold capture and retained
selection materialization remain size-dependent, and boundary generation still
scans active edges per plan; this result does not establish linear scaling in
component count. Those costs remain in existing 32/30.1 work.

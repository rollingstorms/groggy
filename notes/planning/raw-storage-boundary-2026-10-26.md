# Raw owner mutation boundary and explicit pooled attribute failure

Follow-on to native default migration `914476f6` in the 11:18:59 UTC priority
window. Existing 09.4b.2 raw-storage component is complete; its fixed inventory now
has **two completed components and one remaining**. No checklist rows were added.
Counts remain **201 checked / 67 open**.

## Closed boundary

- `Graph::pool_mut` is crate-private, used by trusted prepared source installation/
  journal replay and internal tests. It is unavailable to external native/FFI crates.
- `Graph::space_mut` is a test-only corruption hook; normal builds do not contain it.
  Read-only pool/space inspection remains public. Graph/GraphWriter/GraphBatch own
  supported mutations, including explicit manual tracked edits.
- Audited reader returns: no public owner pool Rc/RefMut or mutable GraphSpace is
  returned. Immutable pool references and shared CSR/read payloads do not grant
  mutable access to the owner. Standalone low-level storage is a separate owner.
- Compile-fail Rustdoc cases verify that mutable Graph callers cannot bypass the
  boundary. No unsafe escape, untracked replacement API or opt-out was introduced.
- Ten native integration files needing deliberate corruption/raw legacy fixtures
  moved to internal unit modules, preserving **all 46 existing cases**. Imports
  change from external `groggy` to `crate`. Their new filter is
  `cargo test --offline --locked --lib api::graph::storage_boundary_tests`.
- The meta-creation benchmark now prepares its frozen retained source using the
  public Graph API. Source setup remains outside its creation timing; old membership-
  only receipts are retained as historical, not a matched speedup comparison.

## Emerging issue fixed inside existing scope

The raw audit found `GraphPool::set_subgraph_attribute` returning Ok without
storing anything; the corresponding getter always returned None. These operations
now explicitly return GraphError::NotImplemented. Generic GraphEntity pooled-group/
neighborhood/component/path/filter routes inherit the rejection and publish
nothing. Real node/meta-node/edge property writes still use the coordinator.

New public regressions verify typed failure/no allocation/no version plus separate
meta/subnode updates and the unchanged frozen source payload. This follows A01.1;
it does not introduce mutable group attributes or automatic source propagation.

API explanation: `../../docs/api/storage-access.md`.
Release meta receipt: `../benchmarks/sealed-meta-creation-2026-10-26.json`.
At 100k retained source nodes, median creation was **3.96 µs manual / 8.08 µs
publishing** in this local fixture. Source construction and output scans are
excluded; there is no speedup, RSS, FFI or whole-engine latency guarantee.

## Remaining fixed 09.4b.2 component

Uncertified/topology multi-stage algorithm acceptance. Certified property builders
and outer pipelines already use one staged acceptance and rollback. Requested
unsupported atomic families reject before execution. Classic wrapper/capability
coverage and topology effects still need completion; broader IR/default execution
policy remains AR7.8/30.1. Other liveness/view/index/conformance parents retain
their individual stated scope. Sealing raw access does not certify all of them.

Final verification totals are recorded in the priority-window summary.

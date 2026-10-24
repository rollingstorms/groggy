# 09.4b.2f / 30.1a — Honor nonpersistent builder attachment

Audit finding: the native builder's node/edge attachment and stroke smoothing
steps ignored Context.persist_results and mutated the graph during persist=False
execution. This violated the public execution flag and the accepted separation
of transient computation from explicit persistence.

## Implemented

- Shared StepScope attachment entry points choose ordinary checked persistence or
  canonical target/kind validation followed by owned output columns. Preserve the
  dedicated persistent node-column path and existing bulk multi-column setters.
- Both node-map/node-column attachment routes, edge attachment and smoothing use
  that native boundary. A disabled persistence flag records no properties,
  mutation versions or deltas. Manual dirty graphs retain their pending changes.
- AlgorithmOutput has typed NodeAttributes/EdgeAttributes variants. Python profile
  outputs expose sorted owned `(ID, value)` lists under separate node_attributes
  and edge_attributes property keys. Repeated keys retain the last emitted column,
  and empty attachment columns remain visible as empty results.
- Validate all columns of each attachment before emitting any, retaining inactive,
  duplicate, reserved-target and kind errors. Check cancellation before accepting
  or emitting. FFI uses the existing value bridge only.
- No full graph clone, private graph overlay or rollback was introduced. Graph
  readers within a nonpersistent pipeline read the unchanged graph; intermediates
  remain pipeline variables. These results are output columns, not private graph
  state or an implicit persistent log.

## Boundaries still open

Persistent multi-step execution can publish individual attachments before a later
step fails. True compound staging, read-after-write semantics, native/default
compatibility, source-state ownership and execution/lowering consolidation remain
in 09.4b.2 / AR7.8 / 30. Public apply wrappers retain their True default; completing
the already accepted transient-default migration remains work, not a new decision
gate. Pipeline runs/provenance are not automatically inserted into graph history.

The matched before/after performance investigation remains follow-up under the
existing performance gates. The owner requested continuing implementation
priorities first. This correctness repair does not claim a speedup: persistent
and transient runs intentionally do different storage/history work.

See [API and executable example](../../docs/api/transient-builder-outputs.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **932 passed,
  52 ignored**, zero failures across native/unit/integration/doc suites.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed.
- Rebuilt the in-tree development extension via offline locked `cargo rustc`
  with the existing macOS dynamic lookup linker options. Three pre-existing FFI
  warnings remain unchanged.
- **13 new Python transient-output cases passed**. Full root API gate excluding
  the unrelated owner algorithm benchmark: **1,076 passed, 1 skipped**. This does
  not include the separate legacy `tests/modules/` survey.
- **36 documentation examples passed**, including the new owned-output example.
  Storage examples ran in a temporary directory.
- `git diff --check` passed. Unrelated owner notes/benchmark data/tests remain
  outside this change.

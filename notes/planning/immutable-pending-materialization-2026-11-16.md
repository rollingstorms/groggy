# Immutable pending materialization boundary — 2026-11-16

Completed the read-kernel extraction following owned-pipeline-effects, within
09.4b.2, AR3/AR7.8 and AR6. Durable invocation/dispatch is still unfinished.

## One shared implementation

PropertyBatch::capture_scoped remains the owner adapter. It verifies owner
identity and the complete current write coordinate, captures/reuses the accepted
root, then delegates explicit node/edge membership and that immutable GraphView to
capture_membership. The latter owns all selected snapshot/topology materialization,
per-target validation and weak capture-authority registration in one internal
capture module. It cannot borrow or mutate a Graph and needs no Subgraph/owner
parameter. Fixed and dynamic frames use the same kernel as before.

No second execution path, writable scratch graph or durable owner escape is added.
The earlier column-copy optimization remains in this kernel. Current wrapper
coordinate validation, one accepted base, geometry/dirty-target cache rules,
retained group catalog and immutable provisional read semantics remain intact.
Preparation/acceptance stays in the separate owned-effects module. This moves
responsibilities out of the staging coordinator rather than introducing parallel
materializers.

## Regression evidence

Three new native tests cover:

1. A checked immutable durable source, staged node/revised endpoint/removal and
   typed properties, followed by closing the durable owner before materialization.
   The resulting view drives the existing AlgorithmRead BFS kernel; it has no
   write_state_id. Reopening succeeds while pins are alive, with identical journal
   bytes, head, accepted geometry and allocation. The reopened owner rejects the
   old coordinate despite matching numeric IDs/head.
2. Exact fixed membership, endpoint closure, added-node selection, dynamic-root
   expansion rejection, removal and immutable earlier output.
3. A staged property on a removed sibling rejects even when that sibling is absent
   from the requested child frame. Materialization cannot conceal invalid writes.

The focused PropertyBatch suite has **22 passing tests**. Full native, strict
Clippy and fresh Python/artifact qualification follow at the final runtime source.
No public API or storage-format change is introduced here.

## Remaining durable execution boundary

The selected read materializer and owned payload conversion are now independent
of mutable owners. Algorithm/StepScope invocation, source history, shared allocation
binding and output continuation still need a durable adapter that does not hand
out a writable Subgraph. It must use the existing journal/schema/coordinator
boundary, with persist/sync before visibility and the established poison/reopen
rules. This extraction alone does not authorize or implement durable pipelines.

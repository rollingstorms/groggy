# Existing-edge revisions to staged nodes — 2026-11-16

Completes a bounded topology composition gap under 09.4b.2, 03.2/AR2, 30.1,
AR3 and AR7.8. No new checklist rows; broad remaining requirements stay open.

## Implemented contract

`graph.set_edge_endpoints` accepts `[accepted_edge_id, source, target]`, where
an endpoint is either an accepted selected node ID or `{"added": position}`.
The latter refers to a node created earlier in the same atomic batch, spanning
multiple creation stages, outer builders and sequential root loop iterations.
It never means the last allocation or a guessed numeric provisional ID.

Creation and revision now share the exact endpoint-reference parser and selected
node resolver. The step resolves all rows before appending revisions. The existing
PropertyBatch uses resolved numeric positions and the canonical publisher retains
edge identity, properties, explicit/default clocks, duplicate/conflict validation,
endpoint lifetimes and cascade ordering. A revision can rescue an edge off a removed
node regardless of declaration order. There is still one final publication and
no transient accepted state. No FFI business logic, writable graph/history clone,
new public native facade or alternate executor is introduced.

The edge being revised must remain an accepted edge in the invocation selection.
Forward/malformed references, unselected accepted IDs, provisional numeric guesses,
invalid lifetimes, duplicate revisions and explicit removal conflicts reject.
New-edge revision/deletion and topology within fixed child frames remain guarded.

## Evidence

- Two new native tests exercise directed/undirected reconstruction, sparse IDs,
  multiple batch-global allocation positions, both cascade/revision declaration
  orders, typed property preservation, exact projection, immutable pending/prior
  captures, archive recovery and late-error output/ID rollback.
- **32 new Python cases** cover local/outer runs, selected node/edge confinement,
  whole-run failure, explicit clocks, forward references, numeric guesses and
  malformed controls even inside zero-iteration loops.
- Focused native endpoint/outer/root-loop suites: **19 passed**.
- Focused Python endpoint/root-loop/pending-path suites: **110 passed**.
- Rebuilt in-tree release extension, with the three established binding warnings;
  strict all-target Clippy passes. Final full gates are recorded in the slice
  closeout at their actual source.
- API examples and neighboring guards are updated in
  `docs/api/pipeline-endpoint-revisions.md`, `staged-topology-additions.md` and
  `root-topology-loops.md`. The new example joins the existing maintained docs gate.

An initial Python test supplied a detached GraphView directly to Pipeline, which
requires an owner-bound Subgraph. It was corrected to the existing induced_subgraph
API before qualification; no implicit writer was added to GraphView.

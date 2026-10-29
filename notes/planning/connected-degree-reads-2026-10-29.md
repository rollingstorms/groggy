# Connected CSR liveness and degree boundary — 2026-10-29

Scope: repair the closing reproduction in the previous priority window under
existing 26.1/AR1/AR2/AR7.5; no added checklist row or broad ownership closure.

## Completed implementation

- Canonical selected liveness/endpoint-closure validation now precedes both
  connected CSR cache reuse and rebuilding. Historical pool endpoints cannot
  reappear as current neighbors after a node/edge removal. Out-of-selection
  endpoint revisions reject as well. Revisions returning to a valid scope can
  read again; equal-version state restoration remains instance-sensitive.
- The native total-degree override now counts incident instances, including
  incoming directed edges and each loop once, matching Graph/GraphView degree.
  Directed neighbors retain their outgoing-traversal definition.
- The three Python degree adapters no longer contain independent per-node edge
  scans or skip invalid rows. They marshal into the native topology-only
  `SubgraphDegreeRead` and release the GIL for immutable row computation.
- Requested IDs preserve order/repetitions and reject as a whole outside the
  original selection. Implicit rows follow ascending node IDs. A stale selection
  rejects even an empty request and even explicit full-owner edge counting.
- Directed incoming counts use incident minus outgoing plus loops; undirected
  incoming/outgoing use incident-instance semantics, consistent with frozen views.
  This intentionally repairs the older stored-endpoint-only undirected adapters.
- Full-owner counts expand topology explicitly without expanding permitted result
  IDs or capturing properties/history. Local preparation shares existing CSR;
  prepared data retains no mutable owner and remains readable on a worker after
  owner mutation/removal/drop. First/invalidated topology preparation still runs
  on the owner thread; no claim that all preparation releases the GIL.

## Evidence and limits

Native regressions cover cold/warm removal, edge removal, closure-breaking and
restored endpoints, equal-numeric-version restores, noninduced parallel/loop
selection, isolates, ordered duplicate requests, bad IDs, empty input, full-owner
counting, worker lifetime and no publication. Python exercises all three modes
for both directedness and full/local domains plus stale and invalid input paths.
The user guide contains an executable current-vs-frozen example.

Release benchmark `connected_degree_reads` measures cold/warm topology preparation,
full-row incident counting and warmed scalar native neighbor/degree calls at
1k/100k chain nodes. It verifies counts and makes no before/after speedup claim.
No per-node whole-edge scan remains in the Python batch degree path. Warm local
preparation reuses instance/version validation and cached CSR; row lookup uses
sorted IDs. Explicit full-owner preparation currently creates temporary wider
membership/CSR and is not a persistent owner-wide degree cache.

Final gate counts and timing receipt are recorded in the priority window log.
The three pre-existing unrelated pipeline FFI warnings are outside this change.

## Still open

The compatibility selection remains owner-connected, so owner changes can
invalidate it. Automatic immutable default/filter returns, effectful facades and
specialized adapters retain their existing 26.1/21.2/AR4/AR7.5 scope. The new
prepared degree input is an immutable topology read, not a replacement history
view or an implicit request to mutate the owner.

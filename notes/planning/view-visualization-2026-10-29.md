# Selected immutable visualization — 2026-10-29

Existing 26.1/AR4/AR7.5 visualization adapter migration. Two actual pre-fix Python
server cases reproduced full-owner scope leakage and missing captured .viz.
GraphDataSource.from_view now consumes selected frozen snapshot buffers through
the same checked from_entities policy as table visualization. Connected .viz
captures/validates the selection and GraphView.viz retains original values; full
Graph.viz delegates through its full selection. No implicit edge induction or
source-group expansion. Payloads share only their own immutable nodes/edges and
stable process-local source identity; they retain no writable owner.

Two native cases verify both directions, parallel/loop exclusion, stable IDs,
exact f64 weights/positions, labels, empty/isolate selection, clone identity and
worker lifetime after graph deletion. Two isolated Python processes exercise real
HTTP/websocket servers and inspect /debug/snapshot after owner mutations, proving
only the selected node/edge identities reach the frontend. Focused existing/table/
source-identity suite: eleven cases. Full gate: **1,508 native / 54 ignored**,
strict all-target core Clippy, **2,567 root Python / 1 skipped**, **251 executed
docs examples**, rebuilt/atomically staged release extension and scoped checks.
Three old unrelated FFI warnings remain; owner benchmark/review work untouched.

Release absolute captured-source medians after separate source capture: sixteen
nodes cost **11.125 µs at a 1k owner / 11.292 µs at a 100k owner**; full source
construction costs **0.668 ms / 150.908 ms**. Five samples, directed chain and f64
node property. Accessor construction is included; server/layout/RSS and incorrect
old full-owner timing are excluded. Raw:
`../benchmarks/view-visualization-2026-10-29.json`.
API: `../../docs/api/view-visualization.md`.

No new checklist rows or parent closure: **211 checked / 57 open**. Remaining main
work is the default frozen read binding and explicit checked execution/write path.

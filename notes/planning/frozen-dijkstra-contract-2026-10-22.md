# Frozen checked Dijkstra — 2026-10-22

AR7.5g propagates the prepared immutable algorithm input into weighted distances.
AlgorithmRead/GraphView.dijkstra_distances return reachable (original node ID,
f64 distance) rows sorted by ID. They read selected captured topology and weights;
owner changes/destruction do not affect results. Python releases the GIL. Native
callers reuse prepared CSR and can supply an atomic cancellation flag. Weight
maps are validated per call; there is no hidden cache or history mutation.

Live and frozen readers share query/weights checked_pairs and one Dijkstra heap
kernel. The live adapter now borrows weight values from the pool and rejects
unreadable references. It retains active/closed selection checks. Pipeline marker,
telemetry and explicit f32 property persistence/nonpersistent f64 output remain
in their existing adapter, outside the frozen read input.

All selected weights are validated, even on unreachable edges: absent attributes
are one, zero is valid, and null/Boolean/text/vector/negative/nonfinite values
fail. Parallel weights collapse to their minimum for distance-only computation;
undirected pairs reverse and directed pairs do not. Output sums are f64 under the
existing numeric conversion contract, not arbitrary-precision integer sums or
matrix-export null/text conversion. Strict improvements and overflow errors stay
intact. Cancellation is checked during heap/neighbor work and before output;
weight validation/preparation has no bounded cancellation-latency guarantee.

Evidence: three new native fixtures cover all 512 small edge sets in both graph
directions/all sources with parallel zero/positive weights against Floyd–Warshall,
sparse IDs, selected excluded shortcuts, owner mutation/drop, worker transfer,
isolates, cancellation and unreachable invalid weights. Full locked offline
native suites and strict all-target core Clippy pass. Rebuilt Python weighted/
traversal/core/read/history/property regressions: 316 passed, five existing skips.
All seven documented read-view examples execute. Existing three FFI warnings
remain. Native release costs: `notes/benchmarks/frozen-dijkstra-2026-10-22.md`.

A*, components, centrality and remaining pipelines still need pinned-reader
migration. This slice does not choose workbook policies, implement checked view
writeback or complete arbitrary effective-time reconstruction.

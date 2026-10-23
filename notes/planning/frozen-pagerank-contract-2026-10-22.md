# Frozen PageRank — 2026-10-22

AR7.5j adds captured PageRank reads after components. Remaining closeness,
betweenness, registry and pipeline migration stays open under AR7.5/28.16.
No pending workbook policy is selected here.

## Shared computation

PageRank now owns one power-iteration implementation for live and frozen input.
Personalization scalar validation and normalization are shared. Prepared
AlgorithmRead reuses its captured CSR and row identity, reads personalization
from captured values without cloning property payloads, and returns owned f64
scores. GraphView convenience methods prepare per call. Python is a thin adapter
that releases the GIL and exposes immutable PageRankResult getters.

This remains unweighted PageRank: selected parallel edge occurrences contribute
separately; undirected edges are traversable both ways and loops occur once.
Preserve missing personalization=one, finite nonnegative numeric domain, finite
positive total, parameter validation, sink redistribution, uniform initial rank,
maximum-coordinate stopping criterion, final mass normalization and live
telemetry/output/persistence behavior. Ordinary property storage remains f32;
this slice does not implement the pending f64 property/default migration.

## Explicit result contract

PageRankResult contains node-ID ordered scores, converged, iterations and
max_update. The last value is the final maximum coordinate update before final
mass normalization: a stopping metric, not a certified fixed-point residual or
error bound. Reaching the iteration limit returns an approximation with false
convergence status. Empty input returns empty scores, true, zero and zero.
Invalid parameters still fail on empty inputs. Existing live output types remain
unchanged and still report convergence in telemetry.

Results survive owner mutation/drop. Reads publish no algorithm properties,
deltas, checkpoints or live caches. Cancellation is checked during captured
personalization collection and in the shared iteration kernel; no partial output.
CSR preparation, normalization, edge-row traversal inside a single row and output
allocation are not bounded interruptible. No new retention/publication decisions.

## Verification and timing

The existing direct Gaussian-elimination reference test now checks frozen scores
across all 512 directed and 512 undirected three-node graph masks, uniform and
nonuniform personalization, and damping zero/0.85. Additional native tests cover
selected parallel occurrences, excluded edges, frozen values, owner drop, empty
input, finite budget and cancellation. Python covers result ownership/immutability,
concurrent reads, missing/null/invalid/zero-total personalization, parameters,
isolates, frozen topology, finite-budget status and no attribute publication.
Public documentation includes an executable example.

Read-state Criterion adds uniform and personalized ring fixtures for prepared
and convenience reads, with a fixed 50-iteration budget and explicit convergence
metadata in the timing receipt. Timings do not establish production latency,
FFI overhead, allocation/RSS or a before/after speedup.

Executed: locked offline full `cargo test`, strict all-target `cargo clippy`,
in-tree FFI rebuild, Python read/algorithm/state/property regression gate
(**362 passed, 5 existing skips**), all **10** public read-view documentation
examples, and release Criterion timings. See
`notes/benchmarks/frozen-pagerank-2026-10-22.md` and
`benches/frozen_pagerank_2026-10-22.json`. Three pre-existing FFI warnings remain.
Black is unavailable locally; no package installs performed.

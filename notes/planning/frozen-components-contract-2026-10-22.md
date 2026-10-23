# Frozen connected components — 2026-10-22

AR7.5i migrates component reads onto the same captured selection/state used by
traversal and weighted paths. It implements the accepted A12.1 orientation
recommendation. Broader centrality/registry/pipeline migration remains open.

## Contract and architecture

- Native AlgorithmRead and GraphView expose connected_components; Python GraphView
  exposes connected_components(mode="undirected") with the GIL released.
- Native callers choose the existing ComponentMode enum. A shared native parser
  validates Python/registry mode strings, case insensitively. No FFI algorithms.
- Undirected (the preserved default) and weak ignore orientation. Strong respects
  direction; on undirected graphs all three modes give the same partition.
- Reuse the existing BFS and iterative Tarjan kernels, preserving live-wrapper
  telemetry. Captured reads have a local telemetry Context with no writer/cache.
- Prepared strong reads reuse CSR. Directed weak/undirected reads build symmetric
  CSR from selected captured edges. No owner mutation/version lookup occurs.
- Results are immutable GraphViews sharing the original pin. Components are
  ordered by minimum node ID; IDs within each view are sorted. Isolates remain
  singleton components; empty input yields an empty list. Group edges in one
  pass, preserving selected internal parallel instances and loops and excluding
  crossing edges. Nested selection cannot reintroduce parent-excluded edges.
- Reads publish no attributes, deltas, checkpoints, or live component caches.
  Mutating/dropping the owner does not alter outputs or captured properties.
- Native cancellation checks entry, kernel traversal, edge grouping and final
  output; no partial result. Preparation, sorting and membership allocation are
  not bounded interruptible. Kernels use u32 labels; captured reads reject node
  counts at or above u32::MAX. No production graph-size guarantee is implied.

## Verification

Native regression compares all three modes with independent transitive closure
on 256 edge subsets in each orientation (512 graphs), including parallel edges,
loops, isolates and owner changes. Additional native coverage checks empty input,
selection closure and cancellation. Python checks mode validation, captured
property/membership lifetime, selected parallels, owner drop, empty/singleton
inputs and concurrent reads. Full core tests and strict all-target Clippy plus
in-tree Python read/algorithm regression gates are recorded with this change.
The public component example executes alongside the existing read-view examples.

Criterion read_state adds prepared/convenience weak/strong component cases on
the existing 1k/10k undirected ring fixture. See the benchmark receipt/note for
bounded measurements; this is not a directed-mode, fragmented-graph, FFI, memory
or production latency benchmark.

Executed: locked offline full `cargo test`, strict all-target `cargo clippy`,
in-tree FFI rebuild, Python read/algorithm/state/property regression gate
(**327 passed, 5 existing skips**), all **9** public read-view documentation
examples, and release Criterion component timings. Three pre-existing FFI
warnings remain. Black is unavailable locally; no package installs performed.

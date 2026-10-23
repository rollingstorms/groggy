# Frozen betweenness — 2026-10-23

AR7.5l migrates node Brandes betweenness onto captured AlgorithmRead/GraphView
input. Live and captured callers share shortest-path enumeration, dependency
accumulation and normalization; Python marshals owned f64 rows with the GIL
released. This completes the PageRank/closeness/betweenness read slices, not broad
algorithm/registry/pipeline migration or AR7.5 as a whole.

## Preserved rules

- Directed outgoing paths; undirected reverse occurrences with raw ordered-pair
  accumulation divided by two. Explicit normalization retains the established
  directed/undirected factors and includes selected isolates in n.
- Unweighted calls reuse prepared CSR and every selected parallel occurrence.
  Explicit weights use shared checked minimum-pair lengths. All selected values,
  including disconnected edges/loops, must be finite strictly positive numeric
  lengths; absent is one. Keep every minimum-length parallel instance in a private
  weighted CSR. Expensive parallel instances are excluded, and positive self-loops
  never enter a shortest path. Empty/isolate selections still return correct rows.
- Captured topology/properties and parent-selected edge scope remain unchanged by
  owner mutation/drop. Outputs are owned node-ID ordered f64 rows; reads publish
  no attributes, deltas, checkpoints or live caches.
- Cancellation now polls inside shortest-path BFS/Dijkstra and dependency
  accumulation as well as source boundaries. No partial output. CSR preparation,
  weight validation and individual adjacency/predecessor row scans remain not
  bounded interruptible. Existing path-count/distance representability errors
  remain explicit; missing checked weights now error instead of fallback.

## Necessary repairs found during migration

The live nonpersistent wrapper discarded computed scores. It now emits the
existing AlgorithmOutput::NodeScores under centrality.betweenness.scores, matching
other centralities. f32 output-range checks apply to persisted properties only;
transient/native rows retain f64. Preserve existing live telemetry and explicit
persistence behavior. Nonfinite/negative accumulated results still fail.

Replace the indexer's unchecked ID-span and dense-row narrowing assumptions with
checked arithmetic and sparse fallback, covering zero/usize::MAX and dense IDs at
the upper boundary. Replace generated-node index unwraps with explicit errors.

## Verification and timings

Independent simple edge-instance path enumeration checks captured directed and
undirected multigraphs across 256 masks, both weight modes and normalization
modes, plus the existing larger parallel/tied/cyclic fixture. Native/Python tests
cover retained values/scope, owner drop, loops, invalid lengths, isolates, empty
input, cancellation, concurrent reads, owned output and transient live rows.
All-source unweighted/explicit-unit-weight Criterion fixtures use selected
100/1000-node paths. Receipt and gate results follow below. No production scale,
FFI, memory or before/after speedup claim.

Executed: locked offline full `cargo test`, strict all-target `cargo clippy`,
in-tree FFI rebuild, Python read/algorithm/state/property regression gate
(**411 passed, 5 existing skips**), all **12** public read-view documentation
examples, and release Criterion timings. Receipt:
`benches/frozen_betweenness_2026-10-23.json`; note:
`notes/benchmarks/frozen-betweenness-2026-10-23.md`. Three pre-existing FFI warnings
remain. Black is unavailable locally; no package installs performed.

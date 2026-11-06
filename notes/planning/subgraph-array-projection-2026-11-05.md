# Captured SubgraphArray projections — 2026-11-05

Repairs the existing **26.1 / AR4 / AR7.5 / 27.2** source-binding requirement.
No new checklist row or ownership model is introduced.

A reproduced regression: capture a neighborhood array with x=1, change owner x
to 9, then request array[["x"]]. The old helper returned 9. Both the single-column
and table-column helpers bypassed the captured source and read current owner
mappings directly; lookup failures were converted to nulls or discarded.

Collection projection now captures each selection's immutable native read source
and delegates columns to canonical entity readers. Table columns align with
ascending original node IDs and preserve missing cells as null. Native
GraphViewEntities.present_column provides ordered present observations, retaining
explicit null while omitting absence. The legacy array["attribute"] API keeps
its node-first/edge-fallback behavior and present-observation shape. Unknown
source attributes raise KeyError even if added later to the owner. Table result
types and detached owned-column semantics are preserved. Column/table assembly
uses the captured data with the GIL released; only Python wrappers are built
under the GIL. Array.table no longer silently skips failed selections.

Focused regressions cover sorted sparse IDs, aligned columns, absence versus
explicit null, node/edge fallback, owner edits/removals and retained meta-node
sources after original nodes are removed. The native present-column test also
keeps its capture after owner changes. Existing numerical and pipeline boundaries
remain unchanged. Generic eager graph iterator placeholders remain separate open
work under 30.1/AR7.8, as documented in the preceding lazy-array receipt.

The same fixed reader inventory contained current-owner bypasses in merge and
visualization. Both now use native GraphView.merge_all: validate one common
capture, union exact node/edge membership once, preserve direction/values/clocks
and retained groups, and never induce extra edges. Owned merge delegates to
existing checked graph materialization, preserving sparse IDs without filler
nodes and retaining meta payloads. Visualization consumes the captured union
directly; no temporary working graph is assembled. Zero-item arrays retain the
existing fresh default-empty graph behavior because they have no source
coordinate. Nonempty arrays of empty views keep their capture.

Collection density summaries delegate to the existing native directed/undirected
formula. Node/edge table and grouping errors propagate; invalid grouping domains
reject rather than return a misleading empty collection. Component callback and
truth-conversion failures propagate; callback side effects themselves are not
rolled back. The older component neighborhood convenience method and generic
eager marker-trait iterators are still unfinished in existing 30.1/AR7.8; this
repair does not claim to implement those unrelated transforms.

## Verification and measured costs

- Full native gate: **1,532 passed / 54 ignored**; strict root all-target Clippy
  passed. Release extension rebuilt in tree; three pre-existing FFI warnings.
- Focused collection/entity/visualization suite: **30 passed** before the final
  late-callback regression. Maintained root Python gate including it:
  **2,591 passed / 1 skipped**. **257 executable Python documentation examples**.
- Bounded quiet release comparison, five samples with captures held and fixture/
  validation excluded, separate wall/process CPU in raw receipts:
  `notes/benchmarks/subgraph-array-projection-{before,after}-2026-11-05.json`.
  For one 100k-node group, two-column table projection improves **35.594 →
  11.132 ms** median wall and observations **29.446 → 6.875 ms**. Parallel
  projection uses more process CPU than wall time.
- At 100 groups of 1k nodes, table projection changes **50.646 → 56.319 ms**,
  observations **14.071 → 39.848 ms**. The captured reader now enforces source
  values and ascending IDs. This fixture exposes a real throughput regression;
  repeated ordering/row snapshot access and group scheduling are candidates for
  profiling, not established causal explanations. It remains existing
  32.2/AR7.5 performance work, without a new checklist row. No universal speedup,
  memory or production throughput guarantee is claimed.

Fixed reader inventory closed here: table-column extraction, present observation
extraction, merge and visualization. Summary density and error propagation reuse
existing native contracts. Other transform placeholders are named above; this
receipt does not count them as implemented.

Follow-up: the component neighborhood convenience placeholder named above is
completed separately in `component-neighborhood-defaults-2026-11-05.md`.

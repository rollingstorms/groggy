# Specialized selection consolidation — 2026-10-29

Existing 26.1/AR2/AR7.5 component. No new checklist row or broad parent closure.

## Confirmed defect

The pre-fix native regression creates owner routes 0→1→2→4 and 0→2, then selects
nodes {0,2,4} with only edge 2→4. FilteredSubgraph::bfs(0) returned {0,2,4} because
it traversed the owner before filtering. Its correct selected result is {0}.
Component and neighborhood implementations had parallel owner-traverse/filter
paths; the component implementation also returned itself without checking actual
connectivity. Specialized induced-node scans could omit parallel instances and
loops. These are actual selected-scope defects, not absent core algorithms.

## Implementation completed

- FilteredSubgraph, ComponentSubgraph and NeighborhoodSubgraph now contain one
  canonical Subgraph selection plus metadata. Separate mutable owner/membership
  fields and duplicate traversal implementations are retired.
- Neighbors, total degree, components, BFS/DFS, shortest hop paths, induction and
  edge selection share canonical validation and selected topology. Cached scope
  and CSR validity inherit the prior restoration/removal fixes.
- Derived selections retain specialized wrapper type and original filter/analysis/
  center context. Component largest flags remain true only when the derived node
  membership retains the original full largest component; smaller results do not
  inherit that claim. Neighborhood IDs retain the existing center/hop/count rule,
  and sampler-specific IDs remain unchanged.
- Internal typed result helpers move canonical Subgraph membership directly into
  metadata wrappers. They avoid cloning output hash sets through trait objects
  or building a second graph/capture. The public SubgraphOperations API is stable.
- Explicit empty node/edge selection returns a valid empty result consistently;
  older filtered/component code rejected some empty results. Induction retains
  every selected parallel instance and loop and cannot expand parent edge scope.
- Canonical connected induction/edge selection now validate and derive topology
  directly. They no longer materialize full owner property values just to return
  connected membership. Induction scans selected edges once; explicit edge
  selection resolves requested selected instances. This avoids introducing the
  existing cold value-capture cost into specialized subset operations.
- Filter complement prunes edges with endpoints outside complementary nodes.
  This prevents its output from violating the canonical closed-scope invariant.
- Native as_subgraph() borrows the shared selection. Clones share Arc membership;
  Python neighborhood conversion now clones the canonical handle rather than both
  sets. Existing neighborhood type labels remain unchanged.
- Native pin_view() captures selected state for all three types; the extension's
  ComponentSubgraph/NeighborhoodSubgraph adapters expose it with curated stubs.
  Public neighborhood-array items retain their existing base Subgraph contract.

## Verification and performance boundary

Native regressions exercise all three variants in both directedness modes,
excluded nodes/instances, actual disconnected components, parallel/loop induction,
empty/invalid selection and address behavior, same-node paths, metadata/clone
sharing, warm removal rejection, frozen property/topology lifetime and workers.
The public Python neighborhood bridge is exercised in both directions, plus
extension adapter registration. Existing exhaustive canonical traversal tests and
previous degree/cache regression suites remain in the full native gates.

Release specialized_selection_scope benchmark measures 1k/100k chains, checksum-
verified warm three-hop traversal across base/filtered/component/neighborhood
inputs, four-node induction/three-edge selection, shared filter cloning and
current wrapper struct sizes. Absolute timing
receipt is under notes/benchmarks. No old owner-wide traversal is used as a
correctness reference and no universal speedup is claimed. The canonical wrapper
adds cache/order metadata to small objects; shared membership avoids full set
copies on clone. First validation/topology preparation still scales with selection.

Full gate counts and benchmark medians are recorded in the priority-window log.

## Remaining original scope

Specialized wrappers still connect to the current owner by default. Capture is
explicit, and the captured GraphView is readonly; metadata-specific filter
reapplication/neighborhood expansion still mean explicit current-owner work.
The legacy defaults/filter return migration, effectful facades, complete viz
adapter migration and general compound topology transaction remain their existing
26.1/21.2/AR4/AR7.5/09.4b.2 boundaries. This component does not close those parents.

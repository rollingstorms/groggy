# 07.3a.2d — Preserve checked algorithm output precision

Dijkstra, closeness, Brandes betweenness and PageRank already compute owned f64
scores. Their mutable wrappers now write Float64 properties rather than rounding
to f32. Finite values above f32 range no longer fail merely because persistence
was selected. Keep existing selection/weight/cancellation/domain validation;
nonfinite computed results still reject before output. No new history publication
or durability contract is introduced by changing result value tags.

Canonical EdgeOperations weight/capacity access accepts stored f64 in addition to
legacy float/integer values. set_weight preserves its declared f64 input. Remove
identical compatibility-edge overrides so legacy and canonical handles inherit
one policy. Absent/nonnumeric attributes still return None; algorithm preflight
retains stricter domain validation rather than treating malformed weights as
missing. This does not promise recovery of precision already lost in f32 inputs.

Native/Python regressions compare stored scores with owned outputs exactly,
exercise large Dijkstra/closeness persistence and verify both edge handle surfaces.
Existing independently referenced algorithms, selection/cache, cancellation,
invalid weights and frozen-read regressions remain gates. Broader step/vector/
display/CSV consumers and ordinary Python float inference stay open.

Verification: full locked offline Rust tests, strict all-target Clippy, in-tree
FFI rebuild and 199 focused Python algorithm/frozen/property/builder/pipeline
checks (2 existing skips). Executed 22 documentation examples; snippets expecting
`selection` use an explicit three-node positive-length fixture. Output-tag
assertions follow the new f64 contract, and the Dijkstra overflow fixture now uses
actual f64 overflow rather than the retired f32 storage limit. Three pre-existing
FFI warnings remain. No kernel optimization/performance gain is claimed.

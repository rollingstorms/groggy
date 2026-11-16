# Sequential root topology loops — 2026-11-16

## Completed bounded execution slice

Root iter.loop bodies can now repeat certified topology and pending-safe reads
through the existing StepExecutor and shared PropertyBatch. Nested root loops
retain global allocation positions, staged columns and one acceptance. No second
executor, writable Graph/history clone, new IR or per-iteration publication was
introduced. The common effect fold now has explicit root-loop versus fixed-child
policy. Legacy Properties mixed with repeated topology reject regardless of order;
later iterations cannot read stale owner-only state. Fixed for_each topology,
unsupported callbacks, streaming and alternate batch plans remain guarded.

Creation maps overwrite their named variable each iteration; literal added
references remain positions in the entire atomic batch, including earlier builder
allocations. Computed removal maps can select distinct surviving targets each
iteration. Repeated literal removals/revisions retain canonical duplicate rejection.
Existing removal/revision operations still require accepted owner targets; this
adds no provisional deletion/revision grammar. Final assignments and effective
clocks belong to one batch; intermediate iterations are not retained history states.

## Verification

- Four native regressions cover root/nested growth, warmed immutable input,
  global added positions, subsequent classic computation, duplicate-removal failure,
  prior output restoration, zero iterations, fixed-child preflight rejection,
  mid-loop cooperative cancellation, allocation reuse and Context reuse.
- 37 new Python cases cover local/outer atomic execution, direction, zero/one/many
  iterations, nested loops, selected/outside rows, global references across builder
  prefixes, different removal targets per iteration, late failure, duplicate
  revisions, child guards and explicit table clocks/archive recovery.
- Old unsupported-loop tests retain coverage with a genuinely unsupported legacy
  body; the two cases rejecting ordinary root topology were retired as obsolete.
- Full locked/offline native: **1,608 passed / 54 ignored / zero failed**, 224 suite
  results. Strict all-target Clippy, formatting and whitespace checks pass.
- Rebuilt in-tree release extension: **3,218 maintained Python passed / one
  skipped / 14 established warnings**, **286 executable docs examples**. CPU-sensitive
  gate ran after native builds/tests. No new artifact-platform or scale claim.

Early harness fixes use the existing public Step re-exports and remove an ambiguous
String conversion. The removal test respects the existing node-only top_k API;
edge removal uses the certified seeded edge sampler. No new edge top_k was invented.

## Emerging issue

The direct core.attach_node_attr/attach_edge_attr factories silently ignore an
`effective_at` parameter; table.attach supports it correctly. The explicit-time
loop regression uses the supported table route. Follow up with explicit timestamp
support and parameter validation at the direct attachment boundary, including
rollback, child confinement and later-write clock replacement. This is existing
09.4b.2 / 07.4 / temporal clock scope, not a new checkbox row.

## Checklist

Advances 09.4b.2, 30.1 and AR7.8 and conformance 01.2/19.2. Broad rows remain open
for child/provisional topology semantics, alternate plans/lowering, streaming,
durable dispatch and their other exit criteria. Fixed inventory remains
**215 checked / 53 unchecked**, including **44 open release rows** and nine deferred.

Follow-up complete: `builder-attachment-times-2026-11-16.md` implements explicit
direct attachment clocks and strict parameter checks through the same coordinator.

# Actual default component neighborhoods — 2026-11-05

Completes the existing component-neighborhood convenience path within
**26.1 / AR2 / 30.1 / AR7.8**. No new checklist row or policy question.

ComponentsArray.neighborhood previously called the neighborhood algorithm but
discarded its result and returned each whole component. Its "first" center was
HashSet iteration order and errors were silently skipped. The corresponding
legacy iterator convenience path had the same placeholder.

Both now share one translation helper and native GraphView.neighborhood_from_first.
The default first node means lowest original ID. Python convenience retains one
hop and the existing both-direction reachability rule. Native callers can pass
hop count/direction. Empty selected views retain their captured source/type;
empty component collections remain empty. Source values, membership and exact
edge instances remain immutable after owner changes. No graph mutation is
published. Native traversal runs with the GIL released; Python wrapping remains
under the GIL. Minimum-center selection scans component membership; this is not
an O(1) neighborhood-latency promise.

Regression coverage checks a component larger than its one-hop neighborhood,
parallel edges, immutable values after owner removal, zero-hop and directed
incoming/outgoing native controls, isolated nodes and empty captures. Generic
eager filter/collapse/expansion marker methods remain outside this completion;
they do not become implemented because this convenience path is repaired.

## Verification

Full native gate **1,535 passed / 54 ignored**; strict root all-target Clippy
passed. Rebuilt in-tree release extension retains three pre-existing FFI warnings.
Focused Python neighborhood/collection/specialized scope suite **14 passed**.
Final maintained root Python gate **2,593 passed / 1 skipped**, plus **258
executable documentation examples**. Final review also extends the existing
parallel captured-column test to verify ordered present observations, null and
absence above the parallel threshold; its targeted test and final Clippy pass.
Scoped formatting repairs the earlier shared collapse-option adapter indentation
without changing behavior.

Fixed checklist remains **215 checked / 53 open / 268 total**. This pass closed
21.2 and completed existing compound/numerical/read components without adding
repair rows. Remaining priority boundaries are generic topology/typed pipeline
execution, the measured small-group projection slowdown, wider numerical and
scale certification, and the already deferred adapter/service work.

# Selected graph materialization — 2026-10-29

Existing 26.1/AR4/AR7.5 conversion component. Six pre-fix Python cases reproduced
empty connected copies and the absent captured converter. Both now use native
Graph.from_view / GraphView.to_graph and existing snapshot installation, identity
validation, complete import staging and automatic publication. Selected IDs,
direction, exact typed values and referenced frozen group roots are retained.
Typed local references outside the selection reject; unrelated groups, source
history/inactive reservations and source engine configuration are outside a
selected copy. Empty copies publish no state; nonempty copies publish one new
initial state with new engine clocks. Full owned recovery remains the separate
owned-snapshot API. No writable owner alias or table/JSON intermediate is used.

Three native and six Python cases verify sparse IDs, parallel-edge exclusion,
exact f64/nulls/direction, independent later mutations/ID allocation, rejected
out-of-scope references, stale connected inputs, empty copies, nested source
lifetime and actual full-archive recovery. Full verification: **1,506 native /
54 ignored**, strict all-target core Clippy, **2,565 root Python / 1 skipped**,
**250 executed docs examples**, rebuilt/atomically staged release extension,
scoped format/syntax/diff checks. Three old unrelated FFI warnings remain. Owner
benchmark/review work is excluded and untouched.

Release Python absolute medians: **2.987 ms at 1k**, **595.559 ms at 100k** for a
directed chain with exact f64 node properties. Five independent copies; source
setup excluded and capture reported separately. Destination construction and one
accepted initial state are included. No comparison against the old incorrect
empty result, no zero-copy/RSS/GIL claim. Raw:
`../benchmarks/view-materialization-2026-10-29.json`.
API: `../../docs/api/selected-graph-copy.md`.

No new checklist rows or broad closure: **211 checked / 57 open**. Next: selected
visualization capture, then default frozen read/checked effectful facade migration.

# Whole-selection Python column facade — 2026-10-29

Existing 26.1 / AR4 / AR7.5 component, using the completed native column reader.

## Problem and implementation

Three pre-fix Python regressions showed unordered/misaligned connected columns,
unsupported immutable indexing, and removed selected IDs silently becoming nulls.
The FFI previously looped over HashSet node IDs, borrowed the owner per row and
converted every scalar read error to a null cell. That duplicated native read
policy and hid invalid identities/corrupt storage behind ordinary absence.

Connected Subgraph["property"] now delegates to the canonical checked Rust bulk
column reader. node_ids/edge_ids, nodes/edges and to_nodes/to_edges use native
ascending ID order, so a column can be zipped against its public row IDs. Missing
properties and explicit null still project to null without dropping rows; inactive
selected IDs and other read failures raise errors. Live ownership remains explicit.

GraphView["property"] delegates to its existing frozen node accessor column, with
GIL release and typed payloads. Its to_nodes/to_edges aliases return readonly
captured entity adapters. No owner clone, new mutation coordinator, intermediate
snapshot or second column engine is introduced. Returned arrays own their buffers.
Other legacy accessor-specific absence/type policies remain their original scope.

## Verification and performance

Eight new Python cases cover sorted full/subset node/edge adapters, absent and
explicit-null rows, wholly absent columns, empty views, invalid key types,
removed-ID rejection, exact f64/vector/string payloads and frozen owner lifetime.
The focused captured-entity/indexing gate passed 26 cases. Full root gate:
**2,545 Python / 1 skipped**, **247 executed docs examples**. Native core is
unchanged from the preceding full **1,498 native / 54 ignored** gate and strict
all-target core Clippy. Full-ABI release extension rebuilt/atomically staged;
three existing unrelated FFI warnings remain. Syntax/diff checks pass. Owner's
untracked benchmark/review work remains excluded and untouched.

Release-extension benchmark uses seven bound __getitem__ calls per size with a
homogeneous exact f64 column. Setup/warmup and to_list validation are excluded;
BaseArray construction is included. Before/after builds are separate quiet runs,
not alternating pairs. Homogeneous values avoid treating old unordered output as
a correctness reference for row identity.

| Nodes | Former scalar FFI loop | Checked native bulk read |
| --- | --- | --- |
| 1,000 | 0.103 ms | 0.113 ms |
| 100,000 | 32.798 ms | 23.702 ms |

The smaller case is slightly slower and the larger case improves on this fixture;
there is no universal speedup, memory or 100ns FFI claim. Both produce full owned
columns with linear work. Raw: `../benchmarks/view-column-indexing-2026-10-29.json`.
API/examples: `../../docs/api/connected-column-alignment.md`.

No new rows or premature closure: **211 checked / 57 open**. The remaining live
read/default/effectful pipeline migration stays in 26.1/AR4/AR7.5/21.2.

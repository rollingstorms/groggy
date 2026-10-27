# Three-way snapshot merge planning — 2026-10-27 UTC

Existing **14.2 component**, after endpoint revisions (04.2) and explicit retained
state selection (13.3). Native planning is complete; owner publication/Python/
durability remain separate required work. No new checkbox rows or inferred policy.

## Completed native component

- Replaced the public merge_snapshots panic with conflict-free three-way planning
  and explicit error return for conflicts. SnapshotMergePlan inventories typed
  base/left/right node, edge, property and endpoint conflicts. Separate IDs preserve
  parallel facts. Exact typed codec comparisons retain float widths, signed zero,
  NaN payloads, collections and group-reference IDs; absence differs from Null.
- Caller resolve_with accepts whole typed entity/property/endpoint answers. Invalid
  answer kinds, missing edge endpoints and endpoint-bearing node records reject.
  Caller failures cannot mutate input/owner state. Resulting topology is recomputed
  after direct answers, so provisional dependencies never silently choose policy.
  Dangling edges require explicit deletion or exact missing-node restoration;
  duplicates/extras reject, and no implicit cascade or ID allocation occurs.
- Inputs are borrowed while merging records; only output/conflict values are owned.
  Deterministic sorted ID/property unions are explicit costs. Resolution scans
  endpoints once and sorts only dangling-edge callbacks, avoiding per-active-edge
  temporary missing-node allocations and needless sorting of every edge ID.
- Completed the public structural validate_snapshot placeholder. Merge planning,
  retained integrity and state restoration share this invariant owner. Full raw
  pool/source/identity/schema/effective-time checks remain Graph responsibilities.
- Corrected common-ancestor selection: the previous purported BFS used a stack and
  could select an older common ancestor via a second parent. merge_bases returns
  all best common ancestors in sorted order, in ancestry/common-edge passes.
  Unique-base compatibility selection rejects ambiguous criss-cross histories.
  Unknown/zero ancestry coordinates reject, including equal unknown IDs.

## Evidence and limits

Eight native integration cases plus one executable Rust API example cover
independent column changes, null/absence, node/edge creation collisions and
removal/modification, divergent endpoints, parallel IDs, unchanged NaN bits,
signed-zero/type conflicts, malformed topology/records, explicit dependency repair,
callback abort, stale/unknown ancestry, older-DFS-base regression and multiple best
bases. The existing root Python and retained/recovery suites remain regression
gates after the shared structural validator changes.

Final gates: **1,351 native passed / 52 ignored**, including the executable
Rust example; strict all-target Clippy; rebuilt/staged in-tree extension with its
three existing unrelated warnings; **2,025 root Python passed / 1 skipped** and
**176 Python documentation examples**; serial SIGKILL parent **1 passed / 18 cases**.
Scoped Rust formatting and diff checks pass. Existing owner benchmark/nested
legacy survey exclusions remain in the Python gate.

Receipt: `../benchmarks/snapshot-merge-2026-10-27.json`, three local native release
samples. Chain with one integer column; alternating independent left/right property
changes; setup and input copies excluded. At 1k nodes, median planning **1.77 ms**
and final structural validation **0.047 ms**. At 100k, **274.29 ms / 6.72 ms**.
This measures pure logical planning, not owner publication, durability, FFI, RSS,
or a production budget. No overall-engine before/after claim.
The native logical planner is not an owning Graph merge API. Callers must supply
same-domain snapshots and a chosen base; the returned logical target ID reserves
nothing. No archive/journal schema change, external install or push.

## Fixed remaining 14.2 inventory

1. Prepared Graph capture and acceptance, guarding owner/head/branch/base and
   current pool/reference/external-identity/schema invariants.
2. Complete exact typed resolution delta relative to the first parent, explicit/
   default effective clocks and two-parent publication with one accepted version.
3. Durable synced merge records/replay/failure/reopen/backup/maintenance evidence.
4. Thin Python plan/conflict/resolution translation and writer APIs; conformance
   examples and shared branch/retention/effective-time checks at that boundary.

**205 checked / 63 open**, same 268 rows. 14.2 advances from open to partial; its
parent stays unchecked until owning acceptance and the public/durable surface
are verified. Tracked inverse-mutation policy remains separately open under13.3.

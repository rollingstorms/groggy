# Secondary indexes — 18.2 exit review

Completed the remaining retained group-membership family and reviewed the fixed
18.2 family inventory. The existing parent can close under the accepted immutable,
transient, explicit-build policy. No new checklist rows or lifecycle policies.

## Completed family inventory

| Existing family | Native owner and public surface | Evidence |
| --- | --- | --- |
| Selected/global property equality, presence and exact numeric/temporal ranges | `GraphQueryIndex`, named column indexes | `captured-query-indexes-2026-10-27.md` |
| Labels, relationship types, endpoints/incidence, neighbors and degrees | `GraphQueryIndex`, captured selection/topology postings | Same receipt; finite pattern integration in `graph-patterns-2026-10-27.md` |
| Per-change effective-time event windows and structural activation intervals | `EffectiveTimeIndex`, shared canonical replay lineage | `effective-time-indexes-2026-10-28.md` |
| Exact typed property-value/absence intervals, including explicit boolean validity | Property/lifecycle postings in `EffectiveTimeIndex` | `property-intervals-2026-10-28.md` |
| Retained group membership, nested group links and reverse root references | `GroupMembershipIndex`, explicit capture/build adapters | Current slice below |

18.3's owner-approved transient lifecycle applies to every family. Explicit
capture/build or rebuild handles changed/reopened state. There are no persisted
definitions, background updates, automatic recovery rebuilds or hidden mutation
maintenance. Managed allocation-lifetime external identities retain their separate
durable 18.1/item 10 contract. Existing primary historical identity scans are not
silently promoted to indexed queries.

## Group implementation and contract

The native owner is `src/query/groups.rs`, exported from the query module.
GraphView, Graph, DurableGraph and connected Subgraph expose
`build_group_index()`. FFI translates strict IDs/results and releases the GIL;
Python exports and stubs describe the same immutable object.

Build indexes typed scalar references in selected root node/edge properties and
visits each reachable frozen source revision once using an iterative work list.
It stores root property postings plus parent/child links and shares frozen source
views. Membership delegates to the existing frozen hash sets. It never joins
source-local node/edge numbers across revisions or follows ordinary integers.
Legacy aliases resolve within each source context to canonical content-addressed
revision IDs. Missing frozen legacy sources reject explicitly.

`group_ids`/`root_groups`, `group_view`, `children`/`parents`, qualified
`contains_node`/`contains_edge`, and direct/recursive `node_references` and
`edge_references` cover the family. Results are deterministic, owned and scoped.
Recursive reference lookup follows ancestor revisions, deduplicates root
entity/property pairs and does not enumerate every route. Unknown/out-of-scope
group revisions fail; unknown members of a known revision return false.

Old captures remain readable after source/root edits, deactivation, branch
switches, owner drop and durable close/reopen. Root effective clocks and separate
nested source pins are preserved. Read access to nested sources does not install
standalone writable references in the root pool; explicit retain is still needed
when reusing such a source in a new root property. The executable guide records
this existing distinction. Index build never changes pool registrations, heads,
history or durable bytes. Root selections share their existing underlying
GraphView state; indexing does not promise to shrink that captured payload.

## Verification

Three native integration tests cover nested/direct node and edge references,
parallel/loop edge membership, independent source-local namespaces, selected root
scope, unreferenced revision exclusion, explicit rebuilds, branches, deactivation,
owner drop and threads. Two native unit cases cover reused local legacy aliases
in different nested source contexts and explicit missing-source rejection versus
ordinary integer properties. Eleven Python cases exercise that public surface,
connected Subgraph capture, exact effective/source pins, immutable owned outputs,
durable bytes/head preservation, reopen/lock lifetime and strict controls.

Final gates: **1,428 native passed / 56 ignored**, **2,285 root Python passed /
1 skipped**, **218 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The CPython 3.9
full-ABI release extension was rebuilt and atomically staged in tree; three
pre-existing FFI warnings remain. Root Python verification excludes the owner's
separate algorithm benchmark and nested historical surveys. No external packages
or graph libraries were installed/used. Owner benchmark/notebook/review changes
remain excluded from this commit.

Commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target` with
offline locked `cargo test`, focused `--test group_membership_index`, strict
`cargo clippy --all-targets -- -D warnings`, and release
`cargo bench --bench group_membership_index`. Python runs the focused/root
contracts and executable docs runner. Guide: `docs/api/group-membership-indexes.md`.

## Performance evidence and boundaries

`notes/benchmarks/group-membership-index-2026-10-28.json` records five alternating
native release samples, with 100 checked direct lookups/scans per sample. A root
selection has 101/10,001 nodes; all but one reference a parent revision and one
directly references its nested leaf. Independent scanning visits selected
membership and borrowed snapshot properties, checks the named field and sorts
matches without per-node FFI calls or owned value copies. Every direct result is
checked; recursive results are checked to include every root once. The benchmark
also times full recursive output separately. Build timing excludes root capture
and starts from an already captured view. It includes all reachable-index work.

Local sample medians:

| Root nodes | Index build | One direct hit | Selected-property scan | All recursive references |
| --- | --- | --- | --- | --- |
| 101 | 18.916 µs | 0.215 µs | 3.604 µs | 6.292 µs |
| 10,001 | 2.031 ms | 0.203 µs | 644.510 µs | 0.779 ms |

See the JSON for complete samples. Narrow reverse lookups skip unrelated root rows. Recursive
lookup remains output-sensitive and sorts/deduplicates matches, so full output
cost grows with the result size. These bounded native fixtures do not measure RSS,
Python marshalling, deep nesting throughput or production/million-node capacity.
Memory adds reference/link postings while sharing values; immutable readers can
retain old state in memory. Index builds and root captures remain explicit work.

## Remaining original priorities

18.2 is complete within the listed family inventory; **211 checked / 57 open**
in the fixed 268-row checklist. Broader legacy pinned query migration remains
21.2; ordinary connected Subgraph ownership remains 26.1/AR4/AR7.5; compound
topology/typed-effect acceptance remains 09.4b.2/AR3/AR7.8. Cross-platform recovery,
conformance/scale/operational certification and Graphiti/service work retain their
separate existing items. No general production-readiness claim or adjacent audit
scope is added by closing the index family.

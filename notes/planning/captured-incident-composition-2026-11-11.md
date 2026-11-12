# Captured incident-edge composition planning — 2026-11-11

Completes the known per-selection whole-graph edge scan in frozen composition
within existing **09.4b.2 / AR3 / 32.2**. No new checklist row is added.

## Change and preserved contract

Frozen composition sources already carry validated immutable incident-edge rows.
Boundary and contraction preparation now gather candidates from those rows,
then sort/deduplicate by original edge ID. Internal edges seen at both endpoints
coalesce; distinct parallel IDs remain distinct; self-loops occur once. Candidate
rows cover the full captured source so boundary edges outside the selected
internal edge set are still considered. Contraction traversal remains limited
to the exact selected internal edges.

All composition entry points validate the frozen owner/history/branch/head
before using its source. Live/native connected compatibility keeps the existing
current-owner scan and does not capture all properties just to find boundaries.
No new long-lived adjacency cache or writable graph clone is introduced.
Preparation remains read-only, with all compositions/retained groups accepted
together by the existing mixed publication boundary. Stale input, late invalid
aggregations and destructive cross-selection ambiguity still reject unchanged.

An exhaustive differential fixture compares indexed preparation to the previous
global scan over every subset of five active candidate nodes in directed and
undirected graphs. It covers sparse IDs, endpoint revision, removed edges/nodes,
parallel edges/loops, selected internal-edge omissions, all five external edge
strategies, first/last/sum aggregation and automatic/none/explicit meta targets.
The existing independent reachability/corridor and atomic collection tests remain
separate guards against both paths sharing an incorrect aggregation rule.

## Quiet release measurements

| Components | Native before → after (ms) | Python before → after (ms) |
|---:|---:|---:|
| 10 | 0.354 → 0.340 | 0.405 → 0.166 |
| 100 | 2.168 → 2.083 | 1.702 → 1.304 |
| 500 | 15.391 → 6.690 | 17.091 → 7.798 |
| 2,000 | 192.406 → 40.904 | 203.355 → 47.797 |

Native median of five samples: disconnected pairs with count aggregation.
Python median of three: disconnected pairs with one integer node property and
sum aggregation. Both include retained-source preparation, destructive topology
effects and one accepted state. Fixture construction and result validation are
excluded. Each before/after pair uses its own unchanged fixture; the native and
Python columns are different workloads and do not measure FFI overhead.
Small cases are noisy/mixed; the 2,000-case gains are about 4.7× native / 4.3×
Python. Raw `incident-composition-{native-before,native-after,before,after}-2026-11-11.json`
receipts retain samples and Python process CPU separately from wall time.

Reproduction: `cargo bench --offline --locked --bench meta_composition -- --collection`
with `GROGGY_COMPOSITION_PROFILE` selecting the output path, and
`benches/compound_composition.py` for the public Python workflow. Builds must be
quiet during timing. No allocation/RSS or general throughput claim is made.
Frozen candidate preparation now scales with selected incidence plus candidate
ordering; the whole collapse still includes aggregation, group retention and
publication, and these measurements do not establish linear total scaling.

## Remaining boundary

Generic read-after-topology-write pipeline stages, typed graph/table effects and
arbitrary custom implementations remain existing 09.4b.2/30.1/AR7.8 work. Broader
history, storage, memory and scale certification remain in 32.2. No contract
decision or owner deferral is inferred from this optimization.

## Verification

Full native gate: **1,538 passed / 54 ignored**. Strict root all-target Clippy
passed. The release extension was rebuilt in tree (three existing FFI warnings).
Maintained Python gate: **2,594 passed / 1 skipped**; **258 executable docs**.
Scoped whitespace checks passed. Checklist remains **215 checked / 53 open**.

# Groggy roadmap status — 2026-10-24

Branch: `groggy-roadmap-2026-10-20`. The [flat checklist](graph-engine-flat-checklist.md)
currently contains **205 checked and 63 unchecked entries**. Entries overlap:
parent workstreams and smaller repair slices both appear, and unchecked entries
include partial, open, review and deferred work. These counts are an inventory,
not a completion percentage or an estimate of remaining implementation effort.

Count correction: the earlier summary omitted three already checked A* entries
because its heading parser treated the literal asterisk as formatting. The earlier atomic-builder pass
added two checked slices; three other entries were a counting correction. The
recent loop/block, structural, filtering, aggregation, transformation, mask-sampling, community and bounded-selector continuation slices are counted individually alongside their open parents. Counts now use checkbox markers independently of headings.

## Fixed remaining-work inventory

The 63 unchecked entries are not 63 independent tasks: 13 are overlapping AR
architecture entries, 50 are numbered entries (including nested publication
parents), and nine of those numbered entries explicitly say Deferred. The old
flat-checklist header had a stale 158 checked count; the current verified count is shown above.
Recent implementations were recorded as new checked slices while broad parents
stayed open. That made the open total insensitive to substantial completed work.

From this pass, audit repairs update existing requirements/evidence. New
capabilities are identified as scope additions. Each current workstream uses a
fixed implementation inventory with stated exits, so completing components reduces
its remaining count without manufacturing broad parent completion.

### Temporal builder family — existing eight primitives

| Existing primitive | Current status | Verified boundary |
| --- | --- | --- |
| snapshot_at | Complete | Retained-state selector certified in 09.4b.2al; effective-time reconstruction remains a separate original requirement. |
| decay | Complete | Checked maps/columns certified in 09.4b.2aj–2ak. |
| diff_nodes | Complete | Retained owner provenance, added/removed/modified maps, property comparison and atomic failure behavior verified. |
| diff_edges | Complete | Same; unrepresented endpoint rewiring rejects under existing 04.2/13.3. |
| window_aggregate | Complete | Correct change-value history, branch/index provenance, sparse state IDs, null/absence and checked reductions. |
| filter | Complete | Correct branch-scoped lifecycle/changes, retained-index provenance and failure/cancellation behavior. |
| mark_changed_nodes | Complete | Exact change-type membership, aggregation over windows and index provenance. |
| window | Complete | State-ID/time units, reactivation/deletion boundaries and branch-scoped existence. |

**Remaining: six → four → zero. All eight primitives complete; no checklist rows
added.** These are the same primitive names present before the audit. This closes
the fixed temporal family under 30.1; other pipeline/intermediate/pin requirements
and arbitrary effective-time reconstruction (16.2) retain their original scope.
See `temporal-window-completion-2026-10-24.md` for implementation/gate evidence.

## Recent completed slices

| Slice | What is now supported |
| --- | --- |
| 26.2 | Checked current-view batch writes, exact source/scope validation, prevalidated refreshed pins and durable journal publication/recovery. |
| 12.3a | Explicit checkpoints label accepted states without creating duplicate mutation deltas. |
| 09.4b.1 | Ordinary Python mutations default to checked publication, including initial properties and bulk operations. Explicit manual compatibility remains. |
| 09.4b.2a | Compound collapse prepares aggregations, retained membership, boundary edges and removals before accepting one version. |
| 09.4b.2c / 27.2a | Explicitly mapped mixed node/edge table property updates validate identity/endpoints and publish one batch. |
| 09.4b.2d / 05.3a | Subgraph/algorithm bulk writes reject complete inputs with inactive targets; all-edge convenience writes publish once. |
| 09.4b.2e / 27.2b | Table graph materialization preserves supported typed values/identities, remaps references and rejects unsupported data without silent downgrade/loss. |
| 09.4b.2f / 30.1a | Builder node/edge attachment and smoothing honor persist=False and return owned typed columns without graph mutation. |
| 09.4b.2g / 30.1b | Explicit atomic property builders stage touched columns, read staged values and accept once; errors/cancellation discard the buffer. |
| 09.4b.2h | Nested property loops recursively validate descendants, share staged values across iterations and discard the whole run on failure. |
| 09.4b.2i | Fused variable kernels and supported ordered message-pass blocks compose with atomic staging; options/direction and arithmetic errors are checked. |
| 09.4b.2j | Conditional blocks honor compare/where semantics with shared masks, early input/operator validation and atomic rollback. |
| 09.4b.2k | Synchronous message passing reads fixed block input and installs buffered outputs together, preserving column order and outer atomic staging. |
| 09.4b.2l | Degree and incident-weight primitives honor selected topology, staged weights and checked input/output rules. |
| 09.4b.2m | Undirected triangle and k-core primitives use checked selected topology, corrected counting/peeling and atomic staging; directed variants reject explicitly. |
| 09.4b.2n | Property filters and ordering honor staged values, exact numeric order, deterministic ties and explicit validation. |
| 09.4b.2o–2q | Shared checked statistics, deterministic categories/owned histogram vectors, five neighbor reducers and public/optimizer lowering parity. |
| 09.4b.2r–2t | Checked expression operators/literals, scoped synchronous/async maps and deterministic selected label/copy transformations; opaque callbacks remain uncertified. |
| 09.4b.2u–2w | Stable isolated node/edge mask sampling and bounded map-key reservoirs with strict controls, selected identity validation and atomic publication. |
| 09.4b.2x–2z | Checked seeding, synchronous propagation and actual directed/undirected local modularity gains verified against independent full-objective moves. |
| 09.4b.2aa–2ac | Ordered bounded selector iteration, selected neighborhood expansion and consistent induced emission; legacy owner connections remain explicit. |
| 09.4b.2ad–2af | Recursive for-each property staging, complete child selector flattening and private merged/per-item owned output frames. |
| 09.4b.2ag–2ai | Checked staged-weight distance maps, loopless Yen node paths and isolated seeded edge-incidence walks. All eight temporal primitives are now complete under 30.1; broader pipeline work remains. |
| 09.4b.2aj–2al | Finite selected decay maps, ordered column parity and retained immutable snapshot builder selection with strict temporal controls. |
| 09.4b.2b | Preset/override precedence, canonical meta identity with semantic labels, explicit meta selectors and real external endpoint contraction. |
| 30.1 temporal family | All eight primitives complete: retained branch indexes, exact lifecycle/change windows, checked aggregation and failure-safe outputs. |



These are extensions and repairs to an already substantial graph engine. They do
not represent building its basic engine contract from scratch.

## Broader areas

| Area | Completed foundation | Still open |
| --- | --- | --- |
| Numerical core and architecture | Canonical matrix/array foundations, entity handles, many arithmetic/statistical/solver/factorization/sparse/operator and algorithm repairs, bounded release receipts. | Remaining promised derivative/composition domains, uncovered consumers and representative performance/memory/scale gates; broad numerical parents remain partial. |
| Mutation/publication | Checked mixed batches, ordinary Python defaults, property clocks, checkpoints, compound collapse and recent table/subgraph boundaries. | Remaining native default/infallible compatibility, mutable pool/space escapes, whole-pipeline boundaries and broader pipeline semantics. |
| Properties, state and history | f64 defaults, typed temporal properties, shared logical change records, retained snapshots and immutable read APIs. | Complete value lifecycle, effective-time reconstruction/validation, rewiring/revert, exact portable snapshots and portable pooled/meta contents. |
| Views and tables | Explicit pinned read views, frozen matrix/table exports, mapped writeback and checked independent materialization. | Migrate legacy connected reads and broader graph/table conversion adapters. Frozen BaseTable selections/projections and generic joins now retain source pins; direct/mixed source property writeback is checked. Generic joins, grouping, dedup and set operations share typed keys and stable ordering; row combinations retain source context, and concat rejects name collisions. TableArray reductions/group-key validation and lossless supplied-table flatten now share a Rust coordinator. All six specialized graph-table merges and federated domain naming now preserve attributes, domain-local reference maps, validation policy and source pins. Collection context now survives empty grouping/selections, schema transforms, callbacks and native joins. Broader graph/table conversions and broad table parents remain partial. |
| Storage, branches and schema | Checked journal/recovery foundations, checkpoint annotations, existing branch/identity behavior and explicit schema checks. | Full pooled retention, lifecycle/compaction/format migration, complete branch lifecycle, three-way merge, purge and schema evolution. |
| Queries, indexes and pipelines | Existing native filter/query/index/builder machinery and many repaired algorithm paths. | Consistent index lifecycle, pinned temporal query results, shared expression/execution boundaries, pipeline intermediates/history and complete algorithm certification. |
| Integration and service | Existing embedded engine and test/benchmark infrastructure. | Graphiti ingest/driver/provenance/retrieval remain deferred behind core; service concurrency, multi-user deployment and broad comparison/scale gates remain open. |

## Continue in this order

1. Finish the remaining native mutation/publication and compound pipeline audits
   under 09.4b.2; composer leaf 09.4b.2b is now complete.
2. Complete legacy view/source-pin migration and retained pool/meta recovery;
   these underpin dependable history and connected table behavior.
3. Finish timestamp reconstruction and timeline validation, then complete
   rewiring/revert, branch/merge/retention and storage lifecycle behavior.
4. Complete query/index/pipeline consistency and remaining architecture moves.
5. Close documented numerical/algorithm/performance/conformance/release gaps with
   representative fixtures and measured limits.
6. Resume Graphiti provenance, ingest, retrieval and driver integration after the
   core behavior is dependable.

This order does not create a new approval gate. Existing workbook answers remain
the implementation authority; unresolved policy decisions stay visible in the
appendix and their corresponding checklist entries.

## Current verification baseline

Latest collection-ownership slice (2026-10-25): **1,079 native tests passed, 51 ignored; 1,812 root
Python API tests passed, 1 skipped; 109 executable documentation examples passed;
strict all-target core Clippy passed.** The root Python gate excludes the unrelated
owner benchmark test and the separate legacy modules survey. Local release
receipts are bounded observations, not production throughput/memory guarantees.

A matched composer before/after comparison now measures the bulk-removal repair
(`bulk-removal-2026-10-25.md`); broader performance comparison remains follow-up
under the existing performance gates. As requested, implementation priorities continue
first; comparison must match workload, precision and history semantics and record
time, allocations, peak memory and retained-history growth.

The explicit nested-sampler child 09.4b.2ad is now complete alongside related
selector-output and owned-frame slices 09.4b.2ae/2af. This closes one unchecked
entry while adding two checked implementation slices. Standalone sampler
transactions and pinned view/source provenance remain open parent work.


### Native graph-table feature conversions — 2026-10-25

Completed native node/edge feature conversion and the existing f64/f32/i64 node
conveniences under 27.1/AR7.10. Fixed row-major layout, empty shape/name retention,
and strict ID validation through one GraphTable-owned implementation. Five new
native tests cover both domains, 32 shapes and conversion/ownership errors.

Latest gate: **1,084 native passed / 51 ignored; 1,812 root Python passed /
1 skipped; 109 executable docs passed; strict all-target Clippy passed**.
Evidence: `graph-table-feature-conformance-2026-10-25.md`; benchmark receipt:
`notes/benchmarks/graph-table-features-2026-10-25.json`.

No new checkbox rows: **191 checked / 77 unchecked**, including overlapping
parents and nine explicitly deferred entries. The feature-matrix conversion
family is closed; the broader graph/table reshaping and bundle adapter boundaries
remain under the existing parent.


### Typed filters and checked reshapes — 2026-10-25

Closed node/edge typed equality filtering, stable generic pivot grouping,
checked melt schema/row counts, specialized pivot/melt identity validation and
the Python melt identifier adapter under 27.1/AR7.10. Existing positional Python
melt calls remain valid; `id_vars` is an optional keyword. General long layouts
use BaseTable; specialized outputs cannot discard or duplicate graph identity.

Evidence: `table-reshape-conformance-2026-10-25.md`. Gates: **1,090 native passed /
51 ignored; 1,819 root Python passed / 1 skipped; 112 executable docs passed;
strict all-target Clippy passed**. Current-cost receipt:
`notes/benchmarks/table-reshaping-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite/bundle adapter scope remains open.


### Typed and verified standalone table bundles — 2026-10-25

Closed the existing save/load/validate/info/verify bundle family under
27.1/AR7.10. New v3 CBOR stores ordered logical inline values and policy using
the shared property codec, exact float bits, checked framing/checksum and atomic
synced replacement of one authoritative file. V2 CSV reads now verify historical
checksums, metadata, manifest and schema/counts; resave migrates to v3. FFI
persistence logic was replaced with native delegation and GIL release.

Pooled refs reject rather than becoming strings; source pins/history are detached,
not represented as portable. Legacy CSV cannot recover lost type information.
Topology validation and encoded integrity remain separate. Whole-table memory
cost and post-publication directory-sync reconciliation are documented.

Evidence: `graph-table-bundle-conformance-2026-10-25.md`; frozen v2 fixture;
8 native / 11 Python regression cases. Gate: **1,098 native passed / 51 ignored;
1,830 root Python passed / 1 skipped; 115 executable docs passed; strict
all-target Clippy passed**. Benchmark receipt:
`notes/benchmarks/graph-table-bundles-2026-10-25.json`. No new checklist rows;
191 checked / 77 unchecked. Composite adapters and separate full retained-pool,
history and storage lifecycle requirements remain open.


### Entity table adapters and 27.2 completion — 2026-10-25

Completed node/edge ID-mapped property attachment, typed structural readers,
linear tuple extraction, exact Python integer ID arrays and source-aware
attribute publication. Unknown/duplicate attribute IDs reject; malformed
structural reads no longer compact rows independently. Existing source capture,
head, schema and durable publication gates are reused.

27.2 is now checked: the immutable source-bound explicit table update contract
is complete, including Python mixed property updates with one delta and explicit
effective time. Remaining legacy live-view migration and composite conversions
stay in 26.1/27.1/AR7.10; full retained-pool portability keeps its original items.

Evidence: `entity-table-adapters-2026-10-25.md`; six native and thirteen Python
regressions. Gate: **1,104 native passed / 51 ignored; 1,843 root Python passed /
1 skipped; 118 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/entity-table-adapters-2026-10-25.json`.
No new checklist rows: **192 checked / 76 unchecked** after closing 27.2.


### Checked bulk table lookups and neighbor reads — 2026-10-25

Closed the existing native get_nodes/get_edges/neighbors adapter family under
27.1/AR7.10. Bulk identity lookup scans IDs once and copies selected first whole
rows; duplicate requests collapse and unknown IDs remain omitted. Structural
errors reject consistently. Neighbors scan borrowed endpoints once, preserving
the existing external-neighbor set semantics without copied edge tables/arrays.
No public API or graph direction/subsetting policy was added.

Evidence: `graph-table-bulk-reads-2026-10-25.md`; four native regressions, including
all 32 subsets of an independent looped/multiedge oracle. Gate: **1,108 native
passed / 51 ignored; 1,843 root Python passed / 1 skipped; 118 executable docs
passed; strict all-target Clippy passed**. Receipt:
`notes/benchmarks/graph-table-bulk-reads-2026-10-25.json`.
At 100k input rows with 100 requested IDs, current medians are 2.222 ms node
lookup, 2.345 ms edge lookup and 3.486 ms external neighbors; no matched speedup,
RSS or FFI claim. No new rows: **192 checked / 76 unchecked** after 27.2 closure.
Composite subsetting/routing and legacy view migration remain open. The induced
node-subset edge policy question remains unanswered; it was not silently applied.


### Legacy table output pin migration — 2026-10-25

Closed existing native/Python Graph, connected Subgraph and full/constrained
accessor table export paths under 26.1/27.1/AR7.10. All share pinned projection,
source row identity, stable selected IDs and explicit-null preservation. Mixed
exports share one capture; stale selections reject; dirty working captures stay
read-only. Removed separate FFI row-building logic. Legacy source objects remain
connected; GraphTable edge-retention policy is unchanged.

Evidence: `legacy-table-export-migration-2026-10-25.md`; 6 native / 12 Python cases.
Gate: **1,114 native passed / 51 ignored; 1,855 root Python passed / 1 skipped;
121 executable docs passed; strict all-target Clippy passed**. Fresh capture costs
are proportional to parent state; current capture/reuse measurements are in
`notes/benchmarks/table-export-capture-2026-10-25.json`. No COW, O(1), RSS or fixed
FFI budget claim. No new rows: **192 checked / 76 unchecked**. Broad legacy-view
and composite adapter exits remain open.


### Shared table schema validation — 2026-10-25

Closed ordered constructor, projection and column-growth invariants under
27.1/AR7.10. Every physical column must appear exactly once in an ordered schema;
projections reject duplicate/unknown requests; declared zero-row schemas cannot
be expanded by changing only one column. Truly empty initialization, sparse
from_columns padding and N-row zero-column projections retain their contracts.
Python node/edge projection delegates to native validation with GIL release.

Evidence: `table-schema-invariants-2026-10-25.md`; 6 native / 6 Python cases.
Gate: **1,120 native passed / 51 ignored; 1,861 root Python passed / 1 skipped;
124 executable docs passed; strict all-target Clippy passed**. These are O(schema
width) checks, without a new performance claim/kernel. No new rows: **192 checked /
76 unchecked**. Composite routing/subsetting and broad legacy migration remain.


### Composite routing and shared component metadata — 2026-10-25

Closed native sort/group/pivot/melt operand routing and GraphTable component
metadata sharing under 27.1/AR7.10. Edge melt now respects its identifier component
when values have shared names. Existing sort/reshape node preference and grouping
ambiguity rejection remain. Clones/groups share table/schema objects above the
already shared BaseArray value buffers; mutable access detaches metadata and
retains the existing column COW/replacement rules. Initial progress commentary
overstated prior value-copy costs; evidence/docs explicitly correct that claim.

Evidence: `graph-table-component-conformance-2026-10-25.md`; 6 routing / 5 sharing
native cases, including component and buffer pointer checks, independent edits
and source publication. Gate: **1,131 native passed / 51 ignored; 1,861 root Python
passed / 1 skipped; 127 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/graph-table-component-sharing-2026-10-25.json`.
Clone/drop timing is amortized over 10000 iterations to exceed clock resolution;
no matched speedup, RSS, whole-graph COW or FFI latency guarantee.

27.1 now has a fixed explicit remainder: node-subset edge policy; composite
metadata/rectangular Table interpretation; existing unsupported to_nodes/to_edges/
to_subgraphs converter contracts. These pre-existing facade contracts remain
within the original parent, not new checklist rows. Legacy object migration stays
in 26.1; full retained-pool/history portability remains separately tracked.
Inventory: **192 checked / 76 unchecked**, unchanged by adding no rows.

### Checked table visualization captures — 2026-10-25

Replaced all three Python table `.viz` graph-rebuilding loops with native checked
entity adaptation (27.1/AR7.10 and 28.15). Sparse/optimized IDs, parallel edges,
loops and typed properties retain their original identity; malformed/duplicate
IDs and dangling endpoints reject before server creation. No temporary graph,
new IDs or history are allocated. Direction is explicit natively; the existing
Python undirected default remains. Empty/oversized GraphDataSource windows are
safe. Evidence/API: `table-visualization-2026-10-25.md` and
`docs/api/table-visualization.md`; 7 native / 8 Python cases and 3 API examples.
Full gate: **1,138 native passed / 51 ignored; 1,869 root Python passed / 1 skipped;
130 executable docs passed; strict all-target Clippy passed**.

At 100k nodes, the current native direct capture median is about 80ms versus
798ms for checked graph materialization plus capture on the same dense fixture.
These are current alternatives, not a speedup against the old incorrect FFI
loop or an RSS/FFI/server guarantee. Receipt:
`notes/benchmarks/table-visualization-2026-10-25.json`.
Existing table DataSource compatibility readers and interactive_viz options,
converter ownership, node-subset and rectangular metadata contracts remain open.
Inventory remains **192 checked / 76 unchecked**; no new rows added.

### Shared immutable visualization captures — 2026-10-25

GraphDataSource clones now share their entire immutable payload and stable
process-local identity; counts derive from stored vectors and duplicate payload
helpers are consolidated. Python viewer server reuse now uses that identity,
replacing a temporary stack-address key that could alias separate captures.
Same accessor reuses its server; a fresh `.viz` accessor is a separate capture.

Evidence: `viz-snapshot-sharing-2026-10-25.md`; one pointer/last-owner test, four
native public-contract cases and one process-isolated Python real HTTP/websocket
server test (including sparse IDs in actual server snapshots). Gate: **1,143
native passed / 51 ignored; 1,870 root Python passed / 1 skipped; 130 executable
docs passed; strict all-target Clippy passed**. Matched native clone/drop at 100k
nodes: median 26.06ms before / 10ns after, amortized over 100 iterations with the
same benchmark source. Initial capture and owned payload exports still copy;
this is not a graph-capture/RSS/FFI/server latency claim. Receipts:
`notes/benchmarks/viz-snapshot-clone-{before,after}-2026-10-25.json`.
Existing 27.1/AR7.10/28.15 scope; **192 checked / 76 unchecked**, no new rows.

### Checked BaseTable entity conversions — 2026-10-25

Closed the existing native BaseTable converter family and exposed thin Python
`to_nodes(uid_key="node_id")` / `to_edges()` adapters (27.1/AR7.10). Custom node
keys now rename correctly; edge aliases use existing native normalization.
Complete identity/endpoint validation rejects malformed or duplicate IDs, while
parallel edges/self loops remain valid. Conversions create table components,
share column payloads, and allocate no graph entities or history.

Writable source binding survives only for the same captured kind/IDs/endpoints;
changed identity/kind/topology retains read-only source captures. Compatible
renames can publish one explicit source update. Import-oriented from_base_table
wrappers remain permissive. Evidence/API: `checked-table-conversions-2026-10-25.md`
and `docs/api/checked-table-conversions.md`; 7 native / 11 Python cases, 3 examples.
Gate: **1,150 native passed / 51 ignored; 1,881 root Python passed / 1 skipped;
133 executable docs passed; strict all-target Clippy passed**. Validation scans
identity columns; no speedup, RSS or FFI latency claim.

The fixed 27.1 remainder still names GraphTable node-subset policy, rectangular
metadata, and its unsupported graph-accessor/subgraph converter contracts.
Those are distinct from the now-complete BaseTable data-only adapters. No new
checklist rows; inventory remains **192 checked / 76 unchecked**.

### 12.3 exit review — explicit checkpoint requirement complete

Closed the original 12.3 parent after tracing native commit/checkpoint, writer,
Python default publication and durable annotation/replay through the existing
passing conformance suites. The former Partial status predated ordinary Python
default migration. Clean labels add no delta or state ID; dirty/manual commit
publishes once for compatibility; stale/dirty/schema failures and durable
rollback/recovery are covered. Evidence: `checkpoint-exit-review-2026-10-25.md`.

This is a status correction, not a new capability or a claim that native default
migration (09.4b.2), materialized recovery checkpoints/compaction (10.4), portable
pooled snapshots (12.2) or effective-time reconstruction (16.2) are complete.
Current gate remains 1,150 native / 1,881 root Python / 133 executable docs,
with 51 native ignored and one Python skip; strict all-target Clippy passed.
Inventory now **193 checked / 75 unchecked**, with no rows added or removed.

### GraphTable owner decisions implemented — 2026-10-25

Owner confirmed induced node subsets, separate component shapes/rectangular
operations through .nodes/.edges, and table-component to_nodes/to_edges aliases.
Implemented checked head/tail/slice and one-pass induced node grouping, exact row
provenance, separate native/Python shapes/schemas and independent component
wrappers. GraphTable no longer implements rectangular Table; native subsets are
fallible and Python raises standard errors. Existing aggregate counts remain
explicitly separate from shape. Also fixed BaseTable's end-versus-length slice
bug found by the new middle/reversed range cases.

Evidence/API: `graph-table-decisions-2026-10-25.md` and
`docs/api/graph-table-components.md`. Gate: **1,157 native passed / 51 ignored;
1,890 root Python passed / 1 skipped; 134 executable docs; strict all-target
Clippy passed**. Current 100k-node/100-group native cost about 31ms; receipt
`notes/benchmarks/graph-table-induced-groups-2026-10-25.json`. The earlier bundle
policy is a different workload; no matched speedup/RSS/FFI claim.

27.1 remaining inventory is now its unresolved to_subgraphs contract and native
legacy display/streaming schema integration. These are existing converter and
metadata boundaries, not new requirements. Broader 26.1/portable history remain
separate. Inventory **193 checked / 75 unchecked**, no new rows.

### Native GraphTable streaming projection and cache identity — 2026-10-25

Closed the remaining native table-schema integration boundary in 27.1/AR7.10.
A private graph_streaming adapter produces one explicitly tagged display
projection with component-prefixed columns, typed values and null padding;
headers/schema/width agree and windows span both components. Core shapes remain
separate. Clone-stable process-local presentation IDs refresh on derived outputs,
mutable component access, policy edits and explicit ID assignment, preventing
same-shaped branches from aliasing cached windows. No graph history is produced.

Evidence: `graph-table-streaming-2026-10-25.md`; four native cases. Full gate:
**1,161 native passed / 51 ignored; 1,890 root Python passed / 1 skipped;
134 executable docs; strict all-target Clippy passed**. Current induced-group
receipt: `notes/benchmarks/graph-table-streaming-identity-2026-10-25.json`;
100k-node/100-group median about 28.2ms, not a speedup/RSS/FFI claim.

The fixed 27.1 remainder is now **to_subgraphs grouping/ownership**, currently
awaiting the owner choice (explicit column groups / connected components /
deferral). Native legacy graph-data/layout adapters are unchanged and remain in
existing AR7.6/28.15 scope. Broader 26.1 and portable-history work remain separate.
Inventory **193 checked / 75 unchecked**, with no new rows.

### Native graph-table layouts preserve isolated nodes — 2026-10-25

Fixed the existing 28.15/AR7.6 compatibility layout adapter: using the complete
checked table capture retains isolated nodes, including a one-node graph, with
original IDs/order. Metadata no longer infers direction by reciprocal-edge
heuristics; the native table compatibility path defaults undirected and the
checked capture offers explicit direction. Removed the quadratic guess without
claiming all legacy graph readers, layout options or solver quality are certified.

Evidence: `graph-table-native-layout-2026-10-25.md`; two native cases. Full gate:
**1,163 native passed / 51 ignored; 1,890 root Python passed / 1 skipped;
134 executable docs; strict all-target Clippy passed**. Native current grid case
on 100k nodes (chain plus isolate, no properties) costs about 43ms including
complete capture, not a speedup versus the old incomplete node set. Receipt:
`notes/benchmarks/graph-table-native-layout-2026-10-25.json`.

The immediate remaining table decision is to_subgraphs grouping/ownership;
the three earlier owner answers are implemented and recorded. Original wider
workbook questions and 16.2 effective-time reconstruction remain open.
Inventory **193 checked / 75 unchecked**, no added rows.

## 2026-10-25 — Close agreed table facade; return to temporal core

27.1 and AR7.10 are complete against their fixed inventory. The owner accepted
explicit `to_subgraphs(by=...)` node-property groups returning immutable views;
native and Python implementations preserve table values and original identity,
sharing one frozen capture. Evidence: `graph-table-grouped-views-2026-10-25.md`.
**195 checked / 73 unchecked**, with no new checklist rows.

Execution correction: recent slices overemphasized adjacent facade/viz audits.
Prioritize usable core workflows: timed bulk mutation → reconstructed immutable
read → query/table → save/reopen. Incidental cleanup is taken only if it blocks
that workflow or resolves a demonstrated correctness/performance failure. Next
core slice is 16.2/16.3; this does not retire any original roadmap scope.

## 2026-10-25 — Effective-time vertical slice

16.3 is complete for precisely timed publication ancestry; 16.2 is partial with
working Graph/DurableGraph timestamp reconstruction, accepted-order corrections,
branch/head capture, optional recorded cutoff, immutable derived/table source
pins, ordinary typed properties and durable reopen. Read-side endpoint interval
validation is present; atomic write-side timeline validation remains the next
correctness priority. Pooled/group history, unset and merge/legacy boundaries
retain their original scopes. **196 checked / 72 unchecked**, no new rows.

Release baseline: 100k-node/99,999-edge/101-version replay about 147 ms median;
preparing owned immutable ancestry about 35 µs. This is full replay, not an index.
See `effective-time-read-slice-2026-10-25.md` and the machine-readable receipt.

Full gate before the final two native ownership/extreme-clock regressions:
1,169 native passed / 51 ignored; 1,899 root Python passed / 1 skipped;
136 executable documentation examples; strict all-target Clippy passed.
Final targeted/rebuilt checks are recorded with the completion gate below.

Completion gate: **1,171 native passed / 51 ignored; 1,900 root Python passed /
1 skipped; 136 executable documentation examples; strict all-target Clippy
passed**. Native/Python logs: /tmp/groggy-effective-native-final.log and
/tmp/groggy-effective-python-final.log; Clippy/build logs use effective-*-final.
Scoped rustfmt, Python stub AST parse and git diff --check passed. In-tree FFI
rebuild retains three unrelated pre-existing warnings. Owner benchmark changes
and untracked benchmark tests/receipts were excluded from staging and execution.

## 2026-10-25 — Atomic effective endpoint publication

Completed the write-side endpoint/lifetime requirement within 16.2. Checked
preparation rejects invalid complete effective intervals before allocation,
working-state changes or durable append. Retained removed edges participate in
retroactive node changes. A history-owned incremental lifecycle index avoids full
history replay for ordinary writes; property-only preparation skips topology
validation. Branch changes rebuild the selected ancestry; GC drops the cache.
Journal recovery prepares with original recorded clocks. Explicit timed topology
against unknown legacy lifetimes rejects; untimed legacy compatibility remains.

Current bounded release edge-publication costs: 2.2–6.7 µs median across 1k/100k
nodes and zero/1k prior property versions. These measure full checked publication,
not a prior-version speedup or high-degree/deep-lifetime/memory/FFI guarantee.
See `effective-topology-publication-2026-10-25.md` and its JSON receipt.

Gate: **1,178 native passed / 51 ignored; 1,903 root Python passed / 1 skipped;
136 executable documentation examples; strict all-target Clippy passed**. Three
additional rebuilt Python checks verify rejection/ID and identity reuse/durable
bytes after the full run. Scoped rustfmt and git diff --check passed. Existing
mixed timed scoped-write tests/docs now use explicit valid endpoint births;
separate tests retain the untimed legacy compatibility boundary. Owner benchmark
changes remain untouched and excluded from staging/test selection.

The count remains **196 checked / 72 open**: this removes a concrete 16.2 blocker
without adding rows or falsely closing the parent. Next core scope is property
unset/retained property history (11.2), followed by retained pooled/group reads
(12.2/16.2) and measured read indexing. Legacy/merge migration remains explicit.

## 2026-10-26 — Complete 16.2 effective-time contract

Closed 16.2 after implementing frozen source revisions/nested referents, typed
durable transfer/reopen/backup, explicit read-local legacy clocks and source
bindings, resolved first-parent merge replay, checked pooled-reference admission,
and cached selected-head event prefixes. Existing atomic endpoint checks remain.
No working values or second-resolution clocks are guessed for missing legacy data.
Known precise clocks/frozen sources reject overrides. See the full criteria and
scope boundaries in `effective-time-exit-review-2026-10-26.md`.

**197 checked / 71 open**, no new rows. Recounting the open categories gives 13 AR
entries and 58 numbered entries, including nine explicitly deferred entries;
the earlier category subtotal was stale even though the headline total was correct.
Property unset/coalescing is next under 11.2; owned/configuration-complete snapshots,
merge planning and storage checkpoints retain their existing parent requirements.

Release receipt distinguishes cold/warm reads: at 100k nodes/101 versions, about
155 ms cold index+reconstruction, 105 ms warm reconstruction and 0.4 µs warm empty
prefix. This is not O(1) full graph materialization or a production/RSS guarantee.

Gate: **1,184 native passed / 51 ignored; 1,913 root Python passed / 1 skipped;
139 executable documentation examples; strict all-target Clippy passed**. Scoped
rustfmt, stub AST parsing and git diff --check passed. In-tree FFI rebuild retains
three unrelated pre-existing warnings. Owner benchmark edits/test/receipt remain
untouched, unstaged and excluded from execution. Logs use /tmp/groggy-162-*-exit.

## 2026-10-26 — property history completed (11.2)

Closed the existing item, with no new rows: **198 checked / 70 open**. Explicit
unsets preserve null/absence through mixed scalar/bulk changes, schema validation,
timed replay, branch switching, immutable pins and durable reopen/backup. Fixed
scalar-after-bulk coalescing. All typed representations and retained references
remain readable; older assignment-only hashes/journal entries remain compatible.
See `property-history-exit-review-2026-10-26.md`. Next is 12.2 owned/configuration
snapshots and repeated-switch allocation, followed by native boundaries 09.4b.2.

## 2026-10-26 — exact owned snapshots completed (12.2)

Closed the existing full configuration / owned raw-pool transfer / repeated-switch
allocation inventory: **199 checked / 69 open**, no new rows. Full capture is
explicit; ordinary snapshots stay lightweight. All retained core storage/history
and counters restore into independent owners. Release 100k-node evidence shows
zero duplicate column growth through six switch pairs; reconstruction still has
substantial costs. See `owned-snapshot-exit-review-2026-10-26.md`. Next is the
whole-pipeline boundary under existing 09.4b.2; persistent archive/import stays 10.3.

### 2026-10-26 — measured reconstruction improvement

Followed the 12.2 benchmark to a quadratic membership scan. One membership set and
one removal filter preserve the old semantics and column bounds. Matched 100k-node
warm switch pairs improve from 1.554 s to 0.564 s, and owned capture from 566 ms
to 100 ms. See `recorded-membership-scaling-2026-10-26.md`. Counts remain
**199 checked / 69 open**; no scope or checklist rows added.

## 2026-10-26 — outer pipeline and import publication boundaries

Completed certified whole-pipeline property acceptance and complete selected
snapshot import baselines under existing 09.4b.2. Errors/cancellation discard
earlier staged updates; children share one acceptance; untouched imported topology
and properties remain in history. **199 checked / 69 open**, no new rows. The
parent now has three explicit open components: native default/infallible migration,
raw pool/space escapes and uncertified/topology algorithm acceptance. Effect-aware
IR remains AR7.8. Evidence/gates: `outer-pipeline-boundary-2026-10-26.md`.

## 2026-10-26 — branch lifecycle completed (14.1)

Closed the existing creation/switching/identity/read/recovery inventory. Recorded
timestamp selection now uses selected first-parent ancestry; siblings and later
unselected retained states cannot leak into it. Frozen groups, dirty manual
workflows, empty branches and durable schema/failure boundaries are verified.
**200 checked / 68 open**, no new rows. See `branch-lifecycle-exit-review-2026-10-26.md`.
Explicit destructive pruning/history and pool reclamation remain 15/10.4.

## 2026-10-26 — explicit history retention component (15.1 / 15.2)

Replaced no-op gc_history and added explicit branch pruning with shared branch/tag/
merge ancestry, open immutable payloads, lifetime reservations and A09 checkpoint
policy preserved. Durable removal receipts, rollback/replay/reopen/backup and a
50k-version benchmark are recorded in `history-retention-slice-2026-10-26.md`.
Raw pool/identity reclamation, complete referent closure and physical compaction/
replacement remain open under 15/10.4. **200 checked / 68 open**, no new rows.

## Consolidated priority pass — 2026-10-26

[Priority slice record](priority-slices-2026-10-26.md) lists the seven implementation
commits including 16.2, final verification, measured costs and the fixed remaining
core inventory. Follow-on movement is **197/71 → 200/68**, closing existing 11.2,
12.2 and 14.1 while advancing native boundaries and history-only retention without
adding checklist rows. No additional owner answers were needed for these slices.

2026-10-26: 10.3 complete retained binary archive/existing-graph journal import is
closed; see `retained-archive-import-2026-10-26.md`. Current inventory 201/67.

## 2026-10-26 — explicit physical storage checkpoint component (10.4)

Added `DurableGraph.checkpoint_storage` with a complete retained baseline, locked
atomic Unix replacement, schema revision preservation and continued replay/backup.
No graph version is created. File/candidate recovery runs outside Python's GIL;
pre/post-replacement failure ownership is verified at five injected boundaries.
Local 10k-update reopen: **73.05 ms → 24.46 ms** with all raw values retained.
See `storage-checkpoint-2026-10-26.md` for evidence and the fixed remaining scope.
**201 checked / 67 open**, no new rows. Chunking/scale/migration/non-Unix/crash
corpora remain; automatic scheduling stays deferred under A02. Pool reclamation
and optimize are still separate 15/06.3 work. No owner decision changed.

## 2026-10-26 — exact floating restoration follow-up

Repaired signed-zero/NaN-payload reuse in retained state restoration under existing
12.2/10.3. Scalar/vector/compressed values retain exact bits through switches,
owned copies, archives and durable checkpoint/recovery; repeated restoration does
not append duplicate values. Public property equality/history hashes are unchanged.
See `restoration-float-bits-2026-10-26.md` for regression and measured scope.
Counts remain **201 checked / 67 open**; no new rows or owner decisions.

## 2026-10-26 — native default migration component completed

Ordinary native Graph construction now publishes, node insertion and custom config
construction return visible errors, and explicit manual factories preserve deferred
assembly. Private fixtures/imports spell out that policy; ordinary bulk creates one
version. Node accessor append/extend are registered and validate complete inputs
before one native bulk creation. See `native-default-migration-2026-10-26.md` for
source migration, tests and release timings. Existing 09.4b.2 inventory shrinks from
three components to two: raw mutable storage escapes and uncertified/topology
compound acceptance. **201 checked / 67 open**, no new rows or owner decisions.
The raw audit found a silent unsupported pooled-attribute setter; it remains part
of that existing boundary repair, with no new checklist task.

## 2026-10-26 — raw owner mutation boundary completed

Public Graph pool/space access is read-only; raw mutation is crate-private/test-only.
All 46 corruption/recovery cases moved to internal native modules without being
skipped. Unsupported pooled attribute getter/setter operations now explicitly fail
instead of pretending to store data. Meta/subnode properties remain independent
coordinated writes and frozen sources remain immutable (A01.1). See
`raw-storage-boundary-2026-10-26.md`. Remaining 09.4b.2 fixed inventory: **one**
component, uncertified/topology multi-stage acceptance; native defaults and raw
escapes are closed. **201 checked / 67 open**, no new rows or owner choices.

### 2026-10-26 — classic algorithm outer property acceptance

All 13 built-in classic registry algorithms now share builder Context property
staging, including their weight/marker/heuristic/personalization/seed reads.
Success publishes once; late failure rolls back. Evidence and remaining limits:
`classic-atomic-algorithms-2026-10-26.md`. Existing 09.4b.2 remains open for topology
compound acceptance and unsupported custom effects; fixed component inventory
remains two complete plus this narrowed final component. No new checklist rows:
**201 checked / 67 open**. Broader IR/default policy remains AR7.8/30.1.

### Fourth priority window — chunked storage baselines (2026-10-26)

Window began 12:52:12 UTC. Existing 10.4 chunked-baseline component is complete:
bounded 8 MiB streaming fragments, ordered/full-stream checksum verification,
scratch-backed recovery and locked atomic initial creation/replacement. A 566 MB
retained journal reopens in 1.30 s / 1.73 GB peak RSS locally; decoded memory
remains material work. Evidence: `chunked-storage-baselines-2026-10-26.md`.
Gates: 1,259 native / 51 ignored; 2,000 root Python / 1 skipped; 158 docs; strict
Clippy. No new checklist rows: **201 checked / 67 open**. 10.4 remains partial
for decoded-memory/migration/platform/process-failure scope; A02 schedule deferred.

### Borrowed archive-state validation completed

Following 242e80e2, selected-state validation now borrows current column values
instead of cloning current properties and encoding two complete maps. Paired
release medians (three alternating pairs, 270 MiB unique text / 566 MB journal):
open **1.464 → 1.306 s**, isolated-process peak RSS **1,723,629,568 →
1,203,109,888 bytes** (~11% less time, ~30% less RSS). This is a local fixture,
not a universal bound. Receipt: `borrowed-archive-validation-2026-10-26.md`.
Gates: **1,261 native / 51 ignored**, **2,000 root Python / 1 skipped**, **158
documentation examples**, strict all-target Clippy and in-tree extension rebuild.
No new checklist rows; **201 checked / 67 open**. Decoded graph/history still
scale with retained data. The owner GraphTable grouped-view answer was reconfirmed
and its stale appendix pending marker corrected in d5bdc24a.

### Checkpoint process-death recovery verified

Existing 10.4/31.4 scope now includes actual SIGKILL/reopen tests at six boundaries,
including partial large-fragment output and before/after atomic replacement.
Complete state/clocks/history/schema/checkpoints, released locks and continued
append are verified. See `checkpoint-process-recovery-2026-10-26.md`. This does
not certify sudden power loss/non-Unix replacement or add orphan maintenance.
Gates: **1,263 native / 51 ignored**, strict all-target Clippy; production API
unchanged. No new rows; **201 checked / 67 open**.

### Recorded-state reconstruction completed without per-version whole-state copies

HistoryForest borrows retained deltas; GraphSnapshot uses one owning replay buffer
and a generation/order membership index. Existing immutable APIs, selected lineage,
IDs/types/clocks and write semantics remain. Native prefix parity and 10k-version
scalar-add regressions pass. Three-pair release fixture (10k nodes / 1k property
versions): **1.025 s → 1.722 ms**, requested allocations **4.18 GB → 4.62 MB**;
ordinary mutation/FFI latency is not measured. Scalar-add scratch peak grows by
~40 KiB in its small fixture. Evidence: `recorded-reconstruction-2026-10-26.md`.
Gates: **1,265 native / 51 ignored**, **2,000 root Python / 1 skipped**, **158
documentation examples**, strict Clippy and rebuilt extension. No new rows;
**201 checked / 67 open**. Full retained integrity scanning is the next 33.2 slice.

### Retained-memory integrity diagnostic completed (33.2 component)

Native/Python Graph and DurableGraph now expose check_integrity, verifying
portable recovery invariants plus every retained first-parent transition, including
unreferenced/nonhead versions. Structured counts and contextual errors are returned
without mutation. Detached verification releases the Python GIL; capture remains
owner-bound. Durable scope is retained memory/current schema, not rereading an open
journal. Evidence and exact limits: `retained-integrity-2026-10-26.md`.
Gates: **1,270 native / 52 ignored**, separate serial SIGKILL gate **1 passed**;
**2,004 root Python / 1 skipped**, **160 docs**, strict Clippy and rebuilt extension.
No new checklist rows; **201 checked / 67 open**. Broad 33.2 retains original
index/cache/physical-file/service diagnostic scope.

### JSON mutation recoverability gap repaired (07.4 / 05.3)

A reproduced native path accepted malformed typed JSON before later save failure.
The shared graph property boundary now applies the same intrinsic syntax rule as
retained storage, rejecting whole scalar/bulk/create/batch/upsert inputs before
values/versions/IDs/reservations change. Valid text and large-number spelling remain
exact; ordinary Text/non-JSON values keep their distinct behavior. Four native
regressions and a release parse-cost receipt are recorded in
`json-property-validation-2026-10-26.md`.
Gates: **1,274 native / 52 ignored**, serial SIGKILL gate **1 passed**; **2,004
root Python / 1 skipped**, **160 docs**, strict Clippy, in-tree extension rebuild.
No new checklist rows; **201 checked / 67 open**.

## 2026-10-26 — real property compaction and explicit orphan reclamation

Existing 15.2/10.4 property maintenance is implemented: `Graph.optimize` is no
longer a no-op; `compact_pool` deduplicates exact portable values and remaps every
working index. Optional `reclaim_unreferenced=True` discards slots absent from
every retained delta before/after value and working mapping. `compact_storage`
performs the same operation with verified locked atomic baseline replacement.
History/pins/groups/inactive IDs/external reservations survive, with explicit
rejection of stale/dirty/foreign captures. Failure coverage includes five injected
phases for both policies and twelve process-death cases. Local receipts and limits:
`pool-compaction-2026-10-26.md`.

Raw entity/group reclamation, complete referent closure and identity release remain
15.1/15.2/06.3. No scheduling/Graphiti/service scope change or new checklist row.
Current inventory remains **201 checked / 67 open**.

## 2026-10-26 — actual physical journal verification

`DurableGraph.verify_storage` reads actual named-file bytes through the shared
recovery parser and verifies all retained transitions/current schema. It never
repairs a tail; incomplete/corrupt/changed sources block the owner until reopen.
Native/Python tests cover byte/semantic corruption, independent cursors/locks,
partial tails, changed paths and chunked baselines. Evidence, explicit audit costs
and limits: `storage-verification-2026-10-26.md`. Derived index/cache and service
diagnostics remain within the original 33.2 scope. No new checklist row.

## 2026-10-26 — explicit retained-storage purge closes core retention

Existing **06.3, 15.1 and 15.2** are complete under the accepted A09 embedded-library
policy: previewable explicit purge reclaims orphan owner entities, group
registrations, raw values and external names, with complete retained-history roots,
independent immutable payloads and monotonic IDs. Sparse endpoint storage reclaims
interior holes and supports later high-ID writes. Full archives, owned captures,
verified durable replacement, reopen and backup retain purge state. No extra
permanent pins, grace period, mandatory backup or automatic scheduling was added.
Numeric ID arrays retain value semantics, not hidden ownership pins.

Evidence, local cost trade-offs and limitations: `retained-purge-2026-10-26.md`.
Gates: **1,308 native / 52 ignored**, dedicated serial **18-case** SIGKILL matrix,
**2,017 root Python / 1 skipped**, **167 docs examples**, strict all-target Clippy
and rebuilt in-tree extension. Existing Python exclusions/FFI warnings remain.
**204 checked / 64 open**, same 268 rows. Next: typed endpoint changes/reversals
(04.2/13.3), then three-way merges (14.2); remaining execution/default, view/index,
numerical/scale/release and deferred integration retain their original scopes.

### Endpoint comparison/safety foundation (04.2 / 13.3)

Endpoint-only snapshot differences previously disappeared; raw restoration could
silently overwrite an existing endpoint pair. Canonical typed endpoint diffs now
preserve sorted stable edge identities/before-after tuples and reach commit diffs
and Python output. Raw dense/sparse reconstruction rejects untracked rewiring,
and snapshot restoration preflights before replacing working membership.
Coordinated rewiring/reversal and recorded/effective/durable mutation remain open;
no wire format or owner policy changed. Fixed remaining inventory and evidence:
`endpoint-difference-foundation-2026-10-26.md`.

Gates: **1,310 native / 52 ignored**, **2,017 root Python / 1 skipped**, **168 docs
examples**, rebuilt extension and strict all-target Clippy. Same exclusions/known
FFI warnings. **204 checked / 64 open**, no new checkbox rows.

### Shared recorded-read plans and detached diffs

Recorded reconstruction and effective reads now share the existing immutable
lineage cache. Native captured diff/read plans survive prune/purge/owner destruction;
Python recorded diff reconstruction/comparison releases the GIL. Cloned plans share
lineage arrays, and identical-state diffs skip reconstruction after validation.
Cold ancestry walks and Python output marshalling remain owner-bound. A local
50k-property-version fixture measures 6.70 ms cold capture, 62.5 ns matching warm
capture and 2.94 ms detached diff execution; this is not whole-engine/FFI latency.
Evidence/limits: `recorded-diff-plans-2026-10-26.md`.

Final gates: **1,315 native / 52 ignored**, serial **18-case** process-kill matrix,
**2,018 root Python / 1 skipped**, **168 docs examples**, strict all-target Clippy
and rebuilt extension. Existing owner/legacy exclusions and known FFI warnings
remain. **204 checked / 64 open**, no new rows. Coordinated endpoint mutation,
reversal and merge planning remain the next core work.

## Endpoint revisions completed — 2026-10-27 UTC

Existing **04.2** is complete: stable-ID scalar/bulk/mixed endpoint changes and
ordered reversal retain properties, names, typed before/after history and explicit/
default clocks. Recorded/effective reads, indexes/caches, branch/older owned restore,
manual coalescing/reset, frozen sources and full durable replay/maintenance are
verified. Legacy membership-only groups explicitly reject unsupported rewiring;
no implicit source migration or new owner decision. Evidence/cost/limits:
`endpoint-revisions-2026-10-27.md`. General revert remains **13.3**; three-way
merge planning remains **14.2**.

**205 checked / 63 open**, same 268 rows. Gates: **1,335 native / 52 ignored**,
**2,022 root Python / 1 skipped**, **172 docs examples**, strict all-target Clippy,
rebuilt extension and serial **18-case** process-kill matrix. Existing owner/legacy
exclusions and known FFI warnings remain. No new rows, external installs or push.

### Explicit retained-state restoration (13.3 component) — 2026-10-27

Graph/DurableGraph and writers now restore an explicitly selected retained
published state, retargeting only the working/current-branch selection. Topology,
typed properties, external bindings and group revisions restore; future history,
other branches, allocation/name reservations, clocks and immutable pins remain.
Dirty/stale/foreign/collected/schema targets reject before mutation. Native detached
reconstruction runs off-owner/GIL, and durable reference records validate and replay
after sync with rollback/recovery/backup/maintenance coverage. No new mutation or
implicit history pruning. Same-state durable restores append no frame.

Existing **13.3 remains partial** for general tracked inverse mutations and their
recording/effective-time policy; this reference switch does not choose that policy.
**205 checked / 63 open**, same 268 rows. Gates: **1,342 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 docs examples**, strict Clippy, in-tree FFI
and serial 18-case recovery matrix. Local 100k-node restoration: **30.27 ms replay /
126.42 ms validation-install**; owner capture **2.54 µs**, no RSS/production claim.
Evidence: `state-restoration-2026-10-27.md`; API: `docs/api/state-restoration.md`.
Next core scope is caller-resolved three-way merges (14.2), with tracked revert
policy still explicit. No new checklist rows or owner-policy assumptions.

### Native three-way merge planner (14.2 component) — 2026-10-27

Native snapshot merge now combines independent node/edge/property/endpoint changes
and inventories typed conflicts with whole base/left/right values. Explicit caller
answers resolve entity/property/endpoints; recomputed dangling dependencies require
explicit edge deletion or exact missing-node retention. Parallel IDs remain distinct;
absence/Null, float types, signed zero and NaN payloads are preserved. Shared
structural validation replaces a public panic and duplicated integrity/restore checks.
Common-ancestor selection now returns all best bases and rejects ambiguous criss-cross
selection rather than silently choosing an older DFS match.

**14.2 is partial**, not complete: owning Graph capture/acceptance, full identity/
schema/source/time guards, one two-parent publication, durable replay and Python
surfaces remain. Evidence and fixed remainder: `snapshot-merge-planning-2026-10-27.md`.
**205 checked / 63 open**, same 268 rows. Gates: **1,351 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 Python docs examples** plus one executable
Rust merge example, strict Clippy, rebuilt FFI and serial 18-case recovery matrix.
Local 100k-node pure planning median **274.29 ms**, final validation **6.72 ms**;
no publication/FFI/RSS/production claim. No new checklist rows.

### Owner-bound merge capture/preflight (14.2 component) — 2026-10-27

Native Graph now captures owner/head/branch/base and two explicit retained parent
versions into a detached read plan. Typed caller resolution produces a privately
provenanced unpublished outcome. Read-only preflight rejects stale/foreign/changed
branches, collected parents/bases, unallocated IDs, missing group references,
conflicting external identity and optional full-target schema violations. Valid
compaction preserves logical results; explicit empty base handles unrelated roots.
Capture/resolution/validation do not allocate IDs/versions or change working state.

**14.2 remains partial**: exact first-parent delta and clock/timeline preparation,
current-policy revalidation, guarded one-version two-parent publication, durable
record/replay/failure evidence and Python APIs remain. There is no unchecked
finish_merge write API. Fixed evidence/remainder: `owner-merge-capture-2026-10-27.md`.
**205 checked / 63 open**, same 268 rows. Gates: **1,356 native / 52 ignored**,
**2,025 root Python / 1 skipped**, **176 Python docs examples**, strict Clippy and
rebuilt/staged in-tree FFI. The preceding planner's 18-case recovery gate remains
the latest storage check; this component has no journal/storage change.

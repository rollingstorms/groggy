# Groggy roadmap status — 2026-10-24

Branch: `groggy-roadmap-2026-10-20`. The [flat checklist](graph-engine-flat-checklist.md)
currently contains **193 checked and 75 unchecked entries**. Entries overlap:
parent workstreams and smaller repair slices both appear, and unchecked entries
include partial, open, review and deferred work. These counts are an inventory,
not a completion percentage or an estimate of remaining implementation effort.

Count correction: the earlier summary omitted three already checked A* entries
because its heading parser treated the literal asterisk as formatting. The earlier atomic-builder pass
added two checked slices; three other entries were a counting correction. The
recent loop/block, structural, filtering, aggregation, transformation, mask-sampling, community and bounded-selector continuation slices are counted individually alongside their open parents. Counts now use checkbox markers independently of headings.

## Fixed remaining-work inventory

The 75 unchecked entries are not 75 independent tasks: 14 are overlapping AR
architecture entries, 61 are numbered entries (including nested publication
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

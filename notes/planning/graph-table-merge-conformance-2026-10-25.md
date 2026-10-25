# Specialized GraphTable merge conformance — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. The six existing specialized merge strategies
and federated-domain naming are complete for the contract below. No new checkbox
rows. Broader collection ownership/conversion adapters remain in their existing
parents; pooled portability stays in 07.3 / 10.3 / 12.2.

## Fixed completed inventory

| Existing strategy/component | Verified behavior |
| --- | --- |
| Fail | Preserve disjoint node/edge IDs, rows and properties; reject collisions. Replaces the path that returned empty tables. |
| KeepFirst | Keep first complete colliding record, including properties/endpoints; stable output row positions. |
| KeepSecond | Keep last complete colliding record; absent columns become Null, with no leaking older attributes. |
| MergeAttributes | Later present columns win, Null included; absent columns retain prior cells. Ordered endpoint conflict rejects explicitly. |
| DomainPrefix | Dense independent IDs in domain/row order, complete local endpoint/reference maps, emitted domain/original-ID metadata. |
| AutoRemap | Preserve first/noncolliding IDs, remap later collisions to lowest unused nonnegative IDs while reserving all input IDs; same metadata for traceability. |
| Structural validation | All inputs, including single-input cases: strict IDs, no duplicate IDs/columns, local endpoint existence, typed referent existence. No cross-domain fallback. |
| Properties and policy | Complete union of attribute columns, precise inline values, global surviving external-ID uniqueness and conservative merged validation policy; no silent repair. |
| Source adapters | Node/edge output components keep borrowed input source captures read-only, without guessed row identity. Python releases the GIL for native merging/loading. |
| Federated naming | Caller domain names validated and actually emitted; no placeholder metadata mutation. Existing loader/format limitations remain explicit. |

A single private Rust `domain_merge` module owns validation, identity planning and
column-buffer construction. The previous paths either discarded collected attrs,
returned empty tables or fell back to IDs from another domain. Source metadata
construction now accepts borrowed table references, avoiding full table clones
solely for provenance. Public MergeOptions/strategy names and call shapes remain.

Typed NodeArray/EdgeArray values use shared materialization reference remapping;
plain integers/vectors/JSON/text remain opaque. Raw SubgraphRef rejects because
portable pool relocation is an existing separate requirement. No graph owner is
mutated or publication created by merging tables. Graph materialization is explicit.

## Verification

- Ten native tests exercise all six strategies, sparse/overlapping/future-reserved
  IDs, node/edge properties and reference arrays, f64 precision, whole-row versus
  attribute policies, missing columns, endpoint conflicts, duplicate/dangling IDs,
  pooled-reference rejection, policy/identity checks, frozen captures, explicit
  materialization and named federated bundles.
- Independent endpoint oracle over all 256 pairs of two-node directed topology
  masks, including loops, extra parallel loops, isolated nodes and empty edges.
- Thirteen Python cases exercise the public six-policy surface, single-input
  validation, values/topology, endpoint conflicts and source writeback rejection.
- Full native: 1,071 passed / 51 ignored.
- Root Python API suite against rebuilt in-tree extension: 1,802 passed / 1 skipped.
- 106 executable documentation examples, including three merge blocks.
- Strict all-target core Clippy, scoped rustfmt, Python/stub AST and diff checks pass.
- In-tree macOS FFI artifact uses dynamic Python symbol lookup; no wheel/package
  installation. Three preexisting unrelated FFI warnings remain.

## Current native costs

`cargo bench --offline --locked --bench graph_table_merge` records five optimized
observations per size/strategy in `notes/benchmarks/graph-table-merge-2026-10-25.json`.
Two chain tables with node integer and precise edge f64 properties. Fail uses
disjoint IDs; other policies use colliding inputs. Input preparation/cloning and
final checks are excluded; validation, ID maps, property transfer and complete
output construction are timed.

| Nodes per input | Fail ms | KeepFirst ms | KeepSecond ms | MergeAttributes ms | DomainPrefix ms | AutoRemap ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3.5968 | 2.6082 | 2.4823 | 2.2040 | 2.8623 | 2.7905 |
| 10,000 | 22.3770 | 15.8566 | 17.5769 | 18.0484 | 24.6720 | 25.7245 |

These are current-cost medians, without matched speedup, RSS/allocation, FFI or
large-scale latency guarantees. Maps and output buffers allocate; frozen sources
remain alive with merged outputs. Legacy CSV bundle loading is not certified as
lossless portable property/history/pool storage by this merge repair.

## Remaining existing scope

Specialized graph-domain merges are removed from the remaining table inventory.
Collection ownership when transformations remove all tables and broader graph/
table conversion adapters remain under 27/AR7.10. Complete pooled contents remain
under 07.3 / 10.3 / 12.2. Broad parent checkboxes remain partial until their full
exits are satisfied; the checklist still has 191 checked / 77 unchecked rows.

Follow-up: [collection ownership](table-collection-ownership-2026-10-25.md) closes
the supported empty selection/schema/callback/join ownership family. Broader
conversion adapters remain open under the existing table parents.

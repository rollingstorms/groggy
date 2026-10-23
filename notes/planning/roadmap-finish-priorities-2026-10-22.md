# Roadmap finish priorities — 2026-10-22

User direction: add module consolidation to the flat checklist and prioritize
completion of the existing roadmap while the owner finishes the workbook.
This is an execution order, not a replacement vision or a claim that unfinished
items are absent from the existing engine. Original 01–34 numbering is retained;
AR7 tracks consolidation. Pending workbook suggestions are not accepted decisions.

## Current execution checkpoint

Implemented: AR7.2 matrix consolidation, AR7.3 canonical arrays/shared numeric
foundations, 29.60–29.63 finite numerical fixes/receipts, AR7.4 entity ownership,
and AR7.5a–h shared read state, explicit immutable selections, frozen matrix/table
exports, scoped query predicates, prepared BFS/DFS and checked Dijkstra/A* reads. Evidence is
linked from the flat checklist. Supported-scope implementation is distinct from
remaining broad capability and production gates (including A13/A15).

Next: migrate component and centrality readers
onto the prepared immutable algorithm input,
retaining a separate working mutation owner. Source-preserving table adapters
remain separate from the completed owned table projections. Ordinary legacy
Subgraph/table reads are not yet certified frozen. Stages 3–6 retain their existing
partial/open work and specific workbook dependencies; no blank policy is accepted.

## Owner answers received — 2026-10-22

See [the answer record](workbook-answers-2026-10-22.md) and updated appendix.
Group revision references/selected-state copying, same-effective-time ordering,
endpoint validity rejection, persistent absence after unset, f64 floating defaults,
UTC instant/date/duration types, staged schema migration, mixed table writeback,
explicit index failure, component/path defaults and transient pipeline runs now
have recorded directions. These decisions are not implementation completion.

The accepted f64 default intentionally supersedes earlier preserve-f32-default
recommendations. Implement tagged f64 properties and bridge/codec/history/schema/
numerical/query migration before declaring the final property format complete;
keep existing tagged data readable. Native matrix f64 support alone does not do
this. Endpoint validity now rejects invalid timelines rather than silently hiding
an edge. No generic silent index-to-scan fallback is authorized.

Automatic-publication defaults, the exact restore source and purge protections
have clarification requests pending. A07.1 is tentative; A08 and other unanswered
fields remain open. Checkpoint scheduling is explicitly deferred until MVP/first
use; opening-time measurement can proceed now without inventing a log budget.
A10.1 concerns duplicated source rows, not automatic creation of graph entities.

## Immediate work, without waiting for the whole workbook

1. **AR7.2 — one matrix module.** Use the current 455 Python regressions, full
   native suites and 25-case release receipt as the migration baseline. Inventory
   exports/callers, move existing implementations into explicit internal modules,
   preserve legacy imports through re-exports, and compare behavior/performance.
   This changes ownership/organization while preserving current semantics.
2. **AR7.3 — array/numeric foundations.** Resolve duplicate BaseArray ownership
   and locate shared scalar/dtype/conversion/accurate arithmetic primitives below
   both arrays and matrices. Preserve typed arrays, precision and null policies.
3. **Close the numerical pass deliberately.** Convert the broad remaining 29
   parents into a finite inventory of supported surfaces and concrete findings.
   Complete the size/sparsity/fill/dispatch sweep, public API/doc and lint gates;
   separate working reference dispatch from actual SIMD acceleration claims.
   Fix discovered failures. Parent completion requires evidence across its stated
   scope; an optional enhancement does not automatically become a new blocker,
   and an unfinished promise requires implementation or explicit owner deferral.
4. **AR7.4/AR7.5 with AR1/AR2/AR4 — entity/view foundations.** Unify duplicate
   handles and shared membership/state ownership; make immutable read views pin
   the selected state and propagate it into numerical/query/table results.
   Current live-parent Subgraph behavior does not satisfy that accepted contract.
   Preserve retained inactive-ID inspection separately from current membership.

## Then follow the dependencies

### 1. State, values, changes and temporal reconstruction

Finish 04.2/05.3/06.3/07.3–07.4/09.4/10.3/11.2/12.2–12.3/13.3/16.2–16.3,
with AR3/AR6 and shared change records (AR7.9). Exact retained referents and typed
changes underpin state restoration and time queries. Checkpoint operations and
reconstruction are separate: accepting a property/bulk mutation records its delta
with explicit effective time or the accepted batch default; reading/computing does
not create a change. A requested snapshot is a checkpoint. Optional recorded_by
remains explicit. Test unambiguous cases and prepare the machinery now; await the
specific unanswered policies before choosing correction/lifecycle semantics.

### 2. Durable lifecycle, branches and retention

Finish 08.3/10.4/14/15. Portable retained storage and exact states precede trustworthy
merges/reclamation. Implement checkpoint/recovery/format-validation mechanisms
separately from choosing automatic scheduling, schema migration, merge resolution
or destructive retention policies. Verify crash recovery before publication and
reclamation claims. Do not silently purge history while the workbook is pending.

### 3. Indexed queries, tables, algorithms and pipelines

Finish 18/21/27/remaining 28/30, alongside AR7.7–AR7.8/AR7.10. Index and query results
must share active/state/time semantics. Tables need explicit identity/writeback.
Pipelines consume corrected primitives and pinned inputs. Remaining Infomap,
non-registry analytics and algorithm certification remain named work. Pipeline
run-history and ambiguous writeback are decision gates, not permission to invent
policy. Independent algorithm/reference work can proceed while other gates wait.

### 4. Core verification, presentation and operations

Carry 01.2/19/31/32/34 through every preceding step. Then finish the complete
comparison/failure/scale reports, display consolidation (AR7.6), smaller facade
work (AR5), compatibility retirement gate (AR7.11) and operational integrity.
09.5/33.3 need an explicit service boundary: embedded correctness does not complete
the multi-user objective. Benchmark target fixtures can be published provisionally;
owner-specific performance budgets remain unanswered until selected.

### 5. Deferred integration remains part of finishing the full vision

After the core milestone, resume 17/20/22–25/32.4: provenance/episodes, Graphiti
round-trip/idempotency, lexical/exact-vector/hybrid retrieval and operation driver.
Provenance/storage primitives can be prepared earlier when their contracts are
settled. Choose the Graphiti release/fixture before adapter compatibility claims.
Completing the core must not be reported as completing all 34 items.

## Workbook reading priority

The owner does not need to complete every page before implementation resumes.
Use the existing open-questions appendix; do not overwrite answers or treat blank
fields as approval.

| Priority | Appendix | What the answers unlock |
| --- | --- | --- |
| First | A01, A03–A06 | Retained group/meta identity/copy rules; migration of legacy writes/names; same-time corrections and endpoint validity; unset/restore semantics; temporal types and property float precision. Most now have owner answers; see the answer record for the remaining publication/restore clarifications. |
| Next | A02, A07–A11 | Checkpoint/upgrade budgets, schema evolution, merge conflicts, retention/purge, table writeback and index loading/fallback. Some answered; schema-branch/merge/purge/rebuild choices still need resolution before their lifecycle behavior is finalized. |
| Alongside algorithm/numerical closure | A12–A13, A16 | Compatibility choices, numerical guarantees and duplicate/retry extensions. Preserve existing established direction, exact-shape, checked precision and rejection rules until an explicit change is accepted. |
| Before pipeline/service commitments | A14–A15 | Intermediate naming/run provenance and actual scale/service budgets. These do not block module moves or independent correctness work. |
| After core, before Graphiti integration | A17 | Compatibility release and correction-versus-parallel-fact adapter identity. |

If a selected task reaches an unanswered policy gate, finish all independent
preparation and report that specific dependency; continue the next independent
checklist item. No elapsed-time assumption or provisional recommendation counts
as the owner's answer.

## Completion discipline

- Each task identifies its stable checklist ID, concrete behavior and exit checks
  before implementation; commit coherent completed items separately.
- Preserve existing functionality, import compatibility and accepted decisions.
- Add verification at the layer changed, including Rust/Python and applicable
  persistence/state/time behavior. Capture before/after performance for structural
  numerical/graph kernel changes.
- Keep broad unchecked parents accurate; historical completed slices retain their
  scope. Finish conformance, public documentation and examples as part of the item.
- Record discovered defects under the responsible item. Distinguish optional new
  capabilities from necessary repairs so finishing does not become endless scope
  expansion. Deferring original vision work requires the owner's direction.


### AR7.5i completed — frozen components

Connected component views now share captured state and existing kernels, with the
accepted default/weak/strong modes. See `frozen-components-contract-2026-10-22.md`.
Next independent read migration: centrality. Restore/purge/publication policy
questions remain pending; this slice does not choose them.


### AR7.5j completed — frozen PageRank

Captured PageRank now shares one validated power-iteration kernel with the live
wrapper. Owned results expose finite-budget convergence status. See
`frozen-pagerank-contract-2026-10-22.md`. Next independent read migrations:
closeness and betweenness; broader AR7.5 remains open.


## AR7.5k — Frozen closeness (2026-10-22)

Completed captured classic/harmonic closeness reads using the existing distance/
scoring kernel and a shared compact incoming CSR transpose. Preserve defaults,
normalization and strict selected weight rules; return owned f64 rows without
publication. See `frozen-closeness-contract-2026-10-22.md`. Betweenness and broad
AR7.5 remain open.

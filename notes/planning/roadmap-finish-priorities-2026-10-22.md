# Roadmap finish priorities — 2026-10-22

User direction: add module consolidation to the flat checklist and prioritize
completion of the existing roadmap while the owner finishes the workbook.
This is an execution order, not a replacement vision or a claim that unfinished
items are absent from the existing engine. Original 01–34 numbering is retained;
AR7 tracks consolidation. Pending workbook suggestions are not accepted decisions.

## Current execution checkpoint

Implemented: AR7.2 matrix consolidation, AR7.3 canonical arrays/shared numeric
foundations, 29.60–29.63 finite numerical fixes/receipts, AR7.4 entity ownership,
and AR7.5a–b shared read state plus explicit immutable selections. Evidence is
linked from the flat checklist. Supported-scope implementation is distinct from
remaining broad capability and production gates (including A13/A15).

Next: migrate existing graph/table adapters and read consumers onto that shared
state/membership, retaining a separate working mutation owner. Ordinary legacy
Subgraph/table reads are not yet certified frozen. Stages 3–6 retain their existing
partial/open work and specific workbook dependencies; no blank policy is accepted.

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
| First | A01, A03–A06 | Retained group/meta identity/copy rules; migration of legacy writes/names; same-time corrections and endpoint validity; unset/restore semantics; temporal types and property float precision. These are the next substantial core history decisions. |
| Next | A02, A07–A11 | Checkpoint/upgrade budgets, schema evolution, merge conflicts, retention/purge, table writeback and index loading/fallback. Needed before corresponding durable/lifecycle behavior is finalized. |
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

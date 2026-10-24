# Owner workbook answers — 2026-10-22

Source: the owner's dictated answers to the open-questions appendix and subsequent
explicit clarification replies in this chat. Accepted directions authorize future
implementation; they do not claim that the code already implements them. Only
answered questions change status. Unanswered or qualified choices remain visible.

## Answer mapping

| Question | Owner answer / clarification | Recorded direction and status |
| --- | --- | --- |
| A01.1 | Subnode changes and meta-node changes are distributed; confirmed new reference per revision. | **Confirmed:** target subnode and meta properties independently. An explicit membership/source-pin revision receives a new immutable reference; older references remain readable. Do not infer automatic propagation to every containing group. |
| A01.2 | Suggested. | **Confirmed:** copy/materialize the selected state by default; transfer history only explicitly. |
| A02.1 | Defer checkpoint schedule until MVP/first use; question a maximum budget; benchmark opening. | **Confirmed:** measure recovery/open time now; defer scheduling thresholds and mandatory log/disk limits. Do not invent an opening SLA. |
| A02.2 | Suggested; asks whether checkpoint is compressed state. | **Confirmed:** verified side-by-side format conversion, retaining original until verification. A checkpoint is saved state/recovery information; compression is optional. |
| A03.1 | Yes, migrate defaults now. | **Confirmed (2026-10-23):** checked automatic publication becomes the default for ordinary Graph mutations now; commit means an explicit checkpoint. Publication errors must remain visible and a failed batch must publish nothing. |
| A03.2 | Check examples, but change everything now. | **Confirmed:** audit real repository examples as compatibility fixtures and proceed with the requested terminology/API migration now. A03.1 now separately confirms automatic-publication defaults. |
| A04.1 | Only explicit changes/effective time; confirmed later accepted write wins equal-time conflicts. | **Confirmed:** mutation events carry effective time; reads/computation create no changes. At identical effective time for the same key, later accepted version wins. Explicit recorded cutoff can reveal the earlier accepted value. |
| A04.2 | Reject it. | **Confirmed, overrides suggestion:** reject an effective edge whose endpoint validity does not support it; do not silently retain/hide invalid topology as the default. Backdated combinations need whole-proposed-timeline validation. |
| A05.1 | Remain absent. | **Confirmed:** unset persists until a later set; no fallback to an older value. |
| A05.2 | Restore unless explicit. | **Confirmed correction (2026-10-23):** restoration requires an explicit retained state; do not select a pre-deactivation state automatically. |
| A06.1 | Suggested. | **Confirmed:** first-class UTC instant, calendar date and duration first; distinct versioned types. Local/timezone-aware calendar types need their own precise rules later. |
| A06.2 | Change default; asks whether f32 was for memory. | **Confirmed:** ordinary Python floats default to f64. This intentionally supersedes the proposed preserve-f32 default. Preserve existing typed data during storage migration; do not pretend widening recovers already-rounded bits. |
| A07.1 | “You could, I guess.” | **Tentative:** no definite choice between branch schema identity and global installed policy. Keep the current policy until clarified. |
| A07.2 | Okay. | **Confirmed:** explicit compatible migration stages, each atomic and separately recorded. |
| A09.1 | No. | **Proposal rejected:** do not infer extra permanent named-checkpoint/audit retention roots. **Confirmed (2026-10-23):** purge is explicit and must preserve shared branches and open immutable views. |
| A09.2 | No. | **Proposal rejected:** the combined grace-period/backup/audit proposal is not accepted. **Confirmed (2026-10-23):** no mandatory grace period or backup; explicit purge still preserves shared branches and open immutable views. This is currently a data science library policy, not a production database retention guarantee. |
| A10.1 | Conditional yes if joining duplicates things. | **Conditional interpretation:** repeated table rows can map to the same source entity; a join does not itself create graph entities. Reject ambiguous/conflicting writeback without explicit target mapping/reduction. No silent last-row-wins update. |
| A10.2 | One batch. | **Confirmed:** one mixed atomic writeback for unambiguously identified node and edge updates. |
| A11.2 | Fail. | **Confirmed, overrides suggestion:** fail when an index required by the selected query execution is unavailable; scanning must be explicit, not an automatic fallback. This does not make today's ordinary scan API require a nonexistent optional index. |
| A12.1 | Okay. | **Confirmed:** preserve documented compatibility default; expose explicit weak/strong component modes. This does not accept unrelated provisional self-loop degree changes. |
| A12.2 | Yes. | **Confirmed:** one deterministic optimum by default; equal-optimum enumeration is a separate bounded operation. |
| A13.2 | Change “S64/X64” default; confirmed f64 floating point. | **Confirmed:** f64 floating-point defaults/support; this is not an integer-width change. Keep real error/convergence domains and residual/tolerance evidence. Existing numerics already largely use f64; property storage/FFI still require migration. |
| A14.2 | Correct. | **Confirmed:** pipeline runs are transient by default; persistence/provenance is explicit. |
| A15.1 | Include compute time and wall time. | **Confirmed measurement addition:** report elapsed computation and end-to-end wall time; label process CPU separately when collected. Specific graph/vector/history/hardware fixtures and budgets remain unspecified. |
| A16.1 | Explicitly. | **Recorded interpretation:** reject duplicate assignments by default; identical coalescing is an explicit option and conflicting assignments still fail. No implicit retry-key policy is chosen. |
| A17.1 | Real use later; use a test example. | **Confirmed sequencing:** use a synthetic documented integration fixture initially; a real Graphiti compatibility release/use workload is not yet selected. Integration stays deferred behind core work; do not claim a hand-made fixture proves upstream compatibility. |

## Explicit clarification replies

- “S64/X64” means **f64 floating point**; ordinary Python floats should default to f64.
- For two explicit writes at the same effective timestamp, **later accepted write wins**.
- Membership/source-pin group revisions use **new immutable references**, retaining the old reference.

- Ordinary mutations adopt checked automatic publication now; commit is an explicit checkpoint.
- Entity restoration requires an explicit retained state.
- Explicit purge preserves shared branches and open immutable views, without extra permanent checkpoint/audit pins or mandatory grace/backup. Scope is currently a data science library.

## Questions not answered by the original message

A08.1–A08.2 (rewiring/merge),
A11.1 (index recovery versus rebuild), A13.1 (shape/broadcast policy), A14.1
(stage naming), A15.2 (service limits), A16.2 (durable retry key), and A17.2
(adapter correction/fact identity). A07.1 remains tentative. A15.1/A17.1 supply
measurement/fixture direction, not complete production targets. A03.1, A05.2 and A09 were clarified on 2026-10-23. A question
being unanswered in this message does not undo earlier accepted behavior; existing
exact-shape operations and other established contracts remain in force.

## Answers to the owner's explanatory questions

### Is a checkpoint a compressed state?

A logical checkpoint identifies a saved state. A physical recovery checkpoint can
materialize that state plus recovery metadata and a journal position, so opening
starts there and replays later changes. Compression is an independent encoding
choice. It need not discard retained history. Current DurableGraph opening still
replays the full journal; the existing full-log backup is not a compact checkpoint.
See `docs/api/durable-storage.md` and A02.

### Was f32 chosen to save memory?

Current `AttrValue::Float`/`FloatVec` store f32. Comments in `src/types.rs` emphasize
embeddings/coordinates and cache-friendly vector storage. Python conversion in
`python-groggy/src/ffi/utils/helpers.rs` narrows incoming floats and float lists.
This supports a compact-storage motivation; an original documented scalar-default
rationale has not been established. f32 numeric elements use 4 bytes versus 8 for
f64. Vector payloads therefore double when widened; total graph memory need not
double because containers, IDs, indexes and enum layouts have other costs.
Historical f32 values must remain faithfully tagged/readable; lost precision
cannot be reconstructed simply by widening them. Future defaults and explicitly
typed storage are distinct from rewriting retained data.

### Does a join duplicate nodes in the graph?

A join can repeat one source node/edge across several table rows. It does not
itself allocate graph entities. Writing multiple differing assignments back to
that one identity needs explicit mapping/reduction; accepted mixed updates are
one atomic batch. This is the condition used to interpret A10.1.

## Implementation impact

- Retained references can proceed with per-revision immutable identity and explicit
  selected-state copies; preserve distributed mutation targets and historical pins.
- Temporal replay now has an accepted equal-time ordering and endpoint rejection
  rule. Validate full proposed timelines atomically, including backdated writes.
- f64 property/default migration and native temporal tags must update the typed
  value bridge, schema, codecs, history/journal and numerical/query adapters.
  These are requested changes, not already completed by f64 matrix support.
- Mixed table writeback, index-failure policy and transient pipelines have clearer
  contracts; source identity/expected-state checks remain required.
- Checkpoint scheduling stays deferred. Measure actual opening cost before setting
  thresholds. No production maximum or SLA is inferred from a small baseline.
- Unanswered merge/rebuild/shape/service/retry/adapter choices remain open. No
  implementation or destructive retention policy is selected by blank fields.

## Initial opening-time measurement

A bounded warm-cache baseline now exists in
[`notes/benchmarks/journal-open-development-2026-10-22.md`](../benchmarks/journal-open-development-2026-10-22.md).
With one live node, median open wall time is approximately 0.130 / 5.684 / 56.260 ms
for 1 / 101 / 1,001 accepted batches. Seven samples per case; process CPU is
recorded separately. The current in-tree development extension replays the whole
journal. This is not release/cold-device/large-graph performance or a production
budget. Checkpoint scheduling and maximum-log limits remain deferred as requested.

## A06.1 implementation — 2026-10-23

07.3b now implements distinct native UTC instant, calendar date and fixed elapsed
duration properties. Rust/Python share exact microseconds, calendar years 1..9999
and the full Python timedelta domain. Aware datetimes normalize to UTC; naive
datetimes reject. Versioned payloads retain values through history, export and
checked journal recovery. See [implementation and verification](temporal-properties-2026-10-23.md).
Local/zoned calendar types and temporal arithmetic remain separate future scope.

## A03.1 checkpoint boundary — 2026-10-23

12.3a now supplies explicit checkpoints of clean published states, including
writer sessions and durable recovery. Clean `commit()` labels that state without
a duplicate mutation delta. Dirty legacy `commit()` retains a publication
compatibility path. 09.4a adds native fallible checked constructors; switching
ordinary Rust/Python defaults and auditing pooled/meta/compound mutations remains
09.4b under the existing authorization. See [implementation record](explicit-checkpoints-2026-10-23.md).

## A03.1 ordinary Python defaults — 2026-10-23

09.4b.1 switches ordinary Python creation/mutation defaults to checked publication.
Bulk calls publish once, ordinary time defaults use the acceptance clock, and
explicit writer/batch effective timestamps remain available. `commit()` records
an explicit checkpoint. `Graph(manual=True)` names the legacy deferred path.
Remaining native infallible/default compatibility and multi-stage operation audits
are 09.4b.2, already authorized. See [implementation record](automatic-publication-2026-10-23.md).

# Owner-bound merge capture and preflight — 2026-10-27 UTC

Existing **14.2 component**, following `7f90d270` native snapshot planning/caller
resolution. This prepares owning acceptance without exposing an unchecked write
path. Parent 14.2 remains partial; no new checklist rows or policy assumptions.

## Delivered

Graph.prepare_merge_read takes explicit right-hand retained state, optional
explicit base and expected current head. It captures owner identity, branch and
both parents with three immutable RecordedReadPlans. Unique best-base selection
is automatic only when unambiguous; non-common/unknown bases reject. Unrelated
retained roots require explicit base zero; identical/unknown/zero parents and
dirty/stale input reject before capture. This does not reserve a merge version.

GraphMergeReadPlan.execute reconstructs and plans without mutable Graph access.
GraphMergePlan exposes typed conflicts and captured base/parents; conflict-free
or explicit caller-resolved outcomes retain the same private provenance coordinate.
All three native public types are Send + Sync. GraphMergeOutcome exposes only an
immutable logical snapshot with unpublished state zero; it has no write authority.

Read-only Graph.validate_merge_outcome rechecks current head/branch, owner identity,
still-retained parents/base, ancestry, structure/direction, allocated/unpurged IDs,
source-group references, external identity bindings and optional complete schema.
Logical values survive intervening valid compaction. Source version collection,
foreign archived/reopened owners, same-numeric-state branch changes and newer
working states reject. Custom resolutions cannot introduce unallocated IDs,
missing pooled sources, stolen external names or schema-invalid results. Capture,
resolve and validation leave versions, allocation high-water marks, clocks, working
state, immutable pins and identity reservations unchanged.

## Verification

Five focused native integration cases cover detached thread execution, Send/Sync,
base/parent reporting, compaction, non-publication/non-allocation, stale/foreign/
branch/collection guards, dirty state, invalid/non-common bases, explicit unrelated
empty-base merge, schema/reference failures, custom nonexistent endpoint restoration
and attempted external-identity theft. Final verification: **1,356 native passed / 52 ignored** (including the existing
executable Rust merge example), strict all-target Clippy, rebuilt/staged in-tree
FFI with the three existing unrelated warnings, **2,025 root Python passed /
1 skipped**, and **176 executable Python docs examples**. The same owner benchmark
and nested legacy survey exclusions remain. Scoped Rust formatting and diff checks
pass. The previous planner commit's serial 18-case recovery matrix passed; this
read-only capture component changes no journal/storage implementation.

No Python API, durable record, storage-format change or implicit mutation is
introduced. The pure/native planner benchmark remains the previous component's
receipt; no capture/publication/RSS or new latency claim is inferred here.

## Fixed 14.2 remainder

1. Lower the resolved outcome to a complete exact typed delta relative to its
   first parent, including existing allocated-ID restoration and group/identity data.
2. Prepare effective/default/explicit clocks and validate full affected topology
   timelines plus current owner/schema/provenance immediately before acceptance.
3. Publish exactly one version with two parents and retarget the selected branch,
   with no partial visible result or accidental allocation/pending manual changes.
4. Durable synced record/replay, failure/reopen/backup/maintenance evidence and
   thin Python plan/resolution/writer translation with conformance examples.

Do not treat read-only preflight as a cached authorization or publish via raw state
installation. There is intentionally no finish_merge write API until the full
coordinator exists. **205 checked / 63 open**, same 268 rows.

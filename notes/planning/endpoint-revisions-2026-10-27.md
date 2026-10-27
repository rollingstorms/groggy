# Coordinated endpoint revisions — 2026-10-27 UTC

Existing scope: **04.2 complete**, endpoint/reversal component of **13.3**.
Branch: `groggy-roadmap-2026-10-20`. This is the next authorized priority slice,
following explicit retention and recorded diff plans. No new checkbox rows.

## Contract delivered

An existing edge retains its ID, properties, semantic kind and external identity
while its ordered endpoint pair changes. Rust/Python scalar, bulk and mixed
GraphBatch writes use one coordinator/acceptance. Source/target handles may refer
to nodes created by the same batch; new edges specify their initial pair through
creation. Duplicate IDs, inactive/missing targets, removal conflicts, invalid
resulting endpoints and schema failures reject before allocation/publication.
Rewiring away from a removed node is supported; implicit cascades use the resulting
topology rather than deleting that rewritten edge.

`EdgeEndpointChange` is canonical across ChangeSet, recorded Delta, snapshot/
commit diffs, legacy DeltaObject output and TemporalDelta. Whole typed before/after
pairs are retained; stable sorting/domain-tagged hashing preserves older no-revision
hashes. IndexDeltaStrategy coalesces manual revisions and updates pending initial
edge pairs, using a lazy position index rather than repeated linear searches.
External custom strategies explicitly opt in; unsupported tracking rejects.

Explicit per-change effective times use `EdgeEndpoints`/`edge_endpoints`. One
engine clock supplies the default per batch. Later accepted writes win ties.
Revisions require an effective live edge and active endpoints; backdated removal
cannot implicitly resurrect an edge through a previously accepted revision.
Recorded before-pairs are independent of effective event ordering. Same-pair writes
remain events, matching existing property semantics.

Ordinary Graph methods respect automatic/manual policy. Writer/batch calls always
publish checked writes; explicit-time Python direct conveniences select that path.
Manual pending initial/repeated revisions coalesce, and reset_hard restores the
accepted pair. Dirty checked writes reject. Empty ordinary bulk/reversal is a no-op;
empty explicit batches retain their existing invalid-input behavior.

## Ownership, caches, indexes and recovery

Raw pool pairs materialize the selected state, not a timeless edge identity.
Untracked raw restoration remains guarded. Trusted recorded installation restores
historical pairs, including older owned selections that retain later versions.
GraphSpace invalidates cached adjacency even when edge membership/IDs stay fixed.
Immutable recorded/effective views, owned captures and frozen group sources keep
independent topology. Directed incoming/outgoing caches, undirected storage order,
loops and parallel instances are covered.

TemporalIndex retains its original compact endpoint map for unchanged edges and
allocates revision timelines only for changed edges. Recorded point queries use
binary search; window queries intersect endpoint/liveness intervals in a linear
merge. TemporalDelta, changed-edge inventories and builder edge-diff outputs include
revisions rather than rejecting or dropping them. FFI diff records share one
translation helper and shape; no graph business logic moved into FFI.

Legacy membership-only groups explicitly reject member rewiring because they have
no pinned source revision. This preserves their existing meaning. Frozen sources
are supported; unattached legacy registrations can be explicitly purged. No implicit
migration or owner policy choice is invented. General connected-view migration
remains in its existing 26/AR scope.

Full retained archives use document v4 when revisions occur. Readers accept v1–v4,
old empty-field shapes/hash values stay readable, and downgrades reject. Journal
batches persist the new operation/original clocks; recovery computes before pairs
from the accepted parent. Reopen, backup, checkpoint, property compaction and explicit
purge retain revisions. Purge protects both old/new endpoint dependencies. Endpoint
append rollback tests cover before header, after header, after payload and after
sync; the generic serial 18-case process-kill matrix remains a separate gate.

## Tests and measured cost

- Native **1,335 passed / 52 ignored**.
- Root Python **2,022 passed / 1 skipped**, with the existing owner benchmark/nested
  legacy exclusions. Strict bool ID/time rejection covers new conveniences.
- **172 executable documentation examples** (four new endpoint examples).
- Strict all-target Clippy; rebuilt/staged in-tree extension, three known unrelated
  FFI warnings; serial process-kill parent **1 passed / 18 cases**.
- Scoped Rust formatting, Python/stub syntax and diff checks. No installs/push.

Coverage includes typed hashing/coalescing/counts, manual pending creation/reset,
directed/undirected/parallel/loop behavior, cached adjacency, temporal indexes and
diffs, explicit/tied/backdated clocks, mixed new-node/removal/property batches,
atomic schema/malformed input rejection, retained false-before detection, source
ownership, older owned captures, sparse storage, pruning/purge closure, journal
rollback/retry/reopen/backup/maintenance and archive-version gates.

Receipt: `../benchmarks/endpoint-revisions-2026-10-27.json`, generated by
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline
--locked --bench endpoint_revisions`. Three local release samples; setup, input
construction and old-view capture excluded. No new edge IDs are allocated.

| Parallel edge instances revised | Checked mutation median | Effective read median | Static temporal-index build | Revised temporal-index build |
| --- | --- | --- | --- | --- |
| 1k | 1.22 ms | 0.96 ms | 0.19 ms | 0.41 ms |
| 100k | 194.36 ms | 125.97 ms | 19.23 ms | 55.87 ms |

This is a new capability cost receipt, not an overall-engine before/after speed
claim. Native bulk mutation, effective replay and index construction are separate
measurements. No isolated RSS/FFI/production budget claim. Bulk checked writes still
run on the owner thread; broader GIL migration remains 19.2.

## Fixed next work

**204 checked / 64 open → 205 checked / 63 open**, same 268 rows. Existing 04.2
closes. Existing 13.3 retains general published-state restoration/revert, dirty
state policy/examples and destructive-versus-tracked recording distinctions.
Endpoint reversal is implemented; it is not a general historical revert. The
accepted decision record already permits an explicit state switch while retaining
history until separately requested destructive pruning. Three-way merge planning
and caller resolution (14.2) follows. No requirements were retired or newly added.

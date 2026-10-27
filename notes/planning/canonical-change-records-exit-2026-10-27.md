# Canonical change records — AR7.9 exit


## Canonical change-record closure — 2026-10-27

Existing AR7.9 is complete within its typed-record/representation/compatibility
scope. One native ChangeSet owner and snapshot difference builder serve recorded
diffs, merge and tracked restore. Exact value comparison also serves archive
validation and delta interning. Signed zero, NaN payloads, dtype and inactive-row
regressions are covered; owned payload buffers move into immutable deltas.

Physical offsets have an explicit borrowed IndexChangeView and complete indexed
DeltaObject projection. Fabricated index_N values and the silently empty adapter
are removed. Pool-independent logical extraction now rejects property offsets;
explicit retained-pool resolution preserves actual values and absence. Raw-space
delta extraction rejects because GraphSpace owns no pending changes. Native
fallibility/strategy compatibility changes are documented; persisted formats and
Python shapes are unchanged.

Verification: 1,383 native / 56 ignored; 2,042 root Python / 1 skipped; strict
all-target Clippy and rebuilt FFI. All **183 executable documentation examples** pass, including the new
change-record guide. Matched alternating-order local release medians:
100k property-free snapshot differences 7.05 → 5.31 ms; 10k nodes with 256-wide
f64 vectors and one change 12.14 → 8.20 ms. These are two fixture comparisons,
not whole-engine/RSS/production guarantees. Receipt:
`../benchmarks/canonical-state-changes-2026-10-27.json`.

Fixed inventory now **208 checked / 60 open**; no new rows. Remaining native
compound topology acceptance and coordinator compatibility guards stay in
09.4b.2/AR3; broader view/index/query/release work remains unchanged.

## Fixed component inventory

| Component | Exit |
| --- | --- |
| Shared property difference identity | AttributeChange/EntityType stay under state::change, with exact legacy re-exports. Diagnostic/sampling entity scopes remain distinct domains. |
| Shared accepted mutation payload | ChangeSet moves from change_tracker to state::change; existing paths re-export the same type and fields. |
| Snapshot-to-change conversion | One deterministic builder covers exact typed before/after properties, membership and ordered endpoints across snapshot/history/merge/restore. |
| Value comparison | Shared normalized portable payload semantics, with native fast paths tested against the codec across every supported family. |
| Physical pending changes | Borrowed typed offsets, actual topology and endpoint revisions; no fabricated logical values or silently dropped index payload. |
| Pool resolution | One shared resolver checks retained indices and preserves first-before/last-after, Null/absence and exact types. |
| Immutable accepted delta conversion | Sort and move buffers; unchanged legacy hash bytes and wire fields. Pointer tests prove large vector/string buffer transfer. |
| Selected/history/effective ownership | GraphSpace, HistoryForest, clock metadata/indexes and column indices remain separate responsibilities. No new persisted schema is introduced. |
| Compatibility/evidence | Existing imports, native fallibility/strategy migration docs, full native/Python tests, archive/replay fixtures and matched release controls. |

## Deliberate native API corrections

- ChangeTracker.create_delta and no-pool create_change_set/create_changeset now
  return GraphResult. Properties require create_changeset_resolved(pool).
- TemporalStorageStrategy exposes indexed_changes; default typed conversion and
  pool resolution replace independent conversion implementations.
- DeltaObject.new_with_indices is checked and preserves actual index data in its
  explicit index_changes field. Equality checks payloads after hash equality;
  malformed unencodable raw objects retain reflexive representation equality.
- GraphSpace.create_change_delta returns an explicit authority error. It never
  had enough information to derive pending mutations from selected state alone.
- GraphSnapshot.diff_with keeps its infallible signature; compared unencodable
  values are conservatively changed. try_diff_with is the checked path used by
  historical readers. Comparison is not an integrity scan.

No history/archive/journal format, acceptance/effective-time rule, ID policy,
retention pin or Python return shape changes. Benchmark control is a frozen copy
of d2abe891's old snapshot diff under benches/support, never a library path.
Finite benchmark fixtures deliberately avoid cases where old logical equality
was wrong. Correctness regressions cover those cases separately.

## Remaining work in existing scopes

AR3 still owns complete coordinator boundaries. Two old tracker helpers for
generic reverse-operation generation and conflict prediction still return empty
placeholders; replace those with explicit supported behavior or visible errors
before certifying that legacy surface. High-level tracked restoration and merge
use the tested owner-bound APIs, not these helpers. Compound topology execution
remains 09.4b.2; view/query/index lifetimes and production gates remain separate.

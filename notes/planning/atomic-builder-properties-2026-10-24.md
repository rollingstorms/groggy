# 09.4b.2g / 30.1b — Explicit atomic property builder boundary

Continues the coordinator/pipeline priority. Existing persistent builders publish
attachments individually, so a later failure can leave a partial graph result.
This slice establishes a supported atomic mode while retaining legacy behavior
for step families whose reads/effects have not yet been migrated.

## Implementation

- Native property-only staging has one owner under algorithms/property_batch.
  Context holds its lifetime; columns are indexed by name/entity with last-write
  values. No full graph clone, tentative pool writes or rollback snapshot is used.
- StepScope attachment methods validate entire columns before staging. Property
  loads, weight scale and smoothing read through the overlay; kind normalization
  agrees with graph setters. Init/arithmetic/normalization/flow families declare
  their read/effect compatibility through the Step trait.
- Custom/unsupported steps default to non-support. The native pipeline rejects an
  atomic request containing them before execution. Sampler pipelines also reject
  this request explicitly. Native nested property transactions are not inferred.
- Success rechecks the captured clean head, then uses one canonical mixed batch
  and acceptance clock. Failure/cancellation discards staging and clears Context.
  Empty assignments accept no version. Explicit atomic runs on manual graphs
  require a clean head and publish once, preserving the ordinary manual policy.
- Python build(atomic=True) serializes an optional native definition flag. It
  retains original statement order and bypasses the current IR optimizer, which
  does not encode graph-property effects. Existing normal builds and their JSON
  remain unchanged when atomic mode is not requested. The FFI adds no business
  logic or graph copies; this boundary is native.
- Nonpersistent execution keeps the existing transient result contract: no
  property overlay is installed, graph reads stay unchanged, and output columns
  remain owned results. An outer sequence of algorithm handles is not made one
  transaction by an inner atomic builder.

## Remaining work and performance scope

09.4b.2 / AR7.8 / 30 remain partial: migrate unsupported nested/loop/fused/graph
step families, certify effect-aware optimization, finish default migration and
outer composition/temporal/input ownership, and retain native compatibility.
The accepted transient-default migration remains open; no new approval is needed.
This does not provide durable publication or multi-user service isolation.

Staging uses ordered maps over touched cells, with column-sized validation/value
storage. It trades temporary memory and ordered-map lookup work for one final
publication. Original-order execution may omit legacy IR optimizations. The
native release receipt `benches/atomic_builder_properties_2026-10-24.json` measures
1k/10k-node runs with two/eight attachment columns, five observations per case,
checking read-after-write and one accepted version. Fixtures/build are excluded.
It is not a matched old/new speedup, FFI, allocation/RSS or production budget.
The broader matched performance comparison stays deferred behind the owner's
requested implementation priorities.

See [API and executable example](../../docs/api/atomic-builder-properties.md).

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **936 passed,
  52 ignored**, zero failures. New native cases cover mixed coalesced values,
  normalization/empty columns, stale/dirty rejection and cancellation after an
  attachment with subsequent Context reuse.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed, including the new benchmark.
- Rebuilt the in-tree extension through offline locked `cargo rustc` with the
  existing macOS dynamic lookup linker options and staged the changed Python
  builder sources. The three pre-existing FFI warnings remain unchanged.
- **Eight new Python atomic-builder cases passed**. Full root API gate excluding
  the unrelated owner algorithm benchmark: **1,084 passed, 1 skipped**. The
  separate legacy `tests/modules/` survey is not part of this gate.
- **37 documentation examples passed**, including ordered read-after-write with
  one accepted state. Storage examples ran in a temporary directory.
- Four release benchmark cases, five samples each, verified read-after-write and
  one clean accepted version. Raw native timing receipt retained; no matched
  speedup or production-budget claim.
- `git diff --check` passed. Owner development notes/benchmark data/tests remain
  outside this change.

# Pending core compute steps — 2026-11-15

## Completed slice

Seventeen existing step types now certify the appropriate pending execution
contract: four filtering/ordering, four numerical map, three fused, three
community and three sampling-mask steps. Filters, top-k, fused arithmetic,
community kernels and map samplers use pending selected membership. Normalizers
consume explicit variable maps without silently refreshing them. Nested compute
loops and outer atomic pipelines retain one canonical acceptance.

Fused neighbor aggregation now reuses the checked selected directional CSR reader
instead of three duplicate pool-based builders. Top-k and sampler liveness checks
honor immutable selected sources, including provisional IDs. Source maps with
removed/foreign IDs still reject where membership is required, including k=0.
Community direction belongs to the selected source. Existing numerical, incidence,
mask and label definitions are retained; stable CSR neighbor order may change
floating accumulation rounding relative to prior unordered runs.

Subgraph-producing sampling retains its separate owner checks and remains guarded
after pending topology. Guard tests now exercise valid iterate-node/edge stages,
rather than relying on formerly unsupported mask samplers.

## Verification

- Locked/offline full native suite: **1,589 passed / 54 ignored / zero failed**,
  223 suite results including doctests. Includes a mixed outer/nested computation
  proving provisional rows, selected endpoints, pure maps, masks and one acceptance.
- Strict all-target Clippy, Rust formatting and whitespace checks passed.
- Rebuilt in-tree release extension, without installing a wheel.
- **2,918 maintained Python tests / one skipped / 14 established warnings**;
  **276 executable documentation examples** passed. The 35 new Python cases cover
  pending node/edge filters, exact ordering, numerical maps, fused direction,
  reproducible masks, community parity against equivalent accepted Groggy graphs,
  removed input rows and late failure rollback. Three established binding build
  warnings remain unchanged. Black/isort/pre-commit remain unavailable.
- Initial test assumptions incorrectly treated edge_id as a synthetic property;
  corrected fixtures use an explicit ordinal property. No public ID API changed.

Native release benchmark: five observations each on a directed ring, one staged
node/two edges and one compute loop with ID load, sum normalization, outgoing fused
aggregation, top-k, seeded mask, four attachments and one publication. Medians:
**1,000 initial nodes: 7.892 ms; 10,000: 74.891 ms**. This includes pending
materialization and validation; fixture/initial capture, compilation, algorithm
construction and assertions are excluded. These are bounded observations, not a
before/after speedup, FFI, memory or scale guarantee. Raw receipt:
`notes/benchmarks/pending-core-compute-2026-11-15.json`. Reproduce with the
atomic_builder_properties bench's `--pending-core` mode and
GROGGY_ATOMIC_BUILDER_PROFILE selecting its output path.

## Existing checklist and open work

Advances **09.4b.2, 30.1 and AR7.8**; inventory stays **215 checked / 53 unchecked**,
including **44 unchecked release rows**. No new checkbox rows.

Subgraph-producing sampling/for-each needs bounded selected-frame refresh without
expanding each child to all provisional nodes or treating siblings' staged writes
as foreign targets. That requires retaining the original execution root separately
from child membership; enabling capabilities alone is insufficient. General
topology loops, alternate plans/streaming, legacy wrapper reads, remaining facade
lowering, journal-owner dispatch and naming/history policy retain existing exit
requirements. Unrelated owner review/benchmark files are untouched.

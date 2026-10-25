# Checked temporal builder bundle — 2026-10-24

Closes **09.4b.2aj / 09.4b.2ak / 09.4b.2al**, extending the existing
**09.4b.2 / 30.1 / AR7.8** parents under accepted workbook decisions.

## Finite selected decay maps (09.4b.2aj)

The existing `DecayStep` previously accepted NaN/infinite controls, multiplied
unchecked values, passed nonnumeric values through and had no builder registry
entry. It now supports property-atomic execution and `temporal.decay` registration.
Half-life is finite positive; explicit elapsed time is finite nonnegative, default
zero. Values must be finite numeric scalars, with integer sources producing f64.
Booleans/null/text/collections reject. The ratio-first formula handles subnormal
half-life with elapsed zero, and very large ratios decay to zero without NaN.

Node/edge outputs preserve sparse keys and entity kind. Validate each source ID
against active invocation membership; edge endpoints must be selected active
nodes. Build outputs privately and install only after final cancellation checks,
including same-name source/output use. Loading earlier attached graph properties
uses the existing staged overlay; decay itself does not infer clocks or mutate
history. Explicit attachments share the enclosing one-acceptance property batch.

## Ordered column parity (09.4b.2ak)

The same helper accepts native `NodeColumn` sources, retains row identity/order
and emits another column rather than converting through a node map. Duplicate
IDs reject to avoid ambiguous rows. Re-export the existing public NodeColumn type
beside StepVariables so native users and the release harness can construct it.
Column validation uses an O(rows) seen set; maps already have unique keys.
Empty inputs still validate controls. No owner/history clone or FFI logic added.

## Retained snapshot builder and strict temporal controls (09.4b.2al)

The existing `SnapshotAtStep` was implemented but described as a placeholder and
unavailable through the native step registry. Register `temporal.snapshot_at`
with output plus exactly one nonnegative integer state_id/timestamp. Certify its
retained-state read boundary and check cancellation after reconstruction before
installing the immutable result. The full-owner snapshot excludes staged writes
and unaccepted manual working changes, does not switch the working graph and
does not publish a checkpoint. Existing immutable state ownership is reused.

Timestamp selection preserves the existing global retained-history selector:
acceptance epoch seconds, greatest state ID on equal seconds, all retained
branches eligible. This is not arbitrary effective-time reconstruction. Explicit
state identity is preferable when branch identity matters. No Python conversion
of native snapshot variables was introduced; the graph's snapshot API remains
available for Python inspection. Reconstruction itself is not interruptible.

Existing temporal factories now reject supplied wrong-typed optional snapshot/
prefix/index/change selectors. Change types validate at preparation. Filter state
coordinates require actual integers: AlgorithmParams::expect_int truncates floats,
so temporal parsing deliberately matches Int directly before checked u64 conversion.
Negative IDs no longer wrap; reversed ranges reject before execution. Omitted
optional text values keep existing defaults.

## Follow-up remains explicit

Other temporal helpers are not blanket-certified. Concrete remaining audit work:

- Diff consumers lack explicit snapshot-source identity/branch compatibility
  validation and need cancellation/output preparation review.
- ensure_temporal_index reuses an untagged supplied index and installs a new one
  during execution. Provenance, invalidation and failure-side effects need review.
- Window aggregation contains legacy `index_` string/pool-reference resolution;
  legitimate text can be confused with a placeholder. Its snapshot fallback walks
  numeric state IDs, which need not be contiguous or represent only changes; it
  also filters nonnumeric history values before numeric reductions. Repair these
  semantics before certification, including state-ID versus timestamp terminology.
- Lifetime/window queries use a single creation/deletion interval, including
  inclusive deletion handling in the window helper. Review reactivation gaps,
  branch ancestry and exact end-boundary semantics.

These are work under existing open history/index/execution parents, not new
contract questions or a claim that the substantial engine is starting over.

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **985 passed,
  52 ignored**, including 11 temporal-step module tests (six new).
- Native regressions cover column order/in-place replacement, invalid controls
  and values preserving prior outputs, finite extreme limits, foreign/inactive/
  duplicate IDs, immutable retained state excluding working changes, acceptance
  timestamp selection, missing states and cancellation.
- `tests/test_builder_temporal_checked.py`: **39 passed**. Covers node/edge and
  persistent/transient parity, one publication, staged loads, prefix rollback,
  subnormal half-life, malformed/unknown snapshot coordinates and strict factories.
- Root Python gate: **1,634 passed, 1 skipped**, excluding the unrelated owner
  benchmark test. **75 executable documentation examples** passed, including three
  in `docs/api/builder-temporal.md`.
- `CARGO_INCREMENTAL=0 cargo clippy --offline --locked --all-targets -- -D warnings`:
  passed. In-tree FFI rebuild passed; existing unrelated FFI warnings remain.

## Bounded release receipt

`benches/temporal_builder_steps.rs` and
`../benchmarks/temporal-builder-bundle-2026-10-24.json` record five observations
per workload. Native optimized kernel-only timing includes validation, decay/
reconstruction and private variable installation. Fixture/variable preparation,
result validation and property publication are excluded. Validate every value,
column order, snapshot identity/existence, and unchanged owner state. Ring fixture
has equal node/edge counts. Median milliseconds:

| Operation | 1,000 nodes | 10,000 nodes |
| --- | ---: | ---: |
| Node-map decay | 0.0367 | 0.4610 |
| Edge-map decay | 0.0888 | 1.1458 |
| Ordered node-column decay | 0.0890 | 0.9244 |
| Retained owner snapshot | 0.7335 | 9.5102 |

Commands:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline --locked --bench temporal_builder_steps --no-run
GROGGY_TEMPORAL_PROFILE=notes/benchmarks/temporal-builder-bundle-2026-10-24.json python-groggy/target/release/deps/temporal_builder_steps-61f9b0668942e115
```

These are bounded observations, not a matched speedup, FFI latency, allocation/RSS
profile or production-scale claim. Decay is linear in source rows and allocates
private outputs. Retained snapshot cost depends on history/checkpoint layout and
state size. Full-engine performance and effective-time reconstruction remain open.

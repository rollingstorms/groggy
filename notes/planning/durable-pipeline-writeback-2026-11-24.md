# Durable computed writeback and topology — 2026-11-24

Follow-up to the completed first durable invocation slice. Existing scope remains
30.1, 09.4b.2, AR3, AR6 and AR7.8; no stable checklist rows are added. The
217 checked / 51 open umbrella inventory remains unchanged because it includes
unported kernel/composition routes and policy decisions.

## Finished component inventory

- [x] Share issued-table attachment across memory and detached invocation; retain
  exact capture authority, row identity, explicit mapping/clocks and canonical
  reserved-key/kind/null/duplicate validation.
- [x] Certify durable property/result writeback from all thirteen captured graph
  algorithm families, with exact accepted values, immutable source tables and
  journal recovery.
- [x] Share accepted-edge endpoint revision with explicit whole-batch staged-node
  references; preserve stable edge identity and existing property values.
- [x] Share accepted node/edge removal; preserve owner-wide incident-edge cascade
  semantics even when incident edges are outside the output selection. Explicit
  view-batch writes retain their separate stricter scope.
- [x] Preserve complete rollback for removed-row attachments, guessed provisional
  removals, duplicate revisions and other canonical failures; consume no accepted
  IDs or journal bytes and restore prior native outputs.
- [x] Preserve latest installed and optional stricter per-call schema checks at
  acceptance. Additional policy is retained with the journal operation for replay
  without installing a schema revision. Explicit policy also checks an empty run
  without creating history.

Contracts/examples: [durable writeback](../../docs/api/durable-pipeline-writeback.md)
and [invocation lifecycle](../../docs/api/durable-pipelines.md).

## Implementation

`PropertyBatch::attach_table_membership` consumes explicit selected membership
and shares the original issued-capture check and value conversion. Both adapters
use canonical pure bulk validation. Endpoint references share the same accepted
membership/Added-position resolver; adapters supply accepted liveness without
lending a mutable owner. Removal/revision kernels and table attachment are sealed
native invocations, not a separate executor or JSON interpreter.

All effects still prepare through one canonical GraphBatch. Durable acceptance
prepares the full accepted output and checks both policies before append/sync/
publication. No new storage format or intermediate versions are introduced.

Still guarded: provisional-entity removal/revision, unported arithmetic/neighbor/
history producers, loops/composition, alternate plans, streaming and custom
effects. A14 naming/cache/run history and provisional-history choices remain open.
This pass makes no policy decision and does not close overlapping parent rows.

## Qualification

Final local gates passed: **1,645 Rust tests / 54 ignored**, including ten
native doctests; separate serial killed-checkpoint recovery passed. Strict
all-target Clippy, root/changed-FFI formatting, diff checks and Python 3.8 syntax
parsing passed. The fresh CPython 3.9 wheel passed **3,409 maintained Python tests /
one established skip / 14 established warnings**, **292 executable documentation
examples**, all 105 public exports and isolated archive/journal workflows. All
62 packaged Python/stub sources match the checkout.

The durable integration inventory is now ten native tests and 24 Python cases,
plus the two journal failure-injection cases from the first slice. Native and
Python tests cover all thirteen captured algorithm families writing exact results;
directed/undirected rewiring/cascades and replay; node/edge property mappings;
removed-row and provisional-target rejection; and installed/additional schemas.

Commands (with `CARGO_INCREMENTAL=0`, `CARGO_TARGET_DIR=python-groggy/target`):

- `cargo test --offline --locked`
- `cargo clippy --offline --locked --all-targets -- -D warnings`
- `cargo test --offline --locked --lib killed_checkpoint_process_recovers_complete_state_and_releases_both_inode_locks -- --ignored --test-threads=1`
- `maturin build --release --out /tmp/groggy-durable-writeback-final-wheel`
- `/Users/michaelroth/Documents/Code/venv/bin/python notes/scripts/verify_release_wheel.py /tmp/groggy-durable-writeback-final-wheel --tests maintained --report notes/benchmarks/durable-writeback-wheel-qualification-2026-11-24.json`

[Local qualification](../benchmarks/durable-writeback-local-qualification-2026-11-24.json)
pins the exact 499-file production tree and artifact/cargo/log hashes. No pip
wheel installation or new dependency installation was used. Cross-platform
qualification subsequently passed; its exact gate levels and artifact evidence
are retained below. The first slice's receipts remain pinned to `aff5e090`;
its timing results are not silently assigned to this changed runtime.

## Commit and platform follow-up

Runtime commit: `b24e86e028962c7fc1dbc11a3ac82bb27ebdbf8d`, pushed on
`groggy-roadmap-2026-10-20`. The current capability matrix pins this code and its
fresh local receipt; production-source verification is separate from historical
first-slice timing evidence. The fixed 268-row inventory has not changed.

[Platform qualification run 45443651638](https://github.com/rollingstorms/groggy/actions/runs/45443651638)
is pinned to this runtime and completed successfully. All 21 required jobs
passed: nineteen wheel combinations, native gates and source rebuild. The manual
publish job was skipped. Linux CPython 3.12 ran the maintained suite; the other
eighteen wheels ran ownership plus isolated export/archive/journal workflows.
The source rebuild ran isolated workflows/exports only.

[Final platform receipt](../benchmarks/durable-writeback-platform-qualification-2026-11-24.json)
retains all twenty qualification reports and forty uploaded artifact identities/
ZIP digests. The 5,380,869-byte source archive matches all 499 production files at
`b24e86e0`, independently checked against Git archive and Git blob contents. Its
combined production hash matches the local receipt. Source-tar and inner-wheel
hashes are distinct from upload ZIP digests; actual regenerated archive lockfile
hashes are retained without claiming identical local dependencies. The archive
precedes later documentation/test-harness-only commits; production code remains
identical, as the capability validator confirms.

## Repeated workload baseline

The unchanged `benches/durable_pipeline.rs` program was repeated on this runtime
in three fresh processes, with 21 observations per size. It exercises the first
slice's creation/property/BFS acceptance program. The harness bytes are identical
at both runtime commits. [Raw observations and comparison](../benchmarks/durable-pipeline-runtime-comparison-2026-11-24.json)
retain both runtime/source identities, ranges and timing conditions.

Median wall times in milliseconds on the follow-up runtime:

| Accepted base nodes | Memory atomic run | Durable prepare | Durable accept | Durable reopen |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.042 | 0.029 | 4.689 | 0.098 |
| 1,000 | 3.184 | 3.146 | 6.889 | 5.702 |
| 10,000 | 32.917 | 33.398 | 31.574 | 54.553 |

The 1-node acceptance median rose from 4.119 to 4.689 ms; acceptance includes
fsync and its distributions overlap. The 1,000/10,000-node phase medians stayed
close or lower in this sample. OS cache, power/frequency and background load were
not controlled, so these descriptive observations establish no causal speedup or
regression guarantee. Complete accepted snapshot preparation and storage
synchronization remain material costs under existing 32.2/AR6 work. No new
latency/memory budget or stable checklist row is introduced.

## Next existing execution priority: alternate plan lowering

This refines the already-open 30.1/AR7.8 scope and adds no checklist item.
The current sources confirm why alternate plans remain guarded:

- `steps/loop_step.rs::LoopStep::apply` parses ordinary body steps before trying
  a batch plan. `try_batch_plan` returns no plan on deserialization failure;
  valid legacy wrappers can consequently fall back to body execution. This is
  not one common validation/lowering boundary.
- `execution/batch_executor.rs::BatchExecutor::execute` receives StepScope but no
  Context, so it does not share checked execution's cancellation/pending refresh
  boundary. Attribute loaders access Subgraph directly.
- Its division instruction substitutes zero for denominator magnitude below
  `1e-9`; numeric property loading/arithmetic uses f64 conversion. Those are
  concrete differences from checked step errors and typed integer handling.
  Python `builder/legacy_compat.py` can attach this plan to loop metadata.

The next finite implementation chunk should lower validated alternate plans to
checked native execution, retaining the existing public facade. Its exits are:
explicit malformed-plan rejection; parity for typed arithmetic/errors and carried
bindings; shared pending read/write/cancellation/output rollback; zero-iteration
and sparse/directed/parallel/loop fixtures; and one accepted version/reopen
parity for supported durable programs. Compare overlapping carries/aliases with
existing step execution rather than inventing A14 naming/history/cache policy.
Keep unsupported effects explicit. Do not enable the alternate atomic flag or
restore JIT merely because a plan parses. Bench the shared implementation after
semantic parity; owner latency/memory targets remain open.

### Test harness cleanup

A test-only follow-up moves pipeline setup before starting the GIL peer and
always releases both synchronization events when readiness fails. The focused
24-case durable Python suite passed afterward. Production sources remain exactly
those qualified at `b24e86e0`; this cleanup adds no engine feature or checklist row.

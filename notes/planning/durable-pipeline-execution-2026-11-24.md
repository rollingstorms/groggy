# Durable pipeline execution — 2026-11-24

**Follow-up:** issued computed-table writeback, accepted-entity removal/revision
and stricter per-call schemas are now implemented in
[the next completed component](durable-pipeline-writeback-2026-11-24.md). The
qualification and original limits below describe the first pinned slice.

Existing scope: 30.1, 09.4b.2, AR3, AR6 and AR7.8. This completes the first
supported durable program from `durable-pipeline-boundary-2026-11-16.md`.
No stable checklist rows are added or closed by narrowing their scope. Broad
parents still include their unported routes and owner policy questions.

## Finished component checks

- [x] Preflight every builder/step before executing a prefix; atomic effect flags
  alone do not grant portable invocation.
- [x] Capture sealed kernels and their metadata before worker execution. Workers
  invoke no custom Step/Algorithm methods and retain no Graph/writable Subgraph.
- [x] Reuse one StepExecutor and existing immutable-read, initialization,
  attachment, topology-creation and algorithm kernels. Canonical property
  validation is shared with ordinary Graph operations.
- [x] Stage sparse/global IDs, parallel instances, loops, exact values and explicit
  clocks; prepare without accepted allocations or publication.
- [x] Certify a creation/property/capture/computation/output program, including
  parity for all thirteen captured graph algorithms.
- [x] Validate owner, branch, head and allocation high-water at acceptance,
  including sibling allocations followed by a return to the original head.
- [x] Check the latest installed schema and complete the immutable accepted output
  before journal append/sync/publication.
- [x] Preserve prior outputs, graph, journal, IDs and state on late execution,
  cancellation or recoverable append failure; clear invocation staging for reuse.
- [x] Preserve poison/reopen behavior for indeterminate append and recover exact
  history, clocks, values and allocation after successful acceptance.
- [x] Expose the existing Python Pipeline builder via `run_durable`, with detached
  preparation under `allow_threads` and immutable accepted results.

## Design and boundaries

See [public contract](../../docs/api/durable-pipelines.md) for the exact registered
inventory and an executable example. The public native two-phase proposal holds
only read pins, identity/clock/allocation expectations and owned effects. It uses
no dummy owner, scratch Graph clone, mutable durable handle or second JSON
interpreter. Complete output projection reuses checked proposed-state snapshot
preparation; its whole-state snapshot cost remains explicit.

Memory/public Step adapters remain available. Legacy/custom implementations have
no sealed kernel by default and keep their current memory-only path. Durable
execution does not support every memory-certified primitive yet. Unported
arithmetic/neighbor/history producers, removal/revision, table attachment,
composition/loops, alternate plans and service scheduling stay guarded. No A14
naming/cache/history or provisional-history decision is invented.

## Qualification

Final local qualification: **1,642 native tests passed / 54 ignored**, ten native
doctests included; the separate serial process-kill checkpoint gate passed.
`cargo clippy --offline --locked --all-targets -- -D warnings`, root formatting,
changed-FFI formatting and `git diff --check` passed. The built CPython 3.9 wheel
passed **3,390 maintained Python tests / one established skip / 14 established
warnings**, **291 executable documentation examples**, all 105 public exports and
the isolated archive/journal application workflows. Five new Python cases and
nine new native cases (seven integration, two journal failure-injection tests)
cover the new boundary. Python 3.8 syntax parsing passed; platform execution
remains a separate gate. All 62 packaged Python/stub sources match the checkout.

Commands used with `CARGO_INCREMENTAL=0` and
`CARGO_TARGET_DIR=python-groggy/target`:

- `cargo test --offline --locked`
- `cargo test --offline --locked --lib killed_checkpoint_process_recovers_complete_state_and_releases_both_inode_locks -- --ignored --test-threads=1`
- `cargo clippy --offline --locked --all-targets -- -D warnings`
- `maturin build --release --out /tmp/groggy-durable-release-wheel`
- `/Users/michaelroth/Documents/Code/venv/bin/python notes/scripts/verify_release_wheel.py /tmp/groggy-durable-release-wheel --tests maintained --report notes/benchmarks/durable-pipeline-wheel-qualification-2026-11-24.json`

The build used `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=0` and the local CPython
3.9 interpreter. No wheel was installed with pip and no dependency was installed.
See the [local receipt](../benchmarks/durable-pipeline-local-qualification-2026-11-24.json)
for the qualified 499-file production tree, cargo inputs, artifact and log hashes.
The earlier platform wheel run qualifies its pinned older runtime; it is not
reused as evidence for this runtime change.

## Performance verification

`benches/durable_pipeline.rs` measures the same supported creation/property/BFS
program at 1, 1,000 and 10,000 accepted base nodes. Program/fixture/journal
initialization is outside timing. Memory execution, detached preparation, durable
acceptance (including proposed-state preparation and fsync), and reopening are
reported separately. The memory result is its ordinary Subgraph continuation;
the durable result is an accepted immutable pin. They are not interchangeable
operations or a historical speedup comparison.

An initial candidate revealed unnecessary pending captures before topology
staging and explicit captures. Both adapters now use the same pending-reader
classification; topology/capture kernels materialize their own inputs. Ordinary
map loads and edge initialization retain unsorted membership iteration; only the
operations requiring deterministic order request ordered nodes. Prototype
receipts are retained separately from the final idle-process observations.
Whole-state accepted snapshot preparation remains an explicit cost and is not
claimed to scale independently of graph size. No owner latency/memory targets
are invented.

### Final local baseline

Three fresh processes after local builds/tests stopped produced 21 observations
per size. Median wall times in milliseconds:

| Accepted base nodes | Memory atomic run | Durable prepare | Durable accept | Durable reopen |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.055 | 0.044 | 4.119 | 0.150 |
| 1,000 | 3.287 | 3.300 | 6.771 | 5.843 |
| 10,000 | 34.479 | 37.463 | 32.909 | 58.285 |

These are complete creation/property/BFS programs on this machine. Acceptance
includes storage synchronization; reopening includes journal replay. Setup and
program compilation are excluded. No production budget or comparison with a
previous library release is claimed. Exact observations and prototype limitations
are linked from the local receipt.

## Commit and platform follow-up

Runtime implementation is committed as `aff5e09007f7fd16f751be930630100935bdd5e1`
and pushed on `groggy-roadmap-2026-10-20`. The capability matrix at first closeout pinned this
runtime and its local qualification receipt; the current matrix now points to
the follow-up runtime. At first closeout, `--check --verify-runtime` passes
for all 499 production sources, 50 feature families, 105 public exports and the
fixed 268-row checklist. No feature-family status is upgraded by this update.

Cross-platform qualification completed successfully for this pinned runtime:
[run 45439515458](https://github.com/rollingstorms/groggy/actions/runs/45439515458).
It is a manual qualification run; publication remains disabled for this event.
All 21 required jobs passed: nineteen wheel combinations, native gates and
source rebuild. The publish job was skipped. Linux CPython 3.12 ran the maintained
suite; the other eighteen wheels ran ownership plus isolated export/archive/
journal workflows. Source rebuild ran isolated workflows/exports only. This
qualifies the first runtime, not the later writeback follow-up.

[Retained platform evidence](../benchmarks/durable-pipeline-platform-qualification-2026-11-24.json)
includes all twenty qualification reports and forty uploaded-artifact identities/
ZIP digests. The 5,371,539-byte source archive matches all 499 production sources
at `aff5e090`. Actual regenerated archive lockfile hashes are retained separately
from local dependency inputs. The combined source hash uses Path-component
ordering, matching the local qualification recipe; an initial aggregate used
string ordering. Individual source-byte verification was unchanged.
No identical dependency resolution or cross-platform binary identity is claimed.

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
qualification is separate and is not yet counted as a pass. The first slice's receipts remain pinned to `aff5e090`;
its timing results are not silently assigned to this changed runtime.

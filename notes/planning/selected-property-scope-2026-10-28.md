# Selected property reads — existing 26.1 / 21.2 component

Connected SubgraphOperations node/edge getters now check selected membership
before reading the current owner's property. A parallel edge is not selected
merely because its endpoints are selected. Algorithm Context/StepScope reads
also guard selected membership before returning staged properties and verify
owner liveness for cache hits. Cached values from an earlier wider scope do not
grant access from a narrower scope or revive deactivated entities.

The repair uses existing native membership sets, property getters and overlay
ownership. No new state owner, query engine, storage representation or FFI
business logic. Valid reads retain exact typed values and absence behavior;
invalid reads reject before exposing owner data. Existing Python connected getter
adapters retain RuntimeError translation. GraphView already enforces its frozen
selection and is unchanged.

## Established writeback contract preserved

An initial extension of this repair to native property-map attachment scope
failed the existing full-owner retained-diff regression on an empty invocation
selection. That test documents intentional behavior: temporal diffs cover owner
states, and explicit native map attachments can address their active owner IDs.
The scope restriction for those writes was removed; mutation/validation paths
are unchanged in this final patch. No regression test was weakened or removed.

Ordinary Python bulk accessors retain their existing selection-bound permissions,
and checked pinned apply_view_batch retains exact selected write scope. Unifying
legacy ownership remains the broader 26.1/AR3 task; it is not inferred from the
new read guards. This is a contract clarification supported by existing tests,
not a new owner policy decision. No unknown map entries are silently dropped.

## Verification

One native integration case covers manual/publishing read scope, excluded active
nodes/parallel edges, in-scope absence, later deactivation, unchanged read heads,
and preservation of existing explicit owner-wide property writeback. One native
unit case directly covers staged values in a wider scope and owner deactivation.
Four Python cases cover selected node/edge reads and existing whole-input bulk
accessor rejection without prefix/history changes. The complete temporal diff
suite retains its empty-invocation owner-wide attachment test.

Final gates: **1,433 native passed / 56 ignored**, **2,309 root Python passed /
1 skipped**, **224 executable documentation examples**, strict all-target core
Clippy, scoped Rust formatting, Python syntax and diff checks. The CPython 3.9
full-ABI release extension was rebuilt and atomically staged in tree; three
pre-existing FFI warnings remain. Root Python verification excludes the owner's
separate algorithm benchmark and nested historical surveys. No external packages
or graph libraries were installed/used. Owner benchmark/notebook/review changes
remain excluded from this commit.

Native commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`
with offline locked cargo test, targeted `--test selected_property_scope`, strict
all-target Clippy and `cargo bench --bench selected_property_scope`. Python uses
the rebuilt extension, root/focused contract suites and executable docs runner.
Guide: `docs/api/selected-property-scope.md`.

## Performance and remaining scope

The valid-input cost is selected membership checks, plus owner liveness before
returning a staged value. The repair creates no snapshots, mutation maintenance,
new publication or storage work. It does not change bulk validation passes.
`notes/benchmarks/selected-property-scope-2026-10-28.json` measures native selected
getters against direct owner getters over 1k/10k sequential scalar rows. Each of
five alternating samples checks a checksum for 100,000 reads; fixture creation is
excluded. The excluded active node also must fail selected access. These local
native measurements do not include Python/FFI overhead or certify the 100ns FFI
budget, arbitrary property payloads, cache behavior or production throughput.

Local sample medians (owned scalar value returned in each call):

| Selected nodes | Selected read | Owner read |
| --- | --- | --- |
| 1,000 | 105.222 ns | 101.715 ns |
| 10,000 | 121.375 ns | 117.542 ns |

The full connected Subgraph remains live. Immutable default ownership/topology
migration stays in 26.1/AR4/AR7.5, legacy query migration in 21.2 and compound
topology/custom effects in 09.4b.2. No new rows or parent closure:
**211 checked / 57 unchecked**.

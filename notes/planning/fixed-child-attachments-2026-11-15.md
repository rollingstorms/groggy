# Fixed child attachment boundaries — 2026-11-15

## Confirmed problem and fix

For-each uses exact live execution frames before pending topology, allowing cheap
shared property-overlay reads. Retained-history diff maps intentionally cover the
full owner. Feeding such a map to a live child attachment previously reached the
owner's active-ID validator without checking the child's membership. A probe with
three singleton children and a historical change on node 2 wrote node 2 while
executing child 0. Frozen pending frames already constrained such writes.

StepScope now validates all supplied IDs before forwarding node columns, node
attribute maps and edge attribute maps to Context. The fixed-frame flag follows
nested loops. Validation precedes staging, publication and transient output
installation; a foreign row rejects the whole input. Parallel edges remain exact
instances. No IDs are filtered or rebound and full-owner diff results remain
readable. Ordinary root explicit owner-ID behavior is retained; public native
Subgraph compatibility setter semantics are unchanged.

This is an execution-scope correction, with no new API, persistence format,
mutation mechanism or Python/FFI business logic.

## Qualification

Locked/offline native suite: **1,600 passed / 54 ignored / zero failed**, 223 suite
results including doctests. Strict all-target Clippy, formatting and whitespace
checks pass. The in-tree release extension was rebuilt without installing a wheel;
three existing binding warnings remain. Focused Python: **55 passed**.

One native test exercises all three attachment boundaries in transient,
non-atomic persistent and atomic persistent modes. Mixed valid/foreign rows leave
no output, staged prefix or graph publication. Sixteen Python regressions cover
node/edge history maps, exact parallel edges, direct/nested execution, transient/
persistent and atomic/non-atomic calls. Atomic prefixes roll back on late child
failure. Rejected runs consume no IDs. The existing full-owner empty-root diff
attachment regression still passes, as do pending child history/view tests.

Full maintained Python: **2,996 passed / one skipped / 14 established warnings**;
all **280 executable documentation examples** pass.
Black/isort/pre-commit remain unavailable; no packages were installed.

## Existing roadmap and remaining work

Advances the execution/source boundaries in 09.4b.2, 30.1, AR7.8 and Python
conformance evidence in 19.2. The fixed **268 rows remain 215 checked / 53 unchecked**,
with **44 unchecked rows in the expanded release scope**. No rows added and broad
unfinished parents remain open. Next: certify reads of accepted snapshots/diffs
independently of unpublished working topology. Selected lifecycle/window readers
still need an explicit provisional-entity contract and remain guarded. General
child topology mutation and alternate-plan execution remain separate requirements.
Unrelated owner files are untouched.

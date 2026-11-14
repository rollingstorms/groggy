# Python owner lifetime repair — 2026-11-14

Existing scope: 19.2 and the bounded 0.6.0 ownership release gate. No new flat
checklist rows. This fixes native finalization, not the entire Python contract
or a multi-client/concurrent-writer service.

## Implementation

The former Rc/RefCell graph, pool and connected-handle ownership could not safely
be destroyed on a foreign thread. PyO3's unsendable guard therefore skipped the
native destructor during foreign-thread cyclic GC, reporting an unraisable error.

Shared native handles now use Arc/OwnerCell. OwnerCell is a small safe Rust wrapper
over nonblocking parking_lot read/write guards plus a creating ThreadId. Borrowing
checks the creator; immutable borrows can nest and conflicting mutable borrows
fail immediately rather than waiting/deadlocking. Shared graph/pool/cache owners
retain the same identity and columnar storage. There is no per-entity lock or
per-call graph copy. Atomic counts make final release independent of access-thread
policy, and the compiler verifies Send/Sync for the actual contained types.

The thread-ID check caches only a copied ThreadId in TLS, avoiding repeated
reference-counted Thread handle clones. It stores no graph data and creates no
deferred cleanup queue or shutdown dependency. State borrowing still rejects
after the creating thread exits; final release still frees the native value.

The migration required explicit Send + Sync temporal-strategy bounds and replacing
the unimplemented MatrixView raw backing pointer with retained Arc<[T]> metadata.
Matrix View operations remain unsupported. The unused raw graph pointer in the
native Python attribute collection is replaced by a retained checked graph handle,
removing two unsafe dereferences rather than asserting their lifetime.

Python wrappers now satisfy PyO3's normal compiler-checked Send requirements.
No unsafe Send implementations, warning filters, global GC changes, hidden thread
registry, owner polling or deferred free are introduced. Explicit durable access,
including close/status/context exit, checks its creating thread separately from
the destructor. Detached GraphView remains the supported worker read interface;
this is not certification of every legacy adapter operation on workers.

Python method signatures/publication/history semantics remain stable. Native
explicit Rc/RefCell handle and Ref/RefMut guard declarations require migration;
use Graph::into_shared for canonical construction. The native compatibility
changes and current PyO3 panic-style foreign-access rejection are documented in
`docs/api/python-owner-lifetimes.md`.

## Lifetime evidence

- Five native tests prove graph/pool/subgraph/matrix shared ownership traits,
  immediate foreign-thread destructor execution with a drop counter, cleanup after
  creator exit, foreign-borrow rejection without poisoning, and nested/conflicting
  borrow behavior.
- Thirty-four Python tests cover 14 graph/owned-value families collected from
  another thread while their creator is alive and after it exits, plus six live
  access/mutation/status/close rejection cases followed by successful creator use.
- Durable cycles reopen the journal immediately after foreign GC, recovering the
  saved property. This requires real native file-lock release, without explicit
  close, owner polling or waiting for the creator to exit.
- The original three-case diagnostic now returns no unraisable errors in any
  case. Tests capture and assert absence of errors; production code changes no
  unraisable hooks or GC settings.

## Verification and performance

Final gates pass with the optimized thread-ID check: **1,566 Rust passed / 54
ignored**, strict all-target Clippy, **2,718 maintained Python passed / 1 skipped /
14 existing builder warnings**, and **267 executable docs examples**, including
the owner-lifetime example. The former finalization warning is absent. The rebuilt
0.6.0 wheel passes the isolated no-site-packages checker and both archive/journal
acceptance workflows. Scoped formatting/import cleanup and diff checks pass;
unrelated pre-existing file-body formatting is preserved. Black/isort are absent
and no dependencies were installed. Three existing FFI compiler warnings remain.

Raw first-pass access measurements remain in
`notes/benchmarks/python-owner-access-initial-2026-11-14.json`. They exposed about
14% overhead in the 1k-property Python batch fixture before thread-ID caching.
Keep those samples rather than replacing them with the optimized result. Native
workflow scale and final public-access measurements retain their separate scopes.

Final access comparison uses three alternating fresh-process pairs against the
previous qualified 0.6.0 wheel, seven samples per operation, with site-packages
disabled and builds/tests stopped. Values are medians of process medians:

| Public operation | Before | After | Observed difference |
| --- | --- | --- | --- |
| Live property read | 308.36 ns | 304.45 ns | -1.3% |
| Captured property read | 267.26 ns | 265.47 ns | -0.7% |
| Warm capture | 215.81 ns | 212.02 ns | -1.8% |
| Checked 1,000-property Python batch | 1.988 ms | 2.039 ms | +2.6% |

Read differences are small timing variation, not claimed speedups. The bulk
fixture exposes a small remaining safety cost. All these times include Python
loop/dispatch, FFI and native work; none measures isolated FFI overhead or certifies
the 100ns FFI budget. Bulk includes Python staging and checked publication, with
the same sequence of retained versions. Initial and optimized comparisons are
separate batches; do not assign every difference solely to the cache change.
All samples and reproduction helper:
`notes/benchmarks/python-owner-access-2026-11-14.json` and
`notes/scripts/benchmark_python_owner_access.py`.

The existing native directed-ring workflow runs in three fresh processes per size
after all builds/tests. At 100k nodes and edges, median bulk acceptance is
**581.888 ms**, cold capture **142.468 ms**, effective read **251.574 ms** and
archive save/recovery **1,452.454 / 859.014 ms**. Whole-process peak RSS ranges
**855.33–882.14 MiB**. Earlier qualification measured 599.632 / 145.124 / 248.510 /
1,418.321 / 840.928 ms respectively, with 836–869 MiB peak RSS. These remain similar
operating costs; no universal speed or memory improvement is claimed. Native
per-operation time, total wall, process CPU and RSS are distinct fields; cold
process/filesystem/allocator effects remain visible. Ring shape, one scalar column
and two versions do not certify deep history, large payloads, hubs or services.
Raw final workflow samples: `notes/benchmarks/python-owner-workflow-2026-11-14.json`.

Reproduce core gates using offline locked Cargo and the shared in-tree target;
build/stage the release extension, run
`notes/scripts/verify_immutable_view_migration.py` and the original diagnostic.
Build `core_workflow_scale` through cargo bench --no-run and run the emitted
executable with 1000/100000 under `/usr/bin/time -l`. Access comparison runs
`python -I -S notes/scripts/benchmark_python_owner_access.py PACKAGE_ROOT` against
an unpacked old wheel and the current in-tree Python root. No wheel is installed.

## Remaining scope

19.2 stays partial: general API/error consistency, GIL and compatibility inventories
remain broader than this defect. The full checklist remains 215 checked / 53 open.
0.6.0 still needs cross-platform distribution qualification and final release
review; no publication or push is performed here. Archive/reconstruction memory,
broader format migration, service concurrency and deferred retrieval retain their
existing roadmap scope.

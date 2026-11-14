# 0.6.0 embedded-core qualification — 2026-11-14

Follow-up: the ownership defect described in this initial qualification is now
fixed; see `python-owner-lifetime-2026-11-14.md`. The original 2,684/15-warning
baseline and pending ownership assessment below are retained as historical
evidence. Current release status incorporates the follow-up verification.

Authorized scope: the proposed 6–10 hour release-preparation allocation. This
receipt records actual results, not hours billed, time spent or a release promise.
Graphiti remains future external plugin work. No tag/upload/publication performed.

## Completed components

1. **Packaging/export consistency** (`1313b4d2`). Cargo crates advance to 0.6.0;
   both pyprojects derive the distribution version from Cargo. Runtime version
   comes from the loaded extension, avoiding stale installed distribution metadata.
   Nested metadata no longer declares description both static and dynamic, and
   both entry points declare the existing typing-extensions dependency.
2. **Standard build backend.** Remove the custom backend that compiled/imported
   the extension and regenerated dynamic stubs before building the wheel. Use
   Maturin directly; stub maintenance is explicit and cross builds need not import
   a target binary. Ship the checked-in stub and typing marker.
3. **Concrete API repair.** Isolated wheel checks exposed two previously missing
   `__all__` bindings. Public `format_data_structure` and `detect_display_type`
   now delegate through thin FFI to the existing canonical native display code.
   All 104 declared exports resolve; four display cases verify native delegation.
4. **One complete acceptance workflow** (`f2f95456`). A shared executable fixture
   and two parameterized tests cover batch creation/external IDs, exact parallel
   edge selection, isolate/loop retention, sorted/projected table source writeback,
   immutable pins, rejected mixed-batch atomicity, effective reads, f64 matrix
   determinant, checkpoint-without-delta, tracked restoration, retained archives,
   durable close/reopen, integrity/storage checks and view lifetime after close.

These complete bounded components within 19.2/31/32/34; they do not complete their
broad parents. Flat checklist remains **215 checked / 53 open / 268 total**.

## Executed evidence

- Root wheel: `maturin build --release --offline --locked` succeeds.
- Source distribution: `maturin sdist` succeeds, then a fresh extracted source
  directory builds offline with locked resolution, independent of the checkout.
- Nested `python-groggy/` wheel build also succeeds.
- Each of the three wheels passes `notes/scripts/verify_release_wheel.py` in an
  isolated subprocess (`-I -S`, site-packages disabled) with only the unpacked
  wheel added to sys.path. Metadata,
  package and native versions agree; package/native paths are inside the temporary
  extraction directory; typing assets, exports, graph/numerical smoke and both
  archive/journal acceptance workflows pass. No optional development dependency
  or installed Groggy supplies a fallback; no wheel was installed.
- Focused Python: **8 passed**.
- Maintained root Python: **2,684 passed / 1 skipped / 15 warnings**; **266**
  executable documentation examples pass. The gate excludes historical nested
  surveys and the owner's separate algorithm benchmark. The standalone core
  example also passes both archive/journal runs.
- `notes/scripts/diagnose_graph_thread_lifetime.py` confirms owner-thread Graph
  GC and worker detached-view GC pass, while worker Graph GC still reports the
  known unraisable error.
- Scoped native formatting and `git diff --check` pass. Black/isort are absent;
  none installed. Existing three FFI compiler warnings remain unchanged.
- Full offline locked `cargo test`: **1,561 passed / 54 ignored / zero failed**
  across 219 suite results, including doctests.
- Offline locked `cargo clippy --all-targets -- -D warnings`: **passed**.

Local platform: macOS arm64, CPython 3.9. This is no certification of every
advertised Python version, wheel tag or OS. Build artifacts/logs are under
`/tmp/groggy-060-*` for this session, not committed binaries.

## Bounded performance baseline

Reuse `benches/core_workflow_scale.rs`; no new benchmark family or performance
gate. Three fresh release processes at each size, after builds/tests finish.
Each runs bulk ring acceptance, capture, indexed/scanned pages, neighborhoods,
one correction, effective reconstruction and complete retained archive recovery.
Native per-operation timers, process wall/CPU and whole-process peak RSS are
retained with every raw sample in
`notes/benchmarks/core-release-060-2026-11-14.json`.

| Nodes and edges each | Bulk acceptance median | Cold capture median | Effective read median | Archive save / recovery medians | Peak RSS range |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 5.048 ms | 0.743 ms | 2.033 ms | 21.060 / 6.581 ms | 16.59–17.14 MiB |
| 100,000 | 599.632 ms | 145.124 ms | 248.510 ms | 1,418.321 / 840.928 ms | 836.00–869.45 MiB |

100k whole-process wall median is 3.761 seconds; its archive is 26,889,511 bytes.
Peak RSS includes every stage, allocator retention and transient copies, not just
retained graph memory. The first small process's startup cost is preserved in raw
results. Fixtures use a directed ring, one repeating scalar property and two
versions; this is not a hub-heavy/deep-history/vector/FFI/journal benchmark.
Compared with the earlier receipts these are the same general operating scale,
but this is not a matched before/after experiment and no regression percentage,
universal latency or memory improvement is claimed. Whole-process memory and
archive costs remain existing 10.4/32.2 follow-ups, not new checklist requirements.

Reproduce: offline locked `cargo bench --bench core_workflow_scale --no-run` with
the shared target directory, then run the emitted release executable under
`/usr/bin/time -l` with 1000 and 100000, three fresh processes each. Build time is
excluded; native operation times, total wall and process CPU are distinct fields.

## Ownership assessment remains open

The existing reproducible defect is a release concern, not fixed by this slice.
PyO3 0.20.3's `ThreadCheckerImpl::can_drop` rejects foreign-thread destruction
before the native destructor, so a Python-level cyclic collection can leak the
native owner even without a cross-thread Graph method call. Core Graph has
Rc/RefCell pool ownership and weak self references; connected wrappers share that
model. Replacing only the top-level FFI wrapper with an Arc/lock does not make
the contained graph and its dependents Send or settle their destruction semantics.

Do not remove unsendable protection, add unsafe Send, disable GC globally or
silence the regression. A safe ownership fix needs a separate bounded design
implementation, or an explicit narrower release decision. Detached immutable
GraphView reads remain the supported worker boundary. The full-suite allocation
path that later triggers GC is still not identified; the A* reporting location
is not proof of an A* defect.

## Remaining release work

- Resolve the ownership defect or explicitly decide the release boundary.
- Run available CI/platform package/import checks before publishing; local
  extracted-wheel imports are not a substitute for cross-platform installations.
- Review the proposed supported scope and migration notes. Keep general topology
  additions, service concurrency, retrieval families and universal old-format
  migration in the future roadmap.
- Use a real application fixture when the owner's project requirements arrive;
  the synthetic acceptance example is useful evidence, not that project itself.

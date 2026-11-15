# Groggy 0.6.0 release qualification

Updated 2026-11-15. **Expanded 0.6.0 scope in progress; earlier candidate qualified. Unreleased.**

## Expanded release target — 2026-11-15

The owner now targets all remaining hierarchical groups except group 8:
**44 unchecked rows in scope; nine retrieval/integration rows deferred**.
This includes provenance and multi-user/concurrency work. The expanded target
remains unfinished. See [scope and next chunk](notes/planning/release-060-expanded-scope-2026-11-15.md).

The successful qualification below applies to the earlier bounded candidate.
Implementation changes toward the expanded target require fresh final gates.

The next implementation chunk, supported outer atomic topology composition, is
complete locally: 1,579 native tests, 2,775 maintained Python tests and 271
executable examples pass, plus strict Clippy and Rust formatting.
See [chunk evidence](notes/planning/outer-atomic-topology-2026-11-15.md).
The 44 broad unchecked release rows retain their remaining exit requirements.

## Qualified bounded candidate

| Component | Status | Evidence |
| --- | --- | --- |
| One complete application workflow | Complete locally | Checked ingestion → immutable graph/table analysis → temporal correction/restore → archive/journal recovery, from unpacked wheels. |
| Performance baseline | Complete for the documented fixtures | Current ownership implementation: three fresh native processes per size, plus matched public-Python access measurements. |
| Scope, migration and release docs | Complete for this candidate | [Release notes](RELEASE_NOTES_0.6.0.md), [application workflow](docs/api/core-release-workflow.md), [qualification runbook](docs/api/release-qualification.md). |
| Package qualification | Complete locally | Fresh root, nested and extracted-source wheels; versions, 105 declared exports, typing assets and both storage workflows pass. |
| Native and artifact-backed regression gates | Passed locally | 1,573 Rust tests; serial checkpoint recovery; strict Clippy; 2,751 Python tests; 270 executable docs examples from the fresh unpacked root wheel. |
| Cross-platform qualification | Passed | All 19 wheel combinations, native gates and extracted-source rebuild pass on native implementation `9aadfee5`, using Rust 1.96.1. Receipts include every wheel hash. |

Committed work has been pushed and the platform workflow dispatched. No tag or
package publication occurred. Manual qualification cannot publish; a published
tag release must pass native, wheel and extracted-source gates before reaching
the existing PyPI environment boundary. See the [coordinated pass receipt](notes/planning/priority-pass-2026-11-14.md)
for precise source, platform and fresh artifact results.

## Supported scope and migration

An embedded Rust/Python graph and analytics library. This candidate includes
checked automatic publication, immutable captures, effective-time reconstruction,
explicit checkpoints/restoration, table writeback, numerical operations and
retained archive/journal recovery within documented boundaries. See the
[scope and migration guide](RELEASE_NOTES_0.6.0.md).

Both package entry points declare CPython `>=3.8,<3.13`, matching the current
PyO3 0.20.3 build limit. CI targets Linux/Windows x86_64 and macOS Intel 3.8–3.12;
macOS arm64 3.9–3.12. Local execution uses macOS arm64 CPython 3.9; all 19 listed combinations have now passed CI.
Newer interpreters and PyPy have no qualification claim in this release receipt.

Creator-checked Arc/OwnerCell handles fix foreign-thread cyclic-GC finalization
and immediate durable file-lock release while preserving creator-bound live
access. Detached immutable GraphView remains the supported worker read interface.
See [Python owner lifetimes](docs/api/python-owner-lifetimes.md) for explicit
native Rust handle migration and the 34 Python/5 native ownership regressions.

Core retrieval expansion,
multi-client services, arbitrary old-format migration, universal power-loss
claims and broad numerical/scale conformance remain outside this bounded release.
They retain their existing roadmap boundaries.

## Executed gates and measured costs

Fresh root, nested and extracted-source wheels pass isolated smoke and the
application workflow with site-packages disabled. The maintained Python/docs gate
also runs against the root unpacked artifact, not an installed/source extension.
No wheel was installed during qualification.

- Maintained Python: **2,751 passed / 1 skipped / 14 established builder warnings**.
- Executable documentation: **270 examples passed**.
- Nested artifact workflow/ownership gate: **36 passed**.
- Rust: **1,573 passed / 54 ignored / zero failed**; the explicit serial checkpoint
  process-recovery gate additionally passes its one test.
- Strict all-target Clippy, Rust formatting and `git diff --check`: **passed**.

At 100k nodes and edges, the existing native directed-ring baseline has median
batch acceptance **0.582 s**, cold capture **0.142 s**, effective read **0.252 s**,
and archive save/recovery **1.452/0.859 s**. Whole-process peak RSS is **855–882
MiB**. These are bounded fixture measurements, not universal performance budgets.
The matched Python access fixture shows small read variation and about **2.6%**
overhead for a checked 1,000-property batch after the ownership fix. Archive/memory
scale work remains open.

The coordinated follow-up adds staged node/edge creation and 13 captured
algorithm-table kernels, explicit compatible schema migration, storage mechanism
reporting and independent finite graph references. Archive validation avoids
400,009 allocations per 100k save/recovery without reducing the measured requested
peaks; larger dense products improve about 8–10% in the recorded corpus. See
[graph memory](notes/planning/priority-memory-2026-11-14.md) and
[numerical measurements](notes/planning/priority-numerical-2026-11-14.md).

Current source, artifact hashes and all platform receipts:

- [Coordinated pass](notes/planning/priority-pass-2026-11-14.md) and
  [machine-readable results](notes/planning/priority-pass-final-gates-2026-11-14.json).
- [Successful platform run](https://github.com/rollingstorms/groggy/actions/runs/43668885370).
  All 21 required jobs passed; publication was skipped for manual qualification.
  Later changes and an actual release tag must run the configured gates again.

Earlier baseline evidence, artifact/lock hashes and reproducible commands:

- [Final local gate receipt](notes/planning/release-060-final-gates-2026-11-14.md)
  and [machine-readable hashes/results](notes/planning/release-060-final-gates-2026-11-14.json).
- [Qualification runbook](docs/api/release-qualification.md).
- [Ownership implementation and performance comparison](notes/planning/python-owner-lifetime-2026-11-14.md).
- [Initial qualification receipt](notes/planning/release-060-qualification-2026-11-14.md), retained historically.

The broader flat checklist remains **215 checked / 53 open / 268 total**; its
parent rows overlap and include deferred scope. This release preparation does not
mark them all complete. Historical [0.5.2 status](notes/planning/release-052-status-historical.md)
is archived separately.

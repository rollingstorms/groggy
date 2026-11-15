# Coordinated core priority pass — 2026-11-14

One coordinated pass through the six proposed priorities. This updates existing
requirements; no additional checklist rows or broad parent checkmarks are added.
The flat inventory remains **215 checked / 53 unchecked / 268 total**. Overlapping
parents and deferred work make this a scope inventory, not a release percentage.

## Completed components

| Priority | Implementation and evidence | Remaining boundary |
| --- | --- | --- |
| Release qualification | Executed the real platform workflow; repaired Python 3.8 verifier paths, Windows slice-width conversion, file/directory sync capability reporting and Windows locked-handle journal identity. Pinned the qualified compiler to Rust 1.96.1. | All 19 wheel jobs plus native and extracted-source gates passed; no release publication. Newer compiler warning cleanliness is not certified. |
| Memory/recovery | Shared scale/allocation fixtures, three fresh processes per size, and validation without disposable read indexes. Each 100k archive save/recovery avoids 400,009 allocations. | Peak requested archive memory is unchanged; larger/deeper histories and recovery budgets remain open. |
| Staged execution | Local atomic node/edge additions, stable added-node references and 13 existing algorithm kernels over immutable pending captures. Late failure rolls back all stages without consuming IDs; acceptance remains one checked batch. | Outer topology programs, unadapted effects, remaining DSL, streaming and alternate plans remain open. |
| Numerical costs | Reused the 55-workload release corpus and measured allocation/throughput before and after contiguous dense-product access. Larger fixtures improve about 8–10%, with unchanged requested peaks and checked arithmetic. | Scalar performance, broad budgets and further acceleration remain open; no external-backend parity claim. |
| Consistency | Empty native bulk columns no longer create ghost columns or underflow. Captured graph kernels agree with independent finite references across 256 directed/undirected fixtures, including sparse IDs, loops, parallel edges, zero weights and inactive owner changes. | This is finite reference evidence, not exhaustive algorithm or cross-backend conformance. Existing creator-bound live ownership remains unchanged. |
| Evolution/recovery | Explicit compatible schema-policy/data/policy workflow verifies atomic failures, retained values, incompatible branch rejection, backup and reopening. Existing corruption/fault/process-recovery gates are retained. | A07.1 global versus branch-specific schema policy remains an unanswered owner choice. No automatic ontology migration or arbitrary format conversion is invented. |

## Contracts retained

- `graph.add_nodes`, `graph.add_edges` and `graph.algorithm_table` use the existing
  native interpreter and canonical mutation/read kernels. Algorithms do not write
  properties implicitly. Provisional tables cannot pretend to be accepted-state
  writeback sources. See [staged topology](../../docs/api/staged-topology-additions.md).
- Storage capabilities describe mechanisms. Windows supports file sync and
  locked-handle verification; directory syncing and physical checked replacement
  remain explicitly Unix-only. No universal power-loss guarantee is claimed.
- Compatible schema stages are separately recorded operations. A caller can
  inspect the installed revision and resume explicitly after interruption.
  Historical data is not rewritten. See [migration](../../docs/api/schema-migration.md).
- Unanswered policy questions are not treated as approval. Current global durable
  schema enforcement remains unchanged pending A07.1.

## Measurements

- [Graph allocation receipt](priority-memory-2026-11-14.md): requested bytes versus
  process RSS, phase retention, before/after allocation counts and exact limits.
- [Numerical receipt](priority-numerical-2026-11-14.md): same-order checked sums,
  row/column layouts, overflow/nonfinite regression tests and raw corpus paths.

## Verification

Native suite: **1,573 passed / 54 ignored / zero failed**, including doctests.
The explicitly selected serial checkpoint process-recovery test also passes.
Strict all-target Clippy passes with Rust 1.96.1. The fresh root wheel passes **2,751 Python tests / 1 skipped / 14 established
builder warnings**, plus **270 executable documentation examples**. All 105
declared exports and both storage workflows resolve from its unpacked artifact.
The nested wheel passes its **36 workflow/ownership tests**. No wheel is installed.
Fresh final artifact receipts identify exported source `9aadfee5`.

The current platform qualification run for native implementation `9aadfee5` is
[43668885370](https://github.com/rollingstorms/groggy/actions/runs/43668885370).
The preceding run passed Linux wheels/source/native gates but exposed a Windows
shared-cursor replay bug. Verification now explicitly rewinds its reader; a
portable native regression reproduces duplicated-handle EOF, subsequent writes
and reopening.
Earlier attempts exposed real failures and were repaired; superseded runs were
cancelled. A workflow configuration or queued job is not a passed qualification.
Manual dispatch cannot publish. No tag or package release was created.

`cargo fmt --all -- --check` passes after formatting the existing Rust sources
(`6886a4f6`, `b4634542`); these follow-ups change no behavior. Black/isort/
pre-commit were unavailable locally and were not installed.

## Commits

`dcdede96`, `c3b3337d`, `99f67731`, `130d60bd`, `93a23ec7`, `9846941e`,
`3242c22d`, `206b2bd7`, `9aadfee5` contain the implementation, benchmarks and regressions.
Qualification receipts identify the exact exported source and artifact hashes.

## Highest remaining risks

1. Large transient archive representations and retained history memory.
2. Unadapted topology/compound execution and remaining native facade parity.
3. Scalar numerical throughput and unchosen workload budgets.
4. Schema/ontology evolution policy and future storage-version migration.
5. Platform-specific filesystem limits; later release tags must repeat qualification.

These remain existing checklist work. The bounded release candidate does not
require completing every deferred engine feature, but it does require accurate
scope and reviewed platform results before publication.

## Fresh local artifact receipt

Root and nested wheels plus an extracted-source rebuild from `9aadfee5` pass
isolated imports, 105 declared exports and both storage workflows. The root wheel
passes the full maintained gate (2,751 Python tests and 270 examples); the nested
wheel passes 36 workflow/ownership tests. The final in-tree extension is rebuilt.

[Machine-readable hashes and gates](priority-pass-final-gates-2026-11-14.json)
include both resolved Cargo lockfiles, the source archive, each wheel, test totals
and the platform-run snapshot. Artifact qualification precedes the behavior-free
Rust formatting follow-ups; subsequent documentation edits do not change the
native implementation. Earlier preview artifacts are superseded by this receipt.

## Final platform outcome

[Run 43668885370](https://github.com/rollingstorms/groggy/actions/runs/43668885370)
completed successfully on native implementation `9aadfee5`: **21 required jobs
passed**, covering **19 wheel combinations**, the native test/serial-recovery/
Clippy gate and extracted-source rebuild. Publication was skipped by the manual
qualification condition. All wheel receipts report version 0.6.0, 105 exports
and successful archive/journal workflows. Linux CPython 3.12 also runs the
maintained Python/docs suite; the other wheel jobs run workflow/ownership tests.

The matrix covers Linux and Windows x86_64 plus macOS Intel on CPython 3.8–3.12,
and macOS arm64 on 3.9–3.12. Full OS/interpreter details and hashes for all 19
wheels plus the source rebuild are retained in the machine-readable receipt.
These are successful tests on the reported runners, not certification of every
filesystem, older OS, compiler or machine.

The six selected components have completed this coordinated pass. Their broader
remaining boundaries stay in the existing checklist. Documentation and formatting
follow-ups do not change native behavior; a future published release tag still
has to pass its configured gates. No tag or package release was created.

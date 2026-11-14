# 0.6.0 application, baseline and final gates — 2026-11-14

Scope: finish the owner's three remaining bounded release-preparation items.
This is an embedded library release candidate. Graphiti remains a future external
plugin; the wider roadmap remains **215 checked / 53 open / 268 total**. No
publication, tag, upload, push or CI dispatch is part of this slice.

## Application acceptance

Reuse the complete application fixture from `f2f95456`, rather than inventing a
second application. It connects checked ingestion and external identity, exact
parallel-edge selection, isolate/loop retention, projected table writeback,
immutable captures, effective-time reconstruction, failed-batch atomicity,
f64 matrix analysis, explicit checkpoints, tracked restoration and recovery.
Both archive and journal paths verify reopened identity/history/properties and
retained views after owner destruction/close. The same fixture runs against
unpacked release wheels with site-packages disabled.

`verify_release_wheel.py` now also accepts an artifact directory, emits successful
SHA256 receipts, checks the declared interpreter boundary and optional release
tag, and can run ownership or maintained Python/docs gates against the unpacked
artifact. The optional test phase enables test dependencies; the separate import
and application smoke remains `-I -S`. No wheel installation is involved.

`verify_release_sdist.py` extracts an actual source distribution into a temporary
directory, rejects links/escaping members, requires both resolved Cargo lockfiles,
builds its wheel with locked resolution, and invokes the isolated wheel gate.

## Build boundary and provenance

Native code is `bada3345` (the ownership repair). Packaging and qualification
changes in this slice introduce no further graph/numerical implementation change.
Fresh root/nested/source-rebuilt artifacts use an export of that committed tree,
overlaid only with this slice's packaging/qualification files and the existing
resolved lockfiles. The owner's unrelated uncommitted review/benchmark work is
excluded. Artifact hashes and lockfile hashes are recorded in the adjacent JSON
receipt; generated binaries remain outside the repository.

Both package entry points declare `>=3.8,<3.13`. The local pinned
`pyo3-ffi-0.20.3/build.rs` explicitly sets CPython's maximum to 3.12. Remove the
nested build's forced forward-compatibility bypass rather than advertising
untested newer interpreters. No PyO3 upgrade or broader compatibility claim.

Cargo locks remain ignored for ordinary library development. Qualification keeps
one existing resolved pair; source rebuilds use the pair shipped in the source
distribution. Fresh CI resolutions can differ, so this is not a promise of
bit-for-bit reproducibility across independently resolved builds.

## Executed local gates

- Fresh root and nested wheels build offline with locked Cargo resolution. An
  independently extracted source distribution also builds with its shipped
  locks. All three pass isolated import/version/exports/typing, numerical smoke
  and both application storage workflows, including the ownership repair.
- Root unpacked wheel: **2,718 passed / 1 skipped / 14 established builder
  warnings**, plus **267 executable documentation examples**. The nested unpacked
  wheel separately passes the **36** workflow/ownership tests. No native-owner
  unraisable errors are reported. The host test environment emits its existing
  urllib3/LibreSSL startup notice; it is separate from the pytest warning count.
- Repeated full `cargo test --offline --locked`: **1,566 passed / 54 ignored /
  zero failed**, across 220 suite results including doctests. Explicit serial
  checkpoint-process recovery: **1 passed**. Strict all-target Clippy: **passed**.
- Empty wheel input, mismatched release tags, stale receipt destinations and
  escaping source members reject. Python syntax checks and `git diff --check`
  pass. Workflow YAML parses; its matrix has 19 combinations and publication
  requires all three qualification jobs. This is local configuration validation,
  not remote workflow execution or an actionlint result.
- The three existing FFI build warnings remain. Black/isort/actionlint are not
  installed locally; none were installed as part of qualification.

Hashes, platform/toolchain, exact artifact smoke receipts and log hashes:
[`release-060-final-gates-2026-11-14.json`](release-060-final-gates-2026-11-14.json).
Transient artifacts/logs are under `/tmp/groggy-060-qualified-*` for this session;
they are not committed or uploaded. The adjacent JSON preserves their identity
after temporary build files are removed.

## Current performance baseline

The current native implementation already has the refreshed ownership baseline
in `notes/benchmarks/python-owner-workflow-2026-11-14.json`. Reuse those six raw
samples: three fresh processes each at 1k/100k, with no concurrent build/test
activity. No new native implementation in this slice changes that baseline.

| Nodes and edges each | Accept batch | Cold capture | Effective read | Archive save / recovery | Whole-process peak RSS |
| --- | --- | --- | --- | --- | --- |
| 1,000 | 4.806 ms | 0.836 ms | 1.867 ms | 35.527 / 7.136 ms | 17.30–18.58 MiB |
| 100,000 | 581.888 ms | 142.468 ms | 251.574 ms | 1,452.454 / 859.014 ms | 855.33–882.14 MiB |

Operation values are medians; RSS is a range across whole-process peaks. Fixture:
directed ring, low degree, one repeating scalar column, two versions, native
archive recovery. It does not measure journal/FFI throughput, dense embeddings,
deep history, hub-heavy graphs or universal memory/latency budgets. Existing
archive/memory scale work remains open under 32.2.

The separate matched public-Python access fixture reports small read variation
and **2.6%** overhead for a checked 1,000-property update after the ownership fix.
Its raw before/after samples are in
`notes/benchmarks/python-owner-access-2026-11-14.json`; these full Python calls
do not isolate the 100 ns FFI budget. No performance improvement is claimed.

## Repeatable release gates

The updated `.github/workflows/publish.yml` replaces retired macOS labels with
currently documented runners, explicitly selects CPython 3.8–3.12 (macOS arm64
3.9–3.12), and qualifies all 19 wheel combinations before uploading them. Each
runs the application/ownership gate; Linux 3.12 also runs maintained Python/docs.
A separate source job rebuilds and tests an extracted distribution. Native tests,
the serial checkpoint-process recovery suite and strict all-target Clippy also
gate publication. The publishing job downloads only distribution artifacts;
qualification receipts are separate artifacts. Tags must match crate versions.

Manual execution can qualify without publishing. Actual cross-platform execution
remains pending: editing YAML and local macOS results are not CI evidence. The
existing PyPI environment remains the publication boundary.

Sources for runner/interpreter configuration:
[GitHub runner inventory](https://github.com/actions/runner-images),
[Maturin action documentation](https://github.com/PyO3/maturin-action).
Local native source, rather than current upstream documentation, supplies the
pinned PyO3 maximum above.

See [the release runbook](../../docs/api/release-qualification.md) for commands.
The full original roadmap and broad parent rows remain open independently of
this bounded release. No service, arbitrary old-format migration, universal
power-loss or every-numerical-domain claim is added.

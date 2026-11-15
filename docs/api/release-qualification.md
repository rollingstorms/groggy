# Qualifying the embedded 0.6.0 release

This runbook qualifies the bounded scope in [the release notes](../../RELEASE_NOTES_0.6.0.md).
The [current status](../../RELEASE_STATUS.md) separates executed local checks from
pending CI evidence. Commands build and inspect artifacts without installing a
Groggy wheel. Publishing is a separate release action.

## Artifact gates

Use a clean checkout of the intended commit. Python, Rust, Maturin 1.9.0 and the
existing test dependencies must be available. Qualification pins Rust 1.96.1
(the locally verified compiler) rather than following new stable lint changes. Select the Python interpreter
explicitly; the current PyO3 dependency is not qualified for Python 3.13 or later.
Both package entry points declare `>=3.8,<3.13`, matching that build limit.

```sh
export GROGGY_PYTHON=/absolute/path/to/venv/bin/python
export GROGGY_RELEASE_OUT=/absolute/path/to/empty/release-output
export CARGO_TARGET_DIR="$PWD/python-groggy/target"
export CARGO_INCREMENTAL=0
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=0
export PYO3_PYTHON="$GROGGY_PYTHON"
```

Cargo lockfiles are currently ignored for library development. Resolve them once
for a fresh checkout, then retain those exact files through qualification. The
source distribution must contain both resolved lockfiles. Offline commands below
require cached dependencies; initial resolution needs network access when the
cache is empty. This is a locked qualification build, not a bit-for-bit build
reproducibility claim across fresh dependency resolutions.

```sh
cargo generate-lockfile
cargo generate-lockfile --manifest-path python-groggy/Cargo.toml
maturin build --release --offline --locked --interpreter "$GROGGY_PYTHON" --out "$GROGGY_RELEASE_OUT/root"
"$GROGGY_PYTHON" notes/scripts/verify_release_wheel.py "$GROGGY_RELEASE_OUT/root" --tests maintained --report "$GROGGY_RELEASE_OUT/root.json"
maturin sdist --out "$GROGGY_RELEASE_OUT/source"
"$GROGGY_PYTHON" notes/scripts/verify_release_sdist.py "$GROGGY_RELEASE_OUT/source" --offline --out "$GROGGY_RELEASE_OUT/rebuilt" --report "$GROGGY_RELEASE_OUT/source.json"
```

Also build from `python-groggy/` using the same interpreter and absolute target
directory, with a fresh output directory, then run the wheel verifier on that
directory. This checks the alternate packaging entry point. The source verifier
extracts to a temporary directory, refuses escaping/link members, requires both
lockfiles, builds with `--locked`, and runs the same isolated wheel smoke.

The wheel gate verifies metadata/runtime/native version agreement, all declared
exports, typing assets, immutable/f64/numerical smoke and the complete
[archive/journal application workflow](core-release-workflow.md). `--tests
maintained` additionally runs the maintained Python suite and curated executable
docs against the extracted artifact. `--tests ownership` runs the smaller
workflow/foreign-finalization suite. Only the test phase enables site-packages
for test dependencies; isolated smoke runs with `-I -S`. Neither installs a wheel.

Receipts are written only after success and include wheel SHA256 values. Existing
report paths reject to avoid retaining a stale success receipt. `--release-tag
v0.6.0` additionally requires the tag and both crate versions to agree.

## Native and performance gates

```sh
cargo test --offline --locked
cargo test --offline --locked --lib killed_checkpoint_process -- --ignored --test-threads=1
cargo clippy --offline --locked --all-targets -- -D warnings
git diff --check
```

The explicit serial checkpoint-process suite is a Unix process-recovery check;
it supplements ordinary tests rather than certifying arbitrary filesystem or
power-loss behavior. Preserve documented platform boundaries.

Reuse `benches/core_workflow_scale.rs`. Build with `cargo bench --offline --locked
--bench core_workflow_scale --no-run`; run the emitted executable in three fresh
processes each with arguments `1000` and `100000`, after builds/tests stop. On
macOS, `/usr/bin/time -l` provides process CPU/RSS in addition to the fixture's
per-operation times. Retain all raw samples and report medians and RSS ranges.
The [ownership receipt](../../notes/planning/python-owner-lifetime-2026-11-14.md)
records the current baseline and matched Python-access comparison. These bounded
measurements are evidence, not a universal performance threshold.

## CI and publication boundary

[The artifact workflow](../../.github/workflows/publish.yml) can be run manually
to qualify without publishing. It builds and exercises wheels for CPython
3.8–3.12 on Linux x86_64, Windows x86_64 and macOS Intel, and 3.9–3.12 on macOS
arm64. There is no native CPython 3.8 macOS arm64 wheel target. PyPy and newer
CPython releases have no qualification claim here. The runner labels follow
[GitHub's runner inventory](https://github.com/actions/runner-images); interpreter
selection follows [Maturin's action](https://github.com/PyO3/maturin-action).

Each wheel must pass the workflow and ownership gate before upload. Linux 3.12
also runs the full maintained Python/docs gate. Native tests, serial checkpoint
recovery, strict Clippy and rebuilding an extracted source distribution must all
pass before publishing can start. Wheel receipts upload separately from package
artifacts; only qualified wheels and the source distribution reach PyPI.

The publish job runs only for a published tag release, requires all qualification
jobs and retains the existing PyPI environment boundary. This configuration is
not evidence of a successful CI run. Review the actual matrix results, artifact
hashes, supported scope and migration notes before creating/publishing the release.

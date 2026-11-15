# Priority-pass source artifact qualification — 2026-11-15

Native implementation source: **a777e4ec8f8fef156d4c4477a4df44ebaca5d6f5**.
The source archive was built after the six implementation commits were pushed.
The working tree contains unrelated owner development/review files; no clean-tree
claim is made and those files were neither edited nor committed by this pass.

## Passed gates

1. Fresh root macOS arm64 CPython 3.9 wheel: isolated `-I -S` smoke, all 105 declared
   exports/version agreement, archive and durable application workflows; then
   **3,089 maintained Python tests / one skipped / 14 established warnings** and
   **283 executable documentation examples** against the unpacked artifact.
2. Fresh source archive retains both Cargo lockfiles. The existing safe source
   verifier extracted it into a temporary directory and built its wheel with
   cached dependencies using `--release --offline --locked` and the same Python.
3. The rebuilt source wheel passes the same isolated export/version smoke and
   archive/durable application workflows. Its receipt has `tests: none`: the
   maintained pytest/docs suite above ran against the fresh root wheel, not twice.

The implementation also passes **1,604 native tests / 54 ignored / zero failed**,
strict all-target Clippy and Rust formatting/whitespace checks. The native classic
pipeline target additionally passes with its earlier live CSR cache warmed before
pending revisions. No wheel, Python package or external dependency was installed.
No release tag or package publication was created. Latest-source cross-platform
qualification remains a future release gate; the older candidate's 19-platform
receipt has its own recorded source.

## Artifact identity

| Artifact | SHA256 |
| --- | --- |
| Root wheel | `9a1eb05bc5c239c8ee5124723922c6b28a3950ffb03dc7b3ef63a83daf9c427f` |
| Source archive | `2d9f2888d70a230325c780db03fb6bfb581bd57ffed203202c6cb5a9028bb4f0` |
| Wheel rebuilt from source | `70b6da3fd723ce00f01395a456696f655220e61982303789149ba3ad057163f5` |

Receipts: `notes/benchmarks/pending-path-wheel-qualification-2026-11-15.json` and
`notes/benchmarks/priority-source-qualification-2026-11-15.json`.
Artifacts remain in `/tmp/groggy-priority-2026-11-15-paths-wheel`,
`/tmp/groggy-priority-2026-11-15-source-dist` and
`/tmp/groggy-priority-2026-11-15-source-wheel` on this workstation.
Hashes identify these payloads, not a bit-for-bit reproducibility guarantee.

## Reproduction

Use the active Python, Maturin 1.9.0 and Rust 1.96.1, with the existing resolved
lockfiles. Build the root wheel using `maturin build --release --offline --locked`,
then run `notes/scripts/verify_release_wheel.py <wheel-directory> --tests maintained`.
Build the source archive with `maturin sdist --out <fresh-directory>` and run
`notes/scripts/verify_release_sdist.py <source-directory> --offline --out <fresh-wheel-directory>`.
Set an absolute CARGO_TARGET_DIR when reusing cached build artifacts across the
extracted temporary project. The source verifier installs nothing.

The fixed checklist stays **215 checked / 53 unchecked**, including **44 unchecked
release rows in scope**. See the pending-execution priority pass for completed
slices and remaining execution/service/scale requirements.

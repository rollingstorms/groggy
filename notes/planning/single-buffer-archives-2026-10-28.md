# Single-buffer archive preparation — 2026-10-28

Measured existing 10.4/32.2 follow-up to `core-workflow-scale-2026-10-28.md`.
No format/API/validation change and no new checklist row.

Archive preparation previously encoded a complete CBOR payload, copied it into a
second framed buffer, and kept the detached retained document alive during full
decode/restore validation. It now reserves the fixed header prefix, appends CBOR
directly into the framed vector, drops the document after encoding, requests
release of geometric growth slack, and fills the existing length/complement/
checksum fields. The prepared buffer remains independently owned.

The full encoded-byte decode/restore check still runs before PreparedArchive or
file publication can succeed. Configuration, portable values, raw referents,
selected state/history consistency, identities, groups and version rules still
validate. Atomic synced sibling/no-clobber file publication is unchanged. All
existing native archive corruption, frozen prepared ownership, old document
versions, collisions/endpoints, recovery and Python archive/journal tests remain
in the full gates. This allocation repair adds no parallel validation path.

## Matched release workflow

Same `benches/core_workflow_scale.rs` executable source on baseline `7779e107`
and the changed core. Two baseline/final processes at 100k nodes/edges and one at
1m, each with a two-version directed ring, bounded read/index/time work, complete
archive save, owner drop and recovery/history assertions. Final runs follow all
builds. Raw evidence: `notes/benchmarks/single-buffer-archives-2026-10-28.json`.

| Nodes / edges | Archive save before → after | Recovery before → after | Whole-process peak RSS before → after |
| --- | --- | --- | --- |
| 100,000 | 1.525 → 1.365 s | 0.868 → 0.821 s | 1,181.969 → 871.469 MiB |
| 1,000,000 | 24.731 → 20.674 s | 13.421 → 13.173 s | 4,062.031 → 4,282.734 MiB |

The 100k medians show 10.5% faster save and 26.3% lower peak RSS. The single
million pair saves 16.4% faster, but peak RSS is **5.4% higher** across the complete
workflow. No large-graph peak-memory improvement is claimed. The process-level
peak includes capture/query/effective replay/recovery, allocator retention and
macOS memory pressure, and is not attributed to a particular stage. Recovery code
is unchanged; its timing difference is not assigned to the encoder repair.
Large reconstruction/recovery memory remains a concrete open priority, with
stage-specific heap/footprint measurement needed before the next memory claim.
These observations are bounded local evidence, not a production budget.

## Verification and remaining scope

Final core gates: **1,442 native passed / 56 ignored**, strict all-target core
Clippy, **2,314 root Python passed / 1 skipped**, **227 executable docs**. The
CPython 3.9 full-ABI release extension was rebuilt and atomically staged; three
pre-existing FFI warnings remain. Offline locked Cargo uses
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target`; full cargo test,
strict Clippy, release FFI build and the workflow benchmark run. Scoped Rust
formatting/diff checks pass. Unrelated owner review/benchmark edits are excluded.

This remains a full buffered export with complete retained validation. It does
not introduce streaming archive decode, remove validation, bound deep-history
replay, change storage formats, schedule checkpoints or certify huge graphs.
10.4/32.2 and legacy connected ownership retain their stated remaining work.
Fixed checklist: **211 checked / 57 unchecked**.

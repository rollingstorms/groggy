# Chunked retained storage baselines — 10.4 component

Fourth priority window started 2026-10-26 12:52:12 UTC. No new checklist rows.
Counts remain **201 checked / 67 open**; 10.4 remains partial.

## Completed

- InitArchive CBOR encoding streams into at most 8 MiB data fragments. Every
  fragment has a checksum and ordinal; a terminator certifies aggregate count,
  byte length and full-stream xxh3 digest. Logical sequence zero remains one
  initialization, so subsequent append/checkpoint/report sequencing is unchanged.
- Small archives keep the old frame bytes. Existing journals remain readable;
  older binaries explicitly reject new chunk markers. Ordinary operations retain
  their existing 256 MiB per-record bound. No format migration was inferred.
- Recovery validates fragments into an automatically cleaned scratch file, then
  buffered decoding/restoration verifies the complete graph/history/schema before
  exposing an owner. Serialized bytes do not accumulate in RAM. Decoded state
  still scales with retained data, and scratch-disk capacity is required.
- Partial/corrupt/reordered/duplicated/spliced fragments, missing terminators,
  aggregate mismatches and duplicate/nonarchive chunked initialization reject
  without rewriting the source. Ordinary incomplete tails after a complete
  baseline retain existing recovery/truncation behavior.
- New journals publish a synced locked sibling temporary without replacing an
  existing file; encoding failures do not leave a named partial destination.
  Checkpoint permissions, atomic replacement and lock ownership are preserved.
  Checkpoint encoding now runs inside detached execute, outside Python's GIL;
  owner capture remains on the owner thread. Initial import/open still use their
  existing owner-bound Python paths, not a new GIL-release claim.

## Evidence

Five native fragment-contract tests include a matrix of truncation offsets and
checksum/order/manifest corruption. Existing 15 journal/checkpoint tests pass.
A Python 9 MiB text workflow exercises real production fragment sizing, immutable
source views, repeated checkpoints, continued writes, backup and reopen.

Release benchmark: `benches/large_storage_baseline.rs`; receipt:
`notes/benchmarks/large-storage-baseline-2026-10-26.json`. The 1/16/270 MiB payload
fixtures produce journals of 2.10/33.56/566.30 MB (retained raw columns and history
both carry values). Largest isolated-process reopen: **1.297 s**, peak RSS
**1,725,808,640 bytes** on this macOS host; checkpoint with sync and candidate
verification **4.829 s**. Warm filesystem cache, one sample per fixture, no SLA.
This demonstrates operation above the old 256 MiB ceiling, not bounded graph
memory or million-node scale. The high decoded-state RSS is documented remaining
work; framing alone cannot remove graph/archive/history copies.

Gates: **1,259 native passed / 51 ignored**, **2,000 root Python passed / 1 skipped**,
**158 documentation examples**, strict all-target Clippy; rebuilt in-tree FFI.
Owner benchmark and legacy nested survey suites remain excluded. No packages
installed; three pre-existing FFI warnings remain.

## Remaining existing scope

10.4: decoded-memory/opening optimization, target/budget decisions deferred A02,
format migrations, non-Unix replacement and additional process/power-loss corpus.
Raw reclamation/identity release stays 15/06.3. Algorithm inspection found no
registered built-in owner-topology mutation steps: general compound topology
execution still needs an explicit staged topology/identity/read contract under
09.4b.2/AR7.8; no speculative pipeline API was introduced to force closure.

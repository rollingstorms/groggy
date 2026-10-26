# Physical journal integrity verification — 2026-10-26

Existing 33.2/10.4 component. `check_integrity` verifies retained memory; the new
native/Python `DurableGraph.verify_storage` reads the actual named journal through
a separate read-only cursor, validates framing/checksums/sequences and semantic
replay, then verifies every retained state and current schema. It returns physical
byte/logical-record counts plus the nested retained integrity report.

## Shared design and failure contract

- Recovery and verification call one factored `read_journal` parser. Parsing never
  truncates/writes; only `open` retains the existing explicit tail-recovery step.
  Existing format/ordinary/chunked framing and replay behavior are unchanged.
- Capture retains the original locked inode, opens an independent read cursor and
  blocks the owner until its outcome is consumed. Execution and file/graph checks
  release Python's GIL. A detached plan/outcome retains the lock after owner drop.
- Verification rejects incomplete tails without repairing them, as well as complete
  corruption, changed paths/symlinks and mismatched captured state/sequence/schema.
  Errors leave the owner blocked until close/reopen. Reopening may recover only an
  incomplete final frame; complete corruption still rejects and is not truncated.
- The source file is never written by verification. Existing large-baseline scratch
  framing remains bounded as in recovery. Full decoded replay and retained audit
  own graph/history memory; this is not a constant-space or per-write operation.
- File validity and captured metadata are checked; a byte-for-byte comparison of
  the existing owner's every in-memory value with the decoded graph is not claimed.
  Derived index/cache consistency, service diagnostics and non-Unix inode checks
  retain their original scope. External tools that ignore locks are not certified.

## Evidence

Six native public cases verify successful stream/baseline reports, schema equality,
unchanged named bytes, continued append, poison/reopen after partial tails,
complete checksum errors, lock lifetime/foreign-outcome rejection, changed paths
before/after capture, and a real 9 MiB chunked baseline. An internal fixture with
valid hashes/framing but an impossible nonhead before-value proves this is a full
retained transition check, not just a checksum scan. Python covers the public
report plus tail/checksum non-repair behavior. Executable API documentation:
`docs/api/storage-verification.md`.

No new checklist row, source format, automatic schedule, external package, push or
Graphiti/service scope change. This closes the physical-file scan component inside
33.2; broader diagnostic parents remain open. Owner benchmark changes are untouched.

## Local cost and final gates

Release fixture: 1k nodes / 1,012 retained versions, three paired warm-cache
samples. Full physical replay plus retained audit median **53.19 ms** for an
11-record journal / 706,317 bytes, **50.27 ms** for a single materialized baseline /
706,953 bytes. Both verify 1,013 states including empty zero. These are explicit
audit costs, not mutation/FFI/isolated-RSS guarantees; the slightly larger baseline
also illustrates that checkpoint replacement does not always shrink a file.
Receipt: `../benchmarks/storage-verification-2026-10-26.json`.

Final gates: **1,292 native passed / 52 ignored**, dedicated serial process-death
matrix **1 passed / twelve cases**, **2,014 root Python passed / 1 skipped**,
**164 executable documentation examples**, strict all-target Clippy and rebuilt
in-tree extension. Scoped formatting/diff checks pass. The owner benchmark and
nested legacy surveys remain excluded from the Python gate; three pre-existing
FFI warnings remain. No external package installation or push.

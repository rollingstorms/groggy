# Native numerical memory ownership — 2026-11-16

Repairs concrete existing 29.4 / 19.2 / AR7.11 ownership and explicit-error scope;
allocation evidence contributes to 29.6/29.26. No checklist rows added or broad
numerical/compatibility parents closed.

## Findings and fixes

1. NativeView::data created a slice under a temporary read lock, then returned it
   after releasing that lock. Safe callers could retain the slice while another
   SharedBuffer write mutated it. BackendView's existing object-safe &[T] return
   cannot carry the source lock guard. Native backend views now own a typed read
   snapshot. Existing SharedBuffer::data/data_mut remain guarded, live, zero-copy
   access. No unsafe Send/lifetime extension is introduced.
2. Native snapshot refresh copies exact same-length input into its owned storage;
   failed size checks retain prior values. data_mut remains unsupported. Snapshot
   sync_to_main explicitly rejects rather than pretending to provide source
   writeback. Signatures/imports remain stable; behavior corrections are documented
   in docs/api/matrix-memory-ownership.md and the release notes.
3. Non-native synchronization previously returned success and changed a private
   backend tag without moving data. All four unavailable backends now reject,
   like their existing view requests. Unused private backend-status/view-cache
   placeholders are removed. Native residency validates the live read guard;
   poison returns an error. Failed mutable borrowing no longer changes generation.
4. NumericType is a safe public trait; zeroed bytes can be invalid for a custom
   implementation. The six known concrete built-ins retain zeroed allocation;
   other types are initialized with T::zero() before exposing slices. Zero-sized
   custom types reject before allocator calls. A NonZeroU8-backed test numeric
   type demonstrates a valid mathematical zero with a nonzero representation.

These changes do not introduce an external accelerator, Python API, storage format,
implicit matrix-to-backend copy or graph/history clone. Normal matrix operations
use SharedBuffer directly; they do not call view_for_backend implicitly. Explicit
backend snapshots now cost O(n), documented and measured separately from live
read guards. This is a safety/semantic correction, not a zero-copy snapshot claim.

## Focused evidence

Seven new native tests cover concurrent source writes while an earlier snapshot
slice is borrowed, source drop and cross-thread snapshot reads; bit-exact f64
NaN/signed-zero values; refresh size/failure and no writeback; all unavailable
backends and unchanged source/generation; poisoned storage and failed-borrow
generation; all six built-in zero/empty domains; custom valid zero representation,
and zero-sized rejection. The memory suite has **12 passing tests**.

Strict all-target Clippy passes with the benchmark extension. A native executable
documentation example covers the public snapshot/live-guard migration. Final full
native, release artifact/Python and measured allocation gates are recorded by the
priority-slice closeout at their tested source. The dedicated benchmark flag is
numerical_completion --backend-views, with existing process-wide allocator counters.

The defects were identified by direct source inspection. The unsafe previous
slice implementation was not exercised with a racy baseline; no undefined-behavior
run is used as a correctness or performance comparison.

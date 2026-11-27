# Portable edge-weight transformation — 2026-11-27

Finite component of existing 09.4b.2, 29.3, 30.1 and AR7.8. The final owner-bound
attribute transformation now delegates to a sealed selected-read kernel, shared
by memory and durable execution. Stable counts remain 217 checked / 51 open.

## Completed component checks

- [x] Share the edge-weight transform through the immutable read boundary.
- [x] Preserve selected pending creation/removal/endpoint revisions and edge IDs.
- [x] Preserve f64 numeric output, unit fallback, nonnumeric pass-through and the
  omitted factor default; reject explicitly malformed/nonfinite factors.
- [x] Reject nonfinite numeric results, preserve previous outputs and cancellation.
- [x] Verify memory/actual worker parity, sparse selection and effective-time recovery.
- [x] Verify pending write/ID/journal rollback and unattached no-publication behavior.

The owner adapter contains no numerical implementation. All attribute-module
steps now expose sealed read kernels; no numerical work moved to FFI. Scalar
numeric scaling retains its established f64 conversion, including precision
limits for large integers. Missing and unrepresentable step-value properties
retain the existing unit fallback; representable nonnumeric values pass through.
This is deliberately a legacy weight transform, not general attribute casting.
Only explicit attachment writes graph properties. New factor/error checks prevent
silent defaults for malformed explicit factors and prevent nonfinite results.

Evidence: two native safety tests, three worker/memory/recovery integration tests,
and 15 public Python cases cover numeric controls, typed values, directed and
undirected edge instances, pending topology, default weights, cancellation and
full rejection. The executable public contract is in
[durable pipelines](../../docs/api/durable-pipelines.md).

Local gates pass **1,700 native tests** (54 ignored), **3,612 packaged Python
tests** (one skipped, 23 expected warnings) and **303 documentation examples**.
Strict all-target Clippy, formatting, whitespace, Python 3.8 syntax and 62 packaged
Python/stub comparisons pass. All 500 production source files match runtime
`a80223251e43e95a20e5faf84049d362ab30fa64`. See [local receipt](../benchmarks/portable-edge-scale-local-qualification-2026-11-27.json)
and [wheel receipt](../benchmarks/portable-edge-scale-wheel-qualification-2026-11-27.json). This migration makes no new
performance or platform claim. Other retained-history producers, collections/
foreach, register/frontends and topology domains remain in existing parents.

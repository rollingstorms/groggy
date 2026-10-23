# 07.3a.1 — f64 property representation and lossless codec foundation

Implements the first dependency of the owner's accepted A06.2/A13.2 decision.
Append distinct scalar/vector f64 variants while preserving every existing f32
variant/tag and legacy read interpretation. Add identity/hash/order, dtype,
formatting, memory accounting and checked explicit numeric conversions. Typed
Python constructors and getters marshal f64 without narrowing. Schema floating
families accept both widths; explicit width constraints remain available.

Typed exports and checked journals use u64 bit-pattern tags, preserving tested
finite precision/range and nonfinite/zero payloads. Snapshot/reset and journal
replay retain width. f32 compression is never applied to f64. Captured weighted
reads, personalization, numeric comparison/input inference, matrix conversion and
algorithm parameter conversion recognize f64. Presentation serializers recognize
the new values, retaining their existing display/JSON policies.

This is a foundation commit, not the default switch or completion of 07.3a. The
remaining legacy table/array predicates/arithmetic, result narrowing, legacy
algorithm persistence/accessors, display/CSV inference and vector consumers need
migration before ordinary Python float/list inference switches. Keep this gate
visible; do not declare all float consumers complete. See
`docs/api/float64-properties.md`.

Correctness verifies typed identity/hash/canonical zero/NaN behavior, explicit
range conversion, retained scalar/vector bits through JSON/history/reset/journal,
explicit Python constructors/schema/weights/personalization, old f32 tags and the
existing regression surfaces. Timings compare one-attribute f32/f64 vector state
capture at 1k/10k elements. Raw payload doubles, not necessarily graph RSS.

Executed: locked offline full native test suite; strict all-target Clippy;
in-tree FFI rebuild; 416 Python property/history/schema/read/algorithm regressions
(5 existing skips); 13 documentation examples; release capture timing receipt.
The additional builder/pipeline gate exercises the parameter conversion change.
One histogram fixture still passed a now-unsupported explicit internal node ID;
update its setup to allocated IDs, preserving the same sequential IDs/values and
histogram assertions. No core identity contract is relaxed. Three pre-existing
FFI warnings remain; no external packages installed.
Additional builder/pipeline gate after the fixture correction: 73 passed,
2 existing skips. The histogram assertions remain unchanged.

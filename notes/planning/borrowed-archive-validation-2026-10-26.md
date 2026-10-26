# Borrowed validation of selected archive state

Existing 10.4/32.2 memory work, following chunked baseline commit 242e80e2.
No checklist rows added; **201 checked / 67 open**.

Archive capture and restore previously cloned a complete current-value snapshot
and encoded both entire property maps for equality against accepted history.
They now compare current column references/topology directly to the reconstructed
accepted snapshot. Text compares by borrowed bytes; other values use the existing
bit/tag-aware portable codec one pair at a time. Current columns and two encoded
whole-map copies are no longer materialized just for validation.

The accepted history snapshot is still reconstructed, and the decoded raw pool,
retained history and frozen sources remain owned. This is a measured copy
reduction, not constant-memory decoding or reclamation. Codec/format, selected
state, branches, identities, graph version/clock and publication policy are
unchanged. Active identity validation uses the already matched retained snapshot.

Tests compare exact portable-codec equality for null, integer storage variants,
f32/f64 signed zero and NaN payloads, vectors, text with NUL/Unicode, bytes, JSON
and group references. Direct current-column validation is compared to a snapshot
oracle for changed values, missing/extra properties, empty rows, topology/state
changes and corrupt indices. Existing complete archive/journal/Python contracts
remain in the full gates.

Paired release evidence: `notes/benchmarks/borrowed-archive-validation-2026-10-26.json`.
Reference code is 242e80e2, compiled separately in a temporary managed checkout;
three alternating before/after pairs use 270 MiB unique text and a 566 MB retained
baseline, with fresh isolated reopen processes and macOS peak RSS. Final measured
summary and gates are recorded in priority-slices-2026-10-26.md. These are local
warm-filesystem fixture measurements; no universal latency/RSS guarantee.

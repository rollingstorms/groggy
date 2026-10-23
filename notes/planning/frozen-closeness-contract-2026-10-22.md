# Frozen closeness — 2026-10-22

AR7.5k adds classic/harmonic closeness reads to AlgorithmRead/GraphView and Python.
The existing live distance/scoring kernel is shared; broader migration remains
open. No pending workbook policy is selected.

Preserve registry defaults (harmonic, outgoing, unnormalized), explicit incoming
mode, classic/harmonic formulas and disconnected normalization. Outputs are owned
f64 rows ordered by node ID. Empty input has no rows, isolates zero. Explicit
weights use shared checked selected-pair validation, missing=one, strictly
positive finite numeric values on all selected edges and minimum parallel length.
Absent weight selection uses hop distances regardless of stored properties.

Prepared outgoing calls reuse captured CSR. Incoming directed calls transpose
that CSR into private contiguous buffers, using one shared linear count/prefix/
fill implementation for live and captured readers. No per-node Vec buffers or
cache orientation pollution. Undirected direction choices coincide.

Values and selected membership stay frozen after owner mutation/drop. Python
releases the GIL. These reads publish no properties, deltas, checkpoints or live
caches. Native cancellation polls entry, traversal and final result, rejecting
partial output. Preparation, weight validation, transpose/allocation and each
single-row traversal are not bounded interruptible. Existing live persistence
and f32 representability checks remain; native/Python read scores stay f64.

The exhaustive Floyd–Warshall suite now checks captured and live results across
512 graph masks, both orientations, weighted/unweighted lengths, classic/harmonic,
incoming/outgoing and normalization modes. Additional fixtures cover retained
scope/values, owner drop, empty/isolate selections, cancellation, invalid weights,
minimum parallel lengths, concurrency, owned output and no property writes.

Executed: locked offline full `cargo test`, strict all-target `cargo clippy`,
in-tree FFI rebuild, Python read/algorithm/state/property regression gate
(**388 passed, 5 existing skips**), all **11** public read-view documentation
examples, and release Criterion timings. Receipt:
`benches/frozen_closeness_2026-10-22.json`; note:
`notes/benchmarks/frozen-closeness-2026-10-22.md`. Three pre-existing FFI warnings
remain. Black is unavailable locally; no package installs performed.

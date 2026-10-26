# Exact floating representation during retained state restoration

Follow-up correctness repair within existing 12.2/10.3/property-history scope.
No new checklist rows; **201 checked / 67 open**.

## Demonstrated problem

The new restoration lookup reused values by public AttrValue equality/hash. That
relation intentionally treats signed zeros as equal and canonicalizes NaN payloads
for property identity. Thus a retained column containing +0 before -0, or one NaN
payload before another, could supply the wrong bits when switching to the latter
accepted version. Snapshot history still held the exact recorded bits. The full
archive's strict selection/history check could then reject this clean graph.
Vector and compressed-f32 representations had the same risk.

## Repair

- Restoration fingerprints/equality now use exact f32/f64 scalar and vector bits,
  including decoded compressed-f32 vectors. Equivalent compressed/plain storage
  representations can still share a retained slot when the actual bits agree.
- Public property equality, query comparison and canonical content hashes remain
  unchanged. This is a storage reconstruction identity rule, not a new numeric or
  query policy. No serialization version or migration is required.
- Collision checks use that same representation-aware relation. Repeated switching
  and independent owned restoration reuse the correct existing slots; ordinary
  writes still append accepted values and retain their logical change events.

Native public regressions verify node/edge scalars, NaN payloads, signed zeros,
vectors, compressed vectors, repeated switches, open pins, exact owned copies and
binary archives, including unchanged column counts. Python workflows additionally
verify durable branch switching, physical checkpoint, reopen and backup.

## Performance scope

Repeated the existing `owned_snapshot_restoration` release fixture; no benchmark
or roadmap rows were added. At 100k nodes/99,999 edges, before/after column lengths
remain **100,000 / 200,000 / 99,999** and estimated score-column/lookup bytes remain
**3,435,304**. Current warm switch-pair median **0.604 s**, previous receipt
**0.564 s**. These separate local runs do not isolate timing noise or establish a
speed improvement; no engine-wide speed claim is made. Exact distinct NaN/zero
representations may require distinct lookup entries, as preserving them requires.
Receipt: `../benchmarks/owned-snapshot-restoration-bits-2026-10-26.json`.

Broader large-graph numerical/temporal throughput and peak-RSS gates remain their
existing scope. No automatic retention or API policy changed.

## Verification

Final native gate: **1,239 passed / 51 ignored**; root Python contract **1,975
passed / 1 skipped**; **156 executable docs examples**. Strict all-target Clippy,
scoped formatting, Python/stub syntax and diff checks pass. The owner benchmark
and legacy nested survey suites remain excluded. The in-tree extension is rebuilt;
three pre-existing FFI warnings remain, and no packages were installed.

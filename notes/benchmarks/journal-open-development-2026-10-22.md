# Warm journal opening baseline — 2026-10-22

Owner A02.1 requests measured opening time before checkpoint scheduling; A15.1
adds elapsed compute/wall timing. This baseline uses the current **development
in-tree Python native extension**, not an optimized release or cold-device run.
One live node, no edges, repeated checked scalar updates. All writes are accepted
and synced before opening measurements. Seven samples after one explicit warmup.

| Accepted batches | Journal bytes | Median open wall | Median open process CPU |
| --- | ---: | ---: | ---: |
| 1 | 1052 | 0.130 ms | 0.130 ms |
| 101 | 43283 | 5.684 ms | 5.681 ms |
| 1001 | 426218 | 56.260 ms | 56.260 ms |

Time covers the open API through its returned handle: lock/header/checksum/replay
and Python/native call overhead. Setup, validation, close and old handle destruction
are excluded. CPU time is process CPU during that call, not kernel-only compute.
Fixture validation checks recovered head and last property value after each sample.
The script uses temporary journals and removes them when complete.

The live graph is fixed while history grows; warm opening grows with replay on
these fixtures. This does not set a checkpoint schedule, maximum log budget or
production SLA. It does not measure cold cache, storage-device bandwidth, release
builds, million-node graphs, retained vectors/meta content, allocations or RSS.
Future tiers should vary graph size/payloads/history and use target hardware before
selecting thresholds. Current storage still replays the full journal; a full-log
backup is not a compact recovery checkpoint.

Reproduce with the in-tree package selected through PYTHONPATH:
`python3 benches/journal_open_profile.py --output /tmp/journal-open.json`.
Raw receipt: `benches/journal_open_development_2026-10-22.json`.

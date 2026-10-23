# Graph table export timing receipts — 2026-10-22

Locked offline release Criterion runner; 200 ms warmup, 1 second measurement,
10 samples. Ring fixtures with default entity_type and one scalar node property.
Graph setup/capture excluded; owned node/edge table construction/drop included.
No competing builds/tests during the retained run. No previous-exporter baseline,
FFI latency, memory/RSS budget or production SLA claim.

| Nodes / edges | Current live GraphTable export | Frozen GraphTable export |
| --- | ---: | ---: |
| 1,000 / 1,000 | 0.208 ms | 0.271 ms |
| 10,000 / 10,000 | 2.189 ms | 5.078 ms |

Frozen projection is slower on these fixtures. It discovers selected names and
reads per-entity captured maps; live projection discovers pooled attribute names
and reads GraphSpace value references. Both produce owned column data/metadata.
Capture is reused, not eliminated: these numbers exclude its separate cost.
Column-oriented frozen storage remains an optimization opportunity. No speedup
against the previous row-assembly code is claimed by this correctness migration.

Receipt: `benches/frozen_table_export_2026-10-22.json`; fixture:
`benches/read_state.rs`. Filter: `read_state/(live_tables|frozen_tables)`.

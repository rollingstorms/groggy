# Shared read-state timing receipts — 2026-10-22

Locked offline release Criterion benchmark `benches/read_state.rs`. Idle runner,
200 ms warmup/1 second measurement/10 samples. Ring fixtures have one current
integer property plus system identity/type properties; setup excluded, captured
output creation/drop included. These are bounded current costs, not an old/new
comparison, allocation/RSS report or production budget.

| Nodes / edges | Fresh capture estimate | Shared pin clone estimate |
| --- | ---: | ---: |
| 1,000 / 1,000 | 574 µs | 10.24 ns |
| 10,000 / 10,000 | 7.63 ms | 10.22 ns |

Estimates are Criterion regression estimates with recorded confidence intervals;
the 10k capture has one mild high outlier. A fresh capture owns values/topology and
is materially more expensive than sharing a pin. Related views should share one
pin, not recapture the complete working graph on every selection. Clone timings
are native Arc ownership costs, **not FFI call latency**.

Receipt: `benches/read_state_2026-10-22.json`. Reproduce with
`CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline --locked --bench read_state -- --noplot`.

# Table statistics release receipts — 2026-10-22

The release corpus now includes 52 cases; the prior 50 checksums match within
1e-13. One warmup and seven idle measured samples, setup excluded, outputs/drop
included; process System heap requests/extra live heap only, not stack or RSS.

| High-offset integer rows | Describe median | Allocation calls | Additional live heap |
| --- | ---: | ---: | ---: |
| 1,700 | 0.180 ms | 160 | 191,392 bytes |
| 17,000 | 1.650 ms | 180 | 2,893,792 bytes |

Each result checks that population spread survives near i64::MAX. This is a
bounded current receipt, not an old/new comparison or universal latency budget.
Describe computes one summary per column; percentile sorting still allocates
working arrays and is not claimed allocation-free. Receipt:
`benches/numerical_table_statistics_2026-10-22.json`.

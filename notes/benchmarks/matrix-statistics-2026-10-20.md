# Matrix axis-statistics baseline - 2026-10-20

Fixture: 1024 rows x 384 columns, f64, sample variance divisor n-1. Each row repeats the same sequence of values modulo 17. Median of five warmed calls per axis; timings include the Python/native call and result construction, excluding matrix construction and Python list conversion. The first result is checked against Python statistics.stdev. Machine: local macOS arm64; debug extension, not a release performance claim.

| Reduction | Correct Welford before dense read guard | One dense read guard |
| --- | ---: | ---: |
| Per row (Groggy axis 0) | 36.21 ms | 16.58 ms |
| Per column (Groggy axis 1) | 36.14 ms | 16.81 ms |
| Whole-process peak RSS | 43,499,520 bytes | 43,483,136 bytes |

This comparison measures the dense read-guard optimization against the already-correct Welford implementation. It does not compare against the earlier incorrect sum placeholder. Whole-process RSS is not per-operation allocation, and these timings do not establish sparse, decomposition or production-scale throughput. The reduction uses f64 accumulators and O(number of outputs) auxiliary storage. Column-major correctness is covered separately by tests.

Reproduce with the in-tree extension on PYTHONPATH:

```sh
python3 benches/matrix_statistics_profile.py --rows 1024 --cols 384 --repeat 5
```

Raw samples: benches/matrix_statistics_before_2026-10-20.json and benches/matrix_statistics_after_2026-10-20.json. Release measurement, broader shapes, larger working sets, dtype/output guarantees and other numerical kernels remain item 29/32 work.

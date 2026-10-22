"""Compare exact-bin dots with an integer fast path in matching dev extensions.
Whole-process RSS is reported; these are not release or per-allocation measurements.
"""
import json
import math
import resource
import statistics
import sys
import time
from groggy import GraphMatrix

results = []
for rows, inner, cols, kind in [(1, 1024, 1, 'integral'), (16, 32, 16, 'integral'),
                                (32, 64, 32, 'integral'), (16, 32, 16, 'fractional')]:
    row = [float(k % 7 - 3) * (1. if kind == 'integral' else .1) for k in range(inner)]
    a = GraphMatrix.from_data([[value] * rows for value in row])
    b = GraphMatrix.from_data([[1.] * inner for _ in range(cols)])
    expected = math.fsum(row)
    a * b
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        product = a * b
        samples.append((time.perf_counter() - start) * 1000)
        assert product.get(0, 0) == expected
    results.append({'shape': [rows, inner, cols], 'kind': kind,
                    'median_ms': statistics.median(samples), 'samples_ms': samples})
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    rss *= 1024
print(json.dumps({'build': 'dev, CARGO_INCREMENTAL=0', 'dtype': 'f64', 'timings': results,
                  'process_peak_rss_bytes': rss}, indent=2))

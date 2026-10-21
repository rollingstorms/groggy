"""Profile in-tree GraphMatrix products; records whole-process RSS, not allocations.
Use the same debug or release extension build for before/after comparisons.
"""
import json
import resource
import statistics
import sys
import time
from groggy import GraphMatrix

results = []
for rows, inner, cols in [(1, 1024, 1), (16, 32, 16), (32, 64, 32)]:
    a = GraphMatrix.from_data([[float((k % 7) - 3)] * rows for k in range(inner)])
    b = GraphMatrix.from_data([[1.] * inner for _ in range(cols)])
    expected = sum(float((k % 7) - 3) for k in range(inner))
    a * b
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        result = a * b
        samples.append((time.perf_counter() - start) * 1000)
        assert result.get(0, 0) == expected
    results.append({'shape': [rows, inner, cols], 'median_ms': statistics.median(samples),
                    'samples_ms': samples})
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    rss *= 1024
print(json.dumps({'build': 'dev, CARGO_INCREMENTAL=0', 'dtype': 'f64', 'timings': results,
                  'process_peak_rss_bytes': rss}, indent=2))

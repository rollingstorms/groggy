"""Matching dev determinant wall-time/RSS fixtures; not release throughput."""
import json
import resource
import statistics
import sys
import time
from groggy import GraphMatrix

results = []
for n in [4, 6]:
    data = [[2. if row == col else (.1 if row > col else 0.) for row in range(n)] for col in range(n)]
    a = GraphMatrix.from_data(data)
    a.determinant()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        result = a.determinant()
        samples.append((time.perf_counter() - start) * 1000)
        assert abs(result / (2.**n) - 1.) < 1e-12
    results.append({'order': n, 'median_ms': statistics.median(samples), 'samples_ms': samples})
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != 'darwin':
    rss *= 1024
print(json.dumps({'build': 'dev, CARGO_INCREMENTAL=0', 'timings': results,
                  'process_peak_rss_bytes': rss}, indent=2))

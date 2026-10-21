"""Time the in-house decompositions and verify residuals against NumPy.

Run against the in-tree extension via PYTHONPATH. Build profile is caller supplied;
these timings include the Python/native call, excluding input construction.
"""
import argparse
import json
import platform
import resource
import statistics
import sys
import time

import numpy as np
from groggy import GraphMatrix


def core(a):
    return GraphMatrix.from_data(a.T.tolist())


def values(a):
    return np.asarray(a.to_numpy())


def timed(call, repeat):
    call()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - start) * 1000)
    return result, {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--build-profile", default="unspecified")
    args = parser.parse_args()
    if args.size < 2 or args.repeat < 1:
        parser.error("size must exceed one and repeat must be positive")
    rng = np.random.default_rng(401)
    records = []
    for shape in [(2 * args.size, args.size), (args.size, 2 * args.size)]:
        a = rng.normal(size=shape)
        matrix = core(a)
        (u, s, vt), timing = timed(matrix.svd, args.repeat)
        u, vt = values(u), values(vt)
        k = len(s)
        residual = float(np.linalg.norm((u[:, :k] * s) @ vt[:k, :] - a) / np.linalg.norm(a))
        assert residual < 1e-10
        assert np.allclose(s, np.linalg.svd(a, compute_uv=False), rtol=1e-10)
        records.append({"operation": "svd", "shape": shape, "relative_residual": residual, **timing})
    a = rng.normal(size=(args.size, args.size))
    a = (a + a.T) / 2
    matrix = core(a)
    (s, v), timing = timed(matrix.eigenvalue_decomposition, args.repeat)
    v = values(v)
    residual = float(np.linalg.norm(a @ v - v * s) / np.linalg.norm(a))
    assert residual < 1e-10
    assert np.allclose(s, np.linalg.eigvalsh(a)[::-1], atol=1e-10)
    records.append({"operation": "symmetric_eigen", "shape": a.shape, "relative_residual": residual, **timing})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    print(json.dumps({"build_profile": args.build_profile, "platform": platform.platform(),
                      "machine": platform.machine(), "numpy_reference_version": np.__version__,
                      "process_peak_rss_bytes": rss, "records": records}, indent=2))


if __name__ == "__main__":
    main()

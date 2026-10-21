# Dense decomposition baseline - 2026-10-21

In-house bounded Jacobi kernels, debug extension, local macOS arm64. Five warmed calls, median wall time including Python/native call and output construction; input construction and NumPy reference checks are excluded. Full U/V factors are requested. NumPy 2.0.2 independently checks spectra and reconstruction. These are first-correct baselines, not comparisons against the old placeholder routines or release throughput claims.

| Kernel | Input shape | Median | Relative residual |
| --- | --- | ---: | ---: |
| SVD | 48 x 24 | 14.67 ms | 3.60e-14 |
| SVD | 24 x 48 | 10.47 ms | 5.27e-14 |
| Symmetric eigen | 24 x 24 | 2.12 ms | 2.79e-14 |

Whole-process peak RSS was 37,273,600 bytes; this includes Python, NumPy and the extension and is not per-operation allocation. Full factors have quadratic output storage. Sparse/large-scale and optimized release measurements remain open.

```sh
python3 benches/matrix_decomposition_profile.py --size 24 --repeat 5 --build-profile debug
```

Use the in-tree extension on PYTHONPATH. Raw samples and environment are in benches/matrix_decomposition_baseline_2026-10-21.json. Numerical domains and tolerances are documented in docs/api/matrix-decompositions.md.

# Leiden levels: bounded debug profile — 2026-10-21

Run `PYTHONPATH=/tmp/groggy-python-check python3 benches/leiden_levels_profile.py` against the explicitly rebuilt in-tree extension. Before: `edc84838`, local moves plus connectivity splitting. After: fast moves, constrained refinement and refined aggregation. Both debug builds use seed 42, 100 node-work units per phase/level and five timed repetitions; graph construction is excluded and final label writes included. Registry setup may affect the first sample.

| Nodes / edges | Phase cap | Before median ms | After median ms | Before → after communities |
|---|---:|---:|---:|---:|
| 48 / 64 | 1 | 1.237 | 0.714 | 16 → 16 |
| 48 / 64 | 10 | 1.681 | 0.784 | 16 → 9 |
| 384 / 512 | 1 | 9.326 | 4.889 | 128 → 128 |
| 384 / 512 | 10 | 11.771 | 6.854 | 128 → 24 |

These algorithms perform different work, and max_iter now bounds queue visits rather than repeated full sweeps. No row establishes an equivalent-work speedup, release performance or general scaling claim. The fixtures are rings of three-node cliques, deliberately small and structured. Original-graph modularity and connectedness are checked separately by native/Python tests; fewer communities alone is not a correctness measure.

Whole-process maximum RSS on macOS was 28,229,632 bytes before and 28,622,848 after, including Python/runtime. This is not Rust-only or per-operation allocation. Shared quotient CSR, queue/membership/refinement buffers and sorted candidate maps use O(V+E) temporary storage; constructing each quotient still costs work. No external package/backend was installed or invoked.

Raw repetitions: `benches/leiden_levels_before_2026-10-21.json` and `benches/leiden_levels_after_2026-10-21.json`. Large-graph memory, release throughput, stochastic quality distributions and external backend comparisons remain open gates.

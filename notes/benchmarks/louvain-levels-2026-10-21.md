# Louvain levels: bounded debug profile — 2026-10-21

Command: `PYTHONPATH=/tmp/groggy-python-check python3 benches/louvain_levels_profile.py`.
Baseline: `f5bfef4e` in-tree extension, before multilevel implementation. After: current multilevel implementation, both built in debug mode with incremental compilation disabled. Same Python script and explicit level limits; five repetitions per case, graph construction excluded and result-column writes included. The first invocation can include registry initialization. No dependencies were installed.

| Nodes / edges | Phase cap | Before median ms | After median ms | Before → after communities |
|---|---:|---:|---:|---:|
| 48 / 64 | 1 | 1.104 | 0.596 | 16 → 16 |
| 48 / 64 | 10 | 1.103 | 0.712 | 16 → 8 |
| 384 / 512 | 1 | 8.098 | 4.367 | 128 → 128 |
| 384 / 512 | 10 | 7.945 | 4.940 | 128 → 32 |

The old phase cap of ten was ignored, so those two rows compare different work and are not speedup measurements. The cap-one cases perform equivalent local optimization on these fixtures; reduced repeated per-candidate adjacency scanning and compact local buffers improve these measured debug timings. Neither result establishes a general or release-build speedup. Native/Python tests separately check higher original-graph modularity after multiple levels; fewer communities alone is not a correctness metric.

The fixture is a ring of three-node cliques. It is deliberately small and structured, not representative of arbitrary million/billion-node graphs. Whole-process maximum RSS was 28,147,712 bytes before and 28,377,088 after on macOS; those include Python/runtime and are not per-operation or Rust-only allocation measurements.

The implementation keeps O(V+E) temporary graph/buffer storage, using sorted pair maps to combine coarse connections with checked u64 edge mass, then weighted CSR for local moves. Coarsening adds work and retains internal loops; it does not mutate graph edge instances or the shared subgraph cache. Large-scale memory, release throughput and backend comparisons remain open gates.

Raw data: `benches/louvain_levels_before_2026-10-21.json` and `benches/louvain_levels_after_2026-10-21.json`. Store all repetitions, not just these medians.

# Groggy Algorithm Benchmark Results

Generated: 2026-03-06 20:55:48

## Connected Components

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0000 | 0.00 | ✓ baseline |
| igraph | 0.0001 | 0.00 | 3.19x |
| groggy | 0.0002 | 0.02 | 6.67x |
| networkx | 0.0004 | 0.00 | 13.70x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0003 | 0.00 | ✓ baseline |
| igraph | 0.0009 | 0.00 | 2.72x |
| groggy | 0.0012 | 0.00 | 3.61x |
| networkx | 0.0041 | 0.00 | 12.76x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0006 | 0.00 | ✓ baseline |
| igraph | 0.0017 | 0.00 | 2.85x |
| groggy | 0.0023 | 0.00 | 3.74x |
| networkx | 0.0124 | 0.00 | 20.14x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0017 | 0.00 | ✓ baseline |
| igraph | 0.0047 | 0.00 | 2.70x |
| groggy | 0.0058 | 1.72 | 3.34x |
| networkx | 0.0456 | 0.08 | 26.16x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0050 | 0.00 | ✓ baseline |
| igraph | 0.0139 | 0.00 | 2.80x |
| groggy | 0.0158 | 0.28 | 3.17x |
| networkx | 0.1471 | 0.03 | 29.63x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0143 | 0.66 | ✓ baseline |
| groggy | 0.0367 | 1.56 | 2.57x |
| igraph | 0.0376 | 0.53 | 2.64x |
| networkx | 0.3520 | 2.56 | 24.66x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.1266 | 0.00 | ✓ baseline |
| igraph | 0.2762 | 0.00 | 2.18x |
| groggy | 0.3775 | 63.69 | 2.98x |
| networkx | 7.1336 | 133.55 | 56.35x |

## Label Propagation

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0003 | 0.00 | ✓ baseline |
| networkit | 0.0007 | 0.02 | 2.36x |
| igraph | 0.0040 | 0.00 | 12.77x |
| networkx | 0.0141 | 0.00 | 45.17x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0022 | 0.05 | ✓ baseline |
| groggy | 0.0028 | 0.00 | 1.31x |
| networkx | 0.1432 | 0.23 | 66.29x |
| igraph | 0.2034 | 0.00 | 94.14x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0039 | 0.00 | ✓ baseline |
| groggy | 0.0084 | 0.00 | 2.19x |
| networkx | 0.2750 | 0.08 | 71.26x |
| igraph | 0.5249 | 0.02 | 136.00x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0081 | 0.03 | ✓ baseline |
| groggy | 0.0315 | 0.36 | 3.89x |
| networkx | 0.9704 | 0.00 | 119.95x |
| igraph | 2.1239 | 0.00 | 262.55x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0168 | 0.02 | ✓ baseline |
| groggy | 0.1224 | 0.00 | 7.29x |
| igraph | 2.0159 | 0.00 | 120.05x |
| networkx | 2.1543 | 0.00 | 128.30x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0402 | 1.09 | ✓ baseline |
| groggy | 0.5037 | 1.55 | 12.53x |
| networkx | 4.6280 | 0.00 | 115.15x |
| igraph | 18.5743 | 0.00 | 462.13x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.1920 | 0.12 | ✓ baseline |
| groggy | 10.5183 | 0.00 | 54.79x |
| networkx | 44.5906 | 0.00 | 232.29x |
| igraph | 811.3402 | 0.00 | 4226.51x |

## Pagerank

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0003 | 0.09 | ✓ baseline |
| igraph | 0.0009 | 0.00 | 3.27x |
| networkx | 0.0032 | 0.00 | 11.25x |
| networkit | 0.0062 | 0.00 | 21.69x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0025 | 0.00 | ✓ baseline |
| networkit | 0.0070 | 0.00 | 2.85x |
| igraph | 0.0087 | 0.00 | 3.53x |
| networkx | 0.0260 | 1.14 | 10.56x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0045 | 0.00 | ✓ baseline |
| networkit | 0.0095 | 0.00 | 2.10x |
| igraph | 0.0196 | 0.00 | 4.33x |
| networkx | 0.0994 | 1.44 | 21.99x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0106 | 0.41 | ✓ baseline |
| networkit | 0.0116 | 0.00 | 1.09x |
| igraph | 0.0494 | 0.31 | 4.66x |
| networkx | 0.2737 | 11.67 | 25.81x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0137 | 0.00 | ✓ baseline |
| groggy | 0.0223 | 0.00 | 1.64x |
| igraph | 0.1117 | 1.75 | 8.18x |
| networkx | 0.8314 | 26.14 | 60.89x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0401 | 0.00 | ✓ baseline |
| groggy | 0.0567 | 0.00 | 1.41x |
| igraph | 0.2318 | 0.45 | 5.78x |
| networkx | 1.7926 | 95.83 | 44.70x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.6717 | 0.00 | ✓ baseline |
| groggy | 0.6922 | 0.00 | 1.03x |
| igraph | 1.7267 | 0.00 | 2.57x |
| networkx | 35.2634 | 0.00 | 52.50x |

## Performance Insights

- Groggy was fastest in **5/21** test cases
- Average speedup when groggy wins: calculated per algorithm

### Key Takeaways

- Groggy's Rust core provides competitive performance across all algorithms
- Memory efficiency remains consistent with other optimized libraries
- The new algorithm pipeline API maintains performance while improving usability

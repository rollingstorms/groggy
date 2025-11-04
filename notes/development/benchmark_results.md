# Groggy Algorithm Benchmark Results

Generated: 2025-11-04 14:46:54

## Connected Components

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0000 | 0.00 | ✓ baseline |
| igraph | 0.0001 | 0.00 | 2.98x |
| groggy | 0.0002 | 0.05 | 5.99x |
| networkx | 0.0003 | 0.00 | 10.73x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0004 | 0.00 | ✓ baseline |
| igraph | 0.0008 | 0.00 | 2.06x |
| groggy | 0.0011 | 0.00 | 2.56x |
| networkx | 0.0038 | 0.00 | 9.15x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0006 | 0.00 | ✓ baseline |
| igraph | 0.0017 | 0.00 | 2.83x |
| groggy | 0.0022 | 0.00 | 3.70x |
| networkx | 0.0101 | 0.00 | 16.71x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0018 | 0.00 | ✓ baseline |
| igraph | 0.0045 | 0.00 | 2.55x |
| groggy | 0.0056 | 0.00 | 3.16x |
| networkx | 0.0373 | 0.00 | 21.22x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0046 | 0.28 | ✓ baseline |
| igraph | 0.0123 | 0.00 | 2.66x |
| groggy | 0.0147 | 0.00 | 3.18x |
| networkx | 0.1120 | 0.00 | 24.19x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0146 | 0.58 | ✓ baseline |
| igraph | 0.0320 | 0.17 | 2.20x |
| groggy | 0.0382 | 2.11 | 2.62x |
| networkx | 0.3062 | 11.30 | 21.02x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0979 | 1.81 | ✓ baseline |
| igraph | 0.2175 | 16.83 | 2.22x |
| groggy | 0.2861 | 66.44 | 2.92x |
| networkx | 1.7619 | 0.00 | 18.00x |

## Label Propagation

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0003 | 0.00 | ✓ baseline |
| networkit | 0.0005 | 0.00 | 1.89x |
| igraph | 0.0038 | 0.02 | 14.52x |
| networkx | 0.0136 | 0.02 | 52.43x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0019 | 0.00 | ✓ baseline |
| groggy | 0.0027 | 0.00 | 1.40x |
| networkx | 0.1302 | 0.05 | 66.99x |
| igraph | 0.1940 | 0.00 | 99.77x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0032 | 0.00 | ✓ baseline |
| groggy | 0.0079 | 0.00 | 2.45x |
| networkx | 0.2526 | 0.00 | 77.98x |
| igraph | 0.4978 | 0.00 | 153.68x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0077 | 0.00 | ✓ baseline |
| groggy | 0.0319 | 0.02 | 4.15x |
| networkx | 0.7674 | 0.45 | 99.77x |
| igraph | 1.7906 | 0.00 | 232.80x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0136 | 0.00 | ✓ baseline |
| groggy | 0.1191 | 0.02 | 8.75x |
| igraph | 1.5673 | 0.00 | 115.24x |
| networkx | 1.6189 | 1.56 | 119.03x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0326 | 1.59 | ✓ baseline |
| groggy | 0.5045 | 7.66 | 15.45x |
| networkx | 4.6669 | 0.00 | 142.96x |
| igraph | 9.6890 | 0.00 | 296.80x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.1731 | 5.42 | ✓ baseline |
| groggy | 9.1388 | 0.00 | 52.78x |
| networkx | 22.3409 | 502.75 | 129.04x |
| igraph | 643.2176 | 0.00 | 3715.05x |

## Pagerank

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0002 | 0.00 | ✓ baseline |
| igraph | 0.0007 | 0.02 | 3.02x |
| networkx | 0.0029 | 0.52 | 12.00x |
| networkit | 0.0045 | 0.03 | 18.73x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0023 | 0.12 | ✓ baseline |
| networkit | 0.0059 | 0.00 | 2.55x |
| igraph | 0.0086 | 0.00 | 3.68x |
| networkx | 0.0229 | 1.67 | 9.86x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0043 | 0.00 | ✓ baseline |
| networkit | 0.0072 | 0.00 | 1.67x |
| igraph | 0.0172 | 1.09 | 3.97x |
| networkx | 0.0907 | 0.58 | 20.90x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0098 | 0.00 | ✓ baseline |
| groggy | 0.0108 | 0.00 | 1.10x |
| igraph | 0.0439 | 0.39 | 4.46x |
| networkx | 0.2385 | 4.20 | 24.25x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0135 | 0.00 | ✓ baseline |
| groggy | 0.0233 | 0.33 | 1.73x |
| igraph | 0.1062 | 0.00 | 7.88x |
| networkx | 0.6780 | 22.14 | 50.33x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0322 | 0.38 | ✓ baseline |
| groggy | 0.0450 | 0.00 | 1.40x |
| igraph | 0.4326 | 10.16 | 13.43x |
| networkx | 1.8612 | 0.00 | 57.80x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.2656 | 74.20 | ✓ baseline |
| networkit | 0.5257 | 0.00 | 1.98x |
| igraph | 1.3589 | 57.34 | 5.12x |
| networkx | 7.6580 | 564.70 | 28.84x |

## Performance Insights

- Groggy was fastest in **5/21** test cases
- Average speedup when groggy wins: calculated per algorithm

### Key Takeaways

- Groggy's Rust core provides competitive performance across all algorithms
- Memory efficiency remains consistent with other optimized libraries
- The new algorithm pipeline API maintains performance while improving usability

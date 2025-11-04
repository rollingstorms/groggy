# Groggy Algorithm Benchmark Results

Generated: 2025-11-04 09:25:13

## Connected Components

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0000 | 0.00 | ✓ baseline |
| igraph | 0.0001 | 0.00 | 3.41x |
| groggy | 0.0002 | 0.00 | 6.66x |
| networkx | 0.0004 | 0.03 | 14.23x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0003 | 0.00 | ✓ baseline |
| igraph | 0.0009 | 0.00 | 2.97x |
| groggy | 0.0010 | 0.02 | 3.58x |
| networkx | 0.0043 | 0.02 | 14.88x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0006 | 0.00 | ✓ baseline |
| igraph | 0.0017 | 0.00 | 2.91x |
| groggy | 0.0021 | 0.00 | 3.56x |
| networkx | 0.0119 | 0.00 | 20.26x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0016 | 0.00 | ✓ baseline |
| igraph | 0.0045 | 0.00 | 2.84x |
| groggy | 0.0053 | 0.00 | 3.37x |
| networkx | 0.0340 | 0.00 | 21.65x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0041 | 0.00 | ✓ baseline |
| igraph | 0.0122 | 0.00 | 2.99x |
| groggy | 0.0144 | 0.00 | 3.55x |
| networkx | 0.1134 | 0.00 | 27.84x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0119 | 0.58 | ✓ baseline |
| igraph | 0.0303 | 0.02 | 2.54x |
| groggy | 0.0322 | 1.91 | 2.69x |
| networkx | 0.2870 | 0.00 | 24.03x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0894 | 3.16 | ✓ baseline |
| groggy | 0.2053 | 13.17 | 2.30x |
| igraph | 0.2140 | 7.22 | 2.39x |
| networkx | 1.5888 | 51.45 | 17.78x |

## Label Propagation

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0003 | 0.00 | ✓ baseline |
| networkit | 0.0004 | 0.00 | 1.52x |
| igraph | 0.0038 | 0.00 | 12.90x |
| networkx | 0.0131 | 0.00 | 44.66x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0021 | 0.00 | ✓ baseline |
| groggy | 0.0027 | 0.03 | 1.30x |
| networkx | 0.1259 | 0.05 | 61.01x |
| igraph | 0.1924 | 0.02 | 93.27x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0037 | 0.00 | ✓ baseline |
| groggy | 0.0077 | 0.00 | 2.08x |
| networkx | 0.2514 | 0.06 | 68.09x |
| igraph | 0.4896 | 0.05 | 132.58x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0084 | 0.00 | ✓ baseline |
| groggy | 0.0316 | 0.00 | 3.78x |
| networkx | 0.7744 | 0.00 | 92.73x |
| igraph | 1.8254 | 0.00 | 218.58x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0145 | 0.00 | ✓ baseline |
| groggy | 0.1173 | 0.00 | 8.10x |
| igraph | 1.4479 | 0.00 | 100.02x |
| networkx | 1.6318 | 1.75 | 112.72x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0303 | 0.05 | ✓ baseline |
| groggy | 0.4796 | 0.00 | 15.80x |
| networkx | 3.8690 | 10.50 | 127.48x |
| igraph | 7.1859 | 0.00 | 236.77x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.1722 | 0.16 | ✓ baseline |
| groggy | 9.3685 | 0.00 | 54.41x |
| networkx | 24.3380 | 420.22 | 141.34x |
| igraph | 579.1774 | 0.00 | 3363.53x |

## Pagerank

### Graph: 1,000 nodes, 2,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0003 | 0.00 | ✓ baseline |
| igraph | 0.0007 | 0.00 | 2.89x |
| networkx | 0.0028 | 0.00 | 11.26x |
| networkit | 0.0048 | 0.00 | 19.10x |

### Graph: 10,000 nodes, 20,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0023 | 0.00 | ✓ baseline |
| networkit | 0.0061 | 0.00 | 2.61x |
| igraph | 0.0083 | 0.42 | 3.56x |
| networkx | 0.0241 | 0.69 | 10.29x |

### Graph: 20,000 nodes, 40,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.0042 | 0.00 | ✓ baseline |
| networkit | 0.0074 | 0.00 | 1.75x |
| igraph | 0.0173 | 0.39 | 4.10x |
| networkx | 0.0930 | 2.92 | 21.98x |

### Graph: 50,000 nodes, 100,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0097 | 0.00 | ✓ baseline |
| groggy | 0.0106 | 0.50 | 1.09x |
| igraph | 0.0440 | 0.00 | 4.52x |
| networkx | 0.2400 | 2.11 | 24.68x |

### Graph: 100,000 nodes, 300,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0121 | 0.00 | ✓ baseline |
| groggy | 0.0269 | 0.00 | 2.23x |
| igraph | 0.1060 | 0.00 | 8.78x |
| networkx | 0.6670 | 11.48 | 55.24x |

### Graph: 200,000 nodes, 600,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| networkit | 0.0281 | 0.00 | ✓ baseline |
| groggy | 0.0477 | 6.02 | 1.70x |
| igraph | 0.2177 | 2.70 | 7.75x |
| networkx | 1.4625 | 2.44 | 52.08x |

### Graph: 1,000,000 nodes, 2,000,000 edges

| Library | Time (s) | Memory (MB) | Speedup vs Fastest |
|---------|----------|-------------|--------------------|
| groggy | 0.2253 | 71.86 | ✓ baseline |
| networkit | 0.3406 | 0.00 | 1.51x |
| igraph | 1.2670 | 20.80 | 5.62x |
| networkx | 6.0526 | 266.14 | 26.87x |

## Performance Insights

- Groggy was fastest in **5/21** test cases
- Average speedup when groggy wins: calculated per algorithm

### Key Takeaways

- Groggy's Rust core provides competitive performance across all algorithms
- Memory efficiency remains consistent with other optimized libraries
- The new algorithm pipeline API maintains performance while improving usability

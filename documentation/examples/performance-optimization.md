# Performance Optimization Guide

This guide demonstrates best practices for achieving optimal performance with Groggy, covering memory management, batch operations, and algorithm selection.

## Understanding Groggy's Performance Architecture

Groggy achieves high performance through several key design decisions:

1. **Rust Core**: All computation happens in native Rust for maximum speed
2. **Columnar Storage**: Memory-efficient attribute storage with cache locality
3. **Lazy Evaluation**: Operations computed on-demand and cached intelligently
4. **Batch Processing**: Vectorized operations for large datasets
5. **Smart Indexing**: Efficient lookup structures for fast queries

## 1. Graph Construction Optimization

### Batch Operations vs Individual Operations

```python
import groggy as gr
import time
import numpy as np

# Create large dataset
n_nodes = 10000
n_edges = 50000

nodes_data = [
    {'id': f'node_{i}', 'value': i, 'category': f'cat_{i % 10}', 'active': i % 3 == 0}
    for i in range(n_nodes)
]

edges_data = [
    {'source': f'node_{i}', 'target': f'node_{(i + 1) % n_nodes}', 'weight': np.random.random()}
    for i in range(n_edges)
]

# ❌ SLOW: Individual operations
def build_graph_individually():
    g = gr.Graph()
    start_time = time.time()
    
    # Add nodes one by one
    for node in nodes_data[:1000]:  # Smaller sample for timing
        g.add_node(**node)
    
    # Add edges one by one
    for edge in edges_data[:1000]:
        g.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    return time.time() - start_time

# ✅ FAST: Batch operations
def build_graph_batch():
    g = gr.Graph()
    start_time = time.time()
    
    # Batch add nodes
    g.add_nodes(nodes_data)
    
    # Batch add edges
    g.add_edges(edges_data)
    
    return time.time() - start_time

individual_time = build_graph_individually()
batch_time = build_graph_batch()

print(f"Individual operations: {individual_time:.3f}s")
print(f"Batch operations: {batch_time:.3f}s")
print(f"Speed improvement: {individual_time/batch_time:.1f}x faster")
```

### Memory-Efficient Large Graph Construction

```python
def create_large_graph_efficiently():
    """Create a large graph with minimal memory overhead"""
    g = gr.Graph()
    
    # Use generators for memory efficiency
    def node_generator(n):
        for i in range(n):
            yield {
                'id': f'node_{i}',
                'value': i,
                'category': f'cat_{i % 100}',  # Limit unique strings
                'score': i * 0.1
            }
    
    def edge_generator(n):
        for i in range(n):
            yield {
                'source': f'node_{i}',
                'target': f'node_{(i + 1) % 10000}',  # Reuse node IDs
                'weight': round(np.random.random(), 3)  # Limit precision
            }
    
    # Process in chunks to control memory usage
    chunk_size = 5000
    
    # Add nodes in chunks
    node_chunk = []
    for i, node in enumerate(node_generator(50000)):
        node_chunk.append(node)
        if len(node_chunk) >= chunk_size:
            g.add_nodes(node_chunk)
            node_chunk = []
    if node_chunk:
        g.add_nodes(node_chunk)
    
    # Add edges in chunks
    edge_chunk = []
    for i, edge in enumerate(edge_generator(100000)):
        edge_chunk.append(edge)
        if len(edge_chunk) >= chunk_size:
            g.add_edges(edge_chunk)
            edge_chunk = []
    if edge_chunk:
        g.add_edges(edge_chunk)
    
    return g

# Create large graph efficiently
print("Creating large graph...")
start_time = time.time()
large_graph = create_large_graph_efficiently()
creation_time = time.time() - start_time

print(f"Created graph with {large_graph.node_count():,} nodes and {large_graph.edge_count():,} edges")
print(f"Creation time: {creation_time:.3f}s")
print(f"Memory usage: ~{large_graph.memory_usage() / 1024**2:.1f} MB")
```

## 2. Query and Filtering Optimization

### Efficient Attribute Queries

```python
# Create test graph
g = gr.Graph()
test_nodes = [
    {'id': f'user_{i}', 'age': 20 + (i % 50), 'salary': 30000 + i * 100, 'department': f'dept_{i % 10}'}
    for i in range(10000)
]
g.add_nodes(test_nodes)

# ✅ FAST: Simple attribute queries
def benchmark_simple_queries():
    start_time = time.time()
    
    # Single attribute equality (uses indexed lookup)
    young_users = g.filter_nodes("age < 30")
    high_earners = g.filter_nodes("salary > 80000")
    engineering = g.filter_nodes("department == 'dept_1'")
    
    return time.time() - start_time

# ✅ FAST: Range queries (optimized for numeric data)
def benchmark_range_queries():
    start_time = time.time()
    
    # Numeric range queries are highly optimized
    mid_career = g.filter_nodes("age >= 30 AND age <= 45")
    salary_range = g.filter_nodes("salary >= 50000 AND salary <= 70000")
    
    return time.time() - start_time

# ⚠️ SLOWER: Complex string operations
def benchmark_complex_queries():
    start_time = time.time()
    
    # String operations are slower than numeric
    # Use sparingly or pre-process data
    dept_pattern = g.filter_nodes("department LIKE 'dept_%'")  # If supported
    
    return time.time() - start_time

simple_time = benchmark_simple_queries()
range_time = benchmark_range_queries()

print(f"Simple queries: {simple_time:.4f}s")
print(f"Range queries: {range_time:.4f}s")
print("Best practices:")
print("- Use numeric comparisons when possible")
print("- Prefer equality over pattern matching")
print("- Combine filters efficiently with AND/OR")
```

### Query Result Caching

```python
def demonstrate_caching():
    """Show how Groggy's caching improves repeated operations"""
    
    # First query - triggers computation
    start_time = time.time()
    young_users = g.filter_nodes("age < 30")
    young_table = young_users.table()
    first_query_time = time.time() - start_time
    
    # Second identical query - uses cache
    start_time = time.time()
    young_users_2 = g.filter_nodes("age < 30")
    young_table_2 = young_users_2.table()
    second_query_time = time.time() - start_time
    
    print(f"First query: {first_query_time:.4f}s")
    print(f"Second query (cached): {second_query_time:.4f}s")
    print(f"Cache speedup: {first_query_time/second_query_time:.1f}x")
    
    # Statistical operations also benefit from caching
    start_time = time.time()
    ages = young_table['age']
    first_stats = ages.describe()
    first_stats_time = time.time() - start_time
    
    start_time = time.time()
    second_stats = ages.describe()  # Cached
    second_stats_time = time.time() - start_time
    
    print(f"First stats calculation: {first_stats_time:.4f}s")
    print(f"Second stats (cached): {second_stats_time:.4f}s")

demonstrate_caching()
```

## 3. Storage View Performance

### Lazy Evaluation Benefits

```python
def demonstrate_lazy_evaluation():
    """Show performance benefits of lazy evaluation"""
    
    # Create large table
    large_table = large_graph.nodes.table()
    
    # ✅ FAST: Lazy operations don't compute until needed
    start_time = time.time()
    filtered_view = large_table.filter_rows(lambda row: row['score'] > 1000)
    sorted_view = filtered_view.sort_by('score', ascending=False)
    lazy_time = time.time() - start_time
    
    print(f"Lazy operations (no computation): {lazy_time:.6f}s")
    
    # Only when we access results does computation happen
    start_time = time.time()
    top_10 = sorted_view.head(10)  # Only computes what's needed
    materialization_time = time.time() - start_time
    
    print(f"Materialization (top 10 only): {materialization_time:.4f}s")
    
    # vs. eager computation of entire result
    start_time = time.time()
    all_filtered = [row for row in large_table.to_dict() if row['score'] > 1000]
    all_sorted = sorted(all_filtered, key=lambda x: x['score'], reverse=True)[:10]
    eager_time = time.time() - start_time
    
    print(f"Eager computation (entire dataset): {eager_time:.4f}s")
    print(f"Lazy speedup: {eager_time/materialization_time:.1f}x")

demonstrate_lazy_evaluation()
```

### Array vs Matrix vs Table Performance

```python
def benchmark_storage_views():
    """Compare performance of different storage views"""
    
    # Sample data
    sample_table = large_graph.nodes.table(attributes=['value', 'score'])
    
    # ✅ FASTEST: Array operations
    start_time = time.time()
    values_array = sample_table['value']
    array_mean = values_array.mean()
    array_std = values_array.std()
    array_time = time.time() - start_time
    
    # ✅ FAST: Matrix operations  
    start_time = time.time()
    matrix = sample_table[['value', 'score']]  # Returns GraphMatrix
    column_means = matrix.mean_axis(axis=0)
    matrix_time = time.time() - start_time
    
    # ⚠️ SLOWER: Full table operations
    start_time = time.time()
    full_stats = sample_table.describe()
    table_time = time.time() - start_time
    
    print(f"Array operations: {array_time:.4f}s")
    print(f"Matrix operations: {matrix_time:.4f}s") 
    print(f"Table operations: {table_time:.4f}s")
    print("\nRecommendations:")
    print("- Use Array for single-column statistics")
    print("- Use Matrix for multi-column numeric operations")
    print("- Use Table for complex analysis and exports")

benchmark_storage_views()
```

## 4. Algorithm Performance

### Graph Algorithm Optimization

```python
def optimize_graph_algorithms():
    """Demonstrate efficient use of graph algorithms"""
    
    # Create connected test graph
    test_graph = gr.Graph()
    
    # Create a connected graph structure
    nodes = [{'id': i, 'value': i} for i in range(1000)]
    test_graph.add_nodes(nodes)
    
    # Add edges to create connected components
    edges = []
    for i in range(0, 1000, 100):  # Create 10 components of 100 nodes each
        for j in range(i, min(i + 100, 1000) - 1):
            edges.append({'source': j, 'target': j + 1, 'weight': 1.0})
    
    test_graph.add_edges(edges)
    
    # ✅ FAST: Built-in algorithms
    start_time = time.time()
    components = test_graph.connected_components()
    components_time = time.time() - start_time
    
    # ✅ FAST: BFS/DFS from specific nodes
    start_time = time.time()
    visited = test_graph.bfs(start_node=0)
    bfs_time = time.time() - start_time
    
    # ✅ FAST: Shortest path
    start_time = time.time()
    try:
        path = test_graph.shortest_path(0, 99)
        path_time = time.time() - start_time
    except:
        path_time = 0
    
    print(f"Connected components: {components_time:.4f}s ({len(components)} components)")
    print(f"BFS traversal: {bfs_time:.4f}s ({len(visited)} nodes visited)")
    if path_time > 0:
        print(f"Shortest path: {path_time:.4f}s")
    
    # Analyze component sizes
    component_sizes = [len(comp.node_ids) for comp in components]
    sizes_array = gr.array(component_sizes)
    print(f"Component sizes - Mean: {sizes_array.mean():.1f}, Max: {sizes_array.max()}")

optimize_graph_algorithms()
```

### Centrality Calculation Performance

```python
def benchmark_centrality():
    """Benchmark centrality calculations"""
    
    # Create scale-free network for realistic centrality analysis
    scale_free_graph = gr.Graph()
    
    # Add nodes
    nodes = [{'id': i, 'importance': np.random.random()} for i in range(500)]
    scale_free_graph.add_nodes(nodes)
    
    # Add edges with preferential attachment
    edges = []
    for i in range(1, 500):
        # Connect to 1-3 existing nodes
        num_connections = min(3, i)
        targets = np.random.choice(i, num_connections, replace=False)
        for target in targets:
            edges.append({'source': i, 'target': target, 'weight': np.random.random()})
    
    scale_free_graph.add_edges(edges)
    
    # Benchmark different centrality measures
    start_time = time.time()
    betweenness = scale_free_graph.centrality.betweenness()
    betweenness_time = time.time() - start_time
    
    start_time = time.time()
    pagerank = scale_free_graph.centrality.pagerank()
    pagerank_time = time.time() - start_time
    
    start_time = time.time()
    closeness = scale_free_graph.centrality.closeness()
    closeness_time = time.time() - start_time
    
    print(f"Betweenness centrality: {betweenness_time:.4f}s")
    print(f"PageRank: {pagerank_time:.4f}s")
    print(f"Closeness centrality: {closeness_time:.4f}s")
    
    # Analyze centrality results
    betweenness_values = list(betweenness.values())
    pagerank_values = list(pagerank.values())
    
    betweenness_array = gr.array(betweenness_values)
    pagerank_array = gr.array(pagerank_values)
    
    print(f"Betweenness - Mean: {betweenness_array.mean():.4f}, Max: {betweenness_array.max():.4f}")
    print(f"PageRank - Mean: {pagerank_array.mean():.4f}, Max: {pagerank_array.max():.4f}")

benchmark_centrality()
```

## 5. Memory Optimization

### Memory Usage Monitoring

```python
def monitor_memory_usage():
    """Monitor and optimize memory usage"""
    
    print("=== Memory Usage Analysis ===")
    
    # Check graph memory usage
    graph_memory = large_graph.memory_usage()
    node_count = large_graph.node_count()
    edge_count = large_graph.edge_count()
    
    print(f"Graph memory: {graph_memory / 1024**2:.1f} MB")
    print(f"Memory per node: {graph_memory / node_count:.0f} bytes")
    print(f"Memory per edge: {graph_memory / edge_count:.0f} bytes")
    
    # Check table memory usage
    table = large_graph.nodes.table()
    table_memory = table.memory_usage()
    
    print(f"Table memory: {table_memory / 1024**2:.1f} MB")
    print(f"Memory efficiency: {graph_memory / table_memory:.2f}x (graph vs table)")
    
    # Check array memory usage
    values_array = table['value']
    array_memory = values_array.memory_usage()
    
    print(f"Array memory: {array_memory / 1024:.1f} KB")
    print(f"Array efficiency: {array_memory / len(values_array):.1f} bytes per element")

monitor_memory_usage()
```

### Memory-Efficient Operations

```python
def memory_efficient_patterns():
    """Demonstrate memory-efficient operation patterns"""
    
    # ✅ GOOD: Stream processing for large results
    def process_large_dataset_streaming():
        # Process in chunks instead of loading everything
        chunk_size = 1000
        total_processed = 0
        
        for i in range(0, large_graph.node_count(), chunk_size):
            # Get chunk of nodes
            chunk_table = large_graph.nodes.table(limit=chunk_size, offset=i)
            
            # Process chunk
            high_value = chunk_table.filter_rows(lambda row: row['value'] > 5000)
            
            # Aggregate results without storing intermediate data
            total_processed += len(high_value)
        
        return total_processed
    
    # ❌ AVOID: Loading entire large datasets into memory
    def process_large_dataset_naive():
        # This loads everything into memory at once
        all_nodes = large_graph.nodes.table()
        high_value_all = all_nodes.filter_rows(lambda row: row['value'] > 5000)
        return len(high_value_all)
    
    # Compare approaches
    start_time = time.time()
    streaming_result = process_large_dataset_streaming()
    streaming_time = time.time() - start_time
    
    print(f"Streaming approach: {streaming_result} results in {streaming_time:.3f}s")
    print("Memory usage: Low (processes in chunks)")
    
    # Note: Naive approach might be too memory-intensive for very large graphs

memory_efficient_patterns()
```

## 6. Best Practices Summary

### Performance Checklist

```python
def performance_best_practices():
    """Summary of performance best practices"""
    
    print("=== Groggy Performance Best Practices ===")
    print()
    
    print("1. GRAPH CONSTRUCTION:")
    print("   ✅ Use batch operations (add_nodes, add_edges)")
    print("   ✅ Process large datasets in chunks")
    print("   ✅ Limit string uniqueness to reduce memory")
    print("   ❌ Avoid individual add_node/add_edge in loops")
    print()
    
    print("2. QUERIES AND FILTERING:")
    print("   ✅ Use numeric comparisons when possible")
    print("   ✅ Leverage query result caching")
    print("   ✅ Combine filters with AND/OR efficiently")
    print("   ❌ Avoid complex string pattern matching")
    print()
    
    print("3. STORAGE VIEWS:")
    print("   ✅ Use GraphArray for single-column statistics")
    print("   ✅ Use GraphMatrix for multi-column numeric ops")
    print("   ✅ Use GraphTable for complex analysis")
    print("   ✅ Leverage lazy evaluation - only compute what you need")
    print()
    
    print("4. ALGORITHMS:")
    print("   ✅ Use built-in graph algorithms when available")
    print("   ✅ Consider algorithm complexity for large graphs")
    print("   ✅ Cache centrality results for repeated use")
    print()
    
    print("5. MEMORY MANAGEMENT:")
    print("   ✅ Monitor memory usage regularly")
    print("   ✅ Use streaming for very large datasets")
    print("   ✅ Clear unnecessary references")
    print("   ❌ Avoid loading entire large tables into memory")
    print()
    
    print("6. DATA TYPES:")
    print("   ✅ Use appropriate numeric types (int vs float)")
    print("   ✅ Limit precision where possible")
    print("   ✅ Consider sparse representations for sparse data")

performance_best_practices()
```

### Performance Measurement Template

```python
import time
import psutil
import os

def measure_performance(operation_name, operation_func):
    """Template for measuring operation performance"""
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024**2  # MB
    
    # Measure operation time
    start_time = time.time()
    result = operation_func()
    end_time = time.time()
    
    # Get final memory usage
    final_memory = process.memory_info().rss / 1024**2  # MB
    memory_delta = final_memory - initial_memory
    
    print(f"=== {operation_name} Performance ===")
    print(f"Execution time: {end_time - start_time:.4f}s")
    print(f"Memory change: {memory_delta:+.1f} MB")
    print(f"Final memory: {final_memory:.1f} MB")
    
    return result, end_time - start_time, memory_delta

# Example usage
def example_operation():
    g = gr.Graph()
    nodes = [{'id': i, 'value': i} for i in range(5000)]
    g.add_nodes(nodes)
    return g

result, exec_time, memory_change = measure_performance(
    "Create 5K node graph", 
    example_operation
)

print(f"Created graph with {result.node_count()} nodes")
```

This guide provides comprehensive strategies for optimizing Groggy performance. The key is to understand the underlying architecture and choose the right approach for your specific use case. Always measure performance for your specific workload and data characteristics.
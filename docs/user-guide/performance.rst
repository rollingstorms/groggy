Performance Optimization
=======================

This guide covers best practices for achieving optimal performance with Groggy, including memory management, algorithm selection, and scaling strategies.

Understanding Groggy's Performance Model
-----------------------------------------

Groggy achieves high performance through several key architectural decisions:

- **Rust Core**: All computation happens in native Rust
- **Columnar Storage**: Memory-efficient attribute storage with cache locality
- **Lazy Evaluation**: Operations computed on-demand and cached
- **Batch Processing**: Vectorized operations for large datasets
- **Smart Indexing**: Efficient lookup structures for fast queries

Graph Construction Optimization
-------------------------------

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr
   import time

   # ❌ SLOW: Individual operations
   def build_graph_slowly(n_nodes=1000):
       g = gr.Graph()
       start = time.time()
       
       for i in range(n_nodes):
           g.add_node(f'node_{i}', value=i, category=f'cat_{i % 10}')
           
       for i in range(n_nodes - 1):
           g.add_edge(f'node_{i}', f'node_{i+1}', weight=1.0)
           
       return time.time() - start

   # ✅ FAST: Batch operations
   def build_graph_quickly(n_nodes=1000):
       g = gr.Graph()
       start = time.time()
       
       # Prepare all node data
       nodes = [
           {'id': f'node_{i}', 'value': i, 'category': f'cat_{i % 10}'}
           for i in range(n_nodes)
       ]
       
       # Prepare all edge data
       edges = [
           {'source': f'node_{i}', 'target': f'node_{i+1}', 'weight': 1.0}
           for i in range(n_nodes - 1)
       ]
       
       # Batch operations
       g.add_nodes(nodes)
       g.add_edges(edges)
       
       return time.time() - start

   slow_time = build_graph_slowly(1000)
   fast_time = build_graph_quickly(1000)
   print(f"Batch operations are {slow_time/fast_time:.1f}x faster")

Memory-Efficient Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def create_large_graph_efficiently(n_nodes=50000):
       """Create large graph with minimal memory overhead"""
       g = gr.Graph()
       
       # Process in chunks to control memory
       chunk_size = 5000
       
       # Use generators to avoid storing all data in memory
       def node_generator(n):
           for i in range(n):
               yield {
                   'id': f'node_{i}',
                   'value': i,
                   'category': f'cat_{i % 100}',  # Limit unique strings
                   'score': round(i * 0.1, 2)     # Limit precision
               }
       
       # Add nodes in chunks
       node_chunk = []
       for node in node_generator(n_nodes):
           node_chunk.append(node)
           if len(node_chunk) >= chunk_size:
               g.add_nodes(node_chunk)
               node_chunk = []
       
       if node_chunk:  # Add remaining nodes
           g.add_nodes(node_chunk)
       
       return g

Query Optimization
------------------

Efficient Filtering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create test graph
   g = gr.Graph()
   nodes = [
       {'id': f'user_{i}', 'age': 20 + (i % 50), 'salary': 30000 + i * 100}
       for i in range(10000)
   ]
   g.add_nodes(nodes)

   # ✅ FAST: Simple numeric comparisons
   young_users = g.filter_nodes("age < 30")               # ~0.1ms
   high_earners = g.filter_nodes("salary > 80000")        # ~0.1ms
   
   # ✅ FAST: Range queries
   mid_career = g.filter_nodes("age >= 30 AND age <= 45") # ~0.2ms
   
   # ⚠️ SLOWER: Complex string operations
   # Use sparingly - prefer categorical data over string patterns
   
   # ✅ OPTIMAL: Pre-process string data into categories
   # Instead of: g.filter_nodes("department LIKE 'Eng%'")
   # Use: g.filter_nodes("dept_category == 'engineering'")

Query Result Caching
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Demonstrate automatic caching
   def show_caching_benefits():
       # First query - computes result
       start = time.time()
       young_users = g.filter_nodes("age < 30")
       first_time = time.time() - start
       
       # Second identical query - uses cache
       start = time.time()
       young_users_2 = g.filter_nodes("age < 30")
       cached_time = time.time() - start
       
       print(f"First query: {first_time*1000:.2f}ms")
       print(f"Cached query: {cached_time*1000:.2f}ms")
       print(f"Cache speedup: {first_time/cached_time:.1f}x")

Storage View Performance
------------------------

Choosing the Right View
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create sample data
   table = g.nodes.table()

   # ✅ FASTEST: Single-column operations with GraphArray
   def array_operations():
       ages = table['age']          # GraphArray
       return ages.mean(), ages.std()

   # ✅ FAST: Multi-column numeric operations with GraphMatrix  
   def matrix_operations():
       numeric_data = table[['age', 'salary']]  # GraphMatrix
       return numeric_data.mean_axis(axis=0)

   # ⚠️ SLOWER: Full table operations (use when needed)
   def table_operations():
       return table.describe()     # Full statistical analysis

   # Benchmark the approaches
   import timeit
   
   array_time = timeit.timeit(array_operations, number=100)
   matrix_time = timeit.timeit(matrix_operations, number=100)
   table_time = timeit.timeit(table_operations, number=10)
   
   print(f"Array ops: {array_time*10:.2f}ms")
   print(f"Matrix ops: {matrix_time*10:.2f}ms") 
   print(f"Table ops: {table_time*100:.2f}ms")

Lazy Evaluation Benefits
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def demonstrate_lazy_evaluation():
       large_table = g.nodes.table()
       
       # ✅ FAST: Lazy operations don't compute until needed
       start = time.time()
       filtered_view = large_table.filter_rows(lambda row: row['salary'] > 50000)
       sorted_view = filtered_view.sort_by('salary', ascending=False)
       lazy_time = time.time() - start
       
       print(f"Lazy operations: {lazy_time*1000:.3f}ms")
       
       # Computation happens only when accessing results
       start = time.time()
       top_10 = sorted_view.head(10)  # Only computes what's needed
       materialization_time = time.time() - start
       
       print(f"Materialization (top 10): {materialization_time*1000:.2f}ms")

Algorithm Performance
---------------------

Graph Algorithm Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create connected test graph
   def create_test_graph(n_nodes=1000):
       g = gr.Graph()
       
       nodes = [{'id': i, 'value': i} for i in range(n_nodes)]
       g.add_nodes(nodes)
       
       # Create connected structure
       edges = []
       for i in range(0, n_nodes, 100):  # 10 components of 100 nodes each
           for j in range(i, min(i + 100, n_nodes) - 1):
               edges.append({'source': j, 'target': j + 1, 'weight': 1.0})
       
       g.add_edges(edges)
       return g

   test_graph = create_test_graph(1000)

   # ✅ FAST: Built-in algorithms are optimized
   def benchmark_algorithms():
       start = time.time()
       components = test_graph.connected_components()
       comp_time = time.time() - start
       
       start = time.time()
       visited = test_graph.bfs(start_node=0)
       bfs_time = time.time() - start
       
       print(f"Connected components: {comp_time*1000:.2f}ms")
       print(f"BFS traversal: {bfs_time*1000:.2f}ms")

Centrality Performance
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def benchmark_centrality(graph_size='medium'):
       if graph_size == 'small':
           g = create_test_graph(500)
       elif graph_size == 'medium':
           g = create_test_graph(2000)
       else:  # large
           g = create_test_graph(5000)
       
       algorithms = {
           'betweenness': lambda: g.centrality.betweenness(),
           'pagerank': lambda: g.centrality.pagerank(),
           'closeness': lambda: g.centrality.closeness()
       }
       
       for name, algo in algorithms.items():
           start = time.time()
           result = algo()
           elapsed = time.time() - start
           print(f"{name.title()}: {elapsed*1000:.1f}ms ({len(result)} nodes)")

Memory Management
-----------------

Memory Usage Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   import os

   def monitor_memory():
       process = psutil.Process(os.getpid())
       memory_mb = process.memory_info().rss / 1024**2
       return memory_mb

   def analyze_memory_usage():
       print("=== Memory Usage Analysis ===")
       
       initial_memory = monitor_memory()
       
       # Create graph
       g = create_large_graph_efficiently(10000)
       after_graph = monitor_memory()
       
       # Create table
       table = g.nodes.table()
       after_table = monitor_memory()
       
       # Create arrays
       ages = table['age']
       values = table['value']
       after_arrays = monitor_memory()
       
       print(f"Initial memory: {initial_memory:.1f} MB")
       print(f"After graph creation: {after_graph:.1f} MB (+{after_graph-initial_memory:.1f})")
       print(f"After table creation: {after_table:.1f} MB (+{after_table-after_graph:.1f})")
       print(f"After array creation: {after_arrays:.1f} MB (+{after_arrays-after_table:.1f})")
       
       # Memory efficiency
       nodes_per_mb = g.node_count() / (after_graph - initial_memory)
       print(f"Memory efficiency: {nodes_per_mb:.0f} nodes per MB")

Memory-Efficient Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def memory_efficient_processing(large_graph):
       """Process large graphs without loading everything into memory"""
       
       # ✅ GOOD: Stream processing
       def process_in_chunks():
           chunk_size = 1000
           results = []
           
           for i in range(0, large_graph.node_count(), chunk_size):
               # Process small chunk
               chunk_nodes = list(large_graph.nodes)[i:i+chunk_size]
               chunk_subgraph = large_graph.subgraph(chunk_nodes)
               chunk_table = chunk_subgraph.table()
               
               # Compute statistics for chunk
               chunk_result = {
                   'count': len(chunk_table),
                   'avg_value': chunk_table['value'].mean()
               }
               results.append(chunk_result)
               
               # Explicitly delete to free memory
               del chunk_table, chunk_subgraph
           
           return results
       
       # ❌ AVOID: Loading entire large dataset
       def process_all_at_once():
           # This could use too much memory for very large graphs
           all_data = large_graph.table()
           return all_data.describe()
       
       return process_in_chunks()

Scaling Strategies
------------------

Large Graph Handling
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def handle_large_graphs(g):
       n_nodes = g.node_count()
       
       if n_nodes < 10000:
           # Small graph - use all algorithms
           return full_analysis(g)
           
       elif n_nodes < 100000:
           # Medium graph - selective algorithms
           return medium_analysis(g)
           
       else:
           # Large graph - approximation and sampling
           return large_analysis(g)

   def full_analysis(g):
       return {
           'centrality': g.centrality.betweenness(),
           'communities': g.communities.louvain(),
           'clustering': g.clustering()
       }

   def medium_analysis(g):
       return {
           'centrality': g.centrality.pagerank(),  # Faster than betweenness
           'communities': g.communities.louvain(),
           'basic_stats': {'density': g.density()}
       }

   def large_analysis(g):
       # Use sampling for very large graphs
       sample_size = min(10000, g.node_count() // 10)
       sample_nodes = g.nodes.sample(sample_size)
       sample_graph = g.subgraph(sample_nodes)
       
       return {
           'sample_centrality': sample_graph.centrality.pagerank(),
           'sample_communities': sample_graph.communities.louvain(),
           'basic_stats': {'density': g.density(), 'sample_size': sample_size}
       }

Parallel Processing Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   import numpy as np

   def parallel_node_analysis(g, node_ids, analysis_func):
       """Process nodes in parallel for CPU-intensive operations"""
       
       def process_chunk(chunk_nodes):
           results = {}
           for node_id in chunk_nodes:
               results[node_id] = analysis_func(g, node_id)
           return results
       
       # Split nodes into chunks
       chunk_size = len(node_ids) // 4  # Use 4 threads
       chunks = [node_ids[i:i+chunk_size] for i in range(0, len(node_ids), chunk_size)]
       
       # Process chunks in parallel
       all_results = {}
       with ThreadPoolExecutor(max_workers=4) as executor:
           futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
           for future in futures:
               all_results.update(future.result())
       
       return all_results

Performance Benchmarking
------------------------

Benchmarking Framework
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def benchmark_operation(operation_name, operation_func, iterations=10):
       """Generic benchmarking function"""
       times = []
       
       for _ in range(iterations):
           start = time.time()
           result = operation_func()
           elapsed = time.time() - start
           times.append(elapsed)
       
       avg_time = np.mean(times)
       std_time = np.std(times)
       
       print(f"{operation_name}:")
       print(f"  Average: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
       print(f"  Min: {min(times)*1000:.2f}ms, Max: {max(times)*1000:.2f}ms")
       
       return avg_time

   # Example usage
   test_graph = create_test_graph(1000)
   
   benchmark_operation("BFS Traversal", lambda: test_graph.bfs(start_node=0))
   benchmark_operation("PageRank", lambda: test_graph.centrality.pagerank())
   benchmark_operation("Connected Components", lambda: test_graph.connected_components())

Performance Testing Suite
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def run_performance_suite():
       """Comprehensive performance testing"""
       
       graph_sizes = [100, 500, 1000, 5000]
       operations = {
           'graph_creation': lambda n: create_test_graph(n),
           'table_creation': lambda g: g.nodes.table(),
           'filtering': lambda g: g.filter_nodes("value > 100"),
           'pagerank': lambda g: g.centrality.pagerank()
       }
       
       results = {}
       
       for size in graph_sizes:
           print(f"\n=== Testing with {size} nodes ===")
           g = create_test_graph(size)
           
           results[size] = {}
           
           for op_name, op_func in operations.items():
               if op_name == 'graph_creation':
                   time_taken = benchmark_operation(op_name, lambda: create_test_graph(size), iterations=5)
               else:
                   time_taken = benchmark_operation(op_name, lambda: op_func(g), iterations=5)
               
               results[size][op_name] = time_taken
       
       # Analyze scaling
       print("\n=== Scaling Analysis ===")
       for op_name in operations.keys():
           print(f"\n{op_name.title()} scaling:")
           for size in graph_sizes:
               time_ms = results[size][op_name] * 1000
               print(f"  {size:4d} nodes: {time_ms:7.2f}ms")

Best Practices Summary
---------------------

Graph Construction
~~~~~~~~~~~~~~~~~~

1. **Use batch operations** for adding nodes and edges
2. **Process large datasets in chunks** to control memory
3. **Limit string uniqueness** to reduce memory overhead
4. **Use appropriate numeric types** (int vs float)

Query Optimization
~~~~~~~~~~~~~~~~~~

1. **Use numeric comparisons** when possible
2. **Leverage automatic caching** for repeated queries
3. **Combine filters efficiently** with AND/OR operations
4. **Pre-process categorical data** instead of string patterns

Storage Views
~~~~~~~~~~~~~

1. **Choose the right view** for your use case:
   - GraphArray for single-column statistics
   - GraphMatrix for multi-column numeric operations  
   - GraphTable for complex relational analysis
2. **Leverage lazy evaluation** - chain operations without materialization
3. **Use column-specific operations** when possible

Algorithm Selection
~~~~~~~~~~~~~~~~~~~

1. **Start with fast algorithms** (PageRank vs Betweenness)
2. **Use approximation** for very large graphs
3. **Sample large graphs** for exploratory analysis
4. **Cache expensive results** for repeated use

Memory Management
~~~~~~~~~~~~~~~~~

1. **Monitor memory usage** during development
2. **Process in chunks** for very large datasets
3. **Use views instead of copies** when possible
4. **Explicitly delete** large intermediate results

This performance guide provides the foundation for building scalable graph analysis workflows. The key is to understand your data characteristics and choose the right combination of techniques for your specific use case.
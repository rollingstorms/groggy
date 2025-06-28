Performance Guide
=================

Groggy is designed for high performance using a Rust backend with Python interface. This guide covers optimization strategies and performance characteristics.

Performance Overview
--------------------

Groggy uses a high-performance Rust backend by default, providing excellent performance for large-scale graph operations:

Benchmark Results
~~~~~~~~~~~~~~~~~

.. list-table:: Groggy Performance Benchmarks
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation
     - Dataset Size
     - Time
     - Memory Usage
   * - Graph Creation
     - 10K nodes + 10K edges
     - 0.12 seconds
     - 25 MB
   * - Graph Creation
     - 100K nodes + 100K edges
     - 1.04 seconds
     - 85 MB
   * - Batch Attribute Update
     - 100 nodes
     - 0.01 seconds
     - Minimal
   * - Node Filtering
     - 10K nodes
     - 0.033 seconds
     - Minimal
   * - State Save/Load
     - 10K nodes graph
     - 0.1-0.2 seconds
     - Efficient
   * - Branch Switching
     - 100K+ nodes
     - 0.1-0.2 seconds
     - ~85 MB/million nodes

Key Performance Features
~~~~~~~~~~~~~~~~~~~~~~~~

**High-Performance Rust Backend:**
- 10-100x faster than pure Python implementations
- Memory efficient: ~85MB per million nodes
- Optimized for large-scale operations

**Batch Operations:**
- `set_nodes_attributes_batch()` and `set_edges_attributes_batch()`
- Significantly faster than individual attribute updates
- Ideal for bulk data processing

**Smart Caching:**
- Lazy-loaded node and edge collections
- Automatic cache invalidation
- Minimal memory overhead

**Efficient State Management:**
- Content-addressed storage with deduplication
- Fast branch switching (0.1-0.2s for large graphs)
- Memory-efficient state persistence

**Use Python Backend When:**
- Rapid prototyping
- Small graphs (<1,000 nodes)
- Debugging graph algorithms
- Rust backend unavailable

.. code-block:: python

   import groggy as gr
   
   # Check available backends
   print("Available backends:", gr.get_available_backends())
   
   # Set backend globally
   gr.set_backend('rust')  # Use Rust for all new graphs
   
   # Or specify per-graph
   fast_graph = Graph(backend='rust')
   debug_graph = Graph(backend='python')

Optimization Strategies
-----------------------

Batch Operations
~~~~~~~~~~~~~~~

Groggy provides high-performance batch operations for bulk processing:

.. list-table:: Batch Operations Performance
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation
     - Individual (10K ops)
     - Batch (10K ops)
     - Speedup
   * - Node Filtering
     - 890ms
     - 24ms
     - 37x faster
   * - Edge Filtering
     - 1,240ms
     - 31ms
     - 40x faster
   * - Attribute Retrieval
     - 234ms
     - 6ms
     - 39x faster
   * - Attribute Updates
     - 187ms
     - 12ms
     - 16x faster

Batch Filtering Examples
^^^^^^^^^^^^^^^^^^^^^^^^

Use batch filtering for efficient queries:

.. code-block:: python

   import groggy as gr
   
   g = gr.Graph(backend='rust')
   
   # Add sample data
   for i in range(10000):
       g.add_node(f"person_{i}", 
                  age=random.randint(18, 65),
                  city=random.choice(['NYC', 'LA', 'Chicago']),
                  occupation=random.choice(['Engineer', 'Teacher', 'Doctor']))
   
   # Efficient batch filtering
   engineers = g.batch_filter_nodes(occupation='Engineer')
   ny_residents = g.batch_filter_nodes(city='NYC')
   
   # Combined filtering
   young_engineers = g.batch_filter_nodes(occupation='Engineer')
   young_engineer_data = g.batch_get_node_attributes(young_engineers)
   filtered = [
       young_engineers[i] for i, attrs in enumerate(young_engineer_data)
       if attrs['age'] < 30
   ]

Bulk Attribute Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

Efficiently update multiple nodes:

.. code-block:: python

   # Get attributes for many nodes at once
   target_nodes = g.batch_filter_nodes(department='engineering')
   all_attrs = g.batch_get_node_attributes(target_nodes)
   
   # Bulk attribute updates
   updates = {
       node_id: {'status': 'active', 'last_update': datetime.now().isoformat()}
       for node_id in target_nodes
   }
   g.batch_set_node_attributes(updates)
   
   # Functional updates
   g.batch_update_node_attributes({
       node_id: lambda attrs: {**attrs, 'experience': attrs.get('experience', 0) + 1}
       for node_id in senior_employees
   })

Context Manager for Bulk Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use batch operations for multiple insertions:

.. code-block:: python

   import time
   from groggy import Graph
   
   g = Graph(backend='rust')
   
   # Inefficient: individual operations
   start = time.time()
   for i in range(10000):
       g.add_node(f"node_{i}", value=i)
   slow_time = time.time() - start
   
   # Efficient: batch operations  
   g2 = Graph(backend='rust')
   start = time.time()
   with g2.batch_operations() as batch:
       for i in range(10000):
           batch.add_node(f"node_{i}", value=i)
   fast_time = time.time() - start
   
   print(f"Individual: {slow_time:.2f}s, Batch: {fast_time:.2f}s")
   print(f"Speedup: {slow_time/fast_time:.1f}x")

Performance Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~
   print(f"Individual ops: {slow_time:.3f}s")
   print(f"Batch ops: {fast_time:.3f}s")
   print(f"Speedup: {slow_time/fast_time:.1f}x")

Memory Management
~~~~~~~~~~~~~~~~

Optimize memory usage for large graphs:

.. code-block:: python

   from groggy import Graph
   import gc
   
   # Use Rust backend for better memory efficiency
   g = Graph(backend='rust')
   
   # For very large graphs, consider periodic cleanup
   for i in range(1000000):
       g.add_node(f"node_{i}", data={"value": i})
       
       # Periodic cleanup for Python objects
       if i % 100000 == 0:
           gc.collect()
           print(f"Added {i} nodes")

Efficient Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build graphs efficiently from different data sources:

.. code-block:: python

   from groggy import Graph
   
   # From edge list (most efficient)
   edges = [("a", "b"), ("b", "c"), ("c", "d")]
   g1 = Graph.from_edge_list(edges, backend='rust')
   
   # From pandas DataFrame (if available)
   try:
       import pandas as pd
       
       # Create sample data
       df = pd.DataFrame({
           'source': ['a', 'b', 'c'],
           'target': ['b', 'c', 'd'],
           'weight': [1.0, 2.0, 1.5]
       })
       
       # Efficient conversion
       edge_list = df[['source', 'target']].values.tolist()
       edge_attrs = {'weight': df['weight'].tolist()}
       
       g2 = Graph.from_edge_list(edge_list, edge_attrs=edge_attrs)
       
   except ImportError:
       print("pandas not available")

Benchmarking Tools
------------------

Groggy includes benchmarking utilities:

.. code-block:: python

   from groggy.utils import create_random_graph
   import time
   import memory_profiler
   
   def benchmark_graph_operations():
       # Test different graph sizes
       sizes = [100, 1000, 10000]
       
       for size in sizes:
           print(f"\\nBenchmarking {size} nodes:")
           
           # Creation time
           start = time.time()
           g = create_random_graph(size, 0.01, use_rust=True)
           creation_time = time.time() - start
           
           # Query time (1000 random queries)
           start = time.time()
           nodes = list(g.nodes)
           for _ in range(1000):
               random_node = nodes[len(nodes) // 2]
               neighbors = g.get_neighbors(random_node)
           query_time = time.time() - start
           
           print(f"  Creation: {creation_time:.3f}s")
           print(f"  1000 queries: {query_time:.3f}s") 
           print(f"  Nodes: {g.node_count()}, Edges: {g.edge_count()}")
   
   # Run benchmark
   benchmark_graph_operations()

Memory Profiling
~~~~~~~~~~~~~~~

Profile memory usage:

.. code-block:: python

   import tracemalloc
   from groggy import Graph
   
   # Start memory tracking
   tracemalloc.start()
   
   # Create large graph
   g = Graph(backend='rust')
   with g.batch_operations() as batch:
       for i in range(100000):
           batch.add_node(f"node_{i}", data={"value": i, "category": i % 10})
   
   # Get memory usage
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
   print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
   
   tracemalloc.stop()

Performance Best Practices
--------------------------

Graph Design
~~~~~~~~~~~~

1. **Minimize Attribute Complexity**: Use simple types when possible
2. **Batch Operations**: Group multiple operations together
3. **Choose Appropriate Data Types**: Use integers for IDs when possible
4. **Avoid Frequent Schema Changes**: Design attributes upfront

.. code-block:: python

   # Good: Simple, consistent attributes
   g.add_node("user1", age=25, active=True, score=100.5)
   
   # Avoid: Complex nested structures for frequently accessed data
   g.add_node("user2", metadata={
       "profile": {"personal": {"age": 25}},
       "settings": {"preferences": {"theme": "dark"}}
   })

Query Optimization
~~~~~~~~~~~~~~~~~

1. **Cache Frequent Queries**: Store results of expensive computations
2. **Use Batch Queries**: Query multiple items at once
3. **Minimize Graph Traversals**: Plan query patterns

.. code-block:: python

   # Cache expensive computations
   neighbor_cache = {}
   
   def get_cached_neighbors(node_id):
       if node_id not in neighbor_cache:
           neighbor_cache[node_id] = g.get_neighbors(node_id)
       return neighbor_cache[node_id]
   
   # Batch queries when possible
   nodes_to_query = ["node1", "node2", "node3"]
   node_data = [g.get_node(node_id) for node_id in nodes_to_query]

Large Graph Strategies
---------------------

For graphs with millions of nodes:

Streaming Operations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def process_large_graph_streaming(filename):
       g = Graph(backend='rust')
       
       # Process in chunks to manage memory
       chunk_size = 10000
       current_chunk = []
       
       with open(filename, 'r') as f:
           for line in f:
               # Parse edge from line
               source, target = line.strip().split(',')
               current_chunk.append((source, target))
               
               # Process chunk when full
               if len(current_chunk) >= chunk_size:
                   with g.batch_operations() as batch:
                       for s, t in current_chunk:
                           batch.add_edge(s, t)
                   current_chunk = []
                   
           # Process remaining items
           if current_chunk:
               with g.batch_operations() as batch:
                   for s, t in current_chunk:
                       batch.add_edge(s, t)
       
       return g

Distributed Processing
~~~~~~~~~~~~~~~~~~~~~

For extremely large graphs, consider distributed approaches:

.. code-block:: python

   def create_subgraphs_for_processing(large_graph, num_partitions=4):
       \"\"\"Split large graph into smaller subgraphs for parallel processing\"\"\"
       
       nodes = list(large_graph.nodes)
       partition_size = len(nodes) // num_partitions
       
       subgraphs = []
       for i in range(num_partitions):
           start_idx = i * partition_size
           end_idx = start_idx + partition_size if i < num_partitions - 1 else len(nodes)
           
           partition_nodes = nodes[start_idx:end_idx]
           subgraph = Graph(backend='rust')
           
           # Add nodes from partition
           with subgraph.batch_operations() as batch:
               for node_id in partition_nodes:
                   node_data = large_graph.get_node(node_id)
                   batch.add_node(node_id, **node_data.attributes)
           
           subgraphs.append(subgraph)
       
       return subgraphs

Performance Monitoring
---------------------

Track performance in production:

.. code-block:: python

   import time
   from collections import defaultdict
   
   class PerformanceMonitor:
       def __init__(self):
           self.operation_times = defaultdict(list)
           self.operation_counts = defaultdict(int)
       
       def time_operation(self, operation_name):
           return self.OperationTimer(self, operation_name)
       
       class OperationTimer:
           def __init__(self, monitor, operation_name):
               self.monitor = monitor
               self.operation_name = operation_name
               self.start_time = None
           
           def __enter__(self):
               self.start_time = time.time()
               return self
           
           def __exit__(self, exc_type, exc_val, exc_tb):
               duration = time.time() - self.start_time
               self.monitor.operation_times[self.operation_name].append(duration)
               self.monitor.operation_counts[self.operation_name] += 1
       
       def get_stats(self):
           stats = {}
           for op_name, times in self.operation_times.items():
               stats[op_name] = {
                   'count': len(times),
                   'total_time': sum(times),
                   'avg_time': sum(times) / len(times),
                   'min_time': min(times),
                   'max_time': max(times)
               }
           return stats
   
   # Usage
   monitor = PerformanceMonitor()
   g = Graph(backend='rust')
   
   # Monitor operations
   with monitor.time_operation('node_addition'):
       for i in range(1000):
           g.add_node(f"node_{i}")
   
   with monitor.time_operation('neighbor_queries'):
       for i in range(100):
           g.get_neighbors(f"node_{i}")
   
   # Get performance stats
   stats = monitor.get_stats()
   for op_name, op_stats in stats.items():
       print(f"{op_name}: {op_stats['avg_time']*1000:.2f}ms avg")

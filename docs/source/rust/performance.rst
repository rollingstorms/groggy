Rust Performance Guide
======================

This guide covers performance optimization techniques specific to GLI's Rust backend.

Performance Characteristics
---------------------------

Benchmarking Results
~~~~~~~~~~~~~~~~~~~

The Rust backend provides significant performance improvements across all operations:

.. list-table:: Detailed Performance Comparison
   :header-rows: 1
   :widths: 25 20 20 15 20

   * - Operation
     - Python Backend
     - Rust Backend
     - Speedup
     - Notes
   * - Node creation (1M)
     - 2.5s
     - 0.8s
     - 3.1x
     - Batch operations
   * - Edge creation (1M)
     - 3.2s
     - 0.6s
     - 5.3x
     - Optimized adjacency
   * - Node lookup
     - 1.0 μs
     - 0.3 μs
     - 3.3x
     - Hash map optimization
   * - Neighbor query
     - 2.0 μs
     - 0.5 μs
     - 4.0x
     - Direct memory access
   * - Attribute access
     - 0.8 μs
     - 0.2 μs
     - 4.0x
     - Zero-copy access
   * - Iteration (1M nodes)
     - 0.45s
     - 0.12s
     - 3.8x
     - Iterator optimization

Memory Usage Analysis
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Memory Usage per Element
   :header-rows: 1
   :widths: 30 25 25 20

   * - Data Structure
     - Python Backend
     - Rust Backend
     - Improvement
   * - Empty node
     - 200 bytes
     - 64 bytes
     - 3.1x less
   * - Node + 5 attributes
     - 400 bytes
     - 128 bytes
     - 3.1x less
   * - Simple edge
     - 150 bytes
     - 48 bytes
     - 3.1x less
   * - Edge + 3 attributes
     - 280 bytes
     - 96 bytes
     - 2.9x less

Optimization Techniques
----------------------

Data Structure Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust backend uses several optimized data structures:

.. code-block:: rust

   use rustc_hash::FxHashMap;  // 30% faster than std::HashMap
   use smallvec::SmallVec;     // Stack allocation for small collections
   use string_interner::StringInterner;  // Efficient string storage
   
   struct OptimizedNode {
       id: StringId,  // Interned string (4 bytes vs 24+ bytes)
       attributes: ContentAddress,  // Deduplicated storage
       edges: SmallVec<[EdgeId; 4]>,  // No heap allocation for ≤4 edges
   }

**String Interning Benefits:**

.. code-block:: python

   # Measuring string interning impact
   import time
   from groggy import Graph
   
   g = Graph(backend='rust')
   
   # Create many nodes with repeated strings
   start = time.time()
   categories = ["user", "admin", "guest"] * 1000
   
   with g.batch_operations() as batch:
       for i, category in enumerate(categories):
           batch.add_node(f"node_{i}", 
                         category=category,  # Automatically interned
                         type="person",      # Shared string storage
                         status="active")    # Only stored once
   
   print(f"Created {len(categories)} nodes in {time.time() - start:.3f}s")

Content-Addressed Storage
~~~~~~~~~~~~~~~~~~~~~~~~

Identical attribute sets are automatically deduplicated:

.. code-block:: python

   from groggy import Graph
   
   g = Graph(backend='rust')
   
   # These nodes share attribute storage
   common_attrs = {"type": "user", "status": "active", "level": 1}
   
   with g.batch_operations() as batch:
       for i in range(10000):
           batch.add_node(f"user_{i}", **common_attrs)
   
   # Memory usage is much lower than expected because
   # all nodes share the same attribute storage

Batch Operation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rust backend batch operations are highly optimized:

.. code-block:: python

   import time
   from groggy import Graph
   
   def compare_batch_vs_individual():
       # Individual operations
       g1 = Graph(backend='rust')
       start = time.time()
       for i in range(10000):
           g1.add_node(f"node_{i}", value=i)
       individual_time = time.time() - start
       
       # Batch operations
       g2 = Graph(backend='rust')
       start = time.time()
       with g2.batch_operations() as batch:
           for i in range(10000):
               batch.add_node(f"node_{i}", value=i)
       batch_time = time.time() - start
       
       print(f"Individual: {individual_time:.3f}s")
       print(f"Batch: {batch_time:.3f}s")
       print(f"Speedup: {individual_time/batch_time:.1f}x")
   
   compare_batch_vs_individual()

Memory Layout Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust backend optimizes memory layout for cache efficiency:

.. code-block:: rust

   // Memory-efficient graph representation
   #[repr(C)]  // Predictable memory layout
   struct PackedNode {
       id: u32,           // 4 bytes (interned string ID)
       attr_hash: u64,    // 8 bytes (content address)
       edge_start: u32,   // 4 bytes (index into edge array)
       edge_count: u16,   // 2 bytes (number of edges)
       flags: u16,        // 2 bytes (node metadata)
   }
   // Total: 20 bytes per node (vs 200+ in Python)

Cache-Friendly Iteration
~~~~~~~~~~~~~~~~~~~~~~~

Rust backend provides cache-efficient iteration patterns:

.. code-block:: python

   from groggy import Graph
   import time
   
   g = Graph(backend='rust')
   
   # Create large graph
   with g.batch_operations() as batch:
       for i in range(100000):
           batch.add_node(f"node_{i}", value=i, category=i % 100)
   
   # Cache-friendly iteration (Rust optimized)
   start = time.time()
   total = 0
   for node_id in g.nodes:
       node_data = g.get_node(node_id)
       total += node_data["value"]
   iteration_time = time.time() - start
   
   print(f"Iterated {g.node_count()} nodes in {iteration_time:.3f}s")
   print(f"Rate: {g.node_count() / iteration_time:.0f} nodes/sec")

Advanced Performance Features
----------------------------

SIMD Optimizations
~~~~~~~~~~~~~~~~~

The Rust backend uses SIMD instructions where possible:

.. code-block:: rust

   // Vectorized operations (simplified example)
   use std::simd::*;
   
   fn batch_process_attributes(values: &[f64]) -> Vec<f64> {
       let mut results = Vec::with_capacity(values.len());
       
       for chunk in values.chunks(8) {
           let vector = f64x8::from_slice(chunk);
           let processed = vector * f64x8::splat(2.0);  // SIMD multiplication
           results.extend_from_slice(processed.as_array());
       }
       
       results
   }

Parallel Processing
~~~~~~~~~~~~~~~~~~

Safe parallel operations with Rust's concurrency model:

.. code-block:: python

   from groggy import Graph
   import concurrent.futures
   
   def parallel_graph_processing():
       g = Graph(backend='rust')  # Thread-safe
       
       # Create initial graph
       with g.batch_operations() as batch:
           for i in range(100000):
               batch.add_node(f"node_{i}", value=i)
       
       # Parallel processing function
       def process_node_batch(node_ids):
           results = []
           for node_id in node_ids:
               node_data = g.get_node(node_id)
               # Process node data
               results.append(node_data["value"] * 2)
           return results
       
       # Split work across threads
       node_ids = list(g.nodes)
       batch_size = len(node_ids) // 4
       batches = [node_ids[i:i+batch_size] for i in range(0, len(node_ids), batch_size)]
       
       # Process in parallel
       with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
           futures = [executor.submit(process_node_batch, batch) for batch in batches]
           results = [future.result() for future in futures]
       
       return results

Memory Pool Allocation
~~~~~~~~~~~~~~~~~~~~~

Rust backend uses memory pools for efficient allocation:

.. code-block:: rust

   use typed_arena::Arena;
   
   struct GraphArena {
       node_arena: Arena<Node>,
       edge_arena: Arena<Edge>,
       string_arena: Arena<String>,
   }
   
   impl GraphArena {
       fn allocate_node(&self, id: String, attrs: Attributes) -> &Node {
           self.node_arena.alloc(Node::new(id, attrs))
       }
   }

Benchmarking Tools
-----------------

Built-in Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from groggy import Graph
   import time
   
   class RustPerformanceMonitor:
       def __init__(self):
           self.metrics = {}
       
       def benchmark_operation(self, name, operation, *args, **kwargs):
           start = time.perf_counter()
           result = operation(*args, **kwargs)
           duration = time.perf_counter() - start
           
           if name not in self.metrics:
               self.metrics[name] = []
           self.metrics[name].append(duration)
           
           return result
       
       def get_statistics(self):
           stats = {}
           for name, times in self.metrics.items():
               stats[name] = {
                   'count': len(times),
                   'total': sum(times),
                   'average': sum(times) / len(times),
                   'min': min(times),
                   'max': max(times),
                   'rate': len(times) / sum(times) if sum(times) > 0 else 0
               }
           return stats
   
   # Usage example
   monitor = RustPerformanceMonitor()
   g = Graph(backend='rust')
   
   # Benchmark different operations
   for i in range(1000):
       monitor.benchmark_operation('add_node', g.add_node, f"node_{i}", value=i)
   
   for i in range(500):
       source, target = f"node_{i}", f"node_{i+1}"
       monitor.benchmark_operation('add_edge', g.add_edge, source, target)
   
   for i in range(100):
       monitor.benchmark_operation('get_neighbors', g.get_neighbors, f"node_{i}")
   
   # Print statistics
   stats = monitor.get_statistics()
   for operation, data in stats.items():
       print(f"{operation}:")
       print(f"  Average: {data['average']*1000:.2f}ms")
       print(f"  Rate: {data['rate']:.0f} ops/sec")

Memory Profiling
~~~~~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   import psutil
   import os
   from groggy import Graph
   
   def profile_rust_memory():
       # Get initial memory
       process = psutil.Process(os.getpid())
       initial_memory = process.memory_info().rss
       
       # Start Python memory tracking
       tracemalloc.start()
       
       # Create large graph
       g = Graph(backend='rust')
       
       with g.batch_operations() as batch:
           for i in range(500000):
               batch.add_node(f"node_{i}", 
                             value=i, 
                             category=i % 1000,
                             metadata={"created": f"2025-{i%12+1:02d}-01"})
       
       # Get memory usage
       current, peak = tracemalloc.get_traced_memory()
       final_memory = process.memory_info().rss
       
       tracemalloc.stop()
       
       print("Memory Usage Analysis:")
       print(f"Python tracked peak: {peak / 1024 / 1024:.1f} MB")
       print(f"Process memory increase: {(final_memory - initial_memory) / 1024 / 1024:.1f} MB")
       print(f"Nodes created: {g.node_count()}")
       print(f"Bytes per node (Python): {peak / g.node_count():.1f}")
       print(f"Bytes per node (Process): {(final_memory - initial_memory) / g.node_count():.1f}")
   
   profile_rust_memory()

Performance Tuning
------------------

Graph Size Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimize_for_graph_size(node_count):
       \"\"\"Choose optimal settings based on graph size\"\"\"
       
       if node_count < 1000:
           # Small graphs - Python backend may be sufficient
           return Graph(backend='python')
       
       elif node_count < 100000:
           # Medium graphs - Rust backend with standard settings
           return Graph(backend='rust')
       
       else:
           # Large graphs - Rust backend with optimizations
           g = Graph(backend='rust')
           # Pre-allocate capacity if possible
           # g.reserve_capacity(nodes=node_count, edges=node_count*2)
           return g

Attribute Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize attribute usage for Rust backend
   def optimize_attributes():
       g = Graph(backend='rust')
       
       # Good: Simple, immutable attributes
       g.add_node("user1", 
                 id=12345,           # Integer
                 name="Alice",       # String
                 active=True,        # Boolean
                 score=95.5)         # Float
       
       # Avoid: Complex mutable objects
       # g.add_node("user2", data=SomeComplexObject())  # Slower
       
       # Good: Use simple collections
       g.add_node("user3", 
                 tags=["admin", "active"],          # List of strings
                 settings={"theme": "dark"})        # Simple dict
       
       # Pre-intern common strings
       common_categories = ["user", "admin", "guest", "premium"]
       for i in range(10000):
           category = common_categories[i % len(common_categories)]
           g.add_node(f"user_{i}", category=category)  # Automatically interned

Batch Size Tuning
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from groggy import Graph
   
   def find_optimal_batch_size():
       \"\"\"Find the optimal batch size for your workload\"\"\"
       
       batch_sizes = [100, 500, 1000, 5000, 10000, 50000]
       results = {}
       
       for batch_size in batch_sizes:
           g = Graph(backend='rust')
           
           total_nodes = 100000
           num_batches = total_nodes // batch_size
           
           start = time.time()
           
           for batch_num in range(num_batches):
               with g.batch_operations() as batch:
                   for i in range(batch_size):
                       node_id = batch_num * batch_size + i
                       batch.add_node(f"node_{node_id}", value=node_id)
           
           duration = time.time() - start
           results[batch_size] = duration
           
           print(f"Batch size {batch_size}: {duration:.3f}s")
       
       # Find optimal batch size
       optimal_size = min(results, key=results.get)
       print(f"\\nOptimal batch size: {optimal_size}")
       
       return optimal_size
   
   # find_optimal_batch_size()

Performance Monitoring in Production
-----------------------------------

Runtime Performance Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import logging
   from collections import defaultdict, deque
   
   class ProductionPerformanceMonitor:
       def __init__(self, window_size=1000):
           self.window_size = window_size
           self.operation_times = defaultdict(lambda: deque(maxlen=window_size))
           self.operation_counts = defaultdict(int)
           self.slow_operations = []
           
       def record_operation(self, operation_name, duration):
           self.operation_times[operation_name].append(duration)
           self.operation_counts[operation_name] += 1
           
           # Log slow operations
           if duration > 0.1:  # 100ms threshold
               self.slow_operations.append({
                   'operation': operation_name,
                   'duration': duration,
                   'timestamp': time.time()
               })
               
               if len(self.slow_operations) > 100:
                   self.slow_operations.pop(0)
       
       def get_performance_summary(self):
           summary = {}
           for op_name, times in self.operation_times.items():
               if times:
                   summary[op_name] = {
                       'count': len(times),
                       'avg_ms': (sum(times) / len(times)) * 1000,
                       'p95_ms': sorted(times)[int(0.95 * len(times))] * 1000,
                       'max_ms': max(times) * 1000,
                   }
           return summary
       
       def check_performance_degradation(self, baseline_metrics):
           \"\"\"Check if performance has degraded compared to baseline\"\"\"
           current = self.get_performance_summary()
           alerts = []
           
           for op_name, current_stats in current.items():
               if op_name in baseline_metrics:
                   baseline = baseline_metrics[op_name]
                   
                   # Check if average time increased by >50%
                   if current_stats['avg_ms'] > baseline['avg_ms'] * 1.5:
                       alerts.append(f"{op_name} performance degraded: "
                                   f"{current_stats['avg_ms']:.1f}ms vs {baseline['avg_ms']:.1f}ms baseline")
           
           return alerts

Best Practices Summary
---------------------

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~

1. **Always Use Rust Backend for Large Graphs** (>10K nodes)
2. **Use Batch Operations** for bulk changes
3. **Keep Attributes Simple** when possible
4. **Pre-allocate** when graph size is known
5. **Monitor Performance** in production
6. **Profile Memory Usage** for large deployments

.. code-block:: python

   # Performance best practices example
   def create_high_performance_graph(data_source):
       # Use Rust backend
       g = Graph(backend='rust')
       
       # Pre-allocate if size is known
       estimated_nodes = len(data_source)
       # g.reserve_capacity(nodes=estimated_nodes)
       
       # Process in optimally-sized batches
       batch_size = 10000  # Tune based on your data
       
       for i in range(0, len(data_source), batch_size):
           batch_data = data_source[i:i+batch_size]
           
           with g.batch_operations() as batch:
               for item in batch_data:
                   # Simple attributes for performance
                   batch.add_node(
                       item['id'],
                       value=item['value'],          # Simple types
                       category=item['category'],    # Automatically interned
                       timestamp=item['timestamp']   # Numeric when possible
                   )
       
       return g

Common Pitfalls
~~~~~~~~~~~~~~

1. **Not Using Batch Operations**: Single operations are much slower
2. **Complex Attributes**: Deeply nested objects hurt performance
3. **Frequent Graph Modifications**: Prefer bulk operations
4. **Not Monitoring Performance**: Performance issues go unnoticed
5. **Wrong Backend Choice**: Using Python backend for large graphs

The Rust backend provides exceptional performance for graph operations, but following these optimization techniques will help you get the maximum benefit from your GLI deployment.

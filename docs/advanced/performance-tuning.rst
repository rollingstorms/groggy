Advanced Performance Tuning
===========================

This guide covers advanced techniques for optimizing Groggy performance in demanding production environments and large-scale graph analysis scenarios.

System Configuration
--------------------

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

**Pool Configuration**:

.. code-block:: python

   import groggy as gr
   
   # Configure memory pools for optimal performance
   pool_config = {
       'small_pool_size': 64 * 1024 * 1024,    # 64MB for small objects
       'medium_pool_size': 256 * 1024 * 1024,  # 256MB for medium objects
       'large_pool_size': 1024 * 1024 * 1024,  # 1GB for large objects
       'string_pool_size': 128 * 1024 * 1024,  # 128MB for strings
       'enable_compaction': True,               # Auto-compaction
       'compaction_threshold': 0.3,             # Compact when 30% fragmented
   }
   
   gr.configure_memory_pools(pool_config)

**Memory Monitoring**:

.. code-block:: python

   def monitor_memory_usage():
       """Monitor and log memory usage patterns"""
       stats = gr.get_memory_stats()
       
       print(f"Total allocated: {stats['total_allocated'] / 1024**2:.1f} MB")
       print(f"Peak usage: {stats['peak_usage'] / 1024**2:.1f} MB")
       print(f"Pool utilization:")
       
       for pool_name, usage in stats['pool_usage'].items():
           utilization = usage['used'] / usage['capacity'] * 100
           print(f"  {pool_name}: {utilization:.1f}% ({usage['used']} / {usage['capacity']})")
       
       # Alert if memory usage is high
       if stats['total_allocated'] > 0.8 * stats['system_memory']:
           print("⚠️  Warning: High memory usage detected")
           
       return stats

Thread Pool Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configure thread pools based on workload
   def configure_threading(workload_type='mixed'):
       if workload_type == 'cpu_intensive':
           # CPU-bound algorithms
           gr.set_thread_count(gr.cpu_count())
           gr.set_thread_affinity(True)
           
       elif workload_type == 'memory_intensive':
           # Memory-bound operations
           gr.set_thread_count(max(4, gr.cpu_count() // 2))
           gr.set_numa_policy('local')
           
       elif workload_type == 'mixed':
           # Balanced workload
           gr.set_thread_count(gr.cpu_count())
           gr.enable_work_stealing(True)
       
       # Configure thread pool behavior
       gr.set_thread_pool_config({
           'queue_size': 10000,
           'idle_timeout': 60,  # seconds
           'stack_size': 8 * 1024 * 1024,  # 8MB stack
       })

Algorithm-Specific Optimizations
--------------------------------

PageRank Optimization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def optimized_pagerank(g, precision_level='balanced'):
       """Optimized PageRank with different precision levels"""
       
       if precision_level == 'fast':
           # Fast but less accurate
           return g.centrality.pagerank(
               alpha=0.85,
               max_iter=50,
               tolerance=1e-4,
               parallel=True,
               use_approximation=True,
               sample_rate=0.1
           )
           
       elif precision_level == 'balanced':
           # Good balance of speed and accuracy
           return g.centrality.pagerank(
               alpha=0.85,
               max_iter=100,
               tolerance=1e-6,
               parallel=True,
               use_power_iteration=True
           )
           
       elif precision_level == 'accurate':
           # Maximum accuracy
           return g.centrality.pagerank(
               alpha=0.85,
               max_iter=1000,
               tolerance=1e-9,
               parallel=True,
               use_conjugate_gradient=True
           )

   # Adaptive PageRank based on graph properties
   def adaptive_pagerank(g):
       node_count = g.node_count()
       edge_count = g.edge_count()
       density = g.density()
       
       if node_count < 10000:
           return optimized_pagerank(g, 'accurate')
       elif density > 0.1:
           # Dense graphs benefit from matrix methods
           return g.centrality.pagerank(use_matrix_method=True)
       else:
           # Sparse graphs use iterative methods
           return optimized_pagerank(g, 'balanced')

Community Detection Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def hierarchical_community_detection(g):
       """Multi-level community detection for large graphs"""
       
       if g.node_count() < 50000:
           # Direct Leiden algorithm
           return g.communities.leiden(resolution=1.0)
       
       # Multi-level approach for large graphs
       communities_hierarchy = []
       current_graph = g
       
       while current_graph.node_count() > 10000:
           # Detect communities at current level
           communities = current_graph.communities.louvain(
               resolution=0.5,  # Lower resolution for coarse grouping
               max_iter=50
           )
           
           communities_hierarchy.append(communities)
           
           # Create contracted graph
           current_graph = contract_graph_by_communities(current_graph, communities)
       
       # Final high-resolution detection
       final_communities = current_graph.communities.leiden(resolution=1.5)
       communities_hierarchy.append(final_communities)
       
       # Expand back to original graph
       return expand_hierarchical_communities(g, communities_hierarchy)

   def contract_graph_by_communities(g, communities):
       """Contract graph by merging nodes in same community"""
       contracted = gr.Graph(directed=g.directed)
       
       # Create super-nodes for each community
       community_map = {}
       for i, community in enumerate(communities):
           super_node_id = f"community_{i}"
           
           # Aggregate attributes
           community_attrs = aggregate_community_attributes(g, community)
           contracted.add_node(super_node_id, **community_attrs)
           
           for node in community:
               community_map[node] = super_node_id
       
       # Add edges between super-nodes
       edge_weights = {}
       for source, target in g.edges:
           source_community = community_map[source]
           target_community = community_map[target]
           
           if source_community != target_community:
               edge_key = (source_community, target_community)
               edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
       
       for (source, target), weight in edge_weights.items():
           contracted.add_edge(source, target, weight=weight)
       
       return contracted

Large Graph Strategies
----------------------

Graph Partitioning
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GraphPartitioner:
       def __init__(self, partition_size=100000):
           self.partition_size = partition_size
           
       def partition_graph(self, g):
           """Partition large graph for distributed processing"""
           
           if g.node_count() <= self.partition_size:
               return [g]  # No partitioning needed
           
           # Use METIS-style partitioning
           partitions = self._metis_partition(g)
           
           return [self._create_subgraph(g, partition) for partition in partitions]
       
       def _metis_partition(self, g):
           """METIS-inspired graph partitioning"""
           # Simplified implementation
           # In practice, use specialized partitioning libraries
           
           node_count = g.node_count()
           num_partitions = (node_count + self.partition_size - 1) // self.partition_size
           
           # Initial random assignment
           partition_assignment = [i % num_partitions for i in range(node_count)]
           
           # Kernighan-Lin refinement
           for iteration in range(10):
               improved = self._kl_refinement(g, partition_assignment, num_partitions)
               if not improved:
                   break
           
           # Group nodes by partition
           partitions = [[] for _ in range(num_partitions)]
           for node_idx, partition_id in enumerate(partition_assignment):
               node_id = g.get_node_id_by_index(node_idx)
               partitions[partition_id].append(node_id)
           
           return partitions
       
       def _kl_refinement(self, g, assignment, num_partitions):
           """Kernighan-Lin refinement step"""
           # Simplified implementation
           improvements = 0
           
           for node_idx in range(g.node_count()):
               current_partition = assignment[node_idx]
               best_partition = current_partition
               best_gain = 0
               
               # Try moving to each other partition
               for target_partition in range(num_partitions):
                   if target_partition == current_partition:
                       continue
                   
                   gain = self._compute_move_gain(g, node_idx, current_partition, target_partition, assignment)
                   
                   if gain > best_gain:
                       best_gain = gain
                       best_partition = target_partition
               
               # Apply best move
               if best_partition != current_partition:
                   assignment[node_idx] = best_partition
                   improvements += 1
           
           return improvements > 0

Streaming Graph Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StreamingGraphProcessor:
       def __init__(self, window_size=10000, overlap=0.1):
           self.window_size = window_size
           self.overlap = int(window_size * overlap)
           self.current_window = gr.Graph()
           self.window_buffer = []
           
       def process_edge_stream(self, edge_stream, analysis_func):
           """Process streaming edges with sliding window"""
           
           results = []
           
           for edge_batch in edge_stream:
               # Add edges to current window
               for source, target, attrs in edge_batch:
                   self.current_window.add_edge(source, target, **attrs)
                   self.window_buffer.append((source, target))
               
               # Check if window is full
               if len(self.window_buffer) >= self.window_size:
                   # Analyze current window
                   result = analysis_func(self.current_window)
                   results.append(result)
                   
                   # Slide window
                   self._slide_window()
           
           return results
       
       def _slide_window(self):
           """Slide the analysis window"""
           # Remove oldest edges (keeping overlap)
           edges_to_remove = self.window_buffer[:-self.overlap]
           
           for source, target in edges_to_remove:
               if self.current_window.has_edge(source, target):
                   self.current_window.remove_edge(source, target)
           
           # Update buffer
           self.window_buffer = self.window_buffer[-self.overlap:]

   # Example usage
   def streaming_centrality_analysis():
       processor = StreamingGraphProcessor(window_size=50000)
       
       def compute_top_nodes(g):
           if g.node_count() < 100:
               return []
           
           centrality = g.centrality.pagerank()
           top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
           return top_nodes
       
       # Simulate edge stream
       edge_stream = generate_edge_stream()  # Your edge source
       
       top_nodes_over_time = processor.process_edge_stream(edge_stream, compute_top_nodes)
       
       return top_nodes_over_time

Approximation Algorithms
-----------------------

Sampling Strategies
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GraphSampler:
       def __init__(self, sampling_ratio=0.1):
           self.sampling_ratio = sampling_ratio
           
       def random_node_sampling(self, g, seed=None):
           """Random node sampling with induced subgraph"""
           if seed:
               import random
               random.seed(seed)
           
           node_count = g.node_count()
           sample_size = int(node_count * self.sampling_ratio)
           
           sampled_nodes = g.nodes.sample(n=sample_size)
           return g.subgraph(sampled_nodes)
       
       def snowball_sampling(self, g, seed_nodes, k_steps=3):
           """Snowball sampling for local neighborhood exploration"""
           sampled_nodes = set(seed_nodes)
           current_frontier = set(seed_nodes)
           
           for step in range(k_steps):
               next_frontier = set()
               
               for node in current_frontier:
                   neighbors = g.neighbors(node)
                   
                   # Sample neighbors based on ratio
                   num_to_sample = max(1, int(len(neighbors) * self.sampling_ratio))
                   sampled_neighbors = random.sample(neighbors, min(num_to_sample, len(neighbors)))
                   
                   next_frontier.update(sampled_neighbors)
                   sampled_nodes.update(sampled_neighbors)
               
               current_frontier = next_frontier - sampled_nodes
               
               if not current_frontier:
                   break
           
           return g.subgraph(list(sampled_nodes))
       
       def degree_stratified_sampling(self, g):
           """Sampling that preserves degree distribution"""
           degrees = g.degree()
           
           # Create degree strata
           degree_strata = self._create_degree_strata(degrees)
           
           sampled_nodes = []
           for stratum, nodes in degree_strata.items():
               stratum_sample_size = max(1, int(len(nodes) * self.sampling_ratio))
               stratum_sample = random.sample(nodes, min(stratum_sample_size, len(nodes)))
               sampled_nodes.extend(stratum_sample)
           
           return g.subgraph(sampled_nodes)

Approximate Algorithms
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def approximate_betweenness_centrality(g, k=None, normalized=True):
       """Approximate betweenness using sampling"""
       
       if k is None:
           # Adaptive k based on graph size
           node_count = g.node_count()
           k = min(node_count, max(100, int(math.sqrt(node_count))))
       
       # Sample k nodes for shortest path computation
       sampled_nodes = g.nodes.sample(n=k)
       
       # Initialize betweenness scores
       betweenness = {node: 0.0 for node in g.nodes}
       
       # Compute shortest paths from sampled nodes
       for source in sampled_nodes:
           # Single-source shortest paths
           distances, predecessors = g.single_source_shortest_paths(source)
           
           # Accumulate betweenness scores
           for target in g.nodes:
               if target == source:
                   continue
               
               paths = g.get_all_shortest_paths(source, target, predecessors)
               
               for path in paths:
                   # Add to betweenness for intermediate nodes
                   for intermediate in path[1:-1]:
                       betweenness[intermediate] += 1.0 / len(paths)
       
       # Scale by sampling factor
       scaling_factor = (g.node_count() * (g.node_count() - 1)) / (k * (k - 1))
       
       for node in betweenness:
           betweenness[node] *= scaling_factor
           
           if normalized and g.node_count() > 2:
               # Normalize by maximum possible betweenness
               max_betweenness = (g.node_count() - 1) * (g.node_count() - 2) / 2
               betweenness[node] /= max_betweenness
       
       return betweenness

   def approximate_clustering_coefficient(g, sample_size=1000):
       """Approximate clustering coefficient using node sampling"""
       
       if g.node_count() <= sample_size:
           return g.clustering()  # Exact computation
       
       sampled_nodes = g.nodes.sample(n=sample_size)
       
       clustering_sum = 0.0
       valid_nodes = 0
       
       for node in sampled_nodes:
           neighbors = g.neighbors(node)
           degree = len(neighbors)
           
           if degree < 2:
               continue  # Skip nodes with degree < 2
           
           # Count triangles
           triangles = 0
           for i, neighbor1 in enumerate(neighbors):
               for neighbor2 in neighbors[i+1:]:
                   if g.has_edge(neighbor1, neighbor2):
                       triangles += 1
           
           # Local clustering coefficient
           possible_triangles = degree * (degree - 1) / 2
           local_clustering = triangles / possible_triangles if possible_triangles > 0 else 0
           
           clustering_sum += local_clustering
           valid_nodes += 1
       
       return clustering_sum / valid_nodes if valid_nodes > 0 else 0.0

Cache Optimization
------------------

Result Caching
~~~~~~~~~~~~~~

.. code-block:: python

   class AdvancedCacheManager:
       def __init__(self, max_memory_mb=1024):
           self.max_memory = max_memory_mb * 1024 * 1024
           self.cache = {}
           self.access_times = {}
           self.cache_sizes = {}
           self.current_memory = 0
           
       def cache_algorithm_result(self, algorithm_name, graph_hash, params, result):
           """Cache algorithm result with intelligent eviction"""
           
           cache_key = self._make_cache_key(algorithm_name, graph_hash, params)
           
           # Estimate result size
           result_size = self._estimate_object_size(result)
           
           # Check if we need to evict
           while self.current_memory + result_size > self.max_memory:
               self._evict_least_recently_used()
           
           # Store result
           self.cache[cache_key] = result
           self.access_times[cache_key] = time.time()
           self.cache_sizes[cache_key] = result_size
           self.current_memory += result_size
       
       def get_cached_result(self, algorithm_name, graph_hash, params):
           """Retrieve cached result if available"""
           
           cache_key = self._make_cache_key(algorithm_name, graph_hash, params)
           
           if cache_key in self.cache:
               # Update access time
               self.access_times[cache_key] = time.time()
               return self.cache[cache_key]
           
           return None
       
       def _evict_least_recently_used(self):
           """Evict least recently used cache entry"""
           
           if not self.cache:
               return
           
           # Find least recently used entry
           lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
           
           # Remove from cache
           del self.cache[lru_key]
           del self.access_times[lru_key]
           self.current_memory -= self.cache_sizes[lru_key]
           del self.cache_sizes[lru_key]

   # Global cache manager
   cache_manager = AdvancedCacheManager(max_memory_mb=2048)

   def cached_pagerank(g, **kwargs):
       """PageRank with intelligent caching"""
       
       graph_hash = g.structure_hash()
       params_hash = hash(frozenset(kwargs.items()))
       
       # Check cache first
       cached_result = cache_manager.get_cached_result('pagerank', graph_hash, params_hash)
       if cached_result is not None:
           return cached_result
       
       # Compute result
       result = g.centrality.pagerank(**kwargs)
       
       # Cache result
       cache_manager.cache_algorithm_result('pagerank', graph_hash, params_hash, result)
       
       return result

NUMA Optimization
-----------------

Memory Locality
~~~~~~~~~~~~~~

.. code-block:: python

   import numa

   def configure_numa_optimization():
       """Configure NUMA-aware memory allocation"""
       
       if not numa.available():
           print("NUMA not available on this system")
           return
       
       # Get NUMA topology
       num_nodes = numa.get_max_node() + 1
       print(f"NUMA nodes available: {num_nodes}")
       
       # Configure memory policy
       numa.set_mempolicy(numa.MPOL_LOCAL)
       
       # Set CPU affinity for threads
       cpu_info = numa.node_to_cpus(0)  # Use first NUMA node
       gr.set_thread_cpu_affinity(cpu_info)
       
       # Configure memory pools per NUMA node
       for node_id in range(num_nodes):
           node_memory = numa.node_size(node_id)
           pool_size = min(node_memory // 4, 1024 * 1024 * 1024)  # 1GB max
           
           gr.create_numa_memory_pool(node_id, pool_size)

   def numa_aware_graph_processing(g, algorithm_func):
       """Process graph with NUMA awareness"""
       
       if not numa.available():
           return algorithm_func(g)
       
       # Get current NUMA node
       current_node = numa.get_current_node()
       
       # Bind memory allocation to current node
       numa.set_mempolicy(numa.MPOL_BIND, [current_node])
       
       try:
           result = algorithm_func(g)
           return result
       finally:
           # Reset memory policy
           numa.set_mempolicy(numa.MPOL_DEFAULT)

Profiling and Monitoring
------------------------

Advanced Profiling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cProfile
   import pstats
   import line_profiler

   class GraphProfiler:
       def __init__(self):
           self.profiles = {}
           
       def profile_operation(self, operation_name, func, *args, **kwargs):
           """Profile a graph operation"""
           
           profiler = cProfile.Profile()
           
           # Run with profiling
           profiler.enable()
           result = func(*args, **kwargs)
           profiler.disable()
           
           # Store profile
           self.profiles[operation_name] = profiler
           
           return result
       
       def generate_report(self, operation_name, sort_by='cumulative', top_n=20):
           """Generate profiling report"""
           
           if operation_name not in self.profiles:
               raise ValueError(f"No profile found for {operation_name}")
           
           stats = pstats.Stats(self.profiles[operation_name])
           stats.sort_stats(sort_by)
           
           print(f"\nProfiling Report for {operation_name}")
           print("=" * 50)
           stats.print_stats(top_n)
           
           # Identify bottlenecks
           print("\nTop Bottlenecks:")
           bottlenecks = stats.get_stats_profile()
           for func, (cc, nc, tt, ct, callers) in list(bottlenecks.items())[:5]:
               print(f"  {func}: {tt:.3f}s total, {ct:.3f}s cumulative")

   # Memory profiling
   def profile_memory_usage(func, *args, **kwargs):
       """Profile memory usage of a function"""
       
       import tracemalloc
       
       tracemalloc.start()
       
       # Get baseline memory
       baseline = tracemalloc.get_traced_memory()[0]
       
       # Run function
       result = func(*args, **kwargs)
       
       # Get peak memory
       current, peak = tracemalloc.get_traced_memory()
       
       tracemalloc.stop()
       
       memory_info = {
           'peak_memory_mb': peak / 1024 / 1024,
           'current_memory_mb': current / 1024 / 1024,
           'memory_increase_mb': (current - baseline) / 1024 / 1024,
       }
       
       return result, memory_info

Real-time Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import threading
   import time
   from collections import deque

   class PerformanceMonitor:
       def __init__(self, history_size=1000):
           self.history_size = history_size
           self.metrics = {
               'cpu_usage': deque(maxlen=history_size),
               'memory_usage': deque(maxlen=history_size),
               'cache_hit_rate': deque(maxlen=history_size),
               'operation_times': deque(maxlen=history_size),
           }
           self.monitoring = False
           self.monitor_thread = None
           
       def start_monitoring(self, interval=1.0):
           """Start continuous performance monitoring"""
           
           self.monitoring = True
           self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
           self.monitor_thread.daemon = True
           self.monitor_thread.start()
           
       def stop_monitoring(self):
           """Stop performance monitoring"""
           
           self.monitoring = False
           if self.monitor_thread:
               self.monitor_thread.join()
       
       def _monitor_loop(self, interval):
           """Main monitoring loop"""
           
           while self.monitoring:
               # Collect metrics
               cpu_usage = self._get_cpu_usage()
               memory_usage = self._get_memory_usage()
               cache_stats = gr.get_cache_statistics()
               
               # Store metrics
               self.metrics['cpu_usage'].append(cpu_usage)
               self.metrics['memory_usage'].append(memory_usage)
               self.metrics['cache_hit_rate'].append(cache_stats['hit_rate'])
               
               # Check for performance issues
               self._check_performance_alerts()
               
               time.sleep(interval)
       
       def _check_performance_alerts(self):
           """Check for performance issues and alert"""
           
           if len(self.metrics['cpu_usage']) < 10:
               return
           
           # High CPU usage alert
           recent_cpu = list(self.metrics['cpu_usage'])[-10:]
           avg_cpu = sum(recent_cpu) / len(recent_cpu)
           
           if avg_cpu > 90:
               print(f"⚠️  High CPU usage: {avg_cpu:.1f}%")
           
           # Low cache hit rate alert
           recent_cache = list(self.metrics['cache_hit_rate'])[-10:]
           avg_cache_hit = sum(recent_cache) / len(recent_cache)
           
           if avg_cache_hit < 0.5:
               print(f"⚠️  Low cache hit rate: {avg_cache_hit:.1%}")

Best Practices Summary
---------------------

1. **Memory Management**:
   - Configure memory pools based on workload characteristics
   - Monitor memory usage and set appropriate limits
   - Use NUMA-aware allocation for large systems

2. **Algorithm Selection**:
   - Choose algorithms based on graph properties (size, density, structure)
   - Use approximation algorithms for very large graphs
   - Implement adaptive algorithms that adjust based on input

3. **Caching Strategy**:
   - Cache expensive algorithm results
   - Implement intelligent cache eviction policies
   - Consider cache coherency for multi-threaded operations

4. **Parallelization**:
   - Configure thread pools based on hardware and workload
   - Use work-stealing for irregular workloads
   - Release GIL for CPU-intensive operations

5. **Monitoring**:
   - Implement comprehensive performance monitoring
   - Profile critical code paths regularly
   - Set up alerts for performance degradation

These advanced techniques enable Groggy to handle demanding production workloads while maintaining optimal performance across a wide range of graph analysis scenarios.
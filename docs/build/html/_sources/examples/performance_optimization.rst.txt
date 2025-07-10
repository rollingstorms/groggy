Performance Optimization Examples
==================================

This section provides practical examples for optimizing Groggy performance in real-world scenarios.

Large Graph Construction
-----------------------

Building Large Graphs Efficiently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import random
   import groggy as gr
   
   def create_large_social_network():
       """Create a social network with 100K users efficiently"""
       
       g = gr.Graph()
       print("Creating 100K node social network...")
       start_time = time.time()
       
       # Generate user data efficiently  
       users_data = []
       cities = ['NYC', 'SF', 'LA', 'CHI', 'BOS']
       interests = ['tech', 'sports', 'music', 'art', 'cooking', 'travel']
       
       for i in range(100000):
           users_data.append({
               'id': f"user_{i}",
               'name': f"User{i}",
               'age': random.randint(18, 65),
               'city': random.choice(cities),
               'interest': random.choice(interests),
               'score': random.randint(0, 1000)
           })
       
       # Single batch operation for maximum efficiency
       g.add_nodes(users_data)
       
       print(f"Added {len(g.nodes)} nodes in {time.time() - start_time:.2f}s")
           
           # Progress reporting
           if batch_start % 200000 == 0:
               elapsed = time.time() - start_time
               print(f"  Added {batch_start + batch_size} nodes in {elapsed:.1f}s")
       
       construction_time = time.time() - start_time
       
       # Add some edges for realism
       edges_data = []
       for i in range(0, len(users_data), 100):  # Connect every 100th user
           if i + 1 < len(users_data):
               edges_data.append({
                   'source': f"user_{i}",
                   'target': f"user_{i+1}",
                   'relationship': 'friend',
                   'strength': random.random()
               })
       
       start_edges = time.time()
       g.add_edges(edges_data)
       
       construction_time = time.time() - start_time
       print(f"Graph construction completed in {construction_time:.2f}s")
       print(f"Rate: {len(g.nodes) / construction_time:.0f} nodes/sec")
       
       return g

High-Performance Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def demonstrate_filtering_performance():
       """Show optimized filtering techniques"""
       
       # Create test graph
       g = create_large_social_network()
       
       # Benchmark different filtering approaches
       import time
       
       # Fast exact matching (bitmap index)
       start = time.time()
       tech_users = g.filter_nodes(interest='tech')
       exact_time = time.time() - start
       print(f"Exact match filtering: {len(tech_users)} results in {exact_time:.4f}s")
       
       # Range queries  
       start = time.time()
       young_users = g.filter_nodes(lambda n, a: 18 <= a.get('age', 0) <= 25)
       range_time = time.time() - start
       print(f"Range filtering: {len(young_users)} results in {range_time:.4f}s")
       
       # Complex filtering
       start = time.time()
       young_tech_sf = g.filter_nodes(
           lambda n, a: (a.get('interest') == 'tech' and 
                        a.get('age', 0) < 30 and
                        a.get('city') == 'SF')
       )
       complex_time = time.time() - start
       print(f"Complex filtering: {len(young_tech_sf)} results in {complex_time:.4f}s")
                   current_batch.append(data)
                   
                   if len(current_batch) >= batch_size:
                       with g.batch_operations() as batch:
                           for item in current_batch:
                               batch.add_node(item['id'], **item.get('attributes', {}))
                       current_batch = []
                       
               except json.JSONDecodeError:
                   continue
       
       # Process remaining items
       if current_batch:
           with g.batch_operations() as batch:
               for item in current_batch:
                   batch.add_node(item['id'], **item.get('attributes', {}))
       
       return g

High-Performance Queries
------------------------

Optimized Graph Traversal
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from groggy import Graph
   import time
   from collections import deque, defaultdict
   
   def optimized_bfs(graph, start_node, max_depth=3):
       \"\"\"Optimized breadth-first search\"\"\"
       
       if not graph.has_node(start_node):
           return []
       
       visited = {start_node}
       queue = deque([(start_node, 0)])
       results = []
       
       # Cache for neighbor lookups
       neighbor_cache = {}
       
       while queue:
           current_node, depth = queue.popleft()
           results.append((current_node, depth))
           
           if depth < max_depth:
               # Use cached neighbors when possible
               if current_node not in neighbor_cache:
                   neighbor_cache[current_node] = graph.get_neighbors(current_node)
               
               neighbors = neighbor_cache[current_node]
               
               for neighbor in neighbors:
                   if neighbor not in visited:
                       visited.add(neighbor)
                       queue.append((neighbor, depth + 1))
       
       return results
   
   def batch_shortest_paths(graph, source_nodes, target_nodes):
       \"\"\"Compute shortest paths for multiple source-target pairs\"\"\"
       
       results = {}
       
       # Group by source to minimize repeated work
       sources_grouped = defaultdict(list)
       for source, target in zip(source_nodes, target_nodes):
           sources_grouped[source].append(target)
       
       for source, targets in sources_grouped.items():
           # Single BFS from source to find paths to all targets
           paths = single_source_shortest_paths(graph, source, set(targets))
           
           for target in targets:
               results[(source, target)] = paths.get(target, None)
       
       return results
   
   def single_source_shortest_paths(graph, source, targets):
       \"\"\"Find shortest paths from source to multiple targets\"\"\"
       
       if not graph.has_node(source):
           return {}
       
       visited = {source}
       queue = deque([(source, [source])])
       paths = {}
       remaining_targets = set(targets)
       
       while queue and remaining_targets:
           current_node, path = queue.popleft()
           
           if current_node in remaining_targets:
               paths[current_node] = path
               remaining_targets.remove(current_node)
           
           for neighbor in graph.get_neighbors(current_node):
               if neighbor not in visited:
                   visited.add(neighbor)
                   queue.append((neighbor, path + [neighbor]))
       
       return paths

Efficient Attribute Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from groggy import Graph
   import numpy as np
   from collections import Counter
   
   def efficient_attribute_analysis(graph):
       \"\"\"Analyze node attributes efficiently\"\"\"
       
       # Collect all attributes in one pass
       all_attributes = {}
       attribute_counts = Counter()
       
       for node_id in graph.nodes:
           node_data = graph.get_node(node_id)
           
           for attr_name, attr_value in node_data.items():
               if attr_name not in all_attributes:
                   all_attributes[attr_name] = []
               
               all_attributes[attr_name].append(attr_value)
               attribute_counts[attr_name] += 1
       
       # Compute statistics efficiently
       statistics = {}
       for attr_name, values in all_attributes.items():
           if all(isinstance(v, (int, float)) for v in values):
               # Numeric attribute
               np_values = np.array(values)
               statistics[attr_name] = {
                   'type': 'numeric',
                   'count': len(values),
                   'mean': np.mean(np_values),
                   'std': np.std(np_values),
                   'min': np.min(np_values),
                   'max': np.max(np_values),
               }
           else:
               # Categorical attribute
               value_counts = Counter(values)
               statistics[attr_name] = {
                   'type': 'categorical',
                   'count': len(values),
                   'unique_values': len(value_counts),
                   'top_values': value_counts.most_common(5),
               }
       
       return statistics
   
   def vectorized_node_scoring(graph, score_function):
       \"\"\"Apply scoring function to all nodes efficiently\"\"\"
       
       # Collect all node data first
       node_data = {}
       for node_id in graph.nodes:
           node_data[node_id] = graph.get_node(node_id)
       
       # Apply scoring function in batch
       scores = {}
       for node_id, data in node_data.items():
           scores[node_id] = score_function(data)
       
       return scores

Memory-Efficient Operations
--------------------------

Streaming Graph Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gc
   from groggy import Graph
   
   def process_graph_in_chunks(large_graph, chunk_size=10000):
       \"\"\"Process large graph in memory-efficient chunks\"\"\"
       
       node_ids = list(large_graph.nodes)
       total_nodes = len(node_ids)
       results = []
       
       for i in range(0, total_nodes, chunk_size):
           chunk_nodes = node_ids[i:i + chunk_size]
           
           # Process chunk
           chunk_results = []
           for node_id in chunk_nodes:
               node_data = large_graph.get_node(node_id)
               # Process individual node
               processed = process_single_node(node_data)
               chunk_results.append(processed)
           
           results.extend(chunk_results)
           
           # Force garbage collection to free memory
           if i % (chunk_size * 10) == 0:
               gc.collect()
               print(f"Processed {i + len(chunk_nodes)} / {total_nodes} nodes")
       
       return results
   
   def memory_efficient_graph_copy(source_graph, filter_func=None):
       \"\"\"Create filtered copy of graph with minimal memory usage\"\"\"
       
       target_graph = Graph(backend='rust')
       
       # Process nodes in batches
       batch_size = 5000
       node_ids = list(source_graph.nodes)
       
       for i in range(0, len(node_ids), batch_size):
           batch_nodes = node_ids[i:i + batch_size]
           
           with target_graph.batch_operations() as batch:
               for node_id in batch_nodes:
                   node_data = source_graph.get_node(node_id)
                   
                   # Apply filter if provided
                   if filter_func is None or filter_func(node_id, node_data):
                       batch.add_node(node_id, **node_data.attributes)
       
       # Copy edges in batches
       for i in range(0, len(node_ids), batch_size):
           batch_nodes = set(node_ids[i:i + batch_size])
           
           with target_graph.batch_operations() as batch:
               for source_node in batch_nodes:
                   if target_graph.has_node(source_node):
                       for target_node in source_graph.get_neighbors(source_node):
                           if (target_graph.has_node(target_node) and 
                               not target_graph.has_edge(source_node, target_node)):
                               
                               edge_data = source_graph.get_edge(source_node, target_node)
                               batch.add_edge(source_node, target_node, **edge_data.attributes)
       
       return target_graph

Parallel Processing Patterns
----------------------------

Multi-threaded Graph Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import concurrent.futures
   import threading
   from groggy import Graph
   
   def parallel_node_processing(graph, process_func, num_workers=4):
       \"\"\"Process nodes in parallel using thread pool\"\"\"
       
       # Note: Only works with Rust backend (thread-safe)
       if graph.backend != 'rust':
           raise ValueError("Parallel processing requires Rust backend")
       
       node_ids = list(graph.nodes)
       chunk_size = len(node_ids) // num_workers
       
       def process_chunk(start_idx, end_idx):
           chunk_results = []
           for i in range(start_idx, min(end_idx, len(node_ids))):
               node_id = node_ids[i]
               node_data = graph.get_node(node_id)
               result = process_func(node_id, node_data)
               chunk_results.append(result)
           return chunk_results
       
       # Execute in parallel
       with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
           futures = []
           
           for i in range(num_workers):
               start_idx = i * chunk_size
               end_idx = start_idx + chunk_size
               future = executor.submit(process_chunk, start_idx, end_idx)
               futures.append(future)
           
           # Collect results
           all_results = []
           for future in concurrent.futures.as_completed(futures):
               chunk_results = future.result()
               all_results.extend(chunk_results)
       
       return all_results
   
   def parallel_graph_statistics(graph):
       \"\"\"Compute graph statistics in parallel\"\"\"
       
       def compute_node_stats(node_id, node_data):
           return {
               'id': node_id,
               'attribute_count': len(node_data.attributes),
               'degree': graph.degree(node_id),
               'has_name': 'name' in node_data
           }
       
       # Process nodes in parallel
       node_stats = parallel_node_processing(graph, compute_node_stats)
       
       # Aggregate results
       total_attributes = sum(stat['attribute_count'] for stat in node_stats)
       total_degree = sum(stat['degree'] for stat in node_stats)
       nodes_with_names = sum(stat['has_name'] for stat in node_stats)
       
       return {
           'total_nodes': len(node_stats),
           'average_attributes': total_attributes / len(node_stats),
           'average_degree': total_degree / len(node_stats),
           'nodes_with_names': nodes_with_names,
           'name_coverage': nodes_with_names / len(node_stats)
       }

Benchmarking and Profiling
-------------------------

Performance Measurement Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import tracemalloc
   import psutil
   import os
   from contextlib import contextmanager
   from groggy import Graph
   
   class PerformanceBenchmark:
       def __init__(self):
           self.results = {}
           self.process = psutil.Process(os.getpid())
       
       @contextmanager
       def measure(self, operation_name):
           # Start measurements
           tracemalloc.start()
           start_time = time.perf_counter()
           start_memory = self.process.memory_info().rss
           
           try:
               yield
           finally:
               # End measurements
               end_time = time.perf_counter()
               end_memory = self.process.memory_info().rss
               current, peak = tracemalloc.get_traced_memory()
               tracemalloc.stop()
               
               # Store results
               self.results[operation_name] = {
                   'duration': end_time - start_time,
                   'memory_delta': end_memory - start_memory,
                   'python_memory_peak': peak,
                   'python_memory_current': current
               }
       
       def print_results(self):
           print("\\nPerformance Results:")
           print("=" * 60)
           
           for operation, metrics in self.results.items():
               print(f"\\n{operation}:")
               print(f"  Duration: {metrics['duration']:.3f}s")
               print(f"  Memory Delta: {metrics['memory_delta'] / 1024 / 1024:.1f} MB")
               print(f"  Python Peak Memory: {metrics['python_memory_peak'] / 1024 / 1024:.1f} MB")
   
   def comprehensive_benchmark():
       \"\"\"Run comprehensive performance benchmark\"\"\"
       
       benchmark = PerformanceBenchmark()
       
       # Test 1: Large graph creation
       with benchmark.measure("Large Graph Creation (100K nodes)"):
           g = Graph(backend='rust')
           with g.batch_operations() as batch:
               for i in range(100000):
                   batch.add_node(f"node_{i}", 
                                 value=i, 
                                 category=i % 100,
                                 active=i % 2 == 0)
       
       # Test 2: Edge addition
       with benchmark.measure("Edge Addition (50K edges)"):
           with g.batch_operations() as batch:
               for i in range(50000):
                   source = f"node_{i}"
                   target = f"node_{(i + 1) % 100000}"
                   batch.add_edge(source, target, weight=1.0)
       
       # Test 3: Random queries
       with benchmark.measure("Random Queries (10K queries)"):
           import random
           node_ids = list(g.nodes)
           for _ in range(10000):
               random_node = random.choice(node_ids)
               neighbors = g.get_neighbors(random_node)
               node_data = g.get_node(random_node)
       
       # Test 4: Graph iteration
       with benchmark.measure("Full Graph Iteration"):
           total_nodes = 0
           total_edges = 0
           for node_id in g.nodes:
               total_nodes += 1
               node_data = g.get_node(node_id)
           
           for source, target in g.edge_pairs():
               total_edges += 1
               edge_data = g.get_edge(source, target)
       
       benchmark.print_results()
       return benchmark.results
   
   # Run the benchmark
   if __name__ == "__main__":
       results = comprehensive_benchmark()

Real-World Integration Examples
------------------------------

Database Integration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import sqlite3
   from groggy import Graph
   
   def load_from_database(db_path, node_query, edge_query):
       \"\"\"Load graph from database efficiently\"\"\"
       
       g = Graph(backend='rust')
       
       with sqlite3.connect(db_path) as conn:
           conn.row_factory = sqlite3.Row  # Enable column access by name
           
           # Load nodes in batches
           cursor = conn.cursor()
           cursor.execute(node_query)
           
           batch_size = 10000
           current_batch = []
           
           while True:
               rows = cursor.fetchmany(batch_size)
               if not rows:
                   break
               
               with g.batch_operations() as batch:
                   for row in rows:
                       node_id = str(row['id'])
                       attributes = {k: v for k, v in dict(row).items() if k != 'id'}
                       batch.add_node(node_id, **attributes)
           
           # Load edges in batches
           cursor.execute(edge_query)
           
           while True:
               rows = cursor.fetchmany(batch_size)
               if not rows:
                   break
               
               with g.batch_operations() as batch:
                   for row in rows:
                       source = str(row['source_id'])
                       target = str(row['target_id'])
                       attributes = {k: v for k, v in dict(row).items() 
                                   if k not in ['source_id', 'target_id']}
                       batch.add_edge(source, target, **attributes)
       
       return g

These examples demonstrate how to achieve optimal performance with Groggy in real-world scenarios. The key principles are:

1. **Use the Rust backend** for large graphs
2. **Batch operations** whenever possible
3. **Process data in chunks** to manage memory
4. **Cache frequently accessed data**
5. **Use parallel processing** when appropriate
6. **Monitor and measure performance** regularly

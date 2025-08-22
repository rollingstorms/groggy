Large Scale Graph Processing
============================

This guide covers techniques and strategies for working with very large graphs that exceed typical memory constraints or require distributed processing.

Scale Definitions
-----------------

Understanding Graph Scales
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Small Graphs** (< 100K nodes):
- Fit entirely in memory
- All algorithms work efficiently
- Interactive analysis possible

**Medium Graphs** (100K - 10M nodes):
- Require memory optimization
- Some algorithms need approximation
- Batch processing recommended

**Large Graphs** (10M - 1B nodes):
- Require careful memory management
- Need specialized algorithms
- Often require sampling or partitioning

**Very Large Graphs** (> 1B nodes):
- Require distributed processing
- Need out-of-core algorithms
- Approximation is essential

Memory Management Strategies
---------------------------

Out-of-Core Processing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr
   import tempfile
   import os

   class OutOfCoreGraph:
       def __init__(self, memory_budget_gb=4):
           self.memory_budget = memory_budget_gb * 1024**3
           self.temp_dir = tempfile.mkdtemp()
           self.node_chunks = []
           self.edge_chunks = []
           self.current_memory = 0
           
       def add_nodes_streaming(self, node_stream, chunk_size=100000):
           """Add nodes from stream with automatic chunking"""
           
           current_chunk = []
           chunk_id = 0
           
           for node_data in node_stream:
               current_chunk.append(node_data)
               
               if len(current_chunk) >= chunk_size:
                   # Save chunk to disk
                   chunk_file = os.path.join(self.temp_dir, f"nodes_chunk_{chunk_id}.pickle")
                   self._save_chunk(current_chunk, chunk_file)
                   self.node_chunks.append(chunk_file)
                   
                   current_chunk = []
                   chunk_id += 1
           
           # Save remaining nodes
           if current_chunk:
               chunk_file = os.path.join(self.temp_dir, f"nodes_chunk_{chunk_id}.pickle")
               self._save_chunk(current_chunk, chunk_file)
               self.node_chunks.append(chunk_file)
       
       def process_chunks(self, analysis_func, combine_func):
           """Process graph chunks and combine results"""
           
           partial_results = []
           
           for chunk_file in self.node_chunks:
               # Load chunk into memory
               chunk_data = self._load_chunk(chunk_file)
               
               # Create temporary graph
               temp_graph = gr.Graph()
               for node_data in chunk_data:
                   temp_graph.add_node(**node_data)
               
               # Analyze chunk
               result = analysis_func(temp_graph)
               partial_results.append(result)
               
               # Clean up memory
               del temp_graph
               del chunk_data
           
           # Combine results
           return combine_func(partial_results)
       
       def _save_chunk(self, data, filename):
           """Save chunk to disk"""
           import pickle
           with open(filename, 'wb') as f:
               pickle.dump(data, f)
       
       def _load_chunk(self, filename):
           """Load chunk from disk"""
           import pickle
           with open(filename, 'rb') as f:
               return pickle.load(f)

Memory-Mapped Graphs
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mmap
   import numpy as np

   class MemoryMappedGraph:
       def __init__(self, filename, mode='r'):
           self.filename = filename
           self.mode = mode
           self.node_data = None
           self.edge_data = None
           
       def create(self, node_count, edge_count, node_attr_size=64):
           """Create memory-mapped graph files"""
           
           # Calculate file sizes
           node_file_size = node_count * node_attr_size
           edge_file_size = edge_count * 16  # 2 * int64 for source/target
           
           # Create node data file
           node_filename = f"{self.filename}_nodes.dat"
           with open(node_filename, 'wb') as f:
               f.write(b'\x00' * node_file_size)
           
           # Create edge data file
           edge_filename = f"{self.filename}_edges.dat"
           with open(edge_filename, 'wb') as f:
               f.write(b'\x00' * edge_file_size)
           
           # Memory map files
           self._map_files()
       
       def _map_files(self):
           """Memory map the graph data files"""
           
           node_filename = f"{self.filename}_nodes.dat"
           edge_filename = f"{self.filename}_edges.dat"
           
           # Map node data
           with open(node_filename, 'r+b') as f:
               self.node_data = mmap.mmap(f.fileno(), 0)
           
           # Map edge data
           with open(edge_filename, 'r+b') as f:
               self.edge_data = mmap.mmap(f.fileno(), 0)
       
       def get_neighbors(self, node_id):
           """Get neighbors using memory-mapped data"""
           
           # This is a simplified example
           # Real implementation would use efficient indexing
           edges = np.frombuffer(self.edge_data, dtype=np.int64).reshape(-1, 2)
           
           # Find edges with this node as source
           neighbor_mask = edges[:, 0] == node_id
           neighbors = edges[neighbor_mask, 1]
           
           return neighbors.tolist()

Graph Compression
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CompressedGraph:
       def __init__(self, compression_level=6):
           self.compression_level = compression_level
           self.compressed_adjacency = None
           self.node_mapping = {}
           
       def compress_graph(self, g):
           """Compress graph using various techniques"""
           
           # 1. Node ID compression
           compressed_graph = self._compress_node_ids(g)
           
           # 2. Adjacency list compression
           self._compress_adjacency_lists(compressed_graph)
           
           # 3. Attribute compression
           self._compress_attributes(compressed_graph)
           
           return compressed_graph
       
       def _compress_node_ids(self, g):
           """Map string node IDs to integers"""
           
           node_id_map = {}
           compressed_graph = gr.Graph(directed=g.directed)
           
           # Create integer mapping
           for i, node_id in enumerate(g.nodes):
               node_id_map[node_id] = i
               self.node_mapping[i] = node_id
           
           # Add nodes with compressed IDs
           for old_id in g.nodes:
               new_id = node_id_map[old_id]
               attrs = g.nodes[old_id]
               compressed_graph.add_node(new_id, **attrs)
           
           # Add edges with compressed IDs
           for source, target in g.edges:
               compressed_source = node_id_map[source]
               compressed_target = node_id_map[target]
               edge_attrs = g.get_edge(source, target)
               compressed_graph.add_edge(compressed_source, compressed_target, **edge_attrs)
           
           return compressed_graph
       
       def _compress_adjacency_lists(self, g):
           """Compress adjacency lists using delta encoding"""
           
           compressed_adj = {}
           
           for node in g.nodes:
               neighbors = sorted(g.neighbors(node))
               
               if not neighbors:
                   compressed_adj[node] = b''
                   continue
               
               # Delta encode neighbor list
               deltas = [neighbors[0]]  # First neighbor
               for i in range(1, len(neighbors)):
                   deltas.append(neighbors[i] - neighbors[i-1])
               
               # Variable-length encode deltas
               compressed_adj[node] = self._varint_encode(deltas)
           
           self.compressed_adjacency = compressed_adj
       
       def _varint_encode(self, values):
           """Variable-length integer encoding"""
           
           encoded = bytearray()
           
           for value in values:
               while value >= 128:
                   encoded.append((value & 127) | 128)
                   value >>= 7
               encoded.append(value & 127)
           
           return bytes(encoded)

Distributed Processing
---------------------

Graph Partitioning
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import List, Dict, Tuple
   import hashlib

   class DistributedGraphProcessor:
       def __init__(self, num_partitions=4):
           self.num_partitions = num_partitions
           self.partitions = [gr.Graph() for _ in range(num_partitions)]
           self.node_to_partition = {}
           
       def partition_graph(self, g, method='hash'):
           """Partition graph across multiple processors"""
           
           if method == 'hash':
               return self._hash_partition(g)
           elif method == 'metis':
               return self._metis_partition(g)
           elif method == 'community':
               return self._community_partition(g)
           else:
               raise ValueError(f"Unknown partitioning method: {method}")
       
       def _hash_partition(self, g):
           """Simple hash-based partitioning"""
           
           # Assign nodes to partitions based on hash
           for node_id in g.nodes:
               partition_id = self._hash_node_id(node_id) % self.num_partitions
               self.node_to_partition[node_id] = partition_id
               
               # Add node to partition
               node_attrs = g.nodes[node_id]
               self.partitions[partition_id].add_node(node_id, **node_attrs)
           
           # Add edges (may create cross-partition edges)
           for source, target in g.edges:
               source_partition = self.node_to_partition[source]
               target_partition = self.node_to_partition[target]
               
               edge_attrs = g.get_edge(source, target)
               
               # Add edge to source partition
               if not self.partitions[source_partition].has_node(target):
                   # Add target as replica
                   target_attrs = g.nodes[target]
                   self.partitions[source_partition].add_node(target, **target_attrs, _replica=True)
               
               self.partitions[source_partition].add_edge(source, target, **edge_attrs)
               
               # Add to target partition if different
               if target_partition != source_partition:
                   if not self.partitions[target_partition].has_node(source):
                       source_attrs = g.nodes[source]
                       self.partitions[target_partition].add_node(source, **source_attrs, _replica=True)
                   
                   self.partitions[target_partition].add_edge(source, target, **edge_attrs)
           
           return self.partitions
       
       def _hash_node_id(self, node_id):
           """Hash function for node IDs"""
           return int(hashlib.md5(str(node_id).encode()).hexdigest(), 16)
       
       def parallel_process(self, algorithm_func, combine_func):
           """Process partitions in parallel"""
           
           import multiprocessing as mp
           
           # Create process pool
           with mp.Pool(processes=self.num_partitions) as pool:
               # Process each partition
               results = pool.map(algorithm_func, self.partitions)
           
           # Combine results
           return combine_func(results)

Message Passing Interface
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MessagePassingProcessor:
       def __init__(self, graph_partitions):
           self.partitions = graph_partitions
           self.num_partitions = len(graph_partitions)
           self.message_queues = [[] for _ in range(self.num_partitions)]
           
       def distributed_pagerank(self, alpha=0.85, max_iterations=100, tolerance=1e-6):
           """Distributed PageRank using message passing"""
           
           # Initialize PageRank values
           pagerank_values = {}
           for partition_id, partition in enumerate(self.partitions):
               for node in partition.nodes:
                   pagerank_values[node] = 1.0 / partition.node_count()
           
           for iteration in range(max_iterations):
               # Clear message queues
               for queue in self.message_queues:
                   queue.clear()
               
               # Compute messages for each partition
               new_pagerank = {}
               
               for partition_id, partition in enumerate(self.partitions):
                   partition_pagerank = self._compute_partition_pagerank(
                       partition, pagerank_values, alpha, partition_id
                   )
                   new_pagerank.update(partition_pagerank)
               
               # Check convergence
               diff = sum(abs(new_pagerank[node] - pagerank_values[node]) 
                         for node in new_pagerank)
               
               if diff < tolerance:
                   break
               
               pagerank_values = new_pagerank
           
           return pagerank_values
       
       def _compute_partition_pagerank(self, partition, global_pagerank, alpha, partition_id):
           """Compute PageRank for a single partition"""
           
           local_pagerank = {}
           
           for node in partition.nodes:
               # Get current PageRank value
               current_pr = global_pagerank.get(node, 0.0)
               
               # Compute contribution from neighbors
               neighbor_contribution = 0.0
               for neighbor in partition.predecessors(node):
                   neighbor_degree = len(list(partition.neighbors(neighbor)))
                   if neighbor_degree > 0:
                       neighbor_contribution += global_pagerank.get(neighbor, 0.0) / neighbor_degree
               
               # Update PageRank
               new_pr = (1 - alpha) / partition.node_count() + alpha * neighbor_contribution
               local_pagerank[node] = new_pr
               
               # Send messages for cross-partition edges
               self._send_cross_partition_messages(node, new_pr, partition, partition_id)
           
           return local_pagerank

Streaming Algorithms
-------------------

Online Graph Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StreamingGraphAnalyzer:
       def __init__(self, window_size=100000, slide_size=10000):
           self.window_size = window_size
           self.slide_size = slide_size
           self.current_graph = gr.Graph()
           self.edge_buffer = []
           self.analysis_history = []
           
       def process_edge_stream(self, edge_iterator, analysis_functions):
           """Process streaming edges with sliding window analysis"""
           
           for edge_data in edge_iterator:
               source, target, attrs = edge_data
               
               # Add edge to current graph
               if not self.current_graph.has_node(source):
                   self.current_graph.add_node(source)
               if not self.current_graph.has_node(target):
                   self.current_graph.add_node(target)
               
               self.current_graph.add_edge(source, target, **attrs)
               self.edge_buffer.append((source, target))
               
               # Check if window is full
               if len(self.edge_buffer) >= self.window_size:
                   # Perform analysis
                   analysis_results = {}
                   for name, func in analysis_functions.items():
                       analysis_results[name] = func(self.current_graph)
                   
                   self.analysis_history.append(analysis_results)
                   
                   # Slide window
                   self._slide_window()
       
       def _slide_window(self):
           """Slide the analysis window"""
           
           # Remove oldest edges
           edges_to_remove = self.edge_buffer[:self.slide_size]
           
           for source, target in edges_to_remove:
               if self.current_graph.has_edge(source, target):
                   self.current_graph.remove_edge(source, target)
           
           # Remove isolated nodes
           isolated_nodes = [node for node in self.current_graph.nodes 
                            if self.current_graph.degree(node) == 0]
           
           for node in isolated_nodes:
               self.current_graph.remove_node(node)
           
           # Update buffer
           self.edge_buffer = self.edge_buffer[self.slide_size:]

Reservoir Sampling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import random

   class ReservoirSampler:
       def __init__(self, reservoir_size=10000):
           self.reservoir_size = reservoir_size
           self.reservoir = []
           self.count = 0
           
       def add_edge(self, source, target, attrs=None):
           """Add edge to reservoir using reservoir sampling"""
           
           self.count += 1
           edge = (source, target, attrs or {})
           
           if len(self.reservoir) < self.reservoir_size:
               # Reservoir not full, add edge
               self.reservoir.append(edge)
           else:
               # Reservoir full, maybe replace random edge
               j = random.randint(1, self.count)
               if j <= self.reservoir_size:
                   self.reservoir[j - 1] = edge
       
       def get_sampled_graph(self):
           """Get graph from sampled edges"""
           
           g = gr.Graph()
           
           for source, target, attrs in self.reservoir:
               if not g.has_node(source):
                   g.add_node(source)
               if not g.has_node(target):
                   g.add_node(target)
               
               g.add_edge(source, target, **attrs)
           
           return g

   def streaming_graph_analysis(edge_stream):
       """Analyze streaming graph using reservoir sampling"""
       
       sampler = ReservoirSampler(reservoir_size=50000)
       
       # Process stream
       for edge_data in edge_stream:
           sampler.add_edge(*edge_data)
       
       # Analyze sample
       sample_graph = sampler.get_sampled_graph()
       
       results = {
           'node_count': sample_graph.node_count(),
           'edge_count': sample_graph.edge_count(),
           'density': sample_graph.density(),
           'clustering': sample_graph.clustering(),
           'components': len(sample_graph.connected_components()),
       }
       
       # Scale results to estimate full graph properties
       scale_factor = sampler.count / sampler.reservoir_size
       results['estimated_edges'] = results['edge_count'] * scale_factor
       
       return results

Approximation Algorithms
-----------------------

Sketching Techniques
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from collections import defaultdict

   class GraphSketcher:
       def __init__(self, sketch_size=1000):
           self.sketch_size = sketch_size
           self.node_sketches = defaultdict(lambda: np.zeros(sketch_size))
           self.hash_functions = self._generate_hash_functions()
           
       def _generate_hash_functions(self):
           """Generate hash functions for sketching"""
           
           hash_functions = []
           for i in range(self.sketch_size):
               # Simple hash function family
               a = np.random.randint(1, 1000000)
               b = np.random.randint(0, 1000000)
               hash_functions.append((a, b))
           
           return hash_functions
       
       def add_edge(self, source, target):
           """Add edge to sketches"""
           
           # Update sketches for both nodes
           self._update_sketch(source, target)
           self._update_sketch(target, source)
       
       def _update_sketch(self, node, neighbor):
           """Update sketch for a node"""
           
           neighbor_hash = hash(neighbor)
           
           for i, (a, b) in enumerate(self.hash_functions):
               hash_value = (a * neighbor_hash + b) % 2
               self.node_sketches[node][i] = max(self.node_sketches[node][i], hash_value)
       
       def estimate_jaccard_similarity(self, node1, node2):
           """Estimate Jaccard similarity between two nodes"""
           
           sketch1 = self.node_sketches[node1]
           sketch2 = self.node_sketches[node2]
           
           # Count matching bits
           matches = np.sum(sketch1 == sketch2)
           
           # Estimate Jaccard similarity
           return matches / self.sketch_size
       
       def find_similar_nodes(self, target_node, threshold=0.5):
           """Find nodes similar to target node"""
           
           similar_nodes = []
           
           for node in self.node_sketches:
               if node == target_node:
                   continue
               
               similarity = self.estimate_jaccard_similarity(target_node, node)
               
               if similarity >= threshold:
                   similar_nodes.append((node, similarity))
           
           return sorted(similar_nodes, key=lambda x: x[1], reverse=True)

HyperLogLog for Cardinality
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import math

   class HyperLogLogCounter:
       def __init__(self, precision=12):
           self.precision = precision
           self.m = 2 ** precision
           self.buckets = [0] * self.m
           self.alpha = self._get_alpha()
           
       def _get_alpha(self):
           """Get alpha constant for HyperLogLog"""
           
           if self.m >= 128:
               return 0.7213 / (1 + 1.079 / self.m)
           elif self.m >= 64:
               return 0.709
           elif self.m >= 32:
               return 0.697
           else:
               return 0.5
       
       def add(self, value):
           """Add value to HyperLogLog counter"""
           
           # Hash the value
           hash_value = hash(value)
           
           # Get bucket index (first p bits)
           bucket_index = hash_value & (self.m - 1)
           
           # Get remaining bits
           remaining_bits = hash_value >> self.precision
           
           # Count leading zeros + 1
           leading_zeros = self._count_leading_zeros(remaining_bits) + 1
           
           # Update bucket with maximum
           self.buckets[bucket_index] = max(self.buckets[bucket_index], leading_zeros)
       
       def _count_leading_zeros(self, value):
           """Count leading zeros in binary representation"""
           
           if value == 0:
               return 32  # Assuming 32-bit hash
           
           count = 0
           for i in range(31, -1, -1):
               if value & (1 << i):
                   break
               count += 1
           
           return count
       
       def estimate_cardinality(self):
           """Estimate cardinality using HyperLogLog"""
           
           # Calculate raw estimate
           raw_estimate = self.alpha * (self.m ** 2) / sum(2 ** (-bucket) for bucket in self.buckets)
           
           # Apply small range correction
           if raw_estimate <= 2.5 * self.m:
               zeros = self.buckets.count(0)
               if zeros != 0:
                   return self.m * math.log(self.m / zeros)
           
           # Apply large range correction
           if raw_estimate <= (1.0/30.0) * (2 ** 32):
               return raw_estimate
           else:
               return -2 ** 32 * math.log(1 - raw_estimate / (2 ** 32))

   def estimate_graph_properties(edge_stream):
       """Estimate graph properties from edge stream"""
       
       node_counter = HyperLogLogCounter(precision=12)
       edge_counter = HyperLogLogCounter(precision=12)
       
       degree_estimators = defaultdict(lambda: HyperLogLogCounter(precision=8))
       
       for source, target in edge_stream:
           # Count unique nodes
           node_counter.add(source)
           node_counter.add(target)
           
           # Count unique edges
           edge_counter.add((source, target))
           
           # Estimate degrees
           degree_estimators[source].add(target)
           degree_estimators[target].add(source)
       
       # Estimate properties
       estimated_nodes = node_counter.estimate_cardinality()
       estimated_edges = edge_counter.estimate_cardinality()
       
       # Estimate average degree
       total_degree = sum(estimator.estimate_cardinality() 
                         for estimator in degree_estimators.values())
       avg_degree = total_degree / len(degree_estimators) if degree_estimators else 0
       
       return {
           'estimated_nodes': estimated_nodes,
           'estimated_edges': estimated_edges,
           'estimated_density': estimated_edges / (estimated_nodes * (estimated_nodes - 1) / 2),
           'estimated_avg_degree': avg_degree,
       }

Performance Monitoring
---------------------

Large Scale Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import psutil
   from collections import deque

   class LargeScaleMonitor:
       def __init__(self, window_size=3600):  # 1 hour window
           self.window_size = window_size
           self.metrics = {
               'throughput': deque(maxlen=window_size),
               'memory_usage': deque(maxlen=window_size),
               'cpu_usage': deque(maxlen=window_size),
               'io_usage': deque(maxlen=window_size),
           }
           self.start_time = time.time()
           self.processed_edges = 0
           
       def record_batch_processed(self, batch_size):
           """Record processing of a batch"""
           
           self.processed_edges += batch_size
           current_time = time.time()
           
           # Calculate throughput (edges per second)
           throughput = batch_size / (current_time - self.start_time)
           self.metrics['throughput'].append(throughput)
           
           # Record system metrics
           self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
           self.metrics['cpu_usage'].append(psutil.cpu_percent())
           
           io_stats = psutil.disk_io_counters()
           if io_stats:
               io_usage = (io_stats.read_bytes + io_stats.write_bytes) / (1024**2)  # MB
               self.metrics['io_usage'].append(io_usage)
           
           self.start_time = current_time
       
       def get_performance_summary(self):
           """Get performance summary"""
           
           if not self.metrics['throughput']:
               return {}
           
           return {
               'avg_throughput': sum(self.metrics['throughput']) / len(self.metrics['throughput']),
               'peak_throughput': max(self.metrics['throughput']),
               'avg_memory_usage': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']),
               'peak_memory_usage': max(self.metrics['memory_usage']),
               'total_processed': self.processed_edges,
           }

Best Practices for Large Graphs
-------------------------------

1. **Memory Management**:
   - Use memory mapping for very large graphs
   - Implement out-of-core processing for graphs that don't fit in memory
   - Monitor memory usage continuously

2. **Algorithm Selection**:
   - Use approximation algorithms for very large graphs
   - Implement streaming algorithms for dynamic graphs
   - Consider distributed algorithms for massive graphs

3. **Data Structures**:
   - Use compressed representations when possible
   - Implement efficient indexing for fast access
   - Consider specialized storage formats

4. **Processing Strategies**:
   - Partition graphs for parallel processing
   - Use sampling for exploratory analysis
   - Implement incremental updates for dynamic graphs

5. **Monitoring and Optimization**:
   - Implement comprehensive performance monitoring
   - Profile critical code paths regularly
   - Optimize based on actual workload characteristics

These techniques enable Groggy to handle graphs of virtually any size, from small interactive networks to massive web-scale graphs with billions of nodes and edges.
Custom Algorithm Development
============================

This guide covers how to develop, implement, and integrate custom graph algorithms into Groggy, including both Python and Rust implementations.

Algorithm Framework Overview
----------------------------

Groggy provides a flexible framework for implementing custom algorithms that can leverage the full performance capabilities of the Rust core while maintaining Python accessibility.

Algorithm Interface
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from abc import ABC, abstractmethod
   import groggy as gr

   class GraphAlgorithm(ABC):
       """Base class for custom graph algorithms"""
       
       def __init__(self, name: str):
           self.name = name
           self.parallel = False
           self.approximate = False
           
       @abstractmethod
       def execute(self, graph: gr.Graph, **kwargs):
           """Execute the algorithm on a graph"""
           pass
       
       def validate_input(self, graph: gr.Graph, **kwargs):
           """Validate algorithm input parameters"""
           if graph.node_count() == 0:
               raise ValueError("Cannot run algorithm on empty graph")
       
       def preprocess(self, graph: gr.Graph, **kwargs):
           """Preprocessing step before algorithm execution"""
           pass
       
       def postprocess(self, result, graph: gr.Graph, **kwargs):
           """Postprocessing step after algorithm execution"""
           return result

Python Algorithm Implementation
------------------------------

Simple Custom Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class EccentricityCentrality(GraphAlgorithm):
       """Calculate eccentricity centrality for each node"""
       
       def __init__(self):
           super().__init__("eccentricity_centrality")
           
       def validate_input(self, graph: gr.Graph, **kwargs):
           super().validate_input(graph, **kwargs)
           if not graph.is_connected():
               raise ValueError("Graph must be connected for eccentricity centrality")
       
       def execute(self, graph: gr.Graph, **kwargs):
           """Calculate eccentricity centrality"""
           
           eccentricity = {}
           
           # For each node, find maximum shortest path distance
           for node in graph.nodes:
               distances = graph.single_source_shortest_paths(node)
               
               # Eccentricity is the maximum distance
               max_distance = max(distances.values())
               eccentricity[node] = max_distance
           
           # Centrality is inverse of eccentricity
           max_eccentricity = max(eccentricity.values())
           centrality = {
               node: max_eccentricity - ecc + 1 
               for node, ecc in eccentricity.items()
           }
           
           return centrality

   # Usage
   def test_eccentricity_centrality():
       g = gr.Graph()
       g.add_nodes(['A', 'B', 'C', 'D'])
       g.add_edges([('A', 'B'), ('B', 'C'), ('C', 'D')])
       
       algorithm = EccentricityCentrality()
       result = algorithm.execute(g)
       
       print("Eccentricity Centrality:", result)

Advanced Algorithm with Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from collections import defaultdict, deque

   class CustomCommunityDetection(GraphAlgorithm):
       """Custom community detection using spectral clustering"""
       
       def __init__(self, k_communities=None, max_iter=100):
           super().__init__("spectral_communities")
           self.k_communities = k_communities
           self.max_iter = max_iter
           self.parallel = True
           
       def execute(self, graph: gr.Graph, **kwargs):
           """Execute spectral community detection"""
           
           # Step 1: Build Laplacian matrix
           laplacian = self._build_laplacian_matrix(graph)
           
           # Step 2: Compute eigenvalues and eigenvectors
           eigenvalues, eigenvectors = self._compute_eigenvectors(laplacian)
           
           # Step 3: Determine number of communities
           k = self._determine_k(eigenvalues) if self.k_communities is None else self.k_communities
           
           # Step 4: K-means clustering on eigenvectors
           communities = self._kmeans_clustering(eigenvectors[:, :k], k)
           
           # Step 5: Map back to node communities
           node_communities = self._map_to_node_communities(graph, communities)
           
           return node_communities
       
       def _build_laplacian_matrix(self, graph):
           """Build normalized Laplacian matrix"""
           
           # Get adjacency matrix
           adj_matrix = graph.adjacency().to_numpy()
           
           # Compute degree matrix
           degrees = np.sum(adj_matrix, axis=1)
           degree_matrix = np.diag(degrees)
           
           # Compute Laplacian: L = D - A
           laplacian = degree_matrix - adj_matrix
           
           # Normalize: L_norm = D^(-1/2) * L * D^(-1/2)
           degree_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-8))  # Add small epsilon
           normalized_laplacian = degree_sqrt_inv @ laplacian @ degree_sqrt_inv
           
           return normalized_laplacian
       
       def _compute_eigenvectors(self, laplacian):
           """Compute eigenvalues and eigenvectors"""
           
           # Use sparse eigensolver for large matrices
           if laplacian.shape[0] > 1000:
               from scipy.sparse.linalg import eigsh
               from scipy.sparse import csr_matrix
               
               sparse_laplacian = csr_matrix(laplacian)
               k = min(50, laplacian.shape[0] - 1)  # Number of eigenvalues to compute
               
               eigenvalues, eigenvectors = eigsh(sparse_laplacian, k=k, which='SM')
           else:
               # Dense eigensolver for small matrices
               eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
           
           # Sort by eigenvalue
           sort_indices = np.argsort(eigenvalues)
           eigenvalues = eigenvalues[sort_indices]
           eigenvectors = eigenvectors[:, sort_indices]
           
           return eigenvalues, eigenvectors
       
       def _determine_k(self, eigenvalues):
           """Determine number of communities using eigengap heuristic"""
           
           # Find largest gap in eigenvalues
           gaps = np.diff(eigenvalues)
           max_gap_index = np.argmax(gaps)
           
           # Number of communities is position of largest gap + 1
           k = max_gap_index + 2
           
           # Bound k to reasonable range
           k = max(2, min(k, len(eigenvalues) // 2))
           
           return k
       
       def _kmeans_clustering(self, embeddings, k):
           """K-means clustering on eigenvector embeddings"""
           
           from sklearn.cluster import KMeans
           
           # Normalize embeddings
           norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
           normalized_embeddings = embeddings / (norms + 1e-8)
           
           # K-means clustering
           kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
           communities = kmeans.fit_predict(normalized_embeddings)
           
           return communities
       
       def _map_to_node_communities(self, graph, communities):
           """Map cluster assignments back to node communities"""
           
           node_list = list(graph.nodes)
           node_communities = []
           
           # Group nodes by community
           community_map = defaultdict(list)
           for i, community_id in enumerate(communities):
               community_map[community_id].append(node_list[i])
           
           # Convert to list of communities
           for community_id in sorted(community_map.keys()):
               node_communities.append(community_map[community_id])
           
           return node_communities

Parallel Algorithm Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import multiprocessing as mp
   from concurrent.futures import ThreadPoolExecutor, as_completed

   class ParallelTriangleCounting(GraphAlgorithm):
       """Parallel triangle counting algorithm"""
       
       def __init__(self, num_threads=None):
           super().__init__("parallel_triangle_counting")
           self.num_threads = num_threads or mp.cpu_count()
           self.parallel = True
           
       def execute(self, graph: gr.Graph, **kwargs):
           """Execute parallel triangle counting"""
           
           # Get all nodes
           nodes = list(graph.nodes)
           
           # Partition nodes for parallel processing
           partitions = self._partition_nodes(nodes, self.num_threads)
           
           # Process partitions in parallel
           with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
               futures = []
               
               for partition in partitions:
                   future = executor.submit(self._count_triangles_partition, graph, partition)
                   futures.append(future)
               
               # Collect results
               total_triangles = 0
               node_triangle_counts = {}
               
               for future in as_completed(futures):
                   partition_triangles, partition_node_counts = future.result()
                   total_triangles += partition_triangles
                   node_triangle_counts.update(partition_node_counts)
           
           # Correct for triple counting (each triangle counted 3 times)
           total_triangles //= 3
           
           return {
               'total_triangles': total_triangles,
               'node_triangle_counts': node_triangle_counts
           }
       
       def _partition_nodes(self, nodes, num_partitions):
           """Partition nodes for parallel processing"""
           
           partition_size = len(nodes) // num_partitions
           partitions = []
           
           for i in range(num_partitions):
               start_idx = i * partition_size
               end_idx = start_idx + partition_size if i < num_partitions - 1 else len(nodes)
               partitions.append(nodes[start_idx:end_idx])
           
           return partitions
       
       def _count_triangles_partition(self, graph, node_partition):
           """Count triangles for a partition of nodes"""
           
           partition_triangles = 0
           node_counts = {}
           
           for node in node_partition:
               neighbors = set(graph.neighbors(node))
               node_triangles = 0
               
               # Check all pairs of neighbors
               neighbor_list = list(neighbors)
               for i, neighbor1 in enumerate(neighbor_list):
                   for neighbor2 in neighbor_list[i+1:]:
                       if graph.has_edge(neighbor1, neighbor2):
                           node_triangles += 1
                           partition_triangles += 1
               
               node_counts[node] = node_triangles
           
           return partition_triangles, node_counts

Rust Algorithm Integration
--------------------------

For maximum performance, algorithms can be implemented in Rust and exposed through the FFI layer.

Rust Algorithm Template
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   // In rust_algorithms.rs
   use pyo3::prelude::*;
   use std::collections::HashMap;
   use crate::graph::GraphCore;
   
   #[pyfunction]
   pub fn rust_betweenness_centrality(
       py: Python,
       graph: &PyAny,
       normalized: bool
   ) -> PyResult<PyObject> {
       // Extract Rust graph core
       let graph_core = extract_graph_core(graph)?;
       
       // Release GIL for computation
       let result = py.allow_threads(|| {
           compute_betweenness_centrality(&graph_core, normalized)
       })?;
       
       // Convert result to Python
       let py_dict = PyDict::new(py);
       for (node_id, centrality) in result {
           py_dict.set_item(node_id, centrality)?;
       }
       
       Ok(py_dict.into())
   }
   
   fn compute_betweenness_centrality(
       graph: &GraphCore,
       normalized: bool
   ) -> Result<HashMap<String, f64>, String> {
       let mut betweenness = HashMap::new();
       let node_count = graph.node_count();
       
       // Initialize betweenness scores
       for node_index in graph.node_indices() {
           let node_id = graph.get_node_id(node_index)?;
           betweenness.insert(node_id, 0.0);
       }
       
       // Brandes algorithm implementation
       for source_index in graph.node_indices() {
           let source_id = graph.get_node_id(source_index)?;
           
           // Single-source shortest paths
           let (distances, predecessors, sigma) = single_source_shortest_paths(graph, source_index)?;
           
           // Accumulation phase
           let mut delta = HashMap::new();
           for node_index in graph.node_indices() {
               let node_id = graph.get_node_id(node_index)?;
               delta.insert(node_id.clone(), 0.0);
           }
           
           // Process nodes in order of decreasing distance
           let mut nodes_by_distance: Vec<_> = distances.iter().collect();
           nodes_by_distance.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
           
           for (node_index, _) in nodes_by_distance {
               let node_id = graph.get_node_id(*node_index)?;
               
               if let Some(preds) = predecessors.get(node_index) {
                   for pred_index in preds {
                       let pred_id = graph.get_node_id(*pred_index)?;
                       let sigma_pred = sigma.get(pred_index).unwrap_or(&0.0);
                       let sigma_node = sigma.get(node_index).unwrap_or(&0.0);
                       
                       if *sigma_node > 0.0 {
                           let contribution = (sigma_pred / sigma_node) * (1.0 + delta[&node_id]);
                           *delta.get_mut(&pred_id).unwrap() += contribution;
                       }
                   }
               }
               
               // Accumulate betweenness (skip source node)
               if node_id != source_id {
                   *betweenness.get_mut(&node_id).unwrap() += delta[&node_id];
               }
           }
       }
       
       // Normalize if requested
       if normalized && node_count > 2 {
           let normalization_factor = if graph.is_directed() {
               (node_count - 1) * (node_count - 2)
           } else {
               (node_count - 1) * (node_count - 2) / 2
           } as f64;
           
           for centrality in betweenness.values_mut() {
               *centrality /= normalization_factor;
           }
       }
       
       Ok(betweenness)
   }

Python Wrapper for Rust Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy.rust_algorithms as rust_alg

   class RustBetweennessCentrality(GraphAlgorithm):
       """Python wrapper for Rust betweenness centrality"""
       
       def __init__(self, normalized=True):
           super().__init__("rust_betweenness_centrality")
           self.normalized = normalized
           self.parallel = True  # Rust implementation is parallel
           
       def execute(self, graph: gr.Graph, **kwargs):
           """Execute Rust betweenness centrality"""
           
           # Validate input
           self.validate_input(graph, **kwargs)
           
           # Call Rust implementation
           result = rust_alg.rust_betweenness_centrality(
               graph._graph_core,  # Access internal Rust graph
               self.normalized
           )
           
           return result

Algorithm Registration System
-----------------------------

Dynamic Algorithm Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AlgorithmRegistry:
       """Registry for custom algorithms"""
       
       def __init__(self):
           self.algorithms = {}
           self.categories = defaultdict(list)
           
       def register(self, algorithm: GraphAlgorithm, category: str = "custom"):
           """Register a custom algorithm"""
           
           self.algorithms[algorithm.name] = algorithm
           self.categories[category].append(algorithm.name)
           
       def get_algorithm(self, name: str) -> GraphAlgorithm:
           """Get algorithm by name"""
           
           if name not in self.algorithms:
               raise ValueError(f"Algorithm '{name}' not found")
           
           return self.algorithms[name]
       
       def list_algorithms(self, category: str = None) -> list:
           """List available algorithms"""
           
           if category is None:
               return list(self.algorithms.keys())
           else:
               return self.categories.get(category, [])
       
       def execute_algorithm(self, name: str, graph: gr.Graph, **kwargs):
           """Execute algorithm by name"""
           
           algorithm = self.get_algorithm(name)
           return algorithm.execute(graph, **kwargs)

   # Global algorithm registry
   algorithm_registry = AlgorithmRegistry()

   # Register custom algorithms
   algorithm_registry.register(EccentricityCentrality(), "centrality")
   algorithm_registry.register(CustomCommunityDetection(), "community")
   algorithm_registry.register(ParallelTriangleCounting(), "structural")

   # Usage
   def run_custom_algorithm():
       g = create_test_graph()
       
       # List available algorithms
       print("Available centrality algorithms:", 
             algorithm_registry.list_algorithms("centrality"))
       
       # Execute custom algorithm
       result = algorithm_registry.execute_algorithm("eccentricity_centrality", g)
       print("Eccentricity centrality:", result)

Algorithm Composition
--------------------

Pipeline Construction
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AlgorithmPipeline:
       """Pipeline for composing multiple algorithms"""
       
       def __init__(self, name: str):
           self.name = name
           self.steps = []
           self.results = {}
           
       def add_step(self, algorithm: GraphAlgorithm, depends_on=None, **kwargs):
           """Add algorithm step to pipeline"""
           
           step = {
               'algorithm': algorithm,
               'depends_on': depends_on or [],
               'kwargs': kwargs,
               'result_key': algorithm.name
           }
           
           self.steps.append(step)
           return self
       
       def execute(self, graph: gr.Graph):
           """Execute the algorithm pipeline"""
           
           self.results = {}
           
           for step in self.steps:
               algorithm = step['algorithm']
               depends_on = step['depends_on']
               kwargs = step['kwargs'].copy()
               
               # Check dependencies
               for dep in depends_on:
                   if dep not in self.results:
                       raise ValueError(f"Dependency '{dep}' not found")
                   
                   # Add dependency results to kwargs
                   kwargs[f"_{dep}_result"] = self.results[dep]
               
               # Execute algorithm
               result = algorithm.execute(graph, **kwargs)
               self.results[step['result_key']] = result
           
           return self.results

   # Example pipeline
   def create_centrality_pipeline():
       pipeline = AlgorithmPipeline("centrality_analysis")
       
       # Add algorithms to pipeline
       pipeline.add_step(
           EccentricityCentrality()
       ).add_step(
           RustBetweennessCentrality(normalized=True)
       ).add_step(
           CustomCommunityDetection(k_communities=5)
       )
       
       return pipeline

   # Usage
   def run_pipeline_analysis():
       g = create_large_test_graph()
       
       pipeline = create_centrality_pipeline()
       results = pipeline.execute(g)
       
       print("Pipeline results:")
       for algorithm_name, result in results.items():
           print(f"  {algorithm_name}: {type(result)}")

Performance Optimization
-----------------------

Algorithm Profiling
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import cProfile
   import pstats

   class AlgorithmProfiler:
       """Profiler for custom algorithms"""
       
       def __init__(self):
           self.profiles = {}
           
       def profile_algorithm(self, algorithm: GraphAlgorithm, graph: gr.Graph, **kwargs):
           """Profile algorithm execution"""
           
           # Setup profiler
           profiler = cProfile.Profile()
           
           # Run with timing
           start_time = time.time()
           profiler.enable()
           
           result = algorithm.execute(graph, **kwargs)
           
           profiler.disable()
           end_time = time.time()
           
           # Store results
           execution_time = end_time - start_time
           
           self.profiles[algorithm.name] = {
               'execution_time': execution_time,
               'profiler': profiler,
               'result': result
           }
           
           return result
       
       def get_performance_report(self, algorithm_name: str):
           """Get performance report for algorithm"""
           
           if algorithm_name not in self.profiles:
               raise ValueError(f"No profile found for {algorithm_name}")
           
           profile_data = self.profiles[algorithm_name]
           stats = pstats.Stats(profile_data['profiler'])
           
           print(f"\nPerformance Report for {algorithm_name}")
           print("=" * 50)
           print(f"Total execution time: {profile_data['execution_time']:.3f} seconds")
           print("\nTop 10 time-consuming functions:")
           stats.sort_stats('cumulative').print_stats(10)

Caching Strategy
~~~~~~~~~~~~~~~

.. code-block:: python

   import hashlib
   import pickle
   from functools import wraps

   def cache_algorithm_result(cache_dir="/tmp/groggy_cache"):
       """Decorator to cache algorithm results"""
       
       def decorator(algorithm_execute_method):
           @wraps(algorithm_execute_method)
           def wrapper(self, graph, **kwargs):
               # Create cache key
               graph_hash = graph.structure_hash()
               kwargs_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()
               cache_key = f"{self.name}_{graph_hash}_{kwargs_hash}"
               
               cache_file = os.path.join(cache_dir, f"{cache_key}.cache")
               
               # Check if cached result exists
               if os.path.exists(cache_file):
                   try:
                       with open(cache_file, 'rb') as f:
                           cached_result = pickle.load(f)
                           print(f"Loaded cached result for {self.name}")
                           return cached_result
                   except:
                       # Cache corrupted, remove file
                       os.remove(cache_file)
               
               # Execute algorithm
               result = algorithm_execute_method(self, graph, **kwargs)
               
               # Cache result
               os.makedirs(cache_dir, exist_ok=True)
               try:
                   with open(cache_file, 'wb') as f:
                       pickle.dump(result, f)
               except:
                   # Failed to cache, continue anyway
                   pass
               
               return result
           
           return wrapper
       
       return decorator

   # Usage with caching
   class CachedEccentricityCentrality(EccentricityCentrality):
       @cache_algorithm_result()
       def execute(self, graph: gr.Graph, **kwargs):
           return super().execute(graph, **kwargs)

Testing Framework
----------------

Algorithm Testing
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import unittest
   import numpy as np

   class AlgorithmTestCase(unittest.TestCase):
       """Base test case for custom algorithms"""
       
       def setUp(self):
           """Set up test graphs"""
           
           # Simple path graph
           self.path_graph = gr.Graph()
           self.path_graph.add_nodes(['A', 'B', 'C', 'D'])
           self.path_graph.add_edges([('A', 'B'), ('B', 'C'), ('C', 'D')])
           
           # Complete graph
           self.complete_graph = gr.Graph()
           nodes = ['A', 'B', 'C', 'D']
           self.complete_graph.add_nodes(nodes)
           for i, node1 in enumerate(nodes):
               for node2 in nodes[i+1:]:
                   self.complete_graph.add_edge(node1, node2)
           
           # Random graph
           self.random_graph = gr.random_graph(50, edge_probability=0.1, seed=42)
       
       def test_algorithm_properties(self, algorithm: GraphAlgorithm):
           """Test basic algorithm properties"""
           
           # Test on empty graph (should fail)
           empty_graph = gr.Graph()
           with self.assertRaises(ValueError):
               algorithm.execute(empty_graph)
           
           # Test on simple graphs
           result_path = algorithm.execute(self.path_graph)
           result_complete = algorithm.execute(self.complete_graph)
           
           # Basic sanity checks
           self.assertIsNotNone(result_path)
           self.assertIsNotNone(result_complete)
           
           # Check result structure (depends on algorithm)
           if isinstance(result_path, dict):
               self.assertTrue(len(result_path) > 0)
               for node in self.path_graph.nodes:
                   self.assertIn(node, result_path)

   class TestCustomAlgorithms(AlgorithmTestCase):
       """Test custom algorithm implementations"""
       
       def test_eccentricity_centrality(self):
           """Test eccentricity centrality algorithm"""
           
           algorithm = EccentricityCentrality()
           
           # Test basic properties
           self.test_algorithm_properties(algorithm)
           
           # Test specific behavior on path graph
           result = algorithm.execute(self.path_graph)
           
           # In a path graph, middle nodes should have higher centrality
           self.assertGreater(result['B'], result['A'])
           self.assertGreater(result['C'], result['A'])
           self.assertEqual(result['B'], result['C'])  # Should be equal
       
       def test_triangle_counting(self):
           """Test triangle counting algorithm"""
           
           algorithm = ParallelTriangleCounting()
           
           # Test on complete graph (known number of triangles)
           result = algorithm.execute(self.complete_graph)
           
           # Complete graph with 4 nodes should have 4 triangles
           expected_triangles = 4  # C(4,3) = 4
           self.assertEqual(result['total_triangles'], expected_triangles)
       
       def test_spectral_communities(self):
           """Test spectral community detection"""
           
           algorithm = CustomCommunityDetection(k_communities=2)
           
           # Create graph with clear community structure
           community_graph = gr.Graph()
           
           # Community 1
           community_graph.add_nodes(['A1', 'A2', 'A3'])
           community_graph.add_edges([('A1', 'A2'), ('A2', 'A3'), ('A3', 'A1')])
           
           # Community 2
           community_graph.add_nodes(['B1', 'B2', 'B3'])
           community_graph.add_edges([('B1', 'B2'), ('B2', 'B3'), ('B3', 'B1')])
           
           # Bridge between communities
           community_graph.add_edge('A1', 'B1')
           
           communities = algorithm.execute(community_graph)
           
           # Should find 2 communities
           self.assertEqual(len(communities), 2)
           
           # Check that nodes from same original community are grouped together
           community_sets = [set(community) for community in communities]
           a_nodes = {'A1', 'A2', 'A3'}
           b_nodes = {'B1', 'B2', 'B3'}
           
           # One community should contain mostly A nodes, other mostly B nodes
           a_overlap = [len(cs.intersection(a_nodes)) for cs in community_sets]
           b_overlap = [len(cs.intersection(b_nodes)) for cs in community_sets]
           
           self.assertTrue(max(a_overlap) >= 2)  # At least 2 A nodes together
           self.assertTrue(max(b_overlap) >= 2)  # At least 2 B nodes together

   # Run tests
   if __name__ == '__main__':
       unittest.main()

This comprehensive framework enables developers to create, optimize, and integrate custom algorithms that leverage Groggy's high-performance infrastructure while maintaining the flexibility and ease of use that Python provides.
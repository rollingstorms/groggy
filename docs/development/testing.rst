Testing Framework
=================

Groggy uses a comprehensive testing framework to ensure reliability, performance, and correctness across all components.

Test Organization
-----------------

Test Structure
~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── unit/                    # Unit tests
   │   ├── rust/               # Rust unit tests
   │   │   ├── core/          # Core algorithm tests
   │   │   ├── ffi/           # FFI layer tests
   │   │   └── utils/         # Utility function tests
   │   └── python/            # Python unit tests
   │       ├── test_graph.py  # Graph class tests
   │       ├── test_storage.py # Storage view tests
   │       └── test_algorithms.py # Algorithm tests
   ├── integration/            # Integration tests
   │   ├── test_workflows.py  # End-to-end workflows
   │   ├── test_memory.py     # Memory management tests
   │   └── test_performance.py # Performance regression tests
   ├── benchmarks/            # Performance benchmarks
   │   ├── test_centrality.py # Centrality algorithm benchmarks
   │   └── test_storage.py    # Storage operation benchmarks
   ├── fixtures/              # Test data and utilities
   │   ├── graphs/           # Sample graph files
   │   └── data/             # Test datasets
   └── conftest.py           # Pytest configuration

Test Categories
~~~~~~~~~~~~~~

**Unit Tests**: Test individual functions and classes in isolation
**Integration Tests**: Test component interactions and workflows
**Performance Tests**: Verify performance characteristics and catch regressions
**Property Tests**: Use property-based testing for algorithm correctness
**Stress Tests**: Test system behavior under extreme conditions

Running Tests
------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   python -m pytest

   # Run with verbose output
   python -m pytest -v

   # Run specific test file
   python -m pytest tests/unit/python/test_graph.py

   # Run specific test function
   python -m pytest tests/unit/python/test_graph.py::TestGraph::test_add_node

   # Run tests matching pattern
   python -m pytest -k "pagerank"

Test Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run with coverage
   python -m pytest --cov=groggy --cov-report=html

   # Run in parallel
   python -m pytest -n auto

   # Run only fast tests (skip slow ones)
   python -m pytest -m "not slow"

   # Run with specific Python version
   python3.9 -m pytest

Rust Tests
~~~~~~~~~

.. code-block:: bash

   # Run all Rust tests
   cargo test

   # Run specific test module
   cargo test graph_core

   # Run with output
   cargo test -- --nocapture

   # Run release mode tests
   cargo test --release

   # Run with specific features
   cargo test --features "parallel,simd"

Writing Unit Tests
-----------------

Python Unit Tests
~~~~~~~~~~~~~~~~

**Basic Test Structure**:

.. code-block:: python

   import pytest
   import groggy as gr

   class TestGraph:
       """Test Graph class functionality"""
       
       def test_empty_graph(self):
           """Test creating empty graph"""
           g = gr.Graph()
           assert g.node_count() == 0
           assert g.edge_count() == 0
           assert g.density() == 0.0
       
       def test_add_node(self):
           """Test adding nodes"""
           g = gr.Graph()
           g.add_node("alice", age=30, role="engineer")
           
           assert g.node_count() == 1
           assert g.has_node("alice")
           assert g.nodes["alice"]["age"] == 30
       
       def test_add_duplicate_node(self):
           """Test adding duplicate node raises error"""
           g = gr.Graph()
           g.add_node("alice")
           
           with pytest.raises(ValueError, match="already exists"):
               g.add_node("alice")

**Parametrized Tests**:

.. code-block:: python

   @pytest.mark.parametrize("directed", [True, False])
   def test_graph_types(self, directed):
       """Test both directed and undirected graphs"""
       g = gr.Graph(directed=directed)
       g.add_nodes(["A", "B", "C"])
       g.add_edge("A", "B")
       
       assert g.directed == directed
       if directed:
           assert not g.has_edge("B", "A")
       else:
           assert g.has_edge("B", "A")

   @pytest.mark.parametrize("size,density", [
       (10, 0.1),
       (100, 0.01),
       (1000, 0.001),
   ])
   def test_random_graphs(self, size, density):
       """Test random graph generation"""
       g = gr.random_graph(size, edge_probability=density)
       
       assert g.node_count() == size
       expected_edges = size * (size - 1) * density / 2
       assert abs(g.edge_count() - expected_edges) < expected_edges * 0.5

**Fixture Usage**:

.. code-block:: python

   @pytest.fixture
   def simple_graph():
       """Create simple test graph"""
       g = gr.Graph()
       g.add_nodes(["A", "B", "C", "D"])
       g.add_edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
       return g

   @pytest.fixture
   def complete_graph():
       """Create complete graph"""
       g = gr.Graph()
       nodes = ["A", "B", "C", "D"]
       g.add_nodes(nodes)
       
       for i, node1 in enumerate(nodes):
           for node2 in nodes[i+1:]:
               g.add_edge(node1, node2)
       
       return g

   def test_centrality_on_complete_graph(self, complete_graph):
       """Test centrality on complete graph"""
       centrality = complete_graph.centrality.pagerank()
       
       # All nodes should have equal centrality
       values = list(centrality.values())
       assert all(abs(v - values[0]) < 1e-6 for v in values)

Rust Unit Tests
~~~~~~~~~~~~~~

**Basic Test Structure**:

.. code-block:: rust

   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::graph::GraphCore;

       #[test]
       fn test_empty_graph() {
           let graph = GraphCore::new(true);
           assert_eq!(graph.node_count(), 0);
           assert_eq!(graph.edge_count(), 0);
           assert_eq!(graph.density(), 0.0);
       }

       #[test]
       fn test_add_node() {
           let mut graph = GraphCore::new(true);
           let node_id = "alice".to_string();
           
           let index = graph.add_node(node_id.clone()).unwrap();
           
           assert_eq!(graph.node_count(), 1);
           assert!(graph.has_node(&node_id));
           assert_eq!(graph.get_node_id(index).unwrap(), node_id);
       }

       #[test]
       #[should_panic(expected = "Node already exists")]
       fn test_add_duplicate_node() {
           let mut graph = GraphCore::new(true);
           let node_id = "alice".to_string();
           
           graph.add_node(node_id.clone()).unwrap();
           graph.add_node(node_id).unwrap(); // Should panic
       }
   }

**Property-Based Tests**:

.. code-block:: rust

   use proptest::prelude::*;

   proptest! {
       #[test]
       fn test_pagerank_properties(
           nodes in prop::collection::vec(any::<u32>(), 1..100),
           edges in prop::collection::vec((any::<usize>(), any::<usize>()), 0..200)
       ) {
           let mut graph = GraphCore::new(true);
           
           // Add nodes
           for node in &nodes {
               let _ = graph.add_node(node.to_string());
           }
           
           // Add valid edges
           for (src_idx, tgt_idx) in edges {
               if src_idx < nodes.len() && tgt_idx < nodes.len() && src_idx != tgt_idx {
                   let src = nodes[src_idx].to_string();
                   let tgt = nodes[tgt_idx].to_string();
                   let _ = graph.add_edge(src, tgt);
               }
           }
           
           if graph.node_count() > 0 {
               let result = pagerank(&graph, 0.85, 100, 1e-6).unwrap();
               
               // Properties that should always hold
               assert!(result.len() == graph.node_count());
               assert!(result.values().all(|&v| v > 0.0));
               
               let sum: f64 = result.values().sum();
               assert!((sum - graph.node_count() as f64).abs() < 1e-3);
           }
       }
   }

Algorithm Testing
----------------

Correctness Tests
~~~~~~~~~~~~~~~~

.. code-block:: python

   class TestPageRank:
       """Test PageRank algorithm correctness"""
       
       def test_pagerank_simple_chain(self):
           """Test PageRank on simple chain graph"""
           g = gr.Graph()
           g.add_nodes(["A", "B", "C"])
           g.add_edges([("A", "B"), ("B", "C")])
           
           result = g.centrality.pagerank(alpha=0.85)
           
           # Node C should have highest PageRank (sink)
           assert result["C"] > result["B"] > result["A"]
       
       def test_pagerank_with_teleportation(self):
           """Test PageRank with different alpha values"""
           g = self._create_test_graph()
           
           # Higher alpha = more focus on structure
           high_alpha = g.centrality.pagerank(alpha=0.95)
           low_alpha = g.centrality.pagerank(alpha=0.5)
           
           # Results should be different
           assert high_alpha != low_alpha
       
       def test_pagerank_convergence(self):
           """Test PageRank convergence"""
           g = self._create_large_graph()
           
           # Should converge within reasonable iterations
           result = g.centrality.pagerank(max_iter=1000, tolerance=1e-8)
           
           # Check sum equals 1 (normalized)
           assert abs(sum(result.values()) - 1.0) < 1e-6

Numerical Stability Tests
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TestNumericalStability:
       """Test numerical stability of algorithms"""
       
       def test_pagerank_numerical_stability(self):
           """Test PageRank numerical stability"""
           g = self._create_test_graph()
           
           # Run multiple times with same parameters
           results = []
           for _ in range(10):
               result = g.centrality.pagerank(alpha=0.85, tolerance=1e-12)
               results.append(result)
           
           # Results should be nearly identical
           for i in range(1, len(results)):
               for node in results[0]:
                   diff = abs(results[0][node] - results[i][node])
                   assert diff < 1e-10, f"Unstable result for node {node}"
       
       def test_edge_case_handling(self):
           """Test handling of edge cases"""
           # Single node
           g1 = gr.Graph()
           g1.add_node("A")
           result1 = g1.centrality.pagerank()
           assert result1["A"] == 1.0
           
           # Disconnected components
           g2 = gr.Graph()
           g2.add_nodes(["A", "B", "C", "D"])
           g2.add_edges([("A", "B"), ("C", "D")])
           result2 = g2.centrality.pagerank()
           
           # Each component should sum to 0.5
           component1_sum = result2["A"] + result2["B"]
           component2_sum = result2["C"] + result2["D"]
           assert abs(component1_sum - 0.5) < 1e-6
           assert abs(component2_sum - 0.5) < 1e-6

Performance Testing
------------------

Benchmark Tests
~~~~~~~~~~~~~~

.. code-block:: python

   import pytest

   class TestPerformance:
       """Performance regression tests"""
       
       @pytest.mark.benchmark(group="pagerank")
       def test_pagerank_small(self, benchmark):
           """Benchmark PageRank on small graph"""
           g = gr.random_graph(1000, edge_probability=0.01, seed=42)
           
           result = benchmark(g.centrality.pagerank)
           
           assert len(result) == 1000
           assert abs(sum(result.values()) - 1.0) < 1e-6
       
       @pytest.mark.benchmark(group="pagerank")
       def test_pagerank_medium(self, benchmark):
           """Benchmark PageRank on medium graph"""
           g = gr.random_graph(10000, edge_probability=0.001, seed=42)
           
           result = benchmark(g.centrality.pagerank)
           
           assert len(result) == 10000
       
       @pytest.mark.slow
       @pytest.mark.benchmark(group="pagerank")
       def test_pagerank_large(self, benchmark):
           """Benchmark PageRank on large graph"""
           g = gr.random_graph(100000, edge_probability=0.0001, seed=42)
           
           result = benchmark(g.centrality.pagerank)
           
           assert len(result) == 100000

Memory Tests
~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   import psutil
   import os

   class TestMemoryUsage:
       """Test memory usage and leaks"""
       
       def test_memory_growth(self):
           """Test for memory leaks in repeated operations"""
           process = psutil.Process(os.getpid())
           initial_memory = process.memory_info().rss
           
           # Perform many operations
           for i in range(100):
               g = gr.random_graph(1000, edge_probability=0.01)
               result = g.centrality.pagerank()
               del g, result  # Explicit cleanup
           
           final_memory = process.memory_info().rss
           memory_growth = final_memory - initial_memory
           
           # Memory growth should be minimal (< 10MB)
           assert memory_growth < 10 * 1024 * 1024
       
       def test_large_graph_memory(self):
           """Test memory usage with large graphs"""
           tracemalloc.start()
           
           g = gr.random_graph(50000, edge_probability=0.0001)
           current, peak = tracemalloc.get_traced_memory()
           
           tracemalloc.stop()
           
           # Memory usage should be reasonable
           # Roughly 50K nodes * 100 bytes/node = 5MB
           expected_memory = 50000 * 100
           assert peak < expected_memory * 10  # Allow 10x overhead

Integration Testing
------------------

Workflow Tests
~~~~~~~~~~~~~

.. code-block:: python

   class TestWorkflows:
       """Test complete analysis workflows"""
       
       def test_social_network_analysis_workflow(self):
           """Test complete social network analysis"""
           # Load or create social network
           g = self._create_social_network()
           
           # Step 1: Basic statistics
           stats = {
               'nodes': g.node_count(),
               'edges': g.edge_count(),
               'density': g.density(),
               'connected': g.is_connected(),
           }
           
           assert stats['nodes'] > 0
           assert stats['edges'] > 0
           
           # Step 2: Centrality analysis
           centrality = {
               'pagerank': g.centrality.pagerank(),
               'betweenness': g.centrality.betweenness(),
           }
           
           assert len(centrality['pagerank']) == g.node_count()
           assert len(centrality['betweenness']) == g.node_count()
           
           # Step 3: Community detection
           communities = g.communities.louvain()
           modularity = g.communities.modularity(communities)
           
           assert len(communities) > 0
           assert modularity >= -1.0 and modularity <= 1.0
           
           # Step 4: Export results
           nodes_table = g.nodes.table()
           
           # Add centrality as attributes
           for node_id, pr in centrality['pagerank'].items():
               nodes_table.loc[node_id, 'pagerank'] = pr
           
           # Verify export
           assert 'pagerank' in nodes_table.columns
           assert nodes_table.shape[0] == g.node_count()

Storage View Tests
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class TestStorageViews:
       """Test storage view integration"""
       
       def test_array_matrix_table_consistency(self):
           """Test consistency across storage views"""
           g = self._create_test_graph_with_attributes()
           
           # Get same data through different views
           age_array = g.nodes.table()['age']
           age_from_matrix = g.nodes.table()[['age', 'salary']]['age']
           
           # Should be identical
           assert age_array.values == age_from_matrix.values
           
           # Statistical operations should match
           assert age_array.mean() == age_from_matrix.mean()
           assert age_array.std() == age_from_matrix.std()
       
       def test_lazy_evaluation_performance(self):
           """Test lazy evaluation improves performance"""
           g = self._create_large_graph_with_attributes()
           
           table = g.nodes.table()
           
           # Chained operations should be fast (lazy)
           start_time = time.time()
           filtered = table.filter_rows(lambda r: r['age'] > 30)
           sorted_filtered = filtered.sort_by('age')
           lazy_time = time.time() - start_time
           
           # Materialization should take longer
           start_time = time.time()
           result = sorted_filtered.head(10)
           materialization_time = time.time() - start_time
           
           # Lazy operations should be much faster
           assert lazy_time < 0.001  # < 1ms
           assert materialization_time > lazy_time

Test Utilities
-------------

Custom Assertions
~~~~~~~~~~~~~~~~

.. code-block:: python

   def assert_graph_equal(g1, g2, check_attributes=True):
       """Assert two graphs are equal"""
       assert g1.node_count() == g2.node_count()
       assert g1.edge_count() == g2.edge_count()
       assert g1.directed == g2.directed
       
       # Check nodes
       assert set(g1.nodes) == set(g2.nodes)
       
       if check_attributes:
           for node in g1.nodes:
               assert g1.nodes[node] == g2.nodes[node]
       
       # Check edges
       assert set(g1.edges) == set(g2.edges)
       
       if check_attributes:
           for edge in g1.edges:
               assert g1.get_edge(*edge) == g2.get_edge(*edge)

   def assert_centrality_properties(centrality, graph):
       """Assert centrality results have expected properties"""
       assert len(centrality) == graph.node_count()
       assert all(score >= 0 for score in centrality.values())
       
       # For PageRank, values should sum to 1
       if abs(sum(centrality.values()) - 1.0) < 1e-6:
           return  # Normalized PageRank
       
       # For other centralities, check reasonable bounds
       max_score = max(centrality.values())
       assert max_score <= graph.node_count()

Graph Generators
~~~~~~~~~~~~~~~

.. code-block:: python

   def create_test_graphs():
       """Create various test graphs for different scenarios"""
       graphs = {}
       
       # Path graph
       graphs['path'] = gr.Graph()
       nodes = [f'node_{i}' for i in range(10)]
       graphs['path'].add_nodes(nodes)
       for i in range(9):
           graphs['path'].add_edge(nodes[i], nodes[i+1])
       
       # Star graph
       graphs['star'] = gr.Graph()
       center = 'center'
       leaves = [f'leaf_{i}' for i in range(10)]
       graphs['star'].add_node(center)
       graphs['star'].add_nodes(leaves)
       for leaf in leaves:
           graphs['star'].add_edge(center, leaf)
       
       # Complete graph
       graphs['complete'] = gr.Graph()
       nodes = [f'node_{i}' for i in range(6)]
       graphs['complete'].add_nodes(nodes)
       for i, node1 in enumerate(nodes):
           for node2 in nodes[i+1:]:
               graphs['complete'].add_edge(node1, node2)
       
       return graphs

Continuous Integration
---------------------

GitHub Actions Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # .github/workflows/test.yml
   name: Tests

   on:
     push:
       branches: [ main, develop ]
     pull_request:
       branches: [ main ]

   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, macos-latest, windows-latest]
           python-version: [3.8, 3.9, '3.10', '3.11']
           rust-version: [stable]

       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: ${{ matrix.python-version }}
       
       - name: Set up Rust
         uses: actions-rs/toolchain@v1
         with:
           toolchain: ${{ matrix.rust-version }}
           override: true
           components: rustfmt, clippy
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install maturin pytest pytest-cov
       
       - name: Build
         run: maturin develop --release
       
       - name: Run tests
         run: |
           python -m pytest tests/ --cov=groggy --cov-report=xml
       
       - name: Run Rust tests
         run: cargo test --all
       
       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./coverage.xml

Test Coverage
~~~~~~~~~~~~

.. code-block:: bash

   # Generate coverage report
   python -m pytest --cov=groggy --cov-report=html --cov-report=term

   # View coverage report
   open htmlcov/index.html

   # Coverage configuration in pyproject.toml
   [tool.coverage.run]
   source = ["groggy"]
   omit = ["tests/*", "benchmarks/*"]

   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "raise AssertionError",
       "raise NotImplementedError",
   ]

This comprehensive testing framework ensures Groggy maintains high quality, performance, and reliability across all supported platforms and use cases.
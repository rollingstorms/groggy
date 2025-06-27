Testing
=======

GLI includes a comprehensive test suite to ensure reliability and correctness across all features and backends.

Running Tests
-------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   cd gli
   python -m pytest
   
   # Run with verbose output
   python -m pytest -v
   
   # Run specific test file
   python -m pytest tests/test_backend_parity.py
   
   # Run tests with coverage
   python -m pytest --cov=gli --cov-report=html

Test Categories
~~~~~~~~~~~~~~

GLI tests are organized into several categories:

.. code-block:: bash

   # Unit tests (fast, isolated)
   python -m pytest tests/test_node_iteration.py
   python -m pytest tests/test_edge_iteration.py
   
   # Backend parity tests
   python -m pytest tests/test_backend_parity.py
   
   # Performance tests
   python -m pytest tests/performance_benchmark.py
   
   # Stress tests
   python -m pytest tests/extreme_stress_test.py

Test Structure
--------------

The test suite follows a structured approach:

.. code-block::

   tests/
   ├── README.md                    # Test documentation
   ├── test_backend_parity.py       # Backend consistency
   ├── test_batch_simple.py         # Batch operations
   ├── test_edge_iteration.py       # Edge iteration
   ├── test_node_iteration.py       # Node iteration
   ├── test_rust_attributes.py      # Rust-specific tests
   ├── performance_benchmark.py     # Performance testing
   ├── complexity_stress_test.py    # Complexity analysis
   ├── extreme_stress_test.py       # Large-scale testing
   ├── gli_tests.ipynb             # Interactive tests
   └── gli_tutorial.ipynb          # Tutorial with tests

Backend Parity Testing
----------------------

One of GLI's key testing strategies is ensuring both backends produce identical results:

.. code-block:: python

   import pytest
   from gli import Graph
   
   @pytest.mark.parametrize("backend", ["python", "rust"])
   def test_basic_operations(backend):
       """Test basic operations work identically across backends"""
       g = Graph(backend=backend)
       
       # Add nodes
       alice = g.add_node(name="Alice", age=30)
       bob = g.add_node(name="Bob", age=25)
       
       # Add edge
       edge_id = g.add_edge(alice, bob, relationship="friends")
       
       # Verify results
       assert g.node_count() == 2
       assert g.edge_count() == 1
       
       alice_data = g.get_node(alice)
       assert alice_data["name"] == "Alice"
       assert alice_data["age"] == 30

Cross-Backend Comparison
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_cross_backend_consistency():
       """Ensure both backends produce identical results for same operations"""
       
       operations = [
           ("add_node", "alice", {"name": "Alice", "age": 30}),
           ("add_node", "bob", {"name": "Bob", "age": 25}),
           ("add_edge", "alice", "bob", {"relationship": "friends"}),
           ("update_node", "alice", {"age": 31}),
       ]
       
       # Execute on both backends
       g_python = Graph(backend="python")
       g_rust = Graph(backend="rust")
       
       for op_type, *args in operations:
           getattr(g_python, op_type)(*args)  
           getattr(g_rust, op_type)(*args)
       
       # Compare final states
       assert g_python.node_count() == g_rust.node_count()
       assert g_python.edge_count() == g_rust.edge_count()
       
       # Compare node data
       for node_id in g_python.nodes:
           python_node = g_python.get_node(node_id)
           rust_node = g_rust.get_node(node_id)
           assert python_node.attributes == rust_node.attributes

Performance Testing
-------------------

GLI includes automated performance testing to catch regressions:

Benchmark Framework
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import pytest
   from gli import Graph
   from gli.utils import create_random_graph
   
   class PerformanceBenchmark:
       def __init__(self):
           self.results = {}
       
       def time_operation(self, name, func, *args, **kwargs):
           start = time.time()
           result = func(*args, **kwargs)
           duration = time.time() - start
           self.results[name] = duration
           return result
       
       def assert_performance(self, operation, max_time):
           actual_time = self.results.get(operation)
           assert actual_time is not None, f"Operation {operation} not benchmarked"
           assert actual_time < max_time, f"{operation} took {actual_time:.3f}s, expected < {max_time}s"

   @pytest.fixture
   def benchmark():
       return PerformanceBenchmark()

   def test_node_addition_performance(benchmark):
       """Test that node addition performance meets requirements"""
       g = Graph(backend="rust")
       
       # Benchmark node addition
       benchmark.time_operation("add_1000_nodes", lambda: [
           g.add_node(f"node_{i}", value=i) for i in range(1000)
       ])
       
       # Assert performance requirement
       benchmark.assert_performance("add_1000_nodes", 0.1)  # 100ms max

Stress Testing
~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.slow
   def test_large_graph_creation():
       """Test creating large graphs doesn't crash or timeout"""
       g = Graph(backend="rust")
       
       # Create large graph with batch operations
       with g.batch_operations() as batch:
           for i in range(100000):
               batch.add_node(f"node_{i}", category=i % 10)
               if i > 0:
                   batch.add_edge(f"node_{i-1}", f"node_{i}", weight=1.0)
       
       assert g.node_count() == 100000
       assert g.edge_count() == 99999
       
       # Test queries still work
       neighbors = g.get_neighbors("node_5000")
       assert len(neighbors) <= 2  # At most previous and next node

Memory Testing
~~~~~~~~~~~~~

.. code-block:: python

   import tracemalloc
   
   def test_memory_usage():
       """Test memory usage stays within reasonable bounds"""
       tracemalloc.start()
       
       g = Graph(backend="rust")
       
       # Add substantial data
       for i in range(10000):
           g.add_node(f"node_{i}", data={"value": i, "category": i % 100})
       
       current, peak = tracemalloc.get_traced_memory()
       tracemalloc.stop()
       
       # Memory should be reasonable (adjust based on requirements)
       assert peak < 50 * 1024 * 1024  # 50MB max
       print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

Interactive Testing
------------------

GLI includes Jupyter notebooks for interactive testing:

Test Notebooks
~~~~~~~~~~~~~

The `tests/gli_tests.ipynb` notebook provides interactive testing capabilities:

.. code-block:: python

   # Cell 1: Setup
   from gli import Graph, set_backend
   import time
   
   # Test both backends
   set_backend('rust')
   g_rust = Graph()
   
   set_backend('python') 
   g_python = Graph()

   # Cell 2: Performance comparison
   def benchmark_backends():
       backends = [('rust', g_rust), ('python', g_python)]
       results = {}
       
       for backend_name, graph in backends:
           start = time.time()
           
           # Add 1000 nodes
           for i in range(1000):
               graph.add_node(f"node_{i}", value=i)
           
           duration = time.time() - start
           results[backend_name] = duration
           
           # Clear graph for next test
           graph.clear()
       
       return results
   
   results = benchmark_backends()
   for backend, duration in results.items():
       print(f"{backend}: {duration:.3f}s")

Debugging Tests
~~~~~~~~~~~~~~

The `tests/debug_delta.py` provides debugging utilities:

.. code-block:: python

   def debug_graph_state(graph, operation_name):
       """Debug helper to inspect graph state"""
       print(f"\\n=== {operation_name} ===")
       print(f"Nodes: {graph.node_count()}")
       print(f"Edges: {graph.edge_count()}")
       print(f"Backend: {graph.backend}")
       
       # Sample some nodes
       nodes = list(graph.nodes)[:5]
       for node_id in nodes:
           node_data = graph.get_node(node_id)
           print(f"  {node_id}: {dict(node_data.attributes)}")

   # Usage in tests
   def test_with_debugging():
       g = Graph()
       debug_graph_state(g, "Initial state")
       
       g.add_node("test", value=42)
       debug_graph_state(g, "After adding node")

Test Configuration
-----------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~~

GLI uses `pytest.ini` for test configuration:

.. code-block:: ini

   [tool:pytest]
   testpaths = tests
   python_files = test_*.py *_test.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       --strict-markers
       --disable-warnings
       -ra
   markers =
       slow: marks tests as slow (deselect with '-m "not slow"')
       integration: marks tests as integration tests
       backend: marks tests that require specific backend

Test Markers
~~~~~~~~~~~

.. code-block:: python

   # Mark slow tests
   @pytest.mark.slow
   def test_million_node_graph():
       pass
   
   # Mark backend-specific tests  
   @pytest.mark.backend("rust")
   def test_rust_specific_feature():
       pass
   
   # Mark integration tests
   @pytest.mark.integration
   def test_with_external_database():
       pass

Running Specific Test Categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run only fast tests
   pytest -m "not slow"
   
   # Run only Rust backend tests
   pytest -m "backend('rust')"
   
   # Run integration tests
   pytest -m integration
   
   # Run with specific backend
   pytest --backend=rust

Continuous Integration
---------------------

GLI uses GitHub Actions for CI/CD:

.. code-block:: yaml

   # .github/workflows/test.yml
   name: Tests
   
   on: [push, pull_request]
   
   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, windows-latest, macos-latest]
           python-version: [3.8, 3.9, '3.10', 3.11]
           
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python ${{ matrix.python-version }}
         uses: actions/setup-python@v3
         with:
           python-version: ${{ matrix.python-version }}
           
       - name: Install Rust
         uses: actions-rs/toolchain@v1
         with:
           toolchain: stable
           
       - name: Install dependencies
         run: |
           pip install -e ".[dev]"
           
       - name: Run tests
         run: |
           pytest --cov=gli --cov-report=xml
           
       - name: Upload coverage
         uses: codecov/codecov-action@v3

Test Data Management
-------------------

For tests requiring sample data:

.. code-block:: python

   import tempfile
   import json
   
   @pytest.fixture
   def sample_graph_data():
       """Provide sample graph data for testing"""
       return {
           "nodes": [
               {"id": "alice", "name": "Alice", "age": 30},
               {"id": "bob", "name": "Bob", "age": 25},
           ],
           "edges": [
               {"source": "alice", "target": "bob", "relationship": "friends"}
           ]
       }
   
   @pytest.fixture
   def temp_graph_file(sample_graph_data):
       """Create temporary file with graph data"""
       with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
           json.dump(sample_graph_data, f)
           return f.name

Contributing Tests
-----------------

Guidelines for contributing tests:

1. **Test Coverage**: Aim for >90% code coverage
2. **Backend Parity**: Test both Python and Rust backends
3. **Performance**: Include performance assertions for critical paths
4. **Documentation**: Document complex test scenarios
5. **Cleanup**: Use fixtures for proper test isolation

.. code-block:: python

   # Good test example
   def test_node_attribute_update():
       """Test that node attributes can be updated correctly"""
       g = Graph()
       
       # Setup
       node_id = g.add_node(name="Alice", age=30)
       
       # Action
       g.update_node(node_id, age=31, city="NYC")
       
       # Verification
       updated_node = g.get_node(node_id)
       assert updated_node["age"] == 31
       assert updated_node["city"] == "NYC"
       assert updated_node["name"] == "Alice"  # Unchanged attribute preserved

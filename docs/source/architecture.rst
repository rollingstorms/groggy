Architecture Overview
====================

Groggy's architecture is designed for performance, flexibility, and ease of use. This section explains the key architectural decisions and components.

System Architecture
-------------------

Groggy uses a unified columnar architecture built in Rust with Python bindings:

.. code-block::

   ┌─────────────────────────────────────┐
   │           Python API                │  <- User Interface
   ├─────────────────────────────────────┤
   │        Graph Abstraction            │  <- Core Logic & Smart Dispatch
   ├─────────────────────────────────────┤
   │      Rust Columnar Backend         │  <- High-Performance Implementation
   ├─────────────────────────────────────┤
   │    Columnar Attribute Storage       │  <- Optimized Data Layout
   ├─────────────────────────────────────┤
   │   Bitmap Indices & Fast Filtering  │  <- Query Optimization
   └─────────────────────────────────────┘

Core Components
---------------

Unified Graph Interface
~~~~~~~~~~~~~~~~~~~~~~

The main `Graph` class provides a unified interface backed by Rust:

.. code-block:: python

   class Graph:
       def __init__(self, directed=False):
           # Initializes Rust backend with columnar storage
           self._core = RustGraph(directed)
           self._filtering_pipeline = FilteringPipeline(self._core)
       
       def add_node(self, **attributes):
           # Delegates to Rust backend with columnar attribute storage
           return self._core.add_node(attributes)

Smart Filtering Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

Groggy uses intelligent query dispatch for optimal performance:

.. code-block:: python

   def filter_nodes(self, **kwargs):
       # Smart dispatch based on query type
       if self._is_simple_numeric_filter(kwargs):
           return self._core.filter_nodes_numeric_fast(**kwargs)
       elif self._is_simple_string_filter(kwargs):
           return self._core.filter_nodes_string_fast(**kwargs)
       else:
           return self._core.filter_nodes_general(**kwargs)

Data Structures
---------------

Unified Type System
~~~~~~~~~~~~~~~~~~~

Groggy uses a unified type system for optimal performance:

.. code-block:: rust

   // Core types for runtime operation
   pub struct NodeData {
       pub id: String,
       pub attributes: HashMap<String, JsonValue>,
   }
   
   pub struct EdgeData {
       pub source: String,
       pub target: String,
       pub attributes: HashMap<String, JsonValue>,
   }
   
   pub struct GraphType {
       pub nodes: HashMap<String, NodeData>,
       pub edges: HashMap<String, EdgeData>,
       pub directed: bool,
   }

Columnar Attribute Storage
~~~~~~~~~~~~~~~~~~~~~~~~~

Attributes are stored in columnar format for efficient querying:

.. code-block:: rust

   pub struct ColumnarStore {
       // Column-oriented storage for fast filtering
       string_columns: HashMap<String, Vec<Option<String>>>,
       numeric_columns: HashMap<String, Vec<Option<f64>>>,
       bool_columns: HashMap<String, Vec<Option<bool>>>,
       
       // Bitmap indices for exact matching
       string_indices: HashMap<(String, String), BitVec>,
       numeric_indices: HashMap<String, BTreeMap<OrderedFloat<f64>, BitVec>>,
   }

Memory Management
~~~~~~~~~~~~~~~~

Groggy implements several optimization strategies:

1. **Columnar Storage**: Attributes stored in column-oriented format for cache efficiency
2. **Bitmap Indexing**: Fast exact-match lookups using compressed bitmaps
3. **Legacy Storage Compatibility**: Efficient conversion between runtime and storage formats

.. code-block:: rust

   // Fast bitmap-based filtering
   impl ColumnarStore {
       pub fn filter_by_string_exact(&self, attr: &str, value: &str) -> BitVec {
           // O(1) lookup using pre-built bitmap index
           self.string_indices
               .get(&(attr.to_string(), value.to_string()))
               .cloned()
               .unwrap_or_else(|| BitVec::new())
       }
       
       pub fn filter_by_numeric_range(&self, attr: &str, min: f64, max: f64) -> BitVec {
           // Efficient range query using sorted index
           let mut result = BitVec::new();
           if let Some(index) = self.numeric_indices.get(attr) {
               for (&val, bitmap) in index.range(min..=max) {
                   result |= bitmap;
               }
           }
           result
       }
   }

Rust Backend Design
-------------------

The unified Rust backend provides maximum performance through columnar storage:

FFI Integration
~~~~~~~~~~~~~~

Groggy uses PyO3 for seamless Python-Rust interoperability:

.. code-block:: rust

   use pyo3::prelude::*;
   
   #[pyclass]
   pub struct RustGraph {
       graph: GraphType,
       columnar_store: ColumnarStore,
   }
   
   #[pymethods]
   impl RustGraph {
       #[new]
       fn new(directed: bool) -> Self {
           RustGraph {
               graph: GraphType::new(directed),
               columnar_store: ColumnarStore::new(),
           }
       }
       
       fn add_node(&mut self, attributes: HashMap<String, JsonValue>) -> PyResult<String> {
           let node_id = self.graph.add_node(attributes.clone())?;
           self.columnar_store.add_node_attributes(&node_id, &attributes)?;
           Ok(node_id)
       }
       
       fn filter_nodes_numeric_fast(&self, attr: &str, value: f64) -> Vec<String> {
           // Fast bitmap-based filtering
           let bitmap = self.columnar_store.filter_by_numeric_exact(attr, value);
           self.bitmap_to_node_ids(&bitmap)
       }
   }

High-Performance Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~

The Rust backend implements multiple optimized filtering strategies:

.. code-block:: rust

   impl RustGraph {
       // O(1) exact matching using bitmap indices
       pub fn filter_nodes_exact(&self, attr: &str, value: &JsonValue) -> Vec<String> {
           match value {
               JsonValue::String(s) => {
                   let bitmap = self.columnar_store.filter_by_string_exact(attr, s);
                   self.bitmap_to_node_ids(&bitmap)
               }
               JsonValue::Number(n) => {
                   let bitmap = self.columnar_store.filter_by_numeric_exact(attr, n.as_f64().unwrap());
                   self.bitmap_to_node_ids(&bitmap)
               }
               _ => self.filter_nodes_general(attr, value)
           }
       }
       
       // Optimized range queries
       pub fn filter_nodes_range(&self, attr: &str, min: f64, max: f64) -> Vec<String> {
           let bitmap = self.columnar_store.filter_by_numeric_range(attr, min, max);
           self.bitmap_to_node_ids(&bitmap)
       }
   }

Performance Characteristics
---------------------------

Algorithmic Complexity
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Operation Complexity
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Time Complexity
     - Notes
   * - Add Node
     - O(1) average
     - Amortized constant time
   * - Add Edge
     - O(1) average
     - Amortized constant time
   * - Get Node Attributes
     - O(1) average
     - Hash table lookup
   * - Filter (Exact Match)
     - O(1)
     - Bitmap index lookup
   * - Filter (Range Query)
     - O(log n + k)
     - BTree range + result size
   * - Filter (General)
     - O(n)
     - Full scan with predicate
   * - Batch Operations
     - O(n)
     - Linear in batch size

Memory Overhead
~~~~~~~~~~~~~~

.. list-table:: Memory Usage Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Groggy (Rust)
     - NetworkX (Python)
   * - 10K Nodes (no attrs)
     - ~2.1 MB
     - ~8.4 MB
   * - 10K Nodes (5 attrs each)
     - ~4.8 MB
     - ~18.2 MB
   * - Filtering Performance
     - 1.2-5.6x faster
     - Baseline

Filtering Performance
~~~~~~~~~~~~~~~~~~~

Groggy's columnar storage provides significant filtering performance improvements:

- **Exact String Matching**: 5.6x faster than NetworkX
- **Numeric Filtering**: 2.1x faster than NetworkX  
- **Range Queries**: 3.2x faster than NetworkX
- **Complex Queries**: 1.2x faster than NetworkX

Concurrency Model
-----------------

Thread Safety
~~~~~~~~~~~~~

Groggy's Rust backend provides thread-safe operations:

**Unified Rust Backend**:
- Uses `Arc<RwLock<T>>` for shared access
- Multiple reader threads supported
- Single writer thread with exclusive access
- Thread-safe columnar operations

.. code-block:: python

   # Thread-safe usage example
   from threading import Thread
   from groggy import Graph
   
   g = Graph()  # Thread-safe by default
   
   def add_nodes(start, end):
       for i in range(start, end):
           g.add_node(value=i, thread_id=threading.current_thread().ident)
   
   # Parallel node addition
   threads = []
   for i in range(4):
       t = Thread(target=add_nodes, args=(i*1000, (i+1)*1000))
       threads.append(t)
       t.start()
   
   for t in threads:
       t.join()
   
   print(f"Total nodes: {len(g.nodes)}")  # Should be 4000

Smart Query Optimization
~~~~~~~~~~~~~~~~~~~~~~~

The filtering pipeline automatically optimizes queries:

.. code-block:: python

   # These are automatically optimized
   engineers = g.filter_nodes(role="engineer")          # Bitmap index
   adults = g.filter_nodes(lambda n, a: a.get('age', 0) > 18)  # General scan
   young_engineers = g.filter_nodes(role="engineer", age=25)   # Index intersection

Extensibility
-------------

Plugin Architecture
~~~~~~~~~~~~~~~~~~

Groggy supports plugins for extending functionality:

.. code-block:: python

   class GraphPlugin:
       def __init__(self, graph):
           self.graph = graph
       
       def install(self):
           # Add methods to graph instance
           self.graph.custom_method = self.custom_method
       
       def custom_method(self):
           # Plugin functionality
           pass
   
   # Usage
   g = Graph()
   plugin = GraphPlugin(g)
   plugin.install()
   g.custom_method()  # Now available

Custom Backends
~~~~~~~~~~~~~~

Advanced users can implement custom backends:

.. code-block:: python

   class CustomBackend:
       def add_node(self, node_id, attributes):
           # Custom implementation
           pass
       
       def get_node(self, node_id):
           # Custom implementation
           pass
   
   # Register custom backend
   Graph.register_backend('custom', CustomBackend)

Error Handling Strategy
----------------------

Groggy uses a layered error handling approach:

Exception Hierarchy
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class GroggyError(Exception):
       """Base exception for Groggy operations"""
       pass
   
   class NodeNotFoundError(GroggyError, KeyError):
       """Raised when a node is not found"""
       pass
   
   class EdgeNotFoundError(GroggyError, KeyError):
       """Raised when an edge is not found"""
       pass
   
   class BackendError(GroggyError):
       """Raised when backend operations fail"""
       pass

Error Recovery
~~~~~~~~~~~~~

Groggy implements graceful error recovery:

.. code-block:: python

   def safe_add_edge(self, source, target, **attributes):
       try:
           return self._backend.add_edge(source, target, attributes)
       except NodeNotFoundError:
           # Auto-create missing nodes
           if not self.has_node(source):
               self.add_node(source)
           if not self.has_node(target):
               self.add_node(target)
           return self._backend.add_edge(source, target, attributes)

Testing Architecture
-------------------

Groggy uses comprehensive testing strategies:

Backend Parity Testing
~~~~~~~~~~~~~~~~~~~~~

Ensures both backends produce identical results:

.. code-block:: python

   def test_backend_parity():
       operations = [
           ('add_node', 'alice', {'age': 30}),
           ('add_node', 'bob', {'age': 25}),
           ('add_edge', 'alice', 'bob', {'weight': 1.0}),
       ]
       
       # Test both backends
       g_python = Graph(backend='python')
       g_rust = Graph(backend='rust')
       
       for op_type, *args in operations:
           getattr(g_python, op_type)(*args)
           getattr(g_rust, op_type)(*args)
       
       # Verify identical results
       assert g_python.node_count() == g_rust.node_count()
       assert g_python.edge_count() == g_rust.edge_count()

Performance Testing
~~~~~~~~~~~~~~~~~~

Automated performance regression testing:

.. code-block:: python

   def test_performance_regression():
       # Baseline performance metrics
       baseline_times = {
           'add_1000_nodes': 0.1,
           'query_1000_neighbors': 0.05,
       }
       
       # Run current tests
       current_times = run_performance_tests()
       
       # Check for regressions
       for test_name, baseline_time in baseline_times.items():
           current_time = current_times[test_name]
           regression_factor = current_time / baseline_time
           
           assert regression_factor < 1.5, f"Performance regression in {test_name}"

Future Architecture Considerations
---------------------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Advanced Indexing**: Spatial indices for geographic data, full-text search
2. **Distributed Backend**: Multi-machine graph processing with columnar replication
3. **Streaming Backend**: Real-time graph updates with incremental index maintenance
4. **GPU Acceleration**: CUDA/OpenCL support for parallel graph algorithms

.. code-block:: python

   # Future distributed backend example
   g = Graph(distributed=True, nodes=['node1', 'node2', 'node3'])
   
   # Future streaming backend example  
   g = Graph(streaming=True, update_buffer=1000)

Scalability Roadmap
~~~~~~~~~~~~~~~~~~

Groggy's columnar architecture enables future scaling:

- **Vertical Scaling**: Better single-machine performance through SIMD optimization
- **Horizontal Scaling**: Distributed columnar storage across machines
- **Cloud Integration**: Native object storage backends (S3, GCS, Azure Blob)
- **Edge Computing**: Lightweight deployments with compressed indices

Query Optimization Roadmap
~~~~~~~~~~~~~~~~~~~~~~~~~

Planned query optimization enhancements:

- **Cost-Based Optimization**: Choose optimal execution strategy based on data statistics
- **Index Intersection**: Combine multiple bitmap indices for complex queries
- **Adaptive Indexing**: Automatically create indices based on query patterns
- **Parallel Query Execution**: Multi-threaded query processing for large datasets

Architecture Overview
====================

Groggy is designed as a high-performance graph analysis library with a hybrid Rust-Python architecture that provides the best of both worlds: Rust's speed and Python's usability.

System Architecture
-------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Python API Layer                         │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │   Graph     │ │ Storage     │ │     Analytics           │ │
   │  │   Class     │ │ Views       │ │     Modules             │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                                │
                                │ PyO3 FFI
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                     Rust Core                               │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │   Graph     │ │  Storage    │ │      Algorithm          │ │
   │  │   Engine    │ │  System     │ │      Implementations    │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │   Memory    │ │   Pool      │ │     Concurrent          │ │
   │  │  Manager    │ │  Allocator  │ │     Execution           │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘

Core Design Principles
----------------------

Performance First
~~~~~~~~~~~~~~~~~

- **Native Rust Core**: All computation happens in Rust for maximum performance
- **Zero-Copy Operations**: Minimize data copying between Python and Rust
- **SIMD Vectorization**: Leverage modern CPU instructions for parallel operations
- **Cache-Friendly Layout**: Columnar storage optimized for cache locality
- **Lazy Evaluation**: Compute results only when needed

Memory Efficiency
~~~~~~~~~~~~~~~~~

- **Pooled Memory**: Custom memory pools reduce allocation overhead
- **Sparse Representations**: Automatic sparse storage for data with many zeros
- **Reference Counting**: Smart pointers manage memory safely across language boundaries  
- **Streaming Operations**: Process large datasets without loading everything into memory

Scalability
~~~~~~~~~~~

- **Parallel Algorithms**: Multi-threaded implementations of expensive operations
- **Approximation Methods**: Sampling-based algorithms for very large graphs
- **Incremental Updates**: Efficient handling of graph modifications
- **Memory Mapping**: Support for graphs larger than available RAM

Usability
~~~~~~~~~

- **Pythonic API**: Familiar interface following pandas/NumPy conventions
- **Storage Views**: Multiple perspectives on the same data (array, matrix, table)
- **Integration**: Seamless conversion to/from pandas, NumPy, NetworkX
- **Error Handling**: Clear, actionable error messages

Language Boundary Design
------------------------

FFI (Foreign Function Interface)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python-Rust boundary is managed through PyO3, which provides:

- **Safe Memory Management**: Automatic handling of Python reference counting
- **Type Conversion**: Efficient conversion between Python and Rust types
- **Exception Propagation**: Rust errors become Python exceptions
- **GIL Management**: Appropriate release of Python's Global Interpreter Lock

Data Passing Strategy
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Python Side
   graph = gr.Graph()
   graph.add_node("alice", age=30)
   
   # This creates:
   # 1. Python Graph object (thin wrapper)
   # 2. Rust GraphCore (actual data storage)
   # 3. Shared references managed by PyO3

Memory Layout
~~~~~~~~~~~~~

Data flows between languages in several ways:

1. **Immutable References**: Python borrows Rust data without copying
2. **Owned Transfer**: Large results moved from Rust to Python ownership
3. **Shared State**: Core graph structure shared with reference counting
4. **Lazy Materialization**: Results computed in Rust, materialized on Python access

Core Subsystems
---------------

Graph Engine (Rust)
~~~~~~~~~~~~~~~~~~~~

The heart of Groggy's performance, implemented in Rust:

**GraphCore Structure**:

.. code-block:: rust

   pub struct GraphCore {
       nodes: NodeStore,
       edges: EdgeStore,
       topology: AdjacencyStructure,
       attributes: AttributeManager,
       indices: IndexManager,
   }

**Key Components**:

- **NodeStore**: Efficient storage of node data with fast lookups
- **EdgeStore**: Edge data with support for directed/undirected graphs  
- **AdjacencyStructure**: Multiple topology representations (list, matrix, hash)
- **AttributeManager**: Columnar attribute storage with type specialization
- **IndexManager**: B-tree and hash indices for fast queries

Storage System
~~~~~~~~~~~~~~

Three-layer storage architecture:

**Pool Layer**:
- Memory pools for different object sizes
- Reduces fragmentation and allocation overhead
- Thread-safe allocation with minimal contention

**Space Layer**:
- Logical organization of related data
- Node space, edge space, attribute spaces
- Enables efficient batch operations

**History Layer**:
- Transaction log for undo/redo operations
- Enables incremental updates and consistency checks
- Optional persistence for durability

Storage Views Implementation
----------------------------

Unified Storage Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

All storage views (Array, Matrix, Table) share a common foundation:

.. code-block:: rust

   trait StorageView<T> {
       fn get(&self, index: Index) -> Option<&T>;
       fn shape(&self) -> Shape;
       fn dtype(&self) -> DataType;
       fn iter(&self) -> impl Iterator<Item = &T>;
   }

**GraphArray Implementation**:

- Wraps a columnar vector with statistical caching
- Lazy computation of aggregates (mean, std, etc.)
- Optimized indexing with support for boolean masks

**GraphMatrix Implementation**:

- 2D columnar layout with row-major or column-major access
- BLAS integration for linear algebra operations
- Automatic sparse representation detection

**GraphTable Implementation**:

- Multiple columns with heterogeneous types
- Relational operations (join, group, filter)
- Integration with graph topology for neighbor queries

Algorithm Architecture
----------------------

Modular Algorithm Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Algorithms are organized into modules with consistent interfaces:

.. code-block:: rust

   trait GraphAlgorithm<Input, Output> {
       fn execute(&self, graph: &GraphCore, input: Input) -> Result<Output>;
       fn parallel_execute(&self, graph: &GraphCore, input: Input) -> Result<Output>;
       fn approximate_execute(&self, graph: &GraphCore, input: Input, precision: f64) -> Result<Output>;
   }

**Centrality Module**:
- Betweenness, closeness, eigenvector, PageRank
- Exact and approximate implementations
- Parallel execution for large graphs

**Community Module**:
- Louvain, Leiden algorithms
- Modularity optimization
- Hierarchical community detection

**Structural Module**:
- Connected components, bridges, articulation points
- Core decomposition, clustering coefficient
- Path analysis and diameter calculation

Parallelization Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

Multi-level parallelization approach:

1. **Algorithm Level**: Parallel implementations of core algorithms
2. **Operation Level**: Vectorized operations using SIMD
3. **Data Level**: Parallel processing of independent data chunks
4. **Graph Level**: Partition-based processing for very large graphs

Memory Management
-----------------

Rust Memory Management
~~~~~~~~~~~~~~~~~~~~~~

**Ownership Model**:
- Clear ownership of graph data in Rust
- Borrowing for temporary access without copying
- Reference counting for shared data structures

**Memory Pools**:

.. code-block:: rust

   struct MemoryPool {
       small_objects: Pool<SmallBlock>,    // < 64 bytes
       medium_objects: Pool<MediumBlock>,  // 64-1024 bytes  
       large_objects: Pool<LargeBlock>,    // > 1024 bytes
       string_pool: StringPool,            // Specialized for strings
   }

**Allocation Strategy**:
- Size-based pool selection
- Thread-local pools to reduce contention
- Bulk allocation for batch operations
- Automatic defragmentation during idle periods

Python Integration
~~~~~~~~~~~~~~~~~~

**Reference Management**:
- PyO3 handles Python reference counting automatically
- Rust objects wrapped in Python smart pointers
- Circular reference detection and cleanup

**Memory Limits**:
- Configurable memory limits for operations
- Graceful degradation when limits exceeded
- Streaming operations for memory-constrained environments

Type System
-----------

Heterogeneous Data Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rust Type System**:

.. code-block:: rust

   #[derive(Clone, Debug)]
   pub enum AttrValue {
       Int8(i8), Int16(i16), Int32(i32), Int64(i64),
       Float32(f32), Float64(f64),
       String(String), Bool(bool),
       Bytes(Vec<u8>), Null,
   }

**Type Coercion Rules**:
- Automatic promotion for numeric operations
- String conversion for display operations
- Null handling with proper three-valued logic

**Python Type Mapping**:
- AttrValue ↔ Python objects via PyO3
- NumPy array integration for bulk data
- Pandas compatibility for DataFrames

Error Handling
--------------

Layered Error Management
~~~~~~~~~~~~~~~~~~~~~~~~

**Rust Error Types**:

.. code-block:: rust

   #[derive(Debug, thiserror::Error)]
   pub enum GraphError {
       #[error("Node {0} not found")]
       NodeNotFound(String),
       
       #[error("Edge ({0}, {1}) not found")]
       EdgeNotFound(String, String),
       
       #[error("Invalid graph state: {0}")]
       InvalidState(String),
       
       #[error("Memory allocation failed")]
       OutOfMemory,
   }

**Python Error Translation**:
- Rust errors automatically become Python exceptions
- Context preservation across language boundary
- Stack traces include both Rust and Python frames

Concurrency Model
-----------------

Thread Safety
~~~~~~~~~~~~~~

**Shared Immutable Data**:
- Graph topology and attributes are immutable once created
- Multiple readers can access data simultaneously
- Copy-on-write for modifications

**Mutable Operations**:
- Graph construction uses interior mutability with locks
- Fine-grained locking to minimize contention
- Lock-free data structures where possible

**Python GIL Handling**:
- Release GIL during expensive Rust operations
- Acquire GIL only for Python callbacks and result conversion
- Parallel algorithms run without GIL when possible

Performance Characteristics
---------------------------

Benchmarking Results
~~~~~~~~~~~~~~~~~~~~

Typical performance characteristics on modern hardware:

**Graph Construction**:
- 1M nodes: ~2 seconds
- 10M edges: ~5 seconds  
- Memory usage: ~50 bytes per node, ~20 bytes per edge

**Algorithm Performance**:
- PageRank (100K nodes): ~50ms
- Connected Components (1M nodes): ~200ms
- Betweenness Centrality (10K nodes): ~1 second

**Storage View Operations**:
- Statistical operations: 10-100x faster than pandas
- Matrix operations: Near NumPy performance
- Table joins: 2-5x faster than pandas

Scaling Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

**Memory Scaling**:
- Linear memory usage with graph size
- Constant overhead: ~10MB base memory
- Sparse graphs: 30-50% memory reduction

**Compute Scaling**:
- Most algorithms scale linearly or near-linearly
- Parallel algorithms show good scaling up to 8-16 cores
- Cache-friendly layouts improve performance on large graphs

Future Architecture
-------------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~~

**Distributed Processing**:
- Graph partitioning for multi-machine processing
- Message-passing algorithms for distributed analysis
- Cloud-native deployment support

**GPU Acceleration**:
- CUDA kernels for embarrassingly parallel algorithms
- GPU memory management for large graphs
- Hybrid CPU-GPU execution

**Streaming Analytics**:
- Online algorithm implementations
- Incremental updates for dynamic graphs
- Real-time analytics with sliding windows

**Storage Backends**:
- Integration with columnar formats (Arrow, Parquet)
- Database backend support (PostgreSQL, Neo4j)
- Cloud storage integration (S3, GCS)

This architecture provides the foundation for high-performance graph analysis while maintaining Python's ease of use. The hybrid design allows for optimal performance where it matters most while preserving the flexibility and ecosystem integration that Python provides.
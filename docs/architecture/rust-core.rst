Rust Core Architecture
======================

The Rust core is the performance engine of Groggy, handling all computational tasks, memory management, and data storage. This document provides an in-depth look at the core Rust implementation.

Core Data Structures
--------------------

GraphCore: The Central Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `GraphCore` struct is the heart of Groggy's implementation:

.. code-block:: rust

   pub struct GraphCore {
       // Node management
       nodes: NodeStorage,
       node_count: AtomicUsize,
       
       // Edge management  
       edges: EdgeStorage,
       edge_count: AtomicUsize,
       
       // Topology representations
       adjacency: AdjacencyManager,
       
       // Attribute storage
       node_attrs: AttributeStorage,
       edge_attrs: AttributeStorage,
       
       // Indexing and caching
       indices: IndexManager,
       cache: CacheManager,
       
       // Memory management
       memory_pool: MemoryPool,
       
       // Configuration
       config: GraphConfig,
       directed: bool,
   }

Node Storage System
~~~~~~~~~~~~~~~~~~~

**NodeStorage Implementation**:

.. code-block:: rust

   pub struct NodeStorage {
       // Primary storage
       id_to_index: HashMap<NodeId, NodeIndex>,
       index_to_id: Vec<NodeId>,
       
       // Fast lookup structures
       id_hash_table: HashTable<NodeId, NodeIndex>,
       
       // Deletion tracking
       deleted_nodes: BitSet,
       free_indices: Vec<NodeIndex>,
       
       // Capacity management
       capacity: usize,
       next_index: AtomicUsize,
   }

**Key Design Decisions**:

- **Stable Indices**: Node indices remain stable across deletions
- **Dense Packing**: Deleted node indices are reused efficiently  
- **Fast Lookup**: O(1) average time for ID ↔ index conversion
- **Memory Efficiency**: Minimal overhead per node (~8 bytes)

Edge Storage System
~~~~~~~~~~~~~~~~~~~

**EdgeStorage Implementation**:

.. code-block:: rust

   pub struct EdgeStorage {
       // Edge data
       edges: Vec<Edge>,
       
       // Source and target mappings
       source_indices: Vec<NodeIndex>,
       target_indices: Vec<NodeIndex>,
       
       // Fast edge lookup
       edge_map: HashMap<(NodeIndex, NodeIndex), EdgeIndex>,
       
       // Deletion tracking
       deleted_edges: BitSet,
       free_indices: Vec<EdgeIndex>,
   }

**Edge Representation**:

.. code-block:: rust

   #[derive(Clone, Debug)]
   pub struct Edge {
       source: NodeIndex,
       target: NodeIndex,
       edge_id: EdgeId,
       
       // Optional weight for algorithms
       weight: Option<f64>,
   }

Adjacency Management
--------------------

Multiple Topology Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Groggy maintains multiple representations of graph topology for different access patterns:

.. code-block:: rust

   pub struct AdjacencyManager {
       // For neighbor iteration
       adjacency_lists: AdjacencyLists,
       
       // For connectivity queries
       adjacency_matrix: Option<AdjacencyMatrix>,
       
       // For path algorithms
       csr_representation: Option<CSRMatrix>,
       
       // Configuration
       auto_build_matrix: bool,
       matrix_density_threshold: f64,
   }

**AdjacencyLists Structure**:

.. code-block:: rust

   pub struct AdjacencyLists {
       // Outgoing edges
       out_neighbors: Vec<Vec<NodeIndex>>,
       out_edge_indices: Vec<Vec<EdgeIndex>>,
       
       // Incoming edges (for directed graphs)
       in_neighbors: Vec<Vec<NodeIndex>>,
       in_edge_indices: Vec<Vec<EdgeIndex>>,
       
       // Caching for degree queries
       out_degrees: Vec<u32>,
       in_degrees: Vec<u32>,
   }

**Automatic Representation Selection**:

- **Sparse graphs** (density < 0.1): Use adjacency lists only
- **Dense graphs** (density > 0.3): Build adjacency matrix automatically
- **Medium graphs**: Build on-demand based on query patterns

Compressed Sparse Row (CSR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For algorithm efficiency, Groggy uses CSR representation:

.. code-block:: rust

   pub struct CSRMatrix {
       // Row pointers (node -> first edge index)
       row_ptrs: Vec<usize>,
       
       // Column indices (target nodes)
       col_indices: Vec<NodeIndex>,
       
       // Edge weights (if present)
       values: Option<Vec<f64>>,
       
       // Dimensions
       num_nodes: usize,
       num_edges: usize,
   }

**Benefits of CSR**:
- Cache-friendly memory layout
- Efficient neighbor iteration
- Compatible with BLAS/LAPACK
- Minimal memory overhead

Attribute Storage
-----------------

Columnar Attribute System
~~~~~~~~~~~~~~~~~~~~~~~~~

Attributes are stored in a columnar format for cache efficiency:

.. code-block:: rust

   pub struct AttributeStorage {
       // Column data by attribute name
       columns: HashMap<String, AttributeColumn>,
       
       // Schema information
       schema: AttributeSchema,
       
       // Null tracking
       null_bitmaps: HashMap<String, BitSet>,
       
       // Statistics caching
       statistics: HashMap<String, ColumnStats>,
   }

**AttributeColumn Types**:

.. code-block:: rust

   pub enum AttributeColumn {
       // Numeric types
       Int8(Vec<i8>),
       Int16(Vec<i16>),
       Int32(Vec<i32>),
       Int64(Vec<i64>),
       Float32(Vec<f32>),
       Float64(Vec<f64>),
       
       // String types
       String(StringColumn),
       
       // Boolean
       Bool(BitSet),
       
       // Binary data
       Bytes(Vec<Vec<u8>>),
       
       // Categorical (for repeated strings)
       Categorical(CategoricalColumn),
   }

String Storage Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**StringColumn Implementation**:

.. code-block:: rust

   pub struct StringColumn {
       // Pooled string storage
       string_pool: StringPool,
       
       // Indices into pool
       string_indices: Vec<StringIndex>,
       
       // Optional dictionary compression
       dictionary: Option<Dictionary>,
   }

   pub struct StringPool {
       // Contiguous string data
       data: Vec<u8>,
       
       // String boundaries
       offsets: Vec<usize>,
       
       // Hash table for deduplication
       string_to_index: HashMap<u64, StringIndex>,
   }

**Benefits**:
- String deduplication reduces memory usage
- Cache-friendly string access
- O(1) string equality comparison by index

Categorical Data
~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct CategoricalColumn {
       // Category definitions
       categories: Vec<String>,
       
       // Category indices (compact representation)
       codes: Vec<CategoryCode>,
       
       // Reverse mapping
       category_to_code: HashMap<String, CategoryCode>,
   }

**Advantages**:
- 8-32x memory reduction for repeated strings
- Fast equality and grouping operations
- Automatic category detection

Memory Management
-----------------

Pool-Based Allocation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct MemoryPool {
       // Size-based pools
       pools: [Pool; 16],  // Powers of 2 from 8 bytes to 256KB
       
       // Large object allocator
       large_allocator: LargeObjectAllocator,
       
       // String-specific pool
       string_pool: StringPool,
       
       // Statistics
       allocation_stats: AllocationStats,
       
       // Configuration
       config: PoolConfig,
   }

**Pool Structure**:

.. code-block:: rust

   pub struct Pool {
       // Free block list
       free_blocks: Vec<*mut u8>,
       
       // Block size for this pool
       block_size: usize,
       
       // Allocated chunks
       chunks: Vec<Chunk>,
       
       // Statistics
       allocated_blocks: usize,
       peak_usage: usize,
   }

**Allocation Strategy**:

1. **Size Classification**: Object size determines pool selection
2. **Fast Path**: Pop from free list (O(1) operation)  
3. **Slow Path**: Allocate new chunk if free list empty
4. **Deallocation**: Push to free list for reuse

Reference Counting
~~~~~~~~~~~~~~~~~~

For shared data structures, Groggy uses reference counting:

.. code-block:: rust

   pub struct SharedData<T> {
       data: Arc<T>,
       weak_refs: AtomicUsize,
   }

   pub struct WeakRef<T> {
       inner: Weak<T>,
   }

**Benefits**:
- Safe sharing across thread boundaries
- Automatic cleanup when no longer referenced
- Cycle detection for complex object graphs

Algorithm Infrastructure
------------------------

Algorithm Trait System
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub trait GraphAlgorithm<Input, Output> {
       type Error;
       
       // Core execution
       fn execute(&self, graph: &GraphCore, input: Input) 
                 -> Result<Output, Self::Error>;
       
       // Optional parallel implementation
       fn execute_parallel(&self, graph: &GraphCore, input: Input) 
                          -> Result<Output, Self::Error> {
           self.execute(graph, input)  // Default to sequential
       }
       
       // Optional approximate implementation
       fn execute_approximate(&self, graph: &GraphCore, input: Input, 
                             precision: f64) -> Result<Output, Self::Error> {
           self.execute(graph, input)  // Default to exact
       }
       
       // Algorithm metadata
       fn name(&self) -> &'static str;
       fn complexity(&self) -> Complexity;
       fn requirements(&self) -> Requirements;
   }

Centrality Algorithms
~~~~~~~~~~~~~~~~~~~~~

**PageRank Implementation**:

.. code-block:: rust

   pub struct PageRank {
       alpha: f64,
       max_iterations: usize,
       tolerance: f64,
       parallel: bool,
   }

   impl GraphAlgorithm<Option<Vec<f64>>, Vec<f64>> for PageRank {
       type Error = AlgorithmError;
       
       fn execute(&self, graph: &GraphCore, personalization: Option<Vec<f64>>) 
                 -> Result<Vec<f64>, Self::Error> {
           let n = graph.node_count();
           let mut ranks = vec![1.0 / n as f64; n];
           let mut new_ranks = vec![0.0; n];
           
           let csr = graph.adjacency.get_csr()?;
           let out_degrees = graph.adjacency.out_degrees();
           
           for iteration in 0..self.max_iterations {
               // Reset ranks
               new_ranks.fill(0.0);
               
               // Distribute rank from each node
               for (node, &rank) in ranks.iter().enumerate() {
                   let degree = out_degrees[node];
                   if degree > 0 {
                       let contribution = rank / degree as f64;
                       
                       // Add to all neighbors
                       for &neighbor in graph.neighbors(node) {
                           new_ranks[neighbor] += contribution;
                       }
                   }
               }
               
               // Apply damping and personalization
               for (i, rank) in new_ranks.iter_mut().enumerate() {
                   let personalization_term = match &personalization {
                       Some(p) => p[i],
                       None => 1.0 / n as f64,
                   };
                   
                   *rank = self.alpha * (*rank) + (1.0 - self.alpha) * personalization_term;
               }
               
               // Check convergence
               let diff: f64 = ranks.iter()
                   .zip(new_ranks.iter())
                   .map(|(old, new)| (old - new).abs())
                   .sum();
               
               if diff < self.tolerance {
                   break;
               }
               
               std::mem::swap(&mut ranks, &mut new_ranks);
           }
           
           Ok(ranks)
       }
   }

Community Detection
~~~~~~~~~~~~~~~~~~~

**Louvain Algorithm Structure**:

.. code-block:: rust

   pub struct LouvainCommunities {
       resolution: f64,
       max_iterations: usize,
       random_seed: Option<u64>,
   }

   impl GraphAlgorithm<(), Vec<CommunityId>> for LouvainCommunities {
       type Error = AlgorithmError;
       
       fn execute(&self, graph: &GraphCore, _: ()) 
                 -> Result<Vec<CommunityId>, Self::Error> {
           
           let mut communities = self.initialize_communities(graph);
           let mut improved = true;
           let mut iteration = 0;
           
           while improved && iteration < self.max_iterations {
               improved = false;
               
               // Phase 1: Local optimization
               for node in graph.nodes() {
                   let current_community = communities[node];
                   let mut best_community = current_community;
                   let mut best_gain = 0.0;
                   
                   // Try moving to neighbor communities
                   for neighbor in graph.neighbors(node) {
                       let neighbor_community = communities[neighbor];
                       if neighbor_community != current_community {
                           let gain = self.modularity_gain(
                               graph, node, current_community, neighbor_community
                           );
                           
                           if gain > best_gain {
                               best_gain = gain;
                               best_community = neighbor_community;
                           }
                       }
                   }
                   
                   // Move if beneficial
                   if best_community != current_community {
                       communities[node] = best_community;
                       improved = true;
                   }
               }
               
               iteration += 1;
           }
           
           Ok(communities)
       }
   }

Parallel Processing
-------------------

Thread Pool Management
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct ThreadPool {
       workers: Vec<Worker>,
       sender: mpsc::Sender<Job>,
       config: ThreadConfig,
   }

   struct Worker {
       id: usize,
       thread: Option<thread::JoinHandle<()>>,
   }

   type Job = Box<dyn FnOnce() + Send + 'static>;

**Work Distribution Strategies**:

1. **Data Parallelism**: Split nodes/edges across threads
2. **Algorithm Parallelism**: Parallel algorithm implementations
3. **Pipeline Parallelism**: Overlap computation and I/O

SIMD Optimization
~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[cfg(target_arch = "x86_64")]
   mod simd {
       use std::arch::x86_64::*;
       
       pub unsafe fn vectorized_dot_product(a: &[f64], b: &[f64]) -> f64 {
           assert_eq!(a.len(), b.len());
           
           let mut sum = _mm256_setzero_pd();
           let chunks = a.len() / 4;
           
           for i in 0..chunks {
               let va = _mm256_loadu_pd(a.as_ptr().add(i * 4));
               let vb = _mm256_loadu_pd(b.as_ptr().add(i * 4));
               let prod = _mm256_mul_pd(va, vb);
               sum = _mm256_add_pd(sum, prod);
           }
           
           // Extract and sum the 4 values
           let mut result = [0.0; 4];
           _mm256_storeu_pd(result.as_mut_ptr(), sum);
           
           let mut total = result.iter().sum::<f64>();
           
           // Handle remaining elements
           for i in (chunks * 4)..a.len() {
               total += a[i] * b[i];
           }
           
           total
       }
   }

Lock-Free Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use crossbeam::atomic::AtomicCell;
   use crossbeam::utils::CachePadded;

   pub struct LockFreeCounter {
       value: CachePadded<AtomicCell<u64>>,
   }

   impl LockFreeCounter {
       pub fn increment(&self) -> u64 {
           self.value.fetch_add(1)
       }
       
       pub fn get(&self) -> u64 {
           self.value.load()
       }
   }

Caching System
--------------

Multi-Level Caching
~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct CacheManager {
       // Algorithm result cache
       algorithm_cache: LruCache<AlgorithmKey, AlgorithmResult>,
       
       // Statistical cache
       stats_cache: HashMap<StatKey, StatResult>,
       
       // Topology cache
       topology_cache: TopologyCache,
       
       // Configuration
       max_memory: usize,
       ttl: Duration,
   }

**Cache Key Generation**:

.. code-block:: rust

   #[derive(Hash, PartialEq, Eq)]
   pub struct AlgorithmKey {
       algorithm_name: String,
       graph_hash: u64,
       parameters_hash: u64,
   }

   impl AlgorithmKey {
       pub fn new<A: GraphAlgorithm<I, O>, I: Hash>(
           algorithm: &A, 
           graph: &GraphCore, 
           input: &I
       ) -> Self {
           let mut hasher = DefaultHasher::new();
           
           algorithm.name().hash(&mut hasher);
           graph.structure_hash().hash(&mut hasher);
           input.hash(&mut hasher);
           
           Self {
               algorithm_name: algorithm.name().to_string(),
               graph_hash: graph.structure_hash(),
               parameters_hash: hasher.finish(),
           }
       }
   }

Cache Invalidation
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   impl CacheManager {
       pub fn invalidate_node_caches(&mut self, node: NodeIndex) {
           // Invalidate caches affected by node changes
           self.stats_cache.retain(|key, _| !key.depends_on_node(node));
           self.algorithm_cache.retain(|key, _| !key.algorithm_affects_node(node));
       }
       
       pub fn invalidate_structure_caches(&mut self) {
           // Invalidate all structure-dependent caches
           self.algorithm_cache.clear();
           self.topology_cache.clear();
       }
   }

Error Handling
--------------

Comprehensive Error Types
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[derive(Debug, thiserror::Error)]
   pub enum GraphError {
       #[error("Node {id} not found")]
       NodeNotFound { id: String },
       
       #[error("Edge ({source}, {target}) not found")]
       EdgeNotFound { source: String, target: String },
       
       #[error("Graph operation invalid: {reason}")]
       InvalidOperation { reason: String },
       
       #[error("Memory allocation failed: {details}")]
       OutOfMemory { details: String },
       
       #[error("Algorithm error: {source}")]
       Algorithm { 
           #[from]
           source: AlgorithmError 
       },
       
       #[error("I/O error: {source}")]
       Io { 
           #[from]
           source: std::io::Error 
       },
   }

**Context Preservation**:

.. code-block:: rust

   use anyhow::{Context, Result};

   impl GraphCore {
       pub fn add_node_with_context(&mut self, id: String, attrs: HashMap<String, AttrValue>) 
                                   -> Result<NodeIndex> {
           
           self.validate_node_id(&id)
               .with_context(|| format!("Invalid node ID: {}", id))?;
           
           let index = self.allocate_node_index()
               .with_context(|| "Failed to allocate node index")?;
           
           self.store_node_attributes(index, attrs)
               .with_context(|| format!("Failed to store attributes for node {}", id))?;
           
           Ok(index)
       }
   }

Performance Monitoring
----------------------

Built-in Profiling
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct PerformanceMonitor {
       // Operation timing
       operation_times: HashMap<String, Vec<Duration>>,
       
       // Memory usage tracking
       memory_usage: Vec<MemorySnapshot>,
       
       // Cache hit rates
       cache_stats: CacheStats,
       
       // Thread utilization
       thread_stats: ThreadStats,
   }

   impl PerformanceMonitor {
       pub fn time_operation<F, R>(&mut self, name: &str, f: F) -> R 
       where F: FnOnce() -> R {
           let start = Instant::now();
           let result = f();
           let duration = start.elapsed();
           
           self.operation_times
               .entry(name.to_string())
               .or_insert_with(Vec::new)
               .push(duration);
               
           result
       }
   }

**Memory Tracking**:

.. code-block:: rust

   #[derive(Debug, Clone)]
   pub struct MemorySnapshot {
       timestamp: Instant,
       total_allocated: usize,
       pool_usage: HashMap<String, usize>,
       gc_pressure: f64,
   }

This Rust core architecture provides the high-performance foundation that makes Groggy's Python API both fast and feature-rich. The careful attention to memory management, caching, and parallelization ensures optimal performance across a wide range of graph analysis workloads.
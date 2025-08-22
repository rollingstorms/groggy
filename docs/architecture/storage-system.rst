Storage System Architecture
===========================

Groggy's storage system provides the foundation for high-performance graph analytics through a sophisticated three-layer architecture: Pool, Space, and History. This design enables efficient memory management, fast data access, and support for complex analytical workloads.

Storage Architecture Overview
-----------------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                     History Layer                           │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │ Transaction │ │   Change    │ │      Versioning         │ │
   │  │    Log      │ │   Tracking  │ │      System             │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      Space Layer                            │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │    Node     │ │    Edge     │ │      Attribute          │ │
   │  │   Space     │ │   Space     │ │       Spaces            │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      Pool Layer                             │
   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
   │  │   Memory    │ │   String    │ │      Object             │ │
   │  │   Pools     │ │    Pool     │ │      Pools              │ │
   │  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
   └─────────────────────────────────────────────────────────────┘

Pool Layer: Memory Management
-----------------------------

The Pool Layer provides efficient, type-aware memory allocation with minimal fragmentation and maximum performance.

Memory Pool Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct MemoryPool {
       // Size-stratified pools
       small_pool: Pool<SmallBlock>,     // 8-64 bytes
       medium_pool: Pool<MediumBlock>,   // 65-1024 bytes
       large_pool: Pool<LargeBlock>,     // 1025-65536 bytes
       huge_pool: HugeObjectAllocator,   // > 65536 bytes
       
       // Specialized pools
       string_pool: StringPool,
       attribute_pool: AttributePool,
       
       // Pool management
       allocator: PoolAllocator,
       statistics: AllocationStatistics,
       
       // Configuration
       config: PoolConfig,
   }

**Pool Implementation**:

.. code-block:: rust

   pub struct Pool<T> {
       // Free block management
       free_blocks: Vec<NonNull<T>>,
       free_count: AtomicUsize,
       
       // Allocated chunks
       chunks: Vec<Chunk<T>>,
       chunk_size: usize,
       
       // Allocation statistics
       total_allocated: AtomicUsize,
       peak_usage: AtomicUsize,
       allocation_count: AtomicU64,
       
       // Thread safety
       free_list_lock: Mutex<()>,
   }

   impl<T> Pool<T> {
       pub fn allocate(&self) -> Option<NonNull<T>> {
           // Fast path: try to pop from free list
           if let Some(ptr) = self.try_pop_free() {
               return Some(ptr);
           }
           
           // Slow path: allocate new chunk if needed
           self.allocate_new_chunk()
       }
       
       pub fn deallocate(&self, ptr: NonNull<T>) {
           // Add to free list for reuse
           self.push_free(ptr);
       }
       
       fn try_pop_free(&self) -> Option<NonNull<T>> {
           let _lock = self.free_list_lock.lock().unwrap();
           self.free_blocks.pop()
       }
       
       fn push_free(&self, ptr: NonNull<T>) {
           let _lock = self.free_list_lock.lock().unwrap();
           self.free_blocks.push(ptr);
           self.free_count.fetch_add(1, Ordering::Relaxed);
       }
   }

String Pool Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct StringPool {
       // Contiguous string storage
       data: Vec<u8>,
       
       // String boundary tracking
       offsets: Vec<u32>,
       
       // Deduplication
       string_to_id: HashMap<u64, StringId>,
       hash_to_offset: HashMap<u64, u32>,
       
       // Allocation tracking
       next_offset: AtomicU32,
       string_count: AtomicU32,
       
       // Garbage collection
       gc_threshold: usize,
       deleted_strings: BitSet,
   }

   impl StringPool {
       pub fn intern_string(&mut self, s: &str) -> StringId {
           let hash = self.hash_string(s);
           
           // Check if string already exists
           if let Some(&id) = self.string_to_id.get(&hash) {
               return id;
           }
           
           // Store new string
           let offset = self.next_offset.load(Ordering::Relaxed);
           let bytes = s.as_bytes();
           
           // Ensure capacity
           if self.data.len() + bytes.len() > self.data.capacity() {
               self.grow_storage();
           }
           
           // Append string data
           self.data.extend_from_slice(bytes);
           self.offsets.push(offset);
           
           let id = StringId(self.string_count.fetch_add(1, Ordering::Relaxed));
           self.string_to_id.insert(hash, id);
           self.hash_to_offset.insert(hash, offset);
           
           id
       }
       
       pub fn get_string(&self, id: StringId) -> &str {
           let offset = self.offsets[id.0 as usize];
           let next_offset = self.offsets.get(id.0 as usize + 1)
               .copied()
               .unwrap_or(self.data.len() as u32);
           
           let bytes = &self.data[offset as usize..next_offset as usize];
           unsafe { std::str::from_utf8_unchecked(bytes) }
       }
   }

Space Layer: Logical Organization
---------------------------------

The Space Layer provides logical organization of related data with efficient batch operations and cache-friendly layouts.

Node Space
~~~~~~~~~~

.. code-block:: rust

   pub struct NodeSpace {
       // Primary node storage
       node_ids: Vec<String>,
       id_to_index: HashMap<String, NodeIndex>,
       
       // Node metadata
       node_metadata: Vec<NodeMetadata>,
       
       // Deletion tracking
       deleted_nodes: BitSet,
       free_indices: Vec<NodeIndex>,
       
       // Batch operation support
       pending_additions: Vec<PendingNode>,
       batch_threshold: usize,
       
       // Statistics
       node_count: AtomicUsize,
       total_capacity: usize,
   }

   #[derive(Clone, Debug)]
   pub struct NodeMetadata {
       creation_time: Timestamp,
       last_modified: Timestamp,
       degree: u32,
       attribute_mask: u64,  // Bitmap of which attributes are set
   }

   impl NodeSpace {
       pub fn add_node(&mut self, id: String) -> Result<NodeIndex, SpaceError> {
           // Check for duplicates
           if self.id_to_index.contains_key(&id) {
               return Err(SpaceError::NodeExists(id));
           }
           
           // Reuse deleted index if available
           let index = if let Some(reused_index) = self.free_indices.pop() {
               reused_index
           } else {
               let new_index = NodeIndex(self.node_ids.len());
               self.node_ids.push(id.clone());
               self.node_metadata.push(NodeMetadata::new());
               new_index
           };
           
           // Update mappings
           self.id_to_index.insert(id, index);
           self.node_count.fetch_add(1, Ordering::Relaxed);
           
           Ok(index)
       }
       
       pub fn batch_add_nodes(&mut self, nodes: Vec<String>) -> Result<Vec<NodeIndex>, SpaceError> {
           // Pre-validate all nodes
           for id in &nodes {
               if self.id_to_index.contains_key(id) {
                   return Err(SpaceError::NodeExists(id.clone()));
               }
           }
           
           // Reserve capacity
           let start_index = self.node_ids.len();
           self.node_ids.reserve(nodes.len());
           self.node_metadata.reserve(nodes.len());
           
           // Batch insert
           let mut indices = Vec::with_capacity(nodes.len());
           for (i, id) in nodes.into_iter().enumerate() {
               let index = NodeIndex(start_index + i);
               self.node_ids.push(id.clone());
               self.node_metadata.push(NodeMetadata::new());
               self.id_to_index.insert(id, index);
               indices.push(index);
           }
           
           self.node_count.fetch_add(indices.len(), Ordering::Relaxed);
           
           Ok(indices)
       }
   }

Edge Space
~~~~~~~~~~

.. code-block:: rust

   pub struct EdgeSpace {
       // Edge storage
       edges: Vec<Edge>,
       
       // Source/target indices for fast access
       source_indices: Vec<NodeIndex>,
       target_indices: Vec<NodeIndex>,
       
       // Fast edge lookup
       edge_map: HashMap<(NodeIndex, NodeIndex), EdgeIndex>,
       
       // Adjacency lists for iteration
       out_adjacency: Vec<Vec<EdgeIndex>>,
       in_adjacency: Vec<Vec<EdgeIndex>>,
       
       // Deletion tracking
       deleted_edges: BitSet,
       free_indices: Vec<EdgeIndex>,
       
       // Configuration
       directed: bool,
       allow_self_loops: bool,
       allow_multi_edges: bool,
   }

   impl EdgeSpace {
       pub fn add_edge(&mut self, source: NodeIndex, target: NodeIndex) 
                      -> Result<EdgeIndex, SpaceError> {
           
           // Check for existing edge (if multi-edges not allowed)
           if !self.allow_multi_edges {
               if self.edge_map.contains_key(&(source, target)) {
                   return Err(SpaceError::EdgeExists(source, target));
               }
           }
           
           // Check for self-loops (if not allowed)
           if !self.allow_self_loops && source == target {
               return Err(SpaceError::SelfLoopNotAllowed);
           }
           
           // Allocate edge index
           let edge_index = if let Some(reused_index) = self.free_indices.pop() {
               reused_index
           } else {
               EdgeIndex(self.edges.len())
           };
           
           // Create edge
           let edge = Edge {
               source,
               target,
               weight: None,
               metadata: EdgeMetadata::new(),
           };
           
           // Store edge
           if edge_index.0 >= self.edges.len() {
               self.edges.push(edge);
               self.source_indices.push(source);
               self.target_indices.push(target);
           } else {
               self.edges[edge_index.0] = edge;
               self.source_indices[edge_index.0] = source;
               self.target_indices[edge_index.0] = target;
           }
           
           // Update adjacency lists
           self.out_adjacency[source.0].push(edge_index);
           if self.directed {
               self.in_adjacency[target.0].push(edge_index);
           } else {
               self.out_adjacency[target.0].push(edge_index);
           }
           
           // Update edge map
           self.edge_map.insert((source, target), edge_index);
           
           Ok(edge_index)
       }
   }

Attribute Spaces
~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct AttributeSpace {
       // Column storage by attribute name
       columns: HashMap<String, AttributeColumn>,
       
       // Schema management
       schema: AttributeSchema,
       null_masks: HashMap<String, BitSet>,
       
       // Statistics and indexing
       column_stats: HashMap<String, ColumnStatistics>,
       indices: HashMap<String, AttributeIndex>,
       
       // Memory management
       column_pool: ColumnPool,
       compaction_threshold: f64,
   }

   #[derive(Clone, Debug)]
   pub enum AttributeColumn {
       Int8(Vec<i8>),
       Int16(Vec<i16>),
       Int32(Vec<i32>),
       Int64(Vec<i64>),
       Float32(Vec<f32>),
       Float64(Vec<f64>),
       String(Vec<StringId>),
       Bool(BitSet),
       Categorical(CategoricalColumn),
   }

   impl AttributeSpace {
       pub fn set_attribute(&mut self, entity_index: usize, name: &str, value: AttrValue) 
                           -> Result<(), SpaceError> {
           
           // Get or create column
           let column = self.columns.entry(name.to_string())
               .or_insert_with(|| self.create_column_for_type(&value));
           
           // Ensure column capacity
           if entity_index >= column.len() {
               column.resize(entity_index + 1, AttrValue::Null);
           }
           
           // Set value
           column.set(entity_index, value)?;
           
           // Update null mask
           let null_mask = self.null_masks.entry(name.to_string())
               .or_insert_with(BitSet::new);
           null_mask.set(entity_index, !value.is_null());
           
           // Invalidate statistics
           self.column_stats.remove(name);
           
           Ok(())
       }
       
       pub fn get_attribute(&self, entity_index: usize, name: &str) -> Option<&AttrValue> {
           self.columns.get(name)?.get(entity_index)
       }
       
       pub fn batch_set_attributes(&mut self, updates: Vec<AttributeUpdate>) 
                                  -> Result<(), SpaceError> {
           // Group updates by column for efficiency
           let mut column_updates: HashMap<String, Vec<(usize, AttrValue)>> = HashMap::new();
           
           for update in updates {
               column_updates.entry(update.attribute_name)
                   .or_insert_with(Vec::new)
                   .push((update.entity_index, update.value));
           }
           
           // Apply updates column by column
           for (column_name, updates) in column_updates {
               let column = self.columns.entry(column_name.clone())
                   .or_insert_with(|| self.create_default_column());
               
               for (index, value) in updates {
                   column.set(index, value)?;
               }
               
               // Invalidate statistics
               self.column_stats.remove(&column_name);
           }
           
           Ok(())
       }
   }

History Layer: Versioning and Transactions
-------------------------------------------

The History Layer provides transaction support, versioning, and consistency guarantees.

Transaction Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct TransactionManager {
       // Active transactions
       active_transactions: HashMap<TransactionId, Transaction>,
       next_transaction_id: AtomicU64,
       
       // Change log
       change_log: Vec<ChangeRecord>,
       log_compaction_threshold: usize,
       
       // Snapshot management
       snapshots: LruCache<SnapshotId, Snapshot>,
       
       // Configuration
       max_active_transactions: usize,
       enable_durability: bool,
   }

   #[derive(Clone, Debug)]
   pub struct Transaction {
       id: TransactionId,
       start_time: Timestamp,
       changes: Vec<Change>,
       state: TransactionState,
       isolation_level: IsolationLevel,
   }

   #[derive(Clone, Debug)]
   pub enum Change {
       NodeAdded { index: NodeIndex, id: String },
       NodeRemoved { index: NodeIndex, id: String },
       EdgeAdded { index: EdgeIndex, source: NodeIndex, target: NodeIndex },
       EdgeRemoved { index: EdgeIndex, source: NodeIndex, target: NodeIndex },
       AttributeChanged { 
           entity_type: EntityType,
           entity_index: usize,
           attribute_name: String,
           old_value: Option<AttrValue>,
           new_value: AttrValue,
       },
   }

   impl TransactionManager {
       pub fn begin_transaction(&mut self, isolation_level: IsolationLevel) 
                               -> Result<TransactionId, TransactionError> {
           
           if self.active_transactions.len() >= self.max_active_transactions {
               return Err(TransactionError::TooManyActiveTransactions);
           }
           
           let transaction_id = TransactionId(
               self.next_transaction_id.fetch_add(1, Ordering::Relaxed)
           );
           
           let transaction = Transaction {
               id: transaction_id,
               start_time: Timestamp::now(),
               changes: Vec::new(),
               state: TransactionState::Active,
               isolation_level,
           };
           
           self.active_transactions.insert(transaction_id, transaction);
           
           Ok(transaction_id)
       }
       
       pub fn commit_transaction(&mut self, transaction_id: TransactionId) 
                                -> Result<(), TransactionError> {
           
           let mut transaction = self.active_transactions.remove(&transaction_id)
               .ok_or(TransactionError::TransactionNotFound)?;
           
           // Validate transaction can be committed
           self.validate_transaction(&transaction)?;
           
           // Apply changes atomically
           for change in &transaction.changes {
               self.apply_change(change)?;
           }
           
           // Add to change log
           let change_record = ChangeRecord {
               transaction_id,
               timestamp: Timestamp::now(),
               changes: transaction.changes.clone(),
           };
           
           self.change_log.push(change_record);
           
           // Compact log if needed
           if self.change_log.len() > self.log_compaction_threshold {
               self.compact_change_log()?;
           }
           
           Ok(())
       }
       
       pub fn rollback_transaction(&mut self, transaction_id: TransactionId) 
                                  -> Result<(), TransactionError> {
           
           let transaction = self.active_transactions.remove(&transaction_id)
               .ok_or(TransactionError::TransactionNotFound)?;
           
           // Undo changes in reverse order
           for change in transaction.changes.into_iter().rev() {
               self.undo_change(&change)?;
           }
           
           Ok(())
       }
   }

Snapshot System
~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct SnapshotManager {
       // Snapshot storage
       snapshots: HashMap<SnapshotId, Snapshot>,
       
       // Incremental snapshots
       incremental_snapshots: HashMap<SnapshotId, IncrementalSnapshot>,
       
       // Snapshot metadata
       snapshot_metadata: HashMap<SnapshotId, SnapshotMetadata>,
       
       // Configuration
       max_snapshots: usize,
       snapshot_interval: Duration,
       compression_enabled: bool,
   }

   #[derive(Clone)]
   pub struct Snapshot {
       id: SnapshotId,
       timestamp: Timestamp,
       
       // Graph state
       node_space: NodeSpace,
       edge_space: EdgeSpace,
       attribute_spaces: HashMap<String, AttributeSpace>,
       
       // Metadata
       graph_config: GraphConfig,
       statistics: GraphStatistics,
   }

   impl SnapshotManager {
       pub fn create_snapshot(&mut self, graph: &GraphCore) -> Result<SnapshotId, SnapshotError> {
           let snapshot_id = SnapshotId::new();
           
           let snapshot = Snapshot {
               id: snapshot_id,
               timestamp: Timestamp::now(),
               node_space: graph.node_space.clone(),
               edge_space: graph.edge_space.clone(),
               attribute_spaces: graph.attribute_spaces.clone(),
               graph_config: graph.config.clone(),
               statistics: graph.compute_statistics(),
           };
           
           // Compress snapshot if enabled
           let final_snapshot = if self.compression_enabled {
               self.compress_snapshot(snapshot)?
           } else {
               snapshot
           };
           
           // Store snapshot
           self.snapshots.insert(snapshot_id, final_snapshot);
           
           // Cleanup old snapshots if needed
           if self.snapshots.len() > self.max_snapshots {
               self.cleanup_old_snapshots();
           }
           
           Ok(snapshot_id)
       }
       
       pub fn restore_snapshot(&self, snapshot_id: SnapshotId, graph: &mut GraphCore) 
                              -> Result<(), SnapshotError> {
           
           let snapshot = self.snapshots.get(&snapshot_id)
               .ok_or(SnapshotError::SnapshotNotFound)?;
           
           // Restore graph state
           graph.node_space = snapshot.node_space.clone();
           graph.edge_space = snapshot.edge_space.clone();
           graph.attribute_spaces = snapshot.attribute_spaces.clone();
           graph.config = snapshot.graph_config.clone();
           
           // Rebuild derived structures
           graph.rebuild_indices()?;
           graph.rebuild_caches()?;
           
           Ok(())
       }
   }

Storage View Integration
------------------------

Unified Access Layer
~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub struct StorageViewManager {
       // Core storage references
       node_space: Arc<RwLock<NodeSpace>>,
       edge_space: Arc<RwLock<EdgeSpace>>,
       attribute_spaces: Arc<RwLock<HashMap<String, AttributeSpace>>>,
       
       // View caches
       array_cache: LruCache<ArrayKey, Arc<ArrayData>>,
       matrix_cache: LruCache<MatrixKey, Arc<MatrixData>>,
       table_cache: LruCache<TableKey, Arc<TableData>>,
       
       // Configuration
       cache_size: usize,
       lazy_loading: bool,
   }

   impl StorageViewManager {
       pub fn get_array_view(&self, attribute_name: &str, entity_type: EntityType) 
                            -> Result<GraphArray, StorageError> {
           
           let cache_key = ArrayKey {
               attribute_name: attribute_name.to_string(),
               entity_type,
           };
           
           // Check cache first
           if let Some(cached_data) = self.array_cache.get(&cache_key) {
               return Ok(GraphArray::from_cached_data(cached_data.clone()));
           }
           
           // Load data from storage
           let data = match entity_type {
               EntityType::Node => {
                   let node_space = self.node_space.read().unwrap();
                   let attr_space = self.attribute_spaces.read().unwrap();
                   
                   if let Some(space) = attr_space.get(attribute_name) {
                       space.get_column_data(attribute_name)?
                   } else {
                       return Err(StorageError::AttributeNotFound(attribute_name.to_string()));
                   }
               },
               EntityType::Edge => {
                   // Similar logic for edges
                   unimplemented!()
               },
           };
           
           // Create array view
           let array_data = Arc::new(ArrayData::from_column_data(data));
           self.array_cache.put(cache_key, array_data.clone());
           
           Ok(GraphArray::from_cached_data(array_data))
       }
       
       pub fn get_table_view(&self, entity_type: EntityType, attributes: Option<Vec<String>>) 
                            -> Result<GraphTable, StorageError> {
           
           let cache_key = TableKey {
               entity_type,
               attributes: attributes.clone(),
           };
           
           // Check cache
           if let Some(cached_data) = self.table_cache.get(&cache_key) {
               return Ok(GraphTable::from_cached_data(cached_data.clone()));
           }
           
           // Load multiple columns
           let table_data = match entity_type {
               EntityType::Node => {
                   let node_space = self.node_space.read().unwrap();
                   let attr_spaces = self.attribute_spaces.read().unwrap();
                   
                   let columns = if let Some(attr_list) = attributes {
                       attr_list
                   } else {
                       attr_spaces.keys().cloned().collect()
                   };
                   
                   let mut table_columns = HashMap::new();
                   for attr_name in columns {
                       if let Some(space) = attr_spaces.get(&attr_name) {
                           let column_data = space.get_column_data(&attr_name)?;
                           table_columns.insert(attr_name, column_data);
                       }
                   }
                   
                   TableData::from_columns(table_columns)
               },
               EntityType::Edge => {
                   // Similar logic for edges
                   unimplemented!()
               },
           };
           
           let table_data_arc = Arc::new(table_data);
           self.table_cache.put(cache_key, table_data_arc.clone());
           
           Ok(GraphTable::from_cached_data(table_data_arc))
       }
   }

Performance Optimizations
-------------------------

Columnar Compression
~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   pub trait ColumnCompression {
       fn compress(&self) -> Result<CompressedColumn, CompressionError>;
       fn decompress(compressed: &CompressedColumn) -> Result<Self, CompressionError>
       where Self: Sized;
   }

   impl ColumnCompression for Vec<i64> {
       fn compress(&self) -> Result<CompressedColumn, CompressionError> {
           // Use delta encoding for sorted sequences
           if self.is_sorted() {
               let deltas = self.windows(2)
                   .map(|w| w[1] - w[0])
                   .collect::<Vec<_>>();
               
               // Use variable-length encoding for small deltas
               let encoded = varint_encode(&deltas)?;
               
               Ok(CompressedColumn::DeltaEncoded {
                   base_value: self[0],
                   encoded_deltas: encoded,
               })
           } else {
               // Use general-purpose compression
               let compressed = zstd::compress(self.as_bytes(), 3)?;
               Ok(CompressedColumn::ZstdCompressed(compressed))
           }
       }
   }

Cache-Aware Layouts
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   // Structure of Arrays (SoA) vs Array of Structures (AoS)
   
   // AoS: Poor cache locality for column operations
   struct NodeAoS {
       id: String,
       x: f64,
       y: f64,
       weight: f64,
   }
   
   // SoA: Optimal cache locality for column operations
   struct NodesSoA {
       ids: Vec<String>,
       x_coords: Vec<f64>,
       y_coords: Vec<f64>,
       weights: Vec<f64>,
   }

   impl NodesSoA {
       // Cache-friendly iteration over coordinates
       pub fn compute_distances(&self) -> Vec<f64> {
           self.x_coords.iter()
               .zip(self.y_coords.iter())
               .map(|(&x, &y)| (x * x + y * y).sqrt())
               .collect()
       }
   }

SIMD-Optimized Operations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[cfg(target_arch = "x86_64")]
   pub mod simd_ops {
       use std::arch::x86_64::*;
       
       pub unsafe fn sum_f64_column(data: &[f64]) -> f64 {
           let mut sum = _mm256_setzero_pd();
           let chunks = data.len() / 4;
           
           // Process 4 elements at a time
           for i in 0..chunks {
               let values = _mm256_loadu_pd(data.as_ptr().add(i * 4));
               sum = _mm256_add_pd(sum, values);
           }
           
           // Extract sum from SIMD register
           let mut result = [0.0; 4];
           _mm256_storeu_pd(result.as_mut_ptr(), sum);
           
           let mut total = result.iter().sum::<f64>();
           
           // Handle remaining elements
           for &value in &data[chunks * 4..] {
               total += value;
           }
           
           total
       }
       
       pub unsafe fn filter_gt_f64(data: &[f64], threshold: f64) -> Vec<usize> {
           let threshold_vec = _mm256_set1_pd(threshold);
           let mut indices = Vec::new();
           
           let chunks = data.len() / 4;
           
           for i in 0..chunks {
               let values = _mm256_loadu_pd(data.as_ptr().add(i * 4));
               let mask = _mm256_cmp_pd(values, threshold_vec, _CMP_GT_OQ);
               let mask_int = _mm256_movemask_pd(mask);
               
               // Extract indices where condition is true
               for j in 0..4 {
                   if (mask_int & (1 << j)) != 0 {
                       indices.push(i * 4 + j);
                   }
               }
           }
           
           // Handle remaining elements
           for (j, &value) in data[chunks * 4..].iter().enumerate() {
               if value > threshold {
                   indices.push(chunks * 4 + j);
               }
           }
           
           indices
       }
   }

This sophisticated storage system architecture enables Groggy to deliver exceptional performance for graph analytics while maintaining data integrity and providing flexible access patterns through the storage view abstraction layer.
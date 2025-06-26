# GLI - Complete Internal Documentation
*Every method, property, and internal mechanism documented*

---

## Content-Addressed Storage System

### `ContentPool`
**Purpose**: Deduplicates nodes and edges using content-addressing to save memory.

#### Public Methods
- `intern_node(node: Node) -> str`: Store node in pool, return content hash
- `intern_edge(edge: Edge) -> str`: Store edge in pool, return content hash  
- `get_node(content_hash: str) -> Optional[Node]`: Retrieve node by hash
- `get_edge(content_hash: str) -> Optional[Edge]`: Retrieve edge by hash
- `release_node(content_hash: str)`: Decrement reference count, cleanup if unused
- `release_edge(content_hash: str)`: Decrement reference count, cleanup if unused

#### Private Methods
- `_node_content_hash(node: Node) -> str`: Compute content hash for node
  - Uses JSON serialization of attributes + fast_hash
  - Caches results in `_node_hash_cache`
  - Cache key: `(node.id, hash(attrs_str))`
  
- `_edge_content_hash(edge: Edge) -> str`: Compute content hash for edge
  - Uses JSON serialization of attributes + fast_hash  
  - Caches results in `_edge_hash_cache`
  - Cache key: `(edge.source, edge.target, hash(attrs_str))`

#### Internal State
- `nodes: Dict[str, Node]`: content_hash -> Node storage
- `edges: Dict[str, Edge]`: content_hash -> Edge storage
- `node_refs: Dict[str, int]`: Reference counting for nodes
- `edge_refs: Dict[str, int]`: Reference counting for edges
- `_node_hash_cache: Dict[tuple, str]`: Hash computation cache
- `_edge_hash_cache: Dict[tuple, str]`: Hash computation cache

**Issues**: Reference counting implementation incomplete, potential memory leaks.

---

## Graph Data Model

### `Node`
**Purpose**: Immutable graph vertex with attributes.

#### Public Methods
- `get_attribute(key: str, default=None)`: Safe attribute access
- `set_attribute(key: str, value: Any) -> Node`: Returns new Node with updated attribute

#### Internal State
- `id: str`: Unique node identifier
- `attributes: Dict[str, Any]`: Node metadata/properties

### `Edge`
**Purpose**: Immutable graph edge with source/target and attributes.

#### Public Methods
- `get_attribute(key: str, default=None)`: Safe attribute access
- `set_attribute(key: str, value: Any) -> Edge`: Returns new Edge with updated attribute

#### Properties
- `id -> str`: Computed edge identifier as `f"{source}->{target}"`

#### Internal State
- `source: str`: Source node ID
- `target: str`: Target node ID  
- `attributes: Dict[str, Any]`: Edge metadata/properties

---

## Change Tracking System

### `GraphDelta`
**Purpose**: Tracks pending changes to avoid full graph copies.

#### Internal State
- `added_nodes: Dict[str, Node]`: node_id -> Node for new nodes
- `removed_nodes: set`: Set of node_ids to remove
- `modified_nodes: Dict[str, Node]`: node_id -> Node for changed nodes
- `added_edges: Dict[str, Edge]`: edge_id -> Edge for new edges
- `removed_edges: set`: Set of edge_ids to remove
- `modified_edges: Dict[str, Edge]`: edge_id -> Edge for changed edges
- `modified_graph_attrs: Dict[str, Any]`: Graph-level attribute changes

### `CompactGraphDelta`
**Purpose**: Memory-efficient delta using content hashes instead of full objects.

#### Public Methods
- `is_empty() -> bool`: Check if delta has any changes

#### Internal State
- `added_nodes: Dict[str, str]`: node_id -> content_hash
- `removed_nodes: set`: node_ids to remove
- `modified_nodes: Dict[str, str]`: node_id -> content_hash  
- `added_edges: Dict[str, str]`: edge_id -> content_hash
- `removed_edges: set`: edge_ids to remove
- `modified_edges: Dict[str, str]`: edge_id -> content_hash
- `modified_graph_attrs: Dict[str, Any]`: Graph attribute changes

---

## State Management

### `Branch`
**Purpose**: Named branch metadata with subgraph support.

#### Internal State
- `name: str`: Branch identifier
- `current_hash: str`: Latest state hash in this branch
- `created_from: str`: Hash where branch was created from
- `created_at: float`: Timestamp of branch creation
- `description: str`: Human-readable branch description
- `is_subgraph: bool`: Whether this branch operates on a subgraph
- `subgraph_filter: Optional[Dict[str, Any]]`: Filter criteria for subgraph
- `parent_branch: Optional[str]`: Parent branch for subgraph branches

### `GraphState`
**Purpose**: Point-in-time graph snapshot with git-like versioning.

#### Public Methods
- `is_root() -> bool`: Returns `parent_hash is None`
- `to_dict(content_pool: ContentPool) -> Dict`: Serialize state for hashing
- `_compute_effective_content()`: **BROKEN** - supposed to resolve delta chains

#### Internal State
- `hash: str`: Unique state identifier
- `parent_hash: Optional[str]`: Previous state (enables history chain)
- `operation: Optional[str]`: Description of change that created this state
- `timestamp: float`: When state was created
- `branch_name: Optional[str]`: Which branch this state belongs to

**Root State Fields** (for full snapshots):
- `nodes: Optional[Dict[str, str]]`: node_id -> content_hash mapping
- `edges: Optional[Dict[str, str]]`: edge_id -> content_hash mapping  
- `graph_attributes: Optional[Dict[str, Any]]`: Graph-level attributes

**Delta State Fields** (currently unused due to bugs):
- `delta: Optional[CompactGraphDelta]`: Changes from parent state

**Subgraph Fields**:
- `is_subgraph_state: bool`: Whether this state represents a subgraph
- `subgraph_metadata: Optional[Dict[str, Any]]`: Subgraph-specific metadata

---

## Graph Operations

### `Graph`
**Purpose**: Main graph class with lazy copy-on-write and batch operations.

#### Public Methods

**Construction**:
- `__init__(nodes=None, edges=None, graph_attributes=None, graph_store=None)`
- `empty(graph_store=None) -> Graph`: Class method to create empty graph

**Basic Operations**:
- `add_node(node_id: str, **attributes) -> Graph`: Add single node
- `add_edge(source: str, target: str, **attributes) -> Graph`: Add single edge
- `batch_add_nodes(node_data: List[tuple]) -> Graph`: Bulk node addition
- `batch_add_edges(edge_data: List[tuple]) -> Graph`: Bulk edge addition
- `snapshot() -> Graph`: Create immutable copy with applied changes

**Subgraph Operations**:
- `create_subgraph(node_filter=None, edge_filter=None, node_ids=None, include_edges=True) -> Graph`
- `get_subgraph_by_attribute(node_attr: str, attr_value: Any) -> Graph`
- `get_connected_component(start_node_id: str) -> Graph`: BFS-based component extraction

**Export Operations**:
- `to_networkx()`: Convert to NetworkX graph (if available)
- `to_graphml() -> str`: Export to GraphML XML format

#### Private Methods

**Copy-on-Write Management**:
- `_init_delta()`: **Lazy initialization** of change tracking
  - Only creates delta when first modification happens
  - Sets `_is_modified = True`
  - Initializes `_pending_delta = GraphDelta()`
  - **Does NOT copy collections immediately** (true lazy)

- `_ensure_writable()`: **True copy-on-write implementation**
  - Only called on first actual write operation
  - Copies `nodes`, `edges`, and `graph_attributes` at write time
  - Most complex part of copy-on-write system

**Effective Data Management**:
- `_get_effective_data() -> Tuple[Dict, Dict, Dict]`: **Core method**
  - Merges base data with pending changes
  - Returns `(effective_nodes, effective_edges, effective_attrs)`
  - Uses `_effective_cache` for performance
  - Cache invalidated by `_invalidate_cache()`
  - **Heavy computation** - applies all delta changes without modifying originals

- `_invalidate_cache()`: Cache invalidation
  - Sets `_effective_cache = None`
  - Sets `_cache_valid = False`
  - Called after every modification

**Change Application**:
- `_apply_pending_changes()`: **Commits pending delta to base collections**
  - Applies all changes from `_pending_delta` to `nodes`/`edges`/`graph_attributes`
  - Updates `node_order` and `edge_order` tracking
  - Clears `_pending_delta = None`
  - **Critical for snapshot creation**

**Utility**:
- `_next_time() -> int`: Increments and returns `_current_time` for ordering

#### Internal State

**Core Data**:
- `nodes: Dict[str, Node]`: Node storage (node_id -> Node)
- `edges: Dict[str, Edge]`: Edge storage (edge_id -> Edge)  
- `graph_attributes: Dict[str, Any]`: Graph-level metadata
- `graph_store: Optional[GraphStore]`: Reference to parent store

**Ordering Tracking**:
- `node_order: Dict[str, int]`: node_id -> insertion_time
- `edge_order: Dict[str, int]`: edge_id -> insertion_time
- `_current_time: int`: Monotonic counter for insertion order

**Copy-on-Write State**:
- `_pending_delta: Optional[GraphDelta]`: Pending changes (None until first modification)
- `_is_modified: bool`: Whether graph has uncommitted changes
- `_effective_cache: Optional[Tuple]`: Cached result of `_get_effective_data()`
- `_cache_valid: bool`: Whether effective cache is valid

**Branch/Subgraph Metadata**:
- `branch_name: Optional[str]`: Which branch this graph belongs to
- `is_subgraph: bool`: Whether this is a subgraph
- `subgraph_metadata: Dict`: Subgraph-specific metadata

---

## Storage and History

### `GraphStore`
**Purpose**: Main controller managing states, branches, content pool, and operations.

#### Public Methods

**Graph Management**:
- `__init__(max_auto_states=10, prune_old_states=True, snapshot_interval=50, enable_disk_cache=False)`
- `get_current_graph() -> Graph`: Get current graph with lazy reconstruction
- `update_graph(new_graph: Graph, operation: str = "update") -> str`: Create new state

**Branch Operations**:
- `create_branch(branch_name: str, from_hash: str = None, description: str = "", from_subgraph: Graph = None) -> str`
- `switch_branch(branch_name: str) -> Graph`: Switch active branch
- `merge_branch(source_branch: str, target_branch: str = None, strategy: str = "auto", message: str = "") -> str`
- `delete_branch(branch_name: str, force: bool = False)`: Remove branch
- `list_branches() -> List[Dict[str, Any]]`: Get all branch info
- `get_branch_diff(branch1: str, branch2: str) -> Dict[str, Any]`: Compare branches

**History Operations**:
- `commit(message: str = "") -> str`: Explicitly commit current state
- `undo() -> Graph`: Revert to previous auto-state
- `get_history(commits_only: bool = False) -> List[Dict]`: Get state history

**Module System**:
- `run_module(module_name: str, **params) -> Graph`: Execute loaded module
- `load_module_config(config_path: str)`: Load module configuration

**Maintenance**:
- `get_storage_stats() -> Dict[str, Any]`: Memory and state statistics

#### Private Methods

**State Creation** (Core Logic):
- `_create_initial_state()`: Creates "initial" empty state as root
- `_create_snapshot_state(graph: Graph, operation: str) -> str`: **Primary state creation**
  - Stores all nodes/edges in content pool via `intern_node()`/`intern_edge()`
  - Creates state data dictionary for hashing
  - Computes state hash via `_compute_hash()`
  - Creates `GraphState` with full snapshot data
  - **Currently used for ALL states** (deltas disabled)

- `_create_compact_delta(old_graph: Graph, new_graph: Graph) -> CompactGraphDelta`: 
  - **Currently unused** due to reconstruction bugs
  - Compares old vs new graphs to find changes
  - Creates compact delta with content hashes
  - **Should be used for memory efficiency**

- `_create_subgraph_snapshot(subgraph: Graph, operation: str) -> str`:
  - Specialized snapshot creation for subgraphs
  - Includes subgraph metadata in state data
  - Forces `subgraph.snapshot()` to apply changes

**State Reconstruction**:
- `_reconstruct_graph_from_state(state_hash: str) -> Graph`: **Critical method**
  - Retrieves nodes/edges from content pool using stored hashes
  - Reconstructs full Graph object
  - Uses weak reference caching in `_reconstructed_cache`
  - **Currently only handles snapshot states**
  - **Does NOT handle delta states** (broken)

- `_apply_delta_to_graph(base_graph: Graph, delta: CompactGraphDelta) -> Graph`:
  - **Unused due to delta reconstruction being broken**
  - Should apply compact delta to base graph
  - Creates new Graph with combined state

**Hash and Optimization**:
- `_compute_hash(graph_data: Dict) -> str`: JSON serialize + fast_hash
- `_should_create_snapshot() -> bool`: **Always returns True** (deltas disabled)
- `_prune_old_states()`: Garbage collection of old states

**Branch Management**:
- `_merge_graphs(source: Graph, target: Graph, strategy: str) -> Graph`: **Unimplemented**
- `_merge_subgraph(subgraph: Graph, target: Graph, strategy: str) -> Graph`: **Unimplemented**

**Module Integration**:
- `_auto_run_module(event_data, module_name)`: Event-triggered module execution

#### Internal State

**State Storage**:
- `states: Dict[str, GraphState]`: All graph states (hash -> GraphState)
- `auto_states: deque`: Recent states with size limit (`maxlen=max_auto_states`)
- `commits: Dict[str, dict]`: Explicitly committed states (hash -> commit_info)
- `current_hash: Optional[str]`: Active state identifier
- `current_graph: Graph`: Current graph instance

**Content Management**:
- `content_pool: ContentPool`: Deduplication system

**Branch System**:
- `branches: Dict[str, Branch]`: All branches (name -> Branch)
- `current_branch: str`: Active branch name (default: "main")
- `branch_heads: Dict[str, str]`: Latest hash per branch (name -> hash)

**Optimization Settings**:
- `max_auto_states: int`: Auto-state retention limit
- `prune_old_states: bool`: Whether to garbage collect
- `snapshot_interval: int`: **Currently ignored** (all snapshots)
- `enable_disk_cache: bool`: **Unimplemented** future feature
- `state_count: int`: Total states created counter

**Caching**:
- `_reconstructed_cache: Dict[str, weakref.ref]`: Weak refs to reconstructed graphs

**Module System**:
- `module_registry: ModuleRegistry`: Dynamic module loader
- `event_bus: EventBus`: Event coordination system

---

## Module System

### `EventBus`
**Purpose**: Simple pub-sub event coordination.

#### Public Methods
- `subscribe(event_name: str, callback: Callable, *args, **kwargs)`: Register event handler
- `emit(event_name: str, data: Any = None)`: Broadcast event to all subscribers

#### Internal State
- `subscribers: Dict[str, List[tuple]]`: event_name -> [(callback, args, kwargs), ...]

### `ModuleRegistry`
**Purpose**: Dynamic module loading and validation.

#### Public Methods
- `__init__(config_path: Optional[str] = None)`: Initialize, optionally load config
- `load_from_config()`: Load modules from YAML configuration
- `load_module(module_config: Dict)`: Load single module from config
- `get_module(name: str)`: Retrieve loaded module info
- `list_modules() -> List[str]`: Get available module names

#### Internal State
- `modules: Dict[str, Dict]`: Loaded modules with metadata
  - Structure: `name -> {'process': func, 'config': dict, 'metadata': dict, 'module_ref': module}`
- `config_path: Optional[str]`: Path to YAML configuration file

**Module Loading Process**:
1. Uses `importlib.util.spec_from_file_location()` for dynamic loading
2. Validates module has required `process` function
3. Extracts optional `METADATA` attribute
4. Stores all module information for later execution

---

## Utility Functions

### Global Functions

#### `fast_hash(data: str) -> str`
**Implementation**:
```python
try:
    import xxhash
    def fast_hash(data: str) -> str:
        return xxhash.xxh64(data.encode()).hexdigest()[:16]
except ImportError:
    import hashlib
    def fast_hash(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()[:16]
```
**Usage**: Content addressing, state hashing, cache keys throughout system.

#### `create_random_graph(n_nodes: int = 10, edge_probability: float = 0.3) -> Graph`
**Implementation Details**:
- Uses batch operations for efficiency
- Creates all nodes first with `batch_add_nodes()`
- Generates edges with random probability
- Uses `batch_add_edges()` for bulk edge creation
- Returns `graph.snapshot()` to ensure immutability

#### Workflow Helpers
- `create_clustering_workflow(store: GraphStore, graph: Graph, algorithms: List[str] = None) -> List[str]`
  - Creates branches for different clustering algorithms
  - Default algorithms: `['kmeans', 'spectral', 'hierarchical']`
  - Returns list of created branch names

- `create_subgraph_branch(store: GraphStore, subgraph: Graph, branch_name: str, description: str = "") -> str`
  - Creates branch specifically from a subgraph
  - Wrapper around `store.create_branch()` with subgraph support

---

## Critical Implementation Issues

### 1. **Broken Delta System**
**Location**: `GraphStore._should_create_snapshot()`
```python
def _should_create_snapshot(self) -> bool:
    # Always create snapshots for now to fix the reconstruction issue
    # TODO: Re-enable delta states once reconstruction is working properly
    return True
```
**Impact**: All states are full snapshots, using excessive memory.

### 2. **Incomplete Reference Counting**
**Location**: `ContentPool.intern_node()`, `ContentPool.release_node()`
- Reference counting logic is present but incomplete
- Potential memory leaks from unreleased content
- No automatic cleanup of unused content

### 3. **Complex Copy-on-Write**
**Location**: `Graph._init_delta()`, `Graph._ensure_writable()`, `Graph._get_effective_data()`
- Three-layer caching system prone to invalidation bugs
- Copy-on-write triggers are scattered throughout code
- Cache invalidation happens frequently, reducing benefits

### 4. **Unimplemented Branch Merging**
**Location**: `GraphStore._merge_graphs()`, `GraphStore._merge_subgraph()`
- Core merge logic is missing
- Branch merging advertised but non-functional
- No conflict resolution strategies

### 5. **Missing Error Handling**
- No exception handling for module loading failures
- State reconstruction failures not handled gracefully
- Branch operations lack validation
- File I/O operations missing error handling

### 6. **Memory Management Issues**
- Weak reference cache may not prevent memory leaks
- Content pool reference counting incomplete
- No explicit cleanup methods for large graphs
- Auto-state pruning logic incomplete

### 7. **Inconsistent State**
**Location**: Various methods throughout `Graph` and `GraphStore`
- Mix of mutable and immutable operations
- Unclear when changes are applied vs pending
- State consistency not guaranteed across operations

---

## Performance Bottlenecks

### 1. **Frequent Cache Invalidation**
- `_invalidate_cache()` called after every modification
- `_get_effective_data()` recomputed frequently
- No incremental cache updates

### 2. **JSON Serialization Overhead**
- Every hash computation involves JSON serialization
- No binary serialization options
- Repeated serialization of same data

### 3. **Memory Duplication**
- Forced snapshots duplicate entire graph state
- Copy-on-write benefits negated by cache invalidation
- No lazy loading of historical states

### 4. **Inefficient Graph Traversal**
- Subgraph operations scan entire graph
- No indexing for attribute-based queries
- BFS implementation creates unnecessary data structures

---

## Testing and Validation Gaps

### 1. **No Unit Tests**
- Complex internal methods untested
- Copy-on-write behavior unvalidated
- State reconstruction logic unverified

### 2. **No Integration Tests**
- Branch operations end-to-end untested
- Module system integration unclear
- Performance characteristics unknown

### 3. **No Benchmarks**
- Memory usage patterns unknown
- Performance comparison with alternatives missing
- Scalability limits undefined

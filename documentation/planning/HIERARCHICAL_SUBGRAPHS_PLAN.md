# Hierarchical Subgraphs Plan: Nodes as Subgraphs

## 🎯 **Core Vision**

Create a powerful hierarchical graph system where subgraphs can be collapsed into meta-nodes and nodes can expand back into their constituent subgraphs. This enables multi-level graph analysis, community aggregation, and sophisticated graph transformations.

## 🏗️ **Core Architecture**

### Primary Interface
```python
# Subgraph to Node conversion
meta_node = subgraph.add_to_graph()           # add subgraph to graph -> node
meta_nodes = g.add_subgraphs(subgraph_list)   # Batch add subgraphs to graph -> nodes

# Node to Subgraph access
subgraph = node[0].subgraph                   # Access contained subgraph
subgraphs = g.nodes.subgraphs                # All subgraph-nodes as array
node[0].subgraph = subgraph                   # set subgraph of node - no automatic edge handling

# Containment and queries
subgraph.contains_nodes([1, 2, 3])           # Check node membership
subgraph_array = PyArray(subgraph_list)      # Array of subgraphs # requires subgraph support in pyarray not yet implemented
                                             # returns from methods like connected_components() will be PyArray[Subgraph]
```

## 🔄 **Bidirectional Transformation System**

### Collapse: Subgraph → Meta-Node
```python
# Basic collapse
community = g.nodes([1, 2, 3, 4])
meta_node = community.add_to_graph(
    attributes={
        "size": len(community),
        "density": community.density(),
        "avg_degree": community.degree().mean()
    }
)

# Batch collapse with aggregation
communities = g.communities.louvain()
meta_nodes = g.add_subgraphs(
    communities,
    agg_functions={
        "size": "count",
        "total_weight": ("weight", "sum"), 
        "avg_centrality": ("betweenness", "mean"),
        "dominant_type": ("node_type", "mode")
    }
)
```

### Expand: Meta-Node → Subgraph
```python
# Access subgraph from meta-node
meta_node = g.nodes[100]
if meta_node.is_subgraph_node:
    original_subgraph = meta_node.subgraph
    print(f"Contains {len(original_subgraph)} nodes")
    
    # # Expand back into main graph
    # expanded_nodes = meta_node.expand()
    
    # Create new separate graph
    separate_graph = meta_node.subgraph.to_graph()
```

## 🗂️ **Subgraph Arrays and Collections**

### PyArray Integration
```python
# Subgraphs as first-class arrays
communities = g.communities.louvain()         # Returns PyArray[Subgraph]

# Array operations on subgraphs
sizes = communities.map(lambda sg: len(sg))
densities = communities.map(lambda sg: sg.density())
large_communities = communities.filter(lambda sg: len(sg) > 10)

# Batch operations
meta_nodes = subgraphs.add_to_graph()          # Collapse all
aggregated = subgraphs.aggregate({
    "total_nodes": "sum:size",
    "avg_clustering": "mean:clustering"
}, inplace=True)
```

### Collection Methods
```python
# Access all subgraph-nodes
subgraph_nodes = g.nodes.subgraphs            # PyArray[Node] where is_subgraph_node=True
```

## 🔗 **Advanced Edge Handling in Hierarchical Graphs**

### Two-Tier Entity System Architecture
```python
# === ACCESSORS (Filtered Views) ===
g.nodes           # All nodes accessor (base + meta)
g.nodes.base      # Base/regular nodes accessor  
g.nodes.meta      # Meta-nodes accessor
g.edges           # All edges accessor (base + meta)
g.edges.base      # Base/regular edges accessor
g.edges.meta      # Meta-edges accessor

# === SUBGRAPHS (Actionable) ===
g.nodes()         # All nodes subgraph
g.nodes.base()    # Base nodes subgraph
g.nodes.meta()    # Meta-nodes subgraph

# === TABLES (Data Views) ===
g.nodes.table()      # All nodes table
g.nodes.base.table() # Base nodes table (auto-sliced NaN columns)
g.nodes.meta.table() # Meta-nodes table (auto-sliced NaN columns)
```

### Meta-Edge Creation During Collapse
```python
# Advanced collapse with comprehensive edge handling
meta_node = subgraph.collapse_to_node(
    agg_functions={"salary": "sum", "age": "mean"},
    defaults={"bonus": 0},           # Default values for missing attributes
    
    # Edge handling options
    edge_to_external="copy",         # copy, aggregate, count
    edge_to_meta="auto",             # auto, explicit, none  
    meta_edge_agg="sum"              # How to aggregate parallel meta-edges
)
```

### Meta-Edge Types and Behavior
```python
# Type 1: Child-to-External Meta-edges (always created)
# When nodes [1,2,3] collapse to meta_node_A:
# - If node 1 → external_node, creates meta_node_A → external_node
# - If node 2 → external_node, aggregates to strengthen meta_node_A → external_node

# Type 2: Meta-to-Meta edges (created when applicable)  
# If collapsed nodes had edges to nodes that are now in other meta-nodes:
# - node 1 → node 4, and node 4 is now in meta_node_B
# - Creates meta_node_A → meta_node_B
```

### Inter-Level Edges
```python
# Edges between different hierarchical levels
g.add_edge(base_node_id, meta_node_id)     # Base node ↔ Meta-node
g.add_edge(meta_node_1, meta_node_2)       # Meta-node ↔ Meta-node

# Edge aggregation during collapse
community_1 = g.nodes([1, 2, 3])
community_2 = g.nodes([4, 5, 6])

# Count edges between communities
inter_edges = g.edges.between_subgraphs(community_1, community_2)
```

## 📊 **Enhanced Aggregation and Feature Engineering**

### Enhanced Missing Attribute Handling
```python
# Strict validation by default (errors on missing attributes)
try:
    meta_node = subgraph.collapse_to_node({
        "salary": "sum",
        "bonus": "mean"      # Error if 'bonus' doesn't exist on any nodes
    })
except MissingAttributeError as e:
    print(f"Attribute '{e.attribute}' not found")

# Advanced usage with defaults for power users
meta_node = subgraph.collapse_to_node(
    agg_functions={"salary": "sum", "age": "mean"},
    defaults={
        "bonus": 0,          # Default value if 'bonus' attribute missing
        "rating": 3.0,       # Default value if 'rating' attribute missing
        "department": "unknown"  # Default for categorical attributes
    }
)

# Default value behavior by aggregation type:
# sum/mean → 0
# count → 0 (but count always works regardless of attributes)
# max/min → None 
# first/last → None
# concat → "" (empty string)
```

### Auto-Optimized Table Views
```python
# Tables automatically slice out NaN-only columns
base_table = g.nodes.base.table()     # Excludes meta-node-only attributes
meta_table = g.nodes.meta.table()     # Excludes base-node-only attributes

# Manual control over column slicing
all_table = g.nodes.table(auto_slice=False)  # Include all columns
clean_table = g.nodes.table(auto_slice=True) # Auto-remove NaN columns

# Prefetch attribute schema for performance
g.nodes.base.prefetch_schema()        # Cache which attributes exist
optimized_table = g.nodes.base.table()  # Faster with prefetched schema
```

### Built-in Aggregation Functions
```python
# Standard aggregation operations
agg_functions = {
    # Numerical aggregations
    "node_count": "count",
    "total_weight": ("weight", "sum"),
    "avg_degree": ("degree", "mean"),
    "max_centrality": ("centrality", "max"),
    
    # Categorical aggregations
    "dominant_type": ("node_type", "mode"),
    "type_diversity": ("node_type", "nunique"),
    
    # Custom aggregations
    "custom_score": lambda sg: sg.custom_metric(),
    
    # Graph-level properties
    "density": lambda sg: sg.density(),
    "clustering": lambda sg: sg.clustering().mean(),
    "diameter": lambda sg: sg.diameter()
}
```

### Advanced Feature Engineering
```python
# Complex aggregations
meta_nodes = g.add_subgraphs(
    communities,
    agg_functions={
        # Statistical summaries
        "degree_stats": ("degree", ["mean", "std", "min", "max"]),
        
        # Network properties
        "structural_features": lambda sg: {
            "density": sg.density(),
            "transitivity": sg.transitivity(), 
            "assortativity": sg.assortativity(),
            "modularity": sg.modularity()
        },
        
        # Temporal aggregations (if time-stamped)
        "activity_pattern": ("timestamp", "histogram"),
        "peak_activity": ("timestamp", "mode"),
        
        # Spatial aggregations (if geo-located)
        "geographic_center": (["lat", "lon"], "centroid"),
        "geographic_spread": (["lat", "lon"], "std")
    }
)
```

## 🎯 **Use Cases and Applications**

### 1. Community Analysis
```python
# Detect communities and analyze at multiple levels
communities = g.communities.louvain()
communities.set("name", [f"community_{i}" for i in range(len(communities))]) # requires a plural set in array not yet implemented -assert that the len(array) == len(communities)
g.add_subgraphs(communities)

# Compare community properties
community_graph = g.nodes.subgraphs.to_graph()
community_centrality = community_graph.centrality.betweenness()
```

### 2. Hierarchical Clustering
```python
# Multi-level hierarchical decomposition
level_1 = g.communities.louvain()
g.add_subgraphs(level_1)

# Cluster the meta-graph
meta_graph = g.nodes.subgraphs.to_graph()
level_2 = meta_graph.communities.louvain()
meta_graph.add_subgraphs(level_2)

# Access full hierarchy
hierarchy = g.hierarchy.all_levels()
```

### 3. Graph Coarsening for Performance
```python
# Coarsen large graph for faster algorithms
large_graph = gr.load("massive_graph.gml")  # 100k nodes

# Coarsen by clustering similar nodes
clusters = large_graph.cluster.spectral(n_clusters=1000)
coarsened = large_graph.add_subgraphs(clusters)

# Run expensive algorithm on coarsened graph
result = coarsened.algorithm.expensive_centrality()

# Project results back to original graph
full_result = result.expand_to_original()
```

### 4. Multi-Scale Network Analysis
```python
# Social network: individuals → groups → organizations
individuals = g  # Original graph
groups = individuals.add_subgraphs(individuals.communities.louvain())
organizations = groups.add_subgraphs(groups.communities.leiden())

# Analyze at each scale
individual_metrics = individuals.centrality.betweenness()
group_influence = groups.centrality.eigenvector()
org_power = organizations.centrality.pagerank()
```

### 5. Biological Pathway Analysis
```python
# Protein interaction → pathways → biological processes
proteins = gr.load("protein_interactions.gml")
pathways = proteins.add_subgraphs(proteins.pathways)
processes = pathways.add_subgraphs(pathways.biological_processes)

# Pathway enrichment analysis
enriched_pathways = pathways.nodes.subgraphs.filter(
    lambda p: p.enrichment_score > threshold
)
```

## ⚠️ **Technical Challenges and Solutions**

### 1. ID Management and Conflicts
**Problem:** Node ID conflicts between hierarchical levels

**Solutions:**
```python
# Namespaced IDs
meta_node = subgraph.add_to_graph(
    id_strategy="namespace",      # prefix with level
    id_prefix="L1_community_"
)

# UUID generation
meta_node = subgraph.add_to_graph(
    id_strategy="uuid"           # Generate unique UUIDs
)

# Custom ID mapping
meta_node = subgraph.add_to_graph(
    id_mapping=custom_id_function
)
```

### 2. Memory Management and Cycles
**Problem:** Memory leaks from circular references, deep nesting

**Solutions:**
```python
# Reference counting with weak references
class SubgraphNode:
    def __init__(self, subgraph):
        self._subgraph_ref = weakref.ref(subgraph)  # Weak reference
        
# Depth limits
g.config.max_subgraph_depth = 5  # Prevent excessive nesting

# Lazy loading
meta_node.subgraph  # Only load when accessed
```

### 3. Consistency and Synchronization
**Problem:** Changes to original graph affecting subgraph-nodes

**Solutions:**
```python
# Immutable snapshots
meta_node = subgraph.add_to_graph(
    mode="snapshot"              # Immutable copy
)

# Live references with notifications
meta_node = subgraph.add_to_graph(
    mode="live",                 # Dynamic reference
    on_change="update_attributes" # Auto-update aggregated attrs
)

# Version tracking
meta_node.subgraph_version       # Track changes
meta_node.sync_if_outdated()     # Manual sync
```

### 4. Serialization and Persistence
**Problem:** Saving/loading hierarchical graphs

**Solutions:**
```python
# Hierarchical format
g.save("hierarchical_graph.h5", 
       format="hierarchical",
       include_subgraphs=True)

# Flatten for standard formats
g.save("flattened_graph.gml",
       format="gml", 
       flatten_hierarchy=True)

# Custom serialization
g.save("custom.json", 
       serializer=HierarchicalJSONSerializer())
```

## 🔍 **Error Handling and Edge Cases**

### 1. Circular Containment
```python
# Detect and prevent cycles
try:
    subgraph_A.add_subgraph(subgraph_B)
    subgraph_B.add_subgraph(subgraph_A)  # Error!
except CircularContainmentError as e:
    print(f"Circular reference detected: {e}")
```

### 2. Orphaned References
```python
# Handle deleted nodes in subgraphs
meta_node = subgraph.add_to_graph()
g.remove_node(1)  # Node 1 was in subgraph # subgraph automatically updates

# 2. Remove subgraph from graph
g.remove_node(meta_node)
# 3. Error on access
try:
    meta_node.subgraph.nodes
except InvalidSubgraphError:
    pass
```

### 3. Type Conflicts
```python
# Handle attribute type conflicts during aggregation
try:
    meta_nodes = g.add_subgraphs(
        communities,
        agg_functions={"mixed_attr": "mean"}  # Mixed str/int types
    )
except AggregationTypeError as e:
    # Fallback strategy
    meta_nodes = g.add_subgraphs(
        communities,
        agg_functions={"mixed_attr": "mode"}  # Use most common value
    )
```

## 🏗️ **Architectural Decisions and Implementation**

### Key Design Decisions

#### 1. Missing Attribute Handling Strategy
**Decision**: Error by default + defaults dict for advanced usage
- **Primary behavior**: `MissingAttributeError` when aggregating non-existent attributes
- **Advanced usage**: Optional `defaults` parameter for power users
- **Rationale**: Strict validation catches typos while remaining flexible

#### 2. Base vs Meta Entity Terminology  
**Decision**: Use "base" for all non-meta entities
- `g.nodes.base` → Regular/original nodes
- `g.nodes.meta` → Meta-nodes  
- `g.edges.base` → Regular edges
- `g.edges.meta` → Meta-edges
- **Rationale**: "base" is clear, semantic, and avoids confusion with "regular"

#### 3. Filtered Accessors vs Subgraphs
**Decision**: Dual-purpose accessor pattern
- **Accessor**: `g.nodes.meta` (no parentheses) → Filtered view for data access
- **Subgraph**: `g.nodes.meta()` (with parentheses) → Actionable subgraph
- **Rationale**: Clean separation between data access and graph operations

#### 4. Auto-Sliced Table Views
**Decision**: Remove NaN-only columns by default in filtered tables
- `g.nodes.base.table()` → Excludes meta-only attributes automatically
- `g.nodes.meta.table()` → Excludes base-only attributes automatically
- `auto_slice=False` → Override for full column access
- **Rationale**: Cleaner data views, better performance, more intuitive

#### 5. Meta-Edge Creation Strategy
**Decision**: Heterogeneous meta-nodes with optional edge aggregation
- **Default**: Copy all edges from collapsed nodes to meta-edges
- **Optional**: Edge aggregation (sum, count, mean) for parallel edges
- **Two types**: Child-to-external (always) + meta-to-meta (conditional)
- **Rationale**: Maximum flexibility while maintaining graph connectivity

#### 6. Worf's Airtight Entity Type System 🛡️
**Decision**: Balance simplicity and safety using columnar storage with type markers
- **Core Approach**: Use `entity_type` attribute to distinguish base vs meta entities
- **Safety Guarantees**: Type-safe creation, immutable types, atomic operations, validation
- **Performance**: Leverage existing efficient columnar storage and attribute indexing
- **Migration**: Safe upgrade path for existing nodes

### Worf's Safety Architecture

#### Type-Safe Entity Creation
```rust
impl Graph {
    /// SAFE: Create base node (default behavior)
    pub fn add_node(&mut self) -> NodeId {
        let node_id = self.pool.allocate_node_id();
        self.pool.set_attr(node_id, "entity_type", AttrValue::Text("base")).unwrap();
        self.space.mark_node_active(node_id);
        node_id
    }

    /// SAFE: Create meta-node atomically with required attributes
    pub fn create_meta_node(&mut self, subgraph_id: SubgraphId) -> GraphResult<NodeId> {
        let node_id = self.pool.allocate_node_id();
        
        // ATOMIC OPERATION: Either all succeed or all fail
        let mut transaction = self.begin_transaction();
        transaction.set_attr(node_id, "entity_type", AttrValue::Text("meta"))?;
        transaction.set_attr(node_id, "contained_subgraph", AttrValue::SubgraphRef(subgraph_id))?;
        transaction.validate_meta_node_requirements(node_id)?;
        transaction.commit()?;
        
        self.space.mark_node_active(node_id);
        Ok(node_id)
    }

    /// FORBIDDEN: Direct entity_type modification
    pub fn set_node_attr(&mut self, node_id: NodeId, attr_name: AttrName, value: AttrValue) -> GraphResult<()> {
        if attr_name == "entity_type" {
            return Err(GraphError::InvalidInput(
                "entity_type is immutable. Use create_meta_node() or add_node()".to_string()
            ));
        }
        // ... rest of implementation
    }
}
```

#### Safety Guarantees
1. **✅ Type Safety**: No invalid entity types possible through API design
2. **✅ Immutability**: Entity types cannot be changed after creation
3. **✅ Atomicity**: All meta-node creation is transactional (all-or-nothing)
4. **✅ Validation**: All entities validated on creation and query operations
5. **✅ Migration**: Safe upgrade path for existing nodes without entity_type
6. **✅ Performance**: Validation cached, type queries use efficient columnar storage

#### Entity Validation System
```rust
pub struct EntityValidator {
    /// REQUIREMENT: Meta-nodes MUST have entity_type = "meta"
    /// REQUIREMENT: Meta-nodes MUST have valid contained_subgraph reference
    /// REQUIREMENT: Base nodes MUST NOT have contained_subgraph (unless migrated)
    /// PERFORMANCE: Validation results cached for repeated queries
}
```

#### Safe Query Interface
```rust
// Type-checked queries using efficient columnar storage
impl Graph {
    pub fn is_meta_node(&self, node_id: NodeId) -> bool
    pub fn is_base_node(&self, node_id: NodeId) -> bool
    pub fn get_meta_nodes(&self) -> Vec<NodeId>          // Only validated meta-nodes
    pub fn get_base_nodes(&self) -> Vec<NodeId>          // Only validated base nodes
    pub fn migrate_entity_types(&mut self) -> GraphResult<()> // Safe migration
}
```

### Implementation Architecture

### Core Data Structures
```rust
/// needs implementation
/// the meta node is a node that contains a ref to the subgraph
/// its just a node with the subgraph as a subgraph attribute, stored in the graph pool

/// questionable structures
pub struct SubgraphNode { /// inherits from node
    pub id: NodeId,
    pub subgraph_ref: SubgraphRef,
    pub aggregated_attrs: HashMap<String, AttrValue>,
    pub metadata: SubgraphMetadata,
}

pub enum SubgraphRef {
    pub subgraph: Subgraph,
}
```

### Python FFI Interface
```python
# Python interface classes
class SubgraphNode(Node):
    @property
    def subgraph(self) -> Subgraph: ...
    @property 
    def is_subgraph_node(self) -> bool: ...

class Subgraph(Graph):
    def add_to_graph(self, **kwargs) -> SubgraphNode: ...
    def contains_nodes(self, node_ids: List[int]) -> bool: ...
    def contains_edges(self, edge_ids: List[int]) -> bool: ...
    def aggregate(self, functions: Dict) -> Dict: ...

class SubgraphArray(PyArray):
    def add_to_graph(self, **kwargs) -> List[SubgraphNode]: ...
    def aggregate(self, functions: Dict) -> PyArray: ...
```

## 🚀 **Development Phases**

### Phase SUB-1: Core Infrastructure (2-3 weeks)
- [ ] Basic subgraph-to-node conversion
- [ ] Simple containment checking
- [ ] Basic aggregation functions
- [ ] ID management system

**Deliverables:**
- `subgraph.add_to_graph()` basic functionality
- `node.subgraph` property access
- Simple aggregation (count, sum, mean)

### Phase SUB-2: Advanced Features (3-4 weeks)
- [ ] Subgraph arrays and batch operations
- [ ] Complex aggregation functions
- [ ] Edge handling strategies
- [ ] Hierarchy navigation

**Deliverables:**
- `g.add_subgraphs()` batch processing
- Advanced aggregation functions
- `g.nodes.subgraphs` collection access
- Edge transformation options

### Phase SUB-3: Production Features (2-3 weeks)
- [ ] Serialization and persistence
- [ ] Performance optimization
- [ ] Error handling and validation
- [ ] Memory management

**Deliverables:**
- Hierarchical file format support
- Large-scale performance optimization
- Comprehensive error handling
- Memory leak prevention

## 🎪 **Extended Possibilities**

### 1. Dynamic Hierarchies
- **Auto-clustering:** Automatically create subgraphs based on metrics
- **Adaptive granularity:** Change hierarchy detail based on analysis needs
- **Temporal hierarchies:** Time-based subgraph evolution

### 2. Cross-Graph Hierarchies
- **Multi-graph systems:** Subgraphs spanning multiple base graphs
- **Graph-of-graphs:** Networks where nodes are entire graphs
- **Federated hierarchies:** Distributed hierarchical graph systems

### 3. Machine Learning Integration
- **Hierarchical embeddings:** Node embeddings that respect hierarchy
- **Multi-scale learning:** Train models at different hierarchy levels
- **Hierarchy-aware algorithms:** Algorithms that leverage structure

### 4. Interactive Hierarchies
- **Visual drill-down:** Interactive exploration of hierarchy levels
- **Dynamic aggregation:** Real-time reaggregation during analysis
- **Collaborative hierarchies:** Multi-user hierarchy construction

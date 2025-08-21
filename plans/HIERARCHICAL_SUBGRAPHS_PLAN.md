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

## 🔗 **Edge Handling in Hierarchical Graphs**

### Inter-Level Edges
```python
# Edges between different hierarchical levels
g.add_edge(regular_node_id, meta_node_id)     # Regular node ↔ Meta-node
g.add_edge(meta_node_1, meta_node_2)          # Meta-node ↔ Meta-node

# Edge aggregation during collapse
community_1 = g.nodes([1, 2, 3])
community_2 = g.nodes([4, 5, 6])

# Count edges between communities
inter_edges = g.edges.between_subgraphs(community_1, community_2)
```

### Edge Transformation Strategies
```python
# Different strategies for handling internal edges
meta_node = subgraph.add_to_graph(
    edge_strategy="aggregate",     # Sum weights of internal edges
    # edge_strategy="preserve",    # Keep internal edges as self-loops
    # edge_strategy="discard",     # Remove internal edges
    # edge_strategy="external_only" # Only preserve edges to outside nodes
)
```

## 📊 **Aggregation and Feature Engineering**

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

## 🏗️ **Implementation Architecture**

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

# Trait-Based Entity Architecture Design

**Date:** 2025-01-09  
**Status:** Approved for Implementation  
**Goal:** Reorganize graph entities using proper trait-based polymorphism for type safety and extensibility

## 🎯 **Problem Statement**

Current issues with the hierarchical entity system:
1. **Type Confusion:** PyNodeView trying to do both regular and meta-node operations (unsafe)
2. **Poor Organization:** Meta-node code scattered across subgraph modules  
3. **Limited Extensibility:** Hard to add new entity types (PathNode, ComponentNode, etc.)
4. **API Inconsistency:** Different return types and access patterns for similar entities

## 🏗️ **Proposed Solution: Trait-Based Entity Architecture**

### **Core Principles:**
1. **Type Safety:** Each entity type returns the correct Python object with appropriate methods
2. **Trait Inheritance:** Meta-entities inherit all regular entity capabilities  
3. **Clean Separation:** Each entity type has its own implementation file
4. **Extensible Design:** Easy to add new specialized entity types
5. **Memory Efficiency:** Each entity carries only the data it needs

## 📁 **New Folder Structure**

### **Rust Core (`src/`):**
```
src/
├── entities/                    # NEW: Core entity implementations  
│   ├── mod.rs                  # Entity module exports
│   ├── node.rs                 # Regular Node struct + NodeOperations impl
│   ├── edge.rs                 # Regular Edge struct + EdgeOperations impl
│   ├── meta_node.rs            # MetaNode struct + NodeOperations + MetaNodeOperations
│   └── meta_edge.rs            # MetaEdge struct + EdgeOperations + MetaEdgeOperations
├── traits/                     # EXISTING: Enhanced trait definitions
│   ├── graph_entity.rs         # ✅ Base GraphEntity trait (keep as-is)
│   ├── node_operations.rs      # ✅ NodeOperations trait (keep as-is)
│   ├── edge_operations.rs      # ✅ EdgeOperations trait (keep as-is)  
│   ├── meta_operations.rs      # NEW: MetaNodeOperations + MetaEdgeOperations
│   └── mod.rs                  # Updated exports
├── storage/                    # EXISTING: Keep low-level storage
│   ├── node.rs                 # Low-level node storage (no changes)
│   └── edge.rs                 # Low-level edge storage (no changes)
```

### **Python FFI (`python-groggy/src/ffi/`):**
```
python-groggy/src/ffi/
├── entities/                   # NEW: Python wrappers for entities
│   ├── mod.rs                  # FFI entity exports
│   ├── node.rs                 # PyNode (wraps core::entities::Node)  
│   ├── edge.rs                 # PyEdge (wraps core::entities::Edge)
│   ├── meta_node.rs            # PyMetaNode (wraps core::entities::MetaNode)
│   └── meta_edge.rs            # PyMetaEdge (wraps core::entities::MetaEdge)
├── storage/                    # EXISTING: Keep accessor system
│   ├── views.rs                # Keep PyNodeView/PyEdgeView for basic views
│   └── accessors.rs            # Enhanced to return correct entity types
├── subgraphs/                  # EXISTING: Clean up
│   ├── hierarchical.rs         # Remove PyMetaNode (moved to entities/)
│   └── ...                     # Keep other subgraph types
```

## 🔧 **Trait Hierarchy Design**

### **Base Traits (Already Excellent):**
```rust
/// Universal entity interface - already well designed
trait GraphEntity {
    fn entity_id(&self) -> EntityId;
    fn entity_type(&self) -> &'static str;
    fn graph_ref(&self) -> Rc<RefCell<Graph>>;
    fn get_attribute(&self, name: &AttrName) -> GraphResult<Option<AttrValue>>;
    fn set_attribute(&self, name: AttrName, value: AttrValue) -> GraphResult<()>;
    // ... other excellent existing methods
}

/// Node operations - already well designed  
trait NodeOperations: GraphEntity {
    fn node_id(&self) -> NodeId;
    fn degree(&self) -> GraphResult<usize>;
    fn neighbors(&self) -> GraphResult<Vec<NodeId>>;
    // ... keep existing excellent methods
}

/// Edge operations - already well designed
trait EdgeOperations: GraphEntity {
    fn edge_id(&self) -> EdgeId;
    fn endpoints(&self) -> GraphResult<(NodeId, NodeId)>;  
    // ... keep existing excellent methods
}
```

### **New Meta Traits:**
```rust
/// Meta-node specific operations (NEW)
trait MetaNodeOperations: NodeOperations {
    /// Check if this meta-node contains a subgraph
    fn has_subgraph(&self) -> bool;
    
    /// Get the ID of the contained subgraph
    fn subgraph_id(&self) -> Option<usize>;
    
    /// Get the contained subgraph  
    fn subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>>;
    
    /// Expand meta-node back to its original subgraph
    fn expand(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>>;
    
    /// Get all meta-edges connected to this meta-node
    fn meta_edges(&self) -> GraphResult<Vec<EdgeId>>;
    
    /// Re-aggregate meta-node attributes with new functions
    fn re_aggregate(&self, agg_functions: HashMap<AttrName, String>) -> GraphResult<()>;
}

/// Meta-edge specific operations (NEW)
trait MetaEdgeOperations: EdgeOperations {  
    /// Check if this is a meta-edge
    fn is_meta_edge(&self) -> bool;
    
    /// Get the count of original edges this meta-edge aggregates
    fn edge_count(&self) -> Option<i64>;
    
    /// Get the IDs of original edges that were aggregated (future enhancement)
    fn aggregated_from(&self) -> GraphResult<Option<Vec<EdgeId>>>;
    
    /// Expand meta-edge back to original edges (future enhancement)
    fn expand(&self) -> GraphResult<Option<Vec<EdgeId>>>;
    
    /// Get meta-edge specific properties as a summary
    fn meta_properties(&self) -> GraphResult<HashMap<String, AttrValue>>;
}
```

## 🏛️ **Implementation Architecture**

### **1. Concrete Entity Structs:**
```rust
// src/entities/node.rs
pub struct Node {
    id: NodeId,
    graph: Rc<RefCell<Graph>>,
}

impl GraphEntity for Node { /* delegate to graph storage */ }
impl NodeOperations for Node { /* delegate to existing algorithms */ }

// src/entities/meta_node.rs
pub struct MetaNode {
    id: NodeId,
    graph: Rc<RefCell<Graph>>,
    // No extra fields - all data in graph storage
}

impl GraphEntity for MetaNode { /* delegate to graph storage */ }
impl NodeOperations for MetaNode { /* inherit all node capabilities */ }
impl MetaNodeOperations for MetaNode { /* meta-specific methods */ }
```

### **2. Smart Accessor Integration:**
```rust
// In PyNodesAccessor::__getitem__
impl PyNodesAccessor {
    fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> {
        match key {
            single_id => {
                let graph = self.graph.borrow();
                
                // Smart type detection based on actual entity properties
                if graph.is_meta_node(single_id) {
                    let meta_node = MetaNode::new(single_id, self.graph.clone())?;
                    Ok(PyMetaNode::from_meta_node(meta_node).into_py(py))
                } else {
                    let node = Node::new(single_id, self.graph.clone())?;  
                    Ok(PyNode::from_node(node).into_py(py))
                }
            }
            array_or_slice => {
                // Return appropriate subgraph type
                /* existing subgraph logic */
            }
        }
    }
}
```

### **3. Python API Design:**
```python
# Type-safe API - users get exactly the right object

# Regular node - only has regular node methods
node = g.nodes[0]              # Returns PyNode
print(node.degree)             # ✅ NodeOperations  
print(node.neighbors)          # ✅ NodeOperations
print(node['name'])            # ✅ GraphEntity attribute access
# node.has_subgraph            # ❌ AttributeError - method doesn't exist

# Meta-node - has regular node methods PLUS meta methods  
meta_node = g.nodes[meta_id]   # Returns PyMetaNode  
print(meta_node.degree)       # ✅ NodeOperations (inherited)
print(meta_node.neighbors)    # ✅ NodeOperations (inherited) 
print(meta_node['size'])      # ✅ GraphEntity attribute access
print(meta_node.has_subgraph) # ✅ MetaNodeOperations
subgraph = meta_node.expand()  # ✅ MetaNodeOperations

# Same pattern for edges
edge = g.edges[0]              # Returns PyEdge
meta_edge = g.edges[meta_id]   # Returns PyMetaEdge
print(meta_edge.edge_count)    # ✅ MetaEdgeOperations
```

## 📈 **Benefits & Advantages**

### **1. Type Safety:**
- Regular nodes cannot call meta-node methods (compile-time safety)
- Users get exactly the capabilities their entity actually has
- No runtime errors from calling unsupported methods

### **2. Clean Architecture:**
- Each entity type has a dedicated implementation file
- Clear separation of concerns
- Easy to understand and maintain

### **3. Extensibility:**
- Adding new entity types (PathNode, ComponentNode, etc.) is straightforward
- Follow the same trait pattern for consistency
- Existing code continues to work unchanged

### **4. Performance:**
- Each entity carries only the data it needs
- All operations delegate to existing optimized algorithms
- No overhead from unused functionality

### **5. API Clarity:**
- Users know exactly what methods are available on each type
- Consistent patterns across all entity types
- Better IDE support and documentation

## 🎯 **Migration Strategy**

### **Phase 1: Core Infrastructure**
1. Create `src/entities/` folder and basic entity structs
2. Create `MetaNodeOperations` and `MetaEdgeOperations` traits
3. Implement traits for all entity types
4. Update module exports

### **Phase 2: Python FFI**  
1. Create `python-groggy/src/ffi/entities/` folder
2. Move and enhance PyMetaNode to new location
3. Create PyNode, PyEdge, PyMetaEdge wrappers
4. Update accessor logic for smart type detection

### **Phase 3: API Integration**
1. Update collapse methods to return proper entity types
2. Test type detection and method availability
3. Update examples and documentation
4. Verify backward compatibility

### **Phase 4: Cleanup**
1. Remove old scattered meta-node code
2. Clean up unused imports and methods
3. Run comprehensive tests
4. Update API documentation

## 🧪 **Validation Approach**

### **Type Safety Verification:**
```python
# These should work
meta_node = g.nodes[meta_id]
assert hasattr(meta_node, 'has_subgraph')  # ✅
assert hasattr(meta_node, 'degree')        # ✅ inherited

# These should fail at the Python level
node = g.nodes[regular_id]  
assert not hasattr(node, 'has_subgraph')   # ✅ type safety
```

### **Functionality Verification:**
```python
# All existing functionality should continue working
subgraph = g.nodes[[0, 1, 2]]
meta_id = subgraph.collapse(node_aggs={"size": "count"})

# New type-safe access
meta_node = g.nodes[meta_id]               # Returns PyMetaNode
assert meta_node.has_subgraph == True      # ✅ Meta-specific method
assert len(meta_node.neighbors) >= 0      # ✅ Inherited node method
assert meta_node.subgraph is not None     # ✅ Meta-specific functionality
```

## 🚀 **Expected Outcomes**

1. **Safer API:** Users cannot accidentally call meta methods on regular nodes
2. **Cleaner Code:** Well-organized entity implementations with clear responsibilities  
3. **Better Extensibility:** Easy to add new entity types following the established pattern
4. **Improved Performance:** Each entity type carries only necessary data and methods
5. **Enhanced User Experience:** Clear, predictable API with excellent IDE support

## 📝 **Implementation Notes**

- All existing storage and algorithms remain unchanged
- Entity structs are lightweight wrappers around IDs + graph references
- All operations delegate to existing optimized implementations  
- Backward compatibility maintained through careful API design
- Migration can be done incrementally without breaking existing code

---

**Next Steps:** Execute this architecture by implementing Phase 1 (Core Infrastructure).
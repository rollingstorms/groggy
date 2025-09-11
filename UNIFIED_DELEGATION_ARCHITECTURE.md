# Unified Delegation Architecture for Groggy

## Vision: Complete Object Interoperability

Transform Groggy into a unified graph ecosystem where every major object type can seamlessly convert to and operate on collections of other types through delegation, eliminating code duplication and maximizing composability.

## Core Philosophy

- **Zero Algorithm Duplication**: All algorithms stay in their optimized core types
- **Maximum Composability**: Any object should be convertible to any other when semantically meaningful
- **Unified Arrays**: Each major type gets a specialized array class with shared delegation traits
- **Graph-Like Connectivity**: Objects connect to each other through natural method chains

## Main Object Types

### 1. Core Objects (Algorithm Holders)
- **Subgraph** - Graph portions with node/edge sets
- **Nodes** (NodesAccessor) - Node collections with attributes  
- **Edges** (EdgesAccessor) - Edge collections with attributes
- **Array** (BaseArray) - Generic collections
- **Table** - Structured data with columns
- **Matrix** - Mathematical graph representations

### 2. Specialized Array Types (Delegation Carriers)
- **SubgraphArray** - Collections of subgraphs
- **NodesArray** - Collections of node sets
- **EdgesArray** - Collections of edge sets  
- **TableArray** - Collections of tables
- **MatrixArray** - Collections of matrices
- **BaseArray** - Generic array operations

## Architecture Components

### Base Delegation Trait
```rust
pub trait ArrayOps<T> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&T>;
    fn iter(&self) -> ArrayIterator<T>;
    
    // Delegation methods that forward to element algorithms
    fn apply_method<R>(&self, method: impl Fn(&T) -> R) -> Vec<R>;
}
```

### Direct Delegation Pattern
```rust
pub trait ArrayOps<T> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&T>;
    
    // Direct apply_on_each delegation methods
    fn table(&self) -> TableArray where T: HasTable;
    fn nodes(&self) -> NodesArray where T: HasNodes;
    fn edges(&self) -> EdgesArray where T: HasEdges;
    fn subgraphs(&self) -> SubgraphArray where T: HasSubgraphs;
    fn filter(&self, predicate: impl Fn(&T) -> bool) -> Self;
    fn sample(&self, k: usize) -> Self;
}
```

## Object Connectivity Graph

```
Graph
├── nodes() → NodesAccessor
│   ├── connected_components() → SubgraphArray
│   ├── filter() → NodesArray  
│   └── table() → Table
├── edges() → EdgesAccessor
│   ├── connected_components() → SubgraphArray
│   ├── filter() → EdgesArray
│   └── table() → Table
├── connected_components() → SubgraphArray
├── table() → Table
└── matrix() → Matrix

SubgraphArray
├── table() → TableArray                    # Direct delegation
├── nodes() → NodesArray                    # Direct delegation  
├── edges() → EdgesArray                    # Direct delegation
├── sample(k) → SubgraphArray               # Direct delegation
├── filter(predicate) → SubgraphArray       # Direct delegation
└── largest() → Subgraph                    # Single element access

TableArray  
├── filter(query) → TableArray              # Direct delegation
├── group_by(cols) → TableArray             # Direct delegation
├── join(other) → TableArray                # Direct delegation
├── agg(spec) → Table                       # Reduce to single table
└── merge() → Table                         # Reduce to single table

NodesArray
├── connected_components() → SubgraphArray  # Direct delegation
├── table() → TableArray                    # Direct delegation
├── filter(query) → NodesArray              # Direct delegation
└── union() → NodesAccessor                 # Reduce to single accessor

EdgesArray
├── connected_components() → SubgraphArray  # Direct delegation
├── table() → TableArray                    # Direct delegation
├── nodes() → NodesArray                    # Direct delegation
├── filter(query) → EdgesArray              # Direct delegation
└── union() → EdgesAccessor                 # Reduce to single accessor

MatrixArray
├── eigen() → BaseArray                     # Direct delegation
├── multiply(other) → MatrixArray           # Direct delegation
├── transform(op) → MatrixArray             # Direct delegation
└── stack() → Matrix                        # Reduce to single matrix
```

## Implementation Phases

### Phase 1: Unify Existing Arrays ✅
- [x] Consolidate ComponentsArray into SubgraphArray
- [x] Create TableArray with delegation
- [x] Implement universal ArrayIterator pattern
- [x] Add collect() methods to all arrays

### Phase 2: Create Missing Specialized Arrays
- [ ] **NodesArray** - Collections of NodesAccessor objects
- [ ] **EdgesArray** - Collections of EdgesAccessor objects  
- [ ] **MatrixArray** - Collections of Matrix objects
- [ ] Update all existing arrays to use unified delegation traits

### Phase 3: Implement Cross-Type Conversions
- [ ] Add conversion methods to each core object:
  - Subgraph → nodes(), edges(), table(), matrix()
  - NodesAccessor → subgraphs(), table(), connected_components()  
  - EdgesAccessor → subgraphs(), table(), connected_components()
  - Table → nodes(), edges(), subgraphs() (where applicable)
  - Matrix → subgraphs(), table()

### Phase 4: Universal Delegation Methods
- [ ] Implement HasTable, HasNodes, HasEdges, HasSubgraphs traits
- [ ] Add delegation methods to ArrayIterator for each trait
- [ ] Enable cross-type method chaining everywhere

### Phase 5: Optimization & Integration
- [ ] Lazy materialization for all array types
- [ ] Zero-copy sharing where possible
- [ ] Performance benchmarking and optimization
- [ ] Complete FFI integration

## Example Usage Patterns

### Current (Limited)
```python
components = g.connected_components()  # ComponentsArray
table_array = components.iter().table().collect()  # TableArray (complex)
```

### Target (Unified & Simplified)
```python
# Direct delegation - no iter().collect()
tables = g.nodes().connected_components().table()  # TableArray

# Cross-type chaining with direct methods
high_degree = g.edges().filter("weight > 0.5").nodes().filter("degree > 10")
communities = high_degree.connected_components().sample(5)

# Array-to-array conversions (direct)
edge_tables = g.connected_components().edges().table()

# Complex analysis chains (streamlined)
results = (g.nodes()
           .connected_components()          # SubgraphArray
           .filter(lambda sg: len(sg) > 100)  # SubgraphArray (direct filter)
           .table()                         # TableArray (direct delegation)
           .group_by(['component_id'])      # TableArray (direct method)
           .agg({'node_count': 'sum'}))     # Table (reduce operation)
```

## Benefits

1. **Infinite Composability**: Every object connects to every other through natural methods
2. **Zero Duplication**: Algorithms stay in optimized core, arrays just delegate
3. **Type Safety**: Compile-time guarantees about method availability  
4. **Performance**: Lazy materialization and zero-copy where possible
5. **Discoverability**: Natural method chaining guides users to functionality
6. **Extensibility**: New types automatically get delegation capabilities

## Technical Considerations

### Memory Management
- Use `Arc<Vec<T>>` for zero-copy sharing between arrays
- Implement lazy materialization for expensive operations
- Cache results where appropriate

### FFI Integration  
- Each specialized array gets Python bindings with `@pyclass`
- Universal iterator methods available in Python
- Proper error handling and type conversion

### Performance
- Benchmark delegation overhead vs direct calls
- Optimize hot paths in ArrayIterator
- Consider parallel processing for large arrays

## Migration Strategy

1. **Backward Compatibility**: Keep existing APIs working
2. **Gradual Migration**: Add new arrays alongside existing ones
3. **Deprecation Path**: Slowly phase out specialized classes like ComponentsArray
4. **Documentation**: Update examples to show new patterns
5. **Testing**: Comprehensive test suite for all conversion paths

---

*This architecture transforms Groggy from a collection of specialized types into a unified graph ecosystem where every object is just a few method calls away from any other.*
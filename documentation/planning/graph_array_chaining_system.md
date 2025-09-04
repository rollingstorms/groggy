# GraphArray Universal Chaining System - Iterative Implementation Plan

## Overview

Implement a unified `GraphArray` system that enables fluent chaining operations (`.iter()`) on any collection of graph-related objects. This replaces our current scattered array implementations with a cohesive, trait-based architecture.

## Current State Analysis

**Existing Array Types:**
- `PyComponentsArray` - lazy connected components collection  
- `PyGraphArray` - statistical array for `AttrValue` collections
- `GraphArrayIterator` - basic Python iterator
- **Missing:** unified chaining, universal base, consistent API

**Problem:** Inconsistent collection APIs, no chaining, manual iteration required

```python
# Current: Manual iteration required
results = []
for component in g.connected_components():
    filtered = component.filter_nodes('age > 25')
    collapsed = filtered.collapse({'avg_age': ('mean', 'age')})
    results.append(collapsed)
```

**Goal:** Unified chaining API across all collection types:

```python  
# Target: Fluent chaining on any collection
results = g.connected_components().iter().filter_nodes('age > 25').collapse({'avg_age': ('mean', 'age')})
filtered_nodes = node_ids.iter().filter_by_degree(min_degree=3).collect()
expanded_subgraphs = meta_nodes.iter().expand().filter_edges('weight > 0.5').collect()
```

## Core Architecture: Trait-Based Method Injection

### Key Innovation: Automatic Method Availability

Instead of manually implementing methods for each type, use **trait-based method injection** where methods automatically become available based on what traits a type implements:

```rust
// Universal iterator - generic over any type
pub struct GraphArrayIterator<T> {
    elements: Vec<T>,
    graph_ref: Option<Rc<RefCell<Graph>>>,
}

// Methods available to ALL types
impl<T> GraphArrayIterator<T> {
    pub fn filter<F>(self, predicate: F) -> Self where F: Fn(&T) -> bool;
    pub fn map<U, F>(self, func: F) -> GraphArrayIterator<U> where F: Fn(T) -> U;
    pub fn collect(self) -> Box<dyn GraphArray<T>>;
}

// Methods automatically available ONLY when T implements certain traits
impl<T: SubgraphLike> GraphArrayIterator<T> {
    pub fn filter_nodes(self, query: &str) -> Self;           // Only for subgraph-like types
    pub fn filter_edges(self, query: &str) -> Self;
    pub fn collapse(self, aggs: PyDict) -> GraphArrayIterator<PyMetaNode>;
}

impl<T: MetaNodeLike> GraphArrayIterator<T> {
    pub fn expand(self) -> GraphArrayIterator<PySubgraph>;    // Only for meta-node types
}

impl<T: NodeIdLike> GraphArrayIterator<T> {
    pub fn filter_by_degree(self, min_degree: usize) -> Self; // Only for node ID types
    pub fn get_neighbors(self) -> GraphArrayIterator<Vec<NodeId>>;
}
```

### Base GraphArray Trait
```rust
pub trait GraphArray<T> {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&T>; 
    fn iter_chain(&self) -> GraphArrayIterator<T>;           // Key method: enables .iter()
}
```

### Type-Safe Method Injection
```python
# Automatically gets the right methods based on type!
components.iter().filter_nodes('age > 25')  # ✅ Available (T: SubgraphLike)
node_ids.iter().filter_by_degree(3)         # ✅ Available (T: NodeIdLike)  
components.iter().expand()                   # ❌ Compile error! (Wrong trait)
```

**Benefits:**
- ✅ **Zero manual implementation** per type - traits provide methods automatically
- ✅ **Type safety** - wrong methods don't exist at compile time  
- ✅ **Extensible** - new types just implement marker traits
- ✅ **Maintainable** - add trait method once, all types get it

### Collection Type Implementations

#### 1. `PyComponentsArray` (Existing)
```python
components = g.connected_components()  # Returns GraphArray<PySubgraph>
results = components.iter().filter_nodes('age > 25').collapse()  # Returns GraphArray<PyMetaNode>
```

#### 2. `PyNodeArray` (New)
```python
node_ids = g.nodes.ids()  # Returns GraphArray<NodeId>  
filtered = node_ids.iter().filter_by_degree(min_degree=3).collect()  # Returns GraphArray<NodeId>
subgraphs = node_ids.iter().to_subgraph().collect()  # Returns GraphArray<PySubgraph>
```

#### 3. `PyMetaNodeArray` (New)
```python
meta_nodes = some_meta_nodes  # Returns GraphArray<PyMetaNode>
expanded = meta_nodes.iter().expand().filter_edges('weight > 0.5').collect()  # Returns GraphArray<PySubgraph>
```

#### 4. Generic `PyGraphArray<T>` (New)
```python
# Works with any collection
custom_array = PyGraphArray([obj1, obj2, obj3])
results = custom_array.iter().map(some_function).filter(some_predicate).collect()
```

## Iterative Implementation Plan

### Architecture Decision: Core Replacement + Trait-Based Extensions

This is a **significant core architecture change** that replaces our scattered array implementations with a unified, trait-based system. We'll implement iteratively to manage risk and maintain functionality.

### Phase 1: Foundation (Start Here) 
**Goal:** Core infrastructure without breaking existing code

1. **Create trait system in new module**
   ```rust
   // src/array_system/mod.rs - New module
   pub trait GraphArray<T> { ... }
   pub struct GraphArrayIterator<T> { ... }
   
   // Marker traits for method injection
   pub trait SubgraphLike { ... }
   pub trait NodeIdLike { ... }
   pub trait MetaNodeLike { ... }
   ```

2. **Implement base iterator with universal methods**
   - `filter()`, `map()`, `collect()` for all types
   - No type-specific methods yet

3. **Add trait-based method injection framework**
   - `impl<T: SubgraphLike> GraphArrayIterator<T>` structure  
   - Framework ready for specific methods

**Deliverable:** Core trait system compiled and tested, no API changes

### Phase 2: First Implementation - Components Array
**Goal:** Working `.iter()` on `PyComponentsArray`

1. **Implement `GraphArray<PySubgraph>` for `PyComponentsArray`**
   - Add `.iter()` method that returns `GraphArrayIterator<PySubgraph>`
   - Maintain all existing functionality

2. **Add subgraph-specific operations**
   ```rust
   impl<T: SubgraphLike> GraphArrayIterator<T> {
       pub fn filter_nodes(self, query: &str) -> Self;
       pub fn collapse(self, aggs: PyDict) -> GraphArrayIterator<PyMetaNode>;
   }
   ```

3. **Python FFI integration**
   - Export new `.iter()` method to Python
   - Ensure chaining works end-to-end

**Deliverable:** `g.connected_components().iter().filter_nodes(...).collapse(...)` working

### Phase 3: Expand to Other Collection Types
**Goal:** Unified system across main collection types

1. **Migrate existing `PyGraphArray` to new system** 
   - Keep statistical methods, add chaining capability
   - Implement `GraphArray<AttrValue>` trait

2. **Create missing collection types**
   - `PyNodeArray` for node ID collections
   - `PyMetaNodeArray` for meta-node collections

3. **Add remaining method injection traits**
   - Node ID operations: `filter_by_degree()`, `get_neighbors()`
   - Meta-node operations: `expand()`

**Deliverable:** All major collection types support `.iter()` chaining

### Phase 4: Integration & Optimization
**Goal:** Complete system integration and performance optimization

1. **Update all APIs to return unified array types**
   - Graph query methods return appropriate `GraphArray<T>`
   - Consistent collection interfaces throughout

2. **Performance optimization**
   - Lazy evaluation optimizations
   - Memory usage improvements
   - Benchmark against old system

3. **Migration cleanup**
   - Remove duplicate/obsolete array implementations
   - Update documentation and examples

**Deliverable:** Complete, optimized, unified GraphArray system

### Risk Management Strategy

**Iterative Benefits:**
- ✅ **Incremental development** - test each phase thoroughly
- ✅ **Backward compatibility** - old APIs work during transition
- ✅ **Risk isolation** - problems contained to current phase
- ✅ **Continuous validation** - real usage feedback at each step

**Rollback Strategy:**
- Each phase maintains existing functionality
- Can pause/rollback at any phase boundary
- Feature flags to enable/disable new system during development

### Success Metrics Per Phase

**Phase 1:** Core traits compile, tests pass, no regressions
**Phase 2:** Components chaining works, performance comparable
**Phase 3:** All collection types unified, comprehensive test coverage
**Phase 4:** Complete system integration, performance improvements documented

## API Examples

### Basic Chaining
```python
# Components → filter → collapse
results = g.connected_components().iter().filter_nodes('age > 25').collapse({
    'avg_age': ('mean', 'age'),
    'total_people': 'count'
})

# Node IDs → filter by degree → create subgraphs  
high_degree_subgraphs = g.nodes.ids().iter().filter_by_degree(min_degree=5).to_subgraph().collect()
```

### Complex Chaining
```python
# Multi-step processing pipeline
processed = (g.connected_components()
    .iter()
    .filter_nodes('department == "engineering"')
    .filter(lambda sg: sg.node_count() > 10)  # Keep large components
    .collapse({'team_size': 'count', 'avg_salary': ('mean', 'salary')})
    .collect())

# Meta-node expansion and re-processing
expanded_and_filtered = (meta_nodes
    .iter()
    .expand()
    .filter_edges('weight > 0.8')  
    .collapse({'strong_connections': 'count'})
    .collect())
```

### Mixed Type Operations
```python
# Start with nodes, end with meta-nodes
pipeline_result = (g.nodes.ids()
    .iter()
    .filter_by_attribute('role', 'manager')
    .get_neighbors()
    .flatten()  # Flatten list of neighbor lists
    .to_subgraph()
    .collapse({'team_lead_influence': 'sum'})
    .collect())
```

## Technical Considerations

### Type Safety
- Use Rust's type system to ensure operations are valid for element types
- Python wrapper provides runtime type checking
- Clear error messages for invalid operation chains

### Performance
- Lazy evaluation: operations only execute when `.collect()` is called
- Bulk operations where possible (vectorized filtering, batch graph queries)
- Memory efficient: avoid materializing intermediate collections

### Extensibility  
- Easy to add new operation methods to `GraphArrayIterator<T>`
- New collection types implement `GraphArray<T>` trait
- Plugin system for custom operations

### Error Handling
- Operations can fail gracefully (e.g., invalid queries, missing attributes)
- Option to skip failed elements vs. abort entire chain
- Detailed error context for debugging

## Migration Strategy

### Backward Compatibility
- Existing APIs continue to work unchanged
- `.iter()` is additive - doesn't break existing code
- Old manual iteration patterns still supported

### Gradual Adoption
1. **Phase 1**: Add `.iter()` to `PyComponentsArray` 
2. **Phase 2**: Extend to other existing array types
3. **Phase 3**: Create new specialized array types  
4. **Phase 4**: Optimize and add advanced operations

## Success Criteria

### Functional
- ✅ Fluent chaining syntax works on all collection types
- ✅ Type-safe operations and result collections  
- ✅ Comprehensive operation coverage (filter, transform, aggregate)
- ✅ Error handling and debugging support

### Performance  
- ✅ Chained operations perform comparably to manual loops
- ✅ Memory usage remains reasonable for large collections
- ✅ Lazy evaluation prevents unnecessary computation

### Developer Experience
- ✅ Intuitive and readable API
- ✅ Clear documentation and examples
- ✅ Good error messages and debugging support
- ✅ Easy to extend with new operations

This system will make batch graph operations significantly more readable and maintainable while maintaining the performance characteristics of the existing codebase.
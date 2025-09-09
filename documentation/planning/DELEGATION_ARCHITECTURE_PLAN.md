# Delegation Architecture Plan: Seamless Object Chaining

## Vision

Create a system where objects can seamlessly transform into other objects through method delegation, enabling infinite combinations like:

```python
g.connected_components().iter().sample(5).neighborhood().table().agg({"weight": "mean"}).collect()
```

**Key Principle**: Separate algorithms from carriers. Algorithms live in core types (optimized once), carriers (arrays, iterators) forward operations via delegation.

## Core Architecture

### 1. Pure "Ops" Traits (Algorithms Layer)

Define capabilities once, implement on core types only:

```rust
pub trait SubgraphOps {
    fn neighborhood(&self, radius: Option<usize>) -> Subgraph;
    fn table(&self) -> NodesTable;
    fn sample(&self, k: usize) -> Subgraph;
    fn filter_nodes(&self, query: &str) -> Subgraph;
    fn edges_table(&self) -> EdgesTable;
    // ... more operations
}

pub trait TableOps {
    fn agg(&self, spec: &AggSpec) -> BaseTable;
    fn filter(&self, expr: &str) -> Self;
    fn group_by(&self, columns: &[&str]) -> GroupedTable;
    fn join(&self, other: &Self, on: &str) -> Self;
    // ... more operations
}

pub trait GraphOps {
    fn connected_components(&self) -> SubgraphArray;
    fn shortest_path(&self, from: NodeId, to: NodeId) -> Option<Subgraph>;
    fn bfs(&self, start: NodeId) -> Subgraph;
    // ... more operations
}
```

**Implementation**: Only on concrete types where optimized algorithms already exist:
```rust
impl SubgraphOps for Subgraph { /* existing optimized code */ }
impl TableOps for NodesTable { /* existing optimized code */ }
impl TableOps for EdgesTable { /* existing optimized code */ }
impl GraphOps for Graph { /* existing optimized code */ }
```

### 2. Generic Delegating Iterator

A single, lazy iterator that forwards operations by mapping over elements:

```rust
pub struct DelegatingIterator<T> {
    inner: Box<dyn Iterator<Item = T>>,
}

impl<T> DelegatingIterator<T> {
    pub fn new<I: Iterator<Item = T> + 'static>(iter: I) -> Self {
        Self { inner: Box::new(iter) }
    }
    
    pub fn map<U, F>(self, f: F) -> DelegatingIterator<U>
    where F: Fn(T) -> U + 'static {
        DelegatingIterator::new(self.inner.map(f))
    }
    
    pub fn flat_map<U, I, F>(self, f: F) -> DelegatingIterator<U>
    where 
        F: Fn(T) -> I + 'static,
        I: IntoIterator<Item = U> + 'static,
    {
        DelegatingIterator::new(self.inner.flat_map(f))
    }
    
    pub fn collect_vec(self) -> Vec<T> { 
        self.inner.collect() 
    }
}
```

### 3. Typed Method Forwarding

Forward methods by mapping scalar operations across iterator elements:

```rust
impl DelegatingIterator<Subgraph> {
    pub fn neighborhood(self, radius: Option<usize>) -> DelegatingIterator<Subgraph> {
        self.map(move |subgraph| subgraph.neighborhood(radius))
    }
    
    pub fn table(self) -> DelegatingIterator<NodesTable> {
        self.map(|subgraph| subgraph.table())
    }
    
    pub fn sample(self, k: usize) -> DelegatingIterator<Subgraph> {
        self.map(move |subgraph| subgraph.sample(k))
    }
}

impl DelegatingIterator<NodesTable> {
    pub fn agg(self, spec: AggSpec) -> DelegatingIterator<BaseTable> {
        self.map(move |table| table.agg(&spec))
    }
    
    pub fn filter(self, expr: String) -> DelegatingIterator<NodesTable> {
        self.map(move |table| table.filter(&expr))
    }
}
```

### 4. Typed Array Carriers

Thin wrappers around vectors that provide iterator access:

```rust
pub struct SubgraphArray {
    inner: Arc<Vec<Subgraph>>,
}

pub struct TableArray {
    inner: Arc<Vec<BaseTable>>,
}

impl SubgraphArray {
    pub fn iter(&self) -> DelegatingIterator<Subgraph> {
        DelegatingIterator::new(self.inner.iter().cloned())
    }
    
    pub fn len(&self) -> usize { self.inner.len() }
}

impl TableArray {
    pub fn iter(&self) -> DelegatingIterator<BaseTable> {
        DelegatingIterator::new(self.inner.iter().cloned())
    }
}

impl From<Vec<BaseTable>> for TableArray {
    fn from(tables: Vec<BaseTable>) -> Self {
        Self { inner: Arc::new(tables) }
    }
}
```

### 5. Python FFI Layer

Mirror the carriers and iterators, forward to Rust implementations:

```rust
#[pyclass(name = "SubgraphArray", unsendable)]
pub struct PySubgraphArray {
    inner: SubgraphArray,
}

#[pymethods]
impl PySubgraphArray {
    fn iter(&self) -> PySubgraphIterator {
        PySubgraphIterator { inner: self.inner.iter() }
    }
    
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "SubgraphIterator", unsendable)]
pub struct PySubgraphIterator {
    inner: DelegatingIterator<Subgraph>,
}

#[pymethods]
impl PySubgraphIterator {
    fn sample(&mut self, k: usize) -> PyResult<Self> {
        Ok(Self { 
            inner: std::mem::take(&mut self.inner).sample(k) 
        })
    }
    
    fn neighborhood(&mut self, radius: Option<usize>) -> PyResult<Self> {
        Ok(Self { 
            inner: std::mem::take(&mut self.inner).neighborhood(radius) 
        })
    }
    
    fn table(&mut self) -> PyResult<PyTableIterator> {
        Ok(PyTableIterator { 
            inner: std::mem::take(&mut self.inner).table() 
        })
    }
    
    fn collect(&mut self) -> PyResult<PySubgraphArray> {
        let vec = std::mem::take(&mut self.inner).collect_vec();
        Ok(PySubgraphArray { inner: vec.into() })
    }
}
```

## Implementation Roadmap

### Phase 1: Core Traits and Iterator
1. Define `SubgraphOps`, `TableOps`, `GraphOps` traits
2. Implement `DelegatingIterator<T>`
3. Create forwarding methods for `DelegatingIterator<Subgraph>`
4. Test basic chaining in Rust

### Phase 2: Array Carriers
1. Implement `SubgraphArray`, `TableArray`
2. Add `From<Vec<T>>` conversions
3. Test array → iterator → collect patterns

### Phase 3: Python FFI Integration
1. Create `PySubgraphArray`, `PyTableArray`
2. Create `PySubgraphIterator`, `PyTableIterator`  
3. Wire up delegation methods
4. Test Python chaining: `components.iter().sample().collect()`

### Phase 4: Full Method Coverage
1. Add all core operations to trait definitions
2. Implement forwarding for all iterator types
3. Add error handling (`try_map`, `Result` propagation)
4. Performance optimization

### Phase 5: Advanced Features
1. Mixed-type arrays (`AnyObject` enum)
2. Dynamic method dispatch for Python flexibility
3. Parallel processing support (`ParDelegatingIterator`)
4. Macro-based boilerplate reduction

## Type Flow Examples

### Example 1: Component Analysis
```
Graph 
  → connected_components() → SubgraphArray
  → .iter() → DelegatingIterator<Subgraph>
  → .sample(5) → DelegatingIterator<Subgraph>
  → .table() → DelegatingIterator<NodesTable>
  → .agg(spec) → DelegatingIterator<BaseTable>
  → .collect() → TableArray
```

### Example 2: Neighborhood Analysis  
```
Graph
  → bfs(start) → Subgraph
  → neighborhood() → Subgraph
  → table() → NodesTable
  → filter("age > 25") → NodesTable
```

### Example 3: Multi-level Analysis
```
Graph
  → connected_components() → SubgraphArray
  → .iter() → DelegatingIterator<Subgraph>  
  → .neighborhood() → DelegatingIterator<Subgraph>
  → .table() → DelegatingIterator<NodesTable>
  → .group_by(["department"]) → DelegatingIterator<GroupedTable>
  → .collect() → GroupedTableArray
```

## Benefits

### 1. Zero Algorithm Duplication
- Algorithms implemented once in core types
- Carriers only forward operations
- Optimized code stays optimized

### 2. Type Safety
- Compile-time method availability checking
- No runtime "method not found" errors
- Clear transformation paths

### 3. Infinite Composability
- Any valid sequence of transformations works
- Repository becomes a "graph of possibilities"
- Easy to discover new patterns

### 4. Performance
- Lazy evaluation in iterators
- Zero-copy where possible
- Parallel processing ready

### 5. Python Ergonomics
- Fluent, chainable API
- Natural method discovery
- Seamless integration with existing code

## Safety and Error Handling

### Compile-time Safety
- Method availability enforced by trait bounds
- Type transformations validated at compile time
- No "method not available on this type" runtime errors

### Runtime Safety
- Result propagation through `try_map`
- Graceful error handling in chains
- Clear error messages with context

### Memory Safety
- Arc/Rc for shared ownership
- Iterator invalidation prevention
- Lazy evaluation prevents excessive memory usage

## Migration Strategy

### 1. Incremental Adoption
- Start with high-value chains (connected_components)
- Gradually expand method coverage
- Maintain backward compatibility

### 2. Existing Code Integration
- Current APIs remain unchanged
- New delegation methods as additions
- Seamless interop between old and new patterns

### 3. Performance Validation
- Benchmark critical paths
- Ensure no performance regression
- Optimize hot paths

## Open Questions

1. **Error Propagation**: How to handle chains where intermediate steps can fail?
2. **Memory Management**: When to materialize vs. stay lazy?
3. **Python Integration**: Balance between type safety and Python flexibility?
4. **Parallel Processing**: When and how to introduce parallel map operations?
5. **Caching**: Should intermediate results be cached in iterators?

## Success Metrics

1. **API Usability**: Can users naturally discover and chain operations?
2. **Performance**: No regression on existing algorithms
3. **Type Safety**: Compile-time prevention of invalid chains
4. **Code Maintainability**: Reduced duplication, clear separation of concerns
5. **Extensibility**: Easy to add new operations and types

---

This architecture transforms the repository into a "graph of transformations" where users can travel along any valid edge (method call) to reach their desired result, with full type safety and zero algorithm duplication.
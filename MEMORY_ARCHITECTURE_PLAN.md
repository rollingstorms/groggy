# Comprehensive Memory Architecture Redesign Plan

## Current Problems (Root Cause Analysis)

### 1. **JSON Serialization Overhead** (Major Issue)
- **Problem**: All attributes are JSON-serialized in Python, sent to Rust, then deserialized
- **Memory Impact**: 2x memory usage for all attribute data
- **Performance Impact**: Serialization/deserialization on every attribute access

### 2. **Multiple Python Wrapper Layers** (Major Issue)
- **Problem**: 3-4 layers of Python objects wrapping Rust objects
- **Current Stack**: 
  ```
  User Code → Python NodeProxy → Python Collection → Rust Proxy → Rust Core
  ```
- **Memory Impact**: Each layer holds references and creates objects

### 3. **Inefficient Data Structures** (Major Issue)
- **Problem**: Rust stores JSON strings instead of native types
- **Memory Impact**: String overhead + parsing overhead
- **Example**: `{"salary": 50000}` becomes `{"salary": "50000"}` (more bytes + string overhead)

### 4. **No Zero-Copy Operations** (Major Issue)
- **Problem**: Data copied between Python and Rust multiple times
- **Memory Impact**: Temporary copies during operations

## Proposed Architecture Changes

### Phase 1: Eliminate JSON Serialization (Weeks 1-2)

#### **1.1 Replace JSON with Native Python/Rust Types**
```rust
// Current (bad):
pub struct AttributeValue {
    json_string: String,
}

// New (good):
pub enum AttributeValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<AttributeValue>),
    Object(HashMap<String, AttributeValue>),
}
```

#### **1.2 Use PyO3 Native Type Conversion**
```rust
// Instead of manual JSON:
impl FromPyObject<'_> for AttributeValue {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            Ok(AttributeValue::String(s))
        } else if let Ok(i) = ob.extract::<i64>() {
            Ok(AttributeValue::Integer(i))
        } // ... etc
    }
}
```

#### **1.3 Direct Memory Mapping**
```rust
// Use PyO3's memory mapping for large datasets
#[pyclass]
pub struct NodeCollection {
    // Direct memory view of Python data
    ids: Vec<String>,
    attributes: HashMap<String, HashMap<String, AttributeValue>>,
}
```

### Phase 2: Reduce Python Wrapper Layers (Weeks 2-3)

#### **2.1 Eliminate Intermediate Python Objects**
```python
# Current (bad):
class NodeCollection:
    def __init__(self, graph):
        self.graph = graph  # Holds reference to entire graph
        self._rust = graph._rust.nodes()  # Wrapper around Rust
        self.attr = NodeAttributeManager(self)  # Another wrapper

# New (good):
# Direct Rust object with minimal Python wrapper
nodes = graph._rust_graph.nodes()  # Direct Rust object
node = nodes.get("n1")  # Direct Rust proxy
value = node.get_attr("salary")  # Direct Rust call
```

#### **2.2 Use Rust Objects Directly**
```rust
#[pyclass]
pub struct Graph {
    nodes: NodeCollection,
    edges: EdgeCollection,
}

#[pymethods]
impl Graph {
    #[getter]
    fn nodes(&self) -> &NodeCollection {
        &self.nodes  // Direct reference, no Python wrapper
    }
}
```

### Phase 3: Optimize Data Structures (Weeks 3-4)

#### **3.1 Columnar Storage for Attributes**
```rust
// Instead of row-based storage:
// {"n1": {"salary": 50000, "role": "engineer"}, "n2": {...}}

// Use columnar storage:
pub struct AttributeStore {
    salary: Vec<Option<i64>>,      // All salaries in one vector
    role: Vec<Option<String>>,     // All roles in one vector
    node_index: HashMap<String, usize>, // Map node_id -> index
}
```

#### **3.2 Memory-Efficient Node/Edge Storage**
```rust
// Use packed structs and ID compression
#[repr(C, packed)]
pub struct NodeId(u32);  // Instead of String

#[repr(C, packed)]
pub struct EdgeId {
    source: NodeId,
    target: NodeId,
}
```

### Phase 4: Implement Zero-Copy Operations (Weeks 4-5)

#### **4.1 Memory Views Instead of Copies**
```rust
// Return memory views instead of copying data
#[pyclass]
pub struct NodeAttributeView {
    data: *const AttributeValue,  // Direct memory pointer
    size: usize,
}

#[pymethods]
impl NodeAttributeView {
    fn __iter__(&self) -> PyResult<PyIterator> {
        // Iterator over memory view, no copying
    }
}
```

#### **4.2 Batch Operations with Memory Mapping**
```rust
// Instead of individual attribute sets:
nodes.set_attr("n1", "salary", 50000);  // Individual Python call

// Use batch operations:
let batch = AttributeBatch::from_python_dict(py_dict)?;
nodes.set_attrs_batch(batch);  // Single Rust call
```

### Phase 5: Python API Redesign (Weeks 5-6)

#### **5.1 Minimal Python Wrapper**
```python
# New clean API that maps directly to Rust
class Graph:
    def __init__(self):
        self._rust = groggy_rust.Graph()
    
    @property
    def nodes(self):
        return self._rust.nodes()  # Direct Rust object
    
    @property  
    def edges(self):
        return self._rust.edges()  # Direct Rust object
```

#### **5.2 Direct Attribute Access**
```python
# Instead of complex proxy objects:
node = graph.nodes.get("n1")
salary = node.salary  # Direct attribute access
node.salary = 60000   # Direct assignment

# Implemented in Rust:
#[pyclass]
pub struct NodeProxy {
    #[pyo3(get, set)]
    pub salary: Option<i64>,
}
```

## Implementation Strategy

### **Week 1: Foundation Changes**
- [ ] Replace JSON serialization with native types in Rust
- [ ] Implement PyO3 native type conversion
- [ ] Add benchmarks for memory usage at each step

### **Week 2: Data Structure Optimization**
- [ ] Implement columnar attribute storage
- [ ] Add memory-efficient node/edge IDs
- [ ] Benchmark memory usage improvements

### **Week 3: Wrapper Layer Reduction**
- [ ] Remove intermediate Python objects
- [ ] Implement direct Rust object access
- [ ] Update Python API to use direct Rust calls

### **Week 4: Zero-Copy Operations**
- [ ] Implement memory views for large datasets
- [ ] Add batch operations with memory mapping
- [ ] Optimize iterator implementations

### **Week 5: API Redesign**
- [ ] Implement new minimal Python wrapper
- [ ] Add direct attribute access
- [ ] Migrate existing tests to new API

### **Week 6: Validation & Optimization**
- [ ] Comprehensive benchmarking
- [ ] Memory leak detection
- [ ] Performance regression testing

## Expected Memory Improvements

### **Target Memory Usage** (10,000 nodes with attributes):
- **Current**: 39.7 MB
- **Phase 1** (No JSON): ~25 MB (-37%)
- **Phase 2** (Fewer wrappers): ~15 MB (-40%)
- **Phase 3** (Columnar storage): ~10 MB (-33%)
- **Phase 4** (Zero-copy): ~8 MB (-20%)
- **Final Target**: ~8 MB (comparable to NetworkX's 9.4 MB)

### **Architecture Comparison**:
```
Current:     [Python] → [JSON] → [Rust] → [JSON] → [Storage]
Proposed:    [Python] → [Direct] → [Rust Storage]
```

## Risk Assessment

### **High Risk**:
- **Breaking API changes**: Existing code will need migration
- **Development time**: 6 weeks of focused work
- **Testing complexity**: Need comprehensive test coverage

### **Medium Risk**:
- **Performance regressions**: Need careful benchmarking
- **Memory leaks**: PyO3 memory management complexity

### **Low Risk**:
- **Rust compilation**: Well-established PyO3 patterns
- **Python compatibility**: Standard PyO3 practices

## Success Metrics

### **Memory Usage** (Primary Goal):
- Groggy ≤ 1.5x NetworkX memory usage (target: ≤14 MB for 10K nodes)
- Groggy ≤ 3x igraph memory usage (target: ≤7 MB for 10K nodes)

### **Performance** (Secondary Goal):
- Graph creation time ≤ 5x NetworkX (currently 13x)
- Attribute access time ≤ 2x NetworkX

### **API Quality** (Tertiary Goal):
- Maintain current API compatibility where possible
- Improve attribute access ergonomics

## Next Steps

1. **Approve this plan** and timeline
2. **Start with Phase 1** (JSON elimination) - highest impact
3. **Create benchmark suite** to track improvements
4. **Implement changes incrementally** with continuous testing
5. **Migrate existing tests** as we progress

This architectural redesign should bring Groggy's memory usage down to competitive levels while maintaining (or improving) performance.
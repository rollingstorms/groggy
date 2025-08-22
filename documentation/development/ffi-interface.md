# FFI Interface Guide

This document explains how Groggy's Foreign Function Interface (FFI) bridges the Rust core with Python, enabling safe and efficient cross-language operations while maintaining performance and memory safety.

## Table of Contents

1. [Overview](#overview)
2. [FFI Architecture](#ffi-architecture)
3. [Type System Integration](#type-system-integration)
4. [Memory Management](#memory-management)
5. [Error Handling](#error-handling)
6. [Performance Considerations](#performance-considerations)
7. [Development Patterns](#development-patterns)

## Overview

Groggy's FFI layer uses PyO3 to create Python bindings for the Rust core. The design prioritizes:

- **Safety**: No unsafe operations exposed to Python
- **Performance**: Minimal overhead for cross-language calls
- **Ergonomics**: Pythonic API that feels natural to Python users
- **Memory Efficiency**: Zero-copy operations where possible

## FFI Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Python User Code                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│              Python API Layer                              │
│        (python-groggy/python/groggy/*.py)                  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                PyO3 FFI Bindings                           │
│           (python-groggy/src/ffi/*.rs)                     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Rust Core                                 │
│                (src/core/*.rs)                             │
└─────────────────────────────────────────────────────────────┘
```

### Core FFI Modules

The FFI layer is organized into specialized modules:

```
python-groggy/src/ffi/
├── mod.rs              # Main FFI module and exports
├── types.rs            # Type conversions and definitions
├── errors.rs           # Error handling and conversion
├── utils.rs            # Common utilities and helpers
├── config.rs           # Configuration management
├── display.rs          # Display and formatting support
├── api/
│   ├── graph.rs        # Main Graph FFI interface
│   ├── graph_query.rs  # Query operations
│   ├── graph_analytics.rs # Analytics and algorithms
│   └── graph_version.rs   # Version and state management
└── core/
    ├── array.rs        # PyGraphArray bindings
    ├── matrix.rs       # PyGraphMatrix bindings
    ├── table.rs        # PyGraphTable bindings
    ├── subgraph.rs     # PySubgraph bindings
    ├── attributes.rs   # Attribute management
    ├── query.rs        # Query interface
    ├── traversal.rs    # Graph traversal
    ├── history.rs      # History and state management
    ├── views.rs        # Graph views and filtering
    └── accessors.rs    # Data accessors
```

## Type System Integration

### Rust to Python Type Mapping

Groggy defines a comprehensive type mapping system:

```rust
// python-groggy/src/ffi/types.rs

#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Null,
}

impl IntoPy<PyObject> for AttrValue {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            AttrValue::String(s) => s.into_py(py),
            AttrValue::Int(i) => i.into_py(py),
            AttrValue::Float(f) => f.into_py(py),
            AttrValue::Bool(b) => b.into_py(py),
            AttrValue::Bytes(b) => PyBytes::new(py, &b).into_py(py),
            AttrValue::Null => py.None(),
        }
    }
}

impl FromPyObject<'_> for AttrValue {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        if obj.is_none() {
            Ok(AttrValue::Null)
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(AttrValue::String(s))
        } else if let Ok(i) = obj.extract::<i64>() {
            Ok(AttrValue::Int(i))
        } else if let Ok(f) = obj.extract::<f64>() {
            Ok(AttrValue::Float(f))
        } else if let Ok(b) = obj.extract::<bool>() {
            Ok(AttrValue::Bool(b))
        } else if let Ok(bytes) = obj.extract::<&PyBytes>() {
            Ok(AttrValue::Bytes(bytes.as_bytes().to_vec()))
        } else {
            Err(PyTypeError::new_err(format!(
                "Cannot convert {} to AttrValue", 
                obj.get_type().name()?
            )))
        }
    }
}
```

### Complex Type Conversions

For more complex types like dictionaries and lists:

```rust
impl FromPyObject<'_> for HashMap<String, AttrValue> {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        let dict = obj.downcast::<PyDict>()?;
        let mut result = HashMap::new();
        
        for (key, value) in dict.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = AttrValue::extract(value)?;
            result.insert(key_str, attr_value);
        }
        
        Ok(result)
    }
}

impl IntoPy<PyObject> for HashMap<String, AttrValue> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new(py);
        for (key, value) in self {
            dict.set_item(key, value.into_py(py)).unwrap();
        }
        dict.into()
    }
}
```

## Memory Management

### Ownership Patterns

Groggy uses several patterns to manage memory safely across the FFI boundary:

#### 1. **Shared Ownership with Arc<RwLock<>>**

```rust
#[pyclass(name = "Graph")]
pub struct PyGraph {
    // Shared ownership of the core graph
    inner: Arc<RwLock<Graph>>,
}

impl PyGraph {
    pub fn with_graph<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&Graph) -> GraphResult<R>,
    {
        let graph = self.inner.read()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
        
        f(&*graph).map_err(|e| e.into())
    }
    
    pub fn with_graph_mut<F, R>(&self, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut Graph) -> GraphResult<R>,
    {
        let mut graph = self.inner.write()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock error: {}", e)))?;
        
        f(&mut *graph).map_err(|e| e.into())
    }
}
```

#### 2. **Reference Tracking for Views**

```rust
#[pyclass(name = "GraphArray")]
pub struct PyGraphArray {
    array: GraphArray,
    // Keep reference to parent to prevent GC
    _parent: Option<Py<PyObject>>,
}

impl PyGraphArray {
    pub fn from_graph_with_parent(
        array: GraphArray,
        parent: Py<PyObject>
    ) -> Self {
        Self {
            array,
            _parent: Some(parent),
        }
    }
}
```

#### 3. **Zero-Copy Views**

```rust
#[pymethods]
impl PyGraphArray {
    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Get reference to underlying data without copying
        let values = &self.array.values;
        
        match self.array.dtype() {
            AttrValueType::Float => {
                let float_values: Vec<f64> = values.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect();
                
                // Create numpy array with zero-copy where possible
                numpy::PyArray1::from_vec(py, float_values).into_py(py)
            }
            // ... other types
        }
    }
}
```

### Memory Safety Guarantees

1. **No Unsafe Code**: All FFI operations use safe Rust patterns
2. **Reference Counting**: Shared data uses Arc for safe concurrent access
3. **Lifetime Management**: Python objects hold references to prevent premature GC
4. **Error Propagation**: All Rust errors are safely converted to Python exceptions

## Error Handling

### Error Conversion System

Groggy implements comprehensive error conversion from Rust to Python:

```rust
// python-groggy/src/ffi/errors.rs

impl From<GraphError> for PyErr {
    fn from(err: GraphError) -> PyErr {
        match err {
            GraphError::NodeNotFound(id) => {
                PyKeyError::new_err(format!("Node not found: {}", id))
            }
            GraphError::EdgeNotFound(id) => {
                PyKeyError::new_err(format!("Edge not found: {}", id))
            }
            GraphError::TypeMismatch { expected, found } => {
                PyTypeError::new_err(format!(
                    "Type mismatch: expected {}, found {}", 
                    expected, found
                ))
            }
            GraphError::InvalidOperation(msg) => {
                PyValueError::new_err(msg)
            }
            GraphError::IoError(msg) => {
                PyIOError::new_err(msg)
            }
            // ... more specific error mappings
        }
    }
}
```

### Error Context Enhancement

```rust
pub trait PyResultExt<T> {
    fn with_context(self, context: &str) -> PyResult<T>;
}

impl<T> PyResultExt<T> for GraphResult<T> {
    fn with_context(self, context: &str) -> PyResult<T> {
        self.map_err(|e| {
            let enhanced_msg = format!("{}: {}", context, e);
            match e {
                GraphError::NodeNotFound(_) => PyKeyError::new_err(enhanced_msg),
                GraphError::TypeMismatch { .. } => PyTypeError::new_err(enhanced_msg),
                _ => PyRuntimeError::new_err(enhanced_msg),
            }
        })
    }
}
```

## Performance Considerations

### Minimizing FFI Overhead

#### 1. **Batch Operations**

```rust
#[pymethods]
impl PyGraph {
    // Instead of individual calls, provide batch operations
    fn add_nodes(&mut self, py: Python<'_>, nodes: PyObject) -> PyResult<()> {
        // Convert Python list to Rust Vec in one go
        let node_data: Vec<(NodeId, HashMap<String, AttrValue>)> = 
            nodes.extract(py)?;
        
        // Single call to Rust core
        self.with_graph_mut(|graph| {
            graph.add_nodes_batch(node_data)
        })
    }
}
```

#### 2. **Lazy Evaluation**

```rust
#[pymethods]
impl PyGraphArray {
    #[getter]
    fn mean(&self) -> PyResult<Option<f64>> {
        // Computation happens in Rust, result cached
        Ok(self.array.mean())
    }
    
    #[getter]
    fn values(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Only convert to Python when explicitly requested
        self.array.values.iter()
            .map(|v| v.clone().into_py(py))
            .collect::<Vec<_>>()
            .into_py(py)
    }
}
```

#### 3. **Reference Reuse**

```rust
#[pyclass(name = "GraphTable")]
pub struct PyGraphTable {
    table: GraphTable,
    // Cache frequently accessed columns
    column_cache: RefCell<HashMap<String, Py<PyGraphArray>>>,
}

#[pymethods]
impl PyGraphTable {
    fn __getitem__(&self, py: Python<'_>, column: &str) -> PyResult<Py<PyGraphArray>> {
        // Check cache first
        if let Some(cached) = self.column_cache.borrow().get(column) {
            return Ok(cached.clone());
        }
        
        // Create new array and cache it
        let array = self.table.get_column(column)
            .ok_or_else(|| PyKeyError::new_err(format!("Column not found: {}", column)))?;
        
        let py_array = PyGraphArray::from_array(array.clone());
        let py_obj = Py::new(py, py_array)?;
        
        self.column_cache.borrow_mut().insert(column.to_string(), py_obj.clone());
        Ok(py_obj)
    }
}
```

### Memory Optimization

#### 1. **Streaming Large Results**

```rust
#[pymethods]
impl PyGraphTable {
    fn iter_rows(&self, py: Python<'_>) -> PyResult<PyRowIterator> {
        // Return iterator instead of materializing all rows
        PyRowIterator::new(py, self.table.clone())
    }
}

#[pyclass]
struct PyRowIterator {
    table: GraphTable,
    current_row: usize,
}

#[pymethods]
impl PyRowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        if self.current_row >= self.table.row_count() {
            Ok(None)
        } else {
            let row = self.table.get_row(self.current_row)?;
            self.current_row += 1;
            Ok(Some(row.into_py(py)))
        }
    }
}
```

## Development Patterns

### Adding a New FFI Method

1. **Define the Rust Core Function**
```rust
// In src/core/table.rs
impl GraphTable {
    pub fn filter_rows<F>(&self, predicate: F) -> GraphResult<GraphTable>
    where
        F: Fn(&HashMap<String, AttrValue>) -> bool,
    {
        // Implementation
    }
}
```

2. **Create FFI Wrapper**
```rust
// In python-groggy/src/ffi/core/table.rs
#[pymethods]
impl PyGraphTable {
    fn filter_rows(&self, py: Python<'_>, predicate: PyObject) -> PyResult<PyGraphTable> {
        // Convert Python predicate to Rust closure
        let rust_predicate = |row: &HashMap<String, AttrValue>| -> bool {
            let py_row = row.clone().into_py(py);
            predicate.call1(py, (py_row,))
                .and_then(|result| result.extract::<bool>(py))
                .unwrap_or(false)
        };
        
        // Call Rust core
        let filtered_table = self.table.filter_rows(rust_predicate)
            .map_err(|e| e.into())?;
        
        Ok(PyGraphTable::from_table(filtered_table))
    }
}
```

3. **Add Python Helper (Optional)**
```python
# In python-groggy/python/groggy/table_extensions.py
def filter(self, condition):
    """
    Filter table rows using a more Pythonic interface.
    
    Args:
        condition: String expression like 'age > 30' or callable
    """
    if isinstance(condition, str):
        # Parse string condition into callable
        predicate = self._parse_condition(condition)
    else:
        predicate = condition
    
    return self.filter_rows(predicate)
```

### Testing FFI Methods

```python
# In tests/test_ffi_table.py
def test_table_filter_rows():
    import groggy as gr
    
    # Create test data
    table = gr.table({
        'name': ['alice', 'bob', 'charlie'],
        'age': [30, 25, 35],
        'active': [True, False, True]
    })
    
    # Test Python predicate
    filtered = table.filter_rows(lambda row: row['age'] > 28)
    assert len(filtered) == 2
    assert 'alice' in filtered['name'].values
    assert 'charlie' in filtered['name'].values
    
    # Test error handling
    with pytest.raises(TypeError):
        table.filter_rows("invalid_predicate")
```

## Best Practices

### 1. **Error Context**
Always provide meaningful error context:
```rust
self.with_graph(|graph| {
    graph.get_node(node_id)
}).with_context(&format!("Getting node {}", node_id))
```

### 2. **Type Validation**
Validate types early:
```rust
fn add_node(&mut self, py: Python<'_>, node_id: PyObject, attrs: PyObject) -> PyResult<()> {
    let node_id: NodeId = node_id.extract(py)
        .map_err(|_| PyTypeError::new_err("node_id must be string or int"))?;
    
    let attributes: HashMap<String, AttrValue> = attrs.extract(py)
        .map_err(|_| PyTypeError::new_err("attributes must be dict"))?;
    
    // ... rest of implementation
}
```

### 3. **Resource Management**
Use RAII patterns and explicit cleanup:
```rust
impl Drop for PyGraph {
    fn drop(&mut self) {
        // Explicit cleanup if needed
        if let Ok(mut graph) = self.inner.write() {
            graph.cleanup_resources();
        }
    }
}
```

### 4. **Documentation**
Provide comprehensive docstrings:
```rust
#[pymethods]
impl PyGraphArray {
    /// Calculate the arithmetic mean of array values.
    /// 
    /// Returns:
    ///     float or None: The mean value, or None if array is empty or contains no numeric values.
    /// 
    /// Examples:
    ///     >>> arr = gr.array([1, 2, 3, 4, 5])
    ///     >>> arr.mean()
    ///     3.0
    #[getter]
    fn mean(&self) -> PyResult<Option<f64>> {
        Ok(self.array.mean())
    }
}
```

This FFI design ensures that Groggy provides a safe, efficient, and Pythonic interface while leveraging the performance benefits of the Rust core.
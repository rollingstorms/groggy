FFI (Foreign Function Interface) Guide
=====================================

This guide covers the Foreign Function Interface (FFI) between Python and Rust in GLI, including how it works, performance implications, and advanced usage.

Overview
--------

GLI uses PyO3 to create seamless bindings between Python and Rust, allowing high-performance Rust code to be called from Python with minimal overhead.

Architecture
~~~~~~~~~~~

.. code-block::

   ┌─────────────────────────────┐
   │         Python Code         │
   │   g = Graph(backend='rust') │
   │   g.add_node("alice")       │
   └─────────────┬───────────────┘
                 │ Python API call
                 ▼
   ┌─────────────────────────────┐
   │      PyO3 Bindings          │
   │   Convert Python → Rust     │
   │   Convert Rust → Python     │
   └─────────────┬───────────────┘
                 │ Native call
                 ▼
   ┌─────────────────────────────┐
   │       Rust Backend          │
   │   FastGraph::add_node()     │
   │   High-performance logic    │
   └─────────────────────────────┘

PyO3 Integration
---------------

Basic FFI Structure
~~~~~~~~~~~~~~~~~~

GLI's Rust backend is structured as a PyO3 module:

.. code-block:: rust

   use pyo3::prelude::*;
   use std::collections::HashMap;
   
   #[pyclass]
   pub struct FastGraph {
       nodes: HashMap<String, RustNode>,
       edges: HashMap<String, RustEdge>,
   }
   
   #[pymethods]
   impl FastGraph {
       #[new]
       fn new() -> Self {
           FastGraph {
               nodes: HashMap::new(),
               edges: HashMap::new(),
           }
       }
       
       fn add_node(&mut self, id: String, attributes: &PyDict) -> PyResult<String> {
           // Convert Python dict to Rust data structures
           let rust_attributes = convert_py_dict_to_rust(attributes)?;
           
           let node = RustNode {
               id: id.clone(),
               attributes: rust_attributes,
           };
           
           self.nodes.insert(id.clone(), node);
           Ok(id)
       }
   }
   
   #[pymodule]
   fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
       m.add_class::<FastGraph>()?;
       Ok(())
   }

Data Type Conversion
~~~~~~~~~~~~~~~~~~~

PyO3 handles automatic conversion between Python and Rust types:

.. code-block:: rust

   // Automatic conversions
   fn rust_function(
       string_val: String,        // Python str → Rust String
       int_val: i64,             // Python int → Rust i64
       float_val: f64,           // Python float → Rust f64
       bool_val: bool,           // Python bool → Rust bool
       list_val: Vec<String>,    // Python list → Rust Vec
       dict_val: HashMap<String, PyObject>, // Python dict → Rust HashMap
   ) -> PyResult<String> {
       // Function implementation
       Ok("success".to_string())
   }

Complex Type Handling
~~~~~~~~~~~~~~~~~~~~

For complex Python objects, GLI uses custom conversion logic:

.. code-block:: rust

   use pyo3::types::{PyDict, PyList, PyAny};
   
   fn convert_python_attributes(py_dict: &PyDict) -> PyResult<AttributeMap> {
       let mut attributes = AttributeMap::new();
       
       for (key, value) in py_dict {
           let key_str: String = key.extract()?;
           
           // Handle different Python types
           let rust_value = if let Ok(s) = value.extract::<String>() {
               AttributeValue::String(s)
           } else if let Ok(i) = value.extract::<i64>() {
               AttributeValue::Integer(i)
           } else if let Ok(f) = value.extract::<f64>() {
               AttributeValue::Float(f)
           } else if let Ok(b) = value.extract::<bool>() {
               AttributeValue::Boolean(b)
           } else if let Ok(list) = value.downcast::<PyList>() {
               AttributeValue::List(convert_python_list(list)?)
           } else if let Ok(dict) = value.downcast::<PyDict>() {
               AttributeValue::Dict(convert_python_attributes(dict)?)
           } else {
               // For complex objects, store as PyObject
               AttributeValue::PyObject(value.into())
           };
           
           attributes.insert(key_str, rust_value);
       }
       
       Ok(attributes)
   }

Performance Characteristics
--------------------------

FFI Overhead Analysis
~~~~~~~~~~~~~~~~~~~~

The FFI layer introduces minimal overhead:

.. list-table:: FFI Performance Overhead
   :header-rows: 1
   :widths: 30 25 25 20

   * - Operation Type
     - Pure Rust Time
     - Python→Rust Time
     - Overhead
   * - Simple function call
     - 5 ns
     - 15 ns
     - 10 ns
   * - String conversion
     - 20 ns
     - 35 ns
     - 15 ns
   * - Dict conversion (5 items)
     - 50 ns
     - 120 ns
     - 70 ns
   * - Large dict (100 items)
     - 500 ns
     - 800 ns
     - 300 ns

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~

1. **Batch Operations**: Minimize FFI calls by batching operations

.. code-block:: python

   # Inefficient: Many FFI calls
   g = Graph(backend='rust')
   for i in range(1000):
       g.add_node(f"node_{i}")  # 1000 FFI calls
   
   # Efficient: Single FFI call
   g = Graph(backend='rust')
   with g.batch_operations() as batch:
       for i in range(1000):
           batch.add_node(f"node_{i}")  # 1 FFI call

2. **Simple Data Types**: Use simple types when possible

.. code-block:: python

   # Slower: Complex nested structures
   g.add_node("user1", profile={
       "personal": {"age": 30, "city": "NYC"},
       "work": {"title": "Engineer", "company": "TechCorp"}
   })
   
   # Faster: Flattened attributes
   g.add_node("user1", 
             age=30, 
             city="NYC", 
             job_title="Engineer", 
             company="TechCorp")

3. **Pre-convert Data**: Convert data to Rust-friendly formats

.. code-block:: python

   # Convert pandas DataFrame to efficient format
   import pandas as pd
   
   df = pd.DataFrame({
       'id': ['node_1', 'node_2', 'node_3'],
       'value': [1, 2, 3],
       'category': ['A', 'B', 'A']
   })
   
   # Convert to simple Python structures
   node_data = [
       (row['id'], {'value': row['value'], 'category': row['category']})
       for _, row in df.iterrows()
   ]
   
   # Single FFI call for all data
   g.batch_add_nodes(node_data)

Memory Management
----------------

Reference Counting
~~~~~~~~~~~~~~~~~

PyO3 uses Python's reference counting for memory management:

.. code-block:: rust

   use pyo3::prelude::*;
   
   #[pyclass]
   struct NodeData {
       #[pyo3(get)]
       id: String,
       
       // PyObject automatically handles reference counting
       attributes: Py<PyDict>,
   }
   
   #[pymethods]
   impl NodeData {
       fn get_attribute(&self, py: Python, key: &str) -> PyResult<PyObject> {
           // Safe access to Python objects from Rust
           let dict = self.attributes.as_ref(py);
           dict.get_item(key).map(|opt| opt.into())
       }
   }

Zero-Copy Operations
~~~~~~~~~~~~~~~~~~~

GLI implements zero-copy operations where possible:

.. code-block:: rust

   // Zero-copy string views
   fn process_node_id(node_id: &str) -> bool {
       // Process string without copying
       node_id.starts_with("user_")
   }
   
   // Zero-copy iteration
   fn iterate_attributes(py_dict: &PyDict) -> PyResult<Vec<String>> {
       let mut keys = Vec::new();
       
       for (key, _) in py_dict {
           // Extract string without copying when possible
           if let Ok(key_str) = key.extract::<&str>() {
               keys.push(key_str.to_string());
           }
       }
       
       Ok(keys)
   }

Error Handling
-------------

Exception Conversion
~~~~~~~~~~~~~~~~~~~

PyO3 automatically converts Rust errors to Python exceptions:

.. code-block:: rust

   use pyo3::exceptions::PyValueError;
   
   #[pymethods]
   impl FastGraph {
       fn add_edge(&mut self, source: &str, target: &str) -> PyResult<String> {
           // Check if nodes exist
           if !self.nodes.contains_key(source) {
               return Err(PyValueError::new_err(
                   format!("Source node '{}' not found", source)
               ));
           }
           
           if !self.nodes.contains_key(target) {
               return Err(PyValueError::new_err(
                   format!("Target node '{}' not found", target)
               ));
           }
           
           // Add edge logic
           let edge_id = format!("{}->{}", source, target);
           Ok(edge_id)
       }
   }

Custom Error Types
~~~~~~~~~~~~~~~~~

Define custom error types for better error handling:

.. code-block:: rust

   use pyo3::create_exception;
   
   create_exception!(gli, NodeNotFoundError, pyo3::exceptions::PyKeyError);
   create_exception!(gli, EdgeAlreadyExistsError, pyo3::exceptions::PyValueError);
   
   #[pymethods]
   impl FastGraph {
       fn get_node(&self, id: &str) -> PyResult<&RustNode> {
           self.nodes.get(id)
               .ok_or_else(|| NodeNotFoundError::new_err(
                   format!("Node '{}' not found", id)
               ))
       }
   }

Advanced FFI Patterns
--------------------

Callback Functions
~~~~~~~~~~~~~~~~

Allow Python callbacks to be called from Rust:

.. code-block:: rust

   use pyo3::types::PyFunction;
   
   #[pymethods]
   impl FastGraph {
       fn filter_nodes(&self, py: Python, predicate: &PyFunction) -> PyResult<Vec<String>> {
           let mut result = Vec::new();
           
           for (node_id, node) in &self.nodes {
               // Call Python function from Rust
               let args = (node_id.clone(), &node.attributes);
               let should_include: bool = predicate.call1(args)?.extract()?;
               
               if should_include {
                   result.push(node_id.clone());
               }
           }
           
           Ok(result)
       }
   }

.. code-block:: python

   # Usage from Python
   g = Graph(backend='rust')
   # ... populate graph ...
   
   # Define Python predicate
   def is_active_user(node_id, attributes):
       return attributes.get('status') == 'active'
   
   # Call Rust function with Python callback
   active_users = g.filter_nodes(is_active_user)

Async Support
~~~~~~~~~~~~

PyO3 supports async operations:

.. code-block:: rust

   use pyo3_asyncio::tokio::future_into_py;
   use tokio;
   
   #[pymethods]
   impl FastGraph {
       fn async_process_graph<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
           let graph_data = self.clone(); // Clone data for async context
           
           future_into_py(py, async move {
               // Async processing logic
               tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
               
               // Return result
               Ok("Processing complete")
           })
       }
   }

.. code-block:: python

   # Async usage from Python
   import asyncio
   
   async def main():
       g = Graph(backend='rust')
       # ... populate graph ...
       
       result = await g.async_process_graph()
       print(result)
   
   asyncio.run(main())

Direct Memory Access
~~~~~~~~~~~~~~~~~~~

For maximum performance, provide direct memory access:

.. code-block:: rust

   use pyo3::buffer::{PyBuffer, ElementType};
   use numpy::{PyArray1, ToPyArray};
   
   #[pymethods]
   impl FastGraph {
       fn get_node_values_as_array<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
           // Collect values efficiently
           let values: Vec<f64> = self.nodes
               .values()
               .filter_map(|node| node.attributes.get("value"))
               .filter_map(|v| v.as_float())
               .collect();
           
           // Return as numpy array (zero-copy when possible)
           Ok(values.to_pyarray(py))
       }
   }

.. code-block:: python

   # Direct numpy array access
   import numpy as np
   
   g = Graph(backend='rust')
   # ... add nodes with numeric values ...
   
   # Get values as numpy array (efficient)
   values = g.get_node_values_as_array()
   result = np.mean(values)  # NumPy operations on Rust data

Debugging and Profiling
-----------------------

Debug Builds
~~~~~~~~~~~

Enable debug information for FFI debugging:

.. code-block:: toml

   # Cargo.toml
   [profile.dev]
   debug = true
   
   [profile.release]
   debug = true  # Keep debug info in release builds

FFI Call Tracing
~~~~~~~~~~~~~~~

Add tracing to understand FFI call patterns:

.. code-block:: rust

   use tracing::{info, instrument};
   
   #[pymethods]
   impl FastGraph {
       #[instrument(skip(self))]
       fn add_node(&mut self, id: String, attributes: &PyDict) -> PyResult<String> {
           info!("Adding node: {}", id);
           
           // Implementation
           let start = std::time::Instant::now();
           let result = self.add_node_impl(id, attributes);
           let duration = start.elapsed();
           
           info!("Node addition took: {:?}", duration);
           result
       }
   }

Performance Profiling
~~~~~~~~~~~~~~~~~~~~

Profile FFI performance:

.. code-block:: python

   import time
   import cProfile
   from gli import Graph
   
   def profile_ffi_calls():
       g = Graph(backend='rust')
       
       # Profile individual operations
       start = time.perf_counter()
       for i in range(10000):
           g.add_node(f"node_{i}", value=i)
       individual_time = time.perf_counter() - start
       
       # Profile batch operations
       g2 = Graph(backend='rust')
       start = time.perf_counter()
       with g2.batch_operations() as batch:
           for i in range(10000):
               batch.add_node(f"node_{i}", value=i)
       batch_time = time.perf_counter() - start
       
       print(f"Individual FFI calls: {individual_time:.3f}s")
       print(f"Batch FFI calls: {batch_time:.3f}s")
       print(f"FFI call overhead: {(individual_time - batch_time):.3f}s")
   
   # Run with profiler
   cProfile.run('profile_ffi_calls()')

Best Practices
-------------

FFI Design Guidelines
~~~~~~~~~~~~~~~~~~~

1. **Minimize FFI Calls**: Batch operations when possible
2. **Use Simple Types**: Avoid complex nested structures
3. **Handle Errors Properly**: Convert Rust errors to appropriate Python exceptions
4. **Manage Memory Carefully**: Be aware of reference counting
5. **Profile Performance**: Measure FFI overhead

.. code-block:: python

   # Good FFI usage
   g = Graph(backend='rust')
   
   # Batch operations to minimize calls
   with g.batch_operations() as batch:
       for item in large_dataset:
           batch.add_node(item['id'], 
                         value=item['value'],     # Simple type
                         category=item['category'] # Simple type
                         )
   
   # Efficient queries
   results = g.batch_query_nodes(node_ids)  # Single FFI call

Common Pitfalls
~~~~~~~~~~~~~~

1. **Too Many Small FFI Calls**: Each call has overhead
2. **Complex Object Conversion**: Nested structures are expensive
3. **Memory Leaks**: Improper reference management
4. **Exception Handling**: Not converting Rust errors properly
5. **Threading Issues**: GIL interactions

.. code-block:: python

   # Avoid: Many small FFI calls
   for node_id in node_ids:
       node_data = g.get_node(node_id)  # Many FFI calls
       process(node_data)
   
   # Better: Batch operation
   all_node_data = g.get_nodes_batch(node_ids)  # Single FFI call
   for node_data in all_node_data:
       process(node_data)

The FFI layer in GLI is designed to provide maximum performance while maintaining Python's ease of use. By understanding how the FFI works and following these best practices, you can get the most out of GLI's Rust backend.

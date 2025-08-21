FFI Layer Architecture
=====================

The Foreign Function Interface (FFI) layer is the bridge between Groggy's Rust core and Python API. Built on PyO3, it provides safe, efficient communication between the two languages while maintaining performance and memory safety.

PyO3 Integration
----------------

Core PyO3 Wrapper Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FFI layer wraps Rust types in Python-compatible objects using PyO3:

.. code-block:: rust

   use pyo3::prelude::*;

   #[pyclass]
   pub struct PyGraph {
       // Internal Rust graph
       inner: Arc<Mutex<GraphCore>>,
       
       // Python-specific state
       py_config: PyGraphConfig,
       
       // Cache for Python objects
       py_cache: RefCell<HashMap<String, PyObject>>,
   }

   #[pymethods]
   impl PyGraph {
       #[new]
       fn new(directed: bool) -> PyResult<Self> {
           let inner = Arc::new(Mutex::new(
               GraphCore::new(directed).map_err(PyGraphError::from)?
           ));
           
           Ok(PyGraph {
               inner,
               py_config: PyGraphConfig::default(),
               py_cache: RefCell::new(HashMap::new()),
           })
       }
       
       fn add_node(&mut self, node_id: &str, attrs: Option<PyDict>) -> PyResult<()> {
           let rust_attrs = self.convert_py_attrs(attrs)?;
           
           self.inner.lock().unwrap()
               .add_node(node_id.to_string(), rust_attrs)
               .map_err(PyGraphError::from)?;
           
           // Invalidate Python caches
           self.invalidate_caches();
           
           Ok(())
       }
   }

Memory Management Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

**Reference Counting Coordination**:

.. code-block:: rust

   use pyo3::types::PyAny;
   use std::sync::{Arc, Weak};

   pub struct SharedGraphData {
       // Rust reference counting
       rust_refs: Arc<GraphCore>,
       
       // Python reference tracking
       py_refs: RefCell<Vec<Weak<PyAny>>>,
       
       // Weak references for cleanup
       weak_refs: RefCell<Vec<Weak<GraphCore>>>,
   }

   impl SharedGraphData {
       pub fn new_python_ref(&self, py: Python, obj: &PyAny) {
           self.py_refs.borrow_mut().push(
               Arc::downgrade(&obj.into())
           );
       }
       
       pub fn cleanup_dead_refs(&self) {
           self.py_refs.borrow_mut().retain(|weak_ref| {
               weak_ref.upgrade().is_some()
           });
       }
   }

Data Type Conversion
--------------------

Rust to Python Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~

**AttrValue Conversion**:

.. code-block:: rust

   impl IntoPy<PyObject> for AttrValue {
       fn into_py(self, py: Python) -> PyObject {
           match self {
               AttrValue::Int8(v) => v.into_py(py),
               AttrValue::Int16(v) => v.into_py(py),
               AttrValue::Int32(v) => v.into_py(py),
               AttrValue::Int64(v) => v.into_py(py),
               AttrValue::Float32(v) => v.into_py(py),
               AttrValue::Float64(v) => v.into_py(py),
               AttrValue::String(v) => v.into_py(py),
               AttrValue::Bool(v) => v.into_py(py),
               AttrValue::Bytes(v) => PyBytes::new(py, &v).into(),
               AttrValue::Null => py.None(),
           }
       }
   }

   impl<'source> FromPyObject<'source> for AttrValue {
       fn extract(obj: &'source PyAny) -> PyResult<Self> {
           // Try different Python types in order
           if let Ok(val) = obj.extract::<i64>() {
               return Ok(AttrValue::Int64(val));
           }
           if let Ok(val) = obj.extract::<f64>() {
               return Ok(AttrValue::Float64(val));
           }
           if let Ok(val) = obj.extract::<String>() {
               return Ok(AttrValue::String(val));
           }
           if let Ok(val) = obj.extract::<bool>() {
               return Ok(AttrValue::Bool(val));
           }
           if let Ok(val) = obj.extract::<&PyBytes>() {
               return Ok(AttrValue::Bytes(val.as_bytes().to_vec()));
           }
           if obj.is_none() {
               return Ok(AttrValue::Null);
           }
           
           Err(PyTypeError::new_err(
               format!("Cannot convert {} to AttrValue", obj.get_type().name()?)
           ))
       }
   }

**Collection Conversions**:

.. code-block:: rust

   pub fn convert_py_dict_to_attrs(py_dict: Option<&PyDict>) -> PyResult<HashMap<String, AttrValue>> {
       let mut attrs = HashMap::new();
       
       if let Some(dict) = py_dict {
           for (key, value) in dict {
               let key_str = key.extract::<String>()?;
               let attr_value = value.extract::<AttrValue>()?;
               attrs.insert(key_str, attr_value);
           }
       }
       
       Ok(attrs)
   }

   pub fn convert_attrs_to_py_dict(py: Python, attrs: &HashMap<String, AttrValue>) -> PyResult<PyObject> {
       let dict = PyDict::new(py);
       
       for (key, value) in attrs {
           dict.set_item(key, value.clone().into_py(py))?;
       }
       
       Ok(dict.into())
   }

NumPy Integration
~~~~~~~~~~~~~~~~~

**Array Conversion**:

.. code-block:: rust

   use numpy::{PyArray1, PyArrayDyn, ToPyArray};
   use ndarray::Array1;

   impl PyGraphArray {
       pub fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
           let data = self.inner.lock().unwrap();
           
           match &data.dtype {
               DataType::Int64 => {
                   let values: Vec<i64> = data.values.iter()
                       .map(|v| v.as_int64().unwrap_or(0))
                       .collect();
                   
                   let array = Array1::from(values);
                   Ok(array.to_pyarray(py).to_object(py))
               },
               DataType::Float64 => {
                   let values: Vec<f64> = data.values.iter()
                       .map(|v| v.as_float64().unwrap_or(0.0))
                       .collect();
                   
                   let array = Array1::from(values);
                   Ok(array.to_pyarray(py).to_object(py))
               },
               _ => Err(PyTypeError::new_err(
                   "Cannot convert non-numeric array to NumPy"
               ))
           }
       }
       
       pub fn from_numpy(py_array: &PyArrayDyn<f64>) -> PyResult<Vec<AttrValue>> {
           let array = unsafe { py_array.as_array() };
           
           let values = array.iter()
               .map(|&x| AttrValue::Float64(x))
               .collect();
           
           Ok(values)
       }
   }

Error Handling
--------------

Error Type Mapping
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use pyo3::exceptions::{PyValueError, PyKeyError, PyMemoryError, PyRuntimeError};

   #[derive(Debug)]
   pub struct PyGraphError(GraphError);

   impl From<GraphError> for PyGraphError {
       fn from(err: GraphError) -> Self {
           PyGraphError(err)
       }
   }

   impl From<PyGraphError> for PyErr {
       fn from(err: PyGraphError) -> Self {
           match err.0 {
               GraphError::NodeNotFound { id } => {
                   PyKeyError::new_err(format!("Node '{}' not found", id))
               },
               GraphError::EdgeNotFound { source, target } => {
                   PyKeyError::new_err(format!("Edge ({}, {}) not found", source, target))
               },
               GraphError::InvalidOperation { reason } => {
                   PyValueError::new_err(reason)
               },
               GraphError::OutOfMemory { details } => {
                   PyMemoryError::new_err(details)
               },
               GraphError::Algorithm { source } => {
                   PyRuntimeError::new_err(format!("Algorithm error: {}", source))
               },
               GraphError::Io { source } => {
                   PyRuntimeError::new_err(format!("I/O error: {}", source))
               },
           }
       }
   }

Context Preservation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   impl PyGraph {
       fn safe_operation<F, R>(&self, operation_name: &str, f: F) -> PyResult<R>
       where
           F: FnOnce(&mut GraphCore) -> Result<R, GraphError>,
       {
           let start = std::time::Instant::now();
           
           let result = {
               let mut core = self.inner.lock().map_err(|_| {
                   PyRuntimeError::new_err("Failed to acquire graph lock")
               })?;
               
               f(&mut core).map_err(|e| {
                   // Add context about the operation
                   let context = format!(
                       "Operation '{}' failed after {:?}: {}",
                       operation_name,
                       start.elapsed(),
                       e
                   );
                   
                   PyRuntimeError::new_err(context)
               })
           };
           
           // Log performance if enabled
           if self.py_config.enable_profiling {
               self.log_operation(operation_name, start.elapsed());
           }
           
           result
       }
   }

GIL Management
--------------

Releasing the GIL
~~~~~~~~~~~~~~~~~

For CPU-intensive operations, the GIL should be released:

.. code-block:: rust

   #[pymethods]
   impl PyGraph {
       fn pagerank(&self, alpha: f64, max_iter: usize, tolerance: f64) -> PyResult<PyObject> {
           let graph = self.inner.clone();
           
           // Release GIL for computation
           let result = Python::with_gil(|py| -> PyResult<Vec<f64>> {
               py.allow_threads(|| {
                   let core = graph.lock().unwrap();
                   let algorithm = PageRank {
                       alpha,
                       max_iterations: max_iter,
                       tolerance,
                       parallel: true,
                   };
                   
                   algorithm.execute(&core, None)
                       .map_err(|e| PyRuntimeError::new_err(e.to_string()))
               })
           })?;
           
           // Convert result back to Python with GIL
           Python::with_gil(|py| {
               let py_dict = PyDict::new(py);
               
               for (i, &score) in result.iter().enumerate() {
                   let node_id = self.inner.lock().unwrap()
                       .get_node_id(NodeIndex(i))
                       .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                   
                   py_dict.set_item(node_id, score)?;
               }
               
               Ok(py_dict.into())
           })
       }
   }

Callback Handling
~~~~~~~~~~~~~~~~~

When Rust needs to call Python code:

.. code-block:: rust

   pub struct PyCallback {
       callback: PyObject,
   }

   impl PyCallback {
       pub fn call_with_node(&self, py: Python, node_id: &str, attrs: &HashMap<String, AttrValue>) 
                            -> PyResult<bool> {
           let py_attrs = convert_attrs_to_py_dict(py, attrs)?;
           
           let result = self.callback.call1(py, (node_id, py_attrs))?;
           
           result.extract::<bool>(py)
       }
   }

   #[pymethods]
   impl PyGraph {
       fn filter_nodes(&self, py: Python, predicate: PyObject) -> PyResult<Vec<String>> {
           let callback = PyCallback { callback: predicate };
           let graph = self.inner.lock().unwrap();
           
           let mut result = Vec::new();
           
           for node_index in graph.node_indices() {
               let node_id = graph.get_node_id(node_index)?;
               let attrs = graph.get_node_attributes(node_index)?;
               
               // Call Python predicate
               if callback.call_with_node(py, &node_id, &attrs)? {
                   result.push(node_id);
               }
           }
           
           Ok(result)
       }
   }

Storage View Integration
------------------------

Zero-Copy Data Sharing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[pyclass]
   pub struct PyGraphArray {
       // Shared reference to Rust data
       inner: Arc<RwLock<ArrayData>>,
       
       // Cached Python objects
       cached_numpy: RefCell<Option<PyObject>>,
       cached_pandas: RefCell<Option<PyObject>>,
   }

   #[pymethods]
   impl PyGraphArray {
       fn __getitem__(&self, py: Python, index: PyObject) -> PyResult<PyObject> {
           let data = self.inner.read().unwrap();
           
           if let Ok(idx) = index.extract::<usize>(py) {
               // Single index access
               if idx >= data.len() {
                   return Err(PyIndexError::new_err("Index out of range"));
               }
               
               Ok(data.values[idx].clone().into_py(py))
               
           } else if let Ok(slice) = index.extract::<PySlice>(py) {
               // Slice access
               let indices = slice.indices(data.len() as i64)?;
               let start = indices.start as usize;
               let stop = indices.stop as usize;
               let step = indices.step as usize;
               
               let mut result_values = Vec::new();
               for i in (start..stop).step_by(step) {
                   result_values.push(data.values[i].clone());
               }
               
               // Create new PyGraphArray for slice
               Ok(PyGraphArray::from_values(result_values)?.into_py(py))
               
           } else {
               Err(PyTypeError::new_err("Invalid index type"))
           }
       }
   }

Lazy Materialization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   impl PyGraphArray {
       fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
           // Check cache first
           if let Some(cached) = self.cached_pandas.borrow().as_ref() {
               return Ok(cached.clone());
           }
           
           // Materialize pandas Series
           let pandas_module = py.import("pandas")?;
           let data = self.inner.read().unwrap();
           
           let py_values: Vec<PyObject> = data.values.iter()
               .map(|v| v.clone().into_py(py))
               .collect();
           
           let series = pandas_module
               .getattr("Series")?
               .call1((py_values,))?;
           
           // Cache the result
           self.cached_pandas.borrow_mut().replace(series.into());
           
           Ok(self.cached_pandas.borrow().as_ref().unwrap().clone())
       }
   }

Performance Optimization
------------------------

Batch Operations
~~~~~~~~~~~~~~~

.. code-block:: rust

   #[pymethods]
   impl PyGraph {
       fn add_nodes(&mut self, py: Python, nodes: &PyList) -> PyResult<()> {
           // Pre-allocate space
           let count = nodes.len();
           let mut rust_nodes = Vec::with_capacity(count);
           
           // Convert all nodes first (fail fast)
           for item in nodes {
               let node_dict = item.extract::<&PyDict>()?;
               
               let id = node_dict.get_item("id")
                   .ok_or_else(|| PyKeyError::new_err("Missing 'id' field"))?
                   .extract::<String>()?;
               
               let attrs = if let Some(attrs_obj) = node_dict.get_item("attrs") {
                   convert_py_dict_to_attrs(Some(attrs_obj.extract::<&PyDict>()?)?)?
               } else {
                   HashMap::new()
               };
               
               rust_nodes.push((id, attrs));
           }
           
           // Batch insert (releases GIL)
           py.allow_threads(|| {
               let mut core = self.inner.lock().unwrap();
               core.add_nodes_batch(rust_nodes)
                   .map_err(|e| PyRuntimeError::new_err(e.to_string()))
           })?;
           
           // Invalidate caches
           self.invalidate_caches();
           
           Ok(())
       }
   }

Memory Pool Integration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[pyclass]
   pub struct PyMemoryPool {
       inner: Arc<MemoryPool>,
   }

   #[pymethods]
   impl PyMemoryPool {
       #[new]
       fn new(config: Option<PyDict>) -> PyResult<Self> {
           let pool_config = if let Some(config_dict) = config {
               PoolConfig::from_py_dict(config_dict)?
           } else {
               PoolConfig::default()
           };
           
           Ok(PyMemoryPool {
               inner: Arc::new(MemoryPool::new(pool_config))
           })
       }
       
       fn stats(&self, py: Python) -> PyResult<PyObject> {
           let stats = self.inner.get_stats();
           
           let py_dict = PyDict::new(py);
           py_dict.set_item("total_allocated", stats.total_allocated)?;
           py_dict.set_item("peak_usage", stats.peak_usage)?;
           py_dict.set_item("pool_usage", stats.pool_usage)?;
           
           Ok(py_dict.into())
       }
   }

Profiling Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[pyclass]
   pub struct PyProfiler {
       profiler: Arc<Mutex<PerformanceMonitor>>,
   }

   #[pymethods]
   impl PyProfiler {
       fn profile_operation(&self, py: Python, operation: PyObject) -> PyResult<PyObject> {
           let start = std::time::Instant::now();
           
           // Execute the operation
           let result = operation.call0(py)?;
           
           let duration = start.elapsed();
           
           // Record timing
           self.profiler.lock().unwrap()
               .record_operation("python_operation", duration);
           
           // Return both result and timing
           let timing_info = PyDict::new(py);
           timing_info.set_item("duration_ms", duration.as_millis())?;
           timing_info.set_item("result", result)?;
           
           Ok(timing_info.into())
       }
   }

Thread Safety
--------------

Safe Concurrency Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use std::sync::RwLock;
   use parking_lot::Mutex;  // Faster than std::sync::Mutex

   #[pyclass]
   pub struct ThreadSafeGraph {
       // Read-heavy data with RwLock
       topology: Arc<RwLock<AdjacencyStructure>>,
       
       // Write-heavy data with Mutex
       attributes: Arc<Mutex<AttributeStorage>>,
       
       // Atomic counters
       node_count: Arc<AtomicUsize>,
       edge_count: Arc<AtomicUsize>,
   }

   #[pymethods]
   impl ThreadSafeGraph {
       fn parallel_pagerank(&self, py: Python) -> PyResult<PyObject> {
           let topology = self.topology.clone();
           let node_count = self.node_count.load(Ordering::Relaxed);
           
           py.allow_threads(|| {
               // Multiple threads can read topology simultaneously
               let topo = topology.read().unwrap();
               
               // Run parallel PageRank
               parallel_pagerank_impl(&topo, node_count)
                   .map_err(|e| PyRuntimeError::new_err(e.to_string()))
           })
       }
   }

Future Improvements
-------------------

Async/Await Support
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   use pyo3_asyncio::tokio::future_into_py;

   #[pymethods]
   impl PyGraph {
       fn async_pagerank<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
           let graph = self.inner.clone();
           
           future_into_py(py, async move {
               let result = tokio::task::spawn_blocking(move || {
                   let core = graph.lock().unwrap();
                   // Run PageRank in background thread
                   run_pagerank(&core)
               }).await?;
               
               Ok(Python::with_gil(|py| result.into_py(py)))
           })
       }
   }

WebAssembly Support
~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   #[cfg(target_arch = "wasm32")]
   use wasm_bindgen::prelude::*;

   #[cfg(target_arch = "wasm32")]
   #[wasm_bindgen]
   pub struct WasmGraph {
       inner: GraphCore,
   }

   #[cfg(target_arch = "wasm32")]
   #[wasm_bindgen]
   impl WasmGraph {
       #[wasm_bindgen(constructor)]
       pub fn new(directed: bool) -> WasmGraph {
           WasmGraph {
               inner: GraphCore::new(directed).unwrap()
           }
       }
       
       #[wasm_bindgen]
       pub fn add_node(&mut self, id: &str, attrs: &JsValue) -> Result<(), JsValue> {
           // Convert JsValue to Rust types
           let rust_attrs = convert_js_to_attrs(attrs)?;
           
           self.inner.add_node(id.to_string(), rust_attrs)
               .map_err(|e| JsValue::from_str(&e.to_string()))?;
           
           Ok(())
       }
   }

The FFI layer provides a robust, safe, and efficient bridge between Rust and Python, enabling Groggy to deliver high performance while maintaining Python's ease of use. Through careful attention to memory management, error handling, and performance optimization, the FFI layer ensures that the benefits of Rust's performance are fully realized in the Python API.
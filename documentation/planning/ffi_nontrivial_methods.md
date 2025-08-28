# Groggy FFI – Non‑Trivial Methods Inventory

_Generated from `/mnt/data/rust_ffi_src.txt` – includes every FFI function with substantive logic beyond simple forwarding or field access._


## ffi/api/graph.rs

### adjacency_matrix_to_py_graph_matrix

**Signature:**

```rust
fn adjacency_matrix_to_py_graph_matrix(
    _py: Python,
    _matrix: groggy::AdjacencyMatrix,
) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 155.

**Implementation:**

```rust

    // TODO: Implement adjacency matrix to GraphMatrix conversion in Phase 2
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "AdjacencyMatrix to GraphMatrix conversion temporarily disabled during Phase 2 unification",
    ))
```

---

### adjacency_matrix_to_py_object

**Signature:**

```rust
fn adjacency_matrix_to_py_object(
    _py: Python,
    _matrix: groggy::AdjacencyMatrixResult,
) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 144.

**Implementation:**

```rust

    // TODO: Implement adjacency matrix to Python object conversion in Phase 2
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "Adjacency matrix functionality temporarily disabled during Phase 2 unification",
    ))
```

---

### new

**Attributes:**

```rust
#[new]
    #[pyo3(signature = (directed = false, _config = None))]
```

**Signature:**

```rust
fn new(directed: bool, _config: Option<&PyDict>) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 145.

**Implementation:**

```rust

        // Create graph with specified directionality
        let rust_graph = if directed {
            RustGraph::new_directed()
        } else {
            RustGraph::new_undirected()
        };
        Ok(Self { inner: Rc::new(RefCell::new(rust_graph)) })
```

---

### add_node

**Attributes:**

```rust
#[pyo3(signature = (**kwargs))]
```

**Signature:**

```rust
fn add_node(&mut self, kwargs: Option<&PyDict>) -> PyResult<NodeId>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 14, token-chars: 341.

**Implementation:**

```rust

        let node_id = self.inner.borrow_mut().add_node();

        // Fast path: if no kwargs, just return the node_id
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                // Only do attribute setting if we actually have attributes
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;

                    self.inner
.borrow_mut()
                        .set_node_attr(node_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }

        Ok(node_id)
```

---

### add_nodes

**Attributes:**

```rust
#[pyo3(signature = (data, uid_key = None))]
```

**Signature:**

```rust
fn add_nodes(&mut self, data: &PyAny, uid_key: Option<String>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 56, token-chars: 1407.

**Implementation:**

```rust

        // Fast path optimization: Check for integer first (most common case)
        if let Ok(count) = data.extract::<usize>() {
            // Old API: add_nodes(5) -> [0, 1, 2, 3, 4] - fastest path
            let node_ids = self.inner.borrow_mut().add_nodes(count);
            return Python::with_gil(|py| Ok(node_ids.to_object(py)));
        }

        // Only use Python::with_gil for complex operations
        Python::with_gil(|py| {
            if let Ok(node_data_list) = data.extract::<Vec<&PyDict>>() {
                // New API: add_nodes([{"id": "alice", "age": 30}, ...], id_key="id")
                let mut id_mapping = std::collections::HashMap::new();

                // Create all nodes first
                let node_ids = self.inner.borrow_mut().add_nodes(node_data_list.len());

                // OPTIMIZATION: Collect attributes by name for bulk operations instead of individual calls
                // This changes complexity from O(N × A × log N) to O(N × A)
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(NodeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

                // Process each node's data
                for (i, node_dict) in node_data_list.iter().enumerate() {
                    let node_id = node_ids[i];

                    // Extract the ID if id_key is provided
                    if let Some(ref key) = uid_key {
                        match node_dict.get_item(key) {
                            Ok(Some(id_value)) => {
                                if let Ok(user_id) = id_value.extract::<String>() {
                                    id_mapping.insert(user_id, node_id);
                                }
                            }
                            Ok(None) => {
                                return Err(PyErr::new::<PyKeyError, _>(format!(
                                    "Missing key: {}",
                                    key
                                )));
                            }
                            Err(e) => return Err(e),
                        }
                    }

                    // Collect all attributes for bulk setting
                    for (attr_key, attr_value) in node_dict.iter() {
                        let attr_name: String = attr_key.extract()?;
                        let attr_val = python_value_to_attr_value(attr_value)?;

                        // Store all attributes including the id_key for later uid_key lookups
                        attrs_by_name
                            .entry(attr_name)
                            .or_insert_with(Vec::new)
                            .push((node_id, attr_val));
                    }
                }

                // OPTIMIZATION: Use bulk attribute setting - O(A) operations instead of O(N × A)
                if !attrs_by_name.is_empty() {
                    self.inner
.borrow_mut()
                        .set_node_attrs(attrs_by_name)
                        .map_err(graph_error_to_py_err)?;
                }

                // Return the mapping if id_key was provided, otherwise return node IDs
                if uid_key.is_some() {
                    Ok(id_mapping.to_object(py))
                } else {
                    Ok(node_ids.to_object(py))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "add_nodes expects either an integer count or a list of dictionaries",
                ))
            }
        })
```

---

### resolve_string_id_to_node

**Signature:**

```rust
fn resolve_string_id_to_node(&self, string_id: &str, uid_key: &str) -> PyResult<NodeId>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 478.

**Implementation:**

```rust

        let node_ids = self.inner.borrow_mut().node_ids();

        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.borrow_mut().get_node_attr(node_id, &uid_key.to_string()) {
                match attr_value {
                    RustAttrValue::Text(s) => {
                        if s == string_id {
                            return Ok(node_id);
                        }
                    }
                    RustAttrValue::CompactText(s) => {
                        if s.as_str() == string_id {
                            return Ok(node_id);
                        }
                    }
                    _ => continue, // Skip non-text attributes
                }
            }
        }

        Err(PyErr::new::<PyKeyError, _>(format!(
            "No node found with {}='{}'",
            uid_key, string_id
        )))
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 5, token-chars: 72.

**Implementation:**

```rust

        format!(
            "Graph(nodes={}, edges={})",
            self.node_count(),
            self.edge_count()
        )
```

---

### density

**Signature:**

```rust
fn density(&self) -> f64
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 365.

**Implementation:**

```rust

        let num_nodes = self.inner.borrow_mut().node_ids().len();
        let num_edges = self.inner.borrow_mut().edge_ids().len();

        if num_nodes <= 1 {
            return 0.0;
        }

        // Calculate maximum possible edges based on graph type
        let max_possible_edges = if self.inner.borrow_mut().is_directed() {
            // For directed graphs: n(n-1)
            num_nodes * (num_nodes - 1)
        } else {
            // For undirected graphs: n(n-1)/2
            (num_nodes * (num_nodes - 1)) / 2
        };

        if max_possible_edges > 0 {
            num_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        }
```

---

### set_node_attribute

**Signature:**

```rust
pub fn set_node_attribute(
        &mut self,
        node: NodeId,
        attr: AttrName,
        value: &PyAttrValue,
    ) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 101.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .set_node_attr(node, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
```

---

### set_node_attr

**Signature:**

```rust
pub fn set_node_attr(&mut self, node: NodeId, attr: AttrName, value: &PyAny) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 144.

**Implementation:**

```rust

        let attr_value = python_value_to_attr_value(value)?;
        self.inner
.borrow_mut()
            .set_node_attr(node, attr, attr_value)
            .map_err(graph_error_to_py_err)
```

---

### set_edge_attribute

**Signature:**

```rust
pub fn set_edge_attribute(
        &mut self,
        edge: EdgeId,
        attr: AttrName,
        value: &PyAttrValue,
    ) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 101.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .set_edge_attr(edge, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
```

---

### set_edge_attr

**Signature:**

```rust
pub fn set_edge_attr(&mut self, edge: EdgeId, attr: AttrName, value: &PyAny) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 144.

**Implementation:**

```rust

        let attr_value = python_value_to_attr_value(value)?;
        self.inner
.borrow_mut()
            .set_edge_attr(edge, attr, attr_value)
            .map_err(graph_error_to_py_err)
```

---

### get_node_attribute

**Signature:**

```rust
pub fn get_node_attribute(
        &self,
        node: NodeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 178.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_edge_attribute

**Signature:**

```rust
pub fn get_edge_attribute(
        &self,
        edge: EdgeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 178.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_edge_attr(edge, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_edge_attributes

**Signature:**

```rust
fn get_edge_attributes(&self, edge: EdgeId, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 276.

**Implementation:**

```rust

        let attrs = self
            .inner
.borrow_mut()
            .get_edge_attrs(edge)
            .map_err(graph_error_to_py_err)?;

        // Convert HashMap to Python dict
        let dict = PyDict::new(py);
        for (attr_name, attr_value) in attrs {
            let py_value = Py::new(py, PyAttrValue { inner: attr_value })?;
            dict.set_item(attr_name, py_value)?;
        }
        Ok(dict.to_object(py))
```

---

### edge_endpoints

**Signature:**

```rust
fn edge_endpoints(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId)>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 75.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .edge_endpoints(edge)
            .map_err(graph_error_to_py_err)
```

---

### node_ids

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 316.

**Implementation:**

```rust

        let node_ids = self.inner.borrow_mut().node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### edge_ids

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 316.

**Implementation:**

```rust

        let edge_ids = self.inner.borrow_mut().edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### set_node_attributes

**Signature:**

```rust
fn set_node_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 65, token-chars: 1946.

**Implementation:**

```rust

        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};

        // HYPER-OPTIMIZED bulk API - minimize PyO3 overhead and allocations
        let mut attrs_values = std::collections::HashMap::with_capacity(attrs_dict.len());

        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;

            // OPTIMIZATION: Extract all fields at once to reduce PyO3 calls
            let (nodes, values_obj, value_type): (Vec<NodeId>, &pyo3::PyAny, String) = {
                let nodes_item = data_dict
                    .get_item("nodes")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'nodes' key"))?;
                let values_item = data_dict
                    .get_item("values")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'values' key"))?;
                let type_item = data_dict
                    .get_item("value_type")?
                    .ok_or_else(|| PyErr::new::<PyKeyError, _>("Missing 'value_type' key"))?;

                (nodes_item.extract()?, values_item, type_item.extract()?)
            };

            let len = nodes.len();

            // OPTIMIZATION: Pre-allocate result vector and use direct indexing
            let mut pairs = Vec::with_capacity(len);

            // OPTIMIZATION: Match on str slice to avoid repeated string comparisons
            match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    // OPTIMIZATION: Direct loop instead of iterator chain
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Text(values[i].clone())));
                    }
                }
                "int" => {
                    let values: Vec<i64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Int(values[i])));
                    }
                }
                "float" => {
                    let values: Vec<f64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Float(values[i] as f32)));
                    }
                }
                "bool" => {
                    let values: Vec<bool> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }

                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Bool(values[i])));
                    }
                }
                _ => return Err(PyErr::new::<PyValueError, _>("Unsupported type")),
            };

            attrs_values.insert(attr, pairs);
        }

        self.inner
.borrow_mut()
            .set_node_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
```

---

### set_edge_attributes

**Signature:**

```rust
fn set_edge_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 118, token-chars: 2739.

**Implementation:**

```rust

        use groggy::AttrValue as RustAttrValue;
        use pyo3::exceptions::{PyKeyError, PyValueError};

        // New efficient columnar API for edges - zero PyAttrValue objects created!
        let mut attrs_values = std::collections::HashMap::new();

        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;

            // Extract components in bulk using the same pattern as node attributes
            let edges: Vec<EdgeId> = if let Ok(Some(item)) = data_dict.get_item("edges") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>(
                    "Missing 'edges' key in attribute data",
                ));
            };
            let value_type: String = if let Ok(Some(item)) = data_dict.get_item("value_type") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>(
                    "Missing 'value_type' key in attribute data",
                ));
            };

            // Batch convert based on known type - no individual type detection!
            let pairs = match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Text(val)))
                        .collect()
                }
                "int" => {
                    let values: Vec<i64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Int(val)))
                        .collect()
                }
                "float" => {
                    let values: Vec<f64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Float(val as f32)))
                        .collect()
                }
                "bool" => {
                    let values: Vec<bool> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>(
                            "Missing 'values' key in attribute data",
                        ));
                    };

                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Mismatched lengths: {} edges vs {} values",
                            edges.len(),
                            values.len()
                        )));
                    }

                    edges
                        .into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Bool(val)))
                        .collect()
                }
                _ => {
                    return Err(PyErr::new::<PyValueError, _>(format!(
                        "Unsupported value_type: {}",
                        value_type
                    )));
                }
            };

            attrs_values.insert(attr, pairs);
        }

        self.inner
.borrow_mut()
            .set_edge_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
```

---

### add_edges

**Signature:**

```rust
fn add_edges(
        &mut self,
        edges: &PyAny,
        node_mapping: Option<std::collections::HashMap<String, NodeId>>,
        uid_key: Option<String>,
    ) -> PyResult<Vec<EdgeId>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 104, token-chars: 2850.

**Implementation:**

```rust

        // Format 1: List of (source, target) tuples - most common case for benchmarks
        if let Ok(edge_pairs) = edges.extract::<Vec<(NodeId, NodeId)>>() {
            return Ok(self.inner.borrow_mut().add_edges(&edge_pairs));
        }
        // Format 2: List of (source, target, attrs_dict) tuples
        else if let Ok(edge_tuples) = edges.extract::<Vec<(&PyAny, &PyAny, Option<&PyDict>)>>() {
            let mut edge_ids = Vec::new();
            let mut edges_with_attrs = Vec::new();

            // First pass: create all edges and collect attribute data
            for (src_any, tgt_any, attrs_opt) in edge_tuples {
                let source: NodeId = src_any.extract()?;
                let target: NodeId = tgt_any.extract()?;

                let edge_id = self
                    .inner
.borrow_mut()
                    .add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);

                // Store edge attributes for bulk processing
                if let Some(attrs) = attrs_opt {
                    edges_with_attrs.push((edge_id, attrs));
                }
            }

            // OPTIMIZATION: Use bulk attribute setting instead of individual calls
            if !edges_with_attrs.is_empty() {
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(EdgeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

                for (edge_id, attrs) in edges_with_attrs {
                    for (key, value) in attrs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;

                        attrs_by_name
                            .entry(attr_name)
                            .or_insert_with(Vec::new)
                            .push((edge_id, attr_value));
                    }
                }

                self.inner
.borrow_mut()
                    .set_edge_attrs(attrs_by_name)
                    .map_err(graph_error_to_py_err)?;
            }

            return Ok(edge_ids);
        }
        // Format 3: List of dictionaries with node mapping
        else if let Ok(edge_dicts) = edges.extract::<Vec<&PyDict>>() {
            let mut edge_ids = Vec::new();
            let mut edges_with_attrs = Vec::new();

            // First pass: create all edges and collect attribute data
            for edge_dict in edge_dicts {
                // Extract source and target
                let source = if let Some(mapping) = &node_mapping {
                    let source_str: String = edge_dict.get_item("source")?.unwrap().extract()?;
                    *mapping.get(&source_str).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Node {} not found in mapping",
                            source_str
                        ))
                    })?
                } else {
                    edge_dict.get_item("source")?.unwrap().extract()?
                };

                let target = if let Some(mapping) = &node_mapping {
                    let target_str: String = edge_dict.get_item("target")?.unwrap().extract()?;
                    *mapping.get(&target_str).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(format!(
                            "Node {} not found in mapping",
                            target_str
                        ))
                    })?
                } else {
                    edge_dict.get_item("target")?.unwrap().extract()?
                };

                // Add the edge
                let edge_id = self
                    .inner
.borrow_mut()
                    .add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                edge_ids.push(edge_id);

                // Store edge and its attributes for bulk processing
                edges_with_attrs.push((edge_id, edge_dict));
            }

            // OPTIMIZATION: Use bulk attribute setting instead of individual calls
            if !edges_with_attrs.is_empty() {
                let mut attrs_by_name: std::collections::HashMap<
                    String,
                    Vec<(EdgeId, RustAttrValue)>,
                > = std::collections::HashMap::new();

                for (edge_id, edge_dict) in edges_with_attrs {
                    for (key, value) in edge_dict.iter() {
                        let key_str: String = key.extract()?;
                        if key_str != "source" && key_str != "target" {
                            let attr_value = python_value_to_attr_value(value)?;
                            attrs_by_name
                                .entry(key_str)
                                .or_insert_with(Vec::new)
                                .push((edge_id, attr_value));
                        }
                    }
                }

                if !attrs_by_name.is_empty() {
                    self.inner
.borrow_mut()
                        .set_edge_attrs(attrs_by_name)
                        .map_err(graph_error_to_py_err)?;
                }
            }

            return Ok(edge_ids);
        }

        // If none of the formats matched, return error
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "add_edges expects a list of (source, target) tuples, (source, target, attrs) tuples, or dictionaries with node_mapping"
        ))
```

---

### filter_nodes

**Signature:**

```rust
fn filter_nodes(slf: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 44, token-chars: 1550.

**Implementation:**

```rust

        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) = Self::validate_node_filter_attributes(&*slf.inner.borrow(), &node_filter) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes.", attr_name)
            ));
        }

        let start = std::time::Instant::now();
        let filtered_nodes = slf
            .inner
.borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        let _elapsed = start.elapsed();

        let start = std::time::Instant::now();
        // O(k) Calculate induced edges using optimized core subgraph method
        use std::collections::HashSet;
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = slf.inner.borrow().get_columnar_topology();
        let mut induced_edges = Vec::new();

        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // O(1) HashSet lookups instead of O(n) Vec::contains
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }

        let _elapsed = start.elapsed();

        Ok(PySubgraph::new(
            filtered_nodes,
            induced_edges,
            "filtered_nodes".to_string(),
            Some(slf.into()),
        ))
```

---

### filter_edges

**Signature:**

```rust
fn filter_edges(slf: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 37, token-chars: 1241.

**Implementation:**

```rust

        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_edge_query")?;
            let parsed_filter: PyEdgeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) = Self::validate_edge_filter_attributes(&*slf.inner.borrow(), &edge_filter) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any edges in the graph. Use graph.edges.table().columns to see available attributes.", attr_name)
            ));
        }

        let filtered_edges = slf
            .inner
.borrow_mut()
            .find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;

        // Calculate nodes that are connected by the filtered edges
        use std::collections::HashSet;
        let mut nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = slf.inner.borrow().edge_endpoints(edge_id) {
                nodes.insert(source);
                nodes.insert(target);
            }
        }

        let node_vec: Vec<NodeId> = nodes.into_iter().collect();

        Ok(PySubgraph::new(
            node_vec,
            filtered_edges,
            "filtered_edges".to_string(),
            Some(slf.into()),
        ))
```

---

### analytics

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn analytics(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<Py<crate::ffi::api::graph_analytics::PyGraphAnalytics>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 189.

**Implementation:**

```rust

        use crate::ffi::api::graph_analytics::PyGraphAnalytics;
        let graph_ref: Py<PyGraph> = slf.into_py(py).extract(py)?;
        let analytics = PyGraphAnalytics { graph: graph_ref };
        Py::new(py, analytics)
```

---

### group_nodes_by_attribute

**Signature:**

```rust
pub fn group_nodes_by_attribute(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 544.

**Implementation:**

```rust

        let results = self
            .inner
.borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
```

---

### remove_nodes

**Signature:**

```rust
fn remove_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 75.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .remove_nodes(&nodes)
            .map_err(graph_error_to_py_err)
```

---

### remove_edges

**Signature:**

```rust
fn remove_edges(&mut self, edges: Vec<EdgeId>) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 75.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .remove_edges(&edges)
            .map_err(graph_error_to_py_err)
```

---

### add_edge

**Attributes:**

```rust
#[pyo3(signature = (source, target, uid_key = None, **kwargs))]
```

**Signature:**

```rust
fn add_edge(
        &mut self,
        _py: Python,
        source: &PyAny,
        target: &PyAny,
        uid_key: Option<String>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<EdgeId>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 48, token-chars: 1179.

**Implementation:**

```rust

        // Try to extract as NodeId first (most common case)
        let source_id = if let Ok(node_id) = source.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = source.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Source must be NodeId or string",
            ));
        };

        let target_id = if let Ok(node_id) = target.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = target.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Target must be NodeId or string",
            ));
        };

        // Add the edge
        let edge_id = self
            .inner
.borrow_mut()
            .add_edge(source_id, target_id)
            .map_err(graph_error_to_py_err)?;

        // Set attributes if provided
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    self.inner
.borrow_mut()
                        .set_edge_attr(edge_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }

        Ok(edge_id)
```

---

### shortest_path

**Attributes:**

```rust
#[pyo3(signature = (source, target, weight_attribute = None, inplace = None, attr_name = None))]
```

**Signature:**

```rust
fn shortest_path(
        slf: PyRef<Self>,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Option<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 153.

**Implementation:**

```rust

        // Delegate to analytics module which has proper graph reference handling
        let analytics = PyGraph::analytics(slf, py)?;
        let result = analytics.borrow(py).shortest_path(
            py,
            source,
            target,
            weight_attribute,
            inplace,
            attr_name,
        );
        result
```

---

### aggregate

**Attributes:**

```rust
#[pyo3(signature = (attribute, operation, target = None, node_ids = None))]
```

**Signature:**

```rust
fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 30, token-chars: 716.

**Implementation:**

```rust

        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                // TODO: Core doesn't have aggregate_nodes_custom, implement if needed
                let result = self.inner.borrow_mut().aggregate_node_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            "edges" => {
                let result = self.inner.borrow_mut().aggregate_edge_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            _ => Err(PyErr::new::<PyTypeError, _>(
                "Target must be 'nodes' or 'edges'",
            )),
        }
```

---

### commit

**Signature:**

```rust
fn commit(&mut self, message: String, author: String) -> PyResult<StateId>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 78.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .commit(message, author)
            .map_err(graph_error_to_py_err)
```

---

### create_branch

**Signature:**

```rust
fn create_branch(&mut self, branch_name: String) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 81.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .create_branch(branch_name)
            .map_err(graph_error_to_py_err)
```

---

### checkout_branch

**Signature:**

```rust
fn checkout_branch(&mut self, branch_name: String) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 83.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .checkout_branch(branch_name)
            .map_err(graph_error_to_py_err)
```

---

### branches

**Signature:**

```rust
fn branches(&self) -> Vec<PyBranchInfo>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 111.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo::new(branch_info))
            .collect()
```

---

### commit_history

**Signature:**

```rust
fn commit_history(&self) -> Vec<PyCommit>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 121.

**Implementation:**

```rust

        // Delegate to core history system
        self.inner
.borrow_mut()
            .commit_history()
            .into_iter()
            .map(|commit_info| PyCommit::from_commit_info(commit_info))
            .collect()
```

---

### historical_view

**Signature:**

```rust
fn historical_view(&self, commit_id: StateId) -> PyResult<PyHistoricalView>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 166.

**Implementation:**

```rust

        // Delegate to core history system
        match self.inner.borrow_mut().view_at_commit(commit_id) {
            Ok(_historical_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_node_mapping

**Attributes:**

```rust
#[pyo3(signature = (uid_key, return_inverse = false))]
```

**Signature:**

```rust
fn get_node_mapping(
        &self,
        py: Python,
        uid_key: String,
        return_inverse: bool,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 21, token-chars: 672.

**Implementation:**

```rust

        let dict = PyDict::new(py);

        // Delegate to core for node IDs and attributes
        let node_ids = self.inner.borrow_mut().node_ids();

        // Use core attribute access for each node
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.borrow_mut().get_node_attr(node_id, &uid_key) {
                // Convert attribute value to appropriate Python type
                let attr_value_py = match attr_value {
                    RustAttrValue::Text(s) => s.to_object(py),
                    RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                    RustAttrValue::Int(i) => i.to_object(py),
                    RustAttrValue::SmallInt(i) => i.to_object(py),
                    RustAttrValue::Float(f) => f.to_object(py),
                    RustAttrValue::Bool(b) => b.to_object(py),
                    _ => continue, // Skip unsupported types
                };

                if return_inverse {
                    // Return node_id -> attribute_value mapping
                    dict.set_item(node_id, attr_value_py)?;
                } else {
                    // Return attribute_value -> node_id mapping (default behavior)
                    dict.set_item(attr_value_py, node_id)?;
                }
            }
        }

        Ok(dict.to_object(py))
```

---

### adjacency_matrix

**Signature:**

```rust
fn adjacency_matrix(&mut self, _py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 130.

**Implementation:**

```rust

        // TODO: Implement adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Adjacency matrix temporarily disabled during Phase 2 unification",
        ))
```

---

### adjacency

**Signature:**

```rust
fn adjacency(&mut self, py: Python) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 391.

**Implementation:**

```rust

        use crate::ffi::core::matrix::PyGraphMatrix;

        // Generate dense adjacency matrix using the public API method
        let adjacency_matrix = self.inner.borrow_mut().dense_adjacency_matrix().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate adjacency matrix: {:?}", e))
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
```

---

### weighted_adjacency_matrix

**Signature:**

```rust
fn weighted_adjacency_matrix(
        &mut self,
        py: Python,
        weight_attr: &str,
    ) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 14, token-chars: 413.

**Implementation:**

```rust

        use crate::ffi::core::matrix::PyGraphMatrix;

        // Generate weighted adjacency matrix using the public API method
        let adjacency_matrix = self
            .inner
.borrow_mut()
            .weighted_adjacency_matrix(weight_attr)
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to generate weighted adjacency matrix: {:?}",
                    e
                ))
            })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
```

---

### dense_adjacency_matrix

**Signature:**

```rust
fn dense_adjacency_matrix(&mut self, _py: Python) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 136.

**Implementation:**

```rust

        // TODO: Implement dense adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Dense adjacency matrix temporarily disabled during Phase 2 unification",
        ))
```

---

### sparse_adjacency_matrix

**Signature:**

```rust
fn sparse_adjacency_matrix(&mut self, _py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 137.

**Implementation:**

```rust

        // TODO: Implement sparse adjacency matrix in Phase 2 (will return PyGraphSparseMatrix)
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Sparse adjacency matrix temporarily disabled during Phase 2 unification",
        ))
```

---

### laplacian_matrix

**Signature:**

```rust
fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 446.

**Implementation:**

```rust

        use crate::ffi::core::matrix::PyGraphMatrix;

        let is_normalized = normalized.unwrap_or(false);

        // Generate Laplacian matrix using the public API method
        let laplacian_matrix = self.inner.borrow_mut().laplacian_matrix(is_normalized).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate Laplacian matrix: {:?}", e))
        })?;

        // Convert AdjacencyMatrix to GraphMatrix
        let graph_matrix = self.adjacency_matrix_to_graph_matrix(laplacian_matrix)?;

        // Wrap in PyGraphMatrix
        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
```

---

### transition_matrix

**Signature:**

```rust
fn transition_matrix(
        &mut self,
        py: Python,
        k: u32,
        weight_attr: Option<&str>,
    ) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 17, token-chars: 653.

**Implementation:**

```rust

        use crate::ffi::core::matrix::PyGraphMatrix;

        // Get adjacency matrix (weighted or unweighted)
        let adjacency_matrix = if let Some(attr) = weight_attr {
            self.inner.borrow_mut().weighted_adjacency_matrix(attr)
        } else {
            self.inner.borrow_mut().dense_adjacency_matrix()
        }
        .map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to generate adjacency matrix: {:?}", e))
        })?;

        // Convert to GraphMatrix for operations
        let mut graph_matrix = self.adjacency_matrix_to_graph_matrix(adjacency_matrix)?;

        // Normalize rows to get transition probabilities
        // TODO: This would benefit from proper row normalization in GraphMatrix
        // For now, just return the k-th power
        for _ in 1..k {
            graph_matrix = graph_matrix.multiply(&graph_matrix).map_err(|e| {
                PyRuntimeError::new_err(format!("Matrix power computation failed: {:?}", e))
            })?;
        }

        let py_graph_matrix = PyGraphMatrix::from_graph_matrix(graph_matrix);
        Ok(Py::new(py, py_graph_matrix)?)
```

---

### subgraph_adjacency_matrix

**Signature:**

```rust
fn subgraph_adjacency_matrix(
        &mut self,
        _py: Python,
        _node_ids: Vec<NodeId>,
    ) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 139.

**Implementation:**

```rust

        // TODO: Implement subgraph adjacency matrix in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subgraph adjacency matrix temporarily disabled during Phase 2 unification",
        ))
```

---

### neighbors

**Attributes:**

```rust
#[pyo3(signature = (nodes = None))]
```

**Signature:**

```rust
fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 54, token-chars: 1611.

**Implementation:**

```rust

        match nodes {
            // Single node case: neighbors(node_id) -> Vec<NodeId> (backward compatibility)
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let neighbors = self.inner.borrow_mut().neighbors(node).map_err(graph_error_to_py_err)?;
                Ok(neighbors.to_object(py))
            }
            // List of nodes case: neighbors([node1, node2, ...]) -> GraphArray
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut neighbor_arrays = Vec::new();

                for node_id in node_ids {
                    match self.inner.borrow_mut().neighbors(node_id) {
                        Ok(neighbors) => {
                            // Convert Vec<NodeId> to comma-separated string representation
                            let neighbor_str = neighbors
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<String>>()
                                .join(",");
                            neighbor_arrays.push(groggy::AttrValue::Text(neighbor_str));
                        }
                        Err(_) => {
                            // For non-existent nodes, return empty string
                            neighbor_arrays.push(groggy::AttrValue::Text(String::new()));
                        }
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(neighbor_arrays);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // All nodes case: neighbors() -> GraphArray
            None => {
                let all_nodes = self.inner.borrow_mut().node_ids();
                let mut neighbor_arrays = Vec::new();

                for node_id in all_nodes {
                    match self.inner.borrow_mut().neighbors(node_id) {
                        Ok(neighbors) => {
                            // Convert Vec<NodeId> to comma-separated string representation
                            let neighbor_str = neighbors
                                .iter()
                                .map(|id| id.to_string())
                                .collect::<Vec<String>>()
                                .join(",");
                            neighbor_arrays.push(groggy::AttrValue::Text(neighbor_str));
                        }
                        Err(_) => {
                            // For any issues, return empty string
                            neighbor_arrays.push(groggy::AttrValue::Text(String::new()));
                        }
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(neighbor_arrays);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // Invalid argument case
            Some(_) => Err(PyTypeError::new_err(
                "neighbors() argument must be a NodeId or list of NodeIds"
            ))
        }
```

---

### degree

**Attributes:**

```rust
#[pyo3(signature = (nodes = None))]
```

**Signature:**

```rust
fn degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 39, token-chars: 1237.

**Implementation:**

```rust

        match nodes {
            // Single node case: degree(node_id) -> int (keep as int for backward compatibility)
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let deg = self.inner.borrow_mut().degree(node).map_err(graph_error_to_py_err)?;
                Ok(deg.to_object(py))
            }
            // List of nodes case: degree([node1, node2, ...]) -> GraphArray
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut degrees = Vec::new();

                for node_id in node_ids {
                    match self.inner.borrow_mut().degree(node_id) {
                        Ok(deg) => {
                            degrees.push(groggy::AttrValue::Int(deg as i64));
                        }
                        Err(_) => {
                            // Skip nodes that don't exist rather than failing
                            continue;
                        }
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // All nodes case: degree() -> GraphArray
            None => {
                let all_nodes = self.inner.borrow_mut().node_ids();
                let mut degrees = Vec::new();

                for node_id in all_nodes {
                    if let Ok(deg) = self.inner.borrow_mut().degree(node_id) {
                        degrees.push(groggy::AttrValue::Int(deg as i64));
                    }
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
```

---

### in_degree

**Attributes:**

```rust
#[pyo3(signature = (nodes = None))]
```

**Signature:**

```rust
fn in_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 34, token-chars: 1210.

**Implementation:**

```rust

        if !self.inner.borrow_mut().is_directed() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "in_degree() is only available for directed graphs. Use degree() for undirected graphs."
            ));
        }

        // Get fresh topology snapshot once
        let (_, _sources, targets) = self.inner.borrow_mut().get_columnar_topology();

        match nodes {
            // Single node case: in_degree(node_id) -> int
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let count = targets.iter().filter(|&&target| target == node).count();
                Ok(count.to_object(py))
            }
            // List of nodes case: in_degree([node1, node2, ...]) -> dict
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let result_dict = pyo3::types::PyDict::new(py);

                for node_id in node_ids {
                    let count = targets.iter().filter(|&&target| target == node_id).count();
                    result_dict.set_item(node_id, count)?;
                }

                Ok(result_dict.to_object(py))
            }
            // All nodes case: in_degree() -> dict
            None => {
                let result_dict = pyo3::types::PyDict::new(py);
                let all_nodes = self.inner.borrow_mut().node_ids();

                for node_id in all_nodes {
                    let count = targets.iter().filter(|&&target| target == node_id).count();
                    result_dict.set_item(node_id, count)?;
                }

                Ok(result_dict.to_object(py))
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "in_degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
```

---

### out_degree

**Attributes:**

```rust
#[pyo3(signature = (nodes = None))]
```

**Signature:**

```rust
fn out_degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 34, token-chars: 1212.

**Implementation:**

```rust

        if !self.inner.borrow_mut().is_directed() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "out_degree() is only available for directed graphs. Use degree() for undirected graphs."
            ));
        }

        // Get fresh topology snapshot once
        let (_, sources, _targets) = self.inner.borrow_mut().get_columnar_topology();

        match nodes {
            // Single node case: out_degree(node_id) -> int
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node = node_arg.extract::<NodeId>()?;
                let count = sources.iter().filter(|&&source| source == node).count();
                Ok(count.to_object(py))
            }
            // List of nodes case: out_degree([node1, node2, ...]) -> dict
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let result_dict = pyo3::types::PyDict::new(py);

                for node_id in node_ids {
                    let count = sources.iter().filter(|&&source| source == node_id).count();
                    result_dict.set_item(node_id, count)?;
                }

                Ok(result_dict.to_object(py))
            }
            // All nodes case: out_degree() -> dict
            None => {
                let result_dict = pyo3::types::PyDict::new(py);
                let all_nodes = self.inner.borrow_mut().node_ids();

                for node_id in all_nodes {
                    let count = sources.iter().filter(|&&source| source == node_id).count();
                    result_dict.set_item(node_id, count)?;
                }

                Ok(result_dict.to_object(py))
            }
            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "out_degree() argument must be a NodeId, list of NodeIds, or None",
            )),
        }
```

---

### neighborhood

**Attributes:**

```rust
#[allow(dead_code)]
```

**Signature:**

```rust
fn neighborhood(
        &mut self,
        nodes: &PyAny,
        k: Option<&PyAny>,
        unified: Option<bool>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 78, token-chars: 2365.

**Implementation:**

```rust

        let py = nodes.py();
        
        // Parse k parameter
        let hop_spec = if let Some(k_val) = k {
            if let Ok(single_k) = k_val.extract::<usize>() {
                HopSpecification::SingleLevel(single_k)
            } else if let Ok(multi_k) = k_val.extract::<Vec<usize>>() {
                HopSpecification::MultiLevel(multi_k)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "k parameter must be an integer or list of integers"
                ));
            }
        } else {
            HopSpecification::SingleLevel(1) // Default to 1-hop
        };
        
        let unified = unified.unwrap_or(false);
        
        // Parse nodes parameter - single node or list of nodes
        if let Ok(single_node) = nodes.extract::<NodeId>() {
            // Single node case
            let result = match hop_spec {
                HopSpecification::SingleLevel(1) => {
                    // Use optimized 1-hop method
                    self.inner.borrow_mut().neighborhood(single_node)
                        .map_err(graph_error_to_py_err)
                        .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                        .map(|py_nbh| py_nbh.into_py(py))
                }
                HopSpecification::SingleLevel(k) => {
                    // Use k-hop method
                    self.inner.borrow_mut().k_hop_neighborhood(single_node, k)
                        .map_err(graph_error_to_py_err)
                        .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                        .map(|py_nbh| py_nbh.into_py(py))
                }
                HopSpecification::MultiLevel(_levels) => {
                    // TODO: Implement multi-level sampling for single node
                    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                        "Multi-level sampling for single node not yet implemented"
                    ));
                }
            }?;
            
            Ok(result)
        } else if let Ok(node_list) = nodes.extract::<Vec<NodeId>>() {
            // Multiple nodes case
            if unified {
                // Return single combined subgraph
                let result = match hop_spec {
                    HopSpecification::SingleLevel(k) => {
                        self.inner.borrow_mut().unified_neighborhood(&node_list, k)
                            .map_err(graph_error_to_py_err)
                            .map(|nbh| PyNeighborhoodSubgraph { inner: nbh })
                            .map(|py_nbh| py_nbh.into_py(py))
                    }
                    HopSpecification::MultiLevel(_levels) => {
                        // TODO: Implement multi-level unified sampling
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "Multi-level unified sampling not yet implemented"
                        ));
                    }
                }?;
                
                Ok(result)
            } else {
                // Return separate subgraphs for each node
                let result = match hop_spec {
                    HopSpecification::SingleLevel(1) => {
                        // Use optimized multi-neighborhood method for 1-hop
                        self.inner.borrow_mut().multi_neighborhood(&node_list)
                            .map_err(graph_error_to_py_err)
                            .map(|result| PyNeighborhoodResult { inner: result })
                            .map(|py_result| py_result.into_py(py))
                    }
                    HopSpecification::SingleLevel(_k) => {
                        // TODO: Implement k-hop for multiple nodes returning separate subgraphs
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "k-hop sampling for multiple nodes (non-unified) not yet implemented"
                        ));
                    }
                    HopSpecification::MultiLevel(_levels) => {
                        // TODO: Implement multi-level sampling for multiple nodes
                        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                            "Multi-level sampling for multiple nodes not yet implemented"
                        ));
                    }
                }?;
                
                Ok(result)
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "nodes parameter must be a NodeId or list of NodeIds"
            ))
        }
```

---

### add_graph

**Signature:**

```rust
pub fn add_graph(&mut self, py: Python, other: &PyGraph) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 40, token-chars: 1460.

**Implementation:**

```rust

        // Get all nodes from the other graph with their attributes
        let other_node_ids = other.inner.borrow().node_ids();
        let other_edge_ids = other.inner.borrow().edge_ids();
        
        // Track ID mappings to handle potential conflicts
        let mut node_id_mapping = std::collections::HashMap::new();
        
        // Add all nodes from other graph
        for &old_node_id in &other_node_ids {
            // Get all attributes for this node
            let mut node_attrs = std::collections::HashMap::new();
            
            // Get attribute names for this node (this is a simplified approach)
            // TODO: This could be more efficient with a proper attribute iteration API
            let sample_attrs = ["name", "label", "type", "value", "weight", "id"]; // Common attribute names
            for attr_name in &sample_attrs {
                if let Ok(Some(attr_value)) = other.inner.borrow().get_node_attr(old_node_id, &attr_name.to_string()) {
                    node_attrs.insert(attr_name.to_string(), attr_value);
                }
            }
            
            // Add the node to this graph
            let new_node_id = self.inner.borrow_mut().add_node();
            node_id_mapping.insert(old_node_id, new_node_id);
            
            // Set all attributes on the new node
            for (attr_name, attr_value) in node_attrs {
                let _ = self.inner.borrow_mut().set_node_attr(new_node_id, attr_name, attr_value);
            }
        }
        
        // Add all edges from other graph
        for &old_edge_id in &other_edge_ids {
            if let Ok((old_source, old_target)) = other.inner.borrow().edge_endpoints(old_edge_id) {
                // Map old node IDs to new node IDs
                if let (Some(&new_source), Some(&new_target)) = (
                    node_id_mapping.get(&old_source),
                    node_id_mapping.get(&old_target)
                ) {
                    // Add the edge
                    match self.inner.borrow_mut().add_edge(new_source, new_target) {
                        Ok(new_edge_id) => {
                            // Copy edge attributes
                            let sample_edge_attrs = ["weight", "label", "type", "capacity"]; // Common edge attribute names
                            for attr_name in &sample_edge_attrs {
                                if let Ok(Some(attr_value)) = other.inner.borrow().get_edge_attr(old_edge_id, &attr_name.to_string()) {
                                    let _ = self.inner.borrow_mut().set_edge_attr(new_edge_id, attr_name.to_string(), attr_value);
                                }
                            }
                        }
                        Err(_) => {
                            // Skip edges that can't be added (e.g., duplicates in undirected graphs)
                            continue;
                        }
                    }
                }
            }
        }
        
        Ok(())
```

---

### view

**Signature:**

```rust
pub fn view(self_: PyRef<Self>, py: Python<'_>) -> PyResult<Py<PySubgraph>>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 276.

**Implementation:**

```rust

        // Pull ids via existing accessors
        let nodes: Vec<NodeId> = self_.inner.borrow().node_ids();
        let edges: Vec<EdgeId> = self_.inner.borrow().edge_ids();

        let mut sg = PySubgraph::new(nodes, edges, "full".to_string(), None);
        let this_graph: Py<PyGraph> = self_.into();
        sg.set_graph_reference(this_graph);
        Py::new(py, sg)
```

---

### is_connected

**Signature:**

```rust
pub fn is_connected(self_: PyRef<Self>, py: Python<'_>) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 104.

**Implementation:**

```rust

        // Create a full-view subgraph and check if it's connected
        let subgraph = Self::view(self_, py)?;
        let subgraph_ref = subgraph.borrow(py);
        subgraph_ref.is_connected()
```

---

### table

**Signature:**

```rust
pub fn table(self_: PyRef<Self>, py: Python<'_>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 100.

**Implementation:**

```rust

        // Forward to g.nodes.table() for consistency
        let nodes_accessor = Self::nodes(self_, py)?;
        let result = nodes_accessor.borrow(py).table(py);
        result
```

---

### edges_table

**Signature:**

```rust
pub fn edges_table(self_: PyRef<Self>, py: Python<'_>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 100.

**Implementation:**

```rust

        // Forward to g.edges.table() for consistency
        let edges_accessor = Self::edges(self_, py)?;
        let result = edges_accessor.borrow(py).table(py);
        result
```

---

### adjacency_matrix_to_graph_matrix

**Signature:**

```rust
fn adjacency_matrix_to_graph_matrix(
        &self,
        adjacency_matrix: groggy::AdjacencyMatrix,
    ) -> PyResult<groggy::GraphMatrix>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 43, token-chars: 1095.

**Implementation:**

```rust

        use groggy::core::array::GraphArray;
        use groggy::core::matrix::GraphMatrix;

        // Extract matrix data and convert to GraphArrays (columns)
        let size = adjacency_matrix.size;
        let mut columns = Vec::with_capacity(size);

        // Create a column for each matrix column
        for col_idx in 0..size {
            let column_values: Vec<groggy::AttrValue> = (0..size)
                .map(|row_idx| {
                    adjacency_matrix
                        .get(row_idx, col_idx)
                        .cloned()
                        .unwrap_or(groggy::AttrValue::Float(0.0))
                })
                .collect();

            let column_name = if let Some(ref labels) = adjacency_matrix.labels {
                format!(
                    "node_{}",
                    labels
                        .get(col_idx)
                        .copied()
                        .unwrap_or(col_idx as groggy::NodeId)
                )
            } else {
                format!("col_{}", col_idx)
            };

            let column = GraphArray::from_vec(column_values).with_name(column_name);
            columns.push(column);
        }

        // Create GraphMatrix from the columns
        let mut graph_matrix = GraphMatrix::from_arrays(columns).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to create GraphMatrix from adjacency matrix: {:?}",
                e
            ))
        })?;

        // Set proper column names using node labels if available
        if let Some(ref labels) = adjacency_matrix.labels {
            let column_names: Vec<String> = labels
                .iter()
                .map(|&node_id| format!("node_{}", node_id))
                .collect();
            graph_matrix.set_column_names(column_names).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set column names: {:?}", e))
            })?;
        }

        Ok(graph_matrix)
```

---

### create_nodes_accessor_internal

**Signature:**

```rust
fn create_nodes_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyNodesAccessor>>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 72.

**Implementation:**

```rust

        Py::new(
            py,
            PyNodesAccessor {
                graph: graph_ref,
                constrained_nodes: None,
            },
        )
```

---

### create_edges_accessor_internal

**Signature:**

```rust
fn create_edges_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyEdgesAccessor>>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 72.

**Implementation:**

```rust

        Py::new(
            py,
            PyEdgesAccessor {
                graph: graph_ref,
                constrained_edges: None,
            },
        )
```

---

### node_attribute_keys

**Signature:**

```rust
pub fn node_attribute_keys(&self, node_id: NodeId) -> Vec<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 122.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_node_attrs(node_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
```

---

### edge_attribute_keys

**Signature:**

```rust
pub fn edge_attribute_keys(&self, edge_id: EdgeId) -> Vec<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 122.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_edge_attrs(edge_id) {
            Ok(attrs) => attrs.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
```

---

### has_node_attribute

**Signature:**

```rust
pub fn has_node_attribute(&self, node_id: NodeId, attr_name: &str) -> bool
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 110.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_node_attr(node_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
```

---

### has_edge_attribute

**Signature:**

```rust
pub fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str) -> bool
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 110.

**Implementation:**

```rust

        match self.inner.borrow_mut().get_edge_attr(edge_id, &attr_name.to_string()) {
            Ok(Some(_)) => true,
            _ => false,
        }
```

---

### get_edge_endpoints

**Signature:**

```rust
pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Result<(NodeId, NodeId), String>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 74.

**Implementation:**

```rust

        self.inner
.borrow_mut()
            .edge_endpoints(edge_id)
            .map_err(|e| e.to_string())
```

---

### get_node_ids

**Signature:**

```rust
pub fn get_node_ids(&self) -> PyResult<Vec<NodeId>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 38.

**Implementation:**

```rust

        Ok(self.inner.borrow_mut().node_ids())
```

---

### get_edge_ids

**Signature:**

```rust
pub fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 38.

**Implementation:**

```rust

        Ok(self.inner.borrow_mut().edge_ids())
```

---

### get_node_ids_array

**Signature:**

```rust
pub fn get_node_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 329.

**Implementation:**

```rust

        let node_ids = self.inner.borrow_mut().node_ids();
        let attr_values: Vec<groggy::AttrValue> = node_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### get_edge_ids_array

**Signature:**

```rust
pub fn get_edge_ids_array(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 329.

**Implementation:**

```rust

        let edge_ids = self.inner.borrow_mut().edge_ids();
        let attr_values: Vec<groggy::AttrValue> = edge_ids
            .into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### group_nodes_by_attribute_internal

**Signature:**

```rust
pub fn group_nodes_by_attribute_internal(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 544.

**Implementation:**

```rust

        let results = self
            .inner
.borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
```

---

### __getattr__

**Signature:**

```rust
fn __getattr__(slf: PyRef<Self>, py: Python, name: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 135.

**Implementation:**

```rust

        // TODO: Complete implementation - temporarily disabled for compilation
        return Err(PyAttributeError::new_err(format!(
            "Attribute '{}' not found. Property-style attribute access is under development.",
            name
        )));
```

---

### _get_node_attribute_column

**Signature:**

```rust
fn _get_node_attribute_column(
        &self,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 463.

**Implementation:**

```rust

        match self
            .inner
.borrow_mut()
            ._get_node_attribute_column(&attr_name.to_string())
        {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<groggy::AttrValue> = values
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(groggy::AttrValue::Null)) // Use Null for missing values
                    .collect();

                // Create GraphArray and convert to PyGraphArray
                let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);
                let py_graph_array = PyGraphArray::from_graph_array(graph_array);

                Py::new(py, py_graph_array)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### validate_node_filter_attributes

**Signature:**

```rust
fn validate_node_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::core::query::NodeFilter,
    ) -> Result<(), String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 25, token-chars: 541.

**Implementation:**

```rust

        use groggy::core::query::NodeFilter;

        match filter {
            NodeFilter::AttributeFilter { name, .. }
            | NodeFilter::AttributeEquals { name, .. }
            | NodeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any nodes
                if !Self::attribute_exists_on_nodes(graph, name) {
                    return Err(name.clone());
                }
            }
            NodeFilter::And(filters) => {
                for f in filters {
                    Self::validate_node_filter_attributes(graph, f)?;
                }
            }
            NodeFilter::Or(filters) => {
                for f in filters {
                    Self::validate_node_filter_attributes(graph, f)?;
                }
            }
            NodeFilter::Not(filter) => {
                Self::validate_node_filter_attributes(graph, filter)?;
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
```

---

### validate_edge_filter_attributes

**Signature:**

```rust
fn validate_edge_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::core::query::EdgeFilter,
    ) -> Result<(), String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 25, token-chars: 541.

**Implementation:**

```rust

        use groggy::core::query::EdgeFilter;

        match filter {
            EdgeFilter::AttributeFilter { name, .. }
            | EdgeFilter::AttributeEquals { name, .. }
            | EdgeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any edges
                if !Self::attribute_exists_on_edges(graph, name) {
                    return Err(name.clone());
                }
            }
            EdgeFilter::And(filters) => {
                for f in filters {
                    Self::validate_edge_filter_attributes(graph, f)?;
                }
            }
            EdgeFilter::Or(filters) => {
                for f in filters {
                    Self::validate_edge_filter_attributes(graph, f)?;
                }
            }
            EdgeFilter::Not(filter) => {
                Self::validate_edge_filter_attributes(graph, filter)?;
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
```

---

### attribute_exists_on_nodes

**Signature:**

```rust
fn attribute_exists_on_nodes(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 169.

**Implementation:**

```rust

        let node_ids = graph.node_ids();
        for node_id in node_ids.iter().take(100) {
            // Sample first 100 nodes for performance
            if let Ok(Some(_)) = graph.get_node_attr(*node_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
```

---

### attribute_exists_on_edges

**Signature:**

```rust
fn attribute_exists_on_edges(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 169.

**Implementation:**

```rust

        let edge_ids = graph.edge_ids();
        for edge_id in edge_ids.iter().take(100) {
            // Sample first 100 edges for performance
            if let Ok(Some(_)) = graph.get_edge_attr(*edge_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
```

---


## ffi/api/graph_analytics.rs

### connected_components

**Attributes:**

```rust
#[pyo3(signature = (inplace = false, attr_name = None))]
```

**Signature:**

```rust
pub fn connected_components(
        &self,
        py: Python,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Vec<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 46, token-chars: 1092.

**Implementation:**

```rust

        let inplace = inplace.unwrap_or(false);
        
        // Delegate to core algorithm - THIN WRAPPER
        let options = TraversalOptions::default();
        let result = {
            let graph = self.graph.borrow(py);
            let components_result = graph
                .inner
                .borrow_mut()
                .connected_components(options)
                .map_err(graph_error_to_py_err)?;
            components_result
        };

        let mut subgraphs = Vec::new();

        // Handle bulk attribute setting if requested
        if inplace {
            if let Some(ref attr_name) = attr_name {
                let mut attrs_values = std::collections::HashMap::new();
                let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                    .components
                    .iter()
                    .enumerate()
                    .flat_map(|(i, component)| {
                        component
                            .nodes
                            .iter()
                            .map(move |&node_id| (node_id, groggy::AttrValue::Int(i as i64)))
                    })
                    .collect();
                attrs_values.insert(attr_name.clone(), node_value_pairs);

                // Set attributes in a separate borrow scope
                let graph = self.graph.borrow(py);
                graph
                    .inner
                    .borrow_mut()
                    .set_node_attrs(attrs_values)
                    .map_err(graph_error_to_py_err)?;
            }
        }

        // Convert core results to FFI wrappers - ZERO-COPY: just use pre-computed edges!
        for (i, component) in result.components.into_iter().enumerate() {
            // 🚀 PERFORMANCE: Use edges already computed by Rust core - no recomputation needed!
            // 🔧 SAFE: Create subgraph with inner RustSubgraph since we have Python context
            let subgraph = PySubgraph::new_with_inner(
                py,
                component.nodes,
                component.edges, // Use pre-computed induced edges from Rust core
                format!("connected_component_{}", i),
                Some(self.graph.clone()),
            );
            subgraphs.push(subgraph);
        }

        Ok(subgraphs)
```

---

### bfs

**Attributes:**

```rust
#[pyo3(signature = (start_node, max_depth = None, inplace = false, attr_name = None))]
```

**Signature:**

```rust
pub fn bfs(
        &self,
        py: Python,
        start_node: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 38, token-chars: 905.

**Implementation:**

```rust

        let inplace = inplace.unwrap_or(false);
        
        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Perform BFS traversal - get inner graph reference directly
        let result = {
            let graph = self.graph.borrow(py);
            let bfs_result = graph
                .inner
                .borrow_mut()
                .bfs(start_node, options)
                .map_err(graph_error_to_py_err)?;
            bfs_result
        };

        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "bfs_distance".to_string());

            // Use bulk attribute setting for performance
            let mut attrs_values = std::collections::HashMap::new();
            let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                .nodes
                .iter()
                .enumerate()
                .map(|(order, &node_id)| (node_id, groggy::AttrValue::Int(order as i64)))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);

            // Set attributes in a separate borrow scope
            let graph = self.graph.borrow(py);
            graph
                .inner
                .borrow_mut()
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        Ok(PySubgraph::new_with_inner(
            py,
            result.nodes,
            result.edges,
            "bfs_traversal".to_string(),
            Some(self.graph.clone()),
        ))
```

---

### dfs

**Attributes:**

```rust
#[pyo3(signature = (start_node, max_depth = None, inplace = false, attr_name = None))]
```

**Signature:**

```rust
pub fn dfs(
        &self,
        py: Python,
        start_node: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 38, token-chars: 902.

**Implementation:**

```rust

        let inplace = inplace.unwrap_or(false);
        
        // Create traversal options
        let mut options = TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Perform DFS traversal - get inner graph reference directly
        let result = {
            let graph = self.graph.borrow(py);
            let dfs_result = graph
                .inner
                .borrow_mut()
                .dfs(start_node, options)
                .map_err(graph_error_to_py_err)?;
            dfs_result
        };

        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "dfs_order".to_string());

            // Use bulk attribute setting for performance
            let mut attrs_values = std::collections::HashMap::new();
            let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = result
                .nodes
                .iter()
                .enumerate()
                .map(|(order, &node_id)| (node_id, groggy::AttrValue::Int(order as i64)))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);

            // Set attributes in a separate borrow scope
            let graph = self.graph.borrow(py);
            graph
                .inner
                .borrow_mut()
                .set_node_attrs(attrs_values)
                .map_err(graph_error_to_py_err)?;
        }

        Ok(PySubgraph::new_with_inner(
            py,
            result.nodes,
            result.edges,
            "dfs_traversal".to_string(),
            Some(self.graph.clone()),
        ))
```

---

### shortest_path

**Attributes:**

```rust
#[pyo3(signature = (source, target, weight_attribute = None, inplace = false, attr_name = None))]
```

**Signature:**

```rust
pub fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<Option<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 47, token-chars: 922.

**Implementation:**

```rust

        let inplace = inplace.unwrap_or(false);
        
        let options = PathFindingOptions {
            weight_attribute,
            max_path_length: None,
            heuristic: None,
        };

        let result = {
            let graph = self.graph.borrow(py);
            let path_result = graph
                .inner
                .borrow_mut()
                .shortest_path(source, target, options)
                .map_err(graph_error_to_py_err)?;
            path_result
        };

        match result {
            Some(path) => {
                if inplace {
                    if let Some(attr_name) = attr_name {
                        // Use bulk attribute setting for performance
                        let mut attrs_values = std::collections::HashMap::new();
                        let node_value_pairs: Vec<(NodeId, groggy::AttrValue)> = path
                            .nodes
                            .iter()
                            .enumerate()
                            .map(|(distance, &node_id)| {
                                (node_id, groggy::AttrValue::Int(distance as i64))
                            })
                            .collect();
                        attrs_values.insert(attr_name, node_value_pairs);

                        // Set attributes in a separate borrow scope
                        let graph = self.graph.borrow(py);
                        graph
                            .inner
                            .borrow_mut()
                            .set_node_attrs(attrs_values)
                            .map_err(graph_error_to_py_err)?;
                    }
                }

                Ok(Some(PySubgraph::new_with_inner(
                    py,
                    path.nodes,
                    path.edges,
                    "shortest_path".to_string(),
                    Some(self.graph.clone()),
                )))
            }
            None => Ok(None),
        }
```

---

### has_path

**Signature:**

```rust
pub fn has_path(&self, py: Python, source: NodeId, target: NodeId) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 292.

**Implementation:**

```rust

        let options = PathFindingOptions {
            weight_attribute: None,
            max_path_length: None,
            heuristic: None,
        };

        let result = {
            let graph = self.graph.borrow(py);
            let path_result = graph
                .inner
                .borrow_mut()
                .shortest_path(source, target, options)
                .map_err(graph_error_to_py_err)?;
            path_result
        };

        Ok(result.is_some())
```

---

### memory_statistics

**Signature:**

```rust
fn memory_statistics(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 1047.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let stats = graph.inner.borrow().memory_statistics();

        // Convert MemoryStatistics to Python dict
        let dict = PyDict::new(py);
        dict.set_item("pool_memory_bytes", stats.pool_memory_bytes)?;
        dict.set_item("space_memory_bytes", stats.space_memory_bytes)?;
        dict.set_item("history_memory_bytes", stats.history_memory_bytes)?;
        dict.set_item(
            "change_tracker_memory_bytes",
            stats.change_tracker_memory_bytes,
        )?;
        dict.set_item("total_memory_bytes", stats.total_memory_bytes)?;
        dict.set_item("total_memory_mb", stats.total_memory_mb)?;

        // Add memory efficiency stats
        let efficiency_dict = PyDict::new(py);
        efficiency_dict.set_item("bytes_per_node", stats.memory_efficiency.bytes_per_node)?;
        efficiency_dict.set_item("bytes_per_edge", stats.memory_efficiency.bytes_per_edge)?;
        efficiency_dict.set_item("bytes_per_entity", stats.memory_efficiency.bytes_per_entity)?;
        efficiency_dict.set_item("overhead_ratio", stats.memory_efficiency.overhead_ratio)?;
        efficiency_dict.set_item("cache_efficiency", stats.memory_efficiency.cache_efficiency)?;
        dict.set_item("memory_efficiency", efficiency_dict)?;

        Ok(dict.to_object(py))
```

---

### get_summary

**Signature:**

```rust
fn get_summary(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 187.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();

        Ok(format!(
            "Graph Analytics: {} nodes, {} edges",
            node_count, edge_count
        ))
```

---


## ffi/api/graph_query.rs

### filter_nodes

**Signature:**

```rust
pub fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 35, token-chars: 1129.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);

        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        let filtered_nodes = graph
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // O(k) Calculate induced edges using optimized core subgraph method
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = graph.inner.borrow().get_columnar_topology();
        let mut induced_edges = Vec::new();

        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // O(1) HashSet lookups instead of O(n) Vec::contains
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }

        Ok(PySubgraph::new(
            filtered_nodes,
            induced_edges,
            "filtered_nodes".to_string(),
            Some(self.graph.clone()),
        ))
```

---

### filter_edges

**Signature:**

```rust
pub fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 32, token-chars: 958.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);

        // Similar pattern to filter_nodes but for edges
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_edge_query")?;
            let parsed_filter: PyEdgeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query",
            ));
        };

        let filtered_edges = graph
            .inner
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;

        // Calculate nodes that are connected by the filtered edges
        let mut nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                nodes.insert(source);
                nodes.insert(target);
            }
        }

        let node_vec: Vec<NodeId> = nodes.into_iter().collect();

        Ok(PySubgraph::new(
            node_vec,
            filtered_edges,
            "filtered_edges".to_string(),
            Some(self.graph.clone()),
        ))
```

---

### filter_subgraph_nodes

**Signature:**

```rust
fn filter_subgraph_nodes(
        &self,
        py: Python,
        subgraph: &PySubgraph,
        filter: &PyAny,
    ) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 42, token-chars: 1297.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);

        // Parse filter same way as filter_nodes
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query",
            ));
        };

        // Apply filter only to nodes in the subgraph
        let subgraph_node_set: HashSet<NodeId> = subgraph.get_nodes().iter().copied().collect();
        let all_filtered_nodes = graph
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // Intersect with subgraph nodes
        let filtered_nodes: Vec<NodeId> = all_filtered_nodes
            .into_iter()
            .filter(|node_id| subgraph_node_set.contains(node_id))
            .collect();

        // Calculate induced edges within the filtered nodes
        let filtered_node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let filtered_edges: Vec<EdgeId> = subgraph
            .get_edges()
            .iter()
            .filter(|&&edge_id| {
                if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                    filtered_node_set.contains(&source) && filtered_node_set.contains(&target)
                } else {
                    false
                }
            })
            .copied()
            .collect();

        Ok(PySubgraph::new(
            filtered_nodes,
            filtered_edges,
            "filtered_subgraph".to_string(),
            Some(self.graph.clone()),
        ))
```

---

### aggregate

**Attributes:**

```rust
#[pyo3(signature = (attribute, operation, target = None, node_ids = None))]
```

**Signature:**

```rust
fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 38, token-chars: 975.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                if let Some(node_list) = node_ids {
                    // Custom node list aggregation
                    self.aggregate_custom_nodes(py, &graph, node_list, attribute)
                } else {
                    // All nodes aggregation
                    let result = graph
                        .inner
                        .borrow()
                        .aggregate_node_attribute(&attribute, &operation)
                        .map_err(graph_error_to_py_err)?;

                    let dict = PyDict::new(py);
                    dict.set_item("value", result.value)?;
                    dict.set_item("operation", &operation)?;
                    dict.set_item("attribute", &attribute)?;
                    dict.set_item("target", "nodes")?;
                    Ok(dict.to_object(py))
                }
            }
            "edges" => {
                // Edge aggregation
                let result = graph
                    .inner
                    .borrow()
                    .aggregate_edge_attribute(&attribute, &operation)
                    .map_err(graph_error_to_py_err)?;

                let dict = PyDict::new(py);
                dict.set_item("value", result.value)?;
                dict.set_item("operation", &operation)?;
                dict.set_item("attribute", &attribute)?;
                dict.set_item("target", "edges")?;
                Ok(dict.to_object(py))
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid target '{}'. Use 'nodes' or 'edges'",
                target
            ))),
        }
```

---

### execute

**Attributes:**

```rust
#[pyo3(signature = (query, **kwargs))]
```

**Signature:**

```rust
fn execute(
        &self,
        py: Python,
        query: &str,
        kwargs: Option<&pyo3::types::PyDict>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 17, token-chars: 637.

**Implementation:**

```rust

        // Parse and execute complex graph queries
        let _graph = self.graph.borrow(py);

        // For now, support basic query patterns
        if query.starts_with("nodes where ") {
            let filter_str = &query[12..]; // Remove "nodes where "
            let filter_py_str = filter_str.to_string().into_py(py);
            self.filter_nodes(py, filter_py_str.as_ref(py))
                .map(|subgraph| Py::new(py, subgraph).unwrap().to_object(py))
        } else if query.starts_with("edges where ") {
            let filter_str = &query[12..]; // Remove "edges where "
            let filter_py_str = filter_str.to_string().into_py(py);
            self.filter_edges(py, filter_py_str.as_ref(py))
                .map(|subgraph| Py::new(py, subgraph).unwrap().to_object(py))
        } else {
            Err(PyValueError::new_err(format!(
                "Unsupported query pattern: {}",
                query
            )))
        }
```

---

### get_stats

**Signature:**

```rust
fn get_stats(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 200.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();

        Ok(format!(
            "Query module ready: {} nodes, {} edges available",
            node_count, edge_count
        ))
```

---

### aggregate_custom_nodes

**Signature:**

```rust
fn aggregate_custom_nodes(
        &self,
        py: Python,
        graph: &crate::ffi::api::graph::PyGraph,
        node_ids: Vec<NodeId>,
        attribute: AttrName,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 67, token-chars: 1455.

**Implementation:**

```rust

        // Use bulk attribute retrieval for much better performance
        let bulk_attributes = graph
            .inner
            .borrow()
            ._get_node_attributes_for_nodes(&node_ids, &attribute)
            .map_err(graph_error_to_py_err)?;
        let mut values = Vec::new();

        // Extract values from bulk result
        for attr_value in bulk_attributes {
            if let Some(value) = attr_value {
                values.push(value);
            }
        }

        // Compute statistics
        let dict = PyDict::new(py);
        dict.set_item("count", values.len())?;

        if !values.is_empty() {
            // Convert first value to determine type for aggregation
            if let Some(first_val) = values.first() {
                match first_val {
                    groggy::AttrValue::Int(_) | groggy::AttrValue::SmallInt(_) => {
                        let int_values: Vec<i64> = values
                            .iter()
                            .filter_map(|v| match v {
                                groggy::AttrValue::Int(i) => Some(*i),
                                groggy::AttrValue::SmallInt(i) => Some(*i as i64),
                                _ => None,
                            })
                            .collect();

                        if !int_values.is_empty() {
                            dict.set_item("sum", int_values.iter().sum::<i64>())?;
                            dict.set_item("min", *int_values.iter().min().unwrap())?;
                            dict.set_item("max", *int_values.iter().max().unwrap())?;
                            dict.set_item(
                                "mean",
                                int_values.iter().sum::<i64>() as f64 / int_values.len() as f64,
                            )?;
                        }
                    }
                    groggy::AttrValue::Float(_) => {
                        let float_values: Vec<f64> = values
                            .iter()
                            .filter_map(|v| match v {
                                groggy::AttrValue::Float(f) => Some(*f as f64),
                                _ => None,
                            })
                            .collect();

                        if !float_values.is_empty() {
                            dict.set_item("sum", float_values.iter().sum::<f64>())?;
                            dict.set_item(
                                "min",
                                float_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            )?;
                            dict.set_item(
                                "max",
                                float_values
                                    .iter()
                                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                            )?;
                            dict.set_item(
                                "mean",
                                float_values.iter().sum::<f64>() / float_values.len() as f64,
                            )?;
                        }
                    }
                    _ => {
                        // For non-numeric types, just provide count
                    }
                }
            }
        }

        Ok(dict.to_object(py))
```

---


## ffi/api/graph_version.rs

### from_commit_info

**Signature:**

```rust
pub fn from_commit_info(info: groggy::api::graph::CommitInfo) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 23, token-chars: 467.

**Implementation:**

```rust

        // Create a simplified Commit from CommitInfo
        // Note: This is a temporary bridge until we have full core integration
        let parents = match info.parent {
            Some(p) => vec![p],
            None => vec![],
        };

        let fake_delta = std::sync::Arc::new(groggy::core::history::Delta {
            content_hash: [0u8; 32],
            nodes_added: Vec::new(),
            nodes_removed: Vec::new(),
            edges_added: Vec::new(),
            edges_removed: Vec::new(),
            node_attr_changes: Vec::new(),
            edge_attr_changes: Vec::new(),
        });

        let commit = groggy::core::history::Commit::new(
            info.id,
            parents,
            fake_delta,
            info.message,
            info.author,
        );

        Self {
            inner: std::sync::Arc::new(commit),
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 104.

**Implementation:**

```rust

        format!(
            "Commit(id={}, message='{}', author='{}')",
            self.inner.id, self.inner.message, self.inner.author
        )
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 109.

**Implementation:**

```rust

        format!(
            "BranchInfo(name='{}', head={}, current={})",
            self.inner.name, self.inner.head, self.inner.is_current
        )
```

---

### commit

**Signature:**

```rust
fn commit(&self, py: Python, message: String, author: String) -> PyResult<StateId>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 151.

**Implementation:**

```rust

        let graph = self.graph.borrow_mut(py);
        let commit_result = graph
            .inner
            .borrow_mut()
            .commit(message, author)
            .map_err(graph_error_to_py_err);
        commit_result
```

---

### create_branch

**Signature:**

```rust
fn create_branch(&self, py: Python, branch_name: String) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 154.

**Implementation:**

```rust

        let graph = self.graph.borrow_mut(py);
        let create_result = graph
            .inner
            .borrow_mut()
            .create_branch(branch_name)
            .map_err(graph_error_to_py_err);
        create_result
```

---

### checkout_branch

**Signature:**

```rust
fn checkout_branch(&self, py: Python, branch_name: String) -> PyResult<()>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 160.

**Implementation:**

```rust

        let graph = self.graph.borrow_mut(py);
        let checkout_result = graph
            .inner
            .borrow_mut()
            .checkout_branch(branch_name)
            .map_err(graph_error_to_py_err);
        checkout_result
```

---

### branches

**Signature:**

```rust
fn branches(&self, py: Python) -> Vec<PyBranchInfo>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 171.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let branches = graph
            .inner
            .borrow()
            .list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo { inner: branch_info })
            .collect();
        branches
```

---

### historical_view

**Signature:**

```rust
fn historical_view(&self, py: Python, commit_id: StateId) -> PyResult<PyHistoricalView>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 234.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let inner = graph.inner.borrow();
        let view_result = inner.view_at_commit(commit_id);
        match view_result {
            Ok(_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### create_snapshot

**Signature:**

```rust
fn create_snapshot(&self, py: Python, name: Option<&str>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 342.

**Implementation:**

```rust

        let snapshot_name = name.unwrap_or("snapshot");
        let author = "system".to_string();
        let message = format!("Snapshot: {}", snapshot_name);

        let state_id = self.commit(py, message, author)?;

        // Return snapshot info as a dictionary
        let dict = PyDict::new(py);
        dict.set_item("state_id", state_id)?;
        dict.set_item("name", snapshot_name)?;
        dict.set_item("type", "snapshot")?;

        Ok(dict.to_object(py))
```

---

### restore_snapshot

**Signature:**

```rust
fn restore_snapshot(&self, py: Python, snapshot_id: &str) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 14, token-chars: 224.

**Implementation:**

```rust

        // Parse snapshot_id as StateId
        match snapshot_id.parse::<StateId>() {
            Ok(state_id) => {
                match self.historical_view(py, state_id) {
                    Ok(_) => {
                        // In a full implementation, you'd actually restore the state
                        // For now, just indicate success if the snapshot exists
                        Ok(true)
                    }
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Err(PyValueError::new_err(format!(
                "Invalid snapshot ID: {}",
                snapshot_id
            ))),
        }
```

---

### get_history

**Signature:**

```rust
fn get_history(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 613.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);

        // Create a summary of version history
        let dict = PyDict::new(py);

        // Get branch information
        let branches = self.branches(py);
        let branch_list: Vec<PyObject> = branches
            .into_iter()
            .map(|branch| Py::new(py, branch).unwrap().to_object(py))
            .collect();
        dict.set_item("branches", branch_list)?;

        // Get commit count (simplified)
        dict.set_item("total_commits", 0)?; // Would be implemented with actual history
        dict.set_item(
            "has_uncommitted_changes",
            graph.inner.borrow().has_uncommitted_changes(),
        )?;

        // Get current state info
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();
        dict.set_item(
            "current_state",
            format!("{} nodes, {} edges", node_count, edge_count),
        )?;

        Ok(dict.to_object(py))
```

---

### get_node_mapping

**Signature:**

```rust
fn get_node_mapping(&self, py: Python, uid_key: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 26, token-chars: 731.

**Implementation:**

```rust

        let dict = PyDict::new(py);
        
        // Get node IDs in isolated borrow scope
        let node_ids = {
            let graph = self.graph.borrow(py);
            let result = graph.inner.borrow().node_ids();
            result
        };

        // Scan all nodes for the specified uid_key attribute using isolated borrows
        for node_id in node_ids {
            let attr_value = {
                let graph = self.graph.borrow(py);
                let result = graph.inner.borrow().get_node_attr(node_id, &uid_key);
                result
            };
            
            if let Ok(Some(attr_value)) = attr_value {
                // Convert attribute value to appropriate Python type
                let key_value = match attr_value {
                    RustAttrValue::Text(s) => s.to_object(py),
                    RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                    RustAttrValue::Int(i) => i.to_object(py),
                    RustAttrValue::SmallInt(i) => i.to_object(py),
                    RustAttrValue::Float(f) => f.to_object(py),
                    RustAttrValue::Bool(b) => b.to_object(py),
                    _ => continue, // Skip unsupported types
                };

                dict.set_item(key_value, node_id)?;
            }
        }

        Ok(dict.to_object(py))
```

---

### get_info

**Signature:**

```rust
fn get_info(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 290.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();
        let has_changes = graph.inner.borrow().has_uncommitted_changes();

        Ok(format!(
            "Version Control: {} nodes, {} edges, uncommitted changes: {}",
            node_count, edge_count, has_changes
        ))
```

---


## ffi/core/accessors.rs

### attr_value_to_python_value

**Signature:**

```rust
fn attr_value_to_python_value(py: Python, attr_value: &AttrValue) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 344.

**Implementation:**

```rust

    match attr_value {
        AttrValue::Int(val) => Ok(val.to_object(py)),
        AttrValue::SmallInt(val) => Ok((*val as i64).to_object(py)),
        AttrValue::Float(val) => Ok(val.to_object(py)),
        AttrValue::Bool(val) => Ok(val.to_object(py)),
        AttrValue::Text(val) => Ok(val.to_object(py)),
        AttrValue::CompactText(val) => Ok(val.as_str().to_object(py)),
        _ => Ok(py.None()),
    }
```

---

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 228.

**Implementation:**

```rust

        if self.index < self.node_ids.len() {
            let node_id = self.node_ids[self.index];
            self.index += 1;

            // Create NodeView for this node
            let node_view = PyGraph::create_node_view_internal(self.graph.clone(), py, node_id)?;
            Ok(Some(node_view.to_object(py)))
        } else {
            Ok(None)
        }
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 189, token-chars: 4965.

**Implementation:**

```rust

        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<NodeId>() {
            let actual_node_id = if let Some(ref constrained) = self.constrained_nodes {
                // Constrained case: treat as index into constrained list
                if (index_or_id as usize) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Node index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id as usize]
            } else {
                // Unconstrained case: treat as actual node ID (existing behavior)
                index_or_id
            };

            // Single node access - return NodeView
            let graph = self.graph.borrow(py);
            if !graph.has_node_internal(actual_node_id) {
                return Err(PyKeyError::new_err(format!(
                    "Node {} does not exist",
                    actual_node_id
                )));
            }

            let node_view =
                PyGraph::create_node_view_internal(self.graph.clone(), py, actual_node_id)?;
            return Ok(node_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow(py);
            let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
                constrained.clone()
            } else {
                { let node_ids = graph.inner.borrow().node_ids(); node_ids }
            };

            // Check if boolean mask length matches node count
            if boolean_mask.len() != all_node_ids.len() {
                return Err(PyIndexError::new_err(format!(
                    "Boolean mask length ({}) must match number of nodes ({})",
                    boolean_mask.len(),
                    all_node_ids.len()
                )));
            }

            // Select nodes where boolean mask is True
            let selected_nodes: Vec<NodeId> = all_node_ids
                .iter()
                .zip(boolean_mask.iter())
                .filter_map(|(&node_id, &include)| if include { Some(node_id) } else { None })
                .collect();

            if selected_nodes.is_empty() {
                return Err(PyIndexError::new_err("Boolean mask selected no nodes"));
            }

            // Validate all selected nodes exist
            for &node_id in &selected_nodes {
                if !graph.has_node_internal(node_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> =
                selected_nodes.iter().copied().collect();
            let (edge_ids, sources, targets) = { 
            let topology = graph.inner.borrow().get_columnar_topology();
            topology
        };
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph with inner RustSubgraph  
            let subgraph = PySubgraph::new_with_inner(
                py,
                selected_nodes,
                induced_edges,
                "boolean_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<NodeId>>() {
            // Batch node access - return Subgraph
            
            // Convert indices to actual node IDs if constrained
            let actual_node_ids: Result<Vec<NodeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_nodes {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index as usize) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Node index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index as usize])
                            }
                        })
                        .collect()
                } else {
                    // Unconstrained case: treat as actual node IDs
                    Ok(indices_or_ids)
                };

            let node_ids = actual_node_ids?;

            // Validate all nodes exist
            for &node_id in &node_ids {
                let exists = {
                    let graph = self.graph.borrow(py);
                    graph.has_node_internal(node_id)
                };
                if !exists {
                    return Err(PyKeyError::new_err(format!(
                        "Node {} does not exist",
                        node_id
                    )));
                }
            }

            // Get induced edges for selected nodes
            let node_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let (edge_ids, sources, targets) = { 
                let graph = self.graph.borrow(py);
                let topology = graph.inner.borrow().get_columnar_topology();
                topology
            };
            let mut induced_edges = Vec::new();

            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                if node_set.contains(&source) && node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph with inner RustSubgraph
            let subgraph = PySubgraph::new_with_inner(
                py,
                node_ids,
                induced_edges,
                "node_batch_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let all_node_ids = {
                let graph = self.graph.borrow(py);  // Only need read access
                { let node_ids = graph.inner.borrow().node_ids(); node_ids }
            };

            // Convert slice to indices
            let slice_info = slice.indices(
                all_node_ids
                    .len()
                    .try_into()
                    .map_err(|_| PyValueError::new_err("Collection too large for slice"))?,
            )?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;

            // Extract nodes based on slice
            let mut selected_nodes = Vec::new();
            let mut i = start;
            while i < stop && i < all_node_ids.len() {
                selected_nodes.push(all_node_ids[i]);
                i += step;
            }

            // 🚀 PERFORMANCE FIX: Use core columnar topology instead of O(E) FFI algorithm
            let selected_node_set: std::collections::HashSet<NodeId> =
                selected_nodes.iter().copied().collect();
            let (edge_ids, sources, targets) = { 
                let graph = self.graph.borrow(py);
                let topology = graph.inner.borrow().get_columnar_topology();
                topology
            };
            let mut induced_edges = Vec::new();

            // O(k) where k = active edges, much better than O(E)
            for i in 0..edge_ids.len() {
                let edge_id = edge_ids[i];
                let source = sources[i];
                let target = targets[i];

                // O(1) HashSet lookups
                if selected_node_set.contains(&source) && selected_node_set.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }

            // Create and return Subgraph with inner RustSubgraph
            let subgraph = PySubgraph::new_with_inner(
                py,
                selected_nodes,
                induced_edges,
                "node_slice_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as string (attribute name access)
        if let Ok(_attr_name) = key.extract::<String>() {
            // TODO: Complete implementation - temporarily disabled
            return Err(PyTypeError::new_err(
                "String attribute access is under development. Use g.age syntax instead."
            ));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Node index must be int, list of ints, slice, or string attribute name. \
            Examples: g.nodes[0], g.nodes[0:10], g.nodes['age'], or g.age for attribute access.",
        ))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(&self, py: Python) -> PyResult<NodesIterator>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 252.

**Implementation:**

```rust

        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            let graph = self.graph.borrow(py);
            let node_ids = graph.inner.borrow().node_ids();
            node_ids
        };

        Ok(NodesIterator {
            graph: self.graph.clone(),
            node_ids,
            index: 0,
        })
```

---

### __len__

**Signature:**

```rust
fn __len__(&self, py: Python) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 145.

**Implementation:**

```rust

        if let Some(ref constrained) = self.constrained_nodes {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.get_node_count())
        }
```

---

### __str__

**Signature:**

```rust
fn __str__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 114.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let count = graph.get_node_count();
        Ok(format!("NodesAccessor({} nodes)", count))
```

---

### attributes

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn attributes(&self, py: Python) -> PyResult<Vec<String>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 18, token-chars: 475.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which nodes to check
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            // Get all node IDs from the graph
            (0..graph.get_node_count() as NodeId).collect()
        };

        // Collect attributes from all nodes
        for &node_id in &node_ids {
            if graph.has_node_internal(node_id) {
                let attrs = graph.node_attribute_keys(node_id);
                for attr in attrs {
                    all_attrs.insert(attr);
                }
            }
        }

        // Convert to sorted vector
        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        Ok(result)
```

---

### table

**Signature:**

```rust
pub fn table(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 481.

**Implementation:**

```rust

        use crate::ffi::core::table::PyGraphTable;

        // Determine node list based on constraints
        let node_data = if let Some(ref constrained) = self.constrained_nodes {
            // Subgraph case: use constrained nodes
            constrained.iter().map(|&id| id as u64).collect()
        } else {
            // Full graph case: get all node IDs from the graph
            let graph = self.graph.borrow(py);
            let node_ids = graph
                .inner
                .borrow()
                .node_ids()
                .into_iter()
                .map(|id| id as u64)
                .collect();
            node_ids
        };

        // Use PyGraphTable::from_graph_nodes to create the table
        let py_table = PyGraphTable::from_graph_nodes(
            py.get_type::<PyGraphTable>(),
            py,
            self.graph.clone(),
            node_data,
            None, // Get all attributes
        )?;

        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### all

**Signature:**

```rust
fn all(&self, py: Python) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 29, token-chars: 781.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        
        // Get all node IDs (respecting constraints if any)
        let all_node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            { let node_ids = graph.inner.borrow().node_ids(); node_ids }
        };
        
        // Get all induced edges for these nodes using columnar topology (high performance)
        let selected_node_set: std::collections::HashSet<NodeId> =
            all_node_ids.iter().copied().collect();
        let (edge_ids, sources, targets) = { 
            let topology = graph.inner.borrow().get_columnar_topology();
            topology
        };
        let mut induced_edges = Vec::new();

        // O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // Include edge if both endpoints are in our node set
            if selected_node_set.contains(&source) && selected_node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }

        // Create and return Subgraph with inner RustSubgraph
        let subgraph = PySubgraph::new_with_inner(
            py,
            all_node_ids,
            induced_edges,
            "all_nodes".to_string(),
            Some(self.graph.clone()),
        );

        Ok(subgraph)
```

---

### _get_node_attribute_column

**Signature:**

```rust
fn _get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 69, token-chars: 1524.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        
        // Determine which nodes to check
        let node_ids = if let Some(ref constrained) = self.constrained_nodes {
            constrained.clone()
        } else {
            { let node_ids = graph.inner.borrow().node_ids(); node_ids }
        };

        if node_ids.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot access attribute '{}': No nodes available", 
                attr_name
            )));
        }

        // Check if the attribute exists on ANY node
        let mut attribute_exists = false;
        for &node_id in &node_ids {
            if graph.has_node_internal(node_id) {
                let attrs = graph.node_attribute_keys(node_id);
                if attrs.contains(&attr_name.to_string()) {
                    attribute_exists = true;
                    break;
                }
            }
        }

        if !attribute_exists {
            return Err(PyKeyError::new_err(format!(
                "Attribute '{}' does not exist on any nodes. Available attributes: {:?}",
                attr_name,
                {
                    let mut all_attrs = std::collections::HashSet::new();
                    for &node_id in &node_ids {
                        if graph.has_node_internal(node_id) {
                            let attrs = graph.node_attribute_keys(node_id);
                            for attr in attrs {
                                all_attrs.insert(attr);
                            }
                        }
                    }
                    let mut result: Vec<String> = all_attrs.into_iter().collect();
                    result.sort();
                    result
                }
            )));
        }

        // Collect attribute values - allow None for nodes without the attribute
        let mut values: Vec<Option<PyObject>> = Vec::new();
        for &node_id in &node_ids {
            if graph.has_node_internal(node_id) {
                match graph.inner.borrow().get_node_attr(node_id, &attr_name.to_string()) {
                    Ok(Some(value)) => {
                        // Convert the attribute value to Python object
                        let py_value = attr_value_to_python_value(py, &value)?;
                        values.push(Some(py_value));
                    }
                    Ok(None) => {
                        values.push(None);
                    }
                    Err(_) => {
                        values.push(None);
                    }
                }
            } else {
                values.push(None);
            }
        }

        // Convert to Python list
        let py_values: Vec<PyObject> = values
            .into_iter()
            .map(|opt_val| match opt_val {
                Some(val) => val,
                None => py.None(),
            })
            .collect();

        Ok(py_values.to_object(py))
```

---

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 228.

**Implementation:**

```rust

        if self.index < self.edge_ids.len() {
            let edge_id = self.edge_ids[self.index];
            self.index += 1;

            // Create EdgeView for this edge
            let edge_view = PyGraph::create_edge_view_internal(self.graph.clone(), py, edge_id)?;
            Ok(Some(edge_view.to_object(py)))
        } else {
            Ok(None)
        }
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 166, token-chars: 4294.

**Implementation:**

```rust

        // Try to extract as single integer
        if let Ok(index_or_id) = key.extract::<EdgeId>() {
            let actual_edge_id = if let Some(ref constrained) = self.constrained_edges {
                // Constrained case: treat as index into constrained list
                if (index_or_id as usize) >= constrained.len() {
                    return Err(PyIndexError::new_err(format!(
                        "Edge index {} out of range (0-{})",
                        index_or_id,
                        constrained.len() - 1
                    )));
                }
                constrained[index_or_id as usize]
            } else {
                // Unconstrained case: treat as actual edge ID (existing behavior)
                index_or_id
            };

            // Single edge access - return EdgeView
            let graph = self.graph.borrow(py);
            if !graph.has_edge_internal(actual_edge_id) {
                return Err(PyKeyError::new_err(format!(
                    "Edge {} does not exist",
                    actual_edge_id
                )));
            }

            let edge_view =
                PyGraph::create_edge_view_internal(self.graph.clone(), py, actual_edge_id)?;
            return Ok(edge_view.to_object(py));
        }

        // Try to extract as boolean array/list (boolean indexing) - CHECK FIRST before integers
        if let Ok(boolean_mask) = key.extract::<Vec<bool>>() {
            let graph = self.graph.borrow(py);
            let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
                constrained.clone()
            } else {
                { let edge_ids = graph.inner.borrow().edge_ids(); edge_ids }
            };

            // Check if boolean mask length matches edge count
            if boolean_mask.len() != all_edge_ids.len() {
                return Err(PyIndexError::new_err(format!(
                    "Boolean mask length ({}) must match number of edges ({})",
                    boolean_mask.len(),
                    all_edge_ids.len()
                )));
            }

            // Select edges where boolean mask is True
            let selected_edges: Vec<EdgeId> = all_edge_ids
                .iter()
                .zip(boolean_mask.iter())
                .filter_map(|(&edge_id, &include)| if include { Some(edge_id) } else { None })
                .collect();

            if selected_edges.is_empty() {
                return Err(PyIndexError::new_err("Boolean mask selected no edges"));
            }

            // Validate all selected edges exist
            for &edge_id in &selected_edges {
                if !graph.has_edge_internal(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from selected edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                let endpoints = {
                    let graph = self.graph.borrow(py);
                    let result = graph.inner.borrow().edge_endpoints(edge_id);
                    result
                };
                if let Ok((source, target)) = endpoints {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph with inner RustSubgraph
            let subgraph = PySubgraph::new_with_inner(
                py,
                endpoint_nodes.into_iter().collect(),
                selected_edges,
                "boolean_edge_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as list of integers (batch access) - CHECK AFTER boolean arrays
        if let Ok(indices_or_ids) = key.extract::<Vec<EdgeId>>() {
            // Batch edge access - return Subgraph with these edges + their endpoints
            let graph = self.graph.borrow(py);

            // Convert indices to actual edge IDs if constrained
            let actual_edge_ids: Result<Vec<EdgeId>, PyErr> =
                if let Some(ref constrained) = self.constrained_edges {
                    // Constrained case: treat as indices into constrained list
                    indices_or_ids
                        .into_iter()
                        .map(|index| {
                            if (index as usize) >= constrained.len() {
                                Err(PyIndexError::new_err(format!(
                                    "Edge index {} out of range (0-{})",
                                    index,
                                    constrained.len() - 1
                                )))
                            } else {
                                Ok(constrained[index as usize])
                            }
                        })
                        .collect()
                } else {
                    // Unconstrained case: treat as actual edge IDs
                    Ok(indices_or_ids)
                };

            let edge_ids = actual_edge_ids?;

            // Validate all edges exist
            for &edge_id in &edge_ids {
                if !graph.has_edge_internal(edge_id) {
                    return Err(PyKeyError::new_err(format!(
                        "Edge {} does not exist",
                        edge_id
                    )));
                }
            }

            // Get all endpoint nodes from these edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph with inner RustSubgraph
            let subgraph = PySubgraph::new_with_inner(
                py,
                endpoint_nodes.into_iter().collect(),
                edge_ids,
                "edge_batch_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<PySlice>() {
            let all_edge_ids = {
                let graph = self.graph.borrow(py);
                { let edge_ids = graph.inner.borrow().edge_ids(); edge_ids }
            };

            // Convert slice to indices
            let slice_info = slice.indices(
                all_edge_ids
                    .len()
                    .try_into()
                    .map_err(|_| PyValueError::new_err("Collection too large for slice"))?,
            )?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;

            // Extract edges based on slice
            let mut selected_edges = Vec::new();
            let mut i = start;
            while i < stop && i < all_edge_ids.len() {
                selected_edges.push(all_edge_ids[i]);
                i += step;
            }

            // Get all endpoint nodes from selected edges
            let mut endpoint_nodes = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                let endpoints = {
                    let graph = self.graph.borrow(py);
                    let result = graph.inner.borrow().edge_endpoints(edge_id);
                    result
                };
                if let Ok((source, target)) = endpoints {
                    endpoint_nodes.insert(source);
                    endpoint_nodes.insert(target);
                }
            }

            // Create and return Subgraph with inner RustSubgraph
            let subgraph = PySubgraph::new_with_inner(
                py,
                endpoint_nodes.into_iter().collect(),
                selected_edges,
                "edge_slice_selection".to_string(),
                Some(self.graph.clone()),
            );

            return Ok(Py::new(py, subgraph)?.to_object(py));
        }

        // If none of the above worked, return error
        Err(PyTypeError::new_err(
            "Edge index must be int, list of ints, or slice. \
            Examples: g.edges[0], g.edges[0:10]. For attribute access use g.edges.weight syntax.",
        ))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(&self, py: Python) -> PyResult<EdgesIterator>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 252.

**Implementation:**

```rust

        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            let graph = self.graph.borrow(py);
            let edge_ids = graph.inner.borrow().edge_ids();
            edge_ids
        };

        Ok(EdgesIterator {
            graph: self.graph.clone(),
            edge_ids,
            index: 0,
        })
```

---

### __len__

**Signature:**

```rust
fn __len__(&self, py: Python) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 145.

**Implementation:**

```rust

        if let Some(ref constrained) = self.constrained_edges {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.get_edge_count())
        }
```

---

### __str__

**Signature:**

```rust
fn __str__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 114.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let count = graph.get_edge_count();
        Ok(format!("EdgesAccessor({} edges)", count))
```

---

### attributes

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn attributes(&self, py: Python) -> PyResult<Vec<String>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 450.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let mut all_attrs = std::collections::HashSet::new();

        // Determine which edges to check
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            // Get all edge IDs from the graph
            { let edge_ids = graph.inner.borrow().edge_ids(); edge_ids }
        };

        // Collect attributes from all edges
        for &edge_id in &edge_ids {
            let attrs = graph.edge_attribute_keys(edge_id);
            for attr in attrs {
                all_attrs.insert(attr);
            }
        }

        // Convert to sorted vector
        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        Ok(result)
```

---

### table

**Signature:**

```rust
pub fn table(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 481.

**Implementation:**

```rust

        use crate::ffi::core::table::PyGraphTable;

        // Determine edge list based on constraints
        let edge_data = if let Some(ref constrained) = self.constrained_edges {
            // Subgraph case: use constrained edges
            constrained.iter().map(|&id| id as u64).collect()
        } else {
            // Full graph case: get all edge IDs from the graph
            let graph = self.graph.borrow(py);
            let edge_ids = graph
                .inner
                .borrow()
                .edge_ids()
                .into_iter()
                .map(|id| id as u64)
                .collect();
            edge_ids
        };

        // Use PyGraphTable::from_graph_edges to create the table
        let py_table = PyGraphTable::from_graph_edges(
            py.get_type::<PyGraphTable>(),
            py,
            self.graph.clone(),
            edge_data,
            None, // Get all attributes
        )?;

        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### all

**Signature:**

```rust
fn all(&self, py: Python) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 27, token-chars: 844.

**Implementation:**

```rust

        let graph = self.graph.borrow_mut(py);
        
        // Get all edge IDs (respecting constraints if any)
        let all_edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            { let edge_ids = graph.inner.borrow().edge_ids(); edge_ids }
        };
        
        // Get all nodes that are endpoints of these edges
        let (_, sources, targets) = graph.inner.borrow().get_columnar_topology();
        let mut connected_nodes = std::collections::HashSet::new();
        let edge_set: std::collections::HashSet<EdgeId> = all_edge_ids.iter().copied().collect();
        
        // Find all nodes connected by our edges
        for i in 0..all_edge_ids.len() {
            let edge_id = all_edge_ids[i];
            if edge_set.contains(&edge_id) {
                let source = sources[i];
                let target = targets[i];
                connected_nodes.insert(source);
                connected_nodes.insert(target);
            }
        }
        
        let connected_node_ids: Vec<NodeId> = connected_nodes.into_iter().collect();
        // Create and return Subgraph with inner RustSubgraph (all edges are included by construction)
        let subgraph = PySubgraph::new_with_inner(
            py,
            connected_node_ids,
            all_edge_ids,
            "all_edges".to_string(),
            Some(self.graph.clone()),
        );

        Ok(subgraph)
```

---

### __getattr__

**Signature:**

```rust
fn __getattr__(&self, _py: Python, name: &str) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 98.

**Implementation:**

```rust

        // TODO: Complete implementation - temporarily disabled
        return Err(PyKeyError::new_err(format!(
            "Edge attribute '{}' access is under development.",
            name
        )));
```

---

### _get_edge_attribute_column

**Signature:**

```rust
fn _get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 61, token-chars: 1384.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        
        // Determine which edges to check
        let edge_ids = if let Some(ref constrained) = self.constrained_edges {
            constrained.clone()
        } else {
            { let edge_ids = graph.inner.borrow().edge_ids(); edge_ids }
        };

        if edge_ids.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot access attribute '{}': No edges available", 
                attr_name
            )));
        }

        // Check if the attribute exists on ANY edge
        let mut attribute_exists = false;
        for &edge_id in &edge_ids {
            let attrs = graph.edge_attribute_keys(edge_id);
            if attrs.contains(&attr_name.to_string()) {
                attribute_exists = true;
                break;
            }
        }

        if !attribute_exists {
            return Err(PyKeyError::new_err(format!(
                "Attribute '{}' does not exist on any edges. Available attributes: {:?}",
                attr_name,
                {
                    let mut all_attrs = std::collections::HashSet::new();
                    for &edge_id in &edge_ids {
                        let attrs = graph.edge_attribute_keys(edge_id);
                        for attr in attrs {
                            all_attrs.insert(attr);
                        }
                    }
                    let mut result: Vec<String> = all_attrs.into_iter().collect();
                    result.sort();
                    result
                }
            )));
        }

        // Collect attribute values - allow None for edges without the attribute
        let mut values: Vec<Option<PyObject>> = Vec::new();
        for &edge_id in &edge_ids {
            match graph.inner.borrow().get_edge_attr(edge_id, &attr_name.to_string()) {
                Ok(Some(value)) => {
                    // Convert the attribute value to Python object
                    let py_value = attr_value_to_python_value(py, &value)?;
                    values.push(Some(py_value));
                }
                Ok(None) => {
                    values.push(None);
                }
                Err(_) => {
                    values.push(None);
                }
            }
        }

        // Convert to Python list
        let py_values: Vec<PyObject> = values
            .into_iter()
            .map(|opt_val| match opt_val {
                Some(val) => val,
                None => py.None(),
            })
            .collect();

        Ok(py_values.to_object(py))
```

---


## ffi/core/array.rs

### new

**Attributes:**

```rust
#[new]
```

**Signature:**

```rust
fn new(values: Vec<PyObject>) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 254.

**Implementation:**

```rust

        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            Ok(PyGraphArray {
                inner: GraphArray::from_vec(attr_values),
            })
        })
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 88, token-chars: 2462.

**Implementation:**

```rust

        // Single integer indexing: arr[5]
        if let Ok(index) = key.extract::<isize>() {
            let len = self.inner.len() as isize;

            // Handle negative indexing (Python-style)
            let actual_index = if index < 0 { len + index } else { index };

            // Check bounds
            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Index out of range"));
            }

            match self.inner.get(actual_index as usize) {
                Some(attr_value) => attr_value_to_python_value(py, attr_value),
                None => Err(PyIndexError::new_err("Index out of range")),
            }
        }
        // Slice indexing: arr[start:end], arr[start:end:step]
        else if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let len = self.inner.len();
            let indices = slice.indices(
                len.try_into()
                    .map_err(|_| PyValueError::new_err("Array too large for slice"))?,
            )?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step;

            let mut result_values = Vec::new();

            if step == 1 {
                // Simple slice [start:stop]
                for i in start..stop.min(len) {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            } else if step > 1 {
                // Step slice [start:stop:step]
                let mut i = start;
                while i < stop.min(len) {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                    i += step as usize;
                }
            } else {
                return Err(PyValueError::new_err("Negative step not supported"));
            }

            // Return new GraphArray with sliced data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // List of indices: arr[[1, 3, 5]]
        else if let Ok(indices) = key.extract::<Vec<isize>>() {
            let len = self.inner.len() as isize;
            let mut result_values = Vec::new();

            for &index in &indices {
                let actual_index = if index < 0 { len + index } else { index };

                if actual_index >= 0 && actual_index < len {
                    if let Some(attr_value) = self.inner.get(actual_index as usize) {
                        result_values.push(attr_value.clone());
                    }
                } else {
                    return Err(PyIndexError::new_err("Index out of range"));
                }
            }

            // Return new GraphArray with selected data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // Boolean mask indexing: arr[mask] where mask is a list of booleans
        else if let Ok(mask) = key.extract::<Vec<bool>>() {
            let len = self.inner.len();
            if mask.len() != len {
                return Err(PyValueError::new_err(
                    "Boolean mask length must match array length",
                ));
            }

            let mut result_values = Vec::new();
            for (i, &include) in mask.iter().enumerate() {
                if include {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            }

            // Return new GraphArray with masked data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        } else {
            Err(PyTypeError::new_err(
                "Index must be int, slice, list of ints, or list of bools",
            ))
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 190.

**Implementation:**

```rust

        // Try rich display formatting first, with graceful fallback
        match self._try_rich_display(py) {
            Ok(formatted) => Ok(formatted),
            Err(_) => {
                // Fallback to simple representation
                let len = self.inner.len();
                let dtype = self._get_dtype();
                Ok(format!("GraphArray(len={}, dtype={})", len, dtype))
            }
        }
```

---

### _try_rich_display

**Signature:**

```rust
fn _try_rich_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 259.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_array function from Python
        let groggy_module = py.import("groggy")?;
        let format_array = groggy_module.getattr("format_array")?;

        // Call the Python formatter
        let result = format_array.call1((display_data,))?;
        let formatted_str: String = result.extract()?;

        Ok(formatted_str)
```

---

### _repr_html_

**Signature:**

```rust
fn _repr_html_(&self, _py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 33, token-chars: 1154.

**Implementation:**

```rust

        // Simple, effective HTML display for GraphArray
        let len = self.inner.len();
        let dtype = self._get_dtype();

        if len == 0 {
            return Ok(format!(
                "<div style='font-family: monospace; color: #666;'><strong>GraphArray:</strong> 0 elements, dtype: {}</div>",
                dtype
            ));
        }

        // Show first few elements
        let display_count = std::cmp::min(len, 10);
        let mut elements = Vec::new();

        for i in 0..display_count {
            if let Some(value) = self.inner.get(i) {
                let display_value = match value {
                    groggy::AttrValue::Int(i) => i.to_string(),
                    groggy::AttrValue::SmallInt(i) => i.to_string(),
                    groggy::AttrValue::Float(f) => format!("{:.3}", f),
                    groggy::AttrValue::Text(s) => format!("'{}'", s),
                    groggy::AttrValue::Bool(b) => b.to_string(),
                    groggy::AttrValue::CompactText(cs) => format!("'{}'", cs.as_str()),
                    _ => format!("{:?}", value),
                };
                elements.push(display_value);
            }
        }

        let elements_str = elements.join(", ");
        let ellipsis = if len > display_count { ", ..." } else { "" };

        Ok(format!(
            "<div style='font-family: monospace; padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9;'>\
            <div style='font-weight: bold; margin-bottom: 4px;'>GraphArray: {} elements, dtype: {}</div>\
            <div style='color: #333;'>[{}{}]</div>\
            </div>",
            len, dtype, elements_str, ellipsis
        ))
```

---

### _try_rich_html_display

**Signature:**

```rust
fn _try_rich_html_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 320.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_array_html function from Python
        let groggy_module = py.import("groggy")?;
        let display_module = groggy_module.getattr("display")?;
        let format_array_html = display_module.getattr("format_array_html")?;

        // Call the Python HTML formatter
        let result = format_array_html.call1((display_data,))?;
        let html_str: String = result.extract()?;

        Ok(html_str)
```

---

### __iter__

**Signature:**

```rust
fn __iter__(slf: PyRef<Self>) -> GraphArrayIterator
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 55.

**Implementation:**

```rust

        GraphArrayIterator {
            array: slf.inner.clone(),
            index: 0,
        }
```

---

### to_list

**Signature:**

```rust
pub fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 168.

**Implementation:**

```rust

        let mut py_values = Vec::with_capacity(self.inner.len());

        for attr_value in self.inner.iter() {
            py_values.push(attr_value_to_python_value(py, attr_value)?);
        }

        Ok(py_values)
```

---

### min

**Signature:**

```rust
fn min(&self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 117.

**Implementation:**

```rust

        match self.inner.min() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
```

---

### max

**Signature:**

```rust
fn max(&self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 117.

**Implementation:**

```rust

        match self.inner.max() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
```

---

### count

**Signature:**

```rust
fn count(&self) -> usize
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 117.

**Implementation:**

```rust

        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .count()
```

---

### has_null

**Signature:**

```rust
fn has_null(&self) -> bool
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 105.

**Implementation:**

```rust

        let values = self.inner.materialize();
        values
            .iter()
            .any(|value| matches!(value, groggy::AttrValue::Null))
```

---

### null_count

**Signature:**

```rust
fn null_count(&self) -> usize
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 116.

**Implementation:**

```rust

        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| matches!(value, groggy::AttrValue::Null))
            .count()
```

---

### drop_na

**Signature:**

```rust
fn drop_na(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 347.

**Implementation:**

```rust

        let values = self.inner.materialize();
        let non_null_values: Vec<groggy::AttrValue> = values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .cloned()
            .collect();

        let new_array = groggy::core::array::GraphArray::from_vec(non_null_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
```

---

### fill_na

**Signature:**

```rust
fn fill_na(&self, py: Python, fill_value: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 463.

**Implementation:**

```rust

        let fill_attr_value = crate::ffi::utils::python_value_to_attr_value(fill_value)?;
        let values = self.inner.materialize();

        let filled_values: Vec<groggy::AttrValue> = values
            .iter()
            .map(|value| {
                if matches!(value, groggy::AttrValue::Null) {
                    fill_attr_value.clone()
                } else {
                    value.clone()
                }
            })
            .collect();

        let new_array = groggy::core::array::GraphArray::from_vec(filled_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
```

---

### percentile

**Signature:**

```rust
fn percentile(&self, p: f64) -> Option<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 68.

**Implementation:**

```rust

        if p < 0.0 || p > 100.0 {
            return None;
        }
        self.inner.quantile(p / 100.0)
```

---

### unique

**Signature:**

```rust
fn unique(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 23, token-chars: 827.

**Implementation:**

```rust

        use std::collections::HashSet;

        // Use HashSet to find unique values
        let mut unique_set = HashSet::new();
        let mut unique_values = Vec::new();

        for attr_value in self.inner.iter() {
            // Create a simple hash key based on the value
            let key = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()), // Simple hash for bytes
                _ => format!("other:{:?}", attr_value),                  // Fallback for other types
            };

            if unique_set.insert(key) {
                // This is a new unique value
                unique_values.push(attr_value.clone());
            }
        }

        // Create new GraphArray with unique values
        let unique_array = GraphArray::from_vec(unique_values);
        let py_unique = PyGraphArray {
            inner: unique_array,
        };

        Ok(Py::new(py, py_unique)?)
```

---

### value_counts

**Signature:**

```rust
fn value_counts(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 26, token-chars: 874.

**Implementation:**

```rust

        use std::collections::HashMap;

        let mut counts: HashMap<String, (i32, RustAttrValue)> = HashMap::new();

        for attr_value in self.inner.iter() {
            // Create a string representation for HashMap key
            let key_str = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()),
                _ => format!("other:{:?}", attr_value),
            };

            match counts.get_mut(&key_str) {
                Some((count, _)) => *count += 1,
                None => {
                    counts.insert(key_str, (1, attr_value.clone()));
                }
            }
        }

        // Convert to Python dict
        let dict = pyo3::types::PyDict::new(py);
        for (_, (count, attr_value)) in counts {
            let py_key = attr_value_to_python_value(py, &attr_value)?;
            dict.set_item(py_key, count)?;
        }

        Ok(dict.to_object(py))
```

---

### describe

**Signature:**

```rust
fn describe(&self, _py: Python) -> PyResult<PyStatsSummary>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 50.

**Implementation:**

```rust

        Ok(PyStatsSummary {
            inner: self.inner.describe(),
        })
```

---

### values

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn values(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 190.

**Implementation:**

```rust

        let materialized = self.inner.materialize();
        let py_values: PyResult<Vec<PyObject>> = materialized
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
```

---

### preview

**Signature:**

```rust
fn preview(&self, py: Python, limit: Option<usize>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 227.

**Implementation:**

```rust

        let limit = limit.unwrap_or(10);
        let preview_values = self.inner.preview(limit);
        let py_values: PyResult<Vec<PyObject>> = preview_values
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
```

---

### to_numpy

**Signature:**

```rust
fn to_numpy(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 253.

**Implementation:**

```rust

        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .values property
        let values = self.values(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (values,))?;
        Ok(array.to_object(py))
```

---

### to_pandas

**Signature:**

```rust
fn to_pandas(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 262.

**Implementation:**

```rust

        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Create Series
        let series = pandas.call_method1("Series", (values,))?;
        Ok(series.to_object(py))
```

---

### to_scipy_sparse

**Signature:**

```rust
fn to_scipy_sparse(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 385.

**Implementation:**

```rust

        // Try to import scipy.sparse
        let scipy_sparse = py.import("scipy.sparse").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "scipy is required for to_scipy_sparse(). Install with: pip install scipy",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Convert to numpy first, then to sparse
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (values,))?;

        // Create CSR matrix (compressed sparse row) from the dense array
        let sparse_matrix = scipy_sparse.call_method1("csr_matrix", (array,))?;
        Ok(sparse_matrix.to_object(py))
```

---

### _get_display_data

**Signature:**

```rust
fn _get_display_data(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 288.

**Implementation:**

```rust

        let dict = pyo3::types::PyDict::new(py);

        // Extract array data - convert to Python list
        let data = self.to_list(py)?;
        dict.set_item("data", data)?;

        // Get array metadata
        dict.set_item("shape", (self.inner.len(),))?;
        dict.set_item("dtype", self._get_dtype())?;
        dict.set_item(
            "name",
            self._get_name().unwrap_or_else(|| "array".to_string()),
        )?;

        Ok(dict.to_object(py))
```

---

### _get_dtype

**Signature:**

```rust
fn _get_dtype(&self) -> String
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 617.

**Implementation:**

```rust

        // Sample first few elements to determine predominant type
        let sample_size = std::cmp::min(self.inner.len(), 5);
        if sample_size == 0 {
            return "object".to_string();
        }

        let mut type_counts = std::collections::HashMap::new();

        for i in 0..sample_size {
            let type_name = match &self.inner[i] {
                RustAttrValue::Int(_) | RustAttrValue::SmallInt(_) => "int64",
                RustAttrValue::Float(_) => "f32",
                RustAttrValue::Bool(_) => "bool",
                RustAttrValue::Text(_) | RustAttrValue::CompactText(_) => "str",
                _ => "object",
            };
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        // Return the most common type
        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, _)| type_name.to_string())
            .unwrap_or_else(|| "object".to_string())
```

---

### __gt__

**Signature:**

```rust
fn __gt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 48, token-chars: 1692.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a > b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a > b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a > b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) > (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) > *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) > (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() > b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a > b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### __lt__

**Signature:**

```rust
fn __lt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 48, token-chars: 1692.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a < b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a < b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a < b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) < (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) < *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) < (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() < b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a < b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### __ge__

**Signature:**

```rust
fn __ge__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 49, token-chars: 1674.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a >= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a >= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a >= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) >= (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) >= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) >= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a >= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Comparison not supported for these types",
                    ))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### __le__

**Signature:**

```rust
fn __le__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 50, token-chars: 1707.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a <= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a <= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a <= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) <= (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) <= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) <= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a <= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### __eq__

**Signature:**

```rust
fn __eq__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 298.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value == &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### __ne__

**Signature:**

```rust
fn __ne__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 298.

**Implementation:**

```rust

        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value != &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
```

---

### min

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn min(&self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 115.

**Implementation:**

```rust

        match &self.inner.min {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
```

---

### max

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn max(&self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 115.

**Implementation:**

```rust

        match &self.inner.max {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
```

---

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 161.

**Implementation:**

```rust

        if self.index < self.array.len() {
            let attr_value = &self.array[self.index];
            self.index += 1;
            Ok(Some(attr_value_to_python_value(py, attr_value)?))
        } else {
            Ok(None)
        }
```

---

### from_py_objects

**Signature:**

```rust
pub fn from_py_objects(values: Vec<PyObject>) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 254.

**Implementation:**

```rust

        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            Ok(PyGraphArray {
                inner: GraphArray::from_vec(attr_values),
            })
        })
```

---


## ffi/core/attributes.rs

### get_node_attribute

**Signature:**

```rust
pub fn get_node_attribute(
        graph: &groggy::Graph,
        node: NodeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 160.

**Implementation:**

```rust

        match graph.get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_edge_attribute

**Signature:**

```rust
pub fn get_edge_attribute(
        graph: &groggy::Graph,
        edge: EdgeId,
        attr: AttrName,
    ) -> PyResult<Option<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 160.

**Implementation:**

```rust

        match graph.get_edge_attr(edge, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_node_attribute_column

**Signature:**

```rust
pub fn get_node_attribute_column(
        graph: &groggy::Graph,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 437.

**Implementation:**

```rust

        match graph._get_node_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<RustAttrValue> = values
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(RustAttrValue::Int(0))) // Use default for None values
                    .collect();

                // Create GraphArray from the attribute values
                let graph_array = groggy::core::array::GraphArray::from_vec(attr_values);

                // Wrap in Python GraphArray
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_edge_attribute_column

**Signature:**

```rust
pub fn get_edge_attribute_column(
        graph: &groggy::Graph,
        py: Python,
        attr_name: &str,
    ) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 339.

**Implementation:**

```rust

        match graph._get_edge_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => {
                            py_values.push(attr_value_to_python_value(py, &attr_value)?)
                        }
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### get_node_attributes_for_nodes

**Signature:**

```rust
pub fn get_node_attributes_for_nodes(
        graph: &groggy::Graph,
        py: Python,
        node_ids: &[NodeId],
        attr_name: &str,
    ) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 353.

**Implementation:**

```rust

        match graph._get_node_attributes_for_nodes(node_ids, &attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => {
                            py_values.push(attr_value_to_python_value(py, &attr_value)?)
                        }
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
```

---

### set_node_attributes_from_dict

**Signature:**

```rust
pub fn set_node_attributes_from_dict(
        graph: &mut groggy::Graph,
        node_id: NodeId,
        attrs: &PyDict,
    ) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 216.

**Implementation:**

```rust

        for (key, value) in attrs.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            graph
                .set_node_attr(node_id, attr_name, attr_value)
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
```

---

### set_edge_attributes_from_dict

**Signature:**

```rust
pub fn set_edge_attributes_from_dict(
        graph: &mut groggy::Graph,
        edge_id: EdgeId,
        attrs: &PyDict,
    ) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 216.

**Implementation:**

```rust

        for (key, value) in attrs.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            graph
                .set_edge_attr(edge_id, attr_name, attr_value)
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
```

---

### get_node_attributes_dict

**Signature:**

```rust
pub fn get_node_attributes_dict(
        graph: &groggy::Graph,
        py: Python,
        node_id: NodeId,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 284.

**Implementation:**

```rust

        let dict = PyDict::new(py);

        // Get all attributes for this node from the core graph
        match graph.get_node_attrs(node_id) {
            Ok(attrs) => {
                for (attr_name, attr_value) in attrs {
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    dict.set_item(attr_name, py_value)?;
                }
            }
            Err(e) => return Err(graph_error_to_py_err(e)),
        }

        Ok(dict.to_object(py))
```

---

### get_edge_attributes_dict

**Signature:**

```rust
pub fn get_edge_attributes_dict(
        graph: &groggy::Graph,
        py: Python,
        edge_id: EdgeId,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 284.

**Implementation:**

```rust

        let dict = PyDict::new(py);

        // Get all attributes for this edge from the core graph
        match graph.get_edge_attrs(edge_id) {
            Ok(attrs) => {
                for (attr_name, attr_value) in attrs {
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    dict.set_item(attr_name, py_value)?;
                }
            }
            Err(e) => return Err(graph_error_to_py_err(e)),
        }

        Ok(dict.to_object(py))
```

---

### set_node_attribute_bulk

**Signature:**

```rust
pub fn set_node_attribute_bulk(
        graph: &mut groggy::Graph,
        attr_name: &str,
        attr_value: RustAttrValue,
        node_ids: &[NodeId],
    ) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 137.

**Implementation:**

```rust

        for &node_id in node_ids {
            graph
                .set_node_attr(node_id, attr_name.to_string(), attr_value.clone())
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
```

---

### set_edge_attribute_bulk

**Signature:**

```rust
pub fn set_edge_attribute_bulk(
        graph: &mut groggy::Graph,
        attr_name: &str,
        attr_value: RustAttrValue,
        edge_ids: &[EdgeId],
    ) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 137.

**Implementation:**

```rust

        for &edge_id in edge_ids {
            graph
                .set_edge_attr(edge_id, attr_name.to_string(), attr_value.clone())
                .map_err(graph_error_to_py_err)?;
        }
        Ok(())
```

---


## ffi/core/history.rs

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 104.

**Implementation:**

```rust

        format!(
            "Commit(id={}, message='{}', author='{}')",
            self.inner.id, self.inner.message, self.inner.author
        )
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 74.

**Implementation:**

```rust

        format!(
            "BranchInfo(name='{}', head={})",
            self.inner.name, self.inner.head
        )
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 154.

**Implementation:**

```rust

        format!(
            "HistoryStatistics(commits={}, branches={}, efficiency={:.2})",
            self.inner.total_commits, self.inner.total_branches, self.inner.storage_efficiency
        )
```

---

### get_node_ids

**Signature:**

```rust
fn get_node_ids(&self) -> PyResult<Vec<NodeId>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 14.

**Implementation:**

```rust

        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
```

---

### get_edge_ids

**Signature:**

```rust
fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 14.

**Implementation:**

```rust

        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
```

---


## ffi/core/matrix.rs

### new

**Attributes:**

```rust
#[new]
```

**Signature:**

```rust
pub fn new(py: Python, arrays: Vec<Py<PyGraphArray>>) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 269.

**Implementation:**

```rust

        // Convert PyGraphArrays to core GraphArrays
        let core_arrays: Vec<GraphArray> = arrays
            .iter()
            .map(|py_array| py_array.borrow(py).inner.clone())
            .collect();

        // Create core GraphMatrix
        let matrix = GraphMatrix::from_arrays(core_arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create matrix: {:?}", e)))?;

        Ok(Self { inner: matrix })
```

---

### zeros

**Attributes:**

```rust
#[classmethod]
```

**Signature:**

```rust
fn zeros(
        _cls: &PyType,
        py: Python,
        rows: usize,
        cols: usize,
        dtype: Option<&str>,
    ) -> PyResult<Py<Self>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 14, token-chars: 447.

**Implementation:**

```rust

        // Parse dtype string to AttrValueType
        let attr_type = match dtype.unwrap_or("float") {
            "int" | "int64" => groggy::AttrValueType::Int,
            "float" | "float64" | "f64" => groggy::AttrValueType::Float,
            "bool" => groggy::AttrValueType::Bool,
            "str" | "string" | "text" => groggy::AttrValueType::Text,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype: {}",
                    dtype.unwrap_or("unknown")
                )))
            }
        };

        let matrix = GraphMatrix::zeros(rows, cols, attr_type);
        Ok(Py::new(py, Self { inner: matrix })?)
```

---

### identity

**Attributes:**

```rust
#[classmethod]
```

**Signature:**

```rust
fn identity(_cls: &PyType, py: Python, size: usize) -> PyResult<Py<Self>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 81.

**Implementation:**

```rust

        let matrix = GraphMatrix::identity(size);
        Ok(Py::new(py, Self { inner: matrix })?)
```

---

### from_graph_attributes

**Attributes:**

```rust
#[classmethod]
```

**Signature:**

```rust
fn from_graph_attributes(
        _cls: &PyType,
        _py: Python,
        _graph: PyObject,
        _attrs: Vec<String>,
        _entities: Vec<u64>,
    ) -> PyResult<Py<Self>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 88.

**Implementation:**

```rust

        // TODO: Implement graph integration in Phase 2
        // For now, return a placeholder error
        Err(PyNotImplementedError::new_err(
            "Graph integration not yet implemented in Phase 2",
        ))
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 414.

**Implementation:**

```rust

        // Multi-index access (row, col) -> single cell value
        if let Ok(indices) = key.extract::<(usize, usize)>() {
            let (row, col) = indices;
            return self.get_cell(py, row, col);
        }

        // Single integer -> row access
        if let Ok(row_index) = key.extract::<usize>() {
            return self.get_row(py, row_index);
        }

        // String -> column access
        if let Ok(col_name) = key.extract::<String>() {
            return self.get_column_by_name(py, col_name);
        }

        Err(PyTypeError::new_err(
            "Key must be: int (row index), string (column name), or (row, col) tuple for multi-index access"
        ))
```

---

### get_cell

**Signature:**

```rust
fn get_cell(&self, py: Python, row: usize, col: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 248.

**Implementation:**

```rust

        match self.inner.get(row, col) {
            Some(attr_value) => attr_value_to_python_value(py, attr_value),
            None => {
                let (rows, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Index ({}, {}) out of range for {}x{} matrix",
                    row, col, rows, cols
                )))
            }
        }
```

---

### get_row

**Signature:**

```rust
fn get_row(&self, py: Python, row: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 281.

**Implementation:**

```rust

        match self.inner.get_row(row) {
            Some(row_array) => {
                let py_array = PyGraphArray::from_graph_array(row_array);
                Ok(Py::new(py, py_array)?.to_object(py))
            }
            None => {
                let (rows, _) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Row index {} out of range for {} rows",
                    row, rows
                )))
            }
        }
```

---

### get_column_by_name

**Signature:**

```rust
fn get_column_by_name(&self, py: Python, name: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 238.

**Implementation:**

```rust

        match self.inner.get_column_by_name(&name) {
            Some(column) => {
                let py_array = PyGraphArray::from_graph_array(column.clone());
                Ok(Py::new(py, py_array)?.to_object(py))
            }
            None => Err(PyKeyError::new_err(format!("Column '{}' not found", name))),
        }
```

---

### get_column

**Signature:**

```rust
fn get_column(&self, py: Python, col: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 292.

**Implementation:**

```rust

        match self.inner.get_column(col) {
            Some(column) => {
                let py_array = PyGraphArray::from_graph_array(column.clone());
                Ok(Py::new(py, py_array)?.to_object(py))
            }
            None => {
                let (_, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Column index {} out of range for {} columns",
                    col, cols
                )))
            }
        }
```

---

### iter_rows

**Signature:**

```rust
fn iter_rows(&self, py: Python) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 350.

**Implementation:**

```rust

        let (rows, _) = self.inner.shape();
        let mut row_arrays = Vec::with_capacity(rows);

        for i in 0..rows {
            match self.inner.get_row(i) {
                Some(row_array) => {
                    let py_array = PyGraphArray::from_graph_array(row_array);
                    row_arrays.push(Py::new(py, py_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Row {} not found", i))),
            }
        }

        Ok(row_arrays)
```

---

### iter_columns

**Signature:**

```rust
fn iter_columns(&self, py: Python) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 364.

**Implementation:**

```rust

        let (_, cols) = self.inner.shape();
        let mut col_arrays = Vec::with_capacity(cols);

        for i in 0..cols {
            match self.inner.get_column(i) {
                Some(col_array) => {
                    let py_array = PyGraphArray::from_graph_array(col_array.clone());
                    col_arrays.push(Py::new(py, py_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Column {} not found", i))),
            }
        }

        Ok(col_arrays)
```

---

### transpose

**Signature:**

```rust
fn transpose(&self, py: Python) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 93.

**Implementation:**

```rust

        let transposed = self.inner.transpose();
        Ok(Py::new(py, PyGraphMatrix { inner: transposed })?)
```

---

### multiply

**Signature:**

```rust
fn multiply(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 235.

**Implementation:**

```rust

        let result_matrix = self.inner.multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Ok(Py::new(py, py_result)?)
```

---

### inverse

**Signature:**

```rust
fn inverse(&self) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 85.

**Implementation:**

```rust

        Err(PyNotImplementedError::new_err(
            "Matrix inverse will be implemented in Phase 5",
        ))
```

---

### power

**Signature:**

```rust
fn power(&self, py: Python, n: u32) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 210.

**Implementation:**

```rust

        let result_matrix = self
            .inner
            .power(n)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix power failed: {:?}", e)))?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Ok(Py::new(py, py_result)?)
```

---

### elementwise_multiply

**Signature:**

```rust
fn elementwise_multiply(
        &self,
        py: Python,
        other: &PyGraphMatrix,
    ) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 252.

**Implementation:**

```rust

        let result_matrix = self.inner.elementwise_multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Elementwise multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Ok(Py::new(py, py_result)?)
```

---

### determinant

**Signature:**

```rust
fn determinant(&self) -> PyResult<Option<f64>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 94.

**Implementation:**

```rust

        Err(PyNotImplementedError::new_err(
            "Determinant calculation will be implemented in Phase 5",
        ))
```

---

### sum_axis

**Signature:**

```rust
fn sum_axis(&self, py: Python, axis: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 340.

**Implementation:**

```rust

        let axis_enum = match axis {
            0 => groggy::core::matrix::Axis::Rows,
            1 => groggy::core::matrix::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.sum_axis(axis_enum);
        let py_array = PyGraphArray::from_graph_array(result_array);
        Ok(Py::new(py, py_array)?.to_object(py))
```

---

### mean_axis

**Signature:**

```rust
fn mean_axis(&self, py: Python, axis: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 341.

**Implementation:**

```rust

        let axis_enum = match axis {
            0 => groggy::core::matrix::Axis::Rows,
            1 => groggy::core::matrix::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.mean_axis(axis_enum);
        let py_array = PyGraphArray::from_graph_array(result_array);
        Ok(Py::new(py, py_array)?.to_object(py))
```

---

### std_axis

**Signature:**

```rust
fn std_axis(&self, py: Python, axis: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 340.

**Implementation:**

```rust

        let axis_enum = match axis {
            0 => groggy::core::matrix::Axis::Rows,
            1 => groggy::core::matrix::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.std_axis(axis_enum);
        let py_array = PyGraphArray::from_graph_array(result_array);
        Ok(Py::new(py, py_array)?.to_object(py))
```

---

### to_pandas

**Signature:**

```rust
fn to_pandas(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 27, token-chars: 710.

**Implementation:**

```rust

        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Convert to dictionary of column name -> values
        let dict = pyo3::types::PyDict::new(py);
        let column_names = self.inner.column_names();
        let (_, cols) = self.inner.shape();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            if col_idx < cols {
                match self.inner.get_column(col_idx) {
                    Some(column) => {
                        let py_array = PyGraphArray::from_graph_array(column.clone());
                        let values = py_array.to_list(py)?;
                        dict.set_item(col_name, values)?;
                    }
                    None => {
                        return Err(PyIndexError::new_err(format!(
                            "Column {} not found",
                            col_idx
                        )))
                    }
                }
            }
        }

        // Create DataFrame
        let dataframe = pandas.call_method1("DataFrame", (dict,))?;
        Ok(dataframe.to_object(py))
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 207.

**Implementation:**

```rust

        // Try rich display formatting first
        match self._try_rich_display(py) {
            Ok(formatted) => Ok(formatted),
            Err(_) => {
                // Fallback to simple representation
                let (rows, cols) = self.inner.shape();
                Ok(format!(
                    "GraphMatrix({} x {}, dtype={})",
                    rows,
                    cols,
                    format!("{:?}", self.inner.dtype())
                ))
            }
        }
```

---

### _repr_html_

**Signature:**

```rust
fn _repr_html_(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 316.

**Implementation:**

```rust

        match self._try_rich_html_display(py) {
            Ok(html) => Ok(html),
            Err(_) => {
                // Fallback to basic HTML
                let (rows, cols) = self.inner.shape();
                Ok(format!(
                    r#"<div style="font-family: monospace; padding: 10px; border: 1px solid #ddd;">
                    <strong>GraphMatrix</strong><br>
                    Shape: {} x {}<br>
                    Dtype: {}
                    </div>"#,
                    rows,
                    cols,
                    format!("{:?}", self.inner.dtype())
                ))
            }
        }
```

---

### _try_rich_display

**Signature:**

```rust
fn _try_rich_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 262.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_matrix function from Python
        let groggy_module = py.import("groggy")?;
        let format_matrix = groggy_module.getattr("format_matrix")?;

        // Call the Python formatter
        let result = format_matrix.call1((display_data,))?;
        let formatted_str: String = result.extract()?;

        Ok(formatted_str)
```

---

### _try_rich_html_display

**Signature:**

```rust
fn _try_rich_html_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 323.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_matrix_html function from Python
        let groggy_module = py.import("groggy")?;
        let display_module = groggy_module.getattr("display")?;
        let format_matrix_html = display_module.getattr("format_matrix_html")?;

        // Call the Python HTML formatter
        let result = format_matrix_html.call1((display_data,))?;
        let html_str: String = result.extract()?;

        Ok(html_str)
```

---

### _get_display_data

**Signature:**

```rust
fn _get_display_data(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 651.

**Implementation:**

```rust

        let dict = pyo3::types::PyDict::new(py);
        let (rows, cols) = self.inner.shape();

        // Convert matrix to nested list for display
        let mut matrix_data = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row_data = Vec::with_capacity(cols);
            for j in 0..cols {
                match self.inner.get(i, j) {
                    Some(attr_value) => {
                        row_data.push(attr_value_to_python_value(py, attr_value)?);
                    }
                    None => row_data.push(py.None()),
                }
            }
            matrix_data.push(row_data);
        }

        dict.set_item("data", matrix_data)?;
        dict.set_item("shape", (rows, cols))?;
        dict.set_item("dtype", self.dtype())?;
        dict.set_item("columns", self.columns())?;
        dict.set_item("is_square", self.is_square())?;
        dict.set_item("is_symmetric", self.is_symmetric())?;

        Ok(dict.to_object(py))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(_slf: PyRef<Self>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 149.

**Implementation:**

```rust

        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Matrix iteration temporarily disabled during Phase 3 - use matrix[i] for row access",
        ))
```

---

### data

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn data(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 229.

**Implementation:**

```rust

        let materialized = self.inner.materialize();
        let py_matrix: PyResult<Vec<Vec<PyObject>>> = materialized
            .iter()
            .map(|row| {
                row.iter()
                    .map(|val| attr_value_to_python_value(py, val))
                    .collect()
            })
            .collect();

        Ok(py_matrix?.to_object(py))
```

---

### preview

**Signature:**

```rust
fn preview(
        &self,
        py: Python,
        row_limit: Option<usize>,
        col_limit: Option<usize>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 184.

**Implementation:**

```rust

        let row_limit = row_limit.unwrap_or(10);
        let col_limit = col_limit.unwrap_or(10);
        let (preview_data, _col_names) = self.inner.preview(row_limit, col_limit);

        Ok(preview_data.to_object(py))
```

---

### dense

**Signature:**

```rust
fn dense(&self, py: Python) -> PyResult<Py<PyGraphMatrix>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 128.

**Implementation:**

```rust

        let dense_matrix = self.inner.dense();
        let py_result = PyGraphMatrix::from_graph_matrix(dense_matrix);
        Ok(Py::new(py, py_result)?)
```

---

### to_numpy

**Signature:**

```rust
fn to_numpy(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 247.

**Implementation:**

```rust

        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .data property
        let data = self.data(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (data,))?;
        Ok(array.to_object(py))
```

---


## ffi/core/neighborhood.rs

### subgraph

**Signature:**

```rust
fn subgraph(&self, py: Python) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 274.

**Implementation:**

```rust

        // Create PySubgraph using the same pattern as connected components
        let subgraph = PySubgraph::new_with_inner(
            py,
            self.inner.node_set().iter().copied().collect(),
            self.inner.edge_set().iter().copied().collect(),
            format!("neighborhood_hops_{}", self.inner.hops()),
            None, // No parent graph reference needed for neighborhood subgraphs
        );
        Ok(subgraph)
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 175.

**Implementation:**

```rust

        format!(
            "NeighborhoodSubgraph(central_nodes={:?}, hops={}, size={}, edges={})",
            self.inner.central_nodes(), self.inner.hops(), self.inner.node_count(), self.inner.edge_count()
        )
```

---

### __str__

**Signature:**

```rust
fn __str__(&self) -> String
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 379.

**Implementation:**

```rust

        if self.inner.central_nodes().len() == 1 {
            format!(
                "Neighborhood of node {} ({}-hop, {} nodes, {} edges)",
                self.inner.central_nodes()[0], self.inner.hops(), self.inner.node_count(), self.inner.edge_count()
            )
        } else {
            format!(
                "Neighborhood of {} nodes ({}-hop, {} nodes, {} edges)",
                self.inner.central_nodes().len(), self.inner.hops(), self.inner.node_count(), self.inner.edge_count()
            )
        }
```

---

### neighborhoods

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn neighborhoods(&self) -> Vec<PyNeighborhoodSubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 94.

**Implementation:**

```rust

        self.inner
            .neighborhoods
            .iter()
            .map(|n| PyNeighborhoodSubgraph { inner: n.clone() })
            .collect()
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, index: usize) -> PyResult<PyNeighborhoodSubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 257.

**Implementation:**

```rust

        self.inner
            .neighborhoods
            .get(index)
            .map(|n| PyNeighborhoodSubgraph { inner: n.clone() })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} out of range for neighborhoods with length {}",
                    index,
                    self.inner.neighborhoods.len()
                ))
            })
```

---

### __iter__

**Signature:**

```rust
fn __iter__(slf: PyRef<Self>) -> PyNeighborhoodResultIterator
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 65.

**Implementation:**

```rust

        PyNeighborhoodResultIterator {
            inner: slf.inner.clone(),
            index: 0,
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 171.

**Implementation:**

```rust

        format!(
            "NeighborhoodResult({} neighborhoods, largest_size={}, time={:.2}ms)",
            self.inner.total_neighborhoods,
            self.inner.largest_neighborhood_size,
            self.execution_time_ms()
        )
```

---

### __next__

**Signature:**

```rust
fn __next__(mut slf: PyRefMut<Self>) -> Option<PyNeighborhoodSubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 175.

**Implementation:**

```rust

        if slf.index < slf.inner.neighborhoods.len() {
            let result = PyNeighborhoodSubgraph {
                inner: slf.inner.neighborhoods[slf.index].clone(),
            };
            slf.index += 1;
            Some(result)
        } else {
            None
        }
```

---

### avg_nodes_per_neighborhood

**Signature:**

```rust
fn avg_nodes_per_neighborhood(&self) -> f64
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 128.

**Implementation:**

```rust

        if self.inner.total_neighborhoods > 0 {
            self.inner.total_nodes_sampled as f64 / self.inner.total_neighborhoods as f64
        } else {
            0.0
        }
```

---

### avg_time_per_neighborhood_ms

**Signature:**

```rust
fn avg_time_per_neighborhood_ms(&self) -> f64
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 111.

**Implementation:**

```rust

        if self.inner.total_neighborhoods > 0 {
            self.total_time_ms() / self.inner.total_neighborhoods as f64
        } else {
            0.0
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 208.

**Implementation:**

```rust

        format!(
            "NeighborhoodStats(neighborhoods={}, nodes={}, time={:.2}ms, avg={:.1} nodes/nbh)",
            self.inner.total_neighborhoods,
            self.inner.total_nodes_sampled,
            self.total_time_ms(),
            self.avg_nodes_per_neighborhood()
        )
```

---


## ffi/core/query.rs

### equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn equals(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 125.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::Equals(attr_value),
        })
```

---

### greater_than

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn greater_than(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 130.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::GreaterThan(attr_value),
        })
```

---

### less_than

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn less_than(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 127.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::LessThan(attr_value),
        })
```

---

### not_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn not_equals(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 128.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::NotEquals(attr_value),
        })
```

---

### greater_than_or_equal

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn greater_than_or_equal(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 137.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::GreaterThanOrEqual(attr_value),
        })
```

---

### less_than_or_equal

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn less_than_or_equal(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 134.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::LessThanOrEqual(attr_value),
        })
```

---

### attribute_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 143.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: NodeFilter::AttributeEquals {
                name,
                value: attr_value,
            },
        })
```

---

### attribute_filter

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 79.

**Implementation:**

```rust

        Self {
            inner: NodeFilter::AttributeFilter {
                name,
                filter: filter.inner.clone(),
            },
        }
```

---

### and_filters

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn and_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 130.

**Implementation:**

```rust

        let rust_filters: Vec<NodeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: NodeFilter::And(rust_filters),
        }
```

---

### or_filters

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn or_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 129.

**Implementation:**

```rust

        let rust_filters: Vec<NodeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: NodeFilter::Or(rust_filters),
        }
```

---

### attribute_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 143.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name,
                value: attr_value,
            },
        })
```

---

### attribute_filter

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 79.

**Implementation:**

```rust

        Self {
            inner: EdgeFilter::AttributeFilter {
                name,
                filter: filter.inner.clone(),
            },
        }
```

---

### and_filters

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn and_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 130.

**Implementation:**

```rust

        let rust_filters: Vec<EdgeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: EdgeFilter::And(rust_filters),
        }
```

---

### or_filters

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn or_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 129.

**Implementation:**

```rust

        let rust_filters: Vec<EdgeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: EdgeFilter::Or(rust_filters),
        }
```

---

### source_attribute_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn source_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 171.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // This is a convenience method that will need to be implemented in the core
        // For now, we'll create a combination filter (this is a placeholder)
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name: format!("source_{}", name),
                value: attr_value,
            },
        })
```

---

### target_attribute_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn target_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 171.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // This is a convenience method that will need to be implemented in the core
        // For now, we'll create a combination filter (this is a placeholder)
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name: format!("target_{}", name),
                value: attr_value,
            },
        })
```

---

### source_or_target_attribute_equals

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn source_or_target_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 352.

**Implementation:**

```rust

        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // Create an OR filter combining source and target attribute filters
        let source_filter = EdgeFilter::AttributeEquals {
            name: format!("source_{}", name),
            value: attr_value.clone(),
        };
        let target_filter = EdgeFilter::AttributeEquals {
            name: format!("target_{}", name),
            value: attr_value,
        };

        Ok(Self {
            inner: EdgeFilter::Or(vec![source_filter, target_filter]),
        })
```

---

### source_or_target_attribute_in

**Attributes:**

```rust
#[staticmethod]
```

**Signature:**

```rust
fn source_or_target_attribute_in(name: AttrName, values: Vec<&PyAny>) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 441.

**Implementation:**

```rust

        let mut filters = Vec::new();

        for value in values {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
            let source_filter = EdgeFilter::AttributeEquals {
                name: format!("source_{}", name),
                value: attr_value.clone(),
            };
            let target_filter = EdgeFilter::AttributeEquals {
                name: format!("target_{}", name),
                value: attr_value,
            };
            filters.push(EdgeFilter::Or(vec![source_filter, target_filter]));
        }

        // Combine all value filters with OR
        Ok(Self {
            inner: EdgeFilter::Or(filters),
        })
```

---


## ffi/core/subgraph.rs

### python_value_to_attr_value

**Signature:**

```rust
fn python_value_to_attr_value(value: &PyAny) -> PyResult<AttrValue>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 400.

**Implementation:**

```rust

    if let Ok(int_val) = value.extract::<i64>() {
        Ok(AttrValue::Int(int_val))
    } else if let Ok(float_val) = value.extract::<f64>() {
        Ok(AttrValue::Float(float_val as f32))
    } else if let Ok(str_val) = value.extract::<String>() {
        Ok(AttrValue::Text(str_val))
    } else if let Ok(bool_val) = value.extract::<bool>() {
        Ok(AttrValue::Bool(bool_val))
    } else {
        Err(PyTypeError::new_err("Unsupported attribute value type"))
    }
```

---

### attr_value_to_python_value

**Signature:**

```rust
fn attr_value_to_python_value(py: Python, attr_value: &AttrValue) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 344.

**Implementation:**

```rust

    match attr_value {
        AttrValue::Int(val) => Ok(val.to_object(py)),
        AttrValue::SmallInt(val) => Ok((*val as i64).to_object(py)),
        AttrValue::Float(val) => Ok(val.to_object(py)),
        AttrValue::Bool(val) => Ok(val.to_object(py)),
        AttrValue::Text(val) => Ok(val.to_object(py)),
        AttrValue::CompactText(val) => Ok(val.as_str().to_object(py)),
        _ => Ok(py.None()),
    }
```

---

### from_core_subgraph

**Signature:**

```rust
pub fn from_core_subgraph(subgraph: RustSubgraph) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 10, token-chars: 227.

**Implementation:**

```rust

        let nodes = subgraph.node_ids();
        let edges = subgraph.edge_ids();
        let subgraph_type = subgraph.subgraph_type().to_string();

        PySubgraph {
            inner: Some(subgraph),
            nodes,
            edges,
            subgraph_type,
            graph: None, // Not needed when we have inner
        }
```

---

### new

**Signature:**

```rust
pub fn new(
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        subgraph_type: String,
        graph: Option<Py<PyGraph>>,
    ) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 7, token-chars: 101.

**Implementation:**

```rust

        // SAFE: Never try to acquire GIL - always create basic structure
        // Inner RustSubgraph will be created on-demand when actually needed
        PySubgraph {
            inner: None,  // Lazy initialization to avoid GIL issues
            nodes,
            edges,
            subgraph_type,
            graph,
        }
```

---

### new_with_inner

**Signature:**

```rust
pub fn new_with_inner(
        py: Python,
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        subgraph_type: String,
        graph: Option<Py<PyGraph>>,
    ) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 337.

**Implementation:**

```rust

        // Safe to create RustSubgraph since we have Python context
        let inner = if let Some(ref graph_py) = graph {
            let graph = graph_py.borrow(py);
            let graph_ref = graph.get_graph_ref();
            
            let rust_subgraph = RustSubgraph::new(
                graph_ref,
                nodes.iter().copied().collect(),
                edges.iter().copied().collect(),
                subgraph_type.clone(),
            );
            
            Some(rust_subgraph)
        } else {
            None
        };
        
        PySubgraph {
            inner,
            nodes,
            edges,
            subgraph_type,
            graph,
        }
```

---

### ensure_inner

**Signature:**

```rust
fn ensure_inner(&mut self, py: Python) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 14, token-chars: 327.

**Implementation:**

```rust

        if self.inner.is_none() {
            if let Some(ref graph_py) = self.graph {
                let graph = graph_py.borrow(py);
                let graph_ref = graph.get_graph_ref();
                
                let rust_subgraph = RustSubgraph::new(
                    graph_ref,
                    self.nodes.iter().copied().collect(),
                    self.edges.iter().copied().collect(),
                    self.subgraph_type.clone(),
                );
                
                self.inner = Some(rust_subgraph);
            }
        }
        Ok(())
```

---

### nodes

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 209.

**Implementation:**

```rust

        if let Some(graph_ref) = &self_.graph {
            Py::new(
                py,
                PyNodesAccessor {
                    graph: graph_ref.clone(),
                    constrained_nodes: Some(self_.nodes.clone()),
                },
            )
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
```

---

### edges

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 209.

**Implementation:**

```rust

        if let Some(graph_ref) = &self_.graph {
            Py::new(
                py,
                PyEdgesAccessor {
                    graph: graph_ref.clone(),
                    constrained_edges: Some(self_.edges.clone()),
                },
            )
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
```

---

### node_ids

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 264.

**Implementation:**

```rust

        let attr_values: Vec<groggy::AttrValue> = self
            .nodes
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### edge_ids

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 264.

**Implementation:**

```rust

        let attr_values: Vec<groggy::AttrValue> = self
            .edges
            .iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 101.

**Implementation:**

```rust

        format!(
            "Subgraph(nodes={}, edges={}, type={})",
            self.nodes.len(),
            self.edges.len(),
            self.subgraph_type
        )
```

---

### __str__

**Signature:**

```rust
fn __str__(&self) -> String
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 26, token-chars: 556.

**Implementation:**

```rust

        let mut info = format!(
            "Subgraph with {} nodes and {} edges",
            self.nodes.len(),
            self.edges.len()
        );

        if !self.subgraph_type.is_empty() {
            info.push_str(&format!("\nType: {}", self.subgraph_type));
        }

        if !self.nodes.is_empty() {
            let node_sample = if self.nodes.len() <= 5 {
                format!("{:?}", self.nodes)
            } else {
                format!(
                    "[{}, {}, {}, ... {} more]",
                    self.nodes[0],
                    self.nodes[1],
                    self.nodes[2],
                    self.nodes.len() - 3
                )
            };
            info.push_str(&format!("\nNodes: {}", node_sample));
        }

        info.push_str(
            "\nAvailable methods: .set(**attrs), .filter_nodes(filter), .table(), .nodes, .edges",
        );
        info
```

---

### density

**Signature:**

```rust
fn density(&self) -> f64
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 239.

**Implementation:**

```rust

        let num_nodes = self.nodes.len();
        let num_edges = self.edges.len();

        if num_nodes <= 1 {
            return 0.0;
        }

        // For an undirected graph, max edges = n(n-1)/2
        // For a directed graph, max edges = n(n-1)
        // Since we don't have easy access to graph type here, we'll assume undirected
        // This is the most common case and matches standard network analysis conventions
        let max_possible_edges = (num_nodes * (num_nodes - 1)) / 2;

        if max_possible_edges > 0 {
            num_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        }
```

---

### degree

**Attributes:**

```rust
#[pyo3(signature = (nodes = None, *, full_graph = false))]
```

**Signature:**

```rust
fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 126, token-chars: 3236.

**Implementation:**

```rust

        // Get graph reference if we need full graph degrees
        let graph_ref = if full_graph {
            self.graph.as_ref().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Cannot compute full graph degrees: subgraph has no parent graph reference",
                )
            })?
        } else {
            // We'll handle local degrees without needing the graph reference
            &self.graph.as_ref().unwrap() // Safe because subgraphs always have graph refs
        };

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.nodes.contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let deg = if full_graph {
                    // Get degree from full graph
                    let graph = graph_ref.borrow(py);
                    let degree = graph.inner.borrow().degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?;
                    degree
                } else {
                    // Calculate local degree within subgraph
                    self.edges
                        .iter()
                        .filter(|&&edge_id| {
                            if let Some(graph_ref) = &self.graph {
                                let graph = graph_ref.borrow(py);
                                let is_connected = if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                                    source == node_id || target == node_id
                                } else {
                                    false
                                };
                                is_connected
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(deg.to_object(py))
            }

            // List of nodes case
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut degrees = Vec::new();

                // Pre-collect edge endpoints ONCE for all degree calculations if not using full_graph
                let edge_endpoints: Vec<(groggy::NodeId, groggy::NodeId)> = if !full_graph {
                    if let Some(graph_ref) = &self.graph {
                        let graph = graph_ref.borrow(py);
                        let inner = graph.inner.borrow();
                        self.edges.iter().filter_map(|&edge_id| {
                            inner.edge_endpoints(edge_id).ok()
                        }).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.nodes.contains(&node_id) {
                        continue; // Skip nodes not in subgraph
                    }

                    let deg = if full_graph {
                        // Get degree from main graph using isolated borrow
                        let degree_result = {
                            let graph = graph_ref.borrow(py);
                            let result = graph.inner.borrow().degree(node_id);
                            result
                        };
                        match degree_result {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree using pre-collected endpoints - O(E) not O(E*N)!
                        edge_endpoints.iter().filter(|(source, target)| {
                            *source == node_id || *target == node_id
                        }).count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut degrees = Vec::new();

                // Pre-collect edge endpoints ONCE for all degree calculations if not using full_graph
                let edge_endpoints: Vec<(groggy::NodeId, groggy::NodeId)> = if !full_graph {
                    if let Some(graph_ref) = &self.graph {
                        let graph = graph_ref.borrow(py);
                        let inner = graph.inner.borrow();
                        self.edges.iter().filter_map(|&edge_id| {
                            inner.edge_endpoints(edge_id).ok()
                        }).collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                };

                for &node_id in &self.nodes {
                    let deg = if full_graph {
                        // Get degree from main graph using isolated borrow
                        let degree_result = {
                            let graph = graph_ref.borrow(py);
                            let result = graph.inner.borrow().degree(node_id);
                            result
                        };
                        match degree_result {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree using pre-collected endpoints - O(E) not O(E*N)!
                        edge_endpoints.iter().filter(|(source, target)| {
                            *source == node_id || *target == node_id
                        }).count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                let graph_array = groggy::GraphArray::from_vec(degrees);
                let py_graph_array = crate::ffi::core::array::PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?.to_object(py))
            }

            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() nodes argument must be a NodeId, list of NodeIds, or None",
            )),
        }
```

---

### filter_edges

**Signature:**

```rust
fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 54, token-chars: 1724.

**Implementation:**

```rust

        // Parse the edge filter - support both EdgeFilter objects and string queries
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_edge_query")?;
            let parsed_filter: PyEdgeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyTypeError::new_err(
                "Edge filter must be an EdgeFilter object or string query"
            ));
        };

        // Get the parent graph reference
        if let Some(parent_graph) = &self.graph {
            // Apply filter using isolated borrow scope
            let all_matching_edges = {
                let graph_ref = parent_graph.borrow(py);
                let result = graph_ref.inner.borrow_mut().find_edges(edge_filter)
                    .map_err(graph_error_to_py_err)?;
                result
            };
            
            // Keep only the edges that are in this subgraph
            let subgraph_edge_set: HashSet<EdgeId> = self.edges.iter().copied().collect();
            let filtered_edges: Vec<EdgeId> = all_matching_edges
                .into_iter()
                .filter(|edge_id| subgraph_edge_set.contains(edge_id))
                .collect();
            
            // Pre-collect all edge endpoints in one borrow operation
            let edge_endpoints: std::collections::HashMap<EdgeId, (NodeId, NodeId)> = {
                let graph_ref = parent_graph.borrow(py);
                let inner = graph_ref.inner.borrow();  // Single borrow for all edge lookups
                filtered_edges.iter().filter_map(|&edge_id| {
                    inner.edge_endpoints(edge_id)
                        .ok()
                        .map(|(s, t)| (edge_id, (s, t)))
                }).collect()
            };
            
            // Find all nodes that are connected by the filtered edges
            let mut connected_nodes = HashSet::new();
            
            for (source, target) in edge_endpoints.values() {
                // Only include nodes that were originally in this subgraph
                if self.nodes.contains(source) {
                    connected_nodes.insert(*source);
                }
                if self.nodes.contains(target) {
                    connected_nodes.insert(*target);
                }
            }
            
            // Convert to Vec for the new subgraph
            let filtered_nodes: Vec<NodeId> = connected_nodes.into_iter().collect();
            
            Ok(PySubgraph::new(
                filtered_nodes,
                filtered_edges,
                format!("{}_edge_filtered", self.subgraph_type),
                self.graph.clone(),
            ))
        } else {
            // Fallback for subgraphs without parent graph reference
            Err(PyRuntimeError::new_err(
                "Cannot filter edges: subgraph has no parent graph reference"
            ))
        }
```

---

### connected_components

**Signature:**

```rust
fn connected_components(&self) -> PyResult<Vec<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 19, token-chars: 533.

**Implementation:**

```rust

        // All subgraphs must have proper inner RustSubgraph
        if let Some(ref inner_subgraph) = self.inner {
            let components = inner_subgraph.connected_components().map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!(
                    "Failed to get connected components: {}",
                    e
                ))
            })?;

            let mut result = Vec::new();
            for component in components.iter() {
                // Create proper PySubgraph from RustSubgraph with parent graph reference
                let mut py_subgraph = PySubgraph::from_core_subgraph(component.clone());
                // Set the parent graph reference so downstream operations work
                py_subgraph.graph = self.graph.clone();
                result.push(py_subgraph);
            }
            Ok(result)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "PySubgraph missing inner RustSubgraph - ensure proper subgraph creation"
            ))
        }
```

---

### is_connected

**Signature:**

```rust
pub fn is_connected(&self) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 290.

**Implementation:**

```rust

        // All subgraphs must have proper inner RustSubgraph
        if let Some(ref inner_subgraph) = self.inner {
            inner_subgraph.is_connected().map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("Failed to check connectivity: {}", e))
            })
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "PySubgraph missing inner RustSubgraph - ensure proper subgraph creation"
            ))
        }
```

---

### set

**Attributes:**

```rust
#[pyo3(signature = (**kwargs))]
```

**Signature:**

```rust
fn set(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<Py<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 53, token-chars: 1266.

**Implementation:**

```rust

        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            if let Some(kwargs) = kwargs {
                for (key, value) in kwargs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;

                    // Use the core Subgraph's bulk set method
                    inner_subgraph
                        .set_node_attribute_bulk(&attr_name, attr_value)
                        .map_err(|e| {
                            PyErr::new::<PyRuntimeError, _>(format!(
                                "Failed to set attribute: {}",
                                e
                            ))
                        })?;
                }
            }

            // Return self for chaining
            let new_subgraph = if let Some(ref inner) = self.inner {
                PySubgraph::from_core_subgraph(inner.clone())
            } else {
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                )
            };
            Ok(Py::new(py, new_subgraph)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            if let Some(kwargs) = kwargs {
                let mut graph = graph_ref.borrow_mut(py);

                // Update all nodes in this subgraph
                for &node_id in &self.nodes {
                    for (key, value) in kwargs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;
                        let py_attr_value = PyAttrValue::from_attr_value(attr_value);

                        graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
                    }
                }
            }

            // Return self for chaining
            Ok(Py::new(
                py,
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                ),
            )?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot set attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
```

---

### update

**Signature:**

```rust
fn update(&mut self, py: Python, data: &PyDict) -> PyResult<Py<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 24, token-chars: 631.

**Implementation:**

```rust

        if let Some(graph_ref) = &self.graph {
            let mut graph = graph_ref.borrow_mut(py);

            // Update all nodes in this subgraph
            for &node_id in &self.nodes {
                for (key, value) in data.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    let py_attr_value = PyAttrValue::from_attr_value(attr_value);

                    graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
                }
            }

            // Return self for chaining
            Ok(Py::new(
                py,
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                ),
            )?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot update attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
```

---

### get_node_attribute_column

**Signature:**

```rust
fn get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 35, token-chars: 996.

**Implementation:**

```rust

        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            let attr_values = inner_subgraph
                .get_node_attribute_column(&attr_name.to_string())
                .map_err(|e| {
                    PyErr::new::<PyRuntimeError, _>(format!(
                        "Failed to get attribute column: {}",
                        e
                    ))
                })?;

            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);

            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            let mut attr_values = Vec::new();

            for &node_id in &self.nodes {
                let attr_value = {
                    let graph = graph_ref.borrow(py);
                    let result = graph.inner.borrow().get_node_attr(node_id, &attr_name.to_string());
                    result
                };
                
                if let Ok(Some(value)) = attr_value {
                    attr_values.push(value);
                } else {
                    // Handle missing attributes with default value
                    attr_values.push(groggy::AttrValue::Int(0));
                }
            }

            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);

            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access attributes on subgraph without graph reference.",
            ))
        }
```

---

### get_edge_attribute_column

**Signature:**

```rust
fn get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Vec<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 21, token-chars: 508.

**Implementation:**

```rust

        if let Some(graph_ref) = &self.graph {
            let mut values = Vec::new();

            for &edge_id in &self.edges {
                let attr_value = {
                    let graph = graph_ref.borrow(py);
                    let result = graph.inner.borrow().get_edge_attr(edge_id, &attr_name.to_string());
                    result
                };
                
                if let Ok(Some(value)) = attr_value {
                    // Convert AttrValue to Python object
                    let py_value = attr_value_to_python_value(py, &value)?;
                    values.push(py_value);
                } else {
                    // Handle missing attributes - use None
                    values.push(py.None());
                }
            }

            Ok(values)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access edge attributes on subgraph without graph reference.",
            ))
        }
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 103, token-chars: 3197.

**Implementation:**

```rust

        // Try single string first (existing behavior)
        if let Ok(attr_name) = key.extract::<String>() {
            // CRITICAL FIX: Route to edge attributes for edge subgraphs
            if self.subgraph_type == "edge_slice_selection" {
                // This is an edge subgraph - route to edge attributes
                if attr_name == "id" {
                    // Special case: edge IDs (the edges themselves)
                    let edge_ids = self
                        .edges
                        .iter()
                        .map(|&edge_id| groggy::AttrValue::Int(edge_id as i64))
                        .collect();
                    let graph_array = groggy::GraphArray::from_vec(edge_ids);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                } else {
                    // Regular edge attributes (strength, weight, etc.)
                    let edge_values = self.get_edge_attribute_column(py, &attr_name)?;
                    // Convert Vec<PyObject> to GraphArray for consistency
                    let mut attr_values = Vec::new();
                    for py_value in edge_values {
                        // Convert Python values back to AttrValue
                        if py_value.is_none(py) {
                            attr_values.push(groggy::AttrValue::Int(0)); // Default for missing
                        } else if let Ok(int_val) = py_value.extract::<i64>(py) {
                            attr_values.push(groggy::AttrValue::Int(int_val));
                        } else if let Ok(float_val) = py_value.extract::<f64>(py) {
                            attr_values.push(groggy::AttrValue::Float(float_val as f32));
                        } else if let Ok(str_val) = py_value.extract::<String>(py) {
                            attr_values.push(groggy::AttrValue::Text(str_val));
                        } else if let Ok(bool_val) = py_value.extract::<bool>(py) {
                            attr_values.push(groggy::AttrValue::Bool(bool_val));
                        } else {
                            attr_values.push(groggy::AttrValue::Int(0)); // Fallback
                        }
                    }
                    let graph_array = groggy::GraphArray::from_vec(attr_values);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                }
            } else {
                // This is a node subgraph - route to node attributes (original behavior)
                let column = self.get_node_attribute_column(py, &attr_name)?;
                return Ok(column.to_object(py));
            }
        }

        // Try list of strings (multi-column access)
        if let Ok(attr_names) = key.extract::<Vec<String>>() {
            if attr_names.is_empty() {
                return Err(PyValueError::new_err("Empty attribute list"));
            }

            // Collect all columns as GraphArrays
            let mut columns = Vec::new();
            let mut num_rows = 0;

            // Type checking for mixed types
            let mut column_types = Vec::new();

            for attr_name in &attr_names {
                let column = self.get_node_attribute_column(py, attr_name)?;

                // Detect column type by borrowing temporarily
                let column_type = {
                    let graph_array = column.borrow(py);

                    // Get the length and detect column type
                    num_rows = graph_array.inner.len();

                    // Sample a few values to determine the predominant type
                    if num_rows > 0 {
                        let sample_size = std::cmp::min(num_rows, 3);
                        let mut type_counts = std::collections::HashMap::new();

                        for i in 0..sample_size {
                            let type_name = match &graph_array.inner[i] {
                                groggy::AttrValue::Int(_) | groggy::AttrValue::SmallInt(_) => "int",
                                groggy::AttrValue::Float(_) => "float",
                                groggy::AttrValue::Bool(_) => "bool",
                                groggy::AttrValue::Text(_) | groggy::AttrValue::CompactText(_) => {
                                    "str"
                                }
                                _ => "mixed",
                            };
                            *type_counts.entry(type_name).or_insert(0) += 1;
                        }

                        // Get the most common type
                        type_counts
                            .into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(type_name, _)| type_name)
                            .unwrap_or("mixed")
                    } else {
                        "empty"
                    }
                }; // Borrow ends here

                column_types.push(column_type);
                columns.push(column);
            }

            // Check for mixed types (GraphMatrix constraint)
            if attr_names.len() > 1 {
                let first_type = column_types[0];
                let has_mixed_types = column_types
                    .iter()
                    .any(|&t| t != first_type && t != "empty");

                if has_mixed_types {
                    let detected_types: Vec<&str> = column_types.into_iter().collect();
                    return Err(PyTypeError::new_err(format!(
                        "Mixed types detected: [{}]. GraphMatrix requires homogeneous types.\n\
                        Use subgraph.nodes.table()[{:?}] for mixed-type data.",
                        detected_types.join(", "),
                        attr_names
                    )));
                }
            }

            // For single column in list form: [['age']] -> return GraphArray (same as 'age')
            if attr_names.len() == 1 {
                return Ok(columns[0].clone_ref(py).to_object(py));
            } else {
                // Multi-column access: return a list of GraphArrays
                // This allows users to work with multiple columns programmatically
                let column_objects: Vec<PyObject> =
                    columns.into_iter().map(|col| col.to_object(py)).collect();
                return Ok(column_objects.to_object(py));
            }
        }

        Err(PyTypeError::new_err(
            "Key must be a string or list of strings",
        ))
```

---

### table

**Signature:**

```rust
fn table(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 37, token-chars: 1142.

**Implementation:**

```rust

        // Get the graph reference
        let graph_py = self
            .graph
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Subgraph is not attached to a graph"))?;
        let graph = graph_py.borrow(py);

        // Get all available node attributes
        let mut all_attrs = std::collections::HashSet::new();
        for &node_id in &self.nodes {
            if let Ok(attrs) = graph.inner.borrow().get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Always include node_id as first column
        let mut column_names = vec!["node_id".to_string()];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        // Create each column
        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "node_id" {
                // Node ID column
                for &node_id in &self.nodes {
                    attr_values.push(groggy::AttrValue::Int(node_id as i64));
                }
            } else {
                // Attribute column
                for &node_id in &self.nodes {
                    if let Ok(Some(attr_value)) = graph.inner.borrow().get_node_attr(node_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Default to null/empty for missing attributes
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            }

            let graph_array = groggy::GraphArray::from_vec(attr_values);
            let py_array = PyGraphArray::from_graph_array(graph_array);
            columns.push(Py::new(py, py_array)?);
        }

        let py_table = PyGraphTable::new(py, columns, Some(column_names))?;
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### edges_table

**Signature:**

```rust
fn edges_table(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 54, token-chars: 1535.

**Implementation:**

```rust

        // Get the graph reference
        let graph_py = self
            .graph
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Subgraph is not attached to a graph"))?;
        let graph = graph_py.borrow(py);

        // Get all available edge attributes
        let mut all_attrs = std::collections::HashSet::new();
        for &edge_id in &self.edges {
            if let Ok(attrs) = graph.inner.borrow().get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Always include edge_id, source, target as first columns
        let mut column_names = vec![
            "edge_id".to_string(),
            "source".to_string(),
            "target".to_string(),
        ];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        // Create each column
        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "edge_id" {
                // Edge ID column
                for &edge_id in &self.edges {
                    attr_values.push(groggy::AttrValue::Int(edge_id as i64));
                }
            } else if column_name == "source" || column_name == "target" {
                // Source/Target columns
                for &edge_id in &self.edges {
                    if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                        let endpoint_id = if column_name == "source" {
                            source
                        } else {
                            target
                        };
                        attr_values.push(groggy::AttrValue::Int(endpoint_id as i64));
                    } else {
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            } else {
                // Attribute column
                for &edge_id in &self.edges {
                    if let Ok(Some(attr_value)) = graph.inner.borrow().get_edge_attr(edge_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Default to null/empty for missing attributes
                        attr_values.push(groggy::AttrValue::Int(0));
                    }
                }
            }

            let graph_array = groggy::GraphArray::from_vec(attr_values);
            let py_array = PyGraphArray::from_graph_array(graph_array);
            columns.push(Py::new(py, py_array)?);
        }

        let py_table = PyGraphTable::new(py, columns, Some(column_names))?;
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### graph

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
pub fn graph(&self) -> PyResult<Option<Py<PyGraph>>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 22.

**Implementation:**

```rust

        Ok(self.graph.clone())
```

---

### _get_node_attribute_column

**Signature:**

```rust
pub fn _get_node_attribute_column(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 483.

**Implementation:**

```rust

        if let Some(ref inner) = self.inner {
            let attr_name = groggy::AttrName::from(name.to_string());
            let arr = inner.get_node_attribute_column(&attr_name).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to get node attribute column: {}",
                    e
                ))
            })?;
            let py_graph_array = PyGraphArray {
                inner: groggy::GraphArray::from_vec(arr),
            };
            return Py::new(py, py_graph_array);
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Subgraph has no inner core; attach a graph-backed subgraph",
        ))
```

---

### _get_edge_attribute_column

**Signature:**

```rust
pub fn _get_edge_attribute_column(
        &self,
        py: Python<'_>,
        name: &str,
    ) -> PyResult<Py<PyGraphArray>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 483.

**Implementation:**

```rust

        if let Some(ref inner) = self.inner {
            let attr_name = groggy::AttrName::from(name.to_string());
            let arr = inner.get_edge_attribute_column(&attr_name).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to get edge attribute column: {}",
                    e
                ))
            })?;
            let py_graph_array = PyGraphArray {
                inner: groggy::GraphArray::from_vec(arr),
            };
            return Py::new(py, py_graph_array);
        }
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Subgraph has no inner core; attach a graph-backed subgraph",
        ))
```

---

### filter_nodes

**Signature:**

```rust
pub fn filter_nodes(&self, py: Python<'_>, filter: &PyAny) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 58, token-chars: 1851.

**Implementation:**

```rust

        // 0) Must have a parent graph to evaluate attributes efficiently
        let Some(graph_ref) = &self.graph else {
            return Err(PyRuntimeError::new_err(
                "Subgraph has no parent graph reference",
            ));
        };

        // 1) Resolve a NodeFilter:
        //    - If caller passed a NodeFilter object -> use it
        //    - Else if they passed a string -> parse via groggy.query_parser.parse_node_query
        //    - Else optional dict[str, AttributeFilter] -> translate to NodeFilter::And(AttributeFilter...)
        let node_filter = if let Ok(py_nf) = filter.extract::<PyNodeFilter>() {
            py_nf.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            let qp = py.import("groggy.query_parser")?;
            let parse = qp.getattr("parse_node_query")?;
            let parsed: PyNodeFilter = parse.call1((query_str,))?.extract()?;
            parsed.inner.clone()
        } else if let Ok(dict) = filter.downcast::<pyo3::types::PyDict>() {
            // Optional: allow dict form {"age": AttributeFilter.greater_than(21), ...}
            use groggy::core::query::NodeFilter as NF;
            use groggy::AttrName;
            let mut clauses = Vec::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                // Expect v is a PyAttributeFilter (FFI), so call .inner on it from Python first if needed.
                // If your AttributeFilter is exposed as a Python class, try to extract it directly:
                let py_attr = v.extract::<crate::ffi::core::query::PyAttributeFilter>()?;
                clauses.push(NF::AttributeFilter {
                    name: AttrName::from(key),
                    filter: py_attr.inner.clone(),
                });
            }
            groggy::core::query::NodeFilter::And(clauses)
        } else {
            return Err(PyTypeError::new_err(
                "filter must be NodeFilter | str | dict[str, AttributeFilter]",
            ));
        };

        // 2) Evaluate filter on the *current subgraph nodes* using the core API
        let g = graph_ref.borrow_mut(py);

        // Get all nodes that match the filter from the entire graph
        let all_filtered_nodes: Vec<groggy::NodeId> = g
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // Intersect with current subgraph's nodes to get nodes that are both:
        // 1) In this subgraph, and 2) Match the filter
        let subgraph_node_set: HashSet<groggy::NodeId> = self.nodes.iter().copied().collect();
        let filtered_nodes: Vec<groggy::NodeId> = all_filtered_nodes
            .into_iter()
            .filter(|node_id| subgraph_node_set.contains(node_id))
            .collect();

        // 3) Induce edges *within this subgraph* only
        let node_set: HashSet<groggy::NodeId> = filtered_nodes.iter().copied().collect();
        let mut induced_edges = Vec::with_capacity(self.edges.len() / 2);
        for &eid in &self.edges {
            if let Ok((s, t)) = g.inner.borrow().edge_endpoints(eid) {
                if node_set.contains(&s) && node_set.contains(&t) {
                    induced_edges.push(eid);
                }
            }
        }

        // 4) Return a new subgraph; preserve graph reference for downstream ops/tables
        let mut out = PySubgraph::new(
            filtered_nodes,
            induced_edges,
            format!("{}_filtered", self.subgraph_type),
            None, // we'll set the graph next
        );
        out.set_graph_reference(graph_ref.clone());
        Ok(out)
```

---

### to_graph

**Signature:**

```rust
pub fn to_graph(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 67, token-chars: 1897.

**Implementation:**

```rust

        // Import the PyGraph class
        let graph_module = py.import("groggy")?;
        let graph_class = graph_module.getattr("Graph")?;

        // Create a new empty graph with the same directed property as parent
        let is_directed = if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);
            let directed = parent_graph.inner.borrow().is_directed();
            directed
        } else {
            false // Default to undirected if no parent reference
        };

        let new_graph = graph_class.call1((is_directed,))?;

        if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);

            // Add nodes with attributes
            let mut node_id_mapping = std::collections::HashMap::new();

            for &old_node_id in &self.nodes {
                // Get node attribute keys
                let attr_keys = parent_graph.node_attribute_keys(old_node_id);

                // Create Python dict for node attributes
                let py_attrs = pyo3::types::PyDict::new(py);

                // Get each attribute using the FFI method
                for attr_key in attr_keys {
                    if let Ok(Some(py_attr_value)) =
                        parent_graph.get_node_attribute(old_node_id, attr_key.clone())
                    {
                        // Convert PyAttrValue to Python object
                        let py_value = crate::ffi::utils::attr_value_to_python_value(
                            py,
                            &py_attr_value.inner,
                        )?;
                        py_attrs.set_item(attr_key, py_value)?;
                    }
                }

                // Add node to new graph - call with **kwargs pattern
                let new_node_id = if py_attrs.len() > 0 {
                    new_graph.call_method("add_node", (), Some(py_attrs))?
                } else {
                    new_graph.call_method0("add_node")?
                };
                let new_id: u64 = new_node_id.extract()?;
                node_id_mapping.insert(old_node_id, new_id);
            }

            // Add edges with attributes
            for &old_edge_id in &self.edges {
                if let Ok((source, target)) = parent_graph.inner.borrow().edge_endpoints(old_edge_id) {
                    if let (Some(&new_source), Some(&new_target)) =
                        (node_id_mapping.get(&source), node_id_mapping.get(&target))
                    {
                        // Get edge attribute keys
                        let attr_keys = parent_graph.edge_attribute_keys(old_edge_id);

                        // Create Python dict for edge attributes
                        let py_attrs = pyo3::types::PyDict::new(py);

                        // Get each attribute using the FFI method
                        for attr_key in attr_keys {
                            if let Ok(Some(py_attr_value)) =
                                parent_graph.get_edge_attribute(old_edge_id, attr_key.clone())
                            {
                                // Convert PyAttrValue to Python object
                                let py_value = crate::ffi::utils::attr_value_to_python_value(
                                    py,
                                    &py_attr_value.inner,
                                )?;
                                py_attrs.set_item(attr_key, py_value)?;
                            }
                        }

                        // Add edge to new graph
                        if py_attrs.len() > 0 {
                            new_graph.call_method(
                                "add_edge",
                                (new_source, new_target),
                                Some(py_attrs),
                            )?;
                        } else {
                            new_graph.call_method1("add_edge", (new_source, new_target))?;
                        };
                    }
                }
            }
        }

        Ok(new_graph.into())
```

---

### to_networkx

**Signature:**

```rust
pub fn to_networkx(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 65, token-chars: 1714.

**Implementation:**

```rust

        // Import networkx
        let nx = py.import("networkx")?;

        // Determine graph type
        let is_directed = if let Some(graph_ref) = &self.graph {
            let parent_graph = graph_ref.borrow(py);
            let directed = parent_graph.inner.borrow().is_directed();
            directed
        } else {
            false // Default to undirected if no parent reference
        };

        // Create appropriate NetworkX graph
        let nx_graph = if is_directed {
            nx.call_method0("DiGraph")?
        } else {
            nx.call_method0("Graph")?
        };

        if let Some(graph_ref) = &self.graph {
            // Add nodes with attributes
            for &node_id in &self.nodes {
                // Get node attribute keys and attributes in isolated scopes
                let attr_keys = {
                    let parent_graph = graph_ref.borrow(py);
                    parent_graph.node_attribute_keys(node_id)
                };

                // Create Python dict for node attributes
                let py_attrs = pyo3::types::PyDict::new(py);

                // Get each attribute using isolated FFI method calls
                for attr_key in attr_keys {
                    let py_attr_value = {
                        let parent_graph = graph_ref.borrow(py);
                        parent_graph.get_node_attribute(node_id, attr_key.clone())
                    };
                    
                    if let Ok(Some(py_attr_value)) = py_attr_value {
                        // Convert PyAttrValue to Python object
                        let py_value = crate::ffi::utils::attr_value_to_python_value(
                            py,
                            &py_attr_value.inner,
                        )?;
                        py_attrs.set_item(attr_key, py_value)?;
                    }
                }

                // Add node to NetworkX graph - NetworkX expects (node_id, **attrs)
                nx_graph.call_method("add_node", (node_id,), Some(py_attrs))?;
            }

            // Add edges with attributes
            for &edge_id in &self.edges {
                let endpoints = {
                    let parent_graph = graph_ref.borrow(py);
                    let result = parent_graph.inner.borrow().edge_endpoints(edge_id);
                    result
                };
                
                if let Ok((source, target)) = endpoints {
                    // Get edge attribute keys in isolated scope
                    let attr_keys = {
                        let parent_graph = graph_ref.borrow(py);
                        parent_graph.edge_attribute_keys(edge_id)
                    };

                    // Create Python dict for edge attributes
                    let py_attrs = pyo3::types::PyDict::new(py);

                    // Get each attribute using isolated FFI method calls
                    for attr_key in attr_keys {
                        let py_attr_value = {
                            let parent_graph = graph_ref.borrow(py);
                            parent_graph.get_edge_attribute(edge_id, attr_key.clone())
                        };
                        
                        if let Ok(Some(py_attr_value)) = py_attr_value {
                            // Convert PyAttrValue to Python object
                            let py_value = crate::ffi::utils::attr_value_to_python_value(
                                py,
                                &py_attr_value.inner,
                            )?;
                            py_attrs.set_item(attr_key, py_value)?;
                        }
                    }

                    // Add edge to NetworkX graph - NetworkX expects (source, target, **attrs)
                    nx_graph.call_method("add_edge", (source, target), Some(py_attrs))?;
                }
            }
        }

        Ok(nx_graph.to_object(py))
```

---

### phase1_clustering_coefficient

**Signature:**

```rust
fn phase1_clustering_coefficient(&self, _py: Python, node_id: Option<usize>) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 252.

**Implementation:**

```rust

        if let Some(subgraph) = &self.inner {
            subgraph.clustering_coefficient(node_id.map(|id| id as NodeId))
                .map_err(|e| PyRuntimeError::new_err(format!("Clustering coefficient error: {}", e)))
        } else {
            Err(PyRuntimeError::new_err("No core subgraph available"))
        }
```

---

### phase1_transitivity

**Signature:**

```rust
fn phase1_transitivity(&self, _py: Python) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 202.

**Implementation:**

```rust

        if let Some(subgraph) = &self.inner {
            subgraph.transitivity()
                .map_err(|e| PyRuntimeError::new_err(format!("Transitivity error: {}", e)))
        } else {
            Err(PyRuntimeError::new_err("No core subgraph available"))
        }
```

---

### phase1_density

**Signature:**

```rust
fn phase1_density(&self) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 126.

**Implementation:**

```rust

        if let Some(subgraph) = &self.inner {
            Ok(subgraph.density())
        } else {
            Err(PyRuntimeError::new_err("No core subgraph available"))
        }
```

---

### merge_with

**Signature:**

```rust
fn merge_with(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 309.

**Implementation:**

```rust

        if let (Some(self_sg), Some(other_sg)) = (&self.inner, &other.inner) {
            let merged = self_sg.merge_with(other_sg)
                .map_err(|e| PyRuntimeError::new_err(format!("Merge error: {}", e)))?;
            Ok(PySubgraph::from_core_subgraph(merged))
        } else {
            Err(PyRuntimeError::new_err("Both subgraphs must have core implementations"))
        }
```

---

### intersect_with

**Signature:**

```rust
fn intersect_with(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 332.

**Implementation:**

```rust

        if let (Some(self_sg), Some(other_sg)) = (&self.inner, &other.inner) {
            let intersection = self_sg.intersect_with(other_sg)
                .map_err(|e| PyRuntimeError::new_err(format!("Intersection error: {}", e)))?;
            Ok(PySubgraph::from_core_subgraph(intersection))
        } else {
            Err(PyRuntimeError::new_err("Both subgraphs must have core implementations"))
        }
```

---

### subtract_from

**Signature:**

```rust
fn subtract_from(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 326.

**Implementation:**

```rust

        if let (Some(self_sg), Some(other_sg)) = (&self.inner, &other.inner) {
            let difference = self_sg.subtract_from(other_sg)
                .map_err(|e| PyRuntimeError::new_err(format!("Subtraction error: {}", e)))?;
            Ok(PySubgraph::from_core_subgraph(difference))
        } else {
            Err(PyRuntimeError::new_err("Both subgraphs must have core implementations"))
        }
```

---

### calculate_similarity

**Signature:**

```rust
fn calculate_similarity(&self, _py: Python, other: &PySubgraph, metric: String) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 681.

**Implementation:**

```rust

        if let (Some(self_sg), Some(other_sg)) = (&self.inner, &other.inner) {
            let similarity_metric = match metric.as_str() {
                "jaccard" => groggy::core::subgraph::SimilarityMetric::Jaccard,
                "dice" => groggy::core::subgraph::SimilarityMetric::Dice,
                "cosine" => groggy::core::subgraph::SimilarityMetric::Cosine,
                "overlap" => groggy::core::subgraph::SimilarityMetric::Overlap,
                _ => return Err(PyRuntimeError::new_err(format!("Unknown similarity metric: {}", metric)))
            };
            
            self_sg.calculate_similarity(other_sg, similarity_metric)
                .map_err(|e| PyRuntimeError::new_err(format!("Similarity calculation error: {}", e)))
        } else {
            Err(PyRuntimeError::new_err("Both subgraphs must have core implementations"))
        }
```

---


## ffi/core/table.rs

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 18, token-chars: 493.

**Implementation:**

```rust

        let (num_rows, _) = self.table.shape();
        if self.index < num_rows {
            let row_dict = pyo3::types::PyDict::new(py);

            // Get column names and data
            let column_names = self.table.columns();

            // Access each column by name
            for (_col_idx, col_name) in column_names.iter().enumerate() {
                // Get the column data by name
                if let Some(array) = self.table.get_column_by_name(col_name) {
                    if self.index < array.len() {
                        let value = &array[self.index];
                        let py_value = attr_value_to_python_value(py, value)?;
                        row_dict.set_item(col_name, py_value)?;
                    }
                }
            }

            self.index += 1;
            Ok(Some(row_dict.to_object(py)))
        } else {
            Ok(None)
        }
```

---

### new

**Attributes:**

```rust
#[new]
    #[pyo3(signature = (arrays, column_names = None))]
```

**Signature:**

```rust
pub fn new(
        py: Python,
        arrays: Vec<Py<PyGraphArray>>,
        column_names: Option<Vec<String>>,
    ) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 248.

**Implementation:**

```rust

        // Convert PyGraphArrays to core GraphArrays
        let core_arrays: Vec<GraphArray> = arrays
            .iter()
            .map(|py_array| py_array.borrow(py).inner.clone())
            .collect();

        // Create core GraphTable
        let table = GraphTable::from_arrays_standalone(core_arrays, column_names)
            .map_err(graph_error_to_py_err)?;

        Ok(Self::from_graph_table(table))
```

---

### from_graph_nodes

**Attributes:**

```rust
#[classmethod]
```

**Signature:**

```rust
pub fn from_graph_nodes(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        nodes: Vec<u64>,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 38, token-chars: 1031.

**Implementation:**

```rust

        let graph_ref = graph.borrow(py);

        // If no attributes specified, discover all available attributes
        let attr_names = attrs.unwrap_or_else(|| {
            // Discover all available node attributes
            let mut all_attrs = std::collections::HashSet::new();
            for &node_id in &nodes {
                if let Ok(attrs) = graph_ref.inner.borrow().get_node_attrs(node_id as usize) {
                    for attr_name in attrs.keys() {
                        all_attrs.insert(attr_name.clone());
                    }
                }
            }

            // Always include node_id as first column
            let mut column_names = vec!["node_id".to_string()];
            column_names.extend(all_attrs.into_iter());
            column_names
        });
        let mut columns = Vec::new();

        for attr_name in &attr_names {
            let mut attr_values = Vec::new();

            if attr_name == "node_id" {
                // Special case: node IDs
                for &node_id in &nodes {
                    attr_values.push(RustAttrValue::Int(node_id as i64));
                }
            } else {
                // Regular node attributes
                for &node_id in &nodes {
                    if let Ok(Some(attr_value)) =
                        graph_ref.inner.borrow().get_node_attr(node_id as usize, attr_name)
                    {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes as Null instead of imputing to 0
                        attr_values.push(RustAttrValue::Null);
                    }
                }
            }

            // Create GraphArray from attribute values
            let graph_array = GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Create GraphTable from arrays
        let table = GraphTable::from_arrays_standalone(columns, Some(attr_names))
            .map_err(graph_error_to_py_err)?;

        Ok(Self::from_graph_table(table))
```

---

### from_graph_edges

**Attributes:**

```rust
#[classmethod]
```

**Signature:**

```rust
pub fn from_graph_edges(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        edges: Vec<u64>,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 55, token-chars: 1418.

**Implementation:**

```rust

        let graph_ref = graph.borrow(py);

        // If no attributes specified, discover all available attributes
        let attr_names = attrs.unwrap_or_else(|| {
            // Discover all available edge attributes
            let mut all_attrs = std::collections::HashSet::new();
            for &edge_id in &edges {
                if let Ok(attrs) = graph_ref.inner.borrow().get_edge_attrs(edge_id as usize) {
                    for attr_name in attrs.keys() {
                        all_attrs.insert(attr_name.clone());
                    }
                }
            }

            // Always include edge_id, source, target as first columns
            let mut column_names = vec![
                "edge_id".to_string(),
                "source".to_string(),
                "target".to_string(),
            ];
            column_names.extend(all_attrs.into_iter());
            column_names
        });
        let mut columns = Vec::new();

        for attr_name in &attr_names {
            let mut attr_values = Vec::new();

            if attr_name == "edge_id" {
                // Special case: edge IDs
                for &edge_id in &edges {
                    attr_values.push(RustAttrValue::Int(edge_id as i64));
                }
            } else if attr_name == "source" || attr_name == "target" {
                // Special case: edge endpoints
                for &edge_id in &edges {
                    if let Ok((source, target)) = graph_ref.inner.borrow().edge_endpoints(edge_id as usize) {
                        let endpoint_id = if attr_name == "source" {
                            source
                        } else {
                            target
                        };
                        attr_values.push(RustAttrValue::Int(endpoint_id as i64));
                    } else {
                        // Handle missing edges with default value
                        attr_values.push(RustAttrValue::Int(0));
                    }
                }
            } else {
                // Regular edge attributes
                for &edge_id in &edges {
                    if let Ok(Some(attr_value)) =
                        graph_ref.inner.borrow().get_edge_attr(edge_id as usize, attr_name)
                    {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes as Null instead of imputing to 0
                        attr_values.push(RustAttrValue::Null);
                    }
                }
            }

            // Create GraphArray from attribute values
            let graph_array = GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Create GraphTable from arrays
        let table = GraphTable::from_arrays_standalone(columns, Some(attr_names))
            .map_err(graph_error_to_py_err)?;

        Ok(Self::from_graph_table(table))
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 174.

**Implementation:**

```rust

        // Try rich display formatting first
        match self._try_rich_display(py) {
            Ok(formatted) => Ok(formatted),
            Err(_) => {
                // Fallback to simple representation
                let (rows, cols) = self.inner.shape();
                Ok(format!("GraphTable({} rows, {} columns)", rows, cols))
            }
        }
```

---

### _repr_html_

**Signature:**

```rust
fn _repr_html_(&self, _py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 109, token-chars: 2509.

**Implementation:**

```rust

        let (rows, cols) = self.inner.shape();
        let columns = self.inner.columns();

        // Create HTML table with data
        let mut html = String::new();
        html.push_str(r#"<div style="max-height: 400px; overflow: auto;">"#);
        html.push_str(
            r#"<table border="1" class="dataframe" style="border-collapse: collapse; margin: 0;">"#,
        );

        // Table header
        html.push_str("<thead><tr style=\"text-align: right;\">");
        html.push_str("<th style=\"padding: 8px; background-color: #f0f0f0;\"></th>"); // Index column
        for column in columns {
            html.push_str(&format!(
                "<th style=\"padding: 8px; background-color: #f0f0f0;\">{}</th>",
                column
            ));
        }
        html.push_str("</tr></thead>");

        // Table body - show first 5 rows for performance
        html.push_str("<tbody>");
        let display_rows = std::cmp::min(rows, 5);

        for row_idx in 0..display_rows {
            html.push_str("<tr>");
            // Index column
            html.push_str(&format!(
                "<th style=\"padding: 8px; background-color: #f9f9f9;\">{}</th>",
                row_idx
            ));

            // Data columns
            if let Some(row_data) = self.inner.iloc(row_idx) {
                for column in columns {
                    let value = row_data
                        .get(column)
                        .cloned()
                        .unwrap_or(groggy::AttrValue::Null);
                    let display_value = match &value {
                        groggy::AttrValue::Int(i) => i.to_string(),
                        groggy::AttrValue::SmallInt(i) => i.to_string(),
                        groggy::AttrValue::Float(f) => {
                            if f.fract() == 0.0 {
                                format!("{:.0}", f)
                            } else {
                                format!("{:.6}", f)
                                    .trim_end_matches('0')
                                    .trim_end_matches('.')
                                    .to_string()
                            }
                        }
                        groggy::AttrValue::Text(s) => s.clone(),
                        groggy::AttrValue::CompactText(compact_str) => {
                            compact_str.as_str().to_string()
                        }
                        groggy::AttrValue::Bool(b) => {
                            if *b {
                                "True".to_string()
                            } else {
                                "False".to_string()
                            }
                        }
                        groggy::AttrValue::Bytes(b) => format!("bytes[{}]", b.len()),
                        groggy::AttrValue::FloatVec(items) => {
                            if items.len() <= 3 {
                                format!(
                                    "[{}]",
                                    items
                                        .iter()
                                        .map(|f| f.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )
                            } else {
                                format!(
                                    "[{}, ... {} items]",
                                    items
                                        .iter()
                                        .take(2)
                                        .map(|f| f.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", "),
                                    items.len()
                                )
                            }
                        }
                        groggy::AttrValue::CompressedText(_compressed) => {
                            // Compressed text - show placeholder for now
                            "compressed_text".to_string()
                        }
                        groggy::AttrValue::CompressedFloatVec(_compressed) => {
                            // Compressed float vector - show placeholder for now
                            "compressed_vec".to_string()
                        }
                        groggy::AttrValue::Null => "NaN".to_string(),
                        _ => format!("{:?}", value),
                    };
                    html.push_str(&format!(
                        "<td style=\"padding: 8px;\">{}</td>",
                        display_value
                    ));
                }
            }
            html.push_str("</tr>");
        }

        // Show truncation message if needed
        if rows > display_rows {
            html.push_str(&format!(
                "<tr><td colspan=\"{}\" style=\"padding: 8px; text-align: center; font-style: italic;\">... {} more rows</td></tr>",
                cols + 1, rows - display_rows
            ));
        }

        html.push_str("</tbody></table></div>");

        // Add summary info
        html.push_str(&format!(
            r#"<div style="margin-top: 8px; font-size: 12px; color: #666;">
            <strong>GraphTable:</strong> {} rows × {} columns
            </div>"#,
            rows, cols
        ));

        Ok(html)
```

---

### _try_rich_display

**Signature:**

```rust
fn _try_rich_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 259.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_table function from Python
        let groggy_module = py.import("groggy")?;
        let format_table = groggy_module.getattr("format_table")?;

        // Call the Python formatter
        let result = format_table.call1((display_data,))?;
        let formatted_str: String = result.extract()?;

        Ok(formatted_str)
```

---

### _try_rich_html_display

**Signature:**

```rust
fn _try_rich_html_display(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 320.

**Implementation:**

```rust

        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_table_html function from Python
        let groggy_module = py.import("groggy")?;
        let display_module = groggy_module.getattr("display")?;
        let format_table_html = display_module.getattr("format_table_html")?;

        // Call the Python HTML formatter
        let result = format_table_html.call1((display_data,))?;
        let html_str: String = result.extract()?;

        Ok(html_str)
```

---

### _get_display_data

**Signature:**

```rust
fn _get_display_data(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 29, token-chars: 862.

**Implementation:**

```rust

        let dict = PyDict::new(py);
        let (rows, cols) = self.inner.shape();

        // Build table data - convert first few rows for display
        let display_rows = std::cmp::min(rows, 100); // Limit for performance
        let mut data_rows = Vec::new();

        for row_idx in 0..display_rows {
            if let Some(row_data) = self.inner.iloc(row_idx) {
                let mut row_values = Vec::new();
                for col_name in self.inner.columns() {
                    let value = row_data
                        .get(col_name)
                        .cloned()
                        .unwrap_or(RustAttrValue::Null);
                    row_values.push(attr_value_to_python_value(py, &value)?);
                }
                data_rows.push(row_values);
            }
        }

        // Set display data
        dict.set_item("data", data_rows)?;
        dict.set_item("columns", self.inner.columns().to_vec())?;
        dict.set_item("shape", (rows, cols))?;
        dict.set_item("source_type", &self.inner.metadata().source_type)?;

        // Add basic dtype detection
        let dtypes: HashMap<String, String> = self
            .inner
            .dtypes()
            .into_iter()
            .map(|(k, v)| (k, format!("{:?}", v).to_lowercase()))
            .collect();
        dict.set_item("dtypes", dtypes)?;

        Ok(dict.to_object(py))
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 126, token-chars: 3742.

**Implementation:**

```rust

        // Handle different key types
        if let Ok(row_index) = key.extract::<isize>() {
            // Single row access: table[2] -> dict
            let (rows, _) = self.inner.shape();
            let len = rows as isize;
            let actual_index = if row_index < 0 {
                len + row_index
            } else {
                row_index
            };

            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Row index out of range"));
            }

            if let Some(row_data) = self.inner.iloc(actual_index as usize) {
                let dict = pyo3::types::PyDict::new(py);
                for (key, value) in row_data {
                    dict.set_item(key, attr_value_to_python_value(py, &value)?)?;
                }
                Ok(dict.to_object(py))
            } else {
                Err(PyIndexError::new_err("Row index out of range"))
            }
        } else if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            // Slice access: table[:2] -> table
            let (rows, _) = self.inner.shape();
            let indices = slice.indices(
                rows.try_into()
                    .map_err(|_| PyValueError::new_err("Table too large for slice"))?,
            )?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step;

            if step != 1 {
                return Err(PyNotImplementedError::new_err("Step slicing not supported"));
            }

            let sliced_table = if start < rows && start <= stop {
                let n = (stop.min(rows)).saturating_sub(start);
                self.inner.head(n) // This is a simplified approach - ideally we'd have a slice method
            } else {
                // Return empty table with same structure
                let empty_arrays = self
                    .inner
                    .columns()
                    .iter()
                    .map(|_| GraphArray::from_vec(vec![]))
                    .collect();
                GraphTable::from_arrays_standalone(
                    empty_arrays,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?
            };

            Ok(Py::new(py, PyGraphTable::from_graph_table(sliced_table))?.to_object(py))
        } else if let Ok(column_name) = key.extract::<String>() {
            // Single column access: table['column'] -> array
            if let Some(column) = self.inner.get_column_by_name(&column_name) {
                let py_array = PyGraphArray::from_graph_array(column.clone());
                Ok(Py::new(py, py_array)?.to_object(py))
            } else {
                Err(PyKeyError::new_err(format!(
                    "Column '{}' not found",
                    column_name
                )))
            }
        } else if let Ok(column_list) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2']] -> table
            let selected_table = self
                .inner
                .select(&column_list.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                .map_err(graph_error_to_py_err)?;

            Ok(Py::new(py, PyGraphTable::from_graph_table(selected_table))?.to_object(py))
        } else if let Ok(mask_array_ref) = key.extract::<PyRef<PyGraphArray>>() {
            // Boolean indexing: table[boolean_mask] -> table
            let mask_values = mask_array_ref.inner.to_list();
            let mut row_indices = Vec::new();

            // Check that mask is boolean and collect true indices
            for (i, value) in mask_values.iter().enumerate() {
                match value {
                    groggy::AttrValue::Bool(true) => row_indices.push(i),
                    groggy::AttrValue::Bool(false) => {} // Skip false values
                    _ => {
                        return Err(PyTypeError::new_err(
                            "Boolean mask must contain only boolean values",
                        ))
                    }
                }
            }

            // Check mask length matches table rows
            let (table_rows, _) = self.inner.shape();
            if mask_values.len() != table_rows {
                return Err(PyValueError::new_err(format!(
                    "Boolean mask length ({}) doesn't match table rows ({})",
                    mask_values.len(),
                    table_rows
                )));
            }

            // Create filtered table by selecting rows
            if row_indices.is_empty() {
                // Return empty table with same structure
                let empty_arrays = self
                    .inner
                    .columns()
                    .iter()
                    .map(|_| GraphArray::from_vec(vec![]))
                    .collect();
                let filtered_table = GraphTable::from_arrays_standalone(
                    empty_arrays,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?;
                Ok(Py::new(py, PyGraphTable::from_graph_table(filtered_table))?.to_object(py))
            } else {
                // Filter by row indices
                let mut filtered_columns = Vec::new();
                for col_name in self.inner.columns() {
                    if let Some(column) = self.inner.get_column_by_name(col_name) {
                        let mut filtered_values = Vec::new();
                        let col_values = column.to_list();
                        for &row_idx in &row_indices {
                            if row_idx < col_values.len() {
                                filtered_values.push(col_values[row_idx].clone());
                            }
                        }
                        filtered_columns.push(GraphArray::from_vec(filtered_values));
                    }
                }

                let filtered_table = GraphTable::from_arrays_standalone(
                    filtered_columns,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?;
                Ok(Py::new(py, PyGraphTable::from_graph_table(filtered_table))?.to_object(py))
            }
        } else {
            Err(PyTypeError::new_err(
                "Key must be: int (row), slice (:), string (column), list of strings (columns), or boolean mask (GraphArray)"
            ))
        }
```

---

### head

**Attributes:**

```rust
#[pyo3(signature = (n = 5))]
```

**Signature:**

```rust
pub fn head(&self, py: Python, n: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 134.

**Implementation:**

```rust

        let head_table = self.inner.head(n);
        let py_table = PyGraphTable::from_graph_table(head_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### tail

**Attributes:**

```rust
#[pyo3(signature = (n = 5))]
```

**Signature:**

```rust
pub fn tail(&self, py: Python, n: usize) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 134.

**Implementation:**

```rust

        let tail_table = self.inner.tail(n);
        let py_table = PyGraphTable::from_graph_table(tail_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### sort_by

**Attributes:**

```rust
#[pyo3(signature = (column, ascending = true))]
```

**Signature:**

```rust
pub fn sort_by(&self, py: Python, column: String, ascending: bool) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 190.

**Implementation:**

```rust

        let sorted_table = self
            .inner
            .sort_by(&column, ascending)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(sorted_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### describe

**Signature:**

```rust
pub fn describe(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 137.

**Implementation:**

```rust

        let desc_table = self.inner.describe();
        let py_table = PyGraphTable::from_graph_table(desc_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### to_dict

**Signature:**

```rust
pub fn to_dict(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 297.

**Implementation:**

```rust

        let dict_data = self.inner.to_dict();
        let py_dict = pyo3::types::PyDict::new(py);

        for (column, values) in dict_data {
            let py_values: Vec<PyObject> = values
                .iter()
                .map(|v| attr_value_to_python_value(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            py_dict.set_item(column, py_values)?;
        }

        Ok(py_dict.to_object(py))
```

---

### group_by

**Signature:**

```rust
pub fn group_by(&self, py: Python, column: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 172.

**Implementation:**

```rust

        let group_by = self
            .inner
            .group_by(&column)
            .map_err(graph_error_to_py_err)?;
        let py_group_by = PyGroupBy::from_group_by(group_by);
        Ok(Py::new(py, py_group_by)?.to_object(py))
```

---

### matrix

**Signature:**

```rust
pub fn matrix(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 163.

**Implementation:**

```rust

        let matrix = self.inner.matrix().map_err(graph_error_to_py_err)?;

        let py_matrix = PyGraphMatrix::from_graph_matrix(matrix);
        Ok(Py::new(py, py_matrix)?.to_object(py))
```

---

### filter_by_degree

**Signature:**

```rust
pub fn filter_by_degree(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        min_degree: Option<usize>,
        max_degree: Option<usize>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 285.

**Implementation:**

```rust

        let graph_ref = graph.borrow(py);

        let filtered_table = self
            .inner
            .filter_by_degree(&*graph_ref.inner.borrow(), &node_id_column, min_degree, max_degree)
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### filter_by_connectivity

**Signature:**

```rust
pub fn filter_by_connectivity(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        target_nodes: Vec<u64>,
        connection_type: String,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 24, token-chars: 686.

**Implementation:**

```rust

        let graph_ref = graph.borrow(py);

        // Convert target nodes to usize
        let target_usize: Vec<usize> = target_nodes.iter().map(|&id| id as usize).collect();

        // Parse connection type
        let connectivity = match connection_type.as_str() {
            "any" => ConnectivityType::ConnectedToAny,
            "all" => ConnectivityType::ConnectedToAll,
            "none" => ConnectivityType::NotConnectedToAny,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid connection type: {}. Use 'any', 'all', or 'none'",
                    connection_type
                )))
            }
        };

        let filtered_table = self
            .inner
            .filter_by_connectivity(
                &*graph_ref.inner.borrow(),
                &node_id_column,
                &target_usize,
                connectivity,
            )
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### filter_by_distance

**Signature:**

```rust
pub fn filter_by_distance(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        target_nodes: Vec<u64>,
        max_distance: usize,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 374.

**Implementation:**

```rust

        let graph_ref = graph.borrow(py);

        // Convert target nodes to usize
        let target_usize: Vec<usize> = target_nodes.iter().map(|&id| id as usize).collect();

        let filtered_table = self
            .inner
            .filter_by_distance(
                &*graph_ref.inner.borrow(),
                &node_id_column,
                &target_usize,
                max_distance,
            )
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### inner_join

**Signature:**

```rust
pub fn inner_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 208.

**Implementation:**

```rust

        let result_table = self
            .inner
            .inner_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### left_join

**Signature:**

```rust
pub fn left_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 207.

**Implementation:**

```rust

        let result_table = self
            .inner
            .left_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### right_join

**Signature:**

```rust
pub fn right_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 208.

**Implementation:**

```rust

        let result_table = self
            .inner
            .right_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### outer_join

**Signature:**

```rust
pub fn outer_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 208.

**Implementation:**

```rust

        let result_table = self
            .inner
            .outer_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### union

**Signature:**

```rust
pub fn union(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 182.

**Implementation:**

```rust

        let result_table = self
            .inner
            .union(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### intersect

**Signature:**

```rust
pub fn intersect(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 186.

**Implementation:**

```rust

        let result_table = self
            .inner
            .intersect(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(&self) -> TableIterator
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 51.

**Implementation:**

```rust

        TableIterator {
            table: self.inner.clone(),
            index: 0,
        }
```

---

### iter

**Signature:**

```rust
fn iter(&self) -> TableIterator
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 51.

**Implementation:**

```rust

        TableIterator {
            table: self.inner.clone(),
            index: 0,
        }
```

---

### data

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn data(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 225.

**Implementation:**

```rust

        let materialized = self.inner.materialize();
        let py_rows: PyResult<Vec<Vec<PyObject>>> = materialized
            .iter()
            .map(|row| {
                row.iter()
                    .map(|val| attr_value_to_python_value(py, val))
                    .collect()
            })
            .collect();

        Ok(py_rows?.to_object(py))
```

---

### preview

**Signature:**

```rust
fn preview(
        &self,
        py: Python,
        row_limit: Option<usize>,
        col_limit: Option<usize>,
    ) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 3, token-chars: 144.

**Implementation:**

```rust

        let row_limit = row_limit.unwrap_or(10);
        let (preview_data, _col_names) = self.inner.preview(row_limit, col_limit);

        Ok(preview_data.to_object(py))
```

---

### to_numpy

**Signature:**

```rust
fn to_numpy(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 247.

**Implementation:**

```rust

        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .data property
        let data = self.data(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (data,))?;
        Ok(array.to_object(py))
```

---

### fill_na

**Attributes:**

```rust
#[pyo3(signature = (fill_value, inplace = false))]
```

**Signature:**

```rust
fn fill_na(&self, py: Python, fill_value: &PyAny, inplace: bool) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 472.

**Implementation:**

```rust

        // Convert Python fill_value to AttrValue
        let attr_fill_value = crate::ffi::utils::python_value_to_attr_value(fill_value)?;

        if inplace {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "fill_na(inplace=True) is not yet implemented. Use fill_na(inplace=False) which returns a new table."
            ));
        }

        // Create new table with filled values
        let filled_table = self
            .inner
            .fill_na(attr_fill_value)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filled_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### drop_na

**Signature:**

```rust
fn drop_na(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 195.

**Implementation:**

```rust

        let filtered_table = self
            .inner
            .drop_na()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### has_null

**Signature:**

```rust
fn has_null(&self) -> bool
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 12, token-chars: 243.

**Implementation:**

```rust

        let (rows, _) = self.inner.shape();
        let _columns = self.inner.columns();
        for row_idx in 0..rows {
            if let Some(row_data) = self.inner.iloc(row_idx) {
                for value in row_data.values() {
                    if matches!(value, groggy::AttrValue::Null) {
                        return true;
                    }
                }
            }
        }
        false
```

---

### null_count

**Signature:**

```rust
fn null_count(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 19, token-chars: 435.

**Implementation:**

```rust

        let dict = pyo3::types::PyDict::new(py);
        let (rows, _) = self.inner.shape();
        let columns = self.inner.columns();

        for column_name in columns {
            let mut null_count = 0;

            for row_idx in 0..rows {
                if let Some(row_data) = self.inner.iloc(row_idx) {
                    if let Some(value) = row_data.get(column_name) {
                        if matches!(value, groggy::AttrValue::Null) {
                            null_count += 1;
                        }
                    } else {
                        // Missing value counts as null
                        null_count += 1;
                    }
                }
            }

            dict.set_item(column_name, null_count)?;
        }

        Ok(dict.to_object(py))
```

---

### to_pandas

**Signature:**

```rust
fn to_pandas(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 378.

**Implementation:**

```rust

        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Get materialized data and column names
        let data = self.data(py)?;
        let columns = self.inner.columns();

        // Create DataFrame
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("columns", columns)?;
        let df = pandas.call_method("DataFrame", (data,), Some(kwargs))?;
        Ok(df.to_object(py))
```

---

### agg

**Signature:**

```rust
pub fn agg(&self, py: Python, operations: HashMap<String, String>) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 25, token-chars: 665.

**Implementation:**

```rust

        // Convert string operations to AggregateOp enum
        let mut ops = HashMap::new();
        for (column, op_str) in operations {
            let op = match op_str.as_str() {
                "sum" => AggregateOp::Sum,
                "mean" => AggregateOp::Mean,
                "count" => AggregateOp::Count,
                "min" => AggregateOp::Min,
                "max" => AggregateOp::Max,
                "std" => AggregateOp::Std,
                "var" => AggregateOp::Var,
                "first" => AggregateOp::First,
                "last" => AggregateOp::Last,
                "unique" => AggregateOp::Unique,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown aggregation operation: {}",
                        op_str
                    )))
                }
            };
            ops.insert(column, op);
        }

        // Apply aggregation
        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### sum

**Signature:**

```rust
pub fn sum(&self, py: Python, column: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 237.

**Implementation:**

```rust

        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Sum);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### mean

**Signature:**

```rust
pub fn mean(&self, py: Python, column: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 238.

**Implementation:**

```rust

        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Mean);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---

### count

**Signature:**

```rust
pub fn count(&self, py: Python, column: String) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 239.

**Implementation:**

```rust

        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Count);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
```

---


## ffi/core/traversal.rs

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 111.

**Implementation:**

```rust

        format!(
            "TraversalResult(nodes={}, edges={}, type='{}')",
            self.nodes.len(),
            self.edges.len(),
            self.traversal_type
        )
```

---

### new

**Signature:**

```rust
pub fn new(
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        distances: Option<Vec<usize>>,
        traversal_type: String,
    ) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 44.

**Implementation:**

```rust

        Self {
            nodes,
            edges,
            distances,
            traversal_type,
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 113.

**Implementation:**

```rust

        format!(
            "AggregationResult(value={}, operation='{}', attribute='{}')",
            self.value, self.operation, self.attribute
        )
```

---

### new

**Signature:**

```rust
pub fn new(value: f64, operation: String, attribute: String, count: usize) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 39.

**Implementation:**

```rust

        Self {
            value,
            operation,
            attribute,
            count,
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 98.

**Implementation:**

```rust

        format!(
            "GroupedAggregationResult(operation='{}', attribute='{}')",
            self.operation, self.attribute
        )
```

---

### new

**Signature:**

```rust
pub fn new(groups: PyObject, operation: String, attribute: String) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 5, token-chars: 34.

**Implementation:**

```rust

        Self {
            groups,
            operation,
            attribute,
        }
```

---


## ffi/core/views.rs

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &str) -> PyResult<PyAttrValue>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 223.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        match graph.get_node_attribute(self.node_id, key.to_string())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!(
                "Attribute '{}' not found on node {}",
                key, self.node_id
            ))),
        }
```

---

### __setitem__

**Signature:**

```rust
fn __setitem__(&mut self, py: Python, key: &str, value: PyAttrValue) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 113.

**Implementation:**

```rust

        let mut graph = self.graph.borrow_mut(py);
        graph.set_node_attribute(self.node_id, key.to_string(), &value)?;
        Ok(())
```

---

### id

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn id(&self) -> PyResult<NodeId>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 16.

**Implementation:**

```rust

        Ok(self.node_id)
```

---

### __contains__

**Signature:**

```rust
fn __contains__(&self, py: Python, key: &str) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 81.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        Ok(graph.has_node_attribute(self.node_id, key))
```

---

### keys

**Signature:**

```rust
fn keys(&self, py: Python) -> PyResult<Vec<String>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 77.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        Ok(graph.node_attribute_keys(self.node_id))
```

---

### values

**Signature:**

```rust
fn values(&self, py: Python) -> PyResult<Vec<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 300.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let node_attrs = graph.inner.borrow().get_node_attrs(self.node_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;

        let values = node_attrs
            .into_iter()
            .map(|(_, value)| PyAttrValue::from_attr_value(value))
            .collect();
        Ok(values)
```

---

### neighbors

**Signature:**

```rust
fn neighbors(&self, py: Python) -> PyResult<Vec<NodeId>>
```

**Why flagged:** body length exceeds trivial threshold. Lines: 5, token-chars: 204.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let neighbors = graph.inner.borrow().neighbors(self.node_id).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to get neighbors: {}", e))
        });
        neighbors
```

---

### items

**Signature:**

```rust
fn items(&self, py: Python) -> PyResult<Vec<(String, PyAttrValue)>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 307.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let node_attrs = graph.inner.borrow().get_node_attrs(self.node_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get node attributes: {}", e))
        })?;

        let items = node_attrs
            .into_iter()
            .map(|(key, value)| (key, PyAttrValue::from_attr_value(value)))
            .collect();
        Ok(items)
```

---

### update

**Signature:**

```rust
fn update(&mut self, py: Python, attributes: &PyDict) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 302.

**Implementation:**

```rust

        let mut graph = self.graph.borrow_mut(py);

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph.set_node_attribute(
                self.node_id,
                key_str,
                &PyAttrValue::from_attr_value(attr_value),
            )?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
```

---

### __str__

**Signature:**

```rust
fn __str__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 18, token-chars: 517.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        if keys.is_empty() {
            Ok(format!("NodeView({})", self.node_id))
        } else {
            let mut attr_parts = Vec::new();
            for key in keys.iter().take(3) {
                // Show first 3 attributes
                if let Ok(Some(value)) = graph.get_node_attribute(self.node_id, key.clone()) {
                    attr_parts.push(format!("{}={}", key, value.__str__()?));
                }
            }

            let attr_str = if keys.len() > 3 {
                format!("{}, ...", attr_parts.join(", "))
            } else {
                attr_parts.join(", ")
            };

            Ok(format!("NodeView({}, {})", self.node_id, attr_str))
        }
```

---

### to_dict

**Signature:**

```rust
fn to_dict(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 268.

**Implementation:**

```rust

        let dict = PyDict::new(py);
        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        for key in keys {
            if let Some(value) = graph
                .get_node_attribute(self.node_id, key.clone())
                .ok()
                .flatten()
            {
                dict.set_item(key, value)?;
            }
        }

        Ok(dict.to_object(py))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(&self, py: Python) -> PyResult<NodeViewIterator>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 284.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let keys = graph.node_attribute_keys(self.node_id);

        let mut items = Vec::new();
        for key in keys {
            if let Some(value) = graph
                .get_node_attribute(self.node_id, key.clone())
                .ok()
                .flatten()
            {
                items.push((key, value));
            }
        }

        Ok(NodeViewIterator { items, index: 0 })
```

---

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 223.

**Implementation:**

```rust

        if self.index < self.items.len() {
            let (key, value) = &self.items[self.index];
            self.index += 1;

            // Return (key, value) tuple
            let tuple = pyo3::types::PyTuple::new(py, [key.to_object(py), value.to_object(py)]);
            Ok(Some(tuple.to_object(py)))
        } else {
            Ok(None)
        }
```

---

### clone

**Signature:**

```rust
fn clone(&self) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 61.

**Implementation:**

```rust

        PyNodeView {
            graph: self.graph.clone(),
            node_id: self.node_id,
        }
```

---

### __getitem__

**Signature:**

```rust
fn __getitem__(&self, py: Python, key: &str) -> PyResult<PyAttrValue>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 223.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        match graph.get_edge_attribute(self.edge_id, key.to_string())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!(
                "Attribute '{}' not found on edge {}",
                key, self.edge_id
            ))),
        }
```

---

### __setitem__

**Signature:**

```rust
fn __setitem__(&mut self, py: Python, key: &str, value: PyAttrValue) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 3, token-chars: 113.

**Implementation:**

```rust

        let mut graph = self.graph.borrow_mut(py);
        graph.set_edge_attribute(self.edge_id, key.to_string(), &value)?;
        Ok(())
```

---

### id

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn id(&self) -> PyResult<EdgeId>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 16.

**Implementation:**

```rust

        Ok(self.edge_id)
```

---

### edge_id

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn edge_id(&self) -> PyResult<EdgeId>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 16.

**Implementation:**

```rust

        Ok(self.edge_id)
```

---

### source

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn source(&self, py: Python) -> PyResult<NodeId>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 198.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let (source, _) = graph
            .inner
            .borrow()
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(source)
```

---

### target

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn target(&self, py: Python) -> PyResult<NodeId>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 198.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let (_, target) = graph
            .inner
            .borrow()
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(target)
```

---

### endpoints

**Signature:**

```rust
fn endpoints(&self, py: Python) -> PyResult<(NodeId, NodeId)>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 199.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let endpoints = graph
            .inner
            .borrow()
            .edge_endpoints(self.edge_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get edge endpoints: {}", e)))?;
        Ok(endpoints)
```

---

### __contains__

**Signature:**

```rust
fn __contains__(&self, py: Python, key: &str) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 81.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        Ok(graph.has_edge_attribute(self.edge_id, key))
```

---

### keys

**Signature:**

```rust
fn keys(&self, py: Python) -> PyResult<Vec<String>>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 77.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        Ok(graph.edge_attribute_keys(self.edge_id))
```

---

### values

**Signature:**

```rust
fn values(&self, py: Python) -> PyResult<Vec<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 300.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let edge_attrs = graph.inner.borrow().get_edge_attrs(self.edge_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;

        let values = edge_attrs
            .into_iter()
            .map(|(_, value)| PyAttrValue::from_attr_value(value))
            .collect();
        Ok(values)
```

---

### items

**Signature:**

```rust
fn items(&self, py: Python) -> PyResult<Vec<(String, PyAttrValue)>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 307.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        // 🚀 PERFORMANCE FIX: Use batch attribute access instead of manual loops
        let edge_attrs = graph.inner.borrow().get_edge_attrs(self.edge_id).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get edge attributes: {}", e))
        })?;

        let items = edge_attrs
            .into_iter()
            .map(|(key, value)| (key, PyAttrValue::from_attr_value(value)))
            .collect();
        Ok(items)
```

---

### update

**Signature:**

```rust
fn update(&mut self, py: Python, attributes: &PyDict) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 302.

**Implementation:**

```rust

        let mut graph = self.graph.borrow_mut(py);

        for (key, value) in attributes.iter() {
            let key_str = key.extract::<String>()?;
            let attr_value = PyAttrValue::extract(value)?.to_attr_value();
            graph.set_edge_attribute(
                self.edge_id,
                key_str,
                &PyAttrValue::from_attr_value(attr_value),
            )?;
        }

        // Return self for chaining
        Ok(self.clone().into_py(py))
```

---

### __str__

**Signature:**

```rust
fn __str__(&self, py: Python) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 28, token-chars: 745.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        // Get endpoints for display
        let (source, target) = match graph.inner.borrow().edge_endpoints(self.edge_id) {
            Ok(endpoints) => endpoints,
            Err(_) => return Ok(format!("EdgeView({}) [invalid]", self.edge_id)),
        };

        if keys.is_empty() {
            Ok(format!(
                "EdgeView({}: {} -> {})",
                self.edge_id, source, target
            ))
        } else {
            let mut attr_parts = Vec::new();
            for key in keys.iter().take(3) {
                // Show first 3 attributes
                if let Ok(Some(value)) = graph.get_edge_attribute(self.edge_id, key.clone()) {
                    attr_parts.push(format!("{}={}", key, value.__str__()?));
                }
            }

            let attr_str = if keys.len() > 3 {
                format!("{}, ...", attr_parts.join(", "))
            } else {
                attr_parts.join(", ")
            };

            Ok(format!(
                "EdgeView({}: {} -> {}, {})",
                self.edge_id, source, target, attr_str
            ))
        }
```

---

### to_dict

**Signature:**

```rust
fn to_dict(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 268.

**Implementation:**

```rust

        let dict = PyDict::new(py);
        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        for key in keys {
            if let Some(value) = graph
                .get_edge_attribute(self.edge_id, key.clone())
                .ok()
                .flatten()
            {
                dict.set_item(key, value)?;
            }
        }

        Ok(dict.to_object(py))
```

---

### __iter__

**Signature:**

```rust
fn __iter__(&self, py: Python) -> PyResult<EdgeViewIterator>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 284.

**Implementation:**

```rust

        let graph = self.graph.borrow(py);
        let keys = graph.edge_attribute_keys(self.edge_id);

        let mut items = Vec::new();
        for key in keys {
            if let Some(value) = graph
                .get_edge_attribute(self.edge_id, key.clone())
                .ok()
                .flatten()
            {
                items.push((key, value));
            }
        }

        Ok(EdgeViewIterator { items, index: 0 })
```

---

### __next__

**Signature:**

```rust
fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 223.

**Implementation:**

```rust

        if self.index < self.items.len() {
            let (key, value) = &self.items[self.index];
            self.index += 1;

            // Return (key, value) tuple
            let tuple = pyo3::types::PyTuple::new(py, [key.to_object(py), value.to_object(py)]);
            Ok(Some(tuple.to_object(py)))
        } else {
            Ok(None)
        }
```

---

### clone

**Signature:**

```rust
fn clone(&self) -> Self
```

**Why flagged:** body length exceeds trivial threshold. Lines: 4, token-chars: 61.

**Implementation:**

```rust

        PyEdgeView {
            graph: self.graph.clone(),
            edge_id: self.edge_id,
        }
```

---


## ffi/display.rs

### new

**Attributes:**

```rust
#[new]
    #[pyo3(signature = (max_rows = None, max_cols = None, max_width = None, precision = None, use_color = None))]
```

**Signature:**

```rust
fn new(
        max_rows: Option<usize>,
        max_cols: Option<usize>,
        max_width: Option<usize>,
        precision: Option<usize>,
        use_color: Option<bool>,
    ) -> Self
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 17, token-chars: 344.

**Implementation:**

```rust

        let mut config = DisplayConfig::default();

        if let Some(rows) = max_rows {
            config.max_rows = rows;
        }
        if let Some(cols) = max_cols {
            config.max_cols = cols;
        }
        if let Some(width) = max_width {
            config.max_width = width;
        }
        if let Some(prec) = precision {
            config.precision = prec;
        }
        if let Some(color) = use_color {
            config.use_color = color;
        }

        Self { inner: config }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 8, token-chars: 195.

**Implementation:**

```rust

        format!(
            "DisplayConfig(max_rows={}, max_cols={}, max_width={}, precision={}, use_color={})",
            self.inner.max_rows,
            self.inner.max_cols,
            self.inner.max_width,
            self.inner.precision,
            self.inner.use_color
        )
```

---

### pydict_to_hashmap

**Signature:**

```rust
fn pydict_to_hashmap(py_dict: &PyDict) -> PyResult<HashMap<String, Value>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 188.

**Implementation:**

```rust

    let mut map = HashMap::new();

    for (key, value) in py_dict.iter() {
        let key_str: String = key.extract()?;
        let json_value = python_to_json_value(value)?;
        map.insert(key_str, json_value);
    }

    Ok(map)
```

---

### python_to_json_value

**Signature:**

```rust
fn python_to_json_value(py_value: &PyAny) -> PyResult<Value>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 23, token-chars: 657.

**Implementation:**

```rust

    if let Ok(s) = py_value.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(i) = py_value.extract::<i64>() {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = py_value.extract::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(Value::Number(num))
        } else {
            Ok(Value::Null)
        }
    } else if let Ok(b) = py_value.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if py_value.is_none() {
        Ok(Value::Null)
    } else if let Ok(list) = py_value.extract::<Vec<&PyAny>>() {
        let mut json_array = Vec::new();
        for item in list {
            json_array.push(python_to_json_value(item)?);
        }
        Ok(Value::Array(json_array))
    } else {
        // Fallback: convert to string
        Ok(Value::String(format!("{:?}", py_value)))
    }
```

---

### py_format_array

**Attributes:**

```rust
#[pyfunction]
#[pyo3(signature = (data, config = None))]
```

**Signature:**

```rust
pub fn py_format_array(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 201.

**Implementation:**

```rust

    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_array(data_map, display_config))
```

---

### py_format_matrix

**Attributes:**

```rust
#[pyfunction]
#[pyo3(signature = (data, config = None))]
```

**Signature:**

```rust
pub fn py_format_matrix(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 202.

**Implementation:**

```rust

    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_matrix(data_map, display_config))
```

---

### py_format_table

**Attributes:**

```rust
#[pyfunction]
#[pyo3(signature = (data, config = None))]
```

**Signature:**

```rust
pub fn py_format_table(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 201.

**Implementation:**

```rust

    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_table(data_map, display_config))
```

---

### py_format_data_structure

**Attributes:**

```rust
#[pyfunction]
#[pyo3(signature = (data, data_type = None, config = None))]
```

**Signature:**

```rust
pub fn py_format_data_structure(
    data: &PyDict,
    data_type: Option<&str>,
    config: Option<&PyDisplayConfig>,
) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 221.

**Implementation:**

```rust

    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_data_structure(data_map, data_type, display_config))
```

---

### py_detect_display_type

**Attributes:**

```rust
#[pyfunction]
```

**Signature:**

```rust
pub fn py_detect_display_type(data: &PyDict) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 2, token-chars: 86.

**Implementation:**

```rust

    let data_map = pydict_to_hashmap(data)?;
    Ok(detect_display_type(&data_map).to_string())
```

---

### register_display_functions

**Signature:**

```rust
pub fn register_display_functions(_py: Python, m: &PyModule) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 15, token-chars: 636.

**Implementation:**

```rust

    // Register display config class
    m.add_class::<PyDisplayConfig>()?;

    // Register formatting functions
    m.add_function(wrap_pyfunction!(py_format_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_table, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_data_structure, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_display_type, m)?)?;

    // Add aliases for Python compatibility
    m.add("format_array", m.getattr("py_format_array")?)?;
    m.add("format_matrix", m.getattr("py_format_matrix")?)?;
    m.add("format_table", m.getattr("py_format_table")?)?;
    m.add(
        "format_data_structure",
        m.getattr("py_format_data_structure")?,
    )?;
    m.add("detect_display_type", m.getattr("py_detect_display_type")?)?;

    Ok(())
```

---


## ffi/traits/subgraph_operations.rs

### nodes

**Signature:**

```rust
fn nodes(&self, py: Python) -> PyResult<Py<PyList>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 152.

**Implementation:**

```rust

        let core = self.core_subgraph()?;
        let node_ids: Vec<usize> = core.node_set()
            .iter()
            .map(|&id| id as usize)
            .collect();
        Ok(PyList::new(py, node_ids).into())
```

---

### edges

**Signature:**

```rust
fn edges(&self, py: Python) -> PyResult<Py<PyList>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 152.

**Implementation:**

```rust

        let core = self.core_subgraph()?;
        let edge_ids: Vec<usize> = core.edge_set()
            .iter()
            .map(|&id| id as usize)
            .collect();
        Ok(PyList::new(py, edge_ids).into())
```

---

### node_count

**Signature:**

```rust
fn node_count(&self) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 38.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.node_count())
```

---

### edge_count

**Signature:**

```rust
fn edge_count(&self) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 38.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.edge_count())
```

---

### is_empty

**Signature:**

```rust
fn is_empty(&self) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 43.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.node_count() == 0)
```

---

### summary

**Signature:**

```rust
fn summary(&self) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 35.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.summary())
```

---

### contains_node

**Signature:**

```rust
fn contains_node(&self, node_id: usize) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 58.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.contains_node(node_id as NodeId))
```

---

### neighbors

**Signature:**

```rust
fn neighbors(&self, py: Python, node_id: usize) -> PyResult<Py<PyList>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 297.

**Implementation:**

```rust

        py.allow_threads(|| {
            let neighbors = self.core_subgraph()?.neighbors(node_id as NodeId)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            let py_neighbors: Vec<usize> = neighbors.into_iter().map(|id| id as usize).collect();
            Ok(PyList::new(py, py_neighbors).into())
        })
```

---

### degree

**Signature:**

```rust
fn degree(&self, py: Python, node_id: usize) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 151.

**Implementation:**

```rust

        py.allow_threads(|| {
            self.core_subgraph()?.degree(node_id as NodeId)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        })
```

---

### contains_edge

**Signature:**

```rust
fn contains_edge(&self, edge_id: usize) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 58.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.contains_edge(edge_id as EdgeId))
```

---

### edge_endpoints

**Signature:**

```rust
fn edge_endpoints(&self, py: Python, edge_id: usize) -> PyResult<(usize, usize)>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 222.

**Implementation:**

```rust

        py.allow_threads(|| {
            let (source, target) = self.core_subgraph()?.edge_endpoints(edge_id as EdgeId)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            Ok((source as usize, target as usize))
        })
```

---

### has_edge_between

**Signature:**

```rust
fn has_edge_between(&self, py: Python, source: usize, target: usize) -> PyResult<bool>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 4, token-chars: 178.

**Implementation:**

```rust

        py.allow_threads(|| {
            self.core_subgraph()?.has_edge_between(source as NodeId, target as NodeId)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        })
```

---

### get_node_attribute

**Signature:**

```rust
fn get_node_attribute(&self, py: Python, node_id: usize, attr_name: String) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 306.

**Implementation:**

```rust

        py.allow_threads(|| {
            let attr_opt = self.core_subgraph()?.get_node_attribute(node_id as NodeId, &attr_name.into())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            
            match attr_opt {
                Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
                None => Ok(None)
            }
        })
```

---

### get_edge_attribute

**Signature:**

```rust
fn get_edge_attribute(&self, py: Python, edge_id: usize, attr_name: String) -> PyResult<Option<PyObject>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 306.

**Implementation:**

```rust

        py.allow_threads(|| {
            let attr_opt = self.core_subgraph()?.get_edge_attribute(edge_id as EdgeId, &attr_name.into())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            
            match attr_opt {
                Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
                None => Ok(None)
            }
        })
```

---

### clustering_coefficient

**Signature:**

```rust
fn clustering_coefficient(&self, py: Python, node_id: Option<usize>) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 375.

**Implementation:**

```rust

        py.allow_threads(|| {
            // Downcast to concrete Subgraph type to access new methods
            // This is temporary until we add these methods to the core trait
            if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
                concrete_subgraph.clustering_coefficient(node_id.map(|id| id as NodeId))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "clustering_coefficient not available for this subgraph type"
                ))
            }
        })
```

---

### transitivity

**Signature:**

```rust
fn transitivity(&self, py: Python) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 10, token-chars: 325.

**Implementation:**

```rust

        py.allow_threads(|| {
            if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
                concrete_subgraph.transitivity()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "transitivity not available for this subgraph type"
                ))
            }
        })
```

---

### density

**Signature:**

```rust
fn density(&self) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 7, token-chars: 215.

**Implementation:**

```rust

        if let Some(concrete_subgraph) = self.try_downcast_to_subgraph() {
            Ok(concrete_subgraph.density())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "density not available for this subgraph type"
            ))
        }
```

---

### merge_with

**Signature:**

```rust
fn merge_with(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 412.

**Implementation:**

```rust

        py.allow_threads(|| {
            if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
                let merged = self_sg.merge_with(other_sg)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
                PySubgraph::from_core_subgraph(merged)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "merge_with requires concrete Subgraph types"
                ))
            }
        })
```

---

### intersect_with

**Signature:**

```rust
fn intersect_with(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 432.

**Implementation:**

```rust

        py.allow_threads(|| {
            if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
                let intersection = self_sg.intersect_with(other_sg)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
                PySubgraph::from_core_subgraph(intersection)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "intersect_with requires concrete Subgraph types"
                ))
            }
        })
```

---

### subtract_from

**Signature:**

```rust
fn subtract_from(&self, py: Python, other: &dyn PySubgraphOperations) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 11, token-chars: 426.

**Implementation:**

```rust

        py.allow_threads(|| {
            if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
                let difference = self_sg.subtract_from(other_sg)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
                PySubgraph::from_core_subgraph(difference)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "subtract_from requires concrete Subgraph types"
                ))
            }
        })
```

---

### calculate_similarity

**Signature:**

```rust
fn calculate_similarity(&self, py: Python, other: &dyn PySubgraphOperations, metric: String) -> PyResult<f64>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 19, token-chars: 709.

**Implementation:**

```rust

        py.allow_threads(|| {
            let similarity_metric = match metric.as_str() {
                "jaccard" => SimilarityMetric::Jaccard,
                "dice" => SimilarityMetric::Dice,
                "cosine" => SimilarityMetric::Cosine,
                "overlap" => SimilarityMetric::Overlap,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown similarity metric: {}", metric)
                ))
            };
            
            if let (Some(self_sg), Some(other_sg)) = (self.try_downcast_to_subgraph(), other.try_downcast_to_subgraph()) {
                self_sg.calculate_similarity(other_sg, similarity_metric)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "calculate_similarity requires concrete Subgraph types"
                ))
            }
        })
```

---

### connected_components

**Signature:**

```rust
fn connected_components(&self, py: Python) -> PyResult<Vec<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 9, token-chars: 318.

**Implementation:**

```rust

        py.allow_threads(|| {
            let components = self.core_subgraph()?.connected_components()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            
            let py_components: PyResult<Vec<PySubgraph>> = components
                .into_iter()
                .map(|component| PySubgraph::from_trait_object(component))
                .collect();
            py_components
        })
```

---

### bfs_subgraph

**Signature:**

```rust
fn bfs_subgraph(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 226.

**Implementation:**

```rust

        py.allow_threads(|| {
            let bfs_result = self.core_subgraph()?.bfs_subgraph(start as NodeId, max_depth)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_trait_object(bfs_result)
        })
```

---

### dfs_subgraph

**Signature:**

```rust
fn dfs_subgraph(&self, py: Python, start: usize, max_depth: Option<usize>) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 5, token-chars: 226.

**Implementation:**

```rust

        py.allow_threads(|| {
            let dfs_result = self.core_subgraph()?.dfs_subgraph(start as NodeId, max_depth)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_trait_object(dfs_result)
        })
```

---

### shortest_path_subgraph

**Signature:**

```rust
fn shortest_path_subgraph(&self, py: Python, source: usize, target: usize) -> PyResult<Option<PySubgraph>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 8, token-chars: 313.

**Implementation:**

```rust

        py.allow_threads(|| {
            let path_opt = self.core_subgraph()?.shortest_path_subgraph(source as NodeId, target as NodeId)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            
            match path_opt {
                Some(path_subgraph) => Ok(Some(PySubgraph::from_trait_object(path_subgraph)?)),
                None => Ok(None)
            }
        })
```

---

### induced_subgraph

**Signature:**

```rust
fn induced_subgraph(&self, py: Python, nodes: Vec<usize>) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 286.

**Implementation:**

```rust

        py.allow_threads(|| {
            let node_ids: Vec<NodeId> = nodes.into_iter().map(|id| id as NodeId).collect();
            let induced = self.core_subgraph()?.induced_subgraph(&node_ids)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_trait_object(induced)
        })
```

---

### subgraph_from_edges

**Signature:**

```rust
fn subgraph_from_edges(&self, py: Python, edges: Vec<usize>) -> PyResult<PySubgraph>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 6, token-chars: 301.

**Implementation:**

```rust

        py.allow_threads(|| {
            let edge_ids: Vec<EdgeId> = edges.into_iter().map(|id| id as EdgeId).collect();
            let edge_subgraph = self.core_subgraph()?.subgraph_from_edges(&edge_ids)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            PySubgraph::from_trait_object(edge_subgraph)
        })
```

---

### collapse_to_node

**Signature:**

```rust
fn collapse_to_node(&self, py: Python, agg_functions: &PyDict) -> PyResult<usize>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 411.

**Implementation:**

```rust

        py.allow_threads(|| {
            let agg_map: HashMap<AttrName, String> = agg_functions
                .iter()
                .map(|(k, v)| {
                    let key = k.extract::<String>().unwrap_or_default().into();
                    let value = v.extract::<String>().unwrap_or_default();
                    (key, value)
                })
                .collect();
            
            let meta_node_id = self.core_subgraph()?.collapse_to_node(agg_map)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
            Ok(meta_node_id as usize)
        })
```

---

### set_node_attrs

**Signature:**

```rust
fn set_node_attrs(&self, py: Python, attrs_values: &PyDict) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 732.

**Implementation:**

```rust

        py.allow_threads(|| {
            let bulk_attrs: PyResult<HashMap<AttrName, Vec<(NodeId, groggy::types::AttrValue)>>> = attrs_values
                .iter()
                .map(|(attr_name, node_values)| {
                    let attr_key = attr_name.extract::<String>()?.into();
                    let values_list = node_values.extract::<Vec<(usize, PyAttrValue)>>()?;
                    let converted_values: PyResult<Vec<(NodeId, groggy::types::AttrValue)>> = values_list
                        .into_iter()
                        .map(|(node_id, py_value)| {
                            let py_obj = py_value.into_py(py);
                            let rust_value = python_value_to_attr_value(py_obj.as_ref(py))?;
                            Ok((node_id as NodeId, rust_value))
                        })
                        .collect();
                    Ok((attr_key, converted_values?))
                })
                .collect();
            
            self.core_subgraph()?.set_node_attrs(bulk_attrs?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        })
```

---

### set_edge_attrs

**Signature:**

```rust
fn set_edge_attrs(&self, py: Python, attrs_values: &PyDict) -> PyResult<()>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 20, token-chars: 732.

**Implementation:**

```rust

        py.allow_threads(|| {
            let bulk_attrs: PyResult<HashMap<AttrName, Vec<(EdgeId, groggy::types::AttrValue)>>> = attrs_values
                .iter()
                .map(|(attr_name, edge_values)| {
                    let attr_key = attr_name.extract::<String>()?.into();
                    let values_list = edge_values.extract::<Vec<(usize, PyAttrValue)>>()?;
                    let converted_values: PyResult<Vec<(EdgeId, groggy::types::AttrValue)>> = values_list
                        .into_iter()
                        .map(|(edge_id, py_value)| {
                            let py_obj = py_value.into_py(py);
                            let rust_value = python_value_to_attr_value(py_obj.as_ref(py))?;
                            Ok((edge_id as EdgeId, rust_value))
                        })
                        .collect();
                    Ok((attr_key, converted_values?))
                })
                .collect();
            
            self.core_subgraph()?.set_edge_attrs(bulk_attrs?)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        })
```

---

### entity_type

**Signature:**

```rust
fn entity_type(&self) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present. Lines: 1, token-chars: 51.

**Implementation:**

```rust

        Ok(self.core_subgraph()?.entity_type().to_string())
```

---


## ffi/types.rs

### py_new

**Attributes:**

```rust
#[new]
```

**Signature:**

```rust
fn py_new(value: &PyAny) -> PyResult<Self>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 23, token-chars: 876.

**Implementation:**

```rust

        let rust_value = if let Ok(b) = value.extract::<bool>() {
            RustAttrValue::Bool(b)
        } else if let Ok(i) = value.extract::<i64>() {
            RustAttrValue::Int(i)
        } else if let Ok(f) = value.extract::<f64>() {
            RustAttrValue::Float(f as f32) // Convert f64 to f32
        } else if let Ok(f) = value.extract::<f32>() {
            RustAttrValue::Float(f)
        } else if let Ok(s) = value.extract::<String>() {
            RustAttrValue::Text(s)
        } else if let Ok(vec) = value.extract::<Vec<f32>>() {
            RustAttrValue::FloatVec(vec)
        } else if let Ok(vec) = value.extract::<Vec<f64>>() {
            // Convert Vec<f64> to Vec<f32>
            let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
            RustAttrValue::FloatVec(f32_vec)
        } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
            RustAttrValue::Bytes(bytes)
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Unsupported attribute value type. Supported types: int, float, str, bool, List[float], bytes"
            ));
        };

        Ok(Self { inner: rust_value })
```

---

### value

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn value(&self, py: Python) -> PyObject
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 852.

**Implementation:**

```rust

        match &self.inner {
            RustAttrValue::Int(i) => i.to_object(py),
            RustAttrValue::Float(f) => f.to_object(py),
            RustAttrValue::Text(s) => s.to_object(py),
            RustAttrValue::Bool(b) => b.to_object(py),
            RustAttrValue::FloatVec(v) => v.to_object(py),
            RustAttrValue::Bytes(b) => b.to_object(py),
            // Handle optimized variants by extracting their underlying value
            RustAttrValue::CompactText(cs) => cs.as_str().to_object(py),
            RustAttrValue::SmallInt(i) => i.to_object(py),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::SubgraphRef(subgraph_id) => subgraph_id.to_object(py),
            RustAttrValue::NodeArray(node_ids) => node_ids.to_object(py),
            RustAttrValue::EdgeArray(edge_ids) => edge_ids.to_object(py),
            RustAttrValue::Null => py.None(),
        }
```

---

### type_name

**Attributes:**

```rust
#[getter]
```

**Signature:**

```rust
fn type_name(&self) -> &'static str
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 16, token-chars: 566.

**Implementation:**

```rust

        match &self.inner {
            RustAttrValue::Int(_) => "int",
            RustAttrValue::Float(_) => "float",
            RustAttrValue::Text(_) => "text",
            RustAttrValue::Bool(_) => "bool",
            RustAttrValue::FloatVec(_) => "float_vec",
            RustAttrValue::Bytes(_) => "bytes",
            RustAttrValue::CompactText(_) => "text",
            RustAttrValue::SmallInt(_) => "int",
            RustAttrValue::CompressedText(_) => "text",
            RustAttrValue::CompressedFloatVec(_) => "float_vec",
            RustAttrValue::SubgraphRef(_) => "subgraph_ref",
            RustAttrValue::NodeArray(_) => "node_array",
            RustAttrValue::EdgeArray(_) => "edge_array",
            RustAttrValue::Null => "null",
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 29, token-chars: 973.

**Implementation:**

```rust

        format!(
            "AttrValue({})",
            match &self.inner {
                RustAttrValue::Int(i) => i.to_string(),
                RustAttrValue::Float(f) => f.to_string(),
                RustAttrValue::Text(s) => format!("\"{}\"", s),
                RustAttrValue::Bool(b) => b.to_string(),
                RustAttrValue::FloatVec(v) => format!("{:?}", v),
                RustAttrValue::Bytes(b) => format!("b\"{:?}\"", b),
                RustAttrValue::CompactText(cs) => format!("\"{}\"", cs.as_str()),
                RustAttrValue::SmallInt(i) => i.to_string(),
                RustAttrValue::CompressedText(cd) => {
                    match cd.decompress_text() {
                        Ok(data) => format!("\"{}\"", data),
                        Err(_) => "compressed(error)".to_string(),
                    }
                }
                RustAttrValue::CompressedFloatVec(cd) => {
                    match cd.decompress_float_vec() {
                        Ok(data) => format!("{:?}", data),
                        Err(_) => "compressed(error)".to_string(),
                    }
                }
                RustAttrValue::SubgraphRef(subgraph_id) => format!("SubgraphRef({})", subgraph_id),
                RustAttrValue::NodeArray(node_ids) => format!("{:?}", node_ids),
                RustAttrValue::EdgeArray(edge_ids) => format!("{:?}", edge_ids),
                RustAttrValue::Null => "None".to_string(),
            }
        )
```

---

### __str__

**Signature:**

```rust
pub fn __str__(&self) -> PyResult<String>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 908.

**Implementation:**

```rust

        Ok(match &self.inner {
            RustAttrValue::Int(i) => i.to_string(),
            RustAttrValue::Float(f) => f.to_string(),
            RustAttrValue::Text(s) => s.clone(),
            RustAttrValue::Bool(b) => b.to_string(),
            RustAttrValue::FloatVec(v) => format!("{:?}", v),
            RustAttrValue::Bytes(b) => format!("{:?}", b),
            RustAttrValue::CompactText(cs) => cs.as_str().to_string(),
            RustAttrValue::SmallInt(i) => i.to_string(),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data,
                Err(_) => "compressed(error)".to_string(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => format!("{:?}", data),
                Err(_) => "compressed(error)".to_string(),
            },
            RustAttrValue::SubgraphRef(subgraph_id) => format!("SubgraphRef({})", subgraph_id),
            RustAttrValue::NodeArray(node_ids) => format!("{:?}", node_ids),
            RustAttrValue::EdgeArray(edge_ids) => format!("{:?}", edge_ids),
            RustAttrValue::Null => "None".to_string(),
        })
```

---

### __hash__

**Signature:**

```rust
fn __hash__(&self) -> u64
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 69, token-chars: 1375.

**Implementation:**

```rust

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Create a hash based on the variant and value
        match &self.inner {
            RustAttrValue::Int(i) => {
                0u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            RustAttrValue::Float(f) => {
                1u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            RustAttrValue::Text(s) => {
                2u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            RustAttrValue::Bool(b) => {
                3u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            RustAttrValue::FloatVec(v) => {
                4u8.hash(&mut hasher);
                for f in v {
                    f.to_bits().hash(&mut hasher);
                }
            }
            RustAttrValue::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            RustAttrValue::CompactText(cs) => {
                6u8.hash(&mut hasher);
                cs.as_str().hash(&mut hasher);
            }
            RustAttrValue::SmallInt(i) => {
                7u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            RustAttrValue::CompressedText(cd) => {
                8u8.hash(&mut hasher);
                if let Ok(text) = cd.decompress_text() {
                    text.hash(&mut hasher);
                }
            }
            RustAttrValue::CompressedFloatVec(cd) => {
                9u8.hash(&mut hasher);
                if let Ok(vec) = cd.decompress_float_vec() {
                    for f in vec {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            }
            RustAttrValue::SubgraphRef(subgraph_id) => {
                10u8.hash(&mut hasher);
                subgraph_id.hash(&mut hasher);
            }
            RustAttrValue::NodeArray(node_ids) => {
                11u8.hash(&mut hasher);
                node_ids.hash(&mut hasher);
            }
            RustAttrValue::EdgeArray(edge_ids) => {
                12u8.hash(&mut hasher);
                edge_ids.hash(&mut hasher);
            }
            RustAttrValue::Null => {
                13u8.hash(&mut hasher);
                // No additional data to hash for Null
            }
        }
        hasher.finish()
```

---

### to_object

**Signature:**

```rust
fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 22, token-chars: 862.

**Implementation:**

```rust

        match &self.inner {
            RustAttrValue::Int(i) => i.to_object(py),
            RustAttrValue::Float(f) => f.to_object(py),
            RustAttrValue::Text(s) => s.to_object(py),
            RustAttrValue::Bool(b) => b.to_object(py),
            RustAttrValue::FloatVec(v) => v.to_object(py),
            RustAttrValue::Bytes(b) => b.to_object(py),
            RustAttrValue::CompactText(cs) => cs.as_str().to_object(py),
            RustAttrValue::SmallInt(i) => (*i as i64).to_object(py),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::SubgraphRef(subgraph_id) => subgraph_id.to_object(py),
            RustAttrValue::NodeArray(node_ids) => node_ids.to_object(py),
            RustAttrValue::EdgeArray(edge_ids) => edge_ids.to_object(py),
            RustAttrValue::Null => py.None(),
        }
```

---

### __repr__

**Signature:**

```rust
fn __repr__(&self) -> String
```

**Why flagged:** body length exceeds trivial threshold. Lines: 6, token-chars: 105.

**Implementation:**

```rust

        format!(
            "ResultHandle(nodes={}, edges={}, type='{}')",
            self.nodes.len(),
            self.edges.len(),
            self.result_type
        )
```

---

### compute_stats

**Signature:**

```rust
fn compute_stats(&self, py: Python) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 64, token-chars: 1319.

**Implementation:**

```rust

        // Safe because we control the lifetime
        let graph = unsafe { &*self.graph_ref };

        let mut values = Vec::new();
        for &node_id in &self.node_ids {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                values.push(attr);
            }
        }

        // Compute statistics in Rust
        {
            let dict = PyDict::new(py);

            // Count
            dict.set_item("count", values.len())?;

            // Type-specific statistics
            if !values.is_empty() {
                match &values[0] {
                    RustAttrValue::Int(_) => {
                        let int_values: Vec<i64> = values
                            .iter()
                            .filter_map(|v| {
                                if let RustAttrValue::Int(i) = v {
                                    Some(*i)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !int_values.is_empty() {
                            let sum: i64 = int_values.iter().sum();
                            let avg = sum as f64 / int_values.len() as f64;
                            let min = *int_values.iter().min().unwrap();
                            let max = *int_values.iter().max().unwrap();

                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    }
                    RustAttrValue::Float(_) => {
                        let float_values: Vec<f32> = values
                            .iter()
                            .filter_map(|v| {
                                if let RustAttrValue::Float(f) = v {
                                    Some(*f)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !float_values.is_empty() {
                            let sum: f32 = float_values.iter().sum();
                            let avg = sum / float_values.len() as f32;
                            let min = float_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let max = float_values
                                .iter()
                                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    }
                    _ => {
                        // For other types, just provide count
                    }
                }
            }

            Ok(dict.to_object(py))
        }
```

---

### sample_values

**Signature:**

```rust
fn sample_values(&self, count: usize) -> PyResult<Vec<PyAttrValue>>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 13, token-chars: 341.

**Implementation:**

```rust

        let graph = unsafe { &*self.graph_ref };
        let mut results = Vec::new();

        let step = if self.node_ids.len() <= count {
            1
        } else {
            self.node_ids.len() / count
        };

        for &node_id in self.node_ids.iter().step_by(step).take(count) {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                results.push(PyAttrValue { inner: attr });
            }
        }

        Ok(results)
```

---


## ffi/utils.rs

### python_value_to_attr_value

**Signature:**

```rust
pub fn python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 102, token-chars: 2839.

**Implementation:**

```rust

    // Check for None/null first
    if value.is_none() {
        return Ok(RustAttrValue::Null);
    }

    // Fast path optimization: Check most common types first

    // Check booleans FIRST (Python bool is a subtype of int, so this must come before int check)
    if let Ok(b) = value.extract::<bool>() {
        return Ok(RustAttrValue::Bool(b));
    }

    // Check integers second (very common in benchmarks)
    if let Ok(i) = value.extract::<i64>() {
        return Ok(RustAttrValue::Int(i));
    }

    // Check strings third (also very common)
    if let Ok(s) = value.extract::<String>() {
        return Ok(RustAttrValue::Text(s));
    }

    // Check floats fourth
    if let Ok(f) = value.extract::<f64>() {
        return Ok(RustAttrValue::Float(f as f32)); // Convert f64 to f32
    }

    // Less common types
    if let Ok(f) = value.extract::<f32>() {
        return Ok(RustAttrValue::Float(f));
    } else if let Ok(vec) = value.extract::<Vec<f32>>() {
        return Ok(RustAttrValue::FloatVec(vec));
    } else if let Ok(vec) = value.extract::<Vec<f64>>() {
        // Convert Vec<f64> to Vec<f32>
        let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
        return Ok(RustAttrValue::FloatVec(f32_vec));
    } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
        return Ok(RustAttrValue::Bytes(bytes));
    }

    // Check for generic list types that we can convert
    if value.hasattr("__iter__").unwrap_or(false)
        && !value.is_instance_of::<pyo3::types::PyString>()
    {
        // This is some kind of iterable (list, tuple, etc.) but not a string

        // Try to extract as a list and convert all elements to the same type
        if let Ok(list) = value.extract::<Vec<&PyAny>>() {
            if list.is_empty() {
                // Empty list - default to empty float vector
                return Ok(RustAttrValue::FloatVec(vec![]));
            }

            // Try to determine the type by sampling the first element
            let first_elem = list[0];

            // Try as integers first
            if first_elem.extract::<i64>().is_ok() {
                let mut int_vec = Vec::with_capacity(list.len());
                for item in list {
                    match item.extract::<i64>() {
                        Ok(i) => int_vec.push(i),
                        Err(_) => {
                            return Err(PyErr::new::<PyTypeError, _>(
                                format!("Mixed types in list not supported. Expected all integers but found: {}", 
                                    item.get_type().name()?)
                            ));
                        }
                    }
                }
                // Store as comma-separated string for now (could add IntVec to AttrValue later)
                let int_str = int_vec
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                return Ok(RustAttrValue::Text(format!("[{}]", int_str)));
            }

            // Try as floats
            if first_elem.extract::<f64>().is_ok() || first_elem.extract::<f32>().is_ok() {
                let mut float_vec = Vec::with_capacity(list.len());
                for item in list {
                    if let Ok(f) = item.extract::<f64>() {
                        float_vec.push(f as f32);
                    } else if let Ok(f) = item.extract::<f32>() {
                        float_vec.push(f);
                    } else {
                        return Err(PyErr::new::<PyTypeError, _>(format!(
                            "Mixed types in list not supported. Expected all floats but found: {}",
                            item.get_type().name()?
                        )));
                    }
                }
                return Ok(RustAttrValue::FloatVec(float_vec));
            }

            // Try as strings
            if first_elem.extract::<String>().is_ok() {
                let mut string_vec = Vec::with_capacity(list.len());
                for item in list {
                    match item.extract::<String>() {
                        Ok(s) => string_vec.push(s),
                        Err(_) => {
                            return Err(PyErr::new::<PyTypeError, _>(
                                format!("Mixed types in list not supported. Expected all strings but found: {}", 
                                    item.get_type().name()?)
                            ));
                        }
                    }
                }
                // Store as JSON-like string representation
                let strings_json = format!(
                    "[{}]",
                    string_vec
                        .iter()
                        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                return Ok(RustAttrValue::Text(strings_json));
            }

            // If we get here, it's a list of an unsupported type
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "Unsupported list element type: {}. Supported list types: [int], [float], [str]",
                first_elem.get_type().name()?
            )));
        }
    }

    // If we get here, it's a completely unsupported type
    Err(PyErr::new::<PyTypeError, _>(
        format!("Unsupported attribute value type: {}. Supported types: int, float, str, bool, bytes, [int], [float], [str]", 
            value.get_type().name()?)
    ))
```

---

### attr_value_to_python_value

**Signature:**

```rust
pub fn attr_value_to_python_value(py: Python, attr_value: &RustAttrValue) -> PyResult<PyObject>
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 17, token-chars: 842.

**Implementation:**

```rust

    let py_value = match attr_value {
        RustAttrValue::Int(i) => i.to_object(py),
        RustAttrValue::Float(f) => f.to_object(py),
        RustAttrValue::Text(s) => s.to_object(py),
        RustAttrValue::Bool(b) => b.to_object(py),
        RustAttrValue::FloatVec(vec) => vec.to_object(py),
        RustAttrValue::Bytes(bytes) => bytes.to_object(py),
        RustAttrValue::CompactText(compact_str) => compact_str.as_str().to_object(py),
        RustAttrValue::SmallInt(i) => i.to_object(py),
        RustAttrValue::CompressedText(_) => "compressed_text".to_object(py), // Placeholder
        RustAttrValue::CompressedFloatVec(_) => vec!["compressed_floats"].to_object(py), // Placeholder
        RustAttrValue::SubgraphRef(subgraph_id) => subgraph_id.to_object(py),
        RustAttrValue::NodeArray(node_ids) => node_ids.to_object(py),
        RustAttrValue::EdgeArray(edge_ids) => edge_ids.to_object(py),
        RustAttrValue::Null => py.None(),
    };
    Ok(py_value)
```

---

### graph_error_to_py_err

**Signature:**

```rust
pub fn graph_error_to_py_err(error: GraphError) -> PyErr
```

**Why flagged:** control-flow / conversions / error handling present; body length exceeds trivial threshold. Lines: 30, token-chars: 749.

**Implementation:**

```rust

    match error {
        GraphError::NodeNotFound {
            node_id,
            operation,
            suggestion,
        } => PyErr::new::<PyValueError, _>(format!(
            "Node {} not found during {}. {}",
            node_id, operation, suggestion
        )),
        GraphError::EdgeNotFound {
            edge_id,
            operation,
            suggestion,
        } => PyErr::new::<PyValueError, _>(format!(
            "Edge {} not found during {}. {}",
            edge_id, operation, suggestion
        )),
        GraphError::InvalidInput(message) => PyErr::new::<PyValueError, _>(message),
        GraphError::NotImplemented {
            feature,
            tracking_issue,
        } => {
            let mut message = format!("Feature '{}' is not yet implemented", feature);
            if let Some(issue) = tracking_issue {
                message.push_str(&format!(". See: {}", issue));
            }
            PyErr::new::<PyRuntimeError, _>(message)
        }
        _ => PyErr::new::<PyRuntimeError, _>(format!("Graph error: {}", error)),
    }
```

---

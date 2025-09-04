//! Python bindings for MetaGraph Composer
//!
//! Clean, intuitive API for meta-node creation that replaces the complex
//! EdgeAggregationConfig system.

use crate::ffi::subgraphs::hierarchical::PyMetaNode;
use groggy::subgraphs::{Subgraph, composer::{EdgeStrategy, MetaNodePlan, ComposerPreview}};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;

/// Python wrapper for EdgeStrategy
#[pyclass(name = "EdgeStrategy")]
#[derive(Clone, Debug)]
pub struct PyEdgeStrategy {
    pub inner: EdgeStrategy,
}

#[pymethods]
impl PyEdgeStrategy {
    #[new]
    fn new(strategy: &str) -> PyResult<Self> {
        let inner = EdgeStrategy::from_str(strategy)
            .map_err(|e| PyValueError::new_err(format!("Invalid edge strategy: {}", e)))?;
        Ok(PyEdgeStrategy { inner })
    }
    
    /// Create aggregate strategy (default)
    #[classmethod]
    fn aggregate(_cls: &PyType) -> Self {
        PyEdgeStrategy { inner: EdgeStrategy::Aggregate }
    }
    
    /// Create keep_external strategy  
    #[classmethod]
    fn keep_external(_cls: &PyType) -> Self {
        PyEdgeStrategy { inner: EdgeStrategy::KeepExternal }
    }
    
    /// Create drop_all strategy
    #[classmethod]
    fn drop_all(_cls: &PyType) -> Self {
        PyEdgeStrategy { inner: EdgeStrategy::DropAll }
    }
    
    /// Create contract_all strategy
    #[classmethod]
    fn contract_all(_cls: &PyType) -> Self {
        PyEdgeStrategy { inner: EdgeStrategy::ContractAll }
    }
    
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
    
    fn __repr__(&self) -> String {
        format!("EdgeStrategy.{:?}", self.inner)
    }
}

/// Python wrapper for ComposerPreview
#[pyclass(name = "ComposerPreview")]
#[derive(Clone)]
pub struct PyComposerPreview {
    pub inner: ComposerPreview,
}

#[pymethods]
impl PyComposerPreview {
    /// Get the meta-node attributes that will be created
    #[getter]
    fn meta_node_attributes(&self) -> HashMap<String, String> {
        self.inner.meta_node_attributes.clone()
    }
    
    /// Get estimated number of meta-edges
    #[getter]
    fn meta_edges_count(&self) -> usize {
        self.inner.meta_edges_count
    }
    
    /// Get the edge strategy
    #[getter]
    fn edge_strategy(&self) -> PyEdgeStrategy {
        PyEdgeStrategy { inner: self.inner.edge_strategy.clone() }
    }
    
    /// Whether edge count will be included
    #[getter]
    fn will_include_edge_count(&self) -> bool {
        self.inner.will_include_edge_count
    }
    
    /// Get the entity type
    #[getter]
    fn entity_type(&self) -> String {
        self.inner.entity_type.clone()
    }
    
    fn __str__(&self) -> String {
        format!(
            "ComposerPreview(attributes={}, meta_edges={}, strategy={:?}, entity_type='{}')",
            self.inner.meta_node_attributes.len(),
            self.inner.meta_edges_count,
            self.inner.edge_strategy,
            self.inner.entity_type
        )
    }
    
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Python wrapper for MetaNodePlan
#[pyclass(name = "MetaNodePlan")]
pub struct PyMetaNodePlan {
    pub inner: MetaNodePlan,
}

#[pymethods]
impl PyMetaNodePlan {
    /// Preview what the plan will create without executing
    fn preview(&self) -> PyComposerPreview {
        PyComposerPreview {
            inner: self.inner.preview()
        }
    }
    
    /// Add a node aggregation to the plan
    /// 
    /// # Arguments
    /// * `target` - Target attribute name
    /// * `function` - Aggregation function ("mean", "sum", "count", etc.)
    /// * `source` - Source attribute name (optional, defaults to target)
    #[pyo3(signature = (target, function, source = None))]
    fn with_node_agg(&mut self, target: String, function: String, source: Option<String>) {
        self.inner = self.inner.clone().with_node_agg(target, function, source);
    }
    
    /// Add an edge aggregation to the plan
    /// 
    /// # Arguments  
    /// * `attr_name` - Edge attribute name to aggregate
    /// * `function` - Aggregation function ("mean", "sum", "concat", etc.)
    fn with_edge_agg(&mut self, attr_name: String, function: String) {
        self.inner = self.inner.clone().with_edge_agg(attr_name, function);
    }
    
    /// Set the edge strategy
    fn with_edge_strategy(&mut self, strategy: &PyEdgeStrategy) {
        self.inner = self.inner.clone().with_edge_strategy(strategy.inner.clone());
    }
    
    /// Set the entity type
    fn with_entity_type(&mut self, entity_type: String) {
        self.inner = self.inner.clone().with_entity_type(entity_type);
    }
    
    /// Apply a preset configuration
    fn with_preset(&mut self, preset_name: String) -> PyResult<()> {
        self.inner = self.inner.clone().with_preset(&preset_name)
            .map_err(|e| PyValueError::new_err(format!("Invalid preset: {}", e)))?;
        Ok(())
    }
    
    /// Execute the plan and create the meta-node
    /// 
    /// This method needs to be called from the subgraph that created the plan.
    /// For now, we'll implement this when we add the collapse method to PySubgraph.
    fn add_to_graph(&self) -> PyResult<PyObject> {
        Err(PyRuntimeError::new_err(
            "add_to_graph() must be called from the subgraph that created this plan. Use subgraph.collapse(...).add_to_graph() instead."
        ))
    }
    
    fn __str__(&self) -> String {
        let preview = self.inner.preview();
        format!(
            "MetaNodePlan(node_aggs={}, edge_aggs={}, strategy={:?})",
            self.inner.node_aggs.len(),
            self.inner.edge_aggs.len(),
            self.inner.edge_strategy
        )
    }
    
    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Utility functions for parsing Python input into composer format
pub fn parse_node_aggs_from_python(node_aggs: &PyAny) -> PyResult<Vec<(String, String, Option<String>)>> {
    let mut result = Vec::new();
    
    // Handle dict format: {"target": "function", "target2": ("function", "source")}
    if let Ok(dict) = node_aggs.downcast::<PyDict>() {
        for (key, value) in dict {
            let target = key.extract::<String>()?;
            
            if let Ok(function) = value.extract::<String>() {
                // Simple format: {"salary": "mean"}
                result.push((target.clone(), function, Some(target)));
            } else if let Ok(tuple) = value.extract::<(String, String)>() {
                // Tuple format: {"avg_salary": ("mean", "salary")}
                let (function, source) = tuple;
                result.push((target, function, Some(source)));
            } else {
                return Err(PyValueError::new_err(format!(
                    "Invalid node aggregation format for '{}'. Expected string or (function, source) tuple.",
                    target
                )));
            }
        }
    }
    // Handle list format: [("target", "function"), ("target", "function", "source")]
    else if let Ok(list) = node_aggs.iter() {
        for item in list {
            let item = item?;
            if let Ok(tuple2) = item.extract::<(String, String)>() {
                let (target, function) = tuple2;
                result.push((target.clone(), function, Some(target)));
            } else if let Ok(tuple3) = item.extract::<(String, String, String)>() {
                let (target, function, source) = tuple3;
                result.push((target, function, Some(source)));
            } else {
                return Err(PyValueError::new_err(
                    "Invalid node aggregation format. Expected (target, function) or (target, function, source) tuples."
                ));
            }
        }
    } else {
        return Err(PyValueError::new_err(
            "node_aggs must be a dict or list of tuples"
        ));
    }
    
    Ok(result)
}

/// Parse edge aggregations from Python dict
pub fn parse_edge_aggs_from_python(edge_aggs: &PyAny) -> PyResult<Vec<(String, String)>> {
    let mut result = Vec::new();
    
    if let Ok(dict) = edge_aggs.downcast::<PyDict>() {
        for (key, value) in dict {
            let attr_name = key.extract::<String>()?;
            let function = value.extract::<String>()?;
            result.push((attr_name, function));
        }
    } else {
        return Err(PyValueError::new_err("edge_aggs must be a dict"));
    }
    
    Ok(result)
}

/// Simplified Python wrapper for immediate execution
/// 
/// Since we can't store Rc<RefCell<Graph>> in a PyClass, we'll execute immediately.
/// This still provides the clean API but without the plan/execute separation.
#[pyclass(name = "MetaNodePlan")]
pub struct PyMetaNodePlanExecutor {
    // Just store preview info since we can't store the actual plan
    preview_info: ComposerPreview,
}

impl PyMetaNodePlanExecutor {
    pub fn new(plan: MetaNodePlan) -> Self {
        Self { 
            preview_info: plan.preview() 
        }
    }
    
    pub fn execute_immediately<T: groggy::traits::SubgraphOperations>(plan: MetaNodePlan, subgraph: &T) -> PyResult<groggy::subgraphs::MetaNode> {
        plan.add_to_graph(subgraph)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create meta-node: {}", e)))
    }
}

#[pymethods]
impl PyMetaNodePlanExecutor {
    /// Preview what was configured  
    fn preview(&self) -> PyComposerPreview {
        PyComposerPreview {
            inner: self.preview_info.clone()
        }
    }
    
    /// This is just a placeholder - actual execution happened already
    fn add_to_graph(&self, _py: Python) -> PyResult<PyObject> {
        Err(PyRuntimeError::new_err(
            "This MetaNodePlan has already been executed. The MetaNode was returned directly from collapse()."
        ))
    }
    
    fn __str__(&self) -> String {
        format!(
            "MetaNodePlan(executed - {} attributes, {} meta-edges, strategy={:?})",
            self.preview_info.meta_node_attributes.len(),
            self.preview_info.meta_edges_count,
            self.preview_info.edge_strategy
        )
    }
    
    fn __repr__(&self) -> String {
        self.__str__()
    }
}